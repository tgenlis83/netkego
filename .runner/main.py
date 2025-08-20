import torch
import torch.nn as nn
import torch.optim as optim
from contextlib import nullcontext
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as T
from tqdm import tqdm
from pathlib import Path
import time
from runner.generated_block import GeneratedBlock, CIN, H, W, GeneratedModel

STOP_PATH = Path(".runner/STOP")
RESUME_CFG = Path(".runner/RESUME.json")
def should_stop():
    try: return STOP_PATH.exists()
    except Exception: return False

def get_resume_request():
    try:
        import json
        if RESUME_CFG.exists():
            data = json.loads(RESUME_CFG.read_text())
            RESUME_CFG.unlink(missing_ok=True)
            return data
    except Exception as e:
        print("WARN: resume read error:", e)
    return None

def get_datasets(root="./data", val_split=0.1, sample_pct=100):
    mean_std = { 'CIFAR10': ([0.4914,0.4822,0.4465],[0.247,0.243,0.261]), 'CIFAR100': ([0.507,0.487,0.441],[0.267,0.256,0.276]), 'MNIST': ([0.1307],[0.3081]), 'FashionMNIST': ([0.2860],[0.3530]), 'STL10': ([0.4467,0.4398,0.4066],[0.2603,0.2566,0.2713]) }
    _fallback = ([0.5]*3, [0.5]*3)
    mean,std = mean_std.get('CIFAR10', _fallback)
    tf_train = T.Compose([T.AutoAugment(policy=T.AutoAugmentPolicy.CIFAR10), T.ToTensor(), T.Normalize(mean, std)])
    tf_test  = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    full = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=tf_train)
    test = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=tf_test)
    # Optional training subset sampling BEFORE val split
    sample_pct = max(1, min(100, int(sample_pct)))
    if sample_pct < 100:
        g = torch.Generator().manual_seed(42)
        idx = torch.randperm(len(full), generator=g)[: max(1, int(len(full) * (sample_pct/100.0)))]
        full = torch.utils.data.Subset(full, idx.tolist())
    n_val = max(1, int(len(full) * val_split))
    n_train = max(1, len(full) - n_val)
    train, val = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    return train, val, test

def get_model(device):
    model = GeneratedModel(in_channels=CIN)
    return model.to(device)

def get_loss():
    return nn.CrossEntropyLoss()

def get_optimizer(model):
    return optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

def get_scheduler(opt):
    return None

def accuracy(logits, targets):
    if logits.dim()>2: logits = logits.mean(dim=(-1,-2))
    preds = logits.argmax(dim=1)
    return (preds==targets).float().mean().item()

def ensure_trainable(model, sample_loader, device, num_classes):
    try:
        nparams = sum(p.numel() for p in model.parameters())
    except Exception:
        nparams = 0
    if nparams>0:
        return model
    print("WARN: Model has no trainable parameters; adding a small linear head.")
    # infer feature dim with a dry forward
    x0,_ = next(iter(sample_loader))
    x0 = x0.to(device)[:1]
    with torch.no_grad():
        y0 = model(x0)
        if y0.dim()>2: y0 = y0.mean(dim=(-1,-2))
        feat = y0.shape[1] if y0.dim()==2 else int(y0.numel())
    class Wrap(nn.Module):
        def __init__(self, base, feat, num_classes):
            super().__init__()
            self.base = base
            self.head = nn.Linear(feat, num_classes)
        def forward(self, x):
            out = self.base(x)
            if out.dim()>2: out = out.mean(dim=(-1,-2))
            return self.head(out)
    w = Wrap(model, int(feat), 10).to(device)
    return w

def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip=0.0, global_step_start=0, precision="fp32", scaler=None, epoch_prefix=""):
    model.train(); total=0.0; global_step=global_step_start
    import os; os.makedirs("checkpoints", exist_ok=True)
    desc = f"{epoch_prefix} - train" if epoch_prefix else "train"
    for x,y in tqdm(loader, desc=desc, leave=False):
        if should_stop():
            print("STOP: requested — exiting train loop.")
            try: STOP_PATH.unlink(missing_ok=True)
            except Exception: pass
            break
        x=x.to(device); y=y.to(device)
        optimizer.zero_grad()
        with get_amp_context(device, precision):
            out = model(x)
            if out.dim()>2: out = out.mean(dim=(-1,-2))
            loss = criterion(out, y)
        if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
            scaler.scale(loss).backward()
            if grad_clip>0: scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip>0: nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        total += loss.item() * x.size(0)
        global_step += 1
        # per-step checkpoint
        ckpt_step={"model": model.state_dict(), "optimizer": optimizer.state_dict(), "global_step": global_step}
        torch.save(ckpt_step, f"checkpoints/step_{global_step:06d}.pt")
        torch.save(ckpt_step, "checkpoints/last_step.pt")
    return total / len(loader.dataset), global_step

def evaluate(model, loader, criterion, device, precision="fp32", epoch_prefix=""):
    model.eval(); total=0.0; accs=0.0
    with torch.no_grad():
        desc = f"{epoch_prefix} - val" if epoch_prefix else "val"
        for x,y in tqdm(loader, desc=desc, leave=False):
            if should_stop():
                print("STOP: requested — exiting val loop.")
                try: STOP_PATH.unlink(missing_ok=True)
                except Exception: pass
                break
            x=x.to(device); y=y.to(device)
            with get_amp_context(device, precision):
                out = model(x)
                if out.dim()>2: out = out.mean(dim=(-1,-2))
                loss = criterion(out, y)
            total += loss.item() * x.size(0)
            accs += accuracy(out, y) * x.size(0)
    return total/len(loader.dataset), accs/len(loader.dataset)

def confusion_matrix(model, loader, device, num_classes):
    import torch
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    model.eval()
    with torch.no_grad():
        for x,y in tqdm(loader, desc="confusion", leave=False):
            x=x.to(device); y=y.to(device)
            out = model(x)
            if out.dim()>2: out = out.mean(dim=(-1,-2))
            pred = out.argmax(dim=1)
            for t,p in zip(y.view(-1), pred.view(-1)):
                cm[t.long(), p.long()] += 1
    return cm

def resolve_device(pref):
    if pref=="cuda" and torch.cuda.is_available(): return torch.device("cuda")
    if pref=="mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return torch.device("mps")
    if pref=="cpu": return torch.device("cpu")
    # fallback to CPU when requested device unavailable
    return torch.device("cpu")

def get_amp_context(device, precision):
    prec = str(precision or "fp32")
    dev = str(device)
    if prec == "amp_fp16" and dev in ("cuda","mps"):
        try: return torch.autocast(device_type=dev, dtype=torch.float16)
        except Exception: return nullcontext()
    if prec == "amp_bf16":
        try: return torch.autocast(device_type=dev, dtype=torch.bfloat16)
        except Exception: return nullcontext()
    return nullcontext()

def reset_peak_mem(device):
    dev = str(device)
    try:
        if dev=="cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

def get_peak_gpu_mem_mb(device):
    dev = str(device)
    try:
        if dev=="cuda" and torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated())/(1024*1024)
        if dev=="mps" and hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
            return float(torch.mps.current_allocated_memory())/(1024*1024)
    except Exception:
        return float("nan")
    return float("nan")

def get_rss_mem_mb():
    try:
        import os, psutil
        return float(psutil.Process(os.getpid()).memory_info().rss)/(1024*1024)
    except Exception:
        return float("nan")

def main():
    device = resolve_device("cuda")
    print("DEVICE:", str(device))
    precision = "amp_fp16"
    train_ds, val_ds, test_ds = get_datasets(sample_pct=100)
    train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True, num_workers=12, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=12)
    test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False, num_workers=12)
    model = get_model(device)
    model = ensure_trainable(model, train_loader, device, 10)
    criterion = get_loss()
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=(precision=="amp_fp16" and str(device)=="cuda"))
    best=0.0
    import os; os.makedirs("checkpoints", exist_ok=True)
    global_step = 0
    # Resume if requested
    resume = get_resume_request()
    if resume:
        from pathlib import Path as _Path
        ckpt_path = _Path(resume.get("path","checkpoints/best.pt"))
        if ckpt_path.exists():
            try:
                payload = torch.load(ckpt_path, map_location=device)
                sd = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
                _incomp = model.load_state_dict(sd, strict=False)
                try:
                    miss = getattr(_incomp, "missing_keys", [])
                    unexp = getattr(_incomp, "unexpected_keys", [])
                except Exception:
                    miss, unexp = [], []
                if miss: print("WARN: missing keys:", miss)
                if unexp: print("WARN: unexpected keys:", unexp)
                if resume.get("mode","full") == "full":
                    if "optimizer" in payload: optimizer.load_state_dict(payload["optimizer"])
                    if "scheduler" in payload and scheduler:
                        try: scheduler.load_state_dict(payload["scheduler"])
                        except Exception: pass
                    global_step = int(payload.get("global_step", 0))
                    start_epoch = int(payload.get("epoch", 0)) + 1
                else:
                    start_epoch = 1
                best = float(payload.get("best", 0.0))
                temp = resume.get("mode", "full")
                print(f"RESUME: loaded {ckpt_path} mode={temp} start_epoch={start_epoch} best={best:.4f} global_step={global_step}")
            except Exception as e:
                print("WARN: failed to resume:", e)
                start_epoch = 1
        else:
            print(f"WARN: checkpoint not found: {ckpt_path}")
            start_epoch = 1
    else:
        start_epoch = 1
    for epoch in range(start_epoch, 30+1):
        if should_stop():
            print("STOP: requested — stopping before new epoch.")
            try: STOP_PATH.unlink(missing_ok=True)
            except Exception: pass
            break
        print("EPOCH:", epoch, "/30")
        reset_peak_mem(device)
        _t0 = time.time()
        tr_loss, global_step = train_one_epoch(model, train_loader, criterion, optimizer, device, grad_clip=0, global_step_start=global_step, precision=precision, scaler=scaler, epoch_prefix=f"Epoch {epoch}/30")
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, precision, epoch_prefix=f"Epoch {epoch}/30")
        epoch_time_sec = max(1e-9, time.time() - _t0)
        gpu_mem_mb = get_peak_gpu_mem_mb(device)
        rss_mem_mb = get_rss_mem_mb()
        try:
            (scheduler.step() if scheduler else None)
        except Exception:
            pass
        # track average epoch time so far
        if epoch == start_epoch: avg_epoch_time_sec = epoch_time_sec
        else: avg_epoch_time_sec = ((epoch - start_epoch) * avg_epoch_time_sec + epoch_time_sec) / max(1, (epoch - start_epoch + 1)) if "avg_epoch_time_sec" in locals() else epoch_time_sec
        try: cur_lr = float(optimizer.param_groups[0]["lr"])
        except Exception: cur_lr = float("nan")
        print(f"METRIC: epoch={epoch} train_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} epoch_time_sec={epoch_time_sec:.3f} avg_epoch_time_sec={avg_epoch_time_sec:.3f} gpu_mem_mb={gpu_mem_mb:.1f} rss_mem_mb={rss_mem_mb:.1f} lr={cur_lr:.6e}")
        improved = val_acc>best
        if improved: best=val_acc; print(f"BEST: val_acc={best:.4f}")
        # save checkpoints each epoch and best
        ckpt={"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "best": best, "global_step": global_step, "val_acc": float(val_acc)}
        if scheduler: ckpt["scheduler"]=scheduler.state_dict()
        fname = f"checkpoints/epoch_{epoch:03d}_val{val_acc:.4f}.pt"
        torch.save(ckpt, fname)
        torch.save(ckpt, "checkpoints/last.pt")
        # save best checkpoint only when improved
        if improved:
            torch.save(ckpt, "checkpoints/best.pt")
            print(f"CKPT: type=best path=checkpoints/best.pt epoch={epoch} val_acc={val_acc:.4f}")
        print(f"CKPT: type=epoch path={fname} epoch={epoch} val_acc={val_acc:.4f}")
    tl, ta = evaluate(model, test_loader, criterion, device, precision)
    print(f"TEST: acc={ta:.4f} loss={tl:.4f}")
    # Save confusion matrix for classification tasks (num_classes>1)
    try:
        if 10 > 1:
            cm = confusion_matrix(model, test_loader, device, 10)
            import json
            import os
            os.makedirs("checkpoints", exist_ok=True)
            counts = cm.tolist()
            # row-normalize
            import math
            norm = []
            for row in counts:
                s = float(sum(row))
                norm.append([ (x/s if s>0 else 0.0) for x in row ])
            Path("checkpoints/confusion.json").write_text(json.dumps({"counts": counts, "normalized": norm}))
    except Exception as e:
        print("WARN: failed to save confusion matrix:", e)

if __name__ == "__main__":
    main()