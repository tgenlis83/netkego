import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as T
from tqdm import tqdm
from runner.generated_block import GeneratedBlock, CIN, H, W, GeneratedModel

def get_datasets(root="./data", val_split=0.1):
    mean_std = { 'CIFAR10': ([0.4914,0.4822,0.4465],[0.247,0.243,0.261]), 'CIFAR100': ([0.507,0.487,0.441],[0.267,0.256,0.276]), 'MNIST': ([0.1307],[0.3081]), 'FashionMNIST': ([0.2860],[0.3530]), 'STL10': ([0.4467,0.4398,0.4066],[0.2603,0.2566,0.2713]) }
    mean,std = mean_std.get('MNIST', ([0.5]*1, [0.5]*1))
    tf_train = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    tf_test  = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    full = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=tf_train)
    test = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=tf_test)
    n_val = max(1, int(len(full) * val_split))
    n_train = len(full) - n_val
    train, val = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    return train, val, test

def get_model(device):
    model = GeneratedModel(in_channels=CIN)
    return model.to(device)

def get_loss():
    return nn.CrossEntropyLoss()

def get_optimizer(model):
    return optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.0001)

def get_scheduler(opt):
    return optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)

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

def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip=0.0, global_step_start=0):
    model.train(); total=0.0; global_step=global_step_start
    import os; os.makedirs("checkpoints", exist_ok=True)
    for x,y in tqdm(loader, desc="train", leave=False):
        x=x.to(device); y=y.to(device)
        optimizer.zero_grad()
        out = model(x)
        if out.dim()>2: out = out.mean(dim=(-1,-2))
        loss = criterion(out, y)
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

def evaluate(model, loader, criterion, device):
    model.eval(); total=0.0; accs=0.0
    with torch.no_grad():
        for x,y in tqdm(loader, desc="val", leave=False):
            x=x.to(device); y=y.to(device)
            out = model(x)
            if out.dim()>2: out = out.mean(dim=(-1,-2))
            loss = criterion(out, y)
            total += loss.item() * x.size(0)
            accs += accuracy(out, y) * x.size(0)
    return total/len(loader.dataset), accs/len(loader.dataset)

def resolve_device(pref="auto"):
    if pref=="cuda" and torch.cuda.is_available(): return torch.device("cuda")
    if pref=="mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return torch.device("mps")
    if pref=="cpu": return torch.device("cpu")
    # auto fallback: CUDA ▶ MPS ▶ CPU
    if torch.cuda.is_available(): return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def main():
    device = resolve_device("auto")
    train_ds, val_ds, test_ds = get_datasets()
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=8)
    model = get_model(device)
    model = ensure_trainable(model, train_loader, device, 10)
    criterion = get_loss()
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)
    best=0.0
    import os; os.makedirs("checkpoints", exist_ok=True)
    global_step = 0
    for epoch in range(1, 10+1):
        print("EPOCH:", epoch, "/10")
        tr_loss, global_step = train_one_epoch(model, train_loader, criterion, optimizer, device, grad_clip=0, global_step_start=global_step)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        try:
            (scheduler.step() if scheduler else None)
        except Exception:
            pass
        print(f"METRIC: epoch={epoch} train_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc>best: best=val_acc; print(f"BEST: val_acc={best:.4f}")
        # save checkpoints each epoch and best
        ckpt={"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "best": best}
        if scheduler: ckpt["scheduler"]=scheduler.state_dict()
        torch.save(ckpt, f"checkpoints/epoch_{epoch:03d}.pt")
        torch.save(ckpt, "checkpoints/last.pt")
        # save best checkpoint only when improved
        if val_acc>=best: torch.save(ckpt, "checkpoints/best.pt")
    tl, ta = evaluate(model, test_loader, criterion, device)
    print(f"TEST: acc={ta:.4f} loss={tl:.4f}")

if __name__ == "__main__":
    main()