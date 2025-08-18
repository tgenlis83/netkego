import os, sys, traceback, importlib.util, types, time, runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEN_PATH = ROOT / '.runner' / 'generated_block.py'
MAIN_PATH = ROOT / '.runner' / 'main.py'

ANSI_RED = "\x1b[31m"; ANSI_GREEN = "\x1b[32m"; ANSI_YELLOW = "\x1b[33m"; ANSI_RESET = "\x1b[0m"

def write(msg: str):
    sys.stdout.write(msg)
    sys.stdout.flush()

def write_line(msg: str = ""):
    write(msg + "\n")

def tee_write(line: str):
    write(line)


def load_generated():
    if not GEN_PATH.exists():
        raise FileNotFoundError(f"Missing {GEN_PATH}. Create it from the PyTorch tab (Copy/Download).")
    spec = importlib.util.spec_from_file_location("generated_block", GEN_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def main():
    # nothing to clear, server accumulates logs
    start = time.time()
    try:
        tee_write("[runner] Starting...\n")
        if MAIN_PATH.exists():
            write_line(ANSI_YELLOW + "[runner] Running .runner/main.py" + ANSI_RESET)
            # Ensure project root on path so `import runner.generated_block` works
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))
            # Execute the script with __name__ == "__main__" so its guard runs
            runpy.run_path(str(MAIN_PATH), run_name="__main__")
            write_line(ANSI_GREEN + "[runner] main.py finished" + ANSI_RESET)
        else:
            write_line(ANSI_YELLOW + "[runner] Loading generated module..." + ANSI_RESET)
            mod = load_generated()
            write_line(ANSI_GREEN + "[runner] Loaded." + ANSI_RESET)

            import torch
            CIN = getattr(mod, 'CIN', 64)
            H = getattr(mod, 'H', 56)
            W = getattr(mod, 'W', 56)
            x = torch.randn(1, CIN, H, W)
            model_cls = getattr(mod, 'GeneratedBlock', None)
            if model_cls is None:
                raise RuntimeError('Generated module does not define GeneratedBlock')
            m = model_cls(in_channels=CIN)
            m.eval()
            with torch.no_grad():
                y = m(x)
            write_line(ANSI_GREEN + f"[runner] OK: output shape = {tuple(y.shape)}" + ANSI_RESET)
            tee_write(f"OK: output shape = {tuple(y.shape)}\n")
    except Exception:
        tb = traceback.format_exc()
        write_line(ANSI_RED + tb + ANSI_RESET)
        tee_write(tb + "\n")
        sys.exit(1)
    finally:
        dt = time.time() - start
        write_line(ANSI_YELLOW + f"[runner] Done in {dt:.3f}s" + ANSI_RESET)
        tee_write(f"Done in {dt:.3f}s\n")

if __name__ == '__main__':
    main()
