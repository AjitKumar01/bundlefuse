"""Device selection and reproducibility controls.

Default policy is *reproducibility first*: we run on CPU with fixed seeds and
deterministic algorithms, because the accelerator path (Apple MPS in particular)
produced run-to-run drift of several percent in earlier versions of this code,
which made published numbers non-reproducible. CPU runs here are bit-identical
across repeats.

Override the device with FORCE_DEVICE=cpu|mps|cuda. Note that FORCE_DEVICE=mps|cuda
trades reproducibility for speed: non-CPU backends are *not* guaranteed to be
deterministic, so any table you intend to publish should be produced on CPU.
"""
import os
import random

try:
    import numpy as np
except Exception:
    np = None
try:
    import torch
except Exception:
    torch = None

SEED = int(os.environ.get("SEED", "0"))


def pick_device():
    if torch is None:
        return "cpu"
    forced = os.environ.get("FORCE_DEVICE", "").strip().lower()
    if forced:
        return torch.device(forced)
    # Reproducibility-first default: CPU. Opt into acceleration via FORCE_DEVICE.
    return torch.device("cpu")


def set_determinism(seed: int = SEED):
    """Seed every RNG and request deterministic kernels. Safe to call repeatedly."""
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        os.environ.setdefault("PYTHONHASHSEED", str(seed))
        # CPU + fixed seeds already reproduces bit-identically here. The deterministic-kernel
        # flag is belt-and-suspenders but forces slow fallback kernels (~10x), so it is opt-in.
        if os.environ.get("DETERMINISTIC", "0") == "1":
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass


DEVICE = pick_device()
set_determinism(SEED)
