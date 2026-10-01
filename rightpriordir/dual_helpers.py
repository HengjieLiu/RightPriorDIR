"""Audited registration/data kernels; see docs/methods.md for protocol scope."""

from __future__ import annotations
import random
import numpy as np
import torch

AUTHOR_ADAMW_WEIGHT_DECAY = 0.01


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | int) -> torch.device:
    if isinstance(device, int) or str(device).isdigit():
        resolved = torch.device(f"cuda:{int(device)}")
    else:
        resolved = torch.device(str(device))
    if resolved.type == "cuda" and (not torch.cuda.is_available()):
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {resolved}")
    return resolved
