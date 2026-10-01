"""Audited registration/data kernels; see docs/methods.md for protocol scope."""

from __future__ import annotations
from typing import Dict, Sequence
import numpy as np
import torch
import torch.nn.functional as F

MARGIN_XYZ = np.asarray([10, 10, 5], dtype=np.int64)

MARGIN_ZYX = MARGIN_XYZ[[2, 1, 0]]


def img_thr(
    vol: np.ndarray, min_val: float = 80.0, max_val: float = 900.0, norm: float = 1.0
) -> np.ndarray:
    out = np.clip(vol.astype(np.float32), min_val, max_val)
    out = out - min_val
    out = out * (norm / (max_val - min_val))
    return out.astype(np.float32, copy=False)


def landmarks_xyz_one_based_to_zyx_zero_based(landmarks_xyz: np.ndarray) -> np.ndarray:
    """Convert DIR-Lab/pTVReg landmark file coordinates to NumPy zyx indices."""
    return landmarks_xyz[:, [2, 1, 0]].astype(np.float64, copy=False) - 1.0


def ptvreg_crop_bounds_zyx_from_landmarks_xyz(
    fixed_xyz: np.ndarray, moving_xyz: np.ndarray, shape_zyx: Sequence[int]
) -> np.ndarray:
    """Return pTVReg crop bounds as Python 0-based inclusive zyx bounds.

    pTVReg computes crop bounds in MATLAB's 1-based xyz array convention:
    crop_v = [max(1, min_xyz - [10,10,5]), min(shape_xyz, max_xyz + [10,10,5])].
    This function converts those inclusive bounds to NumPy's 0-based zyx order.
    """
    both = np.concatenate([fixed_xyz, moving_xyz], axis=0)
    lo1_xyz = np.floor(np.min(both, axis=0)).astype(np.int64)
    hi1_xyz = np.ceil(np.max(both, axis=0)).astype(np.int64)
    shape_xyz = np.asarray(tuple(shape_zyx)[::-1], dtype=np.int64)
    start1_xyz = np.maximum(1, lo1_xyz - MARGIN_XYZ)
    end1_xyz = np.minimum(shape_xyz, hi1_xyz + MARGIN_XYZ)
    start0_zyx = start1_xyz[[2, 1, 0]] - 1
    end0_zyx = end1_xyz[[2, 1, 0]] - 1
    return np.stack([start0_zyx, end0_zyx], axis=1).astype(np.int64)


def legacy_crop_bounds_zyx_from_landmarks_xyz(
    fixed_xyz: np.ndarray, moving_xyz: np.ndarray, shape_zyx: Sequence[int]
) -> np.ndarray:
    """Historical Python crop bounds; required by frozen COPD results."""
    fixed_zyx = fixed_xyz[:, [2, 1, 0]].astype(np.float64, copy=False)
    moving_zyx = moving_xyz[:, [2, 1, 0]].astype(np.float64, copy=False)
    both = np.concatenate([fixed_zyx, moving_zyx], axis=0)
    lo = np.floor(np.min(both, axis=0)).astype(np.int64)
    hi = np.ceil(np.max(both, axis=0)).astype(np.int64)
    shape = np.asarray(shape_zyx, dtype=np.int64)
    start = np.maximum(0, lo - MARGIN_ZYX)
    end = np.minimum(shape - 1, hi + MARGIN_ZYX)
    return np.stack([start, end], axis=1).astype(np.int64)


def crop_zyx(vol: np.ndarray, bounds_zyx: np.ndarray) -> np.ndarray:
    return vol[
        bounds_zyx[0, 0] : bounds_zyx[0, 1] + 1,
        bounds_zyx[1, 0] : bounds_zyx[1, 1] + 1,
        bounds_zyx[2, 0] : bounds_zyx[2, 1] + 1,
    ]


def resize_image_trilinear_zyx(
    vol: np.ndarray, new_shape_zyx: Sequence[int], device: torch.device
) -> np.ndarray:
    with torch.no_grad():
        ten = torch.from_numpy(vol.astype(np.float32, copy=False))[None, None].to(
            device=device
        )
        out = F.interpolate(
            ten,
            size=tuple((int(v) for v in new_shape_zyx)),
            mode="trilinear",
            align_corners=True,
        )
        return out[0, 0].cpu().numpy().astype(np.float32, copy=False)


def resize_mask_nearest_zyx(
    mask: np.ndarray, new_shape_zyx: Sequence[int], device: torch.device
) -> np.ndarray:
    with torch.no_grad():
        ten = torch.from_numpy(mask.astype(np.float32, copy=False))[None, None].to(
            device=device
        )
        out = F.interpolate(
            ten, size=tuple((int(v) for v in new_shape_zyx)), mode="nearest"
        )
        return (out[0, 0].cpu().numpy() >= 0.5).astype(np.uint8)


def transform_landmarks_native0_to_crop(
    landmarks_native0_zyx: np.ndarray, bounds_zyx: np.ndarray
) -> np.ndarray:
    return landmarks_native0_zyx - bounds_zyx[:, 0][None, :].astype(np.float64)


def transform_landmarks_crop_to_registration(
    landmarks_crop_zyx: np.ndarray,
    crop_shape_zyx: Sequence[int],
    reg_shape_zyx: Sequence[int],
) -> np.ndarray:
    crop_max = np.maximum(np.asarray(crop_shape_zyx, dtype=np.float64) - 1.0, 1.0)
    reg_max = np.maximum(np.asarray(reg_shape_zyx, dtype=np.float64) - 1.0, 1.0)
    return landmarks_crop_zyx * (reg_max[None, :] / crop_max[None, :])


def points_inside_shape(
    points_zyx: np.ndarray, shape_zyx: Sequence[int]
) -> Dict[str, object]:
    pts = np.asarray(points_zyx, dtype=np.float64)
    shape = np.asarray(shape_zyx, dtype=np.float64)
    inside = np.all((pts >= 0.0) & (pts <= shape[None, :] - 1.0), axis=1)
    return {
        "all_inside": bool(np.all(inside)),
        "inside_count": int(np.sum(inside)),
        "total_count": int(inside.size),
        "inside_fraction": float(np.mean(inside)) if inside.size else float("nan"),
        "min_zyx": [float(v) for v in np.min(pts, axis=0)],
        "max_zyx": [float(v) for v in np.max(pts, axis=0)],
        "shape_zyx": [int(v) for v in shape_zyx],
    }
