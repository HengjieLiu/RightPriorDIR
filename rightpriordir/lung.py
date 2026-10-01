"""Audited registration/data kernels; see docs/methods.md for protocol scope."""

from __future__ import annotations
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn.functional as F

FOURDCT_SPACING_ZYX: Dict[int, Tuple[float, float, float]] = {
    1: (2.5, 0.97, 0.97),
    2: (2.5, 1.16, 1.16),
    3: (2.5, 1.15, 1.15),
    4: (2.5, 1.13, 1.13),
    5: (2.5, 1.1, 1.1),
    6: (2.5, 0.97, 0.97),
    7: (2.5, 0.97, 0.97),
    8: (2.5, 0.97, 0.97),
    9: (2.5, 0.97, 0.97),
    10: (2.5, 0.97, 0.97),
}


@dataclass(frozen=True)
class LungCTCase:
    case_id: int
    dataset: str
    case_dir: Path
    fixed_label: str
    moving_label: str
    fixed: np.ndarray
    moving: np.ndarray
    fixed_mask: np.ndarray
    moving_mask: np.ndarray
    landmarks_fixed: np.ndarray
    landmarks_moving: np.ndarray
    landmarks_fixed_crop: np.ndarray
    landmarks_moving_crop: np.ndarray
    native_spacing_zyx: Tuple[float, float, float]
    crop_shape_zyx: Tuple[int, int, int]
    preprocess_audit: Dict[str, object]

    @property
    def shape_zyx(self) -> Tuple[int, int, int]:
        return tuple((int(v) for v in self.fixed.shape))


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key, value in row.items():
            if key not in fieldnames and (not isinstance(value, (list, dict))):
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def parse_int_list(value: str) -> List[int]:
    vals = [int(v) for v in value.replace(" ", "").split(",") if v]
    if not vals:
        raise ValueError(f"Expected comma-separated integers, got {value!r}")
    return vals


def parse_float_list(value: str, n: int, default: float) -> List[float]:
    if not value:
        return [float(default)] * n
    vals = [float(v) for v in value.replace(" ", "").split(",") if v]
    if len(vals) == 1:
        return vals * n
    if len(vals) != n:
        raise ValueError(
            f"Expected one value or {n} values, got {len(vals)} from {value!r}"
        )
    return vals


def parse_stage_ints(value: str, n: int) -> List[int]:
    vals = parse_int_list(value)
    if len(vals) == 1:
        return vals * n
    if len(vals) != n:
        raise ValueError(
            f"Expected one value or {n} values, got {len(vals)} from {value!r}"
        )
    return vals


def case_dir(root: Path, case_id: int, dataset: str = "4dct") -> Path:
    dataset_norm = str(dataset).lower()
    dataset_dir = {"4dct": "4DCT", "copd": "COPD"}.get(dataset_norm)
    if dataset_dir is None:
        raise ValueError(f"Unsupported LungCT dataset: {dataset!r}")
    return Path(root) / dataset_dir / f"case{int(case_id):02d}"


def _case_paths(cdir: Path, dataset: str) -> Tuple[Dict[str, Path], str, str]:
    dataset_norm = str(dataset).lower()
    if dataset_norm == "4dct":
        return (
            {
                "fixed": cdir / "fixed_T00_reg1mm_trilinear_zyx.npy",
                "moving": cdir / "moving_T50_reg1mm_trilinear_zyx.npy",
                "fixed_mask": cdir / "fixed_T00_lungmask_reg1mm_nearest_zyx.npy",
                "moving_mask": cdir / "moving_T50_lungmask_reg1mm_nearest_zyx.npy",
                "landmarks_fixed": cdir / "landmarks_fixed_T00_reg1mm_zyx.npy",
                "landmarks_moving": cdir / "landmarks_moving_T50_reg1mm_zyx.npy",
                "landmarks_fixed_crop": cdir / "landmarks_fixed_T00_crop_zyx.npy",
                "landmarks_moving_crop": cdir / "landmarks_moving_T50_crop_zyx.npy",
                "audit": cdir / "preprocess_audit.json",
            },
            "T00",
            "T50",
        )
    if dataset_norm == "copd":
        return (
            {
                "fixed": cdir / "fixed_iBH_reg1mm_trilinear_zyx.npy",
                "moving": cdir / "moving_eBH_reg1mm_trilinear_zyx.npy",
                "fixed_mask": cdir / "fixed_iBH_lungmask_reg1mm_nearest_zyx.npy",
                "moving_mask": cdir / "moving_eBH_lungmask_reg1mm_nearest_zyx.npy",
                "landmarks_fixed": cdir / "landmarks_fixed_iBH_reg1mm_zyx.npy",
                "landmarks_moving": cdir / "landmarks_moving_eBH_reg1mm_zyx.npy",
                "landmarks_fixed_crop": cdir / "landmarks_fixed_iBH_crop_zyx.npy",
                "landmarks_moving_crop": cdir / "landmarks_moving_eBH_crop_zyx.npy",
                "audit": cdir / "preprocess_audit.json",
            },
            "iBH",
            "eBH",
        )
    raise ValueError(f"Unsupported LungCT dataset: {dataset!r}")


def _audit_verification_pass(audit: Dict[str, object]) -> bool:
    if "verification_pass" in audit:
        return bool(audit.get("verification_pass"))
    verification = audit.get("verification")
    if isinstance(verification, dict) and "pass" in verification:
        return bool(verification.get("pass"))
    return True


def load_case(root: Path, case_id: int, dataset: str = "4dct") -> LungCTCase:
    dataset_norm = str(dataset).lower()
    cdir = case_dir(root, case_id, dataset_norm)
    if not cdir.is_dir():
        raise FileNotFoundError(f"Missing pTV-prep case directory: {cdir}")
    (paths, fixed_label, moving_label) = _case_paths(cdir, dataset_norm)
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing pTV-prep files:\n" + "\n".join(missing))
    fixed = np.load(paths["fixed"]).astype(np.float32, copy=False)
    moving = np.load(paths["moving"]).astype(np.float32, copy=False)
    fixed_mask = (np.load(paths["fixed_mask"]) > 0).astype(np.uint8, copy=False)
    moving_mask = (np.load(paths["moving_mask"]) > 0).astype(np.uint8, copy=False)
    landmarks_fixed = np.load(paths["landmarks_fixed"]).astype(np.float64, copy=False)
    landmarks_moving = np.load(paths["landmarks_moving"]).astype(np.float64, copy=False)
    landmarks_fixed_crop = np.load(paths["landmarks_fixed_crop"]).astype(
        np.float64, copy=False
    )
    landmarks_moving_crop = np.load(paths["landmarks_moving_crop"]).astype(
        np.float64, copy=False
    )
    audit = json.loads(paths["audit"].read_text())
    native_spacing = audit.get("native_spacing_zyx_mm")
    if native_spacing is None and dataset_norm == "4dct":
        native_spacing = FOURDCT_SPACING_ZYX.get(int(case_id))
    if native_spacing is None:
        raise ValueError(
            f"Missing native spacing in preprocess audit for {dataset_norm} case {case_id}"
        )
    native_spacing_zyx = tuple((float(v) for v in native_spacing))
    crop_shape = audit.get("crop_shape_zyx")
    if crop_shape is None:
        crop_shape = tuple(
            (int(v) for v in np.ceil(np.max(landmarks_fixed_crop, axis=0) + 1.0))
        )
    crop_shape_zyx = tuple((int(v) for v in crop_shape))
    if fixed.shape != moving.shape:
        raise ValueError(
            f"Image shape mismatch for case {case_id}: {fixed.shape} vs {moving.shape}"
        )
    if fixed.shape != fixed_mask.shape:
        raise ValueError(
            f"Fixed mask shape mismatch for case {case_id}: {fixed.shape} vs {fixed_mask.shape}"
        )
    if fixed.shape != moving_mask.shape:
        raise ValueError(
            f"Moving mask shape mismatch for case {case_id}: {fixed.shape} vs {moving_mask.shape}"
        )
    if landmarks_fixed.shape != landmarks_moving.shape or landmarks_fixed.shape[1] != 3:
        raise ValueError(
            f"Landmark shape mismatch for case {case_id}: {landmarks_fixed.shape} vs {landmarks_moving.shape}"
        )
    if (
        landmarks_fixed_crop.shape != landmarks_moving_crop.shape
        or landmarks_fixed_crop.shape[1] != 3
    ):
        raise ValueError(
            f"Crop landmark shape mismatch for case {case_id}: {landmarks_fixed_crop.shape} vs {landmarks_moving_crop.shape}"
        )
    if int(fixed_mask.sum()) <= 0:
        raise ValueError(f"Fixed lung mask is empty for case {case_id}")
    assert_landmarks_inside(
        landmarks_fixed, fixed.shape, f"case {case_id} fixed landmarks"
    )
    assert_landmarks_inside(
        landmarks_moving, moving.shape, f"case {case_id} moving landmarks"
    )
    assert_landmarks_inside(
        landmarks_fixed_crop, crop_shape_zyx, f"case {case_id} fixed crop landmarks"
    )
    assert_landmarks_inside(
        landmarks_moving_crop, crop_shape_zyx, f"case {case_id} moving crop landmarks"
    )
    return LungCTCase(
        case_id=int(case_id),
        dataset=dataset_norm,
        case_dir=cdir,
        fixed_label=fixed_label,
        moving_label=moving_label,
        fixed=fixed,
        moving=moving,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        landmarks_fixed=landmarks_fixed,
        landmarks_moving=landmarks_moving,
        landmarks_fixed_crop=landmarks_fixed_crop,
        landmarks_moving_crop=landmarks_moving_crop,
        native_spacing_zyx=native_spacing_zyx,
        crop_shape_zyx=crop_shape_zyx,
        preprocess_audit=audit,
    )


def assert_landmarks_inside(
    points_zyx: np.ndarray, shape_zyx: Sequence[int], label: str
) -> None:
    pts = np.asarray(points_zyx, dtype=np.float64)
    shape = np.asarray(shape_zyx, dtype=np.float64)
    inside = np.all((pts >= 0.0) & (pts <= shape[None, :] - 1.0), axis=1)
    if not bool(np.all(inside)):
        bad = np.where(~inside)[0].tolist()
        raise ValueError(f"{label} outside image bounds at landmark indices {bad[:10]}")


def loader_audit(
    root: Path, case_ids: Sequence[int], dataset: str = "4dct"
) -> Dict[str, object]:
    dataset_norm = str(dataset).lower()
    rows: List[Dict[str, object]] = []
    for case_id in case_ids:
        case = load_case(root, int(case_id), dataset=dataset_norm)
        rows.append(
            {
                "dataset": dataset_norm,
                "case_id": int(case_id),
                "case_dir": str(case.case_dir),
                "shape_zyx": list(case.shape_zyx),
                "crop_shape_zyx": list(case.crop_shape_zyx),
                "native_spacing_zyx_mm": list(case.native_spacing_zyx),
                "fixed_label": case.fixed_label,
                "moving_label": case.moving_label,
                "fixed_mask_voxels": int(case.fixed_mask.sum()),
                "fixed_mask_fraction": float(case.fixed_mask.mean()),
                "moving_mask_voxels": int(case.moving_mask.sum()),
                "n_landmarks": int(case.landmarks_fixed.shape[0]),
                "identity_tre_mean_mm": compute_tre_mm(
                    case.landmarks_fixed, case.landmarks_moving, (1.0, 1.0, 1.0)
                )["tre_mean_mm"],
                "native_crop_identity_tre_mean_mm": compute_tre_mm(
                    case.landmarks_fixed_crop,
                    case.landmarks_moving_crop,
                    case.native_spacing_zyx,
                )["tre_mean_mm"],
                "verification_pass": _audit_verification_pass(case.preprocess_audit),
            }
        )
    return {
        "dataset": dataset_norm,
        "root": str(root),
        "case_ids": [int(v) for v in case_ids],
        "n_cases": len(rows),
        "all_cases_loaded": True,
        "all_verification_pass": all((bool(row["verification_pass"]) for row in rows)),
        "rows": rows,
    }


def set_cuda_device(
    gpu: str, *, cudnn_benchmark: bool = False, cudnn_deterministic: bool = True
) -> torch.device:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    torch.backends.cudnn.deterministic = bool(cudnn_deterministic)
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this runner.")
    torch.cuda.set_device(0)
    return torch.device("cuda")


def make_coordinate_tensor(shape: Sequence[int], device: torch.device) -> torch.Tensor:
    vectors = [torch.linspace(-1.0, 1.0, int(s), device=device) for s in shape]
    (zz, yy, xx) = torch.meshgrid(*vectors, indexing="ij")
    return torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3)


def make_coordinate_grid(shape: Sequence[int], device: torch.device) -> torch.Tensor:
    return make_coordinate_tensor(shape, device).reshape(
        *tuple((int(v) for v in shape)), 3
    )


def sample_volume(
    volume: torch.Tensor, coords: torch.Tensor, mode: str = "bilinear"
) -> torch.Tensor:
    sampled = F.grid_sample(
        volume[None, None], coords[None, None, None], mode=mode, align_corners=True
    )
    return sampled.squeeze()


def warp_full(
    volume: torch.Tensor, dense_coords: torch.Tensor, mode: str = "bilinear"
) -> torch.Tensor:
    warped = F.grid_sample(
        volume[None, None], dense_coords[None], mode=mode, align_corners=True
    )
    return warped.squeeze()


class StableStd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor: torch.Tensor) -> torch.Tensor:
        ctx.tensor = tensor.detach()
        result = torch.std(tensor).detach()
        ctx.result = result.detach()
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor]:
        tensor = ctx.tensor.detach()
        result = ctx.result.detach()
        grad = (
            2.0
            / (tensor.numel() - 1.0)
            * (grad_output.detach() / (result.detach() * 2.0 + 1e-06))
            * (tensor.detach() - tensor.mean().detach())
        )
        return (grad,)


def ncc_loss(x1: torch.Tensor, x2: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    if x1.shape != x2.shape:
        raise ValueError(f"NCC shape mismatch: {tuple(x1.shape)} vs {tuple(x2.shape)}")
    cc = ((x1 - x1.mean()) * (x2 - x2.mean())).mean()
    std = StableStd.apply(x1) * StableStd.apply(x2)
    return -(cc / (std + eps)).mean()


def build_siren(hidden_dim, hidden_layers, omega, input_dim=3, *, upstream="IDIR"):
    from .upstream import siren

    return siren(
        upstream,
        [int(input_dim)] + [int(hidden_dim)] * int(hidden_layers) + [3],
        float(omega),
    )


def coordinate_gradient(
    input_coords: torch.Tensor, output: torch.Tensor, *, create_graph: bool = True
) -> torch.Tensor:
    return torch.autograd.grad(
        output,
        [input_coords],
        grad_outputs=torch.ones_like(output),
        create_graph=create_graph,
        retain_graph=True,
        only_inputs=True,
    )[0]


def idir_jacobian_matrix(
    input_coords: torch.Tensor, output: torch.Tensor
) -> torch.Tensor:
    jac = torch.zeros(
        input_coords.shape[0], 3, 3, dtype=output.dtype, device=output.device
    )
    for i in range(3):
        jac[:, i, :] = coordinate_gradient(
            input_coords, output[:, i], create_graph=True
        )
    return jac


def idir_bending_energy(
    input_coords: torch.Tensor, output_rel: torch.Tensor, batch_size: int
) -> torch.Tensor:
    jac = idir_jacobian_matrix(input_coords, output_rel)
    dx_xyz = torch.zeros(
        input_coords.shape[0], 3, 3, dtype=output_rel.dtype, device=output_rel.device
    )
    dy_xyz = torch.zeros_like(dx_xyz)
    dz_xyz = torch.zeros_like(dx_xyz)
    for i in range(3):
        dx_xyz[:, i, :] = coordinate_gradient(
            input_coords, jac[:, i, 0], create_graph=True
        )
        dy_xyz[:, i, :] = coordinate_gradient(
            input_coords, jac[:, i, 1], create_graph=True
        )
        dz_xyz[:, i, :] = coordinate_gradient(
            input_coords, jac[:, i, 2], create_graph=True
        )
    dx_xyz = dx_xyz.square()
    dy_xyz = dy_xyz.square()
    dz_xyz = dz_xyz.square()
    loss = (
        torch.mean(dx_xyz[:, :, 0])
        + torch.mean(dy_xyz[:, :, 1])
        + torch.mean(dz_xyz[:, :, 2])
    )
    loss = loss + 2.0 * torch.mean(dx_xyz[:, :, 1]) + 2.0 * torch.mean(dx_xyz[:, :, 2])
    loss = loss + torch.mean(dy_xyz[:, :, 2])
    return loss / float(batch_size)


def cubic_bspline_value(x: float, derivative: int = 0) -> float:
    t = abs(float(x))
    if derivative == 0:
        if t < 1:
            return 2.0 / 3.0 + 0.5 * t**3 - t**2
        if t < 2:
            return (2.0 - t) ** 3 / 6.0
        return 0.0
    if derivative == 1:
        if t < 1:
            return (1.5 * t - 2.0) * x
        if x < 0:
            return 0.5 * (t - 2.0) ** 2
        return -0.5 * (t - 2.0) ** 2
    if derivative == 2:
        if t < 1:
            return 3.0 * t - 2.0
        if t < 2:
            return -t + 2.0
        return 0.0
    raise ValueError(f"Unsupported derivative: {derivative}")


def cubic_bspline_value_tensor(x: torch.Tensor, derivative: int = 0) -> torch.Tensor:
    t = x.abs()
    zero = torch.zeros_like(x)
    if derivative == 0:
        inner = 2.0 / 3.0 + 0.5 * t**3 - t**2
        outer = (2.0 - t).clamp_min(0.0) ** 3 / 6.0
        return torch.where(t < 1.0, inner, torch.where(t < 2.0, outer, zero))
    if derivative == 1:
        inner = (1.5 * t - 2.0) * x
        outer = torch.where(x < 0.0, 0.5 * (t - 2.0) ** 2, -0.5 * (t - 2.0) ** 2)
        return torch.where(t < 1.0, inner, torch.where(t < 2.0, outer, zero))
    if derivative == 2:
        inner = 3.0 * t - 2.0
        outer = -t + 2.0
        return torch.where(t < 1.0, inner, torch.where(t < 2.0, outer, zero))
    raise ValueError(f"Unsupported derivative: {derivative}")


def cubic_bspline1d(
    stride: int, derivative: int = 0, dtype=None, device=None
) -> torch.Tensor:
    if dtype is None:
        dtype = torch.float32
    kernel = torch.ones(4 * int(stride) - 1, dtype=dtype)
    radius = kernel.shape[0] // 2
    for i in range(kernel.shape[0]):
        kernel[i] = cubic_bspline_value(
            (i - radius) / float(stride), derivative=derivative
        )
    return kernel.to(device=device)


def conv1d(
    data: torch.Tensor,
    *,
    kernel: torch.Tensor,
    dim: int,
    stride: int = 1,
    padding: int = 0,
    transpose: bool = False,
) -> torch.Tensor:
    result = data.type(kernel.dtype)
    result = result.transpose(dim, -1)
    shape = result.size()
    groups = int(np.prod(shape[1:-1]))
    weight = kernel.expand(groups, 1, kernel.shape[-1])
    result = result.reshape(shape[0], groups, shape[-1])
    conv_fn = F.conv_transpose1d if transpose else F.conv1d
    result = conv_fn(result, weight, stride=stride, padding=padding, groups=groups)
    result = result.reshape(shape[0:-1] + result.shape[-1:])
    return result.transpose(-1, dim)


def bspline_control_coords(
    image_shape: Sequence[int], cps: int, device: torch.device
) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    kernel_len = 4 * int(cps) - 1
    pad = (kernel_len - 1) // 2
    padded_shape = tuple((int(s) + 2 * pad for s in image_shape))
    start = int(cps) // 2
    full_coords = make_coordinate_grid(padded_shape, device)
    indices = [
        torch.arange(start, s, step=int(cps), device=device) for s in padded_shape
    ]
    control_grid = full_coords[indices[0]][:, indices[1]][:, :, indices[2]]
    control_shape = tuple((int(v) for v in control_grid.shape[:3]))
    return (control_grid.reshape(-1, 3), control_shape)


def expand_bspline_controls(
    control_values: torch.Tensor,
    control_shape: Sequence[int],
    image_shape: Sequence[int],
    cps: int,
) -> torch.Tensor:
    image_shape = tuple((int(v) for v in image_shape))
    field = (
        control_values.reshape(*tuple((int(v) for v in control_shape)), 3)
        .permute(3, 0, 1, 2)
        .unsqueeze(0)
    )
    for dim in range(3):
        kernel = cubic_bspline1d(cps, dtype=field.dtype, device=field.device)
        padding = (kernel.shape[0] - 1) // 2
        field = conv1d(
            field,
            dim=dim + 2,
            kernel=kernel,
            stride=int(cps),
            padding=padding,
            transpose=True,
        )
    slicer = (slice(None), slice(None)) + tuple(
        (slice(cps, cps + image_shape[i]) for i in range(3))
    )
    field = field[slicer]
    return field.squeeze(0).permute(1, 2, 3, 0).contiguous()


def expand_cubic_bspline_derivative(
    control_values: torch.Tensor,
    control_shape: Sequence[int],
    image_shape: Sequence[int],
    cps: int,
    derivative_orders: Sequence[int],
) -> torch.Tensor:
    image_shape = tuple((int(v) for v in image_shape))
    field = (
        control_values.reshape(*tuple((int(v) for v in control_shape)), 3)
        .permute(3, 0, 1, 2)
        .unsqueeze(0)
    )
    for dim, order in enumerate((int(v) for v in derivative_orders)):
        kernel = cubic_bspline1d(
            cps, derivative=order, dtype=field.dtype, device=field.device
        )
        if order:
            kernel = kernel * (1.0 / float(cps)) ** int(order)
        padding = (kernel.shape[0] - 1) // 2
        field = conv_transpose3d_axis(
            field, kernel=kernel, dim=dim, stride=int(cps), padding=padding
        )
    slicer = (slice(None), slice(None)) + tuple(
        (slice(cps, cps + image_shape[i]) for i in range(3))
    )
    return field[slicer].contiguous()


def conv_transpose3d_axis(
    field: torch.Tensor, *, kernel: torch.Tensor, dim: int, stride: int, padding: int
) -> torch.Tensor:
    if field.ndim != 5:
        raise ValueError(f"Expected [N,C,D,H,W], got {tuple(field.shape)}")
    channels = int(field.shape[1])
    if dim == 0:
        weight = kernel.view(1, 1, -1, 1, 1).repeat(channels, 1, 1, 1, 1)
        stride_3d = (int(stride), 1, 1)
        padding_3d = (int(padding), 0, 0)
    elif dim == 1:
        weight = kernel.view(1, 1, 1, -1, 1).repeat(channels, 1, 1, 1, 1)
        stride_3d = (1, int(stride), 1)
        padding_3d = (0, int(padding), 0)
    elif dim == 2:
        weight = kernel.view(1, 1, 1, 1, -1).repeat(channels, 1, 1, 1, 1)
        stride_3d = (1, 1, int(stride))
        padding_3d = (0, 0, int(padding))
    else:
        raise ValueError(f"Unsupported dim={dim}")
    return F.conv_transpose3d(
        field, weight, stride=stride_3d, padding=padding_3d, groups=channels
    )


def _density_from_terms(
    terms: Sequence[Tuple[torch.Tensor, float]], component_reduction: str
) -> torch.Tensor:
    density: Optional[torch.Tensor] = None
    for term, weight in terms:
        sq = float(weight) * term.square()
        reduced = sq.sum(dim=1) if component_reduction == "sum" else sq.mean(dim=1)
        density = reduced if density is None else density + reduced
    if density is None:
        raise ValueError("No regularity terms were provided")
    return density


def _masked_density_mean(
    density: torch.Tensor, mask: Optional[torch.Tensor]
) -> torch.Tensor:
    if mask is None:
        return density.mean()
    mask_t = mask.to(device=density.device).bool()
    selected = density[..., mask_t]
    if selected.numel() == 0:
        return torch.as_tensor(float("nan"), dtype=density.dtype, device=density.device)
    return selected.mean()


def analytic_bspline_sampled_be(
    control_values: torch.Tensor,
    control_shape: Sequence[int],
    image_shape: Sequence[int],
    cps: int,
    mask: Optional[torch.Tensor] = None,
    *,
    component_reduction: str = "mean",
) -> torch.Tensor:
    terms = [
        (
            expand_cubic_bspline_derivative(
                control_values, control_shape, image_shape, cps, (2, 0, 0)
            ),
            1.0,
        ),
        (
            expand_cubic_bspline_derivative(
                control_values, control_shape, image_shape, cps, (0, 2, 0)
            ),
            1.0,
        ),
        (
            expand_cubic_bspline_derivative(
                control_values, control_shape, image_shape, cps, (0, 0, 2)
            ),
            1.0,
        ),
        (
            expand_cubic_bspline_derivative(
                control_values, control_shape, image_shape, cps, (1, 1, 0)
            ),
            2.0,
        ),
        (
            expand_cubic_bspline_derivative(
                control_values, control_shape, image_shape, cps, (1, 0, 1)
            ),
            2.0,
        ),
        (
            expand_cubic_bspline_derivative(
                control_values, control_shape, image_shape, cps, (0, 1, 1)
            ),
            2.0,
        ),
    ]
    density = _density_from_terms(terms, component_reduction=component_reduction)
    return _masked_density_mean(density, mask)


def flat_indices_to_zyx(
    flat_indices: torch.Tensor, image_shape: Sequence[int]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if flat_indices.ndim != 1:
        raise ValueError(
            f"Expected 1D flat indices, got shape {tuple(flat_indices.shape)}"
        )
    (d, h, w) = (int(v) for v in image_shape)
    z = torch.div(flat_indices, h * w, rounding_mode="floor")
    rem = flat_indices - z * h * w
    y = torch.div(rem, w, rounding_mode="floor")
    x = rem - y * w
    return (z, y, x)


def sample_cubic_bspline_derivative(
    control_values: torch.Tensor,
    control_shape: Sequence[int],
    image_shape: Sequence[int],
    cps: int,
    flat_indices: torch.Tensor,
    derivative_orders: Sequence[int],
) -> torch.Tensor:
    """Evaluate a cubic B-spline derivative at selected voxel-center indices."""
    if len(tuple(derivative_orders)) != 3:
        raise ValueError(f"Expected three derivative orders, got {derivative_orders}")
    control_grid = flat_to_grid(control_values, control_shape)
    (cd, ch, cw) = (int(v) for v in control_shape)
    (z, y, x) = flat_indices_to_zyx(
        flat_indices.to(device=control_values.device, dtype=torch.long), image_shape
    )
    coords = [
        z.to(dtype=control_values.dtype) / float(cps) + 1.0,
        y.to(dtype=control_values.dtype) / float(cps) + 1.0,
        x.to(dtype=control_values.dtype) / float(cps) + 1.0,
    ]
    bases = [torch.floor(coord).to(dtype=torch.long) - 1 for coord in coords]
    sizes = (cd, ch, cw)
    values = torch.zeros(
        (int(flat_indices.numel()), 3),
        dtype=control_values.dtype,
        device=control_values.device,
    )
    for oz in range(4):
        kz = bases[0] + int(oz)
        wz = cubic_bspline_value_tensor(
            coords[0] - kz.to(dtype=control_values.dtype), int(derivative_orders[0])
        )
        if int(derivative_orders[0]):
            wz = wz * (1.0 / float(cps)) ** int(derivative_orders[0])
        vz = (kz >= 0) & (kz < sizes[0])
        kz_clamped = kz.clamp(0, sizes[0] - 1)
        for oy in range(4):
            ky = bases[1] + int(oy)
            wy = cubic_bspline_value_tensor(
                coords[1] - ky.to(dtype=control_values.dtype), int(derivative_orders[1])
            )
            if int(derivative_orders[1]):
                wy = wy * (1.0 / float(cps)) ** int(derivative_orders[1])
            vy = (ky >= 0) & (ky < sizes[1])
            ky_clamped = ky.clamp(0, sizes[1] - 1)
            for ox in range(4):
                kx = bases[2] + int(ox)
                wx = cubic_bspline_value_tensor(
                    coords[2] - kx.to(dtype=control_values.dtype),
                    int(derivative_orders[2]),
                )
                if int(derivative_orders[2]):
                    wx = wx * (1.0 / float(cps)) ** int(derivative_orders[2])
                vx = (kx >= 0) & (kx < sizes[2])
                kx_clamped = kx.clamp(0, sizes[2] - 1)
                weight = wz * wy * wx
                valid = (vz & vy & vx).to(dtype=weight.dtype)
                values = values + control_grid[kz_clamped, ky_clamped, kx_clamped] * (
                    weight * valid
                ).unsqueeze(-1)
    return values


def sampled_cubic_bspline_be(
    control_values: torch.Tensor,
    control_shape: Sequence[int],
    image_shape: Sequence[int],
    cps: int,
    flat_indices: torch.Tensor,
    *,
    component_reduction: str = "mean",
) -> torch.Tensor:
    terms = [
        (
            sample_cubic_bspline_derivative(
                control_values, control_shape, image_shape, cps, flat_indices, (2, 0, 0)
            ),
            1.0,
        ),
        (
            sample_cubic_bspline_derivative(
                control_values, control_shape, image_shape, cps, flat_indices, (0, 2, 0)
            ),
            1.0,
        ),
        (
            sample_cubic_bspline_derivative(
                control_values, control_shape, image_shape, cps, flat_indices, (0, 0, 2)
            ),
            1.0,
        ),
        (
            sample_cubic_bspline_derivative(
                control_values, control_shape, image_shape, cps, flat_indices, (1, 1, 0)
            ),
            2.0,
        ),
        (
            sample_cubic_bspline_derivative(
                control_values, control_shape, image_shape, cps, flat_indices, (1, 0, 1)
            ),
            2.0,
        ),
        (
            sample_cubic_bspline_derivative(
                control_values, control_shape, image_shape, cps, flat_indices, (0, 1, 1)
            ),
            2.0,
        ),
    ]
    density = _density_from_terms(terms, component_reduction=component_reduction)
    return density.mean()


def analytic_bspline_sampled_diffusion(
    control_values: torch.Tensor,
    control_shape: Sequence[int],
    image_shape: Sequence[int],
    cps: int,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    terms = [
        expand_cubic_bspline_derivative(
            control_values, control_shape, image_shape, cps, (1, 0, 0)
        ),
        expand_cubic_bspline_derivative(
            control_values, control_shape, image_shape, cps, (0, 1, 0)
        ),
        expand_cubic_bspline_derivative(
            control_values, control_shape, image_shape, cps, (0, 0, 1)
        ),
    ]
    density = sum((term.square().mean(dim=1) for term in terms)) / 3.0
    return _masked_density_mean(density, mask)


def dense_forward_difference_be(
    disp_dhw3: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    component_reduction: str = "mean",
) -> torch.Tensor:
    if disp_dhw3.ndim != 4 or int(disp_dhw3.shape[-1]) != 3:
        raise ValueError(
            f"Expected [D,H,W,3] displacement, got {tuple(disp_dhw3.shape)}"
        )
    disp = disp_dhw3.permute(3, 0, 1, 2).unsqueeze(0).contiguous()
    dz = disp[:, :, 1:, :, :] - disp[:, :, :-1, :, :]
    dy = disp[:, :, :, 1:, :] - disp[:, :, :, :-1, :]
    dx = disp[:, :, :, :, 1:] - disp[:, :, :, :, :-1]
    terms = [
        (dz[:, :, 1:, :, :] - dz[:, :, :-1, :, :], 1.0),
        (dy[:, :, :, 1:, :] - dy[:, :, :, :-1, :], 1.0),
        (dx[:, :, :, :, 1:] - dx[:, :, :, :, :-1], 1.0),
        (dz[:, :, :, 1:, :] - dz[:, :, :, :-1, :], 2.0),
        (dz[:, :, :, :, 1:] - dz[:, :, :, :, :-1], 2.0),
        (dy[:, :, :, :, 1:] - dy[:, :, :, :, :-1], 2.0),
    ]
    if mask is None:
        masks: List[Optional[torch.Tensor]] = [None for (_term, _weight) in terms]
    else:
        m = mask.to(device=disp_dhw3.device).bool()
        if m.ndim != 3:
            raise ValueError(f"Expected [D,H,W] mask, got {tuple(m.shape)}")
        masks = [
            m[2:, :, :] & m[1:-1, :, :] & m[:-2, :, :],
            m[:, 2:, :] & m[:, 1:-1, :] & m[:, :-2, :],
            m[:, :, 2:] & m[:, :, 1:-1] & m[:, :, :-2],
            m[1:, 1:, :] & m[:-1, 1:, :] & m[1:, :-1, :] & m[:-1, :-1, :],
            m[1:, :, 1:] & m[:-1, :, 1:] & m[1:, :, :-1] & m[:-1, :, :-1],
            m[:, 1:, 1:] & m[:, :-1, 1:] & m[:, 1:, :-1] & m[:, :-1, :-1],
        ]
    loss = torch.zeros((), dtype=disp_dhw3.dtype, device=disp_dhw3.device)
    for (term, weight), term_mask in zip(terms, masks):
        sq = float(weight) * term.square()
        reduced = sq.sum(dim=1) if component_reduction == "sum" else sq.mean(dim=1)
        if term_mask is None:
            loss = loss + reduced.mean()
        else:
            selected = reduced[..., term_mask]
            if selected.numel() == 0:
                loss = loss + torch.as_tensor(
                    float("nan"), dtype=reduced.dtype, device=reduced.device
                )
            else:
                loss = loss + selected.mean()
    return loss


CP_MARGIN = 4


def build_nested_cp_shapes(
    image_shape: Sequence[int],
    base_spacing: int,
    spacings: Iterable[int],
    cp_margin: int = CP_MARGIN,
) -> Dict[int, Tuple[int, int, int]]:
    q_base = [
        int(math.ceil(int(length) / float(base_spacing))) for length in image_shape
    ]
    shapes: Dict[int, Tuple[int, int, int]] = {}
    for spacing in spacings:
        if int(base_spacing) % int(spacing) != 0:
            raise ValueError(
                f"spacing={spacing} must divide base_spacing={base_spacing}"
            )
        ratio = int(base_spacing) // int(spacing)
        shapes[int(spacing)] = tuple((int(ratio * q + cp_margin) for q in q_base))
    return shapes


def assert_nested_schedule(
    shapes: Dict[int, Tuple[int, int, int]], spacings: Sequence[int]
) -> None:
    for coarse, fine in zip(spacings, spacings[1:]):
        if coarse != 2 * fine:
            raise ValueError(
                f"Only dyadic consecutive spacing is supported, got {coarse}->{fine}"
            )
        expected = tuple(
            (int(2 * (int(n) - CP_MARGIN) + CP_MARGIN) for n in shapes[coarse])
        )
        if tuple(shapes[fine]) != expected:
            raise ValueError(
                f"Non-nested CP grid: {coarse}->{fine}, observed={shapes[fine]}, expected={expected}"
            )


def refine_axis_cubic_dyadic_crop_aligned(
    control_grid: torch.Tensor, axis: int
) -> torch.Tensor:
    moved = control_grid.movedim(axis, 0)
    if moved.shape[0] < 3:
        raise ValueError(
            f"Need at least 3 coefficients along axis {axis}, got {moved.shape[0]}"
        )
    even = 0.5 * (moved[:-2] + moved[1:-1])
    odd = (moved[:-2] + 6.0 * moved[1:-1] + moved[2:]) / 8.0
    out = moved.new_empty((2 * moved.shape[0] - 4, *moved.shape[1:]))
    out[0::2] = even
    out[1::2] = odd
    return out.movedim(0, axis)


def cubic_bspline_dyadic_knot_insert_3d(
    coarse_grid: torch.Tensor, target_shape: Sequence[int]
) -> torch.Tensor:
    target = tuple((int(v) for v in target_shape))
    expected = tuple((int(2 * s - CP_MARGIN) for s in coarse_grid.shape[:3]))
    if target != expected:
        raise ValueError(
            f"Target shape {target} is not nested from {tuple(coarse_grid.shape[:3])}; expected {expected}"
        )
    refined = coarse_grid
    for axis in range(3):
        refined = refine_axis_cubic_dyadic_crop_aligned(refined, axis=axis)
    return refined


def resize_control_grid(
    coarse_grid: torch.Tensor, target_shape: Sequence[int]
) -> torch.Tensor:
    grid_5d = coarse_grid.permute(3, 0, 1, 2).unsqueeze(0)
    resized = F.interpolate(
        grid_5d,
        size=tuple((int(v) for v in target_shape)),
        mode="trilinear",
        align_corners=True,
    )
    return resized.squeeze(0).permute(1, 2, 3, 0)


def flat_to_grid(
    control_flat: torch.Tensor, control_shape: Sequence[int]
) -> torch.Tensor:
    return control_flat.reshape(*tuple((int(v) for v in control_shape)), 3)


def grid_to_flat(control_grid: torch.Tensor) -> torch.Tensor:
    return control_grid.reshape(-1, int(control_grid.shape[-1]))


def field_difference_metrics(diff: torch.Tensor, prefix: str = "") -> Dict[str, float]:
    mag = torch.linalg.norm(diff, dim=-1)
    flat = mag.reshape(-1)
    return {
        f"{prefix}mean_abs": float(diff.abs().mean().detach().cpu()),
        f"{prefix}max_abs": float(diff.abs().max().detach().cpu()),
        f"{prefix}mean_mag": float(mag.mean().detach().cpu()),
        f"{prefix}p95_mag": float(torch.quantile(flat, 0.95).detach().cpu()),
        f"{prefix}max_mag": float(mag.max().detach().cpu()),
    }


def gaussian_kernel1d(
    sigma: float, radius_sigma: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    radius = max(1, int(math.ceil(radius_sigma * sigma)))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
    return kernel / kernel.sum()


def gaussian_blur3d(
    volume: torch.Tensor, sigma: float, radius_sigma: float
) -> Tuple[torch.Tensor, int]:
    if sigma <= 0.0:
        return (volume, 1)
    kernel = gaussian_kernel1d(sigma, radius_sigma, volume.device, volume.dtype)
    pad = int(kernel.numel() // 2)
    vol = volume[None, None]
    kx = kernel.view(1, 1, 1, 1, -1)
    ky = kernel.view(1, 1, 1, -1, 1)
    kz = kernel.view(1, 1, -1, 1, 1)
    vol = F.conv3d(F.pad(vol, (pad, pad, 0, 0, 0, 0), mode="replicate"), kx)
    vol = F.conv3d(F.pad(vol, (0, 0, pad, pad, 0, 0), mode="replicate"), ky)
    vol = F.conv3d(F.pad(vol, (0, 0, 0, 0, pad, pad), mode="replicate"), kz)
    return (vol.squeeze(0).squeeze(0), int(kernel.numel()))


def downsample_shape(image_shape: Sequence[int], factor: int) -> Tuple[int, int, int]:
    return tuple((max(2, int(math.ceil(int(s) / float(factor)))) for s in image_shape))


def make_loss_volume(
    volume: torch.Tensor, factor: int, sigma_scale: float, radius_sigma: float
) -> Tuple[torch.Tensor, float, int]:
    if factor <= 1:
        return (volume, 0.0, 1)
    sigma = float(sigma_scale) * float(factor)
    (blurred, kernel_size) = gaussian_blur3d(
        volume, sigma=sigma, radius_sigma=radius_sigma
    )
    target_shape = downsample_shape(tuple((int(v) for v in volume.shape)), factor)
    down = F.interpolate(
        blurred[None, None], size=target_shape, mode="trilinear", align_corners=True
    )
    return (down.squeeze(0).squeeze(0), sigma, kernel_size)


def make_border_mask(
    shape: Sequence[int], border: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    mask = torch.ones(tuple((int(v) for v in shape)), device=device, dtype=dtype)
    if border <= 0:
        return mask
    for axis, size in enumerate(mask.shape):
        if 2 * border >= int(size):
            raise ValueError(
                f"border_mask={border} is too large for axis {axis} with size {size}"
            )
    mask[:border, :, :] = 0
    mask[-border:, :, :] = 0
    mask[:, :border, :] = 0
    mask[:, -border:, :] = 0
    mask[:, :, :border] = 0
    mask[:, :, -border:] = 0
    return mask


def make_loss_mask(
    full_mask: Optional[torch.Tensor],
    factor: int,
    sigma_scale: float,
    radius_sigma: float,
) -> Optional[torch.Tensor]:
    if full_mask is None:
        return None
    if factor <= 1:
        return full_mask
    (mask_l, _sigma, _kernel) = make_loss_volume(
        full_mask, factor, sigma_scale, radius_sigma
    )
    return mask_l.clamp(0.0, 1.0)


def make_nearest_loss_mask(
    full_mask: torch.Tensor, eval_shape: Sequence[int]
) -> torch.Tensor:
    target_shape = tuple((int(v) for v in eval_shape))
    if tuple((int(v) for v in full_mask.shape)) == target_shape:
        return full_mask.bool()
    mask_l = F.interpolate(
        full_mask[None, None].float(), size=target_shape, mode="nearest"
    )
    return mask_l.squeeze(0).squeeze(0).bool()


def stage_pix_zyx(
    reg_pix_zyx: Sequence[float], full_shape: Sequence[int], eval_shape: Sequence[int]
) -> Tuple[float, float, float]:
    pix = []
    for spacing, full, cur in zip(reg_pix_zyx, full_shape, eval_shape):
        if int(cur) <= 1:
            pix.append(float(spacing))
        else:
            pix.append(float(spacing) * float(int(full) - 1) / float(int(cur) - 1))
    return tuple(pix)


def gaussian_blur3d_aniso_channels(
    volumes: torch.Tensor, sigma_zyx: Sequence[float], radius_sigma: float
) -> torch.Tensor:
    if volumes.ndim == 3:
        volumes = volumes.unsqueeze(0)
        squeeze = True
    elif volumes.ndim == 4:
        squeeze = False
    else:
        raise ValueError(
            f"Expected [D,H,W] or [C,D,H,W], got shape {tuple(volumes.shape)}"
        )
    channels = int(volumes.shape[0])
    vol = volumes.unsqueeze(0)
    for axis, sigma in enumerate((float(v) for v in sigma_zyx)):
        if sigma <= 0.0:
            continue
        kernel = gaussian_kernel1d(sigma, radius_sigma, volumes.device, volumes.dtype)
        pad = int(kernel.numel() // 2)
        if axis == 0:
            k = kernel.view(1, 1, -1, 1, 1).repeat(channels, 1, 1, 1, 1)
            padding = (0, 0, 0, 0, pad, pad)
        elif axis == 1:
            k = kernel.view(1, 1, 1, -1, 1).repeat(channels, 1, 1, 1, 1)
            padding = (0, 0, pad, pad, 0, 0)
        else:
            k = kernel.view(1, 1, 1, 1, -1).repeat(channels, 1, 1, 1, 1)
            padding = (pad, pad, 0, 0, 0, 0)
        vol = F.conv3d(F.pad(vol, padding, mode="replicate"), k, groups=channels)
    out = vol.squeeze(0)
    return out.squeeze(0) if squeeze else out


def ptv_lcc_sigma_zyx(
    cur_pix_zyx: Sequence[float], sigma_mm: float, sigma_floor_pix: float
) -> List[float]:
    return [
        max(float(sigma_mm) / float(pix), float(sigma_floor_pix)) for pix in cur_pix_zyx
    ]


def make_ptv_lcc_fixed_cache(
    fixed: torch.Tensor,
    cur_pix_zyx: Sequence[float],
    sigma_mm: float,
    radius_sigma: float,
    sigma_floor_pix: float,
    deps: float = 0.0001,
) -> Dict[str, torch.Tensor | List[float]]:
    sigma_zyx = ptv_lcc_sigma_zyx(cur_pix_zyx, sigma_mm, sigma_floor_pix)
    blurred = gaussian_blur3d_aniso_channels(
        torch.stack((fixed, fixed.square())), sigma_zyx, radius_sigma
    )
    mean_f = blurred[0]
    std_f = torch.sqrt((blurred[1] - mean_f.square()).clamp_min(0.0) + deps)
    return {"fixed": fixed, "mean_f": mean_f, "std_f": std_f, "sigma_zyx": sigma_zyx}


def ptv_local_cc_loss_with_cache(
    warped: torch.Tensor,
    mask: Optional[torch.Tensor],
    cache: Dict[str, torch.Tensor | List[float]],
    cur_pix_zyx: Sequence[float],
    radius_sigma: float,
    deps: float = 0.0001,
) -> torch.Tensor:
    fixed = cache["fixed"]
    mean_f = cache["mean_f"]
    std_f = cache["std_f"]
    sigma_zyx = cache["sigma_zyx"]
    assert isinstance(fixed, torch.Tensor)
    assert isinstance(mean_f, torch.Tensor)
    assert isinstance(std_f, torch.Tensor)
    assert isinstance(sigma_zyx, list)
    blurred = gaussian_blur3d_aniso_channels(
        torch.stack((warped, warped.square(), fixed * warped)), sigma_zyx, radius_sigma
    )
    mean_w = blurred[0]
    std_w = torch.sqrt((blurred[1] - mean_w.square()).clamp_min(0.0) + deps)
    cross = blurred[2] - mean_f * mean_w
    lcc = cross / (std_f * std_w).clamp_min(1e-08)
    loss_map = 1.0 - lcc
    if mask is not None:
        loss_map = loss_map * mask.to(dtype=loss_map.dtype, device=loss_map.device)
    dvol = float(cur_pix_zyx[0]) * float(cur_pix_zyx[1]) * float(cur_pix_zyx[2])
    return loss_map.sum() * dvol


def make_sampled_lcc_offsets(
    image_shape: Sequence[int],
    cur_pix_zyx: Sequence[float],
    sigma_mm: float,
    radius_sigma: float,
    sigma_floor_pix: float,
    max_offsets: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    sigma_zyx = ptv_lcc_sigma_zyx(cur_pix_zyx, sigma_mm, sigma_floor_pix)
    ranges = [
        torch.arange(
            -int(math.ceil(float(radius_sigma) * float(sigma))),
            int(math.ceil(float(radius_sigma) * float(sigma))) + 1,
            device=device,
            dtype=dtype,
        )
        for sigma in sigma_zyx
    ]
    (zz, yy, xx) = torch.meshgrid(*ranges, indexing="ij")
    offsets_zyx = torch.stack((zz, yy, xx), dim=-1).reshape(-1, 3)
    sigma_t = offsets_zyx.new_tensor(sigma_zyx).view(1, 3)
    weights = torch.exp(-0.5 * (offsets_zyx / sigma_t).square().sum(dim=1))
    if int(max_offsets) > 0 and int(max_offsets) < int(weights.numel()):
        keep = torch.topk(
            weights, k=int(max_offsets), largest=True, sorted=False
        ).indices
        offsets_zyx = offsets_zyx[keep]
        weights = weights[keep]
    weights = weights / weights.sum().clamp_min(1e-12)
    shape = [max(int(v) - 1, 1) for v in image_shape]
    offsets_norm_xyz = torch.stack(
        (
            2.0 * offsets_zyx[:, 2] / float(shape[2]),
            2.0 * offsets_zyx[:, 1] / float(shape[1]),
            2.0 * offsets_zyx[:, 0] / float(shape[0]),
        ),
        dim=1,
    )
    meta = {
        "sigma_zyx_pix": [float(v) for v in sigma_zyx],
        "candidate_offsets": int(zz.numel()),
        "selected_offsets": int(offsets_zyx.shape[0]),
        "max_abs_offset_zyx": [
            float(v)
            for v in offsets_zyx.abs().max(dim=0).values.detach().cpu().tolist()
        ],
        "offset_weight_min": float(weights.min().detach().cpu()),
        "offset_weight_max": float(weights.max().detach().cpu()),
    }
    return (offsets_norm_xyz.to(dtype=dtype), weights.to(dtype=dtype), meta)


def sampled_ptv_local_cc_loss(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    patch_coords: torch.Tensor,
    patch_disp: torch.Tensor,
    weights: torch.Tensor,
    domain_voxels: int,
    cur_pix_zyx: Sequence[float],
    deps: float = 0.0001,
) -> torch.Tensor:
    if (
        patch_coords.shape != patch_disp.shape
        or patch_coords.ndim != 3
        or int(patch_coords.shape[-1]) != 3
    ):
        raise ValueError(
            f"Expected patch coords/disp [centers, offsets, 3], got {tuple(patch_coords.shape)} and {tuple(patch_disp.shape)}"
        )
    n_centers = int(patch_coords.shape[0])
    n_offsets = int(patch_coords.shape[1])
    w = weights.to(device=patch_coords.device, dtype=patch_coords.dtype).reshape(
        1, n_offsets
    )
    w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)
    fixed_samples = sample_volume(
        fixed, patch_coords.reshape(-1, 3), mode="bilinear"
    ).reshape(n_centers, n_offsets)
    warped_samples = sample_volume(
        moving, (patch_coords + patch_disp).reshape(-1, 3), mode="bilinear"
    ).reshape(n_centers, n_offsets)
    mean_f = (fixed_samples * w).sum(dim=1)
    mean_w = (warped_samples * w).sum(dim=1)
    var_f = (fixed_samples.square() * w).sum(dim=1) - mean_f.square()
    var_w = (warped_samples.square() * w).sum(dim=1) - mean_w.square()
    std_f = torch.sqrt(var_f.clamp_min(0.0) + float(deps))
    std_w = torch.sqrt(var_w.clamp_min(0.0) + float(deps))
    cross = (fixed_samples * warped_samples * w).sum(dim=1) - mean_f * mean_w
    lcc = cross / (std_f * std_w).clamp_min(1e-08)
    dvol = float(cur_pix_zyx[0]) * float(cur_pix_zyx[1]) * float(cur_pix_zyx[2])
    return (1.0 - lcc).mean() * float(domain_voxels) * dvol


def idir_sampled_tv_physical(
    input_coords: torch.Tensor,
    output_rel: torch.Tensor,
    image_shape: Sequence[int],
    cur_pix_zyx: Sequence[float],
    csqrt: float,
    domain_voxels: Optional[int] = None,
) -> torch.Tensor:
    jac_norm = idir_jacobian_matrix(input_coords, output_rel)
    scale_xyz = output_rel.new_tensor(
        [
            max(int(image_shape[2]) - 1, 1) * float(cur_pix_zyx[2]) / 2.0,
            max(int(image_shape[1]) - 1, 1) * float(cur_pix_zyx[1]) / 2.0,
            max(int(image_shape[0]) - 1, 1) * float(cur_pix_zyx[0]) / 2.0,
        ]
    )
    jac_phys = (
        jac_norm * scale_xyz.view(1, 3, 1) / scale_xyz.view(1, 1, 3).clamp_min(1e-12)
    )
    density = torch.sqrt(jac_phys.square().sum(dim=(1, 2)) + float(csqrt))
    loss = density.mean()
    if domain_voxels is not None:
        dvol = float(cur_pix_zyx[0]) * float(cur_pix_zyx[1]) * float(cur_pix_zyx[2])
        loss = loss * float(domain_voxels) * dvol
    return loss


def domain_cps_from_full(full_cps: int, downsample: int) -> int:
    if int(full_cps) % int(downsample) != 0:
        raise ValueError(
            f"full_cps={full_cps} must be divisible by downsample={downsample}"
        )
    value = int(full_cps) // int(downsample)
    if value < 1:
        raise ValueError(
            f"Invalid domain cps {value} from full_cps={full_cps}, downsample={downsample}"
        )
    return value


def isotv_control_loss(
    control: torch.Tensor,
    control_shape: Sequence[int],
    eval_shape: Sequence[int],
    cur_pix_zyx: Sequence[float],
    domain_cps: int,
    csqrt: float,
) -> torch.Tensor:
    grid = flat_to_grid(control, control_shape)
    scale_xyz = grid.new_tensor(
        [
            (int(eval_shape[2]) - 1) * float(cur_pix_zyx[2]) / 2.0,
            (int(eval_shape[1]) - 1) * float(cur_pix_zyx[1]) / 2.0,
            (int(eval_shape[0]) - 1) * float(cur_pix_zyx[0]) / 2.0,
        ]
    )
    grid_phys = grid * scale_xyz.view(1, 1, 1, 3)
    kspc_zyx = [float(v) * float(domain_cps) for v in cur_pix_zyx]
    grad_sq = torch.zeros(
        grid_phys.shape[:3], device=grid_phys.device, dtype=grid_phys.dtype
    )
    dz = torch.zeros_like(grid_phys)
    dy = torch.zeros_like(grid_phys)
    dx = torch.zeros_like(grid_phys)
    dz[:-1, :, :, :] = (grid_phys[1:, :, :, :] - grid_phys[:-1, :, :, :]) / kspc_zyx[0]
    dy[:, :-1, :, :] = (grid_phys[:, 1:, :, :] - grid_phys[:, :-1, :, :]) / kspc_zyx[1]
    dx[:, :, :-1, :] = (grid_phys[:, :, 1:, :] - grid_phys[:, :, :-1, :]) / kspc_zyx[2]
    grad_sq = (
        grad_sq
        + dz.square().sum(dim=-1)
        + dy.square().sum(dim=-1)
        + dx.square().sum(dim=-1)
    )
    gvol = kspc_zyx[0] * kspc_zyx[1] * kspc_zyx[2]
    return torch.sqrt(grad_sq + float(csqrt)).sum() * float(gvol)


def expand_on_shape(
    control_disp: torch.Tensor,
    control_shape: Sequence[int],
    full_cps: int,
    eval_shape: Sequence[int],
    downsample: int,
) -> torch.Tensor:
    domain_cps = domain_cps_from_full(full_cps, downsample)
    return expand_bspline_controls(
        control_disp,
        tuple((int(v) for v in control_shape)),
        tuple((int(v) for v in eval_shape)),
        domain_cps,
    )


def dense_norm_xyz_to_vox_zyx(disp_norm_xyz: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(disp_norm_xyz, torch.Tensor):
        arr = disp_norm_xyz.detach().cpu().numpy()
    else:
        arr = np.asarray(disp_norm_xyz)
    shape_zyx = np.asarray(arr.shape[:3], dtype=np.float32)
    scale_xyz = np.asarray(
        [(shape_zyx[2] - 1) / 2.0, (shape_zyx[1] - 1) / 2.0, (shape_zyx[0] - 1) / 2.0],
        dtype=np.float32,
    )
    disp_vox_xyz = arr.astype(np.float32, copy=False) * scale_xyz.reshape(1, 1, 1, 3)
    return np.stack(
        (disp_vox_xyz[..., 2], disp_vox_xyz[..., 1], disp_vox_xyz[..., 0]), axis=-1
    ).astype(np.float32)


def compute_tre_mm(
    predicted_zyx: np.ndarray, target_zyx: np.ndarray, spacing_zyx: Sequence[float]
) -> Dict[str, object]:
    if predicted_zyx.shape != target_zyx.shape:
        raise ValueError(
            f"Landmark shape mismatch: {predicted_zyx.shape} vs {target_zyx.shape}"
        )
    spacing = np.asarray(spacing_zyx, dtype=np.float64)
    diff_mm = (
        np.asarray(predicted_zyx, dtype=np.float64)
        - np.asarray(target_zyx, dtype=np.float64)
    ) * spacing[None, :]
    tre = np.sqrt(np.sum(diff_mm**2, axis=1))
    abs_axis = np.abs(diff_mm)
    return {
        "n_landmarks": int(tre.size),
        "tre_mean_mm": float(np.mean(tre)),
        "tre_std_mm": float(np.std(tre)),
        "tre_median_mm": float(np.median(tre)),
        "tre_p95_mm": float(np.percentile(tre, 95)),
        "axis_abs_mean_zyx_mm": [float(v) for v in np.mean(abs_axis, axis=0)],
        "axis_abs_std_zyx_mm": [float(v) for v in np.std(abs_axis, axis=0)],
        "per_landmark_tre_mm": [float(v) for v in tre],
    }


def sample_dense_disp_nearest(
    disp_vox_zyx: np.ndarray, points_zyx: np.ndarray
) -> np.ndarray:
    disp = np.asarray(disp_vox_zyx)
    idx = np.rint(np.asarray(points_zyx, dtype=np.float64)).astype(int)
    idx[:, 0] = np.clip(idx[:, 0], 0, disp.shape[0] - 1)
    idx[:, 1] = np.clip(idx[:, 1], 0, disp.shape[1] - 1)
    idx[:, 2] = np.clip(idx[:, 2], 0, disp.shape[2] - 1)
    return disp[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.float64, copy=False)


def sample_dense_disp_trilinear(
    disp_vox_zyx: np.ndarray, points_zyx: np.ndarray
) -> np.ndarray:
    disp = np.asarray(disp_vox_zyx, dtype=np.float64)
    pts = np.asarray(points_zyx, dtype=np.float64)
    (d, h, w) = disp.shape[:3]
    z = np.clip(pts[:, 0], 0.0, float(d - 1))
    y = np.clip(pts[:, 1], 0.0, float(h - 1))
    x = np.clip(pts[:, 2], 0.0, float(w - 1))
    z0 = np.floor(z).astype(int)
    y0 = np.floor(y).astype(int)
    x0 = np.floor(x).astype(int)
    z1 = np.clip(z0 + 1, 0, d - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    wz = (z - z0)[:, None]
    wy = (y - y0)[:, None]
    wx = (x - x0)[:, None]
    c000 = disp[z0, y0, x0]
    c100 = disp[z1, y0, x0]
    c010 = disp[z0, y1, x0]
    c001 = disp[z0, y0, x1]
    c101 = disp[z1, y0, x1]
    c011 = disp[z0, y1, x1]
    c110 = disp[z1, y1, x0]
    c111 = disp[z1, y1, x1]
    return (
        c000 * (1 - wz) * (1 - wy) * (1 - wx)
        + c100 * wz * (1 - wy) * (1 - wx)
        + c010 * (1 - wz) * wy * (1 - wx)
        + c001 * (1 - wz) * (1 - wy) * wx
        + c101 * wz * (1 - wy) * wx
        + c011 * (1 - wz) * wy * wx
        + c110 * wz * wy * (1 - wx)
        + c111 * wz * wy * wx
    )


def landmark_tre_from_dense_disp(
    disp_vox_zyx: np.ndarray,
    lm_fixed_zyx: np.ndarray,
    lm_moving_zyx: np.ndarray,
    sample_mode: str,
    snap_output: bool,
    spacing_zyx: Sequence[float] = (1.0, 1.0, 1.0),
    dense_disp_units: str = "cropped_1mm_voxel",
) -> Dict[str, object]:
    if sample_mode == "nearest":
        disp_at_points = sample_dense_disp_nearest(disp_vox_zyx, lm_fixed_zyx)
    elif sample_mode == "trilinear":
        disp_at_points = sample_dense_disp_trilinear(disp_vox_zyx, lm_fixed_zyx)
    else:
        raise ValueError(f"Unknown dense displacement sample_mode: {sample_mode}")
    predicted = np.asarray(lm_fixed_zyx, dtype=np.float64) + disp_at_points
    if snap_output:
        predicted = np.rint(predicted)
    out = compute_tre_mm(
        predicted, np.asarray(lm_moving_zyx, dtype=np.float64), spacing_zyx
    )
    out["snap_output"] = bool(snap_output)
    out["sample_mode"] = sample_mode
    out["dense_disp_units"] = dense_disp_units
    return out


def save_landmark_csv(
    path: Path,
    disp_vox_zyx: np.ndarray,
    lm_fixed_zyx: np.ndarray,
    lm_moving_zyx: np.ndarray,
    spacing_zyx: Sequence[float] = (1.0, 1.0, 1.0),
    dense_disp_units: str = "cropped_1mm_voxel",
) -> None:
    nearest = landmark_tre_from_dense_disp(
        disp_vox_zyx,
        lm_fixed_zyx,
        lm_moving_zyx,
        "nearest",
        True,
        spacing_zyx=spacing_zyx,
        dense_disp_units=dense_disp_units,
    )
    sub = landmark_tre_from_dense_disp(
        disp_vox_zyx,
        lm_fixed_zyx,
        lm_moving_zyx,
        "trilinear",
        False,
        spacing_zyx=spacing_zyx,
        dense_disp_units=dense_disp_units,
    )
    snap_tre = np.asarray(nearest["per_landmark_tre_mm"], dtype=np.float64)
    sub_tre = np.asarray(sub["per_landmark_tre_mm"], dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "index",
                "fixed_z",
                "fixed_y",
                "fixed_x",
                "target_z",
                "target_y",
                "target_x",
                "snap_tre_mm",
                "subvoxel_tre_mm",
                "spacing_z_mm",
                "spacing_y_mm",
                "spacing_x_mm",
                "dense_disp_units",
            ]
        )
        spacing = [float(v) for v in spacing_zyx]
        for i in range(lm_fixed_zyx.shape[0]):
            writer.writerow(
                [
                    i,
                    *lm_fixed_zyx[i].tolist(),
                    *lm_moving_zyx[i].tolist(),
                    snap_tre[i],
                    sub_tre[i],
                    *spacing,
                    dense_disp_units,
                ]
            )


def _mean_sq(arr: np.ndarray) -> float:
    return float(np.mean(np.square(arr, dtype=np.float64)))


def _masked_mean_sq(arr: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    vals = np.square(arr, dtype=np.float64)[..., mask]
    return float(np.mean(vals)) if vals.size else float("nan")


def diffusion_loss_vox(
    disp_vox_zyx: np.ndarray, mask: Optional[np.ndarray] = None
) -> float:
    disp = np.moveaxis(disp_vox_zyx, -1, 0)[None]
    dz = disp[:, :, 1:, :, :] - disp[:, :, :-1, :, :]
    dy = disp[:, :, :, 1:, :] - disp[:, :, :, :-1, :]
    dx = disp[:, :, :, :, 1:] - disp[:, :, :, :, :-1]
    if mask is None:
        return float((_mean_sq(dz) + _mean_sq(dy) + _mean_sq(dx)) / 3.0)
    mz = mask[1:, :, :] & mask[:-1, :, :]
    my = mask[:, 1:, :] & mask[:, :-1, :]
    mx = mask[:, :, 1:] & mask[:, :, :-1]
    return float(
        (_masked_mean_sq(dz, mz) + _masked_mean_sq(dy, my) + _masked_mean_sq(dx, mx))
        / 3.0
    )


def bending_energy_vox(
    disp_vox_zyx: np.ndarray, mask: Optional[np.ndarray] = None
) -> float:
    disp = np.moveaxis(disp_vox_zyx, -1, 0)[None]
    dz = disp[:, :, 1:, :, :] - disp[:, :, :-1, :, :]
    dy = disp[:, :, :, 1:, :] - disp[:, :, :, :-1, :]
    dx = disp[:, :, :, :, 1:] - disp[:, :, :, :, :-1]
    dzz = dz[:, :, 1:, :, :] - dz[:, :, :-1, :, :]
    dyy = dy[:, :, :, 1:, :] - dy[:, :, :, :-1, :]
    dxx = dx[:, :, :, :, 1:] - dx[:, :, :, :, :-1]
    dzy = dz[:, :, :, 1:, :] - dz[:, :, :, :-1, :]
    dzx = dz[:, :, :, :, 1:] - dz[:, :, :, :, :-1]
    dyx = dy[:, :, :, :, 1:] - dy[:, :, :, :, :-1]
    if mask is None:
        terms = [_mean_sq(v) for v in (dzz, dyy, dxx, dzy, dzx, dyx)]
    else:
        m_dzz = mask[:-2, :, :] & mask[1:-1, :, :] & mask[2:, :, :]
        m_dyy = mask[:, :-2, :] & mask[:, 1:-1, :] & mask[:, 2:, :]
        m_dxx = mask[:, :, :-2] & mask[:, :, 1:-1] & mask[:, :, 2:]
        m_dzy = (
            mask[:-1, :-1, :] & mask[1:, :-1, :] & mask[:-1, 1:, :] & mask[1:, 1:, :]
        )
        m_dzx = (
            mask[:-1, :, :-1] & mask[1:, :, :-1] & mask[:-1, :, 1:] & mask[1:, :, 1:]
        )
        m_dyx = (
            mask[:, :-1, :-1] & mask[:, 1:, :-1] & mask[:, :-1, 1:] & mask[:, 1:, 1:]
        )
        terms = [
            _masked_mean_sq(dzz, m_dzz),
            _masked_mean_sq(dyy, m_dyy),
            _masked_mean_sq(dxx, m_dxx),
            _masked_mean_sq(dzy, m_dzy),
            _masked_mean_sq(dzx, m_dzx),
            _masked_mean_sq(dyx, m_dyx),
        ]
    return float(
        terms[0] + terms[1] + terms[2] + 2.0 * (terms[3] + terms[4] + terms[5])
    )


def jacobian_determinant_zyx(disp_vox_zyx: np.ndarray) -> np.ndarray:
    disp = np.asarray(disp_vox_zyx, dtype=np.float64)
    grads = []
    for component in range(3):
        grads.append(np.gradient(disp[..., component], edge_order=2))
    jac = np.empty((3, 3) + disp.shape[:3], dtype=np.float64)
    for component in range(3):
        for axis in range(3):
            jac[component, axis] = grads[component][axis]
        jac[component, component] += 1.0
    crop = tuple((slice(2, -2) if s > 4 else slice(None) for s in disp.shape[:3]))
    j = jac[(slice(None), slice(None)) + crop]
    return (
        j[0, 0] * (j[1, 1] * j[2, 2] - j[1, 2] * j[2, 1])
        - j[0, 1] * (j[1, 0] * j[2, 2] - j[1, 2] * j[2, 0])
        + j[0, 2] * (j[1, 0] * j[2, 1] - j[1, 1] * j[2, 0])
    )


def displacement_regularities(
    disp_vox_zyx: np.ndarray,
    foreground_mask: Optional[np.ndarray] = None,
    log_base: str = "e",
) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        "diffusion_l2_full": diffusion_loss_vox(disp_vox_zyx),
        "bending_energy_full": bending_energy_vox(disp_vox_zyx),
    }
    fg = (
        foreground_mask.astype(bool, copy=False)
        if foreground_mask is not None
        else None
    )
    if fg is not None:
        metrics["diffusion_l2_fg"] = diffusion_loss_vox(disp_vox_zyx, fg)
        metrics["bending_energy_fg"] = bending_energy_vox(disp_vox_zyx, fg)
    else:
        metrics["diffusion_l2_fg"] = float("nan")
        metrics["bending_energy_fg"] = float("nan")
    jdet = jacobian_determinant_zyx(disp_vox_zyx)
    crop = tuple(
        (slice(2, -2) if s > 4 else slice(None) for s in disp_vox_zyx.shape[:3])
    )
    if fg is not None:
        jdet_fg = jdet[fg[crop]]
    else:
        jdet_fg = np.asarray([], dtype=jdet.dtype)

    def log_std(vals: np.ndarray) -> float:
        pos = vals[vals > 0]
        if pos.size == 0:
            return float("nan")
        logs = np.log(pos)
        if log_base == "2":
            logs = logs / np.log(2.0)
        return float(np.std(logs))

    metrics.update(
        {
            "jdet_le0_pct_full": float(np.mean(jdet <= 0) * 100.0),
            "jdet_le0_pct_fg": (
                float(np.mean(jdet_fg <= 0) * 100.0) if jdet_fg.size else float("nan")
            ),
            "log_jdet_std_full": log_std(jdet),
            "log_jdet_std_fg": log_std(jdet_fg) if jdet_fg.size else float("nan"),
            "jdet_mean_full": float(np.mean(jdet)),
            "jdet_mean_fg": float(np.mean(jdet_fg)) if jdet_fg.size else float("nan"),
            "ndv_frac_full": float(np.mean(jdet <= 0)),
            "ndv_frac_fg": (
                float(np.mean(jdet_fg <= 0)) if jdet_fg.size else float("nan")
            ),
        }
    )
    mag = np.sqrt(np.sum(np.square(disp_vox_zyx.astype(np.float64)), axis=-1))
    metrics["disp_mag_mean_full_vox"] = float(np.mean(mag))
    metrics["disp_mag_p95_full_vox"] = float(np.percentile(mag, 95))
    metrics["disp_mag_max_full_vox"] = float(np.max(mag))
    if fg is not None and fg.any():
        metrics["disp_mag_mean_fg_vox"] = float(np.mean(mag[fg]))
        metrics["disp_mag_p95_fg_vox"] = float(np.percentile(mag[fg], 95))
        metrics["disp_mag_max_fg_vox"] = float(np.max(mag[fg]))
    else:
        metrics["disp_mag_mean_fg_vox"] = float("nan")
        metrics["disp_mag_p95_fg_vox"] = float("nan")
        metrics["disp_mag_max_fg_vox"] = float("nan")
    return metrics


def resize_reg1mm_disp_to_native_crop_vox(
    disp_vox_reg_zyx: np.ndarray,
    crop_shape_zyx: Sequence[int],
    native_spacing_zyx: Sequence[float],
) -> np.ndarray:
    disp_phys_reg_zyx = np.asarray(disp_vox_reg_zyx, dtype=np.float32)
    field = torch.from_numpy(disp_phys_reg_zyx).permute(3, 0, 1, 2).unsqueeze(0)
    resized = F.interpolate(
        field,
        size=tuple((int(v) for v in crop_shape_zyx)),
        mode="trilinear",
        align_corners=True,
    )
    disp_phys_crop_zyx = (
        resized.squeeze(0).permute(1, 2, 3, 0).numpy().astype(np.float32, copy=False)
    )
    spacing = np.asarray(native_spacing_zyx, dtype=np.float32).reshape(1, 1, 1, 3)
    return (disp_phys_crop_zyx / spacing).astype(np.float32, copy=False)


def final_metrics_from_dense(
    dense_disp_norm_xyz: torch.Tensor,
    case: LungCTCase,
    out_dir: Path,
    save_dense_dvf: bool,
    log_base: str,
    dvf_file_ready_callback: Optional[Callable[[], None]] = None,
) -> Dict[str, object]:
    disp_vox_reg_zyx = dense_norm_xyz_to_vox_zyx(dense_disp_norm_xyz)
    if save_dense_dvf:
        np.save(
            out_dir / "disp_vox_reg1mm_zyx.npy", disp_vox_reg_zyx.astype(np.float32)
        )
        np.save(
            out_dir / "disp_norm_xyz_reg1mm.npy",
            dense_disp_norm_xyz.detach().cpu().numpy().astype(np.float32),
        )
        if dvf_file_ready_callback is not None:
            dvf_file_ready_callback()
    tre_snap = landmark_tre_from_dense_disp(
        disp_vox_reg_zyx,
        case.landmarks_fixed,
        case.landmarks_moving,
        sample_mode="nearest",
        snap_output=True,
    )
    tre_sub = landmark_tre_from_dense_disp(
        disp_vox_reg_zyx,
        case.landmarks_fixed,
        case.landmarks_moving,
        sample_mode="trilinear",
        snap_output=False,
    )
    disp_vox_native_crop_zyx = resize_reg1mm_disp_to_native_crop_vox(
        disp_vox_reg_zyx, case.crop_shape_zyx, case.native_spacing_zyx
    )
    native_tre_snap = landmark_tre_from_dense_disp(
        disp_vox_native_crop_zyx,
        case.landmarks_fixed_crop,
        case.landmarks_moving_crop,
        sample_mode="nearest",
        snap_output=True,
        spacing_zyx=case.native_spacing_zyx,
        dense_disp_units="native_crop_voxel",
    )
    native_tre_sub = landmark_tre_from_dense_disp(
        disp_vox_native_crop_zyx,
        case.landmarks_fixed_crop,
        case.landmarks_moving_crop,
        sample_mode="trilinear",
        snap_output=False,
        spacing_zyx=case.native_spacing_zyx,
        dense_disp_units="native_crop_voxel",
    )
    regularity = displacement_regularities(
        disp_vox_reg_zyx,
        foreground_mask=case.fixed_mask.astype(bool),
        log_base=log_base,
    )
    save_landmark_csv(
        out_dir / "per_landmark_tre.csv",
        disp_vox_reg_zyx,
        case.landmarks_fixed,
        case.landmarks_moving,
    )
    save_landmark_csv(
        out_dir / "per_landmark_tre_native_crop.csv",
        disp_vox_native_crop_zyx,
        case.landmarks_fixed_crop,
        case.landmarks_moving_crop,
        spacing_zyx=case.native_spacing_zyx,
        dense_disp_units="native_crop_voxel",
    )
    return {
        "identity_tre": compute_tre_mm(
            case.landmarks_fixed, case.landmarks_moving, (1.0, 1.0, 1.0)
        ),
        "tre_snap_to_voxel": tre_snap,
        "tre_subvoxel": tre_sub,
        "native_crop_identity_tre": compute_tre_mm(
            case.landmarks_fixed_crop,
            case.landmarks_moving_crop,
            case.native_spacing_zyx,
        ),
        "native_crop_tre_snap_to_voxel": native_tre_snap,
        "native_crop_tre_subvoxel": native_tre_sub,
        "headline_tre_metric": "native_crop_tre_snap_to_voxel",
        "regularity_fg_registration_crop": regularity,
        "dense_field_shape_zyx": list(disp_vox_reg_zyx.shape[:3]),
    }
