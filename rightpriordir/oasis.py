"""Audited registration/data kernels; see docs/methods.md for protocol scope."""

from __future__ import annotations
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import scipy.ndimage

try:
    from surface_distance import metrics as surface_distance_metrics
except ImportError:
    surface_distance_metrics = None

DEFAULT_OASIS_ROOT = Path(os.environ.get("OASIS_ROOT", "data/oasis"))

DEFAULT_PAIR_FILE = (
    Path(__file__).resolve().parents[1] / "splits/pair_test_200_shuffled_seed000003.txt"
)

IMAGE_SHAPE: Tuple[int, int, int] = (160, 224, 192)

OASIS_LABELS = list(range(1, 36))


@dataclass(frozen=True)
class CaseInfo:
    rank: int
    original_case_index: int
    fixed_id: str
    moving_id: str
    pair_file: str
    sorted_pair_file: str


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def parse_pair_file(path: Path) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    pattern = re.compile("\\((\\d+)\\s*,\\s*(\\d+)\\)")
    for line in path.read_text().splitlines():
        match = pattern.search(line)
        if match:
            pairs.append((match.group(1), match.group(2)))
    if not pairs:
        raise ValueError(f"No pairs found in {path}")
    return pairs


def resolve_case(
    *,
    rank: int,
    pair_file: Path = DEFAULT_PAIR_FILE,
    oasis_root: Path = DEFAULT_OASIS_ROOT,
) -> CaseInfo:
    pair_file = Path(pair_file)
    pairs = parse_pair_file(pair_file)
    if rank < 0 or rank >= len(pairs):
        raise IndexError(f"Case rank {rank} outside pair file length {len(pairs)}")
    (fixed_id, moving_id) = pairs[rank]
    sorted_pair_file = Path(oasis_root) / "pair_test_200.txt"
    sorted_pairs = parse_pair_file(sorted_pair_file)
    try:
        original_index = sorted_pairs.index((fixed_id, moving_id))
    except ValueError as exc:
        raise ValueError(
            f"Pair ({fixed_id}, {moving_id}) from {pair_file} is absent from {sorted_pair_file}"
        ) from exc
    return CaseInfo(
        rank=int(rank),
        original_case_index=int(original_index),
        fixed_id=fixed_id,
        moving_id=moving_id,
        pair_file=str(pair_file),
        sorted_pair_file=str(sorted_pair_file),
    )


def reorient_lia_to_ras(array: np.ndarray) -> np.ndarray:
    """Match the legacy OASIS loader orientation used by prior task9 runs."""
    out = np.transpose(array, [0, 2, 1])
    out = np.flip(out, 0)
    out = np.flip(out, 2)
    return np.ascontiguousarray(out)


def load_nifti_reoriented(path: Path, *, dtype: np.dtype) -> np.ndarray:
    arr = np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32))
    arr = reorient_lia_to_ras(arr)
    return np.ascontiguousarray(arr.astype(dtype, copy=False))


def pair_minmax_normalize(
    moving: np.ndarray, fixed: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    lo = float(min(np.nanmin(moving), np.nanmin(fixed)))
    hi = float(max(np.nanmax(moving), np.nanmax(fixed)))
    denom = hi - lo + 1e-12
    return (
        ((moving - lo) / denom).astype(np.float32),
        ((fixed - lo) / denom).astype(np.float32),
    )


def load_oasis_case(
    *,
    rank: int,
    device: torch.device,
    oasis_root: Path = DEFAULT_OASIS_ROOT,
    pair_file: Path = DEFAULT_PAIR_FILE,
) -> Dict[str, torch.Tensor | str | int | Dict[str, object]]:
    case = resolve_case(rank=rank, pair_file=pair_file, oasis_root=oasis_root)
    test_root = Path(oasis_root) / "test"
    fixed_raw = load_nifti_reoriented(
        test_root / f"img{case.fixed_id}.nii.gz", dtype=np.float32
    )
    moving_raw = load_nifti_reoriented(
        test_root / f"img{case.moving_id}.nii.gz", dtype=np.float32
    )
    fixed_seg = load_nifti_reoriented(
        test_root / f"seg{case.fixed_id}.nii.gz", dtype=np.float32
    ).astype(np.int16)
    moving_seg = load_nifti_reoriented(
        test_root / f"seg{case.moving_id}.nii.gz", dtype=np.float32
    ).astype(np.int16)
    for name, array in [
        ("fixed image", fixed_raw),
        ("moving image", moving_raw),
        ("fixed labels", fixed_seg),
        ("moving labels", moving_seg),
    ]:
        if array.shape != IMAGE_SHAPE or not np.isfinite(array).all():
            raise ValueError(
                f"Invalid {name}: expected finite array of shape {IMAGE_SHAPE}, got {array.shape}"
            )
    for labels in (fixed_seg, moving_seg):
        if np.any((labels < 0) | (labels > 35)):
            raise ValueError(
                "Expected the prepared OASIS label map 0–35; do not supply original unremapped labels"
            )
    (moving, fixed) = pair_minmax_normalize(moving_raw, fixed_raw)
    fixed_image_mask = fixed_raw > 0
    moving_image_mask = moving_raw > 0
    fixed_seg_mask = fixed_seg > 0
    mask_audit = {
        "fixed_image_positive_voxels": int(fixed_image_mask.sum()),
        "moving_image_positive_voxels": int(moving_image_mask.sum()),
        "fixed_seg_positive_voxels": int(fixed_seg_mask.sum()),
        "fixed_image_positive_fraction": float(fixed_image_mask.mean()),
        "fixed_seg_positive_fraction": float(fixed_seg_mask.mean()),
        "fixed_image_positive_minus_seg_positive_voxels": int(
            fixed_image_mask.sum() - fixed_seg_mask.sum()
        ),
        "foreground_rule": "fixed raw image > 0",
        "segmentation_mask_role": "dice_hd95_asd_only",
    }
    return {
        "moving": torch.from_numpy(moving).to(device=device, dtype=torch.float32),
        "fixed": torch.from_numpy(fixed).to(device=device, dtype=torch.float32),
        "moving_seg": torch.from_numpy(moving_seg.astype(np.float32)).to(device=device),
        "fixed_seg": torch.from_numpy(fixed_seg.astype(np.float32)).to(device=device),
        "moving_image_mask": torch.from_numpy(moving_image_mask.astype(np.float32)).to(
            device=device
        ),
        "fixed_image_mask": torch.from_numpy(fixed_image_mask.astype(np.float32)).to(
            device=device
        ),
        "fixed_seg_mask": torch.from_numpy(fixed_seg_mask.astype(np.float32)).to(
            device=device
        ),
        "fixed_mask": torch.from_numpy(fixed_image_mask.astype(np.float32)).to(
            device=device
        ),
        "moving_id": case.moving_id,
        "fixed_id": case.fixed_id,
        "case_rank": case.rank,
        "original_case_index": case.original_case_index,
        "pair_file": case.pair_file,
        "sorted_pair_file": case.sorted_pair_file,
        "mask_audit": mask_audit,
    }


def set_cuda_device(
    gpu: str, *, cudnn_benchmark: bool = True, cudnn_deterministic: bool = False
) -> torch.device:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    torch.backends.cudnn.deterministic = bool(cudnn_deterministic)
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this runner.")
    torch.cuda.set_device(0)
    return torch.device("cuda")


def cuda_backend_metadata(device: Optional[torch.device] = None) -> Dict[str, object]:
    """Return reproducibility metadata for the active CUDA/cuDNN backend."""
    metadata: Dict[str, object] = {
        "torch_version": str(torch.__version__),
        "cuda_runtime_version": (
            None if torch.version.cuda is None else str(torch.version.cuda)
        ),
        "cudnn_version": None,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cuda_device_index": None,
        "cuda_device_name": None,
    }
    try:
        metadata["cudnn_version"] = torch.backends.cudnn.version()
    except Exception:
        metadata["cudnn_version"] = None
    target = device
    if target is not None and target.type == "cuda" and torch.cuda.is_available():
        index = int(torch.cuda.current_device())
        metadata["cuda_device_index"] = index
        metadata["cuda_device_name"] = str(torch.cuda.get_device_name(index))
    return metadata


def make_coordinate_tensor(
    shape: Tuple[int, int, int], device: torch.device
) -> torch.Tensor:
    vectors = [torch.linspace(-1.0, 1.0, int(s), device=device) for s in shape]
    grid = torch.meshgrid(*vectors, indexing="ij")
    stacked = torch.stack((grid[2], grid[1], grid[0]), dim=-1)
    return stacked.reshape(-1, 3)


def make_coordinate_grid(
    shape: Tuple[int, int, int], device: torch.device
) -> torch.Tensor:
    return make_coordinate_tensor(shape, device).reshape(*shape, 3)


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
        eps = 1e-06
        grad = (
            2.0
            / (tensor.numel() - 1.0)
            * (grad_output.detach() / (result.detach() * 2 + eps))
            * (tensor.detach() - tensor.mean().detach())
        )
        return (grad,)


def ncc_loss(x1: torch.Tensor, x2: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    if x1.shape != x2.shape:
        raise ValueError(f"NCC shape mismatch: {tuple(x1.shape)} vs {tuple(x2.shape)}")
    cc = ((x1 - x1.mean()) * (x2 - x2.mean())).mean()
    std = StableStd.apply(x1) * StableStd.apply(x2)
    return -(cc / (std + eps)).mean()


class DirectCPGenerator(nn.Module):

    def __init__(self, control_shape: Tuple[int, int, int]):
        super().__init__()
        self.delta = nn.Parameter(torch.zeros(*control_shape, 3))

    def forward(
        self, coords: torch.Tensor, control_shape: Tuple[int, int, int]
    ) -> torch.Tensor:
        del coords, control_shape
        return self.delta.reshape(-1, 3)


class NetworkGenerator(nn.Module):

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self, coords: torch.Tensor, control_shape: Tuple[int, int, int]
    ) -> torch.Tensor:
        del control_shape
        return self.model(coords)


def build_siren(hidden_dim, hidden_layers, omega, input_dim=3, *, upstream="IDIR"):
    from .upstream import siren

    return siren(
        upstream,
        [int(input_dim)] + [int(hidden_dim)] * int(hidden_layers) + [3],
        float(omega),
    )


def cubic_bspline_value(x: float, derivative: int = 0) -> float:
    t = abs(float(x))
    if derivative == 0:
        if t < 1:
            return 2.0 / 3.0 + 0.5 * t**3 - t**2
        if t < 2:
            return (2 - t) ** 3 / 6.0
        return 0.0
    if derivative == 1:
        if t < 1:
            return (1.5 * t - 2.0) * x
        if x < 0:
            return 0.5 * (t - 2) ** 2
        return -0.5 * (t - 2) ** 2
    if derivative == 2:
        if t < 1:
            return 3.0 * t - 2.0
        if t < 2:
            return -t + 2.0
        return 0.0
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
    dilation: int = 1,
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
    result = conv_fn(
        result, weight, stride=stride, dilation=dilation, padding=padding, groups=groups
    )
    result = result.reshape(shape[0:-1] + result.shape[-1:])
    return result.transpose(-1, dim)


def bspline_control_coords(
    image_shape: Tuple[int, int, int], cps: int, device: torch.device
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
    control_shape: Tuple[int, int, int],
    image_shape: Tuple[int, int, int],
    cps: int,
) -> torch.Tensor:
    field = control_values.reshape(*control_shape, 3).permute(3, 0, 1, 2).unsqueeze(0)
    for dim in range(3):
        kernel = cubic_bspline1d(
            cps, derivative=0, dtype=field.dtype, device=field.device
        )
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


CP_MARGIN = 4


def parse_int_list(text: str) -> List[int]:
    values = [int(token.strip()) for token in str(text).split(",") if token.strip()]
    if not values:
        raise ValueError(f"No integer values parsed from {text!r}")
    return values


def parse_stage_ints(text: str, expected_len: int) -> List[int]:
    values = parse_int_list(text)
    if len(values) == 1 and int(expected_len) > 1:
        return values * int(expected_len)
    if len(values) != int(expected_len):
        raise ValueError(
            f"Expected {expected_len} stage values from {text!r}, got {values}"
        )
    return values


def parse_float_list(text: str, expected_len: int, default: float) -> List[float]:
    if str(text).strip():
        values = [
            float(token.strip()) for token in str(text).split(",") if token.strip()
        ]
    else:
        values = [float(default)]
    if len(values) == 1 and int(expected_len) > 1:
        return values * int(expected_len)
    if len(values) != int(expected_len):
        raise ValueError(
            f"Expected {expected_len} stage float values from {text!r}, got {values}"
        )
    return values


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
        shapes[int(spacing)] = tuple((int(ratio * q + int(cp_margin)) for q in q_base))
    return shapes


def assert_nested_schedule(
    shapes: Dict[int, Tuple[int, int, int]], spacings: Sequence[int]
) -> None:
    for coarse, fine in zip(spacings, spacings[1:]):
        if int(coarse) != 2 * int(fine):
            raise ValueError(
                f"Only dyadic consecutive spacing is supported, got {coarse}->{fine}"
            )
        expected = tuple(
            (int(2 * (int(n) - CP_MARGIN) + CP_MARGIN) for n in shapes[int(coarse)])
        )
        if tuple(shapes[int(fine)]) != expected:
            raise ValueError(
                f"Non-nested CP grid: {coarse}->{fine}, observed={shapes[int(fine)]}, expected={expected}"
            )


def flat_to_grid(
    control_flat: torch.Tensor, control_shape: Sequence[int]
) -> torch.Tensor:
    return control_flat.reshape(*tuple((int(v) for v in control_shape)), 3)


def grid_to_flat(control_grid: torch.Tensor) -> torch.Tensor:
    return control_grid.reshape(-1, int(control_grid.shape[-1]))


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
    out = moved.new_empty((2 * moved.shape[0] - CP_MARGIN, *moved.shape[1:]))
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


def expand_on_shape(
    control_disp: torch.Tensor,
    control_shape: Sequence[int],
    full_cps: int,
    eval_shape: Sequence[int],
    downsample: int,
) -> torch.Tensor:
    domain_cps = domain_cps_from_full(int(full_cps), int(downsample))
    return expand_bspline_controls(
        control_disp,
        tuple((int(v) for v in control_shape)),
        tuple((int(v) for v in eval_shape)),
        domain_cps,
    )


def gaussian_kernel1d(
    sigma: float, radius_sigma: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    radius = max(1, int(math.ceil(float(radius_sigma) * float(sigma))))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (coords / float(sigma)) ** 2)
    return kernel / kernel.sum()


def gaussian_blur3d(
    volume: torch.Tensor, sigma: float, radius_sigma: float
) -> Tuple[torch.Tensor, int]:
    if float(sigma) <= 0.0:
        return (volume, 1)
    kernel = gaussian_kernel1d(
        float(sigma), float(radius_sigma), volume.device, volume.dtype
    )
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
    if int(factor) <= 1:
        return (volume, 0.0, 1)
    sigma = float(sigma_scale) * float(factor)
    (blurred, kernel_size) = gaussian_blur3d(
        volume, sigma=sigma, radius_sigma=float(radius_sigma)
    )
    target_shape = downsample_shape(tuple((int(v) for v in volume.shape)), int(factor))
    down = F.interpolate(
        blurred[None, None], size=target_shape, mode="trilinear", align_corners=True
    )
    return (down.squeeze(0).squeeze(0), sigma, kernel_size)


def as_tensor(
    x: torch.Tensor | np.ndarray, *, device: Optional[torch.device] = None
) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device) if device is not None else x
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def dhw3_to_ncdhw(
    disp_dhw3: torch.Tensor | np.ndarray, *, device: Optional[torch.device] = None
) -> torch.Tensor:
    disp = as_tensor(disp_dhw3, device=device).float()
    if disp.ndim != 4 or disp.shape[-1] != 3:
        raise ValueError(f"Expected [D,H,W,3], got {tuple(disp.shape)}")
    return disp.permute(3, 0, 1, 2).unsqueeze(0).contiguous()


def mask_to_dhw(
    mask: torch.Tensor | np.ndarray, *, device: Optional[torch.device] = None
) -> torch.Tensor:
    mask_t = as_tensor(mask, device=device).bool()
    if mask_t.ndim != 3:
        raise ValueError(f"Expected [D,H,W] mask, got {tuple(mask_t.shape)}")
    return mask_t


def norm_grid_dhw3_to_voxel_dhw3(
    disp_norm: torch.Tensor | np.ndarray,
    image_shape: Optional[Tuple[int, int, int]] = None,
) -> torch.Tensor:
    disp = as_tensor(disp_norm).float()
    if image_shape is None:
        (d, h, w) = (int(v) for v in disp.shape[:3])
    else:
        (d, h, w) = (int(v) for v in image_shape)
    factors = torch.tensor(
        [(w - 1) / 2.0, (h - 1) / 2.0, (d - 1) / 2.0],
        dtype=disp.dtype,
        device=disp.device,
    )
    return disp * factors.view(1, 1, 1, 3)


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


def _term_mean(
    term: torch.Tensor, mask: Optional[torch.Tensor], component_reduction: str = "mean"
) -> torch.Tensor:
    if component_reduction == "sum":
        density = term.square().sum(dim=1)
        return _masked_density_mean(density, mask)
    if mask is None:
        return term.square().mean()
    mask_t = mask.to(device=term.device).bool()
    selected = term.square()[..., mask_t]
    if selected.numel() == 0:
        return torch.as_tensor(float("nan"), dtype=term.dtype, device=term.device)
    return selected.mean()


def forward_valid_be(
    disp_ncdhw: torch.Tensor,
    mask: Optional[torch.Tensor | np.ndarray] = None,
    *,
    component_reduction: str = "mean",
) -> torch.Tensor:
    disp = disp_ncdhw.float()
    dz = disp[:, :, 1:, :, :] - disp[:, :, :-1, :, :]
    dy = disp[:, :, :, 1:, :] - disp[:, :, :, :-1, :]
    dx = disp[:, :, :, :, 1:] - disp[:, :, :, :, :-1]
    terms = [
        (dz[:, :, 1:, :, :] - dz[:, :, :-1, :, :], 1.0, "dzz"),
        (dy[:, :, :, 1:, :] - dy[:, :, :, :-1, :], 1.0, "dyy"),
        (dx[:, :, :, :, 1:] - dx[:, :, :, :, :-1], 1.0, "dxx"),
        (dz[:, :, :, 1:, :] - dz[:, :, :, :-1, :], 2.0, "dzy"),
        (dz[:, :, :, :, 1:] - dz[:, :, :, :, :-1], 2.0, "dzx"),
        (dy[:, :, :, :, 1:] - dy[:, :, :, :, :-1], 2.0, "dyx"),
    ]
    if mask is None:
        masks = {name: None for (_, _, name) in terms}
    else:
        m = mask_to_dhw(mask, device=disp.device)
        masks = {
            "dzz": m[2:, :, :] & m[1:-1, :, :] & m[:-2, :, :],
            "dyy": m[:, 2:, :] & m[:, 1:-1, :] & m[:, :-2, :],
            "dxx": m[:, :, 2:] & m[:, :, 1:-1] & m[:, :, :-2],
            "dzy": m[1:, 1:, :] & m[:-1, 1:, :] & m[1:, :-1, :] & m[:-1, :-1, :],
            "dzx": m[1:, :, 1:] & m[:-1, :, 1:] & m[1:, :, :-1] & m[:-1, :, :-1],
            "dyx": m[:, 1:, 1:] & m[:, :-1, 1:] & m[:, 1:, :-1] & m[:, :-1, :-1],
        }
    total = torch.zeros((), dtype=disp.dtype, device=disp.device)
    for term, weight, name in terms:
        total = total + float(weight) * _term_mean(
            term, masks[name], component_reduction
        )
    return total


def forward_valid_diffusion(
    disp_ncdhw: torch.Tensor, mask: Optional[torch.Tensor | np.ndarray] = None
) -> torch.Tensor:
    disp = disp_ncdhw.float()
    dz = disp[:, :, 1:, :, :] - disp[:, :, :-1, :, :]
    dy = disp[:, :, :, 1:, :] - disp[:, :, :, :-1, :]
    dx = disp[:, :, :, :, 1:] - disp[:, :, :, :, :-1]
    if mask is None:
        return (dz.square().mean() + dy.square().mean() + dx.square().mean()) / 3.0
    m = mask_to_dhw(mask, device=disp.device)
    mz = m[1:, :, :] & m[:-1, :, :]
    my = m[:, 1:, :] & m[:, :-1, :]
    mx = m[:, :, 1:] & m[:, :, :-1]
    return (_term_mean(dz, mz) + _term_mean(dy, my) + _term_mean(dx, mx)) / 3.0


def expand_cubic_bspline_derivative(
    control_disp_dhw3: torch.Tensor | np.ndarray,
    image_shape: Tuple[int, int, int],
    cps: int,
    derivative_orders: Sequence[int],
) -> torch.Tensor:
    field = dhw3_to_ncdhw(control_disp_dhw3).float()
    for dim, order in enumerate(derivative_orders):
        kernel = cubic_bspline1d(
            cps, derivative=int(order), dtype=field.dtype, device=field.device
        )
        if order:
            kernel = kernel * (1.0 / float(cps)) ** int(order)
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
        (slice(cps, cps + int(image_shape[i])) for i in range(3))
    )
    return field[slicer].contiguous()


def analytic_bspline_sampled_be(
    control_disp_dhw3: torch.Tensor | np.ndarray,
    image_shape: Tuple[int, int, int],
    cps: int,
    mask: Optional[torch.Tensor | np.ndarray] = None,
    *,
    component_reduction: str = "mean",
) -> torch.Tensor:
    terms = [
        (
            expand_cubic_bspline_derivative(
                control_disp_dhw3, image_shape, cps, (2, 0, 0)
            ),
            1.0,
        ),
        (
            expand_cubic_bspline_derivative(
                control_disp_dhw3, image_shape, cps, (0, 2, 0)
            ),
            1.0,
        ),
        (
            expand_cubic_bspline_derivative(
                control_disp_dhw3, image_shape, cps, (0, 0, 2)
            ),
            1.0,
        ),
        (
            expand_cubic_bspline_derivative(
                control_disp_dhw3, image_shape, cps, (1, 1, 0)
            ),
            2.0,
        ),
        (
            expand_cubic_bspline_derivative(
                control_disp_dhw3, image_shape, cps, (1, 0, 1)
            ),
            2.0,
        ),
        (
            expand_cubic_bspline_derivative(
                control_disp_dhw3, image_shape, cps, (0, 1, 1)
            ),
            2.0,
        ),
    ]
    density = _density_from_terms(terms, component_reduction=component_reduction)
    mask_t = None if mask is None else mask_to_dhw(mask, device=density.device)
    return _masked_density_mean(density, mask_t)


def analytic_bspline_sampled_diffusion(
    control_disp_dhw3: torch.Tensor | np.ndarray,
    image_shape: Tuple[int, int, int],
    cps: int,
    mask: Optional[torch.Tensor | np.ndarray] = None,
) -> torch.Tensor:
    terms = [
        expand_cubic_bspline_derivative(control_disp_dhw3, image_shape, cps, (1, 0, 0)),
        expand_cubic_bspline_derivative(control_disp_dhw3, image_shape, cps, (0, 1, 0)),
        expand_cubic_bspline_derivative(control_disp_dhw3, image_shape, cps, (0, 0, 1)),
    ]
    density = sum((term.square().mean(dim=1) for term in terms)) / 3.0
    mask_t = None if mask is None else mask_to_dhw(mask, device=density.device)
    return _masked_density_mean(density, mask_t)


def be_rows_to_columns(rows: Sequence[Dict[str, object]]) -> Dict[str, float]:
    formula_map = {
        "forward_valid": "be_fd_forward",
        "analytic_bspline_sampled": "be_bspline_analytic",
    }
    out: Dict[str, float] = {}
    for row in rows:
        formula = str(row.get("formula"))
        if formula not in formula_map:
            continue
        if str(row.get("component_reduction")) != "mean":
            continue
        key = f"{formula_map[formula]}_{row['region']}_{row['unit']}"
        out[key] = float(row["value"])
    return out


def diffusion_rows_to_columns(rows: Sequence[Dict[str, object]]) -> Dict[str, float]:
    formula_map = {
        "forward_valid": "diffusion_fd_forward",
        "analytic_bspline_sampled": "diffusion_bspline_analytic",
        "idir_autograd": "diffusion_idir_autograd",
    }
    out: Dict[str, float] = {}
    for row in rows:
        formula = str(row.get("formula"))
        if formula not in formula_map:
            continue
        if str(row.get("component_reduction")) != "mean":
            continue
        key = f"{formula_map[formula]}_{row['region']}_{row['unit']}"
        out[key] = float(row["value"])
    return out


def add_native_analytic_columns(
    reg: Dict[str, object], *, prefix: str, sources: Sequence[str]
) -> None:
    for region in ("full", "fixed_fg"):
        for unit in ("voxel", "normalized_grid"):
            native_key = f"{prefix}_native_analytic_{region}_{unit}"
            for source in sources:
                source_key = f"{source}_{region}_{unit}"
                if source_key in reg:
                    reg[native_key] = reg[source_key]
                    break


def add_native_analytic_be_columns(reg: Dict[str, object]) -> None:
    add_native_analytic_columns(
        reg, prefix="be", sources=("be_idir_autograd", "be_bspline_analytic")
    )


def add_native_analytic_diffusion_columns(reg: Dict[str, object]) -> None:
    add_native_analytic_columns(
        reg,
        prefix="diffusion",
        sources=("diffusion_idir_autograd", "diffusion_bspline_analytic"),
    )


def compute_be_metric_rows(
    dense_disp_norm_dhw3: torch.Tensor | np.ndarray,
    fixed_mask: torch.Tensor | np.ndarray,
    *,
    control_disp_norm_dhw3: Optional[torch.Tensor | np.ndarray] = None,
    image_shape: Tuple[int, int, int] = IMAGE_SHAPE,
    cps: Optional[int] = None,
) -> List[Dict[str, object]]:
    disp_norm_ch = dhw3_to_ncdhw(dense_disp_norm_dhw3)
    disp_vox_ch = dhw3_to_ncdhw(
        norm_grid_dhw3_to_voxel_dhw3(dense_disp_norm_dhw3, image_shape)
    )
    mask_t = mask_to_dhw(fixed_mask)
    rows: List[Dict[str, object]] = []
    for unit, disp_ch in [("normalized_grid", disp_norm_ch), ("voxel", disp_vox_ch)]:
        for region, mask in [("full", None), ("fixed_fg", mask_t)]:
            rows.append(
                {
                    "unit": unit,
                    "formula": "forward_valid",
                    "region": region,
                    "component_reduction": "mean",
                    "value": float(
                        forward_valid_be(disp_ch, mask, component_reduction="mean")
                        .detach()
                        .cpu()
                    ),
                    "notes": "Valid forward second differences.",
                }
            )
    if control_disp_norm_dhw3 is not None:
        if cps is None:
            raise ValueError("cps is required for analytic B-spline BE")
        control_norm = as_tensor(control_disp_norm_dhw3).float()
        control_vox = norm_grid_dhw3_to_voxel_dhw3(control_norm, image_shape)
        for unit, control in [
            ("normalized_grid", control_norm),
            ("voxel", control_vox),
        ]:
            for region, mask in [("full", None), ("fixed_fg", mask_t)]:
                rows.append(
                    {
                        "unit": unit,
                        "formula": "analytic_bspline_sampled",
                        "region": region,
                        "component_reduction": "mean",
                        "value": float(
                            analytic_bspline_sampled_be(
                                control,
                                image_shape,
                                cps,
                                mask,
                                component_reduction="mean",
                            )
                            .detach()
                            .cpu()
                        ),
                        "notes": "Cubic B-spline second derivatives sampled at voxel centers.",
                    }
                )
    return rows


def compute_diffusion_metric_rows(
    dense_disp_norm_dhw3: torch.Tensor | np.ndarray,
    fixed_mask: torch.Tensor | np.ndarray,
    *,
    control_disp_norm_dhw3: Optional[torch.Tensor | np.ndarray] = None,
    image_shape: Tuple[int, int, int] = IMAGE_SHAPE,
    cps: Optional[int] = None,
) -> List[Dict[str, object]]:
    disp_norm_ch = dhw3_to_ncdhw(dense_disp_norm_dhw3)
    disp_vox_ch = dhw3_to_ncdhw(
        norm_grid_dhw3_to_voxel_dhw3(dense_disp_norm_dhw3, image_shape)
    )
    mask_t = mask_to_dhw(fixed_mask)
    rows: List[Dict[str, object]] = []
    for unit, disp_ch in [("normalized_grid", disp_norm_ch), ("voxel", disp_vox_ch)]:
        for region, mask in [("full", None), ("fixed_fg", mask_t)]:
            rows.append(
                {
                    "unit": unit,
                    "formula": "forward_valid",
                    "region": region,
                    "component_reduction": "mean",
                    "value": float(
                        forward_valid_diffusion(disp_ch, mask).detach().cpu()
                    ),
                    "notes": "Valid forward first differences.",
                }
            )
    if control_disp_norm_dhw3 is not None:
        if cps is None:
            raise ValueError("cps is required for analytic B-spline diffusion")
        control_norm = as_tensor(control_disp_norm_dhw3).float()
        control_vox = norm_grid_dhw3_to_voxel_dhw3(control_norm, image_shape)
        for unit, control in [
            ("normalized_grid", control_norm),
            ("voxel", control_vox),
        ]:
            for region, mask in [("full", None), ("fixed_fg", mask_t)]:
                rows.append(
                    {
                        "unit": unit,
                        "formula": "analytic_bspline_sampled",
                        "region": region,
                        "component_reduction": "mean",
                        "value": float(
                            analytic_bspline_sampled_diffusion(
                                control, image_shape, cps, mask
                            )
                            .detach()
                            .cpu()
                        ),
                        "notes": "Cubic B-spline first derivatives sampled at voxel centers.",
                    }
                )
    return rows


def dhw3_norm_to_zyx_voxel_ncdhw(
    disp_norm_dhw3: torch.Tensor | np.ndarray,
) -> np.ndarray:
    disp = np.asarray(as_tensor(disp_norm_dhw3).detach().cpu(), dtype=np.float32)
    (d, h, w, _) = disp.shape
    out = np.empty((1, 3, d, h, w), dtype=np.float32)
    out[0, 0] = disp[..., 2] * ((d - 1) / 2.0)
    out[0, 1] = disp[..., 1] * ((h - 1) / 2.0)
    out[0, 2] = disp[..., 0] * ((w - 1) / 2.0)
    return out


def generate_voxel_coordinate_grid(
    shape: Sequence[int], device: torch.device, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    axes = [
        torch.linspace(
            start=0,
            end=int(dim_size) - 1,
            steps=int(dim_size),
            device=device,
            dtype=dtype,
        )
        for dim_size in shape
    ]
    return torch.stack(torch.meshgrid(axes, indexing="ij"), dim=0)[None]


def optional_add_tensor(
    addable_1: Optional[torch.Tensor], addable_2: Optional[torch.Tensor]
) -> Optional[torch.Tensor]:
    if addable_1 is None:
        return addable_2
    if addable_2 is None:
        return addable_1
    return addable_1 + addable_2


def calculate_jacobian_determinants(ddf: torch.Tensor) -> Mapping[str, torch.Tensor]:
    if ddf.size(1) != 3 or ddf.ndim != 5:
        raise ValueError(
            "Currently only supports 3D deformation fields. The input shape must be (batch_size, 3, dim_1, dim_2, dim_3)."
        )
    trans = ddf + generate_voxel_coordinate_grid(
        ddf.shape[2:], dtype=ddf.dtype, device=ddf.device
    )
    kwargs = {"dtype": trans.dtype, "device": trans.device}
    kernels = {
        "D0x": torch.tensor([-0.5, 0, 0.5], **kwargs).view(1, 1, 3, 1, 1),
        "D+x": torch.tensor([0, -1, 1], **kwargs).view(1, 1, 3, 1, 1),
        "D-x": torch.tensor([-1, 1, 0], **kwargs).view(1, 1, 3, 1, 1),
        "D0y": torch.tensor([-0.5, 0, 0.5], **kwargs).view(1, 1, 1, 3, 1),
        "D+y": torch.tensor([0, -1, 1], **kwargs).view(1, 1, 1, 3, 1),
        "D-y": torch.tensor([-1, 1, 0], **kwargs).view(1, 1, 1, 3, 1),
        "D0z": torch.tensor([-0.5, 0, 0.5], **kwargs).view(1, 1, 1, 1, 3),
        "D+z": torch.tensor([0, -1, 1], **kwargs).view(1, 1, 1, 1, 3),
        "D-z": torch.tensor([-1, 1, 0], **kwargs).view(1, 1, 1, 1, 3),
        "1*xy": torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, 0]], **kwargs).reshape(
            1, 1, 3, 3, 1
        ),
        "1*xz": torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, 0]], **kwargs).reshape(
            1, 1, 3, 1, 3
        ),
        "1*yz": torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, 0]], **kwargs).reshape(
            1, 1, 1, 3, 3
        ),
        "2*xy": torch.tensor([[0, 0, 0], [0, -1, 0], [0, 0, 1]], **kwargs).reshape(
            1, 1, 3, 3, 1
        ),
        "2*xz": torch.tensor([[0, 0, 0], [0, -1, 0], [0, 0, 1]], **kwargs).reshape(
            1, 1, 3, 1, 3
        ),
        "2*yz": torch.tensor([[0, 0, 0], [0, -1, 0], [0, 0, 1]], **kwargs).reshape(
            1, 1, 1, 3, 3
        ),
    }
    weights = {
        "x": torch.cat([kernels[key] for key in ["D0x", "D+x", "D-x"]] * 3, dim=0),
        "y": torch.cat([kernels[key] for key in ["D0y", "D+y", "D-y"]] * 3, dim=0),
        "z": torch.cat([kernels[key] for key in ["D0z", "D+z", "D-z"]] * 3, dim=0),
        "*xy": torch.cat([kernels[key] for key in ["1*xy", "2*xy"]] * 3, dim=0),
        "*xz": torch.cat([kernels[key] for key in ["1*xz", "2*xz"]] * 3, dim=0),
        "*yz": torch.cat([kernels[key] for key in ["1*yz", "2*yz"]] * 3, dim=0),
    }
    partials = {
        "x": F.conv3d(trans, weights["x"], groups=3)[:, :, :, 1:-1, 1:-1],
        "y": F.conv3d(trans, weights["y"], groups=3)[:, :, 1:-1, :, 1:-1],
        "z": F.conv3d(trans, weights["z"], groups=3)[:, :, 1:-1, 1:-1, :],
        "*xy": F.conv3d(trans, weights["*xy"], groups=3)[:, :, :, :, 1:-1],
        "*xz": F.conv3d(trans, weights["*xz"], groups=3)[:, :, :, 1:-1, :],
        "*yz": F.conv3d(trans, weights["*yz"], groups=3)[:, :, 1:-1, :, :],
    }
    jacobians = {
        "000": torch.stack(
            (partials["x"][:, ::3], partials["y"][:, ::3], partials["z"][:, ::3]),
            dim=-1,
        ).permute(0, 2, 3, 4, 1, 5),
        "+++": torch.stack(
            (partials["x"][:, 1::3], partials["y"][:, 1::3], partials["z"][:, 1::3]),
            dim=-1,
        ).permute(0, 2, 3, 4, 1, 5),
        "++-": torch.stack(
            (partials["x"][:, 1::3], partials["y"][:, 1::3], partials["z"][:, 2::3]),
            dim=-1,
        ).permute(0, 2, 3, 4, 1, 5),
        "+-+": torch.stack(
            (partials["x"][:, 1::3], partials["y"][:, 2::3], partials["z"][:, 1::3]),
            dim=-1,
        ).permute(0, 2, 3, 4, 1, 5),
        "+--": torch.stack(
            (partials["x"][:, 1::3], partials["y"][:, 2::3], partials["z"][:, 2::3]),
            dim=-1,
        ).permute(0, 2, 3, 4, 1, 5),
        "-++": torch.stack(
            (partials["x"][:, 2::3], partials["y"][:, 1::3], partials["z"][:, 1::3]),
            dim=-1,
        ).permute(0, 2, 3, 4, 1, 5),
        "-+-": torch.stack(
            (partials["x"][:, 2::3], partials["y"][:, 1::3], partials["z"][:, 2::3]),
            dim=-1,
        ).permute(0, 2, 3, 4, 1, 5),
        "--+": torch.stack(
            (partials["x"][:, 2::3], partials["y"][:, 2::3], partials["z"][:, 1::3]),
            dim=-1,
        ).permute(0, 2, 3, 4, 1, 5),
        "---": torch.stack(
            (partials["x"][:, 2::3], partials["y"][:, 2::3], partials["z"][:, 2::3]),
            dim=-1,
        ).permute(0, 2, 3, 4, 1, 5),
        "j1*": torch.stack(
            (
                partials["*xy"][:, 0::2],
                partials["*xz"][:, 0::2],
                partials["*yz"][:, 0::2],
            ),
            dim=-1,
        ).permute(0, 2, 3, 4, 1, 5),
        "j2*": torch.stack(
            (
                partials["*xy"][:, 1::2],
                partials["*yz"][:, 1::2],
                partials["*xz"][:, 1::2],
            ),
            dim=-1,
        ).permute(0, 2, 3, 4, 1, 5),
    }
    return {key: torch.linalg.det(value) for (key, value) in jacobians.items()}


def calculate_non_diffeomorphic_volume(
    jacobian_determinants: Mapping[str, torch.Tensor],
    mask: Optional[torch.Tensor] = None,
    threshold: float = 0.0,
) -> torch.Tensor:
    return calculate_non_diffeomorphic_volume_map(
        jacobian_determinants, mask=mask, threshold=threshold
    ).sum()


def calculate_non_diffeomorphic_volume_map(
    jacobian_determinants: Mapping[str, torch.Tensor],
    mask: Optional[torch.Tensor] = None,
    threshold: float = 0.0,
) -> torch.Tensor:
    if mask is not None:
        mask = mask[..., 1:-1, 1:-1, 1:-1]
    non_diff_volume_map: Optional[torch.Tensor] = None
    for diff_direction in [
        "+++",
        "++-",
        "+-+",
        "+--",
        "j1*",
        "j2*",
        "-++",
        "-+-",
        "--+",
        "---",
    ]:
        volume_map = (
            -0.5 * jacobian_determinants[diff_direction].clamp(max=threshold) / 6.0
        )
        if mask is not None:
            volume_map = volume_map * mask
        non_diff_volume_map = optional_add_tensor(non_diff_volume_map, volume_map)
    if non_diff_volume_map is None:
        raise RuntimeError("No NDV map terms were computed")
    return non_diff_volume_map


def ndv_metrics_voxel(
    disp_zyx: np.ndarray, fixed_mask: np.ndarray, device: torch.device
) -> Dict[str, float]:
    ddf = torch.from_numpy(disp_zyx).float().to(device)
    mask_t = torch.from_numpy(fixed_mask[None].astype(np.float32)).to(device)
    with torch.no_grad():
        jacdets = calculate_jacobian_determinants(ddf)
        ndv_full = calculate_non_diffeomorphic_volume(jacdets).detach()
        ndv_fg = calculate_non_diffeomorphic_volume(jacdets, mask=mask_t).detach()
        denom_full = float(np.prod(tuple((int(v) for v in jacdets["+++"].shape[1:]))))
        denom_fg = float(mask_t[..., 1:-1, 1:-1, 1:-1].sum().detach().cpu())
    return {
        "ndv_full": float(ndv_full.cpu()),
        "ndv_fg": float(ndv_fg.cpu()),
        "ndv_frac_full": (
            float(ndv_full.cpu()) / denom_full if denom_full else float("nan")
        ),
        "ndv_frac_fg": float(ndv_fg.cpu()) / denom_fg if denom_fg else float("nan"),
    }


def diffusion_loss_voxel(
    disp_zyx: np.ndarray, mask: Optional[np.ndarray] = None
) -> float:
    dz = disp_zyx[:, :, 1:, :, :] - disp_zyx[:, :, :-1, :, :]
    dy = disp_zyx[:, :, :, 1:, :] - disp_zyx[:, :, :, :-1, :]
    dx = disp_zyx[:, :, :, :, 1:] - disp_zyx[:, :, :, :, :-1]
    if mask is None:
        return float((np.mean(dz**2) + np.mean(dy**2) + np.mean(dx**2)) / 3.0)
    mz = mask[1:, :, :] & mask[:-1, :, :]
    my = mask[:, 1:, :] & mask[:, :-1, :]
    mx = mask[:, :, 1:] & mask[:, :, :-1]

    def masked_mean_sq(arr: np.ndarray, m: np.ndarray) -> float:
        if not np.any(m):
            return float("nan")
        vals = arr[..., m]
        return float(np.mean(vals**2)) if vals.size else float("nan")

    return float(
        (masked_mean_sq(dz, mz) + masked_mean_sq(dy, my) + masked_mean_sq(dx, mx)) / 3.0
    )


def jacobian_determinant_zyx(disp_zyx: np.ndarray) -> np.ndarray:
    gradx = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
    grady = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
    gradz = np.array([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)
    gradx_disp = np.stack(
        [
            scipy.ndimage.correlate(disp_zyx[:, i], gradx, mode="constant", cval=0.0)
            for i in range(3)
        ],
        axis=1,
    )
    grady_disp = np.stack(
        [
            scipy.ndimage.correlate(disp_zyx[:, i], grady, mode="constant", cval=0.0)
            for i in range(3)
        ],
        axis=1,
    )
    gradz_disp = np.stack(
        [
            scipy.ndimage.correlate(disp_zyx[:, i], gradz, mode="constant", cval=0.0)
            for i in range(3)
        ],
        axis=1,
    )
    grad_disp = np.concatenate([gradx_disp, grady_disp, gradz_disp], 0)
    jac = grad_disp + np.eye(3, 3).reshape(3, 3, 1, 1, 1)
    jac = jac[:, :, 2:-2, 2:-2, 2:-2]
    return (
        jac[0, 0] * (jac[1, 1] * jac[2, 2] - jac[1, 2] * jac[2, 1])
        - jac[1, 0] * (jac[0, 1] * jac[2, 2] - jac[0, 2] * jac[2, 1])
        + jac[2, 0] * (jac[0, 1] * jac[1, 2] - jac[0, 2] * jac[1, 1])
    )


def log_std_positive(values: np.ndarray) -> float:
    pos = values[values > 0]
    if pos.size == 0:
        return float("nan")
    return float(np.std(np.log(pos)))


def jacobian_metrics_voxel(
    disp_zyx: np.ndarray, fixed_mask: np.ndarray
) -> Dict[str, float]:
    jdet = jacobian_determinant_zyx(disp_zyx.astype(np.float64, copy=False))
    mask_crop = fixed_mask[2:-2, 2:-2, 2:-2]
    fg = jdet[mask_crop]
    return {
        "jdet_lumir_shape_d": float(jdet.shape[0]),
        "jdet_le0_pct_full": float(np.mean(jdet <= 0) * 100.0),
        "jdet_le0_pct_fg": float(np.mean(fg <= 0) * 100.0) if fg.size else float("nan"),
        "log_jdet_std_full": log_std_positive(jdet),
        "log_jdet_std_fg": log_std_positive(fg) if fg.size else float("nan"),
        "log_jdet_pos_frac_full": float(np.mean(jdet > 0)),
        "log_jdet_pos_frac_fg": float(np.mean(fg > 0)) if fg.size else float("nan"),
        "jdet_mean_full": float(np.mean(jdet)),
        "jdet_mean_fg": float(np.mean(fg)) if fg.size else float("nan"),
    }


def regularity_metrics(
    dense_disp_norm: torch.Tensor,
    fixed_mask: torch.Tensor,
    *,
    control_disp_norm: Optional[torch.Tensor] = None,
    cps: Optional[int] = None,
) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
    fixed_mask_np = fixed_mask.detach().cpu().bool().numpy()
    dense_disp_eval = dense_disp_norm.detach()
    disp_zyx = dhw3_norm_to_zyx_voxel_ncdhw(dense_disp_eval.cpu())
    reg: Dict[str, object] = {
        "diffusion_l2_full": diffusion_loss_voxel(disp_zyx),
        "diffusion_l2_fg": diffusion_loss_voxel(disp_zyx, fixed_mask_np),
    }
    reg.update(jacobian_metrics_voxel(disp_zyx, fixed_mask_np))
    reg.update(ndv_metrics_voxel(disp_zyx, fixed_mask_np, dense_disp_eval.device))
    be_rows = compute_be_metric_rows(
        dense_disp_eval,
        fixed_mask.detach().bool(),
        control_disp_norm_dhw3=(
            None if control_disp_norm is None else control_disp_norm.detach()
        ),
        image_shape=IMAGE_SHAPE,
        cps=cps,
    )
    diffusion_rows = compute_diffusion_metric_rows(
        dense_disp_eval,
        fixed_mask.detach().bool(),
        control_disp_norm_dhw3=(
            None if control_disp_norm is None else control_disp_norm.detach()
        ),
        image_shape=IMAGE_SHAPE,
        cps=cps,
    )
    reg.update(be_rows_to_columns(be_rows))
    add_native_analytic_be_columns(reg)
    reg.update(diffusion_rows_to_columns(diffusion_rows))
    add_native_analytic_diffusion_columns(reg)
    return (reg, be_rows, diffusion_rows)


def dice_per_label(
    warped_seg: torch.Tensor,
    fixed_seg: torch.Tensor,
    moving_seg: torch.Tensor | None = None,
) -> List[float]:
    warped_np = warped_seg.detach().cpu().numpy()
    fixed_np = fixed_seg.detach().cpu().numpy()
    moving_np = None if moving_seg is None else moving_seg.detach().cpu().numpy()
    out: List[float] = []
    for label in OASIS_LABELS:
        pred = warped_np == label
        true = fixed_np == label
        if moving_np is not None and (
            not ((moving_np == label).any() and pred.any() and true.any())
        ):
            out.append(float("nan"))
            continue
        denom = int(pred.sum()) + int(true.sum())
        out.append(float(2.0 * np.logical_and(pred, true).sum() / (denom + 1e-05)))
    return out


def surface_metrics_per_label(
    warped_seg: torch.Tensor, fixed_seg: torch.Tensor, moving_seg: torch.Tensor
) -> Dict[str, object]:
    if surface_distance_metrics is None:
        raise RuntimeError("surface_distance is required for HD95/ASD metrics")
    warped_np = warped_seg.detach().cpu().numpy()
    fixed_np = fixed_seg.detach().cpu().numpy()
    moving_np = moving_seg.detach().cpu().numpy()
    hd95: List[float] = []
    asd_warped_to_fixed: List[float] = []
    asd_fixed_to_warped: List[float] = []
    asd_symmetric: List[float] = []
    for label in OASIS_LABELS:
        if not (
            (fixed_np == label).any()
            and (moving_np == label).any()
            and (warped_np == label).any()
        ):
            hd95.append(float("nan"))
            asd_warped_to_fixed.append(float("nan"))
            asd_fixed_to_warped.append(float("nan"))
            asd_symmetric.append(float("nan"))
            continue
        try:
            distances = surface_distance_metrics.compute_surface_distances(
                mask_gt=fixed_np == label,
                mask_pred=warped_np == label,
                spacing_mm=(1.0, 1.0, 1.0),
            )
            hd95.append(
                float(
                    surface_distance_metrics.compute_robust_hausdorff(distances, 95.0)
                )
            )
            (fixed_to_warped_value, warped_to_fixed_value) = (
                surface_distance_metrics.compute_average_surface_distance(distances)
            )
            gt_areas = distances["surfel_areas_gt"]
            pred_areas = distances["surfel_areas_pred"]
            denom = float(np.sum(gt_areas) + np.sum(pred_areas))
            symmetric = (
                float(
                    np.sum(distances["distances_gt_to_pred"] * gt_areas)
                    + np.sum(distances["distances_pred_to_gt"] * pred_areas)
                )
                / denom
                if denom > 0
                else float("nan")
            )
        except Exception:
            hd95.append(float("nan"))
            asd_warped_to_fixed.append(float("nan"))
            asd_fixed_to_warped.append(float("nan"))
            asd_symmetric.append(float("nan"))
            continue
        asd_warped_to_fixed.append(float(warped_to_fixed_value))
        asd_fixed_to_warped.append(float(fixed_to_warped_value))
        asd_symmetric.append(float(symmetric))
    return {
        "hd95": hd95,
        "hd95_mean": safe_nanmean(hd95),
        "asd_backend": "google_deepmind_surface_distance_area_weighted",
        "asd_warped_to_fixed": asd_warped_to_fixed,
        "asd_fixed_to_warped": asd_fixed_to_warped,
        "asd_symmetric": asd_symmetric,
        "asd_warped_to_fixed_mean": safe_nanmean(asd_warped_to_fixed),
        "asd_fixed_to_warped_mean": safe_nanmean(asd_fixed_to_warped),
        "asd_symmetric_mean": safe_nanmean(asd_symmetric),
    }


def safe_nanmean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    return float(np.mean(finite)) if finite.size else float("nan")


def evaluate_from_dense_coords(
    dense_coords: torch.Tensor,
    batch: Dict[str, torch.Tensor | str | int | Dict[str, object]],
) -> Dict[str, object]:
    moving = batch["moving"]
    fixed = batch["fixed"]
    moving_seg = batch["moving_seg"]
    fixed_seg = batch["fixed_seg"]
    assert isinstance(moving, torch.Tensor)
    assert isinstance(fixed, torch.Tensor)
    assert isinstance(moving_seg, torch.Tensor)
    assert isinstance(fixed_seg, torch.Tensor)
    warped = warp_full(moving, dense_coords, mode="bilinear")
    warped_seg = warp_full(moving_seg, dense_coords, mode="nearest")
    dice = dice_per_label(warped_seg, fixed_seg, moving_seg)
    surface = surface_metrics_per_label(warped_seg, fixed_seg, moving_seg)
    metrics: Dict[str, object] = {
        "image_loss": float(
            ncc_loss(fixed[None, None], warped[None, None]).detach().cpu()
        ),
        "dice": dice,
        "dice_mean": safe_nanmean(dice),
        "dense_coords": dense_coords,
        "warped_seg": warped_seg,
    }
    metrics.update(surface)
    return metrics


def accuracy_row(
    metrics: Dict[str, object], fixed_id: str, moving_id: str
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "fixed_id": fixed_id,
        "moving_id": moving_id,
        "dense_image_loss": metrics["image_loss"],
        "dice_mean": metrics["dice_mean"],
        "hd95_mean": metrics["hd95_mean"],
        "asd_backend": metrics["asd_backend"],
        "asd_warped_to_fixed_mean": metrics["asd_warped_to_fixed_mean"],
        "asd_fixed_to_warped_mean": metrics["asd_fixed_to_warped_mean"],
        "asd_symmetric_mean": metrics["asd_symmetric_mean"],
    }
    for label, value in zip(OASIS_LABELS, metrics["dice"]):
        row[f"dice_label_{label:02d}"] = value
    for label, value in zip(OASIS_LABELS, metrics["hd95"]):
        row[f"hd95_label_{label:02d}"] = value
    for key in ["asd_warped_to_fixed", "asd_fixed_to_warped", "asd_symmetric"]:
        for label, value in zip(OASIS_LABELS, metrics[key]):
            row[f"{key}_label_{label:02d}"] = value
    return row


def save_dense_disp(path: Path, dense_coords: torch.Tensor) -> None:
    id_grid = make_coordinate_grid(IMAGE_SHAPE, dense_coords.device)
    disp = (dense_coords - id_grid).detach().cpu().numpy().astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, disp)


def count_trainable_params(module: nn.Module) -> int:
    return int(
        sum((param.numel() for param in module.parameters() if param.requires_grad))
    )
