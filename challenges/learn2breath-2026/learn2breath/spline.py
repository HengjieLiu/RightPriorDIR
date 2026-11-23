"""Numerical primitives for the self-contained MR-D-BSCP P2e rerun.

Source provenance
-----------------
This module is a deliberately small dependency closure extracted from the
historical P2e numerical implementation at inrorspline commit
``97e18208a36fa3699c94e23f8735a8103fa696a4``:
``scripts/experiments/lungct_no_sweep_repro/common.py``
(``sha256=88ff1f53d12cb32d974d9d3fdbbfb432097ea6b14312c979e318a297f474b2d9``).
It contains the functions reached by the historical pair-local runner
(``run_pairlocal.py``
``sha256=6396aa369a84af81a78e95dbdf5204e1a343e42124341e07d2c9ca74bf2c0de6``)
and their numerical dependencies.  Behaviour is intentionally preserved;
the only changes are module-local documentation, typing, and removal of
unrelated data/path helpers.

There is no runtime dependency on the source checkout.  Runtime imports are
restricted to the standard library, NumPy, and PyTorch.

Tensor convention: solver volumes are ``[Z, Y, X]`` and normalized sampling
coordinates/displacements are ordered ``(x, y, z)`` for
``grid_sample(..., align_corners=True)``.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# The four-coefficient crop margin is part of the historical nested control
# lattice and exact dyadic knot-insertion contract.
CP_MARGIN = 4


def make_coordinate_tensor(shape: Sequence[int], device: torch.device) -> torch.Tensor:
    """Return flattened normalized ``(x, y, z)`` coordinates for ``[Z,Y,X]``."""

    vectors = [torch.linspace(-1.0, 1.0, int(s), device=device) for s in shape]
    zz, yy, xx = torch.meshgrid(*vectors, indexing="ij")
    return torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3)


def make_coordinate_grid(shape: Sequence[int], device: torch.device) -> torch.Tensor:
    """Return normalized ``[Z,Y,X,3]`` coordinates in grid-sample order."""

    return make_coordinate_tensor(shape, device).reshape(*tuple(int(v) for v in shape), 3)


def warp_full(
    volume: torch.Tensor, dense_coords: torch.Tensor, mode: str = "bilinear"
) -> torch.Tensor:
    """Warp one ``[Z,Y,X]`` volume using the historical sampling convention."""

    warped = F.grid_sample(
        volume[None, None],
        dense_coords[None],
        mode=mode,
        align_corners=True,
    )
    return warped.squeeze()


def cubic_bspline_value(x: float, derivative: int = 0) -> float:
    """Evaluate the historical centered cubic B-spline basis or derivative."""

    t = abs(float(x))
    if derivative == 0:
        if t < 1:
            return (2.0 / 3.0) + 0.5 * t**3 - t**2
        if t < 2:
            return ((2.0 - t) ** 3) / 6.0
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


def cubic_bspline1d(
    stride: int,
    derivative: int = 0,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Construct the historical sampled cubic B-spline 1-D kernel."""

    if dtype is None:
        dtype = torch.float32
    kernel = torch.ones(4 * int(stride) - 1, dtype=dtype)
    radius = kernel.shape[0] // 2
    for index in range(kernel.shape[0]):
        kernel[index] = cubic_bspline_value((index - radius) / float(stride), derivative=derivative)
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
    """Apply a depth-wise 1-D convolution along an arbitrary tensor axis."""

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


def expand_bspline_controls(
    control_values: torch.Tensor,
    control_shape: Sequence[int],
    image_shape: Sequence[int],
    cps: int,
) -> torch.Tensor:
    """Expand flattened cubic B-spline controls onto a ``[Z,Y,X,3]`` grid."""

    image_shape = tuple(int(v) for v in image_shape)
    field = (
        control_values.reshape(*tuple(int(v) for v in control_shape), 3)
        .permute(3, 0, 1, 2)
        .unsqueeze(0)
    )
    for dim in range(3):
        kernel = cubic_bspline1d(cps, dtype=field.dtype, device=field.device)
        padding = (kernel.shape[0] - 1) // 2
        field = conv1d(
            field, dim=dim + 2, kernel=kernel, stride=int(cps), padding=padding, transpose=True
        )
    slicer = (slice(None), slice(None)) + tuple(
        slice(cps, cps + image_shape[index]) for index in range(3)
    )
    field = field[slicer]
    return field.squeeze(0).permute(1, 2, 3, 0).contiguous()


def build_nested_cp_shapes(
    image_shape: Sequence[int],
    base_spacing: int,
    spacings: Iterable[int],
    cp_margin: int = CP_MARGIN,
) -> Dict[int, Tuple[int, int, int]]:
    """Build the historical crop-aligned nested control-grid schedule."""

    q_base = [int(math.ceil(int(length) / float(base_spacing))) for length in image_shape]
    shapes: Dict[int, Tuple[int, int, int]] = {}
    for spacing in spacings:
        if int(base_spacing) % int(spacing) != 0:
            raise ValueError(f"spacing={spacing} must divide base_spacing={base_spacing}")
        ratio = int(base_spacing) // int(spacing)
        shapes[int(spacing)] = tuple(int(ratio * q + cp_margin) for q in q_base)
    return shapes


def assert_nested_schedule(
    shapes: Dict[int, Tuple[int, int, int]], spacings: Sequence[int]
) -> None:
    """Reject schedules that cannot use the exact dyadic prolongation rule."""

    for coarse, fine in zip(spacings, spacings[1:]):
        if coarse != 2 * fine:
            raise ValueError(f"Only dyadic consecutive spacing is supported, got {coarse}->{fine}")
        expected = tuple(int(2 * (int(n) - CP_MARGIN) + CP_MARGIN) for n in shapes[coarse])
        if tuple(shapes[fine]) != expected:
            raise ValueError(
                f"Non-nested CP grid: {coarse}->{fine}, observed={shapes[fine]}, expected={expected}"
            )


def refine_axis_cubic_dyadic_crop_aligned(control_grid: torch.Tensor, axis: int) -> torch.Tensor:
    """Perform exact crop-aligned cubic dyadic knot insertion along one axis."""

    moved = control_grid.movedim(axis, 0)
    if moved.shape[0] < 3:
        raise ValueError(f"Need at least 3 coefficients along axis {axis}, got {moved.shape[0]}")
    even = 0.5 * (moved[:-2] + moved[1:-1])
    odd = (moved[:-2] + 6.0 * moved[1:-1] + moved[2:]) / 8.0
    out = moved.new_empty((2 * moved.shape[0] - 4, *moved.shape[1:]))
    out[0::2] = even
    out[1::2] = odd
    return out.movedim(0, axis)


def cubic_bspline_dyadic_knot_insert_3d(
    coarse_grid: torch.Tensor, target_shape: Sequence[int]
) -> torch.Tensor:
    """Prolong a nested ``[Z,Y,X,3]`` control lattice without dense DVFs."""

    target = tuple(int(v) for v in target_shape)
    expected = tuple(int(2 * size - CP_MARGIN) for size in coarse_grid.shape[:3])
    if target != expected:
        raise ValueError(
            f"Target shape {target} is not nested from {tuple(coarse_grid.shape[:3])}; expected {expected}"
        )
    refined = coarse_grid
    for axis in range(3):
        refined = refine_axis_cubic_dyadic_crop_aligned(refined, axis=axis)
    return refined


def flat_to_grid(control_flat: torch.Tensor, control_shape: Sequence[int]) -> torch.Tensor:
    """View ``[N,3]`` controls as a ``[Z,Y,X,3]`` grid."""

    return control_flat.reshape(*tuple(int(v) for v in control_shape), 3)


def grid_to_flat(control_grid: torch.Tensor) -> torch.Tensor:
    """View a ``[Z,Y,X,3]`` control grid as ``[N,3]`` controls."""

    return control_grid.reshape(-1, int(control_grid.shape[-1]))


def gaussian_kernel1d(
    sigma: float,
    radius_sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the separable Gaussian kernel used by the frozen pyramid/LCC."""

    radius = max(1, int(math.ceil(radius_sigma * sigma)))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
    return kernel / kernel.sum()


def gaussian_blur3d(
    volume: torch.Tensor, sigma: float, radius_sigma: float
) -> Tuple[torch.Tensor, int]:
    """Blur a scalar ``[Z,Y,X]`` volume with replicated edge padding."""

    if sigma <= 0.0:
        return volume, 1
    kernel = gaussian_kernel1d(sigma, radius_sigma, volume.device, volume.dtype)
    pad = int(kernel.numel() // 2)
    vol = volume[None, None]
    kx = kernel.view(1, 1, 1, 1, -1)
    ky = kernel.view(1, 1, 1, -1, 1)
    kz = kernel.view(1, 1, -1, 1, 1)
    vol = F.conv3d(F.pad(vol, (pad, pad, 0, 0, 0, 0), mode="replicate"), kx)
    vol = F.conv3d(F.pad(vol, (0, 0, pad, pad, 0, 0), mode="replicate"), ky)
    vol = F.conv3d(F.pad(vol, (0, 0, 0, 0, pad, pad), mode="replicate"), kz)
    return vol.squeeze(0).squeeze(0), int(kernel.numel())


def downsample_shape(image_shape: Sequence[int], factor: int) -> Tuple[int, int, int]:
    """Return the historical ceil-preserving anti-aliased pyramid shape."""

    return tuple(max(2, int(math.ceil(int(size) / float(factor)))) for size in image_shape)


def make_loss_volume(
    volume: torch.Tensor,
    factor: int,
    sigma_scale: float,
    radius_sigma: float,
) -> Tuple[torch.Tensor, float, int]:
    """Build the fixed anti-aliased loss pyramid level."""

    if factor <= 1:
        return volume, 0.0, 1
    sigma = float(sigma_scale) * float(factor)
    blurred, kernel_size = gaussian_blur3d(volume, sigma=sigma, radius_sigma=radius_sigma)
    target_shape = downsample_shape(tuple(int(v) for v in volume.shape), factor)
    down = F.interpolate(
        blurred[None, None], size=target_shape, mode="trilinear", align_corners=True
    )
    return down.squeeze(0).squeeze(0), sigma, kernel_size


def make_border_mask(
    shape: Sequence[int], border: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Create the historical full-grid loss border mask."""

    mask = torch.ones(tuple(int(v) for v in shape), device=device, dtype=dtype)
    if border <= 0:
        return mask
    for axis, size in enumerate(mask.shape):
        if 2 * border >= int(size):
            raise ValueError(f"border_mask={border} is too large for axis {axis} with size {size}")
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
    """Anti-alias and downsample the frozen full-grid loss mask."""

    if full_mask is None:
        return None
    if factor <= 1:
        return full_mask
    mask_level, _sigma, _kernel = make_loss_volume(full_mask, factor, sigma_scale, radius_sigma)
    return mask_level.clamp(0.0, 1.0)


def stage_pix_zyx(
    reg_pix_zyx: Sequence[float],
    full_shape: Sequence[int],
    eval_shape: Sequence[int],
) -> Tuple[float, float, float]:
    """Convert full-grid physical spacing to a pyramid-level ``[Z,Y,X]`` spacing."""

    pix = []
    for spacing, full, current in zip(reg_pix_zyx, full_shape, eval_shape):
        if int(current) <= 1:
            pix.append(float(spacing))
        else:
            pix.append(float(spacing) * float(int(full) - 1) / float(int(current) - 1))
    return tuple(pix)  # type: ignore[return-value]


def gaussian_blur3d_aniso_channels(
    volumes: torch.Tensor,
    sigma_zyx: Sequence[float],
    radius_sigma: float,
) -> torch.Tensor:
    """Blur one or more ``[Z,Y,X]`` channels with anisotropic Gaussian kernels."""

    if volumes.ndim == 3:
        volumes = volumes.unsqueeze(0)
        squeeze = True
    elif volumes.ndim == 4:
        squeeze = False
    else:
        raise ValueError(f"Expected [D,H,W] or [C,D,H,W], got shape {tuple(volumes.shape)}")
    channels = int(volumes.shape[0])
    vol = volumes.unsqueeze(0)
    for axis, sigma in enumerate(float(value) for value in sigma_zyx):
        if sigma <= 0.0:
            continue
        kernel = gaussian_kernel1d(sigma, radius_sigma, volumes.device, volumes.dtype)
        pad = int(kernel.numel() // 2)
        if axis == 0:
            kernel_3d = kernel.view(1, 1, -1, 1, 1).repeat(channels, 1, 1, 1, 1)
            padding = (0, 0, 0, 0, pad, pad)
        elif axis == 1:
            kernel_3d = kernel.view(1, 1, 1, -1, 1).repeat(channels, 1, 1, 1, 1)
            padding = (0, 0, pad, pad, 0, 0)
        else:
            kernel_3d = kernel.view(1, 1, 1, 1, -1).repeat(channels, 1, 1, 1, 1)
            padding = (pad, pad, 0, 0, 0, 0)
        vol = F.conv3d(F.pad(vol, padding, mode="replicate"), kernel_3d, groups=channels)
    out = vol.squeeze(0)
    return out.squeeze(0) if squeeze else out


def ptv_lcc_sigma_zyx(
    cur_pix_zyx: Sequence[float],
    sigma_mm: float,
    sigma_floor_pix: float,
) -> List[float]:
    """Return historical pTV LCC sigmas in level pixels."""

    return [max(float(sigma_mm) / float(pix), float(sigma_floor_pix)) for pix in cur_pix_zyx]


def make_ptv_lcc_fixed_cache(
    fixed: torch.Tensor,
    cur_pix_zyx: Sequence[float],
    sigma_mm: float,
    radius_sigma: float,
    sigma_floor_pix: float,
    deps: float = 1e-4,
) -> Dict[str, torch.Tensor | List[float]]:
    """Cache fixed-image pTV-style local-correlation statistics."""

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
    deps: float = 1e-4,
) -> torch.Tensor:
    """Evaluate the historical physical-volume-scaled pTV local-CC loss."""

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
    lcc = cross / (std_f * std_w).clamp_min(1e-8)
    loss_map = 1.0 - lcc
    if mask is not None:
        loss_map = loss_map * mask.to(dtype=loss_map.dtype, device=loss_map.device)
    dvol = float(cur_pix_zyx[0]) * float(cur_pix_zyx[1]) * float(cur_pix_zyx[2])
    return loss_map.sum() * dvol


def domain_cps_from_full(full_cps: int, downsample: int) -> int:
    """Map a full-resolution control spacing to the stage loss grid."""

    if int(full_cps) % int(downsample) != 0:
        raise ValueError(f"full_cps={full_cps} must be divisible by downsample={downsample}")
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
    """Evaluate the historical physical isoTV penalty on control coefficients."""

    grid = flat_to_grid(control, control_shape)
    scale_xyz = grid.new_tensor(
        [
            (int(eval_shape[2]) - 1) * float(cur_pix_zyx[2]) / 2.0,
            (int(eval_shape[1]) - 1) * float(cur_pix_zyx[1]) / 2.0,
            (int(eval_shape[0]) - 1) * float(cur_pix_zyx[0]) / 2.0,
        ]
    )
    grid_phys = grid * scale_xyz.view(1, 1, 1, 3)
    kspc_zyx = [float(value) * float(domain_cps) for value in cur_pix_zyx]
    grad_sq = torch.zeros(grid_phys.shape[:3], device=grid_phys.device, dtype=grid_phys.dtype)
    dz = torch.zeros_like(grid_phys)
    dy = torch.zeros_like(grid_phys)
    dx = torch.zeros_like(grid_phys)
    dz[:-1, :, :, :] = (grid_phys[1:, :, :, :] - grid_phys[:-1, :, :, :]) / kspc_zyx[0]
    dy[:, :-1, :, :] = (grid_phys[:, 1:, :, :] - grid_phys[:, :-1, :, :]) / kspc_zyx[1]
    dx[:, :, :-1, :] = (grid_phys[:, :, 1:, :] - grid_phys[:, :, :-1, :]) / kspc_zyx[2]
    grad_sq = grad_sq + dz.square().sum(dim=-1) + dy.square().sum(dim=-1) + dx.square().sum(dim=-1)
    gvol = kspc_zyx[0] * kspc_zyx[1] * kspc_zyx[2]
    return torch.sqrt(grad_sq + float(csqrt)).sum() * float(gvol)


def expand_on_shape(
    control_disp: torch.Tensor,
    control_shape: Sequence[int],
    full_cps: int,
    eval_shape: Sequence[int],
    downsample: int,
) -> torch.Tensor:
    """Expand controls on a stage or final image grid without altering controls."""

    domain_cps = domain_cps_from_full(full_cps, downsample)
    return expand_bspline_controls(
        control_disp,
        tuple(int(value) for value in control_shape),
        tuple(int(value) for value in eval_shape),
        domain_cps,
    )
