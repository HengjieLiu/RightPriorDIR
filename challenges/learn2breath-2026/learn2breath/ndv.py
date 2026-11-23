"""Pure-Torch, source-pinned SITReg non-diffeomorphic-volume loss.

This module is a small, isolated adaptation of the NDV calculation in SITReg
commit ``a665a46a3553b26ade6ff5f57815e5e8ec43a948``:

* ``src/algorithm/ndv.py``
* ``src/loss/regularity.py``
* ``src/algorithm/dense_deformation.py``

It deliberately has no runtime dependency on the SITReg checkout or its
``composable_mapping`` package.  The loss is the raw, unmasked sum of the ten
tetrahedral non-diffeomorphic-volume contributions at the upstream default
``threshold=0.0``.  In particular, it is *not* a normalised percentage, a
central-Jacobian folding count, or the Learn2Breath practical-equivalence
metric.

Tensor conventions
------------------
The local P2e solver emits a dense field as ``[Z, Y, X, 3]`` with normalized
``(x, y, z)`` components for ``grid_sample(..., align_corners=True)``.  Before
SITReg NDV evaluation it must be converted to a voxel displacement field of
shape ``[1, 3, Z, Y, X]`` with components ``(z, y, x)``.  The conversion helper
below performs exactly that operation.  It is intentionally explicit because
leaving the components in ``(x, y, z)`` can turn an axis reflection into a
shear and silently miss folds.
"""

# MIT License
#
# Copyright (c) 2023 Joel Honkamaa
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor
from torch.linalg import det
from torch.nn.functional import conv3d


SITREG_SOURCE_REVISION = "a665a46a3553b26ade6ff5f57815e5e8ec43a948"
"""Immutable upstream revision used for the adapted NDV semantics."""

SITREG_NDV_THRESHOLD = 0.0
"""Pinned upstream NDV threshold; changing it is outside this sweep contract."""

SITREG_NDV_IMPLEMENTATIONS = ("reference", "fast")
"""Supported project-owned implementations of the pinned loss."""

DEFAULT_SITREG_NDV_IMPLEMENTATION = "fast"
"""Default for new repository work; the convolution path remains the reference."""

SITREG_TETRAHEDRON_DIRECTIONS = (
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
)
"""The eight directional and two starred tetrahedra from pinned SITReg."""


class SitregNdvError(ValueError):
    """Raised when a tensor does not satisfy the pinned NDV input contract."""


def _require_spatial_shape(shape: Sequence[int]) -> tuple[int, int, int]:
    """Validate a three-dimensional spatial shape usable by the 3x3x3 stencils."""

    if len(shape) != 3:
        raise SitregNdvError(f"Expected a three-dimensional spatial shape, got {tuple(shape)!r}.")
    normalized = tuple(int(size) for size in shape)
    if any(size < 3 for size in normalized):
        raise SitregNdvError(
            "SITReg's 3D NDV stencils need every spatial dimension to be at least 3; "
            f"got {normalized!r}."
        )
    return normalized


def _require_sitreg_voxel_ddf(ddf_zyx: Tensor) -> tuple[int, int, int]:
    """Validate the voxel-displacement tensor accepted by pinned SITReg NDV."""

    if not isinstance(ddf_zyx, Tensor):
        raise SitregNdvError(f"Expected a torch.Tensor, got {type(ddf_zyx).__name__}.")
    if ddf_zyx.ndim != 5 or ddf_zyx.size(1) != 3:
        raise SitregNdvError(
            "SITReg NDV expects voxel displacements with shape [B,3,Z,Y,X], "
            f"got {tuple(ddf_zyx.shape)!r}."
        )
    if not ddf_zyx.is_floating_point():
        raise SitregNdvError(
            f"SITReg NDV requires a floating-point displacement tensor, got {ddf_zyx.dtype}."
        )
    return _require_spatial_shape(tuple(int(size) for size in ddf_zyx.shape[2:]))


def _as_batched_normalized_zyx_xyz(dense_norm_zyx_xyz: Tensor) -> tuple[Tensor, bool]:
    """Validate local P2e dense coordinates and add a batch dimension if absent."""

    if not isinstance(dense_norm_zyx_xyz, Tensor):
        raise SitregNdvError(f"Expected a torch.Tensor, got {type(dense_norm_zyx_xyz).__name__}.")
    had_batch_dimension = dense_norm_zyx_xyz.ndim == 5
    if dense_norm_zyx_xyz.ndim == 4:
        dense_norm_zyx_xyz = dense_norm_zyx_xyz.unsqueeze(0)
    elif dense_norm_zyx_xyz.ndim != 5:
        raise SitregNdvError(
            "Local normalized dense displacements must have shape [Z,Y,X,3] or [B,Z,Y,X,3], "
            f"got {tuple(dense_norm_zyx_xyz.shape)!r}."
        )
    if dense_norm_zyx_xyz.size(-1) != 3:
        raise SitregNdvError(
            "Local normalized dense displacements must use final component order (x,y,z), "
            f"got shape {tuple(dense_norm_zyx_xyz.shape)!r}."
        )
    if not dense_norm_zyx_xyz.is_floating_point():
        raise SitregNdvError(
            "Local normalized dense displacements must be floating point, "
            f"got {dense_norm_zyx_xyz.dtype}."
        )
    _require_spatial_shape(tuple(int(size) for size in dense_norm_zyx_xyz.shape[-4:-1]))
    return dense_norm_zyx_xyz, had_batch_dimension


def normalized_zyx_xyz_to_sitreg_voxel_zyx(dense_norm_zyx_xyz: Tensor) -> Tensor:
    """Convert local normalized ``[Z,Y,X,xyz]`` data to SITReg voxel ``[B,zyx,Z,Y,X]``.

    The source P2e field is in normalized sampling coordinates with
    ``align_corners=True``.  One normalized unit consequently represents
    ``(size - 1) / 2`` voxels along that component's physical grid axis.  SITReg
    creates its identity grid in array-dimension order, so both components and
    storage axes must be reordered from local ``xyz`` to SITReg ``zyx``.

    An unbatched local field is returned with a singleton batch dimension.  A
    batched local field preserves its batch dimension.
    """

    dense, _ = _as_batched_normalized_zyx_xyz(dense_norm_zyx_xyz)
    z_size, y_size, x_size = (int(size) for size in dense.shape[-4:-1])
    scales_xyz = dense.new_tensor(((x_size - 1) / 2.0, (y_size - 1) / 2.0, (z_size - 1) / 2.0))
    voxel_zyx_xyz = dense * scales_xyz.view(1, 1, 1, 1, 3)
    # ``[..., [2, 1, 0]]`` converts component order xyz -> zyx before
    # ``permute`` changes the storage layout to channel-first [B,3,Z,Y,X].
    return voxel_zyx_xyz[..., [2, 1, 0]].permute(0, 4, 1, 2, 3).contiguous()


def sitreg_voxel_zyx_to_normalized_zyx_xyz(ddf_zyx: Tensor) -> Tensor:
    """Inverse of :func:`normalized_zyx_xyz_to_sitreg_voxel_zyx` for test/QC use.

    The result always retains its batch dimension and has shape ``[B,Z,Y,X,3]``
    with local P2e normalized ``(x,y,z)`` components.  It is not used by the
    optimizer; preserving it here makes the coordinate adapter independently
    testable without importing the frozen retiming module.
    """

    z_size, y_size, x_size = _require_sitreg_voxel_ddf(ddf_zyx)
    scales_xyz = ddf_zyx.new_tensor(((x_size - 1) / 2.0, (y_size - 1) / 2.0, (z_size - 1) / 2.0))
    voxel_zyx_xyz = ddf_zyx.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]
    return (voxel_zyx_xyz / scales_xyz.view(1, 1, 1, 1, 3)).contiguous()


def generate_voxel_coordinate_grid(
    shape_zyx: Sequence[int],
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> Tensor:
    """Return SITReg's ``[1,3,Z,Y,X]`` voxel identity grid in ``(z,y,x)`` order."""

    z_size, y_size, x_size = _require_spatial_shape(shape_zyx)
    axes = [
        torch.linspace(start=0, end=size - 1, steps=size, dtype=dtype, device=device)
        for size in (z_size, y_size, x_size)
    ]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=0).unsqueeze(0)


def calculate_jacobian_determinants(ddf_zyx: Tensor) -> Mapping[str, Tensor]:
    """Calculate the exact ten-tetrahedron determinant maps from pinned SITReg.

    This is a source-level adaptation of ``algorithm.ndv.calculate_jacobian_determinants``.
    The returned maps have shape ``[B,Z-2,Y-2,X-2]``.  Do not replace the two
    starred terms with a central finite-difference determinant: they are part of
    the upstream digital-diffeomorphism definition.
    """

    shape_zyx = _require_sitreg_voxel_ddf(ddf_zyx)
    grid = generate_voxel_coordinate_grid(shape_zyx, dtype=ddf_zyx.dtype, device=ddf_zyx.device)
    trans = ddf_zyx + grid

    kernels: dict[str, Tensor] = {}
    kwargs = {"dtype": trans.dtype, "device": trans.device}
    # The local names x/y/z are retained verbatim from SITReg's implementation.
    # In its tensor convention they refer to the first/second/third *array*
    # axis, whose coordinate components are correspondingly z/y/x here.
    kernels["D0x"] = torch.tensor([-0.5, 0.0, 0.5], **kwargs).view(1, 1, 3, 1, 1)
    kernels["D+x"] = torch.tensor([0.0, -1.0, 1.0], **kwargs).view(1, 1, 3, 1, 1)
    kernels["D-x"] = torch.tensor([-1.0, 1.0, 0.0], **kwargs).view(1, 1, 3, 1, 1)

    kernels["D0y"] = torch.tensor([-0.5, 0.0, 0.5], **kwargs).view(1, 1, 1, 3, 1)
    kernels["D+y"] = torch.tensor([0.0, -1.0, 1.0], **kwargs).view(1, 1, 1, 3, 1)
    kernels["D-y"] = torch.tensor([-1.0, 1.0, 0.0], **kwargs).view(1, 1, 1, 3, 1)

    kernels["D0z"] = torch.tensor([-0.5, 0.0, 0.5], **kwargs).view(1, 1, 1, 1, 3)
    kernels["D+z"] = torch.tensor([0.0, -1.0, 1.0], **kwargs).view(1, 1, 1, 1, 3)
    kernels["D-z"] = torch.tensor([-1.0, 1.0, 0.0], **kwargs).view(1, 1, 1, 1, 3)

    kernels["1*xy"] = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]], **kwargs
    ).reshape(1, 1, 3, 3, 1)
    kernels["1*xz"] = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]], **kwargs
    ).reshape(1, 1, 3, 1, 3)
    kernels["1*yz"] = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]], **kwargs
    ).reshape(1, 1, 1, 3, 3)

    kernels["2*xy"] = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], **kwargs
    ).reshape(1, 1, 3, 3, 1)
    kernels["2*xz"] = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], **kwargs
    ).reshape(1, 1, 3, 1, 3)
    kernels["2*yz"] = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], **kwargs
    ).reshape(1, 1, 1, 3, 3)

    # The concatenation and group ordering match the pinned implementation:
    # each component channel receives the same three directional stencils.
    weights = {
        "x": torch.cat([kernels[key] for key in ("D0x", "D+x", "D-x")] * 3, dim=0),
        "y": torch.cat([kernels[key] for key in ("D0y", "D+y", "D-y")] * 3, dim=0),
        "z": torch.cat([kernels[key] for key in ("D0z", "D+z", "D-z")] * 3, dim=0),
        "*xy": torch.cat([kernels[key] for key in ("1*xy", "2*xy")] * 3, dim=0),
        "*xz": torch.cat([kernels[key] for key in ("1*xz", "2*xz")] * 3, dim=0),
        "*yz": torch.cat([kernels[key] for key in ("1*yz", "2*yz")] * 3, dim=0),
    }

    partials = {
        "x": conv3d(trans, weights["x"], groups=3)[:, :, :, 1:-1, 1:-1],
        "y": conv3d(trans, weights["y"], groups=3)[:, :, 1:-1, :, 1:-1],
        "z": conv3d(trans, weights["z"], groups=3)[:, :, 1:-1, 1:-1, :],
        "*xy": conv3d(trans, weights["*xy"], groups=3)[:, :, :, :, 1:-1],
        "*xz": conv3d(trans, weights["*xz"], groups=3)[:, :, :, 1:-1, :],
        "*yz": conv3d(trans, weights["*yz"], groups=3)[:, :, 1:-1, :, :],
    }

    def _jacobian_from_partials(first: Tensor, second: Tensor, third: Tensor) -> Tensor:
        return torch.stack((first, second, third), dim=-1).permute(0, 2, 3, 4, 1, 5)

    jacobians = {
        "000": _jacobian_from_partials(
            partials["x"][:, ::3], partials["y"][:, ::3], partials["z"][:, ::3]
        ),
        "+++": _jacobian_from_partials(
            partials["x"][:, 1::3], partials["y"][:, 1::3], partials["z"][:, 1::3]
        ),
        "++-": _jacobian_from_partials(
            partials["x"][:, 1::3], partials["y"][:, 1::3], partials["z"][:, 2::3]
        ),
        "+-+": _jacobian_from_partials(
            partials["x"][:, 1::3], partials["y"][:, 2::3], partials["z"][:, 1::3]
        ),
        "+--": _jacobian_from_partials(
            partials["x"][:, 1::3], partials["y"][:, 2::3], partials["z"][:, 2::3]
        ),
        "-++": _jacobian_from_partials(
            partials["x"][:, 2::3], partials["y"][:, 1::3], partials["z"][:, 1::3]
        ),
        "-+-": _jacobian_from_partials(
            partials["x"][:, 2::3], partials["y"][:, 1::3], partials["z"][:, 2::3]
        ),
        "--+": _jacobian_from_partials(
            partials["x"][:, 2::3], partials["y"][:, 2::3], partials["z"][:, 1::3]
        ),
        "---": _jacobian_from_partials(
            partials["x"][:, 2::3], partials["y"][:, 2::3], partials["z"][:, 2::3]
        ),
        "j1*": _jacobian_from_partials(
            partials["*xy"][:, 0::2], partials["*xz"][:, 0::2], partials["*yz"][:, 0::2]
        ),
        "j2*": _jacobian_from_partials(
            partials["*xy"][:, 1::2], partials["*yz"][:, 1::2], partials["*xz"][:, 1::2]
        ),
    }
    return {key: det(value) for key, value in jacobians.items()}


def calculate_raw_ndv_map(jacobian_determinants: Mapping[str, Tensor]) -> Tensor:
    """Return pinned SITReg's unmasked, threshold-zero NDV map.

    Each negative tetrahedron determinant contributes ``-det / 12`` and a
    nonnegative determinant contributes zero.  Summing all ten contributions
    yields a raw volume on the one-voxel interior, with no batch, voxel-count,
    mask, spacing, or practical-equivalence normalisation.
    """

    missing = tuple(
        direction
        for direction in SITREG_TETRAHEDRON_DIRECTIONS
        if direction not in jacobian_determinants
    )
    if missing:
        raise SitregNdvError(f"Missing SITReg NDV determinant directions: {missing!r}.")
    result: Tensor | None = None
    for direction in SITREG_TETRAHEDRON_DIRECTIONS:
        determinant = jacobian_determinants[direction]
        if not isinstance(determinant, Tensor):
            raise SitregNdvError(f"Determinant {direction!r} is not a torch.Tensor.")
        contribution = -0.5 * determinant.clamp(max=SITREG_NDV_THRESHOLD) / 6.0
        result = contribution if result is None else result + contribution
    if result is None:  # Defensive only; the fixed direction tuple is nonempty.
        raise AssertionError("The pinned SITReg direction set unexpectedly became empty.")
    return result


def sitreg_ndv_loss(ddf_zyx: Tensor) -> Tensor:
    """Return source-pinned unmasked raw SITReg NDV for a voxel ``[B,3,Z,Y,X]`` DDF.

    The upstream ``NDVLoss`` calls ``.mean()`` after its volume helper.  That
    helper already returns a scalar sum, so this preserves the exact scalar
    behavior without introducing an accidental batch or voxel normalization.
    """

    ndv = calculate_raw_ndv_map(calculate_jacobian_determinants(ddf_zyx)).sum()
    return ndv.mean()


def sitreg_ndv_loss_from_normalized_zyx_xyz(dense_norm_zyx_xyz: Tensor) -> Tensor:
    """Evaluate raw SITReg NDV directly from a local P2e normalized dense field."""

    return sitreg_ndv_loss(normalized_zyx_xyz_to_sitreg_voxel_zyx(dense_norm_zyx_xyz))


def _fast_identity_vector(ddf_zyx: Tensor, channel: int, value: float) -> Tensor:
    """Return a scalar-sized identity derivative broadcastable over ``ddf_zyx``."""

    result = ddf_zyx.new_zeros((1, 3, 1, 1, 1))
    result[:, channel] = value
    return result


def _fast_jacobian_partials(ddf_zyx: Tensor) -> dict[str, tuple[Tensor, ...]]:
    """Build SITReg's six direct-slice partial families without a coordinate grid.

    The reference implementation applies 3x3x3 convolutions to ``ddf + grid``.
    These expressions are the same finite differences applied directly to the
    displacement, with the identity-coordinate derivative added analytically.
    The names ``x``, ``y`` and ``z`` retain SITReg's array-axis convention:
    first/depth, second/height, and third/width respectively.
    """

    z_size, y_size, x_size = _require_sitreg_voxel_ddf(ddf_zyx)
    del z_size, y_size, x_size
    center = ddf_zyx[:, :, 1:-1, 1:-1, 1:-1]
    identity_x = _fast_identity_vector(ddf_zyx, 0, 1.0)
    identity_y = _fast_identity_vector(ddf_zyx, 1, 1.0)
    identity_z = _fast_identity_vector(ddf_zyx, 2, 1.0)

    x = (
        ddf_zyx[:, :, 2:, 1:-1, 1:-1] - center + identity_x,
        center - ddf_zyx[:, :, :-2, 1:-1, 1:-1] + identity_x,
    )
    y = (
        ddf_zyx[:, :, 1:-1, 2:, 1:-1] - center + identity_y,
        center - ddf_zyx[:, :, 1:-1, :-2, 1:-1] + identity_y,
    )
    z = (
        ddf_zyx[:, :, 1:-1, 1:-1, 2:] - center + identity_z,
        center - ddf_zyx[:, :, 1:-1, 1:-1, :-2] + identity_z,
    )
    star_xy = (
        ddf_zyx[:, :, :-2, :-2, 1:-1] - center - identity_x - identity_y,
        ddf_zyx[:, :, 2:, 2:, 1:-1] - center + identity_x + identity_y,
    )
    star_xz = (
        ddf_zyx[:, :, :-2, 1:-1, :-2] - center - identity_x - identity_z,
        ddf_zyx[:, :, 2:, 1:-1, 2:] - center + identity_x + identity_z,
    )
    star_yz = (
        ddf_zyx[:, :, 1:-1, :-2, :-2] - center - identity_y - identity_z,
        ddf_zyx[:, :, 1:-1, 2:, 2:] - center + identity_y + identity_z,
    )
    return {"x": x, "y": y, "z": z, "*xy": star_xy, "*xz": star_xz, "*yz": star_yz}


def _fast_det3(first: Tensor, second: Tensor, third: Tensor) -> Tensor:
    """Calculate determinants for Jacobian columns in SITReg's ``zyx`` rows."""

    a, d, g = first[:, 0], first[:, 1], first[:, 2]
    b, e, h = second[:, 0], second[:, 1], second[:, 2]
    c, f, i = third[:, 0], third[:, 1], third[:, 2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _fast_direction_determinants(ddf_zyx: Tensor) -> dict[str, Tensor]:
    """Return the ten NDV determinant maps used by the fast implementation."""

    partials = _fast_jacobian_partials(ddf_zyx)
    xp, xm = partials["x"]
    yp, ym = partials["y"]
    zp, zm = partials["z"]
    xy1, xy2 = partials["*xy"]
    xz1, xz2 = partials["*xz"]
    yz1, yz2 = partials["*yz"]
    return {
        "+++": _fast_det3(xp, yp, zp),
        "++-": _fast_det3(xp, yp, zm),
        "+-+": _fast_det3(xp, ym, zp),
        "+--": _fast_det3(xp, ym, zm),
        "j1*": _fast_det3(xy1, xz1, yz1),
        "j2*": _fast_det3(xy2, yz2, xz2),
        "-++": _fast_det3(xm, yp, zp),
        "-+-": _fast_det3(xm, yp, zm),
        "--+": _fast_det3(xm, ym, zp),
        "---": _fast_det3(xm, ym, zm),
    }


def calculate_jacobian_determinants_fast(ddf_zyx: Tensor) -> Mapping[str, Tensor]:
    """Return fast determinant maps for numerical validation.

    Return the ten non-central maps so each tetrahedron can be compared with
    the source-pinned convolution reference.
    """

    return _fast_direction_determinants(ddf_zyx)


def _fast_raw_ndv_sum(ddf_zyx: Tensor) -> Tensor:
    """Sum the negative-volume contributions of the ten determinants."""

    partials = _fast_jacobian_partials(ddf_zyx)
    xp, xm = partials["x"]
    yp, ym = partials["y"]
    zp, zm = partials["z"]
    xy1, xy2 = partials["*xy"]
    xz1, xz2 = partials["*xz"]
    yz1, yz2 = partials["*yz"]
    determinants = (
        _fast_det3(xp, yp, zp),
        _fast_det3(xp, yp, zm),
        _fast_det3(xp, ym, zp),
        _fast_det3(xp, ym, zm),
        _fast_det3(xm, yp, zp),
        _fast_det3(xm, yp, zm),
        _fast_det3(xm, ym, zp),
        _fast_det3(xm, ym, zm),
        _fast_det3(xy1, xz1, yz1),
        _fast_det3(xy2, yz2, xz2),
    )
    result = ddf_zyx.new_zeros(())
    scale = -0.5 / 6.0
    for determinant in determinants:
        result = result + (scale * determinant.clamp(max=SITREG_NDV_THRESHOLD)).sum()
    return result


def sitreg_ndv_loss_fast(ddf_zyx: Tensor) -> Tensor:
    """Return the eager direct-slice implementation of raw SITReg NDV."""

    _require_sitreg_voxel_ddf(ddf_zyx)
    return _fast_raw_ndv_sum(ddf_zyx)


def sitreg_ndv_loss_fast_from_normalized_zyx_xyz(dense_norm_zyx_xyz: Tensor) -> Tensor:
    """Evaluate fast raw SITReg NDV from local normalized P2e coordinates."""

    return sitreg_ndv_loss_fast(normalized_zyx_xyz_to_sitreg_voxel_zyx(dense_norm_zyx_xyz))


def validate_ndv_implementation(value: str) -> str:
    """Validate and return an explicit NDV implementation name."""

    normalized = str(value)
    if normalized not in SITREG_NDV_IMPLEMENTATIONS:
        raise SitregNdvError(
            f"Unknown SITReg NDV implementation {normalized!r}; "
            f"expected one of {SITREG_NDV_IMPLEMENTATIONS!r}."
        )
    return normalized


def sitreg_ndv_loss_for_implementation(
    ddf_zyx: Tensor,
    implementation: str = DEFAULT_SITREG_NDV_IMPLEMENTATION,
) -> Tensor:
    """Dispatch one voxel DDF through the explicitly selected implementation."""

    selected = validate_ndv_implementation(implementation)
    if selected == "reference":
        return sitreg_ndv_loss(ddf_zyx)
    return sitreg_ndv_loss_fast(ddf_zyx)


__all__ = [
    "DEFAULT_SITREG_NDV_IMPLEMENTATION",
    "SITREG_NDV_THRESHOLD",
    "SITREG_NDV_IMPLEMENTATIONS",
    "SITREG_SOURCE_REVISION",
    "SITREG_TETRAHEDRON_DIRECTIONS",
    "SitregNdvError",
    "calculate_jacobian_determinants",
    "calculate_jacobian_determinants_fast",
    "calculate_raw_ndv_map",
    "generate_voxel_coordinate_grid",
    "normalized_zyx_xyz_to_sitreg_voxel_zyx",
    "sitreg_ndv_loss",
    "sitreg_ndv_loss_fast",
    "sitreg_ndv_loss_fast_from_normalized_zyx_xyz",
    "sitreg_ndv_loss_for_implementation",
    "sitreg_ndv_loss_from_normalized_zyx_xyz",
    "sitreg_voxel_zyx_to_normalized_zyx_xyz",
    "validate_ndv_implementation",
]
