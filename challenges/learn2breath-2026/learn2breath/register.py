#!/usr/bin/env python3
"""CT-only MR-D-BSCP runtime for the Learn2Breath hidden-test container.

The public command line is deliberately limited to three positional paths:
``fixed_insp moving_exp output_disp``.  All development-time switches are
namespaced environment variables so freezing the final defaults does not alter
the challenge interface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
import time
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence

import nibabel as nib
import numpy as np
import torch

from . import spline
from .schedule import (
    SCHEDULE,
    ScheduleConfig,
    adaptive_stop_decision,
)
from .ndv import (
    SITREG_SOURCE_REVISION,
    sitreg_ndv_loss_fast_from_normalized_zyx_xyz,
)


RUNTIME_ID = "mrdbscp_submission_runtime_v1"
RUNTIME_DATE = "2026-09-01"
STRATEGIES = ("ndv_tail", "cp4_ndv_adaptive")
CUDNN_MODES = ("reproducible", "throughput")
EXPECTED_SHAPE_XYZ = (224, 192, 224)
EXPECTED_SPACING_XYZ_MM = (1.5, 1.5, 1.5)
GEOMETRY_ATOL = 1.0e-5
SEED = 1

P2E_LCC_SIGMA_MM = 2.1
P2E_LCC_RADIUS_SIGMA = 2.0
P2E_LCC_SIGMA_FLOOR_PIX = 0.8
P2E_ISOTV_WEIGHT = 0.11
P2E_CSQRT = 5.0e-3
P2E_BORDER_VOXELS = 5
HU_LOW = -920.0
HU_HIGH = -100.0


class RuntimeContractError(ValueError):
    """Raised when an invocation violates the CT, geometry, or field contract."""


@dataclass(frozen=True)
class RuntimeConfig:
    """Validated environment-controlled configuration."""

    strategy: str = "ndv_tail"
    ndv_weight: float = 10000.0
    ndv_tail_steps: int = 100
    cudnn_mode: str = "reproducible"

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> "RuntimeConfig":
        env = os.environ if environ is None else environ
        strategy = str(env.get("L2B_STRATEGY", "ndv_tail"))
        if strategy not in STRATEGIES:
            raise RuntimeContractError(
                f"L2B_STRATEGY must be one of {STRATEGIES}, got {strategy!r}"
            )
        try:
            ndv_weight = float(env.get("L2B_NDV_WEIGHT", "10000"))
        except ValueError as exc:
            raise RuntimeContractError(
                "L2B_NDV_WEIGHT must be a finite nonnegative number"
            ) from exc
        if not math.isfinite(ndv_weight) or ndv_weight < 0.0:
            raise RuntimeContractError("L2B_NDV_WEIGHT must be finite and nonnegative")
        tail_text = str(env.get("L2B_NDV_TAIL_STEPS", "100"))
        try:
            tail_steps = int(tail_text, 10)
        except ValueError as exc:
            raise RuntimeContractError("L2B_NDV_TAIL_STEPS must be a positive integer") from exc
        if str(tail_steps) != tail_text.strip() or tail_steps <= 0:
            raise RuntimeContractError("L2B_NDV_TAIL_STEPS must be a canonical positive integer")
        cudnn_mode = str(env.get("L2B_CUDNN_MODE", "reproducible"))
        if cudnn_mode not in CUDNN_MODES:
            raise RuntimeContractError(
                f"L2B_CUDNN_MODE must be one of {CUDNN_MODES}, got {cudnn_mode!r}"
            )
        return cls(strategy, ndv_weight, tail_steps, cudnn_mode)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NiftiGeometry:
    shape_xyz: tuple[int, int, int]
    spacing_xyz_mm: tuple[float, float, float]
    affine: np.ndarray
    orientation: tuple[str, str, str]
    qform: np.ndarray
    qform_code: int
    sform: np.ndarray
    sform_code: int


@dataclass(frozen=True)
class NativePair:
    fixed_xyz_hu: np.ndarray
    moving_xyz_hu: np.ndarray
    fixed_image: nib.Nifti1Image
    fixed_geometry: NiftiGeometry


def _json_print(event: str, **values: Any) -> None:
    payload = {"event": event, **values}
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")), flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stored_scaling(image: nib.spatialimages.SpatialImage) -> tuple[float, float]:
    proxy = image.dataobj
    slope = float(getattr(proxy, "slope", 1.0))
    intercept = float(getattr(proxy, "inter", 0.0))
    return slope, intercept


def _geometry(image: nib.Nifti1Image, *, role: str) -> NiftiGeometry:
    shape = tuple(int(value) for value in image.shape)
    if shape != EXPECTED_SHAPE_XYZ:
        raise RuntimeContractError(f"{role} must have shape {EXPECTED_SHAPE_XYZ}, got {shape}")
    if np.dtype(image.get_data_dtype()) != np.dtype(np.float32):
        raise RuntimeContractError(
            f"{role} NIfTI storage dtype must be float32, got {image.get_data_dtype()}"
        )
    slope, intercept = _stored_scaling(image)
    if not np.isclose(slope, 1.0, rtol=0.0, atol=0.0) or not np.isclose(
        intercept, 0.0, rtol=0.0, atol=0.0
    ):
        raise RuntimeContractError(
            f"{role} must use identity NIfTI scaling, got slope={slope}, intercept={intercept}"
        )
    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    if not np.all(np.isfinite(spacing)) or not np.allclose(
        spacing, EXPECTED_SPACING_XYZ_MM, rtol=0.0, atol=GEOMETRY_ATOL
    ):
        raise RuntimeContractError(
            f"{role} spacing must be {EXPECTED_SPACING_XYZ_MM} mm, got {spacing}"
        )
    affine = np.asarray(image.affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        raise RuntimeContractError(f"{role} effective affine must be finite 4x4")
    affine_spacing = np.linalg.norm(affine[:3, :3], axis=0)
    if not np.allclose(affine_spacing, EXPECTED_SPACING_XYZ_MM, rtol=0.0, atol=GEOMETRY_ATOL):
        raise RuntimeContractError(
            f"{role} affine column spacing must be {EXPECTED_SPACING_XYZ_MM}, got {tuple(affine_spacing)}"
        )
    orientation = tuple(str(value) for value in nib.aff2axcodes(affine))
    if len(orientation) != 3 or any(value == "None" for value in orientation):
        raise RuntimeContractError(f"{role} affine has an invalid orientation: {orientation}")
    qform, qcode = image.get_qform(coded=True)
    sform, scode = image.get_sform(coded=True)
    qcode_int = int(qcode)
    scode_int = int(scode)
    qmatrix = np.asarray(image.get_qform(coded=False), dtype=np.float64)
    smatrix = np.asarray(image.get_sform(coded=False), dtype=np.float64)
    if qcode_int > 0 and (qform is None or not np.all(np.isfinite(qmatrix))):
        raise RuntimeContractError(f"{role} active qform is invalid")
    if scode_int > 0 and (sform is None or not np.all(np.isfinite(smatrix))):
        raise RuntimeContractError(f"{role} active sform is invalid")
    if qcode_int == 0 and scode_int == 0:
        raise RuntimeContractError(
            f"{role} must have at least one active qform or sform definition"
        )
    return NiftiGeometry(
        shape, spacing, affine, orientation, qmatrix, qcode_int, smatrix, scode_int
    )


def _require_matching_geometry(fixed: NiftiGeometry, moving: NiftiGeometry) -> None:
    mismatches: list[str] = []
    if fixed.shape_xyz != moving.shape_xyz:
        mismatches.append("shape")
    if not np.allclose(fixed.spacing_xyz_mm, moving.spacing_xyz_mm, rtol=0.0, atol=GEOMETRY_ATOL):
        mismatches.append("spacing")
    if not np.allclose(fixed.affine, moving.affine, rtol=0.0, atol=GEOMETRY_ATOL):
        mismatches.append("effective affine")
    if fixed.orientation != moving.orientation:
        mismatches.append("orientation")
    if fixed.qform_code != moving.qform_code:
        mismatches.append("qform code")
    elif fixed.qform_code > 0 and not np.allclose(
        fixed.qform, moving.qform, rtol=0.0, atol=GEOMETRY_ATOL
    ):
        mismatches.append("active qform")
    if fixed.sform_code != moving.sform_code:
        mismatches.append("sform code")
    elif fixed.sform_code > 0 and not np.allclose(
        fixed.sform, moving.sform, rtol=0.0, atol=GEOMETRY_ATOL
    ):
        mismatches.append("active sform")
    if mismatches:
        raise RuntimeContractError(
            "Fixed INSP and moving EXP grids do not match: " + ", ".join(mismatches)
        )


def load_native_pair(fixed_path: Path, moving_path: Path) -> NativePair:
    """Load physical-HU CTs without changing their voxel axes or affine."""

    if fixed_path.resolve() == moving_path.resolve():
        raise RuntimeContractError("fixed_insp and moving_exp must be distinct paths")
    try:
        fixed_image = nib.load(str(fixed_path))
        moving_image = nib.load(str(moving_path))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeContractError(f"Could not load input NIfTI: {exc}") from exc
    if type(fixed_image) is not nib.Nifti1Image or type(moving_image) is not nib.Nifti1Image:
        raise RuntimeContractError("Both inputs must be NIfTI-1 images")
    fixed_geometry = _geometry(fixed_image, role="fixed_insp")
    moving_geometry = _geometry(moving_image, role="moving_exp")
    _require_matching_geometry(fixed_geometry, moving_geometry)
    fixed = np.asanyarray(fixed_image.dataobj).astype(np.float32, copy=False)
    moving = np.asanyarray(moving_image.dataobj).astype(np.float32, copy=False)
    if not np.all(np.isfinite(fixed)) or not np.all(np.isfinite(moving)):
        raise RuntimeContractError("Input CTs must contain only finite physical-HU values")
    return NativePair(
        fixed_xyz_hu=np.ascontiguousarray(fixed, dtype=np.float32),
        moving_xyz_hu=np.ascontiguousarray(moving, dtype=np.float32),
        fixed_image=fixed_image,
        fixed_geometry=fixed_geometry,
    )


def normalize_hu_xyz_to_solver_zyx(volume_xyz: np.ndarray) -> np.ndarray:
    """Clip physical HU and transpose storage only; do not reorient or flip."""

    clipped = np.clip(np.asarray(volume_xyz, dtype=np.float32), HU_LOW, HU_HIGH)
    normalized = (clipped - HU_LOW) / (HU_HIGH - HU_LOW)
    return np.ascontiguousarray(normalized.transpose(2, 1, 0), dtype=np.float32)


def normalized_zyx_to_field_cxyz(dense_norm_zyx_xyz: np.ndarray) -> np.ndarray:
    """Convert normalized P2E `[Z,Y,X,xyz]` displacements to xyz voxel CXYZ."""

    dense = np.asarray(dense_norm_zyx_xyz, dtype=np.float32)
    if dense.ndim != 4 or dense.shape[-1] != 3:
        raise RuntimeContractError(f"Expected dense [Z,Y,X,3], got {dense.shape}")
    z_size, y_size, x_size = (int(value) for value in dense.shape[:3])
    scales_xyz = np.asarray(
        [(x_size - 1) / 2.0, (y_size - 1) / 2.0, (z_size - 1) / 2.0], dtype=np.float32
    )
    voxel_zyx_xyz = dense * scales_xyz.reshape(1, 1, 1, 3)
    voxel_xyz_xyz = voxel_zyx_xyz.transpose(2, 1, 0, 3)
    return np.ascontiguousarray(np.moveaxis(voxel_xyz_xyz, -1, 0), dtype=np.float32)


def configure_torch(config: RuntimeConfig) -> dict[str, Any]:
    """Set seeds and the requested cuDNN policy without overstating determinism."""

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    reproducible = config.cudnn_mode == "reproducible"
    torch.backends.cudnn.benchmark = not reproducible
    torch.backends.cudnn.deterministic = reproducible
    return {
        "seed": SEED,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "bitwise_determinism_claimed": False,
        "cuda_grid_sample_backward_may_be_nondeterministic": True,
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _loss_row(
    *,
    stage_index: int,
    stage_epoch: int,
    global_epoch: int,
    full_cps: int,
    downsample: int,
    domain_cps: int,
    loss: torch.Tensor,
    data_loss: torch.Tensor,
    isotv_loss: torch.Tensor,
    raw_ndv: torch.Tensor,
    ndv_weight: float,
    lr: float,
) -> dict[str, Any]:
    return {
        "stage_index": stage_index,
        "stage_epoch": stage_epoch,
        "global_epoch": global_epoch,
        "full_cps_vox": full_cps,
        "downsample": downsample,
        "domain_cps": domain_cps,
        "loss": float(loss.detach().cpu()),
        "data_loss": float(data_loss.detach().cpu()),
        "isotv_loss": float(isotv_loss.detach().cpu()),
        "isotv_weight": P2E_ISOTV_WEIGHT,
        "weighted_isotv_loss": float((P2E_ISOTV_WEIGHT * isotv_loss).detach().cpu()),
        "sitreg_ndv_loss": float(raw_ndv.detach().cpu()),
        "sitreg_ndv_weight": float(ndv_weight),
        "weighted_sitreg_ndv_loss": float((float(ndv_weight) * raw_ndv).detach().cpu()),
        "lr": float(lr),
    }


def optimize_pair(
    pair: NativePair,
    config: RuntimeConfig,
    *,
    device: torch.device,
    schedule: ScheduleConfig = SCHEDULE,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run one cold five-stage registration and the optional fixed CP4 tail."""

    fixed_zyx = normalize_hu_xyz_to_solver_zyx(pair.fixed_xyz_hu)
    moving_zyx = normalize_hu_xyz_to_solver_zyx(pair.moving_xyz_hu)
    fixed = torch.from_numpy(fixed_zyx).to(device=device, dtype=torch.float32)
    moving = torch.from_numpy(moving_zyx).to(device=device, dtype=torch.float32)
    image_shape = tuple(int(value) for value in fixed.shape)
    spacing_zyx = tuple(reversed(pair.fixed_geometry.spacing_xyz_mm))
    border = spline.make_border_mask(image_shape, P2E_BORDER_VOXELS, device, fixed.dtype)
    nested_shapes = spline.build_nested_cp_shapes(
        image_shape, base_spacing=schedule.full_cps[0], spacings=schedule.full_cps
    )
    spline.assert_nested_schedule(nested_shapes, schedule.full_cps)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    started = time.perf_counter()
    previous_control: torch.Tensor | None = None
    previous_shape: tuple[int, int, int] | None = None
    stage_summaries: list[dict[str, Any]] = []
    global_epoch = 0

    for stage_index, (full_cps, downsample, epoch_cap, lr) in enumerate(
        zip(schedule.full_cps, schedule.downsample, schedule.epoch_caps, schedule.lrs)
    ):
        fixed_level, blur_sigma, blur_kernel = spline.make_loss_volume(fixed, downsample, 0.5, 3.0)
        moving_level, _, _ = spline.make_loss_volume(moving, downsample, 0.5, 3.0)
        level_shape = tuple(int(value) for value in fixed_level.shape)
        level_spacing_zyx = spline.stage_pix_zyx(spacing_zyx, image_shape, level_shape)
        level_border = spline.make_loss_mask(border, downsample, 0.5, 3.0)
        if level_border is None:
            raise RuntimeContractError("P2E loss-border construction failed")
        lcc_cache = spline.make_ptv_lcc_fixed_cache(
            fixed_level,
            level_spacing_zyx,
            P2E_LCC_SIGMA_MM,
            P2E_LCC_RADIUS_SIGMA,
            P2E_LCC_SIGMA_FLOOR_PIX,
        )
        control_shape = tuple(int(value) for value in nested_shapes[full_cps])
        control = torch.nn.Parameter(
            torch.zeros(int(np.prod(control_shape)), 3, device=device, dtype=torch.float32)
        )
        if previous_control is not None and previous_shape is not None:
            with torch.no_grad():
                prolonged = spline.cubic_bspline_dyadic_knot_insert_3d(
                    spline.flat_to_grid(previous_control, previous_shape), control_shape
                )
                control.copy_(spline.grid_to_flat(prolonged))
        domain_cps = spline.domain_cps_from_full(full_cps, downsample)
        identity = spline.make_coordinate_grid(level_shape, device).detach()
        optimizer = torch.optim.Adam([control], lr=lr)
        initial_optimizer_state_entries = len(optimizer.state)
        stage_rows: list[dict[str, Any]] = []
        stop_reason = "epoch_cap_reached"
        stage_started = time.perf_counter()
        ndv_weight = (
            config.ndv_weight if config.strategy == "cp4_ndv_adaptive" and stage_index == 4 else 0.0
        )

        for stage_epoch in range(1, epoch_cap + 1):
            dense = spline.expand_on_shape(
                control, control_shape, full_cps, level_shape, downsample
            )
            warped = spline.warp_full(moving_level, identity + dense, mode="bilinear")
            data_loss = spline.ptv_local_cc_loss_with_cache(
                warped, level_border, lcc_cache, level_spacing_zyx, P2E_LCC_RADIUS_SIGMA
            )
            isotv_loss = spline.isotv_control_loss(
                control, control_shape, level_shape, level_spacing_zyx, domain_cps, P2E_CSQRT
            )
            if stage_index == 4 and config.strategy == "cp4_ndv_adaptive":
                raw_ndv = sitreg_ndv_loss_fast_from_normalized_zyx_xyz(dense)
            else:
                raw_ndv = data_loss.new_zeros(())
            loss = data_loss + P2E_ISOTV_WEIGHT * isotv_loss + float(ndv_weight) * raw_ndv
            if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(raw_ndv)):
                raise RuntimeContractError(f"Nonfinite loss at CP{full_cps}, epoch {stage_epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            global_epoch += 1
            row = _loss_row(
                stage_index=stage_index,
                stage_epoch=stage_epoch,
                global_epoch=global_epoch,
                full_cps=full_cps,
                downsample=downsample,
                domain_cps=domain_cps,
                loss=loss,
                data_loss=data_loss,
                isotv_loss=isotv_loss,
                raw_ndv=raw_ndv,
                ndv_weight=ndv_weight,
                lr=lr,
            )
            stage_rows.append(row)
            if stage_epoch == 1 or stage_epoch % 25 == 0 or stage_epoch == epoch_cap:
                _json_print("optimization_step", **row)
            decision = adaptive_stop_decision(stage_rows, schedule, stage_index)
            if decision.stop:
                stop_reason = str(decision.reason)
                break

        _sync(device)
        stage_summaries.append(
            {
                "stage_index": stage_index,
                "full_cps_vox": full_cps,
                "downsample": downsample,
                "domain_cps": domain_cps,
                "control_shape_zyx": list(control_shape),
                "level_shape_zyx": list(level_shape),
                "epoch_cap": epoch_cap,
                "actual_epochs": len(stage_rows),
                "lr": lr,
                "isotv_weight": P2E_ISOTV_WEIGHT,
                "sitreg_ndv_weight": ndv_weight,
                "ndv_active_from_first_update": bool(
                    ndv_weight > 0.0 or (stage_index == 4 and config.strategy == "cp4_ndv_adaptive")
                ),
                "optimizer_initial_state_entries": initial_optimizer_state_entries,
                "stop_reason": stop_reason,
                "elapsed_sec": time.perf_counter() - stage_started,
                "antialias_sigma": blur_sigma,
                "antialias_kernel_size": blur_kernel,
            }
        )
        previous_control = control.detach()
        previous_shape = control_shape

    if previous_control is None or previous_shape is None:
        raise AssertionError("Five-stage optimizer produced no CP4 controls")

    tail_summary: dict[str, Any] | None = None
    final_control = previous_control
    final_shape = previous_shape
    if config.strategy == "ndv_tail":
        full_cps = schedule.full_cps[-1]
        downsample = 1
        control = torch.nn.Parameter(previous_control.clone())
        optimizer = torch.optim.Adam([control], lr=schedule.lrs[-1])
        initial_optimizer_state_entries = len(optimizer.state)
        identity = spline.make_coordinate_grid(image_shape, device).detach()
        lcc_cache = spline.make_ptv_lcc_fixed_cache(
            fixed, spacing_zyx, P2E_LCC_SIGMA_MM, P2E_LCC_RADIUS_SIGMA, P2E_LCC_SIGMA_FLOOR_PIX
        )
        domain_cps = spline.domain_cps_from_full(full_cps, downsample)
        tail_started = time.perf_counter()
        for step in range(1, config.ndv_tail_steps + 1):
            dense = spline.expand_on_shape(control, final_shape, full_cps, image_shape, downsample)
            warped = spline.warp_full(moving, identity + dense, mode="bilinear")
            data_loss = spline.ptv_local_cc_loss_with_cache(
                warped, border, lcc_cache, spacing_zyx, P2E_LCC_RADIUS_SIGMA
            )
            isotv_loss = spline.isotv_control_loss(
                control, final_shape, image_shape, spacing_zyx, domain_cps, P2E_CSQRT
            )
            raw_ndv = sitreg_ndv_loss_fast_from_normalized_zyx_xyz(dense)
            loss = data_loss + P2E_ISOTV_WEIGHT * isotv_loss + config.ndv_weight * raw_ndv
            if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(raw_ndv)):
                raise RuntimeContractError(f"Nonfinite NDV-tail loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            row = _loss_row(
                stage_index=5,
                stage_epoch=step,
                global_epoch=global_epoch + step,
                full_cps=full_cps,
                downsample=downsample,
                domain_cps=domain_cps,
                loss=loss,
                data_loss=data_loss,
                isotv_loss=isotv_loss,
                raw_ndv=raw_ndv,
                ndv_weight=config.ndv_weight,
                lr=schedule.lrs[-1],
            )
            if step == 1 or step % 10 == 0 or step == config.ndv_tail_steps:
                _json_print("optimization_step", **row)
        _sync(device)
        final_control = control.detach()
        tail_summary = {
            "requested_steps": config.ndv_tail_steps,
            "actual_steps": config.ndv_tail_steps,
            "fresh_adam": True,
            "optimizer_initial_state_entries": initial_optimizer_state_entries,
            "full_resolution": True,
            "full_cps_vox": full_cps,
            "lr": schedule.lrs[-1],
            "isotv_weight": P2E_ISOTV_WEIGHT,
            "sitreg_ndv_weight": config.ndv_weight,
            "elapsed_sec": time.perf_counter() - tail_started,
        }

    with torch.no_grad():
        final_dense = spline.expand_on_shape(final_control, final_shape, 4, image_shape, 1)
        final_raw_ndv = sitreg_ndv_loss_fast_from_normalized_zyx_xyz(final_dense)
    _sync(device)
    field = normalized_zyx_to_field_cxyz(final_dense.detach().cpu().numpy())
    if field.shape != (3,) + EXPECTED_SHAPE_XYZ or not np.all(np.isfinite(field)):
        raise RuntimeContractError(f"Terminal field violates CXYZ contract: shape={field.shape}")
    elapsed = time.perf_counter() - started
    peak_bytes = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    return field, {
        "strategy": config.strategy,
        "schedule": schedule.as_dict(),
        "stage_summaries": stage_summaries,
        "tail_summary": tail_summary,
        "cold_zero_cp64_initialization": True,
        "controls_shared_between_subjects": False,
        "anatomical_masks_used": False,
        "segmentations_used": False,
        "landmarks_used": False,
        "corrfield_used": False,
        "final_raw_sitreg_ndv": float(final_raw_ndv.detach().cpu()),
        "elapsed_sec": elapsed,
        "peak_memory_bytes": peak_bytes,
    }


def _check_written_field(reloaded: nib.Nifti1Image, geometry: NiftiGeometry) -> None:
    qform, qcode = reloaded.get_qform(coded=True)
    sform, scode = reloaded.get_sform(coded=True)
    errors: list[str] = []
    if tuple(reloaded.shape) != (3,) + geometry.shape_xyz:
        errors.append("shape")
    if np.dtype(reloaded.get_data_dtype()) != np.dtype(np.float32):
        errors.append("dtype")
    if not np.allclose(reloaded.affine, geometry.affine, rtol=0.0, atol=GEOMETRY_ATOL):
        errors.append("affine")
    if int(qcode) != geometry.qform_code:
        errors.append("qform_code")
    elif int(qcode) > 0 and not np.allclose(qform, geometry.qform, rtol=0.0, atol=GEOMETRY_ATOL):
        errors.append("qform")
    if int(scode) != geometry.sform_code:
        errors.append("sform_code")
    elif int(scode) > 0 and not np.allclose(sform, geometry.sform, rtol=0.0, atol=GEOMETRY_ATOL):
        errors.append("sform")
    if not np.all(np.isfinite(np.asanyarray(reloaded.dataobj))):
        errors.append("finite")
    if errors:
        raise RuntimeContractError(
            "Written output failed internal validation: " + ", ".join(errors)
        )


def write_displacement(
    field_cxyz: np.ndarray, pair: NativePair, output_path: Path
) -> dict[str, Any]:
    """Write one field while preserving the fixed qform/sform definitions."""

    if not output_path.name.lower().endswith((".nii", ".nii.gz")):
        raise RuntimeContractError("Use a single-file .nii or .nii.gz output")
    if output_path.exists():
        raise RuntimeContractError(f"Refusing to overwrite output path: {output_path}")
    if not output_path.parent.is_dir():
        raise RuntimeContractError(f"Output parent directory does not exist: {output_path.parent}")
    field = np.ascontiguousarray(field_cxyz, dtype=np.float32)
    if field.shape != (3,) + pair.fixed_geometry.shape_xyz or not np.all(np.isfinite(field)):
        raise RuntimeContractError(
            f"Output must be finite float32 [3,X,Y,Z], got {field.shape} {field.dtype}"
        )
    header = pair.fixed_image.header.copy()
    header.set_data_dtype(np.float32)
    header.set_data_shape(field.shape)
    header.set_zooms((*pair.fixed_geometry.spacing_xyz_mm, 1.0))
    header.set_slope_inter(1.0, 0.0)
    output = nib.Nifti1Image(field, pair.fixed_geometry.affine, header=header)
    output.set_qform(pair.fixed_geometry.qform, code=pair.fixed_geometry.qform_code)
    output.set_sform(pair.fixed_geometry.sform, code=pair.fixed_geometry.sform_code)
    # Verify the complete file before making it visible under the requested name.
    with TemporaryDirectory(dir=output_path.parent) as directory:
        staged = Path(directory) / output_path.name
        nib.save(output, str(staged))
        _check_written_field(nib.load(str(staged)), pair.fixed_geometry)
        digest = _sha256(staged)
        try:
            os.link(staged, output_path)
        except FileExistsError as exc:
            raise RuntimeContractError(f"Refusing to overwrite output path: {output_path}") from exc
    return {
        "path": str(output_path),
        "sha256": digest,
        "shape": list(field.shape),
        "dtype": "float32",
        "component_order": "xyz",
        "units": "voxel",
        "semantics": "displacement_only_EXP_to_INSP_on_fixed_grid",
        "qform_code": pair.fixed_geometry.qform_code,
        "sform_code": pair.fixed_geometry.sform_code,
    }


def _provenance(config: RuntimeConfig, torch_policy: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "runtime_id": RUNTIME_ID,
        "runtime_date": RUNTIME_DATE,
        "config": config.as_dict(),
        "schedule": SCHEDULE.as_dict(),
        "torch_policy": dict(torch_policy),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "nibabel": nib.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
        "sitreg_source_revision": SITREG_SOURCE_REVISION,
        "sitreg_ndv_implementation": "fast_direct_slice_eager",
        "orientation_policy": "matched_native_grid_no_canonicalization",
        "input_scaling": "physical_HU_identity_NIfTI_scaling",
        "hu_window": [HU_LOW, HU_HIGH],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register one expiratory CT to its paired inspiratory CT."
    )
    parser.add_argument("fixed_insp", help="fixed inspiratory CT (NIfTI-1)")
    parser.add_argument("moving_exp", help="moving expiratory CT (NIfTI-1)")
    parser.add_argument("output_disp", help="new NIfTI displacement field on the fixed grid")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = RuntimeConfig.from_environment()
    torch_policy = configure_torch(config)
    provenance = _provenance(config, torch_policy)
    _json_print("effective_configuration", **provenance)
    if not torch.cuda.is_available():
        raise RuntimeContractError("The submission runtime requires one CUDA GPU")
    pair = load_native_pair(Path(args.fixed_insp), Path(args.moving_exp))
    _json_print(
        "input_contract",
        fixed_path=str(args.fixed_insp),
        moving_path=str(args.moving_exp),
        shape=list(pair.fixed_geometry.shape_xyz),
        spacing_mm=list(pair.fixed_geometry.spacing_xyz_mm),
        orientation=list(pair.fixed_geometry.orientation),
        qform_code=pair.fixed_geometry.qform_code,
        sform_code=pair.fixed_geometry.sform_code,
        matched_native_grid=True,
        canonicalization_applied=False,
    )
    device = torch.device("cuda:0")
    field, optimization = optimize_pair(pair, config, device=device)
    output = write_displacement(field, pair, Path(args.output_disp))
    _json_print("completed", optimization=optimization, output=output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeContractError as exc:
        _json_print("failed", error_type=type(exc).__name__, error=str(exc))
        raise SystemExit(2) from exc


__all__ = [
    "CUDNN_MODES",
    "EXPECTED_SHAPE_XYZ",
    "EXPECTED_SPACING_XYZ_MM",
    "NativePair",
    "RuntimeConfig",
    "RuntimeContractError",
    "SCHEDULE",
    "STRATEGIES",
    "configure_torch",
    "load_native_pair",
    "normalize_hu_xyz_to_solver_zyx",
    "normalized_zyx_to_field_cxyz",
    "optimize_pair",
    "write_displacement",
]
