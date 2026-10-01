"""Audited registration/data kernels; see docs/methods.md for protocol scope."""

from __future__ import annotations
from typing import Any, Callable, Mapping, Sequence
import numpy as np
import torch
import copy
import hashlib
import torch.nn.functional as torch_f
from .dual_helpers import AUTHOR_ADAMW_WEIGHT_DECAY, resolve_device
from .upstream import load_external_api, AUTHOR_EXTERNAL_COMMIT
from . import lung as mr_common

LEGACY_VARIANTS = ("N0", "N1", "P0", "P1")

LUNGMASK_REVISION_VARIANTS = ("PL1D", "PL1B", "PL1A", "PL1C")

VARIANTS = LEGACY_VARIANTS + LUNGMASK_REVISION_VARIANTS

LOSS_BRANCH_DUAL_NATIVE = "dual_native_reg"

LOSS_BRANCH_IDIR_BE = "idir_type_be"

LOSS_BRANCH_PTV_LCC_TV = "ptv_lcc_isotv"

LOSS_BRANCHES = (LOSS_BRANCH_DUAL_NATIVE, LOSS_BRANCH_IDIR_BE, LOSS_BRANCH_PTV_LCC_TV)

SCHEDULE_IMAGE_PYRAMID = "mr_d_bscp_shaped_image_pyramid"

SCHEDULE_AUTHOR_BLUR = "author_original_blur"

FULL_RESOLUTION_FACTORS = (1, 1, 1, 1)

MULTIRESOLUTION_FACTORS = (8, 4, 2, 1)

LOCKED_STAGE_LENGTHS = (400, 400, 800, 900)

AUTHOR_BLUR_SIGMAS = (4.0, 2.0, 0.0)

AUTHOR_BLUR_TOTAL_STEPS = 3000

DEFAULT_IDIR_BE_WEIGHT = 10.0

DEFAULT_PTV_ISOTV_WEIGHT = 0.055

DEFAULT_PTV_LCC_SIGMA_MM = 2.1

DEFAULT_PTV_LCC_RADIUS_SIGMA = 2.0

DEFAULT_PTV_LCC_SIGMA_FLOOR_PIX = 0.8

DEFAULT_PTV_CSQRT = 0.005

DEFAULT_SAMPLED_LCC_CENTERS = 512

DEFAULT_SAMPLED_LCC_OFFSETS = 64

VARIANT_META: dict[str, dict[str, Any]] = {
    "N0": {
        "protocol": "N",
        "image_factors": FULL_RESOLUTION_FACTORS,
        "loss_branch_identifier": LOSS_BRANCH_DUAL_NATIVE,
    },
    "N1": {
        "protocol": "N",
        "image_factors": MULTIRESOLUTION_FACTORS,
        "loss_branch_identifier": LOSS_BRANCH_DUAL_NATIVE,
    },
    "P0": {
        "protocol": "P",
        "image_factors": FULL_RESOLUTION_FACTORS,
        "loss_branch_identifier": LOSS_BRANCH_DUAL_NATIVE,
    },
    "P1": {
        "protocol": "P",
        "image_factors": MULTIRESOLUTION_FACTORS,
        "loss_branch_identifier": LOSS_BRANCH_DUAL_NATIVE,
    },
    "PL1D": {
        "protocol": "PL",
        "image_factors": MULTIRESOLUTION_FACTORS,
        "loss_branch_identifier": LOSS_BRANCH_DUAL_NATIVE,
    },
    "PL1B": {
        "protocol": "PL",
        "image_factors": MULTIRESOLUTION_FACTORS,
        "loss_branch_identifier": LOSS_BRANCH_IDIR_BE,
        "schedule_name": SCHEDULE_IMAGE_PYRAMID,
    },
    "PL1A": {
        "protocol": "PL",
        "image_factors": (1, 1, 1),
        "loss_branch_identifier": LOSS_BRANCH_DUAL_NATIVE,
        "schedule_name": SCHEDULE_AUTHOR_BLUR,
    },
    "PL1C": {
        "protocol": "PL",
        "image_factors": (1, 1, 1),
        "loss_branch_identifier": LOSS_BRANCH_PTV_LCC_TV,
        "schedule_name": SCHEDULE_AUTHOR_BLUR,
    },
}


def variant_protocol(variant: str) -> str:
    variant = str(variant).upper()
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    return str(VARIANT_META[variant]["protocol"])


def variant_factors(variant: str) -> tuple[int, int, int, int]:
    variant = str(variant).upper()
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    return tuple((int(value) for value in VARIANT_META[variant]["image_factors"]))


def variant_loss_branch(variant: str) -> str:
    variant = str(variant).upper()
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    branch = str(VARIANT_META[variant]["loss_branch_identifier"])
    if branch not in LOSS_BRANCHES:
        raise ValueError(f"unknown loss branch for {variant}: {branch!r}")
    return branch


def variant_schedule(variant: str) -> str:
    variant = str(variant).upper()
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    schedule = str(VARIANT_META[variant].get("schedule_name", SCHEDULE_IMAGE_PYRAMID))
    if schedule not in {SCHEDULE_IMAGE_PYRAMID, SCHEDULE_AUTHOR_BLUR}:
        raise ValueError(f"unknown schedule for {variant}: {schedule!r}")
    return schedule


def stage_starts(stage_lengths: Sequence[int]) -> list[int]:
    lengths = [int(value) for value in stage_lengths]
    if not lengths or any((value <= 0 for value in lengths)):
        raise ValueError(f"stage lengths must be positive, got {lengths}")
    starts = [0]
    for length in lengths[:-1]:
        starts.append(starts[-1] + length)
    return starts


def _assert_released_preset(config: Mapping[str, Any]) -> None:
    expected = {
        ("model", "is_dual"): True,
        ("model", "layers"): [3, 256, 256, 3],
        ("model", "coarse_omega0"): 10,
        ("model", "fine_omega0"): 30,
        ("loss", "laplace_weight"): 0.01,
        ("loss", "jacobian_weight"): 0.01,
        ("loss", "fine_reg_scale"): 0.1,
        ("data", "dense_points_per_batch"): 30000,
        ("data", "sampler"): "continuous",
        ("scheduler", "warmup_steps"): 20,
    }
    mismatches: list[str] = []
    for (section, key), expected_value in expected.items():
        actual = config.get(section, {}).get(key)
        if actual != expected_value:
            mismatches.append(
                f"{section}.{key}: expected {expected_value!r}, found {actual!r}"
            )
    if mismatches:
        raise RuntimeError(
            "The pinned DUAL preset no longer matches the controlled-baseline audit:\n- "
            + "\n- ".join(mismatches)
        )


def build_controlled_config(
    variant: str,
    *,
    stage_lengths: Sequence[int] = LOCKED_STAGE_LENGTHS,
    regularization_multiplier: float = 1.0,
    ptv_isotv_weight: float | None = None,
) -> dict[str, Any]:
    """Resolve all settings used by one controlled variant.

    A nonlocked ``stage_lengths`` value exists solely for isolated smoke tests;
    production callers must use the default and record their run mode.
    """
    variant = str(variant).upper()
    multiplier = float(regularization_multiplier)
    if not np.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError(
            f"regularization_multiplier must be finite and positive, got {multiplier!r}"
        )
    factors = list(variant_factors(variant))
    schedule_name = variant_schedule(variant)
    lengths = [int(value) for value in stage_lengths]
    if schedule_name == SCHEDULE_AUTHOR_BLUR:
        total_steps = (
            AUTHOR_BLUR_TOTAL_STEPS
            if tuple(lengths) == LOCKED_STAGE_LENGTHS
            else max(1, int(sum(lengths)))
        )
        lengths = [total_steps]
        starts = [0]
    elif len(lengths) != len(factors) or any((value <= 0 for value in lengths)):
        raise ValueError(
            f"stage_lengths must contain four positive values, got {lengths}"
        )
    else:
        total_steps = int(sum(lengths))
        starts = stage_starts(lengths)
    api = load_external_api()
    config = copy.deepcopy(api.configs.DUAL_CONFIG)
    _assert_released_preset(config)
    config["model"].update(
        {"coarse_scale": 0.1, "fine_scale": 0.05, "fine_branch_is_active": False}
    )
    config["training"].update(
        {
            "total_steps": int(total_steps),
            "fine_start_stage": 1,
            "stage_lengths": lengths,
            "stage_starts": starts,
            "image_factors": factors,
        }
    )
    if schedule_name == SCHEDULE_AUTHOR_BLUR:
        config["training"]["blur_sigmas"] = [
            float(value) for value in AUTHOR_BLUR_SIGMAS
        ]
        config["training"]["blur_convergence_patience"] = 50
        config["training"]["blur_convergence_threshold"] = 0.005
        config["training"]["blur_convergence_threshold_mode"] = "rel"
    ptv_weight = (
        DEFAULT_PTV_ISOTV_WEIGHT
        if ptv_isotv_weight is None
        else float(ptv_isotv_weight)
    )
    if not np.isfinite(ptv_weight) or ptv_weight <= 0.0:
        raise ValueError(
            f"ptv_isotv_weight must be finite and positive, got {ptv_weight!r}"
        )
    config["loss"].update(
        {
            "mask_weight": 0.0,
            "laplace_weight": 0.01 * multiplier,
            "jacobian_weight": 0.01 * multiplier,
            "fine_reg_scale": 0.1,
            "idir_bending_weight": DEFAULT_IDIR_BE_WEIGHT * multiplier,
            "ptv_isotv_weight": ptv_weight,
            "ptv_lcc_sigma_mm": DEFAULT_PTV_LCC_SIGMA_MM,
            "ptv_lcc_radius_sigma": DEFAULT_PTV_LCC_RADIUS_SIGMA,
            "ptv_lcc_sigma_floor_pix": DEFAULT_PTV_LCC_SIGMA_FLOOR_PIX,
            "ptv_csqrt": DEFAULT_PTV_CSQRT,
            "sampled_lcc_centers": DEFAULT_SAMPLED_LCC_CENTERS,
            "sampled_lcc_offsets": DEFAULT_SAMPLED_LCC_OFFSETS,
        }
    )
    loss_branch = variant_loss_branch(variant)
    if loss_branch in {LOSS_BRANCH_IDIR_BE, LOSS_BRANCH_PTV_LCC_TV}:
        config["loss"].update(
            {"laplace_weight": 0.0, "jacobian_weight": 0.0, "fine_reg_scale": 0.0}
        )
    config["optimizer"] = {
        "type": "AdamW",
        "lr": 0.001,
        "end_lr": 1e-05,
        "weight_decay": AUTHOR_ADAMW_WEIGHT_DECAY,
        "betas": [0.9, 0.999],
        "eps": 1e-08,
        "reset_each_stage": True,
    }
    if schedule_name == SCHEDULE_AUTHOR_BLUR:
        config["optimizer"]["reset_each_stage"] = False
    config["scheduler"].update(
        {"type": "cosine", "warmup_steps": 20, "reset_each_stage": True}
    )
    if schedule_name == SCHEDULE_AUTHOR_BLUR:
        config["scheduler"]["reset_each_stage"] = False
        config["scheduler"]["reset_on_blur_change"] = True
    config["controlled_adapter"] = {
        "schema_version": 1,
        "variant": variant,
        "protocol": variant_protocol(variant),
        "loss_branch_identifier": loss_branch,
        "regularization_multiplier": multiplier,
        "resolved_regularization_weights": {
            "coarse_jacobian_weight": float(config["loss"]["jacobian_weight"]),
            "coarse_laplacian_weight": float(config["loss"]["laplace_weight"]),
            "fine_jacobian_weight": float(config["loss"]["jacobian_weight"])
            * float(config["loss"]["fine_reg_scale"]),
            "fine_laplacian_weight": float(config["loss"]["laplace_weight"])
            * float(config["loss"]["fine_reg_scale"]),
            "idir_bending_weight": (
                float(config["loss"]["idir_bending_weight"])
                if loss_branch == LOSS_BRANCH_IDIR_BE
                else 0.0
            ),
            "ptv_isotv_weight": (
                float(config["loss"]["ptv_isotv_weight"])
                if loss_branch == LOSS_BRANCH_PTV_LCC_TV
                else 0.0
            ),
        },
        "external_commit": AUTHOR_EXTERNAL_COMMIT,
        "schedule_name": schedule_name,
        "image_factors": factors,
        "stage_lengths": lengths,
        "stage_starts": starts,
        "blur_sigmas": (
            [float(value) for value in AUTHOR_BLUR_SIGMAS]
            if schedule_name == SCHEDULE_AUTHOR_BLUR
            else []
        ),
        "pyramid_source": (
            "full_resolution_gaussian_blur"
            if schedule_name == SCHEDULE_AUTHOR_BLUR
            else "each_level_independently_from_full_resolution"
        ),
        "antialias_sigma_scale": 0.5,
        "antialias_radius_sigma": 3.0,
        "resize_mode": "trilinear_align_corners_true",
        "sampling_domain_resolution": "full_registration_grid_at_every_stage",
        "model_state_policy": "persist_across_stages",
        "optimizer_state_policy": (
            "single_adamw_persistent_moments"
            if schedule_name == SCHEDULE_AUTHOR_BLUR
            else "fresh_adamw_each_stage"
        ),
        "scheduler_state_policy": (
            "reset_on_blur_change"
            if schedule_name == SCHEDULE_AUTHOR_BLUR
            else "fresh_warmup_cosine_each_stage"
        ),
        "stage_policy": (
            "author_dynamic_convergence"
            if schedule_name == SCHEDULE_AUTHOR_BLUR
            else "fixed_stage_lengths"
        ),
        "loss_scope": (
            "sampled_global_ncc_plus_released_spatial_regularization"
            if loss_branch == LOSS_BRANCH_DUAL_NATIVE
            else (
                "sampled_ptv_lcc_plus_sampled_physical_isotv"
                if loss_branch == LOSS_BRANCH_PTV_LCC_TV
                else "sampled_global_ncc_plus_idir_coordinate_autograd_bending_energy"
            )
        ),
        "semantic_mask_loss_present": False,
    }
    return config


def _validate_arrays(
    fixed_image_xyz: np.ndarray,
    moving_image_xyz: np.ndarray,
    sampling_mask_xyz: np.ndarray,
) -> None:
    arrays = {
        "fixed_image_xyz": np.asarray(fixed_image_xyz),
        "moving_image_xyz": np.asarray(moving_image_xyz),
        "sampling_mask_xyz": np.asarray(sampling_mask_xyz),
    }
    if arrays["fixed_image_xyz"].ndim != 3:
        raise ValueError("fixed_image_xyz must have shape (X,Y,Z)")
    expected = arrays["fixed_image_xyz"].shape
    for name, array in arrays.items():
        if array.shape != expected:
            raise ValueError(f"{name} shape {array.shape} does not match {expected}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains non-finite values")
    if min(expected) < 2:
        raise ValueError(f"all image dimensions must be >=2, got {expected}")
    mask = arrays["sampling_mask_xyz"] > 0
    if not bool(mask.any()):
        raise ValueError("sampling mask is empty")


def _branch_regularization_components(
    *,
    regularizers_module: Any,
    coords: torch.Tensor,
    output: torch.Tensor,
    jacobian_weight: float,
    laplace_weight: float,
    branch_scale: float,
) -> dict[str, torch.Tensor]:
    (determinant, laplacian) = regularizers_module.compute_jacobian_and_laplacian(
        coords, output, add_identity=True
    )
    jacobian_raw = torch_f.relu(-determinant).mean()
    laplacian_raw = (laplacian**2).mean()
    return {
        "jacobian_raw": jacobian_raw,
        "laplacian_raw": laplacian_raw,
        "jacobian_weighted": branch_scale * jacobian_weight * jacobian_raw,
        "laplacian_weighted": branch_scale * laplace_weight * laplacian_raw,
    }


def _number(value: torch.Tensor | float) -> float:
    return (
        float(value.detach().item())
        if isinstance(value, torch.Tensor)
        else float(value)
    )


def _coordinate_fingerprint(coords: torch.Tensor) -> str:
    sample = coords.detach()[: min(512, int(coords.shape[0]))].cpu().numpy()
    return hashlib.sha256(sample.astype(np.float32, copy=False).tobytes()).hexdigest()


def _xyz_volume_to_zyx(volume_xyz: torch.Tensor) -> torch.Tensor:
    if volume_xyz.ndim != 3:
        raise ValueError(f"expected a 3D XYZ volume, got {tuple(volume_xyz.shape)}")
    return volume_xyz.permute(2, 1, 0).contiguous()


def _make_stage_volume(
    full_volume: torch.Tensor, factor: int, *, sigma_scale: float, radius_sigma: float
) -> tuple[torch.Tensor, float, int]:
    return mr_common.make_loss_volume(
        full_volume,
        factor=int(factor),
        sigma_scale=float(sigma_scale),
        radius_sigma=float(radius_sigma),
    )


def _train_controlled_author_blur(
    *,
    fixed_image_xyz: np.ndarray,
    moving_image_xyz: np.ndarray,
    sampling_mask_xyz: np.ndarray,
    config: dict[str, Any],
    device: str | int | torch.device,
    record_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    api = load_external_api()
    device_t = device if isinstance(device, torch.device) else resolve_device(device)
    variant = str(config["controlled_adapter"]["variant"]).upper()
    if variant_schedule(variant) != SCHEDULE_AUTHOR_BLUR:
        raise ValueError(
            "_train_controlled_author_blur requires an author-blur variant"
        )
    if variant_loss_branch(variant) not in {
        LOSS_BRANCH_DUAL_NATIVE,
        LOSS_BRANCH_PTV_LCC_TV,
    }:
        raise ValueError("author-blur controlled variants require native or pTV loss")
    trainer = api.trainer.DirINR(config=config)
    trainer._cached_mask_indices = {}
    model = trainer._build_model(config).to(device_t)
    model.train()
    fixed_full = torch.as_tensor(
        np.asarray(fixed_image_xyz, dtype=np.float32), device=device_t
    )
    moving_full = torch.as_tensor(
        np.asarray(moving_image_xyz, dtype=np.float32), device=device_t
    )
    sampling_mask = torch.as_tensor(
        np.asarray(sampling_mask_xyz) > 0, device=device_t, dtype=torch.float32
    )
    full_shape = tuple((int(value) for value in fixed_full.shape))
    opt_cfg = config["optimizer"]
    lr = float(opt_cfg["lr"])
    end_lr = float(opt_cfg["end_lr"])
    weight_decay = float(opt_cfg["weight_decay"])
    betas = tuple((float(value) for value in opt_cfg.get("betas", (0.9, 0.999))))
    eps = float(opt_cfg.get("eps", 1e-08))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps
    )
    training_cfg = config["training"]
    blur_sigmas = trainer._resolve_blur_sigmas(config)
    if [float(value) for value in blur_sigmas] != [
        float(value) for value in AUTHOR_BLUR_SIGMAS
    ]:
        raise ValueError(
            f"author-blur schedule requires {AUTHOR_BLUR_SIGMAS}, got {blur_sigmas}"
        )
    total_steps = int(training_cfg["total_steps"])
    fine_start_stage = min(
        max(int(training_cfg.get("fine_start_stage", 1)), 0), len(blur_sigmas) - 1
    )
    scheduler_type = str(config["scheduler"].get("type", "cosine")).lower()
    warmup_steps = int(config["scheduler"]["warmup_steps"])
    reset_scheduler = bool(config["scheduler"].get("reset_on_blur_change", True))
    points_per_step = int(config["data"]["dense_points_per_batch"])
    sampler = str(config["data"]["sampler"])
    loss_cfg = config["loss"]
    if float(loss_cfg.get("mask_weight", 0.0)) != 0.0:
        raise ValueError("controlled author-blur variants prohibit semantic mask loss")
    jacobian_weight = float(loss_cfg["jacobian_weight"])
    laplace_weight = float(loss_cfg["laplace_weight"])
    fine_reg_scale = float(loss_cfg["fine_reg_scale"])
    loss_branch = str(config["controlled_adapter"].get("loss_branch_identifier", ""))
    if loss_branch not in LOSS_BRANCHES:
        raise ValueError(f"unknown loss branch identifier: {loss_branch!r}")
    ptv_isotv_weight = float(loss_cfg.get("ptv_isotv_weight", DEFAULT_PTV_ISOTV_WEIGHT))
    ptv_lcc_sigma_mm = float(loss_cfg.get("ptv_lcc_sigma_mm", DEFAULT_PTV_LCC_SIGMA_MM))
    ptv_lcc_radius_sigma = float(
        loss_cfg.get("ptv_lcc_radius_sigma", DEFAULT_PTV_LCC_RADIUS_SIGMA)
    )
    ptv_lcc_sigma_floor_pix = float(
        loss_cfg.get("ptv_lcc_sigma_floor_pix", DEFAULT_PTV_LCC_SIGMA_FLOOR_PIX)
    )
    ptv_csqrt = float(loss_cfg.get("ptv_csqrt", DEFAULT_PTV_CSQRT))
    sampled_lcc_centers = int(
        loss_cfg.get("sampled_lcc_centers", DEFAULT_SAMPLED_LCC_CENTERS)
    )
    sampled_lcc_offsets = int(
        loss_cfg.get("sampled_lcc_offsets", DEFAULT_SAMPLED_LCC_OFFSETS)
    )
    if loss_branch == LOSS_BRANCH_PTV_LCC_TV:
        if ptv_isotv_weight <= 0.0:
            raise ValueError("ptv_lcc_isotv requires a positive ptv_isotv_weight")
        if sampled_lcc_centers <= 0 or sampled_lcc_offsets <= 0:
            raise ValueError("sampled pTV LCC requires positive centers and offsets")
    full_shape_zyx = tuple((int(value) for value in full_shape[::-1]))
    ptv_offsets_norm_xyz: torch.Tensor | None = None
    ptv_offset_weights: torch.Tensor | None = None
    ptv_offset_meta: dict[str, object] | None = None
    if loss_branch == LOSS_BRANCH_PTV_LCC_TV:
        (ptv_offsets_norm_xyz, ptv_offset_weights, ptv_offset_meta) = (
            mr_common.make_sampled_lcc_offsets(
                full_shape_zyx,
                (1.0, 1.0, 1.0),
                ptv_lcc_sigma_mm,
                ptv_lcc_radius_sigma,
                ptv_lcc_sigma_floor_pix,
                sampled_lcc_offsets,
                device_t,
                fixed_full.dtype,
            )
        )
        config["controlled_adapter"]["sampled_ptv_lcc_offset_meta"] = ptv_offset_meta
        config["controlled_adapter"]["sampled_ptv_lcc_domain_voxels"] = int(
            sampling_mask.sum().detach().item()
        )
    convergence_checker = api.scheduler.ConvergenceChecker(
        patience=int(training_cfg.get("blur_convergence_patience", 50)),
        threshold=float(training_cfg.get("blur_convergence_threshold", 0.005)),
        threshold_mode=str(training_cfg.get("blur_convergence_threshold_mode", "rel")),
    )
    stage_index = 0
    sigma = float(blur_sigmas[stage_index])
    (fixed_stage, moving_stage) = trainer._get_blurred_images(
        fixed_image=fixed_full, moving_image=moving_full, sigma=sigma
    )
    scheduler = trainer._build_scheduler(
        optimizer=optimizer,
        scheduler_type=scheduler_type,
        warmup_steps=warmup_steps,
        n_steps_stage=total_steps,
        end_lr=end_lr,
    )
    history: list[dict[str, Any]] = []
    stage_records: list[dict[str, Any]] = [
        {
            "stage_index": 0,
            "start_step": 0,
            "blur_sigma": sigma,
            "image_factor": 1,
            "stage_shape_xyz": list(full_shape),
            "fine_active_at_stage_start": False,
            "loss_branch_identifier": loss_branch,
            "optimizer_reset": False,
            "scheduler_reset": False,
            "transition_trigger": "initial",
            "coordinate_fingerprint_sha256": None,
        }
    ]
    fine_activation_step: int | None = None
    for zero_step in range(total_steps):
        if trainer.is_dual and stage_index >= fine_start_stage:
            if not model.fine_branch_is_active:
                model.turn_on_fine_branch()
                fine_activation_step = zero_step
                stage_records[-1]["fine_active_at_stage_start"] = True
        optimizer.zero_grad(None)
        coords = trainer._sample_dense_batch(
            sampling_mask=sampling_mask,
            image_shape=full_shape,
            sampler=sampler,
            n_points=(
                sampled_lcc_centers
                if loss_branch == LOSS_BRANCH_PTV_LCC_TV
                else points_per_step
            ),
            device=device_t,
            cache_key="full_resolution_sampling_domain",
        ).requires_grad_(True)
        if stage_records[-1]["coordinate_fingerprint_sha256"] is None:
            stage_records[-1]["coordinate_fingerprint_sha256"] = (
                _coordinate_fingerprint(coords)
            )
        (coarse, fine, total) = trainer._predict_displacement(
            model=model, coords=coords
        )
        zero = total.new_tensor(0.0)
        coarse_reg = {
            "jacobian_raw": zero,
            "laplacian_raw": zero,
            "jacobian_weighted": zero,
            "laplacian_weighted": zero,
        }
        fine_reg = dict(coarse_reg)
        ptv_isotv_raw = zero
        ptv_isotv_weighted = zero
        if loss_branch == LOSS_BRANCH_PTV_LCC_TV:
            if ptv_offsets_norm_xyz is None or ptv_offset_weights is None:
                raise RuntimeError("sampled pTV LCC offsets were not initialized")
            fixed_stage_zyx = _xyz_volume_to_zyx(fixed_stage)
            moving_stage_zyx = _xyz_volume_to_zyx(moving_stage)
            patch_coords = coords[:, None, :] + ptv_offsets_norm_xyz[None, :, :]
            patch_flat = patch_coords.reshape(-1, 3)
            (patch_coarse, patch_fine, patch_total) = trainer._predict_displacement(
                model=model, coords=patch_flat
            )
            del patch_coarse, patch_fine
            patch_disp = patch_total.reshape(
                coords.shape[0], int(ptv_offsets_norm_xyz.shape[0]), 3
            )
            similarity = mr_common.sampled_ptv_local_cc_loss(
                fixed_stage_zyx,
                moving_stage_zyx,
                patch_coords,
                patch_disp,
                ptv_offset_weights,
                int(config["controlled_adapter"]["sampled_ptv_lcc_domain_voxels"]),
                (1.0, 1.0, 1.0),
            )
            ptv_isotv_raw = mr_common.idir_sampled_tv_physical(
                coords,
                total,
                full_shape_zyx,
                (1.0, 1.0, 1.0),
                ptv_csqrt,
                domain_voxels=int(
                    config["controlled_adapter"]["sampled_ptv_lcc_domain_voxels"]
                ),
            )
            ptv_isotv_weighted = ptv_isotv_weight * ptv_isotv_raw
            spatial = ptv_isotv_weighted
        else:
            fixed_voxels = api.utils.get_interpolation_coords(coords, full_shape)
            moving_voxels = api.utils.get_interpolation_coords(
                coords + total, full_shape
            )
            fixed_values = api.utils.trilinear_interpolation(fixed_stage, *fixed_voxels)
            moving_values = api.utils.trilinear_interpolation(
                moving_stage, *moving_voxels
            )
            similarity = api.ncc.lncc(
                moving_values.view(1, -1), fixed_values.view(1, -1)
            )
            if jacobian_weight > 0.0 or laplace_weight > 0.0:
                coarse_reg = _branch_regularization_components(
                    regularizers_module=api.regularizers,
                    coords=coords,
                    output=coarse,
                    jacobian_weight=jacobian_weight,
                    laplace_weight=laplace_weight,
                    branch_scale=1.0,
                )
                if fine is not None:
                    fine_reg = _branch_regularization_components(
                        regularizers_module=api.regularizers,
                        coords=coords,
                        output=fine,
                        jacobian_weight=jacobian_weight,
                        laplace_weight=laplace_weight,
                        branch_scale=fine_reg_scale,
                    )
            spatial = sum(
                (
                    coarse_reg["jacobian_weighted"],
                    coarse_reg["laplacian_weighted"],
                    fine_reg["jacobian_weighted"],
                    fine_reg["laplacian_weighted"],
                ),
                start=zero,
            )
        total_loss = similarity + spatial
        finite = {
            "similarity": similarity,
            "spatial": spatial,
            "total": total_loss,
            "coarse_jacobian_raw": coarse_reg["jacobian_raw"],
            "coarse_laplacian_raw": coarse_reg["laplacian_raw"],
            "fine_jacobian_raw": fine_reg["jacobian_raw"],
            "fine_laplacian_raw": fine_reg["laplacian_raw"],
            "ptv_isotv_raw": ptv_isotv_raw,
        }
        nonfinite = {
            name: _number(value)
            for (name, value) in finite.items()
            if not bool(torch.isfinite(value).all().item())
        }
        if nonfinite:
            raise FloatingPointError(
                f"non-finite loss at step {zero_step}, stage {stage_index}: {nonfinite}"
            )
        lr_before_update = float(optimizer.param_groups[0]["lr"])
        total_loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
            if scheduler._step_count >= scheduler.t_total:
                scheduler = None
        lr_after_schedule = float(optimizer.param_groups[0]["lr"])
        record = {
            "zero_step": zero_step,
            "step": zero_step + 1,
            "stage_index": stage_index,
            "stage_local_step": zero_step - int(stage_records[-1]["start_step"]),
            "image_factor": 1,
            "blur_sigma": sigma,
            "stage_shape_x": full_shape[0],
            "stage_shape_y": full_shape[1],
            "stage_shape_z": full_shape[2],
            "antialias_sigma_vox": 0.0,
            "fine_active": bool(model.fine_branch_is_active),
            "loss_total": _number(total_loss),
            "loss_similarity": _number(similarity),
            "loss_spatial_weighted": _number(spatial),
            "loss_mask_weighted": 0.0,
            "coarse_jacobian_raw": _number(coarse_reg["jacobian_raw"]),
            "coarse_jacobian_weighted": _number(coarse_reg["jacobian_weighted"]),
            "coarse_laplacian_raw": _number(coarse_reg["laplacian_raw"]),
            "coarse_laplacian_weighted": _number(coarse_reg["laplacian_weighted"]),
            "fine_jacobian_raw": _number(fine_reg["jacobian_raw"]),
            "fine_jacobian_weighted": _number(fine_reg["jacobian_weighted"]),
            "fine_laplacian_raw": _number(fine_reg["laplacian_raw"]),
            "fine_laplacian_weighted": _number(fine_reg["laplacian_weighted"]),
            "idir_bending_raw": 0.0,
            "idir_bending_weighted": 0.0,
            "ptv_lcc_raw": (
                _number(similarity) if loss_branch == LOSS_BRANCH_PTV_LCC_TV else 0.0
            ),
            "ptv_isotv_raw": _number(ptv_isotv_raw),
            "ptv_isotv_weighted": _number(ptv_isotv_weighted),
            "ptv_isotv_weight": (
                ptv_isotv_weight if loss_branch == LOSS_BRANCH_PTV_LCC_TV else 0.0
            ),
            "sampled_lcc_centers": (
                int(coords.shape[0]) if loss_branch == LOSS_BRANCH_PTV_LCC_TV else 0
            ),
            "sampled_lcc_offsets": (
                int(ptv_offsets_norm_xyz.shape[0])
                if ptv_offsets_norm_xyz is not None
                else 0
            ),
            "loss_branch_identifier": loss_branch,
            "idir_bending_weight": 0.0,
            "learning_rate_before_update": lr_before_update,
            "learning_rate_after_schedule": lr_after_schedule,
            "coarse_scale": float(model.coarse_scale.detach().item()),
            "fine_scale": float(model.fine_scale.detach().item()),
        }
        history.append(record)
        if record_callback is not None:
            record_callback(record)
        if (
            sigma > 0.0
            and stage_index < len(blur_sigmas) - 1
            and convergence_checker.update(record["loss_total"])
        ):
            convergence_checker.reset()
            stage_records[-1]["end_step_exclusive"] = zero_step + 1
            previous_stage = stage_index
            stage_index += 1
            sigma = float(blur_sigmas[stage_index])
            (fixed_stage, moving_stage) = trainer._get_blurred_images(
                fixed_image=fixed_full, moving_image=moving_full, sigma=sigma
            )
            if reset_scheduler:
                remaining_steps = total_steps - (zero_step + 1)
                scheduler = (
                    None
                    if remaining_steps <= 0
                    else trainer._build_scheduler(
                        optimizer=optimizer,
                        scheduler_type=scheduler_type,
                        warmup_steps=warmup_steps,
                        n_steps_stage=remaining_steps,
                        end_lr=end_lr,
                    )
                )
            stage_records.append(
                {
                    "stage_index": stage_index,
                    "start_step": zero_step + 1,
                    "blur_sigma": sigma,
                    "image_factor": 1,
                    "stage_shape_xyz": list(full_shape),
                    "fine_active_at_stage_start": bool(model.fine_branch_is_active),
                    "loss_branch_identifier": loss_branch,
                    "optimizer_reset": False,
                    "scheduler_reset": bool(reset_scheduler),
                    "transition_trigger": "author_dynamic_convergence",
                    "from_stage": previous_stage,
                    "coordinate_fingerprint_sha256": None,
                }
            )
    stage_records[-1]["end_step_exclusive"] = total_steps
    for stage in stage_records:
        stage["length"] = int(stage["end_step_exclusive"]) - int(stage["start_step"])
    if len(stage_records) != len(AUTHOR_BLUR_SIGMAS):
        raise RuntimeError(
            f"author blur schedule visited {len(stage_records)} stages; expected {len(AUTHOR_BLUR_SIGMAS)}"
        )
    return {
        "model": model,
        "history": history,
        "stages": stage_records,
        "fine_activation_step": fine_activation_step,
        "optimizer_settings": copy.deepcopy(config["optimizer"]),
        "scheduler_settings": copy.deepcopy(config["scheduler"]),
    }


def train_controlled(
    *,
    fixed_image_xyz: np.ndarray,
    moving_image_xyz: np.ndarray,
    sampling_mask_xyz: np.ndarray,
    config: dict[str, Any],
    device: str | int | torch.device,
    record_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Train one fixed-stage controlled DUAL model.

    The sampling mask is used only to draw full-domain coordinates.  No
    semantic or moving-image mask is accepted by this API, which makes the P
    no-lung-mask optimization contract structurally enforceable.
    """
    _validate_arrays(fixed_image_xyz, moving_image_xyz, sampling_mask_xyz)
    variant = str(config["controlled_adapter"]["variant"]).upper()
    if variant_schedule(variant) == SCHEDULE_AUTHOR_BLUR:
        return _train_controlled_author_blur(
            fixed_image_xyz=fixed_image_xyz,
            moving_image_xyz=moving_image_xyz,
            sampling_mask_xyz=sampling_mask_xyz,
            config=config,
            device=device,
            record_callback=record_callback,
        )
    api = load_external_api()
    device_t = device if isinstance(device, torch.device) else resolve_device(device)
    expected_factors = list(variant_factors(variant))
    factors = [int(value) for value in config["training"]["image_factors"]]
    if factors != expected_factors:
        raise ValueError(
            f"variant {variant} requires factors {expected_factors}, got {factors}"
        )
    lengths = [int(value) for value in config["training"]["stage_lengths"]]
    starts = stage_starts(lengths)
    if starts != [int(value) for value in config["training"]["stage_starts"]]:
        raise ValueError("resolved stage starts do not match stage lengths")
    trainer = api.trainer.DirINR(config=config)
    trainer._cached_mask_indices = {}
    model = trainer._build_model(config).to(device_t)
    model.train()
    fixed_full = torch.as_tensor(
        np.asarray(fixed_image_xyz, dtype=np.float32), device=device_t
    )
    moving_full = torch.as_tensor(
        np.asarray(moving_image_xyz, dtype=np.float32), device=device_t
    )
    sampling_mask = torch.as_tensor(
        np.asarray(sampling_mask_xyz) > 0, device=device_t, dtype=torch.float32
    )
    full_shape = tuple((int(value) for value in fixed_full.shape))
    opt_cfg = config["optimizer"]
    lr = float(opt_cfg["lr"])
    end_lr = float(opt_cfg["end_lr"])
    weight_decay = float(opt_cfg["weight_decay"])
    betas = tuple((float(value) for value in opt_cfg.get("betas", (0.9, 0.999))))
    eps = float(opt_cfg.get("eps", 1e-08))
    warmup_steps = int(config["scheduler"]["warmup_steps"])
    scheduler_type = str(config["scheduler"].get("type", "cosine"))
    fine_start_stage = int(config["training"].get("fine_start_stage", 1))
    points_per_step = int(config["data"]["dense_points_per_batch"])
    sampler = str(config["data"]["sampler"])
    sigma_scale = float(config["controlled_adapter"]["antialias_sigma_scale"])
    radius_sigma = float(config["controlled_adapter"]["antialias_radius_sigma"])
    loss_cfg = config["loss"]
    if float(loss_cfg.get("mask_weight", 0.0)) != 0.0:
        raise ValueError("controlled variants prohibit semantic mask loss")
    jacobian_weight = float(loss_cfg["jacobian_weight"])
    laplace_weight = float(loss_cfg["laplace_weight"])
    fine_reg_scale = float(loss_cfg["fine_reg_scale"])
    idir_bending_weight = float(loss_cfg.get("idir_bending_weight", 0.0))
    loss_branch = str(config["controlled_adapter"].get("loss_branch_identifier", ""))
    if loss_branch not in LOSS_BRANCHES:
        raise ValueError(f"unknown loss branch identifier: {loss_branch!r}")
    if loss_branch == LOSS_BRANCH_IDIR_BE and idir_bending_weight <= 0.0:
        raise ValueError("idir_type_be requires a positive idir_bending_weight")
    history: list[dict[str, Any]] = []
    stage_records: list[dict[str, Any]] = []
    global_step = 0
    fine_activation_step: int | None = None
    for stage_index, (factor, stage_length, start) in enumerate(
        zip(factors, lengths, starts)
    ):
        if start != global_step:
            raise RuntimeError(
                f"stage {stage_index} starts at {start}, current {global_step}"
            )
        if trainer.is_dual and stage_index >= fine_start_stage:
            if not model.fine_branch_is_active:
                model.turn_on_fine_branch()
                fine_activation_step = global_step
        (fixed_stage, blur_sigma, blur_kernel) = _make_stage_volume(
            fixed_full, factor, sigma_scale=sigma_scale, radius_sigma=radius_sigma
        )
        (moving_stage, moving_sigma, moving_kernel) = _make_stage_volume(
            moving_full, factor, sigma_scale=sigma_scale, radius_sigma=radius_sigma
        )
        if fixed_stage.shape != moving_stage.shape:
            raise RuntimeError("fixed/moving pyramid level shape mismatch")
        if blur_sigma != moving_sigma or blur_kernel != moving_kernel:
            raise RuntimeError("fixed/moving antialias settings diverged")
        stage_shape = tuple((int(value) for value in fixed_stage.shape))
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps
        )
        scheduler = trainer._build_scheduler(
            optimizer=optimizer,
            scheduler_type=scheduler_type,
            warmup_steps=warmup_steps,
            n_steps_stage=int(stage_length),
            end_lr=end_lr,
        )
        stage_record: dict[str, Any] = {
            "stage_index": stage_index,
            "start_step": global_step,
            "end_step_exclusive": global_step + int(stage_length),
            "length": int(stage_length),
            "image_factor": int(factor),
            "full_shape_xyz": list(full_shape),
            "stage_shape_xyz": list(stage_shape),
            "antialias_sigma_vox": float(blur_sigma),
            "antialias_kernel_size": int(blur_kernel),
            "fine_active_at_stage_start": bool(model.fine_branch_is_active),
            "loss_branch_identifier": loss_branch,
            "optimizer_reset": True,
            "scheduler_reset": True,
            "coordinate_fingerprint_sha256": None,
        }
        for local_step in range(int(stage_length)):
            optimizer.zero_grad(None)
            coords = trainer._sample_dense_batch(
                sampling_mask=sampling_mask,
                image_shape=full_shape,
                sampler=sampler,
                n_points=points_per_step,
                device=device_t,
                cache_key="full_resolution_sampling_domain",
            ).requires_grad_(True)
            if local_step == 0:
                stage_record["coordinate_fingerprint_sha256"] = _coordinate_fingerprint(
                    coords
                )
            (coarse, fine, total) = trainer._predict_displacement(
                model=model, coords=coords
            )
            fixed_voxels = api.utils.get_interpolation_coords(coords, stage_shape)
            moving_voxels = api.utils.get_interpolation_coords(
                coords + total, stage_shape
            )
            fixed_values = api.utils.trilinear_interpolation(fixed_stage, *fixed_voxels)
            moving_values = api.utils.trilinear_interpolation(
                moving_stage, *moving_voxels
            )
            similarity = api.ncc.lncc(
                moving_values.view(1, -1), fixed_values.view(1, -1)
            )
            zero = similarity.new_tensor(0.0)
            empty = {
                "jacobian_raw": zero,
                "laplacian_raw": zero,
                "jacobian_weighted": zero,
                "laplacian_weighted": zero,
            }
            coarse_reg = empty
            fine_reg = empty
            idir_be_raw = zero
            idir_be_weighted = zero
            if loss_branch == LOSS_BRANCH_DUAL_NATIVE and (
                jacobian_weight > 0.0 or laplace_weight > 0.0
            ):
                coarse_reg = _branch_regularization_components(
                    regularizers_module=api.regularizers,
                    coords=coords,
                    output=coarse,
                    jacobian_weight=jacobian_weight,
                    laplace_weight=laplace_weight,
                    branch_scale=1.0,
                )
                if fine is not None:
                    fine_reg = _branch_regularization_components(
                        regularizers_module=api.regularizers,
                        coords=coords,
                        output=fine,
                        jacobian_weight=jacobian_weight,
                        laplace_weight=laplace_weight,
                        branch_scale=fine_reg_scale,
                    )
                spatial = sum(
                    (
                        coarse_reg["jacobian_weighted"],
                        coarse_reg["laplacian_weighted"],
                        fine_reg["jacobian_weighted"],
                        fine_reg["laplacian_weighted"],
                    ),
                    start=zero,
                )
            elif loss_branch == LOSS_BRANCH_IDIR_BE:
                idir_be_raw = mr_common.idir_bending_energy(
                    coords, total, batch_size=points_per_step
                )
                idir_be_weighted = idir_bending_weight * idir_be_raw
                spatial = idir_be_weighted
            else:
                spatial = zero
            total_loss = similarity + spatial
            finite = {
                "similarity": similarity,
                "spatial": spatial,
                "total": total_loss,
                "coarse_jacobian_raw": coarse_reg["jacobian_raw"],
                "coarse_laplacian_raw": coarse_reg["laplacian_raw"],
                "fine_jacobian_raw": fine_reg["jacobian_raw"],
                "fine_laplacian_raw": fine_reg["laplacian_raw"],
                "idir_bending_raw": idir_be_raw,
            }
            nonfinite = {
                name: _number(value)
                for (name, value) in finite.items()
                if not bool(torch.isfinite(value).all().item())
            }
            if nonfinite:
                raise FloatingPointError(
                    f"non-finite loss at step {global_step}, stage {stage_index}: {nonfinite}"
                )
            lr_before_update = float(optimizer.param_groups[0]["lr"])
            total_loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            lr_after_schedule = float(optimizer.param_groups[0]["lr"])
            record = {
                "zero_step": global_step,
                "step": global_step + 1,
                "stage_index": stage_index,
                "stage_local_step": local_step,
                "image_factor": int(factor),
                "stage_shape_x": stage_shape[0],
                "stage_shape_y": stage_shape[1],
                "stage_shape_z": stage_shape[2],
                "antialias_sigma_vox": float(blur_sigma),
                "fine_active": bool(model.fine_branch_is_active),
                "loss_total": _number(total_loss),
                "loss_similarity": _number(similarity),
                "loss_spatial_weighted": _number(spatial),
                "loss_mask_weighted": 0.0,
                "coarse_jacobian_raw": _number(coarse_reg["jacobian_raw"]),
                "coarse_jacobian_weighted": _number(coarse_reg["jacobian_weighted"]),
                "coarse_laplacian_raw": _number(coarse_reg["laplacian_raw"]),
                "coarse_laplacian_weighted": _number(coarse_reg["laplacian_weighted"]),
                "fine_jacobian_raw": _number(fine_reg["jacobian_raw"]),
                "fine_jacobian_weighted": _number(fine_reg["jacobian_weighted"]),
                "fine_laplacian_raw": _number(fine_reg["laplacian_raw"]),
                "fine_laplacian_weighted": _number(fine_reg["laplacian_weighted"]),
                "idir_bending_raw": _number(idir_be_raw),
                "idir_bending_weighted": _number(idir_be_weighted),
                "loss_branch_identifier": loss_branch,
                "idir_bending_weight": (
                    idir_bending_weight if loss_branch == LOSS_BRANCH_IDIR_BE else 0.0
                ),
                "learning_rate_before_update": lr_before_update,
                "learning_rate_after_schedule": lr_after_schedule,
                "coarse_scale": float(model.coarse_scale.detach().item()),
                "fine_scale": float(model.fine_scale.detach().item()),
            }
            history.append(record)
            if record_callback is not None:
                record_callback(record)
            global_step += 1
        stage_records.append(stage_record)
    if global_step != int(config["training"]["total_steps"]):
        raise RuntimeError(
            f"executed {global_step} steps, expected {config['training']['total_steps']}"
        )
    return {
        "model": model,
        "history": history,
        "stages": stage_records,
        "fine_activation_step": fine_activation_step,
        "optimizer_settings": copy.deepcopy(config["optimizer"]),
        "scheduler_settings": copy.deepcopy(config["scheduler"]),
    }
