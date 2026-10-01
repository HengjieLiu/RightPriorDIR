"""Audited registration/data kernels; see docs/methods.md for protocol scope."""

from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
from . import raw_lung as lungct_io
from . import lung as common
from . import dual_data as base

ControlledCase = base.ControlledCase

PTV_BORDER_VOXELS = base.PTV_BORDER_VOXELS

make_border_five_mask_xyz = base.make_border_five_mask_xyz


def _canonical_root(value: Path) -> Path:
    root = value.expanduser().resolve()
    candidates = [root]
    if root.name.lower() == "copd" and root.parent.name == "raw":
        candidates.append(root.parent.parent)
    if (root / "raw" / "COPD").is_dir():
        candidates.append(root)
    for candidate in candidates:
        if (candidate / "raw" / "COPD").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not resolve canonical DIR-Lab root below {root}")


def _resolve_mask(root: Path, case_id: int) -> Path:
    root = root.expanduser().resolve()
    filename = f"case{case_id:02d}_iBH_R231_binary_zyx.npy"
    candidates = (
        root / "COPD" / f"case{case_id:02d}" / filename,
        root / "copd" / f"case{case_id:02d}" / filename,
        root / f"case{case_id:02d}" / filename,
        root
        / "processed"
        / "lungmask_R231_native"
        / "COPD"
        / f"case{case_id:02d}"
        / filename,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve fixed iBH R231 mask {filename} below {root}"
    )


def _resolve_ptv_root(root: Path, case_id: int) -> Path:
    root = root.expanduser().resolve()
    candidates = (
        root,
        root / "pTVReg_iso1mm",
        root / "processed" / "pTVReg_iso1mm",
        root.parent if root.name.lower() == "copd" else root,
    )
    for candidate in candidates:
        if (candidate / "COPD" / f"case{case_id:02d}").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve pTVReg_iso1mm/COPD/case{case_id:02d} below {root}"
    )


def _load_native_case(
    case_id: int, native_raw_root: Path, native_mask_root: Path
) -> ControlledCase:
    data_root = _canonical_root(native_raw_root)
    case = lungct_io.resolve_copd_case(case_id, data_root)
    if case.status != "complete":
        raise FileNotFoundError(
            f"COPD case {case_id:02d} is {case.status}: {case.notes}"
        )
    if any(
        (
            value is None
            for value in (
                case.image_ibh,
                case.image_ebh,
                case.landmarks_ibh,
                case.landmarks_ebh,
            )
        )
    ):
        raise FileNotFoundError(f"COPD case {case_id:02d} has incomplete source paths")
    fixed = base._zyx_to_xyz_array(
        base._load_raw_int16(Path(case.image_ibh), case.shape_zyx)
    )
    moving = base._zyx_to_xyz_array(
        base._load_raw_int16(Path(case.image_ebh), case.shape_zyx)
    )
    fixed_landmarks = np.ascontiguousarray(
        lungct_io.load_landmarks_xyz(Path(case.landmarks_ibh)), dtype=np.float64
    )
    moving_landmarks = np.ascontiguousarray(
        lungct_io.load_landmarks_xyz(Path(case.landmarks_ebh)), dtype=np.float64
    )
    mask_path = _resolve_mask(native_mask_root, case_id)
    mask_zyx = (np.load(mask_path, allow_pickle=False) > 0).astype(np.uint8, copy=False)
    if mask_zyx.shape != tuple(case.shape_zyx):
        raise ValueError(
            f"iBH R231 mask shape {mask_zyx.shape} != native image {case.shape_zyx}"
        )
    fixed_mask = base._zyx_to_xyz_array(mask_zyx)
    provenance: dict[str, Any] = {
        "protocol_handle": "native_full_lungmask_idir",
        "protocol_tags": {
            "dataset": "copd",
            "image_domain": "native_full_volume",
            "spacing": "native",
            "crop_rule": "none",
            "intensity_rule": "native_raw_int16",
            "mask_rule": "fixed_iBH_lung_mask",
            "loss_domain": "mask_sampled",
            "metric_domain": "native_landmarks",
            "metric_resolution": "native_full_or_crop",
            "tre_mode": "both",
            "method_family": "dual_inr",
        },
        "dataset": "copd",
        "fixed_phase": "iBH",
        "moving_phase": "eBH",
        "registration_direction": "eBH_to_iBH",
        "image_domain": "native_full_volume",
        "source_array_order": "zyx",
        "training_array_order": "xyz",
        "vector_component_order": "xyz",
        "spacing_order": "xyz",
        "intensity_rule": "raw_int16_no_shift_clip_or_normalization",
        "crop_rule": "none",
        "spacing_rule": "native_anisotropic",
        "optimization_mask_rule": "fixed_iBH_r231_lung_mask_only",
        "metric_mask_rule": "fixed_iBH_r231_lung_mask",
        "landmark_rule": "raw_copd_xyz_preserved_without_index_shift",
        "shape_zyx": list(case.shape_zyx),
        "shape_xyz": list(fixed.shape),
        "spacing_zyx_mm": list(case.spacing_zyx),
        "spacing_xyz_mm": list(case.spacing_zyx[::-1]),
        "sampling_mask_voxels": int(fixed_mask.sum()),
    }
    result = ControlledCase(
        protocol="N",
        case_id=case_id,
        fixed=fixed,
        moving=moving,
        sampling_mask_xyz=fixed_mask,
        metric_mask_xyz=fixed_mask,
        landmarks_fixed_xyz=fixed_landmarks,
        landmarks_moving_xyz=moving_landmarks,
        spacing_xyz=tuple((float(value) for value in case.spacing_zyx[::-1])),
        source_paths={
            "fixed_image_iBH": str(Path(case.image_ibh).resolve()),
            "moving_image_eBH": str(Path(case.image_ebh).resolve()),
            "fixed_landmarks_iBH": str(Path(case.landmarks_ibh).resolve()),
            "moving_landmarks_eBH": str(Path(case.landmarks_ebh).resolve()),
            "fixed_mask_iBH_r231": str(mask_path.resolve()),
        },
        provenance=provenance,
    )
    base._validate_controlled_case(result)
    return result


def _load_ptv_case(case_id: int, ptv_root: Path) -> ControlledCase:
    common_root = _resolve_ptv_root(ptv_root, case_id)
    ptv = common.load_case(common_root, case_id, dataset="copd")
    fixed = base._zyx_to_xyz_array(ptv.fixed.astype(np.float32, copy=False))
    moving = base._zyx_to_xyz_array(ptv.moving.astype(np.float32, copy=False))
    metric_mask = base._zyx_to_xyz_array(ptv.fixed_mask.astype(np.uint8, copy=False))
    sampling_mask = make_border_five_mask_xyz(fixed.shape)
    audit = ptv.preprocess_audit
    crop_convention = str(
        audit.get("crop_convention", "plan_expected_legacy_python_bounds")
    )
    provenance: dict[str, Any] = {
        "protocol_handle": "ptv_landmark_crop_1mm_no_lungmask",
        "protocol_tags": {
            "dataset": "copd",
            "image_domain": "landmark_crop",
            "spacing": "isotropic_1mm",
            "crop_rule": "legacy_python_bounds_landmark_bbox_margin_xyz_10_10_5",
            "intensity_rule": "ct_clip_80_900_scale_0_1",
            "mask_rule": "none",
            "loss_domain": "border_masked",
            "metric_domain": "native_landmarks_after_crop_conversion",
            "metric_resolution": "native_full_or_crop_and_cropped_1mm_reg",
            "tre_mode": "both",
            "method_family": "dual_inr",
        },
        "dataset": "copd",
        "fixed_phase": "iBH",
        "moving_phase": "eBH",
        "registration_direction": "eBH_to_iBH",
        "image_domain": "landmark_crop",
        "source_array_order": "zyx",
        "training_array_order": "xyz",
        "vector_component_order": "xyz",
        "spacing_order": "xyz",
        "intensity_rule": "ct_clip_80_900_scale_0_1",
        "crop_rule": crop_convention,
        "spacing_rule": "approximately_isotropic_1mm",
        "optimization_mask_rule": "exact_border_five_interior_no_lung_mask",
        "metric_mask_rule": "fixed_iBH_r231_lung_mask_metric_only",
        "landmark_rule": "canonical_ptv_reg1mm_zyx_transposed_to_xyz",
        "shape_zyx": list(ptv.shape_zyx),
        "shape_xyz": list(fixed.shape),
        "spacing_zyx_mm": [1.0, 1.0, 1.0],
        "spacing_xyz_mm": [1.0, 1.0, 1.0],
        "native_spacing_zyx_mm": list(ptv.native_spacing_zyx),
        "crop_shape_zyx": list(ptv.crop_shape_zyx),
        "border_voxels": PTV_BORDER_VOXELS,
        "sampling_mask_voxels": int(sampling_mask.sum()),
        "metric_mask_voxels": int(metric_mask.sum()),
        "preprocess_verification_pass": bool(common._audit_verification_pass(audit)),
        "preprocessing_note": "Legacy Python crop bounds are intentionally reused so P0/P1 match the completed COPD MR-D-BSCP P2e reference.",
    }
    result = ControlledCase(
        protocol="P",
        case_id=case_id,
        fixed=fixed,
        moving=moving,
        sampling_mask_xyz=sampling_mask,
        metric_mask_xyz=metric_mask,
        landmarks_fixed_xyz=np.ascontiguousarray(
            ptv.landmarks_fixed[:, ::-1], dtype=np.float64
        ),
        landmarks_moving_xyz=np.ascontiguousarray(
            ptv.landmarks_moving[:, ::-1], dtype=np.float64
        ),
        spacing_xyz=(1.0, 1.0, 1.0),
        source_paths={
            "ptv_case_dir": str(ptv.case_dir.resolve()),
            "preprocess_audit": str((ptv.case_dir / "preprocess_audit.json").resolve()),
            "fixed_iBH_reg1mm": str(
                (ptv.case_dir / "fixed_iBH_reg1mm_trilinear_zyx.npy").resolve()
            ),
            "moving_eBH_reg1mm": str(
                (ptv.case_dir / "moving_eBH_reg1mm_trilinear_zyx.npy").resolve()
            ),
            "fixed_iBH_mask_metric_only": str(
                (ptv.case_dir / "fixed_iBH_lungmask_reg1mm_nearest_zyx.npy").resolve()
            ),
        },
        provenance=provenance,
        ptv_case=ptv,
    )
    base._validate_controlled_case(result)
    return result


def load_controlled_case(
    *,
    protocol: str,
    case_id: int,
    native_raw_root: Path,
    native_mask_root: Path,
    ptv_root: Path,
) -> ControlledCase:
    protocol = base._validate_protocol(protocol)
    case_id = base._validate_case_id(case_id)
    if protocol == "N":
        return _load_native_case(case_id, native_raw_root, native_mask_root)
    return _load_ptv_case(case_id, ptv_root)
