"""Audited registration/data kernels; see docs/methods.md for protocol scope."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import torch
from . import raw_lung as lungct_io
from . import lung as common

PROTOCOLS = ("N", "P", "PL")

PTV_BORDER_VOXELS = 5


@dataclass(frozen=True)
class ControlledCase:
    """One fixed-T00/moving-T50 case in DUAL's xyz array convention.

    ``sampling_mask_xyz`` defines the optimization domain.  It is the fixed
    T00 R231 lung mask for protocol N, the exact five-voxel pTV interior for
    protocol P, and the fixed T00 pTV 1 mm lung mask for protocol PL.
    ``metric_mask_xyz`` is always the fixed T00 lung mask and is deliberately
    separate because protocol P must not use it for optimization.
    """

    protocol: str
    case_id: int
    fixed: np.ndarray
    moving: np.ndarray
    sampling_mask_xyz: np.ndarray
    metric_mask_xyz: np.ndarray
    landmarks_fixed_xyz: np.ndarray
    landmarks_moving_xyz: np.ndarray
    spacing_xyz: tuple[float, float, float]
    source_paths: Mapping[str, str]
    provenance: Mapping[str, Any]
    ptv_case: common.LungCTCase | None = None

    @property
    def shape_xyz(self) -> tuple[int, int, int]:
        return tuple((int(value) for value in self.fixed.shape))

    @property
    def fixed_image(self) -> np.ndarray:
        """Explicit image-name alias useful to runners."""
        return self.fixed

    @property
    def moving_image(self) -> np.ndarray:
        """Explicit image-name alias useful to runners."""
        return self.moving

    @property
    def sampling_mask(self) -> np.ndarray:
        return self.sampling_mask_xyz

    @property
    def metric_mask(self) -> np.ndarray:
        return self.metric_mask_xyz

    @property
    def fixed_metric_mask_xyz(self) -> np.ndarray:
        return self.metric_mask_xyz

    @property
    def landmarks_fixed(self) -> np.ndarray:
        return self.landmarks_fixed_xyz

    @property
    def landmarks_moving(self) -> np.ndarray:
        return self.landmarks_moving_xyz


def _validate_case_id(case_id: int) -> int:
    case_id = int(case_id)
    if case_id not in range(1, 11):
        raise ValueError(f"DIR-Lab 4DCT case_id must be in 1..10, got {case_id}")
    return case_id


def _validate_protocol(protocol: str) -> str:
    value = str(protocol).strip().upper()
    if value not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}, got {protocol!r}")
    return value


def _dedupe_paths(paths: Sequence[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            result.append(path)
            seen.add(key)
    return result


def _resolve_4dct_root(native_raw_root: Path) -> Path:
    root = Path(native_raw_root).expanduser().resolve()
    candidates = _dedupe_paths([root, root / "4DCT", root / "raw" / "4DCT"])
    matches = [
        candidate
        for candidate in candidates
        if candidate.is_dir() and any(candidate.glob("Case[0-9]*Pack*"))
    ]
    if not matches:
        raise FileNotFoundError(
            f"Could not resolve a raw 4DCT directory below {root}; expected Case{{id}}Pack* folders"
        )
    return matches[0]


def _phase_image_candidates(case_root: Path, case_id: int, phase: str) -> list[Path]:
    phase_token = f"t{phase}".lower()
    candidates = [
        path
        for path in case_root.rglob("*.img")
        if phase_token in path.name.lower() and f"case{case_id}" in path.name.lower()
    ]

    def priority(path: Path) -> tuple[int, int, str]:
        name = path.name.lower()
        if name.endswith(f"_t{phase}_s.img"):
            suffix_rank = 0
        elif name.endswith(f"_t{phase}-ssm.img"):
            suffix_rank = 1
        elif name.endswith(f"_t{phase}.img"):
            suffix_rank = 2
        else:
            suffix_rank = 3
        return (suffix_rank, len(path.parts), str(path))

    return sorted(candidates, key=priority)


def _phase_landmark_candidates(case_root: Path, case_id: int, phase: str) -> list[Path]:
    phase_token = f"t{phase}".lower()
    return sorted(
        (
            path
            for path in case_root.rglob("*.txt")
            if phase_token in path.name.lower()
            and "xyz" in path.name.lower()
            and (f"case{case_id}" in path.name.lower())
        ),
        key=lambda path: (len(path.parts), str(path)),
    )


def _resolve_native_sources(native_raw_root: Path, case_id: int) -> dict[str, Path]:
    fourdct_root = _resolve_4dct_root(native_raw_root)
    pack_roots = sorted(fourdct_root.glob(f"Case{case_id}Pack*"))
    pack_roots += sorted(fourdct_root.glob(f"Case{case_id}Deploy*"))
    if not pack_roots:
        raise FileNotFoundError(
            f"Could not find DIR-Lab case {case_id} below {fourdct_root}"
        )
    scored: list[tuple[int, int, Path, dict[str, list[Path]]]] = []
    for case_root in _dedupe_paths(pack_roots):
        candidates = {
            "fixed_image_t00": _phase_image_candidates(case_root, case_id, "00"),
            "moving_image_t50": _phase_image_candidates(case_root, case_id, "50"),
            "fixed_landmarks_t00": _phase_landmark_candidates(case_root, case_id, "00"),
            "moving_landmarks_t50": _phase_landmark_candidates(
                case_root, case_id, "50"
            ),
        }
        score = sum((bool(values) for values in candidates.values()))
        scored.append((score, -len(case_root.parts), case_root, candidates))
    (_score, _depth, selected_root, selected) = max(
        scored, key=lambda item: (item[0], item[1], str(item[2]))
    )
    missing = [key for (key, values) in selected.items() if not values]
    if missing:
        raise FileNotFoundError(
            f"Incomplete DIR-Lab case {case_id} below {selected_root}; missing {missing}"
        )
    return {
        "case_root": selected_root,
        **{key: values[0] for (key, values) in selected.items()},
    }


def _resolve_native_mask_path(native_mask_root: Path, case_id: int) -> Path:
    root = Path(native_mask_root).expanduser().resolve()
    filename = f"case{case_id:02d}_T00_R231_binary_zyx.npy"
    case_name = f"case{case_id:02d}"
    candidates = _dedupe_paths(
        [
            root / "4DCT" / case_name / filename,
            root / "4dct" / case_name / filename,
            root / case_name / filename,
            root / "processed" / "lungmask_R231_native" / "4DCT" / case_name / filename,
        ]
    )
    matches = [path for path in candidates if path.is_file()]
    if not matches:
        raise FileNotFoundError(
            f"Could not resolve fixed T00 R231 mask {filename} below {root}"
        )
    return matches[0]


def _resolve_ptv_common_root(ptv_root: Path, case_id: int) -> Path:
    root = Path(ptv_root).expanduser().resolve()
    case_name = f"case{case_id:02d}"
    candidates = _dedupe_paths(
        [
            root,
            root / "pTVReg_iso1mm",
            root / "processed" / "pTVReg_iso1mm",
            root.parent if root.name.lower() == "4dct" else root,
        ]
    )
    for candidate in candidates:
        if (candidate / "4DCT" / case_name).is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve canonical pTVReg_iso1mm/4DCT/{case_name} below {root}"
    )


def _load_raw_int16(path: Path, shape_zyx: Sequence[int]) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int16)
    expected = int(np.prod(shape_zyx))
    if raw.size != expected:
        raise ValueError(
            f"{path} has {raw.size} int16 voxels, expected {expected} for shape {tuple((int(v) for v in shape_zyx))}"
        )
    return raw.reshape(tuple((int(v) for v in shape_zyx)))


def _zyx_to_xyz_array(array: np.ndarray) -> np.ndarray:
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D zyx array, got shape {array.shape}")
    return np.ascontiguousarray(np.transpose(array, (2, 1, 0)))


def make_border_five_mask_xyz(shape_xyz: Sequence[int]) -> np.ndarray:
    """Return the exact pTV five-voxel interior, independent of lung masks."""
    shape = tuple((int(value) for value in shape_xyz))
    if len(shape) != 3 or any((value <= 2 * PTV_BORDER_VOXELS for value in shape)):
        raise ValueError(
            f"A border-{PTV_BORDER_VOXELS} mask requires three axes larger than {2 * PTV_BORDER_VOXELS}, got {shape}"
        )
    mask = np.zeros(shape, dtype=np.uint8)
    border = PTV_BORDER_VOXELS
    mask[border:-border, border:-border, border:-border] = 1
    return mask


def _assert_landmarks_inside_xyz(
    landmarks_xyz: np.ndarray, shape_xyz: Sequence[int], label: str
) -> None:
    points = np.asarray(landmarks_xyz, dtype=np.float64)
    shape = np.asarray(shape_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError(f"{label} must be a non-empty Nx3 array, got {points.shape}")
    if not np.isfinite(points).all():
        raise FloatingPointError(f"{label} contains non-finite values")
    inside = np.all((points >= 0.0) & (points <= shape[None, :] - 1.0), axis=1)
    if not bool(np.all(inside)):
        bad = np.flatnonzero(~inside).tolist()
        raise ValueError(
            f"{label} has points outside xyz shape {tuple((int(v) for v in shape))} at indices {bad[:10]}"
        )


def _validate_controlled_case(case: ControlledCase) -> None:
    shape = case.shape_xyz
    if len(shape) != 3 or min(shape) < 2:
        raise ValueError(f"Images must be 3D with axes >=2, got {shape}")
    if case.moving.shape != shape:
        raise ValueError(f"Fixed/moving shape mismatch: {shape} vs {case.moving.shape}")
    if case.sampling_mask_xyz.shape != shape:
        raise ValueError(
            f"Sampling-mask shape mismatch: {case.sampling_mask_xyz.shape} vs {shape}"
        )
    if case.metric_mask_xyz.shape != shape:
        raise ValueError(
            f"Metric-mask shape mismatch: {case.metric_mask_xyz.shape} vs {shape}"
        )
    for label, array in (("fixed", case.fixed), ("moving", case.moving)):
        if not np.isfinite(array).all():
            raise FloatingPointError(f"{label} image contains non-finite values")
    for label, mask in (
        ("sampling", case.sampling_mask_xyz),
        ("metric", case.metric_mask_xyz),
    ):
        values = np.unique(mask)
        if not np.all(np.isin(values, (0, 1))):
            raise ValueError(f"{label} mask is not binary; values={values.tolist()}")
        if int(mask.sum()) <= 0:
            raise ValueError(f"{label} mask is empty")
    if case.landmarks_fixed_xyz.shape != case.landmarks_moving_xyz.shape:
        raise ValueError(
            f"Fixed/moving landmark shape mismatch: {case.landmarks_fixed_xyz.shape} vs {case.landmarks_moving_xyz.shape}"
        )
    _assert_landmarks_inside_xyz(case.landmarks_fixed_xyz, shape, "fixed landmarks")
    _assert_landmarks_inside_xyz(case.landmarks_moving_xyz, shape, "moving landmarks")
    spacing = np.asarray(case.spacing_xyz, dtype=np.float64)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError(
            f"spacing_xyz must contain three finite positive values: {spacing}"
        )
    if case.protocol == "N":
        if case.fixed.dtype != np.dtype(np.int16) or case.moving.dtype != np.dtype(
            np.int16
        ):
            raise TypeError(
                "Protocol N must preserve the raw DIR-Lab int16 image values"
            )
        if not np.array_equal(case.sampling_mask_xyz, case.metric_mask_xyz):
            raise ValueError(
                "Protocol N sampling and metric masks must both be fixed T00 R231"
            )
        if case.ptv_case is not None:
            raise ValueError("Protocol N must not retain a pTV case")
    elif case.protocol == "P":
        expected = make_border_five_mask_xyz(shape)
        if not np.array_equal(case.sampling_mask_xyz, expected):
            raise ValueError(
                "Protocol P sampling mask must be the exact border-five interior"
            )
        if case.ptv_case is None:
            raise ValueError("Protocol P must retain its shared pTV case object")
    elif case.protocol == "PL":
        if not np.array_equal(case.sampling_mask_xyz, case.metric_mask_xyz):
            raise ValueError(
                "Protocol PL sampling and metric masks must both be fixed T00 lung mask"
            )
        if case.ptv_case is None:
            raise ValueError("Protocol PL must retain its shared pTV case object")


def _load_native_case(
    case_id: int, native_raw_root: Path, native_mask_root: Path
) -> ControlledCase:
    source = _resolve_native_sources(native_raw_root, case_id)
    shape_zyx = lungct_io.FOURDCT_SHAPES_ZYX[case_id]
    spacing_zyx = lungct_io.FOURDCT_SPACING_ZYX[case_id]
    fixed = _zyx_to_xyz_array(_load_raw_int16(source["fixed_image_t00"], shape_zyx))
    moving = _zyx_to_xyz_array(_load_raw_int16(source["moving_image_t50"], shape_zyx))
    fixed_landmarks = np.ascontiguousarray(
        lungct_io.load_landmarks_xyz(source["fixed_landmarks_t00"]), dtype=np.float64
    )
    moving_landmarks = np.ascontiguousarray(
        lungct_io.load_landmarks_xyz(source["moving_landmarks_t50"]), dtype=np.float64
    )
    mask_path = _resolve_native_mask_path(native_mask_root, case_id)
    fixed_mask_zyx = (np.load(mask_path) > 0).astype(np.uint8, copy=False)
    if fixed_mask_zyx.shape != tuple(shape_zyx):
        raise ValueError(
            f"Native fixed mask shape {fixed_mask_zyx.shape} does not match raw image shape {shape_zyx}"
        )
    fixed_mask = _zyx_to_xyz_array(fixed_mask_zyx)
    source_paths = {key: str(value.resolve()) for (key, value) in source.items()}
    source_paths["fixed_mask_t00_r231"] = str(mask_path.resolve())
    provenance: dict[str, Any] = {
        "protocol_handle": "native_full_lungmask_idir",
        "protocol_tags": {
            "dataset": "4dct",
            "image_domain": "native_full_volume",
            "spacing": "native",
            "crop_rule": "none",
            "intensity_rule": "native_raw_int16",
            "mask_rule": "fixed_t00_lung_mask",
            "loss_domain": "mask_sampled",
            "metric_domain": "native_landmarks",
            "metric_resolution": "native_full_or_crop",
            "tre_mode": "both",
            "method_family": "dual_inr",
        },
        "dataset": "4dct",
        "fixed_phase": "T00",
        "moving_phase": "T50",
        "registration_direction": "T50_to_T00",
        "image_domain": "native_full_volume",
        "source_array_order": "zyx",
        "training_array_order": "xyz",
        "vector_component_order": "xyz",
        "spacing_order": "xyz",
        "intensity_rule": "raw_int16_no_shift_clip_or_normalization",
        "crop_rule": "none",
        "spacing_rule": "native_anisotropic",
        "optimization_mask_rule": "fixed_t00_r231_lung_mask_only",
        "metric_mask_rule": "fixed_t00_r231_lung_mask",
        "landmark_rule": "raw_dirlab_xyz_preserved_without_index_shift",
        "shape_zyx": [int(value) for value in shape_zyx],
        "shape_xyz": [int(value) for value in fixed.shape],
        "spacing_zyx_mm": [float(value) for value in spacing_zyx],
        "spacing_xyz_mm": [float(value) for value in spacing_zyx[::-1]],
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
        spacing_xyz=tuple((float(value) for value in spacing_zyx[::-1])),
        source_paths=source_paths,
        provenance=provenance,
    )
    _validate_controlled_case(result)
    return result


def _load_ptv_case(
    case_id: int, ptv_root: Path, *, use_fixed_lung_mask_for_sampling: bool = False
) -> ControlledCase:
    common_root = _resolve_ptv_common_root(ptv_root, case_id)
    ptv_case = common.load_case(common_root, case_id, dataset="4dct")
    fixed = _zyx_to_xyz_array(ptv_case.fixed.astype(np.float32, copy=False))
    moving = _zyx_to_xyz_array(ptv_case.moving.astype(np.float32, copy=False))
    metric_mask = _zyx_to_xyz_array(ptv_case.fixed_mask.astype(np.uint8, copy=False))
    protocol = "PL" if use_fixed_lung_mask_for_sampling else "P"
    sampling_mask = (
        metric_mask.copy()
        if use_fixed_lung_mask_for_sampling
        else make_border_five_mask_xyz(fixed.shape)
    )
    fixed_landmarks = np.ascontiguousarray(
        ptv_case.landmarks_fixed[:, ::-1], dtype=np.float64
    )
    moving_landmarks = np.ascontiguousarray(
        ptv_case.landmarks_moving[:, ::-1], dtype=np.float64
    )
    audit_path = ptv_case.case_dir / "preprocess_audit.json"
    source_paths = {
        "ptv_case_dir": str(ptv_case.case_dir.resolve()),
        "preprocess_audit": str(audit_path.resolve()),
        "fixed_image_t00_reg1mm": str(
            (ptv_case.case_dir / "fixed_T00_reg1mm_trilinear_zyx.npy").resolve()
        ),
        "moving_image_t50_reg1mm": str(
            (ptv_case.case_dir / "moving_T50_reg1mm_trilinear_zyx.npy").resolve()
        ),
        "fixed_mask_t00_reg1mm_metric_only": str(
            (ptv_case.case_dir / "fixed_T00_lungmask_reg1mm_nearest_zyx.npy").resolve()
        ),
    }
    provenance: dict[str, Any] = {
        "protocol_handle": (
            "ptv_landmark_crop_1mm_with_fixed_t00_lungmask"
            if use_fixed_lung_mask_for_sampling
            else "ptv_landmark_crop_1mm_no_lungmask"
        ),
        "protocol_tags": {
            "dataset": "4dct",
            "image_domain": "landmark_crop",
            "spacing": "isotropic_1mm",
            "crop_rule": "landmark_bbox_margin_xyz_10_10_5",
            "intensity_rule": "ct_clip_80_900_scale_0_1",
            "mask_rule": (
                "fixed_t00_lung_mask" if use_fixed_lung_mask_for_sampling else "none"
            ),
            "loss_domain": (
                "mask_sampled" if use_fixed_lung_mask_for_sampling else "border_masked"
            ),
            "metric_domain": "native_landmarks_after_crop_conversion",
            "metric_resolution": "native_full_or_crop_and_cropped_1mm_reg",
            "tre_mode": "both",
            "method_family": "dual_inr",
        },
        "dataset": "4dct",
        "fixed_phase": "T00",
        "moving_phase": "T50",
        "registration_direction": "T50_to_T00",
        "image_domain": "landmark_crop",
        "source_array_order": "zyx",
        "training_array_order": "xyz",
        "vector_component_order": "xyz",
        "spacing_order": "xyz",
        "intensity_rule": "ct_clip_80_900_scale_0_1",
        "crop_rule": "landmark_bbox_margin_xyz_10_10_5",
        "spacing_rule": "approximately_isotropic_1mm",
        "optimization_mask_rule": (
            "fixed_t00_lung_mask"
            if use_fixed_lung_mask_for_sampling
            else "exact_border_five_interior_no_lung_mask"
        ),
        "metric_mask_rule": (
            "fixed_t00_lung_mask"
            if use_fixed_lung_mask_for_sampling
            else "fixed_t00_lung_mask_metric_only"
        ),
        "landmark_rule": "canonical_ptv_reg1mm_zyx_transposed_to_xyz",
        "shape_zyx": [int(value) for value in ptv_case.shape_zyx],
        "shape_xyz": [int(value) for value in fixed.shape],
        "spacing_zyx_mm": [1.0, 1.0, 1.0],
        "spacing_xyz_mm": [1.0, 1.0, 1.0],
        "native_spacing_zyx_mm": [
            float(value) for value in ptv_case.native_spacing_zyx
        ],
        "crop_shape_zyx": [int(value) for value in ptv_case.crop_shape_zyx],
        "border_voxels": PTV_BORDER_VOXELS,
        "sampling_mask_voxels": int(sampling_mask.sum()),
        "metric_mask_voxels": int(metric_mask.sum()),
        "preprocess_verification_pass": bool(
            common._audit_verification_pass(ptv_case.preprocess_audit)
        ),
    }
    if float(np.min(fixed)) < -1e-06 or float(np.max(fixed)) > 1.0 + 1e-06:
        raise ValueError(
            f"Canonical pTV fixed image is outside [0,1]: range=({float(np.min(fixed))}, {float(np.max(fixed))})"
        )
    if float(np.min(moving)) < -1e-06 or float(np.max(moving)) > 1.0 + 1e-06:
        raise ValueError(
            f"Canonical pTV moving image is outside [0,1]: range=({float(np.min(moving))}, {float(np.max(moving))})"
        )
    result = ControlledCase(
        protocol=protocol,
        case_id=case_id,
        fixed=fixed,
        moving=moving,
        sampling_mask_xyz=sampling_mask,
        metric_mask_xyz=metric_mask,
        landmarks_fixed_xyz=fixed_landmarks,
        landmarks_moving_xyz=moving_landmarks,
        spacing_xyz=(1.0, 1.0, 1.0),
        source_paths=source_paths,
        provenance=provenance,
        ptv_case=ptv_case,
    )
    _validate_controlled_case(result)
    return result


def load_controlled_case(
    protocol: str,
    case_id: int,
    native_raw_root: Path | str,
    native_mask_root: Path | str,
    ptv_root: Path | str,
) -> ControlledCase:
    """Load N or P while keeping optimization and metric masks explicit.

    All three roots are mandatory even though each protocol consumes only the
    relevant roots.  This keeps one stable launcher contract across N0/N1/P0/P1
    and makes the complete data provenance available at invocation time.
    """
    protocol = _validate_protocol(protocol)
    case_id = _validate_case_id(case_id)
    if protocol == "N":
        return _load_native_case(case_id, Path(native_raw_root), Path(native_mask_root))
    return _load_ptv_case(
        case_id, Path(ptv_root), use_fixed_lung_mask_for_sampling=protocol == "PL"
    )


def normalized_field_xyz_to_voxel_xyz(
    dense_disp_norm_xyz: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """Scale an xyz-grid normalized DUAL field into xyz voxel displacement."""
    if isinstance(dense_disp_norm_xyz, torch.Tensor):
        array = dense_disp_norm_xyz.detach().cpu().numpy()
    else:
        array = np.asarray(dense_disp_norm_xyz)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(
            f"Expected normalized displacement [X,Y,Z,3], got {array.shape}"
        )
    if min(array.shape[:3]) < 2:
        raise ValueError(f"All field axes must be >=2, got {array.shape[:3]}")
    if not np.isfinite(array).all():
        raise FloatingPointError("Normalized dense field contains non-finite values")
    scale = (np.asarray(array.shape[:3], dtype=np.float32) - 1.0) / 2.0
    result = array.astype(np.float32, copy=False) * scale.reshape(1, 1, 1, 3)
    if not np.isfinite(result).all():
        raise FloatingPointError("Voxel dense field contains non-finite values")
    return np.ascontiguousarray(result, dtype=np.float32)


def field_xyz_to_common_zyx(disp_voxel_xyz: np.ndarray) -> np.ndarray:
    """Convert ``[X,Y,Z,(dx,dy,dz)]`` to ``[Z,Y,X,(dz,dy,dx)]``."""
    field = np.asarray(disp_voxel_xyz)
    if field.ndim != 4 or field.shape[-1] != 3:
        raise ValueError(f"Expected voxel displacement [X,Y,Z,3], got {field.shape}")
    if not np.isfinite(field).all():
        raise FloatingPointError("Voxel dense field contains non-finite values")
    return np.ascontiguousarray(
        np.transpose(field, (2, 1, 0, 3))[..., ::-1], dtype=np.float32
    )


def mask_xyz_to_common_zyx(mask_xyz: np.ndarray) -> np.ndarray:
    """Convert a scalar xyz mask into the shared zyx array convention."""
    mask = np.asarray(mask_xyz)
    if mask.ndim != 3:
        raise ValueError(f"Expected mask [X,Y,Z], got {mask.shape}")
    return np.ascontiguousarray(np.transpose(mask, (2, 1, 0)))


def landmarks_xyz_to_common_zyx(landmarks_xyz: np.ndarray) -> np.ndarray:
    points = np.asarray(landmarks_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected landmarks [N,3], got {points.shape}")
    if not np.isfinite(points).all():
        raise FloatingPointError("Landmarks contain non-finite values")
    return np.ascontiguousarray(points[:, ::-1])


def _model_device(
    model: torch.nn.Module, device: torch.device | str | int | None
) -> torch.device:
    if device is not None:
        if isinstance(device, int) or str(device).isdigit():
            return torch.device(f"cuda:{int(device)}")
        return torch.device(str(device))
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _combined_model_output(
    model: torch.nn.Module, coords: torch.Tensor
) -> torch.Tensor:
    output = model(coords)
    if isinstance(output, (tuple, list)):
        if not output:
            raise ValueError("Model returned an empty displacement tuple")
        output = sum(output[1:], output[0])
    if not isinstance(output, torch.Tensor) or output.shape != coords.shape:
        raise ValueError(
            f"Model must return normalized displacement {tuple(coords.shape)}, got {getattr(output, 'shape', type(output))}"
        )
    return output


def decode_dense_normalized_field_xyz(
    model: torch.nn.Module,
    shape_xyz: Sequence[int],
    device: torch.device | str | int | None = None,
    *,
    max_points_per_chunk: int = 262144,
) -> np.ndarray:
    """Decode a model on a full xyz grid without materializing it at once."""
    shape = tuple((int(value) for value in shape_xyz))
    if len(shape) != 3 or min(shape) < 2:
        raise ValueError(f"shape_xyz must have three axes >=2, got {shape}")
    if int(max_points_per_chunk) <= 0:
        raise ValueError("max_points_per_chunk must be >0")
    device_t = _model_device(model, device)
    total = int(np.prod(shape))
    result = np.empty((total, 3), dtype=np.float32)
    was_training = model.training
    model.eval()
    yz = int(shape[1] * shape[2])
    try:
        with torch.no_grad():
            for start in range(0, total, int(max_points_per_chunk)):
                stop = min(start + int(max_points_per_chunk), total)
                flat = torch.arange(start, stop, device=device_t, dtype=torch.long)
                ix = torch.div(flat, yz, rounding_mode="floor")
                remainder = flat - ix * yz
                iy = torch.div(remainder, shape[2], rounding_mode="floor")
                iz = remainder - iy * shape[2]
                coords = torch.stack((ix, iy, iz), dim=1).to(torch.float32)
                scale = coords.new_tensor(shape).sub_(1.0)
                coords = coords.div(scale).mul_(2.0).sub_(1.0)
                output = _combined_model_output(model, coords)
                result[start:stop] = output.detach().to("cpu", torch.float32).numpy()
    finally:
        model.train(was_training)
    result = result.reshape(*shape, 3)
    if not np.isfinite(result).all():
        raise FloatingPointError(
            "Decoded normalized dense field contains non-finite values"
        )
    return result


def decode_dense_voxel_field_xyz(
    model: torch.nn.Module,
    shape_xyz: Sequence[int],
    device: torch.device | str | int | None = None,
    *,
    max_points_per_chunk: int = 262144,
) -> np.ndarray:
    return normalized_field_xyz_to_voxel_xyz(
        decode_dense_normalized_field_xyz(
            model, shape_xyz, device, max_points_per_chunk=max_points_per_chunk
        )
    )


def predict_direct_warped_landmarks_xyz(
    model: torch.nn.Module,
    case: ControlledCase,
    device: torch.device | str | int | None = None,
) -> np.ndarray:
    """Evaluate the INR directly at fixed landmarks and return warped xyz voxels."""
    device_t = _model_device(model, device)
    points = torch.as_tensor(
        case.landmarks_fixed_xyz, device=device_t, dtype=torch.float32
    )
    shape = points.new_tensor(case.shape_xyz)
    normalized = points.div(shape - 1.0).mul(2.0).sub(1.0)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            displacement_norm = _combined_model_output(model, normalized)
    finally:
        model.train(was_training)
    displacement_voxel = displacement_norm * ((shape - 1.0) / 2.0)
    warped = (points + displacement_voxel).detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(warped).all():
        raise FloatingPointError(
            "Direct INR landmark predictions contain non-finite values"
        )
    return warped


def _direct_tre_metrics(
    warped_xyz: np.ndarray, moving_xyz: np.ndarray, spacing_xyz: Sequence[float]
) -> dict[str, object]:
    warped = np.asarray(warped_xyz, dtype=np.float64)
    moving = np.asarray(moving_xyz, dtype=np.float64)
    if warped.shape != moving.shape:
        raise ValueError(
            f"Direct landmark shape mismatch: {warped.shape} vs {moving.shape}"
        )
    spacing_zyx = tuple((float(value) for value in spacing_xyz[::-1]))
    moving_zyx = landmarks_xyz_to_common_zyx(moving)
    return {
        "snap": common.compute_tre_mm(
            np.rint(landmarks_xyz_to_common_zyx(warped)), moving_zyx, spacing_zyx
        ),
        "subvoxel": common.compute_tre_mm(
            landmarks_xyz_to_common_zyx(warped), moving_zyx, spacing_zyx
        ),
    }


def evaluate_native_metrics(
    case: ControlledCase,
    dense_disp_voxel_xyz: np.ndarray,
    *,
    model: torch.nn.Module | None = None,
    direct_warped_xyz: np.ndarray | None = None,
    device: torch.device | str | int | None = None,
) -> dict[str, object]:
    """Evaluate native N data with both dense and (when given) direct INR TRE."""
    if case.protocol != "N":
        raise ValueError(
            f"evaluate_native_metrics requires protocol N, got {case.protocol}"
        )
    field = np.asarray(dense_disp_voxel_xyz)
    if field.shape != case.shape_xyz + (3,):
        raise ValueError(
            f"Dense field shape {field.shape} does not match case {case.shape_xyz + (3,)}"
        )
    field_zyx = field_xyz_to_common_zyx(field)
    fixed_zyx = landmarks_xyz_to_common_zyx(case.landmarks_fixed_xyz)
    moving_zyx = landmarks_xyz_to_common_zyx(case.landmarks_moving_xyz)
    spacing_zyx = tuple((float(value) for value in case.spacing_xyz[::-1]))
    identity = common.compute_tre_mm(fixed_zyx, moving_zyx, spacing_zyx)
    dense_snap = common.landmark_tre_from_dense_disp(
        field_zyx,
        fixed_zyx,
        moving_zyx,
        sample_mode="nearest",
        snap_output=True,
        spacing_zyx=spacing_zyx,
        dense_disp_units="native_full_voxel",
    )
    dense_subvoxel = common.landmark_tre_from_dense_disp(
        field_zyx,
        fixed_zyx,
        moving_zyx,
        sample_mode="trilinear",
        snap_output=False,
        spacing_zyx=spacing_zyx,
        dense_disp_units="native_full_voxel",
    )
    if model is not None and direct_warped_xyz is not None:
        raise ValueError("Pass model or direct_warped_xyz, not both")
    if model is not None:
        direct_warped_xyz = predict_direct_warped_landmarks_xyz(model, case, device)
    direct = (
        _direct_tre_metrics(
            direct_warped_xyz, case.landmarks_moving_xyz, case.spacing_xyz
        )
        if direct_warped_xyz is not None
        else None
    )
    result: dict[str, object] = {
        "protocol": "N",
        "coordinate_frame": "native_full_voxel_xyz",
        "spacing_order": "xyz",
        "identity_tre": identity,
        "dense_dvf": {"snap": dense_snap, "subvoxel": dense_subvoxel},
        "tre_snap_to_voxel": dense_snap,
        "tre_subvoxel": dense_subvoxel,
        "headline_tre_metric": "tre_snap_to_voxel",
        "direct_inr": direct,
        "dense_field_shape_xyz": [int(value) for value in field.shape[:3]],
    }
    if direct is not None:
        result["direct_tre_snap_to_voxel"] = direct["snap"]
        result["direct_tre_subvoxel"] = direct["subvoxel"]
    return result


def evaluate_ptv_metrics(
    case: ControlledCase,
    dense_disp_voxel_xyz: np.ndarray,
    *,
    model: torch.nn.Module | None = None,
    direct_warped_xyz: np.ndarray | None = None,
    device: torch.device | str | int | None = None,
) -> dict[str, object]:
    """Evaluate P on its 1mm grid and in the converted native-crop headline domain."""
    if case.protocol not in {"P", "PL"} or case.ptv_case is None:
        raise ValueError(
            f"evaluate_ptv_metrics requires protocol P/PL, got {case.protocol}"
        )
    field = np.asarray(dense_disp_voxel_xyz)
    if field.shape != case.shape_xyz + (3,):
        raise ValueError(
            f"Dense field shape {field.shape} does not match case {case.shape_xyz + (3,)}"
        )
    field_zyx = field_xyz_to_common_zyx(field)
    ptv = case.ptv_case
    reg_identity = common.compute_tre_mm(
        ptv.landmarks_fixed, ptv.landmarks_moving, (1.0, 1.0, 1.0)
    )
    reg_snap = common.landmark_tre_from_dense_disp(
        field_zyx,
        ptv.landmarks_fixed,
        ptv.landmarks_moving,
        sample_mode="nearest",
        snap_output=True,
        spacing_zyx=(1.0, 1.0, 1.0),
        dense_disp_units="cropped_1mm_voxel",
    )
    reg_subvoxel = common.landmark_tre_from_dense_disp(
        field_zyx,
        ptv.landmarks_fixed,
        ptv.landmarks_moving,
        sample_mode="trilinear",
        snap_output=False,
        spacing_zyx=(1.0, 1.0, 1.0),
        dense_disp_units="cropped_1mm_voxel",
    )
    native_field_zyx = common.resize_reg1mm_disp_to_native_crop_vox(
        field_zyx, ptv.crop_shape_zyx, ptv.native_spacing_zyx
    )
    native_identity = common.compute_tre_mm(
        ptv.landmarks_fixed_crop, ptv.landmarks_moving_crop, ptv.native_spacing_zyx
    )
    native_snap = common.landmark_tre_from_dense_disp(
        native_field_zyx,
        ptv.landmarks_fixed_crop,
        ptv.landmarks_moving_crop,
        sample_mode="nearest",
        snap_output=True,
        spacing_zyx=ptv.native_spacing_zyx,
        dense_disp_units="native_crop_voxel",
    )
    native_subvoxel = common.landmark_tre_from_dense_disp(
        native_field_zyx,
        ptv.landmarks_fixed_crop,
        ptv.landmarks_moving_crop,
        sample_mode="trilinear",
        snap_output=False,
        spacing_zyx=ptv.native_spacing_zyx,
        dense_disp_units="native_crop_voxel",
    )
    if model is not None and direct_warped_xyz is not None:
        raise ValueError("Pass model or direct_warped_xyz, not both")
    if model is not None:
        direct_warped_xyz = predict_direct_warped_landmarks_xyz(model, case, device)
    direct_reg = (
        _direct_tre_metrics(
            direct_warped_xyz, case.landmarks_moving_xyz, case.spacing_xyz
        )
        if direct_warped_xyz is not None
        else None
    )
    return {
        "protocol": case.protocol,
        "coordinate_frame": "ptv_reg1mm_voxel_xyz",
        "spacing_order": "xyz",
        "identity_tre": reg_identity,
        "tre_snap_to_voxel": reg_snap,
        "tre_subvoxel": reg_subvoxel,
        "registration_grid": {
            "identity": reg_identity,
            "dense_snap": reg_snap,
            "dense_subvoxel": reg_subvoxel,
            "direct_inr": direct_reg,
        },
        "native_crop_identity_tre": native_identity,
        "native_crop_tre_snap_to_voxel": native_snap,
        "native_crop_tre_subvoxel": native_subvoxel,
        "native_crop": {
            "identity": native_identity,
            "dense_snap": native_snap,
            "dense_subvoxel": native_subvoxel,
        },
        "headline_tre_metric": "native_crop_tre_snap_to_voxel",
        "dense_field_shape_xyz": [int(value) for value in field.shape[:3]],
        "native_crop_field_shape_zyx": [
            int(value) for value in native_field_zyx.shape[:3]
        ],
    }


def evaluate_controlled_metrics(
    case: ControlledCase,
    dense_disp_voxel_xyz: np.ndarray,
    *,
    model: torch.nn.Module | None = None,
    direct_warped_xyz: np.ndarray | None = None,
    device: torch.device | str | int | None = None,
) -> dict[str, object]:
    """Dispatch landmark evaluation without computing heavyweight regularity."""
    if case.protocol == "N":
        return evaluate_native_metrics(
            case,
            dense_disp_voxel_xyz,
            model=model,
            direct_warped_xyz=direct_warped_xyz,
            device=device,
        )
    return evaluate_ptv_metrics(
        case,
        dense_disp_voxel_xyz,
        model=model,
        direct_warped_xyz=direct_warped_xyz,
        device=device,
    )
