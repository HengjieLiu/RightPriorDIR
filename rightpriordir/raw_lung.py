"""Audited registration/data kernels; see docs/methods.md for protocol scope."""

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

FOURDCT_SHAPES_ZYX: Dict[int, Tuple[int, int, int]] = {
    1: (94, 256, 256),
    2: (112, 256, 256),
    3: (104, 256, 256),
    4: (99, 256, 256),
    5: (106, 256, 256),
    6: (128, 512, 512),
    7: (136, 512, 512),
    8: (128, 512, 512),
    9: (128, 512, 512),
    10: (120, 512, 512),
}

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

COPD_SHAPES_ZYX: Dict[int, Tuple[int, int, int]] = {
    1: (121, 512, 512),
    2: (102, 512, 512),
    3: (126, 512, 512),
    4: (126, 512, 512),
    5: (131, 512, 512),
    6: (119, 512, 512),
    7: (112, 512, 512),
    8: (115, 512, 512),
    9: (116, 512, 512),
    10: (135, 512, 512),
}

COPD_SPACING_ZYX: Dict[int, Tuple[float, float, float]] = {
    1: (2.5, 0.625, 0.625),
    2: (2.5, 0.645, 0.645),
    3: (2.5, 0.652, 0.652),
    4: (2.5, 0.59, 0.59),
    5: (2.5, 0.647, 0.647),
    6: (2.5, 0.633, 0.633),
    7: (2.5, 0.625, 0.625),
    8: (2.5, 0.586, 0.586),
    9: (2.5, 0.664, 0.664),
    10: (2.5, 0.742, 0.742),
}


@dataclass
class FourDCTCaseFiles:
    case_id: int
    status: str
    root: Optional[str]
    image_t00: Optional[str]
    image_t50: Optional[str]
    landmarks_t00: Optional[str]
    landmarks_t50: Optional[str]
    shape_zyx: Tuple[int, int, int]
    spacing_zyx: Tuple[float, float, float]
    notes: List[str]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class COPDCaseFiles:
    case_id: int
    status: str
    root: Optional[str]
    image_ibh: Optional[str]
    image_ebh: Optional[str]
    landmarks_ibh: Optional[str]
    landmarks_ebh: Optional[str]
    shape_zyx: Tuple[int, int, int]
    spacing_zyx: Tuple[float, float, float]
    notes: List[str]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _candidate_case_roots(fourdct_root: Path, case_id: int) -> List[Path]:
    roots = sorted(fourdct_root.glob(f"Case{case_id}Pack*"))
    expanded: List[Path] = []
    for root in roots:
        expanded.append(root)
        expanded.extend(sorted((p for p in root.iterdir() if p.is_dir())))
    return expanded


def _find_subdir(root: Path, names: Sequence[str]) -> Optional[Path]:
    for name in names:
        path = root / name
        if path.is_dir():
            return path
    return None


def _select_phase_image(
    image_dir: Optional[Path], case_id: int, phase: str
) -> Optional[Path]:
    if image_dir is None:
        return None
    patterns = [
        f"case{case_id}_T{phase}_s.img",
        f"case{case_id}_T{phase}-ssm.img",
        f"case{case_id}_T{phase}.img",
        f"Case{case_id}_T{phase}_s.img",
        f"Case{case_id}_T{phase}-ssm.img",
        f"Case{case_id}_T{phase}.img",
    ]
    for pattern in patterns:
        matches = sorted(image_dir.glob(pattern))
        if matches:
            return matches[0]
    wildcard = sorted(image_dir.glob(f"*case{case_id}*T{phase}*.img"))
    return wildcard[0] if wildcard else None


def _select_phase_landmarks(
    landmark_dir: Optional[Path], case_id: int, phase: str
) -> Optional[Path]:
    if landmark_dir is None:
        return None
    patterns = [
        f"Case{case_id}_300_T{phase}_xyz.txt",
        f"case{case_id}_dirLab300_T{phase}_xyz.txt",
        f"Case{case_id}_dirLab300_T{phase}_xyz.txt",
        f"case{case_id}_300_T{phase}_xyz.txt",
    ]
    for pattern in patterns:
        matches = sorted(landmark_dir.glob(pattern))
        if matches:
            return matches[0]
    wildcard = sorted(landmark_dir.glob(f"*case{case_id}*T{phase}*xyz.txt"))
    wildcard += sorted(landmark_dir.glob(f"*Case{case_id}*T{phase}*xyz.txt"))
    return wildcard[0] if wildcard else None


def _candidate_copd_roots(data_root: Path, case_id: int) -> List[Path]:
    roots: List[Path] = []
    for base in (data_root / "raw" / "COPD",):
        roots.extend(
            [base / f"copd{case_id}" / f"copd{case_id}", base / f"copd{case_id}"]
        )
        roots.extend(sorted(base.glob(f"copd{case_id}/copd{case_id}")))
        roots.extend(sorted(base.glob(f"copd{case_id}")))
    deduped: List[Path] = []
    seen = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            deduped.append(root)
            seen.add(key)
    return deduped


def _select_copd_image(root: Path, case_id: int, phase: str) -> Optional[Path]:
    patterns = [f"copd{case_id}_{phase}CT.img", f"*copd{case_id}*_{phase}CT.img"]
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    return None


def _select_copd_landmarks(root: Path, case_id: int, phase: str) -> Optional[Path]:
    patterns = [
        f"copd{case_id}_300_{phase}_xyz_r1.txt",
        f"*copd{case_id}*300*{phase}*xyz*.txt",
    ]
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    return None


def resolve_4dct_case(case_id: int, data_root: Path) -> FourDCTCaseFiles:
    root = data_root / "raw" / "4DCT"
    notes: List[str] = []
    selected_root: Optional[Path] = None
    image_t00 = image_t50 = landmarks_t00 = landmarks_t50 = None
    for candidate in _candidate_case_roots(root, case_id):
        image_dir = _find_subdir(candidate, ("Images", "images"))
        landmark_dir = _find_subdir(candidate, ("ExtremePhases", "extremePhases"))
        t00_img = _select_phase_image(image_dir, case_id, "00")
        t50_img = _select_phase_image(image_dir, case_id, "50")
        t00_lm = _select_phase_landmarks(landmark_dir, case_id, "00")
        t50_lm = _select_phase_landmarks(landmark_dir, case_id, "50")
        score = sum((v is not None for v in (t00_img, t50_img, t00_lm, t50_lm)))
        current_score = sum(
            (
                v is not None
                for v in (image_t00, image_t50, landmarks_t00, landmarks_t50)
            )
        )
        if score > current_score:
            selected_root = candidate
            (image_t00, image_t50) = (t00_img, t50_img)
            (landmarks_t00, landmarks_t50) = (t00_lm, t50_lm)
    if selected_root is None:
        status = "missing"
        notes.append("No case root found.")
    elif all(
        (v is not None for v in (image_t00, image_t50, landmarks_t00, landmarks_t50))
    ):
        status = "complete"
    else:
        status = "partial"
        missing = []
        for name, value in (
            ("image_t00", image_t00),
            ("image_t50", image_t50),
            ("landmarks_t00", landmarks_t00),
            ("landmarks_t50", landmarks_t50),
        ):
            if value is None:
                missing.append(name)
        notes.append("Missing: " + ", ".join(missing))
    return FourDCTCaseFiles(
        case_id=case_id,
        status=status,
        root=str(selected_root) if selected_root else None,
        image_t00=str(image_t00) if image_t00 else None,
        image_t50=str(image_t50) if image_t50 else None,
        landmarks_t00=str(landmarks_t00) if landmarks_t00 else None,
        landmarks_t50=str(landmarks_t50) if landmarks_t50 else None,
        shape_zyx=FOURDCT_SHAPES_ZYX[case_id],
        spacing_zyx=FOURDCT_SPACING_ZYX[case_id],
        notes=notes,
    )


def resolve_copd_case(case_id: int, data_root: Path) -> COPDCaseFiles:
    notes: List[str] = []
    selected_root: Optional[Path] = None
    image_ibh = image_ebh = landmarks_ibh = landmarks_ebh = None
    for candidate in _candidate_copd_roots(data_root, case_id):
        if not candidate.is_dir():
            continue
        ibh_img = _select_copd_image(candidate, case_id, "iBH")
        ebh_img = _select_copd_image(candidate, case_id, "eBH")
        ibh_lm = _select_copd_landmarks(candidate, case_id, "iBH")
        ebh_lm = _select_copd_landmarks(candidate, case_id, "eBH")
        score = sum((v is not None for v in (ibh_img, ebh_img, ibh_lm, ebh_lm)))
        current_score = sum(
            (
                v is not None
                for v in (image_ibh, image_ebh, landmarks_ibh, landmarks_ebh)
            )
        )
        if score > current_score:
            selected_root = candidate
            (image_ibh, image_ebh) = (ibh_img, ebh_img)
            (landmarks_ibh, landmarks_ebh) = (ibh_lm, ebh_lm)
    if selected_root is None:
        status = "missing"
        notes.append("No COPD case root found.")
    elif all(
        (v is not None for v in (image_ibh, image_ebh, landmarks_ibh, landmarks_ebh))
    ):
        status = "complete"
    else:
        status = "partial"
        missing = []
        for name, value in (
            ("image_ibh", image_ibh),
            ("image_ebh", image_ebh),
            ("landmarks_ibh", landmarks_ibh),
            ("landmarks_ebh", landmarks_ebh),
        ):
            if value is None:
                missing.append(name)
        notes.append("Missing: " + ", ".join(missing))
    return COPDCaseFiles(
        case_id=case_id,
        status=status,
        root=str(selected_root) if selected_root else None,
        image_ibh=str(image_ibh) if image_ibh else None,
        image_ebh=str(image_ebh) if image_ebh else None,
        landmarks_ibh=str(landmarks_ibh) if landmarks_ibh else None,
        landmarks_ebh=str(landmarks_ebh) if landmarks_ebh else None,
        shape_zyx=COPD_SHAPES_ZYX[case_id],
        spacing_zyx=COPD_SPACING_ZYX[case_id],
        notes=notes,
    )


def load_raw_lungct_image(path: Path | str, shape_zyx: Sequence[int]) -> np.ndarray:
    arr = np.fromfile(Path(path), dtype=np.int16)
    expected = int(np.prod(shape_zyx))
    if arr.size != expected:
        raise ValueError(
            f"{path} has {arr.size} voxels, expected {expected} for shape {tuple(shape_zyx)}"
        )
    return arr.reshape(tuple((int(v) for v in shape_zyx))).astype(np.float32)


def load_raw_4dct_image(path: Path | str, shape_zyx: Sequence[int]) -> np.ndarray:
    return load_raw_lungct_image(path, shape_zyx)


def load_raw_copd_image(path: Path | str, shape_zyx: Sequence[int]) -> np.ndarray:
    return load_raw_lungct_image(path, shape_zyx)


def load_landmarks_xyz(path: Path | str) -> np.ndarray:
    rows: List[List[float]] = []
    with Path(path).open() as f:
        for line in f:
            if not line.strip():
                continue
            rows.append([float(v) for v in line.strip().split()[:3]])
    arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected Nx3 landmarks in {path}, got {arr.shape}")
    return arr


def landmark_mask_fraction(mask: np.ndarray, landmarks_zyx: np.ndarray) -> float:
    idx = np.rint(landmarks_zyx).astype(int)
    valid = (
        (idx[:, 0] >= 0)
        & (idx[:, 0] < mask.shape[0])
        & (idx[:, 1] >= 0)
        & (idx[:, 1] < mask.shape[1])
        & (idx[:, 2] >= 0)
        & (idx[:, 2] < mask.shape[2])
    )
    if not np.any(valid):
        return float("nan")
    vals = mask[idx[valid, 0], idx[valid, 1], idx[valid, 2]] > 0
    return float(np.mean(vals))


def mask_root_case_dir(mask_root: Path, case_id: int, dataset: str = "4dct") -> Path:
    dataset_norm = str(dataset).lower()
    dataset_dir = {"4dct": "4DCT", "copd": "COPD"}.get(dataset_norm)
    if dataset_dir is None:
        raise ValueError(f"Unsupported LungCT dataset: {dataset!r}")
    canonical_root = mask_root / dataset_dir
    legacy_root = mask_root / dataset_norm
    selected_root = (
        canonical_root
        if canonical_root.is_dir() or not legacy_root.is_dir()
        else legacy_root
    )
    return selected_root / f"case{case_id:02d}"
