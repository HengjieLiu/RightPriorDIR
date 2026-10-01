"""Raw DIR-LAB → versioned approximately 1 mm inputs (no registration)."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from . import geometry as g
from . import raw_lung as raw


def protocol_id(dataset, crop_mode):
    if dataset not in {"4dct", "copd"} or crop_mode not in {"corrected", "legacy"}:
        raise ValueError("Unknown dataset/crop protocol")
    return f"dirlab_{dataset}_{crop_mode}_crop_nominal1mm_v1"


def case_inputs(data_root, dataset, case_id):
    resolver = raw.resolve_4dct_case if dataset == "4dct" else raw.resolve_copd_case
    case = resolver(case_id, Path(data_root))
    if case.status != "complete":
        raise ValueError(f"Incomplete {dataset} case {case_id}: {case.notes}")
    if dataset == "4dct":
        return (
            case,
            ("T00", "T50"),
            (case.image_t00, case.image_t50),
            (case.landmarks_t00, case.landmarks_t50),
        )
    return (
        case,
        ("iBH", "eBH"),
        (case.image_ibh, case.image_ebh),
        (case.landmarks_ibh, case.landmarks_ebh),
    )


def geometry(case, fixed_xyz, moving_xyz, mode):
    if mode not in {"corrected", "legacy"}:
        raise ValueError(mode)
    corrected = g.ptvreg_crop_bounds_zyx_from_landmarks_xyz(
        fixed_xyz, moving_xyz, case.shape_zyx
    )
    legacy = g.legacy_crop_bounds_zyx_from_landmarks_xyz(
        fixed_xyz, moving_xyz, case.shape_zyx
    )
    bounds = corrected if mode == "corrected" else legacy
    shape = bounds[:, 1] - bounds[:, 0] + 1
    reg_shape = np.rint(shape * np.asarray(case.spacing_zyx)).astype(int)
    if np.any(shape < 2) or np.any(reg_shape < 2):
        raise ValueError("Degenerate crop or registration grid")
    return bounds, shape, reg_shape, corrected, legacy


def prepare_case(
    data_root, mask_root, out_root, dataset, case_id, crop_mode, device="cpu"
):
    case, phases, images, landmark_paths = case_inputs(data_root, dataset, case_id)
    fixed_xyz, moving_xyz = [raw.load_landmarks_xyz(p) for p in landmark_paths]
    if fixed_xyz.shape != (300, 3) or moving_xyz.shape != (300, 3):
        raise ValueError("Expected exactly 300 paired DIR-LAB landmarks")
    bounds, crop_shape, reg_shape, corrected, legacy = geometry(
        case, fixed_xyz, moving_xyz, crop_mode
    )
    pid = protocol_id(dataset, crop_mode)
    label = "4DCT" if dataset == "4dct" else "COPD"
    out = Path(out_root) / pid / label / f"case{case_id:02d}"
    if out.exists():
        raise FileExistsError(
            f"Refusing to overwrite prepared case {out}; choose a new output root"
        )
    arrays = {}
    verification = {}
    sources = {}
    loader = raw.load_raw_4dct_image if dataset == "4dct" else raw.load_raw_copd_image
    for role, phase, img_path, points_xyz in zip(
        ("fixed", "moving"), phases, images, (fixed_xyz, moving_xyz)
    ):
        image = loader(img_path, case.shape_zyx)
        mask_path = (
            Path(mask_root)
            / label
            / f"case{case_id:02d}"
            / f"case{case_id:02d}_{phase}_R231_binary_zyx.npy"
        )
        mask = (np.load(mask_path, allow_pickle=False) > 0).astype(np.uint8)
        if mask.shape != tuple(case.shape_zyx) or not np.any(mask):
            raise ValueError(f"Invalid native lung mask {mask_path.name}")
        native = g.img_thr(image)
        cropped = g.crop_zyx(native, bounds)
        mask_crop = g.crop_zyx(mask, bounds)
        reg = g.resize_image_trilinear_zyx(cropped, reg_shape, torch.device(device))
        reg_mask = g.resize_mask_nearest_zyx(mask_crop, reg_shape, torch.device(device))
        points_native = g.landmarks_xyz_one_based_to_zyx_zero_based(points_xyz)
        points_crop = g.transform_landmarks_native0_to_crop(points_native, bounds)
        points_reg = g.transform_landmarks_crop_to_registration(
            points_crop, crop_shape, reg_shape
        )
        for grid, points, shape in [
            ("native", points_native, case.shape_zyx),
            ("crop", points_crop, crop_shape),
            ("reg", points_reg, reg_shape),
        ]:
            inside = g.points_inside_shape(points, shape)
            verification[f"{role}_{grid}_landmarks_inside"] = inside
            if not inside["all_inside"]:
                raise ValueError(f"{role} landmarks outside {grid}")
        for suffix, array in [
            ("native_ptvintensity_zyx", native),
            ("crop_ptvintensity_zyx", cropped),
            ("reg1mm_trilinear_zyx", reg),
            ("lungmask_native_zyx", mask),
            ("lungmask_crop_zyx", mask_crop),
            ("lungmask_reg1mm_nearest_zyx", reg_mask),
        ]:
            arrays[f"{role}_{phase}_{suffix}"] = array
        for grid, points in [
            ("native0", points_native),
            ("crop", points_crop),
            ("reg1mm", points_reg),
        ]:
            arrays[f"landmarks_{role}_{phase}_{grid}_zyx"] = points
        sources[role] = {
            "raw_image_name": Path(img_path).name,
            "raw_image_sha256": hashlib.sha256(Path(img_path).read_bytes()).hexdigest(),
            "native_mask_name": mask_path.name,
            "native_mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
        }
    verification["pass"] = True
    audit = {
        "protocol_name": pid,
        "dataset": dataset,
        "case_id": case_id,
        "crop_mode": crop_mode,
        "fixed_phase": phases[0],
        "moving_phase": phases[1],
        "field_direction": "fixed_to_moving_pull",
        "crop_bounds_zyx_inclusive": bounds.tolist(),
        "corrected_crop_bounds_zyx_inclusive": corrected.tolist(),
        "legacy_crop_bounds_zyx_inclusive": legacy.tolist(),
        "legacy_minus_corrected_bounds_zyx": (legacy - corrected).tolist(),
        "raw_shape_zyx": list(case.shape_zyx),
        "native_spacing_zyx_mm": list(case.spacing_zyx),
        "crop_shape_zyx": crop_shape.tolist(),
        "registration_shape_zyx": reg_shape.tolist(),
        "registration_spacing_zyx_mm": [1.0, 1.0, 1.0],
        "endpoint_spacing_zyx_mm": (
            (crop_shape - 1) * np.asarray(case.spacing_zyx) / (reg_shape - 1)
        ).tolist(),
        "spacing_note": "Nominal 1mm shape rule; endpoint-preserving interpolation is not exact physical 1mm spacing",
        "image_resampling": "torch trilinear align_corners=True",
        "mask_resampling": f"{crop_mode} crop; torch nearest",
        "intensity_rule": "clip raw [80,900], subtract80, divide820; no HU offset for registration",
        "landmark_rule": "source XYZ one-based -> ZYX zero-based -> crop -> endpoint-scaled registration grid",
        "verification": verification,
        "verification_pass": True,
        "sources": sources,
        "arrays": {},
    }
    out.mkdir(parents=True, exist_ok=False)
    for name, array in arrays.items():
        path = out / f"{name}.npy"
        np.save(path, array, allow_pickle=False)
        audit["arrays"][name] = {
            "filename": path.name,
            "shape": list(array.shape),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (out / "preprocess_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--mask-root", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--dataset", choices=["4dct", "copd"], required=True)
    p.add_argument("--crop-mode", choices=["corrected", "legacy"], required=True)
    p.add_argument("--cases", type=int, nargs="+", default=list(range(1, 11)))
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    a = p.parse_args(argv)
    if any(i not in range(1, 11) for i in a.cases):
        p.error("Case IDs must be 1–10")
    for i in a.cases:
        audit = prepare_case(
            a.data_root, a.mask_root, a.out_root, a.dataset, i, a.crop_mode, a.device
        )
        print(
            json.dumps(
                {
                    "case": i,
                    "protocol": audit["protocol_name"],
                    "verified": audit["verification_pass"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
