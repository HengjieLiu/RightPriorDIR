"""Evaluate a saved public-format displacement without registration."""

import argparse
import hashlib
import json
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--field", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    a = p.parse_args(argv)
    import numpy as np
    import torch
    from .register import json_values, check_crop

    metadata = json.loads(a.field.with_suffix(".json").read_text())
    if (
        metadata.get("component_order") != "xyz"
        or metadata.get("units") != "normalized_grid_displacement_align_corners_true"
        or metadata.get("direction") != "fixed_to_moving_pull"
    ):
        raise ValueError("Unsupported field coordinate contract")
    if hashlib.sha256(a.field.read_bytes()).hexdigest() != metadata["sha256"]:
        raise ValueError("Field checksum mismatch")
    field = torch.from_numpy(np.load(a.field, allow_pickle=False)).to(a.device)
    if tuple(field.shape) != tuple(metadata["shape_zyx"]) + (3,):
        raise ValueError("Field shape mismatch")
    if not torch.isfinite(field).all():
        raise ValueError("Field contains nonfinite values")
    if a.out.exists():
        raise FileExistsError("Use a fresh evaluation output directory")
    a.out.mkdir(parents=True, exist_ok=False)
    if metadata["dataset"] == "oasis":
        from . import oasis as common

        batch = common.load_oasis_case(
            rank=metadata["case_rank"],
            device=torch.device(a.device),
            oasis_root=a.data_root,
        )
        if tuple(field.shape[:3]) != tuple(batch["fixed"].shape):
            raise ValueError("Field/data shape mismatch")
        if (
            batch["fixed_id"] != metadata["fixed_id"]
            or batch["moving_id"] != metadata["moving_id"]
        ):
            raise ValueError("Pair mismatch")
        metrics = common.evaluate_from_dense_coords(
            field + common.make_coordinate_grid(common.IMAGE_SHAPE, field.device), batch
        )
        regularity, _, _ = common.regularity_metrics(field, batch["fixed_mask"])
        metrics["regularity"] = regularity
    else:
        from . import lung as common

        case = common.load_case(
            a.data_root, metadata["case_id"], dataset=metadata["dataset"]
        )
        if tuple(field.shape[:3]) != case.shape_zyx:
            raise ValueError("Field/data shape mismatch")
        check_crop(case, metadata["crop_mode"])
        if (
            hashlib.sha256(
                (case.case_dir / "preprocess_audit.json").read_bytes()
            ).hexdigest()
            != metadata["preprocess_audit_sha256"]
        ):
            raise ValueError("Preparation metadata differs from this field")
        metrics = common.final_metrics_from_dense(field, case, a.out, False, "e")
    (a.out / "metrics.json").write_text(
        json.dumps(json_values(metrics), indent=2) + "\n"
    )
    print(f'Wrote {a.out / "metrics.json"}')


if __name__ == "__main__":
    main()
