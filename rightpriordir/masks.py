"""Generate native-grid R231 lung masks; model weights are downloaded separately."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .prepare_lung import case_inputs
from . import raw_lung as raw


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--dataset", choices=["4dct", "copd"], required=True)
    p.add_argument("--cases", type=int, nargs="+", default=list(range(1, 11)))
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Local R231 .pth checkpoint; see docs/data.md",
    )
    a = p.parse_args(argv)
    if any(i not in range(1, 11) for i in a.cases):
        p.error("Case IDs must be 1–10")
    if not a.weights.is_file():
        p.error("R231 checkpoint not found")
    import SimpleITK as sitk
    from lungmask import LMInferer

    inferer = LMInferer(
        modelname="R231", modelpath=str(a.weights), force_cpu=a.device == "cpu"
    )
    loader = raw.load_raw_4dct_image if a.dataset == "4dct" else raw.load_raw_copd_image
    label = "4DCT" if a.dataset == "4dct" else "COPD"
    weight_sha = hashlib.sha256(a.weights.read_bytes()).hexdigest()
    for i in a.cases:
        case, phases, images, _ = case_inputs(a.data_root, a.dataset, i)
        out = a.out_root / label / f"case{i:02d}"
        if out.exists():
            raise FileExistsError(f"Refusing to overwrite masks: {out}")
        out.mkdir(parents=True, exist_ok=False)
        for phase, img_path in zip(phases, images):
            # DIR-LAB raw stored values are shifted CT intensities; R231 expects HU.
            image = (
                loader(img_path, case.shape_zyx).astype(np.float32) - 1024.0
            ).astype(np.int16)
            itk = sitk.GetImageFromArray(image)
            itk.SetSpacing(tuple(reversed(case.spacing_zyx)))
            labels = inferer.apply(itk).astype(np.uint8)
            prefix = f"case{i:02d}_{phase}_R231"
            np.save(out / f"{prefix}_labels_zyx.npy", labels, allow_pickle=False)
            np.save(
                out / f"{prefix}_binary_zyx.npy",
                (labels > 0).astype(np.uint8),
                allow_pickle=False,
            )
            record = {
                "dataset": a.dataset,
                "case_id": i,
                "phase": phase,
                "model": "R231",
                "weights_sha256": weight_sha,
                "hu_offset": -1024,
                "spacing_zyx_mm": list(case.spacing_zyx),
                "shape_zyx": list(labels.shape),
                "binary_rule": "labels > 0",
                "lungmask_version": "0.2.20",
            }
            (out / f"{prefix}_summary.json").write_text(
                json.dumps(record, indent=2) + "\n"
            )
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
