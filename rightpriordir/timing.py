"""One persistent OASIS candidate: cold rank 0, then warm ranks 0..9.

Called in a fresh process by the experiment runner. No accuracy evaluation is
allowed between observations; it would alter the warm-process cache state.
"""

import argparse
import gc
import json
import math
from pathlib import Path
import random
import subprocess
import time

from .experiment import file_hash, write_json
from .register import resolve_config, json_values


def observation_order():
    return [("cold", 0)] + [("warm", rank) for rank in range(10)]


def validate_timing_rows(rows):
    if [(r["condition"], r["case_rank"]) for r in rows] != observation_order():
        raise ValueError("Expected one cold rank0 and ten ordered warm observations")
    for row in rows:
        if (
            not math.isfinite(row["loaded_images_to_saved_dvf_seconds"])
            or row["loaded_images_to_saved_dvf_seconds"] <= 0
        ):
            raise ValueError("Nonfinite/nonpositive timing")
        if not 1 <= row["actual_epochs"] <= 2500:
            raise ValueError("Invalid realized optimizer-step count")
        if row["evaluation_in_timing"] or not row["field_finite"]:
            raise ValueError("Invalid timing endpoint or field")


def gpu_snapshot():
    records = {}
    for key, query in [
        ("devices", "--query-gpu=uuid,name,driver_version,memory.used,utilization.gpu"),
        (
            "compute_processes",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        ),
    ]:
        try:
            output = subprocess.run(
                ["nvidia-smi", query, "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            records[key] = (
                output.stdout.strip() if output.returncode == 0 else "unavailable"
            )
        except (OSError, subprocess.TimeoutExpired):
            records[key] = "unavailable"
    return records


def run_candidate(config, data_root, out, device_name):
    import numpy as np
    import torch
    from . import oasis, oasis_engine as engine

    config, args = resolve_config(config)
    if (
        config["dataset"] != "oasis"
        or config["method"] == "idir_original"
        or args.alpha != 1000
    ):
        raise ValueError(
            "Paired timing is restricted to the five OASIS spline methods at alpha=1000"
        )
    if config["backend"] != {"cudnn_benchmark": False, "cudnn_deterministic": True}:
        raise ValueError(
            "September 16 timing requires cuDNN benchmark=False, deterministic=True"
        )
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for timing")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args.output_mode = "dvf_only"
    args.timing_detail = "coarse"
    args.loss_history_mode = "cpu"
    rows = []
    before = gpu_snapshot()
    for condition, rank in observation_order():
        target = out / f"{condition}-rank{rank:03d}"
        target.mkdir(exist_ok=False)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        batch = oasis.load_oasis_case(
            rank=rank, device=device, oasis_root=Path(data_root)
        )
        timer = engine.TimingRecorder("coarse", device)
        torch.cuda.synchronize(device)
        # Start only after both images/labels are resident on the selected GPU.
        start = time.perf_counter()
        trainer = (
            engine.train_multistage_direct_cp
            if "multistage" in args.method
            else engine.train_spline
        )
        result = trainer(args, batch, device, args.lr, timer=timer, run_dir=target)
        torch.cuda.synchronize(device)
        field = (
            result["dense_disp"].detach().cpu().numpy().astype(np.float32, copy=False)
        )
        np.save(target / "field.npy", field, allow_pickle=False)
        elapsed = time.perf_counter() - start
        # Validation, hashing and bookkeeping are AFTER the serialized-DVF endpoint.
        finite = bool(
            field.ndim == 4 and field.shape[-1] == 3 and np.isfinite(field).all()
        )
        if not finite:
            raise ValueError(
                "Timing candidate produced a malformed/nonfinite displacement"
            )
        metadata = {
            "schema_version": 1,
            "dataset": "oasis",
            "method": args.method,
            "case_rank": rank,
            "fixed_id": batch["fixed_id"],
            "moving_id": batch["moving_id"],
            "protocol_name": "oasis_lia_to_ras_seed3_pair_minmax_v1",
            "shape_zyx": list(field.shape[:3]),
            "array_order": "zyx3",
            "component_order": "xyz",
            "units": "normalized_grid_displacement_align_corners_true",
            "direction": "fixed_to_moving_pull",
            "sha256": file_hash(target / "field.npy"),
        }
        write_json(target / "field.json", metadata)
        row = {
            "condition": condition,
            "case_rank": rank,
            "fixed_id": batch["fixed_id"],
            "moving_id": batch["moving_id"],
            "loaded_images_to_saved_dvf_seconds": elapsed,
            "actual_epochs": int(result["actual_epochs"]),
            "stage_actual_epochs": result["stage_actual_epochs"],
            "field_finite": finite,
            "training_peak_mem_gb": result["training_peak_mem_gb"],
            "evaluation_in_timing": False,
            "field": str((target / "field.npy").relative_to(out)),
            "field_sha256": metadata["sha256"],
        }
        write_json(
            target / "training.json",
            json_values(
                {
                    k: result[k]
                    for k in (
                        "actual_epochs",
                        "stage_actual_epochs",
                        "stage_summaries",
                        "stage_stop_reasons",
                        "history",
                    )
                    if k in result
                }
            ),
        )
        rows.append(row)
        write_json(out / "timing.partial.json", {"config": config, "rows": rows})
        del field, result, batch, timer
        gc.collect()  # Preserve CUDA allocator/backend state between observations.
    validate_timing_rows(rows)
    payload = {
        "config": config,
        "resolved_parameters": vars(args),
        "rows": rows,
        "endpoint": "loaded device images to serialized dense DVF; no fsync guarantee",
        "quality_evaluation_performed": False,
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "cpu_threads": torch.get_num_threads(),
            "gpu_uuid": str(
                getattr(torch.cuda.get_device_properties(device), "uuid", "unavailable")
            ),
            "requested_backend": config["backend"],
            "effective_backend": {
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
            },
        },
        "gpu_snapshot_before": before,
        "gpu_snapshot_after": gpu_snapshot(),
        "gpu_exclusivity_verified": False,
        "note": "Portable same-boundary harness, not a guarantee of identical historical runtime. Use an idle exclusive GPU; inspect process snapshots.",
    }
    write_json(out / "timing.json", json_values(payload))
    return payload


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--execute", action="store_true")
    a = p.parse_args(argv)
    config, _ = resolve_config(a.config)
    if not a.execute:
        print(
            json.dumps(
                {
                    "config": config,
                    "observations": observation_order(),
                    "optimization_requested": False,
                },
                indent=2,
            )
        )
        return
    run_candidate(config, a.data_root, a.out, a.device)


if __name__ == "__main__":
    main()
