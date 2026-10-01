"""One-pair registration interface. Default: inspect configuration, do not train."""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OASIS_METHODS = {
    "idir_original",
    "sinr_cps2_be_bspline_analytic",
    "sinr_cps4_be_bspline_analytic",
    "direct_cp_cps2_be_bspline_analytic",
    "direct_cp_cps4_be_bspline_analytic",
    "direct_cp_8_4_2_be_bspline_analytic_multistage",
}
LUNG_METHODS = {
    "idir_be_default_lungmask",
    "sinr_cps4_global_ncc_fd_be",
    "direct_cp_cps4_global_ncc_fd_be",
    "direct_cp_32_16_8_4_global_ncc_fd_be",
    "direct_cp_32_16_8_4_ptv_lcc_isotv",
    "dual_inr",
}


def resolve_config(path):
    config = json.loads(Path(path).read_text()) if not isinstance(path, dict) else path
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported config schema")
    dataset = config["dataset"]
    method = config["method"]
    if dataset not in {"oasis", "4dct", "copd"}:
        raise ValueError("Unsupported dataset")
    if method not in (OASIS_METHODS if dataset == "oasis" else LUNG_METHODS):
        raise ValueError("Method is outside this release")
    family = "oasis" if dataset == "oasis" else "lung"
    defaults = json.loads((ROOT / f"rightpriordir/{family}_defaults.json").read_text())
    parameters = config.get("parameters", {})
    unknown = set(parameters) - set(defaults) - {"alpha"}
    if unknown:
        raise ValueError(f"Unknown parameters: {sorted(unknown)}")
    deferred = {
        "sample_checkpoints",
        "finest_loss_diagnostic",
        "global_ncc_diagnostic_interval",
        "image_sim_eval_interval",
        "prolongation_diagnostics",
    }
    if any(parameters.get(k) for k in deferred):
        raise ValueError("Diagnostic experiments are deferred")
    defaults.update(parameters)
    if defaults.get("timing_detail", "off") not in {
        "off",
        "coarse",
        "deep",
        "endpoints",
    }:
        raise ValueError("Private diagnostic timing modes are not released")
    if dataset == "oasis" and defaults.get("output_mode") != "full":
        raise ValueError("Use the experiment timing suite for DVF-only benchmarking")
    defaults["method"] = method
    defaults["cudnn_benchmark"] = bool(config["backend"]["cudnn_benchmark"])
    defaults["cudnn_deterministic"] = bool(config["backend"]["cudnn_deterministic"])
    if dataset != "oasis" and config.get("crop_mode") not in {"corrected", "legacy"}:
        raise ValueError("An explicit crop_mode is required for lung configs")
    if int(defaults["epochs"]) != 2500:
        raise ValueError("Base release configuration requires 2500 steps")
    if method.startswith("direct_cp_") and (
        "multistage" in method or "32_16_8_4" in method
    ):
        if sum(map(int, defaults["stage_epochs"].split(","))) != 2500:
            raise ValueError("Stage epochs must sum to 2500")
    if method == "dual_inr":
        if config.get("dual_variant") not in (
            {"PL1A"} if dataset == "4dct" else {"P0", "P1"}
        ):
            raise ValueError("Unsupported Dual-INR variant for this dataset")
    return config, argparse.Namespace(**defaults)


def check_crop(case, expected):
    from .prepare_lung import protocol_id

    observed = case.preprocess_audit
    if (
        observed.get("protocol_name") != protocol_id(case.dataset, expected)
        or observed.get("crop_mode") != expected
    ):
        raise ValueError(
            "Preparation protocol mismatch. Use the versioned prepare-lung command; do not relabel old data."
        )
    if not observed.get("verification_pass"):
        raise ValueError("Preparation did not pass geometry verification")


def json_values(value):
    import numpy as np
    import torch

    if isinstance(value, dict):
        return {
            k: json_values(v)
            for k, v in value.items()
            if not isinstance(v, (torch.nn.Module, torch.Tensor))
        }
    if isinstance(value, (list, tuple)):
        return [json_values(v) for v in value]
    if isinstance(value, np.ndarray):
        return (
            json_values(value.tolist())
            if value.size <= 1000
            else {"shape": list(value.shape), "omitted_array": True}
        )
    if isinstance(value, np.generic):
        return json_values(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def run(config, args, data_root, case_id, out, device):
    import numpy as np
    import torch

    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    torch.backends.cudnn.deterministic = args.cudnn_deterministic
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "Registration uses CUDA. Frozen plotting and geometry checks work on CPU."
        )
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    method = config["method"]
    dataset = config["dataset"]
    if dataset == "oasis":
        from . import oasis as common, oasis_engine as engine

        batch = common.load_oasis_case(
            rank=case_id, device=device, oasis_root=data_root
        )
        lr = float(args.lr if args.lr is not None else engine.default_lr(method))
        if method == "idir_original":
            result = engine.train_idir(args, batch, device, lr)
        elif "multistage" in method:
            result = engine.train_multistage_direct_cp(args, batch, device, lr)
        else:
            result = engine.train_spline(args, batch, device, lr)
        field = result["dense_coords"] - common.make_coordinate_grid(
            common.IMAGE_SHAPE, device
        )
        metrics = {
            k: v
            for k, v in result.items()
            if k
            not in {
                "model",
                "history",
                "dense_coords",
                "warped_seg",
                "generator",
                "control_disp",
            }
        }
        regularity, _, _ = common.regularity_metrics(field, batch["fixed_mask"])
        metrics["regularity"] = regularity
        cohort = {
            "case_rank": case_id,
            "fixed_id": batch["fixed_id"],
            "moving_id": batch["moving_id"],
            "protocol_name": "oasis_lia_to_ras_seed3_pair_minmax_v1",
        }
    else:
        from . import lung as common, lung_engine as engine

        case = common.load_case(data_root, case_id, dataset=dataset)
        check_crop(case, config["crop_mode"])
        args.dataset = dataset
        args.case_id = case_id
        if method == "dual_inr":
            from . import dual_adapter, dual_data
            from . import dual_copd_data

            loader = dual_data if dataset == "4dct" else dual_copd_data
            variant = config["dual_variant"]
            protocol = dual_adapter.variant_protocol(variant)
            dual_case = loader.load_controlled_case(
                protocol=protocol,
                case_id=case_id,
                native_raw_root=data_root,
                native_mask_root=data_root,
                ptv_root=data_root,
            )
            dual_config = dual_adapter.build_controlled_config(
                variant,
                regularization_multiplier=float(
                    config.get("regularization_multiplier", 1.0)
                ),
            )
            dual_config["controlled_adapter"]["seed"] = int(args.seed)
            result = dual_adapter.train_controlled(
                fixed_image_xyz=dual_case.fixed,
                moving_image_xyz=dual_case.moving,
                sampling_mask_xyz=dual_case.sampling_mask_xyz,
                config=dual_config,
                device=device,
            )
            xyz = dual_data.decode_dense_normalized_field_xyz(
                result["model"],
                dual_case.shape_xyz,
                device=device,
                max_points_per_chunk=args.eval_chunk_size,
            )
            # DUAL uses spatial XYZ; the public field schema uses spatial ZYX, components XYZ.
            field = torch.from_numpy(np.ascontiguousarray(xyz.transpose(2, 1, 0, 3)))
            result = {
                "training_config": dual_config,
                "history_tail": result["history"][-10:],
            }
        elif method == "idir_be_default_lungmask":
            result = engine.train_idir(args, case, device, out)
        elif "32_16_8_4" in method:
            result = engine.train_multistage(args, case, device, out)
        else:
            result = engine.train_single_stage_spline(
                args, case, device, out, direct_cp=method.startswith("direct_cp")
            )
        if method != "dual_inr":
            field = result["dense_disp_norm_xyz"]
        metrics = common.final_metrics_from_dense(field, case, out, False, "e")
        metrics["training"] = result.get("training", result.get("training_config", {}))
        cohort = {
            "case_id": case_id,
            "protocol_name": case.preprocess_audit["protocol_name"],
            "crop_mode": config["crop_mode"],
            "preprocess_audit_sha256": hashlib.sha256(
                (case.case_dir / "preprocess_audit.json").read_bytes()
            ).hexdigest(),
        }
    array = field.detach().cpu().numpy().astype(np.float32)
    if array.ndim != 4 or array.shape[-1] != 3 or not np.isfinite(array).all():
        raise ValueError("Registration produced a malformed or nonfinite field")
    np.save(out / "field.npy", array, allow_pickle=False)
    metadata = {
        "schema_version": 1,
        "dataset": dataset,
        "method": method,
        **cohort,
        "shape_zyx": list(array.shape[:3]),
        "array_order": "zyx3",
        "component_order": "xyz",
        "units": "normalized_grid_displacement_align_corners_true",
        "direction": "fixed_to_moving_pull",
        "sha256": hashlib.sha256((out / "field.npy").read_bytes()).hexdigest(),
    }
    (out / "field.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (out / "metrics.json").write_text(json.dumps(json_values(metrics), indent=2) + "\n")
    return metadata


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument(
        "--case", type=int, required=True, help="OASIS rank 0–99; lung case 1–10"
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--alpha", type=float, help="Optional ARC regularization weight override"
    )
    p.add_argument(
        "--execute",
        action="store_true",
        help="Explicitly launch optimization; omitted by default",
    )
    a = p.parse_args(argv)
    config, args = resolve_config(a.config)
    if a.case not in (range(100) if config["dataset"] == "oasis" else range(1, 11)):
        p.error("Case is outside the released cohort")
    if a.alpha is not None:
        if not math.isfinite(a.alpha) or a.alpha < 0:
            p.error("alpha must be finite and nonnegative")
        if config["method"] == "dual_inr":
            config["regularization_multiplier"] = a.alpha
        elif config["method"].endswith("ptv_lcc_isotv"):
            args.ptv_isotv_weight = a.alpha
        elif config["method"] == "idir_be_default_lungmask":
            args.alpha_bending = a.alpha
        else:
            args.alpha = a.alpha
    plan = {
        "config": config,
        "resolved_parameters": vars(args),
        "case": a.case,
        "optimization_requested": a.execute,
        "step_note": "Dual-INR PL1A uses 3000 author-blur steps; all other bundled configurations use 2500",
    }
    if not a.execute:
        print(json.dumps(plan, indent=2))
        return
    if a.out.exists():
        raise FileExistsError("Refusing to overwrite a run; use a new --out directory")
    import torch

    a.out.mkdir(parents=True, exist_ok=False)
    import importlib.metadata

    plan["runtime"] = {
        name: importlib.metadata.version(name)
        for name in ["torch", "numpy", "scipy", "monai"]
    }
    plan["runtime"].update(
        cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        upstream_lock_sha256=hashlib.sha256(
            (ROOT / "third_party/lock.json").read_bytes()
        ).hexdigest(),
        device=str(a.device),
        requested_backend=config["backend"],
    )
    (a.out / "run_config.json").write_text(json.dumps(plan, indent=2) + "\n")
    start = time.time()
    metadata = run(config, args, a.data_root, a.case, a.out, torch.device(a.device))
    plan["runtime"]["effective_backend"] = {
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }
    plan["runtime"]["gpu"] = torch.cuda.get_device_name(torch.device(a.device))
    plan["runtime"]["cpu_threads"] = torch.get_num_threads()
    (a.out / "run_config.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(
        json.dumps(
            {
                "field": str(a.out / "field.npy"),
                "metadata": metadata,
                "elapsed_seconds": time.time() - start,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
