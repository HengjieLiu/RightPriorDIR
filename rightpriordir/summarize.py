"""Aggregate only checksum-verified new runs; never overwrite frozen results."""

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics

from .experiment import ROOT, digest, file_hash, validate_completion, write_json
from .frozen import md_table


def scalar(value, required=False):
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        if required:
            raise ValueError("Missing/nonfinite headline or regularity metric")
        return None
    return float(value)


def extract_metrics(metrics, dataset):
    if dataset == "oasis":
        reg = metrics["regularity"]
        values = {
            "dice": scalar(metrics.get("dice_mean"), True),
            "hd95": scalar(metrics.get("hd95_mean")),
            "asd": scalar(metrics.get("asd_symmetric_mean")),
        }
    else:
        reg = metrics["regularity_fg_registration_crop"]
        values = {
            name: scalar(metrics[key].get("tre_mean_mm"), True)
            for name, key in [
                ("tre_native_snap_mm", "native_crop_tre_snap_to_voxel"),
                ("tre_native_subvoxel_mm", "native_crop_tre_subvoxel"),
                ("tre_reg1mm_snap_mm", "tre_snap_to_voxel"),
                ("tre_reg1mm_subvoxel_mm", "tre_subvoxel"),
            ]
        }
    values.update(
        sdlogj=scalar(reg.get("log_jdet_std_fg"), True),
        folding_pct=scalar(reg.get("jdet_le0_pct_fg"), True),
        diffusion_l2=scalar(reg.get("diffusion_l2_fg")),
        bending_energy=scalar(reg.get("bending_energy_fg")),
    )
    return values


def stats(values, ddof):
    values = [v for v in values if v is not None and math.isfinite(v)]
    n = len(values)
    return {
        "n": n,
        "mean": statistics.mean(values) if n else None,
        "sd": (
            (statistics.stdev(values) if ddof else statistics.pstdev(values))
            if n > ddof
            else None
        ),
    }


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def collect(root, allow_partial=False):
    root = Path(root)
    manifest = json.loads((root / "experiment.json").read_text())
    check = {k: v for k, v in manifest.items() if k != "experiment_sha256"}
    if digest(check) != manifest["experiment_sha256"]:
        raise ValueError("Experiment manifest checksum mismatch")
    jobs = manifest["jobs"]
    if len({j["id"] for j in jobs}) != len(jobs):
        raise ValueError("Duplicate jobs in experiment manifest")
    rows, missing = [], []
    timing_runtime = None
    for job in jobs:
        job_dir = root / "jobs" / job["id"]
        completion = validate_completion(job_dir, job, manifest)
        if completion is None:
            missing.append(job["id"])
            continue
        result = job_dir / completion["attempt"] / "result"
        common = {
            "arm": job["arm"],
            "schedule": job["schedule"],
            "weight": job["weight"],
            "method": job["config"]["method"],
            "dataset": manifest["suite"]["dataset"],
            "seed": job["config"]["parameters"].get("seed", 1),
            "cudnn_benchmark": job["config"]["backend"]["cudnn_benchmark"],
            "cudnn_deterministic": job["config"]["backend"]["cudnn_deterministic"],
            "crop_mode": job["config"].get("crop_mode", "not_applicable"),
        }
        if manifest["suite"]["kind"] == "timing":
            from .timing import validate_timing_rows

            timed = json.loads((result / "timing.json").read_text())
            observed = timed["runtime"]
            identity = {
                k: observed[k]
                for k in (
                    "gpu",
                    "gpu_uuid",
                    "torch",
                    "cuda",
                    "cudnn",
                    "cpu_threads",
                    "effective_backend",
                )
            }
            if timing_runtime is not None and identity != timing_runtime:
                raise ValueError(
                    "Timing candidates used different GPU/runtime/backend identities"
                )
            timing_runtime = identity
            measurements = timed["rows"]
            validate_timing_rows(measurements)
            rows.extend({**common, **r} for r in measurements)
        else:
            metadata = json.loads((result / "field.json").read_text())
            row = {
                **common,
                "case": job["case"],
                "fixed_id": metadata.get("fixed_id", ""),
                "moving_id": metadata.get("moving_id", ""),
                "process_wall_seconds_includes_loading_and_evaluation": completion[
                    "elapsed_process_seconds"
                ],
                **extract_metrics(
                    json.loads((result / "metrics.json").read_text()), common["dataset"]
                ),
            }
            rows.append(row)
    if missing and not allow_partial:
        raise ValueError(
            f"Incomplete experiment: {len(missing)}/{len(jobs)} jobs missing; use --allow-partial only for explicitly partial reports"
        )
    if not rows:
        raise ValueError("No verified completed runs to summarize")
    return manifest, rows, missing


def population_rows(manifest, rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["arm"], row["schedule"], row["weight"])].append(row)
    timing = manifest["suite"]["kind"] == "timing"
    metrics = (
        ["loaded_images_to_saved_dvf_seconds", "actual_epochs", "training_peak_mem_gb"]
        if timing
        else (
            ["dice", "hd95", "asd"]
            if manifest["suite"]["dataset"] == "oasis"
            else [
                "tre_native_snap_mm",
                "tre_native_subvoxel_mm",
                "tre_reg1mm_snap_mm",
                "tre_reg1mm_subvoxel_mm",
            ]
        )
    )
    if not timing:
        metrics += ["sdlogj", "folding_pct", "diffusion_l2", "bending_energy"]
    population = []
    for (arm, schedule, weight), group in groups.items():
        selected = [r for r in group if r["condition"] == "warm"] if timing else group
        cases = [r["case_rank"] if timing else r["case"] for r in selected]
        if len(set(cases)) != len(cases) or not set(cases) <= set(manifest["cases"]):
            raise ValueError("Duplicate or unexpected cases in aggregation")
        arm_spec = next(a for a in manifest["suite"]["arms"] if a["id"] == arm)
        row = {
            "arm": arm,
            "schedule": schedule,
            "weight": weight,
            "n_cases": len(cases),
            "expected_cases": len(manifest["cases"]),
            "sd_ddof": arm_spec.get("sd_ddof", manifest["suite"]["sd_ddof"]),
            "full_cohort": manifest["full_cohort"],
            "dataset": manifest["suite"]["dataset"],
            "crop_mode": group[0]["crop_mode"],
            "seed": group[0]["seed"],
            "cudnn_benchmark": group[0]["cudnn_benchmark"],
            "cudnn_deterministic": group[0]["cudnn_deterministic"],
        }
        for metric in metrics:
            for key, value in stats(
                [r.get(metric) for r in selected], row["sd_ddof"]
            ).items():
                row[f"{metric}_{key}"] = value
        if timing:
            row["cold_seconds"] = next(
                r["loaded_images_to_saved_dvf_seconds"]
                for r in group
                if r["condition"] == "cold"
            )
        population.append(row)
    return population


def comparison_rows(manifest, population):
    """Descriptive new-minus-frozen differences, never an equality acceptance gate."""
    reference = ROOT / manifest["suite"]["reference"]
    if file_hash(reference) != manifest["reference_sha256"]:
        raise ValueError("Frozen comparison source changed")
    with reference.open() as f:
        frozen = list(csv.DictReader(f))
    rows = []
    for p in population:
        suite = manifest["suite"]["id"]
        method = next(
            j["config"]["method"] for j in manifest["jobs"] if j["arm"] == p["arm"]
        )
        candidate, pairs = None, []
        if suite == "oasis-paper":
            method = (
                "idir_original_dense_siren_be_autograd"
                if method == "idir_original"
                else method
            )
            candidate = next(
                (
                    r
                    for r in frozen
                    if r["method_id"] == method
                    and r["alpha"]
                    and math.isclose(float(r["alpha"]), p["weight"])
                ),
                None,
            )
            pairs = [("dice", "dice_mean_mean"), ("sdlogj", "log_jdet_std_fg_mean")]
        elif suite == "4dct-paper":
            method = (
                "dual_inr_pl1a_author_blur_native_loss"
                if method == "dual_inr"
                else method
            )
            candidate = next(
                (
                    r
                    for r in frozen
                    if r["method"] == method
                    and r["reg_value"]
                    and math.isclose(float(r["reg_value"]), p["weight"])
                ),
                None,
            )
            pairs = [
                ("tre_native_snap_mm", "original_snap_tre_mean_mm_mean"),
                ("sdlogj", "fg_logj_std_mean"),
            ]
        elif suite == "copd-extension":
            name = {"dual-p0": "P0", "dual-p1": "P1", "mrdbscp-p2e": "MR-D-BSCP P2e"}[
                p["arm"]
            ]
            candidate = next(r for r in frozen if r["method"] == name)
            pairs = [
                ("tre_native_snap_mm", "headline_native_snap_mm_mean"),
                ("sdlogj", "log_jdet_std_fg_mean"),
            ]
        if candidate:
            for metric, key in pairs:
                if candidate.get(key) and p.get(metric + "_mean") is not None:
                    old = float(candidate[key])
                    rows.append(
                        {
                            "arm": p["arm"],
                            "schedule": p["schedule"],
                            "weight": p["weight"],
                            "metric": metric,
                            "new_mean": p[metric + "_mean"],
                            "frozen_mean": old,
                            "new_minus_frozen": p[metric + "_mean"] - old,
                            "new_n": p["n_cases"],
                            "full_cohort": p["full_cohort"],
                            "interpretation": "descriptive; no automatic equivalence claim",
                        }
                    )
    return rows


def plot_population(out, population, dataset):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    groups = defaultdict(list)
    for row in population:
        groups[(row["arm"], row["schedule"])].append(row)
    y = "dice" if dataset == "oasis" else "tre_native_snap_mm"
    for (arm, schedule), points in groups.items():
        points.sort(key=lambda p: p["weight"])
        counts = ",".join(map(str, sorted({p["n_cases"] for p in points})))
        ax.plot(
            [p["sdlogj_mean"] for p in points],
            [p[y + "_mean"] for p in points],
            "o-",
            label=f"{arm}/{schedule} (N={counts})",
        )
    ax.set_xlabel("Foreground SD(log J)")
    ax.set_ylabel("Dice" if dataset == "oasis" else "Native snapped TRE (mm)")
    ax.set_title(f"New rerun results — {dataset}; not frozen Figure 2")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "rerun_arc.png", dpi=180)
    fig.savefig(out / "rerun_arc.pdf")
    plt.close(fig)


def summarize(root, out, allow_partial=False, plots=True):
    manifest, rows, missing = collect(root, allow_partial)
    population = population_rows(manifest, rows)
    out = Path(out).resolve()
    if out.is_relative_to(ROOT) and not out.is_relative_to(ROOT / "outputs"):
        raise ValueError("Keep new summaries outside release source/frozen results")
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "cases.csv", rows)
    write_csv(out / "population.csv", population)
    comparisons = comparison_rows(manifest, population)
    write_csv(out / "comparison_to_frozen.csv", comparisons)
    timing = manifest["suite"]["kind"] == "timing"
    metric = (
        "loaded_images_to_saved_dvf_seconds"
        if timing
        else "dice" if manifest["suite"]["dataset"] == "oasis" else "tre_native_snap_mm"
    )
    table = [
        [
            p["arm"],
            p["schedule"],
            p["weight"],
            f'{p["n_cases"]}/{p["expected_cases"]}',
            p[metric + "_mean"],
            p[metric + "_sd"],
        ]
        for p in population
    ]
    note = f'Suite: {manifest["suite"]["id"]}. Full cohort: {manifest["full_cohort"]}. Full suite: {manifest["full_suite"]}. Missing jobs: {len(missing)}. SD convention is explicit per row in population.csv (Dual PL1A uses sample SD).'
    (out / "summary.md").write_text(
        "# New experiment results\n\n"
        + note
        + "\n\nThese are new measurements, not replacements for the bundled paper values.\n\n"
        + md_table(
            ["Arm", "Schedule", "Weight", "N/expected", metric + " mean", "SD"], table
        )
    )
    if plots and not timing:
        plot_population(out, population, manifest["suite"]["dataset"])
    if timing:
        pairs = []
        for base in population:
            if base["schedule"] != "base":
                continue
            fast = next(
                (
                    r
                    for r in population
                    if r["arm"] == base["arm"] and r["schedule"] == "fast"
                ),
                None,
            )
            if fast:
                pairs.append(
                    {
                        "arm": base["arm"],
                        "warm_n": base["n_cases"],
                        "base_warm_seconds": base[metric + "_mean"],
                        "fast_warm_seconds": fast[metric + "_mean"],
                        "ratio_of_means": base[metric + "_mean"]
                        / fast[metric + "_mean"],
                    }
                )
        write_csv(out / "paired_timing.csv", pairs)
    write_json(
        out / "summary.json",
        {
            "experiment_sha256": manifest["experiment_sha256"],
            "suite": manifest["suite"]["id"],
            "full_cohort": manifest["full_cohort"],
            "full_suite": manifest["full_suite"],
            "missing_jobs": missing,
            "completed_jobs": len(manifest["jobs"]) - len(missing),
            "optimization_launched_by_summary": False,
            "files": {
                p.name: file_hash(p) for p in sorted(out.iterdir()) if p.is_file()
            },
        },
    )
    return population


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    a = p.parse_args(argv)
    rows = summarize(a.run_root, a.out, a.allow_partial, not a.no_plots)
    print(f"Wrote {len(rows)} population rows to {a.out}")


if __name__ == "__main__":
    main()
