"""Explicit, resumable cohort reruns. Without --execute, only print a plan."""

import argparse
from contextlib import contextmanager
import copy
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import math
import re
from pathlib import Path
import subprocess
from subprocess import run as run_subprocess
import sys
import time

from .register import ROOT, resolve_config

SUITES = (
    "oasis-paper",
    "4dct-paper",
    "copd-extension",
    "oasis-accuracy",
    "oasis-timing",
)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    """Atomic replacement for our own state files, never for medical inputs."""
    import os
    import tempfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        name = f.name
    os.replace(name, path)


@contextmanager
def locked(path, blocking=False):
    """Linux advisory lock: kernel releases it if a worker crashes."""
    with Path(path).open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as e:
            raise RuntimeError(f"Another worker owns {path}") from e
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def selected_cases(text, allowed):
    if not text:
        return list(allowed)
    values = []
    for part in text.split(","):
        if ":" in part:
            start, stop = map(int, part.split(":"))
            values.extend(range(start, stop))
        else:
            values.append(int(part))
    if not values or len(set(values)) != len(values) or not set(values) <= set(allowed):
        raise ValueError(
            "Cases must be unique and inside the suite; ranges use start:stop (stop excluded)"
        )
    return sorted(values)


def apply_weight(config, weight):
    config = copy.deepcopy(config)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("Weights must be finite and nonnegative")
    method = config["method"]
    if method == "dual_inr":
        if weight <= 0:
            raise ValueError("Dual regularization multiplier must be positive")
        config["regularization_multiplier"] = weight
    else:
        key = (
            "ptv_isotv_weight"
            if method.endswith("ptv_lcc_isotv")
            else "alpha_bending" if method == "idir_be_default_lungmask" else "alpha"
        )
        config.setdefault("parameters", {})[key] = weight
    return config


def accelerated(config):
    """The selected August quality / September 16 timing schedules, not CP-BE."""
    config = copy.deepcopy(config)
    p = config["parameters"]
    p["stage_early_stop_mode"] = "adaptive"
    if "multistage" in config["method"]:
        p.update(
            lr=0.01,
            stage_lr="0.03,0.01,0.003",
            stage_min_epochs="40,80,130",
            stage_patience="20,35,50",
            stage_check_interval="25,25,25",
            stage_rel_delta_total="0.0001,0.0001,0.0001",
            stage_rel_delta_data="0.00005,0.00005,0.00005",
        )
    else:
        p.update(
            lr=float(p["lr"]) * 10,
            stage_min_epochs="375",
            stage_patience="150",
            stage_check_interval="1",
            stage_min_delta="0.0001",
        )
    return config


def source_inventory():
    paths = []
    for directory in ("rightpriordir", "configs", "experiments", "splits"):
        paths.extend(
            p
            for p in (ROOT / directory).rglob("*")
            if p.is_file() and p.suffix in {".py", ".json", ".csv", ".txt"}
        )
    paths.extend(
        ROOT / p for p in ("requirements.txt", "Dockerfile", "third_party/lock.json")
    )
    return {p.relative_to(ROOT).as_posix(): file_hash(p) for p in sorted(paths)}


def build_plan(suite_id, cases=None, arms=None):
    if suite_id not in SUITES:
        raise ValueError(f"Unknown suite: {suite_id}")
    suite = json.loads((ROOT / f"experiments/{suite_id}.json").read_text())
    if suite["schema_version"] != 1:
        raise ValueError("Unsupported experiment schema")
    allowed = list(range(suite["cases"][0], suite["cases"][1] + 1))
    cases = selected_cases(cases, allowed)
    if suite["kind"] == "timing" and cases != allowed:
        raise ValueError(
            "Timing requires cold rank0 followed by all ten warm ranks in one process"
        )
    requested = set(arms.split(",")) if arms else {a["id"] for a in suite["arms"]}
    known = {a["id"] for a in suite["arms"]}
    if not requested or not requested <= known:
        raise ValueError(f"Unknown arms; choose from {sorted(known)}")
    jobs = []
    for arm in suite["arms"]:
        if arm["id"] not in requested:
            continue
        for weight in arm["weights"]:
            base = apply_weight(
                json.loads((ROOT / arm["config"]).read_text()), float(weight)
            )
            if base["dataset"] != suite["dataset"] or base.get(
                "crop_mode"
            ) != suite.get("crop_mode"):
                raise ValueError("Suite and configuration data protocols differ")
            if "backend" in suite:
                base["backend"] = suite["backend"]
            for schedule in (
                ["base", "fast"] if suite.get("paired_schedules") else ["base"]
            ):
                config = (
                    accelerated(base) if schedule == "fast" else copy.deepcopy(base)
                )
                resolve_config(config)
                weight_id = (
                    format(float(weight), ".17g").replace(".", "p").replace("+", "")
                )
                candidate = f'{arm["id"]}-{schedule}-w{weight_id}'
                for case in [None] if suite["kind"] == "timing" else cases:
                    jobs.append(
                        {
                            "id": (
                                candidate
                                if case is None
                                else f"{candidate}-case{case:03d}"
                            ),
                            "arm": arm["id"],
                            "schedule": schedule,
                            "weight": float(weight),
                            "case": case,
                            "config": config,
                        }
                    )
    pairs = []
    if suite["dataset"] == "oasis":
        with (ROOT / "splits/oasis100.csv").open() as f:
            all_pairs = list(csv.DictReader(f))
        pairs = [all_pairs[i] for i in cases]
    plan = {
        "schema_version": 1,
        "suite": suite,
        "cases": cases,
        "pairs": pairs,
        "selected_arms": sorted(requested),
        "full_cohort": cases == allowed,
        "full_suite": cases == allowed and requested == known,
        "jobs": jobs,
        "source_files": source_inventory(),
        "reference_sha256": file_hash(ROOT / suite["reference"]),
    }
    plan["plan_sha256"] = digest(plan)
    return plan


def data_inventory(plan, data_root):
    """Hash exact run inputs once per invocation; no tensors or optimization."""
    root = Path(data_root).resolve()
    dataset = plan["suite"]["dataset"]
    paths = set()
    if dataset == "oasis":
        pairs_path = root / "pair_test_200.txt"
        if file_hash(pairs_path) != file_hash(ROOT / "splits/pair_test_200.txt"):
            # Whitespace differences are harmless; order and identities are not.
            from .oasis import parse_pair_file

            if parse_pair_file(pairs_path) != parse_pair_file(
                ROOT / "splits/pair_test_200.txt"
            ):
                raise ValueError("OASIS pair file differs from the released order")
        paths.add(pairs_path)
        for pair in plan["pairs"]:
            for role in ("fixed_id", "moving_id"):
                for prefix in ("img", "seg"):
                    paths.add(root / "test" / f"{prefix}{pair[role]}.nii.gz")
    else:
        from .lung import case_dir, _case_paths
        from .prepare_lung import protocol_id

        for case in plan["cases"]:
            cdir = case_dir(root, case, dataset)
            files, _, _ = _case_paths(cdir, dataset)
            audit = json.loads(files["audit"].read_text())
            if audit.get("protocol_name") != protocol_id(
                dataset, plan["suite"]["crop_mode"]
            ) or not audit.get("verification_pass"):
                raise ValueError(f"Wrong or unverified preparation: {cdir}")
            paths.update(files.values())
    return {
        p.relative_to(root).as_posix(): {
            "sha256": file_hash(p),
            "bytes": p.stat().st_size,
        }
        for p in sorted(paths)
    }


def runtime_inventory():
    return {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}


def validate_completion(job_dir, job, manifest):
    path = job_dir / "complete.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    if (
        result["job_sha256"] != digest(job)
        or result["experiment_sha256"] != manifest["experiment_sha256"]
    ):
        raise ValueError(f"Completion belongs to a different experiment: {job_dir}")
    if not result.get("files"):
        raise ValueError(f"Empty completion: {job_dir}")
    attempt = result.get("attempt", "")
    if not re.fullmatch(r"attempt-\d{4,}", attempt):
        raise ValueError("Invalid completed attempt path")
    names = (
        ("field.npy", "field.json", "metrics.json", "run_config.json")
        if manifest["suite"]["kind"] == "quality"
        else ("timing.json",)
    )
    required = {f"{attempt}/result/{name}" for name in names} | {
        f"{attempt}/config.json"
    }
    if not required <= result["files"].keys():
        raise ValueError("Completion is missing required output checksums")
    for relative, expected in result["files"].items():
        target = (job_dir / relative).resolve()
        if (
            not target.is_relative_to(job_dir.resolve())
            or file_hash(target) != expected
        ):
            raise ValueError(f"Completed output changed: {target}")
    return result


def seal_job(job_dir, attempt, job, manifest, elapsed):
    result = attempt / "result"
    if manifest["suite"]["kind"] == "quality":
        required = [
            result / p
            for p in ("field.npy", "field.json", "metrics.json", "run_config.json")
        ]
        metadata = json.loads((result / "field.json").read_text())
        if metadata["sha256"] != file_hash(result / "field.npy"):
            raise ValueError("Field checksum differs from metadata")
        if (
            metadata["dataset"] != manifest["suite"]["dataset"]
            or metadata["method"] != job["config"]["method"]
        ):
            raise ValueError("Result method/dataset mismatch")
        key = "case_rank" if metadata["dataset"] == "oasis" else "case_id"
        if metadata[key] != job["case"]:
            raise ValueError("Result case mismatch")
        if metadata["dataset"] == "oasis":
            pair = next(p for p in manifest["pairs"] if int(p["rank"]) == job["case"])
            if (metadata["fixed_id"], metadata["moving_id"]) != (
                pair["fixed_id"],
                pair["moving_id"],
            ):
                raise ValueError("Result pair mismatch")
        elif metadata["crop_mode"] != job["config"]["crop_mode"]:
            raise ValueError("Result crop mismatch")
        if (
            json.loads((result / "run_config.json").read_text())["config"]
            != job["config"]
        ):
            raise ValueError("Executed config differs from planned config")
        from .summarize import extract_metrics

        extract_metrics(
            json.loads((result / "metrics.json").read_text()), metadata["dataset"]
        )
    else:
        from .timing import validate_timing_rows

        timed = json.loads((result / "timing.json").read_text())
        if timed["config"] != job["config"]:
            raise ValueError("Timed config differs from planned config")
        validate_timing_rows(timed["rows"])
        required = [result / "timing.json"]
        pairs = {int(p["rank"]): p for p in manifest["pairs"]}
        for row in timed["rows"]:
            pair = pairs[row["case_rank"]]
            if (row["fixed_id"], row["moving_id"]) != (
                pair["fixed_id"],
                pair["moving_id"],
            ):
                raise ValueError("Timing pair mismatch")
            field = (result / row["field"]).resolve()
            if (
                not field.is_relative_to(result.resolve())
                or file_hash(field) != row["field_sha256"]
            ):
                raise ValueError("Timing field checksum/path mismatch")
            required.extend([field, field.with_suffix(".json")])
    required += [attempt / "config.json"]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    # Capture every emitted field/metric/provenance file; logs are advisory only.
    files = [p for p in result.rglob("*") if p.is_file()] + [attempt / "config.json"]
    receipt = {
        "job_sha256": digest(job),
        "experiment_sha256": manifest["experiment_sha256"],
        "attempt": attempt.name,
        "elapsed_process_seconds": elapsed,
        "files": {p.relative_to(job_dir).as_posix(): file_hash(p) for p in files},
    }
    write_json(job_dir / "complete.json", receipt)
    return receipt


def execute_plan(
    plan,
    data_root,
    out,
    device="cuda:0",
    resume=False,
    retry_failed=False,
    shard_index=0,
    shard_count=1,
    keep_going=False,
):
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Require 0 <= shard-index < shard-count")
    if plan["suite"]["kind"] == "timing" and shard_count != 1:
        raise ValueError(
            "Paired timing must be sequential on one exclusive GPU; no cross-GPU sharding"
        )
    out = Path(out).resolve()
    data_root = Path(data_root).resolve()
    if out == data_root or out.is_relative_to(data_root):
        raise ValueError("Outputs must be outside the data root")
    if out.is_relative_to(ROOT) and not out.is_relative_to(ROOT / "outputs"):
        raise ValueError(
            "Do not write runs into release source; use an external mount or outputs/"
        )
    from .upstream import verify

    for name in ("IDIR", "SINR", "DUAL-INR-DIR"):
        verify(name)
    manifest = {
        **plan,
        "data_root": str(data_root),
        "data_files": data_inventory(plan, data_root),
        "runtime_packages": runtime_inventory(),
        "shard_count": shard_count,
    }
    manifest["experiment_sha256"] = digest(manifest)
    out.mkdir(parents=True, exist_ok=True)
    with locked(out / ".manifest.lock", blocking=True):
        target = out / "experiment.json"
        if target.exists():
            if not resume:
                raise FileExistsError(
                    "Existing experiment; use --resume or a new --out"
                )
            if json.loads(target.read_text()) != manifest:
                raise ValueError(
                    "Data, code, environment, cohort or config changed; use a new experiment directory"
                )
        else:
            if any(p.name != ".manifest.lock" for p in out.iterdir()):
                raise FileExistsError("Output is not an empty experiment directory")
            write_json(target, manifest)
    failures = []
    for index, job in enumerate(plan["jobs"]):
        if index % shard_count != shard_index:
            continue
        job_dir = out / "jobs" / job["id"]
        job_dir.mkdir(parents=True, exist_ok=True)
        with locked(job_dir / ".lock"):
            if validate_completion(job_dir, job, manifest):
                print(f'SKIP verified {job["id"]}', flush=True)
                continue
            previous = sorted(job_dir.glob("attempt-*"))
            if previous and not retry_failed:
                raise RuntimeError(
                    f"Incomplete {job['id']}; --retry-failed preserves old attempts and starts a new one"
                )
            attempt = job_dir / f"attempt-{len(previous) + 1:04d}"
            attempt.mkdir(exist_ok=False)
            config_path = attempt / "config.json"
            write_json(config_path, job["config"])
            result = attempt / "result"
            command = [sys.executable, "-m", "rightpriordir"]
            if plan["suite"]["kind"] == "quality":
                command += ["register", "--case", str(job["case"])]
            else:
                command = [sys.executable, "-m", "rightpriordir.timing"]
            command += [
                "--config",
                str(config_path),
                "--data-root",
                str(data_root),
                "--out",
                str(result),
                "--device",
                device,
                "--execute",
            ]
            write_json(attempt / "command.json", {"argv": command, "job": job})
            start = time.perf_counter()
            print(f'RUN {index+1}/{len(plan["jobs"])} {job["id"]}', flush=True)
            try:
                with (attempt / "run.log").open("w") as log:
                    run_subprocess(
                        command, stdout=log, stderr=subprocess.STDOUT, check=True
                    )
                seal_job(job_dir, attempt, job, manifest, time.perf_counter() - start)
            except Exception as e:
                write_json(
                    attempt / "failed.json",
                    {"error": str(e), "elapsed_seconds": time.perf_counter() - start},
                )
                failures.append(job["id"])
                if not keep_going:
                    raise
    if failures:
        raise RuntimeError(f"Failed jobs: {failures}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", choices=SUITES, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cases", help="E.g. 0:10 or 1,2; stop is excluded")
    p.add_argument("--arms", help="Comma-separated arm IDs from the suite JSON")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--keep-going", action="store_true")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--list-jobs", action="store_true")
    p.add_argument(
        "--check-data",
        action="store_true",
        help="Read/hash required inputs, but never optimize without --execute",
    )
    a = p.parse_args(argv)
    plan = build_plan(a.suite, a.cases, a.arms)
    if a.shard_count < 1 or not 0 <= a.shard_index < a.shard_count:
        p.error("Invalid shard index/count")
    if plan["suite"]["kind"] == "timing" and a.shard_count != 1:
        p.error("Timing uses one exclusive GPU; sharding is disabled")
    summary = {
        "suite": a.suite,
        "dataset": plan["suite"]["dataset"],
        "cases": plan["cases"],
        "arms": plan["selected_arms"],
        "full_suite": plan["full_suite"],
        "jobs": len(plan["jobs"]),
        "observations": len(plan["jobs"])
        * (11 if plan["suite"]["kind"] == "timing" else 1),
        "jobs_on_this_shard": sum(
            i % a.shard_count == a.shard_index for i in range(len(plan["jobs"]))
        ),
        "plan_sha256": plan["plan_sha256"],
        "optimization_requested": a.execute,
        "excluded": plan["suite"].get("excluded", []),
    }
    if a.list_jobs:
        summary["job_list"] = plan["jobs"]
    if a.check_data and not a.execute:
        summary["input_files_verified"] = len(data_inventory(plan, a.data_root))
    print(json.dumps(summary, indent=2), flush=True)
    if a.execute:
        execute_plan(
            plan,
            a.data_root,
            a.out,
            a.device,
            a.resume,
            a.retry_failed,
            a.shard_index,
            a.shard_count,
            a.keep_going,
        )


if __name__ == "__main__":
    main()
