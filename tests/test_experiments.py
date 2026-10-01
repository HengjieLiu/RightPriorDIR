"""Experiment orchestration/aggregation with synthetic outputs only."""

import csv
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from rightpriordir import experiment as ex, summarize as report, timing
from rightpriordir.register import resolve_config


@pytest.mark.parametrize(
    "suite,count",
    [
        ("oasis-paper", 3600),
        ("4dct-paper", 370),
        ("copd-extension", 30),
        ("oasis-accuracy", 1000),
        ("oasis-timing", 10),
    ],
)
def test_all_suite_contracts(suite, count):
    plan = ex.build_plan(suite)
    assert len(plan["jobs"]) == count
    assert len({j["id"] for j in plan["jobs"]}) == count
    assert plan["full_suite"] and plan["full_cohort"]
    for job in plan["jobs"]:
        config, args = resolve_config(job["config"])
        assert args.epochs == 2500
        if config["method"] == "dual_inr":
            assert args.seed == 42
        else:
            assert args.seed == 1


@pytest.mark.parametrize("suite", ["oasis-paper", "4dct-paper"])
def test_paper_grids_exactly_match_frozen_rows(suite):
    plan = ex.build_plan(suite)
    with (ex.ROOT / plan["suite"]["reference"]).open() as f:
        rows = list(csv.DictReader(f))
    expected = set()
    for row in rows:
        if suite == "oasis-paper":
            if not row["alpha"]:
                continue
            method = row["method_id"].replace(
                "idir_original_dense_siren_be_autograd", "idir_original"
            )
            value = float(row["alpha"])
        else:
            if not row["reg_value"] or row["legacy_context"] == "True":
                continue
            method = (
                "dual_inr" if row["method"].startswith("dual_inr_") else row["method"]
            )
            value = float(row["reg_value"])
        expected.add((method, value))
    actual = {(j["config"]["method"], j["weight"]) for j in plan["jobs"]}
    assert actual == expected


def test_paired_schedules_and_backend_are_separate():
    quality = ex.build_plan("oasis-accuracy", "0", "mrdbscp,sinr-cps2")
    timed = ex.build_plan("oasis-timing", arms="mrdbscp,sinr-cps2")
    for job in quality["jobs"]:
        assert job["config"]["backend"] == {
            "cudnn_benchmark": True,
            "cudnn_deterministic": False,
        }
    for job in timed["jobs"]:
        _, args = resolve_config(job["config"])
        assert not args.cudnn_benchmark and args.cudnn_deterministic
        if job["schedule"] == "fast" and job["arm"] == "sinr-cps2":
            assert args.lr == pytest.approx(1e-3)
            assert (
                args.stage_min_epochs,
                args.stage_patience,
                args.stage_check_interval,
                args.stage_min_delta,
            ) == ("375", "150", "1", "0.0001")
        if job["schedule"] == "fast" and job["arm"] == "mrdbscp":
            assert args.stage_lr == "0.03,0.01,0.003"
            assert args.stage_min_epochs == "40,80,130"
    with pytest.raises(ValueError, match="Timing requires"):
        ex.build_plan("oasis-timing", "0:2")


@pytest.mark.parametrize("cases", ["0,0", "-1", "100", "2:1"])
def test_invalid_cohort_rejected(cases):
    with pytest.raises(ValueError):
        ex.build_plan("oasis-paper", cases)


def test_all_dry_plans_without_data_torch_or_writes(tmp_path):
    script = "from rightpriordir.experiment import main,SUITES; import sys\nfor suite in SUITES: main(['--suite',suite,'--data-root','/missing','--out',sys.argv[1]])\nassert 'torch' not in sys.modules"
    subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "no-output")],
        check=True,
        capture_output=True,
    )
    assert not (tmp_path / "no-output").exists()


def fake_registration(command, **kwargs):
    """Write a plausible tiny result; never invoke a registration engine."""
    option = lambda name: command[command.index(name) + 1]
    config = json.loads(Path(option("--config")).read_text())
    case = int(option("--case"))
    out = Path(option("--out"))
    out.mkdir(parents=True, exist_ok=False)
    np.save(out / "field.npy", np.zeros((4, 5, 6, 3), np.float32))
    dataset = config["dataset"]
    metadata = {
        "dataset": dataset,
        "method": config["method"],
        "sha256": ex.file_hash(out / "field.npy"),
        "case_rank" if dataset == "oasis" else "case_id": case,
        "fixed_id": "0333",
        "moving_id": "0115",
    }
    if dataset == "oasis":
        with (ex.ROOT / "splits/oasis100.csv").open() as f:
            pair = list(csv.DictReader(f))[case]
        metadata.update(fixed_id=pair["fixed_id"], moving_id=pair["moving_id"])
    else:
        metadata["crop_mode"] = config["crop_mode"]
    ex.write_json(out / "field.json", metadata)
    reg = {
        "log_jdet_std_fg": 0.2 + case * 0.01,
        "jdet_le0_pct_fg": 0.0,
        "diffusion_l2_fg": 0.3,
        "bending_energy_fg": 0.04,
    }
    if dataset == "oasis":
        metrics = {"dice_mean": 0.8 + case * 0.01, "hd95_mean": 2.0, "regularity": reg}
    else:
        metrics = {
            k: {"tre_mean_mm": 1.0 + case * 0.1}
            for k in [
                "native_crop_tre_snap_to_voxel",
                "native_crop_tre_subvoxel",
                "tre_snap_to_voxel",
                "tre_subvoxel",
            ]
        }
        metrics["regularity_fg_registration_crop"] = reg
    ex.write_json(out / "metrics.json", metrics)
    ex.write_json(out / "run_config.json", {"config": config})


@pytest.fixture
def mocked_execution(monkeypatch):
    calls = []

    def invoke(command, **kwargs):
        calls.append(command)
        fake_registration(command, **kwargs)

    monkeypatch.setattr(
        ex,
        "data_inventory",
        lambda *a: {"synthetic": {"sha256": "synthetic", "bytes": 1}},
    )
    monkeypatch.setattr(ex, "run_subprocess", invoke)
    return calls


@pytest.mark.parametrize(
    "suite,cases,arms,n",
    [
        ("oasis-paper", "0:2", "idir", 12),
        ("4dct-paper", "1:3", "dual-pl1a", 12),
        ("copd-extension", "1:3", None, 6),
    ],
)
def test_execute_resume_summary_all_datasets(
    tmp_path, mocked_execution, suite, cases, arms, n
):
    plan = ex.build_plan(suite, cases, arms)
    out = tmp_path / "runs"
    ex.execute_plan(plan, tmp_path / "data", out)
    assert len(mocked_execution) == n
    ex.execute_plan(plan, tmp_path / "data", out, resume=True)
    assert len(mocked_execution) == n  # Verified completions never retrain.
    population = report.summarize(out, tmp_path / "summary", plots=False)
    assert all(p["n_cases"] == 2 and not p["full_cohort"] for p in population)
    expected_ddof = 1  # Dual PL1A overrides the 4DCT default of zero.
    assert all(p["sd_ddof"] == expected_ddof for p in population)
    assert (tmp_path / "summary/comparison_to_frozen.csv").is_file()
    with pytest.raises(FileExistsError):
        report.summarize(out, tmp_path / "summary", plots=False)


def test_shards_partial_reporting_and_drift(tmp_path, mocked_execution):
    plan = ex.build_plan("oasis-paper", "0:2", "idir")
    out = tmp_path / "runs"
    ex.execute_plan(plan, tmp_path / "data", out, shard_count=2)
    assert len(mocked_execution) == 6
    with pytest.raises(ValueError, match="Incomplete"):
        report.collect(out)
    manifest, rows, missing = report.collect(out, allow_partial=True)
    assert len(rows) == 6 and len(missing) == 6
    assert report.stats([1.0], 1)["sd"] is None
    ex.execute_plan(
        plan, tmp_path / "data", out, resume=True, shard_count=2, shard_index=1
    )
    assert len(report.collect(out)[1]) == 12
    changed = ex.build_plan("oasis-paper", "0:3", "idir")
    with pytest.raises(ValueError, match="changed"):
        ex.execute_plan(changed, tmp_path / "data", out, resume=True, shard_count=2)


def test_failed_attempt_preserved_and_explicit_retry(
    tmp_path, monkeypatch, mocked_execution
):
    plan = ex.build_plan("copd-extension", "1", "dual-p0")
    out = tmp_path / "runs"

    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(ex, "run_subprocess", fail)
    with pytest.raises(subprocess.CalledProcessError):
        ex.execute_plan(plan, tmp_path / "data", out)
    job_dir = out / "jobs" / plan["jobs"][0]["id"]
    assert (job_dir / "attempt-0001/failed.json").is_file()
    with pytest.raises(RuntimeError, match="retry-failed"):
        ex.execute_plan(plan, tmp_path / "data", out, resume=True)
    monkeypatch.setattr(ex, "run_subprocess", fake_registration)
    ex.execute_plan(plan, tmp_path / "data", out, resume=True, retry_failed=True)
    assert (job_dir / "attempt-0001/failed.json").is_file()
    assert (job_dir / "attempt-0002/result/field.npy").is_file()
    with (job_dir / "attempt-0002/result/metrics.json").open("a") as f:
        f.write(" ")
    with pytest.raises(ValueError, match="changed"):
        report.collect(out)


def test_statistics_missing_and_denominators():
    assert report.stats([1, 3, None], 0) == {"n": 2, "mean": 2, "sd": 1}
    assert report.stats([1, 3], 1)["sd"] == pytest.approx(2**0.5)
    assert report.stats([], 1) == {"n": 0, "mean": None, "sd": None}
    with pytest.raises(ValueError, match="Missing/nonfinite"):
        report.extract_metrics({"dice_mean": float("nan"), "regularity": {}}, "oasis")


def test_timing_worker_order_no_evaluation(tmp_path, monkeypatch):
    import torch
    from rightpriordir import oasis, oasis_engine

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "synthetic")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *a: SimpleNamespace(uuid="synthetic-uuid"),
    )
    monkeypatch.setattr(timing, "gpu_snapshot", lambda: {})
    order = []

    def load(**kwargs):
        order.append(kwargs["rank"])
        return {"fixed_id": "0333", "moving_id": "0115"}

    monkeypatch.setattr(oasis, "load_oasis_case", load)

    def no_evaluation(*a, **kw):
        raise AssertionError("Timing cannot evaluate accuracy")

    monkeypatch.setattr(oasis, "evaluate_from_dense_coords", no_evaluation)

    def fake_engine(args, *a, **kw):
        assert args.output_mode == "dvf_only" and args.timing_detail == "coarse"
        return {
            "dense_disp": torch.zeros(4, 5, 6, 3),
            "actual_epochs": 2500,
            "stage_actual_epochs": [2500],
            "training_peak_mem_gb": 0.001,
            "stage_summaries": [],
            "history": [],
        }

    monkeypatch.setattr(oasis_engine, "train_spline", fake_engine)
    config = ex.build_plan("oasis-timing", arms="sinr-cps2")["jobs"][0]["config"]
    payload = timing.run_candidate(
        config, tmp_path / "data", tmp_path / "timed", "cuda:0"
    )
    assert order == [0] + list(range(10))
    timing.validate_timing_rows(payload["rows"])
    assert not payload["quality_evaluation_performed"]
    assert len(list((tmp_path / "timed").glob("*/field.json"))) == 11
    with pytest.raises(ValueError):
        timing.validate_timing_rows(payload["rows"][:-1])


def test_data_preflight_exact_files_and_pairs(tmp_path):
    plan = ex.build_plan("oasis-paper", "0", "idir")
    (tmp_path / "test").mkdir()
    (tmp_path / "pair_test_200.txt").write_bytes(
        (ex.ROOT / "splits/pair_test_200.txt").read_bytes()
    )
    for pid in ("0333", "0115"):
        for prefix in ("img", "seg"):
            (tmp_path / "test" / f"{prefix}{pid}.nii.gz").write_bytes(
                b"not-medical-data-test-only"
            )
    assert len(ex.data_inventory(plan, tmp_path)) == 5
    (tmp_path / "test/img0333.nii.gz").unlink()
    with pytest.raises(FileNotFoundError):
        ex.data_inventory(plan, tmp_path)


def test_timing_experiment_completion_and_summary(
    tmp_path, monkeypatch, mocked_execution
):
    plan = ex.build_plan("oasis-timing", arms="sinr-cps2")

    def fake_timing(command, **kwargs):
        value = lambda key: command[command.index(key) + 1]
        config = json.loads(Path(value("--config")).read_text())
        out = Path(value("--out"))
        out.mkdir()
        rows = []
        for condition, rank in timing.observation_order():
            pair = plan["pairs"][rank]
            folder = out / f"{condition}-rank{rank:03d}"
            folder.mkdir()
            np.save(folder / "field.npy", np.zeros((4, 5, 6, 3), np.float32))
            ex.write_json(folder / "field.json", {"synthetic": True})
            rows.append(
                {
                    "condition": condition,
                    "case_rank": rank,
                    "fixed_id": pair["fixed_id"],
                    "moving_id": pair["moving_id"],
                    "loaded_images_to_saved_dvf_seconds": (
                        1.0
                        if config["parameters"].get("stage_early_stop_mode")
                        == "adaptive"
                        else 2.0
                    ),
                    "actual_epochs": 2500,
                    "training_peak_mem_gb": 1.0,
                    "field_finite": True,
                    "evaluation_in_timing": False,
                    "field": str((folder / "field.npy").relative_to(out)),
                    "field_sha256": ex.file_hash(folder / "field.npy"),
                }
            )
        runtime = {
            "gpu": "test",
            "gpu_uuid": "test-uuid",
            "torch": "test",
            "cuda": "test",
            "cudnn": "test",
            "cpu_threads": 6,
            "effective_backend": config["backend"],
        }
        ex.write_json(
            out / "timing.json", {"config": config, "rows": rows, "runtime": runtime}
        )

    monkeypatch.setattr(ex, "run_subprocess", fake_timing)
    out = tmp_path / "timing"
    ex.execute_plan(plan, tmp_path / "data", out)
    pop = report.summarize(out, tmp_path / "timing-summary", plots=False)
    assert len(pop) == 2 and all(p["n_cases"] == 10 for p in pop)
    with (tmp_path / "timing-summary/paired_timing.csv").open() as f:
        paired = list(csv.DictReader(f))
    assert float(paired[0]["ratio_of_means"]) == 2.0
    assert "dice" not in (tmp_path / "timing-summary/summary.md").read_text().lower()
    with pytest.raises(ValueError, match="no cross-GPU"):
        ex.execute_plan(plan, tmp_path / "data", tmp_path / "other", shard_count=2)


def test_new_quality_plot_generated_from_synthetic_results(tmp_path, mocked_execution):
    plan = ex.build_plan("copd-extension", "1:3", "mrdbscp-p2e")
    ex.execute_plan(plan, tmp_path / "data", tmp_path / "runs")
    report.summarize(tmp_path / "runs", tmp_path / "plots")
    assert (tmp_path / "plots/rerun_arc.png").stat().st_size > 1000


def test_completion_requires_every_core_checksum(tmp_path, mocked_execution):
    plan = ex.build_plan("copd-extension", "1", "dual-p0")
    out = tmp_path / "runs"
    ex.execute_plan(plan, tmp_path / "data", out)
    path = out / "jobs" / plan["jobs"][0]["id"] / "complete.json"
    record = json.loads(path.read_text())
    record["files"].pop("attempt-0001/result/field.npy")
    ex.write_json(path, record)
    with pytest.raises(ValueError, match="required output checksums"):
        report.collect(out)


def test_changed_input_content_blocks_resume(tmp_path, monkeypatch, mocked_execution):
    plan = ex.build_plan("copd-extension", "1", "dual-p0")
    ex.execute_plan(plan, tmp_path / "data", tmp_path / "runs")
    monkeypatch.setattr(
        ex,
        "data_inventory",
        lambda *a: {"synthetic": {"sha256": "changed", "bytes": 1}},
    )
    with pytest.raises(ValueError, match="Data, code, environment"):
        ex.execute_plan(plan, tmp_path / "data", tmp_path / "runs", resume=True)


def test_timing_runtime_mismatch_rejected(tmp_path, monkeypatch):
    plan = ex.build_plan("oasis-timing", arms="sinr-cps2")
    manifest = {
        **plan,
        "data_root": "synthetic",
        "data_files": {},
        "runtime_packages": {},
        "shard_count": 1,
    }
    manifest["experiment_sha256"] = ex.digest(manifest)
    root = tmp_path / "runs"
    ex.write_json(root / "experiment.json", manifest)
    monkeypatch.setattr(
        report, "validate_completion", lambda *a: {"attempt": "attempt-0001"}
    )
    for index, job in enumerate(plan["jobs"]):
        rows = [
            {
                "condition": condition,
                "case_rank": rank,
                "loaded_images_to_saved_dvf_seconds": 1.0,
                "actual_epochs": 2500,
                "evaluation_in_timing": False,
                "field_finite": True,
            }
            for condition, rank in timing.observation_order()
        ]
        runtime = {
            "gpu": "same-name",
            "gpu_uuid": f"different-physical-gpu-{index}",
            "torch": "same",
            "cuda": "same",
            "cudnn": "same",
            "cpu_threads": 6,
            "effective_backend": job["config"]["backend"],
        }
        ex.write_json(
            root / "jobs" / job["id"] / "attempt-0001/result/timing.json",
            {"rows": rows, "runtime": runtime},
        )
    with pytest.raises(ValueError, match="different GPU"):
        report.collect(root)
