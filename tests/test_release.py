"""Offline synthetic checks; no optimizer step and no patient data."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from rightpriordir import geometry as g, lung, oasis
from rightpriordir import prepare_lung as prep
from rightpriordir.frozen import ROOT, read_rows, verify_inputs, write_tables
from rightpriordir.register import resolve_config, check_crop
from rightpriordir.upstream import siren, verify


@pytest.fixture(autouse=True)
def forbid_optimization(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Optimization must not run in release acceptance")

    monkeypatch.setattr(torch.optim.Adam, "step", fail)
    monkeypatch.setattr(torch.optim.AdamW, "step", fail)


def test_frozen_checksums_and_pairs():
    assert len(verify_inputs()["inputs"]) >= 16
    pairs = read_rows(ROOT / "splits/oasis100.csv")
    assert len(pairs) == 100
    assert read_rows(ROOT / "splits/oasis10.csv") == pairs[:10]
    assert (pairs[0]["fixed_id"], pairs[0]["moving_id"]) == ("0333", "0115")
    rows = read_rows(ROOT / "results/updated_oasis/timing_cases.csv")
    assert len(rows) == 110
    for row in rows:
        pair = pairs[int(row["case_rank"])]
        assert row["fixed_id"].zfill(4) == pair["fixed_id"]
        assert row["moving_id"].zfill(4) == pair["moving_id"]


def test_copd_coverage_and_aggregation():
    cases = read_rows(ROOT / "results/copd_extension/cases.csv")
    for population in read_rows(ROOT / "results/copd_extension/population.csv"):
        rows = [r for r in cases if r["method"] == population["method"]]
        assert {int(r["case_id"]) for r in rows} == set(range(1, 11))
        values = np.array([float(r["headline_native_snap_mm"]) for r in rows])
        assert values.mean() == pytest.approx(
            float(population["headline_native_snap_mm_mean"])
        )
        assert values.std(ddof=1) == pytest.approx(
            float(population["headline_native_snap_mm_sample_sd"])
        )


def test_tables_keep_cohorts_separate(tmp_path):
    write_tables(tmp_path)
    timing = (tmp_path / "updated_oasis_timing.md").read_text()
    assert "N=10" in timing and "57.72" in timing and "9.15" in timing
    assert "Dice" not in timing
    assert (
        "separate experiment"
        in (tmp_path / "historical_oasis100_accuracy.md").read_text()
    )
    assert "corrected MATLAB crop" in (tmp_path / "copd_extension.md").read_text()


@pytest.mark.parametrize(
    "path",
    sorted((ROOT / "configs").rglob("*.json")),
    ids=lambda p: str(p.relative_to(ROOT)),
)
def test_configuration(path):
    config, args = resolve_config(path)
    assert args.epochs == 2500
    assert config["backend"].keys() == {"cudnn_benchmark", "cudnn_deterministic"}


def test_crop_interior_shift():
    points = np.array([[40, 50, 20], [80, 90, 60]], float)
    corrected = g.ptvreg_crop_bounds_zyx_from_landmarks_xyz(
        points, points, (100, 128, 128)
    )
    legacy = g.legacy_crop_bounds_zyx_from_landmarks_xyz(
        points, points, (100, 128, 128)
    )
    np.testing.assert_array_equal(legacy - corrected, np.ones((3, 2), int))


def test_copd01_clipped_geometry():
    points = np.array([[81, 125, 1], [426, 425, 104]], float)
    corrected = g.ptvreg_crop_bounds_zyx_from_landmarks_xyz(
        points, points, (121, 512, 512)
    )
    legacy = g.legacy_crop_bounds_zyx_from_landmarks_xyz(
        points, points, (121, 512, 512)
    )
    np.testing.assert_array_equal(corrected, [[0, 108], [114, 434], [70, 435]])
    np.testing.assert_array_equal(legacy, [[0, 109], [115, 435], [71, 436]])
    assert corrected[0, 1] - corrected[0, 0] + 1 == 109
    assert legacy[0, 1] - legacy[0, 0] + 1 == 110


@pytest.mark.parametrize("mode", ["corrected", "legacy"])
def test_coordinate_roundtrip(mode):
    points = np.array([[40, 50, 20], [80, 90, 60]], float)
    case = SimpleNamespace(shape_zyx=(100, 128, 128), spacing_zyx=(2.5, 0.97, 0.97))
    bounds, shape, reg, _, _ = prep.geometry(case, points, points, mode)
    native = g.landmarks_xyz_one_based_to_zyx_zero_based(points)
    crop = g.transform_landmarks_native0_to_crop(native, bounds)
    registered = g.transform_landmarks_crop_to_registration(crop, shape, reg)
    restored = registered * (shape - 1) / (reg - 1) + bounds[:, 0]
    np.testing.assert_allclose(restored, native, rtol=0, atol=1e-12)
    assert g.points_inside_shape(registered, reg)["inside_count"] == 2


def test_prepare_synthetic_and_refuse_overwrite(tmp_path, monkeypatch):
    shape = (14, 16, 18)
    points = np.tile([[9.0, 8.0, 7.0]], (300, 1))
    raw_path = tmp_path / "raw.img"
    raw_path.write_bytes(b"synthetic-only")
    case = SimpleNamespace(shape_zyx=shape, spacing_zyx=(2.5, 1.0, 1.0))
    monkeypatch.setattr(
        prep,
        "case_inputs",
        lambda *a: (case, ("iBH", "eBH"), (raw_path, raw_path), ("fixed", "moving")),
    )
    monkeypatch.setattr(prep.raw, "load_landmarks_xyz", lambda p: points)
    monkeypatch.setattr(
        prep.raw, "load_raw_copd_image", lambda *a: np.full(shape, 490, dtype=np.int16)
    )
    mask_dir = tmp_path / "masks/COPD/case01"
    mask_dir.mkdir(parents=True)
    for phase in ("iBH", "eBH"):
        np.save(
            mask_dir / f"case01_{phase}_R231_binary_zyx.npy", np.ones(shape, np.uint8)
        )
    audit = prep.prepare_case(
        tmp_path, tmp_path / "masks", tmp_path / "prepared", "copd", 1, "legacy"
    )
    assert audit["verification_pass"] and audit["mask_resampling"].startswith("legacy")
    root = tmp_path / "prepared" / prep.protocol_id("copd", "legacy")
    loaded = lung.load_case(root, 1, "copd")
    check_crop(loaded, "legacy")
    np.testing.assert_allclose(loaded.fixed, 0.5, rtol=0, atol=1e-7)
    assert loaded.fixed.shape == loaded.fixed_mask.shape
    with pytest.raises(ValueError):
        check_crop(loaded, "corrected")
    with pytest.raises(FileExistsError):
        prep.prepare_case(
            tmp_path, tmp_path / "masks", tmp_path / "prepared", "copd", 1, "legacy"
        )


@pytest.mark.parametrize("name", ["IDIR", "SINR", "DUAL-INR-DIR"])
def test_upstream_pin(name):
    _, commit = verify(name)
    assert len(commit) == 40


def test_siren_same_initialization_and_gradients():
    torch.manual_seed(1)
    a = siren("IDIR", [3, 16, 16, 3], 32).double()
    torch.manual_seed(1)
    b = siren("SINR", [3, 16, 16, 3], 32).double()
    x = torch.randn(11, 3, dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(a(x), b(x), rtol=0, atol=0)
    ga = torch.autograd.grad(a(x).square().sum(), x)[0]
    gb = torch.autograd.grad(b(x).square().sum(), x)[0]
    torch.testing.assert_close(ga, gb, rtol=0, atol=0)
    be = lung.idir_bending_energy(x, a(x), len(x))
    assert torch.isfinite(be)
    grads = torch.autograd.grad(be, list(a.parameters()), allow_unused=True)
    assert any(v is not None and v.abs().sum() > 0 for v in grads)


@pytest.mark.parametrize("module", [oasis, lung])
def test_bspline_knot_insertion_preserves_field(module):
    shape = (13, 15, 17)
    shapes = module.build_nested_cp_shapes(shape, 4, [4, 2])
    torch.manual_seed(3)
    coarse = torch.randn(*shapes[4], 3, dtype=torch.float64)
    fine = module.cubic_bspline_dyadic_knot_insert_3d(coarse, shapes[2])
    a = module.expand_bspline_controls(coarse.reshape(-1, 3), shapes[4], shape, 4)
    b = module.expand_bspline_controls(fine.reshape(-1, 3), shapes[2], shape, 2)
    torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


def test_native_tre_identity_and_translation():
    field = np.zeros((8, 10, 12, 3), np.float32)
    field[..., 0] = 1.0
    fixed = np.array([[2.0, 3.0, 4.0], [4.0, 5.0, 6.0]])
    target = fixed + np.array([1.0, 0.0, 0.0])
    result = lung.landmark_tre_from_dense_disp(
        field,
        fixed,
        target,
        sample_mode="nearest",
        snap_output=True,
        spacing_zyx=(2.5, 1.0, 1.0),
    )
    assert result["tre_mean_mm"] == 0


def test_dual_axis_roundtrip_and_schedules():
    from rightpriordir import dual_data, dual_adapter

    field = np.zeros((7, 8, 9, 3), np.float32)
    field[..., 0] = 0.25
    voxel = dual_data.normalized_field_xyz_to_voxel_xyz(field)
    zyx = dual_data.field_xyz_to_common_zyx(voxel)
    assert zyx.shape == (9, 8, 7, 3)
    np.testing.assert_allclose(zyx[..., 2], 0.75)
    for variant, steps in [("P0", 2500), ("P1", 2500), ("PL1A", 3000)]:
        assert (
            dual_adapter.build_controlled_config(variant)["training"]["total_steps"]
            == steps
        )


@pytest.mark.parametrize(
    "path",
    sorted((ROOT / "configs").rglob("*.json")),
    ids=lambda p: str(p.relative_to(ROOT)),
)
def test_registration_dispatch_without_optimizer(path, tmp_path, monkeypatch):
    """Exercise the public save/metadata dispatch with synthetic engine outputs."""
    from rightpriordir import (
        register,
        oasis_engine,
        lung_engine,
        dual_adapter,
        dual_data,
        dual_copd_data,
    )

    config, args = resolve_config(path)
    dataset = config["dataset"]
    shape = (8, 9, 10)
    field = torch.full((*shape, 3), 0.125)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    called = []
    if dataset == "oasis":
        batch = {
            "fixed_id": "0333",
            "moving_id": "0115",
            "fixed_mask": torch.ones(shape),
        }
        monkeypatch.setattr(oasis, "load_oasis_case", lambda **kw: batch)
        monkeypatch.setattr(
            oasis, "make_coordinate_grid", lambda *a: torch.zeros((*shape, 3))
        )
        monkeypatch.setattr(oasis, "regularity_metrics", lambda *a: ({}, [], []))

        def engine(*a, **kw):
            called.append("engine")
            return {"dense_coords": field, "dice_mean": 0.9}

        for name in ["train_idir", "train_spline", "train_multistage_direct_cp"]:
            monkeypatch.setattr(oasis_engine, name, engine)
        case_id = 0
    else:
        audit = {
            "protocol_name": prep.protocol_id(dataset, config["crop_mode"]),
            "crop_mode": config["crop_mode"],
            "verification_pass": True,
        }
        (tmp_path / "preprocess_audit.json").write_text(json.dumps(audit))
        case = SimpleNamespace(
            dataset=dataset, preprocess_audit=audit, case_dir=tmp_path, shape_zyx=shape
        )
        monkeypatch.setattr(lung, "load_case", lambda *a, **kw: case)
        monkeypatch.setattr(
            lung, "final_metrics_from_dense", lambda *a: {"synthetic_metric": 0.0}
        )

        def engine(*a, **kw):
            called.append("engine")
            return {"dense_disp_norm_xyz": field, "training": {}}

        for name in ["train_idir", "train_single_stage_spline", "train_multistage"]:
            monkeypatch.setattr(lung_engine, name, engine)
        dual_case = SimpleNamespace(
            fixed=np.zeros(shape[::-1]),
            moving=np.zeros(shape[::-1]),
            sampling_mask_xyz=np.ones(shape[::-1]),
            shape_xyz=shape[::-1],
        )
        for mod in [dual_data, dual_copd_data]:
            monkeypatch.setattr(mod, "load_controlled_case", lambda **kw: dual_case)

        def dual_engine(**kw):
            called.append("engine")
            return {"model": torch.nn.Identity(), "history": []}

        monkeypatch.setattr(dual_adapter, "train_controlled", dual_engine)
        monkeypatch.setattr(
            dual_data,
            "decode_dense_normalized_field_xyz",
            lambda *a, **kw: field.numpy().transpose(2, 1, 0, 3),
        )
        case_id = 1
    meta = register.run(
        config, args, tmp_path, case_id, tmp_path, torch.device("cuda:0")
    )
    assert called == ["engine"]
    assert meta["direction"] == "fixed_to_moving_pull" and meta["shape_zyx"] == list(
        shape
    )
    np.testing.assert_array_equal(np.load(tmp_path / "field.npy"), field.numpy())
    assert (tmp_path / "field.json").is_file() and (tmp_path / "metrics.json").is_file()


def test_dry_run_cli_has_no_outputs(tmp_path):
    import subprocess, sys

    target = tmp_path / "must_not_exist"
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "rightpriordir",
            "register",
            "--config",
            str(ROOT / "configs/oasis/mrdbscp.json"),
            "--data-root",
            str(tmp_path / "no_dataset"),
            "--case",
            "0",
            "--out",
            str(target),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(run.stdout)["optimization_requested"] is False
    assert not target.exists()


def test_frozen_import_does_not_import_torch():
    import subprocess, sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            'import sys; import rightpriordir.frozen; assert "torch" not in sys.modules',
        ],
        check=True,
    )
