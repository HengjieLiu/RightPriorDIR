"""Synthetic organizer tests: no medical images or optimization required."""

import csv
import json
import subprocess
import sys

import pytest
from rightpriordir import prepare_oasis as prep


@pytest.fixture
def source(tmp_path, monkeypatch):
    archive = tmp_path / "archive"
    archive.mkdir()
    subjects = ["OASIS_OAS1_0123_MR1", "OASIS_OAS1_0370_MR1"]
    (archive / "subjects.txt").write_text("\n".join(subjects) + "\n")
    mapping = tmp_path / "map.csv"
    pairs = tmp_path / "pairs.txt"
    pairs.write_text("# fixed ID, moving ID\n(0002, 0001)\n")
    with mapping.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["release_id", "neurite_subject", "image_sha256", "seg_sha256"])
        for number, subject in enumerate(subjects, 1):
            folder = archive / subject
            folder.mkdir()
            hashes = []
            for name in ("aligned_norm.nii.gz", "aligned_seg35.nii.gz"):
                path = folder / name
                path.write_bytes(f"synthetic-{number}-{name}".encode())
                hashes.append(prep.sha256(path))
            writer.writerow([f"{number:04d}", subject, *hashes])
    monkeypatch.setattr(prep, "SUBJECT_MAP", mapping)
    monkeypatch.setattr(prep, "PAIR_FILE", pairs)
    monkeypatch.setattr(prep, "SUBJECTS_SHA256", prep.sha256(archive / "subjects.txt"))
    return archive


def test_pinned_public_mapping():
    with prep.SUBJECT_MAP.open() as stream:
        rows = {r["release_id"]: r for r in csv.DictReader(stream)}
    assert len(rows) == 84
    assert rows["0333"]["neurite_subject"] == "OASIS_OAS1_0370_MR1"
    assert rows["0115"]["neurite_subject"] == "OASIS_OAS1_0123_MR1"
    assert all(
        len(r["image_sha256"]) == len(r["seg_sha256"]) == 64 for r in rows.values()
    )


@pytest.mark.parametrize("check", [False, True])
def test_preview_creates_nothing(source, tmp_path, check):
    out = tmp_path / "prepared"
    plan = prep.prepare(source, out, check_data=check)
    assert not out.exists()
    assert plan["subject_count"] == 2
    assert len(plan["files"]) == 4


def test_copy_preserves_bytes_and_ordinal_ids(source, tmp_path):
    out = tmp_path / "prepared"
    plan = prep.prepare(source, out, execute=True)
    for record in plan["files"]:
        assert (out / record["target"]).read_bytes() == (
            source / record["source"]
        ).read_bytes()
    assert (out / "test/img0001.nii.gz").read_bytes() == (
        source / "OASIS_OAS1_0123_MR1/aligned_norm.nii.gz"
    ).read_bytes()
    assert not (out / "test/img0123.nii.gz").exists()
    assert (out / "pair_test_200.txt").read_bytes() == prep.PAIR_FILE.read_bytes()
    assert json.loads((out / "preparation.json").read_text()) == plan
    with pytest.raises(FileExistsError):
        prep.prepare(source, out, execute=True)


@pytest.mark.parametrize("fault", ["order", "content", "missing", "map"])
def test_reject_bad_inputs_without_output(source, tmp_path, fault):
    if fault == "order":
        (source / "subjects.txt").write_text("wrong order\n")
    elif fault == "content":
        (source / "OASIS_OAS1_0123_MR1/aligned_norm.nii.gz").write_bytes(b"changed")
    elif fault == "missing":
        (source / "OASIS_OAS1_0123_MR1/aligned_seg35.nii.gz").unlink()
    else:
        prep.SUBJECT_MAP.write_text(
            prep.SUBJECT_MAP.read_text().replace("0001,", "0123,")
        )
    out = tmp_path / "prepared"
    with pytest.raises((ValueError, FileNotFoundError)):
        prep.prepare(source, out, execute=True)
    assert not out.exists()


def test_reject_output_inside_source_or_repository(source):
    for out in (source / "prepared", prep.ROOT / "prepared-test-must-not-exist"):
        with pytest.raises(ValueError):
            prep.prepare(source, out, execute=True)
        assert not out.exists()


def test_reject_dangling_output_symlink(source, tmp_path):
    out = tmp_path / "link"
    target = tmp_path / "nonexistent"
    out.symlink_to(target)
    with pytest.raises(FileExistsError):
        prep.prepare(source, out, execute=True)
    assert not target.exists()


def test_cli_help_is_lightweight():
    result = subprocess.run(
        [sys.executable, "-m", "rightpriordir", "prepare-oasis", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--check-data" in result.stdout and "--execute" in result.stdout
