"""Organize the pinned Neurite OASIS test subjects without changing voxels."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
SUBJECT_MAP = ROOT / "splits/oasis_subject_map.csv"
PAIR_FILE = ROOT / "splits/pair_test_200.txt"
SUBJECTS_SHA256 = "52a8a9cae5992837140464f8d28a69135b096484786f3cb85c50834958eee4ca"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_plan(source, out):
    if os.path.lexists(out):
        raise FileExistsError(f"Select a new output directory: {out}")
    source, out = Path(source).resolve(), Path(out).resolve()
    if out.is_relative_to(source) or out.is_relative_to(ROOT):
        raise ValueError("Keep prepared data outside the source archive and repository")
    if sha256(source / "subjects.txt") != SUBJECTS_SHA256:
        raise ValueError("subjects.txt differs from the pinned Neurite v1.0 order")
    subjects = (source / "subjects.txt").read_text().splitlines()
    with SUBJECT_MAP.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    ids = {
        value
        for pair in re.findall(r"\((\d{4}),\s*(\d{4})\)", PAIR_FILE.read_text())
        for value in pair
    }
    if not ids or len(rows) != len(ids) or {r["release_id"] for r in rows} != ids:
        raise ValueError("Subject map does not exactly cover the released test pool")
    files = []
    for row in rows:
        sid, subject = row["release_id"], row["neurite_subject"]
        if not re.fullmatch(r"OASIS_OAS1_\d{4}_MR1", subject):
            raise ValueError("Invalid Neurite subject directory")
        if (
            int(sid) < 1
            or int(sid) > len(subjects)
            or subjects[int(sid) - 1] != subject
        ):
            raise ValueError(f"Release ID {sid} is a one-based subjects.txt index")
        for original, prefix, hash_key in [
            ("aligned_norm.nii.gz", "img", "image_sha256"),
            ("aligned_seg35.nii.gz", "seg", "seg_sha256"),
        ]:
            relative = f"{subject}/{original}"
            path = source / relative
            if not path.is_file() or not path.resolve().is_relative_to(source):
                raise FileNotFoundError(
                    f"Missing/inaccessible archive input: {relative}"
                )
            if not re.fullmatch(r"[0-9a-f]{64}", row[hash_key]):
                raise ValueError("Invalid source checksum in subject map")
            files.append(
                {
                    "source": relative,
                    "target": f"test/{prefix}{sid}.nii.gz",
                    "sha256": row[hash_key],
                }
            )
    return {
        "schema_version": 1,
        "protocol": "neurite_oasis_v1_subject_order_test_pool",
        "subject_count": len(rows),
        "subjects_txt_sha256": SUBJECTS_SHA256,
        "subject_map_sha256": sha256(SUBJECT_MAP),
        "pair_file_sha256": sha256(PAIR_FILE),
        "voxel_transforms": "none; byte-preserving copies; orientation handled by registration loader",
        "files": files,
    }


def check_sources(source, plan):
    for record in plan["files"]:
        if sha256(Path(source) / record["source"]) != record["sha256"]:
            raise ValueError(f"Source checksum mismatch: {record['source']}")


def prepare(source, out, *, execute=False, check_data=False):
    plan = build_plan(source, out)
    source, out = Path(source).resolve(), Path(out).resolve()
    if check_data or execute:
        check_sources(source, plan)
    if execute:
        # Preflight completes before creating any output; partial failures are
        # preserved and must be retried using a fresh output root.
        out.mkdir(parents=True, exist_ok=False)
        (out / "test").mkdir()
        for record in plan["files"]:
            target = out / record["target"]
            shutil.copyfile(source / record["source"], target)
            if sha256(target) != record["sha256"]:
                raise ValueError(f"Copy/source changed: {record['source']}")
        shutil.copyfile(PAIR_FILE, out / "pair_test_200.txt")
        if sha256(out / "pair_test_200.txt") != plan["pair_file_sha256"]:
            raise ValueError("Pair file changed during preparation")
        (out / "preparation.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Extracted neurite-oasis.v1.0 directory",
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="New data directory, outside Git"
    )
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="Check all source hashes without copying",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Copy the test pool; never optimize or resample",
    )
    args = parser.parse_args(argv)
    plan = prepare(
        args.source, args.out, execute=args.execute, check_data=args.check_data
    )
    print(
        json.dumps(
            {
                "executed": args.execute,
                "source_hashes_checked": args.check_data or args.execute,
                "out": str(args.out),
                **plan,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
