# Running the release

This guide covers environment, data mounts, frozen artifacts and one-pair runs.
These are local candidate-verification instructions; public publication/image
distribution remains on hold while [SINR permission is pending](../THIRD_PARTY.md).
For full three-dataset experiments, sweeps, recovery, aggregation and the paired
OASIS timing benchmark, see [the two-layer experiment guide](experiments.md).

## 1. Clone and build

Linux x86-64 and Docker are the tested platform. Frozen rendering is CPU-only.
Registration additionally needs an NVIDIA GPU, a compatible host driver and
NVIDIA Container Toolkit; a Dockerfile cannot install the host driver. The
tested device is an RTX 6000 Ada (48 GB). Full evaluation retains several dense
arrays, so table training-peak memory is not the total memory requirement.
Budget disk space for the approximately 14 GB historical base plus build layers,
model weights, prepared volumes and outputs. Data preparation can produce several
GB per cohort; keep it outside Git.

```bash
git clone --recurse-submodules https://github.com/HengjieLiu/RightPriorDIR.git
cd RightPriorDIR
# For an existing checkout:
git submodule update --init --recursive
git submodule status
docker build -t rightpriordir:release .
docker run --rm --network none rightpriordir:release python -m rightpriordir verify
docker run --rm --network none --gpus all rightpriordir:release \
  python -m rightpriordir verify --gpu
```

The Dockerfile pins the historical base by digest and the added dependency
versions in `requirements.txt`. Tested runtime: Python 3.10, torch 2.6.0+cu118,
CUDA 11.8, cuDNN 90100. Unused inherited packages (`itk` metapackage,
`PyGObject`, and the `itk` consumer `icon_registration`) are removed from the
derived image to resolve inherited metadata/dependency problems; `pip check`
must pass. This does not modify your
host Python, research container or the base image. No pip installs occur at run
time. Do not install the three upstream dependency files wholesale.

The image includes source and submodules, but excludes Git metadata, scans,
prepared volumes and weights. Content hashes still verify upstream pins. The
three wrappers do not need a separately installed IDIR/SINR/inrdir package.

## 2. Frozen reproduction — no optimization

From the checkout:

```bash
mkdir -p outputs
docker run --rm --network none --user "$(id -u):$(id -g)" \
  -v "$PWD/outputs:/outputs" rightpriordir:release \
  python -m rightpriordir frozen --out /outputs/paper
docker run --rm --network none rightpriordir:release \
  python -m pytest -q -p no:cacheprovider tests
```

The output directory contains:

```text
figure2.png / figure2.pdf
updated_oasis_timing.md
historical_oasis100_accuracy.md
table1_published.md
copd_extension.md
render_manifest.json
```

Use `--tables-only` to skip plotting. Existing target files are not overwritten;
select a fresh output directory when repeating a command. The PNG is the exact
paper rendering in the tested environment. PDF creation metadata may differ
between invocations even when the plotted content is unchanged.

## 3. Mount data for preparation or your own registration

Choose your own absolute paths; none of these are assumed by the source:

```bash
export OASIS_DATA=/absolute/path/to/oasis/split_v1
export DIRLAB_DATA=/absolute/path/to/DIR-Lab
export RELEASE_WORK=/absolute/path/to/rightpriordir-work
mkdir -p "$RELEASE_WORK"
docker run --rm -it --gpus all --shm-size=8g \
  --user "$(id -u):$(id -g)" \
  -v "$OASIS_DATA:/data/oasis:ro" \
  -v "$DIRLAB_DATA:/data/dirlab:ro" \
  -v "$RELEASE_WORK:/work:rw" \
  rightpriordir:release bash
```

For CPU-only preprocessing omit `--gpus all`; use `--device cpu` for mask
inference. For frozen reproduction none of these data mounts is needed. For
registration on one physical GPU use `--gpus 'device=2'` on `docker run`, then
select **`cuda:0` inside that container**. Do not use a physical host GPU index
as the container-local index.

Follow [data.md](data.md) to run `prepare-oasis` with its pinned ordinal-ID map,
generate R231 masks,
and convert raw DIR-LAB volumes. Copy the bundled `splits/pair_test_200.txt` into
the OASIS root before mounting it read-only only if you did not use
`prepare-oasis`, which writes that list automatically. Raw data mounts are
read-only throughout.

## 4. Inspect a single-pair run

All commands below run inside the container. **Without `--execute`, they only
print resolved configuration; they do not load data, create run files or train.**

```bash
python -m rightpriordir register --config configs/oasis/mrdbscp.json \
  --data-root /data/oasis --case 0 --out /work/runs/oasis-rank000 --device cuda:0

python -m rightpriordir register --config configs/4dct/mrdbscp_ptv.json \
  --data-root /work/prepared/dirlab_4dct_corrected_crop_nominal1mm_v1 \
  --case 1 --out /work/runs/4dct-case01 --device cuda:0

python -m rightpriordir register --config configs/copd/dual_p1.json \
  --data-root /work/prepared/dirlab_copd_legacy_crop_nominal1mm_v1 \
  --case 1 --out /work/runs/copd-dual-p1-case01 --device cuda:0
```

`--case` is a zero-based rank (0–99) for OASIS and a case ID (1–10) for lung.
Change the config to select IDIR, SINR, D-BSCP, MR-D-BSCP or the documented
Dual-INR variant. Alpha/TV/multiplier overrides use `--alpha`; see [methods.md](methods.md).
Copy a config if you deliberately change backend flags, and keep the resulting
protocol separate from published results.

## 5. Explicitly run registration (optional; not needed for paper plots)

Review the previous output, then add `--execute`:

```bash
python -m rightpriordir register --config configs/oasis/mrdbscp.json \
  --data-root /data/oasis --case 0 --out /work/runs/oasis-rank000 \
  --device cuda:0 --execute
```

Outputs are `run_config.json`, `field.npy`, `field.json`, `metrics.json`, plus
method-dependent histories and lung per-landmark metrics. They belong outside
Git. The output directory must not exist, preventing accidental overwrite.
Each field has a checksum and an explicit direction/unit/axis contract. The
registration command evaluates its final field, so wall time is **not** the
frozen table's loaded-images-to-saved-DVF timing endpoint.

These optional training commands were inspected but **not executed for this
release**. Acceptance checked kernels/gradients/geometry and command dispatch;
it did not certify a fresh full-cohort optimization. Do not use these commands
to silently replace the bundled historical values.

## 6. Evaluate an existing public-format field

```bash
python -m rightpriordir evaluate --field /work/runs/oasis-rank000/field.npy \
  --data-root /data/oasis --out /work/evaluations/oasis-rank000 --device cuda:0
```

For lung use the same versioned preparation root as at registration. Evaluation
also works on CPU but may need substantial RAM. `field.json` must be beside the
array. A different crop protocol, changed preparation metadata, wrong pair,
nonfinite values or checksum mismatch is rejected. An arbitrary upstream field
without this coordinate metadata is not accepted automatically.

## Troubleshooting and verification limits

- Missing submodule / checksum mismatch: use the pinned gitlinks; do not update
  upstream HEAD to resolve it.
- Wrong OASIS shape or labels: check prepared Neurite v1.0 data, 0–35 labels and
  orientation; do not resize automatically just to satisfy the shape check.
- Lung protocol mismatch: prepare the requested explicit crop version. Never
  edit an old `preprocess_audit.json` merely to pass the check.
- CUDA unavailable: run the GPU verification command and fix host/container
  GPU access. Plotting and geometry tests do not require CUDA.
- Out of memory: avoid concurrent jobs; reducing evaluation chunk size in a
  copied config helps INR decoding, but dense spline operations still require
  the full field. Changing sample counts, grids or crops changes the experiment.
- Offline R231: supply the local checkpoint with `--weights`. Model weights are
  not shipped, and R231 inference was not rerun in release acceptance; existing
  masks were used for exact preprocessing comparisons.

See [validation.md](validation.md) for the precise release checks. No diagnostic
experiment runner, external-reference environment, or lung acceleration sweep
is part of this release.
