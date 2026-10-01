# Two layers of reproduction

Both layers are shipped. They have different inputs, costs and guarantees.

| Layer | Input → output | GPU / optimization | Entry point |
| --- | --- | --- | --- |
| 1. Frozen artifacts | Bundled numerical results → paper Figure 2 and tables | Neither | `python -m rightpriordir frozen` |
| 2. Experiment reruns | Prepared datasets → registration → per-case metrics → new tables/curves | CUDA; optimization only with `--execute` | `python -m rightpriordir experiment`, then `summarize` |

Layer 2 is not the same as replotting old numbers. New outputs have their own
data/source/config hashes and never replace `results/` or `figures/`. Numerical
and timing equality across hardware is not guaranteed. **The real experiments
were not rerun during this release work**; execution orchestration was tested
with synthetic outputs and optimizer steps prohibited.

## Exact shipped experiment suites

| Suite (`--suite`) | Cohort | Matrix | Registration observations |
| --- | --- | --- | --- |
| `oasis-paper` | Seed-3 ranks 0–99 | 6 methods × 6 weights × 100 pairs | 3,600 |
| `4dct-paper` | Cases 1–10, corrected crop | IDIR 7 weights; SINR/D-BSCP 7 each; MR NCC/pTV 5 each; Dual PL1A 6 | 370 |
| `copd-extension` | Cases 1–10, legacy crop | Dual P0/P1 and MR-D-BSCP P2e, fixed selected settings | 30 |
| `oasis-accuracy` | Seed-3 ranks 0–99 | Five spline methods × base/fast × 100 | 1,000 |
| `oasis-timing` | Seed-3 ranks 0–9 | Five spline methods × base/fast × (1 cold + 10 warm) | 110 |

The explicit, editable specifications live in [experiments/](../experiments/oasis-paper.json):
[OASIS paper](../experiments/oasis-paper.json), [4DCT paper](../experiments/4dct-paper.json),
[COPD extension](../experiments/copd-extension.json), [OASIS accuracy](../experiments/oasis-accuracy.json),
and [OASIS timing](../experiments/oasis-timing.json). `--list-jobs` shows every resolved
configuration, weight and case. Paper grid values are tested against the frozen CSVs.

Scopes deliberately differ: COPD adds three selected configurations, not a new
unreported alpha sweep. External VoxelMorph/TransMorph/VFA/SITReg/Greedy/pTVReg
remain frozen references, not executable methods here. The 4DCT 2400-step legacy
context row at TV weight 0.022 is also frozen-only; it is not silently rerun with
2500 steps. Diagnostic experiments and lung acceleration are still excluded.

Own-method base configurations use 2500 steps. Dual PL1A uses the original
3000-step adaptive blur schedule; P0/P1 use 2500 steps. Dual seed is **42**, as in
the historical controlled-cohort records; the other released methods use seed 1.
OASIS accuracy/base paper configs retain their historical cuDNN true/false
override. The September 16 timing suite uses benchmark **false**, deterministic
**true**. Dual uses false/true as the explicit release backend; historical Dual
backend flags were not established for every frozen row.

## 1. Build, data and mounts

Follow [running.md](running.md) to clone recursively and build
`rightpriordir:release`, and [data.md](data.md) for acquisition/layout, R231 masks
and raw DIR-LAB → nominal 1 mm conversion. Raw scans and model weights are not
included. OASIS needs the specified prepared Neurite data, segmentations and
exact pair list; a different OASIS preparation is not equivalent.

On the **host**, set your own paths and start a container:

```bash
export OASIS_DATA=/absolute/path/to/oasis/split_v1
export DIRLAB_DATA=/absolute/path/to/DIR-Lab
export RELEASE_WORK=/absolute/path/to/rightpriordir-work
mkdir -p "$RELEASE_WORK"
docker run --rm -it --network none --gpus 'device=0' --shm-size=8g \
  --user "$(id -u):$(id -g)" \
  -v "$OASIS_DATA:/data/oasis:ro" \
  -v "$DIRLAB_DATA:/data/dirlab:ro" \
  -v "$RELEASE_WORK:/work:rw" \
  rightpriordir:release bash
```

Download R231 weights beforehand as explained in data.md. Inside this container
the selected physical GPU is **cuda:0**. If running only planning/aggregation,
omit `--gpus`. The commands below run **inside the container**.

If lung preparation has not yet been made, generate/supply masks first, then:

```bash
python -m rightpriordir prepare-lung --dataset 4dct --crop-mode corrected \
  --data-root /data/dirlab --mask-root /work/masks --out-root /work/prepared
python -m rightpriordir prepare-lung --dataset copd --crop-mode legacy \
  --data-root /data/dirlab --mask-root /work/masks --out-root /work/prepared
```

Do not point these at old prepared directories or relabel old metadata. The
resulting versioned roots are the ones used below. Preparation writes new
datasets but does not perform registration optimization.

## 2. Preview and preflight — no optimization

```bash
python -m rightpriordir experiment --suite oasis-paper \
  --data-root /data/oasis --out /work/reruns/oasis-paper
python -m rightpriordir experiment --suite 4dct-paper \
  --data-root /work/prepared/dirlab_4dct_corrected_crop_nominal1mm_v1 \
  --out /work/reruns/4dct-paper
python -m rightpriordir experiment --suite copd-extension \
  --data-root /work/prepared/dirlab_copd_legacy_crop_nominal1mm_v1 \
  --out /work/reruns/copd-extension
```

These commands only print the matrix. They do not require data to exist, create
outputs or import PyTorch. Add `--check-data` to verify required filenames,
split/crop metadata and compute content hashes; this reads data but still does
not optimize. The individual loader additionally checks shapes, labels and
landmarks during each actual run. Hashing a full cohort can take time.

For a small subset, use e.g. `--cases 0:2 --arms mrdbscp` (OASIS ranks 0 and 1)
or `--cases 1,2 --arms dual-p1` (COPD). Ranges are **start-inclusive,
stop-exclusive**. Give subsets separate output roots: they remain labelled
subsets and cannot be resumed later as a different full-cohort plan.

## 3. Rerun each dataset — explicitly opt in

The following commands **really optimize** when the user chooses to run them.
They were not executed during release preparation.

```bash
python -m rightpriordir experiment --suite oasis-paper \
  --data-root /data/oasis --out /work/reruns/oasis-paper \
  --device cuda:0 --execute

python -m rightpriordir experiment --suite 4dct-paper \
  --data-root /work/prepared/dirlab_4dct_corrected_crop_nominal1mm_v1 \
  --out /work/reruns/4dct-paper --device cuda:0 --execute

python -m rightpriordir experiment --suite copd-extension \
  --data-root /work/prepared/dirlab_copd_legacy_crop_nominal1mm_v1 \
  --out /work/reruns/copd-extension --device cuda:0 --execute
```

Every quality job uses a fresh subprocess, fixed configuration and per-case
evaluation. No casewise best-weight selection occurs. Expect substantial compute
and storage: OASIS has 6,881,280 voxels and a float32 three-component field is
about 82.6 MB; 3,600 fields alone are about **297 GB** (decimal), before extra
outputs. This is not a measured minimum GPU-memory requirement. Full trajectories
and runtime on your hardware remain to be validated.

## 4. OASIS accelerated quality versus paired timing

These are separate experiments and outputs:

```bash
python -m rightpriordir experiment --suite oasis-accuracy \
  --data-root /data/oasis --out /work/reruns/oasis-accuracy \
  --device cuda:0 --execute

python -m rightpriordir experiment --suite oasis-timing \
  --data-root /data/oasis --out /work/reruns/oasis-timing \
  --device cuda:0 --execute
```

Both use alpha=1000 and the selected base/fast schedules. Single-stage fast uses
10× LR, at most 2500 steps, minimum 375, patience 150, absolute delta 1e-4 and
**check interval 1**. MR fast uses maximum stages 400/800/1300, LR
0.03/0.01/0.003, minimum stages 40/80/130, patience 20/35/50, check interval 25,
relative total/data thresholds 1e-4/5e-5. They do not use September 24 CP-BE.

Timing starts **after both images/labels are on the GPU**, and ends after dense
field transfer to CPU and `.npy` serialization. No filesystem `fsync` is implied.
Setup, loading, field hashing and accuracy evaluation are outside that endpoint.
There is **no accuracy evaluation between timed observations**. Each candidate
gets a fresh persistent process with cold rank 0 followed by warm ranks 0–9;
rank 0 is intentionally run twice. The 11 observations cannot be split into
independent subprocesses without changing what "warm" means.

Use one idle, exclusive GPU for the whole timing suite; sharding is disabled.
GPU process snapshots are recorded but exclusivity is not automatically certified.
Aggregation rejects mixed GPU UUID/runtime/backend identities across candidates.
The portable harness preserves the endpoint, schedule and coarse instrumentation,
not exact historical wall times or incidental logging overhead. Quality-job
`process_wall_seconds_includes_loading_and_evaluation` is a different metric.
The timing summary contains no borrowed historical Dice values.

## 5. Resume, retries and optional multi-GPU quality sharding

Repeat the **same** command with `--execute --resume`. Only completions with
matching experiment/config identities and every output checksum are skipped.
Changing inputs, source, dependencies, cohort, weights or shard count requires
a fresh experiment directory.

An interrupted/failed job is retained. To retry it, add `--retry-failed` as well:

```bash
python -m rightpriordir experiment --suite copd-extension \
  --data-root /work/prepared/dirlab_copd_legacy_crop_nominal1mm_v1 \
  --out /work/reruns/copd-extension --device cuda:0 \
  --execute --resume --retry-failed
```

Old attempts are not overwritten. A corrupt previously completed output raises
an error rather than silently retraining. By default the runner stops on the
first failure; `--keep-going` attempts independent remaining jobs and still exits
nonzero if any failed. Per-job locks prevent duplicate concurrent writers.

Quality suites can use `--shard-count K --shard-index I`, with `I=0..K-1`.
All workers need the same container-side data/output paths, code and environment.
Launch shard 0 first, wait for `experiment.json`, then launch other workers with
`--resume` and the same `--shard-count`. Give each worker a separate GPU/container;
each such container still uses `--device cuda:0`. Job index modulo K assigns
disjoint jobs. Run every shard before making a complete summary. Do not use
sharding for the paired timing suite or simultaneously run another workload on
its GPU.

## 6. Aggregate, compare and plot new results

```bash
python -m rightpriordir summarize --run-root /work/reruns/oasis-paper \
  --out /work/reports/oasis-paper
python -m rightpriordir summarize --run-root /work/reruns/4dct-paper \
  --out /work/reports/4dct-paper
python -m rightpriordir summarize --run-root /work/reruns/copd-extension \
  --out /work/reports/copd-extension
python -m rightpriordir summarize --run-root /work/reruns/oasis-accuracy \
  --out /work/reports/oasis-accuracy
python -m rightpriordir summarize --run-root /work/reruns/oasis-timing \
  --out /work/reports/oasis-timing
```

Aggregation is CPU-only and never launches an optimizer. Outputs:

- `cases.csv`: every accepted new case, protocol, seed and backend.
- `population.csv`: per-method/weight means, SDs and finite-metric denominators.
- `summary.md` / `summary.json`: coverage, source identity and output hashes.
- `comparison_to_frozen.csv`: descriptive mean differences for paper/COPD suites.
- `rerun_arc.png/pdf`: **new-result** accuracy–regularity curves for quality suites,
  without inserting unrerun external references or claiming to be frozen Figure 2.
- `paired_timing.csv`: base/fast warm mean ratios for the timing suite.

OASIS and COPD use sample SD (ddof=1). 4DCT own-method rows use population SD
(ddof=0), while Dual PL1A uses its source's sample SD (ddof=1). Both conventions
and actual denominators are explicit per row. Missing metrics stay empty, never
zero. Native snapped TRE is the lung headline; other TRE domains remain separate.

Missing jobs block aggregation by default. `--allow-partial` creates an explicitly
partial report with missing job IDs and observed/expected counts; it is not a
full-cohort paper result. `--no-plots` skips image generation. Existing report
directories are never overwritten; use a new directory for each report snapshot.

For independent reevaluation of any saved field, use the `evaluate` command in
[running.md](running.md). This also works on a timing observation's `field.npy`,
but do it after the timing candidate has finished. Never replace the shipped
frozen values with newly aggregated results without a separate scientific review.
