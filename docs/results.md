# Frozen results and provenance

This release separates three records. Rendering them never launches registration.

For the distinct data-to-experiment reproduction layer, use the full cohort
suites in [experiments.md](experiments.md). New runs and summaries stay outside
these frozen inputs; their existence is not a claim that release preparation
reexecuted the experiments.

| Record | Cohort | Interpretation |
| --- | --- | --- |
| `results/paper/*_arc.csv` | OASIS100 / DIR-LAB 4DCT10 | Frozen inputs for paper Figure 2; not replaced by accelerated results |
| `results/updated_oasis/timing_first10.csv` | OASIS seed-3 ranks 0–9 | September 16, 2026 paired base/fast timing; the updated main table |
| `results/paper/oasis100_accuracy.csv` | OASIS seed-3 ranks 0–99 | Historical accuracy, shown separately; **not** accuracy measured in the September 16 timing experiment |
| `results/paper/table1_published.csv` | Published Table 1 | Hand-transcribed rounded camera-ready values, retained as a historical record |
| `results/copd_extension/*` | COPD cases 1–10 | Post-paper extension; existing P0/P1, MR-D-BSCP P2e and pTVReg reference results |

`results/manifest.json` records source identifiers, original input hashes, released
input hashes, row counts and column names. Source identifiers identify historical
experiments, not paths required at runtime. No scans, model weights or deformation
fields are distributed. Figure 1 is a static overview asset. Figure 3 and diagnostic
experiment reproduction are outside this release.

## Timing update (not the published Table 1)

Each of five methods has one base and one fast row, with one cold observation and
10 warm observations. The endpoint is **already loaded images on the selected
device → serialized dense displacement field**. It excludes data loading, initial
device setup and accuracy evaluation. Cold does not mean full-process startup.
Warm standard deviation is a descriptive dispersion, not a confidence interval.
The fast schedules use stopping criteria; their realized iteration counts vary.

The recorded environment is PyTorch 2.6.0+cu118, CUDA runtime 11.8, cuDNN 90100,
NVIDIA RTX 6000 Ada Generation, cuDNN benchmark **false**, deterministic **true**.
Do not combine old accuracy with these timing rows as a newly measured joint
accuracy/speed claim. The September 24 CP-BE/cache/batch experiment is not selected.
Lung accelerated results remain pending.

## Figure 2 and metric conventions

OASIS foreground is the fixed raw image > 0, not segmentation > 0. Dice is averaged
over labels 1–35, then over pairs. The horizontal coordinate is foreground
SD(log J); the CSV preserves the historical metric definition and aggregation.
Ellipses use the exported between-case standard deviations on each axis, not
covariance ellipses, standard errors or confidence regions. The lower-left panel
has no ellipses; the lower-right selected-method panel does.

The lung vertical coordinate is **native-grid snapped TRE (mm)**, not the 1 mm-grid
or subvoxel TRE. Those other columns are retained with explicit names. The field
is a fixed-to-moving pull displacement even when registration is described as
"moving → fixed". Comparisons have different loss domains: fixed-lung sampling
for IDIR, full-crop NCC for spline NCC methods, border-crop LCC for pTV-style loss.
These are protocol-tagged comparisons, not uniformly matched method-only tests.

The base release uses the historical 2500-step configurations. An important
exception is the lung Dual-INR PL1A reference in Figure 2: its adapter specifies
the author's adaptive blur schedule with **3000 steps**. We preserve this
reference instead of relabelling it as a 2500-step run. External DL/Greedy/pTVReg
points are frozen reference evaluations, not rerunnable implementations here.

## COPD and crop provenance

COPD P0/P1 use Dual-INR with the controlled 2500-step schedule; P0 stays at full
resolution and P1 uses image factors 8/4/2/1. MR-D-BSCP P2e uses control spacing
64/32/16/8/4, image factors 16/8/4/2/1, steps 400/400/400/400/900 and TV weight 0.11.
pTVReg has TRE only where field/regularity information was not available. Missing
metrics must remain missing, not be replaced by zero.

Historical 4DCT preparation uses the corrected MATLAB-origin crop. Historical
COPD uses legacy Python crop bounds. The latter are usually shifted one native
voxel and are not geometrically interchangeable. A corrected COPD preparation is
a separate protocol and does not inherit these frozen results. See `docs/data.md`.
