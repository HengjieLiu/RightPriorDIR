# Methods, wrappers and metric definitions

This package contains study-specific registration implementations, not the
complete original application of each third-party baseline. The source
extraction manifest records the research-kernel origins. Official repositories
remain intact as pinned submodules; see [THIRD_PARTY.md](../THIRD_PARTY.md).

| Figure name | Implementation | Released configuration |
| --- | --- | --- |
| INR-Dense / IDIR | Upstream IDIR SIREN, local sampled NCC and coordinate-autograd BE | `configs/{oasis,4dct}/idir.json` |
| INR-BSCP / SINR | Upstream SINR SIREN predicts cubic B-spline controls; local study loss/decoder | `configs/oasis/sinr_cps{2,4}.json`, `configs/4dct/sinr_cps4.json` |
| D-BSCP | Directly optimized spline controls | `configs/oasis/dbscp_cps{2,4}.json`, `configs/4dct/dbscp_cps4.json` |
| MR-D-BSCP | Anti-aliased image pyramid, exact cubic dyadic knot insertion between control grids | `configs/oasis/mrdbscp.json`, `configs/4dct/mrdbscp_{ncc,ptv}.json` |
| MR-INR-Dense / Dual-INR PL1A | Official two-branch model and losses; author adaptive blur, fixed-lung sampling | `configs/4dct/dual_pl1a.json` |
| COPD Dual-INR P0/P1 | Official model/losses, stage-local AdamW and warmup-cosine; full resolution / pyramid | `configs/copd/dual_p{0,1}.json` |
| COPD MR-D-BSCP P2e | Five-stage spline extension, legacy crop | `configs/copd/mrdbscp_p2e.json` |

## Core protocol

OASIS IDIR/SINR networks have three 256-unit hidden layers; omega is 32 for IDIR,
50 for SINR. IDIR samples up to 10,000 fixed foreground coordinates per step.
Spline image loss uses the whole image. Adam uses LR 1e-4 for INR methods and
1e-3 for direct controls. OASIS spline regularization uses analytic cubic
B-spline second derivatives; IDIR uses coordinate autograd. The historical
IDIR BE batch normalization is retained, so alpha scales are not comparable
between parameterizations.

OASIS MR-D-BSCP uses control spacings 8/4/2, image factors 4/2/1, steps
400/800/1300. The 4DCT MR configurations use spacings 32/16/8/4, factors
8/4/2/1 and steps 400/400/800/900. COPD P2e uses spacings 64/32/16/8/4,
factors 16/8/4/2/1 and steps 400/400/400/400/900. The 4DCT NCC spline
configurations use dense finite-difference BE, whereas the pTV branch uses
local correlation and physical-unit isotropic TV with a five-voxel border mask.

The Dual PL1A row is an explicit **3000-step author-blur exception**; P0/P1 use
2500 steps. The latter disable semantic mask loss and use the crop's border-five
interior for sampling, keeping lung masks for evaluation only. PL1A samples the
fixed lung mask. No TotalSegmentator/lobe pipeline is needed for these variants.

Bundled single-pair configs are representative curve points. Full matrices and
batch runners are now supplied in `experiments/` and the [experiment guide](experiments.md). The
OASIS spline alpha is 1000; IDIR is 1. Lung examples specify their weights in
JSON. `register --alpha` overrides the applicable BE/TV weight (or Dual-INR
regularization multiplier). All original Figure 2 sweep weights remain in the
frozen CSV rows. The updated OASIS fast schedule is documented in
`results/updated_oasis/protocol.json`; `experiment --suite oasis-timing` supplies
the paired persistent-process timing harness. `oasis-accuracy` separately runs
base/fast quality on all 100 pairs. Dual historical seeds are 42, not the seed 1
used by the other released methods.

Backend flags are explicit. The historical own-method configs use benchmark
true / deterministic false. The paired timing update uses false / true. Dual
configs use false / true as an explicit release setting; historical backend
flags were not established for every frozen Dual row. No full optimization
was rerun to assert bitwise registration equivalence across these environments.

## Metrics

OASIS labels 1–35 are evaluated only when present in original moving, fixed and
warped segmentations; missing-label metrics are NaN and excluded from the case
mean. Surface metrics use `surface-distance`, unit voxel spacing and area-weighted
surfaces. Foreground is fixed raw image > 0, not segmentation foreground.

OASIS SDlogJ uses positive Jacobian determinants only and a cropped valid
derivative domain; folding percentage is reported separately. Between-pair
OASIS plotting SD uses the historical sample convention (ddof=1). Lung frozen
Figure 2 columns retain their source aggregations (own-method figure aggregation
uses population SD, ddof=0; Dual PL1A source rows use sample SD, ddof=1); COPD extension and updated timing tables
explicitly use sample SD, ddof=1. Re-rendering preserves these exported values
instead of silently standardizing denominators. None of the ellipses are CIs.

Lung headline TRE is native-grid snapped TRE; the subvoxel and nominal 1 mm
variants remain separately named. See [data.md](data.md) for field direction,
native rescaling, cropping and endpoint conventions. pTVReg reference rows have
their own protocol tags and must not be presented as matched preprocessing
comparisons with COPD legacy-crop rows.

## What was verified

Release acceptance prohibits `Adam.step` and `AdamW.step`. Synthetic tests cover
upstream pins, SIREN values/derivatives, spline knot insertion, crop boundary
cases, coordinate round trips, TRE translation, config parsing, table cohort
separation and preprocessing no-overwrite behavior. A private migration audit
also compared extracted kernels against their research sources, all 20 real
lung cases' landmark geometry, and both-phase image/mask arrays for cases 1/2
in each lung dataset. R231 inference and complete registration trajectories
were not rerun. Both reproduction layers are implemented; frozen artifact
equivalence is verified, whereas real experiment trajectories remain unexecuted.
