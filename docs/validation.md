# Release acceptance

Acceptance intentionally does not rerun registration optimization. Frozen
results remain the source of the paper figures/tables.
Technical acceptance is separate from publication clearance: SINR author
permission is pending; see [the release hold](../THIRD_PARTY.md).

Verified during release preparation:

- All frozen input checksums and row counts; all 110 September 16 timing
  observations matched the released ordered OASIS pair IDs.
- OASIS100/first10 are nested, with the exact original sorted 200-pair list.
- All 84 test-pool subject IDs map by original Neurite `subjects.txt` position,
  not by the original subject-number digits. All 168 image/label source files
  matched historical organized inputs byte-for-byte. `prepare-oasis` pins that
  map and refuses changed inputs or existing destinations.
- COPD contains 10 cases for each of four selected rows; reported mean and
  sample SD match the per-case values.
- Figure 2 regenerated offline without raw data/GPU; PNG SHA-256 is
  `eb0db83bbd1f96d34c3afc9c25576e56c7944c07cc04ce74ece6f615d23e3e16`, identical
  to the supplied paper asset.
- All 20 real lung cases matched historical crop bounds, shapes and both sets
  of 300 registration landmarks; coordinate round trips passed.
- Both-phase image and mask arrays for 4DCT/COPD case01/02 matched historical
  arrays exactly (eight phase pairs).
- Private-to-public SIREN initialization, output and coordinate gradients, and
  cps2/cps4 spline expansion, gradients and analytic BE matched exactly.
- Three pinned upstream imports/model construction, 15 public configs, and the
  offline synthetic tests passed. Tests prohibit optimizer steps.
- GPU visibility and scalar CUDA arithmetic passed on RTX 6000 Ada, CUDA 11.8,
  torch 2.6.0+cu118, cuDNN 90100.

Not claimed: rerun paper optimizations, equal end-to-end registration outputs on
all GPUs, re-executed R231 inference, regenerated reference-method predictions,
new lung acceleration results, or corrected-COPD registration results. A
working environment and numerical kernel checks do not establish those claims.

The Docker runtime is built from a pinned historical base; this is an isolated
derived-image build, not a from-source rebuild of PyTorch/CUDA. Final acceptance
uses a clean source snapshot with no private mounts or Git metadata; committing,
tagging and publishing the repository/image are separate maintainer actions.

## Experiment reproduction layer

The experiment runner now covers the full released OASIS and 4DCT weight grids,
COPD's three selected configurations, OASIS100 base/fast accuracy and the separate
N=10 cold/warm timing protocol. Offline checks cover exact frozen-grid coverage,
three-dataset mocked dispatch, no-data/no-output dry runs, verified completion
hashes, source/config/data drift rejection, preserved retries, disjoint shards,
partial-report labels, per-source SD conventions and cold/warm observation order.
No real optimizer steps are used for these checks. End-to-end real optimization,
GPU timing equality, R231 inference and new corrected-COPD results remain
unverified rather than being implied by the additional executable code.
