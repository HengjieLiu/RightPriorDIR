# Third-party code and reference results

The root [MIT license](LICENSE) applies to our own code. It does not replace
dataset terms, model-weight terms, container licenses, or upstream notices.

| Submodule | Fixed commit | Wrapper role | Upstream notice |
| --- | --- | --- | --- |
| [IDIR](https://github.com/MIAGroupUT/IDIR) | `9ab3ccfbbf09dc8fc65fa848332e7d9a0b3b81b3` | Official SIREN; local study protocol/loss/evaluation | [MIT](third_party/IDIR/LICENSE) |
| [SINR](https://github.com/vasl12/SINR) | `1a524ca7ae453b55310595fe957245088a108233` | Official SIREN control generator; local cubic-spline decoder and study protocol | Author permission pending; no root LICENSE at this pinned revision |
| [Dual-INR](https://github.com/IPMI-ICNS-UKE/DUAL-INR-DIR) | `f7c8e833435ec9c6b7ebec6c9a8098b566709146` | Released model/loss/interpolation with explicit PL1A/P0/P1 adapters | [MIT](third_party/DUAL-INR-DIR/LICENSE) |

## Pending release clearance

SINR author permission is **pending**, as confirmed by the maintainer on
September 30, 2026. The earlier statement that permission was confirmed is
superseded. The project MIT license does not grant rights to SINR.

This candidate still includes the pinned SINR submodule and wrappers. Public
release of this candidate, including any image bundling SINR, is on hold until
permission is resolved or a separately reviewed SINR-free scope is approved.
Recording "pending" is not a grant of permission. Technical tests may continue
locally; do not describe their success as publication clearance.

Upstream sources are unmodified git submodules, not relabelled copies. The
content hashes in [third_party/lock.json](third_party/lock.json) allow checking
the exact sources in a Docker image where Git metadata is omitted. Wrappers
are study adaptations, not claims that every upstream default or training
pipeline is reproduced. Do not install each upstream `requirements.txt` on top
of this environment: the wrapper's audited import surface is narrower.

Our implementations also use PyTorch, NumPy, SciPy, nibabel, surface-distance,
Matplotlib, pandas, SimpleITK, MONAI, and lungmask. Their existing notices remain
in the installed distributions. The historical base image includes NVIDIA
components under their own terms; MIT does not relicense the Docker runtime.

VoxelMorph, TransMorph, VFA, SITReg, Greedy, and pTVReg appear as frozen reference
evaluations in Figure 2. Their plotting values and original aggregate source
hashes are distributed in `results/`; their training/inference environments,
weights and raw predictions are not. Cite the original methods as well as this
study; see the [paper references](https://arxiv.org/abs/2608.16146) and the
per-method mapping in [docs/methods.md](docs/methods.md).
