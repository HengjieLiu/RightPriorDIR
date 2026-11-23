# Learn2Breath 2026 · MR-D-BSCP

Our MR-D-BSCP submission placed second in the Learn2Breath task at Learn2Reg 2026. It fits each inspiratory/expiratory CT pair independently on one GPU, using only the two scans. [Challenge page](https://www.codabench.org/competitions/15955/) · [Test-phase report](https://github.com/MFaizyabAli/Learn2Reg26_L2B/tree/main/Learn2Breath_TestPhase_Results) · [Run one pair](#run-one-pair)

## 1. MR-D-BSCP framework and registration objective

MR-D-BSCP directly optimizes the control-point displacements of a cubic B-spline. Given a fixed inspiration scan (INSP) and a moving expiration scan (EXP), we start with zero displacement and sample EXP at `p + u(p)` for each fixed voxel `p`. We clip both scans to [−920, −100] HU and scale to [0, 1].

At each pyramid level we minimize **`LCC loss + 0.11 × isoTV`**, a pTVreg-style objective with two terms.

- **Similarity.** We compute Gaussian-windowed local cross-correlation (LCC) between INSP and warped EXP. We sum `1 − LCC` over the image interior with voxel-volume weighting. A five-voxel border on the full grid defines the support, and we cache the fixed image's local mean and standard deviation at each level.
- **Regularization.** Isotropic total variation (isoTV) penalizes spatial changes in the spline control displacements. We convert controls to millimetres, take finite differences along each lattice axis using physical knot spacing, and sum the smoothed gradient magnitudes with knot-cell-volume weighting. [Spline and loss implementation](learn2breath/spline.py)

The challenge uses **five levels**. Relative to the paper's four-level 4DCT configuration, the extra coarsest level starts at 64-voxel control spacing. We blur before downsampling, fit the controls with Adam, then use exact cubic knot insertion to carry the fitted deformation to the next level. [Paper configuration](../../configs/4dct/mrdbscp_ptv.json) · [Challenge schedule](learn2breath/schedule.py)

| Coarse → fine | 1 | 2 | 3 | 4 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Image downsampling | 16× | 8× | 4× | 2× | 1× |
| Control spacing (full-grid voxels) | 64 | 32 | 16 | 8 | 4 |
| Adam learning rate | 0.03 | 0.03 | 0.01 | 0.01 | 0.003 |

## 2. Learn2Breath adaptations

**Early stopping for the time budget.** The challenge allowed five hours for 200 pairs on one GPU. Once a level reaches its minimum update count, we check progress every 25 updates. For both total loss and LCC loss, we compare the median of the latest 25 updates with the median of a 25-update window ending one lookback earlier. Relative improvement is `(previous − recent) / max(|previous|, 1e−8)`. A level ends when **both** improvements fall below their thresholds (5 × 10⁻⁵ for total loss, 2 × 10⁻⁵ for LCC), or when it reaches its cap.

| Coarse → fine | 1 | 2 | 3 | 4 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Minimum updates | 120 | 90 | 60 | 60 | 60 |
| Lookback (updates) | 50 | 38 | 25 | 25 | 25 |
| Update cap | 800 | 600 | 400 | 400 | 400 |

**NDV refinement.** After level 5, we retain the fitted controls and reset Adam. On the same full-resolution grid we take **exactly 100 additional updates** at learning rate 0.003 with **`LCC loss + 0.11 × isoTV + 10,000 × NDV`**. The non-diffeomorphic volume (NDV) term penalizes folding. It sums `max(−det, 0) / 12` for ten oriented tetrahedral determinants per interior voxel across the full unmasked interior. [Registration loop](learn2breath/register.py)

**NDV acceleration.** Deferring NDV optimization to those final 100 updates limits its cost. The GPU implementation takes finite differences with tensor slices, adds the identity derivatives analytically, and evaluates the ten required determinants with explicit 3×3 arithmetic. On the full challenge grid, the original RTX 6000 Ada benchmark measured **1.84× faster NDV forward/backward** than the convolution reference, with peak allocation falling from **10.76 to 3.09 GiB**. [How the accelerated NDV differs from SITReg](NDV.md) · [NDV implementation](learn2breath/ndv.py)

### Run one pair

From the repository root, set the input directory to the absolute path containing the two NIfTI files. Both examples require Linux x86-64 and an NVIDIA GPU with a CUDA 11.8-compatible driver. The original submission recommended at least 8 GiB of GPU memory.

```bash
cd challenges/learn2breath-2026
CT_DIR=/absolute/path/to/ct_data
mkdir -p outputs
```

**Docker.** With the NVIDIA Container Toolkit installed, build the image and run it offline. The image uses the submitted PyTorch 2.6.0 / CUDA 11.8 / cuDNN 9 base. [Dockerfile](Dockerfile)

```bash
docker build -t rightpriordir/learn2breath:2026 .
docker run --rm --gpus device=0 --network none --user "$(id -u):$(id -g)" \
  -v "$CT_DIR:/input:ro" -v "$PWD/outputs:/output" \
  rightpriordir/learn2breath:2026 \
  /input/NLST_0001_INSP.nii.gz /input/NLST_0001_EXP.nii.gz \
  /output/disp_NLST_0001.nii.gz
```

**Python.** In a fresh Python 3.11 environment, install the CUDA build of PyTorch and the hash-locked wheels. The lock includes NumPy 1.26.4 and NiBabel 5.3.2. [Dependency lock](requirements.lock)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cu118 'torch==2.6.0+cu118'
python -m pip install --only-binary=:all: --no-deps --require-hashes -r requirements.lock
CUDA_VISIBLE_DEVICES=0 python -m learn2breath.register \
  "$CT_DIR/NLST_0001_INSP.nii.gz" "$CT_DIR/NLST_0001_EXP.nii.gz" \
  outputs/disp_NLST_0001_python.nii.gz
```

The three arguments are the fixed INSP image, moving EXP image, and a new output filename in an existing writable directory. Inputs must be finite float32 NIfTI-1 CTs in physical HU with identity scaling, shape `(224, 192, 224)`, 1.5 mm isotropic spacing, and matching native geometry (affine, orientation, and active qform/sform). The output is a float32 displacement field of shape `[3, X, Y, Z]` on the fixed INSP grid. Its channels are x, y, z offsets in voxels and it preserves the fixed affine and forms.

The source is the `Learn2Breath26` submission runtime at revision `491143b10fd89cee9ef2a3aedcfaa4b5d73d78b4`. The commands use the submitted defaults. CUDA interpolation gradients can vary across runs, which can change a stopping point or the resulting field slightly.
