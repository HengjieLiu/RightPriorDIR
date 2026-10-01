# Data setup and coordinate contracts

Datasets and model weights are not bundled. Obtain the data under their original
terms. Frozen figure/table generation needs neither data nor weights.

## OASIS

The experiments used the **Neurite OASIS v1.0 prepared images and 35-label
segmentations**, organized as below. Original OASIS acquisitions, affine-aligned
alternatives, or an unrelated Learn2Reg preparation are not interchangeable.
Use the [Neurite OASIS v1.0 publisher page](https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md),
which provides the **3D v1.0 archive** and its checksum. Follow the
[OASIS data terms](https://www.oasis-brains.org/) and publisher's citation
instructions. Do not substitute the 2D or mesh archives, or redistribute scans
in this repository.

### Download and organize

On the **host**, choose new directories outside Git and download/extract the
archive (approximately 6.6 GB compressed; allow additional extraction space):

```bash
export OASIS_DOWNLOAD=/absolute/path/to/oasis-download
export OASIS_PREPARED=/absolute/path/to/oasis-prepared
mkdir -p "$OASIS_DOWNLOAD" "$OASIS_PREPARED"
curl -fL --output "$OASIS_DOWNLOAD/neurite-oasis.v1.0.tar" \
  https://surfer.nmr.mgh.harvard.edu/ftp/data/neurite/data/neurite-oasis.v1.0.tar
printf '081392a8150ff99ab7a64a9ded377835  %s\n' \
  "$OASIS_DOWNLOAD/neurite-oasis.v1.0.tar" | md5sum --check -
# Only extract after the publisher checksum reports OK; use a fresh destination.
mkdir "$OASIS_DOWNLOAD/extracted"
mkdir "$OASIS_DOWNLOAD/extracted/neurite-oasis.v1.0"
tar -xf "$OASIS_DOWNLOAD/neurite-oasis.v1.0.tar" \
  -C "$OASIS_DOWNLOAD/extracted/neurite-oasis.v1.0"
export NEURITE_SOURCE="$OASIS_DOWNLOAD/extracted/neurite-oasis.v1.0"
```

The archive stores subject folders at its top level; the commands explicitly
create the containing directory. `NEURITE_SOURCE` must contain `subjects.txt`
and subject folders.
MD5 here is the publisher's download-integrity check, not a security signature.
After the [Docker build](running.md), inspect and hash-check all required inputs:

```bash
docker run --rm --network none --user "$(id -u):$(id -g)" \
  -v "$NEURITE_SOURCE:/source:ro" -v "$OASIS_PREPARED:/work" \
  rightpriordir:release python -m rightpriordir prepare-oasis \
  --source /source --out /work/split_v1 --check-data
```

Repeat with `--execute` added to copy the files. This does **not** register,
resample, normalize, reorient or relabel anything. It creates the exact sorted
pair list, `test/imgXXXX.nii.gz`, `test/segXXXX.nii.gz` and `preparation.json`.
Existing output roots are rejected; a failed partial copy is retained, so retry
with a new root. Set `OASIS_DATA="$OASIS_PREPARED/split_v1"` for subsequent runs.
The command prepares 84 unique subjects (168 volumes) covering the full released
200-pair pool, including all paper/timing pairs; no training split is needed.

### Exact ID mapping — not the original OASIS subject number

`XXXX` is the **one-based position in the archive's original `subjects.txt`**.
Do not derive it from the digits in a subject directory or a new directory sort.
The explicit mapping and original file SHA-256 values are in
[oasis_subject_map.csv](../splits/oasis_subject_map.csv). For each selected row:

| Source within the mapped subject directory | Released filename |
| --- | --- |
| `aligned_norm.nii.gz` | `test/imgXXXX.nii.gz` |
| `aligned_seg35.nii.gz` | `test/segXXXX.nii.gz` |

For example, release ID **0333 → OASIS_OAS1_0370_MR1**, and
**0115 → OASIS_OAS1_0123_MR1**. These remain the fixed/moving release IDs below.
All 168 original compressed files matched the historical organized inputs
byte-for-byte during release preparation. The organizer verifies that exact
source content and copies it unchanged; arbitrary re-saved NIfTIs are not accepted
as original archive inputs. The pinned `subjects.txt` SHA-256 is
`52a8a9cae5992837140464f8d28a69135b096484786f3cb85c50834958eee4ca`.

### Prepared layout and registration convention

```text
OASIS_ROOT/
  pair_test_200.txt                  # use the bundled splits/pair_test_200.txt
  test/
    img0333.nii.gz
    seg0333.nii.gz
    img0115.nii.gz
    seg0115.nii.gz
    ...
```

The complete ordered paper pairs are in [oasis100.csv](../splits/oasis100.csv).
The timing subset is [oasis10.csv](../splits/oasis10.csv), ranks 0–9 of the same
seed-3 shuffle. The first column ID is **fixed**, the second **moving**. Leading
zeroes in subject IDs matter. Rank zero is fixed 0333, moving 0115. The original
sorted-list indices of the first ten pairs are 158,62,14,69,47,137,110,6,116,88.

The loader applies the historical LIA→RAS array transform: transpose `(0,2,1)`,
flip axis 0, flip axis 2. The resulting array must be `(160,224,192)` in ZYX
storage. Do not pre-reorient inputs to RAS and then apply this again. It uses
pairwise min/max normalization shared by fixed and moving images; fixed raw
image > 0 defines foreground. Segmentation labels 1–35 are only for label
metrics. HD95 uses the historical unit-voxel metric convention; no additional
resampling is performed here. A matching shape alone does not establish matching
data provenance. No download/registration or OASIS preprocessing is implicit in
the registration command.

## DIR-LAB original inputs

Unpack the original [DIR-LAB](https://www.dir-lab.com/) 4DCT and COPD archives:

```text
DIRLAB_ROOT/raw/
  4DCT/Case1Pack/
    Images/case1_T00_s.img
    Images/case1_T50_s.img
    ExtremePhases/Case1_300_T00_xyz.txt
    ExtremePhases/Case1_300_T50_xyz.txt
  COPD/copd1/copd1/
    copd1_iBHCT.img
    copd1_eBHCT.img
    copd1_300_iBH_xyz_r1.txt
    copd1_300_eBH_xyz_r1.txt
```

Repeat for cases 1–10. The raw loader also handles the distributed lowercase
`extremePhases`, alternative `-ssm.img` names, and nested case-8 directory. Native
shape/spacing are the per-case constants in [raw_lung.py](../rightpriordir/raw_lung.py).
4DCT fixed/moving phases are T00/T50; COPD iBH/eBH. Both landmark files have
300 rows in **XYZ, one-based native voxel coordinates**. Raw arrays use ZYX.

## Masks and preparation

[lungmask R231](https://github.com/JoHof/lungmask) supplies a native binary lung
mask (`labels > 0`). Download the upstream checkpoint once, outside Git:

```bash
mkdir -p "$RELEASE_WORK/weights"
curl -fL --output "$RELEASE_WORK/weights/unet_r231-d5d2fc3d.pth" \
  https://github.com/JoHof/lungmask/releases/download/v0.0/unet_r231-d5d2fc3d.pth
sha256sum "$RELEASE_WORK/weights/unet_r231-d5d2fc3d.pth"
```

The mask command records its full SHA-256. It subtracts **1024** from stored raw
CT values before R231 inference. Registration intensities instead use the raw
values clipped to [80,900], then subtract 80 and divide by 820; do **not** apply
the HU subtraction to that branch. Both phases have masks, but only the fixed
mask defines IDIR sampling; moving masks do not expand the optimization domain.

Commands below run **inside the release container** with `/data` read-only and
`/work` writable; see [running.md](running.md) for the mounts. Existing masks
may be supplied with the same filenames, but record their own provenance.

```bash
python -m rightpriordir masks --dataset 4dct --cases 1 2 3 4 5 6 7 8 9 10 \
  --data-root /data/dirlab --out-root /work/masks \
  --weights /work/weights/unet_r231-d5d2fc3d.pth --device cuda
python -m rightpriordir masks --dataset copd --cases 1 2 3 4 5 6 7 8 9 10 \
  --data-root /data/dirlab --out-root /work/masks \
  --weights /work/weights/unet_r231-d5d2fc3d.pth --device cuda

python -m rightpriordir prepare-lung --dataset 4dct --crop-mode corrected \
  --data-root /data/dirlab --mask-root /work/masks --out-root /work/prepared
python -m rightpriordir prepare-lung --dataset copd --crop-mode legacy \
  --data-root /data/dirlab --mask-root /work/masks --out-root /work/prepared
```

Output roots are respectively
`/work/prepared/dirlab_4dct_corrected_crop_nominal1mm_v1/4DCT/caseXX` and
`/work/prepared/dirlab_copd_legacy_crop_nominal1mm_v1/COPD/caseXX`.
Each case contains native/cropped/resampled images and masks, transformed
landmarks, array checksums, and `preprocess_audit.json`. Existing case directories
are never overwritten. Use a new output root after a partial failure.

## Corrected versus legacy crop

Both modes use the union of fixed/moving landmark bounding boxes and native
XYZ margins [10,10,5]. `corrected` computes the inclusive MATLAB one-based bounds
and then subtracts one before Python slicing. `legacy` historically used the
one-based landmark values directly as zero-based bounds. The landmarks
themselves are converted to zero-based in both modes.

| Frozen results | Required preparation |
| --- | --- |
| 4DCT own methods and Dual PL1A | corrected |
| COPD Dual P0/P1 and MR-D-BSCP P2e | legacy |
| COPD official pTVReg reference | official MATLAB corrected crop; frozen TRE only |

For COPD cases 02–10, legacy bounds are +1 voxel in each axis relative to
corrected bounds. Case01 clips at z=0: corrected ZYX bounds are
`[[0,108],[114,434],[70,435]]`, legacy `[[0,109],[115,435],[71,436]]`.
Its depth changes from 109 to 110 native slices, each 2.5 mm. These are distinct
protocols, not merely different names for the same data. Consistent legacy
geometry does not automatically invalidate a legacy TRE; mixing conventions does.

A corrected COPD preparation is available by choosing `--crop-mode corrected`.
It gets a separate protocol directory and **has no frozen registration results
in this release**. Bundled COPD configurations deliberately reject that protocol.

## Resampling and evaluation

Target shape is `rint(crop_shape * native_spacing_mm)`. Images use PyTorch
trilinear interpolation with `align_corners=True`; masks use nearest-neighbor.
This is the historical **nominal/approximately 1 mm** preparation: endpoint
preservation and integer rounding mean that physical spacing is not exactly
1 mm. Both nominal and endpoint-derived spacing are recorded.

Landmarks follow `XYZ1 → ZYX0 → subtract crop start → multiply
(registration_shape−1)/(crop_shape−1)`. The headline TRE retains the historical
convention: resample the registration displacement to native crop shape with
aligned endpoints, divide voxel components by native spacing, nearest-sample
the displacement and snap the warped landmark to native voxel indices. Subvoxel
TRE is separate. Do not substitute registration-grid TRE for native-grid TRE.

Public `field.npy` uses spatial ZYX, vector components XYZ, normalized-grid
displacements and fixed→moving pull direction (`align_corners=True`). Its
`field.json` must travel with it; evaluation validates the checksum, pair/case,
crop protocol and preparation metadata.
