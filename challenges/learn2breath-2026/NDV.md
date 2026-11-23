# How Learn2Breath accelerates SITReg's NDV loss

The Learn2Breath implementation evaluates the same ten tetrahedra as SITReg's default non-diffeomorphic volume (NDV) loss. The speedup comes from how we compute their signed volumes. We take differences directly on the displacement tensor, evaluate 3×3 determinants with elementwise arithmetic, and reduce each determinant map to a scalar. On the challenge grid this reduced median forward-and-backward time from **753.5 to 409.3 ms**. [Released PyTorch implementations](learn2breath/ndv.py) · [Original SITReg algorithm at the pinned revision](https://github.com/honkamj/SITReg/blob/a665a46a3553b26ade6ff5f57815e5e8ec43a948/src/algorithm/ndv.py)

## Which quantity are we computing?

Let $u(p)$ be a displacement in voxels and $\phi(p)=p+u(p)$ the corresponding sampling position. At each interior voxel, SITReg builds eight tetrahedra from the possible combinations of forward and backward differences along the three axes. Two more, called `j1*` and `j2*`, use face-diagonal differences. Each determinant $d_t(p)$ measures an oriented volume. A negative determinant means the sampled tetrahedron has reversed orientation, and a more negative value contributes a larger penalty.

For the unmasked, zero-threshold loss used in Learn2Breath, the scalar is

$$
L_{\mathrm{NDV}}(u)
= \frac{1}{12} \sum_{b}\sum_{p\,\in\,\mathrm{interior}}\sum_{t=1}^{10}
  \max\!\left(-d_{b,t}(p), 0\right).
$$

The interior excludes one voxel on each face so all neighbors exist. The factor $1/12$ is SITReg's $0.5/6$ tetrahedral weighting. The sum includes every batch element and stays in voxel-volume units. In upstream `NDVLoss`, the final `.mean()` acts on the scalar already returned by the volume sum. SITReg's general helper also supports a mask and a configurable threshold. The challenge implementation specializes the unmasked case at threshold zero. [SITReg loss wrapper](https://github.com/honkamj/SITReg/blob/a665a46a3553b26ade6ff5f57815e5e8ec43a948/src/loss/regularity.py)

Both released paths accept a voxel displacement tensor of shape `[B, 3, Z, Y, X]`, with components ordered `(z, y, x)` to match the spatial axes. The registration solver produces normalized `(x, y, z)` displacements in `[Z, Y, X, 3]`. Its shared adapter multiplies those components by `(X−1)/2`, `(Y−1)/2`, and `(Z−1)/2`, respectively, then reorders the channels. This conversion matters even in a synthetic test. Reversing the first spatial axis must change the displacement component for that same axis.

## Where the implementations differ

| Operation | SITReg convolution reference | Learn2Breath accelerated path |
| --- | --- | --- |
| Coordinate field | Builds a dense identity grid and adds $u$ to obtain $\phi$. | Works on $u$ and adds the known identity contribution to each difference. |
| Neighbor differences | Six grouped `conv3d` calls produce three axial and three diagonal families of differences. | Offset tensor slices select the same neighbors on the common one-voxel interior. |
| Determinants | Stacks three vector columns into a 3×3 matrix at each voxel and calls `torch.linalg.det` for eleven orientations. | Computes the ten determinants consumed by NDV using explicit scalar arithmetic. The reference's eleventh, the central-difference determinant `000`, has zero role in the loss. |
| Reduction | Adds the ten thresholded contributions into a voxelwise NDV map, then sums the map. | Thresholds and sums each determinant map over voxels and batch, then adds the ten scalars. |

### A forward difference, worked through

Let $e_i$ be a one-voxel step along array axis $i$. A forward-difference column computed by convolution on the transformed coordinates is

$$
\phi(p+e_i)-\phi(p)
= \big[p+e_i+u(p+e_i)\big]-\big[p+u(p)\big]
= e_i + u(p+e_i)-u(p).
$$

The right-hand side needs two views into $u$ and one broadcast unit vector. The backward column similarly becomes $e_i+u(p)-u(p-e_i)$. The same identity handles a diagonal offset $q$: its column is $q+u(p+q)-u(p)$, with two signed unit steps in $q$. The accelerated path preserves the column order of each starred tetrahedron, including the swapped last two columns in `j2*`. Swapping those columns changes a determinant's sign.

### Why this uses less GPU memory

For three vector columns $a,b,c$, the accelerated code expands the scalar triple product $a\cdot(b\times c)$ component by component. It therefore avoids allocating dense, per-voxel 3×3 matrices or dispatching a general determinant routine. For scale, **one** float32 tensor of 3×3 matrices on the challenge interior occupies about **321 MiB** ($222\times190\times222\times9$ float32 values).

The implementation runs in eager PyTorch. It still creates the ten determinant maps and uses autograd to differentiate the loss back through the dense field to the spline controls. The smaller peak comes from the intermediate tensors the calculation avoids, including the dense identity grid, stacked Jacobian matrices, and combined NDV map. The benchmark below measures their overall effect.

## Checking the loss and its gradient

A useful analytic check is a reflection of the first array axis. Set $u_0(p)=-2p_0$ and the other components to zero, so $\phi_0(p)=-p_0$. The eight axial determinants are each $-1$, and the two starred determinants are each $-2$. Each interior voxel therefore contributes $(8+2\times2)/12=1$. For one field of shape `7 × 8 × 9`, both implementations should return **210**, the number of interior voxels. Identity and constant translation give zero.

This example runs on CPU from the challenge directory in the documented Python environment. It compares the gradients as well as the scalars, since Adam updates the controls using the gradient.

```python
import torch
from learn2breath.ndv import sitreg_ndv_loss, sitreg_ndv_loss_fast

u = torch.zeros(1, 3, 7, 8, 9, dtype=torch.float64)
u[:, 0] = -2 * torch.arange(7, dtype=u.dtype).view(7, 1, 1)
u.requires_grad_()

reference = sitreg_ndv_loss(u)
fast = sitreg_ndv_loss_fast(u)
torch.testing.assert_close(reference, u.new_tensor(210.0))
torch.testing.assert_close(fast, reference)
grad_ref, = torch.autograd.grad(reference, u)
grad_fast, = torch.autograd.grad(fast, u)
assert grad_ref.abs().max() > 0
torch.testing.assert_close(grad_fast, grad_ref)
```

The two paths are algebraically equivalent. In floating point, forming $p+u(p)$ before differencing can round differently from differencing $u$ first. The determinant evaluation and sum order also differ. In particular, a determinant very close to zero can round to opposite sides of the folding threshold. For general fields, compare per-direction maps, scalar losses, and gradients with numerical tolerances.

## What the benchmark measured

The submission benchmark compared the local convolution reference with the eager accelerated implementation on one RTX 6000 Ada, using PyTorch 2.6.0, CUDA 11.8, and float32 voxel displacement fields of shape `[1, 3, 224, 192, 224]`. Each path ran 10 warm-up iterations followed by 30 timed iterations per measurement with CUDA events. The timing covers NDV on an already dense field.

| Measurement | Convolution reference | Accelerated |
| --- | ---: | ---: |
| Median forward | 77.6 ms | 57.1 ms |
| Median forward + backward | 753.5 ms | 409.3 ms |
| Peak PyTorch CUDA allocation | 10.764 GiB | 3.092 GiB |

This gives a **1.84× speedup** for the NDV forward/backward computation and **71.3% lower peak allocation**. A separate seeded full-resolution field produced exactly `285825.84375` on both paths. The measurements are from the `Learn2Breath26` experiment `2026-09-01_mrdbscp_submission_runtime_v1`, recorded in `ndv_benchmark/ndv_benchmark.json`.

The registration schedule controls **how often** this cost is paid. The submitted run fits five pyramid levels with LCC and isoTV, turns on `10,000 × NDV` for 100 additional full-resolution Adam updates, then evaluates NDV once on the final field for reporting. The table measures the loss implementation alone. [Registration schedule and run instructions](README.md) · [Optimizer](learn2breath/register.py)
