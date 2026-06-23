# Debye vs WAXS 1D Curve Comparison Notes

Date: 2026-06-13

## Scope

This note records the first direct comparison between the exact Debye curve and
the current WAXS 1D curve path.

The three quantities compared are:

1. Exact Debye intensity with unit form factors:

   ```text
   I_Debye(q) = sum_i sum_j sinc(q * |r_i - r_j|)
   ```

   where `sinc(x) = sin(x) / x`.

2. Exact Ewald-ring curve from direct atomistic amplitudes:

   ```text
   I_ring(q) = mean_phi |A(q, phi)|^2
   ```

3. Histogram/circular WAXS 1D curve:

   ```text
   I_hist(q) = PreparedCakePlan.ring_average_intensity()
   ```

This separation is important. Debye is a full 3D orientational average. The
WAXS 1D curve is an azimuthal average over a fixed Ewald ring for a fixed beam
direction. They should agree only for isotropic structures or after sufficient
orientation averaging. For a finite anisotropic structure, the meaningful
algorithmic reference for `I_hist(q)` is `I_ring(q)`, not directly
`I_Debye(q)`.

## Script

The comparison script is:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_debye_curve.py
```

It writes JSON and CSV output under `benchmark_results/`.

The initial runs used:

- synthetic water-density boxes
- unit form factors
- exact Debye backend: `scipy.spatial.distance.pdist`
- `qmin = 0.05 A^-1`
- `Nq = 24`
- `wavelength_nm = 0.1`
- `bin_width_nm = 0.1`
- C++ histogram backend
- float32 histogram
- cubic angle lookup table with size 32
- C++ circular backend
- default 1D curve path: `ring_average_intensity_r_grouped`

## Low-q Initial Result

Command:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_debye_curve.py --atom-counts 250,500,1000,2000 --nq 24 --qmax 2.2 --orientation-samples 4 --output benchmark_results\debye_curve_comparison_initial.json
```

| atoms | n_phi | Debye total | WAXS curve total | speedup | ring vs Debye | curve vs ring | curve vs Debye |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 250 | 180 | 0.0680 s | 0.0058 s | 11.8x | 2.48e-2 | 2.37e-3 | 2.37e-2 |
| 500 | 180 | 0.1530 s | 0.0032 s | 47.3x | 1.25e-2 | 8.17e-4 | 1.22e-2 |
| 1000 | 180 | 0.4639 s | 0.0129 s | 36.1x | 1.51e-2 | 1.40e-3 | 1.38e-2 |
| 2000 | 180 | 2.5635 s | 0.0086 s | 297.7x | 1.55e-2 | 9.97e-4 | 1.59e-2 |

## High-q Initial Result

Command:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_debye_curve.py --atom-counts 500,1000,2000 --nq 24 --qmax 6.3 --orientation-samples 2 --output benchmark_results\debye_curve_comparison_highq_initial.json
```

| atoms | n_phi | Debye total | WAXS curve total | speedup | ring vs Debye | curve vs ring | curve vs Debye |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 186 | 0.1199 s | 0.0062 s | 19.5x | 1.60e-2 | 1.02e-3 | 1.56e-2 |
| 1000 | 224 | 0.4954 s | 0.0048 s | 102.3x | 1.20e-3 | 7.04e-4 | 1.24e-3 |
| 2000 | 274 | 1.8708 s | 0.0094 s | 198.8x | 1.04e-2 | 5.26e-4 | 1.01e-2 |

## Interpretation

The current histogram/circular 1D path is already close to the exact Ewald-ring
curve. In these initial cases, `I_hist(q)` versus `I_ring(q)` is about
`5e-4` to `2e-3` relative L2 error.

The larger discrepancy is usually `I_ring(q)` versus `I_Debye(q)`, about
`1e-3` to `2e-2` in these finite random boxes. This is a physics/averaging
difference, not primarily a solver error.

The first blocked NumPy Debye implementation was a conservative baseline. The
script now defaults to `--debye-backend pdist`, which computes pair distances in
SciPy's C code and is materially faster for these sizes. This reduces the
reported speedup but does not change the scaling interpretation: exact Debye
still scales with pair count, while the WAXS curve path scales through the
cylindrical histogram and harmonic contraction.

Therefore, the paper should not yet claim an unconditional Debye replacement.
The defensible statement is narrower:

```text
For isotropic or orientation-averaged atomistic WAXS curves, the cylindrical
histogram/circular 1D path can approximate the Debye orientational average much
faster, while also producing the detector-relevant fixed-ring curve directly.
```

## Next Checks

Before making a manuscript-level Debye claim, add:

1. Spherical or explicitly isotropic test structures, not only finite boxes.
2. More orientation samples to show convergence from `I_ring(q)` to
   `I_Debye(q)`.
3. Error sweeps over `bin_width_nm`, `N_phi`, q range, and atom count.
4. q-dependent element form factors.
5. A medium-N Debye baseline using pair blocking or sampled-pair approximation.
