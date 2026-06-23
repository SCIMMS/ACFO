# WAXS 1D Curve Optimization Notes

Date: 2026-06-12

## Scope

This note summarizes the current 1D WAXS curve optimization work in the
Atomic WAXS Cake-Map Simulator. The focus is the ring-averaged curve

```text
I(q) = mean_phi |A(q, phi)|^2
```

computed from the same cylindrical histogram and curved-Ewald/circular-FFT
representation used by the cake-map solver.

The current conclusion is:

- The default 1D path should remain the R-grouped circular path.
- The R-dependent harmonic cutoff path is useful as an optional high-q fast
  path, not as the general default.
- For physically scaled 1D WAXS curves, the dedicated curve path is already
  competitive with NUFFT and can be faster, especially when the workflow needs
  WAXS-specific gridding, form-factor changes, and direct 1D output.

## Implemented Methods

### 1. R-Grouped 1D Curve

Current default:

```python
PreparedCakePlan.ring_average_intensity()
```

This calls:

```python
PreparedCakePlan.ring_average_intensity_r_grouped()
```

The contraction order is:

```text
Hhat(e, R, z, h)
  -> B(q, R, h)       z reduction and element/form-factor reduction
  -> Ahat(q, h)       radial kernel contraction
  -> I(q)             Parseval ring average
```

This avoids producing the final full cake-map amplitude when only a 1D curve is
needed.

### 2. R-Dependent Harmonic Bandlimit

Optional path:

```python
PreparedCakePlan.ring_average_intensity_r_dependent_bandlimit(
    margin=16,
    cutoff_bin_size=16,
)
```

For each q and radial shell R, it uses:

```text
h_max(q, R) = ceil(|q_perp(q)| * R + margin)
```

then clips to the angular Nyquist limit and rounds up to `cutoff_bin_size`.
Only Fourier modes in the selected low-order prefix and high-order negative
tail are accumulated.

This is an approximate fast path. The current test/benchmark setting uses
`margin=16` and `cutoff_bin_size=16`.

## Backend Changes

The R-dependent path now has a C++/pybind11 backend:

- `circular_ring_average_r_dependent`
- `circular_ring_average_r_dependent64`

The important low-level optimization is in the worker loop:

- accumulate only selected harmonic modes for each `(q, element, R)` block
- avoid resetting the full harmonic buffer when the cutoff is below Nyquist
- reset only the selected prefix and negative-frequency tail
- multiply by the radial kernel only for the selected modes

This matters because high-q grids can have large `n_phi`, and clearing or
touching all harmonics defeats the purpose of an R-dependent cutoff.

## Validation

Commands run after the implementation:

```powershell
.\.venv\Scripts\python.exe -m py_compile src\waxs_cake\solvers.py scripts\benchmark_physical_scaling.py
.\.venv\Scripts\python.exe setup.py build_ext --inplace
.\.venv\Scripts\python.exe -m pytest tests\test_solvers.py -k "r_dependent or r_grouped or ring_average" -q
.\.venv\Scripts\python.exe -m pytest -q
```

Result:

```text
45 passed
```

The R-dependent curve error versus the dense cake-map ring average was around
`4.6e-8` to `1.4e-7` in the benchmark cases below.

## Benchmark Conditions

Common settings:

- synthetic water-density box
- `bin_width_nm = 0.1`
- `Nq = 40`
- `wavelength_nm = 0.1`
- histogram backend: C++ float32
- angle LUT: cubic, size 32
- circular backend: C++
- R-dependent settings: `margin=16`, `cutoff_bin_size=16`

Benchmark script:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_physical_scaling.py ...
```

## Benchmark Summary

Times are wall-clock totals including histogram construction, plan creation,
and first curve solve.

| atoms | qmax | n_phi | default 1D total | R-dependent total | NUFFT | conclusion |
|---:|---:|---:|---:|---:|---:|---|
| 100k | 2.2 A^-1 | 344 | 0.162 s | 0.244 s | 0.282 s | default 1D wins |
| 100k | 6.3 A^-1 | 924 | 0.444 s | 0.424 s | 4.362 s | R-dependent slightly wins |
| 1M | 2.2 A^-1 | 704 | 0.923 s | 0.968 s | 1.631 s | default 1D wins |
| 1M | 6.3 A^-1 | 1952 | 2.670 s | 2.295 s | failed | R-dependent wins vs default |

The 1M, qmax 6.3 A^-1 NUFFT run failed with:

```text
RuntimeError: FINUFFT general malloc failure
```

so that case should not be presented as a clean NUFFT timing comparison yet.
A fair high-q NUFFT baseline needs target batching or a memory-aware NUFFT
driver.

## Benchmark Files

The benchmark outputs used here are:

- `benchmark_results/physical_scaling_100k_bin0p1_q2p2A_rdep_curve_cpp_loopopt2_m16_b16_with_nufft.json`
- `benchmark_results/physical_scaling_100k_bin0p1_q6p3A_rdep_curve_cpp_loopopt2_m16_b16_with_nufft.json`
- `benchmark_results/physical_scaling_1m_bin0p1_q2p2A_rdep_curve_cpp_loopopt2_m16_b16_with_nufft.json`
- `benchmark_results/physical_scaling_1m_bin0p1_q6p3A_rdep_curve_cpp_loopopt2_m16_b16.json`

## Interpretation

### Where This Beats NUFFT

The dedicated WAXS 1D path is most attractive when:

- the required output is the 1D ring-averaged WAXS curve, not arbitrary
  nonuniform Fourier samples
- the q grid and detector angular sampling follow the curved-Ewald WAXS
  geometry
- form factors or q ranges need to be changed repeatedly after the structure
  has been binned
- the histogram representation is acceptable for the target accuracy
- the high-q NUFFT target set becomes large enough that memory and target
  batching become important

For mid-q 1D curves, the R-grouped path is currently the better default.
For high-q curves, R-dependent bandlimiting can reduce harmonic work enough to
beat the R-grouped default.

### Where NUFFT Is Still Strong

NUFFT remains the better conceptual baseline when:

- arbitrary q targets are needed
- no cylindrical/WAXS-specific structure should be assumed
- one wants a mature, general nonuniform Fourier backend
- the target count is memory-manageable or already batched efficiently

This means the paper claim should not be "we replace NUFFT in general." The
stronger and more defensible claim is:

```text
A WAXS-specific curved-Ewald circular representation can compute cake maps and
1D curves faster than a generic NUFFT baseline in the intended WAXS grid
setting, with controlled error.
```

## Current Default Recommendation

Keep:

```python
ring_average_intensity() -> ring_average_intensity_r_grouped()
```

Do not switch the default to R-dependent cutoff yet.

Use R-dependent cutoff explicitly for high-q tests:

```python
plan.ring_average_intensity_r_dependent_bandlimit(
    margin=16,
    cutoff_bin_size=16,
)
```

Reason:

- at qmax 2.2 A^-1, R-dependent is slower
- at qmax 6.3 A^-1, R-dependent is faster
- the high-q gain is real but not universal

## Next Optimization Directions

1. Add a memory-aware NUFFT baseline.

   The 1M high-q NUFFT failure means the current comparison is incomplete.
   Implement q/phi target batching for NUFFT so high-q cases produce a fair
   timing instead of failing on allocation.

2. Improve R-dependent grouping.

   The current C++ path still loops over `(q, element, R, z, h)` with scalar
   complex accumulation. It could be improved by grouping R rows with the same
   cutoff and using more contiguous batched operations.

3. Avoid generating full `khat` for R-dependent high-q cases.

   The current R-dependent path still calls the full kernel FFT block and then
   skips modes during contraction. A stronger implementation would generate or
   retain only the required harmonic modes.

4. Benchmark complex64 versus complex128 for the 1D paths.

   The histogram already defaults to float32. If the 1D curve error remains
   acceptable, complex64 may give a useful speed and memory improvement.

5. Separate one-shot and reuse benchmarks.

   For WAXS single-shot analysis, the one-shot total is the most relevant
   metric. For form-factor or q-range sweeps, cached histogram/plan timings
   should be reported separately.

## Paper-Framing Note

The strongest first-paper story remains narrow:

- WAXS-specific cake map and 1D curve generation
- curved-Ewald geometry
- cylindrical histogram plus circular harmonic contraction
- direct/NUFFT comparison
- accuracy controlled against dense cake-map or NUFFT references
- practical scaling to large atomistic systems

The 1D curve results are useful because many WAXS workflows ultimately compare
radial curves, and because avoiding the full cake-map output can reduce
unnecessary work. However, the full cake-map method remains important for the
paper because it is the more distinctive WAXS/XFEL detector-facing output.
