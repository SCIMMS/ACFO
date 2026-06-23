# WAXS 2D Cake Benchmark Notes: NUFFT-Success Region

Date: 2026-06-13

## Scope

This note summarizes the physically scaled 2D WAXS cake-map benchmarks in the
region where the FINUFFT baseline still completes successfully. The goal is to
separate two cases:

- NUFFT-success region: compare speed directly against NUFFT.
- NUFFT-failure region: record memory/target-count limitations separately.

The main result is that the R-dependent 2D cake path remains faster than NUFFT
even before reaching the region where NUFFT fails.

## Physical Scaling

The benchmarks use the physical-grid helper, not a fixed arbitrary grid.

For each atom count, the box size is chosen from water-equivalent density:

```text
water molecules per nm^3 = 33.3679
atoms per water molecule = 3
```

The 1M-atom case therefore corresponds to a roughly 20 nm box:

```text
box side = 21.54 nm
r_max    = 15.23 nm
```

Common benchmark settings:

- `bin_width_nm = 0.1`
- `qmin = 0.05 A^-1`
- `Nq = 40`
- `wavelength_nm = 0.1`
- histogram backend: C++ float32
- angle LUT: cubic, size 32
- circular backend: C++
- R-dependent cutoff: `margin=16`, `cutoff_bin_size=16`

The angular grid is not fixed. It grows with the physical box size and q range:

```text
n_phi >= 2 * qmax * r_max + 2 * harmonic_margin
```

This is important because high-q WAXS and 20 nm-scale boxes increase the
required angular harmonic bandwidth.

## Implemented 2D Fast Path

The full cake-map output is:

```text
A(q, phi)
I(q, phi) = |A(q, phi)|^2
```

Unlike the 1D curve path, the 2D cake still needs the final inverse FFT over
harmonics. The optimization is therefore applied before that step, during
`Ahat(q, h)` construction.

The R-dependent cutoff uses:

```text
h_max(q, R) = ceil(|q_perp(q)| * R + margin)
```

Only the selected low-order harmonic prefix and negative-frequency tail are
contracted for each `(q, R)` shell. This reduces unnecessary high-harmonic work
at small R while preserving the full cake-map output grid.

Current API:

```python
plan.circular_fft_r_dependent_bandlimit(
    margin=16,
    cutoff_bin_size=16,
)
```

Implementation points:

- Python API: `PreparedCakePlan.circular_fft_r_dependent_bandlimit`
- Fourier coefficient API: `PreparedCakePlan.circular_ahat_r_dependent_bandlimit`
- C++ backend: `circular_contract_r_dependent`
- complex64 backend: `circular_contract_r_dependent64`

## Validation

The implementation was checked with:

```powershell
.\.venv\Scripts\python.exe -m py_compile src\waxs_cake\solvers.py scripts\benchmark_physical_scaling.py
.\.venv\Scripts\python.exe setup.py build_ext --inplace
.\.venv\Scripts\python.exe -m pytest -q
```

Result:

```text
46 passed
```

The benchmark script records both amplitude error and intensity error versus
the full dense circular cake path. The paper-relevant error is usually the
intensity error, because the measured cake map is intensity.

## NUFFT-Success Benchmark Results

Times are wall-clock totals including histogram construction, plan creation,
and first solve. The R-dependent intensity error is relative to the full dense
circular cake path.

| atoms | qmax | n_phi | full circular | R-dependent 2D | NUFFT | R-dependent vs NUFFT | intensity error |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100k | 2.2 A^-1 | 344 | 0.154 s | 0.095 s | 0.277 s | 2.92x | 2.17e-8 |
| 250k | 2.2 A^-1 | 456 | 0.356 s | 0.259 s | 0.417 s | 1.61x | 1.76e-7 |
| 500k | 2.2 A^-1 | 564 | 0.581 s | 0.341 s | 0.682 s | 2.00x | 4.37e-8 |
| 1M | 2.2 A^-1 | 704 | 0.996 s | 0.541 s | 1.365 s | 2.52x | 7.64e-8 |
| 100k | 6.3 A^-1 | 924 | 0.308 s | 0.280 s | 4.214 s | 15.04x | 1.73e-7 |
| 250k | 6.3 A^-1 | 1242 | 0.934 s | 0.498 s | 23.699 s | 47.59x | 2.76e-6 |
| 500k | 6.3 A^-1 | 1556 | 2.205 s | 0.979 s | 214.364 s | 218.92x | 3.95e-7 |

The `500k, qmax=6.3 A^-1` case was run with `repeats=1` because NUFFT took
more than 200 seconds. The other rows used `repeats=2`.

## Benchmark Files

The benchmark outputs used for the table are:

- `benchmark_results/physical_scaling_nufft_success_q2p2A_rdep_cake_m16_b16.json`
- `benchmark_results/physical_scaling_nufft_success_100k_q6p3A_rdep_cake_m16_b16.json`
- `benchmark_results/physical_scaling_nufft_success_250k_q6p3A_rdep_cake_m16_b16.json`
- `benchmark_results/physical_scaling_nufft_probe_500k_q6p3A_rdep_cake_m16_b16.json`

Prior high-q 1M attempts showed FINUFFT memory failure. That case should be
reported separately as a scaling limitation of the current unbatched NUFFT
baseline, not mixed into the NUFFT-success table.

## Interpretation

### Main Observation

The method is already competitive in the region where NUFFT succeeds.

At `qmax=2.2 A^-1`, the R-dependent 2D cake path is about `1.6x` to `2.9x`
faster than NUFFT over `100k` to `1M` atoms.

At `qmax=6.3 A^-1`, NUFFT still succeeds up to `500k` atoms in these tests, but
its runtime grows rapidly. The R-dependent 2D cake path is about `15x`, `48x`,
and `219x` faster for `100k`, `250k`, and `500k` atoms respectively.

### Why the High-q Advantage Grows

NUFFT sees the full target set. For a WAXS cake map, the target count grows as:

```text
Nq * Nphi
```

and `Nphi` must grow with `qmax * Rmax`.

The circular WAXS-specific method uses the structured Ewald-ring geometry. The
R-dependent path further reduces contraction work because small-R shells do not
need high harmonic modes even when the outer radius sets a large global
`n_phi`.

This is a WAXS-geometry advantage, not a generic Fourier-transform advantage.

### Accuracy

The intensity error versus the full dense circular cake path stayed in the
range:

```text
2e-8 to 3e-6
```

for the tested cases. This is promising, but it is still an algorithmic
reference comparison. Manuscript-level validation should also include real
structures and q-dependent atomic form factors.

## Publication Potential

### Current Assessment

The publication potential is real, especially as a narrow WAXS methods paper.
The strongest claim is not that the method replaces NUFFT generally. The
stronger claim is:

```text
For atomistic WAXS cake maps on curved-Ewald ring grids, a cylindrical
histogram plus circular harmonic contraction can exploit the geometry to
generate 2D cake maps and 1D curves faster than a generic NUFFT baseline, with
controlled intensity error.
```

This is a defensible claim because:

- the output is exactly the WAXS cake-map geometry, not arbitrary Fourier
  targets
- the grid size is selected from physical box size and q range
- the comparison includes a mature generic baseline
- the speed advantage is present before NUFFT fails
- the high-q scaling trend is favorable for the WAXS-specific method
- the error is measured against the dense circular reference

### Novelty Level

The novelty is best described as application-specific and moderate-to-strong:

- not a new general NUFFT algorithm
- not a new scattering theory
- but a specialized WAXS cake-map generator that exploits curved-Ewald ring
  structure, cylindrical binning, harmonic contraction, and R-dependent
  bandwidth

For Journal of Synchrotron Radiation, this can be a plausible methods paper if
the manuscript is framed around XFEL/WAXS practical computation rather than
general Fourier analysis.

### What Is Still Needed Before Submission

The current benchmark is not yet enough by itself. Before writing the paper,
the following should be added:

1. Real molecular or nanoparticle structures.

   Use PDB/MD structures or realistic water/solute boxes instead of only
   uniform random water-density boxes.

2. Element-specific q-dependent form factors.

   The speed story is stronger if form-factor changes are included, because
   this is one of the intended advantages over a generic target-only NUFFT run.

3. Fair memory-aware NUFFT baseline.

   The current NUFFT baseline is useful, but high-q failure should be handled
   with target batching so the comparison does not look unfair.

4. Error-versus-speed sweeps.

   Vary `margin`, `cutoff_bin_size`, `bin_width_nm`, q range, and atom count.
   Report both amplitude and intensity error, but emphasize intensity.

5. A clear scope boundary.

   Avoid claiming general superiority over NUFFT. Claim superiority for the
   intended WAXS cake-map grid and physically scaled atomistic workloads.

6. Reproducibility package.

   Keep the benchmark commands, generated JSON files, and environment setup in
   the repository.

## Recommended Paper Framing

The first paper should stay narrow:

```text
Fast atomistic WAXS cake-map simulation on curved-Ewald grids using cylindrical
histograms and R-dependent circular harmonic contraction.
```

A strong paper structure would be:

1. Motivation: XFEL/WAXS single-shot template generation and large atomistic
   systems.
2. Method: cylindrical histogram, Ewald-ring circular convolution, harmonic
   contraction, R-dependent bandwidth.
3. Accuracy: dense circular, direct small-system, and NUFFT comparisons.
4. Performance: NUFFT-success region, high-q scaling, and memory-limited NUFFT
   region.
5. Applications: realistic molecular/nanoparticle WAXS cake maps and 1D curves.
6. Limitations: binning error, form-factor model, detector effects, and
   non-WAXS target grids.

Broader ideas such as MD convergence monitoring, incremental histogram updates,
valid observation regions, and structured-material variants should be treated
as discussion or future work unless they are fully validated before submission.

## Bottom Line

The method now has a plausible first-paper core:

- physically scaled 20 nm-class atomistic boxes
- direct comparison against NUFFT where NUFFT succeeds
- favorable high-q scaling
- useful 2D cake-map output, not only 1D curves
- controlled intensity error

The next critical step is to replace synthetic random boxes with realistic
structures and form factors. If the speed/error pattern survives that test,
the work is strong enough to shape into a focused WAXS/XFEL methods manuscript.
