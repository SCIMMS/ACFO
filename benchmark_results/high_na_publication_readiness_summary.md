# High-NA publication-readiness summary

This note consolidates the High-NA evidence that is ready to use in the
cross-domain curved-Ewald/operator manuscript. It is a claim-boundary summary,
not a new benchmark run.

## Defensible High-NA claim

The current evidence supports this scoped claim:

> The same circular-harmonic factorization used for curved-Ewald/WAXS-style
> operators recovers known High-NA Debye-Wolf/Richards-Wolf reductions and
> becomes useful for repeated vectorial focal-field evaluation on cylindrical,
> polar, annular, axial, or sparse focal-volume grids. It is not presented as a
> universal dense Cartesian PSF rasterizer.

## Evidence lanes

| lane | strongest local evidence | readout |
|---|---|---|
| Scalar Debye-Wolf correctness | `high_na_pupil_spectrum_option_matrix_summary.md` | Direct-reference vortex rows fail with geometric-only cutoff (`L2 ~ 1`) and recover with adaptive sparse cutoff (`4.3e-10` small, `4.0e-08` representative). |
| Pupil high-harmonic safety | `high_na_pupil_spectrum_option_matrix_summary.md` | `pupil_spectrum="adaptive"` adds only significant beyond-cutoff harmonics; benign rows need no extras and remain near `1e-12` vs FINUFFT. |
| Generic CPU FINUFFT baseline | `high_na_pupil_spectrum_option_matrix_summary.md` | Adaptive sparse total-time speedups vs FINUFFT are about `2.03x`, `1.83x`, `1.46x`, and `2.58x` for representative/large rows. |
| Vectorial Richards-Wolf correctness | `high_na_vectorial_backpropagation_representative_direct_summary.md` | Representative vectorial vortex separable path matches direct forward/adjoint near machine precision (`3.443e-15` forward, `8.843e-16` adjoint). |
| Vectorial GPU hot loop | `high_na_torch_gpu_vectorial_rollup.md` | Representative vectorial vortex batch-32 forward+adjoint hot-loop speedup is `81.12x` vs CPU separable reference with `~1e-7` to `5e-7` field/adjoint L2. |
| GPU cuFINUFFT baseline | `high_na_gpu_cufinufft_baseline_rollup.md` | Structured vectorial GPU path is faster than matched cuFINUFFT on cylindrical targets; representative batch-32 hot pair speedup is `708.39x`, setup-inclusive `35.79x`. |
| External package focal-pattern check | `high_na_pyfocus_vectorial_package_summary.md` | PyFocus/PyCustomFocus XY focal-pattern shape L2 is `1.112e-04`, `1.013e-04`, and `3.478e-05` for linear, circular, and vortex cases. |
| Cartesian stress test | `high_na_external_package_matched_extension_rollup.md` | Against psf-generator, an oversampled cylindrical adapter reaches few-`1e-3` intensity-shape agreement while remaining faster on the tested GPU. |
| Physical demo | `high_na_aberration_correction_summary.md`, phase-mask demos | Phase-only pupil-mask optimization and aberration correction demonstrate that the operator can sit inside repeated optical design loops. |

## Best manuscript placement

| section | role |
|---|---|
| Introduction/background | High-NA as the known wave-optics bridge: circular-harmonic / Fourier-Bessel Debye-Wolf reductions already matter in optics. |
| Methods | Derive scalar and vectorial High-NA forms as the same prepared harmonic operator pattern, with adaptive pupil-spectrum support. |
| Main figure or SI | Show vectorial correctness and GPU hot-loop / cuFINUFFT speedup on native cylindrical targets. |
| SI | PyFocus/PyCustomFocus and psf-generator package checks, Cartesian adapter stress test, phase-mask demos. |

## Claim boundaries

- The circular-harmonic identity itself is not the novelty.
- The High-NA result is strongest as a bridge/example showing that the
  factorization recovers a known optics reduction and can be implemented as a
  prepared repeated operator.
- The strongest native High-NA regime is cylindrical/polar/ROI focal-field
  evaluation with repeated masks, coherent modes, or inverse-design iterations.
- Dense Cartesian PSF replacement remains a limitation; the current adapter is
  promising but should be framed as a stress test.
- Peak process RSS and independent hardware repeats are not yet fully measured.

## Remaining optional work

These are not blockers for moving to the overall paper summary:

1. Add process-level peak RSS instrumentation for GPU and CPU High-NA rows.
2. Add a stricter matched Cartesian package row with complex-field alignment,
   if the paper tries to make a stronger PSF-package comparison.
3. Re-run the representative vectorial GPU rows on a newer GPU if available.

The current High-NA package is sufficient for the cross-domain story when the
claim is kept to prepared structured focal-volume operators rather than generic
PSF rasterization.
