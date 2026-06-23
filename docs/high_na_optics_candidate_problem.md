# High-NA Optics Candidate Problem

Date: 2026-06-15

## Bottom Line

The best high-NA optics problem for this algorithm is:

```text
Fast vectorial Debye-Wolf / Richards-Wolf focal-field evaluation for
non-axisymmetric pupil masks, aberrations, and inverse-design loops on a
cylindrical or sparse 3D focal-region grid.
```

This is the high-NA analogue of the WAXS cake-map problem. The common structure
is not "optics" in general, but repeated Fourier evaluation on a rotationally
structured curved manifold.

The problem should not be framed as replacing standard FFT optics. It should be
framed as accelerating a specific regime:

- high numerical aperture;
- vectorial diffraction, not paraxial scalar FFT only;
- non-axisymmetric pupil amplitude, phase, or polarization;
- many focal-field evaluations for PSF engineering, adaptive optics, or inverse
  design;
- detector or objective functions defined on cylindrical rings, 3D regions of
  interest, or sparse target points rather than only a full Cartesian grid.

## Why This Is The Right Cross-Field Demo

High-NA focusing is commonly described by the Debye-Wolf/Richards-Wolf integral.
For a pupil field on the angular aperture, the focal field can be written in a
form like:

```text
E_c(rho, psi, z)
  = int_theta int_phi
      W_c(theta, phi)
      exp(i k z cos(theta))
      exp(i k rho sin(theta) cos(phi - psi))
      sin(theta) dphi dtheta
```

Here:

- `theta, phi` are pupil/reference-sphere coordinates;
- `rho, psi, z` are focal-region cylindrical coordinates;
- `W_c(theta, phi)` includes pupil phase, amplitude, apodization, aberration,
  and vectorial polarization mixing for component `c`;
- `theta <= theta_max`, with `sin(theta_max) = NA / n`.

Expanding the pupil field in azimuthal harmonics gives:

```text
W_c(theta, phi) -> What_c(theta, h)
```

and the focal field becomes:

```text
E_c(rho, psi, z)
  = sum_h exp(i h psi)
      int_theta
        What_c(theta, h)
        i^h J_h(k rho sin(theta))
        exp(i k z cos(theta))
        sin(theta) dtheta
```

This is directly analogous to the WAXS solver:

| WAXS cake-map solver | high-NA Debye-Wolf solver |
|---|---|
| atom histogram in `(R, z, beta)` | pupil field in `(theta, phi)` |
| beta FFT | pupil azimuth FFT |
| `K_h(q, R) = i^h J_h(q_perp R)` | `K_h(rho, theta) = i^h J_h(k rho sin theta)` |
| z phase `exp(i q_z z_atom)` | defocus phase `exp(i k z cos theta)` |
| output `A(q, phi)` | output `E(rho, psi, z)` |
| R-dependent harmonic cutoff | rho/theta-dependent harmonic cutoff |

The same core idea survives: the angular bandwidth needed at a target radius is
not arbitrary. It is controlled by:

```text
|h| <= k rho sin(theta_max) + margin
```

or more locally:

```text
|h| <= k rho sin(theta) + margin
```

So the algorithm can use occupied harmonic support instead of treating every
pupil sample and every focal target as a generic nonuniform Fourier pair.

## Best Initial Application

The most practical first demo is:

```text
Fast 3D PSF / focal-field library generation for engineered high-NA pupil
masks in microscopy and light shaping.
```

Concrete examples:

- vortex or annular phase masks for STED/MINFLUX-like donut/toroidal foci;
- astigmatic, coma, spherical-aberration, or Zernike-corrected pupils;
- spatial-light-modulator phase masks for arbitrary 3D two-photon excitation
  patterns;
- repeated adaptive-optics correction trials;
- inverse design of non-axisymmetric focal fields.

This is better than starting with a purely axisymmetric Airy focus because
axisymmetric fields already have strong Hankel-transform-style baselines.

## What Problem We Can Claim To Solve

Safe claim:

```text
We accelerate repeated vectorial high-NA focal-field evaluation for
non-axisymmetric pupil functions by using a circular-harmonic Debye-Wolf
factorization with explicit harmonic cutoffs.
```

Stronger claim, only after benchmark:

```text
For high-NA PSF engineering and adaptive-optics loops, the harmonic
factorization can compute controlled-error 3D focal fields faster than direct
Debye-Wolf quadrature and generic NUFFT baselines when the requested focal
grid is cylindrical, ring-like, or reused across many pupil updates.
```

Claims to avoid:

- not a replacement for all Fourier optics propagation;
- not a replacement for FFT-Debye methods on dense Cartesian grids until
  benchmarked directly;
- not a claim that axisymmetric focusing is new;
- not a high-NA microscopy result unless the speedup enables a real design,
  fitting, or adaptive-optics workflow.

## Baselines

Use three baseline levels.

1. Direct Debye-Wolf quadrature

   This is the correctness reference:

   ```text
   for each target (rho, psi, z):
       sum over theta, phi pupil samples
   ```

   It is simple, exact to quadrature, and good for small grids.

2. FINUFFT type-3 baseline

   The pupil samples lie on a spherical cap in k-space:

   ```text
   xi(theta, phi) = k (sin theta cos phi, sin theta sin phi, cos theta)
   ```

   Focal targets are real-space points:

   ```text
   r = (rho cos psi, rho sin psi, z)
   ```

   This makes the direct Debye-Wolf forward model a type-3 NUFFT-like
   comparison:

   ```text
   E(r) = sum_m W_m exp(i xi_m dot r)
   ```

3. Domain-specific optics baseline

   For manuscript-quality optics claims, we eventually need a domain baseline:

   - FFT-Debye / Cartesian focal-volume methods;
   - Hankel or Bessel-transform methods for axisymmetric pupils;
   - existing vectorial focal-field packages for arbitrary masks.

   The first proof-of-fit can start with direct and FINUFFT. The domain-specific
   baseline becomes necessary before making a high-IF cross-field claim.

   Current external-package snapshot:

   - `scripts/benchmark_high_na_pyfocus_vectorial_package.py` compares the local
     vectorial Richards-Wolf implementation against PyFocus/PyCustomFocus
     3.4.0 on PyFocus's native Cartesian XY plane.
   - `benchmark_results/high_na_pyfocus_vectorial_package_summary.md` reports
     scale-fit intensity-shape L2 errors of about `3.5e-05` to `1.1e-04` for
     linear x-polarized, right-circular, and vortex x-polarized cases at
     `NA=0.95`.
   - The local separable solver is timed on an equal-target-count cylindrical
     grid, not on PyFocus's Cartesian grid. The measured hot-loop speedup is
     about `58x` to `70x`; one-shot speedup including plan build is about
     `19x` to `20x` for this small snapshot.
   - This is strong first evidence that the vectorial formulation and physical
     focal patterns are aligned with a domain package. It is not yet a matched
     dense-Cartesian FFT-Debye or GPU PSF-generator comparison.

   Current GPU-backend snapshot:

   - `scripts/benchmark_high_na_torch_gpu.py` adds a PyTorch backend for the
     vectorial separable forward and adjoint/backpropagation contractions.
   - On an NVIDIA GeForce RTX 2070 SUPER, the representative vectorial vortex
     workload with batch 32 gives about `101x` forward hot-loop speedup, `57x`
     adjoint hot-loop speedup, and `81x` combined forward+adjoint speedup over
     the local CPU separable batch reference at `complex64`. Forward and adjoint
     relative L2 errors are about `1.8e-7` and `4.6e-7`.
   - The same workload at batch 8 gives about `9x` combined forward+adjoint
     hot-loop speedup at `complex128`, with near-machine-precision forward and
     adjoint L2 errors.
   - `scripts/benchmark_high_na_gpu_dense_baseline.py` adds a same-device dense
     direct CUDA quadrature reference. On the representative vectorial vortex
     cylindrical workload at `complex64`, batch 8, the separable GPU path gives
     about `5.0x` forward+adjoint hot-loop speedup over dense direct while
     matching it to about `3.8e-6` forward L2 and `7.4e-7` adjoint L2.
   - The same dense direct script also records a Cartesian arbitrary-target GPU
     anchor: batch 4, 5120 spatial targets/mask, about `0.011 s` for a
     forward+adjoint pair.
   - `scripts/benchmark_high_na_psf_generator_baseline.py` adds an external
     `psf-generator` `VectorialCartesianPropagator` timing anchor on the same
     RTX 2070 SUPER. In the representative timing-only row,
     `psf-generator` computes a dense 64x64x5 Cartesian vectorial stack in
     about `4.5 ms`, while the local structured cylindrical backend computes
     a 16x64x5 vectorial grid in about `1.0 ms` for batch 1 and about `2.0 ms`
     for batch 32.
   - `scripts/benchmark_high_na_cylindrical_backprop_design.py` tests the
     design-loop interpretation directly. For an annular cylindrical ROI
     field-matching objective on the representative vectorial vortex workload,
     batch 32, the separable GPU iteration takes about `2.2 ms` versus
     `13.1 ms` for dense direct CUDA. The phase-gradient relative L2 is about
     `5.6e-7`, and one separable phase-gradient step reduces the dense-reference
     loss by about `15%`. With a non-axisymmetric `m=3` ROI weighting, the same
     row gives about `6.5x` iteration speedup and phase-gradient L2 about
     `7.0e-7`.
   - This supports a GPU hot-loop story for repeated mask/coherent-mode
     evaluation and inverse-design screening on cylindrical or ROI-scored
     targets. It is not yet a matched
     dense-Cartesian GPU PSF-generator accuracy comparison.

## First Prototype Scope

Start scalar, then vectorial.

### Phase 1: scalar Debye-Wolf harmonic prototype

Implement:

```text
P(theta, phi)
  -> FFT_phi Phat(theta, h)
  -> contract with J_h(k rho sin theta) exp(i k z cos theta)
  -> IFFT_h E(rho, psi, z)
```

Validate against direct quadrature for:

- clear circular pupil;
- vortex phase mask `exp(i l phi)`;
- astigmatic or coma-like phase mask;
- random low-order Zernike aberration.

### Phase 2: vectorial weights

Add three field components:

```text
E_x, E_y, E_z
```

by folding the Richards-Wolf vectorial polarization/apodization factors into
`W_c(theta, phi)`. The algorithmic structure is unchanged; the field component
is just another channel.

### Phase 3: repeated-loop benchmark

Benchmark two modes:

- cold solve: first pupil mask and field output;
- hot loop: many pupil phase masks or aberration updates on the same
  `(theta, rho, z, h)` geometry.

The hot loop is important because PSF engineering and adaptive optics repeatedly
evaluate related pupil functions on the same optical system.

## Benchmark Variables

Use physically meaningful variables, not just array sizes.

Optical parameters:

- wavelength `lambda`;
- refractive index `n`;
- numerical aperture `NA`;
- `theta_max = asin(NA / n)`;
- pupil angular resolution `Ntheta`, `Nphi`.

Focal-region parameters:

- `rho_max` in units of wavelength;
- `z_range` in units of wavelength;
- `drho`, `dz`;
- `Npsi`, selected from harmonic support rather than a fixed arbitrary number;
- number of pupil masks or optimization iterations.

Important scaling sweeps:

1. Fixed `drho`, `dz`, extend `rho_max`.

   This is analogous to fixed-`dq` WAXS scaling: the output region grows and the
   angular harmonic support grows with `k rho NA/n`.

2. Fixed focal region, increase `NA`.

   This increases `sin(theta_max)` and therefore harmonic support.

3. Fixed geometry, increase number of pupil masks.

   This tests whether precomputed Bessel/defocus kernels and FFT-friendly
   harmonic grids give real amortized benefit.

4. Pupil bandwidth sweep.

   Compare low-order Zernike/vortex masks against high-frequency random SLM
   masks. The method should win most clearly when the pupil's angular harmonic
   content is structured or when the target region does not require all modes.

## Expected Win/Loss Regions

Likely win:

- non-axisymmetric but not fully random pupil fields;
- repeated forward evaluations over the same objective geometry;
- cylindrical focal grids or ring/ROI scoring functions;
- vectorial focal-field libraries where `E_x, E_y, E_z` share geometry;
- inverse design or adaptive optics where many masks are tested.

Uncertain:

- one-shot dense Cartesian 3D focal-volume output;
- very high-bandwidth random SLM masks;
- GPU-optimized FFT-Debye baselines.

Likely weak:

- axisymmetric Airy focus;
- scalar paraxial propagation;
- problems where a single 2D FFT already gives the desired output.

## Minimal Success Criterion

The high-NA demo is worth keeping only if it passes this gate:

```text
On a non-axisymmetric high-NA pupil benchmark, the harmonic Debye-Wolf solver
matches direct quadrature to controlled error and shows increasing advantage
over direct/NUFFT as rho_max, NA, or repeated mask count grows.
```

For the broader paper strategy, the cross-field benchmark needs to show the
same qualitative pattern as WAXS:

- structured curved-manifold geometry;
- explicit harmonic cutoff;
- controlled accuracy;
- fixed-spacing or fixed-error scaling;
- fair direct/NUFFT/domain-specific baseline.

## Source Notes

Useful starting references:

- Richards and Wolf vectorial focusing / Debye-Wolf integral as the physical
  forward model.
- Lin et al., Optics Express 2012, fast vectorial calculation of volumetric
  focused field.
- Caprile et al., PyFocus, arbitrary phase-mask vectorial high-NA field
  calculation and toroidal foci.
- Vishniakou and Seelig, differentiable optimization of the Debye-Wolf integral
  for light shaping and adaptive optics in two-photon microscopy.
