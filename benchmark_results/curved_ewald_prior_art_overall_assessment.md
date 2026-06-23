# Curved Ewald factorization prior-art overall assessment

Generated: 2026-06-22

## Executive conclusion

Across the checked literature layers, no exact S0 collision was found.

The evidence supports a narrow but defensible novelty claim:

> The novelty is not the circular/spherical harmonic identity, Fourier-Bessel or
> Hankel transform, Fourier diffraction theorem, or generic prepared-operator
> idea. The plausible novelty is the prepared forward/adjoint operatorization of
> rotationally structured curved Ewald/cap Fourier evaluation, specialized to
> WAXS detector cake maps and extended to ODT detector/illumination repeated
> workloads.

## Direct answers to the two key questions

### 1. Is this exact factorization already used elsewhere?

Not found in the checked prior art.

Many adjacent structures exist:

- High-NA circular-harmonic focusing reduction exists and must be cited.
- Scattering uses spherical harmonics, Rayleigh expansions, spherical Bessel
  functions, and Debye histogram acceleration.
- ODT/FPDT/OCT already uses Ewald/FDT/backpropagation geometry.
- Cryo-EM, radio interferometry, and photoacoustics already use prepared
  kernels/operators and geometry-aware Fourier methods.
- Numerical analysis already has FFT-in-angle, Hankel/DHT, pseudo-polar
  Fourier, and friendly-grid transforms.

But the specific combination below was not found:

- curved Ewald/cap or WAXS detector/cake measurement,
- cylindrical `q_perp R cos(phi_q - beta) + q_z z` phase split,
- angular harmonic/FFT reuse,
- radial/axial kernel reuse,
- explicit prepared build/hot split,
- repeated forward/adjoint evaluation,
- and ODT detector/illumination `h/l`-type geometry exploitation.

### 2. Does a meaningful ODT structure exist for this algorithm?

Yes.

Layer 2 found strong ODT/FPDT/OCT structures where repeated or geometry-fixed
operators are meaningful:

- generalized incident-field DT and focused-beam DT,
- raster-scan DT with translated focused beams,
- FPDT with repeated angled illumination and Ewald cap filling,
- Fourier-synthesis ODT for high-speed/multiplexed volumetric imaging,
- generalized FDT / filtered-backpropagation work that already uses NDFT/NFFT
  style baselines.

Therefore the ODT application should be framed as:

> We accelerate repeated forward/adjoint evaluation for already meaningful ODT
> measurement geometries, rather than introducing a new ODT Fourier diffraction
> theorem.

## Layer-by-layer result

| Layer | Purpose | Result | Novelty risk |
|---|---|---|---|
| Layer 1 direct collision | Search exact Ewald/Fourier-Bessel/circular-harmonic collision | No S0 found. Boichenko is S1 for High-NA. | High only for High-NA novelty; low for WAXS/ODT exact collision. |
| Layer 2 ODT/FPDT/OCT | Check FDT, raster/focused beam, FPDT, OCT baselines | Strong B1 baselines; no same harmonic prepared operator. | Medium-high as baseline requirement. |
| Layer 3 scattering harmonics | Check WAXS/SAXS/FXS/Debye/aniso-PDF harmonic ancestors | Spherical harmonic and Debye accelerations exist; no cylindrical cake-map prepared operator found. | Medium-high for overbroad WAXS claims. |
| Layer 4 prepared operators | Check cryo-EM/radio/photoacoustic/GPU operator precedents | Prepared build/hot reuse is not new. | High for generic prepared-operator claim. |
| Layer 5 numerical transforms | Check PPFT, DHT, Fourier-Bessel/Hankel transform families | Transform ingredients are established. | High for transform novelty, low for physical operator novelty. |

## Strongest prior-art anchors

| Anchor | Why it matters | How to use it |
|---|---|---|
| Boichenko high-NA circular-harmonic focusing | Closest high-NA reduction; prevents claiming High-NA math novelty. | Use High-NA as bridge/benchmark/practical implementation, not main novelty. |
| Kirisits / Elbau ODT FDT and raster-scan DT | Shows focused/raster ODT geometries are real and active. | Cite as physics baseline; claim computational operatorization. |
| Deumer/CDEF/DEBYER | Pair-distance histogram acceleration is an established Debye baseline. | Compare or discuss for WAXS/SAXS curve workloads; distinguish detector-resolved cake maps. |
| Zhang anisotropic X-ray PDF | Spherical-harmonic X-ray scattering/PDF precedent. | Cite as scattering harmonic ancestor. |
| Wang/Shkolnisky/Singer cryo-EM FIRM | Prepared Toeplitz/NUFFT kernel reuse for iterative reconstruction. | Cite as strongest generic prepared-operator precedent. |
| Pratley/Lucas/Merry radio w-projection | Radial/Hankel/separable kernels for curved Fourier correction. | Cite as adjacent geometry-aware Fourier-kernel acceleration. |
| Zhou/Grisouard DHT polar solver | FFT in angle plus DHT/Fourier-Bessel radial transform. | Cite as numerical transform ancestry. |
| Averbuch PPFT | Friendly non-Cartesian grid can beat generic inversion. | Cite for geometry-friendly Fourier sampling philosophy. |

## Claim boundary for manuscript

Avoid:

- "We discovered a new circular harmonic identity."
- "We introduce Fourier-Bessel/Hankel factorization."
- "We replace NUFFT in general."
- "We introduce a new Fourier diffraction theorem for ODT."
- "We are first to use prepared operators in iterative reconstruction."

Use:

> Classical circular/spherical harmonic and Fourier-Bessel/Hankel decompositions
> are adapted here into a prepared forward/adjoint operator for rotationally
> structured curved Ewald/cap Fourier evaluation. WAXS validates the cylindrical
> detector/cake-map specialization, while ODT demonstrates the payoff of the
> extended detector/illumination factorization in repeated reconstruction and
> backpropagation workloads.

## Benchmark requirements implied by the prior art

The paper needs to separate these timings:

- build/preparation time,
- hot forward time,
- hot adjoint/backprop time,
- amortized time after repeated evaluations,
- memory footprint,
- accuracy versus direct/reference,
- accuracy versus NUFFT/NFFT/cuFINUFFT,
- and sensitivity to detector/cap/grid geometry.

The most important comparisons are:

| Application | Required baseline |
|---|---|
| WAXS | Direct detector evaluation, FINUFFT/NUFFT, and Debye/DEBYER-like pair-distance or curve baselines where applicable. |
| High-NA | Debye-Wolf/Richards-Wolf, PyFocus/vectorial focal-field package, Boichenko-style circular-harmonic reduction if implementable. |
| ODT | Direct/reference, NUFFT/NFFT/cuFINUFFT adjoint, FDT/backprop/gridding-style baseline, and repeated reconstruction timing. |

## Recommended paper story

The strongest NCS-style story is:

1. General method:
   prepared curved Ewald/cap Fourier operator for rotationally structured
   detector/source geometries.
2. WAXS validation:
   correctness and scaling on detector-resolved cake maps, where the cylindrical
   phase split gives a direct validation case.
3. High-NA bridge:
   recover a known circular-harmonic reduction and compare practical
   implementation/performance against existing optics tools.
4. ODT impact:
   repeated forward/adjoint/backpropagation or reconstruction workload, showing
   amortized speed and GPU relevance in experimentally meaningful geometries.

High-NA should not carry the novelty claim. It should connect the method to a
known and important wave-optics reduction.

## Remaining due diligence

Before manuscript submission, still check package-specific primary sources:

- CRYSOL, WAXSiS, FoXS, Pepsi-SAXS,
- DEBUSSY and other Debye-equation packages,
- FXS/Kam original papers if the harmonic-scattering lineage is discussed,
- GISAXS/GIWAXS detector-pattern simulators if surface scattering enters scope,
- and any package that computes full 2D WAXS detector patterns rather than only
  1D orientationally averaged curves.

These are unlikely to overturn the current novelty boundary, but they are needed
for a complete reviewer-proof related-work section.

## Files generated in this check

- `benchmark_results/curved_ewald_prior_art_layer1_direct_collision_check.md`
- `benchmark_results/curved_ewald_prior_art_layer2_odt_fpdt_oct_check.md`
- `benchmark_results/curved_ewald_prior_art_layer3_scattering_harmonic_check.md`
- `benchmark_results/curved_ewald_prior_art_layer4_adjacent_prepared_operator_check.md`
- `benchmark_results/curved_ewald_prior_art_layer5_numerical_transform_check.md`
- `benchmark_results/curved_ewald_prior_art_overall_assessment.md`
