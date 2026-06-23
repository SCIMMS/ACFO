# Curved Ewald factorization prior-art check - Layer 5 numerical transform family

Generated: 2026-06-22

## Question

Layer 5 checks whether the mathematical transform ingredients are already known
outside the physical WAXS/ODT settings.

This layer does not threaten the physics/application claim by itself. It sets
the correct citation boundary for the transform machinery.

The narrow question is:

> Are FFT-in-angle, Hankel/Fourier-Bessel transforms, pseudo-polar Fourier
> grids, and direct non-Cartesian Fourier inversions established numerical
> tools?

The answer is yes. Therefore, the manuscript should not claim a new transform.

## Checked primary sources

Local text was extracted from primary-source PDFs under
`benchmark_results/prior_art_pdfs/` into `benchmark_results/prior_art_text/`.

| ID | Source | Local PDF | Verification status |
|---|---|---|---|
| Averbuch, Shabat, Shkolnisky 2015, direct inversion of 3D PPFT | https://arxiv.org/abs/1507.06174 | `prior_art_pdfs/averbuch_2015_direct_inversion_3d_pseudo_polar_fourier.pdf` | PDF text checked |
| Zhou and Grisouard 2022/2023, polar spectral solver with DHT | https://arxiv.org/abs/2210.09736 | `prior_art_pdfs/zhou_grisouard_2022_polar_spectral_dht.pdf` | PDF text checked |
| Kutyniok, Shahram, Zhuang 2011, ShearLab / pseudo-polar Fourier | https://arxiv.org/abs/1106.1319 | `prior_art_pdfs/kutyniok_2011_shearlab_pseudopolar_fourier.pdf` | PDF text checked |
| Teyfouri et al. 2019, CBCT PPFT / Grangeat | https://arxiv.org/abs/1906.06472 | `prior_art_pdfs/teyfouri_2019_cbct_pseudopolar_grangeat.pdf` | PDF text checked |

## Matrix update

| ID | Domain | What it does | Overlap with our method | Missing relative to S0 | Class | Threat | Manuscript positioning |
|---|---|---|---|---|---|---|---|
| Averbuch/Shabat/Shkolnisky 2015 PPFT | Numerical transform / tomography | Evaluates Fourier transform on a pseudo-polar grid and provides fast direct inversion using 1D resampling. | Strong precedent that specialized non-Cartesian Fourier grids can be more invertible/friendly than generic NUFFT grids. | Near-polar grid, not curved Ewald/cap detector manifold; no scattering/ODT h/l harmonic operator. | A1/C1 | Medium | Cite to support the broader point that geometry-friendly Fourier grids can outperform generic nonuniform sampling. |
| Zhou/Grisouard 2022/2023 DHT polar spectral solver | Numerical PDE / polar coordinates | Uses FFTs in azimuth to isolate angular modes and DHT/Fourier-Bessel transforms radially. | Direct mathematical ancestor of angular FFT + radial Bessel/Hankel mode separation. | Solves polar-coordinate PDEs, not Ewald/cap Fourier measurement; no detector build/hot operator or inverse-imaging workload. | C1/A1 | Medium | Cite as transform ancestry for FFT-in-angle plus Hankel/DHT decomposition. Do not claim this decomposition is mathematically new. |
| Kutyniok/Shahram/Zhuang 2011 ShearLab / PPFT | Numerical harmonic analysis | Uses pseudo-polar grids and weighted PPFT so that adjoints/inverses are usable for digital shearlets. | Precedent for designing digital transforms around friendly non-Cartesian grids and adjoint usability. | Directional representation system, not scattering or ODT measurement operator. | A1/C1 | Medium | Useful for claim boundary around pseudo-polar/friendly-grid transforms and adjoint design. |
| Teyfouri et al. 2019 CBCT PPFT / Grangeat | CT reconstruction | Uses pseudo-polar Fourier/Radon machinery to accelerate cone-beam CT reconstruction. | Application precedent for using PPFT-like friendly grids in a physical reconstruction problem. | CT/Radon geometry, not curved Ewald scattering; no WAXS/ODT harmonic split. | B1/A1 | Low-medium | Optional citation if discussing friendly-grid reconstruction beyond wave scattering. |

## Layer 5 conclusion

No S0 exact collision was found.

However, Layer 5 confirms that these pieces are established:

- FFT along angular/azimuthal direction,
- Fourier-Bessel / discrete Hankel transform per angular mode,
- pseudo-polar or near-polar Fourier grids,
- direct inversion/resampling on special non-Cartesian grids,
- using adjoints/inverses enabled by weighted grid design.

Therefore the paper should not claim:

- a new Fourier-Bessel transform,
- a new Hankel transform,
- a new angular-mode decomposition,
- or a general replacement for NUFFT.

The safe claim is:

> We adapt classical angular-harmonic and Fourier-Bessel/Hankel machinery to a
> prepared curved-Ewald/cap measurement operator. The novelty is not the
> transform identity itself, but the way it is packaged for repeated WAXS
> detector/cake-map and ODT forward/adjoint workloads.

## Relation to the WAXS/ODT algorithm

Layer 5 also clarifies why the algorithm can beat generic NUFFT in certain
regimes:

- NUFFT is general and low-level optimized, but it treats the target/sample set
  as generic nonuniform points.
- Friendly-grid methods win when the nonuniform set has exploitable structure.
- Our structure is not pseudo-polar; it is curved Ewald/cap/cylindrical and, in
  ODT, detector/illumination factored.
- Thus the comparison should be framed as "structured-manifold specialization"
  rather than "NUFFT is slow".

## Final prior-art stance after Layers 1-5

The broad prior-art check supports the following hierarchy:

| Claim component | Novel? | Why |
|---|---|---|
| Circular/spherical harmonic identities | No | Classical and used in high-NA, scattering, XFEL, photoacoustics, polar solvers. |
| Fourier-Bessel/Hankel radial transforms | No | Standard numerical transform family. |
| Prepared build/hot operator reuse | No | Known in cryo-EM FIRM, radio w-projection, NUFFT/NFFT inverse problems, GPU operator frameworks. |
| Ewald/FDT geometry in ODT/FPDT/OCT | No | Established by Fourier diffraction theorem and modern generalized/raster-scan DT work. |
| WAXS detector cake-map cylindrical phase split as a prepared operator | Not found in checked prior art | This remains a plausible novelty axis. |
| ODT detector/illumination h/l factorization as prepared repeated adjoint | Not found in checked prior art | This remains a plausible novelty axis, but must be compared to NUFFT/NFFT and FDT/backprop baselines. |
| General curved Ewald/cap Fourier operatorization across WAXS and ODT | Plausible main novelty | Supported if benchmarks show real speed/accuracy/memory benefit in experimentally meaningful repeated workloads. |

## Recommended citation framing

Use three citation tiers:

1. Mathematical ancestors:
   angular harmonics, Fourier-Bessel/Hankel, pseudo-polar/friendly grids.
2. Domain precedents:
   high-NA Boichenko-style circular harmonic focusing, ODT/FDT/backprop,
   WAXS/SAXS Debye and anisotropic PDF harmonic methods.
3. Adjacent computational precedents:
   cryo-EM Toeplitz/NUFFT, radio w-projection/Hankel, photoacoustic Fourier
   reconstructions, GPU operator frameworks.

Then state the contribution narrowly:

> We provide a prepared operator for rotationally structured curved Ewald/cap
> Fourier evaluation, with WAXS validating the cylindrical detector/cake-map
> specialization and ODT demonstrating repeated forward/adjoint impact in a
> physically meaningful reconstruction workload.
