# Curved Ewald factorization prior-art matrix definition

Generated: 2026-06-22

## Purpose

This matrix is designed to answer a narrow novelty question:

> Is there already an algorithm that uses the same computational structure as our prepared harmonic operator for repeated Fourier evaluation on rotationally structured curved Ewald spheres/caps?

The matrix should not merely group papers by broad mathematical family. It should distinguish:

- classical harmonic/Fourier-Bessel identities,
- high-NA circular-harmonic reductions,
- Ewald/Fourier-diffraction geometry baselines,
- generic NUFFT/gridding baselines,
- adjacent prepared-operator methods,
- and exact or near-exact collisions with the proposed WAXS/ODT operatorization.

## Same-algorithm test

A prior work is considered `same` only if it satisfies most of the following simultaneously.

| Axis | Required signal |
|---|---|
| Physical operator | Evaluates Fourier/scattering/propagation samples on a curved shell, cap, cone, or rotational manifold rather than only a generic Cartesian grid. |
| Phase separation | Uses the structure `q_perp R cos(phi_q - beta) + q_z z` or an equivalent rotational phase split. |
| Harmonic machinery | Uses Jacobi-Anger, circular/spherical harmonics, Fourier-Bessel, Hankel, or equivalent plane-wave expansion as the actual computational path. |
| Angular transform | Uses angular Fourier coefficients, angular FFT, or explicit harmonic-mode coefficients rather than only direct quadrature over the angle. |
| Radial/axial kernel | Reuses Bessel/Hankel/radial and axial phase factors across many targets or updates. |
| Prepared reuse | Separates build/plan cost from hot repeated forward or adjoint execution. |
| Adjoint/inverse workload | Supports repeated adjoint/backpropagation, reconstruction, optimization, or residual updates, not only one-shot forward evaluation. |
| Ewald/cap geometry | Explicitly handles Ewald sphere/cap/diffraction-tomography geometry, not only generic polar transforms. |
| ODT h/l structure | For ODT-like cases, separates detector-cap harmonic `h` and illumination/source harmonic `l` or an equivalent two-angle factorization. |

## Similarity classes

| Class | Meaning | Novelty risk |
|---|---|---|
| S0 exact | Same physical operator, same harmonic factorization, prepared forward/adjoint reuse, same Ewald/cap geometry. | Critical |
| S1 near-same | Same harmonic reduction in a close domain, but missing prepared adjoint, Ewald/cap, or h/l geometry. | High for that domain |
| A1 adjacent operator | Similar prepared-operator or harmonic acceleration, but different physics or manifold. | Medium |
| B1 baseline geometry | Same physical geometry or inverse problem, but solved by FDT, gridding, NUFFT, interpolation, or filtered backpropagation rather than our factorization. | Medium as baseline |
| C1 mathematical ancestor | Plane-wave expansion, Fourier-Bessel/Hankel, spherical harmonics, or cylindrical harmonics as a known identity/tool. | Low, cite for honesty |
| D1 unrelated | Same keywords but different operator/workload. | Low |

## Matrix columns

Use these columns for the full prior-art table.

| Column | Meaning |
|---|---|
| ID | Short citation key. |
| Domain | High-NA, WAXS/SAXS/FXS, ODT/FPDT/OCT, cryo-EM, radio interferometry, photoacoustic/acoustic, SAR/radar, MRI, numerical transform, etc. |
| Source URL / DOI | Stable source link. |
| Physical problem | What operator or inverse problem is being solved. |
| Measurement manifold | Cartesian, polar, cylindrical, spherical, Ewald sphere, Ewald cap, detector ring, focal cylinder, etc. |
| Uses Ewald/cap? | Yes/no/partial. |
| Uses harmonic factorization? | None, circular, spherical, Fourier-Bessel, Hankel, plane-wave expansion, etc. |
| Uses angular FFT/coefficient extraction? | Yes/no/unclear. |
| Reuses radial/axial kernels? | Yes/no/unclear. |
| Prepared build/hot split? | Yes/no/unclear. |
| Forward, adjoint, or both? | Forward only, adjoint only, forward/adjoint, reconstruction. |
| Repeated workload? | One-shot, focal scan, many masks, iterative reconstruction, streaming update, etc. |
| ODT h/l split? | Yes/no/not applicable. |
| Baseline compared | Direct quadrature, NUFFT, package, filtered backprojection, gridding, etc. |
| Similarity class | S0, S1, A1, B1, C1, D1. |
| Novelty threat | Critical/high/medium/low. |
| Positioning sentence | How to cite it in the manuscript. |
| Verification status | Seed only, abstract checked, PDF checked, implementation checked. |

## Seed matrix after broad first-pass search

| ID | Domain | Source | Physical problem / manifold | Harmonic or operator structure | Similarity | Threat | Positioning |
|---|---|---|---|---|---|---|---|
| Boichenko 2022/2023 | High-NA optics | https://arxiv.org/abs/2212.10978 | Richards-Wolf tight focusing, focal cylindrical coordinates | Circularly polarized vortex-mode expansion; focal field factorized into `rho,z` and azimuth; single-integral reduction vs direct double integral | S1 | High for High-NA only | Must cite as known high-NA circular-harmonic/Fourier-Bessel reduction. Our High-NA role should be recovery/practical benchmark, not mathematical novelty. |
| Panova et al. 2020 | High-intensity focusing | https://arxiv.org/abs/2010.00409 | Tight focusing of short pulses via mapping curved far field to periodic space | Spectral solver after mapping; not the same circular-harmonic Ewald operator | A1/D1 | Low-medium | Adjacent example of exploiting focusing geometry for computation, not same factorization. |
| Kirisits/Naujoks/Scherzer 2024 | Diffraction tomography | https://arxiv.org/abs/2403.16835 | Generalized incident field diffraction tomography | Generalized Fourier diffraction theorem for customized/focused incident fields | B1 | Medium | Important ODT theory baseline; not evidence of our prepared h/l harmonic operator unless later details show same computation. |
| Kirisits/Quellmalz/Setterqvist 2024 | Diffraction tomography | https://arxiv.org/abs/2407.01793 | Generalized FDT and filtered backpropagation, Ewald/Fourier coverage | Ewald/cap geometry and explicit reconstruction formula; not circular-harmonic prepared operator | B1 | Medium | Must cite as FDT/backpropagation baseline. Our novelty is geometry-aware operatorization for repeated updates. |
| Zuo et al. 2019 | FPDT / ODT | https://arxiv.org/abs/1904.09386 | Fourier ptychographic diffraction tomography, 3D RI reconstruction | Iterative stitching of variably illuminated images in 3D Fourier space | B1 | Medium | Experimental/application baseline for high-throughput ODT/FPDT geometry. |
| Horstmeyer & Yang 2015 | FPDT | https://arxiv.org/abs/1510.08756 | Fourier ptychographic diffraction tomography | Illumination-angle synthetic aperture / iterative reconstruction | B1 | Medium | Baseline for FPDT geometry and reconstruction, not our h/l harmonic plan. |
| Zhou et al. 2020 | OCT / diffraction tomography | https://arxiv.org/abs/2012.04875 | Unified k-space/OCT theory with Ewald sphere | Fourier diffraction theorem framework across OCT/ODT variants | B1 | Medium | Useful for broader wave-imaging context; not same prepared operator. |
| Chen et al. 2021 | Cryo-EM | https://arxiv.org/abs/2101.11709 | Curved Ewald sphere correction in electron microscopy | Curved-Ewald-aware reconstruction, but not our cylindrical harmonic operator | B1/A1 | Medium | Shows curved Ewald matters in another field; not same algorithm. |
| Yeo et al. 2023 | Cryo-EM | https://arxiv.org/abs/2312.08965 | Phase retrieval diffraction tomography with Ewald curvature | Curved-Ewald reconstruction/phase retrieval | B1 | Medium | Baseline for Ewald-curvature reconstruction, different inverse algorithm. |
| Wang/Shkolnisky/Singer 2013 | Cryo-EM | https://arxiv.org/abs/1307.5824 | Iterative 3D reconstruction from projection images | Precomputed Toeplitz `A*A` operator using NUFFT; fast iterative reconstruction | A1 | Medium-high for prepared-operator framing | Very important adjacent prepared-operator prior art: same build/hot philosophy, different projection geometry and no h/l Ewald harmonic split. |
| Marshall/Mickelin/Shi/Singer 2022 | Cryo-EM | https://arxiv.org/abs/2210.17501 | Fast PCA for cryo-EM images | Fourier-Bessel basis on disk; acceleration of covariance/PCA | A1/C1 | Medium | Adjacent Fourier-Bessel computational basis, not curved Ewald operator. |
| Zhao et al. 2014 | Cryo-EM image analysis | https://arxiv.org/abs/1412.0781 | Steerable PCA / Fourier-Bessel basis | Fourier-Bessel expansion and NUFFT for disk images | A1/C1 | Medium | Cite as Fourier-Bessel image-analysis basis; not same physical operator. |
| Flamant et al. 2016 | XFEL single-particle imaging | https://arxiv.org/abs/1602.01301 | Orientation recovery / shell-by-shell reconstruction | Harmonic analysis on the sphere separates angular and radial degrees | A1/C1 | Medium | Adjacent X-ray harmonic analysis; not WAXS cake-map forward operator. |
| FXS / Kam line | X-ray scattering | https://en.wikipedia.org/wiki/Fluctuation_X-ray_scattering | Angular correlations and spherical harmonics in scattering | Spherical harmonics, Legendre transform, Hankel relation for correlation data | C1/A1 | Low-medium | Important scattering harmonic ancestor; not same prepared WAXS Ewald cake-map evaluation. Need primary references for final manuscript. |
| Zhang et al. 2022 | Anisotropic X-ray PDF | https://arxiv.org/abs/2205.05865 | 2D diffraction pattern to anisotropic PDF | Spherical harmonics method based on 3D diffraction geometry and Fourier transform | A1 | Medium | Adjacent X-ray harmonic/PDF processing; check PDF for exact computational overlap. |
| Pratley et al. 2018/2019 | Radio interferometry | https://arxiv.org/abs/1807.09239 | Wide-field non-coplanar imaging | Radially symmetric w-projection kernel via Hankel transform; distributed w-stacking/projection | A1 | Medium-high as adjacent prepared kernel | Strong adjacent example: radial/Hankel kernel generation for curved/wide-field Fourier correction, but not Ewald scattering or h/l ODT. |
| Merry 2015 | Radio interferometry | https://arxiv.org/abs/1511.07152 | W-projection approximation | Separable approximation to w-projection kernels | A1 | Medium | Adjacent separability/low-rank kernel prior art. |
| Lucas/Skipper/Scaife 2019 | Radio interferometry | https://arxiv.org/abs/1904.08463 | Fast w-projection | Replaces 2D FFT kernel generation with 1D Hankel transform | A1 | Medium | Adjacent Hankel acceleration; useful for claim boundary. |
| Averbuch/Shabat/Shkolnisky 2015 | Numerical transform / tomography | https://arxiv.org/abs/1507.06174 | 3D pseudo-polar Fourier transform inversion | Specialized near-polar Fourier grid with direct inversion via 1D resampling | A1/B1 | Medium | Shows specialized Fourier grids can beat generic nonuniform methods; not same curved Ewald harmonic operator. |
| Zhou/Grisouard 2022 | Numerical PDE / polar spectral solver | https://arxiv.org/abs/2210.09736 | Polar-coordinate spectral solver | FFT in azimuth plus discrete Hankel transform per angular mode | C1/A1 | Medium | Good mathematical/computational ancestor for angular FFT + Hankel structure. Not Ewald/cap measurement. |
| Wang/Anastasio 2012 | Photoacoustic tomography | https://arxiv.org/abs/1208.2262 | Circular/spherical measurement geometry PACT | Fourier-domain exact reconstruction; two orders faster than filtered backprojection | A1/B1 | Medium | Adjacent circular/spherical measurement reconstruction; not Ewald scattering. |
| Kunyansky 2011 | Thermoacoustic tomography | https://arxiv.org/abs/1102.1413 | Circular/spherical/cylindrical acquisition geometries | Fast algorithms exploiting geometry, O(n^3 log n)-type costs | A1/B1 | Medium | Adjacent geometry-aware tomography acceleration. |
| Haltmeier/Scherzer/Zangerl 2008 | Photoacoustic tomography | https://arxiv.org/abs/0808.3510 | Photoacoustic Fourier reconstruction | NUFFT avoids Fourier interpolation artifacts | B1 | Low-medium | Generic NUFFT-type tomography baseline, not harmonic prepared operator. |
| Rafaely 2023 | Acoustics | https://arxiv.org/abs/2310.04169 | Spherical microphone array beamforming | Spherical harmonics-domain beamforming | C1/A1 | Low-medium | Adjacent wave-field harmonic processing. |
| Nguyen et al. 2023 | Acousto-optic sound-field reconstruction | https://arxiv.org/abs/2311.01715 | Concentric circle sampling / exterior sound field | Circular harmonic extension | A1 | Medium | Adjacent circular-harmonic reconstruction from circular sampling. |
| Conway/Cohl 2009 | Helmholtz theory | https://arxiv.org/abs/0910.1193 | Helmholtz Green function in cylindrical coordinates | Exact Fourier expansion in cylindrical coordinates | C1 | Low | Mathematical ancestor for cylindrical harmonic Green functions. |
| Plane-wave expansion / Rayleigh expansion | General wave physics | https://en.wikipedia.org/wiki/Plane-wave_expansion | Plane wave as spherical harmonics and Bessel functions | Classical expansion identity | C1 | Low | Cite only as classical mathematical background if needed. |
| Hankel / Fourier-Bessel transform | General math/numerics | https://en.wikipedia.org/wiki/Hankel_transform | Fourier transform under radial/cylindrical symmetry | Hankel transform and angular harmonic decomposition | C1 | Low | Mathematical ancestor; not a novelty threat by itself. |
| MRI NUFFT operator/Jacobian line | MRI | https://arxiv.org/abs/2111.02912 | Non-Cartesian MRI sampling optimization | NUFFT operator/Jacobian efficiency for iterative recon | B1/A1 | Low-medium | Generic NUFFT/operator baseline for inverse problems; not curved Ewald harmonic geometry. |
| SAR polar format / Stolt line | SAR/radar | https://arxiv.org/abs/2503.07889 | Spotlight SAR polar-format geometry mapping | Polar-format processing; geometry transforms | B1/A1 | Low-medium | Search space to expand; current seed is geometry mapping, not same harmonic operator. |

## Search expansion plan

The broad review should run in layers.

### Layer 1: direct collision search

Search terms:

- `"Ewald" "Fourier-Bessel" "circular harmonics"`
- `"Ewald sphere" "Hankel transform" reconstruction`
- `"Fourier diffraction theorem" "cylindrical harmonics"`
- `"optical diffraction tomography" "Fourier-Bessel"`
- `"diffraction tomography" "circular harmonics"`
- `"Ewald cap" "harmonic" "adjoint"`
- `"cone beam" "Fourier-Bessel" "diffraction tomography"`

Goal: find S0 or S1 collisions.

### Layer 2: ODT / FPDT / OCT baselines

Search terms:

- `"generalized Fourier diffraction theorem" "focused beam"`
- `"filtered backpropagation" "diffraction tomography"`
- `"Fourier ptychographic diffraction tomography" "Ewald sphere"`
- `"ODT" "NUFFT" "Ewald"`
- `"optical coherence tomography" "Ewald sphere" "k-space"`
- `"Bessel beam" "diffraction tomography"`

Goal: define B1 baselines and experimental geometry relevance.

### Layer 3: scattering harmonic ancestors

Search terms:

- `"fluctuation X-ray scattering" "spherical harmonics" "Bessel"`
- `"wide-angle X-ray scattering" "spherical harmonics"`
- `"anisotropic pair distribution function" "spherical harmonics" Fourier transform`
- `"Debye scattering equation" "spherical harmonics"`
- `"X-ray scattering" "Fourier-Bessel" "algorithm"`

Goal: ensure WAXS novelty is not contradicted by existing scattering-specific harmonic solvers.

### Layer 4: adjacent prepared-operator methods

Search terms:

- `"precomputed operator" "NUFFT" "iterative reconstruction"`
- `"Toeplitz" "NUFFT" "adjoint" "reconstruction"`
- `"w-projection" "Hankel transform" "kernel generation"`
- `"radio interferometry" "radially symmetric" "Hankel" "projection"`
- `"photoacoustic tomography" "circular" "Fourier transform" reconstruction`

Goal: position prepared build/hot split as a known computational design pattern, while keeping Ewald h/l factorization as new.

### Layer 5: numerical transform family

Search terms:

- `"fast polar Fourier transform" "Bessel"`
- `"pseudo-polar Fourier transform" direct inversion`
- `"polar Fourier transform" "angular FFT" "Hankel"`
- `"discrete Hankel transform" "FFT in azimuth"`
- `"fast Fourier-Bessel transform" "angular"`

Goal: cite mathematical/numerical ancestors without letting them blur the physical claim.

## Working conclusion after first broad pass

No S0 exact collision has been identified in this first pass.

The strongest novelty threats are:

1. Boichenko for High-NA circular-harmonic Debye/Richards-Wolf reduction. This is S1 and must be cited prominently.
2. ODT/FDT literature for Ewald/cap reconstruction geometry. These are B1 baselines and must be cited as the physics/inverse-problem baseline.
3. Radio-interferometry w-projection/Hankel work and cryo-EM Toeplitz/NUFFT reconstruction as adjacent prepared-operator examples. These are A1 and should be used to show the broader computational lineage.

The safest current novelty statement is:

> Circular/spherical harmonic and Fourier-Bessel factorizations are classical, and closely related reductions are known in high-NA focusing and other wave-imaging settings. The new contribution is the prepared forward/adjoint operatorization of rotationally structured curved Ewald/cap Fourier evaluation, with WAXS as a direct validation case and ODT as an extended detector-illumination h/l factorization for repeated inverse/backpropagation workloads.
