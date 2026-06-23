# Curved Ewald factorization prior-art check: Layer 1 direct collision

Generated: 2026-06-22

## Question

Layer 1 checks whether there is already a direct collision with the proposed method:

> prepared harmonic forward/adjoint operator for repeated Fourier evaluation on rotationally structured curved Ewald spheres/caps, including WAXS-like `q_perp R cos(phi_q - beta) + q_z z` separation or ODT-like detector/illumination `h/l` factorization.

## Search terms used

Exact phrase searches:

- `"Ewald" "Fourier-Bessel" "circular harmonics"`
- `"Ewald sphere" "Hankel transform" reconstruction`
- `"Fourier diffraction theorem" "cylindrical harmonics"`
- `"optical diffraction tomography" "Fourier-Bessel"`
- `"Jacobi-Anger" "Ewald"`
- `"Jacobi Anger" "diffraction tomography"`
- `"circular harmonic" "Ewald"`
- `"circular harmonics" "optical diffraction tomography"`

Broader searches:

- `Ewald Fourier Bessel circular harmonics diffraction tomography`
- `Ewald sphere Hankel transform reconstruction diffraction tomography`
- `Fourier diffraction theorem cylindrical harmonics tomography`
- `optical diffraction tomography Fourier Bessel`
- `diffraction tomography circular harmonics`
- `Ewald cap harmonic adjoint diffraction tomography`
- `cone beam Fourier Bessel diffraction tomography`
- `Bessel beam Fourier diffraction theorem`

## Layer 1 result

No S0 exact collision was found in this pass.

The closest new hit is not a harmonic/Bessel prepared operator, but it is highly relevant to the ODT story:

- Peter Elbau, Noemi Naujoks, Otmar Scherzer, **Raster Scan Diffraction Tomography**, arXiv:2602.17351, 2026.
- Peter Elbau, Noemi Naujoks, **Invertibility of the Fourier Diffraction Relation in Raster Scan Diffraction Tomography**, arXiv:2602.17344, 2026.

These papers should be added as high-priority B1/A1 ODT baselines because they explicitly address focused-beam scanning diffraction tomography using Herglotz waves and Fourier diffraction relations.

## Checked sources

| Source | What was checked | Evidence for similarity | Missing relative to our S0 test | Classification |
|---|---|---|---|---|
| [Boichenko 2022/2023](https://arxiv.org/abs/2212.10978) | Previously checked; retained in Layer 1 because it is the clearest High-NA harmonic precedent. | Circularly polarized vortex-mode expansion; Richards-Wolf field factorized into `rho,z` and azimuth; direct double integral reduced to single integrals. | Not an Ewald/cap scattering operator; no WAXS/ODT; no detector/illumination `h/l`; not mainly prepared adjoint/reconstruction. | S1 for High-NA, not S0 for paper. |
| [Raster Scan Diffraction Tomography 2026](https://arxiv.org/abs/2602.17351) | Abstract/html checked. | Focused beam modeled as Herglotz wave; active scan geometry; new Fourier diffraction theorem for scanning data; analyzes recoverable Fourier coverage. | No circular/Fourier-Bessel/Jacobi-Anger factorization found; no angular FFT; no prepared radial/axial kernel; no h/l harmonic operator. | B1/A1, high relevance for ODT. |
| [Invertibility of the Fourier Diffraction Relation in Raster Scan DT 2026](https://arxiv.org/abs/2602.17344) | PDF checked for Bessel/harmonic/adjoint terms. | Focused beams translated across object; Fourier diffraction relation relates measurements to Fourier coefficients; reconstructibility of Fourier coefficients analyzed. | `Bessel`, `harmonic`, and `adjoint` terms not present in PDF text search; no prepared harmonic execution path. | B1/A1, high relevance for ODT theory. |
| [Kirisits, Quellmalz, Setterqvist 2024](https://arxiv.org/abs/2407.01793) | PDF checked. | Generalized Fourier diffraction theorem; Fourier coverage for object orientation, incident direction, and frequency; filtered backpropagation. | Uses FDT/backpropagation mapping rather than circular/Fourier-Bessel prepared operator; no h/l split. | B1 baseline geometry. |
| [Kirisits, Naujoks, Scherzer 2024](https://arxiv.org/abs/2403.16835) | Abstract checked. | Generalized incident field; focused beams; new forward model and two-step reconstruction. | No evidence yet of circular harmonic prepared operator; PDF still needs full text check. | B1/A1, high relevance for focused-beam ODT. |
| [Muller, Schurmann, Guck 2015/2016](https://arxiv.org/abs/1507.00466) | PDF checked for Bessel/harmonic/Ewald/backpropagation. | Full ODT theory review; derives 2D/3D backpropagation; implementation uses FFT and rotational projection-wise reconstruction. | No Bessel/harmonic/Ewald terms in text search; not our prepared h/l harmonic operator. | B1 baseline. |
| [Zuo et al. 2019](https://arxiv.org/abs/1904.09386) | Abstract checked. | FPDT with variably illuminated images stitched in 3D Fourier space; high-throughput ODT context. | Iterative FPDT/ptychographic reconstruction, not circular-harmonic Ewald operatorization. | B1 application baseline. |
| [Horstmeyer & Yang 2015](https://arxiv.org/abs/1510.08756) | Abstract checked. | LED-array Fourier ptychographic diffraction tomography; intensity-only 3D RI reconstruction. | No evidence of h/l harmonic split or prepared Bessel operator. | B1 application baseline. |
| [Zhou et al. 2020](https://arxiv.org/abs/2012.04875) | Abstract checked. | Unified OCT/ODT k-space theory; Fourier diffraction theorem relates coherent interaction to Ewald sphere; includes Bessel-beam OCT. | Theory/k-space unification, not prepared circular-harmonic operator. | B1 context baseline. |
| [Fourier Synthesis ODT 2023](https://arxiv.org/abs/2309.16912) | Abstract checked. | High-speed ODT using multiplexed illumination angles and inverse computational strategies; kHz volumetric imaging context. | Does not appear to be harmonic/Ewald prepared operator; should be checked as high-speed ODT competitor later. | B1 impact baseline. |

## Classification update

Recommended additions to the master seed matrix:

| ID | Domain | Similarity | Threat | Positioning |
|---|---|---|---|---|
| Elbau/Naujoks/Scherzer 2026 Raster Scan DT | ODT / ultrasound diffraction tomography | B1/A1 | High for ODT framing, not S0 | Must cite as focused-beam scanning DT and Fourier diffraction relation baseline. Our distinction is not scan-FDT theory, but prepared harmonic h/l execution for repeated adjoint/backpropagation. |
| Elbau/Naujoks 2026 Invertibility | ODT / ultrasound diffraction tomography | B1/A1 | Medium-high for ODT theory | Cite for Fourier-coefficient recoverability in raster scan DT. It supports the importance of Fourier coefficients but does not implement harmonic operatorization. |
| Brown et al. 2023 FS-ODT | ODT / high-speed microscopy | B1 | Medium for real-time/high-speed claims | Cite as high-speed ODT/volumetric imaging comparator; not a direct algorithmic collision. |

## Interim conclusion

Layer 1 strengthens the strategy rather than weakening it.

- High-NA factorization remains known via Boichenko.
- ODT Fourier diffraction / focused-beam scanning theory is active and now includes 2026 raster-scan work.
- No direct prior work was found that combines Ewald/cap geometry with our circular/Fourier-Bessel prepared forward/adjoint operator and ODT detector/illumination `h/l` harmonic split.

The manuscript should therefore avoid claiming a new Fourier diffraction theorem for ODT. The claim should be:

> Given Ewald/cap Fourier measurement geometries already described by Fourier diffraction theory, we provide a geometry-aware prepared operator that exploits rotational detector/illumination structure for repeated forward/adjoint evaluation.

## Next layer

Proceed to Layer 2:

- ODT / FPDT / OCT baseline matrix.
- Full PDF checks for Kirisits/Naujoks/Scherzer 2024, Raster Scan DT 2026, FS-ODT 2023, FPDT 2015/2019, OCT k-space 2020.
- Extract whether any implementation uses reusable angular/harmonic structure or only FDT/FFT/gridding/iterative stitching.
