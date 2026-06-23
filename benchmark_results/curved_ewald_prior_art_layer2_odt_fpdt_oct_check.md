# Curved Ewald factorization prior-art check - Layer 2 ODT/FPDT/OCT baselines

Generated: 2026-06-22

## Question

Layer 2 checks whether optical diffraction tomography (ODT), Fourier ptychographic
diffraction tomography (FPDT), optical coherence tomography (OCT), and focused-beam
diffraction-tomography papers already contain the same algorithmic structure as the
proposed prepared curved-Ewald/cap harmonic operator.

The exact-collision question is intentionally narrow:

> Do these works use a circular/Fourier-Bessel or equivalent harmonic factorization
> of curved Ewald/cap Fourier samples, with a prepared build/hot split and repeated
> forward or adjoint execution?

## Checked primary sources

Local text was extracted from primary-source PDFs under
`benchmark_results/prior_art_pdfs/` into `benchmark_results/prior_art_text/`.

| ID | Source | Local PDF | Verification status |
|---|---|---|---|
| Kirisits et al. 2024, generalized incident field DT | https://arxiv.org/abs/2403.16835 | `prior_art_pdfs/kirisits_2024_generalized_incident_field_dt.pdf` | PDF text checked |
| Kirisits et al. 2024, generalized FDT / filtered backprop | https://arxiv.org/abs/2407.01793 | `prior_art_pdfs/kirisits_2024_generalized_fdt_filtered_backprop.pdf` | PDF text checked |
| Elbau et al. 2026, raster scan DT | https://arxiv.org/abs/2602.17351 | `prior_art_pdfs/elbau_2026_raster_scan_dt.pdf` | PDF text checked |
| Elbau and Naujoks 2026, invertibility of raster scan FDR | https://arxiv.org/abs/2602.17344 | `prior_art_pdfs/elbau_2026_invertibility_raster_scan_dt.pdf` | PDF text checked |
| Horstmeyer and Yang 2015, Fourier ptychographic tomography | https://arxiv.org/abs/1510.08756 | `prior_art_pdfs/horstmeyer_2015_fpdt.pdf` | PDF text checked |
| Zuo et al. 2019, FPDT | https://arxiv.org/abs/1904.09386 | `prior_art_pdfs/zuo_2019_fpdt.pdf` | PDF text checked |
| Brown et al. 2023, Fourier synthesis ODT | https://arxiv.org/abs/2309.16912 | `prior_art_pdfs/fourier_synthesis_odt_2023.pdf` | PDF text checked |
| Zhou et al. 2020, unified k-space OCT | https://arxiv.org/abs/2012.04875 | `prior_art_pdfs/zhou_2020_unified_k_space_oct.pdf` | PDF text checked |
| Beinert and Quellmalz 2022, TV phase retrieval for DT | https://arxiv.org/abs/2201.11579 | `prior_art_pdfs/beinert_2022_tv_phase_retrieval_dt.pdf` | PDF text checked |
| Paladhi et al. 2016, generalized backpropagation algorithms | https://arxiv.org/abs/1605.01754 | `prior_art_pdfs/paladhi_2016_generalized_backprop_dt.pdf` | PDF text checked |

## Keyword-level collision scan

The direct harmonic-factorization keywords were searched across the extracted text:

- `Fourier-Bessel`
- `cylindrical harmonic`
- `circular harmonic`
- `Jacobi-Anger`
- `Hankel`
- `Bessel`

Findings:

- No checked ODT/FPDT/OCT source used `Fourier-Bessel`, `cylindrical harmonic`,
  `circular harmonic`, or `Jacobi-Anger` as the computational route for Ewald/cap
  Fourier evaluation.
- `Hankel` appears in several DT/FDT papers as the Helmholtz Green function in
  2D, not as a Hankel-transform acceleration or circular-harmonic operator.
- `Bessel` appears in OCT/FS-ODT contexts, including Bessel-beam OCT and spherical
  particle scattering, but not as the same prepared curved-Ewald harmonic operator.

This is the main Layer 2 result: the ODT literature has strong Ewald/FDT and
backpropagation baselines, but the exact computational factorization under review
was not found in this layer.

## Matrix update

| ID | Domain | What it does | Overlap with our method | Missing relative to S0 | Class | Threat | Manuscript positioning |
|---|---|---|---|---|---|---|---|
| Kirisits et al. 2024 generalized incident field DT | ODT / ultrasound DT | Extends DT to generalized incident fields, with focused beams modeled as superpositions of plane waves and a two-step reconstruction process. | Very relevant physics: focused beams and adapted Fourier diffraction relation. | No circular/Fourier-Bessel Ewald factorization; no prepared hot operator; inverse step is framed through singular-system/TSVD machinery. | B1 | Medium | Cite as focused-beam/generalized-incident-field DT theory. Our claim should not be a new FDT; it is an operatorization of repeated evaluation once the FDT geometry is fixed. |
| Kirisits et al. 2024 generalized FDT / filtered backprop | DT / FDT | Provides generalized Fourier diffraction theorem and filtered backpropagation formulae for broad experimental geometries. Discretization maps filtered backpropagation to adjoint NDFT/NFFT-type evaluation. | Strongest ODT baseline for Ewald geometry and adjoint/backprop computation. | Uses generic NDFT/NFFT/backprop machinery, not the proposed rotational harmonic factorization or h/l split. | B1/A1 | Medium-high as baseline | Must cite. Compare against NFFT/NUFFT-type backprop rather than implying no fast baseline exists. |
| Elbau et al. 2026 raster scan DT | Focused-beam / raster-scan DT | Derives a Fourier diffraction relation for translated focused beams represented by Herglotz waves; analyzes scan geometries and accessible Fourier coefficients. | Very relevant to the question of ODT structures where repeated measurements are physically meaningful. | No prepared circular-harmonic Ewald operator; no detector/source h/l harmonic split; primarily theory/invertibility/coverage. | B1 | Medium | Cite as evidence that focused/raster-scan ODT is an active and physically meaningful geometry. Position our ODT demo as computational acceleration for repeated operators, not as the first scan-geometry theory. |
| Elbau and Naujoks 2026 invertibility raster scan DT | Focused-beam / raster-scan DT | Studies which Fourier coefficients are recoverable from the raster-scan Fourier diffraction relation. | Confirms reconstruction relevance of scan geometry and Fourier coefficient recovery. | No fast prepared evaluation algorithm; no circular-harmonic Ewald factorization. | B1 | Medium | Use to support the experimental relevance of scan/focused-beam ODT structures. |
| Horstmeyer and Yang 2015 FPT | FPDT | Uses angled LED-array illumination and ptychographic phase retrieval to fill 3D Fourier space under Ewald-sphere geometry. | Ewald sphere and repeated angle illumination are directly relevant. | No circular harmonic/Jacobi-Anger split; no prepared Ewald harmonic plan; ptychographic update strategy is different. | B1 | Medium | Baseline for FPDT geometry and iterative ptychographic reconstruction. |
| Zuo et al. 2019 FPDT | FPDT | Iteratively stitches variably illuminated low-resolution images into a high-resolution 3D Fourier-space object; explicitly discusses finite-NA Ewald-sphere effects. | Strong experimental FPDT baseline; demonstrates high-throughput angle illumination and Ewald cap filling. | No same harmonic factorization; algorithm is ptychographic Fourier-space stitching. | B1 | Medium | Cite as FPDT application baseline. Our method should be framed as accelerating a different class of repeated Ewald/cap operator. |
| Brown et al. 2023 FS-ODT | High-speed ODT | Uses multiplexed illumination patterns and inverse computation for kilohertz-rate volumetric imaging. | Very relevant to the real-time/high-throughput motivation. | Multiplexed acquisition and FISTA/multi-slice reconstruction are not the same h/l harmonic operator. | B1/A1 | Medium | Cite as high-speed ODT motivation and comparator for reconstruction speed claims. |
| Zhou et al. 2020 unified k-space OCT | OCT / ODT context | Places OCT and related modalities under a 3D k-space/Ewald-sphere framework. | Strong conceptual bridge: Ewald sphere and k-space wave-imaging interpretation. | Not a prepared harmonic Ewald evaluator; no ODT h/l split. | B1/C1 | Medium | Cite in introduction to show Ewald/k-space geometry is broadly important in wave imaging. |
| Beinert and Quellmalz 2022 TV phase retrieval DT | DT inverse problem | Uses NDFT/NFFT and adjoint NDFT for TV-based reconstruction and phase retrieval under DT models. | Important computational baseline: iterative reconstruction with adjoint NFFT. | Generic NFFT route, not rotational harmonic factorization or prepared h/l reuse. | B1/A1 | Medium-high as baseline | Compare against NUFFT/NFFT families for inverse/reconstruction workloads. |
| Paladhi et al. 2016 generalized backpropagation | DT / microwave DT | Generalizes filtered backpropagation under reduced scan coverage. | Backpropagation baseline and scan-coverage context. | Not a harmonic factorization; no prepared curved-manifold operator. | B1 | Low-medium | Cite if discussing classical/generalized backpropagation lineage. |

## What changed after Layer 2

Layer 2 strengthens two points.

1. ODT has real, active, physically meaningful repeated-measurement geometries.
   Focused-beam and raster-scan DT are not artificial workloads. The 2026
   raster-scan papers are especially relevant because they explicitly connect
   translated focused beams to Fourier coefficients of the scattering potential.

2. The baseline is stronger than a plain direct Debye-Wolf-like quadrature
   comparison. Several DT papers reduce reconstruction/backpropagation to
   NDFT/NFFT or filtered-backpropagation machinery. Therefore, the fair ODT
   comparison is against NUFFT/NFFT/backprop baselines, not against direct
   quadrature only.

## Current novelty boundary

No S0 exact collision was found in Layer 2.

The safer ODT novelty statement is:

> We do not introduce a new Fourier diffraction theorem for ODT. Existing FDT,
> generalized-FDT, FPDT, and focused/raster-scan DT work already define the
> physical Fourier/Ewald measurement geometry. Our contribution is a prepared
> rotationally structured curved-Ewald/cap operator for repeated forward/adjoint
> evaluation, exploiting detector/illumination geometry through harmonic reuse
> rather than generic NFFT/backprop evaluation.

## Implications for experiments

The ODT benchmark should include at least three baseline groups:

| Baseline group | Why needed |
|---|---|
| Direct/reference evaluation | Validates correctness and gives an intuitive lower-level baseline. |
| NUFFT/NFFT or cuFINUFFT-style adjoint | Necessary because DT reconstruction literature already uses NDFT/NFFT-type adjoints. |
| Filtered-backprop / gridding-style reconstruction baseline | Necessary for positioning against established DT practice. |

The most defensible application workload is not a one-shot forward solve. It is:

- repeated adjoint/backpropagation inside iterative reconstruction,
- raster/focused-beam scan updates,
- streaming or warm-started reconstruction from nearby frames,
- or high-throughput ODT where acquisition geometry is fixed and the build cost is amortized.

## Next layer

Layer 3 should check scattering-specific harmonic ancestors:

- fluctuation X-ray scattering and Kam-type spherical harmonic methods,
- wide-angle X-ray scattering harmonic or Fourier-Bessel algorithms,
- anisotropic PDF / spherical harmonic Fourier transform methods,
- Debye scattering equation harmonic accelerations,
- and X-ray scattering algorithms that might already use the same cylindrical
  `q_perp R cos(phi_q - beta) + q_z z` split.

Layer 3 is the most important layer for WAXS-specific novelty risk.
