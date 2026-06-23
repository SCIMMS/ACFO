# Curved Ewald factorization prior-art check - Layer 3 scattering harmonic ancestors

Generated: 2026-06-22

## Question

Layer 3 checks the WAXS/SAXS/FXS/X-ray-scattering side of the prior art.
This is the highest-risk layer for the WAXS validation part of the manuscript,
because harmonic and Debye-equation accelerations are common in scattering.

The narrow collision question is:

> Has the scattering literature already used the same cylindrical
> `q_perp R cos(phi_q - beta) + q_z z` phase split, angular harmonic reuse,
> and prepared forward/adjoint operator for WAXS/Ewald/cake-map evaluation?

## Checked primary sources

Local text was extracted from primary-source PDFs under
`benchmark_results/prior_art_pdfs/` into `benchmark_results/prior_art_text/`.

| ID | Source | Local PDF | Verification status |
|---|---|---|---|
| Zhang et al. 2022, anisotropic X-ray PDF spherical harmonics | https://arxiv.org/abs/2205.05865 | `prior_art_pdfs/zhang_2022_anisotropic_xray_pdf_spherical_harmonics.pdf` | PDF text checked |
| Chen and Pollack 2020/2023, WAXS Frequency Marching | https://arxiv.org/abs/2012.13370 | `prior_art_pdfs/chen_pollack_2020_waxs_frequency_marching.pdf` | PDF text checked |
| Pabit et al. 2015, WAXS vs MD for nucleic acids | https://arxiv.org/abs/1512.08074 | `prior_art_pdfs/pabit_2015_waxs_md_nucleic_acids.pdf` | PDF text checked |
| Deumer et al. 2021/2022, CDEF / Debye scattering formula | https://arxiv.org/abs/2109.06570 | `prior_art_pdfs/deumer_2021_cdef_debye_scattering_formula.pdf` | PDF text checked |
| Shachar and Garay 2023, Rayleigh-Gans-Debye anisotropic inhomogeneities | https://arxiv.org/abs/2305.01542 | `prior_art_pdfs/shachar_2023_rgd_anisotropic_inhomogeneities.pdf` | PDF text checked |
| Nasedkin et al. 2016, S/WAXS protein dynamics | https://arxiv.org/abs/1611.08259 | `prior_art_pdfs/nasedkin_2016_swaxs_protein_dynamics.pdf` | PDF text checked |
| Flamant et al. 2016, XFEL spherical-harmonic EMC | https://arxiv.org/abs/1602.01301 | `prior_art_pdfs/flamant_2016_xfel_harmonic_analysis.pdf` | PDF text checked |

Searches also covered direct-collision phrases such as:

- `q_perp R cos X-ray scattering Bessel`
- `Jacobi-Anger X-ray scattering Debye`
- `Bessel expansion Debye scattering formula`
- `cylindrical coordinates Debye scattering equation`
- `wide-angle X-ray scattering Fourier-Bessel algorithm`

No direct phrase-level collision was found.

## Keyword-level collision scan

Across the checked PDFs, the direct same-algorithm terms were absent or only
appeared in unrelated contexts:

- `Fourier-Bessel`: not found in the checked WAXS/FXS scattering PDFs.
- `cylindrical harmonic`: not found.
- `circular harmonic`: not found.
- `Jacobi-Anger`: not found.
- `q_perp` / equivalent cake-map split language: not found.

Positive harmonic signals were found, but they correspond to different operators:

- spherical harmonics and spherical Bessel functions for anisotropic PDF transforms,
- spherical harmonic analysis on XFEL Ewald shells for orientation/reconstruction,
- Debye pair-distance histogram acceleration for isotropic scattering curves,
- WAXS/SWAXS density refinement via frequency marching,
- Rayleigh-Gans-Debye scattering theory and Ewald-sphere approximations.

## Matrix update

| ID | Domain | What it does | Overlap with our method | Missing relative to S0 | Class | Threat | Manuscript positioning |
|---|---|---|---|---|---|---|---|
| Zhang et al. 2022 anisotropic X-ray PDF | Anisotropic total scattering / PDF | Expands total scattering and PDF functions in spherical harmonics, uses Rayleigh expansion and spherical Bessel functions to relate reciprocal and real space. | Strong harmonic ancestor in X-ray scattering; explicitly uses Ewald-sphere diffraction geometry and spherical harmonics. | Not a WAXS detector cake-map evaluator; no cylindrical `q_perp R cos(phi-beta)+q_z z` split; no prepared repeated forward/adjoint operator; output is anisotropic PDF. | A1/C1 | Medium | Cite as scattering-side harmonic/PDF precedent. It supports the claim boundary: harmonic identities are known; our contribution is the prepared curved-manifold operator for a different measurement workload. |
| Chen and Pollack 2020/2023 Frequency Marching | Solution WAXS/SWAXS reconstruction | Refines 3D electron density from SAXS/WAXS profiles by marching to higher reciprocal frequencies. | WAXS application relevance; iterative reconstruction against wide-angle profiles. | Uses orientationally averaged profile/Debye-style computation, not detector-resolved cylindrical harmonic cake-map evaluation. | B1 | Medium | Useful WAXS motivation and inverse-problem context, but not an algorithmic collision. |
| Pabit et al. 2015 WAXS vs MD | Biomolecular WAXS validation | Compares WAXS experiments with MD simulations for nucleic-acid structural change. | Shows WAXS is experimentally meaningful for biomolecular structure validation. | No new fast scattering operator; no harmonic factorization. | B1/D1 | Low | Cite only if manuscript needs biomolecular WAXS motivation. |
| Deumer et al. 2021/2022 CDEF / DEBYER | SAXS / Debye formula software | Uses quasi-random point clouds and DEBYER for efficient Debye scattering formula evaluation; DEBYER reduces repeated `q` evaluation through pair-distance histogram binning. | Important computational baseline: pair-distance histogram reuse accelerates Debye formula evaluation for many `q` values. | Computes orientation-averaged scattering curves/form factors, not detector/cake-map complex amplitudes; no angular FFT or cylindrical Ewald/cake split. | A1/B1 | Medium-high as WAXS/SAXS baseline | Must cite/compare if discussing Debye-equation acceleration. It is not the same algorithm, but it is the closest scattering-side acceleration pattern found in Layer 3. |
| Shachar and Garay 2023 RGD anisotropic inhomogeneities | General scattering theory | Derives Rayleigh-Gans-Debye model for anisotropic inhomogeneities; discusses Ewald-sphere geometry and large-size approximations. | Ewald-sphere and RGD physical context. | Theory/approximation paper, not a prepared harmonic computational operator. | C1/B1 | Low-medium | Cite if introducing broader RGD/Ewald scattering physics; not a novelty threat to the algorithm. |
| Nasedkin et al. 2016 S/WAXS protein dynamics | S/WAXS application | Uses S/WAXS data and CRYSOL-like settings; mentions spherical harmonics as part of scattering-curve calculation tools. | Confirms spherical-harmonic machinery is normal in SAXS/WAXS software ecosystems. | Not the same detector-resolved WAXS cake-map operator; no build/hot prepared adjoint. | B1/C1 | Low-medium | Background/application citation only. |
| Flamant et al. 2016 XFEL harmonic EMC | XFEL single-particle imaging | Uses spherical harmonics on reciprocal-space shells to separate angular and radial degrees of freedom in EMC reconstruction. | Strong adjacent X-ray harmonic method on Ewald/shell geometry; relevant to claim boundaries. | Works on diffraction-pattern orientation recovery and shell intensity expansion, not real-space atom-to-cake-map WAXS evaluation; no cylindrical cake-map factorization. | A1/C1 | Medium | Cite as X-ray harmonic-analysis precedent. It makes clear that spherical shell harmonic acceleration is known in XFEL, while our operator is a different prepared WAXS/ODT Fourier-evaluation workload. |

## Layer 3 conclusion

No S0 exact collision was found.

The scattering literature contains several important ancestors and baselines:

1. Spherical harmonics and spherical Bessel/Rayleigh expansions are established
   in anisotropic total scattering/PDF and XFEL reconstruction.
2. Debye-equation acceleration by pair-distance histograms is established and
   should be treated as a serious WAXS/SAXS baseline.
3. WAXS/SWAXS reconstruction and validation papers exist, but they generally
   use orientationally averaged profiles, Debye-style profile computation, or
   optimization in density/model space rather than a prepared detector-resolved
   curved-manifold operator.

The strongest WAXS-side novelty boundary is therefore:

> We are not claiming that spherical harmonics, Rayleigh expansions, spherical
> Bessel transforms, Debye histogram acceleration, or WAXS/SWAXS reconstruction
> are new. The new part is the detector/cake-map-specific prepared operator:
> exploiting cylindrical geometry and the `q_perp R cos(phi_q - beta) + q_z z`
> phase split to amortize repeated curved-Ewald Fourier evaluation.

## Benchmark implications

For WAXS, the fair baseline set should include:

| Baseline | Why it matters |
|---|---|
| Direct Debye / direct detector evaluation | Correctness reference. |
| Pair-distance histogram / DEBYER-like curve computation | Closest established Debye acceleration, especially for isotropic `I(q)` profiles. |
| NUFFT / FINUFFT detector-grid evaluation | Generic nonuniform Fourier baseline. |
| Spherical-harmonic/PDF or XFEL harmonic methods | Citation boundary, not necessarily a direct runtime baseline unless adapted. |

The WAXS claim should avoid saying "first harmonic scattering method". A safer
claim is:

> Compared with established isotropic Debye-curve and spherical-harmonic
> scattering tools, our prepared operator targets detector-resolved WAXS
> cake-map evaluation on curved/cylindrical reciprocal manifolds, where angular
> reuse and fixed detector geometry can be amortized across many evaluations.

## Open gaps after Layer 3

The following should still be checked before final manuscript submission:

- primary CRYSOL / WAXSiS / FoXS / Pepsi-SAXS implementation papers,
- DEBUSSY and other Debye-equation package papers,
- FXS/Kam original papers if the manuscript discusses scattering harmonic
  lineage in detail,
- GISAXS / GIWAXS simulation packages if the final scope includes surfaces,
- and any package that computes 2D detector WAXS patterns rather than only
  1D orientationally averaged curves.

Layer 4 should next check adjacent prepared-operator methods, especially
cryo-EM Toeplitz/NUFFT, radio-interferometry w-projection/Hankel, and
photoacoustic circular/spherical measurement reconstructions.
