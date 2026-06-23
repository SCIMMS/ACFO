# Curved Ewald factorization prior-art check - Layer 4 adjacent prepared-operator methods

Generated: 2026-06-22

## Question

Layer 4 checks adjacent computational fields where a forward/adjoint operator,
kernel, or transform plan is prepared once and reused many times.

The goal is not to find the same physical operator. The goal is to avoid
overclaiming the general computational idea of build/hot reuse.

The narrow question is:

> Is prepared operatorization or kernel reuse already a known computational
> design pattern in Fourier inverse problems, and if so, what remains new about
> our WAXS/ODT curved-Ewald operator?

## Checked primary sources

Local text was extracted from primary-source PDFs under
`benchmark_results/prior_art_pdfs/` into `benchmark_results/prior_art_text/`.

| ID | Source | Local PDF | Verification status |
|---|---|---|---|
| Wang, Shkolnisky, Singer 2013, cryo-EM FIRM | https://arxiv.org/abs/1307.5824 | `prior_art_pdfs/wang_2013_cryoem_fast_volumetric_reconstruction.pdf` | PDF text checked |
| Pratley, Johnston-Hollitt, McEwen 2018/2019, radio w-stacking/w-projection | https://arxiv.org/abs/1807.09239 | `prior_art_pdfs/pratley_2018_radio_w_projection_hankel.pdf` | PDF text checked |
| Merry 2015, separable w-projection kernels | https://arxiv.org/abs/1511.07152 | `prior_art_pdfs/merry_2015_w_projection_computationally_efficient.pdf` | PDF text checked |
| Lucas, Thiran, Wiaux 2019, fast w-projection | https://arxiv.org/abs/1904.08463 | `prior_art_pdfs/lucas_2019_fast_image_reconstruction_radio_w_projection.pdf` | PDF text checked |
| Wang and Anastasio 2012, photoacoustic Fourier reconstruction | https://arxiv.org/abs/1208.2262 | `prior_art_pdfs/wang_anastasio_2012_photoacoustic_fourier_reconstruction.pdf` | PDF text checked |
| Kunyansky 2011, fast photoacoustic algorithms | https://arxiv.org/abs/1102.1413 | `prior_art_pdfs/kunyansky_2011_fast_photoacoustic_algorithms.pdf` | PDF text checked |
| Haltmeier, Scherzer, Zangerl 2008/2009, photoacoustic NUFFT | https://arxiv.org/abs/0808.3510 | `prior_art_pdfs/haltmeier_2008_photoacoustic_nufft.pdf` | PDF text checked |
| PyNX 2020, coherent X-ray operator toolkit | https://arxiv.org/abs/2008.11511 | `prior_art_pdfs/pynx_2020_operator_hpc_coherent_xray.pdf` | PDF text checked |

## Matrix update

| ID | Domain | What it does | Overlap with our method | Missing relative to S0 | Class | Threat | Manuscript positioning |
|---|---|---|---|---|---|---|---|
| Wang/Shkolnisky/Singer 2013 FIRM | Cryo-EM reconstruction | Uses the Toeplitz structure of `A*A`; precomputes a convolution kernel and `A*b` using NUFFT, then applies fast FFT-based iterations. | Very strong prepared-operator precedent: build expensive kernel once, accelerate repeated iterative reconstruction. | Projection-slice cryo-EM geometry, not WAXS/ODT curved Ewald/cap detector geometry; no cylindrical cake-map or h/l split. | A1 | High for generic "prepared operator" claim | Must cite. Do not claim build/hot operatorization itself is new. Claim geometry-specific factorization and workload-specific amortization. |
| Pratley et al. 2018/2019 w-stacking/w-projection | Radio interferometry | Develops radially symmetric w-projection kernels using a 1D Hankel transform and distributed w-stacking/w-projection. | Strong adjacent example of exploiting radial symmetry and Hankel transforms to reduce curved/wide-field Fourier correction cost. | Radio sky curvature/non-coplanar baselines, not scattering Ewald cap; kernel correction for interferometric gridding, not atom/density-to-detector WAXS/ODT operator. | A1 | Medium-high | Cite as adjacent curved-manifold Fourier kernel acceleration. It supports our lineage but does not erase Ewald/WAXS/ODT novelty. |
| Merry 2015 separable w-projection | Radio interferometry | Approximates w-projection kernels as separable, reducing memory and making precomputation practical. | Strong adjacent memory/compression/precompute precedent. | Different measurement equation and separability target; no Ewald/cake/h-l harmonic split. | A1 | Medium | Cite as precedent for low-rank/separable kernel compression in Fourier imaging. |
| Lucas et al. 2019 fast w-projection | Radio interferometry | Uses Hankel transform optimization for w-projection kernel generation; reports speedups mainly in kernel generation. | Very relevant to our build-cost discussion: a geometry-aware transform helps primarily in plan/kernel generation. | Radio interferometry only; not a prepared WAXS/ODT scattering operator. | A1 | Medium | Useful analogy for why build-time reduction is a legitimate contribution. |
| Wang and Anastasio 2012 photoacoustic Fourier reconstruction | Photoacoustic tomography | Derives Fourier-transform reconstruction formulae for circular/spherical measurement geometries. | Adjacent circular/spherical measurement geometry and fast Fourier reconstruction. | Direct inverse formula for photoacoustic data, not curved Ewald Fourier evaluation or prepared harmonic operator. | A1/B1 | Medium | Cite only if discussing geometry-aware Fourier inversion beyond optics/scattering. |
| Kunyansky 2011 fast photoacoustic algorithms | Photoacoustic / thermoacoustic tomography | Uses circular/spherical/cylindrical symmetries, spherical harmonics, Hankel transforms, and FFT/NUFFT to build fast reconstructions. | Strong mathematical/algorithmic cousin: symmetry-aware transforms on circular/spherical measurement geometries. | Different wave equation and data model; not Ewald/cap scattering or prepared WAXS/ODT h/l operator. | A1/C1 | Medium-high as adjacent harmonic tomography | Cite as a broad harmonic tomography ancestor; not an exact collision. |
| Haltmeier et al. 2008/2009 photoacoustic NUFFT | Photoacoustic tomography | Uses NUFFT for Fourier reconstruction to avoid interpolation artifacts while keeping fast computation. | Generic nonuniform Fourier baseline in tomography; includes precomputed NUFFT windows. | Generic NUFFT reconstruction, not geometry-specific harmonic factorization. | B1/A1 | Medium | Cite as evidence that NUFFT baselines are standard in inverse imaging. |
| PyNX 2020 | Coherent X-ray imaging / ptychography / CDI | Provides GPU-accelerated operator composition for CDI, ptychography, and wavefront propagation. | Strong software-pattern precedent: operators, GPUs, iterative imaging pipelines. | Different coherent-imaging forward models; no cylindrical Ewald/cake harmonic factorization. | A1 | Medium | Cite if discussing implementation style and GPU operator pipelines, not as mathematical prior art. |

## Layer 4 conclusion

No S0 exact collision was found, but Layer 4 narrows the claim substantially.

The following ideas are not novel by themselves:

- build once, reuse many times;
- precomputing kernels for iterative reconstruction;
- using Toeplitz or convolution structure for `A*A`;
- using NUFFT/NFFT adjoints in inverse problems;
- using Hankel transforms to generate radially symmetric Fourier kernels;
- using separable/low-rank approximations to reduce kernel memory;
- operator-based GPU imaging frameworks.

The remaining defensible novelty is:

> The proposed method specializes this broader prepared-operator design pattern
> to rotationally structured curved Ewald/cap Fourier evaluation, using the
> cylindrical WAXS phase split and ODT detector/illumination harmonic structure
> to reduce repeated forward/adjoint workloads that generic NUFFT/backprop or
> generic kernel-precomputation methods do not specifically exploit.

## Important claim edit

Avoid this claim:

> We introduce a prepared forward/adjoint Fourier operator for iterative imaging.

Safer claim:

> We introduce a geometry-prepared forward/adjoint operator for rotationally
> structured curved Ewald/cap Fourier evaluation, and show that this preparation
> is especially effective in WAXS detector cake maps and repeated ODT
> backpropagation/reconstruction workloads.

## Benchmark implications

Layer 4 means the paper needs two benchmark framings:

1. Against generic NUFFT/cuFINUFFT or NFFT-style operators.
2. Against domain-style baselines where build/precompute cost is separated from
   hot iteration cost.

Report both:

- build or preparation time,
- hot forward/adjoint time,
- amortized time after `N` repeated evaluations,
- memory footprint of the prepared representation,
- and accuracy versus the domain reference.

This is especially important because radio w-projection and cryo-EM FIRM show
that the community already recognizes build/hot separation as a meaningful
algorithmic axis.

## Next layer

Layer 5 should check the numerical-transform family:

- pseudo-polar Fourier transforms,
- polar Fourier / polar spectral solvers,
- FFT in azimuth plus discrete Hankel transforms,
- fast spherical/cylindrical harmonic transforms,
- and general fast Fourier-Bessel/Hankel transform libraries.

Layer 5 will decide how to cite the mathematical transform ancestry without
making the physical WAXS/ODT claim look too broad.
