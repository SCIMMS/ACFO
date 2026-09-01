# ACFO v0.2.0 manuscript evidence

This directory is the public evidence and figure-source bundle for *Geometry factorization and representation dispatch for repeated curved Fourier inference*.

- WAXS: 8.279-fold versus the strongest eligible fused FINUFFT Type-3 comparator (95% paired-bootstrap interval 8.166-8.396), with preserved candidate ordering for the prespecified noise realization.
- NuMagSANS Example 3: 1.687-fold for the shared 800-orientation Fourier backbone versus affine Type-2 (95% orientation-resampling interval 1.684-1.690). The five packing reductions are excluded from this ratio.
- UOB-100(Dy): measured-data correctness example with primary candidate J/T=0.134132 and numerical agreement among ACFO, exact FFT and direct sums. A specialized eligible representation remains faster.
- ODT: native Type-2 forward and Type-1 adjoint comparisons, with setup, hot application and dtype reported separately.
- High-NA Debye-Wolf: Supplementary accuracy-control evidence only.

Run `python scripts/verify_ncs_release_v020.py` from the repository root. Third-party raw archives are not redistributed. `PROVENANCE.json` records original and normalized-copy hashes; normalization changes machine-local paths and line endings only.
