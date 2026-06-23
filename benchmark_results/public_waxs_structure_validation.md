# Public WAXS structure validation

Initial validation using public RCSB structures converted to the repository NPZ contract.
The Debye reference, direct WAXS ring average, and histogram ring-average path use the same form-factor model per row.

| structure | atoms | model | n_phi | Debye s | direct s | histogram s | hist/Debye speedup | direct vs Debye L2 | rotated vs Debye L2 | hist vs direct L2 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| protein_1crn_heavy_centered | 327 | unit | 180 | 0.0903 | 0.0995 | 0.0141 | 6.42x | 4.721e-02 | 5.353e-03 | 6.556e-04 |
| protein_1crn_heavy_centered | 327 | atomic_number | 180 | 0.0883 | 0.1109 | 0.0133 | 6.63x | 4.704e-02 | 5.199e-03 | 8.177e-04 |
| protein_1crn_heavy_centered | 327 | xray_f0 | 180 | 0.0699 | 0.1045 | 0.0153 | 4.57x | 4.672e-02 | 5.156e-03 | 7.010e-04 |
| protein_1ubq_heavy_centered | 602 | unit | 180 | 0.2894 | 0.1930 | 0.0135 | 21.48x | 1.018e-02 | 3.573e-03 | 8.776e-04 |
| protein_1ubq_heavy_centered | 602 | atomic_number | 180 | 0.2482 | 0.1922 | 0.0131 | 18.96x | 1.031e-02 | 3.693e-03 | 7.840e-04 |
| protein_1ubq_heavy_centered | 602 | xray_f0 | 180 | 0.2671 | 0.2238 | 0.0132 | 20.28x | 1.006e-02 | 3.680e-03 | 6.819e-04 |

Interpretation:

- `histogram_rel_l2_vs_direct_ring` is the primary solver correctness check for the current fixed molecular orientation.
- `direct_ring_rel_l2_vs_debye` is expected to be nonzero for a single anisotropic protein orientation; it measures orientation-average mismatch, not a solver failure.
- If `rotated_ring_rel_l2_vs_debye` is present, it is the direct-ring curve averaged over random molecular orientations.
- The `atomic_number` rows are a lightweight multi-element path check.
- The `xray_f0` rows use q-dependent neutral-atom elastic X-ray f0 values from `periodictable`; anomalous dispersion, ionic state, solvent, and Debye-Waller effects are intentionally outside this first validation.

Source artifacts:

- JSON: `benchmark_results/public_waxs_structure_validation.json`
- CSV: `benchmark_results/public_waxs_structure_validation.csv`
