# Public WAXS structure validation

Initial validation using public RCSB structures converted to the repository NPZ contract.
The Debye reference, direct WAXS ring average, and histogram ring-average path use the same constant form-factor model per row.

| structure | atoms | model | n_phi | Debye s | direct s | histogram s | hist/Debye speedup | direct vs Debye L2 | rotated vs Debye L2 | hist vs direct L2 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| protein_1crn_heavy_centered | 327 | unit | 180 | 0.0486 | 0.1172 | 0.0156 | 3.12x | 4.721e-02 | 5.353e-03 | 6.556e-04 |
| protein_1crn_heavy_centered | 327 | atomic_number | 180 | 0.0451 | 0.1190 | 0.0092 | 4.89x | 4.704e-02 | 5.199e-03 | 8.177e-04 |
| protein_1ubq_heavy_centered | 602 | unit | 180 | 0.1496 | 0.1937 | 0.0136 | 11.02x | 1.018e-02 | 3.573e-03 | 8.776e-04 |
| protein_1ubq_heavy_centered | 602 | atomic_number | 180 | 0.1457 | 0.1868 | 0.0130 | 11.20x | 1.031e-02 | 3.693e-03 | 7.840e-04 |

Interpretation:

- `histogram_rel_l2_vs_direct_ring` is the primary solver correctness check for the current fixed molecular orientation.
- `direct_ring_rel_l2_vs_debye` is expected to be nonzero for a single anisotropic protein orientation; it measures orientation-average mismatch, not a solver failure.
- If `rotated_ring_rel_l2_vs_debye` is present, it is the direct-ring curve averaged over random molecular orientations.
- The `atomic_number` rows are a lightweight multi-element path check. They are not a final q-dependent atomic form-factor WAXS model.

Source artifacts:

- JSON: `benchmark_results/public_waxs_structure_validation.json`
- CSV: `benchmark_results/public_waxs_structure_validation.csv`
