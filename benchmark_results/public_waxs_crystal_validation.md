# Public WAXS crystal CIF validation

Initial 2D cake-map validation using public COD CIF structures converted to finite supercell NPZ inputs.
The comparison is direct 2D WAXS cake intensity versus the cylindrical-histogram cake path on the same fixed crystal orientation.

| structure | atoms | model | n_phi | direct cake s | histogram cake s | speedup | 2D intensity L2 | ring L2 | direct anisotropy | histogram anisotropy |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| crystal_nacl_cod1000041_4x4x4 | 512 | unit | 180 | 0.1786 | 0.0210 | 8.51x | 7.306e-03 | 6.554e-03 | 1.641 | 1.308 |
| crystal_nacl_cod1000041_4x4x4 | 512 | atomic_number | 180 | 0.1940 | 0.0161 | 12.08x | 7.305e-03 | 6.554e-03 | 1.550 | 1.290 |
| crystal_quartz_cod1011097_4x4x4 | 576 | unit | 180 | 0.2196 | 0.0253 | 8.68x | 1.931e-03 | 8.645e-04 | 1.605 | 1.431 |
| crystal_quartz_cod1011097_4x4x4 | 576 | atomic_number | 180 | 0.2596 | 0.0265 | 9.80x | 2.272e-03 | 8.896e-04 | 1.593 | 1.437 |
| crystal_silicon_cod1526655_4x4x4 | 512 | unit | 180 | 0.2093 | 0.0137 | 15.29x | 2.728e-03 | 1.792e-03 | 1.693 | 1.488 |
| crystal_silicon_cod1526655_4x4x4 | 512 | atomic_number | 180 | 0.2192 | 0.0119 | 18.36x | 2.729e-03 | 1.792e-03 | 1.693 | 1.488 |

Interpretation:

- `cake_intensity_rel_l2_vs_direct` is the primary 2D fixed-orientation crystal solver check.
- `ring_rel_l2_vs_direct` verifies the 1D reduction after azimuthal averaging.
- The anisotropy columns are not an error metric; they confirm that these CIF supercells exercise non-isotropic cake-map structure.
- The `atomic_number` rows are still a lightweight multi-element check, not a final q-dependent WAXS form-factor model.

Source artifacts:

- JSON: `benchmark_results/public_waxs_crystal_validation.json`
- CSV: `benchmark_results/public_waxs_crystal_validation.csv`
