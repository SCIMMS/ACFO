# Public WAXS crystal CIF validation

Initial 2D cake-map validation using public COD CIF structures converted to finite supercell NPZ inputs.
The comparison is direct 2D WAXS cake intensity versus the cylindrical-histogram cake path on the same fixed crystal orientation.

| structure | atoms | model | n_phi | direct cake s | histogram cake s | speedup | 2D intensity L2 | ring L2 | direct anisotropy | histogram anisotropy |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| crystal_nacl_cod1000041_4x4x4 | 512 | unit | 180 | 0.1890 | 0.0193 | 9.78x | 7.306e-03 | 6.554e-03 | 1.641 | 1.308 |
| crystal_nacl_cod1000041_4x4x4 | 512 | atomic_number | 180 | 0.1769 | 0.0150 | 11.76x | 7.305e-03 | 6.554e-03 | 1.550 | 1.290 |
| crystal_nacl_cod1000041_4x4x4 | 512 | xray_f0 | 180 | 0.1741 | 0.0161 | 10.79x | 6.940e-03 | 6.455e-03 | 1.552 | 1.290 |
| crystal_quartz_cod1011097_4x4x4 | 576 | unit | 180 | 0.1985 | 0.0174 | 11.43x | 1.931e-03 | 8.645e-04 | 1.605 | 1.431 |
| crystal_quartz_cod1011097_4x4x4 | 576 | atomic_number | 180 | 0.1915 | 0.0170 | 11.26x | 2.272e-03 | 8.896e-04 | 1.593 | 1.437 |
| crystal_quartz_cod1011097_4x4x4 | 576 | xray_f0 | 180 | 0.1894 | 0.0160 | 11.84x | 1.898e-03 | 8.368e-04 | 1.596 | 1.438 |
| crystal_silicon_cod1526655_4x4x4 | 512 | unit | 180 | 0.1699 | 0.0111 | 15.25x | 2.728e-03 | 1.792e-03 | 1.693 | 1.488 |
| crystal_silicon_cod1526655_4x4x4 | 512 | atomic_number | 180 | 0.1724 | 0.0113 | 15.33x | 2.729e-03 | 1.792e-03 | 1.693 | 1.488 |
| crystal_silicon_cod1526655_4x4x4 | 512 | xray_f0 | 180 | 0.1728 | 0.0114 | 15.12x | 2.473e-03 | 1.689e-03 | 1.693 | 1.488 |

Interpretation:

- `cake_intensity_rel_l2_vs_direct` is the primary 2D fixed-orientation crystal solver check.
- `ring_rel_l2_vs_direct` verifies the 1D reduction after azimuthal averaging.
- The anisotropy columns are not an error metric; they confirm that these CIF supercells exercise non-isotropic cake-map structure.
- The `atomic_number` rows are a lightweight multi-element path check.
- The `xray_f0` rows use q-dependent neutral-atom elastic X-ray f0 values from `periodictable`; anomalous dispersion, ionic state, solvent, and Debye-Waller effects are intentionally outside this first validation.

Source artifacts:

- JSON: `benchmark_results/public_waxs_crystal_validation.json`
- CSV: `benchmark_results/public_waxs_crystal_validation.csv`
