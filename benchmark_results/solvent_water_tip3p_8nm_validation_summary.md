# TIP3P water-box WAXS validation

This benchmark uses an OpenMM-generated 8 nm TIP3P water box as a realistic
amorphous/solvent WAXS input. The raw OpenMM run output is kept under
`runs/water_tip3p_8nm/` and the derived solver inputs are kept under
`structures/processed/`; both are intentionally ignored because they are
generated data artifacts.

## OpenMM source

| field | value |
|---|---:|
| requested box side | 8.0 nm |
| final box side | 7.9918195675 nm |
| waters | 16810 |
| atoms | 50430 |
| model | TIP3P |
| platform | CUDA |
| equilibration | 200 ps |
| production | 200 ps |
| timestep | 2 fs |
| temperature | 300 K |
| pressure | 1 bar |

## Derived WAXS inputs

| input | atoms | purpose |
|---|---:|---|
| `solvent_water_tip3p_8nm_full_allatom.npz` | 50430 | realistic full-box cake benchmark |
| `solvent_water_tip3p_8nm_full_oonly.npz` | 16810 | cheaper full-box oxygen-only benchmark |
| `solvent_water_tip3p_8nm_sphere_r2p5nm_allatom.npz` | 6474 | Debye-checkable all-atom sphere |
| `solvent_water_tip3p_8nm_sphere_r2p5nm_oonly.npz` | 2158 | cheaper Debye-checkable oxygen-only sphere |

All derived structures are centered in nm and retain the original box vectors
and metadata. Sphere crops are selected residue-wise using the oxygen atom as
the water-molecule anchor, so hydrogens are not cut independently.

## Sphere Debye validation

Command:

```bash
python scripts/validate_public_waxs_structures.py \
  structures/processed/solvent_water_tip3p_8nm_sphere_r2p5nm_oonly.npz \
  structures/processed/solvent_water_tip3p_8nm_sphere_r2p5nm_allatom.npz \
  --repeats 1 --qmax 2.2 --nq 24 --bin-width-nm 0.08 \
  --form-factor-models xray_f0 \
  --output benchmark_results/solvent_water_tip3p_8nm_sphere_validation.json
```

| input | atoms | n_phi | Debye s | direct ring s | histogram s | hist/Debye speedup | direct vs Debye L2 | hist vs direct L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sphere O-only | 2158 | 180 | 5.0141 | 0.9446 | 0.0455 | 110.29x | 1.245e-03 | 3.947e-04 |
| sphere all-atom | 6474 | 180 | 36.3426 | 2.2428 | 0.0674 | 539.55x | 1.175e-03 | 2.644e-04 |

## Full-box cake validation

Command:

```bash
python scripts/validate_public_waxs_crystals.py \
  structures/processed/solvent_water_tip3p_8nm_full_oonly.npz \
  structures/processed/solvent_water_tip3p_8nm_full_allatom.npz \
  --repeats 1 --qmax 2.2 --nq 24 --bin-width-nm 0.08 \
  --form-factor-models xray_f0 \
  --output benchmark_results/solvent_water_tip3p_8nm_full_cake_validation.json
```

| input | atoms | n_phi | direct cake s | histogram cake s | speedup | 2D intensity L2 | ring L2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| full O-only | 16810 | 288 | 8.7626 | 0.0932 | 94.01x | 4.156e-04 | 8.827e-05 |
| full all-atom | 50430 | 288 | 25.9942 | 0.1075 | 241.87x | 3.476e-04 | 7.360e-05 |

## Interpretation

- The Debye-checkable sphere crops show sub-`1e-3` histogram-vs-direct error
  with q-dependent `xray_f0` form factors.
- The full 8 nm all-atom solvent box reaches a `241.87x` speedup over direct
  2D cake evaluation while keeping the 2D intensity error at `3.476e-04`.
- Exact Debye is intentionally not used for the full 50k-atom all-atom box
  because the pair count is too large for a practical reference run.
