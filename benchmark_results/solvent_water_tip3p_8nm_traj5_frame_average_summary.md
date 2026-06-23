# TIP3P water trajectory frame-average WAXS validation

This benchmark checks whether the solvent WAXS validation remains stable over
multiple MD frames rather than relying on a single final snapshot. Five frames
were selected evenly from the OpenMM production DCD:

```text
frames = [0, 4, 9, 14, 19]
```

The trajectory source is `runs/water_tip3p_8nm/water_tip3p_8nm_trajectory.dcd`
from the 8 nm TIP3P CUDA OpenMM run. Derived frame NPZ files are generated data
artifacts under `structures/processed/` and are intentionally ignored.

## Preparation command

```bash
python scripts/prepare_openmm_water_waxs_inputs.py \
  --trajectory-dcd runs/water_tip3p_8nm/water_tip3p_8nm_trajectory.dcd \
  --prefix solvent_water_tip3p_8nm_traj5 \
  --max-frames 5 \
  --sphere-radius-nm 2.5
```

## Sphere Debye validation

Command:

```bash
python scripts/validate_public_waxs_structures.py \
  --glob "structures/processed/solvent_water_tip3p_8nm_traj5_frame*_sphere_r2p5nm_*.npz" \
  --repeats 1 --qmax 2.2 --nq 24 --bin-width-nm 0.08 \
  --form-factor-models xray_f0 \
  --output benchmark_results/solvent_water_tip3p_8nm_traj5_sphere_validation.json
```

| subset | frames | atoms mean | atoms range | Debye s mean | direct ring s mean | histogram s mean | hist/Debye speedup mean | direct vs Debye L2 mean | hist vs direct L2 mean | hist vs direct L2 max |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| sphere all-atom | 5 | 6447 | 6408-6477 | 33.4150 | 2.2130 | 0.0194 | 1778.61x | 1.312e-03 | 2.204e-04 | 3.956e-04 |
| sphere O-only | 5 | 2149 | 2136-2159 | 3.5175 | 0.8062 | 0.0123 | 325.69x | 1.308e-03 | 3.427e-04 | 5.939e-04 |

## Full-box 2D cake validation

Command:

```bash
python scripts/validate_public_waxs_crystals.py \
  --glob "structures/processed/solvent_water_tip3p_8nm_traj5_frame*_full_*.npz" \
  --repeats 1 --qmax 2.2 --nq 24 --bin-width-nm 0.08 \
  --form-factor-models xray_f0 \
  --output benchmark_results/solvent_water_tip3p_8nm_traj5_full_cake_validation.json
```

| subset | frames | atoms | direct cake s mean | histogram cake s mean | speedup mean | speedup range | 2D intensity L2 mean | 2D intensity L2 max | ring L2 mean |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| full all-atom | 5 | 50430 | 26.8237 | 0.1076 | 250.93x | 225.62x-283.84x | 3.404e-04 | 4.922e-04 | 1.472e-04 |
| full O-only | 5 | 16810 | 9.0801 | 0.0544 | 168.24x | 139.94x-196.39x | 4.063e-04 | 6.311e-04 | 1.529e-04 |

## Interpretation

- The 5-frame solvent trajectory check keeps histogram-vs-direct errors below
  `6.0e-4` for Debye-checkable sphere crops and below `6.4e-4` for full-box
  2D cake maps.
- The full all-atom 8 nm water box maintains a mean `250.93x` speedup over
  direct cake evaluation across production frames.
- This reduces the risk that the solvent result is a single-snapshot artifact.
  The remaining limitation is physical modeling scope: TIP3P water and neutral
  `xray_f0(q)` form factors are used, without anomalous dispersion, explicit
  experimental background, or Debye-Waller fitting.
