# Perfect protein-nanocrystal lattice-factorization control

Exact-coordinate unit-cell harmonic amplitude와 finite translation lattice sum을 결합했다.

- unit exact harmonic vs direct NDFT complex L2: `1.010e-12`
- unit harmonic seconds: `0.083`

| supercell | atoms | cells | repetition residual | factorized full-target s | subset direct s | subset complex L2 |
|---|---:|---:|---:|---:|---:|---:|
| 3x3x3 | 216,216 | 27 | 1.522e-12 | 0.084 | 1.473 | 3.682e-11 |
| 5x5x5 | 1,001,000 | 125 | 3.059e-12 | 0.093 | 6.824 | 2.428e-11 |

## Sparse positional-defect correction

- defect atoms: `1,001` (0.1000%)
- displacement RMS: `0.0200 nm`
- full-target delta correction: `0.196 s`
- corrected subset complex L2: `2.425e-11`


- exact periodic-control gates: **PASS**
- 이 경로는 perfect repeated crystal의 표준 구조인자×lattice-sum control이다.
- disorder/defect가 있는 모든 원자구조를 자동으로 해결하지 않으며 ACFO novelty claim이 아니다.
- sparse defects는 perfect background와 explicit delta-scatterer correction으로 확장할 수 있다.
