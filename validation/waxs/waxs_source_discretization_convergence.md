# WAXS high-q source-discretization convergence

Detector target grid를 고정하고 cylindrical source representation만 바꾸어 exact-atom direct NDFT와 비교했다.

## Azimuth-source convergence

| source Nphi | arc at Rmax (nm) | complex L2 | intensity L2 | ring L2 | intensity NCC | storage MiB |
|---:|---:|---:|---:|---:|---:|---:|
| 750 | 4.453e-02 | 2.176e-01 | 2.020e-01 | 1.845e-02 | 0.962337 | 0.275 |
| 1,500 | 2.227e-02 | 1.130e-01 | 1.083e-01 | 7.242e-03 | 0.989715 | 0.275 |
| 3,000 | 1.113e-02 | 5.644e-02 | 5.249e-02 | 3.503e-03 | 0.997477 | 0.275 |
| 6,000 | 5.567e-03 | 2.840e-02 | 2.564e-02 | 3.918e-03 | 0.999398 | 0.275 |
| 12,000 | 2.783e-03 | 1.635e-02 | 1.516e-02 | 1.013e-03 | 0.999789 | 0.275 |
| 24,000 | 1.392e-03 | 1.077e-02 | 1.022e-02 | 7.066e-04 | 0.999905 | 0.275 |
| 48,000 | 6.958e-04 | 8.781e-03 | 8.582e-03 | 1.194e-03 | 0.999933 | 0.275 |
| 96,000 | 3.479e-04 | 8.297e-03 | 8.034e-03 | 9.448e-04 | 0.999941 | 0.275 |

## Radial/axial-bin convergence at finest source Nphi

| bin width (nm) | complex L2 | intensity L2 | ring L2 | intensity NCC | storage MiB |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 9.067e-01 | 7.830e-01 | 2.288e-01 | 0.341891 | 0.275 |
| 0.05 | 5.096e-01 | 4.628e-01 | 1.065e-01 | 0.792856 | 0.275 |
| 0.025 | 2.748e-01 | 2.603e-01 | 6.719e-02 | 0.936411 | 0.275 |
| 0.0125 | 1.289e-01 | 1.204e-01 | 2.073e-02 | 0.986738 | 0.275 |
| 0.00625 | 6.644e-02 | 6.203e-02 | 7.957e-03 | 0.996479 | 0.275 |
| 0.003125 | 3.356e-02 | 3.264e-02 | 1.084e-03 | 0.999025 | 0.275 |
| 0.0015625 | 1.683e-02 | 1.554e-02 | 6.065e-03 | 0.999779 | 0.275 |
| 0.00078125 | 8.297e-03 | 8.034e-03 | 9.448e-04 | 0.999941 | 0.275 |

- finest exploratory intensity <=1%: **PASS**
- finest exploratory ring intensity <=0.5%: **PASS**
- 이 sweep은 operator가 아니라 atom-to-cylinder representation의 수렴성을 측정한다.
- target 수를 고정했으므로 source Nphi 증가와 detector oversampling을 혼동하지 않는다.
