# WAXS high-q hot-reuse AB/BA sweep

- integrity: **PASS**
- tested high-q trend: **SUPPORTED**
- workload: `216,216` atoms / `27` repeated cells / supercell `[3, 3, 3]`
- warm-up / measured pairs per band: `1 / 4`
- paired hot timing excludes setup; setup + K-hot totals below are median-based models, not paired statistics.

| q band (A^-1) | Nq/Nphi | targets | ACFO setup/hot s | FINUFFT setup/hot s | paired median/p05 | ACFO/FINUFFT us target^-1 | break-even K | complex L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05-1.35 | 16/240 | 3,840 | 0.0286/0.0889 | 0.4423/0.8872 | 10.666x/7.041x | 23.15/231.05 | 1 | 1.621e-07 |
| 3.35-4.65 | 16/576 | 9,216 | 0.0086/0.1570 | 0.2188/4.3694 | 27.876x/25.271x | 17.04/474.11 | 1 | 6.477e-07 |
| 6.70-8.00 | 16/864 | 13,824 | 0.0153/0.2048 | 0.4368/16.8474 | 82.680x/61.920x | 14.82/1218.71 | 1 | 6.999e-07 |

## Setup + K hot applications

| q band (A^-1) | K | ACFO total s | FINUFFT total s | FINUFFT/ACFO |
|---|---:|---:|---:|---:|
| 0.05-1.35 | 1 | 0.1175 | 1.3295 | 11.314x |
| 0.05-1.35 | 10 | 0.9175 | 9.3145 | 10.152x |
| 0.05-1.35 | 100 | 8.9179 | 89.1642 | 9.998x |
| 3.35-4.65 | 1 | 0.1656 | 4.5882 | 27.701x |
| 3.35-4.65 | 10 | 1.5789 | 43.9126 | 27.813x |
| 3.35-4.65 | 100 | 15.7112 | 437.1559 | 27.825x |
| 6.70-8.00 | 1 | 0.2201 | 17.2842 | 78.534x |
| 6.70-8.00 | 10 | 2.0636 | 168.9109 | 81.854x |
| 6.70-8.00 | 100 | 20.4985 | 1685.1786 | 82.210x |

## Predeclared trend readout

- high/low paired-median ratio: `7.752x`
- high/low paired-p05 ratio: `8.794x`
- log-speedup slope versus q center: `0.308006 A`
- FINUFFT/ACFO per-target growth ratio: `8.241x`
- A valid run may return `NOT SUPPORTED`; integrity and hypothesis outcome are intentionally separate.

## Claim boundary

- The paired timing claim applies only to the tested finite 216,216-atom exact repeated 1IEE crystal and full uniform polar target rings.
- The separable finite lattice factor is a crystallographic specialization; dense disordered WAXS cannot use that perfect-lattice specialization.
- Prepared ACFO and reusable FINUFFT plan setup are excluded from paired hot timing and are reported separately through setup + K-hot totals.
- Setup + K-hot totals use measured hot medians and are not paired statistics or fully executed K-frame workflows.
- FINUFFT eps=1e-6 is a practical timing baseline; direct-NDFT correctness is established separately.
- Equal q-window width and fixed Nq are held across bands while Nphi increases to meet the band-specific harmonic support; the trend is workload-specific, not a universal asymptotic claim.
- The 6.70-8.00 inverse-angstrom band is a computational extension and may require a custom chamber or detector geometry; it is not evidence for the tested EIGER active region.
- This sweep does not measure detector acquisition, pixel remapping, background correction, or end-to-end XFEL throughput.
