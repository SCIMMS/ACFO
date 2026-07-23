# TIP3P 20-frame dense high-q exact-beta validation

The 8 nm TIP3P trajectory is read directly from DCD without persistent per-frame NPZ files.

- frames: `20`
- atoms/frame: `50,430`
- q: `[5.0, 6.3]` inverse angstrom
- exact-beta backend: `cpp_fused`
- detector Nphi / maximum harmonic: `768 / 378`
- exact-beta complex L2 max: `2.456e-12`
- direct/exact-beta speedup median: `17.907x`
- coarse 0.1 nm intensity L2 mean/max: `0.825 / 0.900`
- fused harmonic output / thread scratch upper bound: `0.0231 / 0.0178 MiB`

| frame | exact L2 | intensity L2 | direct s | exact-beta s | direct/exact | coarse intensity L2 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.538e-12 | 1.073e-12 | 4.430 | 0.238 | 18.627x | 0.787 |
| 1 | 8.088e-13 | 7.072e-13 | 4.414 | 0.235 | 18.818x | 0.831 |
| 2 | 5.399e-13 | 5.723e-13 | 4.260 | 0.237 | 17.975x | 0.900 |
| 3 | 6.355e-13 | 5.215e-13 | 4.129 | 0.230 | 17.977x | 0.802 |
| 4 | 2.976e-13 | 2.546e-13 | 4.068 | 0.233 | 17.470x | 0.811 |
| 5 | 4.013e-13 | 3.585e-13 | 4.098 | 0.235 | 17.458x | 0.839 |
| 6 | 6.339e-13 | 5.121e-13 | 4.173 | 0.235 | 17.724x | 0.841 |
| 7 | 1.760e-12 | 1.420e-12 | 4.112 | 0.232 | 17.737x | 0.881 |
| 8 | 1.886e-13 | 1.577e-13 | 4.113 | 0.233 | 17.652x | 0.857 |
| 9 | 1.087e-12 | 1.029e-12 | 4.321 | 0.230 | 18.785x | 0.810 |
| 10 | 1.552e-13 | 1.473e-13 | 4.119 | 0.233 | 17.667x | 0.860 |
| 11 | 2.212e-13 | 1.777e-13 | 4.070 | 0.235 | 17.333x | 0.758 |
| 12 | 8.122e-13 | 7.264e-13 | 4.122 | 0.236 | 17.438x | 0.799 |
| 13 | 1.386e-12 | 1.210e-12 | 4.155 | 0.232 | 17.885x | 0.829 |
| 14 | 7.226e-13 | 6.363e-13 | 4.072 | 0.227 | 17.928x | 0.828 |
| 15 | 2.489e-13 | 2.259e-13 | 4.147 | 0.255 | 16.242x | 0.838 |
| 16 | 1.338e-12 | 1.131e-12 | 4.253 | 0.236 | 18.031x | 0.780 |
| 17 | 3.564e-13 | 3.028e-13 | 4.240 | 0.235 | 18.011x | 0.787 |
| 18 | 2.456e-12 | 2.160e-12 | 4.675 | 0.257 | 18.216x | 0.807 |
| 19 | 4.121e-13 | 3.939e-13 | 4.381 | 0.232 | 18.871x | 0.863 |

- dense 20-frame correctness gate: **PASS**
- Direct NDFT is the small-case correctness oracle; FINUFFT is not used as truth here.
- Timings are local CPU wall times for this machine and workload, not Nq=512 production evidence.
- The coarse row records the known 0.1 nm whole-object representation failure at high q.
