# Uniaxial PyMeep 3-D asymptotic amplitude decision

- scoped single-component amplitude: **PASS**
- publication full amplitude: **FAIL**

| resolution | extraordinary L2 | NCC | peak error | ordinary calibration | radial scatter | forced ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 12.046% | 0.961174 | 0.0 deg | 6.338% | 1.383% | 1.050 |
| 12 | 7.482% | 0.999216 | 0.0 deg | 3.562% | 0.458% | 0.974 |
| 16 | 4.992% | 0.992041 | 2.0 deg | 5.423% | 0.522% | 1.059 |

- calibrated field L2, 8->12: `0.087346`
- calibrated field L2, 12->16: `0.066702`
- empirical error order: `1.264`
- matched-geometry nonlinear->single-Ex L2 improvement: `15.05x`

The single-Ex/45-deg bridge reaches 4.992% extraordinary complex L2 at resolution 16 with NCC 0.992 and 2-deg peak error, but the 12-to-16 field change is 6.67%, ordinary calibration is 5.42%, and the forced-sphere ratio is 1.06. Therefore this is a scoped representation diagnostic, not a publication-grade full nonlinear-source amplitude PASS.

The next run must change the source/reference contract, not only the domain size.
