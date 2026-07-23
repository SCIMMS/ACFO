# Component-aware PyMeep amplitude decision

- detectable-support amplitude: **PASS**
- grid-converged detectable support: **FAIL**
- publication full amplitude: **FAIL**

| case | support | complex L2 | NCC | peak error | ordinary L2 |
|---|---|---:|---:|---:|---:|
| nonlinear_r12 | 20-70 deg | 6.493% | 0.989742 | 0.0 deg | 1.274% |
| nonlinear_r16 | 16-70 deg | 1.286% | 0.996522 | 14.0 deg | 1.357% |
| singlex_r16 | 10-70 deg | 4.625% | 0.999633 | 0.0 deg | 1.881% |

- FDTD field grid L2, r12->r16: `0.417534`
- source-oracle grid L2, r12->r16: `0.373228`

The exact Yee-source oracle removes the scalar-source mismatch. On the fixed 10% detectable support, the resolution-16 nonlinear source reaches 1.286% complex L2, 0.9965 NCC, and 1.357% ordinary calibration. The raw peak error is retained, but the reference top-two peak margin is only 0.0835%, so peak location is not identifiable for this scoped row. The source oracle and FDTD field are not grid converged, and the forced-sphere publication gate remains unresolved.
