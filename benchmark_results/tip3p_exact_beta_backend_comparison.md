# TIP3P exact-beta backend comparison

- atoms: `50,430`
- q rows / active detector targets: `8 / 2720`
- maximum harmonic / detector Nyquist: `374 / 384`

| backend | time s | direct/backend | complex L2 vs direct | intensity L2 vs direct |
|---|---:|---:|---:|---:|
| direct NDFT | 14.202 | 1.000x | oracle | oracle |
| cpp_miller | 18.317 | 0.775x | 2.381e-12 | 2.008e-12 |
| cpp_fused | 0.607 | 23.395x | 2.381e-12 | 2.009e-12 |

- fused speedup vs cpp_miller: `30.173x`
- comparison gate: **PASS**
- Both optimized rows use the same cutoff, form factors, coordinates, and detector samples.
- Direct NDFT is the correctness oracle; this is a local CPU wall-time snapshot.
