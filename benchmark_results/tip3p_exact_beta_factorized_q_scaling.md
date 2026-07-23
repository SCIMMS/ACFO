# TIP3P factorized exact-beta q-scaling

- atoms: `50,430`
- q range: `[5.0, 6.3]` inverse angstrom
- detector Nphi: `768`
- factorized vs expanded q=2 complex L2: `0.000e+00`

| Nq | targets | max h | median s | min-max s | atom-by-q matrix MiB avoided | accounted arrays MiB |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1,536 | 374 | 0.226 | 0.221-0.227 | 1.5 | 6.8 |
| 8 | 6,144 | 374 | 0.512 | 0.509-0.574 | 6.2 | 6.9 |
| 32 | 24,576 | 374 | 2.044 | 2.041-2.158 | 24.6 | 7.5 |
| 128 | 98,304 | 374 | 8.082 | 8.082-8.082 | 98.5 | 9.7 |
| 512 | 393,216 | 374 | 34.059 | 34.059-34.059 | 394.0 | 18.6 |

- q=512 runtime gate <= 120 s: **PASS**
- q=512 accounted arrays <= 64 MiB: **PASS**
- Timings include coordinate preprocessing, fused contraction, and harmonic evaluation, but exclude file loading and form-factor construction.
- Accounted arrays are not peak RSS; allocator, interpreter, extension, and input-file overhead are excluded.
- This is a local CPU measurement for one TIP3P frame, not a hardware-independent speed claim.
