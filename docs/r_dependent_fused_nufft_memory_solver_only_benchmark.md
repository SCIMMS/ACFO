# R-Dependent Fused vs NUFFT Memory and Solver-Only Benchmark

Date: 2026-06-16

This benchmark checks the R-dependent fused analytic cake-map path against the
memory-safe chunked FINUFFT baseline, while separating representation build time
from solver time.

Representation build means:

- cylindrical histogram construction, `hist_s`;
- `PreparedCakePlan` construction, `plan_s`.

Solver-only time means:

- `r_dependent_cake_first_s` or `r_dependent_cake_cached_s` for the fused
  R-dependent path;
- `nufft_s` for FINUFFT, which starts from raw coordinates and detector targets
  and has no binned representation stage.

Peak memory is the sampled peak RSS delta during the timed solve call. It is not
the full process RSS and does not include persistent representation memory
already resident before the timed solve starts.

## Command

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_physical_scaling.py --atoms 100000 1000000 --qmax 6.3 --q-unit inv_angstrom --nq 40 --repeats 1 --measure-memory --benchmark-r-dependent-cake --r-dependent-margin 16 --r-dependent-cutoff-bin-size 16 --r-dependent-analytic-kernel --r-dependent-fused-analytic-kernel --nufft-q-block-size 2 --out benchmark_results\physical_scaling_highq_rdep_fused_nufft_memory_solver_only.json
```

## Results

High-q case: `qmax = 6.3 A^-1`, `nq = 40`, `bin_width = 0.1 nm`,
`margin = 16`, `cutoff_bin_size = 16`, C++ backend, float32 histogram, fused
Miller analytic kernel.

| atoms | n_phi | hist+plan | fused total | fused first solve | fused cached solve | fused peak delta | NUFFT | NUFFT peak delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100k | 960 | 0.039 s | 0.084 s | 0.045 s | 0.043 s | 0.5 MiB | 6.374 s | 484.3 MiB |
| 1M | 2000 | 0.177 s | 0.505 s | 0.329 s | 0.306 s | 0.5 MiB | 32.499 s | 2142.7 MiB |

| atoms | fused total speedup vs NUFFT | fused first-solve speedup vs NUFFT | fused cached-solve speedup vs NUFFT | fused I rel L2 vs NUFFT | fused I rel L2 vs dense |
|---:|---:|---:|---:|---:|---:|
| 100k | 76.3x | 143.1x | 148.1x | 8.07e-4 | 1.78e-7 |
| 1M | 64.3x | 98.9x | 106.4x | 5.55e-4 | 4.85e-7 |

## Interpretation

The fused R-dependent path is the better high-q CPU benchmark target when the
question is solver throughput after representation construction. At 1M atoms,
the representation build is about `0.177 s`, while the fused solver call is
`0.329 s` first-call and `0.306 s` cached. Compared against chunked NUFFT,
solver-only speedup is about `99x` to `106x`.

The memory result is also decisive for the timed solve call. The fused analytic
kernel avoids materializing the compact `Khat(q,R,h)` array, so the sampled peak
RSS delta is about `0.5 MiB` in both timed cases. Chunked NUFFT avoids the old
single-call FINUFFT allocation failure, but still adds about `0.47 GiB` at 100k
and about `2.09 GiB` at 1M.

The total-time speedup is lower than solver-only speedup because total-time
includes histogram and plan construction for the WAXS-specific method. That is
the right number for one-shot fresh structures; solver-only speedup is the right
number for q/form-factor/rotation-style reuse where the representation already
exists.

Source file:

- `benchmark_results/physical_scaling_highq_rdep_fused_nufft_memory_solver_only.json`
