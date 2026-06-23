# Public aIDT Torch GPU optimization summary

Generated: 2026-06-23

This note summarizes the optimization pass on the public aIDT Diatom I
`700 x 700` real-condition transfer-function reconstruction.

## Baseline

The starting GPU path was a dense geometry-cache implementation:

- CPU streaming reference: about `110 s / volume`
- GPU setup: `0.694 s`
- GPU hot-run median: `0.446 s`
- GPU RHS median: `0.388 s`
- GPU solve median: `0.051 s`
- GPU cache: `661.7 MiB`
- peak allocated: `2767 MiB`

The bottleneck was RHS assembly, which repeatedly regenerated dense PTF/ATF
transfer functions for all detector pixels even though most pixels were outside
the active pupil support.

## Optimizations Applied

| step | what changed | result |
| --- | --- | --- |
| active support probing | measured the union of source-shifted pupil supports | only `26.4%` of detector pixels are active per illumination |
| support RHS mode | compacted per-frame geometry to active support and scattered into the full RHS | removed most zero-support trigonometric work |
| cached support transfer | precomputed conjugated PTF/ATF on the active support | removed hot-loop transfer regeneration |
| skipped output diagnostics | avoided min/max reductions when only reconstruction output and rel-L2 are needed | separated production-like hot timing from reporting diagnostics |

## Full `700 x 700` Results

| mode | setup s | run median s | RHS s | solve s | cache MiB | peak allocated MiB | speedup vs dense GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense baseline | 0.694 | 0.446 | 0.388 | 0.051 | 661.7 | 2767.0 | 1.00x |
| support RHS | 0.656 | 0.166 | 0.110 | 0.049 | 487.4 | 2332.9 | 2.69x |
| support + transfer cache | 0.871 | 0.114 | 0.052 | 0.055 | 2076.2 | 3849.4 | 3.92x |
| support + transfer cache + no output stats | 0.794 | 0.0975 | 0.0465 | 0.0456 | 2076.2 | 3849.4 | 4.58x |

The final optimized run includes CPU comparison:

- CPU streaming reference: `109.75 s`
- optimized GPU run median: `0.09747 s`
- optimized GPU throughput after setup: `10.26 volumes/s`
- run speedup vs CPU streaming: `1126.0x`
- setup+run speedup for one volume: `123.0x`
- rel-L2 vs CPU: `n_re 2.99e-8`, `n_im 8.15e-6`

## Repeated-Volume Readout

| volumes | CPU streaming total | optimized GPU total including setup | effective speedup |
| ---: | ---: | ---: | ---: |
| 1 | `1.83 min` | `0.89 s` | `123.0x` |
| 10 | `18.29 min` | `1.77 s` | `620.3x` |
| 100 | `3.05 h` | `10.54 s` | `1041.1x` |
| 1000 | `30.49 h` | `1.64 min` | `1116.9x` |

## Interpretation

This optimization changes the claim boundary. Before this pass, full-size
`700 x 700` aIDT was a strong high-throughput/offline result but did not reach
`10 Hz` on the local RTX 2070 SUPER. With support transfer caching, the
computational reconstruction core now crosses the `10 Hz` threshold on the same
GPU.

The remaining caveat is that this timing assumes the calibrated geometry and
transfer cache are already prepared and the measurement tensor is resident on
the GPU. Camera transfer, acquisition scheduling, and experiment-specific
preprocessing still need to be measured before claiming end-to-end live
microscope throughput.

## Remaining Headroom

The final hot path is split between:

- RHS scatter/multiply: `0.0465 s`
- frequency solve and inverse FFTs: `0.0456 s`

The next meaningful optimizations are therefore lower-level:

1. fuse cached-transfer multiply and scatter into a custom CUDA/Triton kernel;
2. reduce scatter overhead by grouping active support into row-contiguous spans;
3. fuse or specialize the frequency-domain solve and inverse-transform path;
4. use CUDA graphs for repeated fixed-geometry volume processing.

Solve-coefficient caching was tested but was not promoted: it increased memory
and was slower in the full support-transfer path on this GPU.

## Artifacts

- `benchmark_results/aidt_public_transfer_torch_gpu_optimized_full700_compare.json`
- `benchmark_results/aidt_public_transfer_torch_gpu_optimized_full700_compare.md`
- `benchmark_results/aidt_public_transfer_torch_gpu_support_full700_rerun.json`
- `benchmark_results/aidt_public_transfer_torch_gpu_support_transfer_full700.json`
- `scripts/benchmark_aidt_transfer_torch_gpu.py`
