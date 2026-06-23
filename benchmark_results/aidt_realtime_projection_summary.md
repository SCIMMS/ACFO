# Public aIDT Real-Time Projection

Generated: 2026-06-23

This projection starts from measured local results on the
`NVIDIA GeForce RTX 2070 SUPER` for the full public aIDT condition:

- detector and frame stack: `24 x 700 x 700`
- reconstruction grid: `700 x 700 x 35`
- mode: Torch/CUDA support RHS with cached support transfer
- diagnostics: disabled
- dtype: `complex64`

The purpose is not to claim that end-to-end real-time analysis has already
been demonstrated. The purpose is to quantify how close the measured
pipeline is, and how much speedup or overlap is required to make
real-time analysis practical.

## Measured Baseline

| component | seconds | share of GPU core |
| --- | ---: | ---: |
| measurement FFT | 0.004868 | 5.0% |
| RHS assembly | 0.046507 | 47.9% |
| solve + inverse FFT | 0.045502 | 46.9% |
| other measured overhead | 0.000125 | 0.1% |
| GPU-resident reconstruction core | 0.097002 | 100.0% |
| host-to-GPU copy, preallocated tensor | 0.009735 | n/a |

Current full-condition rates:

| condition | seconds/update | Hz | readout |
| --- | ---: | ---: | --- |
| GPU-resident core | 0.0970 | 10.31 | already above 10 Hz |
| copy + core, sequential | 0.1066 | 9.38 | below 10 Hz |
| copy/core fully overlapped | 0.0970 | 10.31 | above 10 Hz if transfer is hidden |

## Required Speedup

For the measured sequential copy+core path to reach 10 Hz at the original
`700 x 700 x 35` condition:

- If H2D copy stays unchanged, the GPU core needs only `1.075x` speedup.
- If the GPU core stays unchanged, H2D copy alone would need `3.25x` speedup,
  because the current core is already close to the full 100 ms budget.
- If H2D copy is overlapped with reconstruction, the measured core already
  meets the 10 Hz budget.

This is the main practical point: full-condition real-time analysis is not
orders of magnitude away. On the measured older GPU, the sequential
copy-included loop is about 6.6 ms over the 100 ms budget.

## Single-GPU Projection

This table keeps the measured H2D copy cost fixed and scales only the GPU
reconstruction core. It is a required-speedup projection, not a benchmark
of any specific new GPU.

| GPU core speedup | sequential copy+core s | Hz | margin vs 10 Hz |
| ---: | ---: | ---: | ---: |
| 1.00x | 0.1067 | 9.37 | -6.7 ms |
| 1.05x | 0.1021 | 9.79 | -2.1 ms |
| 1.08x | 0.0996 | 10.04 | +0.4 ms |
| 1.10x | 0.0979 | 10.21 | +2.1 ms |
| 1.25x | 0.0873 | 11.45 | +12.7 ms |
| 1.50x | 0.0744 | 13.44 | +25.6 ms |
| 2.00x | 0.0582 | 17.17 | +41.8 ms |
| 3.00x | 0.0421 | 23.77 | +57.9 ms |

## Transfer Improvement Projection

This table keeps the measured GPU core fixed and improves only H2D copy.
It shows that copy acceleration alone is not the cleanest path for the
full 35-slice condition; overlap or compute improvement is more useful.

| H2D copy speedup | sequential copy+core s | Hz | margin vs 10 Hz |
| ---: | ---: | ---: | ---: |
| 1.00x | 0.1067 | 9.37 | -6.7 ms |
| 1.50x | 0.1035 | 9.66 | -3.5 ms |
| 2.00x | 0.1019 | 9.82 | -1.9 ms |
| 3.00x | 0.1002 | 9.98 | -0.2 ms |

## Multi-GPU RHS-Partition Projection

The RHS assembly is the largest single hot-stage and is naturally
partitionable over illumination frames or support blocks. The conservative
projection below scales only the RHS stage across GPUs and leaves H2D copy,
measurement FFT, solve, and overhead unchanged on the measured baseline.
It does not assume a fully distributed inverse FFT or solve.

| GPUs | projected s/update | Hz | margin vs 10 Hz |
| ---: | ---: | ---: | ---: |
| 1 | 0.1067 | 9.37 | -6.7 ms |
| 2 | 0.0835 | 11.98 | +16.5 ms |
| 4 | 0.0719 | 13.92 | +28.1 ms |
| 8 | 0.0660 | 15.14 | +34.0 ms |

## Claim Boundary

A defensible manuscript claim is:

> On an older RTX 2070 SUPER, the prepared GPU operator already reaches
> 10 Hz for the GPU-resident full public aIDT reconstruction core and is
> within a 1.07x compute-speedup of 10 Hz even when ordinary host-to-GPU
> transfer is included. This supports the practical feasibility of
> real-time analysis on current-generation GPUs, overlapped acquisition
> pipelines, or modest multi-GPU deployments.

The claim should not be phrased as a completed end-to-end live microscope
demonstration until acquisition, preprocessing, and transfer scheduling are
measured in the target experimental system.
