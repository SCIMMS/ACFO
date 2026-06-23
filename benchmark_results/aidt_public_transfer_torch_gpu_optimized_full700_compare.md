# Public aIDT Torch GPU geometry-cache benchmark

This benchmark runs the public aIDT PTF/ATF transfer reconstruction with the geometry-cache path on Torch/CUDA.

## Configuration

| key | value |
| --- | --- |
| `crop_size` | `700` |
| `processed_shape` | `[24, 700, 700]` |
| `depth_count` | `35` |
| `n_illum` | `24` |
| `device_name` | `NVIDIA GeForce RTX 2070 SUPER` |
| `dtype` | `complex64` |
| `fft_norm` | `ortho` |
| `rhs_mode` | `support` |
| `cache_support_transfer` | `True` |
| `cache_solve_coeffs` | `False` |
| `collect_output_stats` | `False` |
| `support_active_fraction` | `0.26430408163265307` |
| `alpha` | `100.0` |
| `beta` | `100.0` |

## Runtime

| stage | seconds |
| --- | ---: |
| GPU setup | 0.794498 |
| GPU run median | 0.097470 |
| GPU FFT median | 0.004947 |
| GPU RHS median | 0.046545 |
| GPU solve median | 0.045624 |
| CPU streaming reference | 109.747671 |

## Accuracy vs CPU Reference

| quantity | rel-L2 |
| --- | ---: |
| `n_re` | 2.99028e-08 |
| `n_im` | 8.15352e-06 |
| `v_re` | 8.1187e-06 |
| `v_im` | 8.15293e-06 |

## Memory

| quantity | MiB |
| --- | ---: |
| geometry cache estimate | 2076.213 |
| Torch peak allocated | 3849.371 |
| Torch peak reserved | 4996.000 |

## Repeated-Volume Amortization

These totals use the measured setup and median hot-run values from this benchmark.

| volumes | CPU streaming total | GPU total including one setup | effective speedup |
| ---: | ---: | ---: | ---: |
| 1 | `1.83 min` | `0.89 s` | `123.0x` |
| 10 | `18.29 min` | `1.77 s` | `620.3x` |
| 100 | `3.05 h` | `10.54 s` | `1041.1x` |
| 1000 | `30.49 h` | `1.64 min` | `1116.9x` |

## Optimization Readout

- GPU throughput after setup: `10.26 volumes/s`.
- Active support fraction: `0.264`.
- RHS mode: `support`.
- Cached support transfer: `True`.
- Output min/max diagnostics collected: `False`.
- This timing is the computational reconstruction core with data already on the GPU; camera transfer and acquisition-side preprocessing are not included.
