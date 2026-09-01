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
| GPU setup | 0.914360 |
| GPU run median | 0.097002 |
| GPU FFT median | 0.004868 |
| GPU RHS median | 0.046507 |
| GPU solve median | 0.045502 |

## Memory

| quantity | MiB |
| --- | ---: |
| geometry cache estimate | 2076.213 |
| Torch peak allocated | 3849.371 |
| Torch peak reserved | 4996.000 |
