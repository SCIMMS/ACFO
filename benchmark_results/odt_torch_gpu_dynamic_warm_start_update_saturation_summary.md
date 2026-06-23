# ODT GPU dynamic warm-start benchmark

This benchmark keeps the realistic ring-plus-axis ODT geometry fixed and changes only the synthetic object over a time-lapse sequence. Warm-start rows reuse both the previous reconstruction `x` and the previous prediction `A x`, so each new frame only pays the requested number of update steps.

## Configuration

- device: `NVIDIA GeForce RTX 2070 SUPER`
- torch: `2.12.1+cu126`
- dtype: `complex64`
- total illuminations: `101`
- total q samples: `6619136`
- object bins: `92160`
- cap: `128 x 512`
- frames: `15`
- target FPS: `30.0`
- target frame budget: `33.33` ms
- initial mode: `cold_start`
- initial iterations: `10`
- warmup updates: `3`
- update counts: `[1, 2, 3, 5, 8, 12, 16, 24, 32]`
- synthetic motion fraction: `0.08`
- synthetic phase drift rad: `0.12`
- GPU basis memory: `36.806 MiB`
- GPU peak allocated: `584.33837890625` MiB

## Aggregate Results

| mode | updates/frame | median latency ms | FPS excl. synthetic data | mean object rel-L2 | final object rel-L2 | mean loss rel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| reference | 32 | 254.96 | 3.9 | 0.336 | 0.3388 | 0.05059 |
| warm_start | 1 | 7.48 | 133.7 | 0.4086 | 0.3785 | 0.1426 |
| warm_start | 2 | 15.32 | 65.3 | 0.3662 | 0.3396 | 0.08798 |
| warm_start | 3 | 23.44 | 42.7 | 0.362 | 0.3361 | 0.08834 |
| warm_start | 5 | 36.91 | 27.1 | 0.3373 | 0.3127 | 0.06512 |
| warm_start | 8 | 59.32 | 16.9 | 0.3174 | 0.2946 | 0.04844 |
| warm_start | 12 | 89.30 | 11.2 | 0.3022 | 0.2807 | 0.03782 |
| warm_start | 16 | 127.38 | 7.9 | 0.2922 | 0.2723 | 0.03123 |
| warm_start | 24 | 190.80 | 5.2 | 0.2795 | 0.2629 | 0.02336 |
| warm_start | 32 | 254.02 | 3.9 | 0.2718 | 0.2578 | 0.01877 |

## Warm-Start Readout

- `1` update/frame: median latency `7.48` ms, throughput `133.7` FPS, mean object error `0.4086`; warm/cold error ratio `n/a`.
- `2` update/frame: median latency `15.32` ms, throughput `65.3` FPS, mean object error `0.3662`; warm/cold error ratio `n/a`.
- `3` update/frame: median latency `23.44` ms, throughput `42.7` FPS, mean object error `0.362`; warm/cold error ratio `n/a`.
- `5` update/frame: median latency `36.91` ms, throughput `27.1` FPS, mean object error `0.3373`; warm/cold error ratio `n/a`.
- `8` update/frame: median latency `59.32` ms, throughput `16.9` FPS, mean object error `0.3174`; warm/cold error ratio `n/a`.
- `12` update/frame: median latency `89.30` ms, throughput `11.2` FPS, mean object error `0.3022`; warm/cold error ratio `n/a`.
- `16` update/frame: median latency `127.38` ms, throughput `7.9` FPS, mean object error `0.2922`; warm/cold error ratio `n/a`.
- `24` update/frame: median latency `190.80` ms, throughput `5.2` FPS, mean object error `0.2795`; warm/cold error ratio `n/a`.
- `32` update/frame: median latency `254.02` ms, throughput `3.9` FPS, mean object error `0.2718`; warm/cold error ratio `n/a`.

- figure: `benchmark_results\odt_torch_gpu_dynamic_warm_start_update_saturation.png`

## Interpretation

- The latency columns exclude synthetic data generation because real acquisition would provide the measured field; the synthetic forward time is recorded separately in the CSV/JSON.
- This is still a PyTorch tensor prototype. It demonstrates the algorithmic warm-start path, not the final low-level CUDA ceiling.
