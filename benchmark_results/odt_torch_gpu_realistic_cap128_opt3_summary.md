# ODT Torch GPU reconstruction prototype

This is a tensor-resident GPU prototype for the cone-axis ODT operator. It uses the same ring-plus-axis realistic geometry as the CPU benchmark, and evaluates the harmonic/L-mode contractions with PyTorch tensors. The adjoint decompose stage is reordered into a matmul-centered contraction to avoid the slow four-operand einsum path.

## Configuration

- device: `NVIDIA GeForce RTX 2070 SUPER`
- dtype: `complex64`
- total q samples: `6619136`
- cap: `128 x 512`
- total illuminations: `101`
- object bins: `92160`
- GPU basis memory: `36.806 MiB`
- GPU peak allocated: `583.63623046875` MiB

## Hot Timings

| path | median s |
| --- | ---: |
| forward | 0.002762 |
| adjoint | 0.002703 |
| forward after adjoint pair | 0.004584 |
| one-step update without diagnostics | 0.006429 |
| one iteration | 0.009076 |

## Reconstruction

- iterations: `2`
- final loss rel: `0.542004`
- final object rel-L2: `0.74696`
- CPU/GPU forward rel-L2: `3.2567579288252275e-07`
- CPU/GPU adjoint rel-L2: `1.8886600804026003e-07`

## Readout

- This prototype is still PyTorch-level GPU code, but the dominant adjoint decompose contraction now uses a matmul-centered layout.
- The next GPU step is a fused CUDA/Triton kernel for the remaining compact adjoint and update steps, plus a pruned tensor layout where the active L support is sparse.
