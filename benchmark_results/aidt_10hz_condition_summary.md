# Public aIDT 10 Hz Condition Check

Generated: 2026-06-23

This note records the conditions under which the public aIDT Diatom I
transfer-function reconstruction reaches a 10 Hz update rate on the local
`NVIDIA GeForce RTX 2070 SUPER`.

The benchmark uses the Torch/CUDA support-transfer path:

- measurement: public aIDT Diatom I contract
- illumination frames: 24
- detector condition: original `700 x 700` unless noted
- dtype: `complex64`
- reconstruction mode: support RHS
- cached support transfer: enabled unless noted
- solve coefficient cache: disabled
- FFT normalization: `ortho`

## Results

| condition | z slices | diagnostics | H2D copy included | median s/update | Hz | readout |
| --- | ---: | --- | --- | ---: | ---: | --- |
| `700 x 700`, optimized hot core | 35 | off | no | 0.0970 | 10.31 | crosses 10 Hz, but with narrow margin |
| `700 x 700`, optimized plus output stats | 35 | on | no | 0.1012 | 9.88 | falls just below 10 Hz |
| `700 x 700`, support RHS without transfer cache | 35 | off | no | 0.1655 | 6.04 | transfer cache is required for 10 Hz |
| `700 x 700`, optimized, coarser z | 18 | off | no | 0.0537 | 18.61 | robustly above 10 Hz |
| `700 x 700`, coarser z plus output stats | 18 | on | no | 0.0582 | 17.18 | robustly above 10 Hz |
| `512 x 512`, optimized hot core | 35 | off | no | 0.0501 | 19.97 | robustly above 10 Hz |
| `700 x 700`, optimized with host-to-GPU copy | 35 | off | yes | 0.1066 | 9.38 | below 10 Hz if each volume starts on CPU |
| `700 x 700`, coarser z with host-to-GPU copy | 18 | off | yes | 0.0625 | 15.99 | robustly above 10 Hz with CPU-to-GPU copy |
| `512 x 512`, optimized with host-to-GPU copy | 35 | off | yes | 0.0556 | 18.00 | robustly above 10 Hz with CPU-to-GPU copy |

## Interpretation

The original full public condition, `700 x 700` detector with 35 z slices, does
reach 10 Hz for the GPU-resident computational reconstruction core. The margin
is small: enabling output diagnostics or adding ordinary host-to-GPU copy pushes
the measured loop below 10 Hz on the RTX 2070 SUPER.

The practical 10 Hz conditions on this GPU are therefore:

1. GPU-resident full `700 x 700 x 35` updates, with diagnostics disabled; or
2. full `700 x 700` detector with a coarser 18-slice z grid, even with
   host-to-GPU copy; or
3. reduced detector crop such as `512 x 512 x 35`, even with host-to-GPU copy.

For a strict end-to-end real-time microscope claim at the original full
`700 x 700 x 35` condition, acquisition transfer and preprocessing must be moved
onto the GPU or overlapped with reconstruction. Without that pipeline work, the
stronger claim is a 10 Hz GPU-resident reconstruction core.

## Measured Transfer and Preprocessing Costs

For the full `24 x 700 x 700` float32 measurement tensor:

- tensor size: `44.86 MiB`
- host-to-GPU copy into a preallocated CUDA tensor: `0.0099 s` median
- fresh CUDA tensor allocation plus copy: `0.0116 s` median
- current CPU preprocessing from the resident public contract object:
  `0.1445 s` median

The CPU preprocessing number is not a hard microscope limit because the live
camera path may not use the same public-data frame-mean preprocessing, but it is
large enough that the current Python CPU preprocessing path should not be
included in a 10 Hz end-to-end claim.

## Artifacts

- `aidt_10hz_full700_opt_repeat.json`
- `aidt_10hz_full700_with_stats.json`
- `aidt_10hz_full700_support_no_transfer_cache.json`
- `aidt_10hz_full700_zstep3_opt.json`
- `aidt_10hz_full700_zstep3_with_stats.json`
- `aidt_10hz_crop512_fullz_opt.json`
