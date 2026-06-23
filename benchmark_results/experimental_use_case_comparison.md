# Experimentally meaningful use-case comparison

Generated: 2026-06-23

This note separates two questions:

1. Which experimental ODT/IDT use cases are meaningful enough to motivate the
   algorithm?
2. Among those use cases, which ones can be compared with the current prepared
   curved-Ewald / transfer-function implementation without changing the claim?

The purpose is not another low-level optimization pass. The purpose is to pick
benchmarks whose geometry, repetition pattern, and experimental demand make the
prepared-operator speedup scientifically useful.

## Current measured anchor

The strongest measured anchor is the public annular intensity diffraction
tomography (aIDT) Diatom I condition.

| field | value |
| --- | ---: |
| data contract | `benchmark_results/aidt_diatom_public_contract.npz` |
| detector | `700 x 700` |
| illumination frames | 24 |
| wavelength | `0.515 um` |
| objective/source NA | 0.65 |
| z slices | 35 |
| local GPU | RTX 2070 SUPER, 8 GB |
| corrected CPU streaming reference | `109.75 s` |
| Torch GPU optimized setup | `0.794 s` |
| Torch GPU optimized run median | `0.0975 s` |
| optimized run speedup vs corrected CPU streaming | `1126.0x` |
| optimized setup+run speedup for one volume | `123.0x` |
| optimized GPU geometry/transfer cache | `2076.2 MiB` |
| optimized GPU peak allocated | `3849.4 MiB` |
| rel-L2 vs corrected CPU | `n_re 2.99e-8`, `n_im 8.15e-6` |
| optimized mode | support RHS, cached support transfer, output min/max diagnostics off |

This is the first real-condition result that is both experimentally grounded
and fast enough to support a practical claim.

## Use cases found

| use case | experimental need | geometry repetition | current match | claim strength |
| --- | --- | --- | --- | --- |
| aIDT live / repeated-volume RI imaging | Large-volume, label-free 3D RI imaging at high volume rate; the reported aIDT setup targets dynamic biological samples with annular illumination. | Very high. Same calibrated source ring, pupil, detector, and z grid are reused across volumes. | Direct. We already ingest the public aIDT Diatom I data and solve the same transfer-function reconstruction on the original `700 x 700` condition. | Primary demo. |
| DMD-based 4D ODT / live-cell RI tomography | Multiple illumination angles are measured repeatedly for time-dependent live cells. | High. Illumination directions and detector calibration are fixed for a time-lapse sequence. | Medium. Current synthetic cone-axis ODT operator fits repeated forward/adjoint geometry, but direct public measured-data benchmark is not complete. | Secondary demo. |
| FS-ODT / multiplexed high-speed ODT | High-speed 3D quantitative phase imaging; multiplexed illumination patterns target volumetric dynamics. | High. Pattern set and illumination geometry are fixed, so prepared operators can be reused over frames and iterations. | Medium. Current synthetic fixed-pattern loop exists; real FS-ODT ingestion is not implemented. | Secondary or future demo. |
| FPDT / Fourier ptychographic diffraction tomography | High-throughput wide-FOV 3D RI imaging with many variably illuminated low-resolution images. | Medium-high. LED/source geometry repeats, but ptychographic updates include object/pupil interactions outside the current linear transfer-function path. | Partial. Useful as motivation and future extension, not the cleanest current comparison. | Supporting context. |
| Single-frame tomographic cytometry / xSCYTE-like flow use | Throughput and label-free 3D cell phenotyping are experimentally important. | Medium. Acquisition can be repeated over many cells, but the published model uses a specialized single-frame tomographic phase route. | Indirect. Strong application motivation but not a direct operator comparison yet. | Long-term application. |

## Source-backed experimental relevance

- aIDT is the most direct source-backed use case. Li et al. report annular
  intensity diffraction tomography with 8 intensity images, 10 Hz volume-rate
  motivation, large in-vitro volumes, and public example data/code:
  https://arxiv.org/abs/1904.06004
- DMD ODT is a direct experimental architecture for repeated 3D/4D RI maps of
  live cells under controlled illumination angles:
  https://arxiv.org/abs/1602.03294
- FPDT gives the high-throughput, wide-field, high-resolution 3D microscopy
  motivation, but it is ptychographic and less directly matched to the current
  linear transfer-function benchmark:
  https://arxiv.org/abs/1904.09386
- High-speed Fourier ptychography shows why acquisition speed and repeated
  computational reconstruction matter in large-space-bandwidth microscopy:
  https://arxiv.org/abs/1506.04274
- FS-ODT is a strong high-speed ODT motivation because it targets kilohertz-rate
  volumetric quantitative phase imaging with multiplexed illumination:
  https://arxiv.org/abs/2309.16912
- xSCYTE-like single-frame cytometry is a strong application pull for high
  throughput, but it is not yet the cleanest validation route for this operator:
  https://arxiv.org/abs/2202.03627

## Repeated-volume amortization

Using the measured full `700 x 700` aIDT condition:

| volumes | CPU streaming total | GPU total including one setup | effective speedup |
| ---: | ---: | ---: | ---: |
| 1 | `1.83 min` | `0.89 s` | `123.0x` |
| 10 | `18.29 min` | `1.77 s` | `620.3x` |
| 100 | `3.05 h` | `10.54 s` | `1041.1x` |
| 1000 | `30.49 h` | `1.64 min` | `1116.9x` |

This is the strongest practical story. Once the geometry is fixed, setup cost is
almost irrelevant after a few volumes. A time-lapse or high-throughput dataset
is therefore a better benchmark than a one-shot reconstruction.

## Real-time readout

The optimized full `700 x 700` GPU computational core runs at about
`10.26 volumes/s` on an RTX 2070 SUPER when the calibrated geometry and support
transfer cache are prepared and output min/max diagnostics are skipped. This
crosses the `10 Hz` reconstruction-core target at the original detector size on
the local GPU.

However:

- camera-to-GPU transfer, acquisition scheduling, and any experimental
  preprocessing are not included in this computational-core timing;
- the optimized path uses about `2.08 GiB` of geometry/transfer cache and about
  `3.85 GiB` peak allocated GPU memory on this benchmark;
- the non-transfer-cache support path remains useful when memory is tighter,
  but its full-size run time is about `0.155 s`.

So the current defensible claim is:

> Prepared geometry-cache evaluation turns the public real-condition aIDT
> reconstruction from a minutes-scale CPU workload into a `10 Hz`-class GPU
> computational core for repeated fixed-geometry volumes, with CPU-level
> numerical agreement.

## Comparison decision

| rank | benchmark | why it should be done |
| ---: | --- | --- |
| 1 | Public aIDT repeated-volume benchmark, `N = 1, 10, 100, 1000` volumes | Direct data, direct equation, direct measured speedup, strong setup amortization. |
| 2 | Public aIDT ROI/throughput sweep, for example `128, 256, 512, 700` | Shows where live-rate reconstruction starts and where full-size offline throughput dominates. |
| 3 | DMD/ODT or FS-ODT synthetic-to-contract benchmark with fixed geometry | Shows the same prepared-operator principle in an iterative inverse/backpropagation setting. |
| 4 | cuFINUFFT / NFFT-style baseline on the same measured target | Necessary for paper-grade fairness because DT literature already uses NDFT/NFFT/backprop baselines. |
| 5 | FPDT/xSCYTE-style discussion or adapter | Good motivation, but too indirect for the main claim until the measurement model is implemented. |

## Recommended manuscript positioning

The most defensible story is:

1. WAXS validates the factorization and scaling on a physically familiar
   scattering problem.
2. High-NA optics connects the method to known circular-harmonic reductions in
   wave optics, without claiming that identity as new.
3. aIDT/ODT supplies the experimentally meaningful repeated-operator workload:
   fixed lab geometry, repeated volumes, iterative or streaming reconstruction,
   and strong setup amortization.

The main ODT/aIDT claim should be geometry-specific:

> The method is most valuable when the lab geometry is fixed and many volumes,
> updates, or ROIs are reconstructed under the same curved-Ewald or cap sampling
> structure.

It should not be framed as a universal replacement for NUFFT on arbitrary
unstructured point clouds.

## Next concrete benchmark

Run a hot-repeat aIDT benchmark using the public `700 x 700` geometry:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_aidt_transfer_torch_gpu.py `
  --contract benchmark_results\aidt_diatom_public_contract.npz `
  --crop-size 700 `
  --repeats 20 `
  --compare-cpu `
  --out benchmark_results\aidt_public_repeated_volume_gpu.json `
  --summary-md benchmark_results\aidt_public_repeated_volume_gpu.md
```

Then postprocess the measured setup time, CPU reference time, and median GPU
run time into `N = 1, 10, 100, 1000` volume totals. If a dedicated
`--volume-counts` option is useful later, it should be a thin reporting layer
over the existing geometry cache and should not change the solver math.
