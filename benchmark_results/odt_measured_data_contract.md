# ODT measured-data contract for prepared curved-Ewald operators

This note defines the minimum calibrated data package needed to move the ODT
benchmarks from synthetic self-consistency data to measured experimental data.

It is a solver-facing contract, not a raw microscope-file standard. Raw camera
frames, holograms, or instrument logs should be converted into this format
before calling the prepared curved-Ewald backend.

## Contract Levels

| level | purpose | required for current claims |
| --- | --- | --- |
| L0 calibrated geometry | define wavelength, illumination, detector, and coordinate frame | yes |
| L1 prepared sampling layout | map each measured sample to the curved Ewald manifold or structured ring/cap layout | yes |
| L2 measurement data | provide complex field or intensity data in operator order | yes |
| L3 masks and uncertainty | mark invalid pixels and weight noise/variance | recommended |
| L4 reconstruction state | optional warm start, previous frame, or prior object estimate | real-time demos |

## Common Fields

The preferred portable container is `npz` or HDF5 with the following fields.
Array names use `npz` style here.

| field | shape/type | meaning |
| --- | --- | --- |
| `schema_version` | string | contract version, for example `odt-curved-ewald-v1` |
| `experiment_type` | string | `complex_odt`, `annular_idt`, `fpdt`, `fs_odt`, or `focused_raster_dt` |
| `measurement_model` | string | `complex_field`, `coherent_intensity`, `incoherent_intensity`, or `multiplexed_intensity` |
| `wavelength` or `k` | scalar | optical wavelength or wavenumber in consistent units |
| `units` | string | physical unit for spatial coordinates, for example `um` |
| `illum_dirs` | `(M, 3)` | illumination directions in the reconstruction frame |
| `detector_origin` | `(3,)` | detector origin or reference point in the lab frame |
| `detector_u` | `(3,)` | detector fast-axis unit vector |
| `detector_v` | `(3,)` | detector slow-axis unit vector |
| `detector_pixel_size` | `(2,)` | physical pixel pitch |
| `detector_distance` | scalar | sample-to-detector distance, if using detector-plane calibration |
| `q_layout` | string | `prepared_ring_stack`, `prepared_cap_stack`, `annular_cartesian_stack`, `rotational_sinogram`, or `explicit_q` |
| `data` | model-dependent | measured field or intensity in operator order |
| `mask` | same sample shape as `data` | optional valid-pixel mask |
| `variance` | same sample shape as `data` | optional noise variance or inverse confidence weights |
| `background` | same sample shape as `data` or broadcastable | optional background estimate |
| `flatfield` | same sample shape as `data` or broadcastable | optional detector flatfield correction |

## Structured Layout Fields

For the current ring/cap prepared operators, the compact representation should
prefer structured coordinates over a flattened arbitrary q cloud.

| field | shape/type | meaning |
| --- | --- | --- |
| `cap_radial` | scalar | number of radial detector samples per cap |
| `cap_phi` | scalar | number of angular detector samples per cap |
| `q_radial` | `(cap_radial,)` | radial coordinate for the cap sampling |
| `q_phi` | `(cap_phi,)` | angular coordinate for the cap sampling |
| `q_z_model` | string | full curved-Ewald relation used to build qz |
| `pattern_matrix` | `(P, M)` complex or real | fixed illumination-pattern weights |
| `pattern_model` | string | `coherent`, `incoherent`, or `demultiplexed` |

If the experiment cannot be represented by a structured layout, provide:

| field | shape/type | meaning |
| --- | --- | --- |
| `q_xyz` | `(N, 3)` | explicit reciprocal-space sample coordinates |
| `sample_index` | `(N,)` | optional mapping back to frame/pixel/pattern ids |

The explicit-q path is useful as a baseline but is not the main acceleration
regime. The manuscript claim should prefer experiments that expose repeated
ring, cap, axis, or same-direction structure.

For conventional rotational complex-field ODT sinograms, provide:

| field | shape/type | meaning |
| --- | --- | --- |
| `rotation_angles` | `(M,)` | sample rotation or illumination angle for each complex-field frame |
| `rotation_axis` | `(3,)` | axis used to interpret `rotation_angles`, when available |
| `coordinate_convention` | string | source-specific image-plane and rotation-frame convention |

The `rotational_sinogram` layout is a measured-data ingestion bridge. It
validates that public complex-field ODT data can enter the same contract, but a
direct prepared-operator speed benchmark still needs either a rotational
sinogram adapter or a resampling step into the structured curved-Ewald layout.

For annular intensity diffraction tomography or related fixed-angle intensity
stacks, provide:

| field | shape/type | meaning |
| --- | --- | --- |
| `source_na_xy` | `(M, 2)` | lateral source NA coordinates for each illumination frame |
| `objective_na` | scalar | detection objective NA |
| `frequency_x` | `(W,)` | image-plane Fourier coordinate for detector columns |
| `frequency_y` | `(H,)` | image-plane Fourier coordinate for detector rows |
| `coordinate_convention` | string | source-specific convention for source and image-plane axes |

The `annular_cartesian_stack` layout is closer to the current acceleration
target than a conventional rotational sinogram because the same calibrated
annular illumination geometry is reused across intensity frames. It still needs
a measured-data adapter that maps the Cartesian image stack into the exact
prepared curved-Ewald operator.

## Data Shapes by Experiment

| experiment | `data` shape | operator mapping |
| --- | --- | --- |
| complex ODT/FDT | `(M, R, Phi)` complex | `E_m(q) = A_m x` |
| annular IDT | `(M, H, W)` real | one intensity frame per annular illumination direction |
| FPDT coherent intensity proxy | `(P, R, Phi)` real | `I_p(q) = |sum_m P[p,m] A_m x|^2` |
| FPDT/FS-ODT incoherent proxy | `(P, R, Phi)` real | `I_p(q) = sum_m |P[p,m]|^2 |A_m x|^2` |
| demultiplexed FS-ODT | `(M, R, Phi)` complex or real | after external demultiplexing, same as ODT/FPDT |
| focused/raster DT | `(S, R, Phi)` model-dependent | scan index `S` should be grouped by shared direction when possible |

## Required Validation Checks

Before using a measured package for a benchmark claim, run these checks.

1. Geometry sanity:
   - verify units, wavelength/k, detector pixel pitch, detector distance, and coordinate handedness;
   - verify illumination directions are normalized;
   - verify q samples are inside the intended NA/q range.

2. Shape/order sanity:
   - verify `data`, `mask`, `variance`, and `pattern_matrix` agree with `M`, `P`, `R`, and `Phi`;
   - verify flattened arrays preserve the same operator order used by the prepared plan.

3. Calibration residual:
   - report `||data_measured - data_nominal|| / ||data_nominal||` when a calibration or simulated nominal target exists;
   - for intensity data, report background and flatfield correction norms separately.

4. Baseline parity:
   - compare prepared operator and cuFINUFFT on the same measured target and update rule;
   - report final loss delta and object-update delta, not only speed.

5. Reconstruction claim boundary:
   - distinguish backend speed/parity from physical reconstruction quality;
   - only claim experimental reconstruction after testing real noise, calibration drift, masks, and missing pixels.

## Current Repo Status

Implemented now:

- synthetic complex-field ODT reconstruction loop;
- synthetic warm-start dynamic ODT noise/motion sweep;
- synthetic FPDT/FS-ODT fixed-pattern intensity update loop;
- synthetic detector and pattern-calibration perturbation sweep for the fixed-pattern intensity loop.
- measured-data contract loader/validator in `src/waxs_cake/odt_measured_contract.py`;
- synthetic FS-ODT contract fixture and CLI validation smoke test;
- public ODTbrain HL60 QLSI rotational-sinogram contract conversion and strict
  validation smoke test;
- public aIDT Diatom I annular-intensity contract conversion and strict
  validation smoke test;
- direct prepared-vs-cuFINUFFT measured-target benchmark on the public aIDT
  annular-intensity contract;
- public aIDT measured-target iterative update loop comparing prepared GPU and
  cuFINUFFT GPU backends;
- public aIDT transfer-function reconstruction smoke test using the source
  pupil, phase transfer function, absorption transfer function, background
  correction, and Tikhonov-style regularization from the public implementation;
- same-equation public aIDT comparison between from-scratch streaming
  transfer-function reconstruction, a full prepared PTF/ATF transfer cache,
  a memory-reduced 2D-geometry transfer cache, and a row-blocked low-memory
  geometry path;
- public aIDT original `700 x 700` detector-size transfer benchmark, with
  geometry cache reducing repeated CPU reconstruction time from `104.5 s` to
  `33.0 s` at about `662 MiB` persistent cache;
- Torch/CUDA geometry-cache implementation for public aIDT transfer
  reconstruction, reaching `0.416 s` median run time on the original
  `700 x 700` public-data condition on an RTX 2070 SUPER, with `n_re` rel-L2
  about `3e-8` and `n_im` rel-L2 about `8e-6` versus corrected CPU reference.

Not implemented yet:

- real FPDT/FS-ODT dataset ingestion;
- direct prepared-vs-cuFINUFFT benchmark on the public rotational-sinogram
  contract;
- GPU implementation of the prepared public aIDT transfer-function
  reconstruction layer;
- further factorized transfer cache that reduces memory without reintroducing
  most of the z-dependent trigonometric RHS work;
- GPU kernel fusion and memory scheduling for full detector and larger detector
  regimes;
- MATLAB-level numerical parity check for the public aIDT transfer-function
  reconstruction;
- full ptychographic probe/position update;
- real calibration/mask/variance weighted reconstruction.

## Implemented API

The current code artifact provides:

```text
load_odt_measured_contract(path) -> OdtMeasuredData
validate_odt_measured_contract(data) -> ValidationReport
build_prepared_operator_from_contract(data) -> PreparedOperatorDescriptor
save_odt_measured_contract(path, data) -> None
```

The descriptor is intentionally lightweight. It verifies that a measured
package can be mapped to a structured or explicit-q backend, but it does not
yet instantiate the GPU/FINUFFT reconstruction operator.

## Smoke Fixture

The synthetic fixture below validates the full file round trip:

```powershell
.\.venv\Scripts\python.exe scripts\make_odt_measured_contract_fixture.py --out benchmark_results\odt_measured_contract_fixture.npz --experiment-type fs_odt --measurement-model coherent_intensity --n-illum 8 --n-patterns 4 --active-per-pattern 3 --cap-radial 8 --cap-phi 32 --include-mask --include-variance

.\.venv\Scripts\python.exe scripts\validate_odt_measured_contract.py benchmark_results\odt_measured_contract_fixture.npz --json-out benchmark_results\odt_measured_contract_fixture_validation.json --summary-md benchmark_results\odt_measured_contract_fixture_validation.md --strict
```

The fixture validates as:

- experiment type: `fs_odt`
- measurement model: `coherent_intensity`
- layout: `prepared_ring_stack`
- illumination count: `8`
- pattern count: `4`
- cap shape: `8 x 32`
- q samples per pattern: `256`
- total measured samples: `1024`
- mask and variance: present

## Public Data Smoke: ODTbrain HL60

A real public ODT/QPI dataset has also been converted into the contract:

```powershell
.\.venv\Scripts\python.exe scripts\convert_odtbrain_hl60_to_contract.py --series-h5 benchmark_results\public_data_probe\odtbrain_hl60_extracted\series.h5 --angles-txt benchmark_results\public_data_probe\odtbrain_hl60_extracted\angles.txt --out benchmark_results\odtbrain_hl60_public_contract.npz --summary-md benchmark_results\odtbrain_hl60_public_contract_summary.md

.\.venv\Scripts\python.exe scripts\validate_odt_measured_contract.py benchmark_results\odtbrain_hl60_public_contract.npz --json-out benchmark_results\odtbrain_hl60_public_contract_validation.json --summary-md benchmark_results\odtbrain_hl60_public_contract_validation.md --strict
```

The converted public dataset validates as:

- source: ODTbrain 0.4.12 `examples/data/qlsi_3d_hl60-cell_A140.tar.lzma`;
- original public DOI: `10.6084/m9.figshare.8055407.v1`;
- license in bundled readme: `CC0`;
- sample: HL60 S/4 cell;
- experiment type: `complex_odt`;
- measurement model: `complex_field`;
- layout: `rotational_sinogram`;
- data shape: `140 x 140 x 140` complex64;
- total measured samples: `2,744,000`;
- wavelength: `0.647 um`;
- pixel size: `0.139 um`;
- medium index: `1.335`.

This is real measured public data, not a synthetic fixture. Its geometry is a
conventional rotational complex-field sinogram, so it currently supports
public-data ingestion and contract validation rather than the fixed ring/cap
FPDT/FS-ODT acceleration benchmark.

## Public Data Smoke: aIDT Diatom I

A stronger public candidate for the current ODT acceleration story has also
been converted:

```powershell
.\.venv\Scripts\python.exe scripts\convert_aidt_annular_to_contract.py --raw-mat benchmark_results\public_data_probe\aidt\IRaw_Diatom_I.mat --sorted-pos-mat benchmark_results\public_data_probe\aidt\repo_files\Sorted_Pos.mat --out benchmark_results\aidt_diatom_public_contract.npz --summary-md benchmark_results\aidt_diatom_public_contract_summary.md

.\.venv\Scripts\python.exe scripts\validate_odt_measured_contract.py benchmark_results\aidt_diatom_public_contract.npz --json-out benchmark_results\aidt_diatom_public_contract_validation.json --summary-md benchmark_results\aidt_diatom_public_contract_validation.md --strict
```

The converted public dataset validates as:

- source: `bu-cisl/IDT-using-Annular-Illumination`;
- paper: `High-speed in vitro intensity diffraction tomography`, arXiv `1904.06004`;
- raw data source: public Google Drive folder linked from the repository README;
- data license note: repository is BSD-3-Clause, but a separate raw-data license
  was not found in the checked files;
- experiment type: `annular_idt`;
- measurement model: `coherent_intensity`;
- layout: `annular_cartesian_stack`;
- data shape: `24 x 700 x 700` float32;
- total measured samples: `11,760,000`;
- objective/source NA: `0.65`;
- wavelength: `0.515 um`;
- pixel size: `0.1625 um`;
- medium index: `1.47`.

This is currently the best public measured-data candidate because it exposes a
fixed annular illumination stack, which is much closer to the repeated
ring/annular geometry used in the prepared-operator benchmarks than the HL60
rotational sinogram.

## Public Data Adapter: aIDT Diatom I

The first measured-data adapter benchmark now maps the aIDT Cartesian intensity
stack into a Fourier-domain polar residual and applies that measured residual
to the prepared GPU adjoint and cuFINUFFT GPU Plan baseline.

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_aidt_public_measured_adapter.py --cap-radial 32 --cap-phi 128 --n-beta 256 --n-r 12 --n-z 11 --repeats 3 --warmups 1 --out benchmark_results\aidt_public_measured_adapter_benchmark.json --csv benchmark_results\aidt_public_measured_adapter_benchmark.csv --summary-md benchmark_results\aidt_public_measured_adapter_benchmark.md

.\.venv\Scripts\python.exe scripts\benchmark_aidt_public_measured_adapter.py --cap-radial 64 --cap-phi 256 --n-beta 256 --n-r 12 --n-z 11 --repeats 3 --warmups 1 --out benchmark_results\aidt_public_measured_adapter_large.json --csv benchmark_results\aidt_public_measured_adapter_large.csv --summary-md benchmark_results\aidt_public_measured_adapter_large.md
```

The public measured-data adapter validates as:

| cap | measured residual samples | prepared measured adjoint | cuFINUFFT measured adjoint | adjoint speedup | prepared measured pair | cuFINUFFT measured pair | pair speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 x 128 | 98,304 | 0.981 ms | 17.191 ms | 17.53x | 1.189 ms | 29.177 ms | 24.54x |
| 64 x 256 | 393,216 | 0.875 ms | 45.506 ms | 52.00x | 1.242 ms | 58.036 ms | 46.72x |

The largest relative L2 difference against cuFINUFFT is below `1e-4` in these
runs. This is still an adapter-level operator benchmark, not a full physical
aIDT reconstruction.

## Public Data Update Loop: aIDT Diatom I

The measured residual can now be used as the target in a repeated update loop:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_aidt_public_measured_update_loop.py --cap-radial 32 --cap-phi 128 --n-beta 256 --n-r 12 --n-z 11 --iterations 8 --out benchmark_results\aidt_public_measured_update_loop.json --csv benchmark_results\aidt_public_measured_update_loop_history.csv --summary-md benchmark_results\aidt_public_measured_update_loop.md

.\.venv\Scripts\python.exe scripts\benchmark_aidt_public_measured_update_loop.py --cap-radial 64 --cap-phi 256 --n-beta 256 --n-r 12 --n-z 11 --iterations 8 --out benchmark_results\aidt_public_measured_update_loop_large.json --csv benchmark_results\aidt_public_measured_update_loop_large_history.csv --summary-md benchmark_results\aidt_public_measured_update_loop_large.md
```

The measured-target update loop validates as:

| cap | measured samples | updates | prepared iter | cuFINUFFT iter | iter speedup | final loss delta | object-update rel-L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 x 128 | 98,304 | 8 | 1.643 ms | 25.936 ms | 15.78x | 1.19e-07 | 6.77e-05 |
| 64 x 256 | 393,216 | 8 | 1.962 ms | 58.778 ms | 29.96x | 0 | 5.70e-05 |

The target vector is derived from measured public intensity frames, so the
residual changes during the loop are measured-data driven. This remains a
linear Fourier-domain update model rather than a calibrated nonlinear aIDT
reconstruction.

## Next Implementation Step

The first real-data benchmark should keep the reconstruction update rule fixed
and compare prepared GPU versus cuFINUFFT on the same measured target. The
highest-priority path is now to make the aIDT loop more physical: add
background/flatfield handling, pupil or transfer-function calibration, masks,
regularization, and the nonlinear intensity measurement model. The ODTbrain
HL60 path remains a backup bridge that requires a rotational-sinogram adapter
or calibrated resampling step.
