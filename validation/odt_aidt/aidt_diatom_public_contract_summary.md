# Public aIDT annular-intensity contract conversion

This file records conversion of the public aIDT Diatom I raw intensity stack into the local measured-data contract.

## Source

- repository: `bu-cisl/IDT-using-Annular-Illumination`
- paper: `High-speed in vitro intensity diffraction tomography`
- arXiv: `1904.06004`
- raw data: public Google Drive folder linked from the repository README
- data license note: the repository is BSD-3-Clause; a separate raw-data license was not found in the checked files

## Converted Contract

| key | value |
| --- | --- |
| `experiment_type` | `annular_idt` |
| `measurement_model` | `coherent_intensity` |
| `q_layout` | `annular_cartesian_stack` |
| `n_illum` | `24` |
| `cap_radial` | `700` |
| `cap_phi` | `700` |
| `q_samples` | `490000` |
| `measurement_samples` | `11760000` |
| `data_shape` | `(24, 700, 700)` |
| `data_dtype` | `float32` |
| `objective_na` | `0.65` |
| `source_na_min` | `0.65` |
| `source_na_max` | `0.65` |
| `has_mask` | `False` |

## Source Data Readout

| key | value |
| --- | --- |
| `raw_mat` | `benchmark_results\public_data_probe\aidt\IRaw_Diatom_I.mat` |
| `sorted_pos_mat` | `benchmark_results\public_data_probe\aidt\repo_files\Sorted_Pos.mat` |
| `data_shape` | `(24, 700, 700)` |
| `data_dtype` | `float32` |
| `data_min` | `-0.8053455352783203` |
| `data_max` | `4.397280216217041` |
| `data_mean` | `-0.0025137646589428186` |
| `data_std` | `0.22879533469676971` |
| `wavelength_um` | `0.515` |
| `pixel_size_um` | `0.1625` |
| `objective_na` | `0.65` |
| `medium_index` | `1.47` |
| `source_na_min` | `0.65` |
| `source_na_max` | `0.65` |
| `frequency_x_min` | `-3.076923076923077` |
| `frequency_x_max` | `3.0681318681318683` |
| `frequency_y_min` | `-3.076923076923077` |
| `frequency_y_max` | `3.0681318681318683` |

## Geometry Fit

This is a stronger public-data candidate than a conventional rotational sinogram for the current ODT acceleration story.
It contains measured intensity frames under a fixed annular illumination design, so the acquisition geometry naturally exposes repeated ring/annular structure.
It is still not a finished prepared-operator benchmark: the next step is to map the annular Cartesian image stack into the exact curved-Ewald operator and compare prepared GPU against cuFINUFFT on the same measured update.

## Validation

- valid: `True`
- errors: `0`
- warnings: `0`
