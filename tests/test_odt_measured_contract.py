from __future__ import annotations

import numpy as np

from waxs_cake.odt_measured_contract import (
    SCHEMA_VERSION,
    OdtMeasuredData,
    build_prepared_operator_from_contract,
    load_odt_measured_contract,
    save_odt_measured_contract,
    validate_odt_measured_contract,
)


def _base_fields(*, measurement_model: str = "coherent_intensity") -> dict[str, object]:
    n_illum = 3
    n_patterns = 2
    cap_radial = 4
    cap_phi = 8
    fields: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_type": "fs_odt" if measurement_model != "complex_field" else "complex_odt",
        "measurement_model": measurement_model,
        "k": 5.0,
        "units": "um",
        "illum_dirs": np.array(
            [
                [0.0, 0.0, 1.0],
                [0.1, 0.0, np.sqrt(1.0 - 0.1**2)],
                [0.0, 0.2, np.sqrt(1.0 - 0.2**2)],
            ],
            dtype=np.float64,
        ),
        "detector_origin": np.array([0.0, 0.0, 100.0], dtype=np.float64),
        "detector_u": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "detector_v": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "detector_pixel_size": np.array([0.1, 0.1], dtype=np.float64),
        "detector_distance": 100.0,
        "q_layout": "prepared_ring_stack",
        "cap_radial": cap_radial,
        "cap_phi": cap_phi,
        "q_radial": np.linspace(0.0, 1.0, cap_radial),
        "q_phi": np.linspace(0.0, 2.0 * np.pi, cap_phi, endpoint=False),
        "q_z_model": "full_curved_ewald",
        "mask": np.ones((n_patterns, cap_radial, cap_phi), dtype=np.float32),
        "variance": np.ones((n_patterns, cap_radial, cap_phi), dtype=np.float32),
    }
    if measurement_model == "complex_field":
        fields["data"] = np.ones((n_illum, cap_radial, cap_phi), dtype=np.complex64)
        fields["mask"] = np.ones((n_illum, cap_radial, cap_phi), dtype=np.float32)
        fields["variance"] = np.ones((n_illum, cap_radial, cap_phi), dtype=np.float32)
    else:
        fields["data"] = np.ones((n_patterns, cap_radial, cap_phi), dtype=np.float32)
        fields["pattern_matrix"] = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0 / np.sqrt(2.0), 1j / np.sqrt(2.0)],
            ],
            dtype=np.complex64,
        )
        fields["pattern_model"] = "coherent" if measurement_model == "coherent_intensity" else "incoherent"
    return fields


def test_valid_intensity_contract_round_trips_npz(tmp_path) -> None:
    path = tmp_path / "contract.npz"
    save_odt_measured_contract(path, _base_fields(measurement_model="coherent_intensity"))

    loaded = load_odt_measured_contract(path)
    report = validate_odt_measured_contract(loaded)
    descriptor = build_prepared_operator_from_contract(loaded)

    assert report.ok
    assert report.summary["n_illum"] == 3
    assert report.summary["n_patterns"] == 2
    assert report.summary["q_samples"] == 32
    assert report.summary["measurement_samples"] == 64
    assert descriptor.measurement_model == "coherent_intensity"
    assert descriptor.n_patterns == 2
    assert descriptor.measurement_samples == 64
    assert descriptor.data_shape == (2, 4, 8)


def test_valid_complex_field_contract() -> None:
    measured = OdtMeasuredData(_base_fields(measurement_model="complex_field"))
    report = validate_odt_measured_contract(measured)
    descriptor = build_prepared_operator_from_contract(measured)

    assert report.ok
    assert descriptor.measurement_model == "complex_field"
    assert descriptor.n_illum == 3
    assert descriptor.n_patterns is None
    assert descriptor.measurement_samples == 96
    assert descriptor.data_shape == (3, 4, 8)


def test_invalid_pattern_matrix_shape_is_error() -> None:
    fields = _base_fields(measurement_model="incoherent_intensity")
    fields["pattern_matrix"] = np.ones((2, 4), dtype=np.float32)

    report = validate_odt_measured_contract(fields)

    assert not report.ok
    assert any(issue.field == "pattern_matrix" for issue in report.errors)


def test_invalid_illumination_normalization_is_error() -> None:
    fields = _base_fields(measurement_model="coherent_intensity")
    illum_dirs = np.asarray(fields["illum_dirs"]).copy()
    illum_dirs[1] *= 2.0
    fields["illum_dirs"] = illum_dirs

    report = validate_odt_measured_contract(fields)

    assert not report.ok
    assert any(issue.field == "illum_dirs" for issue in report.errors)


def test_explicit_q_contract_accepts_flat_data() -> None:
    n_q = 5
    fields = {
        "schema_version": SCHEMA_VERSION,
        "experiment_type": "complex_odt",
        "measurement_model": "complex_field",
        "k": 5.0,
        "units": "um",
        "illum_dirs": np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
        "detector_origin": np.array([0.0, 0.0, 100.0], dtype=np.float64),
        "detector_u": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "detector_v": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "detector_pixel_size": np.array([0.1, 0.1], dtype=np.float64),
        "detector_distance": 100.0,
        "q_layout": "explicit_q",
        "q_xyz": np.zeros((n_q, 3), dtype=np.float64),
        "data": np.ones(n_q, dtype=np.complex64),
    }

    report = validate_odt_measured_contract(fields)
    descriptor = build_prepared_operator_from_contract(fields)

    assert report.ok
    assert descriptor.q_layout == "explicit_q"
    assert descriptor.q_samples == n_q
    assert descriptor.measurement_samples == n_q


def test_rotational_sinogram_contract_accepts_complex_field_stack() -> None:
    n_angles = 4
    height = 5
    width = 6
    angles = np.linspace(0.0, np.pi, n_angles, endpoint=False)
    illum_dirs = np.stack(
        [np.sin(angles), np.zeros_like(angles), np.cos(angles)],
        axis=1,
    )
    fields = {
        "schema_version": SCHEMA_VERSION,
        "experiment_type": "complex_odt",
        "measurement_model": "complex_field",
        "wavelength": 0.647,
        "units": "um",
        "illum_dirs": illum_dirs,
        "detector_origin": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        "detector_u": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "detector_v": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "detector_pixel_size": np.array([0.139, 0.139], dtype=np.float64),
        "detector_distance": 1.0,
        "q_layout": "rotational_sinogram",
        "rotation_angles": angles,
        "data": np.ones((n_angles, height, width), dtype=np.complex64),
    }

    report = validate_odt_measured_contract(fields)
    descriptor = build_prepared_operator_from_contract(fields)

    assert report.ok
    assert descriptor.q_layout == "rotational_sinogram"
    assert descriptor.q_samples == height * width
    assert descriptor.measurement_samples == n_angles * height * width


def test_annular_cartesian_stack_accepts_intensity_frames() -> None:
    n_illum = 4
    height = 5
    width = 6
    source_na_xy = np.array(
        [
            [0.5, 0.0],
            [0.0, 0.5],
            [-0.5, 0.0],
            [0.0, -0.5],
        ],
        dtype=np.float64,
    )
    z = np.sqrt(1.0 - np.sum(source_na_xy**2, axis=1))
    illum_dirs = np.column_stack([source_na_xy, z])
    fields = {
        "schema_version": SCHEMA_VERSION,
        "experiment_type": "annular_idt",
        "measurement_model": "coherent_intensity",
        "wavelength": 0.515,
        "k": 2.0 * np.pi * 1.47 / 0.515,
        "units": "um",
        "illum_dirs": illum_dirs,
        "detector_origin": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        "detector_u": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "detector_v": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "detector_pixel_size": np.array([0.1625, 0.1625], dtype=np.float64),
        "detector_distance": 1.0,
        "q_layout": "annular_cartesian_stack",
        "objective_na": 0.65,
        "medium_index": 1.47,
        "source_na_xy": source_na_xy,
        "frequency_x": np.linspace(-1.0, 1.0, width),
        "frequency_y": np.linspace(-1.0, 1.0, height),
        "data": np.ones((n_illum, height, width), dtype=np.float32),
    }

    report = validate_odt_measured_contract(fields)
    descriptor = build_prepared_operator_from_contract(fields)

    assert report.ok
    assert descriptor.q_layout == "annular_cartesian_stack"
    assert descriptor.experiment_type == "annular_idt"
    assert descriptor.q_samples == height * width
    assert descriptor.measurement_samples == n_illum * height * width
    assert report.summary["source_na_max"] == np.linalg.norm(source_na_xy, axis=1).max()
