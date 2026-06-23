from __future__ import annotations

import numpy as np
import pytest

from waxs_cake import (
    cylindrical_flat_indices,
    encode_elements,
    make_cylindrical_histogram,
    make_cylindrical_histogram_from_flat_indices,
    make_cylindrical_histogram_from_indices,
    make_cylindrical_histogram_indexed,
)


def _reference_histogram(
    coords: np.ndarray,
    elements: list[str],
    weights: np.ndarray,
    *,
    n_r: int,
    n_z: int,
    n_phi: int,
    r_max: float,
    z_range: tuple[float, float],
    element_order: tuple[str, ...],
) -> np.ndarray:
    element_to_index = {element: i for i, element in enumerate(element_order)}
    element_indices = np.array([element_to_index[e] for e in elements])
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    radius = np.hypot(x, y)
    beta = np.mod(np.arctan2(y, x), 2.0 * np.pi)
    r_edges = np.linspace(0.0, r_max, n_r + 1)
    z_edges = np.linspace(z_range[0], z_range[1], n_z + 1)
    beta_edges = np.linspace(0.0, 2.0 * np.pi, n_phi + 1)
    r_idx = np.searchsorted(r_edges, radius, side="right") - 1
    z_idx = np.searchsorted(z_edges, z, side="right") - 1
    beta_idx = np.searchsorted(beta_edges, beta, side="right") - 1
    r_idx = np.clip(r_idx, 0, n_r - 1)
    z_idx = np.clip(z_idx, 0, n_z - 1)
    beta_idx = np.mod(beta_idx, n_phi)

    hist = np.zeros((len(element_order), n_r, n_z, n_phi), dtype=np.complex128)
    np.add.at(hist, (element_indices, r_idx, z_idx, beta_idx), weights)
    return hist


def test_fast_histogram_matches_searchsorted_reference() -> None:
    rng = np.random.default_rng(22)
    coords = rng.uniform(-3.0, 3.0, size=(5000, 3))
    edge_coords = np.array(
        [
            [0.0, 0.0, -3.0],
            [3.0, 0.0, 3.0],
            [-3.0, 0.0, 0.0],
            [0.0, -3.0, 1.5],
            [0.0, 3.0, -1.5],
        ]
    )
    coords = np.vstack([coords, edge_coords])
    elements = rng.choice(["C", "N", "O"], size=coords.shape[0]).tolist()
    weights = rng.normal(size=coords.shape[0]) + 1j * rng.normal(size=coords.shape[0])

    kwargs = {
        "n_r": 17,
        "n_z": 19,
        "n_phi": 37,
        "r_max": 5.0,
        "z_range": (-3.0, 3.0),
        "element_order": ("C", "N", "O"),
    }
    got = make_cylindrical_histogram(
        coords,
        elements,
        atom_weights=weights,
        **kwargs,
    ).hist
    expected = _reference_histogram(coords, elements, weights, **kwargs)

    assert np.array_equal(got, expected)


def test_default_histogram_counts_atoms_without_weight_array() -> None:
    rng = np.random.default_rng(23)
    coords = rng.normal(size=(10000, 3))
    hist = make_cylindrical_histogram(coords, n_r=8, n_z=9, n_phi=10)

    assert hist.hist.shape == (1, 8, 9, 10)
    assert np.issubdtype(hist.hist.dtype, np.integer)
    assert hist.hist.sum() == coords.shape[0]


def test_float32_binning_dtype_counts_atoms() -> None:
    rng = np.random.default_rng(24)
    coords = rng.normal(size=(10000, 3))
    hist = make_cylindrical_histogram(
        coords,
        n_r=8,
        n_z=9,
        n_phi=10,
        binning_dtype=np.float32,
    )

    assert hist.hist.sum() == coords.shape[0]


def test_numba_backend_matches_numpy_default_case() -> None:
    try:
        __import__("numba")
    except ImportError:
        pytest.skip("numba is not installed")

    rng = np.random.default_rng(25)
    coords = rng.uniform(-3.0, 3.0, size=(5000, 3))
    kwargs = {
        "n_r": 17,
        "n_z": 19,
        "n_phi": 37,
        "r_max": 6.0,
        "z_range": (-3.0, 3.0),
    }
    expected = make_cylindrical_histogram(coords, **kwargs).hist
    got = make_cylindrical_histogram(coords, backend="numba", **kwargs).hist
    got_parallel = make_cylindrical_histogram(
        coords, backend="numba-parallel", **kwargs
    ).hist

    assert np.array_equal(got, expected)
    assert np.array_equal(got_parallel, expected)


def test_cpp_backend_matches_numpy_default_case() -> None:
    pytest.importorskip("waxs_cake._cpp_histogram")

    rng = np.random.default_rng(26)
    coords = rng.uniform(-3.0, 3.0, size=(5000, 3))
    kwargs = {
        "n_r": 17,
        "n_z": 19,
        "n_phi": 37,
        "r_max": 6.0,
        "z_range": (-3.0, 3.0),
    }
    expected = make_cylindrical_histogram(coords, **kwargs).hist
    got = make_cylindrical_histogram(coords, backend="cpp", **kwargs).hist

    assert np.array_equal(got, expected)
    assert np.issubdtype(got.dtype, np.integer)


def test_cpp_backend_matches_numpy_weighted_multi_element_cases() -> None:
    pytest.importorskip("waxs_cake._cpp_histogram")

    rng = np.random.default_rng(27)
    coords = rng.uniform(-4.0, 4.0, size=(5000, 3))
    elements = rng.choice(["C", "N", "O"], size=coords.shape[0]).tolist()
    real_weights = rng.normal(size=coords.shape[0])
    complex_weights = real_weights + 1j * rng.normal(size=coords.shape[0])
    kwargs = {
        "n_r": 13,
        "n_z": 15,
        "n_phi": 29,
        "r_max": 7.0,
        "z_range": (-4.0, 4.0),
        "element_order": ("C", "N", "O"),
    }

    expected_unweighted = make_cylindrical_histogram(coords, elements, **kwargs).hist
    got_unweighted = make_cylindrical_histogram(
        coords, elements, backend="cpp", **kwargs
    ).hist
    assert np.array_equal(got_unweighted, expected_unweighted)
    assert np.issubdtype(got_unweighted.dtype, np.integer)

    expected_real = make_cylindrical_histogram(
        coords, elements, atom_weights=real_weights, **kwargs
    ).hist
    got_real = make_cylindrical_histogram(
        coords, elements, atom_weights=real_weights, backend="cpp", **kwargs
    ).hist
    assert np.array_equal(got_real, expected_real)
    assert np.issubdtype(got_real.dtype, np.floating)

    expected_complex = make_cylindrical_histogram(
        coords, elements, atom_weights=complex_weights, **kwargs
    ).hist
    got_complex = make_cylindrical_histogram(
        coords, elements, atom_weights=complex_weights, backend="cpp", **kwargs
    ).hist
    assert np.array_equal(got_complex, expected_complex)
    assert np.issubdtype(got_complex.dtype, np.complexfloating)


def test_indexed_histogram_matches_string_element_path() -> None:
    rng = np.random.default_rng(28)
    coords = rng.uniform(-4.0, 4.0, size=(6000, 3))
    element_order = ("C", "N", "O")
    element_indices = rng.integers(0, len(element_order), size=coords.shape[0])
    elements = [element_order[i] for i in element_indices]
    kwargs = {
        "n_r": 13,
        "n_z": 15,
        "n_phi": 29,
        "r_max": 7.0,
        "z_range": (-4.0, 4.0),
        "element_order": element_order,
    }

    expected = make_cylindrical_histogram(coords, elements, **kwargs).hist
    got = make_cylindrical_histogram_indexed(
        coords,
        element_indices,
        n_elements=len(element_order),
        backend="numpy",
        **kwargs,
    ).hist

    assert np.array_equal(got, expected)


def test_encode_elements_prepares_indexed_histogram_inputs() -> None:
    pytest.importorskip("waxs_cake._cpp_histogram")

    rng = np.random.default_rng(128)
    coords = rng.uniform(-4.0, 4.0, size=(6000, 3))
    elements = rng.choice(["O", "C", "N"], size=coords.shape[0]).tolist()
    element_indices, element_order = encode_elements(elements, dtype=np.int64)
    kwargs = {
        "n_r": 13,
        "n_z": 15,
        "n_phi": 29,
        "r_max": 7.0,
        "z_range": (-4.0, 4.0),
    }

    expected = make_cylindrical_histogram(
        coords,
        elements,
        element_order=element_order,
        backend="cpp",
        **kwargs,
    ).hist
    got = make_cylindrical_histogram_indexed(
        coords,
        element_indices,
        n_elements=len(element_order),
        element_order=element_order,
        backend="cpp",
        **kwargs,
    ).hist

    assert element_indices.dtype == np.int64
    assert set(element_order) == {"C", "N", "O"}
    assert np.array_equal(got, expected)


def test_cpp_histogram_fast_dtypes_match_reference() -> None:
    pytest.importorskip("waxs_cake._cpp_histogram")

    rng = np.random.default_rng(29)
    coords = rng.uniform(-4.0, 4.0, size=(6000, 3))
    element_indices = rng.integers(0, 3, size=coords.shape[0], dtype=np.int32)
    real_weights = rng.normal(size=coords.shape[0])
    complex_weights = real_weights + 1j * rng.normal(size=coords.shape[0])
    kwargs = {
        "n_elements": 3,
        "n_r": 13,
        "n_z": 15,
        "n_phi": 29,
        "r_max": 7.0,
        "z_range": (-4.0, 4.0),
    }

    expected_counts = make_cylindrical_histogram_indexed(
        coords,
        element_indices,
        backend="cpp",
        **kwargs,
    ).hist
    got_counts = make_cylindrical_histogram_indexed(
        coords,
        element_indices,
        backend="cpp",
        hist_dtype=np.uint32,
        **kwargs,
    ).hist
    assert got_counts.dtype == np.uint32
    assert np.array_equal(got_counts, expected_counts)

    expected_real = make_cylindrical_histogram_indexed(
        coords,
        element_indices,
        atom_weights=real_weights,
        backend="cpp",
        **kwargs,
    ).hist
    got_real = make_cylindrical_histogram_indexed(
        coords,
        element_indices,
        atom_weights=real_weights,
        backend="cpp",
        hist_dtype=np.float32,
        **kwargs,
    ).hist
    assert got_real.dtype == np.float32
    assert np.allclose(got_real, expected_real, rtol=1e-6, atol=1e-5)

    expected_complex = make_cylindrical_histogram_indexed(
        coords,
        element_indices,
        atom_weights=complex_weights,
        backend="cpp",
        **kwargs,
    ).hist
    got_complex = make_cylindrical_histogram_indexed(
        coords,
        element_indices,
        atom_weights=complex_weights,
        backend="cpp",
        hist_dtype=np.complex64,
        **kwargs,
    ).hist
    assert got_complex.dtype == np.complex64
    assert np.allclose(got_complex, expected_complex, rtol=1e-6, atol=1e-5)


def test_precomputed_bin_indices_match_coordinate_histogram() -> None:
    rng = np.random.default_rng(30)
    coords = rng.uniform(-4.0, 4.0, size=(7000, 3))
    elements = rng.integers(0, 3, size=coords.shape[0], dtype=np.int32)
    weights = rng.normal(size=coords.shape[0])
    n_r, n_z, n_phi = 13, 15, 29
    r_max = 7.0
    z_range = (-4.0, 4.0)

    radius = np.sqrt(coords[:, 0] * coords[:, 0] + coords[:, 1] * coords[:, 1])
    r_idx = (radius * (n_r / r_max)).astype(np.intp)
    z_idx = ((coords[:, 2] - z_range[0]) * (n_z / (z_range[1] - z_range[0]))).astype(
        np.intp
    )
    beta = np.arctan2(coords[:, 1], coords[:, 0])
    beta[beta < 0.0] += 2.0 * np.pi
    beta_idx = (beta * (n_phi / (2.0 * np.pi))).astype(np.intp)
    np.clip(r_idx, 0, n_r - 1, out=r_idx)
    np.clip(z_idx, 0, n_z - 1, out=z_idx)
    np.clip(beta_idx, 0, n_phi - 1, out=beta_idx)

    expected = make_cylindrical_histogram_indexed(
        coords,
        elements,
        n_elements=3,
        atom_weights=weights,
        n_r=n_r,
        n_z=n_z,
        n_phi=n_phi,
        r_max=r_max,
        z_range=z_range,
        backend="cpp",
    ).hist
    got = make_cylindrical_histogram_from_indices(
        r_idx,
        z_idx,
        beta_idx,
        element_indices=elements,
        n_elements=3,
        atom_weights=weights,
        n_r=n_r,
        n_z=n_z,
        n_phi=n_phi,
        r_max=r_max,
        z_range=z_range,
        backend="cpp",
    ).hist

    assert np.array_equal(got, expected)


def test_precomputed_flat_indices_fast_dtype() -> None:
    pytest.importorskip("waxs_cake._cpp_histogram")

    flat = np.array([0, 1, 1, 7, 7, 7], dtype=np.int64)
    got = make_cylindrical_histogram_from_flat_indices(
        flat,
        n_elements=1,
        n_r=2,
        n_z=2,
        n_phi=2,
        r_max=1.0,
        z_range=(-1.0, 1.0),
        hist_dtype=np.uint32,
        backend="cpp",
    ).hist
    got_no_validate = make_cylindrical_histogram_from_flat_indices(
        flat,
        n_elements=1,
        n_r=2,
        n_z=2,
        n_phi=2,
        r_max=1.0,
        z_range=(-1.0, 1.0),
        hist_dtype=np.uint32,
        backend="cpp",
        validate_indices=False,
    ).hist

    assert got.dtype == np.uint32
    assert got.ravel().tolist() == [1, 2, 0, 0, 0, 0, 0, 3]
    assert np.array_equal(got_no_validate, got)


def test_cpp_cylindrical_flat_indices_match_numpy_exact() -> None:
    pytest.importorskip("waxs_cake._cpp_histogram")

    rng = np.random.default_rng(31)
    coords = rng.uniform(-4.0, 4.0, size=(7000, 3))
    elements = rng.integers(0, 3, size=coords.shape[0], dtype=np.int32)
    kwargs = {
        "element_indices": elements,
        "n_elements": 3,
        "n_r": 13,
        "n_z": 15,
        "n_phi": 29,
        "r_max": 7.0,
        "z_range": (-4.0, 4.0),
    }

    expected = cylindrical_flat_indices(coords, backend="numpy", **kwargs)
    got = cylindrical_flat_indices(coords, backend="cpp", **kwargs)

    assert np.array_equal(got, expected)


def test_cpp_cylindrical_flat_indices_lut_conserves_atoms() -> None:
    pytest.importorskip("waxs_cake._cpp_histogram")

    rng = np.random.default_rng(32)
    coords = rng.uniform(-4.0, 4.0, size=(7000, 3))
    flat = cylindrical_flat_indices(
        coords,
        backend="cpp",
        angle_lut_size=4096,
        n_r=13,
        n_z=15,
        n_phi=29,
        r_max=7.0,
        z_range=(-4.0, 4.0),
    )
    hist = make_cylindrical_histogram_from_flat_indices(
        flat,
        n_r=13,
        n_z=15,
        n_phi=29,
        r_max=7.0,
        z_range=(-4.0, 4.0),
        backend="cpp",
    ).hist

    assert hist.sum() == coords.shape[0]


def test_cpp_histogram_angle_lut_conserves_atoms() -> None:
    pytest.importorskip("waxs_cake._cpp_histogram")

    rng = np.random.default_rng(33)
    coords = rng.uniform(-4.0, 4.0, size=(8000, 3))
    hist = make_cylindrical_histogram(
        coords,
        n_r=13,
        n_z=15,
        n_phi=29,
        r_max=7.0,
        z_range=(-4.0, 4.0),
        hist_dtype=np.float32,
        backend="cpp",
        angle_lut_size=4096,
    ).hist

    assert hist.dtype == np.float32
    assert hist.shape == (1, 13, 15, 29)
    assert hist.sum() == coords.shape[0]


def test_cpp_histogram_cubic_angle_lut_conserves_atoms() -> None:
    pytest.importorskip("waxs_cake._cpp_histogram")

    rng = np.random.default_rng(34)
    coords = rng.uniform(-4.0, 4.0, size=(8000, 3))
    hist = make_cylindrical_histogram(
        coords,
        n_r=13,
        n_z=15,
        n_phi=29,
        r_max=7.0,
        z_range=(-4.0, 4.0),
        hist_dtype=np.float32,
        backend="cpp",
        angle_lut_size=512,
        angle_lut_mode="cubic",
    ).hist

    assert hist.dtype == np.float32
    assert hist.sum() == coords.shape[0]
