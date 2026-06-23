from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_adapter_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_aidt_public_measured_adapter.py"
    spec = importlib.util.spec_from_file_location("benchmark_aidt_public_measured_adapter", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_uniform_ring_frame_order_maps_to_counterclockwise_plan_order() -> None:
    adapter = _load_adapter_module()
    angles = np.deg2rad([180.0, 90.0, 0.0, 270.0])
    source_na_xy = np.column_stack([np.cos(angles), np.sin(angles)])

    order = adapter.uniform_ring_frame_order(source_na_xy)

    assert order.tolist() == [2, 1, 0, 3]


def test_interpolate_fft_on_polar_detector_samples_centered_frequency_grid() -> None:
    adapter = _load_adapter_module()
    frequency = np.linspace(-1.0, 1.0, 9)
    xx, yy = np.meshgrid(frequency, frequency, indexing="xy")
    image = (2.0 * xx + 3.0 * yy).astype(np.float64)

    sampled = adapter.interpolate_fft_on_polar_detector(
        image,
        frequency_x=frequency,
        frequency_y=frequency,
        wavelength_um=1.0,
        detector_na=0.5,
        cap_radial=1,
        cap_phi=4,
        interpolation_order=1,
    )

    expected = np.array([[0.5, 0.75, -0.5, -0.75]], dtype=np.complex128)
    np.testing.assert_allclose(sampled, expected, atol=1e-12)
