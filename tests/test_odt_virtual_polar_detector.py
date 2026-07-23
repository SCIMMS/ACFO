from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_odt_virtual_polar_detector import (  # noqa: E402
    CachedBilinearPolarRemap,
    cartesian_q_samples,
)


def test_cached_bilinear_remap_reproduces_affine_complex_field() -> None:
    plan = CachedBilinearPolarRemap(
        torch=torch,
        device=torch.device("cpu"),
        n_xy=17,
        n_radial=7,
        n_phi=24,
        complex_dtype=torch.complex128,
        radial_fraction=0.9,
    )
    axis = torch.linspace(-1.0, 1.0, 17, dtype=torch.float64)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    camera = (1.2 + 0.3j) + (0.7 - 0.2j) * xx + (-0.4 + 0.5j) * yy
    candidate = plan.gather_explicit(camera[None, ...])[0]
    target_x = plan.grid[0, :, :, 0]
    target_y = plan.grid[0, :, :, 1]
    expected = (
        (1.2 + 0.3j)
        + (0.7 - 0.2j) * target_x
        + (-0.4 + 0.5j) * target_y
    )
    torch.testing.assert_close(candidate, expected, rtol=1e-12, atol=1e-12)


def test_explicit_remap_adjoint_and_fused_path_match() -> None:
    torch.manual_seed(12)
    plan = CachedBilinearPolarRemap(
        torch=torch,
        device=torch.device("cpu"),
        n_xy=19,
        n_radial=8,
        n_phi=20,
        complex_dtype=torch.complex128,
    )
    camera = torch.randn(2, 19, 19, dtype=torch.complex128)
    camera.mul_(plan.pupil_mask)
    residual = torch.randn(2, 8, 20, dtype=torch.complex128)
    explicit = plan.gather_explicit(camera, batch_block=1)
    fused = plan.gather_grid_sample(camera, batch_block=1)
    adjoint = plan.adjoint_explicit(residual, batch_block=1)
    torch.testing.assert_close(fused, explicit, rtol=1e-12, atol=1e-12)
    lhs = torch.vdot(explicit.reshape(-1), residual.reshape(-1))
    rhs = torch.vdot(camera.reshape(-1), adjoint.reshape(-1))
    relative = torch.abs(lhs - rhs) / (torch.abs(lhs) + torch.abs(rhs))
    assert float(relative) <= 1e-12


def test_cartesian_q_samples_use_only_circular_pupil() -> None:
    q, mask, active = cartesian_q_samples(
        k=10.0,
        detector_na=0.8,
        n_xy=25,
        illumination=np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
    )
    assert q.count == int(mask.sum()) == active.size
    assert np.all(q.illumination_index == 0)
    assert np.max(np.hypot(q.qx, q.qy)) <= 8.0 + 1e-12


def test_cached_remap_accepts_nonuniform_annular_radial_nodes() -> None:
    nodes = np.array([0.08, 0.21, 0.47, 0.65, 0.78], dtype=np.float64)
    plan = CachedBilinearPolarRemap(
        torch=torch,
        device=torch.device("cpu"),
        n_xy=25,
        n_radial=nodes.size,
        n_phi=32,
        complex_dtype=torch.complex128,
        radial_nodes_fraction=nodes,
    )
    axis = torch.linspace(-1.0, 1.0, 25, dtype=torch.float64)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    camera = torch.complex(0.5 + 0.3 * xx, -0.2 + 0.4 * yy)
    candidate = plan.gather_grid_sample(camera[None, ...])[0]
    target_x = plan.grid[0, :, :, 0]
    target_y = plan.grid[0, :, :, 1]
    expected = torch.complex(0.5 + 0.3 * target_x, -0.2 + 0.4 * target_y)
    torch.testing.assert_close(candidate, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(plan.radial, nodes, rtol=0.0, atol=0.0)
