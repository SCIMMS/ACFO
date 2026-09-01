from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from benchmark_odt_realistic_geometry_reconstruction import build_composite_context  # noqa: E402
from benchmark_odt_torch_gpu_reconstruction import (  # noqa: E402
    TorchCompositeOdtPlan,
    parser as gpu_parser,
)


def rel_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm((candidate - reference).ravel())
        / max(np.linalg.norm(reference.ravel()), np.finfo(np.float64).tiny)
    )


def test_low_memory_adjoint_matches_materialized_kernel() -> None:
    torch = pytest.importorskip("torch")
    config = gpu_parser().parse_args([])
    config.n_beta = 32
    config.n_r = 6
    config.n_z = 5
    config.ring_illum = 3
    config.skip_axis_illumination = False
    config.cap_radial = 6
    config.cap_phi = 16
    config.cpp_threads = 1
    ctx = build_composite_context(config)
    device = torch.device("cpu")
    dense = TorchCompositeOdtPlan.from_context(
        ctx,
        torch=torch,
        device=device,
        dtype="complex128",
        low_memory_adjoint=False,
    )
    low = TorchCompositeOdtPlan.from_context(
        ctx,
        torch=torch,
        device=device,
        dtype="complex128",
        low_memory_adjoint=True,
    )
    rng = np.random.default_rng(20260712)
    coeff_np = rng.standard_normal(ctx.ring.obj.coeff.shape) + 1j * rng.standard_normal(
        ctx.ring.obj.coeff.shape
    )
    residual_np = rng.standard_normal(dense.q_count) + 1j * rng.standard_normal(dense.q_count)
    coeff = torch.as_tensor(coeff_np, dtype=torch.complex128)
    residual = torch.as_tensor(residual_np, dtype=torch.complex128)

    dense_forward = dense.forward(coeff).detach().numpy()
    low_forward = low.forward(coeff).detach().numpy()
    dense_adjoint = dense.adjoint(residual).detach().numpy()
    low_adjoint = low.adjoint(residual).detach().numpy()

    assert rel_l2(low_forward, dense_forward) < 1e-13
    assert rel_l2(low_adjoint, dense_adjoint) < 1e-12
    assert low.basis_mib < dense.basis_mib
