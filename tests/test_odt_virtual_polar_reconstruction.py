from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="PyTorch is an optional ODT dependency")


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_odt_virtual_polar_reconstruction import (  # noqa: E402
    selected_data_lipschitz,
    solve_selected_fista,
)


class IdentitySelectedPlan:
    def __init__(self, shape: tuple[int, int, int]) -> None:
        self.device = torch.device("cpu")
        self.real_dtype = torch.float64
        self.ring = SimpleNamespace(n_r=shape[0], n_beta=shape[2])
        self.shape = shape

    def forward_selected_z(self, value, z_index):
        assert value.shape[1] == z_index.numel()
        return value.reshape(-1)

    def adjoint_selected_z(self, residual, z_index):
        return residual.reshape(self.shape[0], z_index.numel(), self.shape[2])


def test_selected_power_iteration_recovers_identity_norm() -> None:
    plan = IdentitySelectedPlan((3, 2, 4))
    z_index = torch.tensor([2, 3], dtype=torch.long)
    estimate, _ = selected_data_lipschitz(
        torch=torch,
        plan=plan,
        z_index=z_index,
        selected_n_z=2,
        iterations=3,
        seed=7,
    )
    assert estimate == pytest.approx(1.0, abs=1e-12)


def test_selected_nonnegative_fista_is_an_actual_iterative_inverse() -> None:
    plan = IdentitySelectedPlan((3, 1, 4))
    z_index = torch.tensor([1], dtype=torch.long)
    truth_real = torch.linspace(0.1, 1.2, 12, dtype=torch.float64).reshape(3, 1, 4)
    truth = torch.complex(truth_real, torch.zeros_like(truth_real))
    result, reconstruction, history = solve_selected_fista(
        torch=torch,
        plan=plan,
        z_index=z_index,
        data=truth.reshape(-1),
        truth=truth,
        label="identity",
        data_lipschitz=1.0,
        iterations=15,
        record_every=5,
    )
    assert result["object_rel_l2"] < 1e-8
    assert result["data_residual"] < 1e-8
    assert len(history) == 4
    torch.testing.assert_close(reconstruction, truth, rtol=1e-8, atol=1e-8)
