from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

torch = pytest.importorskip("torch")

from benchmark_odt_full_slab_reconstruction import (  # noqa: E402
    pixel_to_modes,
    solve_real_cg_modes,
)
from benchmark_odt_realistic_geometry_reconstruction import (  # noqa: E402
    build_composite_context,
)
from benchmark_odt_torch_gpu_reconstruction import (  # noqa: E402
    TorchCompositeOdtPlan,
    parser,
)


def small_plan() -> TorchCompositeOdtPlan:
    config = parser().parse_args([])
    config.n_beta = 16
    config.n_r = 4
    config.n_z = 3
    config.ring_illum = 5
    config.cap_radial = 4
    config.cap_phi = 16
    config.h_margin = 2
    config.l_margin = 2
    config.cone_l_prune_threshold = 0.0
    config.cpp_threads = 1
    config.skip_native_prepared_adjoint = True
    config.compact_axisymmetric_kernel = True
    context = build_composite_context(config)
    return TorchCompositeOdtPlan.from_context(
        context,
        torch=torch,
        device=torch.device("cpu"),
        dtype="complex128",
        low_memory_adjoint=True,
        radial_block_size=2,
        illumination_block_size=2,
        forward_mode="auto",
        adjoint_mode="auto",
        prune_axis_l0=True,
        axial_lowrank_rank=3,
        ring_adaptive_l_packed_threshold=1e-3,
    )


def test_pixel_to_active_modes_preserves_selected_z_normal_operator() -> None:
    plan = small_plan()
    generator = torch.Generator().manual_seed(20260714)
    coeff = torch.complex(
        torch.randn((4, 3, 16), generator=generator, dtype=torch.float64),
        torch.randn((4, 3, 16), generator=generator, dtype=torch.float64),
    )
    index = torch.arange(3, dtype=torch.long)
    pixel = plan.forward_selected_z(coeff, index)
    direct_modes = plan.forward_selected_z_modes(coeff, index)
    converted_modes = pixel_to_modes(plan, pixel)
    torch.testing.assert_close(converted_modes, direct_modes, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(
        plan.adjoint_selected_z_modes(converted_modes, index),
        plan.adjoint_selected_z(pixel, index),
        rtol=1e-11,
        atol=1e-11,
    )


class IdentityModePlan:
    def __init__(self, shape: tuple[int, int, int]) -> None:
        self.device = torch.device("cpu")
        self.shape = shape

    def forward_selected_z_modes(self, value, index):
        assert value.shape[1] == index.numel()
        return value.reshape(-1)

    def adjoint_selected_z_modes(self, residual, index):
        return residual.reshape(self.shape[0], index.numel(), self.shape[2])


def test_real_mode_cg_reports_complete_reconstruction_time() -> None:
    plan = IdentityModePlan((3, 2, 4))
    index = torch.tensor([1, 2], dtype=torch.long)
    real = torch.linspace(0.1, 2.4, 24, dtype=torch.float64).reshape(3, 2, 4)
    truth = torch.complex(real, torch.zeros_like(real))
    result, history = solve_real_cg_modes(
        torch=torch,
        plan=plan,
        z_index=index,
        truth=truth,
        data_modes=truth.reshape(-1),
        iterations=2,
        preprocessing_s=0.01,
    )
    assert history[0]["object_nrmse"] < 1e-12
    assert history[0]["data_residual"] < 1e-12
    assert result["reconstruction_core_s_including_rhs"] >= result["rhs_adjoint_s"]
    assert result["pixel_input_total_s"] >= result["reconstruction_core_s_including_rhs"]
