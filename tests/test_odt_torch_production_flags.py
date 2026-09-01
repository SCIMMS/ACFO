from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

torch = pytest.importorskip("torch")

from benchmark_odt_realistic_geometry_reconstruction import (  # noqa: E402
    build_composite_context,
)
from benchmark_odt_torch_gpu_reconstruction import (  # noqa: E402
    TorchCompositeOdtPlan,
    parser,
)


def small_context() -> object:
    config = parser().parse_args([])
    config.n_beta = 16
    config.n_r = 4
    config.n_z = 3
    config.ring_illum = 5
    config.skip_axis_illumination = False
    config.cap_radial = 4
    config.cap_phi = 16
    config.h_margin = 2
    config.l_margin = 2
    config.cone_l_prune_threshold = 0.0
    config.cpp_threads = 1
    config.skip_native_prepared_adjoint = True
    config.compact_axisymmetric_kernel = True
    return build_composite_context(config)


def random_pair(q_count: int) -> tuple[object, object]:
    generator = torch.Generator().manual_seed(20260713)
    coeff = torch.complex(
        torch.randn((4, 3, 16), generator=generator, dtype=torch.float64),
        torch.randn((4, 3, 16), generator=generator, dtype=torch.float64),
    )
    residual = torch.complex(
        torch.randn(q_count, generator=generator, dtype=torch.float64),
        torch.randn(q_count, generator=generator, dtype=torch.float64),
    )
    return coeff, residual


def test_accelerator_flags_are_opt_in() -> None:
    args = parser().parse_args([])
    assert args.prune_axis_l0 is False
    assert args.axial_lowrank_rank == 0
    assert args.ring_adaptive_l_packed_threshold == 0.0


def test_production_axis_l0_flag_is_exact() -> None:
    context = small_context()
    common = dict(
        torch=torch,
        device=torch.device("cpu"),
        dtype="complex128",
        low_memory_adjoint=True,
        radial_block_size=2,
        illumination_block_size=2,
        forward_mode="illumination-reduced",
        adjoint_mode="illumination-reduced",
    )
    exact = TorchCompositeOdtPlan.from_context(context, **common)
    pruned = TorchCompositeOdtPlan.from_context(
        context, prune_axis_l0=True, **common
    )
    assert exact.axis is not None and pruned.axis is not None
    assert exact.axis.n_l > 1
    assert pruned.axis.n_l == 1
    coeff, residual = random_pair(exact.q_count)
    torch.testing.assert_close(
        pruned.forward(coeff), exact.forward(coeff), rtol=1e-11, atol=1e-11
    )
    torch.testing.assert_close(
        pruned.adjoint(residual), exact.adjoint(residual), rtol=1e-11, atol=1e-11
    )


def test_full_rank_production_axial_factor_matches_exact_and_is_adjoint() -> None:
    context = small_context()
    common = dict(
        torch=torch,
        device=torch.device("cpu"),
        dtype="complex128",
        low_memory_adjoint=True,
        radial_block_size=2,
        illumination_block_size=2,
        forward_mode="auto",
        adjoint_mode="auto",
        prune_axis_l0=True,
    )
    exact = TorchCompositeOdtPlan.from_context(context, **common)
    factored = TorchCompositeOdtPlan.from_context(
        context, axial_lowrank_rank=3, **common
    )
    assert factored.ring.axial_lowrank_rank == 3
    assert factored.axis is not None and factored.axis.axial_lowrank_rank == 3
    assert factored.ring.resolved_forward_mode == "illumination-reduced"
    assert factored.axis.resolved_adjoint_mode == "illumination-reduced"
    coeff, residual = random_pair(exact.q_count)
    forward = factored.forward(coeff)
    adjoint = factored.adjoint(residual)
    torch.testing.assert_close(forward, exact.forward(coeff), rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(adjoint, exact.adjoint(residual), rtol=1e-11, atol=1e-11)
    lhs = torch.vdot(forward.reshape(-1), residual.reshape(-1))
    rhs = torch.vdot(coeff.reshape(-1), adjoint.reshape(-1))
    dot_error = torch.abs(lhs - rhs) / (torch.abs(lhs) + torch.abs(rhs))
    assert float(dot_error) <= 1e-12


def test_full_rank_selected_z_factor_matches_full_grid_restriction() -> None:
    context = small_context()
    plan = TorchCompositeOdtPlan.from_context(
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
    )
    z_indices = torch.tensor([0, 2], dtype=torch.long)
    generator = torch.Generator().manual_seed(20260714)
    selected = torch.complex(
        torch.randn((4, 2, 16), generator=generator, dtype=torch.float64),
        torch.randn((4, 2, 16), generator=generator, dtype=torch.float64),
    )
    full = torch.zeros((4, 3, 16), dtype=torch.complex128)
    full.index_copy_(1, z_indices, selected)
    residual = torch.complex(
        torch.randn(plan.q_count, generator=generator, dtype=torch.float64),
        torch.randn(plan.q_count, generator=generator, dtype=torch.float64),
    )
    torch.testing.assert_close(
        plan.forward_selected_z(selected, z_indices),
        plan.forward(full),
        rtol=1e-11,
        atol=1e-11,
    )
    torch.testing.assert_close(
        plan.adjoint_selected_z(residual, z_indices),
        plan.adjoint(residual).index_select(1, z_indices),
        rtol=1e-11,
        atol=1e-11,
    )


def test_packed_l_exact_zero_elimination_matches_dense_operator() -> None:
    context = small_context()
    common = dict(
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
    )
    dense = TorchCompositeOdtPlan.from_context(context, **common)
    packed = TorchCompositeOdtPlan.from_context(
        context, ring_adaptive_l_packed_threshold=1e-300, **common
    )
    assert packed.ring.adaptive_l_packed_enabled
    assert packed.ring.adaptive_l_active_fraction <= 1.0
    coeff, residual = random_pair(dense.q_count)
    torch.testing.assert_close(
        packed.forward(coeff), dense.forward(coeff), rtol=1e-11, atol=1e-11
    )
    torch.testing.assert_close(
        packed.adjoint(residual), dense.adjoint(residual), rtol=1e-11, atol=1e-11
    )


def test_approximate_packed_l_pair_is_adjoint_and_selected_z_consistent() -> None:
    context = small_context()
    plan = TorchCompositeOdtPlan.from_context(
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
    coeff, residual = random_pair(plan.q_count)
    forward = plan.forward(coeff)
    adjoint = plan.adjoint(residual)
    lhs = torch.vdot(forward.reshape(-1), residual.reshape(-1))
    rhs = torch.vdot(coeff.reshape(-1), adjoint.reshape(-1))
    dot_error = torch.abs(lhs - rhs) / (torch.abs(lhs) + torch.abs(rhs))
    assert float(dot_error) <= 1e-12

    z_indices = torch.tensor([0, 2], dtype=torch.long)
    selected = coeff.index_select(1, z_indices)
    restricted = torch.zeros_like(coeff)
    restricted.index_copy_(1, z_indices, selected)
    torch.testing.assert_close(
        plan.forward_selected_z(selected, z_indices),
        plan.forward(restricted),
        rtol=1e-11,
        atol=1e-11,
    )
    torch.testing.assert_close(
        plan.adjoint_selected_z(residual, z_indices),
        adjoint.index_select(1, z_indices),
        rtol=1e-11,
        atol=1e-11,
    )
