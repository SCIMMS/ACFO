from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

torch = pytest.importorskip("torch")

from benchmark_odt_realistic_geometry_reconstruction import build_composite_context  # noqa: E402
from benchmark_odt_torch_gpu_reconstruction import (  # noqa: E402
    TorchCompositeOdtPlan,
    parser,
)


def test_radial_illumination_streaming_matches_full_torch_operator() -> None:
    config = parser().parse_args([])
    config.n_beta = 16
    config.n_r = 4
    config.n_z = 3
    config.ring_illum = 3
    config.skip_axis_illumination = False
    config.cap_radial = 4
    config.cap_phi = 16
    config.h_margin = 2
    config.l_margin = 2
    config.cone_l_prune_threshold = 0.0
    config.cpp_threads = 1
    config.skip_native_prepared_adjoint = True
    config.compact_axisymmetric_kernel = True
    context = build_composite_context(config)
    full = TorchCompositeOdtPlan.from_context(
        context,
        torch=torch,
        device=torch.device("cpu"),
        dtype="complex128",
        low_memory_adjoint=True,
    )
    streamed = TorchCompositeOdtPlan.from_context(
        context,
        torch=torch,
        device=torch.device("cpu"),
        dtype="complex128",
        low_memory_adjoint=True,
        radial_block_size=2,
        illumination_block_size=1,
    )
    generator = torch.Generator().manual_seed(20260712)
    coeff = torch.complex(
        torch.randn((4, 3, 16), generator=generator, dtype=torch.float64),
        torch.randn((4, 3, 16), generator=generator, dtype=torch.float64),
    )
    residual = torch.complex(
        torch.randn(full.q_count, generator=generator, dtype=torch.float64),
        torch.randn(full.q_count, generator=generator, dtype=torch.float64),
    )

    torch.testing.assert_close(streamed.forward(coeff), full.forward(coeff), rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(streamed.adjoint(residual), full.adjoint(residual), rtol=1e-11, atol=1e-11)


def test_illumination_reduced_pair_matches_legacy_streaming() -> None:
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
    context = build_composite_context(config)
    legacy = TorchCompositeOdtPlan.from_context(
        context,
        torch=torch,
        device=torch.device("cpu"),
        dtype="complex128",
        low_memory_adjoint=True,
        radial_block_size=2,
        illumination_block_size=2,
        forward_mode="legacy",
        adjoint_mode="legacy",
    )
    reduced = TorchCompositeOdtPlan.from_context(
        context,
        torch=torch,
        device=torch.device("cpu"),
        dtype="complex128",
        low_memory_adjoint=True,
        radial_block_size=2,
        illumination_block_size=2,
        forward_mode="illumination-reduced",
        adjoint_mode="illumination-reduced",
    )
    generator = torch.Generator().manual_seed(20260713)
    residual = torch.complex(
        torch.randn(legacy.q_count, generator=generator, dtype=torch.float64),
        torch.randn(legacy.q_count, generator=generator, dtype=torch.float64),
    )

    coeff = torch.complex(
        torch.randn((4, 3, 16), generator=generator, dtype=torch.float64),
        torch.randn((4, 3, 16), generator=generator, dtype=torch.float64),
    )

    torch.testing.assert_close(
        reduced.forward(coeff), legacy.forward(coeff), rtol=1e-11, atol=1e-11
    )
    torch.testing.assert_close(
        reduced.adjoint(residual), legacy.adjoint(residual), rtol=1e-11, atol=1e-11
    )


def test_auto_adjoint_selects_reduced_ring_and_legacy_axis() -> None:
    config = parser().parse_args([])
    config.n_beta = 32
    config.n_r = 6
    config.n_z = 5
    config.ring_illum = 24
    config.skip_axis_illumination = False
    config.cap_radial = 8
    config.cap_phi = 32
    config.h_margin = 4
    config.l_margin = 4
    config.cpp_threads = 1
    context = build_composite_context(config)
    plan = TorchCompositeOdtPlan.from_context(
        context,
        torch=torch,
        device=torch.device("cpu"),
        dtype="complex128",
        low_memory_adjoint=True,
        radial_block_size=2,
        illumination_block_size=4,
        forward_mode="auto",
        adjoint_mode="auto",
    )

    assert plan.ring.resolved_forward_mode == "illumination-reduced"
    assert plan.ring.resolved_adjoint_mode == "illumination-reduced"
    assert plan.axis is not None
    assert plan.axis.resolved_forward_mode == "legacy"
    assert plan.axis.resolved_adjoint_mode == "legacy"


def test_selected_z_pair_matches_full_grid_restriction() -> None:
    config = parser().parse_args([])
    config.n_beta = 16
    config.n_r = 4
    config.n_z = 5
    config.ring_illum = 5
    config.skip_axis_illumination = False
    config.cap_radial = 4
    config.cap_phi = 16
    config.h_margin = 2
    config.l_margin = 2
    config.cone_l_prune_threshold = 0.0
    config.cpp_threads = 1
    config.compact_axisymmetric_kernel = True
    context = build_composite_context(config)
    plan = TorchCompositeOdtPlan.from_context(
        context,
        torch=torch,
        device=torch.device("cpu"),
        dtype="complex128",
        low_memory_adjoint=True,
        radial_block_size=2,
        illumination_block_size=2,
        forward_mode="illumination-reduced",
        adjoint_mode="illumination-reduced",
    )
    z_indices = torch.tensor([1, 3], dtype=torch.long)
    generator = torch.Generator().manual_seed(20260714)
    coeff_selected = torch.complex(
        torch.randn((4, 2, 16), generator=generator, dtype=torch.float64),
        torch.randn((4, 2, 16), generator=generator, dtype=torch.float64),
    )
    coeff_full = torch.zeros((4, 5, 16), dtype=torch.complex128)
    coeff_full.index_copy_(1, z_indices, coeff_selected)
    residual = torch.complex(
        torch.randn(plan.q_count, generator=generator, dtype=torch.float64),
        torch.randn(plan.q_count, generator=generator, dtype=torch.float64),
    )

    forward_selected = plan.forward_selected_z(coeff_selected, z_indices)
    adjoint_selected = plan.adjoint_selected_z(residual, z_indices)
    torch.testing.assert_close(
        forward_selected,
        plan.forward(coeff_full),
        rtol=1e-11,
        atol=1e-11,
    )
    torch.testing.assert_close(
        adjoint_selected,
        plan.adjoint(residual).index_select(1, z_indices),
        rtol=1e-11,
        atol=1e-11,
    )
    lhs = torch.vdot(forward_selected.reshape(-1), residual.reshape(-1))
    rhs = torch.vdot(coeff_selected.reshape(-1), adjoint_selected.reshape(-1))
    assert float(torch.abs(lhs - rhs) / (torch.abs(lhs) + torch.abs(rhs))) <= 1e-12


def test_selected_detector_harmonic_domain_preserves_norm_and_normal_operator() -> None:
    config = parser().parse_args([])
    config.n_beta = 16
    config.n_r = 4
    config.n_z = 5
    config.ring_illum = 5
    config.skip_axis_illumination = False
    config.cap_radial = 4
    config.cap_phi = 16
    config.h_margin = 2
    config.l_margin = 2
    config.cone_l_prune_threshold = 0.0
    config.cpp_threads = 1
    config.compact_axisymmetric_kernel = True
    context = build_composite_context(config)
    plan = TorchCompositeOdtPlan.from_context(
        context,
        torch=torch,
        device=torch.device("cpu"),
        dtype="complex128",
        low_memory_adjoint=True,
        radial_block_size=2,
        illumination_block_size=2,
        forward_mode="illumination-reduced",
        adjoint_mode="illumination-reduced",
    )
    assert plan.ring.slots_unique
    assert plan.axis is not None and plan.axis.slots_unique
    z_indices = torch.tensor([1, 3], dtype=torch.long)
    generator = torch.Generator().manual_seed(20260715)
    coeff = torch.complex(
        torch.randn((4, 2, 16), generator=generator, dtype=torch.float64),
        torch.randn((4, 2, 16), generator=generator, dtype=torch.float64),
    )
    pixel_data = plan.forward_selected_z(coeff, z_indices)
    mode_data = plan.forward_selected_z_modes(coeff, z_indices)
    torch.testing.assert_close(
        torch.linalg.vector_norm(mode_data),
        torch.linalg.vector_norm(pixel_data),
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        plan.adjoint_selected_z_modes(mode_data, z_indices),
        plan.adjoint_selected_z(pixel_data, z_indices),
        rtol=1e-11,
        atol=1e-11,
    )

    residual = torch.complex(
        torch.randn(plan.mode_q_count, generator=generator, dtype=torch.float64),
        torch.randn(plan.mode_q_count, generator=generator, dtype=torch.float64),
    )
    gradient = plan.adjoint_selected_z_modes(residual, z_indices)
    lhs = torch.vdot(mode_data.reshape(-1), residual.reshape(-1))
    rhs = torch.vdot(coeff.reshape(-1), gradient.reshape(-1))
    assert float(torch.abs(lhs - rhs) / (torch.abs(lhs) + torch.abs(rhs))) <= 1e-12
