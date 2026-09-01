from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

torch = pytest.importorskip("torch")

from benchmark_odt_optimization_sweep import (  # noqa: E402
    axial_svd,
    composite_adjoint_lowrank,
    composite_forward_lowrank,
    lowrank_basis,
    make_torch_plan,
)
from benchmark_odt_realistic_geometry_reconstruction import (  # noqa: E402
    build_composite_context,
)
from benchmark_odt_torch_gpu_reconstruction import parser  # noqa: E402


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
    config.cone_l_prune_threshold = 1e-12
    config.cpp_threads = 1
    config.skip_native_prepared_adjoint = True
    config.compact_axisymmetric_kernel = True
    return build_composite_context(config)


def test_axis_l0_pruning_is_exact() -> None:
    context = small_context()
    full = make_torch_plan(
        context,
        torch=torch,
        device=torch.device("cpu"),
        dtype="complex128",
        radial_block_size=2,
        illumination_block_size=2,
        prune_axis_l0=False,
    )
    pruned = make_torch_plan(
        context,
        torch=torch,
        device=torch.device("cpu"),
        dtype="complex128",
        radial_block_size=2,
        illumination_block_size=2,
        prune_axis_l0=True,
    )
    assert full.axis is not None and pruned.axis is not None
    assert full.axis.n_l > 1
    assert pruned.axis.n_l == 1
    generator = torch.Generator().manual_seed(20260713)
    coeff = torch.complex(
        torch.randn((4, 3, 16), generator=generator, dtype=torch.float64),
        torch.randn((4, 3, 16), generator=generator, dtype=torch.float64),
    )
    residual = torch.complex(
        torch.randn(full.q_count, generator=generator, dtype=torch.float64),
        torch.randn(full.q_count, generator=generator, dtype=torch.float64),
    )
    torch.testing.assert_close(pruned.forward(coeff), full.forward(coeff), rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(pruned.adjoint(residual), full.adjoint(residual), rtol=1e-11, atol=1e-11)


def test_full_rank_axial_factor_matches_exact_operator() -> None:
    context = small_context()
    plan = make_torch_plan(
        context,
        torch=torch,
        device=torch.device("cpu"),
        dtype="complex128",
        radial_block_size=2,
        illumination_block_size=2,
        prune_axis_l0=True,
    )
    ring_svd = axial_svd(plan.ring)
    ring_basis = lowrank_basis(plan.ring, ring_svd, rank=plan.ring.n_z)
    assert plan.axis is not None
    axis_svd = axial_svd(plan.axis)
    axis_basis = lowrank_basis(plan.axis, axis_svd, rank=plan.axis.n_z)
    generator = torch.Generator().manual_seed(20260714)
    coeff = torch.complex(
        torch.randn((4, 3, 16), generator=generator, dtype=torch.float64),
        torch.randn((4, 3, 16), generator=generator, dtype=torch.float64),
    )
    residual = torch.complex(
        torch.randn(plan.q_count, generator=generator, dtype=torch.float64),
        torch.randn(plan.q_count, generator=generator, dtype=torch.float64),
    )
    torch.testing.assert_close(
        composite_forward_lowrank(plan, coeff, ring_basis, axis_basis),
        plan.forward(coeff),
        rtol=1e-11,
        atol=1e-11,
    )
    torch.testing.assert_close(
        composite_adjoint_lowrank(plan, residual, ring_basis, axis_basis),
        plan.adjoint(residual),
        rtol=1e-11,
        atol=1e-11,
    )
