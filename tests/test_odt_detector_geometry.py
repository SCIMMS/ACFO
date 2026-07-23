from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_odt_cone_axis_decomposition import (  # noqa: E402
    build_cone_axis_decomposition,
    build_exact_cone_factorization,
    decomposed_forward,
)
from benchmark_odt_cone_illumination import (  # noqa: E402
    cone_illumination_directions,
)
from benchmark_odt_ewald_cap_operator import (  # noqa: E402
    StructuredOdtPlan,
    detector_radial_nodes,
    make_cylindrical_object,
    recommended_h_cutoff,
    structured_forward_shifted_axis_fft_factored,
)
from benchmark_odt_cone_illumination import cone_q_samples  # noqa: E402


def test_detector_radial_nodes_preserve_bounds_and_bias_outer_spacing() -> None:
    uniform = detector_radial_nodes(
        detector_na=0.9,
        cap_radial=16,
        sampling="uniform_rho",
    )
    theta = detector_radial_nodes(
        detector_na=0.9,
        cap_radial=16,
        sampling="uniform_theta",
    )
    outer = detector_radial_nodes(
        detector_na=0.9,
        cap_radial=16,
        sampling="outer_power",
        outer_power=2.0,
        min_fraction=0.25,
        max_fraction=0.9,
    )
    assert np.all(np.diff(uniform) > 0.0)
    assert np.all(np.diff(theta) > 0.0)
    assert np.all(np.diff(outer) > 0.0)
    assert np.diff(theta)[-1] < np.diff(uniform)[-1]
    assert np.diff(outer)[-1] < np.diff(outer)[0]
    assert outer[0] > 0.25 * 0.9
    assert outer[-1] < 0.9 * 0.9


def test_nonuniform_radial_nodes_preserve_cone_factorization() -> None:
    obj = make_cylindrical_object(
        n_r=5,
        n_z=4,
        n_beta=32,
        r_max=1.0,
        z_max=0.7,
        phantom="beads",
        seed=123,
    )
    illumination = cone_illumination_directions(
        n_illum=5, illumination_na=0.15
    )[0]
    _, base_q = cone_q_samples(
        k=4.0,
        detector_na=0.5,
        cap_radial=5,
        cap_phi=16,
        illumination=illumination,
        radial_sampling="uniform_theta",
    )
    plan = StructuredOdtPlan.build(
        r_axis=obj.r_axis,
        z_axis=obj.z_axis,
        beta_axis=obj.beta_axis,
        h_cutoff=recommended_h_cutoff(base_q, 1.0, 32, 6),
    )
    exact = build_exact_cone_factorization(
        plan,
        k=4.0,
        detector_na=0.5,
        cap_radial=5,
        cap_phi=16,
        illumination_na=0.15,
        n_illum=5,
        radial_sampling="uniform_theta",
    )
    decomp = build_cone_axis_decomposition(
        plan,
        k=4.0,
        detector_na=0.5,
        cap_radial=5,
        cap_phi=16,
        illumination_na=0.15,
        n_illum=5,
        l_cutoff=8,
        radial_sampling="uniform_theta",
    )
    exact_forward = structured_forward_shifted_axis_fft_factored(
        plan, obj.coeff, exact, backend="numpy"
    )
    candidate = decomposed_forward(
        obj.coeff, decomp, backend="numpy", cpp_threads=0
    )
    relative = np.linalg.norm(candidate - exact_forward) / np.linalg.norm(exact_forward)
    assert relative < 1e-12
