from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_odt_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_odt_ewald_cap_operator.py"
    spec = importlib.util.spec_from_file_location("benchmark_odt_ewald_cap_operator", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load ODT benchmark script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_cone_module():
    odt = _load_odt_module()
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    script = scripts_dir / "benchmark_odt_cone_illumination.py"
    spec = importlib.util.spec_from_file_location("benchmark_odt_cone_illumination", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load cone illumination benchmark script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.modules["benchmark_odt_ewald_cap_operator"] = odt
    spec.loader.exec_module(module)
    return module


def _load_cone_intensity_module():
    cone = _load_cone_module()
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    script = scripts_dir / "benchmark_odt_cone_cmd_intensity.py"
    spec = importlib.util.spec_from_file_location("benchmark_odt_cone_cmd_intensity", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load cone CMD intensity benchmark script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.modules["benchmark_odt_cone_illumination"] = cone
    spec.loader.exec_module(module)
    return module


def _load_cone_axis_module():
    cone = _load_cone_module()
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    script = scripts_dir / "benchmark_odt_cone_axis_decomposition.py"
    spec = importlib.util.spec_from_file_location("benchmark_odt_cone_axis_decomposition", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load cone axis-decomposition benchmark script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.modules["benchmark_odt_cone_illumination"] = cone
    spec.loader.exec_module(module)
    return module


def _load_same_direction_module():
    _load_cone_axis_module()
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    script = scripts_dir / "benchmark_odt_same_direction_zperp.py"
    spec = importlib.util.spec_from_file_location("benchmark_odt_same_direction_zperp", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load same-direction zperp benchmark script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_execute_only_module():
    _load_cone_axis_module()
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    script = scripts_dir / "profile_odt_cone_axis_execute_only.py"
    spec = importlib.util.spec_from_file_location("profile_odt_cone_axis_execute_only", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load execute-only cone-axis profile script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _small_shifted_case():
    odt = _load_odt_module()
    obj = odt.make_cylindrical_object(
        n_r=5,
        n_z=4,
        n_beta=32,
        r_max=1.0,
        z_max=0.7,
        phantom="beads",
        seed=123,
    )
    q = odt.ewald_cap_q_samples(
        k=4.0,
        detector_na=0.3,
        cap_radial=3,
        cap_phi=8,
        geometry="shifted",
        n_illum=3,
        illumination_na=0.12,
    )
    plan = odt.StructuredOdtPlan.build(
        r_axis=obj.r_axis,
        z_axis=obj.z_axis,
        beta_axis=obj.beta_axis,
        h_cutoff=odt.recommended_h_cutoff(q, 1.0, 32, 12),
    )
    return odt, obj, q, plan


def test_structured_odt_forward_adjoint_match_direct() -> None:
    odt, obj, q, plan = _small_shifted_case()
    kernel = odt.build_structured_kernel(plan, q)
    residual = odt.random_residual(q, seed=456)

    direct = odt.direct_forward(obj, q, chunk_q=32)
    structured = odt.structured_forward(plan, obj.coeff, q, kernel=kernel)
    direct_adj = odt.direct_adjoint(obj, q, residual, chunk_q=32)
    structured_adj = odt.structured_adjoint(plan, q, residual, kernel=kernel)

    assert odt.relative_l2(structured, direct) < 1e-12
    assert odt.relative_l2(structured_adj, direct_adj) < 1e-12
    assert odt.relative_complex_error(
        odt.complex_dot(structured, residual),
        odt.complex_dot(obj.coeff, structured_adj),
    ) < 1e-12


def test_structured_odt_cached_matches_uncached() -> None:
    odt, obj, q, plan = _small_shifted_case()
    residual = odt.random_residual(q, seed=789)
    kernel = odt.build_structured_kernel(plan, q)

    cached_forward = odt.structured_forward(plan, obj.coeff, q, kernel=kernel)
    uncached_forward = odt.structured_forward(plan, obj.coeff, q)
    cached_adjoint = odt.structured_adjoint(plan, q, residual, kernel=kernel)
    uncached_adjoint = odt.structured_adjoint(plan, q, residual)

    assert np.allclose(cached_forward, uncached_forward, rtol=0.0, atol=0.0)
    assert np.allclose(cached_adjoint, uncached_adjoint, rtol=0.0, atol=0.0)


def test_shifted_axis_factorization_matches_direct() -> None:
    odt, obj, q, _ = _small_shifted_case()
    residual = odt.random_residual(q, seed=1357)
    base_q = odt.ewald_cap_q_samples(
        k=4.0,
        detector_na=0.3,
        cap_radial=3,
        cap_phi=8,
        geometry="axis",
        n_illum=1,
        illumination_na=0.0,
    )
    plan = odt.StructuredOdtPlan.build(
        r_axis=obj.r_axis,
        z_axis=obj.z_axis,
        beta_axis=obj.beta_axis,
        h_cutoff=odt.recommended_h_cutoff(base_q, 1.0, 32, 12),
    )
    factorization = odt.build_shifted_axis_factorization(
        plan,
        k=4.0,
        detector_na=0.3,
        cap_radial=3,
        cap_phi=8,
        n_illum=3,
        illumination_na=0.12,
    )

    direct = odt.direct_forward(obj, q, chunk_q=32)
    factored = odt.structured_forward_shifted_axis_factored(
        plan,
        obj.coeff,
        factorization,
        backend="numpy",
    )
    fft_factored = odt.structured_forward_shifted_axis_fft_factored(
        plan,
        obj.coeff,
        factorization,
    )
    direct_adj = odt.direct_adjoint(obj, q, residual, chunk_q=32)
    factored_adj = odt.structured_adjoint_shifted_axis_factored(
        plan,
        factorization,
        residual,
        backend="numpy",
    )
    fft_factored_adj = odt.structured_adjoint_shifted_axis_fft_factored(
        plan,
        factorization,
        residual,
    )

    assert odt.relative_l2(factored, direct) < 1e-12
    assert odt.relative_l2(fft_factored, direct) < 1e-12
    assert odt.relative_l2(factored_adj, direct_adj) < 1e-12
    assert odt.relative_l2(fft_factored_adj, direct_adj) < 1e-12
    assert odt.relative_complex_error(
        odt.complex_dot(factored, residual),
        odt.complex_dot(obj.coeff, factored_adj),
    ) < 1e-12
    assert odt.relative_complex_error(
        odt.complex_dot(fft_factored, residual),
        odt.complex_dot(obj.coeff, fft_factored_adj),
    ) < 1e-12

    cpp_odt = odt._cpp_odt_module(required=False)
    if cpp_odt is not None and hasattr(cpp_odt, "phase_selected_dft"):
        selected_dft = odt.structured_forward_shifted_axis_fft_factored(
            plan,
            obj.coeff,
            factorization,
            backend="cpp",
            phase_backend="selected-dft",
        )
        selected_dft_adj = odt.structured_adjoint_shifted_axis_fft_factored(
            plan,
            factorization,
            residual,
            backend="cpp",
            phase_backend="selected-dft",
        )
        assert odt.relative_l2(selected_dft, direct) < 1e-12
        assert odt.relative_l2(selected_dft_adj, direct_adj) < 1e-12


def test_cone_cmd_modes_match_flat_shifted_q_list() -> None:
    cone = _load_cone_module()
    args = argparse.Namespace(
        n_beta=32,
        n_r=5,
        n_z=4,
        r_max=1.0,
        z_max=0.7,
        phantom="beads",
        seed=123,
        k=4.0,
        detector_na=0.3,
        cap_radial=3,
        cap_phi=8,
        h_cutoff=None,
        h_margin=20,
        cpp_threads=0,
    )
    obj = cone.make_cylindrical_object(
        n_r=args.n_r,
        n_z=args.n_z,
        n_beta=args.n_beta,
        r_max=args.r_max,
        z_max=args.z_max,
        phantom=args.phantom,
        seed=args.seed,
    )
    cache = cone.build_cone_mode_cache(
        args,
        obj=obj,
        n_illum=6,
        cmd_rank=3,
        illumination_na=0.15,
    )
    residual = cone.random_mode_residual(
        rank=3,
        base_count=cache.base_q.count,
        seed=2468,
    )
    flat_forward, flat_adjoint = cone.flat_mode_pair(
        args,
        obj=obj,
        cache=cache,
        residual_modes=residual,
        backend="numpy",
    )
    cone_forward, cone_adjoint = cone.cone_mode_pair(
        args,
        obj=obj,
        cache=cache,
        residual_modes=residual,
        backend="numpy",
    )

    assert cone.relative_l2(cone_forward, flat_forward) < 1e-12
    assert cone.relative_l2(cone_adjoint, flat_adjoint) < 1e-12
    assert cone.relative_complex_error(
        cone.complex_dot(cone_forward, residual),
        cone.complex_dot(obj.coeff, cone_adjoint),
    ) < 1e-12


def test_cone_cmd_intensity_matches_direct_csd() -> None:
    intensity = _load_cone_intensity_module()
    args = argparse.Namespace(
        n_beta=32,
        n_r=5,
        n_z=4,
        r_max=1.0,
        z_max=0.7,
        phantom="beads",
        seed=123,
        k=4.0,
        detector_na=0.3,
        cap_radial=3,
        cap_phi=8,
        h_cutoff=None,
        h_margin=20,
        cpp_threads=0,
        gaussian_sigma=1.25,
    )
    obj = intensity.make_cylindrical_object(
        n_r=args.n_r,
        n_z=args.n_z,
        n_beta=args.n_beta,
        r_max=args.r_max,
        z_max=args.z_max,
        phantom=args.phantom,
        seed=args.seed,
    )
    partial_cache = intensity.build_cone_mode_cache(
        args,
        obj=obj,
        n_illum=6,
        cmd_rank=3,
        illumination_na=0.15,
    )
    partial_eigs = intensity.mode_eigenvalues(
        family="cmd-gaussian",
        mode_orders=partial_cache.mode_orders,
        n_illum=6,
        gaussian_sigma=args.gaussian_sigma,
    )
    flat = intensity.flat_fields(args, obj=obj, cache=partial_cache, backend="numpy")
    cone_modes = intensity.cone_mode_fields(
        args,
        obj=obj,
        cache=partial_cache,
        backend="numpy",
    )
    partial = intensity.intensity_case_from_fields(
        flat=flat,
        cone_modes=cone_modes,
        weights=partial_cache.weights,
        eigenvalues=partial_eigs,
    )
    assert partial["mode_field_l2_vs_flat_modes"] < 1e-12
    assert partial["flat_mode_intensity_l2_vs_direct"] < 1e-12
    assert partial["cone_intensity_l2_vs_direct"] < 1e-12

    incoherent_cache = intensity.build_cone_mode_cache(
        args,
        obj=obj,
        n_illum=6,
        cmd_rank=6,
        illumination_na=0.15,
    )
    incoherent_eigs = intensity.mode_eigenvalues(
        family="incoherent",
        mode_orders=incoherent_cache.mode_orders,
        n_illum=6,
        gaussian_sigma=args.gaussian_sigma,
    )
    flat_full = intensity.flat_fields(args, obj=obj, cache=incoherent_cache, backend="numpy")
    cone_modes_full = intensity.cone_mode_fields(
        args,
        obj=obj,
        cache=incoherent_cache,
        backend="numpy",
    )
    incoherent = intensity.intensity_case_from_fields(
        flat=flat_full,
        cone_modes=cone_modes_full,
        weights=incoherent_cache.weights,
        eigenvalues=incoherent_eigs,
    )
    assert incoherent["cone_intensity_l2_vs_direct"] < 1e-12
    assert incoherent["incoherent_average_l2_vs_direct"] < 1e-12


def test_cone_axis_z0_zperp_decomposition_matches_phase_ramp() -> None:
    decomp_mod = _load_cone_axis_module()
    args = argparse.Namespace(
        n_beta=48,
        n_r=5,
        n_z=4,
        r_max=1.0,
        z_max=0.7,
        phantom="beads",
        seed=123,
        k=4.0,
        detector_na=0.3,
        cap_radial=3,
        cap_phi=8,
        h_cutoff=None,
        h_margin=20,
        cpp_threads=0,
    )
    obj = decomp_mod.make_cylindrical_object(
        n_r=args.n_r,
        n_z=args.n_z,
        n_beta=args.n_beta,
        r_max=args.r_max,
        z_max=args.z_max,
        phantom=args.phantom,
        seed=args.seed,
    )
    illumination, _ = decomp_mod.cone_illumination_directions(
        n_illum=6,
        illumination_na=0.15,
    )
    flat_q, base_q = decomp_mod.cone_q_samples(
        k=args.k,
        detector_na=args.detector_na,
        cap_radial=args.cap_radial,
        cap_phi=args.cap_phi,
        illumination=illumination,
    )
    plan = decomp_mod.StructuredOdtPlan.build(
        r_axis=obj.r_axis,
        z_axis=obj.z_axis,
        beta_axis=obj.beta_axis,
        h_cutoff=decomp_mod.recommended_h_cutoff(base_q, args.r_max, args.n_beta, 20),
    )
    exact = decomp_mod.build_exact_cone_factorization(
        plan,
        k=args.k,
        detector_na=args.detector_na,
        cap_radial=args.cap_radial,
        cap_phi=args.cap_phi,
        illumination_na=0.15,
        n_illum=6,
    )
    decomp = decomp_mod.build_cone_axis_decomposition(
        plan,
        k=args.k,
        detector_na=args.detector_na,
        cap_radial=args.cap_radial,
        cap_phi=args.cap_phi,
        illumination_na=0.15,
        n_illum=6,
        l_cutoff=13,
    )
    residual = decomp_mod.random_residual(flat_q, seed=9753)
    exact_forward = decomp_mod.structured_forward_shifted_axis_fft_factored(
        plan,
        obj.coeff,
        exact,
        backend="numpy",
    )
    decomposed_forward = decomp_mod.decomposed_forward(
        obj.coeff,
        decomp,
        backend="numpy",
        cpp_threads=0,
    )
    exact_adjoint = decomp_mod.structured_adjoint_shifted_axis_fft_factored(
        plan,
        exact,
        residual,
        backend="numpy",
    )
    decomposed_adjoint = decomp_mod.decomposed_adjoint(
        residual,
        decomp,
        backend="numpy",
        cpp_threads=0,
    )

    assert decomp_mod.relative_l2(decomposed_forward, exact_forward) < 1e-12
    assert decomp_mod.relative_l2(decomposed_adjoint, exact_adjoint) < 1e-12
    assert decomp_mod.relative_complex_error(
        decomp_mod.complex_dot(decomposed_forward, residual),
        decomp_mod.complex_dot(obj.coeff, decomposed_adjoint),
    ) < 1e-12

    cpp_odt = decomp_mod._cpp_odt_module(required=False)
    if cpp_odt is not None and hasattr(cpp_odt, "cone_axis_decompose_forward"):
        decomposed_forward_cpp = decomp_mod.decomposed_forward(
            obj.coeff,
            decomp,
            backend="cpp",
            cpp_threads=0,
            forward_mode="two-step",
        )
        decomposed_adjoint_cpp = decomp_mod.decomposed_adjoint(
            residual,
            decomp,
            backend="cpp",
            cpp_threads=0,
            adjoint_mode="two-step",
        )
        assert decomp_mod.relative_l2(decomposed_forward_cpp, exact_forward) < 1e-12
        assert decomp_mod.relative_l2(decomposed_adjoint_cpp, exact_adjoint) < 1e-12
        assert decomp_mod.relative_complex_error(
            decomp_mod.complex_dot(decomposed_forward_cpp, residual),
            decomp_mod.complex_dot(obj.coeff, decomposed_adjoint_cpp),
        ) < 1e-12
        if hasattr(cpp_odt, "cone_axis_forward_fold"):
            decomposed_forward_fused = decomp_mod.decomposed_forward(
                obj.coeff,
                decomp,
                backend="cpp",
                cpp_threads=0,
                forward_mode="fused",
            )
            assert decomp_mod.relative_l2(decomposed_forward_fused, exact_forward) < 1e-12
            assert decomp_mod.relative_complex_error(
                decomp_mod.complex_dot(decomposed_forward_fused, residual),
                decomp_mod.complex_dot(obj.coeff, decomposed_adjoint_cpp),
            ) < 1e-12
        if hasattr(cpp_odt, "cone_axis_adjoint_unfold_scatter"):
            decomposed_adjoint_fused = decomp_mod.decomposed_adjoint(
                residual,
                decomp,
                backend="cpp",
                cpp_threads=0,
                adjoint_mode="fused",
            )
            assert decomp_mod.relative_l2(decomposed_adjoint_fused, exact_adjoint) < 1e-12
            assert decomp_mod.relative_complex_error(
                decomp_mod.complex_dot(decomposed_forward_cpp, residual),
                decomp_mod.complex_dot(obj.coeff, decomposed_adjoint_fused),
            ) < 1e-12
        if (
            hasattr(cpp_odt, "cone_axis_forward_fold_pruned")
            and hasattr(cpp_odt, "cone_axis_adjoint_unfold_scatter_pruned")
        ):
            decomp_pruned = decomp_mod.build_cone_axis_decomposition(
                plan,
                k=args.k,
                detector_na=args.detector_na,
                cap_radial=args.cap_radial,
                cap_phi=args.cap_phi,
                illumination_na=0.15,
                n_illum=6,
                l_cutoff=13,
                adaptive_l_threshold=1e-14,
            )
            assert decomp_pruned.active_l_indices is not None
            assert decomp_pruned.active_l_indices.size < decomp_pruned.transverse_coeff.size
            decomposed_forward_pruned = decomp_mod.decomposed_forward(
                obj.coeff,
                decomp_pruned,
                backend="cpp",
                cpp_threads=0,
                forward_mode="fused",
            )
            if hasattr(cpp_odt, "cone_axis_forward_fold_pruned_partitioned"):
                import profile_odt_cone_axis_bottleneck as profile

                decomp_partitioned = decomp_pruned
                radial, axial, mode_phase, slots = profile.axis_factor_pack(decomp_partitioned)
                if np.unique(slots).size != slots.size:
                    decomp_partitioned = decomp_mod.build_cone_axis_decomposition(
                        plan,
                        k=args.k,
                        detector_na=args.detector_na,
                        cap_radial=args.cap_radial,
                        cap_phi=64,
                        illumination_na=0.15,
                        n_illum=6,
                        l_cutoff=13,
                        adaptive_l_threshold=1e-14,
                    )
                    radial, axial, mode_phase, slots = profile.axis_factor_pack(decomp_partitioned)
                assert np.unique(slots).size == slots.size
                compact_partitioned_reference = decomp_mod.decomposed_forward(
                    obj.coeff,
                    decomp_partitioned,
                    backend="cpp",
                    cpp_threads=0,
                    forward_mode="fused",
                )
                coeff_h_full = np.ascontiguousarray(
                    np.fft.ifft(obj.coeff, axis=2) * float(plan.n_beta)
                )
                folded_partitioned = cpp_odt.cone_axis_forward_fold_pruned_partitioned(
                    coeff_h_full,
                    radial,
                    axial,
                    mode_phase,
                    slots,
                    decomp_partitioned.transverse_coeff,
                    decomp_partitioned.psi_phase,
                    decomp_partitioned.axial_phase,
                    decomp_partitioned.source_slots,
                    decomp_partitioned.active_l_offsets,
                    decomp_partitioned.active_l_indices,
                    int(decomp_partitioned.factorization.cap_phi),
                    0,
                )
                forward_partitioned = np.fft.fft(folded_partitioned, axis=2).reshape(
                    decomp_partitioned.illumination_phi.size * decomp_partitioned.base_q.count
                )
                assert decomp_mod.relative_l2(
                    forward_partitioned,
                    compact_partitioned_reference,
                ) < 1e-12
            decomposed_adjoint_pruned = decomp_mod.decomposed_adjoint(
                residual,
                decomp_pruned,
                backend="cpp",
                cpp_threads=0,
                adjoint_mode="fused",
            )
            assert decomp_mod.relative_l2(decomposed_forward_pruned, decomposed_forward_cpp) < 1e-12
            assert decomp_mod.relative_l2(decomposed_adjoint_pruned, decomposed_adjoint_cpp) < 1e-12
            assert decomp_mod.relative_complex_error(
                decomp_mod.complex_dot(decomposed_forward_pruned, residual),
                decomp_mod.complex_dot(obj.coeff, decomposed_adjoint_pruned),
            ) < 1e-12


def test_same_direction_zperp_grouped_matches_phase_ramp() -> None:
    same_dir = _load_same_direction_module()
    obj = same_dir.make_cylindrical_object(
        n_r=5,
        n_z=4,
        n_beta=64,
        r_max=1.0,
        z_max=0.7,
        phantom="beads",
        seed=2468,
    )
    magnitudes = np.array([0.03, 0.09, 0.15], dtype=float)
    tmp_illum = same_dir.same_direction_illumination(
        magnitudes=magnitudes,
        direction_phi=0.35,
    )
    _, base_q = same_dir.same_direction_q_samples(
        k=4.0,
        detector_na=0.3,
        cap_radial=3,
        cap_phi=8,
        illumination=tmp_illum,
    )
    plan = same_dir.StructuredOdtPlan.build(
        r_axis=obj.r_axis,
        z_axis=obj.z_axis,
        beta_axis=obj.beta_axis,
        h_cutoff=same_dir.recommended_h_cutoff(base_q, 1.0, 64, 12),
    )
    decomp = same_dir.build_same_direction_decomposition(
        plan,
        k=4.0,
        detector_na=0.3,
        cap_radial=3,
        cap_phi=8,
        magnitudes=magnitudes,
        direction_phi=0.35,
        l_cutoff=10,
    )
    exact_factorization = same_dir.build_exact_factorization(plan, decomp, k=4.0)
    residual = same_dir.random_residual(decomp.flat_q, seed=8642)
    phase_forward = same_dir.structured_forward_shifted_axis_fft_factored(
        plan,
        obj.coeff,
        exact_factorization,
        backend="cpp",
        cpp_threads=0,
    )
    phase_adjoint = same_dir.structured_adjoint_shifted_axis_fft_factored(
        plan,
        exact_factorization,
        residual,
        backend="cpp",
        cpp_threads=0,
    )
    grouped_forward = same_dir.same_direction_forward(
        obj.coeff,
        decomp,
        backend="cpp",
        cpp_threads=0,
        grouped=True,
    )
    grouped_adjoint = same_dir.same_direction_adjoint(
        residual,
        decomp,
        backend="cpp",
        cpp_threads=0,
        grouped=True,
    )
    split_forward = same_dir.same_direction_forward(
        obj.coeff,
        decomp,
        backend="cpp",
        cpp_threads=0,
        grouped=False,
    )
    split_adjoint = same_dir.same_direction_adjoint(
        residual,
        decomp,
        backend="cpp",
        cpp_threads=0,
        grouped=False,
    )
    compression = same_dir.build_magnitude_svd(decomp)
    compressed_forward = same_dir.compressed_forward(
        obj.coeff,
        decomp,
        compression,
        rank=magnitudes.size,
        backend="cpp",
        cpp_threads=0,
    )
    compressed_adjoint = same_dir.compressed_adjoint(
        residual,
        decomp,
        compression,
        rank=magnitudes.size,
        backend="cpp",
        cpp_threads=0,
    )

    assert same_dir.relative_l2(grouped_forward, phase_forward) < 1e-12
    assert same_dir.relative_l2(grouped_adjoint, phase_adjoint) < 1e-12
    assert same_dir.relative_l2(split_forward, grouped_forward) < 1e-12
    assert same_dir.relative_l2(split_adjoint, grouped_adjoint) < 1e-12
    assert same_dir.relative_l2(compressed_forward, grouped_forward) < 1e-12
    assert same_dir.relative_l2(compressed_adjoint, grouped_adjoint) < 1e-12
    assert same_dir.relative_complex_error(
        same_dir.complex_dot(grouped_forward, residual),
        same_dir.complex_dot(obj.coeff, grouped_adjoint),
    ) < 1e-12


def test_cone_axis_prepared_batch_adjoint_matches_single_loop() -> None:
    profile = _load_execute_only_module()
    cpp_odt = profile._cpp_odt_module(required=False)
    if (
        cpp_odt is None
        or not hasattr(cpp_odt, "cone_axis_prepare_adjoint_pruned")
        or not hasattr(cpp_odt, "cone_axis_adjoint_unfold_scatter_pruned_prepared_batch")
    ):
        pytest.skip("native prepared cone-axis batch adjoint extension is unavailable")

    n_beta = 48
    obj = profile.make_cylindrical_object(
        n_r=5,
        n_z=4,
        n_beta=n_beta,
        r_max=1.0,
        z_max=0.7,
        phantom="beads",
        seed=97531,
    )
    illumination = profile.cone_illumination_directions(
        n_illum=6,
        illumination_na=0.15,
    )[0]
    flat_q, base_q = profile.cone_q_samples(
        k=4.0,
        detector_na=0.3,
        cap_radial=3,
        cap_phi=8,
        illumination=illumination,
    )
    plan = profile.StructuredOdtPlan.build(
        r_axis=obj.r_axis,
        z_axis=obj.z_axis,
        beta_axis=obj.beta_axis,
        h_cutoff=profile.recommended_h_cutoff(base_q, 1.0, n_beta, 12),
    )
    l_cutoff = profile.default_l_cutoff(
        k=4.0,
        illumination_na=0.15,
        r_max=1.0,
        margin=10,
        n_beta=n_beta,
    )
    decomp = profile.build_cone_axis_decomposition(
        plan,
        k=4.0,
        detector_na=0.3,
        cap_radial=3,
        cap_phi=8,
        illumination_na=0.15,
        n_illum=6,
        l_cutoff=l_cutoff,
        adaptive_l_threshold=1e-14,
    )
    if decomp.active_l_offsets is None or decomp.active_l_indices is None:
        pytest.skip("adaptive l-pruned cone-axis tables are unavailable")

    radial, axial, mode_phase, slots = profile.axis_factor_pack(decomp)
    native_prepared_tables = cpp_odt.cone_axis_prepare_adjoint_pruned(
        radial,
        axial,
        mode_phase,
        decomp.transverse_coeff,
        decomp.psi_phase,
        decomp.axial_phase,
        decomp.source_slots,
        decomp.active_l_offsets,
        decomp.active_l_indices,
    )
    ctx = profile.PreparedAdjointExecute(
        decomp=decomp,
        radial=radial,
        axial=axial,
        mode_phase=mode_phase,
        slots=slots,
        native_prepared_tables=native_prepared_tables,
        native_prepared_plan=None,
        native_prepared_plan_mode="direct",
        native_prepared_gather_threshold=8192,
        cap_radial=decomp.factorization.cap_radial,
        cap_phi=decomp.factorization.cap_phi,
        n_illum=int(decomp.illumination_phi.size),
        cpp_threads=0,
        cpp_odt=cpp_odt,
    )

    residual_batch = profile.random_residual_batch(flat_q, seed=24680, batch=3)
    batched = profile.prepared_adjoint_execute_batch(ctx, residual_batch)
    looped = profile.prepared_adjoint_execute_loop(ctx, residual_batch)

    assert profile.relative_l2(batched, looped) < 1e-12

    if hasattr(cpp_odt, "ConeAxisPreparedAdjointPlan"):
        native_plan = cpp_odt.ConeAxisPreparedAdjointPlan(
            slots,
            decomp.active_l_offsets,
            *native_prepared_tables,
            int(decomp.plan.n_beta),
        )
        residual_modes = profile.detector_ifft(ctx, residual_batch[0])
        tuple_out_h = cpp_odt.cone_axis_adjoint_unfold_scatter_pruned_prepared(
            residual_modes,
            slots,
            decomp.active_l_offsets,
            *native_prepared_tables,
            int(decomp.plan.n_beta),
            0,
        )
        plan_out_h = native_plan.execute(residual_modes, 0)
        gathered_out_h = native_plan.execute_gathered(residual_modes, 0)
        zmajor_out_h = native_plan.execute_gathered_zmajor(residual_modes, 0)
        assert profile.relative_l2(plan_out_h, tuple_out_h) < 1e-12
        assert profile.relative_l2(gathered_out_h, tuple_out_h) < 1e-12
        assert profile.relative_l2(zmajor_out_h, tuple_out_h) < 1e-12
