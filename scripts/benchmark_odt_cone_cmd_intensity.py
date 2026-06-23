from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from benchmark_odt_cone_illumination import (
    ROOT,
    build_cone_mode_cache,
    mode_order_sequence,
    parse_float_list,
    parse_int_list,
    source_mode_weights,
)
from benchmark_odt_ewald_cap_operator import (
    _axis_grid_forward_fft,
    make_cylindrical_object,
    relative_l2,
    resolve_structured_backend,
    structured_forward,
)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    value_f = float(value)
    if value_f == 0.0:
        return "0"
    if abs(value_f) < 1e-3 or abs(value_f) >= 1e3:
        return f"{value_f:.{digits}e}"
    return f"{value_f:.{digits}f}"


def median_time(func, *, repeats: int) -> tuple[Any, float, list[float]]:
    result = None
    times: list[float] = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        result = func()
        times.append(time.perf_counter() - start)
    if result is None:
        raise RuntimeError("timed function did not run")
    return result, float(median(times)), times


def speedup(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return numerator / denominator


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mode_eigenvalues(
    *,
    family: str,
    mode_orders: np.ndarray,
    n_illum: int,
    gaussian_sigma: float,
) -> np.ndarray:
    if family == "coherent":
        if mode_orders.shape != (1,) or int(mode_orders[0]) != 0:
            raise ValueError("coherent family requires only the zero mode")
        return np.ones(1, dtype=float)
    if family == "incoherent":
        if mode_orders.size != n_illum:
            raise ValueError("incoherent family requires the full discrete Fourier basis")
        return np.full(mode_orders.size, 1.0 / float(n_illum), dtype=float)
    if family == "cmd-gaussian":
        if gaussian_sigma <= 0.0:
            raise ValueError("gaussian_sigma must be positive")
        raw = np.exp(-0.5 * (mode_orders.astype(float) / float(gaussian_sigma)) ** 2)
        return raw / max(float(np.sum(raw)), 1e-300)
    raise ValueError("unknown mode family")


def source_csd_from_modes(weights: np.ndarray, eigenvalues: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.complex128)
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    if weights.ndim != 2 or eigenvalues.shape != (weights.shape[0],):
        raise ValueError("weights/eigenvalues shape mismatch")
    return np.ascontiguousarray((weights.T * eigenvalues[None, :]) @ weights.conj())


def direct_csd_intensity(fields: np.ndarray, csd: np.ndarray) -> np.ndarray:
    fields = np.asarray(fields, dtype=np.complex128)
    csd = np.asarray(csd, dtype=np.complex128)
    if fields.ndim != 2 or csd.shape != (fields.shape[0], fields.shape[0]):
        raise ValueError("field/CSD shape mismatch")
    intensity = np.einsum("ib,ij,jb->b", fields, csd, np.conj(fields), optimize=True)
    return np.real_if_close(intensity, tol=1000).real


def mode_intensity(mode_fields: np.ndarray, eigenvalues: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mode_fields = np.asarray(mode_fields, dtype=np.complex128)
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    if mode_fields.ndim != 2 or eigenvalues.shape != (mode_fields.shape[0],):
        raise ValueError("mode field/eigenvalue shape mismatch")
    contribution = eigenvalues[:, None] * np.abs(mode_fields) ** 2
    return np.sum(contribution, axis=0), contribution


def flat_fields(args: argparse.Namespace, *, obj: Any, cache: Any, backend: str) -> np.ndarray:
    return structured_forward(
        cache.flat_plan,
        obj.coeff,
        cache.flat_q,
        kernel=cache.flat_kernel,
        backend=backend,
        cpp_threads=args.cpp_threads,
    ).reshape(cache.weights.shape[1], cache.base_q.count)


def cone_mode_fields(args: argparse.Namespace, *, obj: Any, cache: Any, backend: str) -> np.ndarray:
    coeff_modes = cache.source_factors * obj.coeff[None, :, :, :]
    coeff_h_full = np.fft.ifft(coeff_modes, axis=3) * float(cache.cone_plan.n_beta)
    coeff_h_all = np.ascontiguousarray(coeff_h_full[:, :, :, cache.cone_plan.h_indices])
    return _axis_grid_forward_fft(
        cache.cone_plan,
        coeff_h_all,
        cache.factorization,
        backend=backend,
        cpp_threads=args.cpp_threads,
    ).reshape(cache.source_factors.shape[0], cache.base_q.count)


def intensity_case_from_fields(
    *,
    flat: np.ndarray,
    cone_modes: np.ndarray,
    weights: np.ndarray,
    eigenvalues: np.ndarray,
) -> dict[str, Any]:
    mode_from_flat = weights @ flat
    csd = source_csd_from_modes(weights, eigenvalues)
    direct = direct_csd_intensity(flat, csd)
    mode_from_flat_intensity, flat_contrib = mode_intensity(mode_from_flat, eigenvalues)
    cone_intensity, cone_contrib = mode_intensity(cone_modes, eigenvalues)
    incoherent_average = np.mean(np.abs(flat) ** 2, axis=0)
    total = max(float(np.sum(cone_intensity)), 1e-300)
    return {
        "direct": direct,
        "mode_from_flat": mode_from_flat_intensity,
        "cone": cone_intensity,
        "mode_field_l2_vs_flat_modes": relative_l2(cone_modes, mode_from_flat),
        "flat_mode_intensity_l2_vs_direct": relative_l2(mode_from_flat_intensity, direct),
        "cone_intensity_l2_vs_direct": relative_l2(cone_intensity, direct),
        "cone_intensity_l2_vs_flat_modes": relative_l2(cone_intensity, mode_from_flat_intensity),
        "incoherent_average_l2_vs_direct": relative_l2(incoherent_average, direct),
        "mode_power": np.sum(cone_contrib, axis=1),
        "mode_power_fraction": np.sum(cone_contrib, axis=1) / total,
        "flat_mode_power_fraction": np.sum(flat_contrib, axis=1)
        / max(float(np.sum(mode_from_flat_intensity)), 1e-300),
    }


def case_specs(args: argparse.Namespace, n_illum: int) -> list[tuple[str, int]]:
    specs: list[tuple[str, int]] = [("coherent", 1)]
    specs.extend(("cmd-gaussian", rank) for rank in args.cmd_ranks if rank <= n_illum)
    if args.include_incoherent:
        specs.append(("incoherent", n_illum))
    out: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for item in specs:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def benchmark_case(
    args: argparse.Namespace,
    *,
    n_illum: int,
    family: str,
    rank: int,
    illumination_na: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    obj = make_cylindrical_object(
        n_r=args.n_r,
        n_z=args.n_z,
        n_beta=args.n_beta,
        r_max=args.r_max,
        z_max=args.z_max,
        phantom=args.phantom,
        seed=args.seed,
    )
    backend = resolve_structured_backend(args.structured_backend)
    cache = build_cone_mode_cache(
        args,
        obj=obj,
        n_illum=n_illum,
        cmd_rank=rank,
        illumination_na=illumination_na,
        include_flat_kernel=True,
    )
    eigenvalues = mode_eigenvalues(
        family=family,
        mode_orders=cache.mode_orders,
        n_illum=n_illum,
        gaussian_sigma=args.gaussian_sigma,
    )
    if family == "incoherent":
        # For a full Fourier basis, this is exactly the same as cache.weights. Keeping
        # this explicit makes the normalization contract clear for the CSD check.
        cache_weights = source_mode_weights(
            np.linspace(0.0, 2.0 * np.pi, n_illum, endpoint=False, dtype=float),
            cache.mode_orders,
        )
    else:
        cache_weights = cache.weights

    flat, flat_hot_s, flat_times = median_time(
        lambda: flat_fields(args, obj=obj, cache=cache, backend=backend),
        repeats=args.hot_repeats,
    )
    cone_modes, cone_hot_s, cone_times = median_time(
        lambda: cone_mode_fields(args, obj=obj, cache=cache, backend=backend),
        repeats=args.hot_repeats,
    )
    result = intensity_case_from_fields(
        flat=flat,
        cone_modes=cone_modes,
        weights=cache_weights,
        eigenvalues=eigenvalues,
    )
    label = f"{family}:N{n_illum}:R{rank}"
    row: dict[str, Any] = {
        "status": "ok",
        "case": label,
        "family": family,
        "n_illum": n_illum,
        "cmd_rank": rank,
        "mode_orders": " ".join(str(int(item)) for item in cache.mode_orders),
        "mode_eigenvalues": " ".join(f"{float(item):.9g}" for item in eigenvalues),
        "illumination_na": illumination_na,
        "n_beta": args.n_beta,
        "n_r": args.n_r,
        "n_z": args.n_z,
        "cap_radial": args.cap_radial,
        "cap_phi": args.cap_phi,
        "flat_q_samples": cache.flat_q.count,
        "axis_q_samples": cache.base_q.count,
        "flat_hot_field_s": flat_hot_s,
        "cone_hot_field_s": cone_hot_s,
        "flat_vs_cone_field_speedup": speedup(flat_hot_s, cone_hot_s),
        "mode_field_l2_vs_flat_modes": result["mode_field_l2_vs_flat_modes"],
        "flat_mode_intensity_l2_vs_direct": result["flat_mode_intensity_l2_vs_direct"],
        "cone_intensity_l2_vs_direct": result["cone_intensity_l2_vs_direct"],
        "cone_intensity_l2_vs_flat_modes": result["cone_intensity_l2_vs_flat_modes"],
        "incoherent_average_l2_vs_direct": result["incoherent_average_l2_vs_direct"],
        "dominant_mode_power_fraction": float(np.max(result["mode_power_fraction"])),
        "flat_hot_field_times_s": " ".join(f"{item:.9g}" for item in flat_times),
        "cone_hot_field_times_s": " ".join(f"{item:.9g}" for item in cone_times),
    }
    mode_rows: list[dict[str, Any]] = []
    for local_index, (order, eigenvalue, fraction, flat_fraction) in enumerate(
        zip(
            cache.mode_orders,
            eigenvalues,
            result["mode_power_fraction"],
            result["flat_mode_power_fraction"],
            strict=True,
        )
    ):
        mode_rows.append(
            {
                "case": label,
                "family": family,
                "n_illum": n_illum,
                "cmd_rank": rank,
                "local_mode_index": local_index,
                "mode_order": int(order),
                "eigenvalue": float(eigenvalue),
                "cone_power_fraction": float(fraction),
                "flat_power_fraction": float(flat_fraction),
            }
        )
    return row, mode_rows


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    mode_rows = payload["mode_rows"]
    lines = [
        "# ODT cone CMD intensity-mode check",
        "",
        "This benchmark checks intensity formation for conical illumination modes.",
        "",
        "- Direct CSD intensity: `I(q)=sum_ij C_ij E_i(q) conj(E_j(q))` from individual illumination fields.",
        "- CMD intensity: `I(q)=sum_m lambda_m |E_m(q)|^2` using Fourier/CMD source modes.",
        "- Cone path: computes the same `E_m` fields directly from source-mode factors and axis-cap detector samples.",
        "- The incoherent case uses the full Fourier basis with equal eigenvalues, so it is equivalent to the average of angle-wise intensities.",
        "",
        "## Results",
        "",
        "| case | flat q | axis q | flat field s | cone field s | speedup | mode field err | flat CMD I err | cone I err | incoh avg err | dominant mode |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {case} | {fq} | {aq} | `{fs}` | `{cs}` | `{sp}` | `{mfe}` | `{fie}` | `{cie}` | `{iae}` | `{dom}` |".format(
                case=row["case"],
                fq=row["flat_q_samples"],
                aq=row["axis_q_samples"],
                fs=fmt(row["flat_hot_field_s"], 5),
                cs=fmt(row["cone_hot_field_s"], 5),
                sp=fmt(row["flat_vs_cone_field_speedup"], 4),
                mfe=fmt(row["mode_field_l2_vs_flat_modes"], 4),
                fie=fmt(row["flat_mode_intensity_l2_vs_direct"], 4),
                cie=fmt(row["cone_intensity_l2_vs_direct"], 4),
                iae=fmt(row["incoherent_average_l2_vs_direct"], 4),
                dom=fmt(row["dominant_mode_power_fraction"], 4),
            )
        )
    lines.extend(["", "## Mode Contributions", ""])
    for case in [row["case"] for row in rows]:
        subset = [item for item in mode_rows if item["case"] == case]
        subset.sort(key=lambda item: item["local_mode_index"])
        parts = [
            f"h={item['mode_order']}: lambda={item['eigenvalue']:.4g}, power={item['cone_power_fraction']:.4g}"
            for item in subset[: min(9, len(subset))]
        ]
        suffix = "" if len(subset) <= 9 else f" ... ({len(subset)} modes total)"
        lines.append(f"- `{case}`: " + "; ".join(parts) + suffix)
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Coherent cone illumination is rank 1 only when the illumination phases are locked into the uniform source mode.",
            "- A rotationally symmetric partial-coherence CSD is diagonal in these discrete Fourier modes; the intensity is then an incoherent sum over mode intensities.",
            "- Fully incoherent simultaneous illumination is not rank 1. It requires the full discrete Fourier basis and equals the average of angle-wise intensities.",
            "- The mode split is therefore a physics assumption about the source CSD, not a symmetry rule that automatically removes interference.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(path_png: Path, path_svg: Path, payload: dict[str, Any]) -> None:
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
    import matplotlib.pyplot as plt

    rows = payload["rows"]
    mode_rows = payload["mode_rows"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), dpi=180)
    ax_speed, ax_modes = axes
    ax_speed.axhline(1.0, color="#444444", lw=1.0, ls="--", alpha=0.75)
    ax_speed.grid(alpha=0.25)
    ax_speed.set_ylabel("flat / cone field speedup")
    ax_speed.set_xlabel("case")
    x = np.arange(len(rows))
    ax_speed.bar(x, [row["flat_vs_cone_field_speedup"] for row in rows], color="#2f78b7")
    ax_speed.set_xticks(x)
    ax_speed.set_xticklabels([row["case"].replace(":", "\n") for row in rows], rotation=0, fontsize=7)
    ax_speed.set_title("A. Field evaluation")

    selected_case = rows[-1]["case"] if rows else ""
    selected_modes = [row for row in mode_rows if row["case"] == selected_case]
    selected_modes.sort(key=lambda item: item["local_mode_index"])
    ax_modes.grid(alpha=0.25)
    ax_modes.bar(
        [str(item["mode_order"]) for item in selected_modes],
        [item["cone_power_fraction"] for item in selected_modes],
        color="#6d9f3a",
    )
    ax_modes.set_xlabel("source Fourier mode order")
    ax_modes.set_ylabel("power fraction")
    ax_modes.set_title(f"B. Mode power: {selected_case}")
    fig.tight_layout()
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png)
    fig.savefig(path_svg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check intensity formation for conical illumination CMD modes."
    )
    parser.add_argument("--output-prefix", default="benchmark_results/odt_cone_cmd_intensity")
    parser.add_argument("--n-illum-values", default="16")
    parser.add_argument("--cmd-ranks", default="3,5,9")
    parser.add_argument("--illumination-na-values", default="0.2")
    parser.add_argument("--include-incoherent", dest="include_incoherent", action="store_true")
    parser.add_argument("--no-incoherent", dest="include_incoherent", action="store_false")
    parser.set_defaults(include_incoherent=True)
    parser.add_argument("--gaussian-sigma", type=float, default=1.35)
    parser.add_argument("--phantom", choices=["beads", "random_beads", "shell"], default="beads")
    parser.add_argument("--n-r", type=int, default=16)
    parser.add_argument("--n-z", type=int, default=15)
    parser.add_argument("--n-beta", type=int, default=192)
    parser.add_argument("--r-max", type=float, default=1.5)
    parser.add_argument("--z-max", type=float, default=1.2)
    parser.add_argument("--k", type=float, default=8.0)
    parser.add_argument("--detector-na", type=float, default=0.45)
    parser.add_argument("--cap-radial", type=int, default=8)
    parser.add_argument("--cap-phi", type=int, default=32)
    parser.add_argument("--h-cutoff", type=int, default=None)
    parser.add_argument("--h-margin", type=int, default=18)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--structured-backend", choices=["auto", "numpy", "cpp"], default="cpp")
    parser.add_argument("--cpp-threads", type=int, default=0)
    parser.add_argument("--hot-repeats", type=int, default=3)
    args = parser.parse_args()
    args.n_illum_values = parse_int_list(args.n_illum_values)
    args.cmd_ranks = parse_int_list(args.cmd_ranks)
    args.illumination_na_values = parse_float_list(args.illumination_na_values)
    if args.gaussian_sigma <= 0.0:
        raise ValueError("gaussian-sigma must be positive")
    if args.cpp_threads < 0:
        raise ValueError("cpp-threads must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    mode_rows: list[dict[str, Any]] = []
    for illumination_na in args.illumination_na_values:
        for n_illum in args.n_illum_values:
            for family, rank in case_specs(args, n_illum):
                row, modes = benchmark_case(
                    args,
                    n_illum=n_illum,
                    family=family,
                    rank=rank,
                    illumination_na=illumination_na,
                )
                rows.append(row)
                mode_rows.extend(modes)
    payload = {
        "config": {
            **vars(args),
            "n_illum_values": args.n_illum_values,
            "cmd_ranks": args.cmd_ranks,
            "illumination_na_values": args.illumination_na_values,
        },
        "rows": rows,
        "mode_rows": mode_rows,
    }
    output_prefix = ROOT / args.output_prefix
    write_json(output_prefix.with_suffix(".json"), payload)
    write_csv(output_prefix.with_suffix(".csv"), rows)
    write_csv(output_prefix.with_name(output_prefix.name + "_modes.csv"), mode_rows)
    write_summary(output_prefix.with_name(output_prefix.name + "_summary.md"), payload)
    write_plot(
        output_prefix.with_name(output_prefix.name + ".png"),
        output_prefix.with_name(output_prefix.name + ".svg"),
        payload,
    )
    print(
        json.dumps(
            {
                "json": str(output_prefix.with_suffix(".json")),
                "csv": str(output_prefix.with_suffix(".csv")),
                "modes_csv": str(output_prefix.with_name(output_prefix.name + "_modes.csv")),
                "summary_md": str(output_prefix.with_name(output_prefix.name + "_summary.md")),
                "png": str(output_prefix.with_name(output_prefix.name + ".png")),
                "svg": str(output_prefix.with_name(output_prefix.name + ".svg")),
                "rows": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
