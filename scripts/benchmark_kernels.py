from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np
from scipy import special

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from waxs_cake import (  # noqa: E402
    PreparedCakePlan,
    estimate_bessel_cutoff,
    make_cylindrical_histogram,
)
from waxs_cake.metrics import relative_l2  # noqa: E402


def unit_complex_from_phase(phase: np.ndarray) -> np.ndarray:
    out = np.empty(phase.shape, dtype=np.complex128)
    out.real = np.cos(phase)
    out.imag = np.sin(phase)
    return out


@dataclass(frozen=True)
class KernelCase:
    name: str
    atoms: int
    nq: int
    nphi: int
    nr: int
    nz: int
    qmax: float
    radius: float = 20.0
    height: float = 20.0
    qmin: float = 0.05


CASES = [
    KernelCase("base_48_180", 1_000_000, 40, 180, 48, 48, 2.2),
    KernelCase("fine_96_180", 1_000_000, 40, 180, 96, 96, 2.2),
    KernelCase("high_phi_48_720", 1_000_000, 40, 720, 48, 48, 2.2),
    KernelCase("low_q_48_180", 1_000_000, 40, 180, 48, 48, 0.5),
    KernelCase("high_q_48_180", 1_000_000, 40, 180, 48, 48, 5.0),
]


def synthetic_cylinder(n_atoms: int, radius: float, height: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = radius * np.sqrt(rng.random(n_atoms))
    beta = 2.0 * np.pi * rng.random(n_atoms)
    z = height * (rng.random(n_atoms) - 0.5)
    coords = np.empty((n_atoms, 3), dtype=np.float64)
    coords[:, 0] = r * np.cos(beta)
    coords[:, 1] = r * np.sin(beta)
    coords[:, 2] = z
    return coords


def timed(func):
    start = time.perf_counter()
    value = func()
    return value, time.perf_counter() - start


def median_time(func, repeats: int):
    values = []
    last = None
    for _ in range(repeats):
        last, elapsed = timed(func)
        values.append(elapsed)
    return last, float(median(values)), values


def solve_with_kernel_blocks(plan: PreparedCakePlan, kernel_func) -> np.ndarray:
    indices = np.arange(plan.q.size)
    out = np.empty((indices.size, plan.binned.n_phi), dtype=np.complex128)
    for start in range(0, indices.size, plan.q_block_size):
        local = slice(start, min(start + plan.q_block_size, indices.size))
        sel = indices[local]
        khat = kernel_func(sel)
        z_reduced = np.einsum(
            "bz,rzh->brh", plan.z_phase[sel], plan.hhat[0], optimize=True
        )
        ahat = plan.form_factors[0, sel, None] * np.einsum(
            "brh,brh->bh", z_reduced, khat, optimize=True
        )
        out[local] = np.fft.ifft(ahat, axis=-1)
    return out


def analytic_bessel_khat(plan: PreparedCakePlan, sel: np.ndarray) -> np.ndarray:
    nphi = plan.binned.n_phi
    modes = np.fft.fftfreq(nphi, d=1.0 / nphi).astype(np.int64)
    x = plan.q_perp[sel, None, None] * plan.binned.r_centers[None, :, None]
    values = special.jv(modes[None, None, :], x)
    phase = np.exp(0.5j * np.pi * modes)
    return nphi * values * phase[None, None, :]


def analytic_cutoff_solve(plan: PreparedCakePlan) -> tuple[np.ndarray, np.ndarray]:
    nphi = plan.binned.n_phi
    modes = np.fft.fftfreq(nphi, d=1.0 / nphi).astype(np.int64)
    abs_modes = np.abs(modes)
    phase = np.exp(0.5j * np.pi * modes)
    cutoffs = np.array(
        [
            estimate_bessel_cutoff(x, tol=1e-8, n_phi=nphi)
            for x in plan.q_perp * plan.binned.r_max
        ],
        dtype=int,
    )
    out = np.empty((plan.q.size, nphi), dtype=np.complex128)
    for iq in range(plan.q.size):
        active = abs_modes <= cutoffs[iq]
        mode_values = modes[active]
        active_indices = np.flatnonzero(active)
        h_active = np.take(plan.hhat[0], active_indices, axis=-1)
        z_reduced = np.einsum(
            "z,rzh->rh",
            plan.z_phase[iq],
            h_active,
            optimize=True,
        )
        x = plan.q_perp[iq] * plan.binned.r_centers[:, None]
        khat = nphi * special.jv(mode_values[None, :], x) * phase[active][None, :]
        ahat = np.zeros(nphi, dtype=np.complex128)
        ahat[active_indices] = plan.form_factors[0, iq] * np.sum(
            z_reduced * khat, axis=0
        )
        out[iq] = np.fft.ifft(ahat)
    return out, cutoffs


class KernelInterpolationTable:
    def __init__(self, nphi: int, x_max: float, dx: float) -> None:
        self.nphi = int(nphi)
        self.dx = float(dx)
        self.x_grid = np.arange(0.0, x_max + 2.0 * dx, dx)
        angles = np.arange(nphi) * (2.0 * np.pi / nphi)
        phase = self.x_grid[:, None] * np.cos(angles)[None, :]
        kernel = unit_complex_from_phase(phase)
        self.table = np.fft.fft(kernel, axis=-1)

    def khat(self, plan: PreparedCakePlan, sel: np.ndarray) -> np.ndarray:
        x = plan.q_perp[sel, None] * plan.binned.r_centers[None, :]
        scaled = x / self.dx
        lower = np.floor(scaled).astype(np.int64)
        lower = np.clip(lower, 0, self.table.shape[0] - 2)
        frac = scaled - lower
        return (
            self.table[lower] * (1.0 - frac[..., None])
            + self.table[lower + 1] * frac[..., None]
        )


def run_case(case: KernelCase, repeats: int, seed: int, dx_values: list[float]) -> dict:
    coords = synthetic_cylinder(case.atoms, case.radius, case.height, seed)
    q = np.linspace(case.qmin, case.qmax, case.nq)
    binned = make_cylindrical_histogram(
        coords,
        n_r=case.nr,
        n_z=case.nz,
        n_phi=case.nphi,
        r_max=case.radius,
        z_range=(-0.5 * case.height, 0.5 * case.height),
        backend="numba-parallel",
    )
    plan = PreparedCakePlan(binned, q, 1.0)

    reference, reference_s, _ = median_time(lambda: plan.circular_fft(), repeats)
    cached_plan, cache_prepare_s = timed(
        lambda: PreparedCakePlan(binned, q, 1.0, cache_kernel_fft=True)
    )
    cached, cached_s, _ = median_time(lambda: cached_plan.circular_fft(), repeats)

    bessel_full, bessel_full_s, _ = median_time(
        lambda: solve_with_kernel_blocks(plan, lambda sel: analytic_bessel_khat(plan, sel)),
        max(1, min(repeats, 2)),
    )
    cutoff, cutoff_s, _ = median_time(lambda: analytic_cutoff_solve(plan)[0], repeats)

    rows = {
        "case": case.__dict__,
        "reference_s": reference_s,
        "cache_prepare_s": cache_prepare_s,
        "cached_s": cached_s,
        "cached_rel_l2": relative_l2(cached, reference),
        "bessel_full_s": bessel_full_s,
        "bessel_full_rel_l2": relative_l2(bessel_full, reference),
        "bessel_cutoff_s": cutoff_s,
        "bessel_cutoff_rel_l2": relative_l2(cutoff, reference),
        "interpolation": [],
    }

    x_max = float(np.max(plan.q_perp) * binned.r_max)
    for dx in dx_values:
        table, build_s = timed(lambda dx=dx: KernelInterpolationTable(case.nphi, x_max, dx))
        interp, interp_s, _ = median_time(
            lambda table=table: solve_with_kernel_blocks(
                plan, lambda sel, table=table: table.khat(plan, sel)
            ),
            repeats,
        )
        rows["interpolation"].append(
            {
                "dx": dx,
                "table_points": int(table.x_grid.size),
                "build_s": build_s,
                "solve_s": interp_s,
                "total_s": build_s + interp_s,
                "rel_l2": relative_l2(interp, reference),
            }
        )
    return rows


def print_case(result: dict) -> None:
    case = result["case"]
    print(
        f"{case['name']}: exact={result['reference_s']:.4f}s "
        f"cached={result['cached_s']:.4f}s "
        f"cache_build={result['cache_prepare_s']:.4f}s "
        f"bessel={result['bessel_full_s']:.4f}s "
        f"cutoff={result['bessel_cutoff_s']:.4f}s"
    )
    for item in result["interpolation"]:
        print(
            f"  interp dx={item['dx']}: build={item['build_s']:.4f}s "
            f"solve={item['solve_s']:.4f}s total={item['total_s']:.4f}s "
            f"rel={item['rel_l2']:.3g}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--cases", nargs="*", default=[case.name for case in CASES])
    parser.add_argument("--dx", nargs="*", type=float, default=[0.2, 0.1, 0.05])
    parser.add_argument("--out", type=Path, default=Path("benchmark_results/kernels.json"))
    args = parser.parse_args()

    selected = [case for case in CASES if case.name in set(args.cases)]
    if not selected:
        raise ValueError("no cases selected")

    make_cylindrical_histogram(
        synthetic_cylinder(256, 20.0, 20.0, args.seed),
        n_r=48,
        n_z=48,
        n_phi=180,
        r_max=20.0,
        z_range=(-10.0, 10.0),
        backend="numba-parallel",
    )

    results = []
    for i, case in enumerate(selected):
        print(f"\n[{i + 1}/{len(selected)}] {case.name}")
        result = run_case(case, args.repeats, args.seed + i, args.dx)
        results.append(result)
        print_case(result)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
