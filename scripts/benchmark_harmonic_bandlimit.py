from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from waxs_cake import PreparedCakePlan, make_cylindrical_histogram, nufft_amplitude  # noqa: E402
from waxs_cake.metrics import intensity, relative_l2  # noqa: E402


@dataclass(frozen=True)
class BandCase:
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
    wavelength: float = 1.0


CASES = [
    BandCase("base", 1_000_000, 40, 180, 48, 48, 2.2),
    BandCase("high_phi", 1_000_000, 40, 720, 48, 48, 2.2),
    BandCase("low_q", 1_000_000, 40, 180, 48, 48, 0.5),
    BandCase("dense_q", 1_000_000, 200, 180, 48, 48, 2.2),
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


def median_time(func, repeats: int):
    value = None
    times = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        value = func()
        times.append(time.perf_counter() - start)
    return value, float(median(times)), times


def run_case(
    case: BandCase,
    *,
    margins: list[int],
    repeats: int,
    seed: int,
    hist_backend: str,
    skip_nufft: bool,
) -> dict:
    coords = synthetic_cylinder(case.atoms, case.radius, case.height, seed)
    q = np.linspace(case.qmin, case.qmax, case.nq)
    phi = (np.arange(case.nphi) + 0.5) * (2.0 * np.pi / case.nphi)

    if hist_backend.startswith("numba"):
        make_cylindrical_histogram(
            coords[:512],
            n_r=case.nr,
            n_z=case.nz,
            n_phi=case.nphi,
            r_max=case.radius,
            z_range=(-0.5 * case.height, 0.5 * case.height),
            backend=hist_backend,
        )

    binned, hist_s, hist_times = median_time(
        lambda: make_cylindrical_histogram(
            coords,
            n_r=case.nr,
            n_z=case.nz,
            n_phi=case.nphi,
            r_max=case.radius,
            z_range=(-0.5 * case.height, 0.5 * case.height),
            backend=hist_backend,
        ),
        repeats,
    )
    exact_plan, plan_s, plan_times = median_time(
        lambda: PreparedCakePlan(binned, q, case.wavelength),
        repeats,
    )
    exact, exact_s, exact_times = median_time(exact_plan.circular_fft, repeats)

    nufft = None
    nufft_s = None
    nufft_times: list[float] = []
    if not skip_nufft:
        nufft, nufft_s, nufft_times = median_time(
            lambda: nufft_amplitude(coords, q, case.wavelength, phi),
            repeats,
        )

    rows = []
    for margin in margins:
        plan = PreparedCakePlan(
            binned,
            q,
            case.wavelength,
            harmonic_bandlimit_margin=margin,
        )
        amp, solve_s, solve_times = median_time(plan.circular_fft, repeats)
        hsel = plan._harmonic_indices_for_block(np.arange(q.size))
        active_modes = case.nphi if hsel is None else int(hsel.size)
        row = {
            "margin": margin,
            "active_modes": active_modes,
            "active_fraction": active_modes / case.nphi,
            "solve_s": solve_s,
            "solve_times": solve_times,
            "speedup_vs_exact_solve": exact_s / solve_s if solve_s else float("inf"),
            "amp_rel_l2_vs_exact_circular": relative_l2(amp, exact),
            "intensity_rel_l2_vs_exact_circular": relative_l2(
                intensity(amp),
                intensity(exact),
            ),
        }
        if nufft is not None:
            row["amp_rel_l2_vs_nufft"] = relative_l2(amp, nufft)
            row["intensity_rel_l2_vs_nufft"] = relative_l2(intensity(amp), intensity(nufft))
        rows.append(row)

    return {
        "case": asdict(case),
        "hist_backend": hist_backend,
        "hist_s": hist_s,
        "hist_times": hist_times,
        "plan_s": plan_s,
        "plan_times": plan_times,
        "exact_s": exact_s,
        "exact_times": exact_times,
        "nufft_s": nufft_s,
        "nufft_times": nufft_times,
        "margins": rows,
    }


def print_case(result: dict) -> None:
    case = result["case"]
    print(
        f"{case['name']}: hist={result['hist_s']:.4f}s "
        f"exact={result['exact_s']:.4f}s nufft={result['nufft_s']}"
    )
    print("margin\tmodes\tfrac\tsolve\tx_exact\tint_err_exact\tint_err_nufft")
    for row in result["margins"]:
        int_nufft = row.get("intensity_rel_l2_vs_nufft")
        print(
            "\t".join(
                [
                    str(row["margin"]),
                    str(row["active_modes"]),
                    f"{row['active_fraction']:.3f}",
                    f"{row['solve_s']:.4f}",
                    f"{row['speedup_vs_exact_solve']:.2f}",
                    f"{row['intensity_rel_l2_vs_exact_circular']:.3g}",
                    "" if int_nufft is None else f"{int_nufft:.3g}",
                ]
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="*", default=[case.name for case in CASES])
    parser.add_argument("--margins", nargs="+", type=int, default=[8, 16, 24, 32, 48])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument(
        "--hist-backend",
        choices=["numpy", "numba", "numba-parallel", "cpp"],
        default="numba-parallel",
    )
    parser.add_argument("--skip-nufft", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmark_results/harmonic_bandlimit.json"),
    )
    args = parser.parse_args()

    selected = [case for case in CASES if case.name in set(args.cases)]
    if not selected:
        raise ValueError("no cases selected")

    results = []
    for i, case in enumerate(selected):
        print(f"\n[{i + 1}/{len(selected)}] {case.name}")
        result = run_case(
            case,
            margins=args.margins,
            repeats=args.repeats,
            seed=args.seed + i,
            hist_backend=args.hist_backend,
            skip_nufft=args.skip_nufft,
        )
        results.append(result)
        print_case(result)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
