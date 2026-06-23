from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from waxs_cake import (  # noqa: E402
    encode_elements,
    make_cylindrical_histogram,
    make_cylindrical_histogram_indexed,
)


@dataclass(frozen=True)
class HistCase:
    name: str
    use_elements: bool
    weight_kind: str


CASES = [
    HistCase("single_unweighted", False, "none"),
    HistCase("single_real_weighted", False, "real"),
    HistCase("multi_unweighted", True, "none"),
    HistCase("multi_real_weighted", True, "real"),
    HistCase("multi_complex_weighted", True, "complex"),
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


def parse_hist_dtype(value: str) -> np.dtype | None:
    return None if value == "default" else np.dtype(value)


def max_abs_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    if candidate.shape != reference.shape:
        return float("inf")
    if candidate.size == 0:
        return 0.0
    return float(np.max(np.abs(candidate - reference)))


def make_case_inputs(
    case: HistCase,
    n_atoms: int,
    seed: int,
    *,
    include_strings: bool,
) -> tuple[list[str] | None, np.ndarray | None, tuple[str, ...] | None, np.ndarray | None]:
    rng = np.random.default_rng(seed)
    elements = None
    element_indices = None
    element_order = None
    if case.use_elements:
        element_order = ("C", "N", "O")
        element_indices = rng.integers(
            0,
            len(element_order),
            size=n_atoms,
            dtype=np.int64,
        )
        if include_strings:
            elements = [element_order[int(i)] for i in element_indices]

    weights = None
    if case.weight_kind == "real":
        weights = rng.normal(size=n_atoms)
    elif case.weight_kind == "complex":
        weights = rng.normal(size=n_atoms) + 1j * rng.normal(size=n_atoms)
    return elements, element_indices, element_order, weights


def run_case(
    case: HistCase,
    coords: np.ndarray,
    *,
    backends: list[str],
    element_input: str,
    repeats: int,
    radius: float,
    height: float,
    n_r: int,
    n_z: int,
    n_phi: int,
    seed: int,
    angle_lut_size: int,
    angle_lut_mode: str,
    hist_dtype: np.dtype | None,
) -> list[dict]:
    elements, element_indices, element_order, weights = make_case_inputs(
        case,
        coords.shape[0],
        seed,
        include_strings=element_input == "strings",
    )
    encode_elapsed = None
    encode_times = []
    if elements is not None:
        encoded, encode_elapsed, encode_times = median_time(
            lambda: encode_elements(elements, element_order=("C", "N", "O")),
            repeats,
        )
        element_indices, element_order = encoded

    kwargs = {
        "n_r": n_r,
        "n_z": n_z,
        "n_phi": n_phi,
        "r_max": radius,
        "z_range": (-0.5 * height, 0.5 * height),
    }
    if element_order is not None:
        kwargs["element_order"] = element_order

    if element_indices is not None:
        reference = make_cylindrical_histogram_indexed(
            coords,
            element_indices,
            n_elements=len(element_order),
            atom_weights=weights,
            backend="numpy",
            **kwargs,
        ).hist
    else:
        reference = make_cylindrical_histogram(
            coords,
            elements,
            atom_weights=weights,
            backend="numpy",
            **kwargs,
        ).hist

    rows = []
    for backend in backends:
        if backend.startswith("numba") and case != CASES[0]:
            note = "numba falls back to numpy for this case"
        else:
            note = ""

        def build_histogram(backend=backend):
            if element_input == "indexed" and element_indices is not None:
                return make_cylindrical_histogram_indexed(
                    coords,
                    element_indices,
                    n_elements=len(element_order),
                    atom_weights=weights,
                    backend=backend,
                    hist_dtype=hist_dtype,
                    angle_lut_size=angle_lut_size if backend == "cpp" else 0,
                    angle_lut_mode=angle_lut_mode,
                    **kwargs,
                ).hist
            return make_cylindrical_histogram(
                coords,
                elements,
                atom_weights=weights,
                backend=backend,
                hist_dtype=hist_dtype,
                angle_lut_size=angle_lut_size if backend == "cpp" else 0,
                angle_lut_mode=angle_lut_mode,
                **kwargs,
            ).hist

        try:
            if backend.startswith("numba"):
                preview = min(coords.shape[0], 256)
                if element_input == "indexed" and element_indices is not None:
                    make_cylindrical_histogram_indexed(
                        coords[:preview],
                        element_indices[:preview],
                        n_elements=len(element_order),
                        atom_weights=None if weights is None else weights[:preview],
                        backend=backend,
                        hist_dtype=hist_dtype,
                        angle_lut_size=0,
                        angle_lut_mode=angle_lut_mode,
                        **kwargs,
                    )
                else:
                    make_cylindrical_histogram(
                        coords[:preview],
                        None if elements is None else elements[:preview],
                        atom_weights=None if weights is None else weights[:preview],
                        backend=backend,
                        hist_dtype=hist_dtype,
                        angle_lut_size=0,
                        angle_lut_mode=angle_lut_mode,
                        **kwargs,
                    )

            got, elapsed, times = median_time(
                build_histogram,
                repeats,
            )
            rows.append(
                {
                    "case": case.name,
                    "element_input": element_input if case.use_elements else "single",
                    "preprocessing_included": bool(
                        case.use_elements and element_input == "strings"
                    ),
                    "backend": backend,
                    "seconds": elapsed,
                    "times": times,
                    "dtype": str(got.dtype),
                    "max_abs_error_vs_numpy": max_abs_error(got, reference),
                    "encode_seconds": encode_elapsed if elements is not None else None,
                    "encode_times": encode_times if elements is not None else [],
                    "note": note,
                }
            )
        except ImportError as exc:
            rows.append(
                {
                    "case": case.name,
                    "element_input": element_input if case.use_elements else "single",
                    "preprocessing_included": bool(
                        case.use_elements and element_input == "strings"
                    ),
                    "backend": backend,
                    "seconds": None,
                    "times": [],
                    "dtype": None,
                    "max_abs_error_vs_numpy": None,
                    "encode_seconds": encode_elapsed if elements is not None else None,
                    "encode_times": encode_times if elements is not None else [],
                    "note": f"skipped: {exc}",
                }
            )
    return rows


def print_rows(rows: list[dict]) -> None:
    print("case\tinput\tbackend\tseconds\tprep_included\tencode_s\tdtype\terror\tnote")
    for row in rows:
        seconds = "skip" if row["seconds"] is None else f"{row['seconds']:.5f}"
        encode_seconds = (
            ""
            if row.get("encode_seconds") is None
            else f"{row['encode_seconds']:.5f}"
        )
        error = "" if row["max_abs_error_vs_numpy"] is None else f"{row['max_abs_error_vs_numpy']:.3g}"
        print(
            "\t".join(
                [
                    row["case"],
                    row["element_input"],
                    row["backend"],
                    seconds,
                    "yes" if row.get("preprocessing_included") else "no",
                    encode_seconds,
                    str(row["dtype"]),
                    error,
                    row["note"],
                ]
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atoms", type=int, default=1_000_000)
    parser.add_argument("--radius", type=float, default=20.0)
    parser.add_argument("--height", type=float, default=20.0)
    parser.add_argument("--nr", type=int, default=48)
    parser.add_argument("--nz", type=int, default=48)
    parser.add_argument("--nphi", type=int, default=180)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--angle-lut-size", type=int, default=0)
    parser.add_argument(
        "--angle-lut-mode",
        choices=["nearest", "cubic"],
        default="nearest",
    )
    parser.add_argument(
        "--hist-dtype",
        choices=["default", "int64", "uint32", "float32", "float64", "complex64", "complex128"],
        default="default",
    )
    parser.add_argument(
        "--element-input",
        choices=["strings", "indexed", "both"],
        default="indexed",
        help=(
            "For multi-element cases, use pre-parsed integer element IDs by "
            "default. Use 'strings' only to audit label preprocessing overhead."
        ),
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["numpy", "numba", "numba-parallel", "cpp"],
        default=["numpy", "numba-parallel", "cpp"],
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=[case.name for case in CASES],
        default=[case.name for case in CASES],
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmark_results/histogram_backends.json"),
    )
    args = parser.parse_args()

    selected = [case for case in CASES if case.name in set(args.cases)]
    coords = synthetic_cylinder(args.atoms, args.radius, args.height, args.seed)

    rows = []
    for i, case in enumerate(selected):
        element_inputs = [args.element_input]
        if not case.use_elements:
            element_inputs = ["strings"]
        elif args.element_input == "both":
            element_inputs = ["strings", "indexed"]
        for element_input in element_inputs:
            print(f"\n[{i + 1}/{len(selected)}] {case.name} ({element_input})")
            case_rows = run_case(
                case,
                coords,
                backends=args.backends,
                element_input=element_input,
                repeats=args.repeats,
                radius=args.radius,
                height=args.height,
                n_r=args.nr,
                n_z=args.nz,
                n_phi=args.nphi,
                seed=args.seed + i + 1,
                angle_lut_size=args.angle_lut_size,
                angle_lut_mode=args.angle_lut_mode,
                hist_dtype=parse_hist_dtype(args.hist_dtype),
            )
            rows.extend(case_rows)
            print_rows(case_rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
