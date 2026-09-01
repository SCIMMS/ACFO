#!/usr/bin/env python3
"""Run and package the frozen representation-native ODT comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "ODT_NATIVE_REPRESENTATION_RTX3090_PROTOCOL.json"
EVIDENCE = ROOT / "evidence"
RETURN_ROOT = ROOT / "odt_native_representation_external_return"
RETURN_ARCHIVE = ROOT / "odt_native_representation_external_return.zip"
os.environ.setdefault("CUPY_CACHE_DIR", str(ROOT / ".cupy_cache"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def verify_package_manifest() -> dict[str, Any]:
    path = ROOT / "PACKAGE_MANIFEST.json"
    manifest = read_json(path)
    missing: list[str] = []
    mismatched: list[dict[str, str]] = []
    for relative, record in manifest["files"].items():
        candidate = ROOT / relative
        if not candidate.is_file():
            missing.append(relative)
            continue
        observed = sha256(candidate)
        if observed != record["sha256"]:
            mismatched.append(
                {"path": relative, "expected": record["sha256"], "observed": observed}
            )
    return {
        "manifest_sha256": sha256(path),
        "file_count": len(manifest["files"]),
        "missing": missing,
        "mismatched": mismatched,
        "passed": not missing and not mismatched,
    }


def environment_receipt() -> dict[str, Any]:
    import finufft
    import numpy
    import scipy
    import torch
    from benchmark_odt_cufinufft_gpu_baseline import import_cufinufft_modules

    cupy, cufinufft = import_cufinufft_modules()
    return {
        "generated_at_utc": utc_now(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cupy": cupy.__version__,
        "cufinufft": getattr(cufinufft, "__version__", None),
        "finufft": finufft.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cpu_count": os.cpu_count(),
    }


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "CUPY_CACHE_DIR": str(ROOT / ".cupy_cache")},
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def acfo_command(
    *, dtype: str, n: int, warmups: int, repeats: int, output: Path
) -> list[str]:
    dot_threshold = "1e-10" if dtype == "complex128" else "2e-5"
    return [
        sys.executable,
        "scripts/benchmark_odt_native_acfo_operator.py",
        "--dtype", dtype,
        "--n-beta", str(n),
        "--n-r", str(n),
        "--n-z", str(n),
        "--h-cutoff", "28",
        "--variant", "banded_inner96_outer64",
        "--timing-warmups", str(warmups),
        "--timing-repeats", str(repeats),
        "--dot-threshold", dot_threshold,
        "--output-native", str(output),
    ]


def nufft_command(
    *,
    dtype: str,
    n: int,
    warmups: int,
    repeats: int,
    output: Path,
    protocol: dict[str, Any],
) -> list[str]:
    track = protocol["precision_tracks"][dtype]
    dot_threshold = "1e-10" if dtype == "complex128" else "2e-5"
    return [
        sys.executable,
        "scripts/benchmark_odt_native_cartesian_cufinufft.py",
        "--dtype", dtype,
        "--n", str(n),
        "--cufinufft-eps", str(track["cufinufft_eps"]),
        "--variant", "banded_inner96_outer64",
        "--timing-warmups", str(warmups),
        "--timing-repeats", str(repeats),
        "--dot-threshold", dot_threshold,
        "--direct-forward-q-count", str(protocol["accuracy"]["cufinufft_direct_forward_q_count"]),
        "--direct-adjoint-voxel-count", str(protocol["accuracy"]["cufinufft_direct_adjoint_voxel_count"]),
        "--direct-threshold", str(track["direct_threshold"]),
        "--output", str(output),
    ]


def pair_summary(acfo: dict[str, Any], nufft: dict[str, Any]) -> dict[str, Any]:
    acfo_pair = float(acfo["timing"]["forward_adjoint_pair"]["median_s"])
    nufft_pair = float(nufft["timing"]["forward_adjoint_pair"]["median_s"])
    acfo_setup = float(acfo["preparation"]["total_s"])
    nufft_setup = float(nufft["preparation"]["method_total_s"])
    saved_per_pair = nufft_pair - acfo_pair
    break_even = None
    if saved_per_pair > 0.0:
        break_even = max(0.0, acfo_setup - nufft_setup) / saved_per_pair
    acfo_hash = acfo["problem"]["q_contract"]["q_sha256_qx_qy_qz_float64"]
    nufft_hash = nufft["problem"]["geometry"]["q_sha256_qx_qy_qz_float64"]
    gates = {
        "same_shape": acfo["problem"]["shape"] == nufft["problem"]["shape"],
        "same_degrees_of_freedom": acfo["problem"]["degrees_of_freedom"] == nufft["problem"]["degrees_of_freedom"],
        "same_q_count": acfo["problem"]["q_count"] == nufft["problem"]["q_count"],
        "same_q_hash": acfo_hash == nufft_hash,
        "same_dtype": acfo["problem"]["dtype"] == nufft["problem"]["dtype"],
        "acfo_dot_passed": bool(acfo["accuracy"]["passed"]),
        "nufft_direct_and_dot_passed": bool(nufft["accuracy"]["passed"]),
        "toeplitz_excluded": bool(
            not acfo["representation"]["toeplitz_used"]
            and not nufft["representation"]["toeplitz_used"]
        ),
    }
    return {
        "dtype": acfo["problem"]["dtype"],
        "n": int(acfo["problem"]["shape"][0]),
        "q_count": int(acfo["problem"]["q_count"]),
        "q_hash": acfo_hash,
        "acfo_setup_s": acfo_setup,
        "nufft_setup_s": nufft_setup,
        "acfo_forward_s": float(acfo["timing"]["forward"]["median_s"]),
        "acfo_adjoint_s": float(acfo["timing"]["adjoint"]["median_s"]),
        "acfo_pair_s": acfo_pair,
        "nufft_forward_s": float(nufft["timing"]["forward"]["median_s"]),
        "nufft_adjoint_s": float(nufft["timing"]["adjoint"]["median_s"]),
        "nufft_pair_s": nufft_pair,
        "acfo_speedup_over_nufft": float(nufft_pair / acfo_pair),
        "continuous_break_even_pairs_if_acfo_hot_faster": break_even,
        "acfo_resident_setup_mib": float(acfo["memory"]["resident_delta_after_setup_mib"]),
        "nufft_resident_setup_mib": float(nufft["memory"]["resident_delta_after_setup_mib"]),
        "acfo_dot_error": float(acfo["accuracy"]["forward_adjoint_dot_relative_error"]),
        "nufft_dot_error": float(nufft["accuracy"]["forward_adjoint_dot_relative_error"]),
        "nufft_direct_probe": nufft["accuracy"]["direct_probe"],
        "gates": gates,
        "integrity_passed": bool(all(gates.values())),
    }


def write_summary(path: Path, aggregate: dict[str, Any]) -> None:
    lines = [
        "# ODT representation-native ACFO versus Cartesian cuFINUFFT",
        "",
        "Toeplitz is excluded. Values above 1 in the speedup column mean ACFO has the faster hot forward-adjoint pair.",
        "",
        "| dtype | N | ACFO setup (s) | NUFFT setup (s) | ACFO pair (ms) | NUFFT pair (ms) | NUFFT/ACFO | break-even pairs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate["rows"]:
        break_even = row["continuous_break_even_pairs_if_acfo_hot_faster"]
        lines.append(
            f"| {row['dtype']} | {row['n']} | {row['acfo_setup_s']:.4f} | {row['nufft_setup_s']:.4f} | "
            f"{1e3 * row['acfo_pair_s']:.3f} | {1e3 * row['nufft_pair_s']:.3f} | "
            f"{row['acfo_speedup_over_nufft']:.3f}x | "
            + ("— |" if break_even is None else f"{break_even:.1f} |")
        )
    lines.extend(
        [
            "",
            "Performance is an observed outcome, not an integrity gate. The two methods use different native object discretizations under the same physical q contract.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def return_manifest() -> dict[str, Any]:
    files: dict[str, Any] = {}
    for path in sorted(RETURN_ROOT.rglob("*")):
        if path.is_file() and path.name != "RETURN_MANIFEST.json":
            files[path.relative_to(RETURN_ROOT).as_posix()] = {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
    manifest = {
        "schema": "odt-native-representation-return-manifest-v1",
        "generated_at_utc": utc_now(),
        "files": files,
    }
    write_json(RETURN_ROOT / "RETURN_MANIFEST.json", manifest)
    return manifest


def archive_return() -> None:
    if RETURN_ARCHIVE.exists():
        RETURN_ARCHIVE.unlink()
    with zipfile.ZipFile(RETURN_ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(RETURN_ROOT.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(ROOT).as_posix())
    RETURN_ARCHIVE.with_suffix(RETURN_ARCHIVE.suffix + ".sha256").write_text(
        f"{sha256(RETURN_ARCHIVE)}  {RETURN_ARCHIVE.name}\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--allow-reference-machine", action="store_true")
    args = parser.parse_args()
    protocol = read_json(PROTOCOL_PATH)
    package_audit = verify_package_manifest()
    environment = environment_receipt()
    if not environment["cuda_available"]:
        raise RuntimeError("CUDA is unavailable")
    expected_device = protocol["environment"]["required_device_name"]
    if environment["device_name"] != expected_device and not args.allow_reference_machine:
        raise RuntimeError(
            f"expected {expected_device}, observed {environment['device_name']}"
        )
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    write_json(EVIDENCE / "environment.json", environment)
    write_json(EVIDENCE / "package_audit.json", package_audit)

    sizes = [64] if args.mode == "smoke" else protocol["shared_physical_contract"]["grid_sizes"]
    dtypes = ["complex64"] if args.mode == "smoke" else ["complex64", "complex128"]
    warmups = protocol["timing"][f"{args.mode}_warmups"]
    repeats = protocol["timing"][f"{args.mode}_repeats"]
    rows: list[dict[str, Any]] = []
    for dtype in dtypes:
        for n in sizes:
            label = f"{dtype}_n{n}"
            acfo_path = EVIDENCE / f"acfo_{label}.json"
            nufft_path = EVIDENCE / f"cufinufft_{label}.json"
            run_logged(
                acfo_command(dtype=dtype, n=n, warmups=warmups, repeats=repeats, output=acfo_path),
                EVIDENCE / "logs" / f"acfo_{label}.log",
            )
            run_logged(
                nufft_command(
                    dtype=dtype,
                    n=n,
                    warmups=warmups,
                    repeats=repeats,
                    output=nufft_path,
                    protocol=protocol,
                ),
                EVIDENCE / "logs" / f"cufinufft_{label}.log",
            )
            rows.append(pair_summary(read_json(acfo_path), read_json(nufft_path)))

    anchor_path = ROOT / protocol["accuracy"]["acfo_complex128_anchor"]
    anchor = read_json(anchor_path)
    aggregate = {
        "schema": "odt-native-representation-external-aggregate-v1",
        "generated_at_utc": utc_now(),
        "mode": args.mode,
        "environment": environment,
        "package_audit": package_audit,
        "acfo_complex128_accuracy_anchor": {
            "path": protocol["accuracy"]["acfo_complex128_anchor"],
            "sha256": sha256(anchor_path),
            "selected_cufinufft_eps": anchor.get("selected_cufinufft_eps"),
            "worst_acfo_direct_relative_l2": anchor.get("worst_across_strata", {}).get("acfo"),
        },
        "rows": rows,
        "integrity_passed": bool(package_audit["passed"] and all(row["integrity_passed"] for row in rows)),
        "performance_is_not_gate": True,
        "toeplitz_included": False,
    }
    write_json(EVIDENCE / "aggregate.json", aggregate)
    write_summary(EVIDENCE / "summary.md", aggregate)

    if RETURN_ROOT.exists():
        import shutil

        shutil.rmtree(RETURN_ROOT)
    RETURN_ROOT.mkdir(parents=True)
    import shutil

    shutil.copy2(PROTOCOL_PATH, RETURN_ROOT / PROTOCOL_PATH.name)
    shutil.copytree(EVIDENCE, RETURN_ROOT / "evidence")
    return_manifest()
    archive_return()
    print(
        json.dumps(
            {
                "schema": aggregate["schema"],
                "mode": args.mode,
                "device": environment["device_name"],
                "integrity_passed": aggregate["integrity_passed"],
                "rows": rows,
                "archive": str(RETURN_ARCHIVE),
                "archive_sha256": sha256(RETURN_ARCHIVE),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not aggregate["integrity_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
