from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "WAXS_100_STATE_EPS_FRONTIER_PROTOCOL.json"
REQUEST_MANIFEST = ROOT / "PACKAGE_MANIFEST.json"
RETURN_NAME = "waxs_100_state_eps_frontier_external_return"
RETURN_ARCHIVE_NAME = f"{RETURN_NAME}.zip"
RETURN_MANIFEST_SCHEMA = "waxs-affine-library-eps-frontier-return-manifest-v2"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_run_bindings() -> dict[str, Any]:
    request_sha256 = os.environ.get("ACFO_REQUEST_SHA256", "").strip().lower()
    expected_manifest_sha256 = os.environ.get(
        "ACFO_REQUEST_MANIFEST_SHA256", ""
    ).strip().lower()
    run_id = os.environ.get("ACFO_RUN_ID", "").strip()
    return_dir_raw = os.environ.get("ACFO_RETURN_DIR", "").strip()
    if SHA256_PATTERN.fullmatch(request_sha256) is None:
        raise RuntimeError("ACFO_REQUEST_SHA256 must be a lowercase SHA-256 digest")
    if SHA256_PATTERN.fullmatch(expected_manifest_sha256) is None:
        raise RuntimeError(
            "ACFO_REQUEST_MANIFEST_SHA256 must be a lowercase SHA-256 digest"
        )
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise RuntimeError("ACFO_RUN_ID is missing or contains unsafe characters")
    if not return_dir_raw:
        raise RuntimeError("ACFO_RETURN_DIR is required")
    if not REQUEST_MANIFEST.is_file():
        raise RuntimeError(f"request manifest is missing: {REQUEST_MANIFEST}")
    actual_manifest_sha256 = sha256(REQUEST_MANIFEST)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise RuntimeError("request manifest does not match the launcher binding")
    return {
        "request_archive_sha256": request_sha256,
        "request_manifest_sha256": actual_manifest_sha256,
        "run_id": run_id,
        "return_dir": Path(return_dir_raw).expanduser().resolve(),
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def verify_manifest(bindings: dict[str, Any]) -> dict[str, Any]:
    path = REQUEST_MANIFEST
    manifest = load_json(path)
    failures = []
    for relative, expected in manifest["files"].items():
        candidate = ROOT / relative
        if not candidate.is_file():
            failures.append({"path": relative, "reason": "missing"})
            continue
        if candidate.stat().st_size != int(expected["bytes"]):
            failures.append({"path": relative, "reason": "bytes"})
        if sha256(candidate) != expected["sha256"]:
            failures.append({"path": relative, "reason": "sha256"})
    gates = {
        "schema": manifest.get("schema")
        == "waxs-affine-library-eps-frontier-request-manifest-v1",
        "package_name": manifest.get("package_name")
        == "waxs_100_state_eps_frontier_external_request_v1",
        "manifest_binding": sha256(path)
        == bindings["request_manifest_sha256"],
        "protocol_binding": manifest.get("protocol_sha256") == sha256(PROTOCOL),
        "files": not failures,
    }
    return {
        "schema": "waxs-eps-frontier-package-audit-v1",
        "file_count": len(manifest["files"]),
        "failures": failures,
        "gates": gates,
        "passed": all(gates.values()),
        "manifest_sha256": sha256(path),
        "protocol_sha256": sha256(PROTOCOL),
    }


def gpu_name() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.splitlines()[0].strip() if result.stdout.strip() else None


def environment_receipt() -> dict[str, Any]:
    import finufft
    import numpy
    import scipy

    return {
        "generated_at_utc": utc_now(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "finufft": getattr(finufft, "__version__", None),
        "cpu_count": os.cpu_count(),
        "gpu_name": gpu_name(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
    }


def run_logged(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            console_encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
            printable = line.encode(
                console_encoding, errors="backslashreplace"
            ).decode(console_encoding, errors="replace")
            print(printable, end="", flush=True)
            handle.write(line)
        return int(process.wait())


def _contains_timing_field(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if "timing" in lowered or "second" in lowered or lowered.endswith("_s"):
                return True
            if _contains_timing_field(child):
                return True
    elif isinstance(value, list):
        return any(_contains_timing_field(child) for child in value)
    return False


def audit_result(
    result: dict[str, Any], protocol: dict[str, Any], environment: dict[str, Any], mode: str
) -> dict[str, Any]:
    selection = result.get("epsilon_selection", {})
    rows = selection.get("rows", [])
    timing = result.get("final_timing") or {}
    harmonic = result.get("acfo_harmonic_accuracy", {})
    candidates = protocol["finufft"]["epsilon_candidates_loose_to_tight"]
    first_passing = next(
        (float(row["eps"]) for row in rows if row.get("accuracy_pass") is True), None
    )
    expected_samples = (
        int(protocol["final_timing"]["samples_per_arm"])
        if mode == "full"
        else int(protocol["smoke_overrides"]["samples_per_arm"])
    )
    expected_warmups = (
        int(protocol["final_timing"]["warmups"])
        if mode == "full"
        else int(protocol["smoke_overrides"]["warmups"])
    )
    independent = (
        mode != "full"
        or environment.get("gpu_name")
        == protocol["environment"]["preferred_workstation_gpu"]
    )
    gates = {
        "schema": result.get("schema")
        == "waxs-affine-library-eps-frontier-result-v1",
        "mode": result.get("mode") == mode,
        "protocol_sha256": result.get("protocol", {}).get("sha256")
        == sha256(PROTOCOL),
        "independent_machine": independent,
        "epsilon_order": [float(row.get("eps")) for row in rows]
        == [float(value) for value in candidates],
        "no_frontier_timing_fields": bool(rows)
        and not any(_contains_timing_field(row) for row in rows),
        "selected_is_first_passing": selection.get("selected_eps") == first_passing,
        "accuracy_pass": selection.get("accuracy_pass") is True,
        "harmonic_accuracy_pass": harmonic.get("accuracy_pass") is True,
        "fused_type3_n_trans_8": timing.get("finufft_plan_type") == 3
        and timing.get("finufft_n_trans") == 8,
        "selected_eps_retimed": timing.get("selected_eps")
        == selection.get("selected_eps"),
        "abba": timing.get("ordering") == "ABBA",
        "warmups": timing.get("warmup_count") == expected_warmups,
        "raw_samples": timing.get("samples_per_arm") == expected_samples
        and len(timing.get("acfo_seconds", {}).get("samples_s", []))
        == expected_samples
        and len(timing.get("baseline_seconds", {}).get("samples_s", []))
        == expected_samples,
        "full_external_contract": mode != "full"
        or result.get("decision", {}).get("external_contract_complete") is True,
    }
    return {
        "schema": "waxs-affine-library-eps-frontier-external-audit-v1",
        "generated_at_utc": utc_now(),
        "gates": gates,
        "passed": all(gates.values()),
        "performance_sign_is_gate": False,
        "selected_eps": selection.get("selected_eps"),
        "speedup": timing.get("baseline_over_acfo_speedup"),
    }


def write_return_archive(
    return_root: Path,
    *,
    bindings: dict[str, Any],
    mode: str,
    verdict: str,
    runner_exit_status: int,
) -> tuple[Path, Path]:
    if mode not in {"smoke", "full"} or verdict not in {"PASS", "FAIL"}:
        raise RuntimeError("return mode or verdict is invalid")
    if runner_exit_status not in {0, 2}:
        raise RuntimeError("runner exit status must be 0 or 2")
    if SHA256_PATTERN.fullmatch(
        str(bindings.get("request_archive_sha256", ""))
    ) is None:
        raise RuntimeError("request archive binding is not a SHA-256 digest")
    if bindings.get("request_manifest_sha256") != sha256(REQUEST_MANIFEST):
        raise RuntimeError("request manifest changed after run bindings were created")
    if RUN_ID_PATTERN.fullmatch(str(bindings.get("run_id", ""))) is None:
        raise RuntimeError("return run ID contains unsafe characters")
    shutil.copy2(PROTOCOL, return_root / PROTOCOL.name)
    shutil.copy2(REQUEST_MANIFEST, return_root / "REQUEST_PACKAGE_MANIFEST.json")
    manifest: dict[str, dict[str, Any]] = {}
    for path in sorted(item for item in return_root.rglob("*") if item.is_file()):
        relative = path.relative_to(return_root).as_posix()
        manifest[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    write_json(
        return_root / "RETURN_MANIFEST.json",
        {
            "schema": RETURN_MANIFEST_SCHEMA,
            "generated_at_utc": utc_now(),
            "request_archive_sha256": bindings["request_archive_sha256"],
            "request_manifest_sha256": bindings["request_manifest_sha256"],
            "protocol_sha256": sha256(PROTOCOL),
            "run_id": bindings["run_id"],
            "mode": mode,
            "verdict": verdict,
            "runner_exit_status": runner_exit_status,
            "files": manifest,
        },
    )
    return_dir = Path(bindings["return_dir"])
    return_dir.mkdir(parents=True, exist_ok=True)
    archive = return_dir / RETURN_ARCHIVE_NAME
    sidecar = return_dir / f"{RETURN_ARCHIVE_NAME}.sha256"
    for output in (archive, sidecar):
        if output.exists():
            raise RuntimeError(f"refusing to overwrite an existing return: {output}")
    with zipfile.ZipFile(
        archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as handle:
        for path in sorted(item for item in return_root.rglob("*") if item.is_file()):
            handle.write(
                path, f"{RETURN_NAME}/{path.relative_to(return_root).as_posix()}"
            )
    sidecar.write_text(f"{sha256(archive)}  {archive.name}\n", encoding="utf-8")
    return archive, sidecar


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    args = parser.parse_args()
    bindings = require_run_bindings()
    return_root = ROOT / f".{RETURN_NAME}_{bindings['run_id']}"
    if return_root.exists():
        raise RuntimeError(f"refusing to reuse return staging root: {return_root}")
    return_root.mkdir()
    evidence = return_root / "evidence"
    evidence.mkdir()
    protocol = load_json(PROTOCOL)
    package_audit = verify_manifest(bindings)
    environment = environment_receipt()
    write_json(evidence / "package_audit.json", package_audit)
    write_json(evidence / "environment.json", environment)

    build_status = run_logged(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        evidence / "build.log",
    )
    test_status = (
        run_logged(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_waxs_100_state_eps_frontier.py",
                "tests/test_waxs_100_state_eps_frontier_return.py",
            ],
            evidence / "tests.log",
        )
        if build_status == 0
        else 99
    )
    result_path = evidence / "waxs_100_state_eps_frontier.json"
    benchmark_status = (
        run_logged(
            [
                sys.executable,
                "scripts/benchmark_waxs_100_state_eps_frontier.py",
                "--protocol",
                str(PROTOCOL),
                "--mode",
                args.mode,
                "--output",
                str(result_path),
            ],
            evidence / "benchmark.log",
        )
        if test_status == 0
        else 99
    )
    if benchmark_status == 0 and result_path.is_file():
        audit = audit_result(load_json(result_path), protocol, environment, args.mode)
    else:
        audit = {
            "schema": "waxs-affine-library-eps-frontier-external-audit-v1",
            "generated_at_utc": utc_now(),
            "gates": {},
            "passed": False,
            "performance_sign_is_gate": False,
            "failure": "benchmark did not complete",
        }
    audit["process_status"] = {
        "build": build_status,
        "tests": test_status,
        "benchmark": benchmark_status,
    }
    audit["package_audit_passed"] = package_audit["passed"]
    audit["passed"] = bool(
        audit["passed"]
        and package_audit["passed"]
        and build_status == 0
        and test_status == 0
        and benchmark_status == 0
    )
    write_json(evidence / "EXTERNAL_RETURN_AUDIT.json", audit)
    verdict = "PASS" if audit["passed"] else "FAIL"
    runner_exit_status = 0 if audit["passed"] else 2
    summary = {
        "schema": "waxs-affine-library-eps-frontier-external-summary-v2",
        "generated_at_utc": utc_now(),
        "mode": args.mode,
        "verdict": verdict,
        "passed": audit["passed"],
        "runner_exit_status": runner_exit_status,
        "run_binding": {
            "request_archive_sha256": bindings["request_archive_sha256"],
            "request_manifest_sha256": bindings["request_manifest_sha256"],
            "protocol_sha256": sha256(PROTOCOL),
            "run_id": bindings["run_id"],
        },
        "selected_eps": audit.get("selected_eps"),
        "return_archive": RETURN_ARCHIVE_NAME,
        "return_sidecar": f"{RETURN_ARCHIVE_NAME}.sha256",
        "external_audit_sha256": sha256(
            evidence / "EXTERNAL_RETURN_AUDIT.json"
        ),
        "result_sha256": sha256(result_path) if result_path.is_file() else None,
    }
    write_json(evidence / "SUMMARY.json", summary)
    archive, sidecar = write_return_archive(
        return_root,
        bindings=bindings,
        mode=args.mode,
        verdict=verdict,
        runner_exit_status=runner_exit_status,
    )
    summary["return_archive_sha256"] = sha256(archive)
    summary["return_sidecar"] = sidecar.name
    print(json.dumps(summary, indent=2))
    shutil.rmtree(return_root)
    raise SystemExit(runner_exit_status)


if __name__ == "__main__":
    main()
