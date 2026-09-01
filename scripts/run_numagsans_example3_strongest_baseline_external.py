"""Run and package the prospective Example 3 strongest-baseline closure."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import threading
from time import perf_counter
from typing import Any
import zipfile


ROOT = Path(__file__).resolve().parents[1]
RETURN_NAME = "numagsans_example3_strongest_baseline_external_return"
RETURN_ARCHIVE_NAME = f"{RETURN_NAME}.zip"
RETURN_MANIFEST_SCHEMA = (
    "numagsans-example3-strongest-baseline-return-manifest-v1"
)
REQUEST_MANIFEST = ROOT / "MANIFEST.json"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_run_bindings() -> dict[str, Any]:
    request_sha256 = os.environ.get("ACFO_REQUEST_SHA256", "").strip().lower()
    run_id = os.environ.get("ACFO_RUN_ID", "").strip()
    return_dir_raw = os.environ.get("ACFO_RETURN_DIR", "").strip()
    if SHA256_PATTERN.fullmatch(request_sha256) is None:
        raise RuntimeError("ACFO_REQUEST_SHA256 must be a lowercase SHA-256 hex digest")
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise RuntimeError("ACFO_RUN_ID is missing or contains unsafe characters")
    if not return_dir_raw:
        raise RuntimeError("ACFO_RETURN_DIR is required")
    if not REQUEST_MANIFEST.is_file():
        raise RuntimeError(f"request manifest is missing: {REQUEST_MANIFEST}")
    return {
        "request_archive_sha256": request_sha256,
        "request_manifest_sha256": sha256(REQUEST_MANIFEST),
        "run_id": run_id,
        "return_dir": Path(return_dir_raw).expanduser().resolve(),
    }


def run_step(name: str, command: list[str], timeout: float) -> dict[str, Any]:
    start = perf_counter()
    log_dir = ROOT / "evidence/logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    tail: list[str] = []
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )

        def relay() -> None:
            assert process.stdout is not None
            with log_path.open("w", encoding="utf-8", errors="replace") as log:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                    tail.append(line)
                    while sum(len(item) for item in tail) > 12000:
                        tail.pop(0)

        thread = threading.Thread(target=relay, daemon=True)
        thread.start()
        timed_out = False
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            returncode = process.wait()
        thread.join(timeout=30)
        return {
            "name": name,
            "command": command,
            "returncode": returncode,
            "seconds": perf_counter() - start,
            "log": str(log_path.relative_to(ROOT)),
            "output_tail": "".join(tail)[-12000:],
            "timed_out": timed_out,
            "pass": returncode == 0 and not timed_out,
        }
    except Exception as exc:
        return {
            "name": name,
            "command": command,
            "returncode": None,
            "seconds": perf_counter() - start,
            "error": repr(exc),
            "log": str(log_path.relative_to(ROOT)),
            "pass": False,
        }


def package_return(
    protocol_path: Path,
    preflight_path: Path,
    *,
    mode: str,
    verdict: str,
    bindings: dict[str, Any],
) -> tuple[Path, Path]:
    evidence = ROOT / "evidence"
    if not evidence.is_dir():
        raise RuntimeError(f"evidence directory is missing: {evidence}")
    if not protocol_path.is_file():
        raise RuntimeError(f"protocol is missing: {protocol_path}")
    if not preflight_path.is_file():
        raise RuntimeError(f"preflight audit is missing: {preflight_path}")
    if not REQUEST_MANIFEST.is_file():
        raise RuntimeError(f"request manifest is missing: {REQUEST_MANIFEST}")
    if bindings.get("request_manifest_sha256") != sha256(REQUEST_MANIFEST):
        raise RuntimeError("request manifest changed after run bindings were created")
    if SHA256_PATTERN.fullmatch(
        str(bindings.get("request_archive_sha256", ""))
    ) is None:
        raise RuntimeError("request archive binding is not a SHA-256 digest")
    if RUN_ID_PATTERN.fullmatch(str(bindings.get("run_id", ""))) is None:
        raise RuntimeError("return run ID contains unsafe characters")
    if mode not in {"smoke", "full", "fatal"} or verdict not in {"PASS", "FAIL"}:
        raise RuntimeError("return mode or verdict is invalid")
    return_dir = Path(bindings["return_dir"])
    return_dir.mkdir(parents=True, exist_ok=True)
    archive = return_dir / RETURN_ARCHIVE_NAME
    sidecar = return_dir / f"{RETURN_ARCHIVE_NAME}.sha256"
    for output in (archive, sidecar):
        if output.exists():
            raise RuntimeError(f"refusing to overwrite an existing return: {output}")
    return_root = ROOT / f".{RETURN_NAME}_{bindings['run_id']}"
    if return_root.exists():
        raise RuntimeError(f"refusing to reuse return staging root: {return_root}")
    return_root.mkdir()
    try:
        shutil.copytree(evidence, return_root / "evidence")
        shutil.copy2(protocol_path, return_root / protocol_path.name)
        shutil.copy2(preflight_path, return_root / preflight_path.name)
        shutil.copy2(REQUEST_MANIFEST, return_root / "REQUEST_MANIFEST.json")
        files: dict[str, dict[str, Any]] = {}
        for path in sorted(item for item in return_root.rglob("*") if item.is_file()):
            relative = path.relative_to(return_root).as_posix()
            files[relative] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        manifest = {
            "schema": RETURN_MANIFEST_SCHEMA,
            "created_utc": utc_now(),
            "request_archive_sha256": bindings["request_archive_sha256"],
            "request_manifest_sha256": bindings["request_manifest_sha256"],
            "protocol_sha256": sha256(protocol_path),
            "run_id": bindings["run_id"],
            "mode": mode,
            "verdict": verdict,
            "files": files,
        }
        (return_root / "RETURN_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        with zipfile.ZipFile(
            archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as handle:
            for path in sorted(
                item for item in return_root.rglob("*") if item.is_file()
            ):
                relative = path.relative_to(return_root).as_posix()
                handle.write(path, f"{RETURN_NAME}/{relative}")
        archive_sha256 = sha256(archive)
        sidecar.write_text(
            f"{archive_sha256}  {archive.name}\n", encoding="utf-8"
        )
    finally:
        if return_root.exists():
            shutil.rmtree(return_root)
    print(f"return_archive={archive}", flush=True)
    print(f"return_archive_sha256={archive_sha256}", flush=True)
    print(f"return_sidecar={sidecar}", flush=True)
    return archive, sidecar


def command_output(command: list[str]) -> str:
    try:
        return subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        ).stdout.strip()
    except Exception as exc:
        return repr(exc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT / "external_data/numagsans_example3"
    )
    parser.add_argument("--skip-acquisition", action="store_true")
    args = parser.parse_args()
    bindings = require_run_bindings()
    protocol_path = ROOT / "NUMAGSANS_EXAMPLE3_STRONGEST_BASELINE_PROTOCOL.json"
    preflight_path = ROOT / "preflight/PROSPECTIVE_PROTOCOL_AUDIT.json"
    evidence = ROOT / "evidence"
    if evidence.exists():
        shutil.rmtree(evidence)
    evidence.mkdir(parents=True)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    preflight_pass = bool(
        preflight["verdict"] == "PASS_PROSPECTIVE_PROTOCOL_FROZEN"
        and all(preflight["checks"].values())
    )
    steps: list[dict[str, Any]] = [
        {
            "name": "prospective_protocol_preflight",
            "pass": preflight_pass,
            "verdict": preflight["verdict"],
        }
    ]
    steps.append(
        run_step(
            "tests",
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_numagsans_example3_ensemble.py",
                "tests/test_numagsans_affine_fft.py",
                "tests/test_numagsans_example3_strongest_baseline_return.py",
                "tests/test_magnetic_sans_torch.py",
                "tests/test_voxel_fft_torch.py",
                "-q",
            ],
            1800,
        )
    )
    if args.mode == "full" and not args.skip_acquisition:
        steps.append(
            run_step(
                "archive_acquisition",
                [
                    sys.executable,
                    "scripts/acquire_numagsans_example3_archives.py",
                    "--protocol",
                    str(protocol_path),
                    "--output-dir",
                    str(args.data_dir),
                    "--receipt",
                    str(ROOT / "evidence/archive_acquisition.json"),
                ],
                14400,
            )
        )
    source_archive = args.data_dir / protocol["dataset"]["source_archive"]["name"]
    output_archive = args.data_dir / protocol["dataset"]["output_archive"]["name"]
    if args.mode == "full" and all(step.get("pass") for step in steps):
        steps.append(
            run_step(
                "gpu_resource_limited_smoke",
                [
                    sys.executable,
                    "scripts/benchmark_numagsans_example3_strongest_baseline.py",
                    "--protocol",
                    str(protocol_path),
                    "--reduced-dir",
                    str(ROOT / "inputs/numagsans_example3_reduced"),
                    "--max-orientations",
                    "1",
                    "--smoke-q-nodes",
                    "16",
                    "--smoke-unique-theta",
                    "127",
                    "--maximum-fft-pad-factor",
                    "12",
                    "--skip-archive-oracle",
                    "--output",
                    str(ROOT / "evidence/gpu_smoke.json"),
                ],
                1800,
            )
        )
    result_path = ROOT / "evidence/numagsans_example3_strongest_baseline.json"
    command = [
        sys.executable,
        "scripts/benchmark_numagsans_example3_strongest_baseline.py",
        "--protocol",
        str(protocol_path),
        "--output",
        str(result_path),
    ]
    if args.mode == "full":
        command.extend(
            [
                "--source-archive",
                str(source_archive),
                "--output-archive",
                str(output_archive),
            ]
        )
    else:
        command.extend(
            [
                "--reduced-dir",
                str(ROOT / "inputs/numagsans_example3_reduced"),
                "--max-orientations",
                "1",
                "--smoke-q-nodes",
                "16",
                "--smoke-unique-theta",
                "127",
                "--maximum-fft-pad-factor",
                "12",
                "--skip-archive-oracle",
            ]
        )
    if all(step.get("pass") for step in steps):
        steps.append(
            run_step(
                "strongest_baseline_benchmark",
                command,
                86400 if args.mode == "full" else 1800,
            )
        )
    else:
        steps.append(
            {
                "name": "strongest_baseline_benchmark",
                "pass": False,
                "skipped": True,
                "reason": "preflight, tests, acquisition, or smoke failed",
            }
        )
    audit_path = ROOT / "evidence/AUDIT.json"
    if result_path.is_file() and args.mode == "full":
        steps.append(
            run_step(
                "result_verification",
                [
                    sys.executable,
                    "scripts/verify_numagsans_example3_strongest_baseline.py",
                    str(result_path),
                    "--protocol",
                    str(protocol_path),
                    "--output",
                    str(audit_path),
                ],
                300,
            )
        )
    environment = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "nvidia_smi": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "pip_freeze": command_output([sys.executable, "-m", "pip", "freeze"]),
    }
    (evidence / "environment.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )
    audit = (
        json.loads(audit_path.read_text(encoding="utf-8"))
        if audit_path.is_file()
        else None
    )
    passed = bool(
        preflight_pass
        and all(step.get("pass") for step in steps)
        and (
            args.mode != "full"
            or (audit is not None and audit["verdict"].startswith("PASS"))
        )
    )
    summary = {
        "schema": "numagsans-example3-strongest-baseline-external-summary-v1",
        "created_utc": utc_now(),
        "mode": args.mode,
        "verdict": "PASS" if passed else "FAIL",
        "run_binding": {
            "request_archive_sha256": bindings["request_archive_sha256"],
            "request_manifest_sha256": bindings["request_manifest_sha256"],
            "protocol_sha256": sha256(protocol_path),
            "run_id": bindings["run_id"],
        },
        "steps": steps,
        "audit": audit,
    }
    (evidence / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    package_return(
        protocol_path,
        preflight_path,
        mode=args.mode,
        verdict=summary["verdict"],
        bindings=bindings,
    )
    print(json.dumps(summary, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        evidence = ROOT / "evidence"
        evidence.mkdir(parents=True, exist_ok=True)
        fatal = {
            "schema": "numagsans-example3-strongest-baseline-fatal-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(exc).__name__,
            "error": repr(exc),
        }
        (evidence / "FATAL_ERROR.json").write_text(
            json.dumps(fatal, indent=2), encoding="utf-8"
        )
        print(json.dumps(fatal, indent=2), file=sys.stderr, flush=True)
        try:
            bindings = require_run_bindings()
            package_return(
                ROOT / "NUMAGSANS_EXAMPLE3_STRONGEST_BASELINE_PROTOCOL.json",
                ROOT / "preflight/PROSPECTIVE_PROTOCOL_AUDIT.json",
                mode="fatal",
                verdict="FAIL",
                bindings=bindings,
            )
        except BaseException as package_exc:
            print(f"return packaging failed: {package_exc!r}", file=sys.stderr)
        raise SystemExit(3)
