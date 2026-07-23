from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark_results"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Step:
    label: str
    command: tuple[str, ...]
    outputs: tuple[Path, ...]
    timeout_s: int
    stdout_artifact: Path | None = None


def output(run_dir: Path, name: str) -> Path:
    return run_dir / name


def build_step_plan(
    python: str,
    run_dir: Path,
    *,
    mode: str,
    resume: bool = False,
) -> list[Step]:
    py = str(python)
    steps: list[Step] = [
        Step(
            "environment",
            (
                py,
                "scripts/collect_acfo_ncs_v14_environment.py",
                "--output",
                str(output(run_dir, "environment.json")),
            ),
            (output(run_dir, "environment.json"),),
            120,
        ),
        Step(
            "pip_freeze",
            (py, "-m", "pip", "freeze"),
            (output(run_dir, "pip_freeze.txt"),),
            120,
            stdout_artifact=output(run_dir, "pip_freeze.txt"),
        ),
        Step(
            "build_cpp_extensions",
            (py, "setup.py", "build_ext", "--inplace", "--force"),
            (output(run_dir, "build_ext_receipt.json"),),
            1200,
        ),
        Step(
            "pytest",
            (py, "-m", "pytest", "-q"),
            (output(run_dir, "pytest_receipt.json"),),
            900,
        ),
        Step(
            "odt_integrated_probe",
            (
                py,
                "scripts/benchmark_odt_banded_cartesian_final_packed.py",
                "--device",
                "cuda",
                "--output",
                str(output(run_dir, "odt_banded_cartesian_final_packed_probe.json")),
            ),
            (output(run_dir, "odt_banded_cartesian_final_packed_probe.json"),),
            1800,
        ),
    ]
    if mode == "quick":
        return steps

    waxs_prepared = [
        py,
        "scripts/benchmark_protein_lattice_finufft_512_abba.py",
        "--finufft-threads",
        "4",
        "--warmup-pairs",
        "10",
        "--measured-pairs",
        "30",
        "--factorized-backend",
        "prepared_fused",
        "--lattice-backend",
        "separable",
        "--output",
        str(output(run_dir, "waxs_prepared_1m_abba.json")),
        "--summary-md",
        str(output(run_dir, "waxs_prepared_1m_abba.md")),
    ]
    if resume:
        waxs_prepared.append("--resume")
    steps.extend(
        [
            Step(
                "waxs_prepared_1m_abba",
                tuple(waxs_prepared),
                (
                    output(run_dir, "waxs_prepared_1m_abba.json"),
                    output(run_dir, "waxs_prepared_1m_abba.md"),
                ),
                14400,
            ),
            Step(
                "waxs_detector_nq512_abba",
                (
                    py,
                    "scripts/benchmark_protein_nanocrystal_finufft_fair.py",
                    "structures/processed/protein_nanocrystal_lysozyme_1iee_3x3x3_fixed.npz",
                    "--source-mode",
                    "same_binned",
                    "--qmin",
                    "0.05",
                    "--qmax",
                    "6.3",
                    "--q-unit",
                    "inv_angstrom",
                    "--nq",
                    "512",
                    "--wavelength-nm",
                    "0.08",
                    "--bin-width-nm",
                    "0.1",
                    "--nphi-min",
                    "2250",
                    "--form-factor-model",
                    "xray_f0",
                    "--finufft-eps",
                    "1e-6",
                    "--finufft-threads",
                    "4",
                    "--finufft-q-block-size",
                    "8",
                    "--detector-label",
                    "EIGER2_X_4M_15p5keV_100mm",
                    "--detector-active-width-mm",
                    "155.1",
                    "--detector-active-height-mm",
                    "162.15",
                    "--detector-distance-mm",
                    "100.0",
                    "--warmups",
                    "10",
                    "--repeats",
                    "30",
                    "--timing-order",
                    "alternating",
                    "--output",
                    str(output(run_dir, "waxs_detector_nq512_abba.json")),
                ),
                (output(run_dir, "waxs_detector_nq512_abba.json"),),
                14400,
            ),
            Step(
                "odt_integrated_scale",
                (
                    py,
                    "scripts/benchmark_odt_banded_cartesian_final_packed_full_timing.py",
                    "--device",
                    "cuda",
                    "--accuracy-probe",
                    str(output(run_dir, "odt_banded_cartesian_final_packed_probe.json")),
                    "--output",
                    str(
                        output(
                            run_dir,
                            "odt_banded_cartesian_final_packed_full_timing.json",
                        )
                    ),
                ),
                (
                    output(
                        run_dir,
                        "odt_banded_cartesian_final_packed_full_timing.json",
                    ),
                ),
                3600,
            ),
            Step(
                "odt_direct_audit_c64",
                (
                    py,
                    "scripts/validate_odt_cufinufft_matched_error.py",
                    "--device",
                    "cuda",
                    "--cufinufft-dtype",
                    "complex64",
                    "--out",
                    str(
                        output(
                            run_dir,
                            "odt_cufinufft_matched_error_direct_subset_c64.json",
                        )
                    ),
                    "--summary-md",
                    str(
                        output(
                            run_dir,
                            "odt_cufinufft_matched_error_direct_subset_c64.md",
                        )
                    ),
                ),
                (
                    output(
                        run_dir,
                        "odt_cufinufft_matched_error_direct_subset_c64.json",
                    ),
                    output(
                        run_dir,
                        "odt_cufinufft_matched_error_direct_subset_c64.md",
                    ),
                ),
                3600,
            ),
            Step(
                "odt_direct_audit_c128",
                (
                    py,
                    "scripts/validate_odt_cufinufft_matched_error.py",
                    "--device",
                    "cuda",
                    "--cufinufft-dtype",
                    "complex128",
                    "--eps-values",
                    "1e-4,3e-5,1e-5,3e-6,1e-6,3e-7,1e-7,3e-8,1e-8",
                    "--out",
                    str(
                        output(
                            run_dir,
                            "odt_cufinufft_matched_error_direct_subset_c128.json",
                        )
                    ),
                    "--summary-md",
                    str(
                        output(
                            run_dir,
                            "odt_cufinufft_matched_error_direct_subset_c128.md",
                        )
                    ),
                ),
                (
                    output(
                        run_dir,
                        "odt_cufinufft_matched_error_direct_subset_c128.json",
                    ),
                    output(
                        run_dir,
                        "odt_cufinufft_matched_error_direct_subset_c128.md",
                    ),
                ),
                3600,
            ),
            Step(
                "odt_same_dtype_abba30",
                (
                    py,
                    "scripts/benchmark_odt_cufinufft_gpu_baseline.py",
                    "--device",
                    "cuda",
                    "--dtype",
                    "complex64",
                    "--cufinufft-dtype",
                    "same",
                    "--low-memory-adjoint",
                    "--radial-block-size",
                    "32",
                    "--illumination-block-size",
                    "4",
                    "--prune-axis-l0",
                    "--axial-lowrank-rank",
                    "16",
                    "--ring-adaptive-l-packed-threshold",
                    "1e-6",
                    "--skip-native-prepared-adjoint",
                    "--compact-axisymmetric-kernel",
                    "--n-beta",
                    "256",
                    "--n-r",
                    "256",
                    "--n-z",
                    "256",
                    "--r-max",
                    "1.0",
                    "--z-max",
                    "0.8",
                    "--phantom",
                    "random_beads",
                    "--seed",
                    "123",
                    "--k",
                    "17.307319527958313",
                    "--detector-na",
                    "0.9240924092409241",
                    "--illumination-angle-deg",
                    "49",
                    "--ring-illum",
                    "120",
                    "--cap-radial",
                    "256",
                    "--cap-phi",
                    "256",
                    "--h-cutoff",
                    "28",
                    "--h-margin",
                    "20",
                    "--l-margin",
                    "18",
                    "--cone-l-prune-threshold",
                    "1e-12",
                    "--cpp-threads",
                    "4",
                    "--forward-execute-mode",
                    "prepared",
                    "--forward-kernel-mode",
                    "partitioned",
                    "--finufft-eps",
                    "1e-12",
                    "--finufft-q-batch-size",
                    "1048576",
                    "--cufinufft-eps",
                    "1e-6",
                    "--cufinufft-plan-mode",
                    "plan",
                    "--repeats",
                    "1",
                    "--warmups",
                    "1",
                    "--pair-repeats",
                    "30",
                    "--pair-warmups",
                    "5",
                    "--out",
                    str(output(run_dir, "odt_same_dtype_abba30.json")),
                    "--csv",
                    str(output(run_dir, "odt_same_dtype_abba30.csv")),
                    "--summary-md",
                    str(output(run_dir, "odt_same_dtype_abba30.md")),
                ),
                (
                    output(run_dir, "odt_same_dtype_abba30.json"),
                    output(run_dir, "odt_same_dtype_abba30.csv"),
                    output(run_dir, "odt_same_dtype_abba30.md"),
                ),
                10800,
            ),
            Step(
                "odt_matched_c128_full_pair5",
                (
                    py,
                    "scripts/benchmark_odt_cufinufft_only.py",
                    "--device",
                    "cuda",
                    "--acfo-reference",
                    str(output(run_dir, "odt_same_dtype_abba30.json")),
                    "--accuracy-audit",
                    str(
                        output(
                            run_dir,
                            "odt_cufinufft_matched_error_direct_subset_c128.json",
                        )
                    ),
                    "--warmups",
                    "1",
                    "--repeats",
                    "1",
                    "--pair-warmups",
                    "2",
                    "--pair-repeats",
                    "5",
                    "--out",
                    str(
                        output(
                            run_dir,
                            "odt_cufinufft_matched_c128_full_pair5.json",
                        )
                    ),
                ),
                (
                    output(
                        run_dir,
                        "odt_cufinufft_matched_c128_full_pair5.json",
                    ),
                ),
                7200,
            ),
            Step(
                "odt_temporal_warm_start",
                (
                    py,
                    "scripts/benchmark_odt_banded_cartesian_temporal_warm_start.py",
                    "--device",
                    "cuda",
                    "--output",
                    str(
                        output(
                            run_dir,
                            "odt_banded_cartesian_temporal_warm_start.json",
                        )
                    ),
                    "--summary-md",
                    str(
                        output(
                            run_dir,
                            "odt_banded_cartesian_temporal_warm_start.md",
                        )
                    ),
                ),
                (
                    output(
                        run_dir,
                        "odt_banded_cartesian_temporal_warm_start.json",
                    ),
                    output(
                        run_dir,
                        "odt_banded_cartesian_temporal_warm_start.md",
                    ),
                ),
                7200,
            ),
        ]
    )
    return steps


def verify_manifest(root: Path) -> dict[str, Any]:
    path = root / "MANIFEST.json"
    if not path.is_file():
        return {
            "schema": "acfo-ncs-v14-manifest-verification-v1",
            "passed": False,
            "error": "MANIFEST.json is missing; run from an extracted v14 release root.",
        }
    manifest = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for item in manifest.get("files", []):
        candidate = root / item["path"]
        actual = sha256(candidate) if candidate.is_file() else None
        rows.append(
            {
                "path": item["path"],
                "exists": candidate.is_file(),
                "expected_sha256": item["sha256"],
                "actual_sha256": actual,
                "matches": actual == item["sha256"],
            }
        )
    return {
        "schema": "acfo-ncs-v14-manifest-verification-v1",
        "generated_at_utc": utc_now(),
        "release_schema": manifest.get("schema"),
        "manifest_sha256": sha256(path),
        "file_count": len(rows),
        "mismatch_count": sum(not row["matches"] for row in rows),
        "mismatches": [row for row in rows if not row["matches"]],
        "passed": bool(rows) and all(row["matches"] for row in rows),
    }


def step_complete(record: dict[str, Any], step: Step) -> bool:
    if record.get("returncode") != 0:
        return False
    expected = record.get("outputs", {})
    for path in step.outputs:
        if not path.is_file() or expected.get(str(path), {}).get("sha256") != sha256(path):
            return False
    return True


def run_step(step: Step, log_dir: Path) -> dict[str, Any]:
    started = utc_now()
    start = time.perf_counter()
    log_path = log_dir / f"{step.label}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    for path in step.outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    timed_out = False
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write(f"started_at_utc={started}\n")
        log.write("command=" + json.dumps(step.command, ensure_ascii=False) + "\n\n")
        log.flush()
        try:
            completed = subprocess.run(
                list(step.command),
                cwd=ROOT,
                stdout=subprocess.PIPE if step.stdout_artifact else log,
                stderr=subprocess.STDOUT,
                timeout=step.timeout_s,
                check=False,
                text=bool(step.stdout_artifact),
                encoding="utf-8" if step.stdout_artifact else None,
                errors="replace" if step.stdout_artifact else None,
            )
            returncode = int(completed.returncode)
            if step.stdout_artifact is not None:
                text = completed.stdout or ""
                step.stdout_artifact.write_text(text, encoding="utf-8")
                log.write(text)
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            returncode = 124
            log.write(f"\nTIMEOUT after {step.timeout_s} s: {exc}\n")
    duration = time.perf_counter() - start
    outputs = {
        str(path): {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in step.outputs
        if path.is_file()
    }
    return {
        "label": step.label,
        "command": list(step.command),
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "duration_s": duration,
        "timeout_s": step.timeout_s,
        "timed_out": timed_out,
        "returncode": returncode,
        "log": str(log_path),
        "outputs": outputs,
        "passed": returncode == 0 and len(outputs) == len(step.outputs),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def create_return_zip(run_dir: Path) -> tuple[Path, str]:
    target = run_dir.parent / f"{run_dir.name}_return_package.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(run_dir.rglob("*")):
            if path.is_file():
                archive.write(path, f"{run_dir.name}/{path.relative_to(run_dir).as_posix()}")
    return target, sha256(target)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the ACFO NCS v14 external validation and return one evidence ZIP."
    )
    parser.add_argument("--mode", choices=("quick", "full"), default="full")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--refresh-receipts", action="store_true")
    parser.add_argument("--allow-reference-machine", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir
    if run_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = RESULTS / f"external_acfo_ncs_v14_{stamp}_{args.mode}"
    elif not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    steps = build_step_plan(
        args.python,
        run_dir,
        mode=args.mode,
        resume=args.resume,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "run_dir": str(run_dir),
                    "steps": [
                        {
                            "label": step.label,
                            "command": list(step.command),
                            "outputs": [str(path) for path in step.outputs],
                            "timeout_s": step.timeout_s,
                        }
                        for step in steps
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if run_dir.exists() and not args.resume:
        raise SystemExit(
            f"run directory already exists: {run_dir}; use --resume or choose --run-dir"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / "logs"
    manifest_receipt = verify_manifest(ROOT)
    write_json(run_dir / "manifest_verification.json", manifest_receipt)
    if not manifest_receipt.get("passed"):
        raise SystemExit(manifest_receipt.get("error", "release manifest verification failed"))

    steps_path = run_dir / "steps.json"
    prior = load_json(steps_path) if args.resume and steps_path.is_file() else {"steps": []}
    prior_by_label = {row["label"]: row for row in prior.get("steps", [])}
    records: list[dict[str, Any]] = []
    overall_start = time.perf_counter()
    first_failure: str | None = None
    for step in steps:
        previous = prior_by_label.get(step.label)
        refresh_receipt = args.refresh_receipts and step.label in {
            "environment",
            "pip_freeze",
        }
        if (
            args.resume
            and previous
            and step_complete(previous, step)
            and not refresh_receipt
        ):
            record = {**previous, "resumed_skip": True}
            print(f"SKIP {step.label}: prior successful output hashes match", flush=True)
        else:
            print(f"RUN  {step.label}", flush=True)
            record = run_step(step, log_dir)
            if step.label in {"build_cpp_extensions", "pytest"}:
                receipt_name = (
                    "build_ext_receipt.json"
                    if step.label == "build_cpp_extensions"
                    else "pytest_receipt.json"
                )
                receipt_schema = (
                    "acfo-ncs-v14-build-ext-receipt-v1"
                    if step.label == "build_cpp_extensions"
                    else "acfo-ncs-v14-pytest-receipt-v1"
                )
                write_json(
                    run_dir / receipt_name,
                    {
                        "schema": receipt_schema,
                        "generated_at_utc": utc_now(),
                        "passed": record["returncode"] == 0,
                        "returncode": record["returncode"],
                        "duration_s": record["duration_s"],
                        "log": record["log"],
                    },
                )
                record["outputs"] = {
                    str(run_dir / receipt_name): {
                        "bytes": (run_dir / receipt_name).stat().st_size,
                        "sha256": sha256(run_dir / receipt_name),
                    }
                }
                record["passed"] = record["returncode"] == 0
        records.append(record)
        write_json(
            steps_path,
            {
                "schema": "acfo-ncs-v14-step-receipt-v1",
                "mode": args.mode,
                "run_dir": str(run_dir),
                "steps": records,
            },
        )
        if not record.get("passed"):
            first_failure = step.label
            print(f"FAIL {step.label}; packaging partial evidence", flush=True)
            break

    from validate_external_acfo_ncs_v14 import evaluate_run

    validation = evaluate_run(
        run_dir,
        mode=args.mode,
        allow_reference_machine=args.allow_reference_machine,
    )
    write_json(run_dir / "validation.json", validation)
    receipt = {
        "schema": "external-acfo-ncs-v14-run-receipt-v1",
        "generated_at_utc": utc_now(),
        "mode": args.mode,
        "run_dir": str(run_dir),
        "resume": args.resume,
        "refresh_receipts": args.refresh_receipts,
        "allow_reference_machine": args.allow_reference_machine,
        "duration_s": time.perf_counter() - overall_start,
        "first_failed_step": first_failure,
        "step_count_completed": len(records),
        "step_count_planned": len(steps),
        "steps_passed": all(record.get("passed") for record in records)
        and len(records) == len(steps),
        "validation": {
            key: validation.get(key)
            for key in (
                "package_smoke_pass",
                "functional_correctness_pass",
                "performance_replication_pass",
                "independent_machine_replication_pass",
                "publication_replication_pass",
                "execution_pass",
            )
        },
    }
    write_json(run_dir / "run_receipt.json", receipt)
    return_zip, return_hash = create_return_zip(run_dir)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    print(f"return_package={return_zip}")
    print(f"return_package_sha256={return_hash}")
    if not validation.get("execution_pass", False):
        raise SystemExit(2)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
