from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STEP_LABEL = "waxs_detector_nq512_abba"
RESULT_JSON = "waxs_detector_nq512_abba.json"
RESULT_MD = "waxs_detector_nq512_abba.md"
TIMEOUT_S = 14400


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def build_detector_command(python: Path, release_root: Path, output: Path) -> list[str]:
    return [
        str(python),
        str(release_root / "scripts/benchmark_protein_nanocrystal_finufft_fair.py"),
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
        str(output),
    ]


def amend_step_records(
    steps_payload: dict[str, Any],
    replacement: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = list(steps_payload.get("steps", []))
    matches = [index for index, row in enumerate(rows) if row.get("label") == STEP_LABEL]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {STEP_LABEL!r} record, found {len(matches)}")
    index = matches[0]
    original = rows[index]
    rows[index] = replacement
    return {**steps_payload, "steps": rows}, original


def preserve_original(target: Path, name: str, preserved_name: str) -> Path:
    source = target / name
    if not source.is_file():
        raise FileNotFoundError(source)
    preserved = target / preserved_name
    shutil.copy2(source, preserved)
    return preserved


def create_return_zip(run_dir: Path) -> tuple[Path, str]:
    target = run_dir.parent / f"{run_dir.name}_return_package.zip"
    with zipfile.ZipFile(
        target,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for path in sorted(run_dir.rglob("*")):
            if path.is_file():
                archive.write(
                    path,
                    f"{run_dir.name}/{path.relative_to(run_dir).as_posix()}",
                )
    return target, sha256(target)


def resolve_under_root(release_root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else release_root / path
    return candidate.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rerun only the frozen n_phi=2250 WAXS detector case and create an "
            "amended ACFO NCS v14 return package."
        )
    )
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--original-run-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    release_root = args.release_root.resolve()
    original = resolve_under_root(release_root, args.original_run_dir)
    python = (
        args.python.resolve()
        if args.python is not None
        else release_root / ".venv/Scripts/python.exe"
    )
    target = (
        resolve_under_root(release_root, args.output_dir)
        if args.output_dir is not None
        else original.parent / f"{original.name}_waxs_detector_nphi2250_amended"
    )
    command = build_detector_command(python, release_root, target / RESULT_JSON)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "release_root": str(release_root),
                    "original_run_dir": str(original),
                    "output_dir": str(target),
                    "command": command,
                    "timeout_s": TIMEOUT_S,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    required = [
        release_root / "MANIFEST.json",
        release_root / "scripts/benchmark_protein_nanocrystal_finufft_fair.py",
        release_root / "scripts/validate_external_acfo_ncs_v14.py",
        python,
        original / "run_receipt.json",
        original / "validation.json",
        original / "steps.json",
        original / RESULT_JSON,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"missing required files: {missing}")
    if target.exists():
        raise SystemExit(f"output directory already exists: {target}")

    shutil.copytree(original, target)
    originals = {
        "run_receipt": preserve_original(
            target,
            "run_receipt.json",
            "run_receipt_original_before_nphi2250.json",
        ),
        "validation": preserve_original(
            target,
            "validation.json",
            "validation_original_before_nphi2250.json",
        ),
        "steps": preserve_original(
            target,
            "steps.json",
            "steps_original_before_nphi2250.json",
        ),
        "detector_json": preserve_original(
            target,
            RESULT_JSON,
            "waxs_detector_nq512_abba_original_nphi2160.json",
        ),
    }
    original_md = target / RESULT_MD
    if original_md.is_file():
        originals["detector_md"] = preserve_original(
            target,
            RESULT_MD,
            "waxs_detector_nq512_abba_original_nphi2160.md",
        )

    log = target / "logs/waxs_detector_nq512_abba_nphi2250_amendment.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    start = time.perf_counter()
    env = os.environ.copy()
    env.setdefault("CUPY_CACHE_DIR", str(release_root / ".cupy_cache"))
    with log.open("w", encoding="utf-8", errors="replace") as handle:
        handle.write(f"started_at_utc={started}\n")
        handle.write("command=" + json.dumps(command, ensure_ascii=False) + "\n\n")
        handle.flush()
        try:
            completed = subprocess.run(
                command,
                cwd=release_root,
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=TIMEOUT_S,
                check=False,
                env=env,
            )
            returncode = int(completed.returncode)
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            returncode = 124
            timed_out = True
            handle.write(f"\nTIMEOUT after {TIMEOUT_S} s: {exc}\n")
    duration = time.perf_counter() - start

    output_files = [target / RESULT_JSON]
    if (target / RESULT_MD).is_file():
        output_files.append(target / RESULT_MD)
    outputs = {
        str(path): {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in output_files
        if path.is_file()
    }
    replacement = {
        "label": STEP_LABEL,
        "command": command,
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "duration_s": duration,
        "timeout_s": TIMEOUT_S,
        "timed_out": timed_out,
        "returncode": returncode,
        "log": str(log),
        "outputs": outputs,
        "passed": returncode == 0 and (target / RESULT_JSON).is_file(),
        "supplemental_rerun": True,
        "frozen_nphi_min": 2250,
    }
    steps_payload = load_json(target / "steps_original_before_nphi2250.json")
    amended_steps, original_step = amend_step_records(steps_payload, replacement)
    amended_steps["run_dir"] = str(target)
    write_json(target / "steps.json", amended_steps)

    amendment = {
        "schema": "acfo-ncs-v14-waxs-detector-nphi2250-amendment-v1",
        "generated_at_utc": utc_now(),
        "original_run_dir": str(original),
        "amended_run_dir": str(target),
        "reason": (
            "The original runner requested nphi_min=1024 while the frozen validator "
            "required realized n_phi=2250; the fresh environment realized n_phi=2160."
        ),
        "changed_condition_only": {
            "parameter": "nphi_min",
            "original_requested": 1024,
            "original_realized": load_json(originals["detector_json"]).get("n_phi"),
            "amended_requested": 2250,
        },
        "original_step": original_step,
        "replacement_step": replacement,
        "original_artifacts": {
            name: artifact(path) for name, path in originals.items()
        },
        "source_artifacts": {
            "release_manifest": artifact(release_root / "MANIFEST.json"),
            "benchmark_script": artifact(
                release_root / "scripts/benchmark_protein_nanocrystal_finufft_fair.py"
            ),
            "validator": artifact(
                release_root / "scripts/validate_external_acfo_ncs_v14.py"
            ),
            "supplement_runner": artifact(Path(__file__).resolve()),
        },
    }
    if returncode != 0:
        amendment["passed"] = False
        write_json(target / "amendment_receipt.json", amendment)
        return_zip, return_hash = create_return_zip(target)
        print(f"return_package={return_zip}")
        print(f"return_package_sha256={return_hash}")
        raise SystemExit(returncode)

    detector = load_json(target / RESULT_JSON)
    amendment["amended_realized_n_phi"] = detector.get("n_phi")
    amendment["amended_complex_l2"] = detector.get("complex_l2_acfo_vs_finufft")

    sys.path.insert(0, str(release_root / "scripts"))
    from validate_external_acfo_ncs_v14 import evaluate_run

    validation = evaluate_run(target, mode="full", allow_reference_machine=False)
    write_json(target / "validation.json", validation)
    records = amended_steps.get("steps", [])
    receipt = {
        "schema": "external-acfo-ncs-v14-amended-run-receipt-v1",
        "generated_at_utc": utc_now(),
        "mode": "full",
        "run_dir": str(target),
        "original_run_dir": str(original),
        "amendment": "waxs_detector_nphi2250_only",
        "first_failed_step": None,
        "step_count_completed": len(records),
        "step_count_planned": 13,
        "steps_passed": len(records) == 13 and all(row.get("passed") for row in records),
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
    write_json(target / "run_receipt.json", receipt)
    amendment["validation"] = receipt["validation"]
    amendment["passed"] = bool(validation.get("execution_pass"))
    write_json(target / "amendment_receipt.json", amendment)
    return_zip, return_hash = create_return_zip(target)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    print(f"return_package={return_zip}")
    print(f"return_package_sha256={return_hash}")
    if not validation.get("execution_pass", False):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
