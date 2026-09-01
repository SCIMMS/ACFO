from __future__ import annotations

import json
import platform
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import build_acfo_ncs_release_candidate as v13  # noqa: E402


OUTPUT = ROOT / "docs/ACFO_NCS_validation_release_candidate_2026-07-14_v14.zip"
RECEIPT = ROOT / "benchmark_results/acfo_ncs_release_candidate_v14_manifest.json"
PREFIX = "ACFO_NCS_validation_release_candidate_v14"
LEGACY_ZIP = ROOT / "docs/ACFO_NCS_validation_release_candidate_2026-07-13_v13.zip"
LEGACY_PREFIX = "ACFO_NCS_validation_release_candidate"


EXTRA_FILES = [
    "benchmark_results/acfo_ncs_release_candidate_manifest.json",
    "benchmark_results/odt_banded_cartesian_final_packed_probe.json",
    "benchmark_results/odt_banded_cartesian_final_packed_full_timing.json",
    "benchmark_results/odt_cufinufft_matched_error_direct_subset.json",
    "benchmark_results/odt_cufinufft_matched_error_direct_subset_c128.json",
    "benchmark_results/odt_h28_rank16_adaptive_l1e-6_vs_cufinufft_abba30_final.json",
    "benchmark_results/odt_cufinufft_matched_c128_full_pair5.json",
    "benchmark_results/odt_cufinufft_c128_full_plan_diagnostic.json",
    "benchmark_results/odt_banded_cartesian_temporal_warm_start.json",
    "benchmark_results/waxs_protein_exact_beta_followup_decision.json",
    "benchmark_results/acfo_claim_artifact_manifest.json",
    "benchmark_results/aidt_10hz_full700_opt_repeat.json",
    "benchmark_results/linbo3_one_way_shg_cascade_holdout_angle25_r24.json",
    "docs/acfo_claim_artifact_manifest_ko.md",
    "docs/acfo_ncs_external_rerun_v14_ko.md",
    "reports/acfo_ncs_validation_closure_20260714/ACFO_NCS_validation_closure_report_ko.pdf",
    "reports/acfo_ncs_validation_closure_20260714/ACFO_NCS_validation_closure_report_ko.md",
    "reports/acfo_ncs_validation_closure_20260714/report_source_inventory.json",
    "reports/acfo_ncs_validation_closure_20260714/verification_receipt.json",
]


RELEASE_README = r"""# ACFO NCS validation release candidate v14

## Scope

This archive is the clean external-machine handoff for the frozen WAXS and latest integrated ODT claims in the 2026-07-14 closure report. It contains a one-command installer/runner, an embedded file-hash manifest, frozen structures, code, tests, local reference artifacts, and explicit gate definitions. Packaged historical JSON is context only: every external PASS is computed from new timestamped outputs under `benchmark_results/external_acfo_ncs_v14_*`.

## One command

Run from the extracted release root on Windows with Python launcher, MSVC Build Tools, an NVIDIA CUDA-compatible driver, and at least 8 GiB VRAM:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_and_run_external_acfo_ncs_v14.ps1 -Mode full
```

The full run installs dependencies, verifies every manifest hash, forcibly rebuilds the C++ extensions, runs all tests, reruns WAXS 1M prepared and detector-aware protocols, runs the integrated ODT probe/scale/direct/matched/temporal protocols, applies frozen gates, and creates one return ZIP with its SHA-256.

On Windows, the installer prefers Python 3.12, then 3.11, with 3.13 as a fallback. It explicitly installs the official PyTorch 2.12.1 CUDA 12.6 wheel and CuPy 14.1.1 with the `[ctk]` CUDA component wheels. It stops before validation unless both `torch.cuda.is_available()` and a CuPy JIT elementwise probe pass. The NVIDIA driver may report a newer CUDA compatibility level; the wheels carry the CUDA runtime and headers used by this package.

Use quick mode before the long run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_and_run_external_acfo_ncs_v14.ps1 -Mode quick
```

Quick mode is package QA only and is never counted as publication-scale independent replication. Resume a full run by supplying the printed run directory with `-Resume -RunDir <path>`.

After changing or repairing packages in an existing environment, add `-RefreshEnvironment` to the resume command. This reinstalls the frozen GPU dependencies and refreshes the environment and `pip freeze` receipts while retaining hash-verified completed numerical steps.

## Independence and memory

`publication_replication_pass` requires a machine fingerprint different from the local RTX 2070 SUPER reference. `-AllowReferenceMachine` exists only for local runner QA and cannot make this gate true. On an 8 GiB GPU the matched complex128 cuFINUFFT plan is timed in a separate process from the ACFO plan; preserve that caveat. A 24 GiB or larger GPU is recommended for an additional co-resident AB/BA experiment.

## Evidence

The runner returns environment and package receipts, build/test logs, exact commands, raw per-repeat timing files, accuracy JSON, validation groups, source/output hashes and a partial evidence ZIP even when a gate fails. Do not edit, rename selectively, or rerun only failed rows without returning the original receipt.

See `docs/acfo_ncs_external_rerun_v14_ko.md` and `VALIDATION_CONTRACT.json` for the complete protocol.
"""


VALIDATION_CONTRACT = {
    "schema": "acfo-ncs-v14-external-validation-contract-v1",
    "quick": {
        "manifest_hashes_match": True,
        "cuda_required": True,
        "cupy_and_cufinufft_required": True,
        "forced_cpp_build_required": True,
        "full_pytest_required": True,
        "odt_operator_rel_l2_max": 2e-6,
        "odt_pixel_mode_rel_l2_max": 2e-6,
        "odt_remap_reconstruction_rel_l2_max": 2e-3,
    },
    "full": {
        "waxs_prepared": {
            "warmup_pairs": 10,
            "measured_pairs": 30,
            "balanced_ab_ba": [15, 15],
            "complex_l2_max": 2e-6,
            "legacy_l2_max": 1e-12,
            "paired_speedup_median_min": 3.0,
            "paired_speedup_p05_min": 3.0,
            "ab_ba_median_relative_gap_max": 0.10,
        },
        "waxs_detector": {
            "nq": 512,
            "nphi": 2250,
            "warmups": 10,
            "repeats": 30,
            "complex_l2_max": 1e-6,
            "intensity_row_median_max": 1e-3,
            "intensity_row_p99_max": 5e-3,
            "speedup_min_exclusive": 1.0,
            "whole_process_memory_ratio_min": 4.0,
        },
        "odt": {
            "direct_dot_error_max": 1e-12,
            "acfo_worst_direct_l2_max": 2e-6,
            "same_dtype_warmups": 5,
            "same_dtype_repeats": 30,
            "same_dtype_speedup_min": 3.0,
            "matched_dtype": "complex128",
            "matched_eps": 1e-7,
            "matched_pair_warmups": 2,
            "matched_pair_repeats": 5,
            "matched_speedup_min": 3.0,
            "temporal_saved_gates_must_all_pass": True,
        },
    },
    "claim_boundary": [
        "Quick mode is not publication replication.",
        "Independent replication requires a different machine fingerprint.",
        "GPU-resident hot paths exclude acquisition, host transfer and hologram demodulation.",
        "Matched complex128 timing is separate-process on memory-limited GPUs.",
    ],
}


def selected_files() -> list[Path]:
    claim_manifest = json.loads(
        (ROOT / "benchmark_results/acfo_claim_artifact_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    claim_paths = {
        artifact["path"]
        for claim in claim_manifest.get("claims", [])
        for artifact in claim.get("artifacts", [])
    }
    claim_paths.update(
        artifact["path"]
        for artifact in claim_manifest.get("gate_closure_artifacts", [])
    )
    missing_claim_paths = [
        path for path in sorted(claim_paths) if not (ROOT / path).is_file()
    ]
    if missing_claim_paths:
        raise RuntimeError(
            f"claim manifest references missing files: {missing_claim_paths}"
        )
    candidates = [
        *v13.selected_files(),
        *(ROOT / item for item in EXTRA_FILES),
        *(ROOT / item for item in sorted(claim_paths)),
    ]
    unique = {path.resolve(): path for path in candidates if path.is_file()}
    return sorted(unique.values(), key=lambda path: path.relative_to(ROOT).as_posix())


def generated_entry(path: str, payload: bytes) -> dict[str, object]:
    import hashlib

    return {
        "path": path,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "generated": True,
    }


def legacy_recovery_payloads() -> dict[str, bytes]:
    missing = [
        item
        for item in (
            v13.ROOT_FILES
            + v13.STRUCTURES
            + v13.WATER_SOURCES
            + v13.RESULT_FILES
            + v13.DOC_FILES
        )
        if not (ROOT / item).is_file()
    ]
    if not missing:
        return {}
    if not LEGACY_ZIP.is_file():
        raise RuntimeError(
            f"v13 source archive is required to recover missing files: {missing}"
        )
    recovered: dict[str, bytes] = {}
    with zipfile.ZipFile(LEGACY_ZIP) as archive:
        for item in missing:
            member = f"{LEGACY_PREFIX}/{item}"
            try:
                recovered[item] = archive.read(member)
            except KeyError as exc:
                raise RuntimeError(
                    f"missing required file locally and in v13 archive: {item}"
                ) from exc
    return recovered


def main() -> None:
    missing = [item for item in EXTRA_FILES if not (ROOT / item).is_file()]
    if missing:
        raise RuntimeError(f"missing required v14 files: {missing}")
    files = selected_files()
    recovered = legacy_recovery_payloads()
    entries: list[dict[str, object]] = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": v13.sha256(path),
        }
        for path in files
    ]
    readme_bytes = RELEASE_README.encode("utf-8")
    contract_bytes = (
        json.dumps(VALIDATION_CONTRACT, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    entries.extend(
        [
            generated_entry("README_RELEASE.md", readme_bytes),
            generated_entry("VALIDATION_CONTRACT.json", contract_bytes),
        ]
    )
    entries.extend(
        {
            **generated_entry(path, payload),
            "generated": False,
            "recovered_from": LEGACY_ZIP.relative_to(ROOT).as_posix(),
        }
        for path, payload in recovered.items()
    )
    local_env = json.loads(
        (ROOT / "benchmark_results/local_prepared_waxs_machine_environment.json").read_text(
            encoding="utf-8"
        )
    )
    def git_value(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return completed.stdout.strip()

    status_lines = [
        line
        for line in git_value("status", "--porcelain=v1").splitlines()
        if line
    ]
    manifest = {
        "schema": "acfo-ncs-validation-release-candidate-v2",
        "release": "v14",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_prefix": PREFIX,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "source_revision": {
            "commit": git_value("rev-parse", "HEAD"),
            "branch": git_value("branch", "--show-current"),
            "tracked_changes": sum(
                not line.startswith("??") for line in status_lines
            ),
            "untracked_entries": sum(line.startswith("??") for line in status_lines),
            "total_status_entries": len(status_lines),
            "clean": not status_lines,
        },
        "reference_machine_fingerprint_sha256": local_env.get(
            "machine_fingerprint_sha256"
        ),
        "file_count": len(entries),
        "payload_bytes": sum(int(item["bytes"]) for item in entries),
        "files": entries,
        "one_command": (
            "powershell -ExecutionPolicy Bypass -File "
            "scripts\\install_and_run_external_acfo_ncs_v14.ps1 -Mode full"
        ),
        "limitations": VALIDATION_CONTRACT["claim_boundary"],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr(f"{PREFIX}/README_RELEASE.md", readme_bytes)
        archive.writestr(f"{PREFIX}/VALIDATION_CONTRACT.json", contract_bytes)
        archive.writestr(
            f"{PREFIX}/MANIFEST.json",
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        )
        for path, payload in recovered.items():
            archive.writestr(f"{PREFIX}/{path}", payload)
        for path in files:
            archive.write(path, f"{PREFIX}/{path.relative_to(ROOT).as_posix()}")
    receipt = {
        **manifest,
        "zip": OUTPUT.relative_to(ROOT).as_posix(),
        "zip_bytes": OUTPUT.stat().st_size,
        "zip_sha256": v13.sha256(OUTPUT),
    }
    RECEIPT.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in receipt.items() if key != "files"},
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"wrote {OUTPUT} and {RECEIPT}")


if __name__ == "__main__":
    main()
