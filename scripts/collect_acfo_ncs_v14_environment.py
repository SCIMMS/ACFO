from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def total_physical_memory_bytes() -> int | None:
    if os.name == "nt":
        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
    if hasattr(os, "sysconf"):
        try:
            return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
        except (OSError, ValueError):
            return None
    return None


def nvidia_smi_receipt() -> dict[str, Any]:
    fields = (
        "name,uuid,memory.total,driver_version,pstate,temperature.gpu,"
        "clocks.current.graphics,clocks.current.memory"
    )
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
            encoding="utf-8",
            errors="replace",
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": type(exc).__name__}
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return {
        "available": completed.returncode == 0 and bool(rows),
        "returncode": completed.returncode,
        "query_fields": fields.split(","),
        "rows": [
            [part.strip() for part in row.split(",")]
            for row in rows
        ],
        "stderr": completed.stderr.strip(),
    }


def torch_receipt() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on external environment
        return {"imported": False, "error": f"{type(exc).__name__}: {exc}"}
    cuda_available = bool(torch.cuda.is_available())
    devices = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_bytes": int(props.total_memory),
                    "major": int(props.major),
                    "minor": int(props.minor),
                    "multi_processor_count": int(props.multi_processor_count),
                }
            )
    return {
        "imported": True,
        "version": getattr(torch, "__version__", None),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": cuda_available,
        "device_count": len(devices),
        "devices": devices,
    }


def build_receipt() -> dict[str, Any]:
    machine = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "processor_identifier": os.environ.get("PROCESSOR_IDENTIFIER"),
        "cpu_count": os.cpu_count(),
        "total_physical_memory_bytes": total_physical_memory_bytes(),
    }
    fingerprint_source = json.dumps(
        machine, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    manifest = ROOT / "MANIFEST.json"
    return {
        "schema": "acfo-ncs-v14-machine-environment-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "machine": machine,
        "machine_fingerprint_sha256": hashlib.sha256(fingerprint_source).hexdigest(),
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "packages": {
                name: package_version(name)
                for name in (
                    "numpy",
                    "scipy",
                    "finufft",
                    "torch",
                    "cupy-cuda12x",
                    "cufinufft",
                    "nvidia-cuda-runtime-cu12",
                    "nvidia-cuda-nvrtc-cu12",
                    "pybind11",
                    "pytest",
                )
            },
        },
        "torch": torch_receipt(),
        "nvidia_smi": nvidia_smi_receipt(),
        "release": {
            "manifest_path": "MANIFEST.json" if manifest.is_file() else None,
            "manifest_sha256": sha256(manifest),
        },
        "source_sha256": {
            path: sha256(ROOT / path)
            for path in (
                "scripts/run_external_acfo_ncs_validation_v14.py",
                "scripts/validate_external_acfo_ncs_v14.py",
                "scripts/benchmark_odt_banded_cartesian_final_packed.py",
                "scripts/validate_odt_cufinufft_matched_error.py",
                "src/waxs_cake/_cpp_solvers.cpp",
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect the external-machine and source receipt for ACFO NCS v14."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_receipt()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
