from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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
    elif hasattr(os, "sysconf"):
        try:
            return int(os.sysconf("SC_PAGE_SIZE")) * int(
                os.sysconf("SC_PHYS_PAGES")
            )
        except (OSError, ValueError):
            pass
    return None


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_receipt() -> dict:
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
    source_hashes = {
        "benchmark_driver": file_sha256(
            ROOT / "scripts/benchmark_protein_lattice_finufft_512_abba.py"
        ),
        "cpp_solvers": file_sha256(ROOT / "src/waxs_cake/_cpp_solvers.cpp"),
        "exact_harmonic": file_sha256(ROOT / "src/waxs_cake/exact_harmonic.py"),
        "manifest": file_sha256(ROOT / "MANIFEST.json"),
    }
    return {
        "schema": "prepared-waxs-machine-environment-v1",
        "passed": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "machine": machine,
        "machine_fingerprint_sha256": hashlib.sha256(fingerprint_source).hexdigest(),
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "packages": {
                name: package_version(name)
                for name in ("numpy", "scipy", "finufft", "pybind11")
            },
        },
        "source_sha256": source_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect a machine/source fingerprint for independent prepared-WAXS timing."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/local_prepared_waxs_machine_environment.json"),
    )
    args = parser.parse_args()
    result = build_receipt()
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
