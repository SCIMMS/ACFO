from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark_results"
OUTPUT = RESULTS / "acfo_ncs_fresh_dependency_rerun.json"


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    suite_path = RESULTS / "acfo_ncs_reduced_release_suite.json"
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    pytest_step = next(row for row in suite["commands"] if row["label"] == "full_pytest")

    import torch

    extensions = sorted((ROOT / "src/waxs_cake").glob("_cpp_*.pyd"))
    result = {
        "schema": "acfo-ncs-fresh-dependency-rerun-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python_executable": sys.executable,
            "python": sys.version,
            "platform": platform.platform(),
            "venv": str(Path(sys.prefix).relative_to(ROOT)),
            "packages": {
                name: version(name)
                for name in (
                    "numpy",
                    "scipy",
                    "finufft",
                    "numba",
                    "pybind11",
                    "torch",
                    "pytest",
                    "reportlab",
                    "PyMuPDF",
                    "matplotlib",
                )
            },
            "torch_cuda_version": torch.version.cuda,
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "rebuilt_cpp_extensions": [path.name for path in extensions],
        "suite": {
            "duration_s": suite["duration_s"],
            "passed": suite["passed"],
            "pytest_stdout_tail": pytest_step["stdout_tail"],
            "pdf_sha256": suite["pdf"]["sha256"],
            "suite_sha256": digest(suite_path),
        },
        "gates": {
            "isolated_venv": ".venv_fresh_20260713" in sys.executable,
            "four_cpp_extensions_rebuilt": len(extensions) == 4,
            "cuda_torch_available": bool(torch.cuda.is_available()),
            "full_pytest_passed": pytest_step["passed"],
            "reduced_suite_passed": suite["passed"],
            "pdf_rebuilt": suite["pdf"]["status"] == "passed",
        },
        "scope": (
            "Fresh dependency install and forced C++ rebuild in an isolated venv, followed by the local reduced GPU suite."
        ),
        "limitations": [
            "This used the same physical Windows machine and RTX 2070 SUPER; it is not an independent-machine rerun.",
            "The optional cuFINUFFT/CuPy baseline extra was not installed in the fresh reduced-suite venv; saved production baseline evidence was schema-checked.",
            "The 216k WAXS and 256-cubed ODT production workloads were not rerun by the reduced suite.",
        ],
    }
    result["passed"] = all(result["gates"].values())
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
