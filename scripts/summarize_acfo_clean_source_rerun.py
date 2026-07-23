from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "benchmark_results/acfo_ncs_clean_source_rerun.json"
ZIP = ROOT / "docs/ACFO_NCS_validation_release_candidate_2026-07-13_v13.zip"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    candidates = sorted((ROOT / "runs").glob("release_clean_source_*/ACFO_NCS_validation_release_candidate"))
    if not candidates:
        raise RuntimeError("no clean-source rerun directory found")
    repo = candidates[-1]
    build_log_path = repo / "clean_source_build_ext.log"
    if not build_log_path.exists():
        raise RuntimeError("clean-source forced-build log is missing")
    build_log = build_log_path.read_text(encoding="utf-16", errors="replace")
    required_build_markers = [
        "building 'waxs_cake._cpp_histogram' extension",
        "building 'waxs_cake._cpp_solvers' extension",
        "building 'waxs_cake._cpp_high_na' extension",
        "building 'waxs_cake._cpp_odt' extension",
    ]
    missing_build_markers = [
        marker for marker in required_build_markers if marker not in build_log
    ]
    if missing_build_markers:
        raise RuntimeError(f"forced-build log is incomplete: {missing_build_markers}")
    suite_path = repo / "benchmark_results/acfo_ncs_reduced_release_suite.json"
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    if not suite.get("passed"):
        raise RuntimeError("clean-source reduced suite did not pass")
    extensions = sorted((repo / "src/waxs_cake").glob("_cpp_*.pyd"))
    if len(extensions) != 4:
        raise RuntimeError(f"expected four rebuilt C++ extensions, found {len(extensions)}")
    pytest_step = next(row for row in suite["commands"] if row["label"] == "full_pytest")
    result = {
        "schema": "acfo-ncs-clean-source-rerun-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "release_zip": ZIP.relative_to(ROOT).as_posix(),
        "release_zip_sha256": digest(ZIP),
        "extracted_repo": repo.relative_to(ROOT).as_posix(),
        "rebuilt_cpp_extensions": [path.name for path in extensions],
        "forced_build_log": {
            "path": build_log_path.relative_to(ROOT).as_posix(),
            "sha256": digest(build_log_path),
            "required_markers_present": True,
        },
        "pytest_stdout_tail": pytest_step["stdout_tail"],
        "reduced_suite_duration_s": suite["duration_s"],
        "reduced_suite_pdf_sha256": suite["pdf"]["sha256"],
        "clean_suite_environment": suite["environment"],
        "gates": {
            "cpp_extensions_rebuilt": True,
            "full_pytest_passed": pytest_step["passed"],
            "reduced_suite_passed": suite["passed"],
            "pdf_rebuilt": suite["pdf"]["status"] == "passed",
        },
        "scope": "Fresh final archive extraction, forced four-extension source rebuild, and reduced-suite rerun using the clean dependency environment created during the immediately preceding archive check.",
        "limitations": [
            "The dependency environment was freshly created during the immediately preceding archive checks, then reused after reinstalling the final archive source; dependencies were not downloaded a second time for the final archive.",
            "Production TIP3P, WAXS, ODT, and PyMeep rows are schema-checked saved evidence unless explicitly listed among reduced-suite commands.",
            "The rerun used the same physical machine and a CPU-only PyTorch build; it is not an independent-machine validation.",
        ],
        "passed": True,
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
