from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/ACFO_NCS_v14_WAXS_detector_nphi2250_supplement_2026-07-14.zip"
PREFIX = "ACFO_NCS_v14_waxs_detector_nphi2250_supplement"
FILES = {
    "run_waxs_detector_only.py": ROOT
    / "scripts/run_external_acfo_ncs_v14_waxs_detector_only.py",
    "run_waxs_detector_only.ps1": ROOT
    / "scripts/run_external_acfo_ncs_v14_waxs_detector_only.ps1",
    "README_KO.md": ROOT
    / "docs/acfo_ncs_v14_waxs_detector_nphi2250_supplement_ko.md",
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def build() -> dict[str, object]:
    missing = [str(path) for path in FILES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    payloads = {name: path.read_bytes() for name, path in FILES.items()}
    manifest = {
        "schema": "acfo-ncs-v14-waxs-detector-nphi2250-supplement-manifest-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Rerun only the frozen WAXS detector n_phi=2250 case and amend an "
            "otherwise completed external v14 return package."
        ),
        "files": [
            {
                "path": name,
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
            }
            for name, payload in sorted(payloads.items())
        ],
    }
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        OUTPUT,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name, payload in sorted(payloads.items()):
            archive.writestr(f"{PREFIX}/{name}", payload)
        archive.writestr(f"{PREFIX}/MANIFEST.json", manifest_bytes)
    receipt = {
        "output": str(OUTPUT),
        "bytes": OUTPUT.stat().st_size,
        "sha256": sha256(OUTPUT),
        "file_count": len(payloads),
    }
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return receipt


if __name__ == "__main__":
    build()
