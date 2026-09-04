"""Small, stdlib-only immutable request/return utilities."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def safe_path(root, name):
    if "\\" in name or ":" in name:
        raise ValueError(f"Unsafe path: {name}")
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise ValueError(f"Unsafe path: {name}")
    target = (Path(root)/name).resolve()
    if Path(root).resolve() not in target.parents:
        raise ValueError(f"Path escapes root: {name}")
    return target


def extract(archive, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        seen = set()
        for info in z.infolist():
            safe_path(destination, info.filename)
            if info.filename in seen or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("Duplicate or symlink ZIP entry")
            seen.add(info.filename)
        z.extractall(destination)


def records(root, exclude=(), ignore_top=()):
    return {p.relative_to(root).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size}
            for p in sorted(Path(root).rglob("*")) if p.is_file()
            and p.relative_to(root).as_posix() not in exclude
            and p.relative_to(root).parts[0] not in ignore_top}


def audit(root, manifest="PACKAGE_MANIFEST.json", exact=True, ignore_top=()):
    root = Path(root)
    m = load(root/manifest)
    errors = []
    for name, r in m["files"].items():
        p = safe_path(root, name)
        if not p.is_file() or p.stat().st_size != r["bytes"] or sha(p) != r["sha256"]:
            errors.append(name)
    if exact:
        errors += sorted(set(records(root, ignore_top=ignore_top))-set(m["files"])-{manifest})
    return {"passed": not errors, "errors": errors, "file_count": len(m["files"]), "manifest_sha256": sha(root/manifest)}


def zip_tree(root, archive, prefix=""):
    # 'x' deliberately preserves any existing archive.
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(Path(root).rglob("*")):
            if p.is_file():
                z.write(p, (PurePosixPath(prefix)/p.relative_to(root).as_posix()).as_posix())


def validate_request(root):
    root = Path(root)
    result = audit(root, ignore_top=("_runs", "__pycache__"))
    m, p, c = (load(root/n) for n in ("PACKAGE_MANIFEST.json", "PROTOCOL.json", "CLAIM_FREEZE.json"))
    checks = {
        "protocol_hash": sha(root/"PROTOCOL.json") == m["protocol_sha256"],
        "claim_hash": sha(root/"CLAIM_FREEZE.json") == m["claim_freeze_sha256"],
        "protocol_schema": p["schema"] == "acfo-operator-storyline-protocol-v2",
        "freeze_schema": c["schema"] == "acfo-operator-storyline-claim-freeze-v2",
        "frozen": c["status"] == "frozen_before_external_benchmark",
        "no_selection": c["immutability"]["post_result_arm_selection_allowed"] is False,
        "no_retuning": c["immutability"]["post_result_threshold_change_allowed"] is False,
        "all_systems": p["story_order"] == ["high_na", "waxs", "odt", "cpswf", "composite"],
        "profiles": set(p["profiles"]) == set(c["machine_specific_allowed_arms"]),
    }
    result["checks"] = checks
    result["passed"] &= all(checks.values())
    return result
