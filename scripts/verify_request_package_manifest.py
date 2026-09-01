from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative_path(value: str) -> bool:
    if not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return bool(
        not path.is_absolute()
        and all(part not in ("", ".", "..") for part in path.parts)
    )


def audit_manifest(
    root: Path,
    manifest_path: Path,
    *,
    require_exact: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = (
        manifest_path.resolve()
        if manifest_path.is_absolute()
        else (root / manifest_path).resolve()
    )
    if manifest_path.parent != root:
        raise RuntimeError("manifest must be a direct child of the package root")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared = payload.get("files")
    if not isinstance(declared, dict):
        raise RuntimeError("manifest files must be an object")
    failures: list[str] = []
    for relative, expected in declared.items():
        if not isinstance(relative, str) or not _safe_relative_path(relative):
            failures.append(f"unsafe declared path: {relative!r}")
            continue
        if not isinstance(expected, dict):
            failures.append(f"invalid record: {relative}")
            continue
        path = root.joinpath(*PurePosixPath(relative).parts)
        try:
            resolved = path.resolve(strict=False)
            resolved.relative_to(root)
        except (OSError, ValueError):
            failures.append(f"path escaped package root: {relative}")
            continue
        if not path.is_file() or path.is_symlink():
            failures.append(f"missing or non-regular: {relative}")
            continue
        if path.stat().st_size != expected.get("bytes"):
            failures.append(f"byte mismatch: {relative}")
        if sha256(path) != expected.get("sha256"):
            failures.append(f"sha256 mismatch: {relative}")
    manifest_relative = manifest_path.relative_to(root).as_posix()
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    unsafe_symlinks = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_symlink()
    )
    undeclared_files = sorted(actual_files - set(declared) - {manifest_relative})
    missing_declared_files = sorted(set(declared) - actual_files)
    if require_exact:
        failures.extend(f"undeclared: {relative}" for relative in undeclared_files)
        failures.extend(
            f"declared but absent: {relative}" for relative in missing_declared_files
        )
        failures.extend(f"symlink not permitted: {relative}" for relative in unsafe_symlinks)
    return {
        "schema": "acfo-request-package-manifest-audit-v2",
        "file_count": len(declared),
        "actual_file_count_excluding_manifest": len(
            actual_files - {manifest_relative}
        ),
        "require_exact": require_exact,
        "undeclared_files": undeclared_files,
        "missing_declared_files": missing_declared_files,
        "unsafe_symlinks": unsafe_symlinks,
        "failures": failures,
        "passed": not failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path, default=Path("PACKAGE_MANIFEST.json"))
    parser.add_argument("--require-exact", action="store_true")
    args = parser.parse_args()
    result = audit_manifest(
        args.root,
        args.manifest,
        require_exact=args.require_exact,
    )
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
