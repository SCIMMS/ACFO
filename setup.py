from __future__ import annotations

import os
import shutil
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


def _find_msvc_tool(tool: str) -> str | None:
    candidates: list[Path] = []
    vctools = os.environ.get("VCToolsInstallDir")
    if vctools:
        candidates.extend(Path(vctools).glob(f"bin/Hostx64/x64/{tool}"))
        candidates.extend(Path(vctools).glob(f"bin/HostX64/x64/{tool}"))

    roots = [
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
        os.environ.get("ProgramFiles", r"C:\Program Files"),
    ]
    for root in roots:
        base = Path(root) / "Microsoft Visual Studio"
        candidates.extend(base.glob(f"**/VC/Tools/MSVC/*/bin/Hostx64/x64/{tool}"))
        candidates.extend(base.glob(f"**/VC/Tools/MSVC/*/bin/HostX64/x64/{tool}"))

    existing = sorted(candidate for candidate in candidates if candidate.exists())
    if existing:
        return str(existing[-1])

    found = shutil.which(tool)
    if found is None:
        return None

    # Git for Windows also ships a Unix-style link.exe. It accepts completely
    # different flags from MSVC link.exe and can shadow the Visual Studio linker
    # on GitHub Actions runners.
    found_path = Path(found)
    normalized_parts = {part.lower() for part in found_path.parts}
    if tool.lower() == "link.exe" and {"git", "usr", "bin"}.issubset(
        normalized_parts
    ):
        return None
    return found


def _find_windows_sdk_tool(tool: str) -> str | None:
    found = shutil.which(tool)
    if found is not None:
        return found

    candidates: list[Path] = []
    windows_sdk = os.environ.get("WindowsSdkDir")
    if windows_sdk:
        candidates.extend(Path(windows_sdk).glob(f"bin/*/x64/{tool}"))

    sdk_root = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
    candidates.extend((sdk_root / "Windows Kits" / "10" / "bin").glob(f"*/x64/{tool}"))

    existing = sorted(candidate for candidate in candidates if candidate.exists())
    return str(existing[-1]) if existing else None


def _prepend_tool_path(compiler, tool: str | None) -> None:
    if tool is None:
        return
    tool_dir = str(Path(tool).parent)
    current_paths = getattr(compiler, "_paths", "")
    if tool_dir not in current_paths.split(os.pathsep):
        compiler._paths = tool_dir + os.pathsep + current_paths


class BuildExt(build_ext):
    c_opts = {
        "msvc": ["/O2", "/std:c++17", "/utf-8"],
        "unix": ["-O3", "-std=c++17"],
    }

    def build_extensions(self) -> None:
        import pybind11

        compiler_type = self.compiler.compiler_type
        opts = list(self.c_opts.get(compiler_type, []))
        opt_mode = os.environ.get("WAXS_CPP_OPT", "").strip().lower()
        valid_opt_modes = {"", "avx2", "native", "fast"}
        if opt_mode not in valid_opt_modes:
            raise ValueError(
                "WAXS_CPP_OPT must be one of: avx2, native, fast"
            )
        if compiler_type == "msvc":
            if not getattr(self.compiler, "initialized", False):
                self.compiler.initialize()
            cl = _find_msvc_tool("cl.exe")
            link = _find_msvc_tool("link.exe")
            lib = _find_msvc_tool("lib.exe")
            rc = _find_windows_sdk_tool("rc.exe")
            mc = _find_windows_sdk_tool("mc.exe")
            mt = _find_windows_sdk_tool("mt.exe")
            if cl is not None:
                self.compiler.cc = cl
            if link is not None:
                self.compiler.linker = link
            if lib is not None:
                self.compiler.lib = lib
            if rc is not None:
                self.compiler.rc = rc
            if mc is not None:
                self.compiler.mc = mc
            if mt is not None:
                self.compiler.mt = mt
            for tool in (cl, link, lib, rc, mc, mt):
                _prepend_tool_path(self.compiler, tool)
            if opt_mode in {"avx2", "native", "fast"}:
                opts.append("/arch:AVX2")
            if opt_mode == "fast":
                opts.append("/fp:fast")
        if compiler_type == "unix":
            opts.extend(["-fvisibility=hidden"])
            if opt_mode in {"native", "fast"}:
                opts.append("-march=native")
            if opt_mode == "fast":
                opts.append("-ffast-math")

        for ext in self.extensions:
            ext.include_dirs.append(pybind11.get_include())
            ext.extra_compile_args = opts
        super().build_extensions()


all_ext_modules = [
    Extension(
        "waxs_cake._cpp_histogram",
        ["src/waxs_cake/_cpp_histogram.cpp"],
        language="c++",
    ),
    Extension(
        "waxs_cake._cpp_solvers",
        ["src/waxs_cake/_cpp_solvers.cpp"],
        language="c++",
    ),
    Extension(
        "waxs_cake._cpp_high_na",
        ["src/waxs_cake/_cpp_high_na.cpp"],
        language="c++",
    ),
    Extension(
        "waxs_cake._cpp_odt",
        ["src/waxs_cake/_cpp_odt.cpp"],
        language="c++",
    ),
]


def _select_ext_modules() -> list[Extension]:
    requested_raw = os.environ.get("WAXS_CPP_EXTENSIONS", "").strip()
    if not requested_raw:
        return all_ext_modules

    requested = {
        item.strip().lower()
        for item in requested_raw.replace(";", ",").split(",")
        if item.strip()
    }
    if not requested:
        return all_ext_modules

    aliases = {
        "histogram": "waxs_cake._cpp_histogram",
        "hist": "waxs_cake._cpp_histogram",
        "_cpp_histogram": "waxs_cake._cpp_histogram",
        "solvers": "waxs_cake._cpp_solvers",
        "solver": "waxs_cake._cpp_solvers",
        "_cpp_solvers": "waxs_cake._cpp_solvers",
        "high_na": "waxs_cake._cpp_high_na",
        "high-na": "waxs_cake._cpp_high_na",
        "_cpp_high_na": "waxs_cake._cpp_high_na",
        "odt": "waxs_cake._cpp_odt",
        "_cpp_odt": "waxs_cake._cpp_odt",
    }
    by_name = {ext.name.lower(): ext for ext in all_ext_modules}
    selected_names: set[str] = set()
    unknown: set[str] = set()
    for item in requested:
        canonical = aliases.get(item, item)
        if canonical.lower() in by_name:
            selected_names.add(canonical.lower())
        else:
            unknown.add(item)
    if unknown:
        valid = sorted(set(aliases) | set(by_name))
        raise ValueError(
            "unknown WAXS_CPP_EXTENSIONS entries "
            f"{sorted(unknown)}; valid entries include {valid}"
        )
    return [ext for ext in all_ext_modules if ext.name.lower() in selected_names]


setup(
    ext_modules=_select_ext_modules(),
    cmdclass={"build_ext": BuildExt},
)
