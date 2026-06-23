# Development Guide

This repository has three layers:

- `src/waxs_cake/`: importable solver and operator code.
- `scripts/`: benchmark, demo, report, and conversion entry points.
- `benchmark_results/`: curated markdown summaries only. Generated arrays,
  downloaded data, PDFs, images, and raw probes stay out of Git.

The package is named `curved-ewald-operators`, but the current import package is
still `waxs_cake`. That keeps the validated WAXS code path stable while the
repository expands to high-NA and ODT/aIDT operator benchmarks.

## Minimal Development Setup

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
```

The editable install builds the pybind11 C++ extensions when a compiler is
available. Tests that require optional native modules skip themselves if those
modules are not built.

For the broader local benchmark environment, install the historical dependency
file:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

That file includes optional benchmark/report dependencies such as FINUFFT,
numba, optics packages, and report-generation tools. It is intentionally broader
than the minimal CI environment.

## Tests

Run the committed unit/regression tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The CI workflow runs the same command after `pip install -e ".[test]"` on
Python 3.11. It is a repository health check, not a full benchmark suite.

## Native Extension Controls

The C++ extension build can be narrowed with `WAXS_CPP_EXTENSIONS`:

```powershell
$env:WAXS_CPP_EXTENSIONS = "histogram,solvers"
.\.venv\Scripts\python.exe setup.py build_ext --inplace
```

Valid aliases include `histogram`, `solvers`, `high_na`, and `odt`.

Optimization flags are controlled with `WAXS_CPP_OPT`:

```powershell
$env:WAXS_CPP_OPT = "avx2"
.\.venv\Scripts\python.exe setup.py build_ext --inplace
```

Valid values are empty/default, `avx2`, `native`, and `fast`. Use default or
`avx2` for portable benchmark comparisons; use `native` or `fast` only when the
result is explicitly machine-specific.

## Benchmark Artifacts

Large generated files are intentionally ignored:

- raw public-data probes
- `.npz`, `.mat`, `.h5`, `.tar`, and other bulky data products
- generated PDFs, PNGs, and benchmark plots
- local build outputs and virtual environments

Keep benchmark conclusions in curated markdown summaries under
`benchmark_results/` or `docs/`, and keep the commands that produced them in
scripts or in the summary text.
