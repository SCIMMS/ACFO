"""Frozen five-system orchestration; local_check is never external benchmark evidence."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
import traceback
from acfo_storyline_package_io import audit, extract, load, records, sha, validate_request, write, zip_tree

ROOT = Path(__file__).resolve().parents[1]
CPU_TESTS = ["test_finite_hankel_cpswf.py", "test_radial_cpswf.py", "test_radial_aidt.py", "test_prepared_cpswf_aidt.py"]


def command_snapshot(command):
    try:
        r = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=10)
        return {"returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr}
    except Exception as exc:
        return {"returncode": None, "error": repr(exc)}


def environment(profile):
    packages = {}
    for name in ("numpy", "scipy", "pytest", "finufft", "torch", "cupy-cuda12x", "cufinufft"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu = command_snapshot(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"])
    apps = command_snapshot(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"])
    active = [line for line in apps.get("stdout", "").splitlines() if line.strip() and "no running processes" not in line.lower()]
    return {"profile": profile, "python": sys.version, "platform": platform.platform(), "cpu_count": os.cpu_count(),
        "packages": packages, "gpu": gpu, "gpu_processes": apps, "no_competing_gpu_processes": apps["returncode"] == 0 and not active,
        "cpu_process_snapshot": command_snapshot(["ps", "-eo", "pid,user,pcpu,pmem,comm", "--sort=-pcpu"]) if os.name != "nt" else {"note": "Windows local package QA; no external CPU idle claim"},
        "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None}


def plan(profile, roots, outputs):
    """One explicit stage list, shared by execution and expected-coverage audit."""
    ext, waxs, odt, comp = (roots[k] for k in ("extension", "waxs", "odt", "composite"))
    local = profile == "local_check"
    stages = []
    def add(name, cwd, args, *, expected=None, deps=(), hours=1, kind="numerical", omp="1"):
        stages.append({"id": name, "cwd": str(cwd), "command": [sys.executable, *map(str, args)],
            "expected": str(expected) if expected else None, "depends": list(deps), "timeout": hours*3600,
            "kind": kind, "omp": omp})
    add("high_na_correspondence", ext, ["scripts/validate_high_na_si_correspondence.py", "--backend", "numpy", "--output", "outputs/high_na_correspondence.json"], expected=ext/"outputs/high_na_correspondence.json")
    add("high_na_support", ext, ["scripts/validate_high_na_harmonic_support_risk.py", "--output", "outputs/high_na_support.json"], expected=ext/"outputs/high_na_support.json")
    if not local:
        args = ["scripts/run_waxs_100_state_external_validation.py", "--mode", "full", "--no-resume"]
        if profile == "server59_replication": args.append("--allow-reference-machine-full")
        add("waxs_full", waxs, args, hours=30, kind="legacy", omp="4")
    add("planar_boundary", waxs, [ROOT/"scripts/validate_waxs_planar_dispatch_boundary.py", "--structure",
        waxs/"structures/processed/protein_nanocrystal_lysozyme_1iee_1x1x1_fixed.npz", "--output", outputs/"planar.json",
        "--source-atoms", "64" if local else "512", "--grid-sizes", *( ["16"] if local else ["16", "24", "32"] ),
        "--warmup", "0" if local else "2", "--repeats", "1" if local else "7"], expected=outputs/"planar.json", omp="4")
    if not local:
        args = ["scripts/run_odt_toeplitz_si_external_validation.py", "--mode", "full", "--run-parent", odt/"runs"]
        if profile == "server59_replication": args.append("--allow-non-rtx3090")
        add("odt_cartesian", odt, args, hours=24, kind="legacy", omp="4")
    add("extension_unit_tests", ext, ["-m", "pytest", "-q", "-p", "no:cacheprovider",
        *["tests/"+name for name in CPU_TESTS], "--junitxml=outputs/tests.xml"], kind="unit")
    for key, script, folder, depends in (
        ("common_radial", "acfo_common_radial_cpswf", "acfo_common_radial_cpswf_v1", ()),
        ("physical_modal_normal", "acfo_radial_physical_transfer", "acfo_radial_physical_transfer_v1", ("common_radial_audit",)),
        ("geometry_api", "acfo_geometry_preparation_api", "acfo_geometry_preparation_api_v1", ("physical_modal_normal_audit",)),
    ):
        add(key, ext, [f"scripts/validate_{script}.py"], expected=ext/"reports"/folder/"summary.json", deps=depends)
        add(key+"_audit", ext, [f"scripts/audit_{script}.py"], expected=ext/"reports"/folder/"audit.json", deps=(key,))
    add("support_frontier", ext, ["scripts/validate_acfo_storyline_support_frontier.py", "--protocol", ROOT/"PROTOCOL.json", "--output", "outputs/support_frontier.json"], expected=ext/"outputs/support_frontier.json", deps=("physical_modal_normal",), kind="accounting")
    if not local:
        add("composite_full", comp, [], hours=6, kind="legacy", omp="4")
        stages[-1]["command"] = ["bash", "run_ubuntu_rtx3090.sh"]
    add("composite_regimes", comp, [ROOT/"scripts/benchmark_acfo_composite_regime.py", "--protocol",
        ROOT/"legacy/ACFO_FINAL_MULTISYSTEM_PROTOCOL.json", "--output", outputs/"composite_regimes.json",
        *( ["--only-case", "small"] if local else [] )], expected=outputs/"composite_regimes.json", hours=8, omp="4")
    return stages


def assessed_json(path, kind):
    if not path or not Path(path).is_file():
        return None
    p = load(path)
    if kind == "accounting":
        return p.get("accounting_complete") is True and p.get("kernel_sampling_sufficient") is True
    if "passed" in p:
        return p["passed"] is True
    if "case_count" in p:
        if p.get("schema") == "acfo-radial-physical-transfer-v1":
            return (p["case_count"] == p["passed_case_count"] == 2
                    and [r["id"] for r in p["cases"]] == ["single_illumination_single_depth", "three_illuminations_five_depths"]
                    and all(r["passed"] for r in p["cases"]))
        return p["case_count"] == 45 and p["passed_case_count"] == 45
    return None


def run_stage(stage, outputs):
    logs = outputs/"logs"
    logs.mkdir(exist_ok=True)
    name = stage["id"]
    env = os.environ.copy()
    cwd = Path(stage["cwd"])
    env.update(OMP_NUM_THREADS=stage["omp"], MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", BLIS_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=os.pathsep.join((str(cwd/"src"), str(cwd/"scripts"))), PYTHONUNBUFFERED="1")
    if name == "waxs_full": env.update(WAXS_CPP_EXTENSIONS="histogram,solvers", WAXS_CPP_OPT="native")
    if name == "odt_cartesian": env.update(WAXS_CPP_EXTENSIONS="odt", WAXS_CPP_OPT="native", CUPY_CACHE_DIR=str(cwd/".cupy_cache"))
    started = time.perf_counter()
    row = {"id": name, "kind": stage["kind"], "started_utc": datetime.now(timezone.utc).isoformat(), "command": stage["command"],
           "thread_contract": {k: env[k] for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS")}}
    print(f"START {name}", flush=True)
    try:
        with (logs/(name+".stdout.log")).open("w", encoding="utf-8") as stdout, (logs/(name+".stderr.log")).open("w", encoding="utf-8") as stderr:
            result = subprocess.run(stage["command"], cwd=cwd, env=env, stdout=stdout, stderr=stderr, timeout=stage["timeout"])
        row.update(returncode=result.returncode, execution_completed=True)
        row["numerical_or_accounting_passed"] = assessed_json(stage["expected"], stage["kind"])
        if stage["kind"] == "unit":
            row["numerical_or_accounting_passed"] = result.returncode == 0
        row["accepted_for_dependencies"] = result.returncode == 0 and (row["numerical_or_accounting_passed"] is True or stage["kind"] == "legacy")
        if stage["kind"] == "legacy":
            row["scientific_status"] = "component_return_audit_required" if result.returncode == 0 else "failed_or_prespecified_memory_outcome_requires_audit"
    except Exception as exc:
        row.update(returncode=None, execution_completed=False, accepted_for_dependencies=False, error=repr(exc), traceback=traceback.format_exc())
    row["wall_seconds"] = time.perf_counter()-started
    write(logs/(name+".json"), row)
    print(f"END {name}: rc={row['returncode']}, seconds={row['wall_seconds']:.2f}", flush=True)
    return row


def collect(roots, outputs):
    """Preserve output paths and names, avoiding basename collisions."""
    for key, root in roots.items():
        if key == "extension":
            for folder in ("reports", "outputs"):
                if (root/folder).exists():
                    shutil.copytree(root/folder, outputs/key/folder)
        else:
            # Partial receipts/logs survive even if a component crashes before making its ZIP.
            folders = {"waxs": ("evidence",), "odt": ("runs",), "composite": ("outputs",)}[key]
            for folder in folders:
                if not (root/folder).exists():
                    continue
                for p in (root/folder).rglob("*"):
                    if p.is_file() and p.suffix.lower() in {".json", ".csv", ".log", ".md", ".txt"}:
                        dest = outputs/"partial_component_evidence"/key/p.relative_to(root)
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(p, dest)
            for p in root.rglob("*return*.zip*"):
                if p.is_file():
                    dest = outputs/"component_returns"/key/p.relative_to(root)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, dest)


def execute(profile, preflight=False, confirm_idle=False):
    req = validate_request(ROOT)
    if not req["passed"]: raise RuntimeError(req)
    env = environment(profile)
    protocol = load(ROOT/"PROTOCOL.json")
    checks = {"python_311_or_newer": sys.version_info >= (3, 11), "cpu_dependencies": all(env["packages"][n] for n in ("numpy", "scipy", "pytest", "finufft"))}
    if profile != "local_check":
        expected = protocol["profiles"][profile]["gpu"]
        names = [line.split(",")[0].strip() for line in env["gpu"].get("stdout", "").splitlines()]
        checks.update(gpu_matches=names == [expected], no_competing_gpu_processes=env["no_competing_gpu_processes"],
            linux=platform.system() == "Linux", user_confirmed_idle=confirm_idle or preflight,
            gpu_python_modules=all(importlib.util.find_spec(n) is not None for n in ("torch", "cupy", "cufinufft")))
    if preflight:
        print(__import__("json").dumps({"request": req, "environment": env, "checks": checks, "benchmark_started": False}, indent=2))
        return 0 if all(checks.values()) else 2
    if not all(checks.values()):
        print(__import__("json").dumps({"preflight_refused": checks, "environment": env}, indent=2))
        return 2
    run = ROOT/"_runs"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+"_"+profile)
    outputs = run/"outputs"
    outputs.mkdir(parents=True, exist_ok=False)
    write(outputs/"request_audit.json", req)
    write(outputs/"environment.json", env)
    index = load(ROOT/"COMPONENT_INDEX.json")
    roots, audits = {}, {}
    for key, rec in index["components"].items():
        extract(ROOT/rec["archive"], run/"work"/key)
        roots[key] = run/"work"/key/rec["root"]
        audits[key] = audit(roots[key], rec["manifest"])
        audits[key]["expected_manifest_matches"] = audits[key]["manifest_sha256"] == rec["packaged_manifest_sha256"]
    write(outputs/"component_audits.json", audits)
    if not all(a["passed"] and a["expected_manifest_matches"] for a in audits.values()):
        raise RuntimeError("Component audit failed before science")
    stages = plan(profile, roots, outputs)
    write(outputs/"execution_plan.json", {"stages": stages})
    rows = []
    for stage in stages:
        prior = {r["id"]: r for r in rows}
        if any(not prior[d]["accepted_for_dependencies"] for d in stage["depends"]):
            row = {"id": stage["id"], "kind": stage["kind"], "execution_completed": False,
                   "accepted_for_dependencies": False, "blocked_by": stage["depends"]}
            rows.append(row)
        else:
            rows.append(run_stage(stage, outputs))
        write(outputs/"progress.json", {"steps": rows})
    collect(roots, outputs)
    complete = all(r["execution_completed"] for r in rows)
    numeric = all(r.get("accepted_for_dependencies", False) for r in rows)
    # Return existence is necessary, not a proxy for numerical or performance success.
    archives = list((outputs/"component_returns").rglob("*.zip")) if (outputs/"component_returns").exists() else []
    has_returns = profile == "local_check" or all(any(key in p.parts for p in archives) for key in ("waxs", "odt", "composite"))
    summary = {"schema": "acfo-operator-storyline-return-v2", "profile": profile,
        "scope": "local_package_QA" if profile == "local_check" else "prospective_external_replication",
        "request_manifest_sha256": sha(ROOT/"PACKAGE_MANIFEST.json"), "claim_freeze_sha256": sha(ROOT/"CLAIM_FREEZE.json"),
        "protocol_sha256": sha(ROOT/"PROTOCOL.json"), "technical_execution_completed": complete and has_returns,
        "all_stage_acceptance_passed": numeric, "scientific_component_audit_pending": profile != "local_check",
        "main_text_eligible": False, "full_original_grid_support_numerically_validated": False,
        "expected_stage_ids": [s["id"] for s in stages], "steps": rows}
    write(outputs/"summary.json", summary)
    bundle = run/"return_bundle"
    shutil.copytree(outputs, bundle/"outputs")
    for name in ("PACKAGE_MANIFEST.json", "PROTOCOL.json", "CLAIM_FREEZE.json"):
        shutil.copy2(ROOT/name, bundle/name)
    write(bundle/"RETURN_MANIFEST.json", {"schema": "acfo-storyline-return-files-v2", "files": records(bundle)})
    archive = run/("acfo_operator_storyline_return_"+profile+".zip")
    zip_tree(bundle, archive)
    Path(str(archive)+".sha256").write_text(sha(archive)+"  "+archive.name+"\n", encoding="utf-8")
    # Never mutate an archived summary after its manifest is finalized.
    write(run/"run_receipt.json", {"archive": str(archive), "sha256": sha(archive), "summary": summary})
    print(__import__("json").dumps({"archive": str(archive), "execution_complete": complete, "stage_acceptance": numeric}, indent=2))
    return 0 if complete and numeric and has_returns else 2


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=("server36_full", "server59_replication", "local_check"), required=True)
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--confirm-idle", action="store_true")
    a = p.parse_args()
    raise SystemExit(execute(a.profile, a.preflight, a.confirm_idle))
