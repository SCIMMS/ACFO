from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "r32_i4_first": ROOT / "benchmark_results/odt_streaming_256cubed_first.json",
    "r16_i4_repeat2": ROOT / "benchmark_results/odt_streaming_256cubed_r16_i4_repeat2.json",
}
ACFO_100 = ROOT / "benchmark_results/odt_torch_256cubed_100pair.json"
CUFINUFFT_100 = ROOT / "benchmark_results/odt_cufinufft_256cubed_100pair.json"
MATCHED_ACCURACY = ROOT / "benchmark_results/odt_cufinufft_gpu_256cubed_plan_pair2.json"
OUTPUT = ROOT / "benchmark_results/odt_256cubed_streaming_decision.json"
DOC = ROOT / "docs/acfo_ncs_odt_256cubed_streaming_ko.md"


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["summary"]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    rows = {name: load_summary(path) for name, path in FILES.items()}
    selected = rows["r16_i4_repeat2"]
    acfo100 = load(ACFO_100)
    cu100 = load(CUFINUFFT_100)
    matched = load(MATCHED_ACCURACY)["summary"]

    streaming_gates = {
        "object_256cubed": selected["object_bins"] == 256**3,
        "illumination_count_121": selected["total_illumination_count"] == 121,
        "detector_256_squared": selected["cap_radial"] == 256 and selected["cap_phi"] == 256,
        "peak_allocated_le_4gib": selected["gpu_peak_allocated_mib"] <= 4096.0,
        "complex64_dot_error_with_128accum_le_1e_8": selected[
            "forward_adjoint_dot_error_complex128_accum"
        ]
        <= 1e-8,
        "finite_forward_adjoint_pair": 0.0 < selected["gpu_forward_adjoint_pair_hot_s"] < 60.0,
    }
    workflow_gates = {
        "acfo_100_pairs_executed": acfo100["pair_timing"]["count"] == 100,
        "cufinufft_100_pairs_executed": cu100["pair_timing"]["count"] == 100,
        "matched_object_and_q_counts": (
            acfo100["object_bins"] == cu100["object_bins"] == 256**3
            and acfo100["q_samples"] == cu100["q_samples"] == 7_929_856
        ),
        "matched_illumination_count": (
            acfo100["total_illumination_count"] == cu100["total_illumination_count"] == 121
        ),
        "cufinufft_forward_l2_vs_acfo_le_1e_3": matched[
            "cufinufft_forward_rel_l2_vs_ours"
        ]
        <= 1e-3,
        "cufinufft_adjoint_l2_vs_acfo_le_1e_3": matched[
            "cufinufft_adjoint_rel_l2_vs_ours"
        ]
        <= 1e-3,
    }

    acfo_wall = float(acfo100["measured_workflow_wall_s"])
    cu_wall = float(cu100["measured_workflow_wall_s"])
    steady_speedup = cu_wall / acfo_wall
    acfo_setup_total = float(acfo100["setup_s"] + acfo_wall)
    cu_setup_total = float(cu100["setup_s"] + cu_wall)
    setup_inclusive_speedup = cu_setup_total / acfo_setup_total
    t1_acfo = float(acfo100["setup_s"] + acfo100["first_forward_adjoint_pair_s"])
    t1_cu = float(cu100["setup_s"] + cu100["first_forward_adjoint_pair_s"])

    comparative_performance_gates = {
        "steady_100pair_speedup_ge_3": steady_speedup >= 3.0,
        "setup_inclusive_100pair_speedup_ge_3": setup_inclusive_speedup >= 3.0,
    }
    result = {
        "schema": "odt-256cubed-streaming-decision-v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected": selected,
        "block_comparison": rows,
        "streaming_gates": streaming_gates,
        "workflow_gates": workflow_gates,
        "comparative_performance_gates": comparative_performance_gates,
        "streaming_feasibility_pass": all(streaming_gates.values()),
        "comparative_accuracy_pass": all(workflow_gates.values()),
        "comparative_performance_pass": all(comparative_performance_gates.values()),
        "passed": (
            all(streaming_gates.values())
            and all(workflow_gates.values())
            and all(comparative_performance_gates.values())
        ),
        "actual_100pair": {
            "acfo_source": ACFO_100.relative_to(ROOT).as_posix(),
            "cufinufft_source": CUFINUFFT_100.relative_to(ROOT).as_posix(),
            "acfo_measured_wall_s": acfo_wall,
            "cufinufft_measured_wall_s": cu_wall,
            "acfo_pair_median_s": acfo100["pair_timing"]["median_s"],
            "cufinufft_pair_median_s": cu100["pair_timing"]["median_s"],
            "acfo_setup_s": acfo100["setup_s"],
            "cufinufft_setup_s": cu100["setup_s"],
            "acfo_first_pair_s": acfo100["first_forward_adjoint_pair_s"],
            "cufinufft_first_pair_s": cu100["first_forward_adjoint_pair_s"],
            "baseline_over_acfo_steady_speedup": steady_speedup,
            "acfo_slowdown_vs_cufinufft": acfo_wall / cu_wall,
            "acfo_setup_plus_100pair_s": acfo_setup_total,
            "cufinufft_setup_plus_100pair_s": cu_setup_total,
            "baseline_over_acfo_setup_inclusive_speedup": setup_inclusive_speedup,
            "t1_setup_plus_first_pair_acfo_s": t1_acfo,
            "t1_setup_plus_first_pair_cufinufft_s": t1_cu,
            "baseline_over_acfo_t1_speedup": t1_cu / t1_acfo,
            "acfo_peak_allocated_mib": acfo100["gpu_peak_allocated_mib"],
            "cufinufft_pool_total_mib": cu100["cupy_pool_total_mib"],
            "memory_instrumentation_caveat": (
                "PyTorch max allocated and CuPy pool total are allocator-specific and are not a strict apples-to-apples peak comparison."
            ),
        },
        "matched_accuracy": {
            "source": MATCHED_ACCURACY.relative_to(ROOT).as_posix(),
            "cufinufft_eps": matched["eps"],
            "forward_rel_l2_vs_acfo": matched["cufinufft_forward_rel_l2_vs_ours"],
            "adjoint_rel_l2_vs_acfo": matched["cufinufft_adjoint_rel_l2_vs_ours"],
            "note": (
                "Accuracy was checked in a combined process; timing claims use isolated standalone processes to avoid cross-allocator interference."
            ),
        },
        "decision": (
            "256-cubed streaming feasibility and matched cuFINUFFT accuracy pass, but the comparative production gate fails. "
            f"ACFO required {acfo_wall:.2f} s for 100 measured pairs versus {cu_wall:.2f} s for cuFINUFFT, "
            f"so baseline/ACFO speedup is {steady_speedup:.3f}x and ACFO is {acfo_wall / cu_wall:.3f}x slower."
        ),
        "claim_action": (
            "Keep ODT as a memory-feasible proof-of-concept unless the structured adjoint is materially optimized; "
            "do not claim an ODT speed advantage from the current implementation."
        ),
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    r32 = rows["r32_i4_first"]
    lines = [
        "# ACFO NCS ODT 256³ streaming 검증",
        "",
        "## 판정",
        "",
        "**Streaming scale·memory·adjoint consistency PASS. 독립 cuFINUFFT comparative performance FAIL.**",
        "",
        "- object: `256 × 256 × 256` = `16,777,216` coefficients",
        "- illumination: ring 120 + axis 1 = `121`",
        "- detector: `256 × 256`; q samples `7,929,856`",
        "- GPU: RTX 2070 SUPER 8 GiB, complex64",
        "- ACFO: compact axisymmetric kernel, radial block 16, illumination block 4",
        "- baseline: cuFINUFFT 2.5.1 reusable type-3 plans, eps=1e-6",
        "",
        "## 실제 100-pair 결과",
        "",
        "| backend | setup | first pair | measured 100 pairs | pair median |",
        "|---|---:|---:|---:|---:|",
        f"| ACFO | {acfo100['setup_s']:.2f} s | {acfo100['first_forward_adjoint_pair_s']:.3f} s | {acfo_wall:.2f} s | {acfo100['pair_timing']['median_s']:.3f} s |",
        f"| cuFINUFFT | {cu100['setup_s']:.2f} s | {cu100['first_forward_adjoint_pair_s']:.3f} s | {cu_wall:.2f} s | {cu100['pair_timing']['median_s']:.3f} s |",
        "",
        f"- steady 100-pair baseline/ACFO speedup: `{steady_speedup:.3f}×`",
        f"- ACFO slowdown versus cuFINUFFT: `{acfo_wall / cu_wall:.3f}×`",
        f"- setup-inclusive 100-pair baseline/ACFO speedup: `{setup_inclusive_speedup:.3f}×`",
        f"- T=1 setup+first-pair: ACFO `{t1_acfo:.2f} s`, cuFINUFFT `{t1_cu:.2f} s`",
        "",
        "따라서 기존 995.3 s 선형 projection은 폐기한다. 실제 ACFO 100-pair 시간은 1,107.95 s였으며, cuFINUFFT가 859.55 s로 더 빨랐다.",
        "",
        "## 정확도와 메모리",
        "",
        f"- matched forward L2, cuFINUFFT vs ACFO: `{matched['cufinufft_forward_rel_l2_vs_ours']:.3e}`",
        f"- matched adjoint L2, cuFINUFFT vs ACFO: `{matched['cufinufft_adjoint_rel_l2_vs_ours']:.3e}`",
        f"- ACFO forward-adjoint dot error, complex128 accumulation: `{selected['forward_adjoint_dot_error_complex128_accum']:.3e}`",
        f"- ACFO peak allocated: `{acfo100['gpu_peak_allocated_mib']:.1f} MiB`",
        f"- cuFINUFFT CuPy pool total: `{cu100['cupy_pool_total_mib']:.1f} MiB`",
        "",
        "메모리 수치는 allocator별 계측 정의가 달라 엄밀한 peak 비교로 사용하지 않는다. 두 backend의 accuracy는 같은 프로세스에서 확인했지만 timing은 allocator 간섭을 피하기 위해 별도 standalone 프로세스로 측정했다.",
        "",
        "## Block trade-off와 후속 조치",
        "",
        f"radial 32/illumination 4는 peak `{r32['gpu_peak_allocated_mib']:.1f} MiB`, pair `{r32['gpu_forward_adjoint_pair_hot_s']:.3f} s`였다. radial 16은 peak를 `{selected['gpu_peak_allocated_mib']:.1f} MiB`로 낮췄다.",
        "",
        "현재 구현에서는 forward는 ACFO가 빠르지만 adjoint가 cuFINUFFT보다 느려 pair 전체 이점을 잃는다. ODT는 memory-feasible proof-of-concept로 유지하고, structured adjoint를 실질적으로 최적화하기 전에는 ODT speed advantage를 주장하지 않는다.",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"wrote {OUTPUT} and {DOC}")


if __name__ == "__main__":
    main()
