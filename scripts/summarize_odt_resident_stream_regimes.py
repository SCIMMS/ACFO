from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark_results"
DOCS = ROOT / "docs"
OUTPUT_JSON = RESULTS / "odt_resident_stream_regime_summary_20260713.json"
OUTPUT_MD = DOCS / "odt_resident_stream_regime_benchmark_ko.md"


FILES = {
    "q_heavy_resident": RESULTS / "odt_resident_stream_20260713_fit_resident.json",
    "q_heavy_stream": RESULTS / "odt_resident_stream_20260713_fit_forced_stream.json",
    "cube128_resident": RESULTS / "odt_resident_stream_20260713_128_resident.json",
    "cube128_stream": RESULTS / "odt_resident_stream_20260713_128_forced_stream.json",
    "cube128_equivalence": RESULTS / "odt_resident_stream_20260713_128_equivalence.json",
    "small_complex128_equivalence": RESULTS / "odt_resident_stream_20260713_64_complex128_equivalence.json",
    "cube256_resident_gate": RESULTS / "odt_resident_stream_20260713_256_resident_gate.json",
    "cube256_r8_i4": RESULTS / "odt_resident_stream_20260713_256_stream_r8_i4.json",
    "cube256_r16_i4": RESULTS / "odt_resident_stream_20260713_256_stream.json",
    "cube256_r32_i4": RESULTS / "odt_resident_stream_20260713_256_stream_r32_i4.json",
    "cube256_r64_i4": RESULTS / "odt_resident_stream_20260713_256_stream_r64_i4.json",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def timing_row(payload: dict) -> dict:
    return {
        "repeats": payload["pair_timing"]["count"],
        "setup_s": payload["setup_s"],
        "median_pair_s": payload["pair_timing"]["median_s"],
        "p05_pair_s": payload["pair_timing"]["p05_s"],
        "p95_pair_s": payload["pair_timing"]["p95_s"],
        "pair_rate_hz": 1.0 / payload["pair_timing"]["median_s"],
        "peak_allocated_mib": payload["gpu_peak_allocated_mib"],
        "peak_reserved_mib": payload["gpu_peak_reserved_mib"],
        "dot_error_complex64": payload["forward_adjoint_dot_error_complex64"],
        "passed": payload["passed"],
    }


def main() -> None:
    raw = {name: load(path) for name, path in FILES.items()}

    q_resident = timing_row(raw["q_heavy_resident"])
    q_stream = timing_row(raw["q_heavy_stream"])
    c128_resident = timing_row(raw["cube128_resident"])
    c128_stream = timing_row(raw["cube128_stream"])
    frontier = {
        block: timing_row(raw[key])
        for block, key in (
            (8, "cube256_r8_i4"),
            (16, "cube256_r16_i4"),
            (32, "cube256_r32_i4"),
            (64, "cube256_r64_i4"),
        )
    }
    gate = raw["cube256_resident_gate"]
    eq128 = raw["cube128_equivalence"]
    eq64 = raw["small_complex128_equivalence"]

    summary = {
        "schema": "odt-resident-stream-regime-summary-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "machine": {
            "device_name": raw["cube256_r16_i4"]["device_name"],
            "vram_mib": gate["total_vram_mib"],
            "torch_version": raw["cube256_r16_i4"]["torch_version"],
            "torch_cuda_version": raw["cube256_r16_i4"]["torch_cuda_version"],
            "dtype": raw["cube256_r16_i4"]["config"]["dtype"],
        },
        "q_heavy_fit_case": {
            "problem": {
                "object_bins": raw["q_heavy_resident"]["object_bins"],
                "q_samples": raw["q_heavy_resident"]["q_samples"],
                "q_per_object": raw["q_heavy_resident"]["q_samples"]
                / raw["q_heavy_resident"]["object_bins"],
            },
            "resident": q_resident,
            "forced_stream": q_stream,
            "resident_speedup_over_stream": q_stream["median_pair_s"]
            / q_resident["median_pair_s"],
            "decision": "resident",
        },
        "object_heavy_fit_case_128cubed": {
            "problem": {
                "object_bins": raw["cube128_resident"]["object_bins"],
                "q_samples": raw["cube128_resident"]["q_samples"],
                "q_per_object": raw["cube128_resident"]["q_samples"]
                / raw["cube128_resident"]["object_bins"],
            },
            "resident": c128_resident,
            "stream": c128_stream,
            "stream_speedup_over_resident": c128_resident["median_pair_s"]
            / c128_stream["median_pair_s"],
            "allocated_memory_reduction": c128_resident["peak_allocated_mib"]
            / c128_stream["peak_allocated_mib"],
            "decision": "stream-reduced-even-though-resident-fits",
        },
        "nonresident_case_256cubed": {
            "problem": gate["problem"],
            "resident_gate": gate,
            "stream_frontier_illumination_block_4": frontier,
            "balanced_default_radial_block": 16,
            "fastest_measured_radial_block": min(
                frontier, key=lambda block: frontier[block]["median_pair_s"]
            ),
            "lowest_memory_measured_radial_block": min(
                frontier, key=lambda block: frontier[block]["peak_allocated_mib"]
            ),
        },
        "accuracy": {
            "complex64_128cubed": eq128["accuracy"],
            "complex64_strict_2e_minus_6_gate_passed": eq128["passed"],
            "complex128_64cubed": eq64["accuracy"],
            "complex128_gate_passed": eq64["passed"],
            "interpretation": (
                "The 128^3 complex64 forward difference is 2.51e-6 and narrowly misses the historical "
                "2.0e-6 sentinel, while both dot errors remain O(1e-8). The complex128 cross-path result "
                "is O(1e-15), identifying floating-point accumulation order rather than an algebraic mismatch."
            ),
        },
        "decision": {
            "resident_rule": (
                "Use resident for q-heavy, small-object repeated workloads after the measured fit gate passes."
            ),
            "stream_rule": (
                "Use illumination-reduced streaming when resident does not fit, and also for object-heavy "
                "multi-angle workloads where the reduced contraction removes materialized illumination tensors."
            ),
            "dispatcher_caveat": (
                "Memory fit alone is insufficient. The current measurements bracket but do not locate the "
                "crossover between q/object ratios 0.945 and 71.82."
            ),
        },
        "sources": {name: str(path.relative_to(ROOT)) for name, path in FILES.items()},
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    q = summary["q_heavy_fit_case"]
    c128 = summary["object_heavy_fit_case_128cubed"]
    c256 = summary["nonresident_case_256cubed"]
    acc = summary["accuracy"]
    lines = [
        "# ODT resident / stream-batched regime benchmark",
        "",
        "## 결론",
        "",
        "RTX 2070 SUPER 8 GiB에서 실행 정책은 `tensor가 메모리에 들어가는가`만으로 결정할 수 없다. q-heavy·small-object 조건에서는 resident가 유리하지만, object-heavy multi-angle 조건에서는 resident가 들어가더라도 illumination-reduced streaming이 계산량과 메모리를 함께 줄였다. 256³에서는 resident가 실제 OOM으로 실패했고 stream-batched가 안정적으로 실행됐다.",
        "",
        "## 측정 환경",
        "",
        f"- GPU: `{summary['machine']['device_name']}`, `{summary['machine']['vram_mib']:.1f} MiB`",
        f"- PyTorch: `{summary['machine']['torch_version']}`, CUDA runtime `{summary['machine']['torch_cuda_version']}`",
        "- dtype: `complex64`; setup과 hot forward-adjoint pair를 분리",
        "",
        "## 1. resident가 유리한 q-heavy fit 조건",
        "",
        f"- object `{q['problem']['object_bins']:,}`, q `{q['problem']['q_samples']:,}`, q/object `{q['problem']['q_per_object']:.2f}`",
        "",
        "| policy | hot pair median | pair rate | peak allocated | setup |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| resident | {q['resident']['median_pair_s']:.6f} s | {q['resident']['pair_rate_hz']:.1f}/s | {q['resident']['peak_allocated_mib']:.1f} MiB | {q['resident']['setup_s']:.3f} s |",
        f"| forced stream | {q['forced_stream']['median_pair_s']:.6f} s | {q['forced_stream']['pair_rate_hz']:.1f}/s | {q['forced_stream']['peak_allocated_mib']:.1f} MiB | {q['forced_stream']['setup_s']:.3f} s |",
        "",
        f"resident가 `{q['resident_speedup_over_stream']:.2f}x` 빠르다.",
        "",
        "## 2. resident가 들어가지만 stream이 유리한 128³ 조건",
        "",
        f"- object `{c128['problem']['object_bins']:,}`, q `{c128['problem']['q_samples']:,}`, q/object `{c128['problem']['q_per_object']:.3f}`",
        "",
        "| policy | hot pair median | pair rate | peak allocated | peak reserved | setup |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| resident | {c128['resident']['median_pair_s']:.6f} s | {c128['resident']['pair_rate_hz']:.2f}/s | {c128['resident']['peak_allocated_mib']:.1f} MiB | {c128['resident']['peak_reserved_mib']:.1f} MiB | {c128['resident']['setup_s']:.3f} s |",
        f"| stream r16/i4 | {c128['stream']['median_pair_s']:.6f} s | {c128['stream']['pair_rate_hz']:.2f}/s | {c128['stream']['peak_allocated_mib']:.1f} MiB | {c128['stream']['peak_reserved_mib']:.1f} MiB | {c128['stream']['setup_s']:.3f} s |",
        "",
        f"stream이 `{c128['stream_speedup_over_resident']:.2f}x` 빠르고 allocated memory는 `{c128['allocated_memory_reduction']:.2f}x` 작다. 이 이득은 단순한 작은 batch 효과가 아니라 illumination 축을 먼저 줄이는 contraction 재배열에서 온다.",
        "",
        "## 3. resident 불가 256³ 조건",
        "",
        f"- object `{c256['problem']['object_bins']:,}`, q `{c256['problem']['q_samples']:,}`",
        f"- resident probe: `{c256['resident_gate']['status']}`; 단일 allocation 요구 `{9.12:.2f} GiB` 대 GPU `8.00 GiB`",
        f"- stream r16/i4 100-pair median: `{frontier[16]['median_pair_s']:.6f} s`, `{frontier[16]['pair_rate_hz']:.2f} pair/s`, peak `{frontier[16]['peak_allocated_mib']:.1f} MiB`, setup `{frontier[16]['setup_s']:.3f} s`",
        "",
        "| radial block | repeats | hot pair median | p05-p95 | peak allocated | 판단 |",
        "| ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    labels = {
        8: "minimum-memory",
        16: "balanced default",
        32: "fastest measured",
        64: "memory increase without speed win",
    }
    for block in (8, 16, 32, 64):
        row = frontier[block]
        lines.append(
            f"| {block} | {row['repeats']} | {row['median_pair_s']:.6f} s | {row['p05_pair_s']:.6f}-{row['p95_pair_s']:.6f} | {row['peak_allocated_mib']:.1f} MiB | {labels[block]} |"
        )
    lines.extend(
        [
            "",
            "r32가 r16보다 약 0.8% 빠르지만 allocated memory가 약 44% 크다. 따라서 8 GiB 기본값은 r16/i4, 더 큰 문제나 작은 GPU는 r8/i4, 충분한 VRAM에서 latency만 최소화할 때 r32/i4가 합리적이다.",
            "",
            "## 정확도 판정",
            "",
            f"- 128³ complex64 stream vs resident: forward rel-L2 `{acc['complex64_128cubed']['stream_forward_rel_l2_vs_resident']:.3e}`, adjoint rel-L2 `{acc['complex64_128cubed']['stream_adjoint_rel_l2_vs_resident']:.3e}`.",
            f"- resident/stream dot error: `{acc['complex64_128cubed']['resident_dot_error_complex128_accum']:.3e}` / `{acc['complex64_128cubed']['stream_dot_error_complex128_accum']:.3e}`.",
            "- forward 차이는 기존 2.0e-6 sentinel을 약간 넘으므로 이 엄격한 gate에는 FAIL로 기록한다. 단, 64³ complex128 교차검증은 forward `2.232e-15`, adjoint `3.550e-16`으로 PASS했다. 따라서 관측 차이는 수식 불일치가 아니라 complex64 누산 순서 차이다.",
            "",
            "## 구현·제품 정책",
            "",
            "1. `resident`: q/object가 매우 크고 object tensor가 작은 반복 geometry용.",
            "2. `stream-reduced`: resident OOM 조건과 object-heavy multi-angle 조건용.",
            "3. dispatcher는 resident memory estimate와 workload ratio를 모두 보아야 한다. 현재 두 측정점 q/object `0.945`와 `71.82` 사이의 정확한 crossover는 후속 sweep으로 정한다.",
            "4. 논문에서는 256³을 `resident-equivalent speed`가 아니라 `8 GiB에서 실행 가능한 batched proof`로 제시하고, hot pair와 setup을 분리한다.",
            "",
            "## 원시 결과",
            "",
        ]
    )
    for name, path in FILES.items():
        lines.append(f"- `{name}`: `{path.relative_to(ROOT)}`")
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary_json": str(OUTPUT_JSON), "summary_md": str(OUTPUT_MD)}, indent=2))


if __name__ == "__main__":
    main()
