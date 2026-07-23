from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark_results"
DOCS = ROOT / "docs"


def load(name: str) -> dict[str, Any]:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def main() -> None:
    validation = load("odt_torch_reduced_contraction_256cubed.json")
    legacy = load("odt_torch_256cubed_100pair.json")
    optimized = load("odt_torch_256cubed_reduced_100pair.json")
    cufinufft = load("odt_cufinufft_256cubed_100pair.json")
    xlarge_guard = load("odt_torch_reduced_contraction_xlarge_auto_guard.json")

    legacy_hot = float(legacy["pair_timing"]["median_s"])
    optimized_hot = float(optimized["pair_timing"]["median_s"])
    cufinufft_hot = float(cufinufft["pair_timing"]["median_s"])
    optimized_100 = float(optimized["pair_timing"]["total_s"])
    legacy_100 = float(legacy["pair_timing"]["total_s"])
    cufinufft_100 = float(cufinufft["pair_timing"]["total_s"])
    optimized_setup_total = float(optimized["setup_s"]) + optimized_100
    legacy_setup_total = float(legacy["setup_s"]) + legacy_100
    cufinufft_setup_total = float(cufinufft["setup_s"]) + cufinufft_100

    decision = {
        "schema": "odt-reduced-contraction-optimization-decision-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "RTX 2070 SUPER, complex64, 256^3 object, 121 illuminations, 256^2 detector",
        "algorithm": {
            "legacy": "nested radial/illumination streaming with repeated compact contractions",
            "optimized": "illumination-reduced forward and adjoint mapped to matmul plus source-slot index_add",
            "auto_guard": "select reduced only for blocked streaming with at least 8 illuminations; keep unblocked prepared and one-angle axis paths legacy",
        },
        "accuracy": validation["accuracy"],
        "individual_operator": validation["timing"],
        "pair_workflow": {
            "legacy_hot_median_s": legacy_hot,
            "optimized_hot_median_s": optimized_hot,
            "cufinufft_hot_median_s": cufinufft_hot,
            "optimized_speedup_vs_legacy_hot": legacy_hot / optimized_hot,
            "optimized_speedup_vs_cufinufft_hot": cufinufft_hot / optimized_hot,
            "legacy_100pair_s": legacy_100,
            "optimized_100pair_s": optimized_100,
            "cufinufft_100pair_s": cufinufft_100,
            "optimized_speedup_vs_legacy_100pair": legacy_100 / optimized_100,
            "optimized_speedup_vs_cufinufft_100pair": cufinufft_100 / optimized_100,
            "optimized_setup_plus_100pair_s": optimized_setup_total,
            "optimized_speedup_vs_legacy_setup_plus_100pair": legacy_setup_total
            / optimized_setup_total,
            "optimized_speedup_vs_cufinufft_setup_plus_100pair": cufinufft_setup_total
            / optimized_setup_total,
            "optimized_pair_p05_s": float(optimized["pair_timing"]["p05_s"]),
            "optimized_pair_p95_s": float(optimized["pair_timing"]["p95_s"]),
        },
        "memory": {
            "legacy_peak_allocated_mib": float(legacy["gpu_peak_allocated_mib"]),
            "optimized_peak_allocated_mib": float(optimized["gpu_peak_allocated_mib"]),
            "legacy_peak_reserved_mib": float(legacy["gpu_peak_reserved_mib"]),
            "optimized_peak_reserved_mib": float(optimized["gpu_peak_reserved_mib"]),
        },
        "xlarge_guard": {
            "object_bins": int(xlarge_guard["object_bins"]),
            "q_samples": int(xlarge_guard["q_samples"]),
            "resolved_modes": xlarge_guard["resolved_modes"],
            "passed": bool(xlarge_guard["passed"]),
        },
        "sources": {
            "validation": "benchmark_results/odt_torch_reduced_contraction_256cubed.json",
            "legacy_100pair": "benchmark_results/odt_torch_256cubed_100pair.json",
            "optimized_100pair": "benchmark_results/odt_torch_256cubed_reduced_100pair.json",
            "cufinufft_100pair": "benchmark_results/odt_cufinufft_256cubed_100pair.json",
            "xlarge_guard": "benchmark_results/odt_torch_reduced_contraction_xlarge_auto_guard.json",
        },
    }
    decision["passed"] = bool(
        validation["passed"]
        and optimized["passed"]
        and xlarge_guard["passed"]
        and decision["pair_workflow"]["optimized_speedup_vs_cufinufft_hot"] > 1.0
    )

    output = RESULTS / "odt_reduced_contraction_optimization_decision.json"
    output.write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    accuracy = decision["accuracy"]
    operator = decision["individual_operator"]
    pair = decision["pair_workflow"]
    memory = decision["memory"]
    lines = [
        "# ODT streaming contraction 최적화 결과",
        "",
        "## 결론",
        "",
        "256³ object, 121 illuminations, 256² detector의 low-memory streaming 조건에서 illumination 축을 먼저 축약하는 forward/adjoint 재배열이 정확도 gate를 유지하면서 병목을 제거했다. 이 최적화는 모든 ODT 조건의 대체 경로가 아니라 multi-angle streaming 전용 경로이며, 기존 unblocked prepared 경로는 그대로 유지한다.",
        "",
        "## 구현 변경",
        "",
        "- adjoint: illumination 합을 `(u,h,l)`에 먼저 모은 뒤 radial/axial contraction을 수행한다.",
        "- forward: object의 `(r,z)` 축을 먼저 축약한 뒤 illumination synthesis를 수행한다.",
        "- 큰 4차원 compact tensor를 반복 생성하지 않고 cuBLAS matmul과 1-D source-slot `index_add_`를 사용한다.",
        "- `legacy`, `auto`, `illumination-reduced` 모드를 제공한다. `auto`는 blocked streaming이면서 조명이 8개 이상일 때만 reduced를 선택한다.",
        "",
        "## 256³ 정확도",
        "",
        "| metric | value | gate |",
        "| --- | ---: | ---: |",
        f"| optimized forward vs legacy rel-L2 | {accuracy['optimized_forward_rel_l2_vs_legacy']:.3e} | ≤ {accuracy['relative_l2_tolerance']:.1e} |",
        f"| optimized adjoint vs legacy rel-L2 | {accuracy['optimized_adjoint_rel_l2_vs_legacy']:.3e} | ≤ {accuracy['relative_l2_tolerance']:.1e} |",
        f"| legacy dot error | {accuracy['legacy_dot_error']:.3e} | ≤ 1e-6 |",
        f"| optimized dot error | {accuracy['optimized_dot_error']:.3e} | ≤ 1e-6 |",
        "",
        "## 개별 operator 시간",
        "",
        "| operator | legacy s | optimized s | speedup |",
        "| --- | ---: | ---: | ---: |",
        f"| forward | {operator['legacy_forward_median_s']:.6f} | {operator['optimized_forward_median_s']:.6f} | {operator['forward_speedup']:.2f}× |",
        f"| adjoint | {operator['legacy_adjoint_median_s']:.6f} | {operator['optimized_adjoint_median_s']:.6f} | {operator['adjoint_speedup']:.2f}× |",
        "",
        "## 100 forward-adjoint workflow",
        "",
        "| backend | hot pair median s | 100-pair s |",
        "| --- | ---: | ---: |",
        f"| legacy ACFO | {pair['legacy_hot_median_s']:.6f} | {pair['legacy_100pair_s']:.3f} |",
        f"| optimized ACFO | {pair['optimized_hot_median_s']:.6f} | {pair['optimized_100pair_s']:.3f} |",
        f"| cuFINUFFT | {pair['cufinufft_hot_median_s']:.6f} | {pair['cufinufft_100pair_s']:.3f} |",
        "",
        f"- optimized hot speedup: legacy 대비 `{pair['optimized_speedup_vs_legacy_hot']:.2f}×`, cuFINUFFT 대비 `{pair['optimized_speedup_vs_cufinufft_hot']:.2f}×`.",
        f"- setup+100 pair speedup: legacy 대비 `{pair['optimized_speedup_vs_legacy_setup_plus_100pair']:.2f}×`, cuFINUFFT 대비 `{pair['optimized_speedup_vs_cufinufft_setup_plus_100pair']:.2f}×`.",
        f"- optimized p05–p95: `{pair['optimized_pair_p05_s']:.6f}–{pair['optimized_pair_p95_s']:.6f} s`.",
        "",
        "## 메모리",
        "",
        f"- peak allocated: `{memory['legacy_peak_allocated_mib']:.1f} → {memory['optimized_peak_allocated_mib']:.1f} MiB`.",
        f"- peak reserved: `{memory['legacy_peak_reserved_mib']:.1f} → {memory['optimized_peak_reserved_mib']:.1f} MiB`.",
        "",
        "## Sentinel과 주장 경계",
        "",
        "- 이전 xlarge 조건(92,160 object bins, 6,619,136 q)은 unblocked prepared 경로가 더 빠르므로 auto가 legacy를 유지한다.",
        "- 256³ 결과는 full general ODT의 10 Hz를 뜻하지 않는다. hot pair는 약 2.25 pair/s이며 setup은 별도다.",
        "- 현재 비교는 동일 RTX 2070 SUPER의 저장된 cuFINUFFT 100-pair 기준이다. 조건별 crossover와 독립 machine 재실행이 다음 단계다.",
        "",
        "## Raw artifacts",
        "",
    ]
    lines.extend(f"- `{path}`" for path in decision["sources"].values())
    doc = DOCS / "odt_reduced_contraction_optimization_ko.md"
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"decision": str(output), "doc": str(doc), "passed": decision["passed"]},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
