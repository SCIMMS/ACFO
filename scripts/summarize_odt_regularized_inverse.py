from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALUES = ("1e6", "1e7", "1e8", "1e9", "1e10")
OUTPUT = ROOT / "benchmark_results/odt_128cubed_beads_30db_gradient_sweep.json"
DOC = ROOT / "docs/acfo_ncs_odt_regularized_inverse_ko.md"


def main() -> None:
    rows = []
    for value in VALUES:
        path = ROOT / f"benchmark_results/odt_128cubed_beads_30db_grad_{value}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "lambda": payload["problem"]["tikhonov"],
            "object_nrmse": payload["final"]["object_nrmse"],
            "data_residual": payload["final"]["data_residual"],
            "normal_residual": payload["final"]["normal_residual"],
            "solve_s": payload["timing_s"]["solve"],
            "peak_mib": payload["memory"]["gpu_peak_allocated_mib"],
            "source": path.relative_to(ROOT).as_posix(),
        })
    best = min(rows, key=lambda row: row["object_nrmse"])
    result = {
        "schema": "odt-128cubed-30db-gradient-sweep-v1",
        "problem": "128^3 real beads, 61 illuminations, 128^2 detector, 30 dB complex noise",
        "rows": rows,
        "best": best,
        "gate_nrmse_le_5pct": best["object_nrmse"] <= 0.05,
        "passed": False,
        "decision": "Quadratic gradient regularization alone is insufficient for the single-axis missing-cone ambiguity.",
        "next": "Test nonnegative structured priors and/or a second acquisition axis; do not claim physical inverse recovery from this sweep.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# ACFO NCS ODT regularized inverse: 30 dB beads",
        "",
        "## 판정",
        "",
        "**FAIL.** 단일 축 ring+axis geometry에서 quadratic spatial-gradient regularization만으로는 30 dB beads NRMSE 5% gate를 통과하지 못했다.",
        "",
        "| lambda | object NRMSE | data residual | normal residual |",
        "|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['lambda']:.1e} | {100*row['object_nrmse']:.2f}% | {100*row['data_residual']:.3f}% | {row['normal_residual']:.3e} |")
    lines.extend([
        "",
        f"최적값은 lambda={best['lambda']:.1e}, object NRMSE={100*best['object_nrmse']:.2f}%였다. 이는 기존 무정규화 무잡음 beads 18.6%보다 작은 개선이지만 30 dB 목표 5%와는 큰 차이가 있다.",
        "",
        "## 해석과 다음 단계",
        "",
        "낮은 data residual과 높은 object error가 함께 남으므로 operator 오류보다 missing-cone/null-space 식별성 문제가 지배적이다. 다음 실험은 (1) nonnegative structured prior 또는 TV, (2) second-axis/multi-axis acquisition을 분리 비교해야 한다. 이 결과를 physical inverse PASS로 사용하지 않는다.",
    ])
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
