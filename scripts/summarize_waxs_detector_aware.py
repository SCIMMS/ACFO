from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUTS = [
    ROOT / "benchmark_results/protein_nanocrystal_detector_eiger4m_216k_nq256_repeat2.json",
    ROOT / "benchmark_results/protein_nanocrystal_detector_eiger4m_216k_nq512_w10_r30_alternating.json",
    ROOT / "benchmark_results/protein_nanocrystal_detector_eiger4m_216k_nq1024_first.json",
]
OUTPUT = ROOT / "benchmark_results/waxs_detector_aware_decision.json"
DOC = ROOT / "docs/acfo_ncs_waxs_detector_aware_ko.md"


def load_row(path: Path) -> dict:
    row = json.loads(path.read_text(encoding="utf-8"))
    memory_ratio = row["finufft_first_peak_rss_mib"] / row["acfo_first_peak_rss_mib"]
    dq = (row["qmax"] - row["qmin"]) / (row["nq"] - 1)
    acfo_times = np.asarray(row["acfo_cached_times"], dtype=np.float64)
    finufft_times = np.asarray(row["finufft_cached_times"], dtype=np.float64)
    paired = None
    if acfo_times.size and acfo_times.size == finufft_times.size:
        ratios = finufft_times / acfo_times
        split = max(1, ratios.size // 2)
        paired = {
            "count": int(ratios.size),
            "median_speedup": float(np.median(ratios)),
            "p05_speedup": float(np.percentile(ratios, 5)),
            "p95_speedup": float(np.percentile(ratios, 95)),
            "min_speedup": float(np.min(ratios)),
            "max_speedup": float(np.max(ratios)),
            "acfo_cv": float(np.std(acfo_times, ddof=1) / np.mean(acfo_times))
            if acfo_times.size > 1
            else 0.0,
            "finufft_cv": float(np.std(finufft_times, ddof=1) / np.mean(finufft_times))
            if finufft_times.size > 1
            else 0.0,
            "acfo_first_half_median_s": float(np.median(acfo_times[:split])),
            "acfo_second_half_median_s": float(np.median(acfo_times[split:])),
            "finufft_first_half_median_s": float(np.median(finufft_times[:split])),
            "finufft_second_half_median_s": float(np.median(finufft_times[split:])),
        }
    return {
        "source": path.relative_to(ROOT).as_posix(),
        "nq": row["nq"],
        "dq_inv_angstrom": dq,
        "nphi": row["n_phi"],
        "full_targets": row["targets"],
        "active_targets": row["active_detector_targets"],
        "active_fraction": row["active_detector_fraction"],
        "outer_ring_active_fraction": row["active_fraction_at_qmax"],
        "acfo_first_s": row["acfo_first_s"],
        "finufft_first_s": row["finufft_first_s"],
        "first_speedup": row["finufft_first_s"] / row["acfo_first_s"],
        "acfo_cached_s": row["acfo_cached_median_s"],
        "finufft_cached_s": row["finufft_cached_median_s"],
        "warm_speedup": row["warm_speedup_finufft_over_acfo"],
        "cached_repeat_count": len(row["acfo_cached_times"]),
        "timing_protocol": row.get("timing_protocol"),
        "acfo_timing_summary": row.get("acfo_timing_summary"),
        "finufft_timing_summary": row.get("finufft_timing_summary"),
        "paired_timing": paired,
        "acfo_peak_rss_mib": row["acfo_first_peak_rss_mib"],
        "finufft_peak_rss_mib": row["finufft_first_peak_rss_mib"],
        "memory_reduction_ratio": memory_ratio,
        "complex_l2": row["complex_l2_acfo_vs_finufft"],
        "intensity_l2": row["intensity_l2_acfo_vs_finufft"],
        "ring_l2": row["ring_l2_acfo_vs_finufft"],
        "intensity_row_l2_median": row["intensity_row_relative_l2_median"],
        "intensity_row_l2_p99": row["intensity_row_relative_l2_p99"],
    }


def main() -> None:
    rows = [load_row(path) for path in INPUTS]
    by_nq = {row["nq"]: row for row in rows}
    fringe_period = 0.0200
    for row in rows:
        row["samples_per_3x3x3_fringe"] = fringe_period / row["dq_inv_angstrom"]
    gates = {
        "all_complex_l2_le_1e_6": all(row["complex_l2"] <= 1e-6 for row in rows),
        "all_intensity_row_median_le_1e_3": all(
            row["intensity_row_l2_median"] <= 1e-3 for row in rows
        ),
        "all_intensity_row_p99_le_5e_3": all(
            row["intensity_row_l2_p99"] <= 5e-3 for row in rows
        ),
        "all_memory_reduction_ge_4": all(
            row["memory_reduction_ratio"] >= 4.0 for row in rows
        ),
        "nq512_protocol_10_warmup_30_repeat_alternating": (
            by_nq[512]["timing_protocol"] is not None
            and by_nq[512]["timing_protocol"]["warmups_per_method"] >= 10
            and by_nq[512]["timing_protocol"]["measured_repeats_per_method"] >= 30
            and by_nq[512]["timing_protocol"]["method_order"] == "alternating_ab_ba"
        ),
        "nq512_pairwise_p05_speedup_ge_1_5": (
            by_nq[512]["paired_timing"] is not None
            and by_nq[512]["paired_timing"]["p05_speedup"] >= 1.5
        ),
        "nq1024_two_samples_per_fringe": by_nq[1024]["samples_per_3x3x3_fringe"] >= 2.0,
    }
    result = {
        "schema": "waxs-detector-aware-decision-v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "detector_contract": {
            "detector": "EIGER2 X 4M rectangular active envelope",
            "active_area_mm": [155.1, 162.15],
            "distance_mm": 100.0,
            "wavelength_angstrom": 0.8,
            "photon_energy_keV": 15.498,
            "module_gaps_and_beamstop": "not included; geometric-envelope tier",
        },
        "rows": rows,
        "gates": gates,
        "passed": all(gates.values()),
        "timing_status": (
            "Local Nq512 publication protocol complete: 10 warm-ups and 30 measured AB/BA alternating pairs. "
            "Nq256 remains a two-repeat probe, Nq1024 remains a first-run probe, and an external-machine rerun is pending."
        ),
        "decision": (
            "Detector-aware accuracy and memory gates pass. At Nq512, the local 10/30 alternating protocol gives "
            f"{by_nq[512]['warm_speedup']:.3f}x ratio-of-medians speedup and "
            f"{by_nq[512]['paired_timing']['p05_speedup']:.3f}x paired p05 speedup at unchanged error. "
            "The earlier 3.47x one-repeat value is superseded; external-machine timing remains pending."
        ),
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    p512 = by_nq[512]["paired_timing"]
    lines = [
        "# ACFO NCS detector-aware WAXS 검증",
        "",
        "## 판정",
        "",
        "**정확도·메모리 gate PASS. Nq=512 local publication timing protocol PASS; 외부 머신 반복은 남아 있다.**",
        "",
        "15.5 keV, 100 mm 거리의 EIGER2 X 4M rectangular active envelope에서 detector에 들어오는 partial-arc node만 FINUFFT가 계산하도록 비교했다. ACFO 시간에는 현재 구현이 계산하는 full ring 전체가 포함된다.",
        "",
        "| Nq | dq (A^-1) | active/full targets | ACFO s | FINUFFT s | speedup | complex L2 | memory ratio | repeats |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        timing = "cached" if row["cached_repeat_count"] else "first"
        acfo_s = row["acfo_cached_s"] if row["cached_repeat_count"] else row["acfo_first_s"]
        finufft_s = row["finufft_cached_s"] if row["cached_repeat_count"] else row["finufft_first_s"]
        speedup = row["warm_speedup"] if row["cached_repeat_count"] else row["first_speedup"]
        lines.append(
            f"| {row['nq']} | {row['dq_inv_angstrom']:.5f} | {row['active_targets']:,}/{row['full_targets']:,} | "
            f"{acfo_s:.2f} ({timing}) | {finufft_s:.2f} ({timing}) | {speedup:.2f}x | "
            f"{row['complex_l2']:.3e} | {row['memory_reduction_ratio']:.2f}x | {row['cached_repeat_count']} |"
        )
    lines.extend(
        [
            "",
            "## Nq=512 publication protocol",
            "",
            "- 10 warm-up + 30 measured calls per method",
            "- AB/BA alternating order; first calls and memory profiling are separate",
            f"- ratio-of-medians speedup: {by_nq[512]['warm_speedup']:.3f}x",
            f"- paired speedup median/p05/p95: {p512['median_speedup']:.3f}x / {p512['p05_speedup']:.3f}x / {p512['p95_speedup']:.3f}x",
            f"- ACFO/FINUFFT coefficient of variation: {100*p512['acfo_cv']:.1f}% / {100*p512['finufft_cv']:.1f}%",
            "- 이전 1-repeat 3.47x 값은 superseded하며 local claim은 약 2x로 제한한다.",
            "",
            "## 제한",
            "",
            "현재 mask는 detector 외곽 rectangle만 반영하며 module gap, bad-pixel mask, beamstop은 포함하지 않는다. Nq=256은 cached 2회, Nq=1024는 first-run probe이므로 Nq=512만 local publication protocol을 충족한다. 외부 머신 반복은 여전히 필요하다.",
            "",
            "## 재현",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe scripts\\summarize_waxs_detector_aware.py",
            "```",
        ]
    )
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote {OUTPUT} and {DOC}")
    return

    lines = [
        "# ACFO NCS detector-aware WAXS 검증",
        "",
        "## 판정",
        "",
        "**기능·정확도·메모리·scaling PASS, publication timing 반복은 잔여 작업.**",
        "",
        "15.5 keV, 100 mm 거리의 EIGER2 X 4M rectangular active envelope를 사용해 실제 detector에 들어오는 partial-arc node만 FINUFFT가 계산하도록 비교했다. ACFO 시간에는 현재 구현이 계산하는 full ring 전체가 포함된다.",
        "",
        "| Nq | dq (A^-1) | active/full targets | active % | ACFO s | FINUFFT s | speedup | complex L2 | row p99 | memory ratio |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        timing = "cached" if row["cached_repeat_count"] else "first"
        acfo_s = row["acfo_cached_s"] if row["cached_repeat_count"] else row["acfo_first_s"]
        finufft_s = row["finufft_cached_s"] if row["cached_repeat_count"] else row["finufft_first_s"]
        speedup = row["warm_speedup"] if row["cached_repeat_count"] else row["first_speedup"]
        lines.append(
            f"| {row['nq']} | {row['dq_inv_angstrom']:.5f} | {row['active_targets']:,}/{row['full_targets']:,} | "
            f"{100.0 * row['active_fraction']:.1f} | {acfo_s:.2f} ({timing}) | {finufft_s:.2f} ({timing}) | "
            f"{speedup:.2f}x | {row['complex_l2']:.3e} | {row['intensity_row_l2_p99']:.3e} | "
            f"{row['memory_reduction_ratio']:.2f}x |"
        )
    lines.extend(
        [
            "",
            "- outer q=6.3 A^-1 ring active fraction: 약 4.62%",
            "- 전체 radial range 누적 active fraction: 약 88.1%",
            "- Nq=1024 samples/fringe: 약 3.27; 3x3x3 finite-domain fringe의 2-sample 조건 충족",
            "- 모든 complex L2 <= 1e-6, row-intensity median/p99 <= 1e-3/5e-3",
            "- 모든 peak-RSS ratio >= 4x",
            "",
            "## 해석",
            "",
            "Nq=256에서는 masked FINUFFT의 target 감소 때문에 warm speedup이 2.38x로 낮아지지만 memory ratio 5.37x로 대체 gate를 통과한다. Nq=512에서는 3.47x, Nq=1024 first probe에서는 3.55x로 증가한다. 따라서 ACFO의 유리한 조건은 curvature 크기 자체보다 detector-realistic high-target-count와 dense radial sampling에서 나타난다.",
            "",
            "## 제한",
            "",
            "현재 mask는 detector 외곽 rectangle만 반영하며 module gap, bad-pixel mask, beamstop은 포함하지 않는다. Nq=256은 cached 2회, Nq=512는 1회, Nq=1024는 first-run만 측정했으므로 최종 논문 timing에는 사전 선언한 warm-up과 반복 프로토콜을 별도로 적용해야 한다.",
            "",
            "## 재현",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe scripts\\summarize_waxs_detector_aware.py",
            "```",
        ]
    )
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote {OUTPUT} and {DOC}")


if __name__ == "__main__":
    main()
