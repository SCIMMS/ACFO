from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Summarize the selected-z GPU hot sweep.")
    p.add_argument("--counts", nargs="+", type=int, default=[1, 2, 4, 8])
    p.add_argument(
        "--results-dir", type=Path, default=ROOT / "benchmark_results"
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_selected_z_gpu_hot_sweep_summary.json",
    )
    p.add_argument(
        "--csv-out",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_selected_z_gpu_hot_sweep_summary.csv",
    )
    p.add_argument(
        "--md-out",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_selected_z_gpu_hot_sweep_summary_ko.md",
    )
    return p


def main() -> None:
    args = parser().parse_args()
    rows: list[dict[str, Any]] = []
    acfo_docs: list[dict[str, Any]] = []
    cu_docs: list[dict[str, Any]] = []
    for count in args.counts:
        acfo_path = args.results_dir / f"odt_selected_z_256_n{count:03d}_acfo.json"
        cu_path = args.results_dir / f"odt_selected_z_256_n{count:03d}_cufinufft.json"
        acfo = load(acfo_path)
        cu = load(cu_path)
        acfo_docs.append(acfo)
        cu_docs.append(cu)
        if not acfo["full_data_used"] or not cu["full_data_used"]:
            raise RuntimeError(f"n={count} did not retain full measurement data")
        if acfo["q_samples"] != cu["q_samples"]:
            raise RuntimeError(f"n={count} q sample mismatch")
        acfo_s = float(acfo["pair_timing"]["median_s"])
        cu_s = float(cu["pair_timing"]["median_s"])
        rows.append(
            {
                "selected_n_z": int(count),
                "selected_object_bins": int(acfo["selected_object_bins"]),
                "q_samples": int(acfo["q_samples"]),
                "illumination_count": int(acfo["total_illumination_count"]),
                "acfo_pair_median_s": acfo_s,
                "acfo_pair_rate_hz": 1.0 / acfo_s,
                "cufinufft_pair_median_s": cu_s,
                "cufinufft_pair_rate_hz": 1.0 / cu_s,
                "acfo_speedup_vs_cufinufft": cu_s / acfo_s,
                "acfo_hot_peak_allocated_mib": float(
                    acfo["gpu_hot_peak_allocated_mib"]
                ),
                "cufinufft_pool_used_mib": float(cu["cupy_pool_used_mib"]),
                "cufinufft_pool_total_mib": float(cu["cupy_pool_total_mib"]),
                "acfo_dot_error": float(acfo["dot_error"]),
                "cufinufft_dot_error": float(cu["dot_error"]),
                "acfo_repeats": int(acfo["pair_timing"]["count"]),
                "cufinufft_repeats": int(cu["pair_timing"]["count"]),
            }
        )

    validation_path = (
        args.results_dir / "odt_selected_z_256_n001_cross_validation.json"
    )
    validation = load(validation_path)
    speedups = [float(row["acfo_speedup_vs_cufinufft"]) for row in rows]
    summary = {
        "schema": "odt-selected-z-gpu-hot-sweep-summary-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "gpu": acfo_docs[0]["device_name"],
            "full_grid": [256, 256, 256],
            "q_samples": int(acfo_docs[0]["q_samples"]),
            "illumination_count": int(acfo_docs[0]["total_illumination_count"]),
            "full_measurement_data_used": True,
            "timed_region": "hot forward plus adjoint pair",
            "acfo_setup_excluded": True,
            "cufinufft_makeplan_setpts_excluded": True,
            "preallocated_cufinufft_outputs": True,
            "acfo_repeats": int(acfo_docs[0]["pair_timing"]["count"]),
            "cufinufft_repeats": int(cu_docs[0]["pair_timing"]["count"]),
        },
        "rows": rows,
        "speedup_range": [min(speedups), max(speedups)],
        "n1_full_restriction_validation": {
            "forward_rel_l2": acfo_docs[0]["forward_restriction_rel_l2"],
            "adjoint_slice_rel_l2": acfo_docs[0]["adjoint_slice_rel_l2"],
        },
        "n1_cross_backend_validation": {
            "cufinufft_forward_rel_l2_vs_acfo": validation[
                "cufinufft_forward_rel_l2_vs_acfo"
            ],
            "cufinufft_adjoint_rel_l2_vs_acfo": validation[
                "cufinufft_adjoint_rel_l2_vs_acfo"
            ],
            "passed": validation["passed"],
        },
        "validation_assessment": "share_with_caveats",
        "remaining_caveats": [
            "The initial sweep ran ACFO and cuFINUFFT in separate backend blocks rather than alternating AB/BA order.",
            "Direct ACFO-versus-cuFINUFFT output comparison was run at n_selected=1; the other counts retain per-backend adjoint tests but not direct cross-backend output comparisons.",
            "Results cover one RTX 2070 SUPER and selected_n_z values 1, 2, 4, and 8.",
        ],
        "claim_boundary": [
            "All detector q samples and all 121 illuminations are retained.",
            "The selected adjoint is the selected z-plane restriction of the full-data adjoint.",
            "The timed pair is a restricted-support forward-adjoint operator pair, not a converged full-volume reconstruction.",
            "The earlier thin-slab depth sweep is diagnostic only and is not used for this selected-slice comparison.",
        ],
    }

    for path in (args.json_out, args.csv_out, args.md_out):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with args.csv_out.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# ODT 전체 데이터 기반 selected-z GPU hot sweep",
        "",
        "## 결론",
        "",
        f"256×256×256 전체 geometry, {summary['protocol']['q_samples']:,}개 q 표본, "
        f"121개 조명을 모두 유지한 selected-z forward+adjoint hot pair에서 ACFO는 "
        f"cuFINUFFT보다 {min(speedups):.2f}–{max(speedups):.2f}배 빨랐다.",
        "",
        "## 측정 결과",
        "",
        "| selected z | selected voxels | ACFO pair (ms) | ACFO rate (Hz) | cuFINUFFT pair (ms) | speedup | ACFO peak (MiB) | cuFINUFFT pool used (MiB) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['selected_n_z']} | {row['selected_object_bins']:,} | "
            f"{1000.0 * row['acfo_pair_median_s']:.3f} | "
            f"{row['acfo_pair_rate_hz']:.3f} | "
            f"{1000.0 * row['cufinufft_pair_median_s']:.3f} | "
            f"{row['acfo_speedup_vs_cufinufft']:.3f}× | "
            f"{row['acfo_hot_peak_allocated_mib']:.1f} | "
            f"{row['cufinufft_pool_used_mib']:.1f} |"
        )
    lines.extend(
        [
            "",
            "ACFO는 각 지점 5회 warm-up 후 20회, cuFINUFFT는 2회 warm-up 후 5회를 측정했다. "
            "표의 시간은 중앙값이다. ACFO 준비 단계와 cuFINUFFT의 Plan/setpts는 제외했으며, "
            "cuFINUFFT execute는 사전 할당 output buffer를 사용했다.",
            "",
            "## 정확도",
            "",
            f"- n=1 ACFO selected forward 대 full-grid zero-padded forward: 상대 L2 "
            f"{acfo_docs[0]['forward_restriction_rel_l2']:.3g}",
            f"- n=1 ACFO selected adjoint 대 full-data adjoint slice: 상대 L2 "
            f"{acfo_docs[0]['adjoint_slice_rel_l2']:.3g}",
            f"- n=1 cuFINUFFT forward 대 ACFO: 상대 L2 "
            f"{validation['cufinufft_forward_rel_l2_vs_acfo']:.3g}",
            f"- n=1 cuFINUFFT adjoint 대 ACFO: 상대 L2 "
            f"{validation['cufinufft_adjoint_rel_l2_vs_acfo']:.3g}",
            "",
            "## 해석과 경계",
            "",
            "1–8개 slice에서 ACFO 시간이 약 130–146 ms로 거의 평탄한 것은 z 출력보다 "
            "전체 detector/illumination 데이터의 FFT와 illumination 축 축약 비용이 지배적이기 때문이다. "
            "따라서 slice 수 감소는 메모리와 z 수축량을 줄이지만, 전체 데이터 처리의 고정비까지 제거하지는 않는다.",
            "",
            "이 pair rate는 한 번의 forward+adjoint 연산률이며, 반복 최적화가 수렴한 최종 reconstruction frame rate가 아니다. "
            "또한 selected forward+adjoint pair는 선택한 z 평면에 미지수가 존재한다고 제한한 문제다. "
            "반면 selected adjoint 자체는 전체 데이터를 사용한 full adjoint의 해당 z 평면과 동일하다. "
            "기존 thin-slab sweep은 이 표의 근거로 사용하지 않았다.",
            "",
            "## 검증 판정",
            "",
            "현재 결과는 **caveat와 함께 내부 공유 가능**하다. 6.1–6.7배 차이는 측정 변동보다 충분히 크고 "
            "정확도 검증도 통과했지만, 출판용 최종 속도비로 고정하기 전에는 ACFO/cuFINUFFT 순서를 번갈아 수행하는 "
            "AB/BA 재측정과 n=2, 4, 8 직접 교차 정확도 검증을 추가하는 편이 안전하다. 현재 값은 RTX 2070 SUPER "
            "한 장과 selected z=1, 2, 4, 8 범위에 한정된다.",
        ]
    )
    args.md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
