from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Summarize virtual polar ODT results.")
    p.add_argument(
        "--hot",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_virtual_polar_detector_gpu.json",
    )
    p.add_argument(
        "--accuracy",
        type=Path,
        default=ROOT
        / "benchmark_results"
        / "odt_virtual_polar_detector_full_accuracy.json",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "benchmark_results"
        / "odt_virtual_polar_detector_decision.json",
    )
    p.add_argument(
        "--summary-md",
        type=Path,
        default=ROOT
        / "benchmark_results"
        / "odt_virtual_polar_detector_decision_ko.md",
    )
    return p


def main() -> None:
    args = parser().parse_args()
    hot = load(args.hot)
    accuracy = load(args.accuracy)
    accuracy_by_size = {
        int(row["camera_n_xy"]): row
        for row in accuracy["physics_accuracy"]["rows"]
    }
    rows: list[dict[str, Any]] = []
    for hot_row in hot["hot_sweep"]:
        size = int(hot_row["camera_n_xy"])
        accuracy_row = accuracy_by_size[size]
        n1 = hot_row["selected_slice_pipeline_estimates"]["n_selected_1"]
        n8 = hot_row["selected_slice_pipeline_estimates"]["n_selected_8"]
        rows.append(
            {
                "camera_n_xy": size,
                "active_pixels_per_view": int(
                    hot_row["camera_active_pixels_per_view"]
                ),
                "active_to_polar_sample_ratio": float(
                    accuracy_row["active_to_polar_sample_ratio"]
                ),
                "input_mib_121_views": float(hot_row["camera_input_mib"]),
                "remap_median_s": float(hot_row["hot_timing"]["median_s"]),
                "remap_p05_s": float(hot_row["hot_timing"]["p05_s"]),
                "remap_p95_s": float(hot_row["hot_timing"]["p95_s"]),
                "remap_peak_mib": float(hot_row["hot_peak_allocated_mib"]),
                "full_pupil_rel_l2": float(accuracy_row["full_pupil_rel_l2"]),
                "inner_95pct_rel_l2": float(
                    accuracy_row["inner_pupil_rel_l2"]
                ),
                "n1_remap_overhead_fraction": float(
                    n1["remap_overhead_fraction_of_pair"]
                ),
                "n1_one_pair_pipeline_hz": float(n1["one_pair_pipeline_hz"]),
                "n8_remap_overhead_fraction": float(
                    n8["remap_overhead_fraction_of_pair"]
                ),
                "n8_one_pair_pipeline_hz": float(n8["one_pair_pipeline_hz"]),
            }
        )

    summary = {
        "schema": "odt-virtual-polar-detector-decision-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": hot["device_name"],
        "protocol": {
            "hot_views": int(hot["protocol"]["full_measurement_frames"]),
            "polar_shape": hot["protocol"]["polar_shape_per_view"],
            "hot_repeats": int(hot["protocol"]["repeats"]),
            "gpu_resident": True,
            "setup_excluded": True,
            "pupil_boundary_weight_normalization": hot["protocol"][
                "pupil_boundary_weight_normalization"
            ],
            "accuracy_reference": accuracy["physics_accuracy"]["metadata"],
        },
        "operator_sanity": hot["operator_sanity"],
        "rows": rows,
        "recommended_initial_camera_n_xy": 320,
        "recommendation_reason": [
            "The circular-pupil active sample count exceeds the 256x256 polar output count.",
            "The measured full-pupil interpolation error is approximately 0.1 percent.",
            "The remap adds about 2 percent to the selected-slice ACFO hot pair while using less memory than the 384 and 512 cases.",
        ],
        "claim_boundary": [
            "This validates a GPU-resident virtual polar detector preprocessing step, not a custom physical polar sensor.",
            "The remap is paid once per acquired complex-field stack and is reusable across iterative updates.",
            "Transfer, hologram demodulation, image-plane FFT, noise propagation, and reconstruction-level quality are not included.",
            "The polar samples are correlated interpolants and must not be counted as independent detector measurements.",
            "Exact Cartesian detector-space likelihood requires including the remap and its adjoint inside the reconstruction operator.",
        ],
        "validation_assessment": "promising; proceed to noisy reconstruction validation",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# ODT virtual polar detector 결정 요약",
        "",
        "## 결론",
        "",
        "일반 Cartesian 카메라의 원형 pupil 데이터를 GPU에서 cached polar grid로 변환하는 경로는 "
        "성능과 정확도 모두 유망하다. 첫 구현 후보로는 **320×320 Cartesian ROI → 256×256 polar cap**을 권장한다.",
        "",
        "## 통합 결과",
        "",
        "| Cartesian camera | active/view | active/polar | remap ms | full-pupil error | input MiB (121 views) | n=1 overhead | n=1 pipeline Hz |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['camera_n_xy']}² | {row['active_pixels_per_view']:,} | "
            f"{row['active_to_polar_sample_ratio']:.3f} | "
            f"{1000.0 * row['remap_median_s']:.3f} | "
            f"{100.0 * row['full_pupil_rel_l2']:.4f}% | "
            f"{row['input_mib_121_views']:.1f} | "
            f"{100.0 * row['n1_remap_overhead_fraction']:.2f}% | "
            f"{row['n1_one_pair_pipeline_hz']:.3f} |"
        )
    lines.extend(
        [
            "",
            "320²가 균형점인 이유는 원형 pupil 내부 raw pixel이 79,900개로 polar 출력 65,536개보다 많고, "
            "전체 pupil 오차가 약 0.1025%이며, 121-view remap이 약 2.7 ms에 불과하기 때문이다. "
            "384²는 오차를 약 0.0720%로 낮추지만 입력 메모리가 94.5 MiB에서 136.1 MiB로 증가한다.",
            "",
            "## 연산자 검증",
            "",
            f"- prepared gather/scatter adjoint 오차: {hot['operator_sanity']['dot_error']:.3g}",
            f"- fused CUDA grid_sample 대 명시적 bilinear gather 상대 L2: {hot['operator_sanity']['grid_sample_rel_l2']:.3g}",
            "- 정확도 reference: 같은 ODT 물체를 Cartesian 및 polar detector q 좌표에서 cuFINUFFT type-3로 각각 직접 계산",
            "",
            "## 남은 검증",
            "",
            "다음 단계는 photon/read noise를 포함한 Cartesian complex-field data를 생성하고, "
            "(a) virtual-polar likelihood와 (b) remap forward/adjoint를 포함한 정확한 Cartesian likelihood의 "
            "selected-slice reconstruction 품질과 반복 시간을 비교하는 것이다.",
            "",
            "현재 polar 값은 보간된 상관 표본이므로 독립 detector pixel 수로 세면 안 된다. "
            "또한 GPU 전송, hologram demodulation, image-plane FFT는 이번 시간에 포함되지 않았다.",
        ]
    )
    args.summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
