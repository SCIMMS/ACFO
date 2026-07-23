from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def percent(value: float) -> str:
    return f"{100.0 * float(value):.4f}%"


def milliseconds(value: float) -> str:
    return f"{1000.0 * float(value):.3f} ms"


def build_summary(payload: dict) -> str:
    problem = payload["problem"]
    timing = payload["timing_scope"]
    rows = payload["cases"]
    lines = [
        "# Cartesian detector → polar remap → ACFO selected-z 실제 복원",
        "",
        "## 결론",
        "",
        "320×320 Cartesian complex detector 데이터를 256×256 polar grid로 remap한 뒤, "
        "121개 조명 view 전체와 detector sample 전체를 사용해 selected-z ACFO 반복 복원을 실제로 연결했다. "
        "Ideal polar 입력과 remap 입력의 최종 복원 차이는 n=1과 n=8 모두 약 0.034%였으므로, "
        "이 조건에서는 detector remap이 복원 품질을 제한하는 병목이 아니다.",
        "",
        "## 측정 조건",
        "",
        f"- GPU: `{payload['device_name']}`",
        f"- full object grid: `{problem['full_object_shape'][0]} × {problem['full_object_shape'][1]} × {problem['full_object_shape'][2]}`",
        f"- Cartesian detector: `{problem['cartesian_detector_shape'][0]} × {problem['cartesian_detector_shape'][1]}`",
        f"- virtual polar detector: `{problem['polar_detector_shape'][0]} × {problem['polar_detector_shape'][1]}`",
        f"- illumination: ring `{problem['ring_illumination_count']}` + axis `1` = `{problem['total_illumination_count']}` views",
        f"- solver: `{problem['solver']}`, `{problem['iterations']}` iterations, `{problem['constraint']}` constraint",
        "- noise: 없음. hologram demodulation 및 host-to-device acquisition transfer도 포함하지 않음",
        "- Cartesian reference: direct cuFINUFFT type-3 forward를 illumination block으로 생성하고 즉시 remap",
        "",
        "## 주요 결과",
        "",
        "| selected z | unknown bins | q samples | remap input vs ideal | ideal object error | remap object error | remap vs ideal reconstruction | core iteration median |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        ideal = row["ideal_polar"]
        remap = row["cartesian_remap"]
        lines.append(
            "| "
            f"{row['selected_n_z']} | "
            f"{row['selected_object_bins']:,} | "
            f"{row['q_samples']:,} | "
            f"{percent(row['cartesian_remap_rel_l2_vs_ideal_polar'])} | "
            f"{percent(ideal['object_rel_l2'])} | "
            f"{percent(remap['object_rel_l2'])} | "
            f"{percent(row['remap_vs_ideal_reconstruction_rel_l2'])} | "
            f"{milliseconds(ideal['core_iteration_timing']['median_s'])} |"
        )
    lines.extend(
        [
            "",
            "각 core iteration은 GPU-resident data에 대한 ACFO forward + adjoint + FISTA update이다. "
            "진단용 forward, geometry/setup, direct reference 생성, remap은 core timing에서 제외했다.",
            "",
            "## 수렴과 해석",
            "",
        ]
    )
    for row in rows:
        ideal = row["ideal_polar"]
        remap = row["cartesian_remap"]
        lines.extend(
            [
                f"### n={row['selected_n_z']}",
                "",
                f"- ideal polar: object error `{percent(ideal['object_rel_l2'])}`, data residual `{percent(ideal['data_residual'])}`",
                f"- Cartesian remap: object error `{percent(remap['object_rel_l2'])}`, data residual `{percent(remap['data_residual'])}`",
                f"- remap 입력 자체의 차이: `{percent(row['cartesian_remap_rel_l2_vs_ideal_polar'])}`",
                f"- 두 복원 결과 간 차이: `{percent(row['remap_vs_ideal_reconstruction_rel_l2'])}`",
                f"- 120회 core solve: ideal `{ideal['core_solve_s']:.3f} s`, remap `{remap['core_solve_s']:.3f} s`",
                "",
            ]
        )
    lines.extend(
        [
            "n=8에서 object error가 n=1보다 큰데도 ideal/remap 결과가 거의 같다는 점이 중요하다. "
            "따라서 이 차이는 Cartesian detector 구조나 interpolation 오차가 아니라, 여러 z plane을 동시에 푸는 "
            "inverse conditioning 및 제한된 데이터 모드의 영향으로 해석하는 것이 맞다.",
            "",
            "## 시간 분해",
            "",
            f"- context build: `{timing['context_build_s']:.3f} s`",
            f"- ACFO GPU plan build: `{timing['acfo_plan_build_s']:.3f} s`",
        ]
    )
    for row in rows:
        ref = row["reference_generation"]
        lines.append(
            f"- n={row['selected_n_z']} validation reference: total `{ref['generation_total_s']:.3f} s` "
            f"(direct cuFINUFFT 합 `{ref['direct_cufinufft_total_s']:.3f} s`, chunked remap 합 `{ref['remap_total_s']:.3f} s`)"
        )
    lines.extend(
        [
            f"- ACFO/PyTorch allocator peak: `{payload['gpu_peak_allocated_mib']:.1f} MiB`",
            f"- peak 범위: `{payload.get('gpu_peak_scope', 'PyTorch allocator only')}`",
            "",
            "Direct Cartesian reference 시간은 정확성 검증을 위한 one-off 비용이다. 실제 반복 측정에서는 같은 "
            "geometry와 ACFO plan을 재사용하고, 이미 GPU에 존재하는 새 detector frame을 remap한 뒤 core iteration을 수행한다.",
            "",
            "## 주장 경계",
            "",
            "- 이것은 단순 forward/adjoint timing이 아니라 실제로 120회 수렴시킨 selected-slice reconstruction이다.",
            "- 모든 121 views와 7,929,856 polar samples를 사용했다.",
            "- 다만 full 256³ unknown을 푼 것은 아니다. n=1은 중앙 한 plane, n=8은 중앙 8개 plane만 unknown이다.",
            "- 첫 chained validation이므로 noise, hologram demodulation, regularization sweep은 아직 포함하지 않았다.",
            "- 현재 데이터로 방어 가능한 결론은 '320² Cartesian detector와 cached polar remap을 사용해도 selected-z ACFO 복원 결과가 ideal polar 결과와 사실상 동일하다'이다.",
            "",
            "## 다음 단계",
            "",
            "1. n=8에서 regularization 및 iteration sweep으로 6.86% floor가 conditioning인지 조기 종료인지 분리한다.",
            "2. controlled complex Gaussian noise 조건을 추가하되 이를 photon noise라고 부르지 않는다.",
            "3. 실제 hologram demodulation/FFT 출력 형식을 입력 contract로 고정한다.",
            "4. 필요한 경우 exact Cartesian likelihood를 구현해 polar interpolation 기반 inverse와 비교한다.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize chained virtual-polar selected-z ODT reconstruction."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT
        / "benchmark_results"
        / "odt_virtual_polar_selected_z_reconstruction.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "benchmark_results"
        / "odt_virtual_polar_selected_z_reconstruction_summary_ko.md",
    )
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_summary(payload), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
