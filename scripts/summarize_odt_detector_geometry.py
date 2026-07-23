from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def cases(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {case["label"]: case for case in payload["cases"]}


def selected(case: dict[str, Any], n: int) -> dict[str, Any]:
    return next(row for row in case["selected"] if row["selected_n_z"] == n)


def pct(value: float, digits: int = 3) -> str:
    return f"{100.0 * float(value):.{digits}f}%"


def ms(value: float) -> str:
    return f"{1000.0 * float(value):.3f}"


def build(args: argparse.Namespace) -> str:
    radial = load(args.radial)
    banded = load(args.banded)
    harmonic = load(args.harmonic)
    cartesian = load(args.cartesian)
    hot = load(args.hot_recheck)
    original = load(args.original_cartesian)
    radial_cases = cases(radial)
    banded_cases = cases(banded)
    harmonic_cases = cases(harmonic)

    baseline = radial_cases["rho256"]
    outer = radial_cases["outer160_p2"]
    outer128 = banded_cases["outer160_phi128"]
    bands = banded_cases["banded_inner96_outer64"]
    geometry_rows = [
        ("uniform ρ 256×256", baseline),
        ("outer-power 160×256", outer),
        ("outer-power 160×128", outer128),
        ("banded 96×128 + 64×256", bands),
    ]

    lines = [
        "# ODT detector-geometry 최적화 결과",
        "",
        "## 결론",
        "",
        "단순 Cartesian→polar 좌표 변환보다 detector radial node와 band 구조를 ACFO contraction에 맞추는 것이 훨씬 효과적이었다. "
        "최종 후보는 내부 `96×128`과 고곡률 외곽 `64×256`을 결합한 banded detector이다. "
        "기존 uniform `256×256` polar detector보다 sample/view를 56.25% 줄이면서, ideal-data 120-iteration 복원에서 "
        "n=1은 1.73×, n=8은 1.70× 빨랐고 object error도 악화되지 않았다.",
        "",
        "## Geometry sweep",
        "",
        "| geometry | samples/view | 감소 | n=1 object | n=1 ms/iter | speedup | n=8 object | n=8 ms/iter | speedup |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    base_n1 = selected(baseline, 1)["solve"]["core_iteration_timing"]["median_s"]
    base_n8 = selected(baseline, 8)["solve"]["core_iteration_timing"]["median_s"]
    for label, case in geometry_rows:
        n1 = selected(case, 1)["solve"]
        n8 = selected(case, 8)["solve"]
        samples = int(case["samples_per_view"])
        reduction = 1.0 - samples / 65536.0
        lines.append(
            f"| {label} | {samples:,} | {pct(reduction, 2)} | "
            f"{pct(n1['object_rel_l2'])} | {ms(n1['core_iteration_timing']['median_s'])} | "
            f"{base_n1 / n1['core_iteration_timing']['median_s']:.2f}× | "
            f"{pct(n8['object_rel_l2'])} | {ms(n8['core_iteration_timing']['median_s'])} | "
            f"{base_n8 / n8['core_iteration_timing']['median_s']:.2f}× |"
        )
    lines.extend(
        [
            "",
            "`outer-power` node는 낮은 NA 중심부를 성기게, Ewald 곡률이 큰 pupil 외곽을 조밀하게 배치한다. "
            "현재 harmonic cutoff는 전체/외곽 `h=36`(73 modes), 내부 band `h=32`(65 modes)이므로 내부 `nφ=128`은 alias 없이 충분하다.",
            "",
            "단일 `160×128`은 data를 가장 많이 줄였지만 `160×256`보다 계산이 빨라지지 않았다. "
            "따라서 핵심 병목은 φ pixel/FFT가 아니라 radial/object contraction이며, q sample 수만으로 실행 시간을 예측하면 안 된다.",
            "",
            "## Harmonic-domain 판정",
            "",
            "| geometry | pixel q | active-mode q | data 감소 | n=1 speedup | n=8 speedup | normal-op error |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for label in ("rho256_phi256", "outer160_phi128", "banded_inner96_outer64"):
        case = harmonic_cases[label]
        n1 = selected(case, 1)
        n8 = selected(case, 8)
        reduction = 1.0 - case["mode_q_count"] / case["pixel_q_count"]
        normal_error = max(
            n1["pixel_vs_mode_normal_rel_l2"],
            n8["pixel_vs_mode_normal_rel_l2"],
        )
        lines.append(
            f"| {label} | {case['pixel_q_count']:,} | {case['mode_q_count']:,} | "
            f"{pct(reduction, 2)} | {n1['iteration_speedup_vs_pixel']:.3f}× | "
            f"{n8['iteration_speedup_vs_pixel']:.3f}× | {normal_error:.2e} |"
        )
    lines.extend(
        [
            "",
            "Active harmonic representation은 normal operator를 약 `1.3–1.5×10⁻⁷` 내에서 보존하고 120회 복원 결과도 pixel-domain과 동일했다. "
            "그러나 iteration 가속은 약 -3%에서 +3% 범위였다. 따라서 harmonic-domain은 GPU/네트워크 전송 및 residual 저장 압축에는 유효하지만, 현재 kernel의 계산 병목을 제거하지는 않는다.",
            "",
            "## 실제 320² Cartesian camera chain",
            "",
            "최종 banded detector를 실제 특수 sensor로 가정하지 않고, 기존 320×320 Cartesian complex camera에서 cached bilinear remap해 다시 검증했다.",
            "",
            "| selected z | banded data vs ideal | ideal object | remap object | 두 복원 간 차이 | core ms/iter | remap ms | 합계 ms | update Hz |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    hot_n1 = float(hot["cases"][0]["remap_timing"]["median_s"])
    hot_by_n = {1: hot_n1, 8: float(cartesian["cases"][1]["remap_timing"]["median_s"])}
    for row in cartesian["cases"]:
        n = int(row["selected_n_z"])
        core = float(row["cartesian_remap"]["core_iteration_timing"]["median_s"])
        remap = hot_by_n[n]
        total = core + remap
        lines.append(
            f"| {n} | {pct(row['cartesian_remap_rel_l2_vs_ideal'])} | "
            f"{pct(row['ideal']['object_rel_l2'])} | {pct(row['cartesian_remap']['object_rel_l2'])} | "
            f"{pct(row['remap_vs_ideal_reconstruction_rel_l2'])} | {ms(core)} | "
            f"{ms(remap)} | {ms(total)} | {1.0 / total:.2f} |"
        )
    original_n1 = next(row for row in original["cases"] if row["selected_n_z"] == 1)
    original_n8 = next(row for row in original["cases"] if row["selected_n_z"] == 8)
    original_hot = 0.002691
    original_total_n1 = (
        original_n1["cartesian_remap"]["core_iteration_timing"]["median_s"]
        + original_hot
    )
    original_total_n8 = (
        original_n8["cartesian_remap"]["core_iteration_timing"]["median_s"]
        + original_hot
    )
    new_total_n1 = (
        cartesian["cases"][0]["cartesian_remap"]["core_iteration_timing"]["median_s"]
        + hot_n1
    )
    new_total_n8 = (
        cartesian["cases"][1]["cartesian_remap"]["core_iteration_timing"]["median_s"]
        + hot_by_n[8]
    )
    lines.extend(
        [
            "",
            f"기존 uniform-polar Cartesian chain 대비 remap+iteration 합계는 n=1에서 `{1000*original_total_n1:.1f}→{1000*new_total_n1:.1f} ms` "
            f"({original_total_n1/new_total_n1:.2f}×), n=8에서 `{1000*original_total_n8:.1f}→{1000*new_total_n8:.1f} ms` "
            f"({original_total_n8/new_total_n8:.2f}×)로 개선됐다.",
            "",
            "n=1 첫 hot-remap 측정의 11.45 ms는 독립 warmup 20/repeat 100 재측정에서 median 1.852 ms로 재현되지 않았다. "
            "n=8의 2.036 ms와 함께 `1.85–2.04 ms`를 방어 가능한 범위로 사용한다.",
            "",
            "## Setup과 구현 상태",
            "",
            f"- banded Cartesian context build: `{cartesian['context_build_s']:.3f} s`",
            f"- GPU plan build: `{cartesian['plan_build_s']:.3f} s`",
            f"- PyTorch allocator peak: `{cartesian['pytorch_peak_allocated_mib']:.1f} MiB` (CuPy direct-reference allocation 제외)",
            "- 현재 banded plan은 두 context를 독립적으로 준비하고 두 forward에서 object β-IFFT를 중복 수행하는 prototype이다.",
            "- setup table, object transform, adjoint β-FFT를 band 간 공유하는 fused banded kernel의 추가 최적화 여지가 남아 있다.",
            "",
            "## 주장 경계",
            "",
            "- 12.5 Hz와 11.1 Hz는 GPU-resident processing-side iteration/update rate이지, 120회 수렴된 volume frame rate가 아니다.",
            "- 120회 remap 복원 core 합계는 n=1 약 10.61 s, n=8 약 11.75 s이다.",
            "- 모든 결과는 noise-free complex field이며 acquisition transfer와 hologram demodulation은 제외했다.",
            "- detector geometry의 품질 이득은 high-q weighting 변화도 포함하므로, 다음 단계에서 동일 noise/covariance 조건으로 확인해야 한다.",
            "",
            "## 다음 단계",
            "",
            "1. 최종 banded geometry에서 controlled complex Gaussian noise 및 transformed covariance를 비교한다.",
            "2. shared-setup/fused-band forward-adjoint를 구현해 현재 1.70× 이상의 추가 가속 가능성을 측정한다.",
            "3. 실제 hologram demodulation 출력 contract와 Cartesian camera transfer를 별도 timing 층으로 추가한다.",
            "4. Harmonic-domain은 compute claim이 아니라 data movement/VRAM 절감 옵션으로 유지한다.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ODT detector geometry experiments.")
    parser.add_argument(
        "--radial",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_radial_geometry_shortlist_120.json",
    )
    parser.add_argument(
        "--banded",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_banded_detector.json",
    )
    parser.add_argument(
        "--harmonic",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_harmonic_detector.json",
    )
    parser.add_argument(
        "--cartesian",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_banded_cartesian_reconstruction.json",
    )
    parser.add_argument(
        "--hot-recheck",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_banded_cartesian_remap_hot_recheck.json",
    )
    parser.add_argument(
        "--original-cartesian",
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
        / "odt_detector_geometry_summary_ko.md",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build(args), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
