from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class FullConditionTiming:
    """Measured full public aIDT timing components on the local RTX 2070 SUPER."""

    run_s: float = 0.09700215001066681
    fft_s: float = 0.004867700001341291
    rhs_s: float = 0.04650739999487996
    solve_s: float = 0.04550194999319501
    h2d_copy_s: float = 0.009735499988892116
    h2d_copy_compute_s: float = 0.1065680000174325

    @property
    def other_s(self) -> float:
        return max(0.0, self.run_s - self.fft_s - self.rhs_s - self.solve_s)


def hz(seconds: float) -> float:
    return 1.0 / seconds


def fmt_s(seconds: float) -> str:
    return f"{seconds:.4f}"


def fmt_hz(seconds: float) -> str:
    return f"{hz(seconds):.2f}"


def build_summary(path: Path) -> None:
    t = FullConditionTiming()
    required_compute_speedup = t.run_s / (0.1 - t.h2d_copy_s)
    required_copy_speedup = t.h2d_copy_s / (0.1 - t.run_s)

    compute_speedups = [1.0, 1.05, 1.08, 1.10, 1.25, 1.50, 2.0, 3.0]
    copy_speedups = [1.0, 1.5, 2.0, 3.0]
    gpu_counts = [1, 2, 4, 8]

    lines: list[str] = [
        "# Public aIDT Real-Time Projection",
        "",
        "Generated: 2026-06-23",
        "",
        "This projection starts from measured local results on the",
        "`NVIDIA GeForce RTX 2070 SUPER` for the full public aIDT condition:",
        "",
        "- detector and frame stack: `24 x 700 x 700`",
        "- reconstruction grid: `700 x 700 x 35`",
        "- mode: Torch/CUDA support RHS with cached support transfer",
        "- diagnostics: disabled",
        "- dtype: `complex64`",
        "",
        "The purpose is not to claim that end-to-end real-time analysis has already",
        "been demonstrated. The purpose is to quantify how close the measured",
        "pipeline is, and how much speedup or overlap is required to make",
        "real-time analysis practical.",
        "",
        "## Measured Baseline",
        "",
        "| component | seconds | share of GPU core |",
        "| --- | ---: | ---: |",
        f"| measurement FFT | {t.fft_s:.6f} | {100.0 * t.fft_s / t.run_s:.1f}% |",
        f"| RHS assembly | {t.rhs_s:.6f} | {100.0 * t.rhs_s / t.run_s:.1f}% |",
        f"| solve + inverse FFT | {t.solve_s:.6f} | {100.0 * t.solve_s / t.run_s:.1f}% |",
        f"| other measured overhead | {t.other_s:.6f} | {100.0 * t.other_s / t.run_s:.1f}% |",
        f"| GPU-resident reconstruction core | {t.run_s:.6f} | 100.0% |",
        f"| host-to-GPU copy, preallocated tensor | {t.h2d_copy_s:.6f} | n/a |",
        "",
        "Current full-condition rates:",
        "",
        "| condition | seconds/update | Hz | readout |",
        "| --- | ---: | ---: | --- |",
        f"| GPU-resident core | {fmt_s(t.run_s)} | {fmt_hz(t.run_s)} | already above 10 Hz |",
        f"| copy + core, sequential | {fmt_s(t.h2d_copy_compute_s)} | {fmt_hz(t.h2d_copy_compute_s)} | below 10 Hz |",
        f"| copy/core fully overlapped | {fmt_s(max(t.run_s, t.h2d_copy_s))} | {fmt_hz(max(t.run_s, t.h2d_copy_s))} | above 10 Hz if transfer is hidden |",
        "",
        "## Required Speedup",
        "",
        "For the measured sequential copy+core path to reach 10 Hz at the original",
        "`700 x 700 x 35` condition:",
        "",
        f"- If H2D copy stays unchanged, the GPU core needs only `{required_compute_speedup:.3f}x` speedup.",
        f"- If the GPU core stays unchanged, H2D copy alone would need `{required_copy_speedup:.2f}x` speedup,",
        "  because the current core is already close to the full 100 ms budget.",
        "- If H2D copy is overlapped with reconstruction, the measured core already",
        "  meets the 10 Hz budget.",
        "",
        "This is the main practical point: full-condition real-time analysis is not",
        "orders of magnitude away. On the measured older GPU, the sequential",
        "copy-included loop is about 6.6 ms over the 100 ms budget.",
        "",
        "## Single-GPU Projection",
        "",
        "This table keeps the measured H2D copy cost fixed and scales only the GPU",
        "reconstruction core. It is a required-speedup projection, not a benchmark",
        "of any specific new GPU.",
        "",
        "| GPU core speedup | sequential copy+core s | Hz | margin vs 10 Hz |",
        "| ---: | ---: | ---: | ---: |",
    ]

    for speedup in compute_speedups:
        total = t.h2d_copy_s + t.run_s / speedup
        margin_ms = (0.1 - total) * 1000.0
        lines.append(
            f"| {speedup:.2f}x | {fmt_s(total)} | {fmt_hz(total)} | {margin_ms:+.1f} ms |"
        )

    lines.extend(
        [
            "",
            "## Transfer Improvement Projection",
            "",
            "This table keeps the measured GPU core fixed and improves only H2D copy.",
            "It shows that copy acceleration alone is not the cleanest path for the",
            "full 35-slice condition; overlap or compute improvement is more useful.",
            "",
            "| H2D copy speedup | sequential copy+core s | Hz | margin vs 10 Hz |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for speedup in copy_speedups:
        total = t.run_s + t.h2d_copy_s / speedup
        margin_ms = (0.1 - total) * 1000.0
        lines.append(
            f"| {speedup:.2f}x | {fmt_s(total)} | {fmt_hz(total)} | {margin_ms:+.1f} ms |"
        )

    lines.extend(
        [
            "",
            "## Multi-GPU RHS-Partition Projection",
            "",
            "The RHS assembly is the largest single hot-stage and is naturally",
            "partitionable over illumination frames or support blocks. The conservative",
            "projection below scales only the RHS stage across GPUs and leaves H2D copy,",
            "measurement FFT, solve, and overhead unchanged on the measured baseline.",
            "It does not assume a fully distributed inverse FFT or solve.",
            "",
            "| GPUs | projected s/update | Hz | margin vs 10 Hz |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for n_gpu in gpu_counts:
        total = t.h2d_copy_s + t.fft_s + t.rhs_s / n_gpu + t.solve_s + t.other_s
        margin_ms = (0.1 - total) * 1000.0
        lines.append(
            f"| {n_gpu} | {fmt_s(total)} | {fmt_hz(total)} | {margin_ms:+.1f} ms |"
        )

    lines.extend(
        [
            "",
            "## Acquisition-Hardware Boundary",
            "",
            "The host-to-GPU transfer term should be treated as an experimental",
            "hardware and acquisition-pipeline boundary, not as an intrinsic limit of",
            "the prepared reconstruction operator. High-rate X-ray facilities already",
            "use detector and control architectures designed around burst acquisition,",
            "front-end buffering, parallel readout, and online calibration.",
            "",
            "For example, the AGIPD detector developed for European XFEL reports",
            "burst-mode storage of 352 images at up to 6.5 MHz, compatible with the",
            "4.5 MHz intra-train frame rate, and the European XFEL pulse structure",
            "contains up to 2700 pulses per train at approximately 4.5 MHz. Detector",
            "control and calibration processing at European XFEL has also been framed",
            "around distributed computing and GPU-accelerated near-real-time",
            "processing. These examples do not prove an end-to-end ODT/aIDT live",
            "pipeline for this repository, but they justify treating data movement as",
            "a solvable facility/pipeline engineering layer rather than the core",
            "algorithmic bottleneck.",
            "",
            "The appropriate claim is therefore processing-side feasibility: once a",
            "frame stack is available on the GPU, the prepared operator can update the",
            "full public-condition reconstruction at 10 Hz on older hardware, and the",
            "copy-included path is close enough that current GPUs, overlapped",
            "acquisition, or multi-GPU processing should make real-time analysis",
            "practical.",
            "",
            "Relevant detector/acquisition references:",
            "",
            "- Allahgholi et al., `The Adaptive Gain Integrating Pixel Detector at the European XFEL`, arXiv:1808.00256.",
            "- Munnich et al., `Integrated Detector Control and Calibration Processing at the European XFEL`, arXiv:1601.01794.",
            "",
            "## Claim Boundary",
            "",
            "A defensible manuscript claim is:",
            "",
            "> On an older RTX 2070 SUPER, the prepared GPU operator already reaches",
            "> 10 Hz for the GPU-resident full public aIDT reconstruction core and is",
            "> within a 1.07x compute-speedup of 10 Hz even when ordinary host-to-GPU",
            "> transfer is included. This supports the practical feasibility of",
            "> real-time analysis on current-generation GPUs, overlapped acquisition",
            "> pipelines, or modest multi-GPU deployments.",
            "",
            "The claim should not be phrased as a completed end-to-end live microscope",
            "demonstration until acquisition, preprocessing, and transfer scheduling are",
            "measured in the target experimental system. The current result is instead",
            "evidence that the processing side has entered the real-time regime.",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "benchmark_results" / "aidt_realtime_projection_summary.md",
    )
    args = parser.parse_args()
    build_summary(args.out)
    print(args.out)


if __name__ == "__main__":
    main()
