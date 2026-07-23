from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "runs" / "mplconfig"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot measured, censored, and extrapolated high-q timing evidence."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("benchmark_results/protein_lattice_highq_threshold_strategy.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/protein_lattice_highq_threshold_strategy.png"),
    )
    args = parser.parse_args()
    data = json.loads((ROOT / args.input).read_text(encoding="utf-8"))

    positions = data["position_sweep"]["rows"]
    resolution = data["resolution_sweep"]["rows"]
    centers = np.asarray([row["q_center_inv_angstrom"] for row in positions])
    speedups = np.asarray(
        [row["measured_speedup_or_lower_bound"] for row in positions]
    )

    plt.rcParams.update({"font.size": 9, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), constrained_layout=True)

    ax = axes[0]
    ax.plot(centers, speedups, "o-", color="#245da8", linewidth=2, markersize=6)
    for x, y, row in zip(centers, speedups, positions):
        ax.annotate(
            f"{y:.1f}x\nNphi={row['nphi']}",
            (x, y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
        )
    ax.set_xlabel(r"q-window center ($\AA^{-1}$), width = 1.3 $\AA^{-1}$")
    ax.set_ylabel("Measured FINUFFT / ACFO first-total")
    ax.set_title("Equal-width q-window position sweep")
    ax.grid(True, alpha=0.25)
    ax.set_ylim(0, float(np.max(speedups)) * 1.18)

    ax = axes[1]
    nq = np.asarray([row["nq"] for row in resolution], dtype=float)
    acfo = np.asarray([row["factorized_first_total_seconds"] for row in resolution])
    ax.plot(nq, acfo, "o-", color="#245da8", label="ACFO measured", linewidth=2)

    measured_x = []
    measured_y = []
    censored_x = []
    censored_y = []
    projected_x = []
    projected_y = []
    projected_low = []
    projected_high = []
    for row in resolution:
        if row["finufft_wall_seconds"] is not None:
            if row["finufft_censored"]:
                censored_x.append(row["nq"])
                censored_y.append(row["finufft_wall_seconds"])
            else:
                measured_x.append(row["nq"])
                measured_y.append(row["finufft_wall_seconds"])
        extrapolation = row["extrapolation"]
        if extrapolation.get("used"):
            projected_x.append(row["nq"])
            projected_y.append(extrapolation["predicted_full_finufft_seconds"])
            projected_low.append(extrapolation["prediction_interval_seconds"][0])
            projected_high.append(extrapolation["prediction_interval_seconds"][1])
    ax.plot(
        measured_x,
        measured_y,
        "o",
        color="#d95f02",
        markersize=7,
        label="FINUFFT measured complete",
    )
    ax.scatter(
        censored_x,
        censored_y,
        marker="^",
        s=70,
        color="#d95f02",
        label="FINUFFT 180 s lower bound",
        zorder=4,
    )
    for x, y in zip(censored_x, censored_y):
        ax.annotate(
            "",
            xy=(x, y * 1.35),
            xytext=(x, y),
            arrowprops={"arrowstyle": "->", "color": "#d95f02", "lw": 1.4},
        )
    if projected_x:
        order = np.argsort(projected_x)
        px = np.asarray(projected_x)[order]
        py = np.asarray(projected_y)[order]
        low = np.asarray(projected_low)[order]
        high = np.asarray(projected_high)[order]
        ax.plot(
            px,
            py,
            "--",
            color="#d95f02",
            linewidth=1.8,
            label="FINUFFT holdout-gated projection",
        )
        ax.fill_between(px, low, high, color="#d95f02", alpha=0.15)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(nq)
    ax.set_xticklabels([str(int(value)) for value in nq])
    ax.set_xlabel(r"Nq at q = 6.7-8.0 $\AA^{-1}$")
    ax.set_ylabel("First-total / streamed wall time (s)")
    ax.set_title("Resolution sweep with explicit censoring")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7.2, loc="upper left")

    fig.suptitle(
        "1.001M-atom repeated protein crystal; FINUFFT eps=1e-6, 4 threads, q-block=2",
        fontsize=10,
        fontweight="bold",
    )
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
