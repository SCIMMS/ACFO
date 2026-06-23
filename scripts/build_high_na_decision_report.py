from __future__ import annotations

import html
import os
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MPL_CACHE = ROOT / ".matplotlib_cache"
MPL_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Patch


RESULTS = ROOT / "benchmark_results"
OUT = RESULTS / "high_na_decision_report.html"
ASSETS = RESULTS / "high_na_decision_report_assets"
NOTES = RESULTS / "high_na_decision_report_source_notes.md"

OPTION_AUTO = RESULTS / "high_na_pupil_spectrum_option_matrix_eps1e-12.csv"
OPTION_THREADS1 = RESULTS / "high_na_pupil_spectrum_option_matrix_eps1e-12_threads1.csv"
WORKLOAD_SUMMARY = RESULTS / "high_na_workload_matrix_summary.md"
OPTION_SUMMARY = RESULTS / "high_na_pupil_spectrum_option_matrix_summary.md"
BUILD_CACHE = RESULTS / "high_na_build_cache_large_grid.csv"
REGIME = RESULTS / "high_na_workload_matrix_regime.csv"
CANDIDATE_MEMO = ROOT / "docs" / "high_na_optics_candidate_problem.md"

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}
NEUTRAL = {
    "open": TOKENS["panel"],
    "xlight": "#F4F5F7",
    "light": "#E2E5EA",
    "base": "#C5CAD3",
    "mid": "#7A828F",
    "dark": "#464C55",
}
COLORS = {
    "blue": {"xlight": "#EAF1FE", "light": "#CEDFFE", "base": "#A3BEFA", "mid": "#5477C4", "dark": "#2E4780"},
    "gold": {"xlight": "#FFF4C2", "light": "#FFEA8F", "base": "#FFE15B", "mid": "#B8A037", "dark": "#736422"},
    "orange": {"xlight": "#FFEDDE", "light": "#FFBDA1", "base": "#F0986E", "mid": "#CC6F47", "dark": "#804126"},
    "olive": {"xlight": "#D8ECBD", "light": "#BEEB96", "base": "#A3D576", "mid": "#71B436", "dark": "#386411"},
    "pink": {"xlight": "#FCDAD6", "light": "#F5BACC", "base": "#F390CA", "mid": "#BD569B", "dark": "#8A3A6F"},
}


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for column in df.columns:
        if df[column].dtype == object:
            converted = pd.to_numeric(df[column], errors="ignore")
            df[column] = converted
    return df


def fmt_float(value: float, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def fmt_sci(value: float, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}e}"


def fmt_seconds(value: float) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    value = float(value)
    if value < 0.01:
        return f"{value * 1000:.1f} ms"
    return f"{value:.3f} s"


def use_chart_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "figure.edgecolor": "none",
            "savefig.facecolor": TOKENS["surface"],
            "savefig.edgecolor": "none",
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": ["Aptos", "Inter", "Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
            "font.monospace": ["Consolas", "DejaVu Sans Mono", "monospace"],
            "patch.linewidth": 1.0,
        },
    )


def add_chart_header(fig, ax, title: str, subtitle: str, title_width: int = 74, subtitle_width: int = 108) -> None:
    title = textwrap.fill(title.strip(), width=title_width, break_long_words=False)
    subtitle = textwrap.fill(subtitle.strip(), width=subtitle_width, break_long_words=False)
    title_lines = title.count("\n") + 1
    subtitle_lines = subtitle.count("\n") + 1
    ax.set_title("")
    fig.subplots_adjust(top=max(0.63, 0.86 - 0.045 * (title_lines - 1) - 0.03 * (subtitle_lines - 1)))
    left = ax.get_position().x0
    fig.text(left, 0.985, title, ha="left", va="top", fontsize=13, fontweight="semibold", color=TOKENS["ink"], linespacing=1.08)
    fig.text(left, 0.93 - 0.045 * (title_lines - 1), subtitle, ha="left", va="top", fontsize=9, color=TOKENS["muted"], linespacing=1.18)
    sns.despine(ax=ax)


def save_fig(fig, name: str) -> Path:
    png = ASSETS / f"{name}.png"
    svg = ASSETS / f"{name}.svg"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return png


def chart_accuracy(option_auto: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    workloads = ["vortex_requires_extra_h_small", "vortex_requires_extra_h_representative"]
    variants = ["geometric_only", "adaptive_sparse", "adaptive_dense_prefix"]
    labels = {
        "geometric_only": "Geometric only",
        "adaptive_sparse": "Adaptive sparse",
        "adaptive_dense_prefix": "Dense prefix",
        "vortex_requires_extra_h_small": "Vortex h18 small",
        "vortex_requires_extra_h_representative": "Vortex h30 representative",
    }
    df = option_auto[
        option_auto["workload"].isin(workloads)
        & option_auto["variant"].isin(variants)
        & option_auto["field_l2_vs_direct"].notna()
    ].copy()
    df["accuracy_digits"] = -np.log10(df["field_l2_vs_direct"].clip(lower=1e-16))
    df["workload_label"] = df["workload"].map(labels)
    df["variant_label"] = df["variant"].map(labels)

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    palette = {
        "Geometric only": COLORS["orange"]["base"],
        "Adaptive sparse": COLORS["blue"]["base"],
        "Dense prefix": COLORS["olive"]["base"],
    }
    sns.barplot(
        data=df,
        x="accuracy_digits",
        y="workload_label",
        hue="variant_label",
        palette=palette,
        ax=ax,
        edgecolor=TOKENS["ink"],
        linewidth=0.8,
    )
    ax.set_xlabel("Accuracy digits recovered (-log10 relative field L2)")
    ax.set_ylabel("")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False, ncol=3, borderaxespad=0)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=3, fontsize=8, color=TOKENS["ink"])
    add_chart_header(
        fig,
        ax,
        "Adaptive cutoff fixes the high-azimuthal validity failure",
        "Direct-reference vortex rows; higher is better. Geometric-only has near-zero recovered digits, adaptive sparse reaches 7-9 digits.",
    )
    return save_fig(fig, "accuracy_recovery"), df


def chart_speed(option_auto: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    workloads = [
        "benign_mixed_representative_modes16",
        "vortex_extra_h_representative_modes16",
        "vortex_extra_h_large_volume_modes16",
        "benign_mixed_large_modes64",
    ]
    short = {
        "benign_mixed_representative_modes16": "Benign mixed\n16 modes",
        "vortex_extra_h_representative_modes16": "Vortex extra h\n16 modes",
        "vortex_extra_h_large_volume_modes16": "Vortex large\n16 modes",
        "benign_mixed_large_modes64": "Benign mixed\n64 modes",
    }
    df = option_auto[
        (option_auto["variant"] == "adaptive_sparse")
        & option_auto["workload"].isin(workloads)
        & option_auto["speedup_total_vs_finufft"].notna()
    ].copy()
    df["workload_label"] = df["workload"].map(short)
    df = df.set_index("workload").loc[workloads].reset_index()

    fig, ax = plt.subplots(figsize=(9.2, 5.1))
    sns.barplot(data=df, x="workload_label", y="speedup_total_vs_finufft", ax=ax, color=COLORS["blue"]["base"], edgecolor=COLORS["blue"]["dark"], linewidth=1.0)
    ax.axhline(1.0, color=TOKENS["ink"], linestyle=":", linewidth=1.0)
    ax.set_xlabel("")
    ax.set_ylabel("Total-time speedup vs FINUFFT")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1fx"))
    for patch, value in zip(ax.patches, df["speedup_total_vs_finufft"]):
        ax.text(patch.get_x() + patch.get_width() / 2, value + 0.04, f"{value:.2f}x", ha="center", va="bottom", fontsize=8, color=TOKENS["ink"])
    add_chart_header(
        fig,
        ax,
        "Adaptive sparse is already faster than FINUFFT in the repeated-mode rows",
        "Auto-thread option matrix, finufft_eps=1e-12. Speedup includes current build and evaluation path for each row.",
    )
    return save_fig(fig, "speedup_vs_finufft"), df


def chart_sparse_work(option_threads1: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    workloads = ["vortex_extra_h_representative_modes16", "vortex_extra_h_large_volume_modes16"]
    labels = {
        "vortex_extra_h_representative_modes16": "Representative vortex",
        "vortex_extra_h_large_volume_modes16": "Large-volume vortex",
        "adaptive_sparse": "Adaptive sparse",
        "adaptive_dense_prefix": "Dense prefix",
    }
    df = option_threads1[
        option_threads1["workload"].isin(workloads)
        & option_threads1["variant"].isin(["adaptive_sparse", "adaptive_dense_prefix"])
    ].copy()
    df["workload_label"] = df["workload"].map(labels)
    df["variant_label"] = df["variant"].map(labels)

    fig, ax = plt.subplots(figsize=(8.8, 5.1))
    palette = {"Adaptive sparse": COLORS["blue"]["base"], "Dense prefix": COLORS["gold"]["base"]}
    sns.barplot(data=df, x="workload_label", y="mode_rho_work", hue="variant_label", palette=palette, ax=ax, edgecolor=TOKENS["ink"], linewidth=0.8)
    ax.set_xlabel("")
    ax.set_ylabel("Mode-rho work")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False, ncol=2, borderaxespad=0)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=3, fontsize=8, color=TOKENS["ink"])
    add_chart_header(
        fig,
        ax,
        "Sparse adaptive adds only the missing harmonics instead of growing every prefix",
        "Single-thread option matrix. Use this view for algorithmic work comparison because auto-thread scheduling changes task shape.",
    )
    return save_fig(fig, "sparse_vs_dense_work"), df


def chart_cache(build_cache: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    df = build_cache[build_cache["variant"] == "cached_all_bins_summary"].copy()
    df["rho_label"] = df["rho_max"].map(lambda v: f"rho_max={float(v):.2g}")

    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    sns.barplot(data=df, x="rho_label", y="first_total_speedup_vs_uncached", ax=ax, color=COLORS["olive"]["base"], edgecolor=COLORS["olive"]["dark"], linewidth=1.0)
    ax.axhline(1.0, color=TOKENS["ink"], linestyle=":", linewidth=1.0)
    ax.set_xlabel("")
    ax.set_ylabel("All-bin cached speedup vs uncached")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1fx"))
    for patch, value in zip(ax.patches, df["first_total_speedup_vs_uncached"]):
        ax.text(patch.get_x() + patch.get_width() / 2, value + 0.02, f"{value:.2f}x", ha="center", va="bottom", fontsize=8, color=TOKENS["ink"])
    add_chart_header(
        fig,
        ax,
        "Caching helps when multiple build settings share the same geometry",
        "Large-grid build/cache CSV, cached_all_bins_summary rows. Treat as repeated-setting guidance, not a single-plan hot-path win.",
    )
    return save_fig(fig, "cache_reuse_speedup"), df


def build_html(
    accuracy_png: Path,
    speed_png: Path,
    sparse_png: Path,
    cache_png: Path,
    speed_df: pd.DataFrame,
    sparse_df: pd.DataFrame,
    cache_df: pd.DataFrame,
) -> str:
    img = lambda path: html.escape(path.relative_to(RESULTS).as_posix())
    speed_range = (speed_df["speedup_total_vs_finufft"].min(), speed_df["speedup_total_vs_finufft"].max())
    sparse_rep = sparse_df[sparse_df["workload"] == "vortex_extra_h_representative_modes16"].set_index("variant")
    sparse_large = sparse_df[sparse_df["workload"] == "vortex_extra_h_large_volume_modes16"].set_index("variant")
    rep_reduction = 1 - sparse_rep.loc["adaptive_sparse", "mode_rho_work"] / sparse_rep.loc["adaptive_dense_prefix", "mode_rho_work"]
    large_reduction = 1 - sparse_large.loc["adaptive_sparse", "mode_rho_work"] / sparse_large.loc["adaptive_dense_prefix", "mode_rho_work"]
    cache_range = (cache_df["first_total_speedup_vs_uncached"].min(), cache_df["first_total_speedup_vs_uncached"].max())

    title = "High-NA Debye-Wolf Benchmark Decision Report"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      --surface: #fcfcfd;
      --panel: #ffffff;
      --ink: #1f2430;
      --muted: #60697a;
      --rule: #dfe4ee;
      --blue: #2e4780;
      --blue-soft: #eaf1fe;
      --orange: #804126;
      --olive: #386411;
    }}
    body {{ margin: 0; background: var(--surface); color: var(--ink); font-family: "Segoe UI", Aptos, Arial, sans-serif; }}
    main {{ max-width: 980px; margin: 0 auto; padding: 42px 22px 68px; }}
    header, section {{ margin-bottom: 34px; }}
    h1 {{ font-size: 34px; line-height: 1.12; margin: 0; letter-spacing: 0; }}
    h2 {{ font-size: 23px; line-height: 1.2; margin: 0 0 13px; letter-spacing: 0; }}
    h3 {{ font-size: 17px; margin: 18px 0 8px; letter-spacing: 0; }}
    p, li {{ font-size: 15.5px; line-height: 1.62; }}
    a {{ color: var(--blue); }}
    .summary {{ background: var(--panel); border: 1px solid var(--rule); border-left: 5px solid var(--blue); border-radius: 8px; padding: 20px 22px; }}
    .summary ul {{ margin: 0; padding-left: 21px; }}
    .summary li + li {{ margin-top: 11px; }}
    .note {{ background: var(--blue-soft); border: 1px solid #cedffe; border-radius: 8px; padding: 14px 16px; }}
    figure {{ margin: 20px 0 8px; }}
    figure img {{ width: 100%; height: auto; border: 1px solid var(--rule); border-radius: 8px; background: #fff; }}
    figcaption {{ color: var(--muted); font-size: 13px; line-height: 1.45; margin-top: 8px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 14px 0; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid var(--rule); padding: 9px 8px; text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 600; }}
    .kpi-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin: 17px 0 5px; }}
    .kpi {{ border: 1px solid var(--rule); border-radius: 8px; padding: 14px 15px; background: var(--panel); }}
    .kpi strong {{ display: block; font-size: 22px; line-height: 1.2; margin-bottom: 4px; }}
    .kpi span {{ color: var(--muted); font-size: 13px; line-height: 1.35; display: block; }}
    @media (max-width: 720px) {{
      main {{ padding: 30px 16px 52px; }}
      h1 {{ font-size: 28px; }}
      .kpi-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
<main data-report-audience="product stakeholders">
  <header data-contract-section="title">
    <h1>{title}</h1>
  </header>

  <section class="summary" data-contract-section="executive-summary">
    <h2>Executive Summary</h2>
    <ul>
      <li><strong>Make `pupil_spectrum=\"adaptive\"` the safe default for this benchmark line.</strong> The direct-reference vortex checks show geometric-only cutoff is not merely inaccurate; it is invalid for high-azimuthal pupil content, with relative field L2 near 1.0. Adaptive sparse adds the missing harmonics and recovers direct-reference agreement to about 4e-10 to 4e-08.</li>
      <li><strong>The method is already competitive where the target regime matches the factorization.</strong> In the auto-thread option matrix, adaptive sparse is {fmt_float(speed_range[0])}x to {fmt_float(speed_range[1])}x faster than FINUFFT on the representative and large repeated-mode rows while preserving the corrected field accuracy.</li>
      <li><strong>The next optimization should focus on production API, thread policy, and amortization, not a wholesale algorithm change.</strong> Sparse adaptive cuts mode-rho work by about {rep_reduction:.0%} to {large_reduction:.0%} versus dense-prefix correction in single-thread vortex rows, and cache reuse helps only when several build settings share geometry.</li>
    </ul>
  </section>

  <section data-contract-section="key-findings">
    <h2>Adaptive cutoff turns the main failure mode into a controlled correction</h2>
    <p><strong>The decisive correctness result is the high-azimuthal vortex case.</strong> Geometric-only cutoff is fine for benign low-bandlimit pupils, but it misses isolated high pupil harmonics. In direct-reference rows, adaptive sparse adds the missing harmonic support rather than increasing every prefix, so it fixes the invalid result with limited extra work.</p>
    <figure>
      <img src="{img(accuracy_png)}" alt="Bar chart comparing recovered accuracy digits for geometric-only, adaptive sparse, and dense-prefix variants on direct-reference vortex workloads.">
      <figcaption>Source: option matrix CSV with direct Debye-Wolf reference rows, finufft_eps=1e-12 run. Higher recovered digits mean lower relative field L2 error.</figcaption>
    </figure>
  </section>

  <section data-contract-section="key-findings">
    <h2>The speed win appears when the workload looks like repeated high-NA propagation</h2>
    <p><strong>The competitive regime is structured focal grids with repeated coherent modes or pupil states.</strong> On the representative option-matrix rows, adaptive sparse beats FINUFFT total time even though FINUFFT is the low-level optimized generic baseline. The largest measured advantage here is the 64-mode benign mixed row, where build overhead is amortized better.</p>
    <div class="kpi-grid">
      <div class="kpi"><strong>{fmt_float(speed_range[0])}x-{fmt_float(speed_range[1])}x</strong><span>Adaptive sparse total-time speedup vs FINUFFT across representative/large auto-thread rows.</span></div>
      <div class="kpi"><strong>7-9 digits</strong><span>Direct-reference accuracy recovered on high-vortex checks after adding sparse pupil harmonics.</span></div>
      <div class="kpi"><strong>{rep_reduction:.0%}-{large_reduction:.0%}</strong><span>Single-thread mode-rho work avoided versus dense-prefix correction on vortex rows.</span></div>
    </div>
    <figure>
      <img src="{img(speed_png)}" alt="Bar chart of adaptive sparse total-time speedup versus FINUFFT for four representative workloads.">
      <figcaption>Source: auto-thread option matrix CSV. These are local build and machine timings from the 2026-06-20 benchmark snapshot.</figcaption>
    </figure>
  </section>

  <section data-contract-section="key-findings">
    <h2>Sparse adaptive is the right correction shape for isolated high harmonics</h2>
    <p><strong>The sparse path is not just a correctness patch; it keeps the work model aligned with the actual pupil spectrum.</strong> On the single-thread vortex rows, sparse adaptive uses 388 mode-rho units instead of 466 for the representative case and 988 instead of 1336 for the large-volume case. That is the evidence that the adaptive rule should add required harmonics selectively rather than growing every dense prefix.</p>
    <figure>
      <img src="{img(sparse_png)}" alt="Grouped bar chart showing mode-rho work for adaptive sparse and dense-prefix variants.">
      <figcaption>Source: single-thread option matrix CSV. Single-thread rows are the right evidence for sparse-vs-dense algorithmic work because auto-thread scheduling can change wall-time ordering.</figcaption>
    </figure>
  </section>

  <section data-contract-section="key-findings">
    <h2>Cache reuse is useful, but only for repeated build-setting sweeps</h2>
    <p><strong>Do not make caching the universal default yet.</strong> The workload matrix already says compact positive-rho no-copy is the safer default for fixed geometry and many coherent modes. The build/cache CSV shows a different use case: when the same geometry is evaluated across several cutoff-bin or margin settings, the cached all-bin summary beats uncached by {fmt_float(cache_range[0])}x to {fmt_float(cache_range[1])}x in the tested large-grid rows. That supports a cache-reuse path for sweeps, not a claim that the cached hot loop is always better.</p>
    <figure>
      <img src="{img(cache_png)}" alt="Bar chart showing all-bin cached speedup versus uncached across rho_max settings.">
      <figcaption>Source: large-grid build/cache CSV. The cached_all_bins_summary rows are repeated-setting guidance and should not be mixed with independent single-plan measurements.</figcaption>
    </figure>
  </section>

  <section data-contract-section="recommended-next-steps">
    <h2>Recommended Next Steps</h2>
    <ol>
      <li><strong>Promote adaptive pupil-spectrum handling into the production-facing path.</strong> Keep `off` for controlled low-bandlimit experiments and `warn` for diagnostics, but make `adaptive` the safe option when pupil harmonic support is unknown.</li>
      <li><strong>Separate runtime reporting into build, hot, and amortized totals.</strong> The current evidence says many-mode and repeated-state workloads are where the algorithm wins; small one-shot runs still need explicit build-overhead caveats.</li>
      <li><strong>Add thread-scaling and peak-RSS sweeps before tightening claims.</strong> Auto-thread scheduling can obscure sparse-vs-dense work, and `basis_mib` does not capture transient memory.</li>
      <li><strong>Add a domain optics baseline only after the scalar benchmark story is stable.</strong> Direct Debye-Wolf and FINUFFT are enough for proof-of-fit, but manuscript-level optics claims need FFT-Debye/vectorial package comparisons.</li>
    </ol>
  </section>

  <section data-contract-section="further-questions">
    <h2>Further Questions</h2>
    <ul>
      <li>How does adaptive sparse scale when the pupil spectrum has broad high-order support rather than isolated extra harmonics?</li>
      <li>What thread policy keeps sparse adaptive faster than dense-prefix under auto-thread production settings?</li>
      <li>How much peak memory is used by compact, cached, adaptive sparse, and FINUFFT paths on larger focal volumes?</li>
      <li>Does the same advantage survive vectorial Richards-Wolf weights and a domain-specific FFT-Debye baseline?</li>
    </ul>
  </section>

  <section data-contract-section="caveats-and-assumptions">
    <h2>Caveats and Assumptions</h2>
    <div class="note">
      <p>This report uses the local 2026-06-20 benchmark snapshot. The strongest claims are limited to scalar High-NA Debye-Wolf propagation on structured `(rho, psi, z)` focal grids with direct or FINUFFT reference checks. It should not be read as a universal NUFFT replacement, dense Cartesian FFT-Debye replacement, or vectorial microscopy result until the missing domain baselines, peak-RSS checks, and vectorial validation are added.</p>
    </div>
  </section>
</main>
</body>
</html>
"""


def build_notes(chart_paths: dict[str, Path]) -> str:
    source_list = [
        OPTION_AUTO,
        OPTION_THREADS1,
        BUILD_CACHE,
        REGIME,
        OPTION_SUMMARY,
        WORKLOAD_SUMMARY,
        CANDIDATE_MEMO,
    ]
    chart_map = [
        ("Accuracy recovery", "Comparison & Ranking / grouped bar", "Direct-reference field L2 converted to -log10 accuracy digits", chart_paths["accuracy_recovery"]),
        ("FINUFFT speedup", "Comparison & Ranking / bar", "Adaptive sparse total-time speedup vs FINUFFT for representative rows", chart_paths["speedup_vs_finufft"]),
        ("Sparse vs dense work", "Comparison & Ranking / grouped bar", "Single-thread mode-rho work for adaptive sparse and dense prefix", chart_paths["sparse_vs_dense_work"]),
        ("Cache reuse", "Comparison & Ranking / bar", "cached_all_bins_summary speedup vs uncached across rho_max settings", chart_paths["cache_reuse_speedup"]),
    ]
    lines = [
        "# High-NA Decision Report Source Notes",
        "",
        "## Report Job",
        "",
        "- Audience: product stakeholders.",
        "- Delivery mode: static HTML with Seaborn-generated PNG charts.",
        "- Question: what the High-NA Debye-Wolf benchmark says, which drivers matter, and where the team should focus next.",
        "- Snapshot date: 2026-06-20 local benchmark artifacts.",
        "",
        "## Sources Used",
        "",
    ]
    lines.extend(f"- `{path}`" for path in source_list)
    lines.extend(
        [
            "",
            "## Chart Map",
            "",
            "| Segment | Chart Family | Supports Claim | Artifact |",
            "| --- | --- | --- | --- |",
        ]
    )
    for segment, family, claim, path in chart_map:
        lines.append(f"| {segment} | {family} | {claim} | `{path}` |")
    lines.extend(
        [
            "",
            "## Validation Notes",
            "",
            "- Direct-reference claims use option-matrix rows where `field_l2_vs_direct` is present.",
            "- FINUFFT speedup claims use option-matrix rows generated with `finufft_eps=1e-12`.",
            "- Sparse-vs-dense work claims use the single-thread option matrix because auto-thread scheduling can change wall-time ordering.",
            "- Cache reuse claims use `cached_all_bins_summary` rows from `high_na_build_cache_large_grid.csv`; these are repeated-setting guidance, not independent single-plan hot-path measurements.",
            "- Scope is scalar High-NA Debye-Wolf on structured focal grids; external FFT-Debye/vectorial baselines and peak RSS remain open.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    use_chart_theme()

    option_auto = read_csv(OPTION_AUTO)
    option_threads1 = read_csv(OPTION_THREADS1)
    build_cache = read_csv(BUILD_CACHE)

    accuracy_png, accuracy_df = chart_accuracy(option_auto)
    speed_png, speed_df = chart_speed(option_auto)
    sparse_png, sparse_df = chart_sparse_work(option_threads1)
    cache_png, cache_df = chart_cache(build_cache)

    html_text = build_html(accuracy_png, speed_png, sparse_png, cache_png, speed_df, sparse_df, cache_df)
    OUT.write_text(html_text, encoding="utf-8")

    notes = build_notes(
        {
            "accuracy_recovery": accuracy_png,
            "speedup_vs_finufft": speed_png,
            "sparse_vs_dense_work": sparse_png,
            "cache_reuse_speedup": cache_png,
        }
    )
    NOTES.write_text(notes, encoding="utf-8")

    print(f"report={OUT}")
    print(f"notes={NOTES}")
    for path in [accuracy_png, speed_png, sparse_png, cache_png]:
        print(f"chart={path}")


if __name__ == "__main__":
    main()
