import csv
import html
import math
from pathlib import Path
from typing import Any


TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

COLORS = {
    "geometric_only": ("#C5CAD3", "#464C55"),
    "adaptive_sparse": ("#A3BEFA", "#2E4780"),
    "adaptive_dense_prefix": ("#FFE15B", "#736422"),
    "finufft": ("#F0986E", "#804126"),
}

LABELS = {
    "benign_mixed_small": "Mixed small",
    "vortex_requires_extra_h_small": "Vortex small",
    "vortex_requires_extra_h_representative": "Vortex direct",
    "benign_mixed_representative_modes16": "Mixed modes16",
    "vortex_extra_h_representative_modes16": "Vortex modes16",
    "vortex_extra_h_large_volume_modes16": "Vortex large",
    "benign_mixed_large_modes64": "Mixed modes64",
    "geometric_only": "Geometric only",
    "adaptive_sparse": "Adaptive sparse",
    "adaptive_dense_prefix": "Dense prefix",
}


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        'role="img" xmlns="http://www.w3.org/2000/svg">'
    ]


def hbar_chart(
    rows: list[dict[str, Any]],
    *,
    title: str,
    subtitle: str,
    value_key: str,
    value_label: str,
    max_value: float | None = None,
    width: int = 940,
    row_height: int = 28,
    left: int = 190,
    right: int = 92,
    top: int = 86,
) -> str:
    height = top + row_height * len(rows) + 44
    max_value = max_value or max(float(row[value_key]) for row in rows)
    plot_width = width - left - right
    out = svg_header(width, height)
    out.append(f'<rect width="{width}" height="{height}" rx="0" fill="{TOKENS["panel"]}"/>')
    out.append(
        f'<text x="{left}" y="28" font-size="18" font-weight="650" fill="{TOKENS["ink"]}">'
        f"{html.escape(title)}</text>"
    )
    out.append(
        f'<text x="{left}" y="52" font-size="12" fill="{TOKENS["muted"]}">'
        f"{html.escape(subtitle)}</text>"
    )
    for tick in range(5):
        x = left + plot_width * tick / 4.0
        value = max_value * tick / 4.0
        out.append(
            f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top - 12}" y2="{height - 32}" '
            f'stroke="{TOKENS["grid"]}" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{x:.1f}" y="{height - 12}" font-size="10" text-anchor="middle" '
            f'fill="{TOKENS["muted"]}">{value:.1f}</text>'
        )
    for i, row in enumerate(rows):
        y = top + i * row_height
        label = row.get("label", "")
        variant = row.get("variant", "adaptive_sparse")
        fill, stroke = COLORS.get(variant, COLORS["adaptive_sparse"])
        value = float(row[value_key])
        bar_width = 0.0 if max_value == 0 else plot_width * value / max_value
        out.append(
            f'<text x="{left - 10}" y="{y + 16}" font-size="11" text-anchor="end" '
            f'fill="{TOKENS["ink"]}">{html.escape(label)}</text>'
        )
        out.append(
            f'<rect x="{left}" y="{y + 3}" width="{bar_width:.1f}" height="17" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{left + bar_width + 8:.1f}" y="{y + 16}" font-size="11" '
            f'fill="{TOKENS["ink"]}">{html.escape(str(row[value_label]))}</text>'
        )
    out.append("</svg>")
    return "\n".join(out)


def grouped_work_chart(rows: list[dict[str, Any]]) -> str:
    chart_rows: list[dict[str, Any]] = []
    for row in rows:
        if row["variant"] not in {"geometric_only", "adaptive_sparse", "adaptive_dense_prefix"}:
            continue
        if "vortex" not in row["workload"]:
            continue
        workload = LABELS.get(row["workload"], row["workload"])
        variant = row["variant"]
        work = int(float(row["mode_rho_work"]))
        chart_rows.append(
            {
                "label": f"{workload} - {LABELS[variant]}",
                "variant": variant,
                "work": work,
                "work_label": f"{work}",
            }
        )
    return hbar_chart(
        chart_rows,
        title="Sparse adaptive keeps extra harmonic work local",
        subtitle="Mode-rho work from single-thread run; lower is better for equal accuracy.",
        value_key="work",
        value_label="work_label",
        max_value=max(float(row["work"]) for row in chart_rows) * 1.08,
    )


def accuracy_chart(rows: list[dict[str, Any]]) -> str:
    chart_rows: list[dict[str, Any]] = []
    wanted = {"vortex_requires_extra_h_small", "vortex_requires_extra_h_representative"}
    for row in rows:
        if row["workload"] not in wanted:
            continue
        if row["variant"] not in {"geometric_only", "adaptive_sparse", "adaptive_dense_prefix"}:
            continue
        l2 = to_float(row["field_l2_vs_direct"])
        if l2 is None:
            continue
        score = -math.log10(max(l2, 1e-16))
        chart_rows.append(
            {
                "label": f"{LABELS[row['workload']]} - {LABELS[row['variant']]}",
                "variant": row["variant"],
                "digits": score,
                "digits_label": f"L2 {l2:.1e}",
            }
        )
    return hbar_chart(
        chart_rows,
        title="Adaptive spectrum recovers high-azimuthal accuracy",
        subtitle="-log10(field L2 vs direct Debye-Wolf); higher is better.",
        value_key="digits",
        value_label="digits_label",
        max_value=10.5,
    )


def speedup_chart(rows: list[dict[str, Any]]) -> str:
    chart_rows: list[dict[str, Any]] = []
    for row in rows:
        if row["variant"] != "adaptive_sparse":
            continue
        speed = to_float(row["speedup_total_vs_finufft"])
        if speed is None:
            continue
        chart_rows.append(
            {
                "label": LABELS.get(row["workload"], row["workload"]),
                "variant": "adaptive_sparse",
                "speedup": speed,
                "speedup_label": f"{speed:.2f}x",
            }
        )
    return hbar_chart(
        chart_rows,
        title="Adaptive path usually stays ahead of FINUFFT total time",
        subtitle="Total-time speedup vs FINUFFT, eps=1e-12, auto-thread run; >1x favors this method.",
        value_key="speedup",
        value_label="speedup_label",
        max_value=max(float(row["speedup"]) for row in chart_rows) * 1.12,
    )


def summary_table(rows: list[dict[str, Any]]) -> str:
    selected = [
        row
        for row in rows
        if row["variant"] == "adaptive_sparse"
        and row["workload"]
        in {
            "benign_mixed_representative_modes16",
            "vortex_extra_h_representative_modes16",
            "vortex_extra_h_large_volume_modes16",
            "benign_mixed_large_modes64",
        }
    ]
    body = []
    for row in selected:
        l2 = row["field_l2_vs_direct"] or row["field_l2_vs_finufft"]
        body.append(
            "<tr>"
            f"<td>{html.escape(LABELS.get(row['workload'], row['workload']))}</td>"
            f"<td>{html.escape(row['required_h_values'] or 'none')}</td>"
            f"<td>{int(float(row['mode_rho_work']))}</td>"
            f"<td>{float(row['total_s']):.4f}s</td>"
            f"<td>{float(row['speedup_total_vs_finufft']):.2f}x</td>"
            f"<td>{float(l2):.1e}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Workload</th><th>Extra h</th><th>Mode-rho work</th>"
        "<th>Total time</th><th>Vs FINUFFT</th><th>L2 error</th></tr></thead><tbody>"
        + "\n".join(body)
        + "</tbody></table>"
    )


def main() -> None:
    out_dir = Path("benchmark_results")
    auto_path = out_dir / "high_na_pupil_spectrum_option_matrix_eps1e-12.csv"
    threads_path = out_dir / "high_na_pupil_spectrum_option_matrix_eps1e-12_threads1.csv"
    auto_rows = read_rows(auto_path)
    thread_rows = read_rows(threads_path)
    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>High-NA pupil-spectrum adaptive benchmark</title>
<style>
body {{
  margin: 0;
  background: {TOKENS["surface"]};
  color: {TOKENS["ink"]};
  font-family: Aptos, Inter, "Segoe UI", Arial, sans-serif;
}}
main {{ max-width: 1040px; margin: 28px auto 48px; padding: 0 22px; }}
h1 {{ font-size: 24px; margin: 0 0 8px; }}
p {{ color: {TOKENS["muted"]}; line-height: 1.5; }}
.chart {{ background: {TOKENS["panel"]}; border: 1px solid {TOKENS["axis"]}; margin: 22px 0; padding: 12px; }}
table {{ border-collapse: collapse; width: 100%; background: {TOKENS["panel"]}; margin-top: 16px; }}
th, td {{ border-bottom: 1px solid {TOKENS["grid"]}; padding: 9px 10px; text-align: left; font-size: 13px; }}
th {{ color: {TOKENS["muted"]}; font-weight: 650; }}
code {{ font-family: Consolas, "SF Mono", monospace; }}
</style>
</head>
<body>
<main>
<h1>High-NA pupil-spectrum adaptive benchmark</h1>
<p>Source: <code>{html.escape(str(auto_path))}</code> and <code>{html.escape(str(threads_path))}</code>. FINUFFT uses eps=1e-12. Direct Debye-Wolf is used where feasible for correctness.</p>
<div class="chart">{accuracy_chart(auto_rows)}</div>
<div class="chart">{grouped_work_chart(thread_rows)}</div>
<div class="chart">{speedup_chart(auto_rows)}</div>
<h2>Adaptive sparse summary</h2>
{summary_table(auto_rows)}
</main>
</body>
</html>
"""
    out_path = out_dir / "high_na_pupil_spectrum_option_matrix_report.html"
    out_path.write_text(html_text, encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
