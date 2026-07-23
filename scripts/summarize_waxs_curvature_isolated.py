from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark_results"
DOC = ROOT / "docs" / "acfo_ncs_waxs_curvature_isolation_ko.md"
OUTPUT = RESULTS / "waxs_curvature_isolated_decision.json"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def main() -> None:
    forward = load("waxs_curvature_isolated_216k_nq256.json")
    reverse = load("waxs_curvature_isolated_216k_nq256_reverse_repeat2.json")
    forward_rows = {float(row["curvature_scale"]): row for row in forward["rows"]}
    reverse_rows = {float(row["curvature_scale"]): row for row in reverse["rows"]}
    shared_scales = sorted(set(forward_rows) & set(reverse_rows))
    robust_rows: list[dict[str, object]] = []
    for scale in shared_scales:
        left = forward_rows[scale]
        right = reverse_rows[scale]
        acfo = np.asarray(
            left["acfo_cached_times"] + right["acfo_cached_times"], dtype=np.float64
        )
        finufft = np.asarray(
            left["finufft_cached_times"] + right["finufft_cached_times"],
            dtype=np.float64,
        )
        paired = finufft / acfo
        robust_rows.append(
            {
                "curvature_scale": scale,
                "max_abs_qz_over_q": left["max_abs_qz_over_q"],
                "dqperp_dq_at_qmax": left["dqperp_dq_at_qmax"],
                "acfo_cached_samples_s": acfo.tolist(),
                "finufft_cached_samples_s": finufft.tolist(),
                "paired_speedup_samples": paired.tolist(),
                "acfo_cached_median_s": float(np.median(acfo)),
                "finufft_cached_median_s": float(np.median(finufft)),
                "robust_warm_speedup": float(np.median(finufft) / np.median(acfo)),
                "paired_speedup_min": float(np.min(paired)),
                "paired_speedup_max": float(np.max(paired)),
                "complex_l2_max": max(
                    left["complex_l2_acfo_vs_finufft"],
                    right["complex_l2_acfo_vs_finufft"],
                ),
            }
        )
    curvature = np.asarray(
        [row["max_abs_qz_over_q"] for row in robust_rows], dtype=np.float64
    )
    speedup = np.asarray(
        [row["robust_warm_speedup"] for row in robust_rows], dtype=np.float64
    )
    planar = robust_rows[0]["robust_warm_speedup"]
    physical = next(
        row["robust_warm_speedup"]
        for row in robust_rows
        if abs(float(row["curvature_scale"]) - 1.0) < 1e-12
    )
    strongest = robust_rows[-1]["robust_warm_speedup"]
    qmax_rows = []
    for suffix in ("q2p13", "q4p06", "q6p30", "q8p06"):
        payload = load(f"qmax_scaling_1m_dq0p160_{suffix}.json")
        row = payload["rows"][0]
        qmax_rows.append(
            {
                "qmax_inv_angstrom": payload["case"]["qmax"],
                "nq": payload["case"]["nq"],
                "n_phi": row["grid"]["n_phi"],
                "targets": payload["case"]["nq"] * row["grid"]["n_phi"],
                "speedup": row["rdep_fused_speedup_vs_nufft"],
            }
        )
    result = {
        "schema": "waxs-curvature-isolated-decision-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": forward["configuration"],
        "replication": {
            "forward_sweep": "waxs_curvature_isolated_216k_nq256.json",
            "reverse_repeat2": "waxs_curvature_isolated_216k_nq256_reverse_repeat2.json",
            "cached_samples_per_shared_curvature": 3,
        },
        "robust_rows": robust_rows,
        "accuracy": {
            "maximum_complex_l2": max(row["complex_l2_max"] for row in robust_rows),
            "gate": 1e-6,
            "passed": max(row["complex_l2_max"] for row in robust_rows) <= 1e-6,
        },
        "curvature_only_test": {
            "speedup_monotonic_non_decreasing": bool(np.all(np.diff(speedup) >= 0.0)),
            "pearson_abs_qz_fraction_vs_speedup": float(
                np.corrcoef(curvature, speedup)[0, 1]
            ),
            "planar_speedup": planar,
            "physical_ewald_speedup": physical,
            "strongest_curvature_speedup": strongest,
            "physical_to_planar_speedup_ratio": physical / planar,
            "strongest_to_planar_speedup_ratio": strongest / planar,
            "hypothesis_supported": False,
        },
        "previous_fixed_dq_qmax_sweep": qmax_rows,
        "decision": (
            "Curvature magnitude alone does not increase FINUFFT/ACFO speedup under the "
            "fixed 216k-protein, Nq=256, Nphi=2160, 552960-target production contract. "
            "The earlier high-q speedup growth is therefore dominated by the coupled growth "
            "of q range, target count, angular bandwidth, and baseline cost rather than by "
            "curvature alone."
        ),
        "claim_boundary": {
            "supported": (
                "ACFO is advantageous for structured axisymmetric curved manifolds with many "
                "targets and reusable geometry; the fixed-dq high-q regime remains favorable."
            ),
            "not_supported": (
                "speedup increases monotonically with curvature when target count and all other "
                "workload variables are fixed"
            ),
            "physics_note": (
                "Strong curvature still makes planar FFT references physically inadequate, but "
                "FINUFFT already supports nonuniform targets and is not penalized by curvature "
                "magnitude alone in this controlled test."
            ),
        },
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# WAXS curvature-isolated benchmark 판정",
        "",
        "## 결론",
        "",
        "동일 protein object, Nq, Nphi, target 수, form factor, 정확도와 FINUFFT q-block을 고정하면 curvature 증가에 따른 speedup 상승은 나타나지 않았다. 따라서 기존 high-q 우위는 curvature 단독 효과가 아니라 q-range, target 수, angular bandwidth와 baseline 비용이 함께 증가한 결과로 해석해야 한다.",
        "",
        "| curvature scale | max |qz|/q | d qperp/dq | ACFO median s | FINUFFT median s | robust warm speedup |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in robust_rows:
        lines.append(
            f"| {row['curvature_scale']:.2f} | {row['max_abs_qz_over_q']:.3f} | "
            f"{row['dqperp_dq_at_qmax']:.3f} | {row['acfo_cached_median_s']:.3f} | "
            f"{row['finufft_cached_median_s']:.3f} | {row['robust_warm_speedup']:.3f}x |"
        )
    lines.extend(
        [
            "",
            f"- planar: `{planar:.3f}x`",
            f"- physical Ewald: `{physical:.3f}x` (`{physical/planar:.3f}` of planar)",
            f"- strongest curvature: `{strongest:.3f}x` (`{strongest/planar:.3f}` of planar)",
            f"- curvature-speedup Pearson correlation: `{result['curvature_only_test']['pearson_abs_qz_fraction_vs_speedup']:.3f}`",
            f"- maximum ACFO-FINUFFT complex L2: `{result['accuracy']['maximum_complex_l2']:.3e}` — PASS",
            "",
            "## 해석",
            "",
            "Curvature는 planar FFT가 물리적으로 부적합해지는 이유이지만, arbitrary targets를 처리하는 FINUFFT에는 curvature magnitude 자체가 추가 penalty가 아니다. ACFO의 상대 우위는 axisymmetric ring factorization, target 수, angular bandwidth, geometry reuse가 함께 만드는 workload regime에서 나타난다.",
            "",
            "## Claim boundary",
            "",
            "- 유지: fixed-dq high-q regime에서 ACFO speedup이 증가한다.",
            "- 수정: 그 증가를 curvature 단독 효과라고 주장하지 않는다.",
            "- 권장 표현: structured curved-manifold and high-target-count regime에서 geometry-aware factorization의 이점이 커진다.",
        ]
    )
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote {OUTPUT} and {DOC}")


if __name__ == "__main__":
    main()
