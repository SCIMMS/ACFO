from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark_results"
DOCS = ROOT / "docs"
SUPPORT = DOCS / "waxs_aidt_odt_progress_report_20260713_support"
ARTIFACT = SUPPORT / "artifact.json"
REPORT_DATA = SUPPORT / "report_data.json"
REPORT_SQL = SUPPORT / "report_data.sql"
REPORT_DB = SUPPORT / "report_data.sqlite"
SOURCE_NOTES = SUPPORT / "source_notes.md"


def load(name: str) -> dict[str, Any]:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def source(
    source_id: str,
    label: str,
    path: str,
    description: str,
    files: list[str],
    generated_at: str,
    metric_definitions: list[str] | None = None,
    sql: str | None = None,
) -> dict[str, Any]:
    query = {
        "engine": "local benchmark artifacts",
        "description": description,
        "language": "JSON and Markdown",
        "executed_at": generated_at,
        "tables_used": files,
        "filters": [
            "Only saved benchmark artifacts inspected for this report are included.",
            "Local timings use the RTX 2070 SUPER unless a row states otherwise.",
            "Measured results and projections are kept separate.",
        ],
        "metric_definitions": metric_definitions or [],
    }
    if sql is not None:
        query["sql"] = sql
        query["language"] = "SQL (SQLite)"
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": query,
    }


def main() -> None:
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    SUPPORT.mkdir(parents=True, exist_ok=True)

    direct = load("waxs_direct_reference_sweep.json")
    exact_beta = load("waxs_exact_beta_harmonic_bridge.json")
    detector = load("waxs_detector_aware_decision.json")
    q_sampling = load("protein_lattice_q_sampling_decision.json")
    curvature = load("waxs_curvature_isolated_decision.json")
    high_q_case = next(row for row in direct["cases"] if row["name"] == "high_q_physical")
    direct_cases = [row for row in direct["cases"] if row["purpose"] in {"q-band", "q-band/anchor", "curvature"}]
    production_errors = [row["acfo_vs_direct_binned"]["complex_l2"] for row in direct_cases]
    full_harmonic_errors = [
        row["acfo_full_harmonic_vs_direct_binned"]["complex_l2"] for row in direct_cases
    ]
    detector_512 = next(row for row in detector["rows"] if row["nq"] == 512)
    q_rows = q_sampling["fixed_dq_range_sweep"]["rows"]
    resolution_rows = q_sampling["fixed_range_resolution_sweep"]["rows"]
    curvature_rows = curvature["robust_rows"]
    physical_curvature = next(row for row in curvature_rows if row["curvature_scale"] == 1.0)
    planar_curvature = next(row for row in curvature_rows if row["curvature_scale"] == 0.0)
    exact_beta_fine = exact_beta["rz_quantization_sweep"][-1]

    aidt_hot = load("aidt_10hz_full700_opt_repeat.json")["summary"]
    aidt_stats = load("aidt_10hz_full700_with_stats.json")["summary"]
    aidt_no_cache = load("aidt_10hz_full700_support_no_transfer_cache.json")["summary"]
    aidt_z18 = load("aidt_10hz_full700_zstep3_opt.json")["summary"]
    aidt_z18_stats = load("aidt_10hz_full700_zstep3_with_stats.json")["summary"]
    aidt_512 = load("aidt_10hz_crop512_fullz_opt.json")["summary"]
    aidt_dense = load("aidt_public_transfer_torch_gpu_optimized_full700_compare.json")
    aidt_adapter = load("aidt_public_measured_adapter_large.json")
    aidt_update = load("aidt_public_measured_update_loop_large.json")

    odt_validation = load("odt_torch_reduced_contraction_256cubed.json")
    odt_legacy = load("odt_torch_256cubed_100pair.json")
    odt_optimized = load("odt_torch_256cubed_reduced_100pair.json")
    odt_cufinufft = load("odt_cufinufft_256cubed_100pair.json")
    odt_regime = load("odt_resident_stream_regime_summary_20260713.json")
    odt_128_gate = load("odt_128cubed_gate_decision.json")

    aidt_full_s = float(aidt_hot["gpu_run_median_s"])
    aidt_full_hz = 1.0 / aidt_full_s
    odt_legacy_pair = float(odt_legacy["pair_timing"]["median_s"])
    odt_optimized_pair = float(odt_optimized["pair_timing"]["median_s"])
    odt_cufinufft_pair = float(odt_cufinufft["pair_timing"]["median_s"])
    odt_vs_legacy = odt_legacy_pair / odt_optimized_pair
    odt_vs_cufinufft = odt_cufinufft_pair / odt_optimized_pair

    summary_rows = [{
        "waxs_detector_speedup": detector_512["warm_speedup"],
        "waxs_operator_error_ppm": max(production_errors) * 1e6,
        "aidt_full_hz": aidt_full_hz,
        "odt256_pair_rate": 1.0 / odt_optimized_pair,
    }]

    waxs_fixed_dq = [
        {
            "q_label": f"qmax {row['q_max_inv_angstrom']:.2f} Å⁻¹",
            "qmax": row["q_max_inv_angstrom"],
            "nq": row["nq"],
            "nphi": row["nphi"],
            "target_count": row["target_count"],
            "acfo_first_total_s": row["factorized_first_total_seconds"],
            "finufft_chunked_s": row["chunked_finufft_streamed_wall_seconds"],
            "speedup": row["first_total_speedup"],
            "complex_l2": row["cross_complex_l2"],
            "regime": "1M repeated-crystal, full-ring, fixed-dq, q-block=2",
        }
        for row in q_rows
    ]

    aidt_conditions = [
        {
            "condition": "700×700×35 GPU-resident core",
            "seconds": aidt_full_s,
            "hz": 1.0 / aidt_full_s,
            "copy_included": "no",
            "diagnostics": "off",
            "claim": "10 Hz core PASS",
        },
        {
            "condition": "700×700×35 + output diagnostics",
            "seconds": float(aidt_stats["gpu_run_median_s"]),
            "hz": 1.0 / float(aidt_stats["gpu_run_median_s"]),
            "copy_included": "no",
            "diagnostics": "on",
            "claim": "near 10 Hz",
        },
        {
            "condition": "700×700×35 + sequential H2D",
            "seconds": 0.1067,
            "hz": 9.37,
            "copy_included": "yes",
            "diagnostics": "off",
            "claim": "end-to-end not established",
        },
        {
            "condition": "700×700×18 + H2D",
            "seconds": 0.0625,
            "hz": 15.99,
            "copy_included": "yes",
            "diagnostics": "off",
            "claim": "10 Hz margin",
        },
        {
            "condition": "512×512×35 + H2D",
            "seconds": 0.0556,
            "hz": 18.00,
            "copy_included": "yes",
            "diagnostics": "off",
            "claim": "10 Hz margin",
        },
        {
            "condition": "700×700×35, no transfer cache",
            "seconds": float(aidt_no_cache["gpu_run_median_s"]),
            "hz": 1.0 / float(aidt_no_cache["gpu_run_median_s"]),
            "copy_included": "no",
            "diagnostics": "off",
            "claim": "cache required",
        },
    ]

    q_heavy = odt_regime["q_heavy_fit_case"]
    cube128 = odt_regime["object_heavy_fit_case_128cubed"]
    odt_speedups = [
        {
            "comparison": "q-heavy: resident vs forced stream",
            "speedup": q_heavy["resident_speedup_over_stream"],
            "selected_path": "resident",
            "baseline": "forced stream",
            "object_bins": q_heavy["problem"]["object_bins"],
            "q_samples": q_heavy["problem"]["q_samples"],
        },
        {
            "comparison": "128³: stream vs resident",
            "speedup": cube128["stream_speedup_over_resident"],
            "selected_path": "stream-reduced",
            "baseline": "resident",
            "object_bins": cube128["problem"]["object_bins"],
            "q_samples": cube128["problem"]["q_samples"],
        },
        {
            "comparison": "256³: optimized vs legacy ACFO",
            "speedup": odt_vs_legacy,
            "selected_path": "illumination-reduced",
            "baseline": "legacy streaming",
            "object_bins": odt_optimized["object_bins"],
            "q_samples": odt_optimized["q_samples"],
        },
        {
            "comparison": "256³: optimized vs cuFINUFFT",
            "speedup": odt_vs_cufinufft,
            "selected_path": "illumination-reduced",
            "baseline": "cuFINUFFT reusable type-3",
            "object_bins": odt_optimized["object_bins"],
            "q_samples": odt_optimized["q_samples"],
        },
    ]

    waxs_evidence = [
        {
            "item": "1. Same-source operator",
            "condition": "8,008-atom 1IEE; direct complex128 NDFT",
            "result": f"full harmonic {min(full_harmonic_errors):.2e}–{max(full_harmonic_errors):.2e}; production {min(production_errors):.2e}–{max(production_errors):.2e}",
            "status": "PASS",
        },
        {
            "item": "2. High-q atomistic representation",
            "condition": "q=5.0–6.3 Å⁻¹; 0.1 nm / source Nphi=750",
            "result": f"pixel intensity {high_q_case['exact_atom_reference']['acfo_vs_exact_atom']['intensity_l2']*100:.2f}%; ring {high_q_case['exact_atom_reference']['acfo_vs_exact_atom']['ring_intensity_l2']*100:.2f}%",
            "status": "FAIL",
        },
        {
            "item": "3. Exact-beta bridge",
            "condition": "detector Nphi=720; streamed atom chunk 256",
            "result": f"direct complex L2 {exact_beta['exact_coordinate_harmonic']['complex_l2']:.2e}; fine-Rz pixel {exact_beta_fine['intensity_l2']*100:.3f}%",
            "status": "PASS on unit-cell reference",
        },
        {
            "item": "4. Detector-aware timing",
            "condition": "EIGER2 X 4M envelope; Nq=512; 10 warm-up + 30 AB/BA",
            "result": f"ACFO {detector_512['acfo_cached_s']:.2f} s; FINUFFT {detector_512['finufft_cached_s']:.2f} s; {detector_512['warm_speedup']:.3f}×; memory {detector_512['memory_reduction_ratio']:.2f}×",
            "status": "local PASS; external pending",
        },
        {
            "item": "5. Curvature isolation",
            "condition": "same object, targets, Nq/Nphi and q-block",
            "result": f"planar {planar_curvature['robust_warm_speedup']:.3f}×; physical {physical_curvature['robust_warm_speedup']:.3f}×; correlation {curvature['curvature_only_test']['pearson_abs_qz_fraction_vs_speedup']:.3f}",
            "status": "curvature-only claim rejected",
        },
    ]

    aidt_evidence = [
        {
            "condition": row["condition"],
            "seconds": row["seconds"],
            "hz": row["hz"],
            "copy": row["copy_included"],
            "diagnostics": row["diagnostics"],
            "boundary": row["claim"],
        }
        for row in aidt_conditions
    ]

    frontier = odt_regime["nonresident_case_256cubed"]["stream_frontier_illumination_block_4"]
    odt_regimes = [
        {
            "regime": "q-heavy fit",
            "object_bins": q_heavy["problem"]["object_bins"],
            "q_samples": q_heavy["problem"]["q_samples"],
            "q_per_object": q_heavy["problem"]["q_per_object"],
            "resident_pair_s": q_heavy["resident"]["median_pair_s"],
            "stream_pair_s": q_heavy["forced_stream"]["median_pair_s"],
            "peak_mib": q_heavy["resident"]["peak_allocated_mib"],
            "decision": "resident; 2.53× faster",
        },
        {
            "regime": "128³ object-heavy fit",
            "object_bins": cube128["problem"]["object_bins"],
            "q_samples": cube128["problem"]["q_samples"],
            "q_per_object": cube128["problem"]["q_per_object"],
            "resident_pair_s": cube128["resident"]["median_pair_s"],
            "stream_pair_s": cube128["stream"]["median_pair_s"],
            "peak_mib": cube128["stream"]["peak_allocated_mib"],
            "decision": "stream; 9.87× faster, 12.66× less allocated",
        },
        {
            "regime": "256³ nonresident",
            "object_bins": odt_regime["nonresident_case_256cubed"]["problem"]["object_bins"],
            "q_samples": odt_regime["nonresident_case_256cubed"]["problem"]["q_samples"],
            "q_per_object": odt_regime["nonresident_case_256cubed"]["problem"]["q_samples"] / odt_regime["nonresident_case_256cubed"]["problem"]["object_bins"],
            "resident_pair_s": None,
            "stream_pair_s": frontier["16"]["median_pair_s"],
            "peak_mib": frontier["16"]["peak_allocated_mib"],
            "decision": "resident OOM (9.12 GiB allocation); r16/i4 default",
        },
    ]

    validation_status = [
        {
            "domain": "WAXS operator numerics",
            "validated": "direct NDFT; full harmonic 2.5e-15–6.7e-14; production ≤9.11e-7",
            "remaining": "external timing replication",
            "not_claimed": "coarse-bin high-q atomistic cake accuracy",
        },
        {
            "domain": "WAXS atomistic source",
            "validated": "exact-beta unit-cell bridge 1.01e-12; fine-Rz pixel 0.779%",
            "remaining": "dense disorder / MD production contraction",
            "not_claimed": "96k detector output requirement",
        },
        {
            "domain": "aIDT",
            "validated": "700×700×35 GPU-resident core 10.31 Hz",
            "remaining": "GPU preprocessing, transfer overlap, acquisition scheduling",
            "not_claimed": "10 Hz end-to-end live microscope",
        },
        {
            "domain": "ODT operator",
            "validated": "128³ direct subset; 256³ stream 0.444 s/pair; resident OOM gate",
            "remaining": "q/object crossover and independent machine",
            "not_claimed": "all-regime resident superiority or general 10 Hz ODT",
        },
        {
            "domain": "ODT physical inverse",
            "validated": "in-range numerical control NRMSE 0.997%",
            "remaining": "missing-cone-aware prior and true multi-axis acquisition",
            "not_claimed": "arbitrary bead/object recovery",
        },
    ]

    datasets = {
        "summary": summary_rows,
        "waxs_fixed_dq": waxs_fixed_dq,
        "aidt_conditions": aidt_conditions,
        "odt_speedups": odt_speedups,
        "waxs_evidence": waxs_evidence,
        "aidt_evidence": aidt_evidence,
        "odt_regimes": odt_regimes,
        "validation_status": validation_status,
    }

    report_query = (
        "SELECT dataset_id, row_index, row_json\n"
        "FROM report_snapshot_rows\n"
        "ORDER BY dataset_id, row_index;"
    )
    if REPORT_DB.exists():
        REPORT_DB.unlink()
    with sqlite3.connect(REPORT_DB) as connection:
        connection.execute(
            "CREATE TABLE report_snapshot_rows ("
            "dataset_id TEXT NOT NULL, row_index INTEGER NOT NULL, row_json TEXT NOT NULL, "
            "PRIMARY KEY (dataset_id, row_index))"
        )
        for dataset_id, rows in datasets.items():
            connection.executemany(
                "INSERT INTO report_snapshot_rows(dataset_id, row_index, row_json) VALUES (?, ?, ?)",
                [
                    (dataset_id, index, json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                    for index, row in enumerate(rows)
                ],
            )
        selected_rows = connection.execute(report_query).fetchall()
    sql_datasets: dict[str, list[dict[str, Any]]] = {}
    for dataset_id, _row_index, row_json in selected_rows:
        sql_datasets.setdefault(dataset_id, []).append(json.loads(row_json))
    if sql_datasets != datasets:
        raise RuntimeError("SQLite provenance round-trip changed report datasets")
    datasets = sql_datasets
    REPORT_SQL.write_text(report_query + "\n", encoding="utf-8")

    report_data = {
        "schema": "waxs-aidt-odt-progress-report-data-v1",
        "generated_at_utc": generated_at,
        "datasets": datasets,
        "derived_metrics": {
            "waxs_production_max_complex_l2": max(production_errors),
            "waxs_full_harmonic_max_complex_l2": max(full_harmonic_errors),
            "waxs_detector_nq512_speedup": detector_512["warm_speedup"],
            "aidt_full_core_seconds": aidt_full_s,
            "aidt_full_core_hz": aidt_full_hz,
            "odt_256_legacy_pair_s": odt_legacy_pair,
            "odt_256_optimized_pair_s": odt_optimized_pair,
            "odt_256_cufinufft_pair_s": odt_cufinufft_pair,
            "odt_optimized_vs_legacy": odt_vs_legacy,
            "odt_optimized_vs_cufinufft": odt_vs_cufinufft,
        },
    }
    REPORT_DATA.write_text(json.dumps(report_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    sources = [
        source(
            "report_data",
            "Reviewed WAXS, aIDT and ODT benchmark extracts",
            "docs/waxs_aidt_odt_progress_report_20260713_support/report_data.sql",
            "Executed SQLite query that reads the reviewed benchmark extracts used by the report charts, cards and exact tables.",
            ["report_snapshot_rows"],
            generated_at,
            [
                "Speedup is baseline median time divided by ACFO median time under the stated protocol.",
                "Hot timing excludes setup; first-total includes the named setup and first execution.",
                "q/object is the number of detector q samples divided by object coefficients.",
                "Complex L2 is ||candidate-reference||2 / ||reference||2.",
            ],
            report_query,
        ),
        source(
            "waxs_direct",
            "WAXS direct-reference and source-representation validation",
            "docs/acfo_ncs_waxs_direct_reference_ko.md",
            "Reviewed direct NDFT, FINUFFT tolerance, source discretization and exact-beta validation narrative.",
            [
                "benchmark_results/waxs_direct_reference_sweep.json",
                "benchmark_results/waxs_source_discretization_convergence.json",
                "benchmark_results/waxs_exact_beta_harmonic_bridge.json",
            ],
            generated_at,
        ),
        source(
            "aidt_summary",
            "Public aIDT 10 Hz condition summary",
            "benchmark_results/aidt_10hz_condition_summary.md",
            "Reviewed public Diatom I full-condition timings and the GPU-resident versus copy-included boundary.",
            [
                "benchmark_results/aidt_10hz_full700_opt_repeat.json",
                "benchmark_results/aidt_10hz_full700_with_stats.json",
                "benchmark_results/aidt_10hz_full700_support_no_transfer_cache.json",
                "benchmark_results/aidt_10hz_full700_zstep3_opt.json",
                "benchmark_results/aidt_10hz_crop512_fullz_opt.json",
            ],
            generated_at,
        ),
        source(
            "odt_summary",
            "ODT reduced-contraction and resident/stream regime validation",
            "docs/odt_resident_stream_regime_benchmark_ko.md",
            "Reviewed current 256-cubed optimized timing, resident OOM gate, block frontier and 128-cubed crossover evidence.",
            [
                "docs/odt_reduced_contraction_optimization_ko.md",
                "benchmark_results/odt_torch_reduced_contraction_256cubed.json",
                "benchmark_results/odt_torch_256cubed_reduced_100pair.json",
                "benchmark_results/odt_cufinufft_256cubed_100pair.json",
                "benchmark_results/odt_resident_stream_regime_summary_20260713.json",
            ],
            generated_at,
        ),
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "ACFO WAXS·aIDT·ODT 검증 및 성능 진행 보고서",
            "description": "2026년 7월 13일까지 저장된 수치와 최신 RTX 2070 SUPER 재측정을 종합한 기술 보고서",
            "generatedAt": generated_at,
            "cards": [
                {
                    "id": "waxs_detector_speedup",
                    "description": "EIGER2 X 4M detector-aware Nq=512 local AB/BA timing.",
                    "dataset": "summary",
                    "sourceId": "report_data",
                    "metrics": [{"label": "WAXS detector-aware speedup (×)", "field": "waxs_detector_speedup", "format": "number"}],
                },
                {
                    "id": "waxs_operator_error",
                    "description": "Maximum production ACFO complex L2 against same-source direct NDFT in the reviewed sweep.",
                    "dataset": "summary",
                    "sourceId": "report_data",
                    "metrics": [{"label": "WAXS max operator error (ppm)", "field": "waxs_operator_error_ppm", "format": "number"}],
                },
                {
                    "id": "aidt_full_rate",
                    "description": "GPU-resident full public-condition reconstruction core rate.",
                    "dataset": "summary",
                    "sourceId": "report_data",
                    "metrics": [{"label": "aIDT full core (Hz)", "field": "aidt_full_hz", "format": "number"}],
                },
                {
                    "id": "odt256_rate",
                    "description": "Optimized 256-cubed stream-reduced hot forward-adjoint pair rate.",
                    "dataset": "summary",
                    "sourceId": "report_data",
                    "metrics": [{"label": "ODT 256³ hot pair (pair/s)", "field": "odt256_pair_rate", "format": "number"}],
                },
            ],
            "charts": [
                {
                    "id": "waxs_fixed_dq_speedup",
                    "title": "WAXS fixed-dq q-range speedup",
                    "subtitle": "1M repeated-crystal full-ring first-total benchmark; q-block=2 memory-safe FINUFFT baseline.",
                    "type": "horizontalBar",
                    "dataset": "waxs_fixed_dq",
                    "sourceId": "report_data",
                    "encodings": {
                        "x": {"field": "q_label", "type": "ordinal", "label": "q range"},
                        "y": {"field": "speedup", "type": "quantitative", "label": "FINUFFT / ACFO", "unit": "×"},
                        "tooltip": [
                            {"field": "nq", "type": "quantitative", "label": "Nq"},
                            {"field": "nphi", "type": "quantitative", "label": "Nphi"},
                            {"field": "acfo_first_total_s", "type": "quantitative", "label": "ACFO s"},
                            {"field": "finufft_chunked_s", "type": "quantitative", "label": "FINUFFT s"},
                            {"field": "complex_l2", "type": "quantitative", "label": "complex L2"},
                        ],
                    },
                    "yAxisTitle": "Speedup (×)",
                    "valueFormat": "number",
                    "unit": "×",
                    "layout": "full",
                },
                {
                    "id": "aidt_condition_rates",
                    "title": "aIDT update rate by execution condition",
                    "subtitle": "Public Diatom I conditions on RTX 2070 SUPER; 10 Hz is reached only under the stated pipeline boundary.",
                    "type": "horizontalBar",
                    "dataset": "aidt_conditions",
                    "sourceId": "report_data",
                    "encodings": {
                        "x": {"field": "condition", "type": "nominal", "label": "Condition"},
                        "y": {"field": "hz", "type": "quantitative", "label": "Updates per second", "unit": "Hz"},
                        "tooltip": [
                            {"field": "seconds", "type": "quantitative", "label": "Seconds/update"},
                            {"field": "copy_included", "type": "nominal", "label": "H2D included"},
                            {"field": "claim", "type": "nominal", "label": "Boundary"},
                        ],
                    },
                    "yAxisTitle": "Update rate (Hz)",
                    "valueFormat": "number",
                    "unit": "Hz",
                    "layout": "full",
                },
                {
                    "id": "odt_path_speedups",
                    "title": "ODT selected-path speedup by workload regime",
                    "subtitle": "Each bar names its own baseline; values are not a single shared-baseline scaling curve.",
                    "type": "horizontalBar",
                    "dataset": "odt_speedups",
                    "sourceId": "report_data",
                    "encodings": {
                        "x": {"field": "comparison", "type": "nominal", "label": "Comparison"},
                        "y": {"field": "speedup", "type": "quantitative", "label": "Selected path / named baseline", "unit": "×"},
                        "tooltip": [
                            {"field": "selected_path", "type": "nominal", "label": "Selected path"},
                            {"field": "baseline", "type": "nominal", "label": "Baseline"},
                            {"field": "object_bins", "type": "quantitative", "label": "Object bins"},
                            {"field": "q_samples", "type": "quantitative", "label": "q samples"},
                        ],
                    },
                    "yAxisTitle": "Speedup (×)",
                    "valueFormat": "number",
                    "unit": "×",
                    "layout": "full",
                },
            ],
            "tables": [
                {
                    "id": "waxs_evidence_table",
                    "title": "WAXS numerical, representation and detector evidence",
                    "subtitle": "Operator accuracy and atomistic source accuracy are separate gates.",
                    "dataset": "waxs_evidence",
                    "sourceId": "report_data",
                    "defaultSort": {"field": "item", "direction": "asc"},
                    "density": "comfortable",
                    "layout": "full",
                    "columns": [
                        {"field": "item", "label": "Evidence layer", "type": "text"},
                        {"field": "condition", "label": "Condition", "type": "text"},
                        {"field": "result", "label": "Measured result", "type": "text"},
                        {"field": "status", "label": "Decision", "type": "text"},
                    ],
                },
                {
                    "id": "aidt_conditions_table",
                    "title": "aIDT execution-condition timing",
                    "subtitle": "Exact update times and rates for the public-condition variants used in the claim boundary.",
                    "dataset": "aidt_evidence",
                    "sourceId": "report_data",
                    "defaultSort": {"field": "hz", "direction": "desc"},
                    "density": "comfortable",
                    "layout": "full",
                    "columns": [
                        {"field": "condition", "label": "Condition", "type": "text"},
                        {"field": "seconds", "label": "s/update", "type": "number", "format": "number"},
                        {"field": "hz", "label": "Hz", "type": "number", "format": "number"},
                        {"field": "copy", "label": "H2D", "type": "text"},
                        {"field": "diagnostics", "label": "Diagnostics", "type": "text"},
                        {"field": "boundary", "label": "Claim boundary", "type": "text"},
                    ],
                },
                {
                    "id": "odt_regimes_table",
                    "title": "ODT resident and stream regimes",
                    "subtitle": "Same RTX 2070 SUPER; hot pair time excludes setup.",
                    "dataset": "odt_regimes",
                    "sourceId": "report_data",
                    "defaultSort": {"field": "object_bins", "direction": "asc"},
                    "density": "comfortable",
                    "layout": "full",
                    "columns": [
                        {"field": "regime", "label": "Regime", "type": "text"},
                        {"field": "object_bins", "label": "Object bins", "type": "number", "format": "compact"},
                        {"field": "q_samples", "label": "q samples", "type": "number", "format": "compact"},
                        {"field": "q_per_object", "label": "q/object", "type": "number", "format": "number"},
                        {"field": "resident_pair_s", "label": "Resident s", "type": "number", "format": "number"},
                        {"field": "stream_pair_s", "label": "Stream s", "type": "number", "format": "number"},
                        {"field": "peak_mib", "label": "Selected peak MiB", "type": "number", "format": "number"},
                        {"field": "decision", "label": "Decision", "type": "text"},
                    ],
                },
                {
                    "id": "validation_status_table",
                    "title": "Current claim and validation boundary",
                    "subtitle": "Validated evidence, remaining work and statements not supported by the current artifacts.",
                    "dataset": "validation_status",
                    "sourceId": "report_data",
                    "defaultSort": {"field": "domain", "direction": "asc"},
                    "density": "spacious",
                    "layout": "full",
                    "columns": [
                        {"field": "domain", "label": "Domain", "type": "text"},
                        {"field": "validated", "label": "Validated", "type": "text"},
                        {"field": "remaining", "label": "Remaining", "type": "text"},
                        {"field": "not_claimed", "label": "Not supported", "type": "text"},
                    ],
                },
            ],
            "sources": sources,
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# ACFO WAXS·aIDT·ODT 검증 및 성능 진행 보고서"},
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "body": (
                        "## 기술 요약\n\n"
                        "- **WAXS:** 동일 source와 curved target을 사용한 operator 검증은 direct NDFT 대비 full-harmonic `2.5e-15–6.7e-14`, production `1.8e-7–9.1e-7`로 통과했다. 그러나 coarse `0.1 nm / Nphi=750` source는 `q=5.0–6.3 Å⁻¹` exact atoms 대비 pixel intensity `77.65%`, ring `22.55%` 오차로 실패했다. Exact-beta unit-cell bridge는 이 병목을 분리할 수 있음을 보였다.\n"
                        "- **aIDT:** public `24×700×700 → 700×700×35` GPU-resident reconstruction core는 `0.0970 s`, `10.31 Hz`다. 일반 H2D copy를 순차 포함하면 `0.1067 s`, `9.37 Hz`이며 현재 CPU preprocessing `0.1445 s`는 별도다. 따라서 processing-side real-time feasibility는 지지하지만 end-to-end live microscope는 아직 아니다.\n"
                        f"- **ODT:** 256³ legacy pair `11.098 s`를 illumination-reduced contraction으로 `0.444 s`까지 줄였다. 이는 legacy 대비 `{odt_vs_legacy:.2f}×`, 저장된 cuFINUFFT `8.598 s` 대비 `{odt_vs_cufinufft:.2f}×`다. 다만 8 GiB에서 resident는 `9.12 GiB` allocation OOM이며, 최적 경로는 workload regime에 따라 resident와 stream으로 갈린다.\n"
                        "- **종합 판정:** 현재 증거는 fixed-geometry prepared operator의 정확도·재사용 이점을 지지한다. `universal NUFFT replacement`, coarse-bin high-q atomistic correctness, full aIDT end-to-end 10 Hz, 모든 ODT regime의 단일 정책은 지지하지 않는다."
                    ),
                },
                {
                    "id": "definitions",
                    "type": "markdown",
                    "body": (
                        "## 비교 범위와 지표 정의\n\n"
                        "`hot` 시간은 geometry/setup 이후 반복 실행 중앙값이고, `first-total`은 해당 표에 명시된 setup과 첫 실행을 합한 값이다. Speedup은 항상 **표에 적힌 baseline 시간 / ACFO 시간**이다. Complex L2는 `||candidate-reference||₂ / ||reference||₂`이며, WAXS operator error와 exact-atom source representation error는 서로 다른 gate다. ODT의 `q/object`는 detector q sample 수를 object coefficient 수로 나눈 workload 지표다. 별도 언급이 없는 GPU 수치는 RTX 2070 SUPER 8 GiB의 local measurement다."
                    ),
                },
                {"id": "headline_metrics", "type": "metric-strip", "cardIds": ["waxs_detector_speedup", "waxs_operator_error", "aidt_full_rate", "odt256_rate"]},
                {
                    "id": "waxs_accuracy",
                    "type": "markdown",
                    "sourceId": "waxs_direct",
                    "body": (
                        "## WAXS operator는 정확하지만 atomistic source가 현재 high-q gate다\n\n"
                        "Small-case direct NDFT는 FINUFFT tolerance와 무관한 correctness oracle 역할을 했다. Full-harmonic complex128 경로는 모든 q/curvature case에서 direct sum과 `≤6.7e-14`로 일치했고 production R-dependent 경로도 `≤9.11e-7`이었다. 반면 동일 production bins를 exact atoms와 비교하면 high-q pixel intensity가 `77.65%`, ring intensity가 `22.55%` 달랐다. 따라서 same-bin FINUFFT agreement를 end-to-end atomistic accuracy로 해석하지 않는다.\n\n"
                        "Per-atom beta를 harmonic phase에 직접 넣은 exact-beta bridge는 detector Nphi를 `720`으로 유지하면서 direct NDFT complex L2 `1.01e-12`를 달성했다. Fine R,z quantization에서는 pixel intensity `0.779%`, ring `0.105%`였다. 다만 이는 unit-cell reference이며 dense disorder·MD production contraction은 남아 있다."
                    ),
                },
                {"id": "waxs_evidence", "type": "table", "tableId": "waxs_evidence_table", "layout": "full"},
                {
                    "id": "waxs_performance",
                    "type": "markdown",
                    "body": (
                        "## WAXS speedup은 curvature 단독이 아니라 target 수·bandwidth·reuse가 만든다\n\n"
                        "1M repeated-crystal full-ring fixed-dq sweep에서 qmax `2.13 → 8.06 Å⁻¹`로 늘면 first-total speedup은 `47.0× → 147.8×`로 증가했고 cross complex L2는 `6.89e-7–7.57e-7`였다. 반대로 q range를 고정하고 radial resolution만 높이면 hot speedup은 Nq `32/128/512`에서 `314.3×/70.4×/33.48×`로 감소했다. 즉 q resolution 증가 자체는 FINUFFT에 상대적으로 유리하다.\n\n"
                        f"더 현실적인 EIGER2 X 4M partial-arc Nq=512 protocol에서는 ACFO/FINUFFT가 `{detector_512['acfo_cached_s']:.2f}/{detector_512['finufft_cached_s']:.2f} s`, ratio-of-medians `{detector_512['warm_speedup']:.3f}×`, memory ratio `{detector_512['memory_reduction_ratio']:.2f}×`였다. Curvature-isolated physical/planar speedup은 `{physical_curvature['robust_warm_speedup']:.3f}×/{planar_curvature['robust_warm_speedup']:.3f}×`이고 correlation은 `{curvature['curvature_only_test']['pearson_abs_qz_fraction_vs_speedup']:.3f}`이므로 high-q 이득을 curvature alone으로 설명하지 않는다."
                    ),
                },
                {"id": "waxs_chart", "type": "chart", "chartId": "waxs_fixed_dq_speedup", "layout": "full"},
                {
                    "id": "aidt_result",
                    "type": "markdown",
                    "sourceId": "aidt_summary",
                    "body": (
                        "## aIDT는 GPU-resident core에서 10 Hz를 넘었지만 pipeline 전체는 남아 있다\n\n"
                        "Dense GPU baseline `0.446 s`에서 active support `26.4%`를 사용하고 cached support transfer를 적용해 hot core를 `0.0975 s`로 줄였다(`4.58×`). 최종 cache는 약 `2076 MiB`, peak allocated는 약 `3849 MiB`다. 독립 repeat의 full public condition은 `0.0970 s`, `10.31 Hz`였고, output statistics를 포함하면 `0.1012 s`, `9.88 Hz`였다.\n\n"
                        "Full-condition H2D copy `9.7–9.9 ms`를 순차 합치면 `0.1067 s`, `9.37 Hz`다. Core를 `1.075×`만 더 가속하거나 copy와 compute를 overlap하면 10 Hz budget에 들어가지만, 현재 CPU preprocessing `0.1445 s`와 acquisition scheduling은 포함되지 않았다. 따라서 표현은 `GPU-resident prepared reconstruction core의 10 Hz`와 `processing-side feasibility`로 제한한다.\n\n"
                        "별도 measured-data adapter는 prepared pair `1.242 ms` 대 cuFINUFFT `58.036 ms`(`46.7×`), 8-step update는 `1.962 ms/iter` 대 `58.778 ms/iter`(`30.0×`)였지만 calibrated nonlinear full aIDT reconstruction은 아니다. Full transfer 10 Hz 결과와 이 proxy speedup을 하나의 주장으로 합치지 않는다."
                    ),
                },
                {"id": "aidt_chart", "type": "chart", "chartId": "aidt_condition_rates", "layout": "full"},
                {"id": "aidt_table", "type": "table", "tableId": "aidt_conditions_table", "layout": "full"},
                {
                    "id": "odt_result",
                    "type": "markdown",
                    "sourceId": "odt_summary",
                    "body": (
                        "## ODT는 illumination-reduced contraction으로 256³ 병목을 제거했지만 단일 정책은 아니다\n\n"
                        f"256³, 121 illuminations, detector 256²에서 illumination을 먼저 축약하는 matmul/index_add 경로는 forward/adjoint를 `4.187/5.502 s`에서 `0.245/0.206 s`로 줄였다. 100-pair hot median은 legacy `{odt_legacy_pair:.3f} s`, optimized `{odt_optimized_pair:.3f} s`, cuFINUFFT `{odt_cufinufft_pair:.3f} s`다. Optimized-vs-legacy forward/adjoint rel-L2는 `{odt_validation['accuracy']['optimized_forward_rel_l2_vs_legacy']:.3e}/{odt_validation['accuracy']['optimized_adjoint_rel_l2_vs_legacy']:.3e}`이고 dot errors는 `≤3.54e-8`이다.\n\n"
                        "Policy는 memory fit만으로 결정되지 않는다. q/object `71.82`인 q-heavy case에서는 resident가 stream보다 `2.53×` 빠르지만, q/object `0.945`인 128³에서는 stream이 resident보다 `9.87×` 빠르고 allocated memory가 `12.66×` 작다. 256³ resident는 `9.12 GiB` allocation으로 OOM이었고 r16/i4 stream은 `0.4445 s/pair`, `2.25 pair/s`, peak `1513 MiB`, setup `27.15 s`였다.\n\n"
                        "128³ complex64 cross-layout forward 차이 `2.509e-6`은 기존 `2.0e-6` sentinel을 근소하게 넘지만 complex128 64³ 교차검증은 `2.23e-15`였다. Algebraic mismatch보다 accumulation order로 해석하되 strict complex64 gate는 FAIL로 남긴다."
                    ),
                },
                {"id": "odt_chart", "type": "chart", "chartId": "odt_path_speedups", "layout": "full"},
                {"id": "odt_table", "type": "table", "tableId": "odt_regimes_table", "layout": "full"},
                {
                    "id": "robustness",
                    "type": "markdown",
                    "body": (
                        "## 정확도, inverse identifiability와 외부 검증은 분리해 판정한다\n\n"
                        "ODT 128³ complex128 full dot error는 `9.688e-16`, independent subset forward/adjoint L2는 `4.155e-12/3.895e-12`였다. Adjoint-range truth는 NRMSE `0.997%`로 통과했지만 30 dB beads + gradient prior 최선은 `17.11%`였고 illumination angle 70°에서도 `5.65%`였다. 이는 operator 오류와 missing-cone/null-space 문제를 분리해야 함을 보여 준다.\n\n"
                        "현재 publication-level timing의 공통 미완료 항목은 독립 머신 반복이다. WAXS detector Nq=512와 1M prepared Nq=512는 local 10/30 AB/BA gate를 통과했지만 external rerun이 남아 있고, ODT 최신 0.444 s도 같은 GPU의 저장된 cuFINUFFT baseline과 비교한 값이다."
                    ),
                },
                {"id": "validation_table", "type": "table", "tableId": "validation_status_table", "layout": "full"},
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## 다음 단계는 claim을 넓히기보다 남은 gate를 닫는 순서가 효율적이다\n\n"
                        "1. **WAXS source gate:** exact-beta/sub-bin dense-disorder contraction을 실제 protein nanocrystal 또는 MD snapshot에 적용하고 detector-aware mask에서 exact-atom subset을 검증한다.\n"
                        "2. **WAXS timing gate:** Nq=512 detector-aware와 1M prepared AB/BA protocol을 독립 머신에서 재실행한다.\n"
                        "3. **aIDT pipeline gate:** pinned memory/CUDA stream overlap과 GPU preprocessing을 구현해 `700×700×35` copy-included 10 Hz를 직접 측정한다.\n"
                        "4. **ODT dispatcher gate:** q/object `0.945–71.82` 사이의 crossover sweep을 수행하고 CLI에 resident, balanced stream(r16/i4), low-memory stream(r8/i4)을 명시한다.\n"
                        "5. **ODT physical inverse gate:** operator benchmark와 별도로 true second-axis/multi-axis acquisition 및 nonnegative/TV prior를 평가한다."
                    ),
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "body": (
                        "## 남은 결정 질문\n\n"
                        "- WAXS experimental object는 dilute single protein이 아니라 **protein crystal, single crystal 또는 oriented nanocrystal** 중 무엇으로 고정할 것인가? 현재 photon-count argument상 이 정의가 experimental feasibility를 좌우한다.\n"
                        "- WAXS main claim의 대표 timing을 detector-aware 약 `2×`로 둘지, high-q full-ring의 큰 regime speedup을 supplementary regime map으로 둘지 최종 논문 구조에서 확정해야 한다.\n"
                        "- ODT complex64 publication tolerance를 기존 `2e-6`로 유지할지, cross-layout accumulation을 고려한 사전 정의 tolerance와 complex128 sentinel을 병행할지 결정해야 한다.\n"
                        "- aIDT는 실제 acquisition pipeline과 연결할 수 있는 raw-frame preprocessing contract를 확보해야 end-to-end claim으로 이동할 수 있다."
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": datasets,
        },
        "sources": sources,
    }
    ARTIFACT.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    source_notes = f"""# WAXS/aIDT/ODT progress report source notes

## Reporting job

- Question: summarize the current WAXS and aIDT/ODT validation work with concrete numbers in a PDF.
- Audience: technical.
- Decision supported: determine which claims are closed, conditional, failed, or still require external/experimental validation.
- Time frame: saved artifacts through 2026-07-13.
- Baselines: direct complex128 NDFT, FINUFFT/cuFINUFFT under named protocols, legacy ACFO paths, and resident/stream paired conditions.
- Success criterion: every headline number resolves to a saved artifact and every modality keeps its claim boundary visible.

## Required-structure mapping

1. Title -> title block.
2. Technical summary -> technical summary block.
3. Key findings with visual evidence -> WAXS, aIDT and ODT result sections plus three charts.
4. Scope, data and metric definitions -> comparison-scope block placed before headline cards so definitions precede evidence.
5. Methodology -> embedded in each modality section and source metadata; no standalone long methods section because the report is a progress synthesis.
6. Limitations, uncertainty and robustness -> robustness section and validation-boundary table.
7. Recommended next steps -> next-steps section.
8. Further questions -> final decision-questions section.

## Chart map

| section | question | form | rows | fields | supported claim | palette |
|---|---|---|---:|---|---|---|
| WAXS | How does fixed-dq first-total speedup change with qmax? | horizontal bar | 4 | q label, speedup, Nq/Nphi, times, L2 | high-q regime advantage grows in the ideal repeated-crystal full-ring benchmark | single blue root; direct labels/table fallback |
| aIDT | Which execution conditions cross 10 Hz? | horizontal bar | 6 | condition, Hz, seconds, copy, diagnostics | full GPU-resident core crosses 10 Hz; sequential copy does not | single blue root; exact timing table |
| ODT | Which selected path wins in each measured regime? | horizontal bar | 4 | named comparison, speedup, workload | resident/stream choice is workload-dependent; optimized 256-cubed reverses the legacy result | single blue root; each row names its baseline |

Curvature-isolation, direct-reference errors, and block-size frontier remain tables/narrative because their exact lookup and caveat structure is more important than visual shape. A scatter for the four ODT block points was rejected as underpowered.

## Source inventory and transformations

- WAXS direct/source: `benchmark_results/waxs_direct_reference_sweep.json`, `benchmark_results/waxs_source_discretization_convergence.json`, `benchmark_results/waxs_exact_beta_harmonic_bridge.json`.
- WAXS performance: `benchmark_results/waxs_detector_aware_decision.json`, `benchmark_results/protein_lattice_q_sampling_decision.json`, `benchmark_results/waxs_curvature_isolated_decision.json`.
- aIDT: `benchmark_results/aidt_10hz_*`, `benchmark_results/aidt_public_measured_adapter_large.json`, `benchmark_results/aidt_public_measured_update_loop_large.json`.
- ODT: `benchmark_results/odt_torch_reduced_contraction_256cubed.json`, `benchmark_results/odt_torch_256cubed_reduced_100pair.json`, `benchmark_results/odt_cufinufft_256cubed_100pair.json`, `benchmark_results/odt_resident_stream_regime_summary_20260713.json`, `benchmark_results/odt_128cubed_gate_decision.json`.
- Derived rates use `1 / median seconds`; speedups use named baseline median divided by selected-path median. No extrapolated FINUFFT timeout is used as an exact timing.
- Reviewed rows are staged in `report_data.sqlite`; `report_data.sql` is executed and its ordered JSON rows are round-tripped into the artifact datasets. `report_data.json` is the human-readable processed dataset; `artifact.json` is the canonical report input.

## Known caveats

- WAXS fixed-dq speedup chart is a 1M repeated-crystal, full-ring, q-block=2 benchmark and is not presented as the detector-aware headline.
- aIDT full transfer 10 Hz and the prepared measured-update/cuFINUFFT speedups are separate workloads.
- ODT latest cuFINUFFT comparison reuses the stored same-GPU 100-pair baseline; external replication is pending.
- Complex64 128-cubed cross-layout forward L2 misses the historical 2e-6 sentinel; the report does not relabel it as a pass.
"""
    SOURCE_NOTES.write_text(source_notes, encoding="utf-8")
    print(json.dumps({
        "artifact": str(ARTIFACT),
        "report_data": str(REPORT_DATA),
        "source_notes": str(SOURCE_NOTES),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
