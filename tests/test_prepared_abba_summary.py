from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_abba_distribution_reports_tail_and_cv() -> None:
    module = load_module(
        "prepared_abba_driver_for_test",
        "scripts/benchmark_protein_lattice_finufft_512_abba.py",
    )
    summary = module.distribution([1.0, 2.0, 3.0, 4.0])
    assert summary["count"] == 4
    assert summary["median"] == 2.5
    assert summary["p05"] < summary["median"] < summary["p95"]
    assert summary["cv"] > 0.0


def test_abba_order_summary_and_relative_gap() -> None:
    module = load_module(
        "prepared_abba_summary_for_test",
        "scripts/summarize_protein_lattice_prepared_abba.py",
    )
    rows = [
        {
            "order": "AB",
            "factorized_seconds": 3.0,
            "finufft_seconds": 99.0,
            "paired_speedup": 33.0,
        },
        {
            "order": "AB",
            "factorized_seconds": 3.2,
            "finufft_seconds": 102.4,
            "paired_speedup": 32.0,
        },
        {
            "order": "BA",
            "factorized_seconds": 3.1,
            "finufft_seconds": 102.3,
            "paired_speedup": 33.0,
        },
    ]
    ab = module.order_summary(rows, "AB")
    ba = module.order_summary(rows, "BA")
    assert ab["count"] == 2
    assert ba["count"] == 1
    assert module.relative_median_gap(
        ab["paired_speedup_median"], ba["paired_speedup_median"]
    ) < 0.02
