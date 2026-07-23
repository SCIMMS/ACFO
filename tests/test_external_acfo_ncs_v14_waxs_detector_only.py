from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from scripts.run_external_acfo_ncs_validation_v14 import build_step_plan
from scripts.run_external_acfo_ncs_v14_waxs_detector_only import (
    STEP_LABEL,
    amend_step_records,
    build_detector_command,
)


ROOT = Path(__file__).resolve().parents[1]


def test_full_runner_freezes_detector_nphi_2250(tmp_path: Path) -> None:
    steps = build_step_plan("python", tmp_path, mode="full")
    detector = next(step for step in steps if step.label == STEP_LABEL)
    index = detector.command.index("--nphi-min")
    assert detector.command[index + 1] == "2250"


def test_supplement_command_changes_only_the_frozen_detector_sampling(
    tmp_path: Path,
) -> None:
    command = build_detector_command(
        tmp_path / ".venv/Scripts/python.exe",
        tmp_path,
        tmp_path / "amended/waxs_detector_nq512_abba.json",
    )
    index = command.index("--nphi-min")
    assert command[index + 1] == "2250"
    assert command[command.index("--nq") + 1] == "512"
    assert command[command.index("--warmups") + 1] == "10"
    assert command[command.index("--repeats") + 1] == "30"
    assert command[command.index("--timing-order") + 1] == "alternating"


def test_amend_step_records_replaces_only_detector_row() -> None:
    payload = {
        "schema": "steps",
        "steps": [
            {"label": "environment", "passed": True},
            {"label": STEP_LABEL, "passed": True, "old": True},
            {"label": "odt_temporal_warm_start", "passed": True},
        ],
    }
    replacement = {"label": STEP_LABEL, "passed": True, "new": True}
    amended, original = amend_step_records(payload, replacement)
    assert original["old"] is True
    assert amended["steps"][0] == payload["steps"][0]
    assert amended["steps"][1] == replacement
    assert amended["steps"][2] == payload["steps"][2]


def test_amend_step_records_rejects_missing_or_duplicate_detector_rows() -> None:
    with pytest.raises(ValueError):
        amend_step_records({"steps": []}, {"label": STEP_LABEL})
    with pytest.raises(ValueError):
        amend_step_records(
            {"steps": [{"label": STEP_LABEL}, {"label": STEP_LABEL}]},
            {"label": STEP_LABEL},
        )


def test_supplement_builder_emits_hash_manifest() -> None:
    from scripts import build_acfo_ncs_v14_waxs_detector_supplement as builder

    receipt = builder.build()
    output = Path(receipt["output"])
    assert output.is_file()
    with zipfile.ZipFile(output) as archive:
        prefix = builder.PREFIX + "/"
        manifest = json.loads(archive.read(prefix + "MANIFEST.json"))
        names = {row["path"] for row in manifest["files"]}
        assert names == set(builder.FILES)
        for row in manifest["files"]:
            payload = archive.read(prefix + row["path"])
            assert len(payload) == row["bytes"]
            assert hashlib.sha256(payload).hexdigest() == row["sha256"]
