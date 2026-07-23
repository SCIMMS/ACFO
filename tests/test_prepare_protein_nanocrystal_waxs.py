from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.prepare_protein_nanocrystal_waxs import (
    build_supercell,
    euler_rotation_zyx,
    expand_unit_cell,
    parse_pdb_crystal,
    parse_supercell,
)
from scripts.prepare_public_waxs_cif_structures import cell_matrix


PDB_FIXTURE = """\
CRYST1   10.000   10.000   10.000  90.00  90.00  90.00 P 1           2
REMARK 290   SMTRY1   1  1.000000  0.000000  0.000000        0.00000
REMARK 290   SMTRY2   1  0.000000  1.000000  0.000000        0.00000
REMARK 290   SMTRY3   1  0.000000  0.000000  1.000000        0.00000
REMARK 290   SMTRY1   2  1.000000  0.000000  0.000000        5.00000
REMARK 290   SMTRY2   2  0.000000  1.000000  0.000000        0.00000
REMARK 290   SMTRY3   2  0.000000  0.000000  1.000000        0.00000
ATOM      1  CA  GLY A   1       1.000   2.000   3.000  1.00 10.00           C
END
"""


def write_fixture(path: Path) -> Path:
    path.write_text(PDB_FIXTURE, encoding="utf-8")
    return path


def test_parse_supercell_accepts_scalar_and_triplet() -> None:
    assert parse_supercell("3") == (3, 3, 3)
    assert parse_supercell("2x3x4") == (2, 3, 4)


def test_crystal_symmetry_and_supercell_expansion(tmp_path: Path) -> None:
    crystal = parse_pdb_crystal(
        write_fixture(tmp_path / "TEST.pdb"),
        pdb_id="TEST",
        include_hetatm=False,
        include_waters=False,
        include_hydrogen=False,
    )
    elements, fractional = expand_unit_cell(crystal)

    assert crystal.space_group == "P 1"
    assert len(crystal.symmetry_rotations) == 2
    assert elements.tolist() == ["C", "C"]
    np.testing.assert_allclose(
        fractional,
        np.asarray([[0.1, 0.2, 0.3], [0.6, 0.2, 0.3]]),
        atol=1e-12,
    )

    cell = cell_matrix(crystal.cell_lengths_angstrom, crystal.cell_angles_deg)
    expanded_elements, coords = build_supercell(elements, fractional, cell, (2, 1, 1))
    assert expanded_elements.size == 4
    np.testing.assert_allclose(np.sort(coords[:, 0]), [1.0, 6.0, 11.0, 16.0], atol=1e-12)


def test_euler_rotation_is_proper_orthogonal() -> None:
    rotation = euler_rotation_zyx((17.0, 31.0, 43.0))
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-12)
