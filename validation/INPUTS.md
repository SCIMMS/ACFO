# External and generated input data

The compact validation release does not duplicate large generated arrays or
third-party raw measurements. This file records how the omitted inputs are
obtained and the hashes used in the frozen runs.

## WAXS protein nanocrystal inputs

The protein-crystal arrays are deterministic derivatives of RCSB PDB entry
`1IEE`. Recreate the three frozen inputs from the repository root:

```powershell
python scripts/prepare_protein_nanocrystal_waxs.py --pdb-id 1IEE --supercell 1,1,1 --output structures/processed/protein_nanocrystal_lysozyme_1iee_1x1x1_fixed.npz
python scripts/prepare_protein_nanocrystal_waxs.py --pdb-id 1IEE --supercell 3,3,3 --output structures/processed/protein_nanocrystal_lysozyme_1iee_3x3x3_fixed.npz
python scripts/prepare_protein_nanocrystal_waxs.py --pdb-id 1IEE --supercell 5,5,5 --output structures/processed/protein_nanocrystal_lysozyme_1iee_5x5x5_fixed.npz
```

Frozen input hashes:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `protein_nanocrystal_lysozyme_1iee_1x1x1_fixed.npz` | 337,672 | `bbb52722463406b0c94f061932c1986edf44fe5902a148347b19bf01a465eb6c` |
| `protein_nanocrystal_lysozyme_1iee_3x3x3_fixed.npz` | 5,130,031 | `963a7c033cddc6661f1694c6586a2a65bafb6826639244696bc60e0bba050562` |
| `protein_nanocrystal_lysozyme_1iee_5x5x5_fixed.npz` | 23,079,492 | `64a66f70237c09939b96b216fa9bff34fa730c22f039d58a2801be9a4d02fffe` |

The script records the RCSB source URL, crystallographic expansion,
orientation, filtering, and atom counts in an adjacent JSON sidecar.

## Public aIDT input

The local contract was converted from the public Diatom I intensity stack
linked by `bu-cisl/IDT-using-Annular-Illumination`. The repository code is
BSD-3-Clause, but a separate raw-data license was not found. The 47 MB converted
array is therefore not redistributed here.

Use `scripts/convert_aidt_annular_to_contract.py` after downloading
`IRaw_Diatom_I.mat` and `Sorted_Pos.mat` from the public source. The frozen
contract is:

- file: `aidt_diatom_public_contract.npz`
- bytes: `47,061,848`
- SHA-256: `e60c79eb6d49edcc38031c42d78ddca819ab7817d9aa5101fe0743a40ccd53da`
- shape: `24 × 700 × 700`
- dtype: `float32`

Its validation receipt and human-readable conversion summary are included in
`odt_aidt/`.

## Raw independent-run bundles

Machine-local raw return bundles are not committed because they contain local
filesystem paths and device identifiers. Their immutable hashes and curated
scientific receipts are retained in this release:

- external RTX 3090 return:
  `1a558be711da3b4fe1623e3a1a31fb013fd82cb783c7e235a8a59a49e9acdca4`
- independent Intel Core i5-12400F return:
  `be7f80a2b1477cf8baa42fcb4ea30b5f896e825dd3838f2a800d2dec5c68d61f`
