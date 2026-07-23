# WAXS Structure Inputs

This directory is reserved for downloaded or generated atomistic structure inputs
used by WAXS validation scripts.

The benchmark-facing contract is `structures/processed/*.npz` with at least:

- `coords`: `(n_atoms, 3)` float64 coordinates in nm
- `elements`: per-atom element symbols
- `atomic_numbers`: optional int16 atomic numbers
- `structure_id`: string scalar
- `metadata_json`: compact JSON provenance

Raw downloads, processed NPZ/XYZ files, and metadata sidecars are generated
artifacts and are ignored by git. Recreate the initial public protein set with:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_public_waxs_structures.py --pdb-ids 1CRN,1UBQ --write-xyz
```

Recreate the initial COD crystal CIF supercell set with:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_public_waxs_cif_structures.py --cod-ids nacl=1000041,silicon=1526655,quartz=1011097 --supercell 4,4,4 --write-xyz
```

Recreate the RCSB PDB `1IEE` protein-crystal inputs used by the ACFO
manuscript validation:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_protein_nanocrystal_waxs.py --pdb-id 1IEE --supercell 1,1,1 --output structures\processed\protein_nanocrystal_lysozyme_1iee_1x1x1_fixed.npz
.\.venv\Scripts\python.exe scripts\prepare_protein_nanocrystal_waxs.py --pdb-id 1IEE --supercell 3,3,3 --output structures\processed\protein_nanocrystal_lysozyme_1iee_3x3x3_fixed.npz
.\.venv\Scripts\python.exe scripts\prepare_protein_nanocrystal_waxs.py --pdb-id 1IEE --supercell 5,5,5 --output structures\processed\protein_nanocrystal_lysozyme_1iee_5x5x5_fixed.npz
```

Frozen hashes and the public aIDT input contract are recorded in
[`validation/INPUTS.md`](../validation/INPUTS.md).
