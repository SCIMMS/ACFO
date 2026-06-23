# Benchmark Structure Generation Plan

Date: 2026-06-14

## Purpose

This document defines how to generate and store structure files for the WAXS
cake-map and 1D curve benchmarks.

The immediate goal is to support two manuscript claims:

1. For isotropic or sufficiently orientation-averaged systems, the 1D WAXS
   curve path can be compared against the Debye orientational average.
2. For anisotropic or partially ordered systems, the 2D cake-map path should be
   benchmarked as a detector-relevant WAXS simulator rather than as a Debye
   replacement.

The recommended benchmark input format is a self-contained `.npz` file. Raw MD
and source files should be retained separately for reproducibility.

## Recommended Directory Layout

```text
structures/
  raw/
    methanol_10k/
      charmm_gui/
      gromacs/
      openmm/
      trajectories/
  processed/
    methanol_10k_cube.npz
    methanol_10k_sphere.npz
    methanol_10k_cube.xyz
    methanol_10k_sphere.xyz
  metadata/
    methanol_10k.json
  README.md
```

`raw/` should contain generated or downloaded source files. These may be large
and should not necessarily be committed to git.

`processed/` should contain small, benchmark-ready snapshots. The `.npz` files
are the canonical solver inputs. The `.xyz` files are human-inspection helpers.

`metadata/` should contain a plain JSON sidecar for every generated structure.

## NPZ Schema

Each benchmark `.npz` file should contain:

```text
coords
  dtype: float64
  shape: (n_atoms, 3)
  units: nm

elements
  dtype: unicode string
  shape: (n_atoms,)
  examples: "H", "C", "N", "O", "S"

atomic_numbers
  dtype: int16
  shape: (n_atoms,)
  optional but recommended

box_vectors
  dtype: float64
  shape: (3, 3)
  units: nm
  for non-periodic cropped structures, keep the parent MD box and set
  periodic = false in metadata

structure_id
  dtype: unicode string scalar

metadata_json
  dtype: unicode string scalar
  compact JSON copy of the sidecar metadata
```

The benchmark code should treat `coords` and `elements` as required. All other
fields are for validation, provenance, and future form-factor handling.

## Metadata Sidecar

Each structure should have a sidecar JSON with at least:

```json
{
  "structure_id": "methanol_10k_cube",
  "system_type": "isotropic_liquid",
  "molecule": "methanol",
  "n_atoms": 10000,
  "n_molecules": 2500,
  "units": "nm",
  "periodic": true,
  "source_workflow": "CHARMM-GUI/CGenFF + GROMACS",
  "force_field": "CHARMM/CGenFF",
  "parameter_source": "CHARMM-GUI small molecule workflow",
  "temperature_K": 300.0,
  "pressure_bar": 1.0,
  "density_g_cm3": null,
  "equilibration": {
    "minimization": true,
    "nvt_steps": null,
    "npt_steps": null
  },
  "production": {
    "steps": null,
    "timestep_fs": null,
    "snapshot_time_ps": null
  },
  "software": {
    "gromacs": null,
    "openmm": null,
    "ase": null,
    "python": null
  },
  "random_seed": null,
  "parent_files": [],
  "notes": ""
}
```

Use `null` for unknown values, but avoid leaving important provenance unknown in
final manuscript benchmarks.

## Structure Sets

### 1. Isotropic Debye Benchmark Set

These are the main files for the Debye-replacement claim.

Recommended first systems:

```text
water_2k_cube.npz
water_2k_sphere.npz
water_5k_cube.npz
water_5k_sphere.npz
water_10k_cube.npz
water_10k_sphere.npz

methanol_2k_cube.npz
methanol_2k_sphere.npz
methanol_5k_cube.npz
methanol_5k_sphere.npz
methanol_10k_cube.npz
methanol_10k_sphere.npz
```

Next organic systems:

```text
acetonitrile_10k_cube.npz
acetonitrile_10k_sphere.npz

acetone_10k_cube.npz
acetone_10k_sphere.npz

dmso_10k_cube.npz
dmso_10k_sphere.npz

benzene_10k_cube.npz
benzene_10k_sphere.npz
```

Rationale:

- `cube` snapshots represent the natural periodic MD output.
- `sphere` crops reduce finite cube anisotropy and are cleaner for Debye
  orientational-average comparisons.
- Exact Debye should be feasible up to a few thousand atoms, while 10k atoms are
  useful for stress-testing exact or sampled Debye timing.

### 2. Scaling Set

These files are for performance scaling rather than exact Debye validation.

```text
water_50k_cube.npz
water_100k_cube.npz
methanol_50k_cube.npz
methanol_100k_cube.npz
```

Exact Debye may be too expensive here. Use direct Ewald-ring checks, sampled
Debye, or smaller cropped subsets.

### 3. Protein Orientation-Average Set

Use small globular proteins from PDB/wwPDB.

Recommended files:

```text
protein_ubiquitin_heavy_cube_or_centered.npz
protein_ubiquitin_h_added_centered.npz
protein_ubiquitin_rotated_016.npz
protein_ubiquitin_rotated_064.npz
protein_ubiquitin_rotated_256.npz
```

Important distinction:

- `heavy` files keep only atoms present in the PDB/mmCIF.
- `h_added` files include hydrogens after a reproducible protonation workflow.

The rotated ensembles are used to show convergence from fixed-ring azimuthal
averaging toward the Debye orientational average.

### 4. Negative-Control / Anisotropic Set

These files are not meant to match Debye. They define the boundary of the 1D
Debye claim and support the 2D cake-map claim.

```text
nanorod_from_crystal_cube_or_free.npz
thin_slab_from_crystal.npz
small_bulk_crystal_supercell.npz
partially_crystalline_plus_amorphous.npz
```

Source structures can come from ICSD or COD CIF files. The recommended workflow
is:

```text
CIF -> supercell -> rod/slab/sphere cut -> NPZ
```

This is more reproducible than searching for a pre-made nanorod structure.

## MD Generation Workflows

### Preferred Liquid Workflow: CHARMM-GUI/CGenFF + GROMACS

This is a good fit if the server already has GROMACS and GPU acceleration.

```text
1. Generate small-molecule topology and parameters with CHARMM-GUI/CGenFF.
2. Export or convert to GROMACS-compatible .itp/.top/.gro files.
3. Pack the liquid box with GROMACS tools or Packmol.
4. Energy minimize.
5. Run short NVT equilibration.
6. Run NPT density equilibration.
7. Run production and save snapshots.
8. Convert selected snapshots to .npz and .xyz.
9. Generate sphere crops from the same snapshots.
```

Record CGenFF penalty scores when available. High penalty scores should be
flagged in metadata.

### Alternative Workflow: CHARMM-GUI/CGenFF + OpenMM

Use this if PDB/PSF/RTF/PRM files are more convenient.

```text
1. Generate PDB, PSF, and CHARMM parameter files.
2. Load with OpenMM CharmmPsfFile and CharmmParameterSet.
3. Set PME, cutoff, constraints, integrator, and barostat.
4. Minimize, equilibrate, and save snapshots.
5. Convert to benchmark .npz and .xyz.
```

This route is clean for OpenMM, but GROMACS may be more convenient for large
liquid boxes.

### ASE Role

ASE should be used mainly for:

```text
- reading and writing XYZ/CIF/PDB helper files
- centering structures
- converting units
- supercell construction
- sphere, slab, or rod cuts
- quick geometry sanity checks
```

ASE is not the preferred MD engine for these liquid benchmarks.

## Sphere Crop Rule

For a periodic cubic MD snapshot with side length `L`, define the box center as
`c`. Choose a crop radius `r_crop`, for example:

```text
r_crop = 0.45 * L
```

Keep atom `i` if:

```text
|coords_i - c| <= r_crop
```

Then recenter the retained coordinates around the origin.

The sphere crop should have:

```text
periodic = false
parent_structure_id = "<cube structure id>"
crop_shape = "sphere"
crop_radius_nm = ...
```

in metadata.

## First Milestone

Generate the following first:

```text
water_2k_cube.npz
water_2k_sphere.npz
water_5k_cube.npz
water_5k_sphere.npz
water_10k_cube.npz
water_10k_sphere.npz

methanol_2k_cube.npz
methanol_2k_sphere.npz
methanol_5k_cube.npz
methanol_5k_sphere.npz
methanol_10k_cube.npz
methanol_10k_sphere.npz
```

These are enough for the first Debye-vs-1D WAXS validation:

```text
I_Debye(q)
I_direct_ring(q)
I_histogram_ring(q)
```

The required benchmark outputs are:

```text
relative L2: direct ring vs Debye
relative L2: histogram ring vs direct ring
relative L2: histogram ring vs Debye
runtime: Debye exact
runtime: direct ring
runtime: histogram 1D curve
```

## Second Milestone

Add:

```text
methanol_50k_cube.npz
acetonitrile_10k_cube.npz
acetone_10k_cube.npz
one small globular protein heavy-atom NPZ
one small crystal supercell NPZ
one nanorod cut NPZ
```

This extends the benchmark from the Debye-replacement claim into the broader
WAXS cake-map story.

## Stop Rules

Do not scale to 500k or 1M atoms until the 2k, 5k, and 10k structures pass:

```text
1. metadata completeness check
2. element and coordinate sanity check
3. density check against expected liquid density
4. Debye vs direct-ring comparison
5. histogram convergence sweep over bin_width_nm
```

Large structures are useful only after the small structures establish that the
comparison is physically meaningful.
