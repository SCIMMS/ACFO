# Third-party input provenance

## Annular intensity-diffraction-tomography geometry

The archived input metadata identifies the source as
[bu-cisl/IDT-using-Annular-Illumination](https://github.com/bu-cisl/IDT-using-Annular-Illumination)
and its linked Diatom I acquisition. The historical acquisition note records
BSD-3-Clause for the source repository and no separately verified license for
the linked raw intensity data. A repository code license is not assumed to grant
rights over externally hosted measurements.

Accordingly, this release excludes `data.npy` from the historical
`aidt_diatom_public_contract.npz` input. All remaining geometry/provenance arrays
are retained byte-for-byte, with original input and removed-field hashes in
`PUBLICATION_PROVENANCE.json`. The declared physical-transfer validation reads
`source_na_xy`, not the intensity stack. Its other system parameters are recorded
in the packaged JSON protocols; its numerical inputs are synthetic. No third-
party raw measurement or article PDF is redistributed by this new supplement.

## WAXS structural input

The WAXS request contains processed unit-cell and repeated-cell coordinate inputs
derived from [Protein Data Bank entry 1IEE](https://www.rcsb.org/structure/1IEE).
Structure-generation code and per-input provenance are included with the WAXS
component. The candidate occupancy/B-factor library is a manufactured benchmark,
separate from experimental structure inference.

## Referenced software and mathematical methods

NumPy, SciPy, FINUFFT, cuFINUFFT, PyTorch, CuPy and pytest are dependencies, not
vendored binaries. Their terms remain those of their respective projects.
Known Debye–Wolf/Richards–Wolf and specialized vortex-series formulations, and
the finite-Hankel/CPSWF lineage, are acknowledged as antecedents; this release
does not assert authorship of those mathematical constructions.
