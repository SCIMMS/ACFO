# ACFO manuscript validation data

This directory contains the compact, machine-readable evidence used to prepare
the ACFO manuscript. It is organized by the role each dataset plays in the
paper.

## Evidence map

| Directory | Manuscript role | Evidence boundary |
| --- | --- | --- |
| `waxs/` | Validation axis | Accuracy, prepared-operator timing, detector geometry, and local regime probes for the tested WAXS workloads |
| `odt_aidt/` | First extension | GPU-resident processing-core measurements; acquisition, host transfer, hologram demodulation, and end-to-end microscope throughput are excluded |
| `general_curvature/` | Second extension | CPU correctness for the tested finite axisymmetric curve families and prescribed homogeneous-bulk undepleted uniaxial nonlinear-optics problem; no GPU claim |
| `high_na_si/` | Prior-art/Supplementary validation | Scalar structured-grid cutoff-safety correspondence; not a general replacement for vectorial Richards-Wolf or domain-specific FFT-Debye methods |
| `provenance/` | Independent-run audit | Curated validation status and original package hashes |

`claim_metrics.csv` gives the paper-facing quantitative summary.
`evidence_architecture.csv` records how the four evidence groups are used.
`headline_metrics.json` is a compact, machine-readable claim-boundary ledger.
Large or third-party inputs and their frozen hashes are documented in
[`INPUTS.md`](INPUTS.md).

## Publication normalization

Machine-local absolute paths were replaced in nine JSON files by
`scripts/normalize_validation_public_paths.py`. Non-finite aIDT output-statistic
fields that were deliberately not collected are represented by JSON `null`.
Each modified JSON contains an `_publication.source_sha256` field that
identifies the corresponding raw receipt. Scientific timing and accuracy values
were not changed.

The raw external RTX 3090 return package is identified by SHA-256
`1a558be711da3b4fe1623e3a1a31fb013fd82cb783c7e235a8a59a49e9acdca4`.
The independent Intel Core i5-12400F return bundle is identified by SHA-256
`be7f80a2b1477cf8baa42fcb4ea30b5f896e825dd3838f2a800d2dec5c68d61f`.
The compact general-curvature replication package itself is included under
`general_curvature/replication_package_v1_1/`.

## Verify

From the repository root:

```powershell
python scripts/verify_validation_release.py
```

The verifier checks the manifest, JSON readability, selected headline values,
package checksums, local-path normalization, and common credential patterns.

## Citation and rights

Copyright in the original ACFO materials is held by Minsu Kim. Use is permitted
under the citation-required terms in the repository `LICENSE`. Use the
repository citation in `CITATION.cff`; the paper DOI will be added after it is
available.
