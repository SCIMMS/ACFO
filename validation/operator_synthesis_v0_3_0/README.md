# ACFO v0.3.0: geometry-prepared operator validation

This additive release archives the 4 September 2026 two-machine validation of
geometry-prepared Fourier actions, radial representations and compatible
coefficient-space compositions. The experimental implementations are isolated
in the self-contained request components; the existing top-level `waxs_cake`
API and v0.2.0 evidence are retained.

**Assessment: share with explicit scope.** Constructive numerical evidence is
supported for the tested geometries and subspaces. Several prespecified
publication requirements remain unfulfilled. Read [scope and results](SCOPE_AND_RESULTS.md)
before interpreting a timing ratio or treating a saved audit as a universal claim.

## Contents

| Directory/file | Purpose |
|---|---|
| `request/` | Prespecified protocols, original top-level claim freeze, orchestration code and four self-contained code/input archives |
| `evidence/server36_original_return.public.zip` | RTX 3090 machine's complete return, including arrays, raw timings, logs and original audit receipts |
| `evidence/server59_original_return.public.zip` | RTX 3080 Ti machine's original return, including its ODT loader failure |
| `evidence/server59_corrective_odt.public.zip` | Same-input ODT-only rerun after a library-search-path repair |
| `audits/` | Independent audits, cross-machine comparison, repair provenance and numerical source-data checks |
| `source_data/` | Thirteen numerical CSV tables; filenames retain their source figure identifiers, without publishing a manuscript draft |
| `PUBLICATION_PROVENANCE.json` | Original/public hashes, file-level curation and numerical/array preservation checks |
| `PUBLIC_MANIFEST.json` | SHA-256 and byte count for each distributed file |

`server36` and `server59` are historical machine labels, not accessible network
endpoints. Actual addresses, personal filesystem paths and process-account names
have been normalized. CPU/GPU models, library versions, numeric measurements,
seeds, tolerances, errors and failed outcomes are preserved.

The GitHub source archive contains this entire directory, so GitHub-linked
Zenodo preservation also includes the evidence arrays and replay components.

## Verify the publication without running a benchmark

From the repository root, using Python 3.11 or newer:

```bash
python validation/operator_synthesis_v0_3_0/verify.py
```

This checks public file hashes, nested archive integrity, request/component
manifests, numeric source tables and the presence of the original failure and
documented correction. It uses the Python standard library. A successful
**distribution verification** coexists with the recorded failed scientific or
publication criteria; it never changes those outcomes to pass.

For focused CPU unit tests (NumPy, SciPy, pytest and periodictable required):

```bash
python validation/operator_synthesis_v0_3_0/verify.py --smoke
```

The optional test run extracts the extension component into a new temporary
directory and runs its four focused test files. It does not run external GPU
benchmarks or alter archived evidence.

## Replay the complete prespecified campaign

Use a separate copy of `request/` and an idle, explicitly allocated Linux machine.
The historical profiles constrain the GPU model as well as the workload. They
are not SSH deployment commands. Both recorded environments ultimately used the
name `acfo-ncs-gpu`; the prespecified profile metadata retains the earlier
environment-name label for provenance.

```bash
cd validation/operator_synthesis_v0_3_0/request
python scripts/run_acfo_operator_storyline_external.py --profile server36_full --preflight
# Only on a matching, explicitly allocated idle host:
python scripts/run_acfo_operator_storyline_external.py --profile server36_full --confirm-idle
```

Use `server59_replication` for the RTX 3080 Ti profile. Outputs are created under
`request/_runs/` in a new timestamped directory. No dependencies are automatically
downloaded or installed. The component source code lives under
`components/{waxs,odt,extension,composite}_request.zip`; its local `src` takes
precedence during replay. The archived entrypoints, seeds, supports and acceptance
criteria remain unchanged.

Recorded environments used Python 3.11.15, NumPy 2.4.6, SciPy 1.17.1,
pytest 9.1.1, FINUFFT/cuFINUFFT 2.5.1, PyTorch 2.11.0+cu128 and CuPy 14.1.1.
See the per-machine environment JSON in the returns for compiler, GPU driver,
thread and timing details. ODT also needs a C++ toolchain and usable CUDA runtime
and cuFFT shared-library search paths. The corrective receipt documents the
second machine's environment-local loader repair; it changed neither packages
nor drivers nor scientific parameters. CPU and GPU comparisons are interpreted
within a machine, never as cross-machine speed ratios.

## Input rights and hash semantics

The historical `aidt_diatom_public_contract.npz` filename now contains **geometry
and provenance only**: its third-party `data` raw-intensity field is excluded.
The declared radial physical stages read `source_na_xy`, whose bytes are
unchanged; they generate synthetic operator inputs. This distribution does not
support a claim of measured-data CPSWF reconstruction. Unrelated historical
helpers that expect raw intensity require separate acquisition and permission.
See [third-party input notes](THIRD_PARTY_INPUTS.md).

The top-level `request/CLAIM_FREEZE.json` is byte-identical to its pre-execution
version. Request manifests were rebuilt after metadata normalization and removal
of the unused intensity field. **Original manifests, `.sha256` sidecars and
receipts inside evidence ZIPs authenticate the pre-publication originals**, not
the privacy-normalized ZIPs. Use `PUBLIC_MANIFEST.json` for this distribution;
`PUBLICATION_PROVENANCE.json` links both identities. The original local evidence
was preserved unchanged. No new external benchmark was run for publication.

The repository's ACFO Citation-Required License applies to ACFO-authored material.
Third-party inputs and referenced software retain their own terms. The license
and the published author/ORCID metadata are unchanged from v0.2.0.
