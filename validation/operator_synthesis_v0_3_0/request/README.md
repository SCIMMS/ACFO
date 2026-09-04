# Geometry-prepared representations and composed operators — external request v2.1

Story: High NA → WAXS → ODT → CPSWF representation preparation → composite operator synthesis.
The exact claim freeze is `CLAIM_FREEZE.json`; inherited comparison contracts are in `legacy/`.
Old exploratory results informed this prospective replication. No new server results are included.

## Run (only after separate upload/execution approval and confirming the machine is idle)

Activate the existing environment (`acfo-ncs-gpu` on 36; `pythonProject` on 59).
Python >=3.11, NumPy, SciPy, pytest, FINUFFT, CUDA-enabled torch/CuPy/cuFINUFFT and
the inherited native-extension build tools must already be installed. No downloader or installer is run.

```bash
python scripts/run_acfo_operator_storyline_external.py --profile server36_full --preflight
python scripts/run_acfo_operator_storyline_external.py --profile server36_full --confirm-idle
# use server59_replication on the 12-GiB machine; identical dimensions and arms
```

The full-run preflight refuses a GPU mismatch or competing GPU process. CPU sharing must
also be checked by the operator before --confirm-idle. Process/load snapshots are recorded.
There is no automatic process killing, downsizing, fallback-arm selection or overwrite.
Each run creates a new `_runs/<timestamp-profile>/` directory. Logs stream to files and
each completed stage is announced. Failed outcomes remain in the return archive.

For local package QA (not publication timing):

```bash
python scripts/run_acfo_operator_storyline_external.py --profile local_check
```

This executes the same CPU extension cases, 68 negative/numerical unit tests, the planar
smoke check and the small composite regime. Full GPU ODT and full WAXS/composite remain untested.

## Representation and evidence boundaries

- WAXS, Cartesian ODT and composite payloads retain the earlier manifested bytes.
- The radial chain is rebuilt from geometry in this run; no old rank/accuracy results are
  substituted for new results. The original measurement contract is included for geometry
  provenance, but measured intensities are not used by the new radial probes.
- A coupled modal normal `R* G R` and a Cartesian Toeplitz convolution are distinct representations.
- CPSWF cases retain the original 9 signed modes; the original-grid full-support item is
  an analytic byte estimate, not a numerical execution. Dense-normal scalability remains open.
- High NA is a known-case/prior-art correspondence, with no novelty or performance headline.
- Fixed local pilot timing designs are replicated descriptively; they do not preauthorize a
  new CPSWF speed headline. Stored array bytes are never relabeled peak RSS.
- Retain all SVD/dense comparators and negative controls. Failure to show a speed win does
  not invalidate file integrity. It prevents the corresponding speed claim.

## Return verification

```bash
python scripts/verify_acfo_operator_storyline_return.py --archive <return.zip> --request-root .
```

Keep the adjacent SHA-256 sidecar. Verification checks an exact file inventory, embedded
request/claim/protocol hashes and expected stage coverage. It does not promote any result to
a manuscript claim. Both machine returns and system-specific numerical/performance audits
are needed before manuscript-ready source data; raw outputs are intentionally kept first.
