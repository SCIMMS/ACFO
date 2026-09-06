# ACFO v0.4.0: structure guided reuse and composition

This supplement records how rotational coefficient structure identifies reusable
factors, intermediate basis spaces, and correction supports when an operator changes.
It contains executable validation adapters, frozen numerical conditions, independent
two-host returns, earlier local evidence, and figure source data.

## Evidence and scope

| Case | Computational question | Evidence and observed limits |
|---|---|---|
| Elliptic perturbation | Reuse, sparse correction, or direct rebuild? | 512 fresh workers, 1,024 base/target checks, two tolerances and prospective boundary probes. Small corrections can be cheaper; large corrections can lose to direct computation. This is one fixed elliptic family. |
| CPSWF preparation | Can the commuting structure reduce preparation cost? | Six process pairs for each of four grid/bandwidth settings against matrix-free randomized preparation. At the largest fixed-bandwidth grid the median ratio is 1.843; smaller grids favor randomized preparation. |
| CPSWF operator closure | Does an input basis remain sufficient after composition? | 432 comparison rows. Fixed input spaces pass 65/108 tested composites; expanded intermediate spaces pass 108/108. Required intermediate order can exceed input rank. |
| Uniaxial slab | Which factors survive a material update? | Two hosts, five fresh pairs each. One-RHS warm update favors shared preparation; 64-RHS warm comparison favors the modal baseline. Physical fields and derivatives use independent references. |
| Radial 4f relay | When does a compact middle space amortize conversion? | Repeated large workloads favor compact propagation; single-use cold workloads can lose. Nyström remains a strong comparator. |
| Curved-detector WAXS | Can coordinate tangents share prepared contractions? | Full tangents improve production-relative timing; coordinate-field comparison is close to parity. Streaming provides a separate memory/time tradeoff. Direct amplitude and finite-difference checks are retained. |
| Weighted ODT | When is the weighted normal core cheap to update? | Harmonic and view-rank cases improve core time, while whole-operation ratios remain near one. Sparse Fourier coupling loses. Agreement is against the same prepared normal action. |

External replication covers slab, relay, WAXS and ODT on two Linux hosts (RTX 3090
and RTX 3080 Ti), with 20 workers and 18 focused test passes per host. Elliptic,
CPSWF and affine-family controls are separately identified local records. Hot calls
within a worker are correlated; fresh processes define the timing sample unit.

## Contents

- `replay/`: isolated source tree, model adapters, tests, contracts and local records.
  The public package API at repository root is unchanged.
- `evidence/server*_confirm.public.zip`: privacy-normalized original returns,
  including per-worker inputs, results, environment records and logs.
- `source_data/`: figure values, all paired external samples, elliptic samples,
  closure rows and host environments.
- `provenance/`: original request inventory and original elliptic freeze records.
- `PUBLICATION_PROVENANCE.json`: original/public hashes and every changed source.
- `PUBLIC_MANIFEST.json`: SHA-256 and length of every distributed file.

The original request archive SHA-256 is
`10add816bc0cf061d7cc523e19b8432d693e441ff692ff0b9b5f6814a742e26e`.
Public metadata suppresses private machine addresses and personal paths. Numerical
JSON leaves and binary numerical arrays are checked for preservation during curation.
Historical manifests inside returned archives authenticate the original bytes;
`PUBLIC_MANIFEST.json` authenticates these public derivatives. The public replay
inventory and elliptic freeze hashes are rebound to public bytes, with original
registrations retained. This publication step introduces no new preregistration.

## Verify the archive

From this directory, with Python 3.11 or later:

```bash
python verify.py
python -m pip install -r replay/requirements-core.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python verify.py --smoke
```

The verifier checks all public hashes, re-evaluates all 40 returned workers against
the frozen numerical gates, verifies elliptic confirmation coverage, and checks
CPSWF factor/result coverage. `--smoke` additionally executes focused CPU tests in
a temporary copy. It does not repeat the large performance campaign. Torch, CuPy,
CUDA and a C++ compiler are required for GPU replication; install compatible builds
for the target machine separately.

The CPU smoke suite also imports PyTorch; the command above installs its CPU build.
For GPU replication, use a CUDA-compatible PyTorch build instead of the CPU build.

## Reproduce the four externally replicated cases

Run from `replay/`, sequentially on an otherwise idle host:

```bash
python scripts/run_acfo_structure_replication.py --profile cpu --preflight
python scripts/run_acfo_structure_replication.py --profile cpu --phase smoke
python scripts/run_acfo_structure_replication.py --profile cpu --phase confirm
python scripts/run_acfo_structure_replication.py --build-native
python scripts/run_acfo_structure_replication.py --profile gpu --preflight
python scripts/run_acfo_structure_replication.py --profile gpu --phase smoke
python scripts/run_acfo_structure_replication.py --profile gpu --phase confirm
```

New runs go to `_runs/`; retained observations stay unchanged. Validation accepts
the frozen numerical conditions independently of whether a speed advantage occurs.

## Historical experiments

Run these commands from this supplement directory. Each output directory must be new.

```bash
# One fresh CPSWF preparation worker; cases 0..3, arms cpswf/randomized, repeats 0..5.
python replay_historical.py preparation --case 2 --arm cpswf --repeat 0 --output /tmp/acfo-prep
# One elliptic update worker; case labels are in the frozen pilot contract.
python replay_historical.py elliptic --case e8_07 --arm cap_update --repeat 0 --output /tmp/acfo-elliptic
# Basis closure, using the preserved numerical protocol.
cd replay
python scripts/experiment_cpswf_operator_algebra.py --protocol validation_contracts/cpswf_operator_algebra_pilot_v1.json --output /tmp/acfo-closure
```

Use an absolute writable path appropriate for your platform in place of `/tmp/...`.
The historical adapter selects the recorded experiment without overwriting its
evidence. Preparation confirmation comprises six alternating arm-order process
pairs across four cases; both tolerances within each process share a sample.
Elliptic cases, arms and pair order are preserved in the sweep driver and contracts.
Frozen reference arrays are included, including references reused from prior runs.

Local evidence paths under `replay/reports/` include:

- `acfo_elliptic_perturbation_sweep_20260905_v1/`: raw pilot/confirmation rows,
  independent references, forecast and prospective contract.
- `acfo_research_confirmation_20260905_v1/`: preparation A-worker results and factors
  plus original source snapshots. Other historical A/B/C/D investigations are outside
  this supplement's preparation comparison.
- `acfo_cpswf_operator_algebra_20260905_v2/`: complete closure comparisons.
- `acfo_mixed_basis_family_20260906_confirm_v2/`: affine-family negative controls.
- The four `*_20260906_confirm_v*` folders: local baseline data, kept distinct from
  the two-host results in `evidence/`.

Archive verification establishes consistency of the distributed records. Replaying
the experiments on new hardware provides additional computational replication.
Original submission drafts and internal editorial planning are outside this release.

License: ACFO Citation-Required License (see `LICENSE`), source-available.
