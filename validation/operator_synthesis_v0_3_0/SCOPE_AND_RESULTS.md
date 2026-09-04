# Evidence scope and results — 4 September 2026

The common procedure prepares a representation from geometry and a tolerance,
then explicitly constructs compatible forward, adjoint, normal, composite or
derivative actions. Here *operator synthesis* means this explicit construction
from known primitives. Automatic symbolic discovery or compilation of arbitrary
operator graphs remains outside the implemented scope.

The numerical campaign connects the WAXS-derived rotational restriction core
to broader preparation and composition workflows. Known High-NA expansions
provide a correspondence example; layered vector-wave composition provides a
constructive example of the general composition principle.

## Recorded evidence

The bounds below are observed maxima across the two recorded machines, with the
documented ODT corrective rerun used for the second machine. They are empirical
results in the prescribed finite geometries, norms and subspaces.

| Validation | Recorded result | Interpretation |
|---|---|---|
| High-NA scalar/vector restriction | Same-node direct-quadrature relative L2 at most 2.44e-15; effective-vector support stress error at most 4.34e-9 | Correspondence with known Debye–Wolf/Richards–Wolf formulations, including source-spectrum support; no speed or expansion-novelty headline |
| WAXS repeated curved evaluation | 100 candidates × 3 seeds × 2 machines; Spearman 1 and matching top-1/top-5; intensity relative L2 at most 2.91e-7 | Ranking preservation for the manufactured occupancy/B-factor candidate library |
| Cartesian ODT | Forward/adjoint, weighted normal and CG checks; 256³ GPU normal-action sample error at most 3.76e-6 | Native Type-2 forward, Type-1 adjoint and cached Cartesian Toeplitz comparison |
| Common radial CPSWF | 45/45 prescribed cases on each machine; identical rank tables | Ranks within 0–1 of matched shared-SVD and individual-SVD lower-bound ranks for the tested weight family |
| Coupled modal normal | Dense-reference probe relative L2 at most 7.50e-9; compiled/composed difference at most 6.77e-16 | Preserves required off-diagonal mode coupling in the selected-mode subspace |
| Geometry-prepared API | Full-chain probe relative L2 at most 9.92e-10; checked fresh/cache agreement | Basis preparation, cache reuse, forward/adjoint and normal actions form a reproducible workflow |
| Layered vector-wave composite | Direct/materialized relative L2 at most 3.07e-16; JVP/finite-difference error at most 1.56e-10 | Fixed-channel translation, propagating-spectrum layered Green action and restriction, with adjoint/JVP/VJP/normal checks |

Removing the intermode coupling terms produced 11.4–23.3% relative action error.
The geometry-prepared ranks for absolute modes `[0,4,16,64,128]` were
`[152,150,144,122,94]` on both machines. A separate five-channel noise-free
translation-recovery example used four updates and reached absolute translation
L2 error at most 9.82e-18. This is a synthetic optimizer-use demonstration.

## Performance must retain its comparison boundary

- **WAXS:** this specific archived payload compares a unit-cell-plus-exact-lattice
  ACFO implementation with full-supercell FINUFFT. Its large stored workflow
  ratios are diagnostic. A strongest comparator with the same available lattice
  contraction is missing, so those ratios are excluded from speed headlines.
  Type-3 eligibility for curved q coordinates does not establish strongest-
  implementation eligibility. The planar Type-1 experiment uses a different
  geometry and supplies a correctness boundary only.
- **Cartesian ODT:** the fastest GPU cuFINUFFT Toeplitz preparation was faster
  than ACFO-derived preparation on both machines (comparator/ACFO setup ratios
  approximately 0.911 and 0.795). The shared hot-action benefit arises from
  caching the same fixed normal. The inherited timing does not supply the
  top-level prespecified paired hot-speed interval; no new speed headline is
  assigned to this evidence.
- **Composite CPU hot action:** own unfused chain/materialized ratios were
  22.045 `[21.436,22.464]` and 20.275 `[16.660,21.275]`. These are within-experiment
  repeat-resampling intervals from the recorded four-process, 31-repeat ABBA
  design. They describe this fixed chain, not an independently optimized external
  solver, full retrieval pipeline, or uncertainty across independent campaigns.
  All prescribed small/medium/large cases remain in the archive.
- **Retained arrays:** 640,032 bytes for the unfused chain versus 373,248 bytes
  for the materialized action, a 41.68% reduction. This is retained-array byte
  accounting, separate from peak process RSS or peak GPU memory.

Historical v0.2.0 WAXS and NuMagSANS headline comparisons belong to different
protocols retained in `validation/ncs_v0_2_0`. They must not be substituted for
the missing comparator or interval in the present campaign. UOB measured-data
ranking and NuMagSANS were not rerun in this five-system campaign.

## Failures and limits retained

1. The RTX 3080 Ti original full run exited with a standalone cuFINUFFT loader
   failure. The ODT-only same-input rerun after a library-search-path repair is
   supplied separately. The original failure is retained.
2. Original Linux radial audits passed 523/523 on each machine. Independent
   Windows re-audits passed 508/523: fifteen bitwise `scipy.special.jv` matrix
   equality tests failed. The maximum Windows/server relative Frobenius
   difference was approximately 7.05e-15. The strict criteria were not relaxed;
   the failed Windows audit remains failed.
3. With qR about 441.38, kernel support 442, data support 128 and padding 48,
   the required kernel expansion support is 490; 1,024 angular samples satisfy
   its Nyquist check. This support differs from the nine signed input modes in
   the executed physical/API subspace. Dense coupled-Gram sizes of 34.45 GiB
   and 501.91 GiB are allocation-free estimates, not observed OOM events,
   full-support numerical validations, or peak-memory measurements.
4. Measured-data CPSWF reconstruction, unrestricted SO(3), evanescent and
   multiple-scattering full-wave models, globally optimal basis selection and
   arbitrary-graph compilation were not validated here.

The machine audits therefore retain `publication_ready: false` and the combined
audit retains `all_frozen_publication_requirements_fulfilled: false`, while
supporting scoped constructive numerical evidence. Publishing the evidence
archive does not change those scientific conclusions.
