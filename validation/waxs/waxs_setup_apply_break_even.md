# Frozen external WAXS setup/apply/break-even summary

Validation: **PASS**. This is deterministic accounting from frozen receipts, not a new benchmark.

## Protocol distinction

- **Prepared 1M reusable-plan:** `T(K) = S + K H`. Both ACFO and FINUFFT plans are prepared once; totals use the measured hot medians.
- **Detector-active q-blocked:** `T(K) = S + F + (K-1)H`. FINUFFT block-plan setup is embedded in every evaluation, so its separate setup field is zero.
- Break-even is the first positive integer `K` for which `T_FINUFFT(K) >= T_ACFO(K)`.

## Setup, apply and break-even

| case | ACFO setup s | FINUFFT setup s | ACFO apply s | FINUFFT apply s | break-even K |
|---|---:|---:|---:|---:|---:|
| Prepared 1M repeated-crystal WAXS | 0.079044 | 0.631565 | 0.640995 hot median | 17.577265 hot median | 1 |
| Detector-active WAXS with memory-safe q-blocked FINUFFT | 0.021841 | 0.000000 | 11.876353 first / 12.277774 cached | 25.029076 first / 30.299071 cached | 1 |

## Setup-inclusive total-time model

| case | K | ACFO total s | FINUFFT total s | FINUFFT/ACFO |
|---|---:|---:|---:|---:|
| prepared_1m_reusable_plan | 1 | 0.720039 | 18.208830 | 25.289x |
| prepared_1m_reusable_plan | 10 | 6.488997 | 176.404218 | 27.185x |
| prepared_1m_reusable_plan | 100 | 64.178569 | 1758.358100 | 27.398x |
| detector_active_q_blocked | 1 | 11.898193 | 25.029076 | 2.104x |
| detector_active_q_blocked | 10 | 122.398161 | 297.720716 | 2.432x |
| detector_active_q_blocked | 100 | 1227.397839 | 3024.637120 | 2.464x |

## Detector-active memory

- First-call peak RSS: ACFO `114.090 MiB`; FINUFFT `621.828 MiB`; ratio `5.450x`.
- Incremental first-call peak: ACFO `18.262 MiB`; FINUFFT `491.242 MiB`; ratio `26.900x`.
- The prepared-1M frozen receipt did not record comparable RSS stages.

## Claim boundary

- Prepared-1M totals combine separately measured setup with hot medians and are not paired statistics.
- Detector-active totals preserve the frozen q-blocked FINUFFT policy; they must not be described as reusable whole-plan FINUFFT totals.
- K=10 and K=100 rows are modeled from measured stage values rather than directly timed K-call workflows.

## Frozen sources

- `reports/acfo_ncs_external_validation_20260714_rtx3090/receipts/waxs_prepared_1m_abba.json` — SHA-256 `798b9360564dac18d4ad58859bf92248739c14d07eb53c1b12edb77b664a6b24`
- `reports/acfo_ncs_external_validation_20260714_rtx3090/receipts/waxs_detector_nq512_abba.json` — SHA-256 `45e393afc50f89bc73efbbbad4fc47f5c6443a01047b37f21484c7f269d6fb7c`
