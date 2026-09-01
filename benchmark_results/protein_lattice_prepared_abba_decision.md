# Prepared fused 1M repeated-crystal 10/30 AB/BA decision

- local prepared timing gate: **PASS**
- prepared / FINUFFT median: `3.143 / 106.945 s`
- paired speedup median / p05: `33.480x / 24.243x`
- prepared vs FINUFFT complex L2: `3.156e-07`
- prepared vs legacy complex L2: `4.811e-14`
- legacy/prepared factorized median improvement: `4.427x`
- AB/BA speedup-median relative gap: `2.04%`
- setup-based first-total ratio using measured medians: `32.425x` (not paired)
- independent-machine publication timing: **PENDING**

| protocol | factorized median s | FINUFFT median s | paired median | paired p05 |
|---|---:|---:|---:|---:|
| legacy 10/30 | 13.913 | 98.985 | 7.138x | 5.817x |
| prepared fused 10/30 | 3.143 | 106.945 | 33.480x | 24.243x |

The prepared local timing gate is closed. Independent-machine replication remains the publication timing gate.
