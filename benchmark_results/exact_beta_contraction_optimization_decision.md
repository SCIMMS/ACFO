# Exact-beta contraction optimization decision

- baseline / fused / cached coefficient medians: `4.323 / 3.593 / 3.141 s`
- baseline/fused paired median and p05: `1.209x / 1.161x`
- fused/cached paired median and p05: `1.169x / 1.023x`
- cached phase table: `42.89 MiB`; setup `0.225 s`
- selected full path legacy / fused hot: `14.515 / 3.652 s` (`3.975x`)
- legacy-selected complex L2: `4.811e-14`
- selected backend: **fused_phase**; cached_phase remains an explicit repeated-hot option.
- optimized 10/30 AB/BA and independent-machine rerun remain pending.
