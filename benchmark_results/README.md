# Benchmark Result Policy

This directory is primarily a local output area for generated benchmark files.

The repository tracks only curated markdown summaries that document the current
claim boundary, validation status, and reproducible benchmark commands. Large
or regenerated artifacts are intentionally excluded from Git, including:

- `.npz`, `.h5`, `.mat`, and other measured or converted data packages;
- raw public-data probes and downloaded prior-art PDFs;
- large JSON/CSV sweep outputs;
- generated figures and report PDFs;
- temporary smoke-test outputs.

If a result is needed for a paper or review, keep a concise markdown summary in
this directory and store heavy data artifacts in an external data/release
location with explicit provenance.

Currently curated summaries include:

- `aidt_torch_gpu_optimization_summary.md`
- `aidt_public_transfer_torch_gpu_optimized_full700_compare.md`
- `aidt_public_transfer_real_condition_summary.md`
- `aidt_10hz_condition_summary.md`
- `aidt_realtime_projection_summary.md`
- `experimental_use_case_comparison.md`
- `odt_measured_data_contract.md`
- `curved_ewald_prior_art_*.md`
