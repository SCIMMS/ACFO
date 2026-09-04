# Axisymmetric Curved-Fourier Operator (ACFO)

[![DOI — all versions](https://zenodo.org/badge/DOI/10.5281/zenodo.22239162.svg)](https://doi.org/10.5281/zenodo.22239162)

Prepared forward-adjoint operators for repeated curved Fourier inference.
ACFO applies analytic, symmetry and dimensional contractions first, selects the
lowest eligible FFT/NUFFT representation, and uses SO(2) geometry factorization
when the workload contains repeated complete rotational orbits.

## Version 0.3.0 — geometry-prepared operator validation

Archived release: [Zenodo DOI 10.5281/zenodo.22298532](https://doi.org/10.5281/zenodo.22298532).
Scientific release commit: `01ee59422399a554ff099877f8ad551535a41fe8`.

The [new validation supplement](validation/operator_synthesis_v0_3_0/README.md)
contains the two-machine High-NA, WAXS, ODT, radial CPSWF and layered-composite
campaign: component code, prespecified claim freeze, numerical arrays, raw timings,
original failures, corrective evidence and independent audits. It connects the
rotational-restriction core to geometry-prepared coefficient-space actions and
explicit compatible operator composition. Experimental implementations remain
isolated inside the replay component archives; the established package API is
unchanged.

The evidence supports the tested constructive workflows with
[explicit comparison and support boundaries](validation/operator_synthesis_v0_3_0/SCOPE_AND_RESULTS.md).
In particular, this campaign's WAXS timings lack an equally contracted strongest
comparator, ODT lacks the prespecified paired hot-speed interval, and Windows
strict bitwise radial failures are retained. No universal acceleration or
full-support reconstruction claim is assigned to this release.

```bash
python validation/operator_synthesis_v0_3_0/verify.py
```

See the [v0.3.0 release notes](docs/releases/v0.3.0.md). The all-versions DOI above
tracks the release series; the previous version-specific DOI is recorded below.

## Version 0.2.0 manuscript release (preserved historical evidence)

Archived release: [Zenodo DOI 10.5281/zenodo.22239163](https://doi.org/10.5281/zenodo.22239163).

The manuscript-facing code, figures, source data, normalized external evidence
and numerical audits are in [`validation/ncs_v0_2_0`](validation/ncs_v0_2_0/README.md).
The NuMagSANS headline is explicitly limited to the shared 800-orientation
Fourier backbone; its five packing reductions are excluded from the speed
ratio. Recompute the headline intervals and validate all public hashes with:

```bash
python scripts/verify_ncs_release_v020.py
```

This repository is evolving from an atomistic WAXS cake-map solver prototype
into a broader curved-Ewald operator testbed. The current Python package name is
still `waxs_cake` because the validated WAXS solver path remains the historical
core, while newer scripts and benchmarks cover high-NA optics and ODT/aIDT
operator workloads.

The WAXS core compares atomistic cake-map forward solvers:

- direct phase-sum reference on Ewald rings
- binned Ewald-ring circular FFT
- binned Jacobi-Anger harmonic expansion
- q-dependent hybrid planner
- optional FINUFFT type-3 baseline

The core output is a caked single-shot WAXS map `A(q, phi)` or
`I(q, phi) = |A(q, phi)|^2`, not a raw detector image.

The newer ODT/aIDT GPU benchmark path demonstrates repeated fixed-geometry
transfer reconstruction on public annular IDT geometry. Curated benchmark
receipts used by the legacy validation scripts live under `benchmark_results/`.
The paper-facing, normalized evidence and its provenance ledger live under
`validation/`. Large or third-party inputs are excluded and identified by
SHA-256 in `validation/INPUTS.md`; the compact general-curvature field archive
and replication package are included because they are original ACFO outputs.

## Manuscript validation release

The compact manuscript evidence is published under [`validation/`](validation/README.md):

- WAXS is the primary validation axis.
- ODT/aIDT is the first processing-core extension.
- General-curvature uniaxial optics is the second extension.
- High-NA optics is prior-art correspondence and Supplementary cutoff-safety
  validation.

The validation release contains machine-readable results, independent-run
provenance, claim boundaries, checksums, and a CPU replication package. Raw
machine paths were normalized for publication without changing scientific
values. Run the release audit with:

```powershell
python scripts/verify_validation_release.py
```

## Citation and license

Copyright in the original ACFO materials is held by Minsu Kim. Use is permitted
under the citation-required terms in [`LICENSE`](LICENSE). Cite the repository
using [`CITATION.cff`](CITATION.cff); the associated paper DOI will be added
when available.

## Setup

Minimal development setup:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
```

Broader local benchmark setup:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` includes optional benchmark/report dependencies. The CI and
unit-test path uses the narrower editable install above.

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Development details, C++ extension controls, and artifact policy are documented
in `docs/development.md`.

## Compare Methods

Small direct-reference case:

```powershell
.\.venv\Scripts\python.exe scripts\compare_methods.py --atoms 1500 --nq 40 --nphi 180 --nr 48 --nz 48 --qmax 2.2 --radius 20 --height 20
```

Large NUFFT-reference case:

```powershell
.\.venv\Scripts\python.exe scripts\compare_methods.py --atoms 1000000 --skip-direct --nq 40 --nphi 180 --nr 48 --nz 48 --qmax 2.2 --radius 20 --height 20
```

For faster approximate bin assignment, use float32 binning:

```powershell
.\.venv\Scripts\python.exe scripts\compare_methods.py --atoms 1000000 --skip-direct --nq 40 --nphi 180 --nr 48 --nz 48 --qmax 2.2 --radius 20 --height 20 --binning-dtype float32
```

This only changes the coordinate arithmetic used for assigning atoms to
`(R, z, beta)` bins. It can move atoms that lie very close to bin boundaries, so
compare the resulting error against the float64/default path before using it for
accuracy-sensitive results.

For the fastest single-element, unweighted histogram path, use the optional
numba backend:

```powershell
.\.venv\Scripts\python.exe scripts\compare_methods.py --atoms 1000000 --skip-direct --nq 40 --nphi 180 --nr 48 --nz 48 --qmax 2.2 --radius 20 --height 20 --hist-backend numba-parallel
```

The benchmark script warms up numba before timing. In application code, the
first numba call includes JIT compilation overhead; subsequent calls measure the
compiled kernel.

For a no-JIT C++ histogram backend with weighted and multi-element support,
build the pybind11 extension in place:

```powershell
.\.venv\Scripts\python.exe setup.py build_ext --inplace
.\.venv\Scripts\python.exe scripts\compare_methods.py --atoms 1000000 --skip-direct --nq 40 --nphi 180 --nr 48 --nz 48 --qmax 2.2 --radius 20 --height 20 --hist-backend cpp
```

For the current fast one-shot path, combine the C++ histogram with a reduced
histogram dtype:

```powershell
.\.venv\Scripts\python.exe scripts\compare_methods.py --atoms 1000000 --skip-direct --nq 40 --nphi 180 --nr 48 --nz 48 --qmax 2.2 --radius 20 --height 20 --hist-backend cpp --hist-dtype float32 --circular-backend cpp --complex-dtype auto
```

The same production fast-path options are available as a preset on the main
comparison and benchmark scripts:

```powershell
.\.venv\Scripts\python.exe scripts\compare_methods.py --fast-preset production --atoms 1000000 --skip-direct --nq 40 --nphi 180 --nr 48 --nz 24 --qmax 2.2 --radius 20 --height 20
.\.venv\Scripts\python.exe scripts\benchmark_bottleneck.py --fast-preset production --nphi 180 --nr 48 --nz 24 --skip-nufft
.\.venv\Scripts\python.exe scripts\benchmark_bottleneck.py --fast-preset production-bandlimited --nphi 720 --nr 48 --nz 48 --skip-nufft
```

`production` expands to the C++ histogram, `hist_dtype=float32`, cubic32 angle
LUT, C++ circular solver, and auto complex dtype. `production-bandlimited` adds
`harmonic_bandlimit_margin=16`; this is approximate relative to the full
sampled circular solver and should be validated against the exact circular path
for a target q range.

`hist_dtype=uint32` preserves exact integer counts while reducing histogram
memory traffic. `hist_dtype=float32` is usually faster for the full circular
path because the angular histogram FFT can use `complex64`; counts remain exact
as long as no bin exceeds the exact integer range of float32. For high-precision
validation, keep the default `int64/complex128` path.

The C++ histogram can also replace per-atom `atan2` with a quadrant lookup table.
The default LUT mode is nearest-neighbor lookup:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_matrix.py --preset quick --hist-backend cpp --hist-dtype float32 --angle-lut-size 4096 --circular-backend cpp --complex-dtype auto
```

`--angle-lut-size 4096` is a practical starting point. On the 1M-atom synthetic
cylinder, direct C++ histogram timing improved from about `0.047 s` to
`0.025 s` for `Nphi=180`, and from about `0.100 s` to `0.042 s` for `Nphi=720`.
The added intensity error versus exact-atan2 binned circular was about `2e-5`
in that test. This is approximate binning, so keep exact `angle_lut_size=0` for
reference validation.

For a smaller table with lower angular approximation error, use the cubic mode:

```powershell
.\.venv\Scripts\python.exe scripts\compare_methods.py --atoms 1000000 --skip-direct --nq 40 --nphi 180 --nr 48 --nz 48 --qmax 2.2 --radius 20 --height 20 --hist-backend cpp --hist-dtype float32 --angle-lut-mode cubic --angle-lut-size 32 --circular-backend cpp --complex-dtype auto
```

The cubic LUT stores one quadrant of `atan(t)` plus analytic derivatives and
evaluates a cubic Hermite segment. In the 1M-atom synthetic-cylinder check,
`angle_lut_mode=cubic, angle_lut_size=16` already reduced the final intensity
error below the nearest-4096 LUT for `Nphi=180, 720, 1440`; size `32` is the
more conservative default candidate. Exact `angle_lut_size=0` should still be
used for reference runs and tolerance calibration.

If element labels are already encoded as small integer IDs, application code can
skip string mapping with `make_cylindrical_histogram_indexed(...)`. This is the
preferred production path for multi-element structures loaded from a trajectory
or structure parser. If a parser returns element labels as strings, encode them
once after loading the structure:

```python
from waxs_cake import encode_elements, make_cylindrical_histogram_indexed

element_indices, element_order = encode_elements(
    atom_elements,
    element_order=("C", "N", "O", "S"),
)
binned = make_cylindrical_histogram_indexed(
    coords,
    element_indices,
    n_elements=len(element_order),
    element_order=element_order,
    backend="cpp",
    hist_dtype="float32",
    angle_lut_mode="cubic",
    angle_lut_size=32,
)
```

For method validation against NUFFT, use the indexed path and exclude string
label parsing from the timed region. NUFFT also needs the same structure
ingestion and element/form-factor preparation, so string-to-index conversion is
tracked as preprocessing rather than as part of the WAXS cake-map method
runtime.

For repeated weighted histograms on the same atom-to-bin assignment, skip the
coordinate transform entirely with
`make_cylindrical_histogram_from_indices(...)` or
`make_cylindrical_histogram_from_flat_indices(...)`. These APIs assume the
caller has already computed the `(R, z, beta)` bin indices, so they measure the
accumulation-only cost and avoid repeated `sqrt/atan2` work.
When the flat indices come from `cylindrical_flat_indices(...)` or another
validated source, pass `validate_indices=False` to remove the extra range-check
scan in repeated accumulation loops.

To precompute those flat indices in C++, use `cylindrical_flat_indices(...)`.
Its exact mode uses the same `atan2` binning as the coordinate histogram. The
experimental `angle_lut_size` option replaces per-atom `atan2` with a quadrant
lookup table:

```python
flat = cylindrical_flat_indices(
    coords,
    n_r=48,
    n_z=24,
    n_phi=180,
    r_max=20.0,
    z_range=(-10.0, 10.0),
    backend="cpp",
    angle_lut_mode="cubic",
    angle_lut_size=32,
)
```

The LUT path is approximate; compare against exact binning for the intended
q-range before using it for final numbers.

The C++ backend can also be benchmarked directly against NumPy and numba:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_histogram_backends.py --atoms 1000000 --repeats 3 --backends numpy numba-parallel cpp --element-input indexed
```

Numba remains useful for the single-element, unweighted path. The C++ backend is
mainly for build-time compiled execution and for weighted or multi-element
histograms, where numba currently falls back to the NumPy path. Use
`--element-input strings` only when auditing preprocessing overhead, not for
NUFFT-relative method timing.

The solver also has an optional C++ circular-contraction backend. It can be
forced with `--circular-backend cpp`, and `complex_dtype=auto` uses a small
heuristic: float32 and complex64 histograms use `complex64`, while the default
integer-count validation path uses `complex128`. For repeated
form-factor/template sweeps, caching both kernels and z-reduced factors remains
the most effective path:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_reuse.py --atoms 1000000 --nq 40 --nphi 180 --nr 48 --nz 48 --qmax 2.2 --sweeps 20 --cache-kernels --cache-z --circular-backend auto
```

For fresh-structure one-shot calculations, the default path is optimized for the
exact circular solver: Jacobi-Anger cutoffs are computed lazily only when the
Jacobi/hybrid solvers are requested, the histogram angular FFT uses SciPy's FFT
backend, and the sampled circular-kernel FFT also uses SciPy's FFT backend. The
exact circular kernel is generated as `cos(theta) + i sin(theta)` instead of via
complex `exp(i theta)`, which reduces kernel-construction cost without changing
the numerical method. For the complex64 C++ path, moderate angular grids and
band-limited high-phi grids use a block-local z-reduction followed by the C++
contraction; full high-phi grids keep the fused C++ contraction to avoid extra
temporary memory traffic.

## Kernel Strategies

The exact circular solver builds
`Khat(q, R, h) = FFT_phi(exp(i q_perp R cos(phi)))` on demand. For repeated
template generation on the same `q/R/phi` grid, precompute the exact kernel FFTs:

```powershell
.\.venv\Scripts\python.exe scripts\compare_methods.py --atoms 1000000 --skip-direct --nq 40 --nphi 180 --nr 48 --nz 48 --qmax 2.2 --radius 20 --height 20 --hist-backend numba-parallel --cache-kernels
```

For an approximate faster kernel path, use a 1D interpolation table in
`x = q_perp R`:

```powershell
.\.venv\Scripts\python.exe scripts\compare_methods.py --atoms 1000000 --skip-direct --nq 40 --nphi 180 --nr 48 --nz 48 --qmax 2.2 --radius 20 --height 20 --hist-backend numba-parallel --kernel-interp-dx 0.05
```

Smaller `--kernel-interp-dx` is more accurate and more expensive. In early tests,
`dx=0.05` added roughly `1e-4` relative amplitude error versus the exact circular
kernel path on representative cases.

For high angular-resolution outputs, an optional harmonic band-limit can skip
z-reduction and contraction for modes outside
`|h| <= ceil(max(q_perp) * r_max + margin)` while still using the sampled exact
kernel FFT. This is an approximation to the exact circular path, so compare it
against the unbandlimited circular result before using it for final numbers:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_harmonic_bandlimit.py --cases base high_phi low_q --margins 8 16 24 32 48
.\.venv\Scripts\python.exe scripts\benchmark_matrix.py --preset default --harmonic-bandlimit-margin 16
```

In representative tests, margin `16` kept the added intensity error versus exact
circular below the existing binning error while improving high-phi and low-q
cases.

## Binning Frontier

The radial and axial bin counts do not need to match. Since the Ewald-ring
`q_z` sensitivity is often weaker than the in-plane `q_perp` sensitivity,
smaller `n_z` can reduce histogram and z-reduction cost with little additional
error:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_binning_frontier.py --bins 48x48 48x32 48x24 48x16 64x24 64x16 96x32 96x24
.\.venv\Scripts\python.exe scripts\benchmark_binning_frontier.py --bins 48x48 48x24 64x24 --hist-backend cpp --hist-dtype float32 --circular-backend cpp --complex-dtype auto
.\.venv\Scripts\python.exe scripts\benchmark_binning_frontier.py --harmonic-bandlimit-margin 16
```

On the 1M-atom, `Nq=40`, `Nphi=180` benchmark, `48x24` consistently reduced
runtime versus `48x48` with essentially the same NUFFT-relative intensity error
in the synthetic cylinder case. Treat this as a frontier to measure for each
target q range and sample size, not a universal default.

## Physical Scaling

Fixed-bin benchmarks are useful for optimization, but they are not sufficient
for NUFFT-relative scaling claims. If atom count grows because the simulated
water-equivalent volume grows, the box size and histogram grid must grow too.
For water density, `1,000,000` atoms correspond to about `333,333` water
molecules and a cube side length of about `21.5 nm`; `1,000,000` water
molecules would be about `31 nm`.

Use `benchmark_physical_scaling.py` to derive the simulation box and cylindrical
grid from density, q range, and target real-space bin width:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_physical_scaling.py --atoms 10000 100000 1000000 --bin-width-nm 0.1 --qmax 2.2 --q-unit inv_angstrom --dry-run
```

The script treats `--nphi-detector` as a lower bound. The timed circular output
uses the larger physical grid `n_phi` whenever `qmax * rmax` requires more
angular modes. This makes physical scaling more conservative than the earlier
fixed `48x24x180` smoke benchmarks. After the physical angular minimum is
selected, `n_phi` is rounded upward to an FFT-friendly even length so the grid
avoids large-prime transform sizes such as `1556 = 4 * 389`.

## Active RZ Sparse Path

The exact circular solver also has an experimental active-cell contraction:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_precomputed_sparse.py --atoms 1000000 --nq 40 --nphi 180 --nr 48 --nz 24 --height 20 --occupied-height 2 --hist-dtype float32 --circular-backend cpp
```

`PreparedCakePlan.circular_fft_sparse_rz()` contracts only non-empty
`(element, R, z)` beta profiles. It is exact relative to the same binned dense
circular solver. It helps only when the number of active `R/z` profiles is much
smaller than the dense `element * n_r * n_z` grid; for a fully occupied cylinder
the dense path remains faster.

## Sparse Source-Projection Path

For high-q grids with fine `z` and angular binning, use the experimental sparse
source-projection path:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_physical_scaling.py --atoms 100000 --qmax 6.3 --q-unit inv_angstrom --skip-nufft --benchmark-sparse-source-projection
```

`PreparedCakePlan.circular_fft_sparse_source_projection()` groups active
`(element, R, z, beta)` bins by `(element, R)`, projects `z` into a q-dependent
complex `(q, element, R, beta)` source row, then applies the beta FFT once per
active `(element, R)` row. It is exact relative to the same dense binned circular
solver. This path is intended for fine high-q grids where `z`/`beta` occupancy
stays sparse as the dense grid grows.

To sweep source-profile chunk sizes while reusing the same histogram and plan:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_sparse_source_projection_sweep.py --atoms 250000 500000 1000000 --chunks 8 16 32 64 128
```

The high-q synthetic water sweep favored chunk `64` for first-solve timing in
the 100k, 250k, and 500k cases, and chunk `32`/`64` in the 1M case. The default
source-profile chunk size is therefore `64`.

The sparse source projection can also be combined with the R-dependent harmonic
cutoff:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_physical_scaling.py --atoms 100000 250000 500000 1000000 --qmax 6.3 --q-unit inv_angstrom --skip-nufft --benchmark-r-dependent-cake --benchmark-sparse-source-projection --benchmark-sparse-source-r-dependent --r-dependent-margin 16 --r-dependent-cutoff-bin-size 16
```

`PreparedCakePlan.circular_fft_sparse_source_r_dependent()` first builds the
q-dependent sparse source rows, then applies the same R-dependent harmonic
cutoff used by `circular_fft_r_dependent_bandlimit()`. The current
implementation still performs a full beta FFT before cutoff, so it should be
treated as an experimental contender rather than a blanket replacement for the
R-dependent cake path. The combined path avoids the prototype's full-R kernel
chunk and cutoff-mask temporaries, and `benchmark_physical_scaling.py` can
sample first-solve RSS with `--measure-memory`. The same compact analytic
kernel used by the R-dependent cake path can be enabled with
`--r-dependent-analytic-kernel`; this reduces sampled-kernel work but does not
remove the combined path's sparse-profile cache cost on the first call. See
`docs/sparse_source_r_dependent_combined_benchmark.md` for the current scaling
and memory numbers.

For the standalone R-dependent analytic cake path, the default C++ half-spectrum
contraction now reuses an existing full circular FFT cache directly instead of
materializing a second compact positive-mode copy. In the 1M high-q benchmark
this reduced the method peak RSS from the old `254.7 MiB` measurement to
`40.5 MiB` while keeping the fastest CPU first solve. For strict working-set
limits, pass `fused_analytic_kernel=True` to
`PreparedCakePlan.circular_fft_r_dependent_bandlimit()` or
`--r-dependent-fused-analytic-kernel` to `benchmark_physical_scaling.py`; this
generates the Miller analytic kernel inside the C++ contraction and measured
`0.9 MiB` method peak RSS in the same case. `--r-dependent-r-block-size` is
available for R-slab streaming experiments, but it is not the CPU default when a
full FFT cache is already resident.

See `docs/r_dependent_analytic_final_summary.md` for the final direct scaling
comparison and default-path recommendation.

## Reuse Benchmarks

For repeated form-factor/template sweeps on the same structure and `q/R/phi`
grid, cache both the exact kernel FFTs and the z-reduced structure factors:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_reuse.py --atoms 1000000 --nq 40 --nphi 180 --nr 48 --nz 48 --qmax 2.2 --sweeps 20 --cache-kernels --cache-z
```

In a representative 1M-atom, `48x48`-bin case, 20 form-factor variants improved
from about `0.083 s/template` without reuse to about `0.016 s/template` with
kernel and z-reduction caches.

The benchmark uses `PreparedCakePlan`, which caches the histogram FFT,
`q_z z` phase table, Bessel cutoffs, angular cosine table, and Jacobi-Anger
harmonic tables. To also precompute the sampled circular-kernel FFT table, add:

```powershell
.\.venv\Scripts\python.exe scripts\compare_methods.py --atoms 1000000 --skip-direct --nq 40 --nphi 180 --nr 48 --nz 48 --qmax 2.2 --radius 20 --height 20 --cache-kernels
```

High angular-resolution case:

```powershell
.\.venv\Scripts\python.exe scripts\compare_methods.py --atoms 1000000 --skip-direct --nq 40 --nphi 720 --nr 48 --nz 48 --qmax 2.2 --radius 20 --height 20
```

## Current Interpretation

The circular FFT and Jacobi-Anger paths agree to numerical precision when they
use the same cylindrical histogram. Their shared error against direct/NUFFT is
therefore currently dominated by `(R, z, beta)` binning.

FINUFFT is a strong baseline for one-off exact evaluations. The binned methods
become interesting when the histogram can be reused, the atom count is large,
or the WAXS cake-map grid and error tolerance make the domain-specific
factorization cheaper than exact nonuniform Fourier evaluation.
