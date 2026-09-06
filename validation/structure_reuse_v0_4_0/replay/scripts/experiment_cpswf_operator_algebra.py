"""Finite-Hankel CPSWF algebra pilot with independent physical-space references.

This is an experiment adapter, not a change to public ACFO APIs. All operator
norm statements are on the explicitly retained input space. Full curves and
failed cases are saved. Derivatives are dimensionless differential actions,
not an elliptic boundary-value solver.
"""
from __future__ import annotations

import os
for _thread_key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_thread_key] = "1"

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy import linalg, special

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from waxs_cake.finite_hankel import (
    commuting_tridiagonal, gauss_unit_interval, hankel_basis_images,
    radial_jacobi_basis, x_squared_tridiagonal,
)


def norm(a):
    return float(linalg.svdvals(a)[0]) if min(a.shape) else 0.0


def relative(a, b):
    return norm(a-b) / max(norm(b), 1e-300)


def tri(d, e):
    return np.diag(d) + np.diag(e, 1) + np.diag(e, -1)


def accurate_gauss(count):
    """Golub-Welsch weights avoid endpoint loss seen in the initial pilot.

    On this local SciPy build roots_legendre(384) gave ~2e-10 orthogonality
    error for degree-128 radial polynomials. Eigenvector weights reduce it.
    Both are the same Gauss rule; this does not change the integration norm.
    """
    k = np.arange(1, count, dtype=float)
    nodes, vectors = linalg.eigh_tridiagonal(np.zeros(count), k/np.sqrt(4*k*k-1))
    return (nodes+1)/2, vectors[0]**2


def ladder_matrices(m, n):
    """Exact radial Jacobi ladder coefficients (no quadrature construction).

    Strict/non-strict upper triangular outer products. Their actions also
    admit weighted suffix sums in O(n), despite a dense stored triangle.
    """
    k = np.arange(n)
    a, b = np.sqrt(2*k+m+1), np.sqrt(2*k+m+2)
    return 2*np.triu(b[:, None]*a[None, :], 1), 2*np.triu(a[:, None]*b[None, :])


def ladder_apply(m, values, *, lower=False):
    n = values.shape[0]
    k = np.arange(n)
    a, b = np.sqrt(2*k+m+1), np.sqrt(2*k+m+2)
    weights = b if lower else a
    z = weights[:, None]*values
    tail = np.cumsum(z[::-1], axis=0)[::-1]
    if not lower:
        tail = np.vstack((tail[1:], np.zeros_like(tail[:1])))
    return 2*(a if lower else b)[:, None]*tail


def eigensystem(m, c, n):
    chi, u = linalg.eigh_tridiagonal(*commuting_tridiagonal(m, c, n))
    u *= np.sign(u[np.argmax(abs(u), axis=0), np.arange(n)])
    return chi, u


def differential_columns(m, n, x, action):
    """sqrt(r) times physical D+, D- or Laplacian of radial Jacobi columns.

    P=P_k^(0,m)(2r^2-1). Analytic polynomial derivatives avoid cancellation
    of the regular r^m branch in D+. No grid differentiation is used.
    """
    k = np.arange(n)
    t = 2*x[:, None]**2-1
    p = special.eval_jacobi(k[None, :], 0, m, t)
    pt = np.zeros_like(p)
    ptt = np.zeros_like(p)
    pt[:, 1:] = (k[1:]+m+1)/2 * special.eval_jacobi(k[None, 1:]-1, 1, m+1, t)
    ptt[:, 2:] = ((k[2:]+m+1)*(k[2:]+m+2)/4
                  * special.eval_jacobi(k[None, 2:]-2, 2, m+2, t))
    scale = np.sqrt(2*(2*k+m+1))[None, :]
    r = x[:, None]
    if action == "raise":
        return scale * 4*r**(m+1.5)*pt
    if action == "lower":
        if m < 1:
            raise ValueError("lower requires positive order")
        return scale*(2*m*r**(m-0.5)*p + 4*r**(m+1.5)*pt)
    if action == "laplacian":
        return scale*(8*(m+1)*r**(m+0.5)*pt + 16*r**(m+2.5)*ptt)
    raise ValueError(action)


def profile(name, x):
    if name == "x2":
        return x*x
    if name == "smooth_exp":
        return np.exp(-3*x*x)
    if name == "edge_erf":
        return 0.2+0.4*(1+special.erf(18*(x*x-0.72)))
    raise ValueError(name)


def physical_map(m, c, x, w, y, v):
    xy = y[:, None]*x[None, :]
    return np.sqrt(v[:, None])*c*np.sqrt(xy)*special.jv(m, c*xy)*np.sqrt(w[None, :])


def first_pass(values, tol):
    passed = np.flatnonzero(np.asarray(values) <= tol)
    return int(passed[0]) if passed.size else None


def prefix_curves(vectors, middle, downstream, reference, middle_norm):
    """No monotonicity assumption for the observed composite error."""
    coordinates = vectors.T @ middle
    left = downstream @ vectors
    approximation = np.zeros_like(reference)
    reconstructed = np.zeros_like(middle)
    final_errors, leakage = [], []
    scale = norm(reference)
    for k in range(vectors.shape[1]+1):
        if k:
            approximation += left[:, k-1:k] @ coordinates[k-1:k]
            reconstructed += vectors[:, k-1:k] @ coordinates[k-1:k]
        final_errors.append(relative(approximation, reference))
        leakage.append(norm(middle-reconstructed)/middle_norm)
    return final_errors, leakage


def run_case(m, c, design):
    start = perf_counter()
    n, q = design["dimension"], design["quadrature"]
    x, w = accurate_gauss(q)
    xa, wa = accurate_gauss(design["quadrature_audit"])
    y, v = accurate_gauss(design["output_quadrature"])
    ya, va = accurate_gauss(2*design["output_quadrature"])
    sw, swa = np.sqrt(w)[:, None], np.sqrt(wa)[:, None]
    phis = {j: radial_jacobi_basis(j, n, x) for j in (m, m+1)}
    phisa = {j: radial_jacobi_basis(j, n, xa) for j in (m, m+1)}
    qs = {j: sw*phis[j] for j in phis}
    qsa = {j: swa*phisa[j] for j in phis}
    eig = {j: eigensystem(j, c, n) for j in phis}
    h = {j: np.sqrt(v[:, None])*hankel_basis_images(j, c, n, y) for j in phis}
    ha = {j: np.sqrt(va[:, None])*hankel_basis_images(j, c, n, ya) for j in phis}
    kernel = {j: physical_map(j, c, x, w, y, v) for j in phis}
    kernela = {j: physical_map(j, c, xa, wa, y, v) for j in phis}
    kernelouta = {j: physical_map(j, c, xa, wa, ya, va) for j in phis}
    hsvd = {j: linalg.svd(h[j], full_matrices=False) for j in phis}
    sv = {j: hsvd[j][1] for j in phis}
    audit = {
        "orthogonality": max(relative(qs[j].T@qs[j], np.eye(n)) for j in phis),
        "analytic_hankel_vs_direct": max(relative(h[j], kernel[j]@qs[j]) for j in phis),
        "hankel_quadrature_doubling": max(relative(h[j], kernela[j]@qsa[j]) for j in phis),
    }
    dplus_values = sw*differential_columns(m, n, x, "raise")
    dminus_values = sw*differential_columns(m+1, n, x, "lower")
    dp, dm = ladder_matrices(m, n)
    audit["raise_algebra_vs_physical"] = relative(qs[m+1]@dp, dplus_values)
    audit["lower_algebra_vs_physical"] = relative(qs[m]@dm, dminus_values)
    audit["ladder_composition"] = relative(qs[m]@dm@dp, sw*differential_columns(m, n, x, "laplacian"))
    audit["suffix_sum_raise"] = relative(ladder_apply(m, eig[m][1]), dp@eig[m][1])
    audit["suffix_sum_lower"] = relative(ladder_apply(m, eig[m+1][1], lower=True), dm@eig[m+1][1])
    x2 = tri(*x_squared_tridiagonal(m, n))
    audit["x2_tridiagonal_vs_physical"] = relative(x2, qs[m].T@(x[:, None]**2*qs[m]))
    matrix_build_seconds = perf_counter()-start
    ranks, rows, external = [], [], []
    for tol in design["tolerances"]:
        u = eig[m][1]
        # Direct operator residual, not concentration eigenvalues alone.
        hankel_errors = [relative(h[m]@u[:, :k]@u[:, :k].T, h[m]) for k in range(n+1)]
        r = first_pass(hankel_errors, tol)
        pin = u[:, :r]
        chi_pad, u_pad = eigensystem(m, c, n+design["dimension_audit_padding"])
        low_pad = radial_jacobi_basis(m, u_pad.shape[0], xa)@u_pad[:, :r]
        low = phisa[m]@pin
        low_pad *= np.sign(np.sum(wa[:, None]*low_pad*low, axis=0))
        audit[f"padded_basis_tol_{tol}"] = relative(swa*low_pad, swa*low)
        ranks.append({"tolerance": tol, "cpswf_input_rank": r,
                      "hankel_svd_rank": int(np.count_nonzero(sv[m] > tol*sv[m][0])),
                      "hankel_error": hankel_errors[r],
                      "next_order_hankel_svd_rank": int(np.count_nonzero(sv[m+1] > tol*sv[m+1][0]))})
        for action in design["actions"]:
            out_order = m+1 if action.startswith("raise") else m
            if action == "identity":
                mid_values, mid_a = qs[m]@pin, qsa[m]@pin
                downstream = h[m]
                ref = kernel[m]@mid_values
                ref_a = kernela[m]@mid_a
                ref_outa = kernelouta[m]@mid_a
            elif action.startswith("raise"):
                mid_values = dplus_values@pin
                mid_a = swa*differential_columns(m, n, xa, "raise")@pin
                if action == "raise_once":
                    downstream = h[m+1]
                    ref = kernel[m+1]@mid_values
                    ref_a = kernela[m+1]@mid_a
                    ref_outa = kernelouta[m+1]@mid_a
                else:
                    downstream = h[m]@dm
                    full = sw*differential_columns(m, n, x, "laplacian")@pin
                    full_a = swa*differential_columns(m, n, xa, "laplacian")@pin
                    ref = kernel[m]@full
                    ref_a = kernela[m]@full_a
                    ref_outa = kernelouta[m]@full_a
            else:
                name, count = action.rsplit("_", 1)
                a, aa = profile(name, x), profile(name, xa)
                mid_values = a[:, None]*(qs[m]@pin)
                mid_a = aa[:, None]*(qsa[m]@pin)
                if count == "once":
                    downstream = h[m]
                    ref = kernel[m]@mid_values
                    ref_a = kernela[m]@mid_a
                    ref_outa = kernelouta[m]@mid_a
                else:
                    downstream = kernel[m]@(a[:, None]*qs[m])
                    ref = kernel[m]@(a[:, None]*mid_values)
                    ref_a = kernela[m]@(aa[:, None]*mid_a)
                    ref_outa = kernelouta[m]@(aa[:, None]*mid_a)
            middle = qs[out_order].T@mid_values
            if action.startswith("raise"):
                # Construct the ladder in coefficient space; quadrature is
                # retained only as an independent audit below.
                middle = dp@pin
            elif action.startswith("x2"):
                middle = x2@pin
            elif action == "identity":
                middle = pin.copy()
            middle_a = qsa[out_order].T@mid_a
            mid_norm = norm(mid_values)
            # Basis-independent floor of the finite Jacobi workspace.
            floor = norm(mid_values-qs[out_order]@middle)/mid_norm
            action_u, action_s, _ = linalg.svd(middle, full_matrices=False)
            basis_options = {"CPSWF": eig[out_order][1], "Jacobi": np.eye(n),
                             "Hankel_SVD": hsvd[out_order][2].T, "action_SVD": action_u}
            common = {"order": m, "bandwidth": c, "tolerance": tol, "action": action,
                      "input_rank": r, "intermediate_angular_order": out_order,
                      "final_angular_order": m+1 if action == "raise_once" else m,
                      "action_image_numerical_rank": int(np.count_nonzero(action_s > tol*action_s[0])),
                      "intermediate_workspace_floor": floor,
                      "reference_quadrature_error": relative(ref, ref_a),
                      "middle_quadrature_error": relative(middle, middle_a),
                      "output_norm_quadrature_error": abs(norm(ref_a)-norm(ref_outa))/norm(ref_outa),
                      "full_workspace_composite_error": relative(downstream@middle, ref_a)}
            for label, vectors in basis_options.items():
                errors, leak = prefix_curves(vectors, middle, downstream, ref_a, mid_norm)
                # Add the independently measured workspace residual to avoid
                # declaring continuum closure from a finite section alone.
                conservative_leak = [value+floor for value in leak]
                kout, kraw = first_pass(errors, tol), first_pass(conservative_leak, tol)
                fixed_k = min(r, vectors.shape[1])
                row = dict(common, basis=label, observed_required_modes=kout,
                           intermediate_required_modes=kraw,
                           fixed_input_rank_observation_error=errors[fixed_k],
                           fixed_input_rank_leakage_bound=conservative_leak[fixed_k],
                           observation_error_curve=errors,
                           intermediate_error_bound_curve=conservative_leak,
                           selected_observation_error=None if kout is None else errors[kout],
                           selected_intermediate_error=None if kout is None else conservative_leak[kout],
                           core_entries=None if kout is None else kout*r,
                           basis_entries=None if kout is None else n*kout,
                           final_error_increases=sum(errors[j+1] > errors[j]*1.01+1e-13 for j in range(len(errors)-1)))
                if kout:
                    core = vectors[:, :kout].T@middle
                    row["core_fraction_above_1e_8_relative_max"] = float(np.mean(abs(core) > np.max(abs(core))*1e-8))
                rows.append(row)
            # Analytic inputs do not depend on the selected CPSWF basis.
            # Check source truncation separately for multiplier composites.
            if action in ("identity", "smooth_exp_return", "edge_erf_return"):
                for source in ("smooth_exp", "edge_erf"):
                    f = swa[:, 0]*xa**(m+0.5)*profile(source, xa)
                    f = f[:, None]/linalg.norm(f)
                    coeff = (qsa[m]@pin).T@f
                    fp = qsa[m]@pin@coeff
                    multiplier = np.ones_like(xa) if action == "identity" else profile(action.rsplit("_", 1)[0], xa)**2
                    actual = kernela[m]@(multiplier[:, None]*f)
                    approximated = kernela[m]@(multiplier[:, None]*fp)
                    external.append({"order": m, "bandwidth": c, "tolerance": tol,
                                     "action": action, "source": source,
                                     "source_projection_error": relative(fp, f),
                                     "observation_error_from_input_projection_only": relative(approximated, actual)})
        print(f"completed m={m} c={c:g} tol={tol:g} r={r}", flush=True)
    return {"order": m, "bandwidth": c, "audits": audit, "ranks": ranks,
            "diagnostic_matrix_build_seconds": matrix_build_seconds,
            "elapsed_seconds": perf_counter()-start}, rows, external


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"reports/acfo_cpswf_operator_algebra_20260905_v2")
    parser.add_argument("--protocol", type=Path, default=ROOT/"validation_contracts/cpswf_operator_algebra_pilot_v1.json")
    args = parser.parse_args()
    design = json.loads(args.protocol.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"source_snapshot.py").write_bytes(Path(__file__).read_bytes())
    (args.output/"protocol.json").write_text(json.dumps(design, indent=2), encoding="utf-8")
    result = {"schema": design["schema"], "cases": [], "rows": [], "external_inputs": []}
    for c in design["bandwidths"]:
        for m in design["orders"]:
            case, rows, external = run_case(m, c, design)
            result["cases"].append(case)
            result["rows"].extend(rows)
            result["external_inputs"].extend(external)
            (args.output/"results.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    threshold = design["reference_threshold"]
    max_audit = max(v for case in result["cases"] for v in case["audits"].values())
    fields = ["reference_quadrature_error", "middle_quadrature_error", "output_norm_quadrature_error",
              "full_workspace_composite_error", "intermediate_workspace_floor"]
    verification = {"maximum_base_audit": max_audit,
                    "maxima": {f: max(r[f] for r in result["rows"]) for f in fields},
                    "case_count": len(result["cases"]), "row_count": len(result["rows"]),
                    "external_input_count": len(result["external_inputs"])}
    verification["all_reference_checks_pass"] = max([max_audit, *verification["maxima"].values()]) <= threshold
    (args.output/"verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    manifest = {}
    for path in [Path(__file__), args.protocol, args.output/"results.json", args.output/"verification.json"]:
        manifest[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    main()
