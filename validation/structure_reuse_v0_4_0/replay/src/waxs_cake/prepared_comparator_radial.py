"""Matched radial-only preparation/cache adapters for the prespecified comparison.

CPSWF follows the existing geometry API's sufficient-rank and +32 audit.
Shared SVD uses its own metric-matched input subspace. Both return source-
coordinate factors, share serialization/checks, and leave physical transfer
construction outside radial timing. No full normal cores are retained here.
"""
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import hashlib
import json

import numpy as np
from scipy import linalg
from .radial_cpswf import (PreparedRadialProjection, radial_block, sampled_cpswf_basis,
                           smallest_shared_prefix, weighted_families)


def array_hash(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode()+a.tobytes()).hexdigest()


def descriptor(g, weights, metric, tolerance, orders):
    return dict(schema="matched-radial-cache-v1", radius=g.radius_um.tolist(),
                outer_radius=float(g.r_edges_um[-1]), q=g.q_perp.tolist(), area=g.area_scale.tolist(),
                orders=list(orders), metric=metric, tolerance=tolerance, dtype="complex128",
                weights=[np.asarray(w).tolist() for w in weights], dimension_padding=48,
                dimension_audit_increment=32)


@dataclass(frozen=True)
class PreparedComparisonRadial:
    method: str
    descriptor: dict
    blocks: dict
    records: list

    @property
    def retained_bytes(self):
        return sum(b.forward_bytes for b in self.blocks.values())

    def save(self, path):
        arrays = {}
        for m, b in self.blocks.items():
            arrays[f"basis_{m}"] = b.basis
            arrays[f"reduced_{m}"] = b.reduced
        metadata = dict(method=self.method, descriptor=self.descriptor, records=self.records,
                        hashes={k: array_hash(a) for k, a in arrays.items()})
        with Path(path).open("xb") as f:
            # Uncompressed NPZ: identical policy, filesystem page-cache state reported.
            np.savez(f, metadata=np.asarray(json.dumps(metadata, allow_nan=False)), **arrays)

    @classmethod
    def load(cls, path, expected, method):
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["metadata"]))
            if meta["descriptor"] != expected or meta["method"] != method:
                raise ValueError("radial descriptor mismatch")
            nr, nq = len(expected["radius"]), len(expected["q"])
            blocks = {}
            if sorted(r["order"] for r in meta["records"]) != sorted(expected["orders"]):
                raise ValueError("radial order mismatch")
            for rec in meta["records"]:
                m, k = rec["order"], rec["rank"]
                p, c = z[f"basis_{m}"], z[f"reduced_{m}"]
                for name, a, shape in ((f"basis_{m}", p, (nr, k)), (f"reduced_{m}", c, (nq, k))):
                    if (a.dtype != np.complex128 or a.shape != shape or np.any(a.imag)
                            or not np.all(np.isfinite(a)) or array_hash(a) != meta["hashes"][name]):
                        raise ValueError("radial cache integrity mismatch")
                metric_basis = p*np.asarray(expected["area"])[:, None] if method == "svd" and expected["metric"] == "density" else p
                if linalg.norm(metric_basis.T@metric_basis-np.eye(k), 2) > 1e-10:
                    raise ValueError("radial cache orthogonality mismatch")
                p.setflags(write=False)
                c.setflags(write=False)
                blocks[m] = PreparedRadialProjection(p, c, ())
        return cls(method, expected, blocks, meta["records"])


def prepare_radial(g, weights, metric="density", tolerance=1e-8, method="cpswf", orders=None):
    if method not in ("cpswf", "svd") or metric not in ("source_strength", "density"):
        raise ValueError("unknown method/metric")
    orders = sorted(set(abs(m) for m in g.modes)) if orders is None else list(orders)
    r, q, area = g.radius_um, g.q_perp, g.area_scale
    if r.size*q.size*8 > 128*2**20:
        raise MemoryError("radial matrix budget exceeded")
    blocks, records = {}, []
    for m in orders:
        all_start = perf_counter()
        tick = perf_counter()
        a = radial_block(q, r, m)
        times = {"bessel_calibration": perf_counter()-tick}
        scale = area if metric == "density" else np.ones(r.size)
        if method == "cpswf":
            tick = perf_counter()
            dim = max(r.size, int(np.ceil(q[-1]*g.r_edges_um[-1]))+m+48)
            if dim+32 > 4096:
                raise MemoryError("basis dimension budget exceeded")
            b, info = sampled_cpswf_basis(r, g.r_edges_um[-1], q[-1], m, dim)
            padded, _ = sampled_cpswf_basis(r, g.r_edges_um[-1], q[-1], m, dim+32)
            times["basis_and_padding_basis"] = perf_counter()-tick
            tick = perf_counter()
            weighted = [np.sqrt(w)[:, None]*a for w in weights]
            src_norm = [float(linalg.norm(aw, 2)) for aw in weighted]
            den_norm = [float(linalg.norm(aw*area, 2)) for aw in weighted]
            norms = src_norm if metric == "source_strength" else den_norm
            if min(src_norm+den_norm) <= 0:
                raise ArithmeticError("zero block normalization")
            search = [aw/s for aw, s in zip(weighted, src_norm)] if metric == "source_strength" else [aw*(max(area)/s) for aw, s in zip(weighted, den_norm)]
            times["normalization"] = perf_counter()-tick
            tick = perf_counter()
            k, checked = smallest_shared_prefix(search, b, tolerance)
            times["rank_search"] = perf_counter()-tick
            if k is None or padded.shape[1] < k:
                raise ArithmeticError(f"unresolved CPSWF order {m}")
            tick = perf_counter()
            p = b[:, :k]
            approximation = (a@p)@p.T
            errors = [float(linalg.norm(np.sqrt(w)[:, None]*(a-approximation)*scale, 2)/s) for w, s in zip(weights, norms)]
            dp = padded[:, :k]
            delta = approximation-(a@dp)@dp.T
            pad_error = max(float(linalg.norm(np.sqrt(w)[:, None]*delta*scale, 2)/s) for w, s in zip(weights, norms))
            orth = float(linalg.norm(p.T@p-np.eye(k), 2))
            times["assessment"] = perf_counter()-tick
            if max(errors) > tolerance+5e-13 or orth > 1e-10 or pad_error > tolerance/4+5e-13:
                raise ArithmeticError(f"CPSWF acceptance failed {m}: residual={errors}, padding={pad_error}")
            tick = perf_counter()
            block = PreparedRadialProjection.build(a, p, [])
            times["reduced_factors"] = perf_counter()-tick
            extra = dict(dimension=int(dim), resolved_prefix=info["resolved_prefix"], padding_error=pad_error,
                         previous_bound=checked.get(k-1))
        else:
            tick = perf_counter()
            target = a*scale[None, :]
            normalized, norms = weighted_families(target, weights)
            times["normalization"] = perf_counter()-tick
            tick = perf_counter()
            _, _, vh = linalg.svd(np.vstack(normalized), full_matrices=False)
            b = vh.T
            times["svd_basis"] = perf_counter()-tick
            tick = perf_counter()
            k, checked = smallest_shared_prefix(normalized, b, tolerance)
            times["rank_search"] = perf_counter()-tick
            if k is None:
                raise ArithmeticError("unresolved shared SVD")
            tick = perf_counter()
            p = b[:, :k]
            errors = checked[k]
            orth = float(linalg.norm(p.T@p-np.eye(k), 2))
            if max(errors) > tolerance+5e-13 or orth > 1e-10:
                raise ArithmeticError("SVD acceptance failed")
            times["assessment"] = perf_counter()-tick
            tick = perf_counter()
            source_p = np.asarray(p/scale[:, None], dtype=complex)
            reduced = np.asarray(target@p, dtype=complex)
            source_p.setflags(write=False)
            reduced.setflags(write=False)
            block = PreparedRadialProjection(source_p, reduced, ())
            times["reduced_factors"] = perf_counter()-tick
            extra = dict(previous_bound=checked.get(k-1), padding_error=None)
        blocks[m] = block
        records.append(dict(order=int(m), rank=int(k), metric=metric, residuals=errors,
                            orthogonality=orth, stage_seconds=times, preparation_seconds=perf_counter()-all_start,
                            retained_bytes=block.forward_bytes, **extra))
    return PreparedComparisonRadial(method, descriptor(g, weights, metric, tolerance, orders), blocks, records)
