"""Metric-matched shared-SVD preparation for radial comparator experiments.

Source-strength and potential-density inputs have different target matrices.
The density comparator is free to project in density-input coordinates. It is
not forced into the CPSWF source-projector form A P P* D. This module prepares
one absolute-order block, not a replacement for the complete aIDT adapter.
"""
from dataclasses import dataclass
import hashlib
import json

import numpy as np
from scipy import linalg

from .radial_cpswf import (
    PreparedRadialProjection, radial_block, smallest_shared_prefix, weighted_families,
)


@dataclass(frozen=True)
class PreparedSharedSvdRadial:
    action: PreparedRadialProjection
    input_scale: np.ndarray
    report: dict

    @classmethod
    def build(cls, *, radius, q_perp, order, weights, tolerance=1e-8,
              metric="source_strength", area_scale=None, max_dense_bytes=128*1024**2):
        r, q = np.asarray(radius, dtype=float), np.asarray(q_perp, dtype=float)
        if r.ndim != 1 or q.ndim != 1 or not r.size or not q.size:
            raise ValueError("nonempty radial and frequency vectors required")
        if metric not in ("source_strength", "density"):
            raise ValueError("unknown input metric")
        if not 0 < tolerance < 1 or not weights:
            raise ValueError("tolerance and calibration weights required")
        # This named-array guard is deliberately not a peak-memory estimate.
        if r.size*q.size*8 > max_dense_bytes:
            raise MemoryError("dense radial calibration exceeds named-array budget")
        scale = np.ones(r.size)
        if metric == "density":
            scale = np.array(area_scale, dtype=float, copy=True)
            if scale.shape != r.shape or np.any(scale <= 0) or not np.all(np.isfinite(scale)):
                raise ValueError("density metric requires positive finite cell areas")
        matrix = radial_block(q, r, order)*scale[None, :]
        normalized, norms = weighted_families(matrix, weights)
        _, _, vh = linalg.svd(np.vstack(normalized), full_matrices=False)
        basis = vh.T
        rank, checked = smallest_shared_prefix(normalized, basis, tolerance)
        if rank is None:
            raise ArithmeticError("shared SVD failed the common residual criterion")
        selected = basis[:, :rank]
        residuals = checked[rank]
        orth = float(linalg.norm(selected.T@selected-np.eye(rank), 2)) if rank else 0.0
        if max(residuals) > tolerance+5e-13 or orth > 1e-10:
            raise ArithmeticError("radial preparation acceptance failed")
        action = PreparedRadialProjection.build(matrix, selected, weights)
        scale.setflags(write=False)
        return cls(action, scale, {"metric": metric, "rank": rank, "order": int(order),
            "relative_spectral_residuals": residuals, "normalization_scales": norms.tolist(),
            "orthogonality_error": orth, "tolerance": tolerance,
            "projection_coordinates": "density" if metric == "density" else "source_strength"})

    def forward(self, x):
        return self.action.forward(x)

    def adjoint(self, y):
        return self.action.adjoint(y)

    def normal(self, x, profile=0):
        return self.action.normal(x, profile)

    def as_source_strength_block(self):
        """Compatible with ModalRadialMap, which multiplies by area first.

        For density SVD: left=AD V, right=V* D^-1. Its adjoint reverses the
        same factors. The returned source-coordinate basis is not orthonormal;
        PreparedRadialProjection actions require real factors, not orthogonality.
        """
        p = np.array(self.action.basis/self.input_scale[:, None], copy=True)
        p.setflags(write=False)
        return PreparedRadialProjection(p, self.action.reduced, self.action.normal_cores)

    @property
    def retained_bytes(self):
        return int(self.action.all_operator_bytes+self.input_scale.nbytes)


@dataclass(frozen=True)
class PreparedSharedSvdAidt:
    """Independent physical-chain construction with an in-memory radial cache."""
    operator: object
    radial_key: str
    report: dict


def prepare_shared_svd_aidt(geometry, *, rank_metric="density", radial_tolerance=1e-8,
                            radial_weights=None, radial_cache=None, limits=None, q_chunk=32):
    # Shared physical primitives are called from geometry; no CPSWF object is
    # constructed, inspected or needed for the SVD preparation path.
    from .aidt_acfo import aidt_transfer_functions_on_points
    from .radial_aidt import ModalRadialMap, PhysicalModalTransfer, PreparedRadialAidt
    from .prepared_cpswf_aidt import PreparationLimits

    weights = [np.ones(len(geometry.q_perp))] if radial_weights is None else radial_weights
    if not weights or rank_metric not in ("density", "source_strength") or not 0 < radial_tolerance < 1:
        raise ValueError("invalid metric, weights or tolerance")
    limits = PreparationLimits() if limits is None else limits
    descriptor = dict(radius=geometry.radius_um.tolist(), q=geometry.q_perp.tolist(),
        area=geometry.area_scale.tolist(), orders=sorted(set(abs(m) for m in geometry.modes)),
        metric=rank_metric, tolerance=radial_tolerance, dtype="complex128",
        weights=[np.asarray(w, dtype=float).tolist() for w in weights])
    key = hashlib.sha256(json.dumps(descriptor, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if radial_cache is not None and (not isinstance(radial_cache, PreparedSharedSvdAidt) or radial_cache.radial_key != key):
        raise ValueError("radial cache mismatch")
    if radial_cache is None:
        blocks, records = {}, []
        for m in descriptor["orders"]:
            block = PreparedSharedSvdRadial.build(radius=geometry.radius_um, q_perp=geometry.q_perp,
                order=m, weights=weights, tolerance=radial_tolerance, metric=rank_metric,
                area_scale=geometry.area_scale, max_dense_bytes=limits.max_radial_matrix_bytes)
            blocks[m] = block.as_source_strength_block()
            records.append(block.report)
    else:
        blocks = radial_cache.operator.radial.blocks
        records = radial_cache.report["blocks"]
    freq, phi = geometry.radial_frequency_um_inv, geometry.phi
    # The transfer budget is checked before the physical arrays are allocated.
    transfer_bytes = 16*(len(geometry.source_na_xy)*len(freq)*len(phi)*len(geometry.depth_values_um)*2+len(phi)*len(geometry.modes))
    if transfer_bytes > limits.max_transfer_bytes:
        raise MemoryError("physical transfer exceeds named-array budget")
    h, a, _ = aidt_transfer_functions_on_points(
        frequency_x=freq[:, None]*np.cos(phi), frequency_y=freq[:, None]*np.sin(phi),
        source_na_xy=geometry.source_na_xy, wavelength_um=geometry.wavelength_um,
        medium_index=geometry.medium_index, objective_na=geometry.objective_na,
        depth_values_um=geometry.depth_values_um, dz_um=geometry.dz_um)
    transfer = PhysicalModalTransfer.build(geometry.modes, phi, h, a, q_chunk=q_chunk)
    radial = ModalRadialMap.build(geometry.modes, geometry.area_scale, blocks, len(geometry.depth_values_um))
    op = PreparedRadialAidt(radial, transfer)
    rng = np.random.default_rng(613)
    x = rng.normal(size=radial.input_shape)+1j*rng.normal(size=radial.input_shape)
    y = rng.normal(size=transfer.data_shape)+1j*rng.normal(size=transfer.data_shape)
    fx, ay = op.forward(x), op.adjoint(y)
    dot = float(abs(np.vdot(fx, y)-np.vdot(x, ay))/max(np.linalg.norm(fx)*np.linalg.norm(y), 1e-300))
    if not np.isfinite(dot) or dot > 1e-12:
        raise ArithmeticError("assembled SVD adjoint check failed")
    return PreparedSharedSvdAidt(op, key, dict(blocks=records, cache_hit=radial_cache is not None,
        adjoint_dot_error=dot, radial_retained_bytes=radial.cache_bytes,
        transfer_retained_bytes=transfer.cache_bytes, metric=rank_metric,
        scope="Independent cold construction; explicit selected modes; in-memory radial reuse. Disk-cache timing is not implemented."))
