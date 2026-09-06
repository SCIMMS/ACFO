"""Geometry-to-operator preparation for the selected-mode CPSWF/aIDT prototype.

Radial tolerance is an operator-norm contract in the declared input metric.
It is not a relative-error bound for an ill-conditioned full physical chain.
No experimental data, saved basis or manually chosen radial rank is required.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import linalg

from .aidt_acfo import aidt_transfer_functions_on_points
from .radial_aidt import ModalRadialMap, PhysicalModalTransfer, PreparedRadialAidt, PreparedModalNormal
from .radial_cpswf import PreparedRadialProjection, radial_block, sampled_cpswf_basis, smallest_shared_prefix

SCHEMA = "prepared-cpswf-aidt-radial-v1"


class PreparationError(ValueError):
    """Geometry, accuracy contract or cached artifact cannot be accepted."""


class PreparationBudgetError(MemoryError):
    """Requested representation exceeds an explicit preparation limit."""


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _key(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _array_hash(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256(_json({"shape": a.shape, "dtype": a.dtype.str}).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _vector(value, name, *, positive=False, nonnegative=False):
    a = np.array(value, dtype=float, copy=True)
    if a.ndim != 1 or a.size == 0 or not np.all(np.isfinite(a)):
        raise PreparationError(f"{name} must be a finite nonempty vector")
    if positive and np.any(a <= 0) or nonnegative and np.any(a < 0):
        raise PreparationError(f"{name} has an invalid sign")
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class PreparationLimits:
    max_basis_dimension: int = 4096
    max_radial_matrix_bytes: int = 128*2**20
    max_transfer_bytes: int = 512*2**20
    max_normal_bytes: int = 128*2**20

    def __post_init__(self):
        if any(isinstance(v, bool) or int(v) != v or v < 1 for v in asdict(self).values()):
            raise PreparationError("preparation limits must be positive integers")


@dataclass(frozen=True)
class AidtCpswfGeometry:
    r_edges_um: np.ndarray
    radial_frequency_um_inv: np.ndarray
    n_phi: int
    modes: tuple[int, ...]
    depth_values_um: np.ndarray
    dz_um: float
    source_na_xy: np.ndarray
    wavelength_um: float
    medium_index: float
    objective_na: float
    angular_padding: int = 48
    data_bandlimit: int | None = None

    def __post_init__(self):
        edges = _vector(self.r_edges_um, "r_edges", nonnegative=True)
        freq = _vector(self.radial_frequency_um_inv, "radial frequencies", nonnegative=True)
        depth = _vector(self.depth_values_um, "depth")
        if edges.size < 2 or edges[0] != 0 or np.any(np.diff(edges) <= 0):
            raise PreparationError("strictly increasing radial edges starting at zero required")
        if np.any(np.diff(freq) <= 0) or freq[-1] <= 0:
            raise PreparationError("strictly increasing frequencies with positive bandwidth required")
        modes = tuple(self.modes)
        if (not modes or len(set(modes)) != len(modes)
                or any(isinstance(m, (bool, np.bool_)) or int(m) != m for m in modes)):
            raise PreparationError("explicit unique integer angular modes required")
        if isinstance(self.n_phi, bool) or int(self.n_phi) != self.n_phi or self.n_phi < 2:
            raise PreparationError("n_phi must be an integer >=2")
        if isinstance(self.angular_padding, bool) or int(self.angular_padding) != self.angular_padding or self.angular_padding < 0:
            raise PreparationError("angular padding must be a nonnegative integer")
        data_h = max(abs(m) for m in modes) if self.data_bandlimit is None else self.data_bandlimit
        if isinstance(data_h, bool) or int(data_h) != data_h or data_h < max(abs(m) for m in modes):
            raise PreparationError("data_bandlimit must cover every requested mode")
        # Sampling check only. It never silently changes the user's mode subset.
        h_req = max(int(np.ceil(2*np.pi*freq[-1]*edges[-1])), int(data_h))+int(self.angular_padding)
        if self.n_phi <= 2*h_req:
            raise PreparationError(f"angular under-resolution: n_phi={self.n_phi}, requires > {2*h_req}")
        source = np.array(self.source_na_xy, dtype=float, copy=True)
        if source.ndim != 2 or source.shape[1] != 2 or source.shape[0] == 0 or not np.all(np.isfinite(source)):
            raise PreparationError("source_na_xy must have finite shape (sources,2)")
        if np.any(np.sum(source*source, axis=1) >= 1):
            raise PreparationError("grazing/evanescent source unsupported by the reused physical model")
        for name in ("dz_um", "wavelength_um", "medium_index", "objective_na"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise PreparationError(f"{name} must be finite and positive")
        source.setflags(write=False)
        for name, val in (("r_edges_um", edges), ("radial_frequency_um_inv", freq),
                          ("depth_values_um", depth), ("source_na_xy", source),
                          ("modes", tuple(int(m) for m in modes)), ("data_bandlimit", int(data_h)),
                          ("n_phi", int(self.n_phi)), ("angular_padding", int(self.angular_padding))):
            object.__setattr__(self, name, val)
        for name in ("dz_um", "wavelength_um", "medium_index", "objective_na"):
            object.__setattr__(self, name, float(getattr(self, name)))

    @property
    def radius_um(self):
        return (self.r_edges_um[1:]+self.r_edges_um[:-1])/2

    @property
    def area_scale(self):
        return np.pi*np.diff(self.r_edges_um**2)

    @property
    def q_perp(self):
        return 2*np.pi*self.radial_frequency_um_inv

    @property
    def phi(self):
        return (np.arange(self.n_phi)+.5)*2*np.pi/self.n_phi

    def descriptor(self):
        return {key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in asdict(self).items()}


@dataclass(frozen=True)
class PreparedCpswfNormal:
    """Standalone normal evaluator: owns no physical transfer or data cache."""
    radial: ModalRadialMap
    modal_normal: PreparedModalNormal
    key: str
    preparation_seconds: float

    def apply(self, potential):
        return self.radial.adjoint(self.modal_normal.apply(self.radial.forward(potential)))

    @property
    def retained_bytes(self):
        return self.radial.cache_bytes+self.modal_normal.cache_bytes


@dataclass(frozen=True)
class PreparedCpswfAidt:
    geometry: AidtCpswfGeometry
    operator: PreparedRadialAidt
    report: dict
    radial_descriptor: dict
    limits: PreparationLimits

    @property
    def input_shape(self):
        return self.operator.radial.input_shape

    @property
    def data_shape(self):
        return self.operator.transfer.data_shape

    def forward(self, potential):
        return self.operator.forward(potential)

    def adjoint(self, data):
        return self.operator.adjoint(data)

    def normal(self, potential, weights=None):
        return self.operator.normal(potential, _data_weights(weights, self.data_shape))

    def prepare_normal(self, weights=None):
        w = _data_weights(weights, self.data_shape)
        shape = self.operator.radial.modal_shape
        required = 16*shape[0]*int(np.prod(shape[1:]))**2
        if required > self.limits.max_normal_bytes:
            raise PreparationBudgetError(f"normal requires {required} retained bytes; limit={self.limits.max_normal_bytes}. Modes unchanged.")
        start = perf_counter()
        normal = self.operator.transfer.compile_normal(w)
        key = _key({"geometry": self.geometry.descriptor(), "radial_key": self.report["radial_key"],
                    "weight": "identity" if w is None else _array_hash(w), "schema": SCHEMA})
        return PreparedCpswfNormal(self.operator.radial, normal, key, perf_counter()-start)

    def save_radial_cache(self, path):
        """Export calibrated radial arrays only; physical transfer remains rebuildable.

        Single NPZ, no pickle, per-array digests, create-only. Loading validates
        the full radial descriptor, precision, shapes, hashes and orthogonality.
        """
        arrays = {}
        for m, block in self.operator.radial.blocks.items():
            arrays[f"basis_{m}"] = block.basis
            arrays[f"reduced_{m}"] = block.reduced
        metadata = {"schema": SCHEMA, "descriptor": self.radial_descriptor,
                    "radial_key": self.report["radial_key"], "blocks": self.report["blocks"],
                    "array_hashes": {k: _array_hash(v) for k, v in arrays.items()}}
        with Path(path).open("xb") as handle:
            np.savez_compressed(handle, metadata_json=np.asarray(_json(metadata)), **arrays)


def _data_weights(weights, shape):
    if weights is None:
        return None
    if np.iscomplexobj(weights):
        raise PreparationError("data weights must be real")
    w = np.array(weights, dtype=float, copy=True)
    if w.shape != shape or not np.all(np.isfinite(w)) or np.any(w < 0):
        raise PreparationError("data weights must match data shape and be finite nonnegative")
    return w


def _load_radial_cache(path, descriptor):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        if (metadata["schema"] != SCHEMA or metadata["radial_key"] != _key(descriptor)
                or _json(metadata["descriptor"]) != _json(descriptor)):
            raise PreparationError("radial cache geometry/metric/tolerance/version mismatch")
        blocks = {}
        nr, nq = len(descriptor["radius_um"]), len(descriptor["q_perp"])
        for entry in metadata["blocks"]:
            m, k = entry["mode_abs"], entry["rank"]
            b, c = data[f"basis_{m}"], data[f"reduced_{m}"]
            for key, a, shape in ((f"basis_{m}", b, (nr, k)), (f"reduced_{m}", c, (nq, k))):
                if (a.dtype != np.complex128 or a.shape != shape or np.any(a.imag)
                        or not np.all(np.isfinite(a)) or _array_hash(a) != metadata["array_hashes"][key]):
                    raise PreparationError("radial cache array integrity/shape/precision mismatch")
            if linalg.norm(b.T@b-np.eye(k), 2) > 1e-10:
                raise PreparationError("radial cache basis orthogonality failed")
            b.setflags(write=False)
            c.setflags(write=False)
            blocks[m] = PreparedRadialProjection(b, c, ())
        if sorted(blocks) != descriptor["orders"]:
            raise PreparationError("radial cache is missing requested orders")
        return blocks, metadata["blocks"]


def prepare_cpswf_aidt(geometry: AidtCpswfGeometry, *, radial_tolerance=1e-8,
                      rank_metric="density", radial_weights=None, radial_cache=None,
                      limits=None, q_chunk=25):
    """Build a checked forward/adjoint pair from physical geometry.

    Explicit angular modes define the operator domain; automatic *radial*
    rank selection never removes angular modes. Default density-rank selection
    uses a conservative monotone source-space bound then verifies ||Delta A D||.
    ``source_strength`` allows exact reproduction of the earlier radial pilot.
    """
    if not isinstance(geometry, AidtCpswfGeometry):
        raise PreparationError("an AidtCpswfGeometry instance is required")
    if not np.isfinite(radial_tolerance) or not 1e-12 <= radial_tolerance < 1:
        raise PreparationError("radial tolerance must be finite in [1e-12,1)")
    if rank_metric not in ("density", "source_strength"):
        raise PreparationError("rank_metric must be density or source_strength")
    if isinstance(q_chunk, bool) or int(q_chunk) != q_chunk or q_chunk < 1:
        raise PreparationError("q_chunk must be a positive integer")
    limits = PreparationLimits() if limits is None else limits
    r, q, area = geometry.radius_um, geometry.q_perp, geometry.area_scale
    nr, nq = len(r), len(q)
    weights = [np.ones(nq)] if radial_weights is None else list(radial_weights)
    if not weights:
        raise PreparationError("at least one radial error profile is required")
    clean_weights = []
    for w in weights:
        if np.iscomplexobj(w):
            raise PreparationError("radial weights must be real")
        w = _vector(w, "radial weights", nonnegative=True)
        if w.shape != (nq,) or not np.any(w > 0):
            raise PreparationError("radial weights must match q and be nonzero")
        clean_weights.append(w)
    orders = sorted(set(abs(m) for m in geometry.modes))
    dimensions = {m: max(nr, int(np.ceil(q[-1]*geometry.r_edges_um[-1]))+m+48) for m in orders}
    if max(dimensions.values())+32 > limits.max_basis_dimension:
        raise PreparationBudgetError("CPSWF basis dimension (including padding audit) exceeds limit; no rank/mode truncation applied")
    if 8*nq*nr > limits.max_radial_matrix_bytes:
        raise PreparationBudgetError("dense radial calibration matrix exceeds limit")
    transfer_bytes = 16*(2*geometry.source_na_xy.shape[0]*nq*geometry.n_phi*len(geometry.depth_values_um)
                         +geometry.n_phi*len(geometry.modes))
    if transfer_bytes > limits.max_transfer_bytes:
        raise PreparationBudgetError(f"physical transfer requires {transfer_bytes} retained bytes; limit={limits.max_transfer_bytes}")
    descriptor = {"schema": SCHEMA, "radius_um": r.tolist(), "q_perp": q.tolist(),
                  "outer_radius_um": float(geometry.r_edges_um[-1]), "area_scale": area.tolist(),
                  "orders": orders, "rank_metric": rank_metric, "tolerance": radial_tolerance,
                  "weights": [w.tolist() for w in clean_weights], "dtype": "complex128",
                  "dimension_padding": 48, "dimension_audit_increment": 32,
                  "qr_dependence_tolerance": 1e-12, "roundoff_slack": 5e-13}
    all_start = perf_counter()
    radial_start = perf_counter()
    if radial_cache is not None:
        blocks, records = _load_radial_cache(radial_cache, descriptor)
    else:
        blocks, records = {}, []
        for m in orders:
            started = perf_counter()
            a = radial_block(q, r, m)
            b, info = sampled_cpswf_basis(r, geometry.r_edges_um[-1], q[-1], m, dimensions[m])
            padded, _ = sampled_cpswf_basis(r, geometry.r_edges_um[-1], q[-1], m, dimensions[m]+32)
            if b.shape[1] == 0:
                raise PreparationError(f"CPSWF sampled basis unresolved at mode {m}")
            weighted = [np.sqrt(w)[:, None]*a for w in clean_weights]
            source_scales = [float(linalg.norm(a_w, 2)) for a_w in weighted]
            density_scales = [float(linalg.norm(a_w*area, 2)) for a_w in weighted]
            if min(source_scales+density_scales) <= 0:
                raise PreparationError(f"zero/underflowed block at mode {m}; relative rank undefined")
            if rank_metric == "source_strength":
                search = [aw/s for aw, s in zip(weighted, source_scales)]
            else:
                # ||A_w(I-P)D|| <= ||A_w(I-P)|| ||D||. This is monotone in
                # nested P. Direct density residuals need not be monotone.
                search = [aw*(max(area)/s) for aw, s in zip(weighted, density_scales)]
            k, checked = smallest_shared_prefix(search, b, radial_tolerance)
            if k is None or padded.shape[1] < k:
                raise PreparationError(f"radial tolerance unresolved at mode {m}; no fallback rank silently accepted")
            p = b[:, :k]
            approximation = (a@p)@p.T
            residual = a-approximation
            source_errors = [float(linalg.norm(np.sqrt(w)[:, None]*residual, 2)/s) for w, s in zip(clean_weights, source_scales)]
            density_errors = [float(linalg.norm(np.sqrt(w)[:, None]*residual*area, 2)/s) for w, s in zip(clean_weights, density_scales)]
            dp = padded[:, :k]
            delta = approximation-(a@dp)@dp.T
            metric_scale = np.ones(nr) if rank_metric == "source_strength" else area
            scales = source_scales if rank_metric == "source_strength" else density_scales
            pad_error = max(float(linalg.norm(np.sqrt(w)[:, None]*delta*metric_scale, 2)/s) for w, s in zip(clean_weights, scales))
            orth = float(linalg.norm(p.T@p-np.eye(k), 2))
            errors = source_errors if rank_metric == "source_strength" else density_errors
            if max(errors) > radial_tolerance+5e-13 or orth > 1e-10 or pad_error > radial_tolerance/4+5e-13:
                raise PreparationError(f"radial accuracy/padding/orthogonality check failed at mode {m}")
            blocks[m] = PreparedRadialProjection.build(a, p, [])
            records.append({"mode_abs": m, "rank": k, "dimension": dimensions[m],
                            "resolved_prefix": info["resolved_prefix"], "orthogonality_error": orth,
                            "source_relative_operator_errors": source_errors, "density_relative_operator_errors": density_errors,
                            "padding_change_relative_operator_error": pad_error,
                            "rank_search_bound": checked[k], "previous_prefix_bound": checked.get(k-1),
                            "preparation_seconds": perf_counter()-started})
    radial_seconds = perf_counter()-radial_start
    transfer_start = perf_counter()
    freq, phi = geometry.radial_frequency_um_inv, geometry.phi
    h, a, _ = aidt_transfer_functions_on_points(
        frequency_x=freq[:, None]*np.cos(phi), frequency_y=freq[:, None]*np.sin(phi),
        source_na_xy=geometry.source_na_xy, wavelength_um=geometry.wavelength_um,
        medium_index=geometry.medium_index, objective_na=geometry.objective_na,
        depth_values_um=geometry.depth_values_um, dz_um=geometry.dz_um)
    transfer = PhysicalModalTransfer.build(geometry.modes, phi, h, a, q_chunk=q_chunk)
    radial = ModalRadialMap.build(geometry.modes, area, blocks, len(geometry.depth_values_um))
    operator = PreparedRadialAidt(radial, transfer)
    transfer_seconds = perf_counter()-transfer_start
    # Fixed independent complex probes verify the assembled API's Euclidean
    # adjoint. They do not substitute for the spectral radial contract above.
    rng = np.random.default_rng(613)
    x = rng.normal(size=radial.input_shape)+1j*rng.normal(size=radial.input_shape)
    y = rng.normal(size=transfer.data_shape)+1j*rng.normal(size=transfer.data_shape)
    fx, ay = operator.forward(x), operator.adjoint(y)
    dot = float(abs(np.vdot(fx, y)-np.vdot(x, ay))/max(np.linalg.norm(fx)*np.linalg.norm(y), 1e-300))
    if not np.isfinite(dot) or dot > 1e-12:
        raise PreparationError("assembled adjoint test failed; prepared object withheld")
    report = {"schema": SCHEMA, "radial_key": _key(descriptor), "geometry_key": _key(geometry.descriptor()),
              "cache_hit": radial_cache is not None, "rank_metric": rank_metric,
              "radial_tolerance": radial_tolerance, "blocks": records, "adjoint_dot_error": dot,
              "radial_preparation_or_load_seconds": radial_seconds, "transfer_preparation_seconds": transfer_seconds,
              "total_preparation_seconds": perf_counter()-all_start,
              "radial_retained_bytes": radial.cache_bytes, "transfer_retained_bytes": transfer.cache_bytes,
              "selected_modes": list(geometry.modes),
              "angular_sampling_required_half_band": max(int(np.ceil(q[-1]*geometry.r_edges_um[-1])), geometry.data_bandlimit)+geometry.angular_padding,
              "scope": "Explicit selected angular domain; automatically prepared radial basis/rank. Radial spectral tolerance is not a full-chain relative-error guarantee. Dense calibration matrices and SVD residual checks are preparation costs. No matrix-free preparation or peak-memory claim."}
    return PreparedCpswfAidt(geometry, operator, report, descriptor, limits)
