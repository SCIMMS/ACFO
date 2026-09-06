"""Bounded, evidence-backed realization selection for the existing q-local G.

Known mode-difference structure is checked by the existing analyzer. Cost
records are explicitly G-only and hardware/problem scoped. This module neither
changes support nor discovers identities from arbitrary operator graphs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import time

import numpy as np
from scipy import fft

from .modal_normal_structure import analyze_modal_normal, prepare_structured_normal, _weights
from .radial_aidt import PhysicalModalTransfer
from .radial_modal_normal import PreparedTransferNormal, prepare_transfer_normal

ARMS = ("composed", "dense", "lag_direct", "lag_fft", "fft_composed", "angular_fft")


def array_digest(*arrays):
    h = hashlib.sha256()
    for value in arrays:
        a = np.ascontiguousarray(value)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(memoryview(a).cast("B"))
    return h.hexdigest()


@dataclass(frozen=True)
class ProblemSpec:
    modes: tuple[int, ...]
    n_q: int
    n_phi: int
    n_source: int
    n_depth: int
    q_chunk: int
    family: str

    def __post_init__(self):
        if (not isinstance(self.modes, tuple) or not self.modes or
                len(set(self.modes)) != len(self.modes) or
                any(isinstance(m, bool) or int(m) != m for m in self.modes)):
            raise ValueError("nonempty unique integer mode tuple required")
        for value in (self.n_q, self.n_phi, self.n_source, self.n_depth, self.q_chunk):
            if isinstance(value, bool) or int(value) != value or value <= 0:
                raise ValueError("positive integer dimensions required")
        if len(set(m % self.n_phi for m in self.modes)) != len(self.modes):
            raise ValueError("aliased modes; support will not be changed")
        if not self.family:
            raise ValueError("explicit physical family required")

    @property
    def key(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    def memory(self, arm):
        """Exact retained counts plus a conservative *array* working estimate.

        Working estimate is an engineering guard, not a proven process-RSS bound.
        Common transfer remains resident for weight refresh in this lifecycle.
        """
        if arm not in ARMS:
            raise ValueError("unknown realization")
        q, p, s, k, m = self.n_q, self.n_phi, self.n_source, 2*self.n_depth, len(self.modes)
        chunk = min(q, self.q_chunk)
        d = len({n-v for n in self.modes for v in self.modes})
        span = max(self.modes)-min(self.modes)+1
        length = fft.next_fast_len(2*span-1)
        h = 16*s*q*p*k
        e = 16*p*m
        w = 8*s*q*p
        transfer = h+e
        core = {"composed": 0, "dense": 16*q*(m*k)**2,
                "lag_direct": 16*q*d*k*k, "lag_fft": 16*q*length*k*k,
                "fft_composed": 0, "angular_fft": 16*q*p*k*k}[arm]
        if arm == "composed":
            owned, extra = transfer+w, w
        elif arm == "fft_composed":
            owned, extra = h+w+24*m, w+24*m
        else:
            owned = core+(0 if arm == "dense" else 8*m+8*d*(arm == "lag_direct"))
            extra = owned
        lift = 16*s*p*m*k
        c = 16*chunk*p*k*k
        workspace = {
            "composed": 16*(8*chunk*p*k+4*s*q*p)+e,
            "dense": 3*lift+32*(m*k)**2,
            "lag_direct": 4*c+16*chunk*m*k*k,
            "lag_fft": 4*c+48*chunk*length*k*k,
            "fft_composed": 16*(8*chunk*p*k+4*s*chunk*p),
            "angular_fft": 3*c+16*8*chunk*p*k,
        }[arm]
        refresh = transfer+extra
        working = math.ceil(1.25*(refresh+workspace+32*q*m*k))
        return {"core_bytes": core, "normal_owned_bytes": owned,
                "refresh_ready_bytes": refresh, "estimated_working_array_bytes": working,
                "unique_lags": d, "mode_span": span}


def describe_transfer(transfer, phi, family):
    if not isinstance(transfer, PhysicalModalTransfer):
        raise TypeError("only the declared PhysicalModalTransfer model is supported")
    info = analyze_modal_normal(transfer, phi)
    spec = ProblemSpec(tuple(transfer.modes), transfer.modal_shape[0], info["n_phi"],
                       transfer.ptf.shape[0], transfer.modal_shape[2], transfer.q_chunk, family)
    digest = array_digest(np.asarray(spec.modes), phi, transfer.ptf, transfer.atf)
    return spec, info, digest


@dataclass(frozen=True)
class CostRecord:
    problem_key: str
    hardware: str
    arm: str
    weight: str
    setup_seconds: float
    apply_seconds: float
    tested_relative_error: float
    source: str
    scope: str = "G_only"
    kind: str = "saved_measurement"

    def __post_init__(self):
        values = (self.setup_seconds, self.apply_seconds, self.tested_relative_error)
        if (self.arm not in ARMS or not self.weight or not self.source or
                not self.hardware or not self.problem_key or
                any(not math.isfinite(v) or v < 0 for v in values)):
            raise ValueError("finite nonnegative, identified G cost record required")
        if self.scope != "G_only":
            raise ValueError("R*GR timings cannot enter the G-only cost model")


@dataclass(frozen=True)
class Epoch:
    weight: str
    calls: int

    def __post_init__(self):
        if not self.weight or isinstance(self.calls, bool) or int(self.calls) != self.calls or self.calls < 1:
            raise ValueError("named weight and positive integer call count required")


@dataclass(frozen=True)
class ResourceBudget:
    core_bytes: int
    normal_owned_bytes: int
    working_array_bytes: int

    def __post_init__(self):
        if any(isinstance(v, bool) or int(v) != v or v <= 0 for v in asdict(self).values()):
            raise ValueError("positive integer resource budgets required")


class NoFeasiblePlan(RuntimeError):
    def __init__(self, candidates):
        super().__init__("no supported, evidenced realization fits; modes and tolerance unchanged")
        self.candidates = candidates


@dataclass(frozen=True)
class NormalDecision:
    spec: ProblemSpec
    selected_arm: str
    hardware: str
    operator_digest: str | None
    weight_digests: dict[str, str]
    candidates: tuple[dict, ...]
    selection_seconds: float
    external_planning_seconds: float
    budget: ResourceBudget
    epochs: tuple[Epoch, ...]
    tolerance: float

    @property
    def selected(self):
        return next(c for c in self.candidates if c["arm"] == self.selected_arm)


def choose_normal(spec, records, epochs, budget, *, hardware, tolerance=1e-11,
                  tie_fraction=.03, external_planning_seconds=0., operator_digest=None,
                  weight_digests=None):
    """Select one realization for a fixed-geometry, single-active-weight lifecycle.

    Each change of weight requires complete G preparation; unchanged consecutive
    epochs reuse it. Geometry changes require a new decision. Error values are
    evidence screening, not a spectral error certificate for arbitrary inputs.
    """
    start = time.perf_counter()
    if (not epochs or any(not isinstance(e, Epoch) for e in epochs) or
            not math.isfinite(tolerance) or tolerance <= 0 or
            not math.isfinite(tie_fraction) or not 0 <= tie_fraction <= 1 or
            not math.isfinite(external_planning_seconds) or external_planning_seconds < 0):
        raise ValueError("invalid workload, accuracy or selection policy")
    lookup = {}
    for r in records:
        if r.problem_key != spec.key or r.hardware != hardware:
            continue
        key = (r.arm, r.weight)
        if key in lookup:
            raise ValueError("aggregate repeated cost records before selection")
        lookup[key] = r
    candidates = []
    for arm in ARMS:
        mem = spec.memory(arm)
        reasons = []
        for field, limit in (("core_bytes", budget.core_bytes),
                             ("normal_owned_bytes", budget.normal_owned_bytes),
                             ("estimated_working_array_bytes", budget.working_array_bytes)):
            if mem[field] > limit:
                reasons.append(field+" exceeds budget")
        total = 0.; setups = 0; previous = None; sources = []
        for epoch in epochs:
            r = lookup.get((arm, epoch.weight))
            if r is None:
                reasons.append("missing G cost evidence for "+epoch.weight)
                continue
            sources.append({"source": r.source, "kind": r.kind})
            if r.tested_relative_error > tolerance:
                reasons.append("tested accuracy exceeds requested tolerance")
            if epoch.weight != previous:
                total += r.setup_seconds; setups += 1
            total += epoch.calls*r.apply_seconds
            previous = epoch.weight
        candidates.append({"arm": arm, **mem, "eligible": not reasons,
                           "reasons": sorted(set(reasons)), "setup_count": setups,
                           "predicted_operator_seconds": total if not reasons else None,
                           "evidence": sources})
    feasible = [c for c in candidates if c["eligible"]]
    if not feasible:
        raise NoFeasiblePlan(candidates)
    best = min(c["predicted_operator_seconds"] for c in feasible)
    tied = [c for c in feasible if c["predicted_operator_seconds"] <= best*(1+tie_fraction)]
    chosen = min(tied, key=lambda c: (c["normal_owned_bytes"], c["predicted_operator_seconds"], c["arm"]))
    elapsed = time.perf_counter()-start
    for c in feasible:
        c["predicted_total_seconds"] = c["predicted_operator_seconds"]+elapsed+external_planning_seconds
        c["within_tie_band"] = c in tied
    return NormalDecision(spec, chosen["arm"], hardware, operator_digest,
                          dict(weight_digests or {}), tuple(candidates), elapsed,
                          external_planning_seconds, budget, tuple(epochs), tolerance)


def prepare_realization(arm, transfer, phi, weights, max_core_bytes):
    """Named backend construction, also used by independent comparator workers."""
    if arm in ("lag_direct", "lag_fft"):
        action = prepare_structured_normal(transfer, phi, weights,
                    representation=arm, max_core_bytes=max_core_bytes)
        return PreparedTransferNormal(tuple(transfer.modes), action, arm, action.cache_bytes)
    names = {"composed": "explicit_composed", "dense": "dense_gram",
             "fft_composed": "fft_composed", "angular_fft": "angular_gram"}
    if arm not in names:
        raise ValueError("unknown realization")
    return prepare_transfer_normal(transfer, phi, weights, representation=names[arm],
                                   max_core_bytes=max_core_bytes)


def prepare_decision(decision, transfer, phi, weights, *, weight_name, hardware):
    """Construct only the selected backend; reject stale transfer/weight evidence.

    The returned PreparedTransferNormal can also bind a compatible radial map.
    That does not turn G-only cost predictions into R*GR predictions.
    """
    spec, _, digest = describe_transfer(transfer, phi, decision.spec.family)
    w = _weights(transfer, weights)
    if (hardware != decision.hardware or spec != decision.spec or
            decision.operator_digest is None or digest != decision.operator_digest or
            decision.weight_digests.get(weight_name) != array_digest(w) or
            weight_name not in {e.weight for e in decision.epochs}):
        raise ValueError("stale/undeclared geometry, support, hardware or data weight; replan required")
    if decision.selected_arm not in ARMS or not decision.selected["eligible"]:
        raise ValueError("invalid selected realization")
    plan = prepare_realization(decision.selected_arm, transfer, phi, w, decision.budget.core_bytes)
    if plan.cache_bytes != decision.spec.memory(decision.selected_arm)["normal_owned_bytes"]:
        raise RuntimeError("retained-array contract disagrees with prepared object")
    return plan


def change_policy(changed):
    """Explicit reuse boundary, independent of a generic symbolic graph system."""
    fields = set(changed)
    if fields <= {"input_values"}:
        return "reuse_prepared_G"
    if fields <= {"input_values", "data_weight"}:
        return "recheck_cost_evidence_and_rebuild_G; radial_basis_reusable"
    if fields <= {"input_values", "radial_basis", "radial_rank", "density_metric"}:
        return "reuse_G_if_modal_interface_matches; rebind_and_revalidate_RstarGR"
    return "new_geometry_or_support_decision_required"
