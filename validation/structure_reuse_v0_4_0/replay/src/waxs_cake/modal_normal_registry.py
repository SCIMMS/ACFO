"""Exact-state cost registry and immutable contexts for the bounded G planner.

One-time copies/checks permit repeat preparation without hashing mutable caller
arrays. Cost reuse is exact-state and declared-runtime scoped; no interpolation
or automatic calibration occurs on a miss. Old planner/backends remain intact.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from types import MappingProxyType

import numpy as np
import scipy
from scipy import fft

from .modal_normal_planner import (
    ARMS, CostRecord, Epoch, ResourceBudget, array_digest, choose_normal,
    describe_transfer,
)
from .modal_normal_structure import StructuredModalNormal, _angular_gram_chunk, _weights
from .modal_normal_full_support import PreparedFftComposedNormal
from .radial_aidt import PhysicalModalTransfer
from .radial_modal_normal import ExplicitTransferNormal, PreparedTransferNormal

SCHEMA = "acfo-exact-state-g-cost-registry-v1"
THREAD_KEYS = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "BLIS_NUM_THREADS")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest_json(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def declared_threads():
    return tuple(os.environ.get(key) for key in THREAD_KEYS)


def runtime_stamp(revision="default"):
    """Declared environment identity, not live native BLAS thread introspection.

    Callers changing native thread pools outside environment variables must use
    a new revision/context. Importing old costs needs a separate provenance audit.
    """
    names = ("modal_normal_registry.py", "modal_normal_planner.py", "modal_normal_structure.py",
             "modal_normal_full_support.py", "radial_modal_normal.py", "radial_aidt.py")
    return {"node": platform.node(), "system": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(), "python": sys.version, "numpy": np.__version__,
            "scipy": scipy.__version__, "declared_threads": list(declared_threads()),
            "runtime_revision": str(revision),
            "implementation_sha256": {name: file_digest(Path(__file__).with_name(name)) for name in names}}


def immutable_array(value, dtype=None):
    a = np.ascontiguousarray(value, dtype=dtype)
    # Immutable bytes base prevents both assignment and setflags(write=True).
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


@dataclass(frozen=True)
class FrozenNormalContext:
    transfer: PhysicalModalTransfer
    phi: np.ndarray
    weights: object
    spec: object
    operator_digest: str
    weight_digests: object
    runtime_json: str
    runtime_id: str
    arrays_bytes: int
    _memory: object
    _lags: np.ndarray
    _modes: np.ndarray
    _bins: np.ndarray
    _phases: np.ndarray
    _fft_length: int
    _threads: tuple
    _nonce: object = field(default_factory=object, repr=False, compare=False)

    @classmethod
    def create(cls, transfer, phi, weights, family, *, runtime_revision="default", max_snapshot_bytes=512*2**20):
        if not isinstance(transfer, PhysicalModalTransfer) or not weights:
            raise ValueError("physical transfer and named weights required")
        if (transfer.ptf.ndim != 4 or transfer.ptf.shape != transfer.atf.shape or
                transfer.angular.shape != (transfer.ptf.shape[2], len(transfer.modes)) or
                any(x.dtype != np.dtype(complex) or not np.all(np.isfinite(x))
                    for x in (transfer.angular, transfer.ptf, transfer.atf))):
            raise ValueError("finite complex128 transfer arrays with compatible shapes required")
        if int(max_snapshot_bytes) != max_snapshot_bytes or max_snapshot_bytes <= 0:
            raise ValueError("positive snapshot byte budget required")
        checked = {name: _weights(transfer, w) for name, w in weights.items()}
        if any(not isinstance(name, str) or not name for name in checked):
            raise ValueError("nonempty weight names required")
        # Guard the large copies before allocation; this is not total process RSS.
        size = transfer.cache_bytes+np.asarray(phi).nbytes+sum(w.nbytes for w in checked.values())
        if 2*size > max_snapshot_bytes:
            raise MemoryError("source plus immutable snapshot exceeds copy budget")
        e, h, a, angles = (immutable_array(x) for x in (transfer.angular, transfer.ptf, transfer.atf, phi))
        copied = PhysicalModalTransfer(tuple(transfer.modes), e, h, a, transfer.q_chunk)
        spec, info, digest = describe_transfer(copied, angles, family)
        wcopy = {name: immutable_array(w) for name, w in checked.items()}
        modes = immutable_array(spec.modes, np.int64)
        lags = immutable_array(info["lags"], np.int64)
        bins = immutable_array(modes % spec.n_phi)
        phases = immutable_array(np.exp(1j*modes*info["phi_offset"]))
        arrays = copied.cache_bytes+angles.nbytes+sum(w.nbytes for w in wcopy.values())
        arrays += sum(x.nbytes for x in (modes, lags, bins, phases))
        memory = {}
        for arm in ARMS:
            m = spec.memory(arm)
            # Context retains all registered weights and shared metadata. Only
            # the chosen core adds arrays; normal-owned counts remain comparable.
            resident = arrays+m["core_bytes"]
            m["context_plus_core_bytes"] = resident
            m["estimated_working_array_bytes"] += max(0, resident-m["refresh_ready_bytes"])
            memory[arm] = MappingProxyType(m)
        runtime = canonical(runtime_stamp(runtime_revision))
        return cls(copied, angles, MappingProxyType(wcopy), spec, digest,
                   MappingProxyType({n: array_digest(w) for n, w in wcopy.items()}), runtime,
                   hashlib.sha256(runtime.encode()).hexdigest(), arrays, MappingProxyType(memory),
                   lags, modes, bins, phases, info["fft_length"], declared_threads())

    @property
    def registry_key(self):
        return digest_json([self.spec.key, self.operator_digest, self.runtime_id])

    def check_runtime(self):
        if declared_threads() != self._threads:
            raise ValueError("declared thread environment changed; new context required")

    def memory(self, arm):
        return dict(self._memory[arm])

    def _prepare(self, arm, weight_name):
        """Use structure checked at construction; retain existing apply kernels."""
        t, w = self.transfer, self.weights[weight_name]
        if arm == "composed":
            action = ExplicitTransferNormal(t, w)
            return PreparedTransferNormal(t.modes, action, arm, w.nbytes)
        if arm == "dense":
            action = t.compile_normal(w)
            return PreparedTransferNormal(t.modes, action, arm, action.cache_bytes)
        if arm == "fft_composed":
            action = PreparedFftComposedNormal(t.modal_shape, t.ptf, t.atf, w,
                                              self._bins, self._phases, t.q_chunk)
            return PreparedTransferNormal(t.modes, action, arm, action.incremental_array_bytes)
        nq, _, nz, nc = t.modal_shape
        p, k = self.spec.n_phi, nz*nc
        lags = self._lags if arm == "lag_direct" else immutable_array([], np.int64)
        length = {"lag_direct": len(lags), "lag_fft": self._fft_length, "angular_fft": p}[arm]
        core = np.empty((nq, length, k, k), dtype=complex)
        for start in range(0, nq, t.q_chunk):
            sl = slice(start, min(start+t.q_chunk, nq))
            c = _angular_gram_chunk(t, w, sl)
            if arm == "angular_fft":
                core[sl] = c
            else:
                coeff = fft.ifft(c, axis=1)
                if arm == "lag_direct":
                    core[sl] = coeff[:, lags % p]*np.exp(1j*lags*self.phi[0])[None, :, None, None]
                else:
                    span = self.memory(arm)["mode_span"]
                    shifts = np.arange(1-span, span); d = -shifts
                    embedded = np.zeros((sl.stop-sl.start, length, k, k), dtype=complex)
                    embedded[:, shifts % length] = coeff[:, d % p]*np.exp(1j*d*self.phi[0])[None, :, None, None]
                    core[sl] = fft.fft(embedded, axis=1)
        core.setflags(write=False)
        action = StructuredModalNormal(t.modal_shape, self._modes, core, arm, p,
                                       float(self.phi[0]), t.q_chunk, lags, self._fft_length)
        return PreparedTransferNormal(t.modes, action, arm, action.cache_bytes)


class RegistryMiss(LookupError):
    pass


@dataclass(frozen=True)
class RegistryChoice:
    selected_arm: str
    payload_json: str
    _context_nonce: object = field(repr=False, compare=False)
    _registry_nonce: object = field(repr=False, compare=False)
    _generation: int = field(repr=False)

    @property
    def payload(self):
        return json.loads(self.payload_json)


class _CachedSpecView:
    def __init__(self, context):
        self.key = context.spec.key
        self._context = context

    def memory(self, arm):
        return self._context.memory(arm)


class CostRegistry:
    """Persistent exact-state entries; per-instance workload decisions are cached."""
    def __init__(self):
        self._entries = {}
        self._choices = {}
        self._issued = set()
        self._nonce = object()
        self._generation = 0

    def register(self, context, records, *, provenance):
        context.check_runtime()
        records = tuple(records)
        if not records or not provenance:
            raise ValueError("nonempty cost evidence and provenance required")
        if context.registry_key in self._entries:
            raise ValueError("entry exists; create a new registry version instead of overwriting")
        pairs = set()
        for r in records:
            if (not isinstance(r, CostRecord) or r.problem_key != context.spec.key or
                    r.hardware != context.runtime_id or r.weight not in context.weights or
                    (r.arm, r.weight) in pairs):
                raise ValueError("cost scope/runtime/weight/duplicate mismatch")
            pairs.add((r.arm, r.weight))
        # JSON snapshot eliminates mutable caller aliases in provenance and rows.
        entry = {"spec": asdict(context.spec), "operator_digest": context.operator_digest,
                 "runtime_json": context.runtime_json, "runtime_id": context.runtime_id,
                 "weight_digests": dict(context.weight_digests),
                 "records": [asdict(r) for r in records], "provenance": provenance}
        self._entries[context.registry_key] = canonical(entry)
        self._generation += 1; self._choices.clear(); self._issued.clear()

    def lookup(self, context, epochs):
        context.check_runtime()
        text = self._entries.get(context.registry_key)
        if text is None:
            raise RegistryMiss("exact geometry/support/runtime has no cost entry; no automatic calibration")
        entry = json.loads(text)
        for name in {e.weight for e in epochs}:
            if (name not in context.weight_digests or
                    entry["weight_digests"].get(name) != context.weight_digests[name]):
                raise RegistryMiss("data-weight content changed or unregistered")
        return tuple(CostRecord(**r) for r in entry["records"])

    def select(self, context, epochs, budget, *, tolerance=1e-11, tie_fraction=.03):
        context.check_runtime()
        epochs = tuple(epochs)
        if not epochs or any(not isinstance(e, Epoch) for e in epochs):
            raise ValueError("nonempty Epoch workload required")
        if not isinstance(budget, ResourceBudget):
            raise ValueError("explicit resource budget required")
        # Include contents even for a new context with the same operator key.
        signature = canonical([context.registry_key, dict(context.weight_digests),
                               [asdict(e) for e in epochs], asdict(budget), tolerance, tie_fraction])
        cached = self._choices.get(signature)
        if cached is not None:
            payload, arm = cached
        else:
            records = self.lookup(context, epochs)
            decision = choose_normal(_CachedSpecView(context), records, epochs, budget,
                        hardware=context.runtime_id, tolerance=tolerance, tie_fraction=tie_fraction,
                        operator_digest=context.operator_digest, weight_digests=dict(context.weight_digests))
            payload = canonical(asdict(replace(decision, spec=context.spec)))
            arm = decision.selected_arm
            self._choices[signature] = (payload, arm)
        self._issued.add((context._nonce, payload, arm))
        return RegistryChoice(arm, payload, context._nonce, self._nonce, self._generation)

    def prepare(self, context, choice, weight_name):
        context.check_runtime()
        if (not isinstance(choice, RegistryChoice) or choice._context_nonce is not context._nonce or
                choice._registry_nonce is not self._nonce or choice._generation != self._generation or
                (context._nonce, choice.payload_json, choice.selected_arm) not in self._issued):
            raise ValueError("choice belongs to a different context or registry version")
        payload = choice.payload
        if (payload["selected_arm"] != choice.selected_arm or
                weight_name not in {e["weight"] for e in payload["epochs"]}):
            raise ValueError("invalid choice or undeclared workload weight")
        selected = next(c for c in payload["candidates"] if c["arm"] == choice.selected_arm)
        if not selected["eligible"]:
            raise ValueError("ineligible preparation")
        plan = context._prepare(choice.selected_arm, weight_name)
        if plan.cache_bytes != selected["normal_owned_bytes"]:
            raise RuntimeError("prepared retained-array contract mismatch")
        return plan

    def save(self, path):
        payload = {"schema": SCHEMA, "entries": {k: json.loads(v) for k, v in self._entries.items()}}
        with Path(path).open("x", encoding="utf-8") as f:
            json.dump({"payload": payload, "payload_sha256": digest_json(payload)}, f, indent=2, allow_nan=False)
            f.write("\n")

    @classmethod
    def load(cls, path):
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        payload = document["payload"]
        if document["payload_sha256"] != digest_json(payload) or payload["schema"] != SCHEMA:
            raise ValueError("registry checksum/schema mismatch")
        registry = cls()
        for key, entry in payload["entries"].items():
            from .modal_normal_planner import ProblemSpec
            spec_data = dict(entry["spec"]); spec_data["modes"] = tuple(spec_data["modes"])
            spec = ProblemSpec(**spec_data)
            if not entry["records"] or not entry["provenance"]:
                raise ValueError("nonempty cost evidence and provenance required")
            if (key != digest_json([spec.key, entry["operator_digest"], entry["runtime_id"]]) or
                    hashlib.sha256(entry["runtime_json"].encode()).hexdigest() != entry["runtime_id"]):
                raise ValueError("registry key/runtime mismatch")
            pairs = set()
            for row in entry["records"]:
                r = CostRecord(**row)
                if (r.problem_key != spec.key or r.hardware != entry["runtime_id"] or
                        r.weight not in entry["weight_digests"] or (r.arm, r.weight) in pairs):
                    raise ValueError("registry evidence mismatch")
                pairs.add((r.arm, r.weight))
            registry._entries[key] = canonical(entry)
        registry._generation = 1
        return registry
