"""Structure-aware, calibrated cost prediction for the existing finite G model.

All six kernels and the immutable context are reused unchanged. Predictions are
restricted to a declared physical family and the calibration dimension box on
the same native runtime. They are estimates, not exact-state timing records or
accuracy certificates. A selected action is probe-checked before public use.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy import fft
from scipy.optimize import nnls

from .modal_normal_planner import ARMS, Epoch, NoFeasiblePlan, ProblemSpec, ResourceBudget
from .modal_normal_registry import FrozenNormalContext, canonical, digest_json, file_digest

SCHEMA = "acfo-structure-aware-g-cost-model-v1"
DESCRIPTOR_FIELDS = ("n_q", "n_phi", "n_source", "n_depth", "q_chunk", "n_modes", "mode_span")


def native_runtime(context):
    """Require live BLAS/OpenMP introspection; never silently infer native threads."""
    from threadpoolctl import threadpool_info
    context.check_runtime()
    pools = sorted((dict(p) for p in threadpool_info()), key=lambda p: p.get("filepath", ""))
    if not pools or any(not isinstance(p.get("num_threads"), int) for p in pools):
        raise RuntimeError("native thread-pool inspection unavailable")
    return {"context_runtime": json.loads(context.runtime_json), "native_pools": pools,
            "predictor_sha256": file_digest(__file__), "device": "CPU", "dtype": "complex128"}


def describe_spec(spec):
    """Finite support descriptors; no inference about continuous qR convergence."""
    modes = sorted(spec.modes)
    differences = {n-m for m in modes for n in modes}
    gaps = np.diff(modes)
    step = math.gcd(*(int(v) for v in gaps)) if gaps.size else 0
    span = modes[-1]-modes[0]+1
    return {**{key: getattr(spec, key) for key in DESCRIPTOR_FIELDS[:5]},
            "n_modes": len(modes), "mode_span": span, "unique_lags": len(differences),
            "mode_step_gcd": step, "regular_spacing": bool(gaps.size == 0 or np.all(gaps == gaps[0])),
            "fft_length": fft.next_fast_len(2*span-1), "channels": 2*spec.n_depth,
            "angular_gram_rank_upper_bound": min(spec.n_source, 2*spec.n_depth),
            "q_blocks": math.ceil(spec.n_q/spec.q_chunk),
            "reindexing": "descriptive only; existing FFT kernels use the original mode span"}


def cost_features(spec, arm, phase):
    """Named, nonnegative work proxies for the unchanged implementation.

    These count leading contraction/transform work and loop overhead, not CPU
    FLOPs or guaranteed complexity of every library implementation.
    """
    if arm not in ARMS or phase not in ("setup", "apply"):
        raise ValueError("known arm and setup/apply phase required")
    d = describe_spec(spec)
    q, p, s, k, m, blocks, length = (spec.n_q, spec.n_phi, spec.n_source,
        2*spec.n_depth, len(spec.modes), d["q_blocks"], d["fft_length"])
    if phase == "setup":
        if arm in ("composed", "fft_composed"):
            return {"object_overhead": 1.}
        if arm == "dense":
            return {"q_loops": q, "lift_entries": q*s*p*m*k, "gram_work": q*s*p*(m*k)**2}
        features = {"source_chunk_loops": blocks*s, "angular_gram_work": q*p*s*k*k}
        if arm in ("lag_direct", "lag_fft"):
            features["angular_fft_work"] = q*k*k*p*math.log2(p)
        if arm == "lag_fft":
            features["lag_fft_work"] = q*k*k*length*math.log2(length)
        features["retained_entries"] = q*k*k*{"lag_direct": d["unique_lags"],
            "lag_fft": length, "angular_fft": p}[arm]
        return features
    if arm == "dense":
        return {"call_overhead": 1., "matrix_entries": q*(m*k)**2, "vector_entries": q*m*k}
    if arm == "lag_direct":
        return {"mode_chunk_loops": blocks*m, "gather_contract_work": q*m*m*k*k,
                "lag_entries": q*d["unique_lags"]*k*k}
    if arm == "composed":
        return {"chunk_loops": blocks, "angular_lift_work": q*p*m*k,
                "factor_work": q*p*s*k, "modal_entries": q*m*k}
    n = length if arm == "lag_fft" else p
    return {"chunk_loops": blocks, "fft_work": q*k*n*math.log2(n),
            "pointwise_work": q*n*(s*k if arm == "fft_composed" else k*k),
            "modal_entries": q*m*k}


def _spec(data):
    data = dict(data)
    data["modes"] = tuple(data["modes"])
    return ProblemSpec(**data)


class PredictionUnavailable(ValueError):
    pass


@dataclass(frozen=True)
class ShapeCostModel:
    payload_json: str

    @property
    def payload(self):
        return json.loads(self.payload_json)

    @property
    def model_id(self):
        return digest_json(self.payload)

    @classmethod
    def fit(cls, rows, runtime, *, calibration_relative_limit=1e-11):
        """Fit per-arm relative-error NNLS; validation data are never accepted here.

        Rows are caller-supplied measured calibration evidence. Role tags and
        checksums support provenance; they cannot authenticate external data.
        """
        rows = json.loads(canonical(list(rows)))
        if not rows or not runtime or not np.isfinite(calibration_relative_limit) or calibration_relative_limit <= 0:
            raise ValueError("calibration evidence, runtime and positive tolerance required")
        family = _spec(rows[0]["spec"]).family
        runtime_id = digest_json(runtime)
        pairs = set()
        for r in rows:
            spec = _spec(r["spec"])
            pair = (spec.key, r["arm"], r["weight"])
            if (r.get("role") != "calibration" or r.get("scope") != "G_only" or
                    spec.family != family or r["runtime_id"] != runtime_id or r["arm"] not in ARMS or
                    not r.get("source") or not r.get("weight") or not r.get("passed") or pair in pairs):
                raise ValueError("invalid/duplicate calibration role, scope, runtime or source")
            if (any(not np.isfinite(r[key]) or r[key] <= 0 for key in ("setup_seconds", "apply_seconds")) or
                    not np.isfinite(r["relative_error"]) or not 0 <= r["relative_error"] <= calibration_relative_limit):
                raise ValueError("invalid timing or failed calibration accuracy")
            pairs.add(pair)
        descriptors = [describe_spec(_spec(r["spec"])) for r in rows]
        bounds = {key: [min(d[key] for d in descriptors), max(d[key] for d in descriptors)] for key in DESCRIPTOR_FIELDS}
        fits = {}
        for arm in ARMS:
            selected = [r for r in rows if r["arm"] == arm]
            if len({_spec(r["spec"]).key for r in selected}) < 3:
                continue
            fits[arm] = {}
            for phase in ("setup", "apply"):
                values = [cost_features(_spec(r["spec"]), arm, phase) for r in selected]
                names = list(values[0])
                x = np.array([[v[n] for n in names] for v in values], dtype=float)
                y = np.array([r[phase+"_seconds"] for r in selected])
                scale = np.maximum(x.max(axis=0), 1.)
                # Unit response makes this relative rather than absolute least squares.
                design = (x/scale)/y[:, None]
                norm = np.maximum(np.linalg.norm(design, axis=0), 1e-300)
                coefficient, _ = nnls(design/norm, np.ones(y.size), maxiter=1000)
                coefficient = coefficient/norm
                prediction = (x/scale)@coefficient
                fits[arm][phase] = {"names": names, "scale": scale.tolist(),
                    "coefficients": coefficient.tolist(), "calibration_ratio": (prediction/y).tolist()}
        if not fits:
            raise ValueError("at least three distinct calibration domains per fitted arm required")
        return cls(canonical({"schema": SCHEMA, "family": family, "runtime": runtime,
            "runtime_id": runtime_id, "bounds": bounds, "fits": fits, "rows": rows,
            "calibration_relative_limit": calibration_relative_limit,
            "method": "fixed analytical work proxies; relative NNLS, nonnegative coefficients",
            "scope": "G_only; dimension-box interpolation of costs, no error certificate"}))

    def predict(self, spec, arm, runtime):
        p = self.payload
        if spec.family != p["family"] or digest_json(runtime) != p["runtime_id"]:
            raise PredictionUnavailable("new physical family or native runtime requires calibration")
        d = describe_spec(spec)
        if any(not lo <= d[key] <= hi for key, (lo, hi) in p["bounds"].items()):
            raise PredictionUnavailable("outside calibration dimension box; no automatic extrapolation")
        if arm not in p["fits"]:
            raise PredictionUnavailable("missing arm calibration")
        result = {}
        for phase, fit in p["fits"][arm].items():
            features = cost_features(spec, arm, phase)
            if list(features) != fit["names"]:
                raise ValueError("feature schema mismatch")
            value = sum(features[n]/s*c for n, s, c in zip(fit["names"], fit["scale"], fit["coefficients"]))
            if not np.isfinite(value) or value <= 0:
                raise PredictionUnavailable("nonpositive/nonfinite cost estimate")
            result[phase+"_seconds"] = value
        return result

    def save(self, path):
        with Path(path).open("x", encoding="utf-8") as f:
            json.dump({"payload": self.payload, "sha256": self.model_id}, f, indent=2, allow_nan=False)
            f.write("\n")

    @classmethod
    def load(cls, path):
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        p = document["payload"]
        if p["schema"] != SCHEMA or document["sha256"] != digest_json(p):
            raise ValueError("model schema/checksum mismatch")
        rebuilt = cls.fit(p["rows"], p["runtime"], calibration_relative_limit=p["calibration_relative_limit"])
        if rebuilt.payload_json != canonical(p):
            raise ValueError("model differs from its calibration evidence")
        return rebuilt


@dataclass(frozen=True)
class PredictiveChoice:
    payload_json: str
    _context_nonce: object = field(repr=False, compare=False)
    _planner_nonce: object = field(repr=False, compare=False)

    @property
    def payload(self):
        return json.loads(self.payload_json)

    @property
    def selected_arm(self):
        return self.payload["selected_arm"]


@dataclass(frozen=True)
class VerifiedNormal:
    prepared: object
    verification_json: str
    preparation_seconds: float
    verification_seconds: float

    def apply(self, values):
        # No selection, hashing, calibration or reference action on this path.
        return self.prepared.action.apply(values)

    def bind_radial(self, radial):
        return self.prepared.bind_radial(radial)


class StructureAwarePlanner:
    def __init__(self, model):
        if not isinstance(model, ShapeCostModel):
            raise TypeError("a calibrated shape model is required")
        self._model = model
        self._nonce = object()
        self._issued = set()

    def select(self, context, epochs, budget, *, tolerance=1e-11, tie_fraction=.03):
        start = time.perf_counter()
        if not isinstance(context, FrozenNormalContext) or not isinstance(budget, ResourceBudget):
            raise TypeError("checked immutable context and explicit resource budget required")
        epochs = tuple(epochs)
        if (not epochs or any(not isinstance(e, Epoch) or e.weight not in context.weights for e in epochs) or
                not np.isfinite(tolerance) or tolerance <= 0 or
                not np.isfinite(tie_fraction) or not 0 <= tie_fraction <= 1):
            raise ValueError("declared workload and finite positive accuracy required")
        runtime = native_runtime(context)
        candidates = []
        setups = sum(i == 0 or epoch.weight != epochs[i-1].weight for i, epoch in enumerate(epochs))
        calls = sum(e.calls for e in epochs)
        for arm in ARMS:
            memory = context.memory(arm)
            reasons = [key+" exceeds budget" for key, bound in (
                ("core_bytes", budget.core_bytes), ("normal_owned_bytes", budget.normal_owned_bytes),
                ("estimated_working_array_bytes", budget.working_array_bytes)) if memory[key] > bound]
            prediction = None
            try:
                prediction = self._model.predict(context.spec, arm, runtime)
            except PredictionUnavailable as exc:
                reasons.append(str(exc))
            total = None if reasons else setups*prediction["setup_seconds"]+calls*prediction["apply_seconds"]
            candidates.append({"arm": arm, "eligible": not reasons, "reasons": reasons, **memory,
                "prediction": prediction, "predicted_operator_seconds": total})
        feasible = [c for c in candidates if c["eligible"]]
        if not feasible:
            raise NoFeasiblePlan(candidates)
        best = min(c["predicted_operator_seconds"] for c in feasible)
        tied = [c for c in feasible if c["predicted_operator_seconds"] <= best*(1+tie_fraction)]
        selected = min(tied, key=lambda c: (c["normal_owned_bytes"], c["predicted_operator_seconds"], c["arm"]))
        payload = canonical({"selected_arm": selected["arm"], "model_id": self._model.model_id,
            "runtime_id": digest_json(runtime), "spec": asdict(context.spec),
            "structure": describe_spec(context.spec), "operator_digest": context.operator_digest,
            "weight_digests": dict(context.weight_digests), "epochs": [asdict(e) for e in epochs],
            "candidates": candidates, "budget": asdict(budget), "setup_count": setups, "apply_count": calls,
            "tolerance": tolerance, "tie_fraction": tie_fraction,
            "selection_seconds": time.perf_counter()-start,
            "accuracy_status": "structure checked; selected action requires independent finite-probe validation",
            "timing_scope": "G setup and apply; context, calibration and verification charged separately"})
        self._issued.add((context._nonce, payload))
        return PredictiveChoice(payload, context._nonce, self._nonce)

    def prepare_verified(self, context, choice, weight, probes):
        if (not isinstance(choice, PredictiveChoice) or choice._context_nonce is not context._nonce or
                choice._planner_nonce is not self._nonce or (context._nonce, choice.payload_json) not in self._issued):
            raise ValueError("choice not issued for this context/planner")
        p = choice.payload
        if p["runtime_id"] != digest_json(native_runtime(context)) or p["model_id"] != self._model.model_id:
            raise ValueError("native runtime or model changed; new selection required")
        if weight not in {e["weight"] for e in p["epochs"]}:
            raise ValueError("weight outside selected workload")
        x = np.asarray(probes, dtype=complex)
        if (x.shape != (2, *context.transfer.modal_shape) or not np.all(np.isfinite(x)) or
                any(np.linalg.norm(v) == 0 for v in x) or
                abs(np.vdot(x[0], x[1])) >= (1-1e-12)*np.linalg.norm(x[0])*np.linalg.norm(x[1])):
            raise ValueError("two finite nonzero independent modal probes required")
        start = time.perf_counter()
        plan = context._prepare(p["selected_arm"], weight)
        setup = time.perf_counter()-start
        if plan.cache_bytes != context.memory(p["selected_arm"])["normal_owned_bytes"]:
            raise RuntimeError("prepared retained-array count differs from contract")
        start = time.perf_counter()
        ys = [plan.action.apply(v) for v in x]
        refs = [context.transfer.adjoint(context.weights[weight]*context.transfer.forward(v)) for v in x]
        errors = [float(np.linalg.norm(y-r)/max(np.linalg.norm(r), 1e-300)) for y, r in zip(ys, refs)]
        hermitian = float(abs(np.vdot(x[0], ys[1])-np.vdot(ys[0], x[1]))/
            max(np.linalg.norm(x[0])*np.linalg.norm(ys[1])+np.linalg.norm(ys[0])*np.linalg.norm(x[1]), 1e-300))
        if not np.all(np.isfinite(errors+[hermitian])) or max(errors+[hermitian]) > p["tolerance"]:
            raise ArithmeticError("selected action failed finite-probe validation; no silent fallback")
        verification = {"relative_errors": errors, "hermitian_bilinear": hermitian,
            "criterion": p["tolerance"], "scope": "two probes, same discrete G; not a spectral-norm certificate"}
        return VerifiedNormal(plan, canonical(verification), setup, time.perf_counter()-start)
