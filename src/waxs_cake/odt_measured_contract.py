from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = "odt-curved-ewald-v1"

EXPERIMENT_TYPES = {
    "complex_odt",
    "annular_idt",
    "fpdt",
    "fs_odt",
    "focused_raster_dt",
}

MEASUREMENT_MODELS = {
    "complex_field",
    "coherent_intensity",
    "incoherent_intensity",
    "multiplexed_intensity",
}

Q_LAYOUTS = {
    "annular_cartesian_stack",
    "prepared_ring_stack",
    "prepared_cap_stack",
    "explicit_q",
    "rotational_sinogram",
}

PATTERN_MODELS = {
    "coherent",
    "incoherent",
    "demultiplexed",
    "multiplexed",
}


@dataclass(frozen=True)
class ValidationIssue:
    level: str
    field: str
    message: str


@dataclass(frozen=True)
class ValidationReport:
    errors: tuple[ValidationIssue, ...] = ()
    warnings: tuple[ValidationIssue, ...] = ()
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            joined = "; ".join(f"{issue.field}: {issue.message}" for issue in self.errors)
            raise ValueError(f"invalid ODT measured-data contract: {joined}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": [issue.__dict__ for issue in self.errors],
            "warnings": [issue.__dict__ for issue in self.warnings],
            "summary": dict(self.summary),
        }


@dataclass(frozen=True)
class OdtMeasuredData:
    fields: dict[str, Any]
    source_path: Path | None = None

    def __contains__(self, key: str) -> bool:
        return key in self.fields

    def __getitem__(self, key: str) -> Any:
        return self.fields[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.fields.get(key, default)

    @property
    def schema_version(self) -> str:
        return _as_str(self.fields.get("schema_version", ""))

    @property
    def experiment_type(self) -> str:
        return _as_str(self.fields.get("experiment_type", ""))

    @property
    def measurement_model(self) -> str:
        return _as_str(self.fields.get("measurement_model", ""))

    @property
    def q_layout(self) -> str:
        return _as_str(self.fields.get("q_layout", ""))

    @property
    def data(self) -> np.ndarray:
        return np.asarray(self.fields["data"])


@dataclass(frozen=True)
class PreparedOperatorDescriptor:
    q_layout: str
    measurement_model: str
    experiment_type: str
    n_illum: int
    cap_radial: int | None
    cap_phi: int | None
    n_patterns: int | None
    q_samples: int
    measurement_samples: int
    data_shape: tuple[int, ...]
    has_mask: bool
    has_variance: bool
    source_path: Path | None = None


def load_odt_measured_contract(path: str | Path) -> OdtMeasuredData:
    path = Path(path)
    with np.load(path, allow_pickle=False) as loaded:
        fields = {key: _normalize_npz_value(loaded[key]) for key in loaded.files}
    return OdtMeasuredData(fields=fields, source_path=path)


def save_odt_measured_contract(path: str | Path, data: OdtMeasuredData | Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = data.fields if isinstance(data, OdtMeasuredData) else dict(data)
    np.savez(path, **fields)


def validate_odt_measured_contract(data: OdtMeasuredData | Mapping[str, Any]) -> ValidationReport:
    measured = data if isinstance(data, OdtMeasuredData) else OdtMeasuredData(dict(data))
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    def err(field: str, message: str) -> None:
        errors.append(ValidationIssue("error", field, message))

    def warn(field: str, message: str) -> None:
        warnings.append(ValidationIssue("warning", field, message))

    fields = measured.fields
    required = [
        "schema_version",
        "experiment_type",
        "measurement_model",
        "units",
        "illum_dirs",
        "detector_origin",
        "detector_u",
        "detector_v",
        "detector_pixel_size",
        "detector_distance",
        "q_layout",
        "data",
    ]
    for key in required:
        if key not in fields:
            err(key, "missing required field")
    if "k" not in fields and "wavelength" not in fields:
        err("k", "one of 'k' or 'wavelength' is required")

    if errors:
        return ValidationReport(tuple(errors), tuple(warnings), {})

    schema_version = _as_str(fields["schema_version"])
    experiment_type = _as_str(fields["experiment_type"])
    measurement_model = _as_str(fields["measurement_model"])
    q_layout = _as_str(fields["q_layout"])
    units = _as_str(fields["units"])
    data_array = np.asarray(fields["data"])

    if schema_version != SCHEMA_VERSION:
        warn("schema_version", f"expected {SCHEMA_VERSION!r}, got {schema_version!r}")
    if experiment_type not in EXPERIMENT_TYPES:
        err("experiment_type", f"unsupported experiment type {experiment_type!r}")
    if measurement_model not in MEASUREMENT_MODELS:
        err("measurement_model", f"unsupported measurement model {measurement_model!r}")
    if q_layout not in Q_LAYOUTS:
        err("q_layout", f"unsupported q layout {q_layout!r}")
    if not units:
        err("units", "must be a non-empty string")
    if not data_array.size:
        err("data", "must not be empty")
    if not np.all(np.isfinite(data_array)):
        err("data", "contains non-finite values")

    illum_dirs = _require_array(fields, "illum_dirs", errors, ndim=2, shape_tail=(3,))
    n_illum = int(illum_dirs.shape[0]) if illum_dirs is not None else 0
    if illum_dirs is not None:
        norms = np.linalg.norm(illum_dirs, axis=1)
        if np.any(norms <= 0.0) or not np.all(np.isfinite(norms)):
            err("illum_dirs", "illumination directions must be finite nonzero vectors")
        elif not np.allclose(norms, 1.0, rtol=0.0, atol=1e-3):
            err("illum_dirs", "illumination directions must be normalized to unit length")

    _require_vector(fields, "detector_origin", errors, length=3)
    detector_u = _require_vector(fields, "detector_u", errors, length=3)
    detector_v = _require_vector(fields, "detector_v", errors, length=3)
    if detector_u is not None and detector_v is not None:
        norm_u = float(np.linalg.norm(detector_u))
        norm_v = float(np.linalg.norm(detector_v))
        if norm_u <= 0.0 or norm_v <= 0.0:
            err("detector_u", "detector axes must be nonzero")
        else:
            if not np.isclose(norm_u, 1.0, rtol=0.0, atol=1e-3):
                warn("detector_u", "detector fast axis is not unit normalized")
            if not np.isclose(norm_v, 1.0, rtol=0.0, atol=1e-3):
                warn("detector_v", "detector slow axis is not unit normalized")
            dot = float(np.dot(detector_u, detector_v) / (norm_u * norm_v))
            if abs(dot) > 1e-3:
                warn("detector_u", "detector axes are not orthogonal")

    pixel_size = _require_vector(fields, "detector_pixel_size", errors, length=2)
    if pixel_size is not None and np.any(pixel_size <= 0.0):
        err("detector_pixel_size", "pixel sizes must be positive")
    detector_distance = _as_float(fields.get("detector_distance"))
    if detector_distance is None or detector_distance <= 0.0:
        err("detector_distance", "must be a positive scalar")
    if "k" in fields:
        k_value = _as_float(fields["k"])
        if k_value is None or k_value <= 0.0:
            err("k", "must be a positive scalar")
    if "wavelength" in fields:
        wavelength = _as_float(fields["wavelength"])
        if wavelength is None or wavelength <= 0.0:
            err("wavelength", "must be a positive scalar")

    cap_radial: int | None = None
    cap_phi: int | None = None
    n_patterns: int | None = None
    q_samples: int | None = None

    if q_layout in {"prepared_ring_stack", "prepared_cap_stack"}:
        cap_radial = _as_positive_int(fields.get("cap_radial"))
        cap_phi = _as_positive_int(fields.get("cap_phi"))
        if cap_radial is None:
            err("cap_radial", "positive integer required for structured layouts")
        if cap_phi is None:
            err("cap_phi", "positive integer required for structured layouts")
        if "q_z_model" not in fields or not _as_str(fields.get("q_z_model")):
            err("q_z_model", "non-empty q_z_model is required for structured layouts")
        q_radial = _require_array(fields, "q_radial", errors, ndim=1)
        q_phi = _require_array(fields, "q_phi", errors, ndim=1)
        if cap_radial is not None and q_radial is not None and q_radial.shape != (cap_radial,):
            err("q_radial", f"expected shape ({cap_radial},), got {q_radial.shape}")
        if cap_phi is not None and q_phi is not None and q_phi.shape != (cap_phi,):
            err("q_phi", f"expected shape ({cap_phi},), got {q_phi.shape}")
        if q_radial is not None and not np.all(np.isfinite(q_radial)):
            err("q_radial", "contains non-finite values")
        if q_phi is not None and not np.all(np.isfinite(q_phi)):
            err("q_phi", "contains non-finite values")
        if cap_radial is not None and cap_phi is not None:
            q_samples = cap_radial * cap_phi
            _validate_structured_data_shape(
                fields,
                errors,
                warnings,
                data_array=data_array,
                measurement_model=measurement_model,
                n_illum=n_illum,
                cap_radial=cap_radial,
                cap_phi=cap_phi,
            )
            n_patterns = _structured_pattern_count(fields, measurement_model)
    elif q_layout == "explicit_q":
        q_xyz = _require_array(fields, "q_xyz", errors, ndim=2, shape_tail=(3,))
        if q_xyz is not None:
            if not np.all(np.isfinite(q_xyz)):
                err("q_xyz", "contains non-finite values")
            q_samples = int(q_xyz.shape[0])
            flat_samples = int(np.prod(data_array.shape))
            if data_array.shape[0] != q_samples and flat_samples != q_samples:
                err(
                    "data",
                    f"explicit_q data must have first dimension or flattened size {q_samples}, got {data_array.shape}",
                )
        if "sample_index" in fields:
            sample_index = np.asarray(fields["sample_index"])
            if q_samples is not None and sample_index.shape != (q_samples,):
                err("sample_index", f"expected shape ({q_samples},), got {sample_index.shape}")
    elif q_layout == "rotational_sinogram":
        if measurement_model != "complex_field":
            err("measurement_model", "rotational_sinogram currently supports complex_field data only")
        if data_array.ndim != 3:
            err("data", f"rotational_sinogram data must have shape (angles, height, width), got {data_array.shape}")
        elif data_array.shape[0] != n_illum:
            err("data", f"expected first dimension {n_illum} from illum_dirs, got {data_array.shape[0]}")
        elif not np.iscomplexobj(data_array):
            warn("data", "rotational_sinogram complex_field data are not complex-valued")
        angles = _require_array(fields, "rotation_angles", errors, ndim=1)
        if angles is not None and data_array.ndim == 3 and angles.shape != (data_array.shape[0],):
            err("rotation_angles", f"expected shape ({data_array.shape[0]},), got {angles.shape}")
        if data_array.ndim == 3:
            cap_radial = int(data_array.shape[1])
            cap_phi = int(data_array.shape[2])
            q_samples = cap_radial * cap_phi
    elif q_layout == "annular_cartesian_stack":
        if measurement_model == "complex_field":
            err("measurement_model", "annular_cartesian_stack is intended for intensity data")
        if data_array.ndim != 3:
            err("data", f"annular_cartesian_stack data must have shape (illum, height, width), got {data_array.shape}")
        elif data_array.shape[0] != n_illum:
            err("data", f"expected first dimension {n_illum} from illum_dirs, got {data_array.shape[0]}")
        elif np.iscomplexobj(data_array):
            err("data", "annular_cartesian_stack intensity data must be real-valued")
        source_na = _require_array(fields, "source_na_xy", errors, ndim=2, shape_tail=(2,))
        if source_na is not None and source_na.shape[0] != n_illum:
            err("source_na_xy", f"expected first dimension {n_illum}, got {source_na.shape[0]}")
        frequency_x = _require_array(fields, "frequency_x", errors, ndim=1)
        frequency_y = _require_array(fields, "frequency_y", errors, ndim=1)
        if data_array.ndim == 3:
            cap_radial = int(data_array.shape[1])
            cap_phi = int(data_array.shape[2])
            q_samples = cap_radial * cap_phi
            if frequency_y is not None and frequency_y.shape != (cap_radial,):
                err("frequency_y", f"expected shape ({cap_radial},), got {frequency_y.shape}")
            if frequency_x is not None and frequency_x.shape != (cap_phi,):
                err("frequency_x", f"expected shape ({cap_phi},), got {frequency_x.shape}")
        objective_na = _as_float(fields.get("objective_na"))
        if objective_na is None or objective_na <= 0.0:
            err("objective_na", "positive scalar required for annular_cartesian_stack")

    _validate_optional_sample_array(fields, errors, "mask", data_array.shape, allow_complex=False)
    _validate_optional_sample_array(fields, errors, "variance", data_array.shape, allow_complex=False)
    if "variance" in fields and np.any(np.asarray(fields["variance"]) < 0.0):
        err("variance", "must be non-negative")
    _validate_optional_sample_array(fields, errors, "background", data_array.shape, allow_complex=False)
    _validate_optional_sample_array(fields, errors, "flatfield", data_array.shape, allow_complex=False)
    if "flatfield" in fields and np.any(np.asarray(fields["flatfield"]) <= 0.0):
        err("flatfield", "must be positive where provided")

    summary = {
        "schema_version": schema_version,
        "experiment_type": experiment_type,
        "measurement_model": measurement_model,
        "q_layout": q_layout,
        "n_illum": n_illum,
        "n_patterns": n_patterns,
        "cap_radial": cap_radial,
        "cap_phi": cap_phi,
        "q_samples": q_samples,
        "measurement_samples": int(data_array.size),
        "data_shape": tuple(int(v) for v in data_array.shape),
        "data_dtype": str(data_array.dtype),
        "has_mask": "mask" in fields,
        "has_variance": "variance" in fields,
    }
    if q_layout == "rotational_sinogram" and "rotation_angles" in fields:
        angles = np.asarray(fields["rotation_angles"])
        if angles.size:
            summary["rotation_angle_min"] = float(np.min(angles))
            summary["rotation_angle_max"] = float(np.max(angles))
    if q_layout == "annular_cartesian_stack":
        if "objective_na" in fields:
            summary["objective_na"] = _as_float(fields["objective_na"])
        if "source_na_xy" in fields:
            source_na = np.asarray(fields["source_na_xy"])
            if source_na.size:
                summary["source_na_min"] = float(np.min(np.linalg.norm(source_na, axis=1)))
                summary["source_na_max"] = float(np.max(np.linalg.norm(source_na, axis=1)))
    return ValidationReport(tuple(errors), tuple(warnings), summary)


def build_prepared_operator_from_contract(data: OdtMeasuredData | Mapping[str, Any]) -> PreparedOperatorDescriptor:
    measured = data if isinstance(data, OdtMeasuredData) else OdtMeasuredData(dict(data))
    report = validate_odt_measured_contract(measured)
    report.raise_for_errors()
    summary = report.summary
    return PreparedOperatorDescriptor(
        q_layout=str(summary["q_layout"]),
        measurement_model=str(summary["measurement_model"]),
        experiment_type=str(summary["experiment_type"]),
        n_illum=int(summary["n_illum"]),
        cap_radial=_none_or_int(summary["cap_radial"]),
        cap_phi=_none_or_int(summary["cap_phi"]),
        n_patterns=_none_or_int(summary["n_patterns"]),
        q_samples=int(summary["q_samples"]) if summary["q_samples"] is not None else int(np.prod(measured.data.shape)),
        measurement_samples=int(summary["measurement_samples"]),
        data_shape=tuple(int(v) for v in summary["data_shape"]),
        has_mask=bool(summary["has_mask"]),
        has_variance=bool(summary["has_variance"]),
        source_path=measured.source_path,
    )


def _normalize_npz_value(value: np.ndarray) -> Any:
    if value.shape == ():
        item = value.item()
        if isinstance(item, bytes):
            return item.decode("utf-8")
        return item
    return np.asarray(value)


def _as_str(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _as_str(value.item())
    return str(value)


def _as_float(value: Any) -> float | None:
    try:
        arr = np.asarray(value)
        if arr.shape != ():
            return None
        parsed = float(arr.item())
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


def _as_positive_int(value: Any) -> int | None:
    try:
        arr = np.asarray(value)
        if arr.shape != ():
            return None
        parsed = int(arr.item())
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _none_or_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _require_array(
    fields: Mapping[str, Any],
    key: str,
    errors: list[ValidationIssue],
    *,
    ndim: int,
    shape_tail: tuple[int, ...] | None = None,
) -> np.ndarray | None:
    if key not in fields:
        errors.append(ValidationIssue("error", key, "missing required field"))
        return None
    arr = np.asarray(fields[key])
    if arr.ndim != ndim:
        errors.append(ValidationIssue("error", key, f"expected {ndim} dimensions, got {arr.ndim}"))
        return None
    if shape_tail is not None and tuple(arr.shape[-len(shape_tail) :]) != tuple(shape_tail):
        errors.append(ValidationIssue("error", key, f"expected trailing shape {shape_tail}, got {arr.shape}"))
        return None
    if not np.all(np.isfinite(arr)):
        errors.append(ValidationIssue("error", key, "contains non-finite values"))
    return arr


def _require_vector(
    fields: Mapping[str, Any],
    key: str,
    errors: list[ValidationIssue],
    *,
    length: int,
) -> np.ndarray | None:
    arr = _require_array(fields, key, errors, ndim=1)
    if arr is not None and arr.shape != (length,):
        errors.append(ValidationIssue("error", key, f"expected shape ({length},), got {arr.shape}"))
        return None
    return arr


def _structured_pattern_count(fields: Mapping[str, Any], measurement_model: str) -> int | None:
    if measurement_model == "complex_field":
        return None
    if "pattern_matrix" not in fields:
        return None
    return int(np.asarray(fields["pattern_matrix"]).shape[0])


def _validate_structured_data_shape(
    fields: Mapping[str, Any],
    errors: list[ValidationIssue],
    warnings: list[ValidationIssue],
    *,
    data_array: np.ndarray,
    measurement_model: str,
    n_illum: int,
    cap_radial: int,
    cap_phi: int,
) -> None:
    if data_array.ndim != 3:
        errors.append(ValidationIssue("error", "data", f"structured data must be 3D, got {data_array.shape}"))
        return
    if data_array.shape[1:] != (cap_radial, cap_phi):
        errors.append(
            ValidationIssue(
                "error",
                "data",
                f"expected trailing shape ({cap_radial}, {cap_phi}), got {data_array.shape[1:]}",
            )
        )
    if measurement_model == "complex_field":
        if data_array.shape[0] != n_illum:
            errors.append(ValidationIssue("error", "data", f"expected first dimension {n_illum}, got {data_array.shape[0]}"))
        if not np.iscomplexobj(data_array):
            warnings.append(ValidationIssue("warning", "data", "complex_field data are not complex-valued"))
        return

    if np.iscomplexobj(data_array):
        errors.append(ValidationIssue("error", "data", "intensity data must be real-valued"))
    if "pattern_matrix" not in fields:
        errors.append(ValidationIssue("error", "pattern_matrix", "required for intensity and multiplexed models"))
        return
    pattern = np.asarray(fields["pattern_matrix"])
    if pattern.ndim != 2:
        errors.append(ValidationIssue("error", "pattern_matrix", f"expected 2D pattern matrix, got {pattern.shape}"))
        return
    if pattern.shape[1] != n_illum:
        errors.append(
            ValidationIssue(
                "error",
                "pattern_matrix",
                f"expected second dimension {n_illum}, got {pattern.shape[1]}",
            )
        )
    if data_array.shape[0] != pattern.shape[0]:
        errors.append(
            ValidationIssue(
                "error",
                "data",
                f"expected first dimension {pattern.shape[0]} from pattern_matrix, got {data_array.shape[0]}",
            )
        )
    if not np.all(np.isfinite(pattern)):
        errors.append(ValidationIssue("error", "pattern_matrix", "contains non-finite values"))
    pattern_model = _as_str(fields.get("pattern_model", ""))
    if pattern_model not in PATTERN_MODELS:
        errors.append(ValidationIssue("error", "pattern_model", f"unsupported or missing pattern model {pattern_model!r}"))


def _validate_optional_sample_array(
    fields: Mapping[str, Any],
    errors: list[ValidationIssue],
    key: str,
    data_shape: tuple[int, ...],
    *,
    allow_complex: bool,
) -> None:
    if key not in fields:
        return
    arr = np.asarray(fields[key])
    try:
        np.broadcast_shapes(arr.shape, data_shape)
    except ValueError:
        errors.append(ValidationIssue("error", key, f"shape {arr.shape} is not broadcastable to data shape {data_shape}"))
        return
    if not allow_complex and np.iscomplexobj(arr):
        errors.append(ValidationIssue("error", key, "must be real-valued"))
    if not np.all(np.isfinite(arr)):
        errors.append(ValidationIssue("error", key, "contains non-finite values"))
