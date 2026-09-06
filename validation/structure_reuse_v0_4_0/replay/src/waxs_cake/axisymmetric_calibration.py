"""Geometry-only calibration models for axisymmetric Fourier sampling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np
from scipy.interpolate import BSpline

from .axisymmetric_manifold import AxisymmetricManifold
from .axisymmetric_operator import PreparedAxisymmetricOperator

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from .histogram import BinnedStructure


class CurvatureModel(Protocol):
    """Minimal real-parameter curvature model used by the optimizer."""

    @property
    def n_parameters(self) -> int: ...

    def manifold(self, parameters: "ArrayLike") -> AxisymmetricManifold: ...

    def geometry_jacobians(
        self,
        parameters: "ArrayLike",
    ) -> tuple["NDArray[np.float64]", "NDArray[np.float64]"]: ...

    def regularization(
        self,
        parameters: "ArrayLike",
        strength: float,
    ) -> tuple[float, "NDArray[np.float64]"]: ...


def _parameter_vector(
    values: "ArrayLike",
    size: int,
    *,
    field_name: str = "parameters",
) -> "NDArray[np.float64]":
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{field_name} must have shape ({size},)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} must contain only finite values")
    return array


@dataclass(frozen=True)
class EllipsoidCurvatureModel:
    """Two-parameter meridional ellipse with a fixed angular coordinate."""

    u: "ArrayLike"
    name: str = "calibrated-ellipsoid"

    def __post_init__(self) -> None:
        u = np.array(self.u, dtype=np.float64, copy=True)
        if u.ndim != 1 or u.size == 0 or not np.all(np.isfinite(u)):
            raise ValueError("u must be a non-empty finite vector")
        if u.size > 1 and np.any(np.diff(u) <= 0.0):
            raise ValueError("u must be strictly increasing")
        u.setflags(write=False)
        object.__setattr__(self, "u", u)

    @property
    def n_parameters(self) -> int:
        return 2

    def manifold(self, parameters: "ArrayLike") -> AxisymmetricManifold:
        a, c = _parameter_vector(parameters, self.n_parameters)
        if a < 0.0:
            raise ValueError("ellipsoid radial scale must be non-negative")
        return AxisymmetricManifold(
            self.u,
            a * np.sin(self.u),
            c * (np.cos(self.u) - 1.0),
            name=self.name,
        )

    def geometry_jacobians(
        self,
        parameters: "ArrayLike",
    ) -> tuple["NDArray[np.float64]", "NDArray[np.float64]"]:
        _parameter_vector(parameters, self.n_parameters)
        zeros = np.zeros_like(self.u)
        return (
            np.column_stack((np.sin(self.u), zeros)),
            np.column_stack((zeros, np.cos(self.u) - 1.0)),
        )

    def regularization(
        self,
        parameters: "ArrayLike",
        strength: float,
    ) -> tuple[float, "NDArray[np.float64]"]:
        values = _parameter_vector(parameters, self.n_parameters)
        if strength < 0.0 or not np.isfinite(strength):
            raise ValueError("regularization strength must be finite and non-negative")
        return 0.0, np.zeros_like(values)


@dataclass(frozen=True)
class AnchoredBSplineCurvatureModel:
    """Clamped B-spline curve whose first ``(Q_perp, Q_z)`` control is fixed."""

    u: "ArrayLike"
    n_control: int = 5
    degree: int = 3
    anchor_q_perp: float = 0.0
    anchor_q_z: float = 0.0
    name: str = "calibrated-anchored-bspline"
    basis: "NDArray[np.float64]" = field(init=False, repr=False)
    second_difference: "NDArray[np.float64]" = field(init=False, repr=False)

    def __post_init__(self) -> None:
        u = np.array(self.u, dtype=np.float64, copy=True)
        if u.ndim != 1 or u.size < 2 or not np.all(np.isfinite(u)):
            raise ValueError("u must be a finite vector with at least two samples")
        if np.any(np.diff(u) <= 0.0):
            raise ValueError("u must be strictly increasing")
        if self.degree < 1 or self.n_control <= self.degree:
            raise ValueError("n_control must exceed the positive spline degree")
        if not np.isfinite(self.anchor_q_perp) or self.anchor_q_perp < 0.0:
            raise ValueError("anchor_q_perp must be finite and non-negative")
        if not np.isfinite(self.anchor_q_z):
            raise ValueError("anchor_q_z must be finite")

        interior_count = self.n_control - self.degree - 1
        interior = np.linspace(u[0], u[-1], interior_count + 2)[1:-1]
        knots = np.concatenate(
            (
                np.repeat(u[0], self.degree + 1),
                interior,
                np.repeat(u[-1], self.degree + 1),
            )
        )
        basis = np.column_stack(
            [
                BSpline(knots, np.eye(self.n_control)[index], self.degree)(u)
                for index in range(self.n_control)
            ]
        )
        second_difference = np.zeros((self.n_control - 2, self.n_control))
        for index in range(self.n_control - 2):
            second_difference[index, index : index + 3] = (1.0, -2.0, 1.0)
        u.setflags(write=False)
        basis.setflags(write=False)
        second_difference.setflags(write=False)
        object.__setattr__(self, "u", u)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "second_difference", second_difference)

    @property
    def n_parameters(self) -> int:
        return 2 * (self.n_control - 1)

    def controls(
        self,
        parameters: "ArrayLike",
    ) -> tuple["NDArray[np.float64]", "NDArray[np.float64]"]:
        values = _parameter_vector(parameters, self.n_parameters)
        split = self.n_control - 1
        q_perp = np.concatenate(([self.anchor_q_perp], values[:split]))
        q_z = np.concatenate(([self.anchor_q_z], values[split:]))
        if np.any(q_perp < 0.0):
            raise ValueError("Q_perp control points must be non-negative")
        return q_perp, q_z

    def parameters_from_controls(
        self,
        q_perp: "ArrayLike",
        q_z: "ArrayLike",
    ) -> "NDArray[np.float64]":
        q_perp_array = _parameter_vector(q_perp, self.n_control, field_name="q_perp")
        q_z_array = _parameter_vector(q_z, self.n_control, field_name="q_z")
        if not np.isclose(q_perp_array[0], self.anchor_q_perp) or not np.isclose(
            q_z_array[0], self.anchor_q_z
        ):
            raise ValueError("first control point must equal the fixed anchor")
        values = np.concatenate((q_perp_array[1:], q_z_array[1:]))
        self.controls(values)
        return values

    def manifold(self, parameters: "ArrayLike") -> AxisymmetricManifold:
        q_perp_control, q_z_control = self.controls(parameters)
        return AxisymmetricManifold(
            self.u,
            self.basis @ q_perp_control,
            self.basis @ q_z_control,
            name=self.name,
        )

    def geometry_jacobians(
        self,
        parameters: "ArrayLike",
    ) -> tuple["NDArray[np.float64]", "NDArray[np.float64]"]:
        self.controls(parameters)
        free_basis = self.basis[:, 1:]
        zeros = np.zeros_like(free_basis)
        return (
            np.column_stack((free_basis, zeros)),
            np.column_stack((zeros, free_basis)),
        )

    def regularization(
        self,
        parameters: "ArrayLike",
        strength: float,
    ) -> tuple[float, "NDArray[np.float64]"]:
        values = _parameter_vector(parameters, self.n_parameters)
        if strength < 0.0 or not np.isfinite(strength):
            raise ValueError("regularization strength must be finite and non-negative")
        if strength == 0.0:
            return 0.0, np.zeros_like(values)
        q_perp, q_z = self.controls(values)
        d2_perp = self.second_difference @ q_perp
        d2_z = self.second_difference @ q_z
        penalty = 0.5 * strength * (
            float(np.dot(d2_perp, d2_perp)) + float(np.dot(d2_z, d2_z))
        )
        full_gradient_perp = strength * self.second_difference.T @ d2_perp
        full_gradient_z = strength * self.second_difference.T @ d2_z
        gradient = np.concatenate((full_gradient_perp[1:], full_gradient_z[1:]))
        return penalty, gradient


def geometry_parameter_jacobian(
    template: "BinnedStructure",
    model: CurvatureModel,
    parameters: "ArrayLike",
    *,
    object_values: "ArrayLike | None" = None,
) -> "NDArray[np.complex128]":
    """Return the complex data Jacobian with shape ``(parameter, u, phi)``."""

    values = _parameter_vector(parameters, model.n_parameters)
    operator = PreparedAxisymmetricOperator(template, model.manifold(values))
    object_array = template.hist if object_values is None else object_values
    derivative_perp, derivative_z = operator.geometry_derivatives(object_array)
    jacobian_perp, jacobian_z = model.geometry_jacobians(values)
    jacobian = (
        jacobian_perp.T[:, :, None] * derivative_perp[None, :, :]
        + jacobian_z.T[:, :, None] * derivative_z[None, :, :]
    )
    return jacobian.astype(np.complex128, copy=False)


def curvature_loss_and_gradient(
    template: "BinnedStructure",
    model: CurvatureModel,
    parameters: "ArrayLike",
    data: "ArrayLike",
    *,
    object_values: "ArrayLike | None" = None,
    smoothness: float = 0.0,
) -> tuple[float, "NDArray[np.float64]"]:
    """Evaluate a normalized complex least-squares calibration objective."""

    values = _parameter_vector(parameters, model.n_parameters)
    operator = PreparedAxisymmetricOperator(template, model.manifold(values))
    object_array = template.hist if object_values is None else object_values
    observed = np.asarray(data, dtype=np.complex128)
    if observed.shape != operator.data_shape or not np.all(np.isfinite(observed)):
        raise ValueError(f"data must be finite with shape {operator.data_shape}")
    residual = operator.forward(object_array) - observed
    scale = max(float(np.vdot(observed, observed).real), np.finfo(np.float64).tiny)
    data_loss = 0.5 * float(np.vdot(residual, residual).real) / scale
    gradient_perp, gradient_z = operator.geometry_loss_gradient(object_array, residual)
    jacobian_perp, jacobian_z = model.geometry_jacobians(values)
    gradient = (jacobian_perp.T @ gradient_perp + jacobian_z.T @ gradient_z) / scale
    penalty, penalty_gradient = model.regularization(values, smoothness)
    return data_loss + penalty, gradient + penalty_gradient


def geometry_identifiability(
    template: "BinnedStructure",
    model: CurvatureModel,
    parameters: "ArrayLike",
    *,
    object_values: "ArrayLike | None" = None,
) -> dict[str, object]:
    """Return singular values and condition number for real geometry parameters."""

    jacobian = geometry_parameter_jacobian(
        template,
        model,
        parameters,
        object_values=object_values,
    )
    matrix = np.concatenate((jacobian.real.reshape(model.n_parameters, -1).T,
                             jacobian.imag.reshape(model.n_parameters, -1).T), axis=0)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    threshold = np.finfo(np.float64).eps * max(matrix.shape) * singular_values[0]
    rank = int(np.count_nonzero(singular_values > threshold))
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values[-1] > threshold
        else float("inf")
    )
    return {
        "rank": rank,
        "n_parameters": model.n_parameters,
        "condition_number": condition,
        "singular_values": singular_values.tolist(),
    }
