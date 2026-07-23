"""Small regularized inverse solvers for prepared axisymmetric operators."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


class ForwardAdjointOperator(Protocol):
    object_shape: tuple[int, ...]
    data_shape: tuple[int, ...]

    def forward(self, object_values: "ArrayLike") -> "NDArray[np.complexfloating]": ...

    def adjoint_euclidean(
        self,
        data_values: "ArrayLike",
    ) -> "NDArray[np.complexfloating]": ...


@dataclass(frozen=True)
class ConjugateGradientReconstruction:
    """Result of a complex Hermitian positive-definite normal-equation solve."""

    reconstruction: "NDArray[np.complex128]"
    history: tuple[dict[str, float | int], ...]
    converged: bool
    iterations: int
    elapsed_seconds: float
    working_set_bytes: int
    regularization: float


def conjugate_gradient_tikhonov(
    operator: ForwardAdjointOperator,
    data: "ArrayLike",
    *,
    regularization: float,
    max_iterations: int = 100,
    relative_tolerance: float = 1e-10,
    initial: "ArrayLike | None" = None,
    truth: "ArrayLike | None" = None,
) -> ConjugateGradientReconstruction:
    """Solve ``(A^H A + lambda I)x = A^H y`` with logged convergence.

    The implementation uses only the declared forward and Euclidean-adjoint
    actions. It is therefore shared by ACFO, direct-matrix, and NUFFT adapters.
    """

    regularization = float(regularization)
    max_iterations = int(max_iterations)
    relative_tolerance = float(relative_tolerance)
    if not np.isfinite(regularization) or regularization < 0.0:
        raise ValueError("regularization must be finite and non-negative")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if not np.isfinite(relative_tolerance) or relative_tolerance <= 0.0:
        raise ValueError("relative_tolerance must be finite and positive")

    observed = np.asarray(data, dtype=np.complex128)
    if observed.shape != operator.data_shape or not np.all(np.isfinite(observed)):
        raise ValueError(f"data must be finite with shape {operator.data_shape}")
    if initial is None:
        reconstruction = np.zeros(operator.object_shape, dtype=np.complex128)
    else:
        reconstruction = np.array(initial, dtype=np.complex128, copy=True)
        if reconstruction.shape != operator.object_shape or not np.all(
            np.isfinite(reconstruction)
        ):
            raise ValueError(f"initial must be finite with shape {operator.object_shape}")
    truth_array = None
    if truth is not None:
        truth_array = np.asarray(truth, dtype=np.complex128)
        if truth_array.shape != operator.object_shape or not np.all(np.isfinite(truth_array)):
            raise ValueError(f"truth must be finite with shape {operator.object_shape}")

    start = perf_counter()
    prediction = operator.forward(reconstruction)
    right_hand_side = operator.adjoint_euclidean(observed)
    normal_residual = right_hand_side - (
        operator.adjoint_euclidean(prediction) + regularization * reconstruction
    )
    direction = normal_residual.copy()
    residual_norm_squared = float(np.vdot(normal_residual, normal_residual).real)
    initial_normal_norm = np.sqrt(residual_norm_squared)
    data_norm = max(float(np.linalg.norm(observed)), np.finfo(np.float64).tiny)
    truth_norm = (
        max(float(np.linalg.norm(truth_array)), np.finfo(np.float64).tiny)
        if truth_array is not None
        else None
    )
    history: list[dict[str, float | int]] = []
    converged = initial_normal_norm == 0.0

    for iteration in range(1, max_iterations + 1):
        if converged:
            break
        iteration_start = perf_counter()
        forward_direction = operator.forward(direction)
        normal_direction = (
            operator.adjoint_euclidean(forward_direction) + regularization * direction
        )
        curvature = float(np.vdot(direction, normal_direction).real)
        if not np.isfinite(curvature) or curvature <= 0.0:
            raise RuntimeError("normal operator is not numerically positive definite")
        step = residual_norm_squared / curvature
        reconstruction += step * direction
        prediction += step * forward_direction
        normal_residual -= step * normal_direction
        new_residual_norm_squared = float(
            np.vdot(normal_residual, normal_residual).real
        )
        relative_normal = np.sqrt(new_residual_norm_squared) / max(
            initial_normal_norm,
            np.finfo(np.float64).tiny,
        )
        data_residual = float(np.linalg.norm(prediction - observed) / data_norm)
        objective = 0.5 * float(np.vdot(prediction - observed, prediction - observed).real)
        objective += 0.5 * regularization * float(
            np.vdot(reconstruction, reconstruction).real
        )
        record: dict[str, float | int] = {
            "iteration": iteration,
            "relative_normal_residual": float(relative_normal),
            "relative_data_residual": data_residual,
            "objective": objective,
            "iteration_seconds": perf_counter() - iteration_start,
        }
        if truth_array is not None and truth_norm is not None:
            record["reconstruction_relative_l2"] = float(
                np.linalg.norm(reconstruction - truth_array) / truth_norm
            )
        history.append(record)
        if relative_normal <= relative_tolerance:
            converged = True
            residual_norm_squared = new_residual_norm_squared
            break
        beta = new_residual_norm_squared / residual_norm_squared
        direction = normal_residual + beta * direction
        residual_norm_squared = new_residual_norm_squared

    working_set_bytes = sum(
        array.nbytes
        for array in (
            reconstruction,
            prediction,
            right_hand_side,
            normal_residual,
            direction,
        )
    )
    return ConjugateGradientReconstruction(
        reconstruction=reconstruction,
        history=tuple(history),
        converged=bool(converged),
        iterations=len(history),
        elapsed_seconds=perf_counter() - start,
        working_set_bytes=int(working_set_bytes),
        regularization=regularization,
    )
