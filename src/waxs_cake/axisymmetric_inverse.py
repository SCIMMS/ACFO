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


class TVGradientOperator(Protocol):
    object_shape: tuple[int, ...]
    dual_shape: tuple[int, ...]
    dual_weights: "NDArray[np.floating]"

    def forward(self, object_values: "ArrayLike") -> "NDArray[np.complexfloating]": ...

    def adjoint_euclidean(
        self,
        dual_values: "ArrayLike",
    ) -> "NDArray[np.complexfloating]": ...

    def value_u(self, object_values: "ArrayLike") -> float: ...


class SymmetricVolumeOperator:
    """Expose ``u=sqrt(V)f`` for an integrated-coefficient operator.

    If the wrapped operator acts on ``c=Vf``, this adapter implements
    ``B u = A_c sqrt(V) u`` and ``B**H y = sqrt(V) A_c**H y``.  Ordinary
    Euclidean geometry in ``u`` is therefore the cylindrical physical L2
    geometry.  The adapter is diagonal and does not alter the prepared ACFO
    plan or its harmonic contract.
    """

    def __init__(
        self,
        coefficient_operator: ForwardAdjointOperator,
        volume_weights: "ArrayLike",
    ) -> None:
        self.coefficient_operator = coefficient_operator
        self.object_shape = tuple(coefficient_operator.object_shape)
        self.data_shape = tuple(coefficient_operator.data_shape)
        volume = np.asarray(volume_weights, dtype=np.float64)
        if volume.shape != self.object_shape:
            try:
                volume = np.broadcast_to(volume, self.object_shape)
            except ValueError as exc:
                raise ValueError(
                    f"volume_weights must broadcast to {self.object_shape}"
                ) from exc
        if not np.all(np.isfinite(volume)) or np.any(volume <= 0.0):
            raise ValueError("volume_weights must be finite and positive")
        self.volume_weights = np.array(volume, dtype=np.float64, copy=True)
        self.sqrt_volume_weights = np.sqrt(self.volume_weights)
        self.volume_weights.setflags(write=False)
        self.sqrt_volume_weights.setflags(write=False)

    def _object_array(self, values: "ArrayLike", name: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.complex128)
        if array.shape != self.object_shape or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite with shape {self.object_shape}")
        return array

    def forward(self, symmetric_values: "ArrayLike") -> np.ndarray:
        values = self._object_array(symmetric_values, "symmetric_values")
        return np.asarray(
            self.coefficient_operator.forward(values * self.sqrt_volume_weights),
            dtype=np.complex128,
        )

    def adjoint_euclidean(self, data_values: "ArrayLike") -> np.ndarray:
        coefficient_adjoint = np.asarray(
            self.coefficient_operator.adjoint_euclidean(data_values),
            dtype=np.complex128,
        )
        if coefficient_adjoint.shape != self.object_shape:
            raise ValueError(
                "wrapped coefficient adjoint returned an unexpected object shape"
            )
        return coefficient_adjoint * self.sqrt_volume_weights

    def density_from_symmetric(self, symmetric_values: "ArrayLike") -> np.ndarray:
        values = self._object_array(symmetric_values, "symmetric_values")
        return values / self.sqrt_volume_weights

    def symmetric_from_density(self, density_values: "ArrayLike") -> np.ndarray:
        values = self._object_array(density_values, "density_values")
        return values * self.sqrt_volume_weights

    def coefficient_from_symmetric(self, symmetric_values: "ArrayLike") -> np.ndarray:
        values = self._object_array(symmetric_values, "symmetric_values")
        return values * self.sqrt_volume_weights


class SquareRootDataWeightOperator:
    """Compose a real diagonal precision factor ``sqrt(W)`` with an operator."""

    def __init__(
        self,
        operator: ForwardAdjointOperator,
        data_weights: "ArrayLike",
    ) -> None:
        self.operator = operator
        self.object_shape = tuple(operator.object_shape)
        self.data_shape = tuple(operator.data_shape)
        weights = np.asarray(data_weights, dtype=np.float64)
        if (
            weights.ndim == 1
            and self.data_shape
            and weights.size == self.data_shape[0]
        ):
            weights = weights.reshape(
                (weights.size,) + (1,) * (len(self.data_shape) - 1)
            )
        try:
            weights = np.broadcast_to(weights, self.data_shape)
        except ValueError as exc:
            raise ValueError(
                f"data_weights must broadcast to {self.data_shape}"
            ) from exc
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("data_weights must be finite and non-negative")
        if not np.any(weights > 0.0):
            raise ValueError("data_weights must contain a positive entry")
        self.sqrt_data_weights = np.sqrt(
            np.array(weights, dtype=np.float64, copy=True)
        )
        self.sqrt_data_weights.setflags(write=False)

    def forward(self, object_values: "ArrayLike") -> np.ndarray:
        return self.sqrt_data_weights * np.asarray(
            self.operator.forward(object_values), dtype=np.complex128
        )

    def adjoint_euclidean(self, data_values: "ArrayLike") -> np.ndarray:
        data = np.asarray(data_values, dtype=np.complex128)
        if data.shape != self.data_shape or not np.all(np.isfinite(data)):
            raise ValueError(f"data_values must be finite with shape {self.data_shape}")
        return np.asarray(
            self.operator.adjoint_euclidean(data * self.sqrt_data_weights),
            dtype=np.complex128,
        )

    def weight_data(self, data_values: "ArrayLike") -> np.ndarray:
        data = np.asarray(data_values, dtype=np.complex128)
        if data.shape != self.data_shape or not np.all(np.isfinite(data)):
            raise ValueError(f"data_values must be finite with shape {self.data_shape}")
        return data * self.sqrt_data_weights


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


@dataclass(frozen=True)
class PrimalDualTVReconstruction:
    """Result of matrix-free complex cylindrical-TV PDHG."""

    reconstruction: "NDArray[np.complex128]"
    history: tuple[dict[str, float | int], ...]
    converged: bool
    iterations: int
    elapsed_seconds: float
    norm_estimation_seconds: float
    working_set_bytes: int
    tv_weight: float
    ridge_regularization: float
    stacked_operator_norm: float
    data_operator_norm: float
    gradient_operator_norm: float
    primal_step: float
    dual_step: float
    data_dual_step: float
    gradient_dual_step: float
    gradient_scaling: float
    final_fixed_point_scaled_residual: float
    fixed_point_certification_count: int
    solver_forward_calls: int
    solver_adjoint_calls: int
    norm_forward_calls: int
    norm_adjoint_calls: int


@dataclass(frozen=True)
class ProjectedGradientReconstruction:
    """Result of matrix-free accelerated projected quadratic retrieval."""

    reconstruction: "NDArray[np.complex128]"
    history: tuple[dict[str, float | int], ...]
    converged: bool
    iterations: int
    elapsed_seconds: float
    norm_estimation_seconds: float
    working_set_bytes: int
    ridge_regularization: float
    operator_norm: float
    initial_lipschitz_constant: float
    lipschitz_constant: float
    step_size: float
    constraint: str
    restart_count: int
    monotone_restart_count: int
    backtracking_increase_count: int
    kkt_certification_count: int
    final_projected_gradient_absolute: float
    final_projected_gradient_relative: float
    final_objective: float
    solver_forward_calls: int
    solver_adjoint_calls: int
    norm_forward_calls: int
    norm_adjoint_calls: int


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


def estimate_stacked_operator_norm(
    operator: ForwardAdjointOperator,
    *,
    gradient: TVGradientOperator | None = None,
    iterations: int = 20,
    seed: int = 0,
) -> float:
    """Estimate ``||[A; Gamma]||`` by deterministic power iteration."""

    iterations = int(iterations)
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if gradient is not None and tuple(gradient.object_shape) != tuple(
        operator.object_shape
    ):
        raise ValueError("operator and gradient object shapes must match")
    rng = np.random.default_rng(int(seed))
    vector = rng.normal(size=operator.object_shape) + 1j * rng.normal(
        size=operator.object_shape
    )
    vector = np.asarray(vector, dtype=np.complex128)
    vector /= np.linalg.norm(vector)
    eigenvalue = 0.0
    for _ in range(iterations):
        forward = np.asarray(operator.forward(vector), dtype=np.complex128)
        normal = np.asarray(
            operator.adjoint_euclidean(forward), dtype=np.complex128
        )
        if gradient is not None:
            gradient_values = np.asarray(
                gradient.forward(vector), dtype=np.complex128
            )
            normal += np.asarray(
                gradient.adjoint_euclidean(gradient_values),
                dtype=np.complex128,
            )
        norm = float(np.linalg.norm(normal))
        if not np.isfinite(norm) or norm == 0.0:
            raise RuntimeError("stacked-operator power iteration failed")
        eigenvalue = float(np.vdot(vector, normal).real)
        vector = normal / norm
    if not np.isfinite(eigenvalue) or eigenvalue <= 0.0:
        raise RuntimeError("stacked-operator norm estimate is not positive")
    return float(np.sqrt(eigenvalue))


def estimate_tv_gradient_norm(
    gradient: TVGradientOperator,
    *,
    iterations: int = 20,
    seed: int = 0,
) -> float:
    """Estimate ``||Gamma||`` by deterministic power iteration."""

    iterations = int(iterations)
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    rng = np.random.default_rng(int(seed))
    vector = rng.normal(size=gradient.object_shape) + 1j * rng.normal(
        size=gradient.object_shape
    )
    vector = np.asarray(vector, dtype=np.complex128)
    vector /= np.linalg.norm(vector)
    eigenvalue = 0.0
    for _ in range(iterations):
        forward = np.asarray(gradient.forward(vector), dtype=np.complex128)
        normal = np.asarray(
            gradient.adjoint_euclidean(forward), dtype=np.complex128
        )
        norm = float(np.linalg.norm(normal))
        if not np.isfinite(norm) or norm == 0.0:
            raise RuntimeError("TV-gradient power iteration failed")
        eigenvalue = float(np.vdot(vector, normal).real)
        vector = normal / norm
    if not np.isfinite(eigenvalue) or eigenvalue <= 0.0:
        raise RuntimeError("TV-gradient norm estimate is not positive")
    return float(np.sqrt(eigenvalue))


def _project_primal_constraint(
    values: np.ndarray,
    *,
    constraint: str,
    support_mask: np.ndarray | None,
) -> np.ndarray:
    if constraint == "none":
        projected = values
    elif constraint == "real":
        projected = values.real.astype(np.complex128)
    elif constraint == "nonnegative":
        projected = np.maximum(values.real, 0.0).astype(np.complex128)
    else:
        raise ValueError("constraint must be 'none', 'real', or 'nonnegative'")
    if support_mask is not None:
        projected = np.where(support_mask, projected, 0.0)
    return projected


def accelerated_projected_quadratic(
    operator: ForwardAdjointOperator,
    data: "ArrayLike",
    *,
    ridge_regularization: float = 0.0,
    max_iterations: int = 1000,
    minimum_iterations: int = 10,
    projected_gradient_tolerance: float = 1e-7,
    constraint: str = "none",
    support_mask: "ArrayLike | None" = None,
    initial: "ArrayLike | None" = None,
    truth: "ArrayLike | None" = None,
    operator_norm: float | None = None,
    norm_iterations: int = 20,
    norm_seed: int = 0,
    norm_inflation: float = 1.05,
    step_safety: float = 0.99,
    adaptive_restart: bool = True,
    backtracking_factor: float = 2.0,
    backtracking_tolerance: float = 1e-12,
) -> ProjectedGradientReconstruction:
    """Solve a constrained quadratic inverse problem with monotone FISTA.

    ``operator_norm`` initializes the trial Lipschitz constant but is not
    trusted as a certified bound.  Exact quadratic backtracking checks each
    accepted step.  The returned stopping certificate is the projected-
    gradient KKT residual evaluated at the final feasible iterate.
    """

    ridge_regularization = float(ridge_regularization)
    max_iterations = int(max_iterations)
    minimum_iterations = int(minimum_iterations)
    projected_gradient_tolerance = float(projected_gradient_tolerance)
    norm_inflation = float(norm_inflation)
    step_safety = float(step_safety)
    backtracking_factor = float(backtracking_factor)
    backtracking_tolerance = float(backtracking_tolerance)
    if not np.isfinite(ridge_regularization) or ridge_regularization < 0.0:
        raise ValueError("ridge_regularization must be finite and non-negative")
    if max_iterations <= 0 or minimum_iterations < 0:
        raise ValueError("iteration counts are invalid")
    if minimum_iterations > max_iterations:
        raise ValueError("minimum_iterations must not exceed max_iterations")
    if (
        not np.isfinite(projected_gradient_tolerance)
        or projected_gradient_tolerance <= 0.0
    ):
        raise ValueError(
            "projected_gradient_tolerance must be finite and positive"
        )
    if not np.isfinite(norm_inflation) or norm_inflation < 1.0:
        raise ValueError("norm_inflation must be finite and at least one")
    if not np.isfinite(step_safety) or not 0.0 < step_safety <= 1.0:
        raise ValueError("step_safety must lie in (0, 1]")
    if not np.isfinite(backtracking_factor) or backtracking_factor <= 1.0:
        raise ValueError("backtracking_factor must be greater than one")
    if not np.isfinite(backtracking_tolerance) or backtracking_tolerance < 0.0:
        raise ValueError("backtracking_tolerance must be finite and non-negative")

    observed = np.asarray(data, dtype=np.complex128)
    if observed.shape != operator.data_shape or not np.all(np.isfinite(observed)):
        raise ValueError(f"data must be finite with shape {operator.data_shape}")
    support = None
    if support_mask is not None:
        support = np.asarray(support_mask, dtype=bool)
        if support.shape != operator.object_shape:
            raise ValueError(f"support_mask must have shape {operator.object_shape}")
    if initial is None:
        reconstruction = np.zeros(operator.object_shape, dtype=np.complex128)
    else:
        reconstruction = np.array(initial, dtype=np.complex128, copy=True)
        if reconstruction.shape != operator.object_shape or not np.all(
            np.isfinite(reconstruction)
        ):
            raise ValueError(f"initial must be finite with shape {operator.object_shape}")
    reconstruction = _project_primal_constraint(
        reconstruction,
        constraint=constraint,
        support_mask=support,
    )
    truth_array = None
    if truth is not None:
        truth_array = np.asarray(truth, dtype=np.complex128)
        if truth_array.shape != operator.object_shape or not np.all(
            np.isfinite(truth_array)
        ):
            raise ValueError(f"truth must be finite with shape {operator.object_shape}")

    norm_start = perf_counter()
    if operator_norm is None:
        resolved_operator_norm = norm_inflation * estimate_stacked_operator_norm(
            operator,
            iterations=norm_iterations,
            seed=norm_seed,
        )
        norm_forward_calls = norm_iterations
        norm_adjoint_calls = norm_iterations
    else:
        resolved_operator_norm = float(operator_norm)
        norm_forward_calls = 0
        norm_adjoint_calls = 0
    norm_estimation_seconds = perf_counter() - norm_start
    if not np.isfinite(resolved_operator_norm) or resolved_operator_norm <= 0.0:
        raise ValueError("operator_norm must be finite and positive")
    lipschitz = (
        resolved_operator_norm**2 + ridge_regularization
    ) / step_safety
    initial_lipschitz = lipschitz

    def objective(values: np.ndarray, prediction: np.ndarray) -> float:
        residual = prediction - observed
        return 0.5 * float(np.vdot(residual, residual).real) + 0.5 * (
            ridge_regularization * float(np.vdot(values, values).real)
        )

    solver_start = perf_counter()
    prediction = np.asarray(
        operator.forward(reconstruction), dtype=np.complex128
    )
    solver_forward_calls = 1
    solver_adjoint_calls = 0
    current_objective = objective(reconstruction, prediction)
    kkt_scale = max(
        resolved_operator_norm * float(np.linalg.norm(observed)),
        ridge_regularization * float(np.linalg.norm(reconstruction)),
        np.sqrt(np.finfo(np.float64).tiny),
    )
    previous_reconstruction = reconstruction.copy()
    previous_prediction = prediction.copy()
    momentum = 1.0
    restart_count = 0
    monotone_restart_count = 0
    backtracking_increase_count = 0
    kkt_certification_count = 0
    final_projected_gradient_absolute = float("inf")
    final_projected_gradient_relative = float("inf")
    history: list[dict[str, float | int]] = []
    converged = False

    for iteration in range(1, max_iterations + 1):
        iteration_start = perf_counter()
        next_momentum = 0.5 * (
            1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum)
        )
        beta = (momentum - 1.0) / next_momentum
        search_point = reconstruction + beta * (
            reconstruction - previous_reconstruction
        )
        search_prediction = prediction + beta * (
            prediction - previous_prediction
        )
        immediate_restart = False

        while True:
            search_gradient = np.asarray(
                operator.adjoint_euclidean(search_prediction - observed),
                dtype=np.complex128,
            )
            solver_adjoint_calls += 1
            search_gradient += ridge_regularization * search_point

            while True:
                candidate = _project_primal_constraint(
                    search_point - search_gradient / lipschitz,
                    constraint=constraint,
                    support_mask=support,
                )
                displacement = candidate - search_point
                displacement_prediction = np.asarray(
                    operator.forward(displacement), dtype=np.complex128
                )
                solver_forward_calls += 1
                curvature = float(
                    np.vdot(displacement_prediction, displacement_prediction).real
                ) + ridge_regularization * float(
                    np.vdot(displacement, displacement).real
                )
                allowed = lipschitz * float(
                    np.vdot(displacement, displacement).real
                ) * (1.0 + backtracking_tolerance)
                if curvature <= allowed or not np.any(displacement):
                    break
                lipschitz *= backtracking_factor
                backtracking_increase_count += 1

            candidate_prediction = search_prediction + displacement_prediction
            candidate_objective = objective(candidate, candidate_prediction)
            roundoff_tolerance = 64.0 * np.finfo(np.float64).eps * max(
                1.0, abs(current_objective)
            )
            if candidate_objective <= current_objective + roundoff_tolerance:
                break
            if immediate_restart:
                raise RuntimeError(
                    "monotone projected-gradient restart failed to decrease "
                    "the objective"
                )
            immediate_restart = True
            monotone_restart_count += 1
            restart_count += 1
            momentum = 1.0
            next_momentum = 0.5 * (1.0 + np.sqrt(5.0))
            search_point = reconstruction
            search_prediction = prediction

        mapping_at_search = lipschitz * float(np.linalg.norm(displacement))
        mapping_at_search_relative = mapping_at_search / kkt_scale
        change = float(np.linalg.norm(candidate - reconstruction))
        relative_change = change / max(
            float(np.linalg.norm(candidate)),
            float(np.linalg.norm(reconstruction)),
            np.sqrt(np.finfo(np.float64).tiny),
        )

        should_certify = (
            iteration == max_iterations
            or (
                iteration >= minimum_iterations
                and mapping_at_search_relative
                <= projected_gradient_tolerance
            )
        )
        certified_this_iteration = False
        if should_certify:
            candidate_gradient = np.asarray(
                operator.adjoint_euclidean(candidate_prediction - observed),
                dtype=np.complex128,
            )
            solver_adjoint_calls += 1
            kkt_certification_count += 1
            candidate_gradient += ridge_regularization * candidate
            projected_candidate = _project_primal_constraint(
                candidate - candidate_gradient / lipschitz,
                constraint=constraint,
                support_mask=support,
            )
            final_projected_gradient_absolute = lipschitz * float(
                np.linalg.norm(candidate - projected_candidate)
            )
            final_projected_gradient_relative = (
                final_projected_gradient_absolute / kkt_scale
            )
            certified_this_iteration = True

        record: dict[str, float | int] = {
            "iteration": iteration,
            "objective": candidate_objective,
            "projected_gradient_at_search_relative": (
                mapping_at_search_relative
            ),
            "final_feasible_projected_gradient_relative": (
                final_projected_gradient_relative
                if certified_this_iteration
                else float("nan")
            ),
            "relative_change": relative_change,
            "lipschitz_constant": lipschitz,
            "iteration_seconds": perf_counter() - iteration_start,
        }
        if truth_array is not None:
            record["reconstruction_relative_l2"] = float(
                np.linalg.norm(candidate - truth_array)
                / max(
                    float(np.linalg.norm(truth_array)),
                    np.sqrt(np.finfo(np.float64).tiny),
                )
            )
        history.append(record)

        old_reconstruction = reconstruction
        old_prediction = prediction
        reconstruction = candidate
        prediction = candidate_prediction
        current_objective = candidate_objective
        previous_reconstruction = old_reconstruction
        previous_prediction = old_prediction

        if (
            certified_this_iteration
            and final_projected_gradient_relative
            <= projected_gradient_tolerance
        ):
            converged = True
            break
        if certified_this_iteration:
            momentum = 1.0
            restart_count += 1
        elif adaptive_restart and float(
            np.vdot(
                search_point - candidate,
                candidate - old_reconstruction,
            ).real
        ) > 0.0:
            momentum = 1.0
            restart_count += 1
        else:
            momentum = next_momentum

    if not np.isfinite(final_projected_gradient_relative):
        final_gradient = np.asarray(
            operator.adjoint_euclidean(prediction - observed),
            dtype=np.complex128,
        )
        solver_adjoint_calls += 1
        kkt_certification_count += 1
        final_gradient += ridge_regularization * reconstruction
        projected_final = _project_primal_constraint(
            reconstruction - final_gradient / lipschitz,
            constraint=constraint,
            support_mask=support,
        )
        final_projected_gradient_absolute = lipschitz * float(
            np.linalg.norm(reconstruction - projected_final)
        )
        final_projected_gradient_relative = (
            final_projected_gradient_absolute
            / kkt_scale
        )

    working_set_bytes = sum(
        array.nbytes
        for array in (
            reconstruction,
            previous_reconstruction,
            prediction,
            previous_prediction,
        )
    )
    iterations_completed = len(history)
    return ProjectedGradientReconstruction(
        reconstruction=reconstruction,
        history=tuple(history),
        converged=converged,
        iterations=iterations_completed,
        elapsed_seconds=perf_counter() - solver_start,
        norm_estimation_seconds=norm_estimation_seconds,
        working_set_bytes=int(working_set_bytes),
        ridge_regularization=ridge_regularization,
        operator_norm=resolved_operator_norm,
        initial_lipschitz_constant=initial_lipschitz,
        lipschitz_constant=lipschitz,
        step_size=1.0 / lipschitz,
        constraint=constraint,
        restart_count=restart_count,
        monotone_restart_count=monotone_restart_count,
        backtracking_increase_count=backtracking_increase_count,
        kkt_certification_count=kkt_certification_count,
        final_projected_gradient_absolute=final_projected_gradient_absolute,
        final_projected_gradient_relative=final_projected_gradient_relative,
        final_objective=current_objective,
        solver_forward_calls=solver_forward_calls,
        solver_adjoint_calls=solver_adjoint_calls,
        norm_forward_calls=norm_forward_calls,
        norm_adjoint_calls=norm_adjoint_calls,
    )


def primal_dual_cylindrical_tv(
    operator: ForwardAdjointOperator,
    data: "ArrayLike",
    gradient: TVGradientOperator,
    *,
    tv_weight: float,
    ridge_regularization: float = 0.0,
    max_iterations: int = 500,
    relative_tolerance: float = 1e-6,
    minimum_iterations: int = 10,
    constraint: str = "none",
    support_mask: "ArrayLike | None" = None,
    initial: "ArrayLike | None" = None,
    truth: "ArrayLike | None" = None,
    norm_iterations: int = 20,
    norm_seed: int = 0,
    norm_inflation: float = 1.05,
    step_safety: float = 0.95,
    stacked_operator_norm: float | None = None,
    data_operator_norm: float | None = None,
    gradient_operator_norm: float | None = None,
    normalize_gradient: bool = True,
) -> PrimalDualTVReconstruction:
    """Solve a complex least-squares plus cylindrical-TV problem with PDHG.

    The objective is

    ``0.5||A u-y||^2 + lambda*TV(u) + 0.5*mu||u||^2 + I_C(u)``.

    The data residual and TV gradient are both dualized.  The implementation
    caches the current and previous forward/gradient values, so every solver
    iteration uses exactly one declared forward, one adjoint, one gradient,
    and one gradient-adjoint action.
    """

    tv_weight = float(tv_weight)
    ridge_regularization = float(ridge_regularization)
    max_iterations = int(max_iterations)
    minimum_iterations = int(minimum_iterations)
    relative_tolerance = float(relative_tolerance)
    norm_inflation = float(norm_inflation)
    step_safety = float(step_safety)
    if not np.isfinite(tv_weight) or tv_weight < 0.0:
        raise ValueError("tv_weight must be finite and non-negative")
    if not np.isfinite(ridge_regularization) or ridge_regularization < 0.0:
        raise ValueError("ridge_regularization must be finite and non-negative")
    if max_iterations <= 0 or minimum_iterations < 0:
        raise ValueError("iteration counts are invalid")
    if minimum_iterations > max_iterations:
        raise ValueError("minimum_iterations must not exceed max_iterations")
    if not np.isfinite(relative_tolerance) or relative_tolerance <= 0.0:
        raise ValueError("relative_tolerance must be finite and positive")
    if not np.isfinite(norm_inflation) or norm_inflation < 1.0:
        raise ValueError("norm_inflation must be finite and at least one")
    if not np.isfinite(step_safety) or not 0.0 < step_safety < 1.0:
        raise ValueError("step_safety must lie strictly between zero and one")
    if tuple(gradient.object_shape) != tuple(operator.object_shape):
        raise ValueError("operator and gradient object shapes must match")

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
        if truth_array.shape != operator.object_shape or not np.all(
            np.isfinite(truth_array)
        ):
            raise ValueError(f"truth must be finite with shape {operator.object_shape}")
    support = None
    if support_mask is not None:
        support = np.asarray(support_mask, dtype=bool)
        if support.shape != operator.object_shape:
            raise ValueError(f"support_mask must have shape {operator.object_shape}")

    dual_weights = np.asarray(gradient.dual_weights, dtype=np.float64)
    try:
        dual_weights = np.broadcast_to(
            dual_weights, tuple(gradient.dual_shape[1:])
        )
    except ValueError as exc:
        raise ValueError("gradient dual_weights do not match its dual shape") from exc
    if not np.all(np.isfinite(dual_weights)) or np.any(dual_weights <= 0.0):
        raise ValueError("gradient dual_weights must be finite and positive")

    gradient_active = tv_weight > 0.0
    if stacked_operator_norm is not None and (
        data_operator_norm is not None or gradient_operator_norm is not None
    ):
        raise ValueError(
            "stacked_operator_norm cannot be combined with separate operator norms"
        )
    norm_estimation_start = perf_counter()
    use_common_steps = stacked_operator_norm is not None
    if use_common_steps:
        resolved_norm = float(stacked_operator_norm)
        resolved_data_norm = resolved_norm
        resolved_gradient_norm = resolved_norm if gradient_active else 0.0
        gradient_scaling = 1.0
        effective_gradient_norm = resolved_gradient_norm
        norm_forward_calls = 0
        norm_adjoint_calls = 0
    else:
        if data_operator_norm is None:
            resolved_data_norm = norm_inflation * estimate_stacked_operator_norm(
                operator,
                iterations=norm_iterations,
                seed=norm_seed,
            )
            norm_forward_calls = int(norm_iterations)
            norm_adjoint_calls = int(norm_iterations)
        else:
            resolved_data_norm = float(data_operator_norm)
            norm_forward_calls = 0
            norm_adjoint_calls = 0
        if gradient_active:
            if gradient_operator_norm is None:
                resolved_gradient_norm = norm_inflation * estimate_tv_gradient_norm(
                    gradient,
                    iterations=norm_iterations,
                    seed=norm_seed + 1,
                )
            else:
                resolved_gradient_norm = float(gradient_operator_norm)
        else:
            resolved_gradient_norm = 0.0
        gradient_scaling = (
            resolved_gradient_norm
            if gradient_active and bool(normalize_gradient)
            else 1.0
        )
        effective_gradient_norm = (
            resolved_gradient_norm / gradient_scaling
            if gradient_active
            else 0.0
        )
        resolved_norm = float(
            np.hypot(resolved_data_norm, effective_gradient_norm)
        )
    norm_estimation_seconds = perf_counter() - norm_estimation_start
    if not np.isfinite(resolved_norm) or resolved_norm <= 0.0:
        raise ValueError("stacked_operator_norm must be finite and positive")
    if not np.isfinite(resolved_data_norm) or resolved_data_norm <= 0.0:
        raise ValueError("data_operator_norm must be finite and positive")
    if gradient_active and (
        not np.isfinite(resolved_gradient_norm) or resolved_gradient_norm <= 0.0
    ):
        raise ValueError("gradient_operator_norm must be finite and positive")
    if use_common_steps:
        primal_step = step_safety / resolved_norm
        data_dual_step = step_safety / resolved_norm
        gradient_dual_step = data_dual_step
    elif gradient_active:
        primal_step = step_safety / (
            resolved_data_norm + effective_gradient_norm
        )
        data_dual_step = step_safety / resolved_data_norm
        gradient_dual_step = step_safety / effective_gradient_norm
    else:
        primal_step = step_safety / resolved_data_norm
        data_dual_step = step_safety / resolved_data_norm
        gradient_dual_step = 0.0
    dual_step = data_dual_step

    prediction = np.asarray(operator.forward(reconstruction), dtype=np.complex128)
    if prediction.shape != operator.data_shape:
        raise ValueError("operator forward returned an unexpected data shape")
    previous_prediction = prediction.copy()
    gradient_values = np.asarray(
        gradient.forward(reconstruction), dtype=np.complex128
    )
    if gradient_values.shape != gradient.dual_shape:
        raise ValueError("gradient forward returned an unexpected dual shape")
    previous_gradient = gradient_values.copy()
    # For g(z)=0.5||z-y||^2 the optimal data dual is z-y.  Initializing it
    # from the current residual is a valid arbitrary PDHG dual start and is
    # especially important when ``initial`` is already a CGLS warm start.
    dual_data = prediction - observed
    dual_gradient = np.zeros(gradient.dual_shape, dtype=np.complex128)
    data_norm = max(float(np.linalg.norm(observed)), np.finfo(np.float64).tiny)
    truth_norm = (
        max(float(np.linalg.norm(truth_array)), np.finfo(np.float64).tiny)
        if truth_array is not None
        else None
    )
    history: list[dict[str, float | int]] = []
    converged = False
    final_fixed_point_scaled_residual = float("inf")
    fixed_point_certification_count = 0
    start = perf_counter()

    def fixed_point_certificate() -> float:
        next_dual_data = (
            dual_data + data_dual_step * (prediction - observed)
        ) / (1.0 + data_dual_step)
        if gradient_active:
            next_dual_gradient = dual_gradient + (
                gradient_dual_step * gradient_values / gradient_scaling
            )
            next_point_norm = np.sqrt(
                np.sum(np.abs(next_dual_gradient) ** 2, axis=0)
            )
            next_radius = tv_weight * gradient_scaling * dual_weights
            next_dual_gradient = next_dual_gradient / np.maximum(
                1.0,
                next_point_norm / next_radius,
            )[None, ...]
        else:
            next_dual_gradient = dual_gradient
        next_direction = np.asarray(
            operator.adjoint_euclidean(next_dual_data), dtype=np.complex128
        )
        if gradient_active:
            next_direction += np.asarray(
                gradient.adjoint_euclidean(next_dual_gradient),
                dtype=np.complex128,
            ) / gradient_scaling
        next_reconstruction = _project_primal_constraint(
            (reconstruction - primal_step * next_direction)
            / (1.0 + primal_step * ridge_regularization),
            constraint=constraint,
            support_mask=support,
        )
        primal_residual = float(
            np.linalg.norm(next_reconstruction - reconstruction)
        ) / max(
            float(np.linalg.norm(next_reconstruction)),
            float(np.linalg.norm(reconstruction)),
            1.0,
        )
        data_dual_residual = float(
            np.linalg.norm(next_dual_data - dual_data)
        ) / max(
            float(np.linalg.norm(next_dual_data)),
            float(np.linalg.norm(dual_data)),
            float(np.linalg.norm(prediction - observed)),
            1.0,
        )
        if gradient_active:
            gradient_dual_residual = float(
                np.linalg.norm(next_dual_gradient - dual_gradient)
            ) / max(
                float(np.linalg.norm(next_dual_gradient)),
                float(np.linalg.norm(dual_gradient)),
                1.0,
            )
        else:
            gradient_dual_residual = 0.0
        return max(
            primal_residual,
            data_dual_residual,
            gradient_dual_residual,
        )

    for iteration in range(1, max_iterations + 1):
        iteration_start = perf_counter()
        extrapolated_prediction = 2.0 * prediction - previous_prediction
        dual_data = (
            dual_data + data_dual_step * (extrapolated_prediction - observed)
        ) / (1.0 + data_dual_step)

        if gradient_active:
            extrapolated_gradient = 2.0 * gradient_values - previous_gradient
            dual_gradient += (
                gradient_dual_step * extrapolated_gradient / gradient_scaling
            )
            point_norm = np.sqrt(np.sum(np.abs(dual_gradient) ** 2, axis=0))
            radius = tv_weight * gradient_scaling * dual_weights
            projection_scale = np.maximum(
                1.0,
                point_norm / radius,
            )
            dual_gradient /= projection_scale[None, ...]

        primal_direction = np.asarray(
            operator.adjoint_euclidean(dual_data), dtype=np.complex128
        )
        if gradient_active:
            primal_direction += np.asarray(
                gradient.adjoint_euclidean(dual_gradient),
                dtype=np.complex128,
            ) / gradient_scaling
        candidate = (
            reconstruction - primal_step * primal_direction
        ) / (1.0 + primal_step * ridge_regularization)
        candidate = _project_primal_constraint(
            candidate,
            constraint=constraint,
            support_mask=support,
        )
        change = float(np.linalg.norm(candidate - reconstruction))
        relative_change = change / max(
            float(np.linalg.norm(reconstruction)),
            float(np.linalg.norm(candidate)),
            np.sqrt(np.finfo(np.float64).tiny),
        )

        previous_prediction = prediction
        previous_gradient = gradient_values
        reconstruction = candidate
        prediction = np.asarray(
            operator.forward(reconstruction), dtype=np.complex128
        )
        gradient_values = np.asarray(
            gradient.forward(reconstruction), dtype=np.complex128
        )
        residual = prediction - observed
        data_fidelity = 0.5 * float(np.vdot(residual, residual).real)
        tv_value = float(
            np.sum(
                dual_weights
                * np.sqrt(np.sum(np.abs(gradient_values) ** 2, axis=0))
            )
        )
        ridge_value = 0.5 * ridge_regularization * float(
            np.vdot(reconstruction, reconstruction).real
        )
        should_certify = iteration == max_iterations or (
            iteration >= minimum_iterations
            and relative_change <= relative_tolerance
        )
        if should_certify:
            final_fixed_point_scaled_residual = fixed_point_certificate()
            fixed_point_certification_count += 1

        record: dict[str, float | int] = {
            "iteration": iteration,
            "relative_change": relative_change,
            "relative_data_residual": float(np.linalg.norm(residual) / data_norm),
            "data_fidelity": data_fidelity,
            "tv_value": tv_value,
            "objective": data_fidelity + tv_weight * tv_value + ridge_value,
            "fixed_point_scaled_residual": (
                final_fixed_point_scaled_residual
                if should_certify
                else float("nan")
            ),
            "iteration_seconds": perf_counter() - iteration_start,
        }
        if truth_array is not None and truth_norm is not None:
            record["reconstruction_relative_l2"] = float(
                np.linalg.norm(reconstruction - truth_array) / truth_norm
            )
        history.append(record)
        if (
            should_certify
            and final_fixed_point_scaled_residual <= relative_tolerance
        ):
            converged = True
            break

    working_set_bytes = sum(
        array.nbytes
        for array in (
            reconstruction,
            prediction,
            previous_prediction,
            gradient_values,
            previous_gradient,
            dual_data,
            dual_gradient,
        )
    )
    iterations_completed = len(history)
    return PrimalDualTVReconstruction(
        reconstruction=reconstruction,
        history=tuple(history),
        converged=converged,
        iterations=iterations_completed,
        elapsed_seconds=perf_counter() - start,
        norm_estimation_seconds=norm_estimation_seconds,
        working_set_bytes=int(working_set_bytes),
        tv_weight=tv_weight,
        ridge_regularization=ridge_regularization,
        stacked_operator_norm=resolved_norm,
        data_operator_norm=resolved_data_norm,
        gradient_operator_norm=resolved_gradient_norm,
        primal_step=primal_step,
        dual_step=dual_step,
        data_dual_step=data_dual_step,
        gradient_dual_step=gradient_dual_step,
        gradient_scaling=gradient_scaling,
        final_fixed_point_scaled_residual=final_fixed_point_scaled_residual,
        fixed_point_certification_count=fixed_point_certification_count,
        solver_forward_calls=iterations_completed + 1,
        solver_adjoint_calls=(
            iterations_completed + fixed_point_certification_count
        ),
        norm_forward_calls=norm_forward_calls,
        norm_adjoint_calls=norm_adjoint_calls,
    )
