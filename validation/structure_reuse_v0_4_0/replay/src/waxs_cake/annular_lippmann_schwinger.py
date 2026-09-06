"""Self-consistent annular Lippmann--Schwinger solves above a substrate."""

from __future__ import annotations

from typing import Any, Callable


class ScalarAnnularLippmannSchwingerSolver:
    r"""Solve ``E = E_inc + G(chi E)`` with a square annular Green action.

    The supplied annular operator already contains the source quadrature
    weights.  ``susceptibility`` is therefore a pointwise material response,
    not a second integration weight.  Forward and adjoint fixed-point solves
    reuse the operator's paired modal contractions.
    """

    def __init__(self, operator: Any, susceptibility: Any) -> None:
        if operator.object_shape != operator.data_shape:
            raise ValueError("self-consistent interaction requires a square operator")
        self.operator = operator
        self.torch = operator.torch
        self.device = operator.device
        self.susceptibility = self.torch.as_tensor(
            susceptibility, device=self.device
        ).to(dtype=operator.complex_dtype)
        if tuple(self.susceptibility.shape) != operator.object_shape or not bool(
            self.torch.all(self.torch.isfinite(self.susceptibility)).item()
        ):
            raise ValueError(
                f"susceptibility must be finite with shape {operator.object_shape}"
            )

    def interaction(self, field: Any) -> Any:
        value = self.operator._object_tensor(field)
        return self.operator.forward(self.susceptibility * value)

    def interaction_adjoint(self, field: Any) -> Any:
        value = self.operator._data_tensor(field)
        return self.torch.conj(self.susceptibility) * self.operator.adjoint(value)

    def residual(self, field: Any, incident_field: Any) -> Any:
        value = self.operator._object_tensor(field)
        incident = self.operator._object_tensor(incident_field)
        return value - incident - self.interaction(value)

    def adjoint_residual(self, field: Any, cotangent: Any) -> Any:
        value = self.operator._object_tensor(field)
        right = self.operator._object_tensor(cotangent)
        return value - right - self.interaction_adjoint(value)

    def _fixed_point(
        self,
        right_hand_side: Any,
        action: Callable[[Any], Any],
        residual: Callable[[Any, Any], Any],
        *,
        maximum_iterations: int,
        relative_tolerance: float,
        relaxation: float,
    ) -> dict[str, Any]:
        right = self.operator._object_tensor(right_hand_side)
        maximum_iterations = int(maximum_iterations)
        relative_tolerance = float(relative_tolerance)
        relaxation = float(relaxation)
        if maximum_iterations <= 0 or relative_tolerance <= 0.0:
            raise ValueError("iteration count and tolerance must be positive")
        if not 0.0 < relaxation <= 1.0:
            raise ValueError("relaxation must lie in (0,1]")
        field = right.clone()
        denominator = self.torch.linalg.vector_norm(right).clamp_min(1e-300)
        history: list[float] = []
        converged = False
        for _ in range(maximum_iterations):
            fixed_point = right + action(field)
            updated = (1.0 - relaxation) * field + relaxation * fixed_point
            relative_update = self.torch.linalg.vector_norm(updated - field) / denominator
            history.append(float(relative_update.detach().cpu()))
            field = updated
            if history[-1] <= relative_tolerance:
                converged = True
                break
        final_residual = residual(field, right)
        relative_residual = self.torch.linalg.vector_norm(final_residual) / denominator
        return {
            "field": field,
            "converged": converged,
            "iterations": len(history),
            "relative_update_history": history,
            "relative_residual": float(relative_residual.detach().cpu()),
        }

    def solve(
        self,
        incident_field: Any,
        *,
        maximum_iterations: int = 200,
        relative_tolerance: float = 1e-12,
        relaxation: float = 1.0,
    ) -> dict[str, Any]:
        return self._fixed_point(
            incident_field,
            self.interaction,
            self.residual,
            maximum_iterations=maximum_iterations,
            relative_tolerance=relative_tolerance,
            relaxation=relaxation,
        )

    def solve_adjoint(
        self,
        cotangent: Any,
        *,
        maximum_iterations: int = 200,
        relative_tolerance: float = 1e-12,
        relaxation: float = 1.0,
    ) -> dict[str, Any]:
        return self._fixed_point(
            cotangent,
            self.interaction_adjoint,
            self.adjoint_residual,
            maximum_iterations=maximum_iterations,
            relative_tolerance=relative_tolerance,
            relaxation=relaxation,
        )

    def susceptibility_vjp(self, field: Any, adjoint_field: Any) -> Any:
        """Return the VJP for a real-valued susceptibility perturbation."""

        value = self.operator._object_tensor(field)
        dual = self.operator._object_tensor(adjoint_field)
        green_adjoint = self.operator.adjoint(dual)
        return self.torch.real(self.torch.conj(green_adjoint) * value)

    def substrate_thickness_vjp(self, field: Any, adjoint_field: Any) -> Any:
        """Return the implicit real VJP with respect to slab thickness."""

        source = self.susceptibility * self.operator._object_tensor(field)
        dual = self.operator._object_tensor(adjoint_field)
        return self.operator.substrate_thickness_vjp(source, dual)
