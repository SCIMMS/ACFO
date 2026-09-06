"""Matrix-free real-parameter Gauss--Newton operators for ACFO models.

The data may be real or complex, while the optimized parameter vector is
real.  All adjoints therefore use the real Euclidean pairing
``Re(sum(conj(left) * right))``.  The implementation accepts analytic JVP/VJP
callbacks, but can also obtain them from PyTorch autograd.  Explicit dense
Jacobians are constructed only for validation and small identifiability
audits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence


def _real_inner(torch: Any, left: Any, right: Any) -> Any:
    return torch.real(torch.sum(torch.conj(left) * right))


@dataclass(frozen=True)
class GaussNewtonSpectrum:
    """Small-parameter Fisher/Gauss--Newton identifiability summary."""

    eigenvalues: list[float]
    singular_values: list[float]
    numerical_rank: int
    condition_number: float
    covariance: list[list[float]]
    correlation: list[list[float]]
    standard_errors: list[float]
    profile_95_half_widths: list[float]
    parameter_labels: list[str]
    damping: float

    def to_dict(self) -> dict[str, object]:
        return {
            "eigenvalues": self.eigenvalues,
            "singular_values": self.singular_values,
            "numerical_rank": self.numerical_rank,
            "condition_number": self.condition_number,
            "covariance": self.covariance,
            "correlation": self.correlation,
            "standard_errors": self.standard_errors,
            "profile_95_half_widths": self.profile_95_half_widths,
            "parameter_labels": self.parameter_labels,
            "damping": self.damping,
        }


class TorchMatrixFreeGaussNewton:
    """Apply ``J* W J + lambda D`` without materializing ``J``.

    Parameters
    ----------
    model
        Callable from one real parameter tensor to a real or complex data
        tensor.
    parameters
        Linearization point.  It is detached and frozen by the operator.
    weights
        Non-negative diagonal data precision.  A scalar or an array matching
        the model output is accepted.
    damping, damping_diagonal
        Levenberg--Marquardt damping ``damping * diag(damping_diagonal)``.
    analytic_jvp, analytic_vjp
        Optional callbacks with signatures ``(parameters, tangent)`` and
        ``(parameters, data_cotangent)``.  Supplying both keeps the normal
        product on the analytic ACFO derivative path.
    """

    def __init__(
        self,
        model: Callable[[Any], Any],
        parameters: Any,
        *,
        weights: Any | None = None,
        damping: float = 0.0,
        damping_diagonal: Any | None = None,
        analytic_jvp: Callable[[Any, Any], Any] | None = None,
        analytic_vjp: Callable[[Any, Any], Any] | None = None,
        parameter_labels: Sequence[str] | None = None,
        torch: Any | None = None,
    ) -> None:
        if torch is None:
            import torch as torch_module

            torch = torch_module
        self.torch = torch
        self.model = model
        theta = torch.as_tensor(parameters)
        if theta.ndim != 1 or theta.numel() == 0 or torch.is_complex(theta):
            raise ValueError("parameters must be a non-empty real vector")
        if not bool(torch.all(torch.isfinite(theta)).item()):
            raise ValueError("parameters must be finite")
        self.parameters = theta.detach().clone()
        self.n_parameters = int(theta.numel())
        self.damping = float(damping)
        if not self.damping >= 0.0:
            raise ValueError("damping must be non-negative")
        if (analytic_jvp is None) != (analytic_vjp is None):
            raise ValueError("analytic_jvp and analytic_vjp must be supplied together")
        self.analytic_jvp = analytic_jvp
        self.analytic_vjp = analytic_vjp

        with torch.no_grad():
            prediction = model(self.parameters)
        if prediction.numel() == 0 or not bool(torch.all(torch.isfinite(prediction)).item()):
            raise ValueError("model output must be non-empty and finite")
        self.data_shape = tuple(prediction.shape)
        self.data_dtype = prediction.dtype
        self.device = prediction.device

        if weights is None:
            weight = torch.ones(self.data_shape, dtype=self.parameters.dtype, device=self.device)
        else:
            weight = torch.as_tensor(weights, dtype=self.parameters.dtype, device=self.device)
            if weight.numel() == 1:
                weight = torch.full(self.data_shape, float(weight), dtype=self.parameters.dtype, device=self.device)
            elif tuple(weight.shape) != self.data_shape:
                try:
                    weight = torch.broadcast_to(weight, self.data_shape).clone()
                except RuntimeError as error:
                    raise ValueError(
                        "weights must be scalar or broadcast to the model output"
                    ) from error
        if not bool(torch.all(torch.isfinite(weight)).item()) or bool(torch.any(weight < 0).item()):
            raise ValueError("weights must be finite and non-negative")
        self.weights = weight

        if damping_diagonal is None:
            diagonal = torch.ones_like(self.parameters)
        else:
            diagonal = torch.as_tensor(
                damping_diagonal,
                dtype=self.parameters.dtype,
                device=self.parameters.device,
            )
            if tuple(diagonal.shape) != tuple(self.parameters.shape):
                raise ValueError("damping_diagonal must match parameters")
        if not bool(torch.all(torch.isfinite(diagonal)).item()) or bool(torch.any(diagonal <= 0).item()):
            raise ValueError("damping_diagonal must be finite and positive")
        self.damping_diagonal = diagonal

        if parameter_labels is None:
            labels = [f"parameter_{index}" for index in range(self.n_parameters)]
        else:
            labels = [str(value) for value in parameter_labels]
            if len(labels) != self.n_parameters or len(set(labels)) != len(labels):
                raise ValueError("parameter_labels must be unique and match parameters")
        self.parameter_labels = labels

    def prediction(self) -> Any:
        """Evaluate the model at the frozen linearization point."""

        return self.model(self.parameters)

    def _parameter_vector(self, values: Any, *, name: str) -> Any:
        tensor = self.torch.as_tensor(
            values,
            dtype=self.parameters.dtype,
            device=self.parameters.device,
        )
        if tuple(tensor.shape) != tuple(self.parameters.shape):
            raise ValueError(f"{name} must match parameters")
        if not bool(self.torch.all(self.torch.isfinite(tensor)).item()):
            raise ValueError(f"{name} must be finite")
        return tensor

    def _data_tensor(self, values: Any, *, name: str) -> Any:
        tensor = self.torch.as_tensor(values, dtype=self.data_dtype, device=self.device)
        if tuple(tensor.shape) != self.data_shape:
            raise ValueError(f"{name} must have shape {self.data_shape}")
        if not bool(self.torch.all(self.torch.isfinite(tensor)).item()):
            raise ValueError(f"{name} must be finite")
        return tensor

    def jvp(self, tangent: Any) -> Any:
        """Apply the Jacobian to a real parameter tangent."""

        vector = self._parameter_vector(tangent, name="tangent")
        if self.analytic_jvp is not None:
            result = self.analytic_jvp(self.parameters, vector)
        else:
            _, result = self.torch.autograd.functional.jvp(
                self.model,
                self.parameters,
                vector,
                create_graph=False,
                strict=True,
            )
        return self._data_tensor(result, name="JVP result")

    def vjp(self, cotangent: Any) -> Any:
        """Apply the real adjoint Jacobian to a data cotangent."""

        dual = self._data_tensor(cotangent, name="cotangent")
        if self.analytic_vjp is not None:
            result = self.analytic_vjp(self.parameters, dual)
            return self._parameter_vector(result, name="VJP result")
        theta = self.parameters.detach().clone().requires_grad_(True)
        prediction = self.model(theta)
        pairing = _real_inner(self.torch, dual, prediction)
        (gradient,) = self.torch.autograd.grad(pairing, theta, create_graph=False)
        return self._parameter_vector(gradient, name="VJP result")

    def weighted_jvp(self, tangent: Any) -> Any:
        return self.weights * self.jvp(tangent)

    def normal_matvec(self, tangent: Any, *, include_damping: bool = True) -> Any:
        """Apply ``J* W J`` and optional Levenberg--Marquardt damping."""

        vector = self._parameter_vector(tangent, name="tangent")
        result = self.vjp(self.weights * self.jvp(vector))
        if include_damping and self.damping:
            result = result + self.damping * self.damping_diagonal * vector
        return result

    def damped_hvp(self, tangent: Any) -> Any:
        """Alias for the damped Gauss--Newton Hessian-vector product."""

        return self.normal_matvec(tangent, include_damping=True)

    def dense_jacobian(self) -> Any:
        """Materialize complex/real Jacobian columns for a small validation case."""

        eye = self.torch.eye(
            self.n_parameters,
            dtype=self.parameters.dtype,
            device=self.parameters.device,
        )
        return self.torch.stack([self.jvp(eye[index]) for index in range(self.n_parameters)], dim=-1)

    def dense_real_jacobian(self, *, weighted: bool = False) -> Any:
        """Return the real-stacked Jacobian used by Fisher diagnostics."""

        jacobian = self.dense_jacobian().reshape(-1, self.n_parameters)
        weight = self.weights.reshape(-1)
        if weighted:
            root = self.torch.sqrt(weight)[:, None]
        else:
            root = self.torch.ones((weight.numel(), 1), dtype=weight.dtype, device=weight.device)
        if self.torch.is_complex(jacobian):
            return self.torch.cat((root * jacobian.real, root * jacobian.imag), dim=0)
        return root * jacobian

    def fisher_matrix(self, *, include_damping: bool = False) -> Any:
        """Materialize a small Fisher matrix by matrix-free normal products."""

        eye = self.torch.eye(
            self.n_parameters,
            dtype=self.parameters.dtype,
            device=self.parameters.device,
        )
        columns = [
            self.normal_matvec(eye[index], include_damping=include_damping)
            for index in range(self.n_parameters)
        ]
        matrix = self.torch.stack(columns, dim=1)
        return 0.5 * (matrix + matrix.T)

    def spectrum(
        self,
        *,
        noise_variance: float = 1.0,
        rcond: float | None = None,
    ) -> GaussNewtonSpectrum:
        """Return rank, correlations and local profile-width diagnostics."""

        torch = self.torch
        variance = float(noise_variance)
        if not variance > 0.0:
            raise ValueError("noise_variance must be positive")
        fisher = self.fisher_matrix(include_damping=False)
        eigenvalues = torch.linalg.eigvalsh(fisher)
        maximum = float(torch.max(torch.abs(eigenvalues)).detach().cpu())
        threshold = (
            float(rcond) * maximum
            if rcond is not None
            else torch.finfo(fisher.dtype).eps * max(fisher.shape) * max(maximum, 1.0)
        )
        positive = eigenvalues > threshold
        rank = int(torch.count_nonzero(positive).detach().cpu())
        if rank == self.n_parameters:
            condition = float((eigenvalues[-1] / eigenvalues[0]).detach().cpu())
        else:
            condition = float("inf")

        damped = fisher + self.damping * torch.diag(self.damping_diagonal)
        covariance = variance * torch.linalg.pinv(damped, rcond=(rcond or 1e-12), hermitian=True)
        diagonal = torch.clamp(torch.diag(covariance), min=0.0)
        standard = torch.sqrt(diagonal)
        denominator = standard[:, None] * standard[None, :]
        correlation = torch.where(
            denominator > 0,
            covariance / denominator,
            torch.zeros_like(covariance),
        )
        singular_values = torch.sqrt(torch.clamp(eigenvalues, min=0.0))
        return GaussNewtonSpectrum(
            eigenvalues=eigenvalues.detach().cpu().tolist(),
            singular_values=singular_values.detach().cpu().tolist(),
            numerical_rank=rank,
            condition_number=condition,
            covariance=covariance.detach().cpu().tolist(),
            correlation=correlation.detach().cpu().tolist(),
            standard_errors=standard.detach().cpu().tolist(),
            profile_95_half_widths=(1.959963984540054 * standard).detach().cpu().tolist(),
            parameter_labels=list(self.parameter_labels),
            damping=self.damping,
        )

    def adjoint_error(self, tangent: Any, cotangent: Any) -> float:
        """Relative error in ``<Jv,u>_R = <v,J*u>``."""

        vector = self._parameter_vector(tangent, name="tangent")
        dual = self._data_tensor(cotangent, name="cotangent")
        left = _real_inner(self.torch, dual, self.jvp(vector))
        right = self.torch.dot(vector, self.vjp(dual))
        denominator = (self.torch.abs(left) + self.torch.abs(right)).clamp_min(1e-300)
        return float((self.torch.abs(left - right) / denominator).detach().cpu())
