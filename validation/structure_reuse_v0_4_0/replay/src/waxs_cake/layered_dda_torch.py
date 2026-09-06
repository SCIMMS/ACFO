"""GPU-resident matvecs and restarted GMRES for prepared layered DDA systems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .layered_dda import PreparedLayeredDDAOperator


@dataclass(frozen=True)
class TorchDDASolveResult:
    dipoles: Any
    converged: bool
    iterations: int
    cycles: int
    relative_residual: float
    restart: int
    preconditioner: str


class TorchPreparedLayeredDDAOperator:
    """Torch-resident counterpart of :class:`PreparedLayeredDDAOperator`."""

    def __init__(
        self,
        *,
        polarizability: Any,
        total_green: Any,
        reflected_height_derivative: Any,
        interaction_scale: Any,
        block_jacobi_inverse: Any,
        block_jacobi_adjoint_inverse: Any,
        torch: Any,
    ) -> None:
        self.torch = torch
        self.polarizability = polarizability
        self.total_green = total_green
        self.reflected_height_derivative = reflected_height_derivative
        self.interaction_scale = interaction_scale
        self.block_jacobi_inverse = block_jacobi_inverse
        self.block_jacobi_adjoint_inverse = block_jacobi_adjoint_inverse
        self.device = polarizability.device
        self.complex_dtype = polarizability.dtype
        self.real_dtype = (
            torch.float32 if self.complex_dtype == torch.complex64 else torch.float64
        )
        self.field_shape = (int(polarizability.shape[0]), 3)

    @classmethod
    def from_numpy(
        cls,
        operator: PreparedLayeredDDAOperator,
        *,
        torch: Any | None = None,
        device: Any = "cpu",
        complex_dtype: Any = "complex128",
    ) -> "TorchPreparedLayeredDDAOperator":
        if torch is None:
            import torch as torch_module

            torch = torch_module
        device = torch.device(device)
        if complex_dtype in {"complex64", np.complex64, torch.complex64}:
            dtype = torch.complex64
        elif complex_dtype in {"complex128", np.complex128, torch.complex128}:
            dtype = torch.complex128
        else:
            raise ValueError("complex_dtype must be complex64 or complex128")

        def tensor(values: Any) -> Any:
            # Some prepared NumPy caches are intentionally read-only.  Copying here
            # gives Torch owned, writable storage and avoids undefined behaviour if
            # a caller later mutates a resident tensor in place.
            array = np.array(values, copy=True, order="C")
            return torch.as_tensor(array, dtype=dtype, device=device)

        return cls(
            polarizability=tensor(operator.polarizability),
            total_green=tensor(operator.total_green_kernel()),
            reflected_height_derivative=tensor(
                operator.reflected_height_derivative
            ),
            interaction_scale=torch.as_tensor(
                operator.interaction_scale, dtype=dtype, device=device
            ),
            block_jacobi_inverse=tensor(
                operator.block_jacobi_inverse(adjoint=False)
            ),
            block_jacobi_adjoint_inverse=tensor(
                operator.block_jacobi_inverse(adjoint=True)
            ),
            torch=torch,
        )

    @property
    def resident_bytes(self) -> int:
        tensors = (
            self.polarizability,
            self.total_green,
            self.reflected_height_derivative,
            self.block_jacobi_inverse,
            self.block_jacobi_adjoint_inverse,
        )
        return int(sum(value.numel() * value.element_size() for value in tensors))

    def matvec(self, dipoles: Any) -> Any:
        values = self._field(dipoles, "dipoles")
        field = self.interaction_scale * self.torch.einsum(
            "iajb,jb->ia", self.total_green, values
        )
        interaction = self.torch.einsum(
            "iab,ib->ia", self.polarizability, field
        )
        return values - interaction

    def adjoint_matvec(self, cotangent: Any) -> Any:
        values = self._field(cotangent, "cotangent")
        alpha_adjoint = self.torch.einsum(
            "iba,ib->ia", self.torch.conj(self.polarizability), values
        )
        interaction = self.torch.conj(self.interaction_scale) * self.torch.einsum(
            "iajb,ia->jb", self.torch.conj(self.total_green), alpha_adjoint
        )
        return values - interaction

    def right_hand_side(self, incident_field: Any) -> Any:
        incident = self._field(incident_field, "incident_field")
        return self.torch.einsum("iab,ib->ia", self.polarizability, incident)

    def common_height_system_jvp(
        self,
        dipoles: Any,
        *,
        incident_rhs_height_derivative: Any | None = None,
    ) -> Any:
        values = self._field(dipoles, "dipoles")
        derivative_field = 2.0 * self.interaction_scale * self.torch.einsum(
            "iajb,jb->ia", self.reflected_height_derivative, values
        )
        right = self.torch.einsum(
            "iab,ib->ia", self.polarizability, derivative_field
        )
        if incident_rhs_height_derivative is not None:
            right = right + self._field(
                incident_rhs_height_derivative, "incident_rhs_height_derivative"
            )
        return right

    def rotation_green_derivative(self) -> Any:
        omega = self.torch.as_tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=self.complex_dtype,
            device=self.device,
        )
        return self.torch.einsum(
            "ac,icjb->iajb", omega, self.total_green
        ) - self.torch.einsum("iajc,cb->iajb", self.total_green, omega)

    def rotation_polarizability_derivative(self) -> Any:
        omega = self.torch.as_tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=self.complex_dtype,
            device=self.device,
        )
        return self.torch.einsum(
            "ac,icb->iab", omega, self.polarizability
        ) - self.torch.einsum("iac,cb->iab", self.polarizability, omega)

    def rotation_system_jvp(
        self,
        dipoles: Any,
        *,
        incident_rhs_rotation_derivative: Any | None = None,
    ) -> Any:
        """Apply the rotation derivative without materializing ``dG`` or ``dalpha``.

        The commutators are contracted as
        ``(Omega G - G Omega)p = Omega(Gp) - G(Omega p)``.  This keeps the
        derivative application matrix-free and avoids two pair-tensor-sized
        temporaries on the resident device.
        """

        values = self._field(dipoles, "dipoles")

        def omega_action(field: Any) -> Any:
            return self.torch.stack(
                (-field[..., 1], field[..., 0], self.torch.zeros_like(field[..., 2])),
                dim=-1,
            )

        base_field = self.interaction_scale * self.torch.einsum(
            "iajb,jb->ia", self.total_green, values
        )
        green_omega_dipoles = self.interaction_scale * self.torch.einsum(
            "iajb,jb->ia", self.total_green, omega_action(values)
        )
        derivative_field = omega_action(base_field) - green_omega_dipoles
        alpha_base_field = self.torch.einsum(
            "iab,ib->ia", self.polarizability, base_field
        )
        polarizability_commutator_action = omega_action(
            alpha_base_field
        ) - self.torch.einsum(
            "iab,ib->ia", self.polarizability, omega_action(base_field)
        )
        right = polarizability_commutator_action + self.torch.einsum(
            "iab,ib->ia", self.polarizability, derivative_field
        )
        if incident_rhs_rotation_derivative is not None:
            right = right + self._field(
                incident_rhs_rotation_derivative, "incident_rhs_rotation_derivative"
            )
        return right

    def solve(
        self,
        incident_field: Any,
        *,
        relative_tolerance: float = 1.0e-8,
        restart: int = 200,
        maximum_cycles: int = 20,
        preconditioner: bool = True,
    ) -> TorchDDASolveResult:
        return self.solve_right_hand_side(
            self.right_hand_side(incident_field),
            adjoint=False,
            relative_tolerance=relative_tolerance,
            restart=restart,
            maximum_cycles=maximum_cycles,
            preconditioner=preconditioner,
        )

    def solve_adjoint(
        self,
        cotangent: Any,
        *,
        relative_tolerance: float = 1.0e-8,
        restart: int = 200,
        maximum_cycles: int = 20,
        preconditioner: bool = True,
    ) -> TorchDDASolveResult:
        return self.solve_right_hand_side(
            cotangent,
            adjoint=True,
            relative_tolerance=relative_tolerance,
            restart=restart,
            maximum_cycles=maximum_cycles,
            preconditioner=preconditioner,
        )

    def solve_right_hand_side(
        self,
        right_hand_side: Any,
        *,
        adjoint: bool,
        relative_tolerance: float,
        restart: int,
        maximum_cycles: int,
        preconditioner: bool,
    ) -> TorchDDASolveResult:
        """Restarted, left-preconditioned complex GMRES on the resident device."""

        torch = self.torch
        right = self._field(right_hand_side, "right_hand_side")
        tolerance = float(relative_tolerance)
        restart = int(restart)
        maximum_cycles = int(maximum_cycles)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("relative_tolerance must be finite and positive")
        if restart <= 0 or maximum_cycles <= 0:
            raise ValueError("restart and maximum_cycles must be positive")
        size = int(np.prod(self.field_shape))
        restart = min(restart, size)
        action = self.adjoint_matvec if adjoint else self.matvec
        inverse_blocks = (
            self.block_jacobi_adjoint_inverse
            if adjoint
            else self.block_jacobi_inverse
        )

        def precondition(value: Any) -> Any:
            if not preconditioner:
                return value
            return torch.einsum("iab,ib->ia", inverse_blocks, value)

        x = torch.zeros_like(right)
        denominator = torch.linalg.vector_norm(right).clamp_min(1.0e-300)
        iterations = 0
        relative_residual = float("inf")
        completed_cycles = 0
        for cycle in range(maximum_cycles):
            residual = right - action(x)
            relative_residual = float(
                (torch.linalg.vector_norm(residual) / denominator).detach().cpu()
            )
            if relative_residual <= tolerance:
                completed_cycles = cycle
                break
            preconditioned = precondition(residual).reshape(-1)
            beta = torch.linalg.vector_norm(preconditioned)
            if float(beta.detach().cpu()) == 0.0:
                completed_cycles = cycle
                break
            basis = torch.empty(
                (size, restart + 1),
                dtype=self.complex_dtype,
                device=self.device,
            )
            hessenberg = torch.zeros(
                (restart + 1, restart),
                dtype=self.complex_dtype,
                device=self.device,
            )
            basis[:, 0] = preconditioned / beta
            used = restart
            for column in range(restart):
                vector = precondition(
                    action(basis[:, column].reshape(self.field_shape))
                ).reshape(-1)
                active = basis[:, : column + 1]
                coefficients = torch.matmul(torch.conj(active).T, vector)
                vector = vector - torch.matmul(active, coefficients)
                correction = torch.matmul(torch.conj(active).T, vector)
                vector = vector - torch.matmul(active, correction)
                coefficients = coefficients + correction
                hessenberg[: column + 1, column] = coefficients
                next_norm = torch.linalg.vector_norm(vector)
                hessenberg[column + 1, column] = next_norm
                iterations += 1
                if float(next_norm.detach().cpu()) <= 1.0e-14 * max(
                    float(beta.detach().cpu()), 1.0
                ):
                    used = column + 1
                    break
                basis[:, column + 1] = vector / next_norm
            target = torch.zeros(
                used + 1, dtype=self.complex_dtype, device=self.device
            )
            target[0] = beta
            solution = torch.linalg.lstsq(
                hessenberg[: used + 1, :used], target
            ).solution
            x = x + torch.matmul(basis[:, :used], solution).reshape(self.field_shape)
            completed_cycles = cycle + 1
        residual = right - action(x)
        relative_residual = float(
            (torch.linalg.vector_norm(residual) / denominator).detach().cpu()
        )
        return TorchDDASolveResult(
            dipoles=x,
            converged=bool(relative_residual <= tolerance),
            iterations=iterations,
            cycles=completed_cycles,
            relative_residual=relative_residual,
            restart=restart,
            preconditioner="block_jacobi" if preconditioner else "none",
        )

    def _field(self, values: Any, name: str) -> Any:
        tensor = self.torch.as_tensor(
            values, dtype=self.complex_dtype, device=self.device
        )
        if tuple(tensor.shape) != self.field_shape or not bool(
            self.torch.all(self.torch.isfinite(tensor)).item()
        ):
            raise ValueError(f"{name} must be finite with shape {self.field_shape}")
        return tensor
