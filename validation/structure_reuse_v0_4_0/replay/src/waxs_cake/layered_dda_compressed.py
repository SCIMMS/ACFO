"""Lattice-compressed layered DDA operators.

Regular voxel indices make homogeneous-space interactions depend only on an
integer displacement and reflected interactions depend only on lateral
displacement plus the source--target height sum.  This module stores those
unique dyadic blocks and gathers them in target chunks during matrix-free
applications, avoiding three dense ``N x 3 x N x 3`` tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .layered_dda import (
    PreparedLayeredDDAOperator,
    _polarizability_blocks,
    free_space_electric_dyadic,
)
from .layered_dda_torch import TorchPreparedLayeredDDAOperator
from .layered_dyadic_green import PreparedLayeredDyadicGreenOperator


Array = np.ndarray


@dataclass(frozen=True)
class PreparedLatticeCompressedLayeredDDAOperator:
    """Prepared DDA system backed by unique regular-lattice Green blocks."""

    positions: Array
    lattice_indices: Array
    lattice_shape: tuple[int, int, int]
    lattice_steps: tuple[float, float, float]
    lattice_rotation: float
    bottom_height: float
    polarizability: Array
    free_space_kernel_grid: Array
    reflected_kernel_grid: Array
    reflected_height_derivative_grid: Array
    interaction_scale: complex
    green_contracts: tuple[dict[str, int | float | bool | str], ...]
    target_chunk_size: int

    @classmethod
    def build(
        cls,
        *,
        positions: Any,
        lattice_indices: Any,
        lattice_shape: tuple[int, int, int],
        lattice_steps: tuple[float, float, float],
        lattice_rotation: float,
        bottom_height: float,
        polarizability: Any,
        radial_wavenumbers: Any,
        quadrature_weights: Any,
        phi: Any,
        upper_wavenumber: float,
        lower_wavenumber: float,
        damping: float,
        interaction_scale: complex | None = None,
        green_chunk_size: int = 256,
        target_chunk_size: int = 32,
        kernel_tolerance: float = 1.0e-12,
        tensor_tail_tolerance: float = 1.0e-13,
        mode_padding: int = 8,
        miller_margin: int = 32,
    ) -> "PreparedLatticeCompressedLayeredDDAOperator":
        points = np.asarray(positions, dtype=np.float64)
        indices = np.asarray(lattice_indices, dtype=np.int64)
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or points.shape[0] == 0
            or not np.all(np.isfinite(points))
        ):
            raise ValueError("positions must be a non-empty finite (n,3) array")
        if indices.shape != points.shape or np.any(indices < 0):
            raise ValueError("lattice_indices must be non-negative with shape (n,3)")
        shape = tuple(int(value) for value in lattice_shape)
        if len(shape) != 3 or any(value <= 0 for value in shape):
            raise ValueError("lattice_shape must contain three positive integers")
        if np.any(indices >= np.asarray(shape)[None, :]):
            raise ValueError("lattice_indices exceed lattice_shape")
        steps = tuple(float(value) for value in lattice_steps)
        rotation = float(lattice_rotation)
        bottom = float(bottom_height)
        if (
            len(steps) != 3
            or any(not np.isfinite(value) or value <= 0.0 for value in steps)
            or not np.isfinite(rotation)
            or not np.isfinite(bottom)
            or bottom <= 0.0
        ):
            raise ValueError("lattice steps, rotation and bottom height are invalid")
        green_chunk_size = int(green_chunk_size)
        target_chunk_size = int(target_chunk_size)
        if green_chunk_size <= 0 or target_chunk_size <= 0:
            raise ValueError("chunk sizes must be positive")
        alpha = _polarizability_blocks(polarizability, points.shape[0])
        scale = (
            complex(float(upper_wavenumber) ** 2)
            if interaction_scale is None
            else complex(interaction_scale)
        )

        nx, ny, nz = shape
        dx_index = np.arange(-(nx - 1), nx, dtype=np.int64)
        dy_index = np.arange(-(ny - 1), ny, dtype=np.int64)
        dz_index = np.arange(-(nz - 1), nz, dtype=np.int64)
        ddx, ddy, ddz = np.meshgrid(
            dx_index, dy_index, dz_index, indexing="ij"
        )
        local_x = ddx.astype(np.float64) * steps[0]
        local_y = ddy.astype(np.float64) * steps[1]
        cosine = np.cos(rotation)
        sine = np.sin(rotation)
        displacement = np.stack(
            (
                cosine * local_x - sine * local_y,
                sine * local_x + cosine * local_y,
                ddz.astype(np.float64) * steps[2],
            ),
            axis=-1,
        )
        free = np.zeros(displacement.shape[:-1] + (3, 3), dtype=np.complex128)
        nonzero = np.any(np.stack((ddx, ddy, ddz), axis=-1) != 0, axis=-1)
        free[nonzero] = free_space_electric_dyadic(
            displacement[nonzero], complex(float(upper_wavenumber), 0.0)
        )

        z_sum_index = np.arange(2 * nz - 1, dtype=np.int64)
        rdx, rdy, rsum = np.meshgrid(
            dx_index, dy_index, z_sum_index, indexing="ij"
        )
        reflected_local_x = rdx.astype(np.float64) * steps[0]
        reflected_local_y = rdy.astype(np.float64) * steps[1]
        reflected_x = cosine * reflected_local_x - sine * reflected_local_y
        reflected_y = sine * reflected_local_x + cosine * reflected_local_y
        reflected_height = 2.0 * bottom + (rsum.astype(np.float64) + 1.0) * steps[2]
        flat_geometry = np.column_stack(
            (reflected_x.ravel(), reflected_y.ravel(), reflected_height.ravel())
        )
        reflected_flat = np.empty((flat_geometry.shape[0], 3, 3), dtype=np.complex128)
        height_flat = np.empty_like(reflected_flat)
        contracts: list[dict[str, int | float | bool | str]] = []
        for start in range(0, flat_geometry.shape[0], green_chunk_size):
            stop = min(start + green_chunk_size, flat_geometry.shape[0])
            geometry = flat_geometry[start:stop]
            plan = PreparedLayeredDyadicGreenOperator.build(
                radial_wavenumbers=radial_wavenumbers,
                quadrature_weights=quadrature_weights,
                phi=phi,
                x=geometry[:, 0],
                y=geometry[:, 1],
                height=geometry[:, 2],
                upper_wavenumber=upper_wavenumber,
                lower_wavenumber=lower_wavenumber,
                damping=damping,
                kernel_tolerance=kernel_tolerance,
                tensor_tail_tolerance=tensor_tail_tolerance,
                mode_padding=mode_padding,
                miller_margin=miller_margin,
            )
            reflected_flat[start:stop] = np.transpose(plan.kernel(), (2, 0, 1))
            height_flat[start:stop] = np.transpose(
                plan.derivative_kernels()[2], (2, 0, 1)
            )
            contracts.append(plan.contract.to_dict())
        reflected_shape = (2 * nx - 1, 2 * ny - 1, 2 * nz - 1, 3, 3)
        return cls(
            positions=_readonly(points),
            lattice_indices=_readonly(indices.astype(np.int32)),
            lattice_shape=shape,
            lattice_steps=steps,
            lattice_rotation=rotation,
            bottom_height=bottom,
            polarizability=_readonly(alpha),
            free_space_kernel_grid=_readonly(free),
            reflected_kernel_grid=_readonly(reflected_flat.reshape(reflected_shape)),
            reflected_height_derivative_grid=_readonly(
                height_flat.reshape(reflected_shape)
            ),
            interaction_scale=scale,
            green_contracts=tuple(contracts),
            target_chunk_size=target_chunk_size,
        )

    @property
    def dipole_count(self) -> int:
        return int(self.positions.shape[0])

    @property
    def field_shape(self) -> tuple[int, int]:
        return (self.dipole_count, 3)

    @property
    def kernel_bytes(self) -> int:
        return int(
            self.free_space_kernel_grid.nbytes
            + self.reflected_kernel_grid.nbytes
            + self.reflected_height_derivative_grid.nbytes
        )

    @property
    def dense_three_kernel_bytes(self) -> int:
        return int(3 * self.dipole_count**2 * 9 * np.dtype(np.complex128).itemsize)

    @property
    def compression_ratio(self) -> float:
        return float(self.dense_three_kernel_bytes / max(self.kernel_bytes, 1))

    def block_jacobi_inverse(self, *, adjoint: bool = False) -> Array:
        indices = self.lattice_indices
        nx, ny, _ = self.lattice_shape
        reflected = self.reflected_kernel_grid[
            nx - 1, ny - 1, 2 * indices[:, 2]
        ]
        interaction = self.interaction_scale * np.einsum(
            "iac,icb->iab", self.polarizability, reflected, optimize=True
        )
        blocks = np.eye(3, dtype=np.complex128)[None, :, :] - interaction
        if adjoint:
            blocks = np.conjugate(np.swapaxes(blocks, 1, 2))
        return np.linalg.inv(blocks)

    def right_hand_side(self, incident_field: Any) -> Array:
        incident = self._field(incident_field, "incident_field")
        return np.einsum("iab,ib->ia", self.polarizability, incident, optimize=True)

    def green_action(self, dipoles: Any) -> Array:
        values = self._field(dipoles, "dipoles")
        return self._kernel_action(values, self.reflected_kernel_grid)

    def matvec(self, dipoles: Any) -> Array:
        values = self._field(dipoles, "dipoles")
        field = self.interaction_scale * self._kernel_action(
            values, self.reflected_kernel_grid
        )
        return values - np.einsum(
            "iab,ib->ia", self.polarizability, field, optimize=True
        )

    def adjoint_matvec(self, cotangent: Any) -> Array:
        values = self._field(cotangent, "cotangent")
        alpha_adjoint = np.einsum(
            "iba,ib->ia", np.conjugate(self.polarizability), values, optimize=True
        )
        output = np.zeros_like(values)
        for start in range(0, self.dipole_count, self.target_chunk_size):
            stop = min(start + self.target_chunk_size, self.dipole_count)
            green = self._gather_total(start, stop, self.reflected_kernel_grid)
            output += np.conjugate(self.interaction_scale) * np.einsum(
                "tjab,ta->jb",
                np.conjugate(green),
                alpha_adjoint[start:stop],
                optimize=True,
            )
        return values - output

    def reflected_height_action(self, dipoles: Any) -> Array:
        values = self._field(dipoles, "dipoles")
        output = np.zeros_like(values)
        for start in range(0, self.dipole_count, self.target_chunk_size):
            stop = min(start + self.target_chunk_size, self.dipole_count)
            reflected = self._gather_reflected(
                start, stop, self.reflected_height_derivative_grid
            )
            output[start:stop] = np.einsum(
                "tjab,jb->ta", reflected, values, optimize=True
            )
        return output

    def _kernel_action(self, values: Array, reflected_grid: Array) -> Array:
        output = np.empty_like(values)
        for start in range(0, self.dipole_count, self.target_chunk_size):
            stop = min(start + self.target_chunk_size, self.dipole_count)
            green = self._gather_total(start, stop, reflected_grid)
            output[start:stop] = np.einsum(
                "tjab,jb->ta", green, values, optimize=True
            )
        return output

    def _gather_total(self, start: int, stop: int, reflected_grid: Array) -> Array:
        return self._gather_free(start, stop) + self._gather_reflected(
            start, stop, reflected_grid
        )

    def _gather_free(self, start: int, stop: int) -> Array:
        indices = self.lattice_indices
        delta = indices[start:stop, None, :] - indices[None, :, :]
        nx, ny, nz = self.lattice_shape
        return self.free_space_kernel_grid[
            delta[..., 0] + nx - 1,
            delta[..., 1] + ny - 1,
            delta[..., 2] + nz - 1,
        ]

    def _gather_reflected(self, start: int, stop: int, grid: Array) -> Array:
        indices = self.lattice_indices
        delta_xy = indices[start:stop, None, :2] - indices[None, :, :2]
        z_sum = indices[start:stop, None, 2] + indices[None, :, 2]
        nx, ny, _ = self.lattice_shape
        return grid[
            delta_xy[..., 0] + nx - 1,
            delta_xy[..., 1] + ny - 1,
            z_sum,
        ]

    def _field(self, values: Any, name: str) -> Array:
        array = np.asarray(values, dtype=np.complex128)
        if array.shape != self.field_shape or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite with shape {self.field_shape}")
        return array


class TorchPreparedLatticeCompressedDDAOperator(TorchPreparedLayeredDDAOperator):
    """GPU-resident gather-and-contract counterpart of the compressed plan."""

    def __init__(
        self,
        *,
        lattice_indices: Any,
        lattice_shape: tuple[int, int, int],
        polarizability: Any,
        free_space_kernel_grid: Any,
        reflected_kernel_grid: Any,
        reflected_height_derivative_grid: Any,
        interaction_scale: Any,
        block_jacobi_inverse: Any,
        block_jacobi_adjoint_inverse: Any,
        target_chunk_size: int,
        torch: Any,
    ) -> None:
        self.torch = torch
        self.lattice_indices = lattice_indices
        self.lattice_shape = lattice_shape
        self.polarizability = polarizability
        self.free_space_kernel_grid = free_space_kernel_grid
        self.reflected_kernel_grid = reflected_kernel_grid
        self.reflected_height_derivative_grid = reflected_height_derivative_grid
        self.interaction_scale = interaction_scale
        self.block_jacobi_inverse = block_jacobi_inverse
        self.block_jacobi_adjoint_inverse = block_jacobi_adjoint_inverse
        self.target_chunk_size = int(target_chunk_size)
        self.device = polarizability.device
        self.complex_dtype = polarizability.dtype
        self.real_dtype = (
            torch.float32 if self.complex_dtype == torch.complex64 else torch.float64
        )
        self.field_shape = (int(polarizability.shape[0]), 3)

    @classmethod
    def from_numpy(
        cls,
        operator: PreparedLatticeCompressedLayeredDDAOperator,
        *,
        torch: Any | None = None,
        device: Any = "cpu",
        complex_dtype: Any = "complex128",
        target_chunk_size: int | None = None,
    ) -> "TorchPreparedLatticeCompressedDDAOperator":
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
            return torch.as_tensor(
                np.array(values, copy=True, order="C"), dtype=dtype, device=device
            )

        return cls(
            lattice_indices=torch.as_tensor(
                np.array(operator.lattice_indices, copy=True),
                dtype=torch.long,
                device=device,
            ),
            lattice_shape=operator.lattice_shape,
            polarizability=tensor(operator.polarizability),
            free_space_kernel_grid=tensor(operator.free_space_kernel_grid),
            reflected_kernel_grid=tensor(operator.reflected_kernel_grid),
            reflected_height_derivative_grid=tensor(
                operator.reflected_height_derivative_grid
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
            target_chunk_size=(
                operator.target_chunk_size
                if target_chunk_size is None
                else int(target_chunk_size)
            ),
            torch=torch,
        )

    @property
    def resident_bytes(self) -> int:
        tensors = (
            self.lattice_indices,
            self.polarizability,
            self.free_space_kernel_grid,
            self.reflected_kernel_grid,
            self.reflected_height_derivative_grid,
            self.block_jacobi_inverse,
            self.block_jacobi_adjoint_inverse,
        )
        return int(sum(value.numel() * value.element_size() for value in tensors))

    def matvec(self, dipoles: Any) -> Any:
        values = self._field(dipoles, "dipoles")
        field = self.interaction_scale * self._kernel_action(
            values, self.reflected_kernel_grid
        )
        return values - self.torch.einsum(
            "iab,ib->ia", self.polarizability, field
        )

    def adjoint_matvec(self, cotangent: Any) -> Any:
        values = self._field(cotangent, "cotangent")
        alpha_adjoint = self.torch.einsum(
            "iba,ib->ia", self.torch.conj(self.polarizability), values
        )
        output = self.torch.zeros_like(values)
        for start in range(0, self.field_shape[0], self.target_chunk_size):
            stop = min(start + self.target_chunk_size, self.field_shape[0])
            green = self._gather_total(start, stop, self.reflected_kernel_grid)
            output = output + self.torch.conj(self.interaction_scale) * self.torch.einsum(
                "tjab,ta->jb", self.torch.conj(green), alpha_adjoint[start:stop]
            )
        return values - output

    def common_height_system_jvp(
        self,
        dipoles: Any,
        *,
        incident_rhs_height_derivative: Any | None = None,
    ) -> Any:
        values = self._field(dipoles, "dipoles")
        derivative = self._reflected_action(
            values, self.reflected_height_derivative_grid
        )
        right = self.torch.einsum(
            "iab,ib->ia",
            self.polarizability,
            2.0 * self.interaction_scale * derivative,
        )
        if incident_rhs_height_derivative is not None:
            right = right + self._field(
                incident_rhs_height_derivative, "incident_rhs_height_derivative"
            )
        return right

    def rotation_system_jvp(
        self,
        dipoles: Any,
        *,
        incident_rhs_rotation_derivative: Any | None = None,
    ) -> Any:
        values = self._field(dipoles, "dipoles")

        def omega_action(field: Any) -> Any:
            return self.torch.stack(
                (-field[..., 1], field[..., 0], self.torch.zeros_like(field[..., 2])),
                dim=-1,
            )

        base_field = self.interaction_scale * self._kernel_action(
            values, self.reflected_kernel_grid
        )
        green_omega = self.interaction_scale * self._kernel_action(
            omega_action(values), self.reflected_kernel_grid
        )
        derivative_field = omega_action(base_field) - green_omega
        alpha_base = self.torch.einsum(
            "iab,ib->ia", self.polarizability, base_field
        )
        alpha_commutator = omega_action(alpha_base) - self.torch.einsum(
            "iab,ib->ia", self.polarizability, omega_action(base_field)
        )
        right = alpha_commutator + self.torch.einsum(
            "iab,ib->ia", self.polarizability, derivative_field
        )
        if incident_rhs_rotation_derivative is not None:
            right = right + self._field(
                incident_rhs_rotation_derivative, "incident_rhs_rotation_derivative"
            )
        return right

    def _kernel_action(self, values: Any, reflected_grid: Any) -> Any:
        output = self.torch.empty_like(values)
        for start in range(0, self.field_shape[0], self.target_chunk_size):
            stop = min(start + self.target_chunk_size, self.field_shape[0])
            green = self._gather_total(start, stop, reflected_grid)
            output[start:stop] = self.torch.einsum(
                "tjab,jb->ta", green, values
            )
        return output

    def _reflected_action(self, values: Any, grid: Any) -> Any:
        output = self.torch.empty_like(values)
        for start in range(0, self.field_shape[0], self.target_chunk_size):
            stop = min(start + self.target_chunk_size, self.field_shape[0])
            reflected = self._gather_reflected(start, stop, grid)
            output[start:stop] = self.torch.einsum(
                "tjab,jb->ta", reflected, values
            )
        return output

    def _gather_total(self, start: int, stop: int, reflected_grid: Any) -> Any:
        indices = self.lattice_indices
        delta = indices[start:stop, None, :] - indices[None, :, :]
        nx, ny, nz = self.lattice_shape
        free = self.free_space_kernel_grid[
            delta[..., 0] + nx - 1,
            delta[..., 1] + ny - 1,
            delta[..., 2] + nz - 1,
        ]
        return free + self._gather_reflected(start, stop, reflected_grid)

    def _gather_reflected(self, start: int, stop: int, grid: Any) -> Any:
        indices = self.lattice_indices
        delta_xy = indices[start:stop, None, :2] - indices[None, :, :2]
        z_sum = indices[start:stop, None, 2] + indices[None, :, 2]
        nx, ny, _ = self.lattice_shape
        return grid[
            delta_xy[..., 0] + nx - 1,
            delta_xy[..., 1] + ny - 1,
            z_sum,
        ]


def _readonly(values: Array) -> Array:
    array = np.ascontiguousarray(values)
    array.setflags(write=False)
    return array
