"""Prepared GPU 3-D FFT and fixed-target trilinear interpolation."""

from __future__ import annotations

from typing import Any

import numpy as np


class TorchPreparedPeriodicCoefficientFFT3D:
    """Zero-padded 3-D FFT for a discrete Cartesian coefficient lattice.

    Targets are dimensionless coordinates conjugate to integer lattice
    indices.  Trilinear interpolation is periodic at the ``[-pi, pi)`` seam,
    which is required when physical targets exceed the first Brillouin zone
    and are wrapped before evaluation.  Unlike the voxel-density interpolator,
    this class applies neither a cell-volume factor nor a voxel sinc factor.
    """

    def __init__(
        self,
        shape: tuple[int, int, int],
        targets: Any,
        *,
        torch: Any,
        device: Any,
        dtype: str = "complex64",
        phase_sign: int = -1,
        pad_factor: int = 2,
    ) -> None:
        if phase_sign not in (-1, 1):
            raise ValueError("phase_sign must be -1 or +1")
        if dtype == "complex64":
            complex_dtype, real_dtype = torch.complex64, torch.float32
        elif dtype == "complex128":
            complex_dtype, real_dtype = torch.complex128, torch.float64
        else:
            raise ValueError("dtype must be 'complex64' or 'complex128'")
        grid_shape = np.asarray(shape, dtype=np.int64)
        if grid_shape.shape != (3,) or np.any(grid_shape <= 0):
            raise ValueError("shape must contain three positive integers")
        pad_factor = int(pad_factor)
        if pad_factor < 1:
            raise ValueError("pad_factor must be at least one")
        q = np.asarray(targets, dtype=np.float64)
        if q.ndim < 2 or q.shape[-1] != 3 or not np.all(np.isfinite(q)):
            raise ValueError("targets must have shape (..., 3) and be finite")

        self.torch = torch
        self.device = device
        self.complex_dtype = complex_dtype
        self.real_dtype = real_dtype
        self.shape = tuple(int(v) for v in grid_shape)
        self.padded_shape = tuple(int(pad_factor * v) for v in grid_shape)
        if any(size % 2 for size in self.padded_shape):
            raise ValueError(
                "periodic coefficient FFT currently requires even padded axes"
            )
        self.output_shape = tuple(int(v) for v in q.shape[:-1])
        self.phase_sign = int(phase_sign)

        flat = q.reshape(-1, 3)
        wrapped = np.remainder(flat + np.pi, 2.0 * np.pi) - np.pi
        lower_indices: list[np.ndarray] = []
        fractions: list[np.ndarray] = []
        for axis, size in enumerate(self.padded_shape):
            step = 2.0 * np.pi / float(size)
            coordinate = (wrapped[:, axis] + np.pi) / step
            lower = np.floor(coordinate).astype(np.int64)
            fractions.append(coordinate - lower)
            lower_indices.append(np.remainder(lower, size))

        ny, nz = self.padded_shape[1:]
        gather_indices = []
        gather_weights = []
        for dx in (0, 1):
            ix_shifted = np.remainder(
                lower_indices[0] + dx, self.padded_shape[0]
            )
            ix = np.remainder(
                ix_shifted + self.padded_shape[0] // 2,
                self.padded_shape[0],
            )
            wx = fractions[0] if dx else 1.0 - fractions[0]
            for dy in (0, 1):
                iy_shifted = np.remainder(
                    lower_indices[1] + dy, self.padded_shape[1]
                )
                iy = np.remainder(
                    iy_shifted + self.padded_shape[1] // 2,
                    self.padded_shape[1],
                )
                wy = fractions[1] if dy else 1.0 - fractions[1]
                for dz in (0, 1):
                    iz_shifted = np.remainder(
                        lower_indices[2] + dz, self.padded_shape[2]
                    )
                    iz = np.remainder(
                        iz_shifted + self.padded_shape[2] // 2,
                        self.padded_shape[2],
                    )
                    wz = fractions[2] if dz else 1.0 - fractions[2]
                    gather_indices.append((ix * ny + iy) * nz + iz)
                    gather_weights.append(wx * wy * wz)
        self.gather_indices = torch.as_tensor(
            np.ascontiguousarray(np.stack(gather_indices)),
            dtype=torch.long,
            device=device,
        )
        self.gather_weights = torch.as_tensor(
            np.ascontiguousarray(np.stack(gather_weights)),
            dtype=real_dtype,
            device=device,
        )

    @property
    def resident_bytes(self) -> int:
        return int(
            self.gather_indices.nelement() * self.gather_indices.element_size()
            + self.gather_weights.nelement() * self.gather_weights.element_size()
        )

    @property
    def transformed_grid_bytes(self) -> int:
        return int(
            4
            * np.prod(self.padded_shape, dtype=np.int64)
            * (8 if self.complex_dtype == self.torch.complex64 else 16)
        )

    def prepare_grid(self, values: Any) -> Any:
        array = self.torch.as_tensor(
            values, dtype=self.complex_dtype, device=self.device
        )
        scalar = array.ndim == 3
        if scalar:
            array = array.unsqueeze(0)
        if tuple(array.shape[-3:]) != self.shape:
            raise ValueError(f"values must end in shape {self.shape}")
        batch_shape = tuple(array.shape[:-3])
        array = array.reshape(-1, *self.shape)
        padded = self.torch.zeros(
            (array.shape[0], *self.padded_shape),
            dtype=self.complex_dtype,
            device=self.device,
        )
        padded[:, : self.shape[0], : self.shape[1], : self.shape[2]] = array
        if self.phase_sign == -1:
            transformed = self.torch.fft.fftn(padded, dim=(-3, -2, -1))
        else:
            transformed = self.torch.fft.ifftn(padded, dim=(-3, -2, -1))
            transformed = transformed * float(np.prod(self.padded_shape))
        return (
            transformed.reshape(*batch_shape, *self.padded_shape)
            if not scalar
            else transformed[0]
        )

    def interpolate_grid(self, transformed: Any) -> Any:
        grid = self.torch.as_tensor(
            transformed, dtype=self.complex_dtype, device=self.device
        )
        scalar = grid.ndim == 3
        if scalar:
            grid = grid.unsqueeze(0)
        if tuple(grid.shape[-3:]) != self.padded_shape:
            raise ValueError(
                f"transformed grid must end in shape {self.padded_shape}"
            )
        batch_shape = tuple(grid.shape[:-3])
        flat = grid.reshape(-1, int(np.prod(self.padded_shape)))
        gathered = flat[:, self.gather_indices]
        out = self.torch.sum(
            gathered * self.gather_weights.unsqueeze(0), dim=1
        )
        out = out.reshape(*batch_shape, *self.output_shape)
        return out[0] if scalar else out

    def forward(self, values: Any) -> Any:
        return self.interpolate_grid(self.prepare_grid(values))


class TorchPreparedVoxelFFTInterpolator:
    """GPU-resident counterpart of ``interpolated_voxel_fft``.

    The reciprocal target cloud is fixed at construction, so the eight corner
    indices and interpolation weights are prepared once.  Both a changing-
    volume path (FFT plus interpolation) and a fixed-volume path (interpolation
    only) are exposed for fair reuse studies.
    """

    def __init__(
        self,
        shape: tuple[int, int, int],
        bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
        q_xyz: Any,
        *,
        torch: Any,
        device: Any,
        dtype: str = "complex64",
        phase_sign: int = 1,
        pad_factor: int = 2,
        continuous_voxels: bool = True,
    ) -> None:
        if phase_sign not in (-1, 1):
            raise ValueError("phase_sign must be -1 or +1")
        if dtype == "complex64":
            complex_dtype, real_dtype = torch.complex64, torch.float32
            np_complex = np.complex64
        elif dtype == "complex128":
            complex_dtype, real_dtype = torch.complex128, torch.float64
            np_complex = np.complex128
        else:
            raise ValueError("dtype must be 'complex64' or 'complex128'")
        grid_shape = np.asarray(shape, dtype=np.int64)
        if grid_shape.shape != (3,) or np.any(grid_shape <= 0):
            raise ValueError("shape must contain three positive integers")
        lower = np.asarray([item[0] for item in bounds], dtype=np.float64)
        upper = np.asarray([item[1] for item in bounds], dtype=np.float64)
        if np.any(upper <= lower):
            raise ValueError("bounds must be increasing")
        q = np.asarray(q_xyz, dtype=np.float64)
        if q.ndim < 2 or q.shape[-1] != 3 or not np.all(np.isfinite(q)):
            raise ValueError("q_xyz must have shape (..., 3) and be finite")
        pad_factor = int(pad_factor)
        if pad_factor < 1:
            raise ValueError("pad_factor must be at least one")

        self.torch = torch
        self.device = device
        self.complex_dtype = complex_dtype
        self.real_dtype = real_dtype
        self.np_complex = np_complex
        self.shape = tuple(int(item) for item in grid_shape)
        self.padded_shape = tuple(int(pad_factor * item) for item in grid_shape)
        self.output_shape = tuple(int(item) for item in q.shape[:-1])
        self.phase_sign = int(phase_sign)
        self.continuous_voxels = bool(continuous_voxels)
        self.spacing = (upper - lower) / grid_shape
        self.first_center = lower + 0.5 * self.spacing
        self.voxel_volume = float(np.prod(self.spacing))

        axes = tuple(
            np.fft.fftshift(
                2.0 * np.pi * np.fft.fftfreq(self.padded_shape[axis], d=self.spacing[axis])
            )
            for axis in range(3)
        )
        flat_q = q.reshape(-1, 3)
        corner_indices: list[np.ndarray] = []
        fractions: list[np.ndarray] = []
        for axis in range(3):
            values = axes[axis]
            if np.any(flat_q[:, axis] < values[0]) or np.any(flat_q[:, axis] > values[-1]):
                raise ValueError("q target lies outside the zero-padded FFT interpolation grid")
            lower_index = np.searchsorted(values, flat_q[:, axis], side="right") - 1
            np.clip(lower_index, 0, values.size - 2, out=lower_index)
            fraction = (flat_q[:, axis] - values[lower_index]) / (
                values[lower_index + 1] - values[lower_index]
            )
            corner_indices.append(lower_index.astype(np.int64, copy=False))
            fractions.append(fraction)

        ny, nz = self.padded_shape[1:]
        gather_indices = []
        gather_weights = []
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    linear = (
                        (corner_indices[0] + dx) * ny + corner_indices[1] + dy
                    ) * nz + corner_indices[2] + dz
                    weight = (
                        (fractions[0] if dx else 1.0 - fractions[0])
                        * (fractions[1] if dy else 1.0 - fractions[1])
                        * (fractions[2] if dz else 1.0 - fractions[2])
                    )
                    gather_indices.append(linear)
                    gather_weights.append(weight)
        self.gather_indices = torch.as_tensor(
            np.ascontiguousarray(np.stack(gather_indices)),
            dtype=torch.long,
            device=device,
        )
        self.gather_weights = torch.as_tensor(
            np.ascontiguousarray(np.stack(gather_weights)),
            dtype=real_dtype,
            device=device,
        )
        if self.continuous_voxels:
            factor = np.prod(
                np.sinc(flat_q * self.spacing[None, :] / (2.0 * np.pi)), axis=1
            )
        else:
            factor = np.ones(flat_q.shape[0], dtype=np.float64)
        self.voxel_factor = torch.as_tensor(
            np.ascontiguousarray(factor), dtype=real_dtype, device=device
        )

        unshifted_axes = tuple(
            2.0 * np.pi * np.fft.fftfreq(self.padded_shape[axis], d=self.spacing[axis])
            for axis in range(3)
        )
        phase_factors = []
        for axis, values in enumerate(unshifted_axes):
            phase_factors.append(
                torch.as_tensor(
                    np.exp(
                        1j * self.phase_sign * values * self.first_center[axis]
                    ).astype(np_complex, copy=False),
                    dtype=complex_dtype,
                    device=device,
                )
            )
        self.phase_x = phase_factors[0].reshape(1, -1, 1, 1)
        self.phase_y = phase_factors[1].reshape(1, 1, -1, 1)
        self.phase_z = phase_factors[2].reshape(1, 1, 1, -1)

    @property
    def resident_bytes(self) -> int:
        tensors = (
            self.gather_indices,
            self.gather_weights,
            self.voxel_factor,
            self.phase_x,
            self.phase_y,
            self.phase_z,
        )
        return int(sum(item.nelement() * item.element_size() for item in tensors))

    def prepare_grid(self, values: Any) -> Any:
        """Transform one volume or leading batch of volumes to the shifted grid."""

        array = self.torch.as_tensor(values, dtype=self.complex_dtype, device=self.device)
        scalar = array.ndim == 3
        if scalar:
            array = array.unsqueeze(0)
        if tuple(array.shape[-3:]) != self.shape:
            raise ValueError(f"values must end in shape {self.shape}")
        batch_shape = tuple(array.shape[:-3])
        array = array.reshape(-1, *self.shape)
        padded = self.torch.zeros(
            (array.shape[0], *self.padded_shape),
            dtype=self.complex_dtype,
            device=self.device,
        )
        padded[
            :,
            : self.shape[0],
            : self.shape[1],
            : self.shape[2],
        ] = array * self.voxel_volume
        if self.phase_sign == 1:
            transformed = self.torch.fft.ifftn(padded, dim=(-3, -2, -1))
            transformed = transformed * float(np.prod(self.padded_shape))
        else:
            transformed = self.torch.fft.fftn(padded, dim=(-3, -2, -1))
        transformed = transformed * self.phase_x * self.phase_y * self.phase_z
        transformed = self.torch.fft.fftshift(transformed, dim=(-3, -2, -1))
        return transformed.reshape(*batch_shape, *self.padded_shape) if not scalar else transformed[0]

    def interpolate_grid(self, transformed: Any) -> Any:
        """Interpolate a prepared reciprocal grid at the fixed target cloud."""

        grid = self.torch.as_tensor(
            transformed, dtype=self.complex_dtype, device=self.device
        )
        scalar = grid.ndim == 3
        if scalar:
            grid = grid.unsqueeze(0)
        if tuple(grid.shape[-3:]) != self.padded_shape:
            raise ValueError(f"transformed grid must end in shape {self.padded_shape}")
        batch_shape = tuple(grid.shape[:-3])
        flat = grid.reshape(-1, int(np.prod(self.padded_shape)))
        gathered = flat[:, self.gather_indices]
        out = self.torch.sum(
            gathered * self.gather_weights.unsqueeze(0), dim=1
        ) * self.voxel_factor.unsqueeze(0)
        out = out.reshape(*batch_shape, *self.output_shape)
        return out[0] if scalar else out

    def forward(self, values: Any) -> Any:
        """Perform the changing-volume FFT plus fixed-target interpolation."""

        return self.interpolate_grid(self.prepare_grid(values))


class TorchPreparedPlanarFFTInterpolator:
    """Prepared 2-D zero-padded FFT with fixed-target bilinear interpolation.

    ``values`` are discrete Fourier coefficients located at the Cartesian
    sample centres implied by ``bounds``.  Unlike the voxel-volume class above,
    this class intentionally applies no cell-area or sinc factor: it evaluates
    the discrete sum

    ``sum[j, k] values[j, k] exp(sign * i * q . r[j, k])``.

    That coefficient convention is the one needed after an exact contraction
    of a flat-detector source axis (``q_z == 0``).
    """

    def __init__(
        self,
        shape: tuple[int, int],
        bounds: tuple[tuple[float, float], tuple[float, float]],
        q_xy: Any,
        *,
        torch: Any,
        device: Any,
        dtype: str = "complex64",
        phase_sign: int = 1,
        pad_factor: int = 2,
        padded_shape: tuple[int, int] | None = None,
    ) -> None:
        if phase_sign not in (-1, 1):
            raise ValueError("phase_sign must be -1 or +1")
        if dtype == "complex64":
            complex_dtype, real_dtype = torch.complex64, torch.float32
            np_complex = np.complex64
        elif dtype == "complex128":
            complex_dtype, real_dtype = torch.complex128, torch.float64
            np_complex = np.complex128
        else:
            raise ValueError("dtype must be 'complex64' or 'complex128'")
        grid_shape = np.asarray(shape, dtype=np.int64)
        if grid_shape.shape != (2,) or np.any(grid_shape <= 0):
            raise ValueError("shape must contain two positive integers")
        lower = np.asarray([item[0] for item in bounds], dtype=np.float64)
        upper = np.asarray([item[1] for item in bounds], dtype=np.float64)
        if np.any(upper <= lower):
            raise ValueError("bounds must be increasing")
        q = np.asarray(q_xy, dtype=np.float64)
        if q.ndim < 2 or q.shape[-1] != 2 or not np.all(np.isfinite(q)):
            raise ValueError("q_xy must have shape (..., 2) and be finite")
        pad_factor = int(pad_factor)
        if pad_factor < 1:
            raise ValueError("pad_factor must be at least one")
        if padded_shape is None:
            normalized_padded_shape = pad_factor * grid_shape
        else:
            normalized_padded_shape = np.asarray(
                padded_shape, dtype=np.int64
            )
            if normalized_padded_shape.shape != (2,) or np.any(
                normalized_padded_shape < grid_shape
            ):
                raise ValueError(
                    "padded_shape must contain two integers no smaller than shape"
                )

        self.torch = torch
        self.device = device
        self.complex_dtype = complex_dtype
        self.real_dtype = real_dtype
        self.np_complex = np_complex
        self.shape = tuple(int(item) for item in grid_shape)
        self.padded_shape = tuple(
            int(item) for item in normalized_padded_shape
        )
        self.output_shape = tuple(int(item) for item in q.shape[:-1])
        self.phase_sign = int(phase_sign)
        self.spacing = (upper - lower) / grid_shape
        self.first_center = lower + 0.5 * self.spacing

        axes = tuple(
            np.fft.fftshift(
                2.0
                * np.pi
                * np.fft.fftfreq(
                    self.padded_shape[axis], d=self.spacing[axis]
                )
            )
            for axis in range(2)
        )
        flat_q = q.reshape(-1, 2)
        corner_indices: list[np.ndarray] = []
        fractions: list[np.ndarray] = []
        for axis in range(2):
            values = axes[axis]
            if np.any(flat_q[:, axis] < values[0]) or np.any(
                flat_q[:, axis] > values[-1]
            ):
                raise ValueError(
                    "q target lies outside the zero-padded planar FFT "
                    "interpolation grid"
                )
            lower_index = np.searchsorted(
                values, flat_q[:, axis], side="right"
            ) - 1
            np.clip(lower_index, 0, values.size - 2, out=lower_index)
            fraction = (flat_q[:, axis] - values[lower_index]) / (
                values[lower_index + 1] - values[lower_index]
            )
            corner_indices.append(lower_index.astype(np.int64, copy=False))
            fractions.append(fraction)

        ny = self.padded_shape[1]
        gather_indices = []
        gather_weights = []
        for dx in (0, 1):
            for dy in (0, 1):
                gather_indices.append(
                    (corner_indices[0] + dx) * ny
                    + corner_indices[1]
                    + dy
                )
                gather_weights.append(
                    (fractions[0] if dx else 1.0 - fractions[0])
                    * (fractions[1] if dy else 1.0 - fractions[1])
                )
        self.gather_indices = torch.as_tensor(
            np.ascontiguousarray(np.stack(gather_indices)),
            dtype=torch.long,
            device=device,
        )
        self.gather_weights = torch.as_tensor(
            np.ascontiguousarray(np.stack(gather_weights)),
            dtype=real_dtype,
            device=device,
        )

        unshifted_axes = tuple(
            2.0
            * np.pi
            * np.fft.fftfreq(
                self.padded_shape[axis], d=self.spacing[axis]
            )
            for axis in range(2)
        )
        phase_factors = []
        for axis, values in enumerate(unshifted_axes):
            phase_factors.append(
                torch.as_tensor(
                    np.exp(
                        1j * self.phase_sign * values * self.first_center[axis]
                    ).astype(np_complex, copy=False),
                    dtype=complex_dtype,
                    device=device,
                )
            )
        self.phase_x = phase_factors[0].reshape(1, -1, 1)
        self.phase_y = phase_factors[1].reshape(1, 1, -1)

    @property
    def resident_bytes(self) -> int:
        tensors = (
            self.gather_indices,
            self.gather_weights,
            self.phase_x,
            self.phase_y,
        )
        return int(sum(item.nelement() * item.element_size() for item in tensors))

    def prepare_grid(self, values: Any) -> Any:
        """Transform one coefficient plane or a leading batch of planes."""

        array = self.torch.as_tensor(
            values, dtype=self.complex_dtype, device=self.device
        )
        scalar = array.ndim == 2
        if scalar:
            array = array.unsqueeze(0)
        if tuple(array.shape[-2:]) != self.shape:
            raise ValueError(f"values must end in shape {self.shape}")
        batch_shape = tuple(array.shape[:-2])
        array = array.reshape(-1, *self.shape)
        padded = self.torch.zeros(
            (array.shape[0], *self.padded_shape),
            dtype=self.complex_dtype,
            device=self.device,
        )
        padded[:, : self.shape[0], : self.shape[1]] = array
        if self.phase_sign == 1:
            transformed = self.torch.fft.ifftn(padded, dim=(-2, -1))
            transformed = transformed * float(np.prod(self.padded_shape))
        else:
            transformed = self.torch.fft.fftn(padded, dim=(-2, -1))
        transformed = transformed * self.phase_x * self.phase_y
        transformed = self.torch.fft.fftshift(transformed, dim=(-2, -1))
        return (
            transformed.reshape(*batch_shape, *self.padded_shape)
            if not scalar
            else transformed[0]
        )

    def interpolate_grid(self, transformed: Any) -> Any:
        """Interpolate a prepared reciprocal plane at the fixed targets."""

        grid = self.torch.as_tensor(
            transformed, dtype=self.complex_dtype, device=self.device
        )
        scalar = grid.ndim == 2
        if scalar:
            grid = grid.unsqueeze(0)
        if tuple(grid.shape[-2:]) != self.padded_shape:
            raise ValueError(
                f"transformed grid must end in shape {self.padded_shape}"
            )
        batch_shape = tuple(grid.shape[:-2])
        flat = grid.reshape(-1, int(np.prod(self.padded_shape)))
        gathered = flat[:, self.gather_indices]
        out = self.torch.sum(
            gathered * self.gather_weights.unsqueeze(0), dim=1
        )
        out = out.reshape(*batch_shape, *self.output_shape)
        return out[0] if scalar else out

    def forward(self, values: Any) -> Any:
        """Perform the coefficient-plane FFT and fixed-target interpolation."""

        return self.interpolate_grid(self.prepare_grid(values))
