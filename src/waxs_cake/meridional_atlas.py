"""Tensor-product meridional atlas for arbitrary three-dimensional targets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse.linalg import spsolve

from .axisymmetric_manifold import AxisymmetricManifold
from .axisymmetric_operator import PreparedAxisymmetricOperator

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from .histogram import BinnedStructure
    from .solvers import FormFactors


InterpolationMethod = Literal["linear", "cubic"]


def _strict_grid(
    values: "ArrayLike",
    *,
    field_name: str,
    nonnegative: bool = False,
) -> "NDArray[np.float64]":
    grid = np.array(values, dtype=np.float64, copy=True)
    if grid.ndim != 1 or grid.size < 2 or not np.all(np.isfinite(grid)):
        raise ValueError(f"{field_name} must be a finite vector with at least two entries")
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError(f"{field_name} must be strictly increasing")
    if nonnegative and np.any(grid < 0.0):
        raise ValueError(f"{field_name} must be non-negative")
    grid.setflags(write=False)
    return grid


class MeridionalFourierAtlas:
    """Spline-interpolated ``(Q_perp, Q_z, harmonic)`` Fourier atlas.

    Construction evaluates the existing ring operator on a tensor grid in the
    meridional half-plane.  Evaluation interpolates all azimuthal Fourier
    coefficients and synthesizes them at arbitrary cylindrical or Cartesian
    target points.  Thus disks, volumes, and non-axisymmetric surfaces share
    one prepared object-specific atlas.

    The current cubic path is a forward PoC.  A production inverse/optimization
    path needs an interpolation implementation that exposes its exact transpose
    scatter rather than relying on SciPy's opaque spline object.
    """

    def __init__(
        self,
        template: "BinnedStructure",
        q_perp_grid: "ArrayLike",
        q_z_grid: "ArrayLike",
        *,
        object_values: "ArrayLike | None" = None,
        form_factors: "FormFactors" = None,
        interpolation: InterpolationMethod = "cubic",
        complex_dtype: np.dtype | str = np.complex128,
    ) -> None:
        self.q_perp_grid = _strict_grid(
            q_perp_grid,
            field_name="q_perp_grid",
            nonnegative=True,
        )
        self.q_z_grid = _strict_grid(q_z_grid, field_name="q_z_grid")
        if interpolation not in {"linear", "cubic"}:
            raise ValueError("interpolation must be 'linear' or 'cubic'")
        if interpolation == "cubic" and (
            self.q_perp_grid.size < 4 or self.q_z_grid.size < 4
        ):
            raise ValueError("cubic interpolation requires at least four points per axis")
        self.interpolation = interpolation

        q_perp_mesh, q_z_mesh = np.meshgrid(
            self.q_perp_grid,
            self.q_z_grid,
            indexing="ij",
        )
        n_nodes = q_perp_mesh.size
        manifold = AxisymmetricManifold(
            np.arange(n_nodes, dtype=np.float64),
            q_perp_mesh.ravel(),
            q_z_mesh.ravel(),
            name="meridional-tensor-atlas",
            interpretation="sampling",
        )
        self.operator = PreparedAxisymmetricOperator(
            template,
            manifold,
            form_factors=form_factors,
            complex_dtype=complex_dtype,
        )
        values = template.hist if object_values is None else object_values
        coefficients = self.operator.forward_fourier(values).reshape(
            self.q_perp_grid.size,
            self.q_z_grid.size,
            self.operator.phi.size,
        )
        coefficients = np.asarray(coefficients, dtype=self.operator.complex_dtype)
        coefficients.setflags(write=False)
        self.coefficients = coefficients
        self.angular_modes = self.operator.angular_modes
        self.phi_origin = float(self.operator.phi[0])
        spline_options = {"solver": spsolve} if self.interpolation == "cubic" else {}
        self._interpolator = RegularGridInterpolator(
            (self.q_perp_grid, self.q_z_grid),
            self.coefficients,
            method=self.interpolation,
            bounds_error=True,
            **spline_options,
        )

    @property
    def complex_dtype(self) -> np.dtype:
        return self.operator.complex_dtype

    @property
    def prepared_nbytes(self) -> int:
        return int(self.coefficients.nbytes)

    def _cylindrical_arrays(
        self,
        q_perp: "ArrayLike",
        phi: "ArrayLike",
        q_z: "ArrayLike",
    ) -> tuple[
        "NDArray[np.float64]",
        "NDArray[np.float64]",
        "NDArray[np.float64]",
    ]:
        rho, angle, axial = np.broadcast_arrays(
            np.asarray(q_perp, dtype=np.float64),
            np.asarray(phi, dtype=np.float64),
            np.asarray(q_z, dtype=np.float64),
        )
        if not np.all(np.isfinite(rho)) or not np.all(np.isfinite(angle)) or not np.all(
            np.isfinite(axial)
        ):
            raise ValueError("target coordinates must be finite")
        if np.any(rho < 0.0):
            raise ValueError("q_perp must be non-negative")
        return rho, angle, axial

    def evaluate_coefficients(
        self,
        q_perp: "ArrayLike",
        q_z: "ArrayLike",
    ) -> "NDArray[np.complexfloating]":
        """Interpolate Fourier coefficients before arbitrary-angle synthesis."""

        rho, axial = np.broadcast_arrays(
            np.asarray(q_perp, dtype=np.float64),
            np.asarray(q_z, dtype=np.float64),
        )
        if not np.all(np.isfinite(rho)) or not np.all(np.isfinite(axial)):
            raise ValueError("target coordinates must be finite")
        if np.any(rho < 0.0):
            raise ValueError("q_perp must be non-negative")
        points = np.column_stack((rho.ravel(), axial.ravel()))
        values = np.asarray(self._interpolator(points), dtype=self.complex_dtype)
        return values.reshape(rho.shape + (self.operator.phi.size,))

    def evaluate_cylindrical(
        self,
        q_perp: "ArrayLike",
        phi: "ArrayLike",
        q_z: "ArrayLike",
        *,
        max_abs_mode: int | None = None,
    ) -> "NDArray[np.complexfloating]":
        """Evaluate arbitrary cylindrical target points inside the atlas box."""

        rho, angle, axial = self._cylindrical_arrays(q_perp, phi, q_z)
        coefficients = self.evaluate_coefficients(rho, axial)
        modes = self.angular_modes
        if max_abs_mode is not None:
            cutoff = int(max_abs_mode)
            if cutoff < 0 or cutoff > self.operator.phi.size // 2:
                raise ValueError("max_abs_mode must be in [0, n_phi // 2]")
            mask = np.abs(modes) <= cutoff
            coefficients = coefficients[..., mask]
            modes = modes[mask]
        phase = np.exp(
            1j * (angle[..., None] - self.phi_origin) * modes
        ).astype(self.complex_dtype, copy=False)
        values = np.sum(coefficients * phase, axis=-1) / self.operator.phi.size
        return values.astype(self.complex_dtype, copy=False)

    def evaluate_cartesian(
        self,
        q_points: "ArrayLike",
        *,
        max_abs_mode: int | None = None,
    ) -> "NDArray[np.complexfloating]":
        """Evaluate Cartesian target points with final dimension ``(qx, qy, qz)``."""

        points = np.asarray(q_points, dtype=np.float64)
        if points.ndim < 1 or points.shape[-1] != 3:
            raise ValueError("q_points must have final dimension 3")
        if not np.all(np.isfinite(points)):
            raise ValueError("q_points must contain only finite values")
        q_perp = np.hypot(points[..., 0], points[..., 1])
        phi = np.arctan2(points[..., 1], points[..., 0])
        return self.evaluate_cylindrical(
            q_perp,
            phi,
            points[..., 2],
            max_abs_mode=max_abs_mode,
        )
