"""Prepared ACFO forward-adjoint pair for the linear aIDT transfer model.

The public aIDT reconstruction writes each illumination measurement as a
linear combination of the lateral Fourier transforms of the real and
imaginary scattering-potential channels at every axial plane.  This module
keeps those published PTF/ATF transfer coefficients and evaluates the lateral
Fourier transforms on a polar frequency grid with the prepared circular
operator.

The object is represented on cylindrical finite-volume cells.  Cell areas are
included explicitly in the forward map, and the adjoint returned here is the
Euclidean adjoint with respect to the unweighted cell values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .axisymmetric_manifold import AxisymmetricManifold
from .axisymmetric_operator import PreparedAxisymmetricOperator, normalized_adjoint_error
from .histogram import BinnedStructure

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def _readonly_vector(value: "ArrayLike", *, name: str) -> "NDArray[np.float64]":
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _complex_dtype(dtype: np.dtype | str) -> np.dtype:
    result = np.dtype(dtype)
    if result not in {np.dtype(np.complex64), np.dtype(np.complex128)}:
        raise ValueError("complex_dtype must be complex64 or complex128")
    return result


def aidt_transfer_functions_on_points(
    *,
    frequency_x: "ArrayLike",
    frequency_y: "ArrayLike",
    source_na_xy: "ArrayLike",
    wavelength_um: float,
    medium_index: float,
    objective_na: float,
    depth_values_um: "ArrayLike",
    dz_um: float,
    evanescent_eps: float = 1e-12,
    complex_dtype: np.dtype | str = np.complex128,
) -> tuple[
    "NDArray[np.complexfloating]",
    "NDArray[np.complexfloating]",
    dict[str, "NDArray[np.float64]"],
]:
    """Return public-aIDT PTF/ATF coefficients at arbitrary lateral frequencies.

    This is the point-sampled counterpart of
    ``scripts/reconstruct_aidt_public_transfer_function.py``.  The equations,
    sign convention and scale are intentionally identical; only the Cartesian
    frequency mesh is replaced by caller-supplied points.

    The returned arrays have shape ``(illumination, *frequency_shape, depth)``.
    """

    dtype = _complex_dtype(complex_dtype)
    fx = np.asarray(frequency_x, dtype=np.float64)
    fy = np.asarray(frequency_y, dtype=np.float64)
    if fx.shape != fy.shape or fx.ndim == 0:
        raise ValueError("frequency_x and frequency_y must have matching array shapes")
    if not np.all(np.isfinite(fx)) or not np.all(np.isfinite(fy)):
        raise ValueError("frequency points must be finite")
    source = np.asarray(source_na_xy, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 2 or source.shape[0] == 0:
        raise ValueError("source_na_xy must have shape (n_illumination, 2)")
    if not np.all(np.isfinite(source)):
        raise ValueError("source_na_xy must be finite")
    depth = _readonly_vector(depth_values_um, name="depth_values_um")

    wavelength_um = float(wavelength_um)
    medium_index = float(medium_index)
    objective_na = float(objective_na)
    dz_um = float(dz_um)
    evanescent_eps = float(evanescent_eps)
    if wavelength_um <= 0.0 or medium_index <= 0.0 or objective_na <= 0.0:
        raise ValueError("wavelength_um, medium_index, and objective_na must be positive")
    if dz_um <= 0.0 or evanescent_eps < 0.0:
        raise ValueError("dz_um must be positive and evanescent_eps non-negative")

    k0 = 2.0 * np.pi / wavelength_um
    k_medium = k0 * medium_index
    max_frequency2 = (objective_na / wavelength_um) ** 2
    output_shape = (source.shape[0],) + fx.shape + (depth.size,)
    ptf = np.empty(output_shape, dtype=dtype)
    atf = np.empty(output_shape, dtype=dtype)
    valid_forward = np.empty(source.shape[0], dtype=np.float64)
    valid_conjugate = np.empty(source.shape[0], dtype=np.float64)
    source_axial_values = np.empty(source.shape[0], dtype=np.float64)
    z_shape = (1,) * fx.ndim + (depth.size,)
    z = depth.reshape(z_shape)

    def axial_green(
        delta_fx: "NDArray[np.float64]",
        delta_fy: "NDArray[np.float64]",
    ) -> tuple[
        "NDArray[np.float64]",
        "NDArray[np.float64]",
        "NDArray[np.bool_]",
    ]:
        radial2 = delta_fx * delta_fx + delta_fy * delta_fy
        inside = 1.0 - wavelength_um * wavelength_um * radial2
        valid = (inside > evanescent_eps) & (radial2 <= max_frequency2)
        axial = np.zeros_like(inside)
        axial[valid] = np.sqrt(inside[valid])
        green = np.zeros_like(inside)
        green[valid] = 1.0 / (k_medium * axial[valid])
        return axial, green, valid

    scale = 0.5 * dz_um * k0 * k0
    for illumination, source_na in enumerate(source):
        source_fx = float(source_na[0]) / wavelength_um
        source_fy = float(source_na[1]) / wavelength_um
        source_inside = 1.0 - wavelength_um * wavelength_um * (
            source_fx * source_fx + source_fy * source_fy
        )
        if source_inside <= evanescent_eps:
            raise ValueError(f"source NA is evanescent or grazing: {source_na}")
        source_axial = float(np.sqrt(source_inside))
        source_axial_values[illumination] = source_axial

        uv1, green1, pupil1 = axial_green(fx - source_fx, fy - source_fy)
        uv2, green2, pupil2 = axial_green(fx + source_fx, fy + source_fy)
        valid_forward[illumination] = float(np.mean(pupil1))
        valid_conjugate[illumination] = float(np.mean(pupil2))
        phase1 = k_medium * z * (uv1[..., None] - source_axial)
        phase2 = k_medium * z * (uv2[..., None] - source_axial)
        green1z = green1[..., None]
        green2z = green2[..., None]
        pupil1z = pupil1[..., None].astype(np.float64)
        pupil2z = pupil2[..., None].astype(np.float64)

        ptf_value = (
            pupil1z * np.sin(phase1) * green1z
            + pupil2z * np.sin(phase2) * green2z
        ) + 1j * (
            pupil1z * np.cos(phase1) * green1z
            - pupil2z * np.cos(phase2) * green2z
        )
        atf_value = -(
            pupil1z * np.cos(phase1) * green1z
            + pupil2z * np.cos(phase2) * green2z
        ) + 1j * (
            pupil1z * np.sin(phase1) * green1z
            - pupil2z * np.sin(phase2) * green2z
        )
        ptf[illumination] = (scale * ptf_value).astype(dtype, copy=False)
        atf[illumination] = (scale * atf_value).astype(dtype, copy=False)

    stats = {
        "valid_forward_fraction": valid_forward,
        "valid_conjugate_fraction": valid_conjugate,
        "source_axial": source_axial_values,
    }
    return ptf, atf, stats


def _single_slice_template(
    *,
    r_edges: "NDArray[np.float64]",
    beta_edges: "NDArray[np.float64]",
    complex_dtype: np.dtype,
) -> BinnedStructure:
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    beta_centers = 0.5 * (beta_edges[:-1] + beta_edges[1:])
    z_edges = np.array([-0.5, 0.5], dtype=np.float64)
    z_centers = np.array([0.0], dtype=np.float64)
    hist = np.zeros(
        (1, r_centers.size, 1, beta_centers.size),
        dtype=complex_dtype,
    )
    return BinnedStructure(
        hist=hist,
        r_centers=r_centers,
        z_centers=z_centers,
        beta_centers=beta_centers,
        elements=("aIDT",),
        r_edges=np.array(r_edges, copy=True),
        z_edges=z_edges,
        beta_edges=np.array(beta_edges, copy=True),
    )


@dataclass(frozen=True)
class PreparedAidtAcfoOperator:
    """Prepared physical PTF/ATF aIDT operator on a cylindrical object grid."""

    lateral_operator: PreparedAxisymmetricOperator
    r_edges_um: "NDArray[np.float64]"
    depth_values_um: "NDArray[np.float64]"
    radial_frequency_um_inv: "NDArray[np.float64]"
    source_na_xy: "NDArray[np.float64]"
    cell_area_um2: "NDArray[np.float64]"
    ptf: "NDArray[np.complexfloating]"
    atf: "NDArray[np.complexfloating]"
    transfer_stats: dict[str, "NDArray[np.float64]"]

    @classmethod
    def build(
        cls,
        *,
        r_edges_um: "ArrayLike",
        n_phi: int,
        depth_values_um: "ArrayLike",
        radial_frequency_um_inv: "ArrayLike",
        source_na_xy: "ArrayLike",
        wavelength_um: float,
        medium_index: float,
        objective_na: float,
        dz_um: float,
        evanescent_eps: float = 1e-12,
        complex_dtype: np.dtype | str = np.complex128,
    ) -> "PreparedAidtAcfoOperator":
        dtype = _complex_dtype(complex_dtype)
        r_edges = _readonly_vector(r_edges_um, name="r_edges_um")
        if r_edges.size < 2 or r_edges[0] < 0.0 or np.any(np.diff(r_edges) <= 0.0):
            raise ValueError("r_edges_um must be increasing, non-negative cell edges")
        n_phi = int(n_phi)
        if n_phi <= 0 or n_phi % 2:
            raise ValueError("n_phi must be a positive even integer")
        depth = _readonly_vector(depth_values_um, name="depth_values_um")
        radial_frequency = _readonly_vector(
            radial_frequency_um_inv,
            name="radial_frequency_um_inv",
        )
        if np.any(radial_frequency < 0.0) or np.any(np.diff(radial_frequency) <= 0.0):
            raise ValueError(
                "radial_frequency_um_inv must be strictly increasing and non-negative"
            )
        source = np.array(source_na_xy, dtype=np.float64, copy=True)
        if source.ndim != 2 or source.shape[1] != 2 or source.shape[0] == 0:
            raise ValueError("source_na_xy must have shape (n_illumination, 2)")

        beta_edges = np.linspace(0.0, 2.0 * np.pi, n_phi + 1)
        template = _single_slice_template(
            r_edges=r_edges,
            beta_edges=beta_edges,
            complex_dtype=dtype,
        )
        manifold = AxisymmetricManifold(
            u=radial_frequency,
            q_perp=2.0 * np.pi * radial_frequency,
            q_z=np.zeros_like(radial_frequency),
            name="aIDT-lateral-polar-frequency-grid",
            frequency_units="radian_per_micrometre",
        )
        lateral_operator = PreparedAxisymmetricOperator(
            template,
            manifold,
            complex_dtype=dtype,
        )
        phi = template.beta_centers
        rr, pp = np.meshgrid(radial_frequency, phi, indexing="ij")
        ptf, atf, transfer_stats = aidt_transfer_functions_on_points(
            frequency_x=rr * np.cos(pp),
            frequency_y=rr * np.sin(pp),
            source_na_xy=source,
            wavelength_um=wavelength_um,
            medium_index=medium_index,
            objective_na=objective_na,
            depth_values_um=depth,
            dz_um=dz_um,
            evanescent_eps=evanescent_eps,
            complex_dtype=dtype,
        )
        annular_area = 0.5 * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)
        cell_area = annular_area[:, None] * (2.0 * np.pi / n_phi)

        for array in (
            r_edges,
            depth,
            radial_frequency,
            source,
            cell_area,
            ptf,
            atf,
        ):
            array.setflags(write=False)
        return cls(
            lateral_operator=lateral_operator,
            r_edges_um=r_edges,
            depth_values_um=depth,
            radial_frequency_um_inv=radial_frequency,
            source_na_xy=source,
            cell_area_um2=cell_area,
            ptf=ptf,
            atf=atf,
            transfer_stats=transfer_stats,
        )

    @property
    def complex_dtype(self) -> np.dtype:
        return self.lateral_operator.complex_dtype

    @property
    def n_r(self) -> int:
        return int(self.r_edges_um.size - 1)

    @property
    def n_phi(self) -> int:
        return int(self.lateral_operator.phi.size)

    @property
    def n_z(self) -> int:
        return int(self.depth_values_um.size)

    @property
    def n_illumination(self) -> int:
        return int(self.source_na_xy.shape[0])

    @property
    def n_frequency(self) -> int:
        return int(self.radial_frequency_um_inv.size)

    @property
    def object_shape(self) -> tuple[int, int, int]:
        return (self.n_r, self.n_phi, self.n_z)

    @property
    def data_shape(self) -> tuple[int, int, int]:
        return (self.n_illumination, self.n_frequency, self.n_phi)

    @property
    def beta_centers(self) -> "NDArray[np.float64]":
        return self.lateral_operator.phi

    @property
    def r_centers_um(self) -> "NDArray[np.float64]":
        return self.lateral_operator.r_centers

    def _object_array(
        self,
        values: "ArrayLike",
        *,
        name: str,
    ) -> "NDArray[np.complexfloating]":
        array = np.asarray(values, dtype=self.complex_dtype)
        if array.shape != self.object_shape:
            raise ValueError(f"{name} must have shape {self.object_shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
        return array

    def _data_array(self, values: "ArrayLike") -> "NDArray[np.complexfloating]":
        array = np.asarray(values, dtype=self.complex_dtype)
        if array.shape != self.data_shape:
            raise ValueError(f"data values must have shape {self.data_shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError("data values must contain only finite values")
        return array

    def lateral_forward(
        self,
        object_values: "ArrayLike",
    ) -> "NDArray[np.complexfloating]":
        """Return negative-sign lateral Fourier samples for every axial plane."""

        values = self._object_array(object_values, name="object_values")
        output = np.empty(
            (self.n_frequency, self.n_phi, self.n_z),
            dtype=self.complex_dtype,
        )
        half_turn = self.n_phi // 2
        for z_index in range(self.n_z):
            coefficients = np.zeros(
                self.lateral_operator.object_shape,
                dtype=self.complex_dtype,
            )
            coefficients[0, :, 0, :] = values[:, :, z_index] * self.cell_area_um2
            positive = self.lateral_operator.forward(coefficients)
            output[:, :, z_index] = np.roll(positive, -half_turn, axis=1)
        return output

    def lateral_adjoint(
        self,
        spectrum_values: "ArrayLike",
    ) -> "NDArray[np.complexfloating]":
        """Apply the Euclidean adjoint of :meth:`lateral_forward`."""

        spectrum = np.asarray(spectrum_values, dtype=self.complex_dtype)
        expected = (self.n_frequency, self.n_phi, self.n_z)
        if spectrum.shape != expected:
            raise ValueError(f"spectrum_values must have shape {expected}")
        output = np.empty(self.object_shape, dtype=self.complex_dtype)
        half_turn = self.n_phi // 2
        for z_index in range(self.n_z):
            positive_data = np.roll(spectrum[:, :, z_index], half_turn, axis=1)
            coefficient_gradient = self.lateral_operator.adjoint_euclidean(positive_data)
            output[:, :, z_index] = (
                coefficient_gradient[0, :, 0, :] * self.cell_area_um2
            )
        return output

    def forward(
        self,
        potential_real: "ArrayLike",
        potential_imag: "ArrayLike",
    ) -> "NDArray[np.complexfloating]":
        """Apply the linear aIDT PTF/ATF forward map."""

        real_channel = self.lateral_forward(
            self._object_array(potential_real, name="potential_real")
        )
        imag_channel = self.lateral_forward(
            self._object_array(potential_imag, name="potential_imag")
        )
        data = np.einsum(
            "sqpz,qpz->sqp",
            self.ptf,
            real_channel,
            optimize=True,
        )
        data += np.einsum(
            "sqpz,qpz->sqp",
            self.atf,
            imag_channel,
            optimize=True,
        )
        return data.astype(self.complex_dtype, copy=False)

    def adjoint(
        self,
        data_values: "ArrayLike",
    ) -> tuple[
        "NDArray[np.complexfloating]",
        "NDArray[np.complexfloating]",
    ]:
        """Apply the Euclidean adjoint of :meth:`forward`."""

        data = self._data_array(data_values)
        spectrum_real = np.einsum(
            "sqpz,sqp->qpz",
            np.conj(self.ptf),
            data,
            optimize=True,
        )
        spectrum_imag = np.einsum(
            "sqpz,sqp->qpz",
            np.conj(self.atf),
            data,
            optimize=True,
        )
        return (
            self.lateral_adjoint(spectrum_real),
            self.lateral_adjoint(spectrum_imag),
        )

    def adjoint_test(
        self,
        potential_real: "ArrayLike",
        potential_imag: "ArrayLike",
        data_values: "ArrayLike",
    ) -> dict[str, complex | float]:
        """Return the two dot products and their normalized mismatch."""

        real_channel = self._object_array(potential_real, name="potential_real")
        imag_channel = self._object_array(potential_imag, name="potential_imag")
        data = self._data_array(data_values)
        left = complex(np.vdot(self.forward(real_channel, imag_channel), data))
        adjoint_real, adjoint_imag = self.adjoint(data)
        right = complex(
            np.vdot(real_channel, adjoint_real)
            + np.vdot(imag_channel, adjoint_imag)
        )
        return {
            "left": left,
            "right": right,
            "normalized_error": normalized_adjoint_error(left, right),
        }


def direct_lateral_fourier(
    operator: PreparedAidtAcfoOperator,
    object_values: "ArrayLike",
) -> "NDArray[np.complexfloating]":
    """Direct reference for the negative-sign polar lateral transform."""

    values = operator._object_array(object_values, name="object_values")
    r = operator.r_centers_um[:, None]
    beta = operator.beta_centers[None, :]
    x = r * np.cos(beta)
    y = r * np.sin(beta)
    rho = operator.radial_frequency_um_inv[:, None]
    phi = operator.beta_centers[None, :]
    fx = rho * np.cos(phi)
    fy = rho * np.sin(phi)
    phase = np.exp(
        -2j
        * np.pi
        * (
            fx[:, :, None, None] * x[None, None, :, :]
            + fy[:, :, None, None] * y[None, None, :, :]
        )
    ).astype(operator.complex_dtype, copy=False)
    weighted = values * operator.cell_area_um2[:, :, None]
    return np.einsum("qprb,rbz->qpz", phase, weighted, optimize=True).astype(
        operator.complex_dtype,
        copy=False,
    )


def direct_lateral_adjoint(
    operator: PreparedAidtAcfoOperator,
    spectrum_values: "ArrayLike",
) -> "NDArray[np.complexfloating]":
    """Direct Euclidean adjoint reference for :func:`direct_lateral_fourier`."""

    spectrum = np.asarray(spectrum_values, dtype=operator.complex_dtype)
    expected = (operator.n_frequency, operator.n_phi, operator.n_z)
    if spectrum.shape != expected:
        raise ValueError(f"spectrum_values must have shape {expected}")
    r = operator.r_centers_um[:, None]
    beta = operator.beta_centers[None, :]
    x = r * np.cos(beta)
    y = r * np.sin(beta)
    rho = operator.radial_frequency_um_inv[:, None]
    phi = operator.beta_centers[None, :]
    fx = rho * np.cos(phi)
    fy = rho * np.sin(phi)
    phase_conjugate = np.exp(
        2j
        * np.pi
        * (
            fx[:, :, None, None] * x[None, None, :, :]
            + fy[:, :, None, None] * y[None, None, :, :]
        )
    ).astype(operator.complex_dtype, copy=False)
    output = np.einsum(
        "qprb,qpz->rbz",
        phase_conjugate,
        spectrum,
        optimize=True,
    )
    return (output * operator.cell_area_um2[:, :, None]).astype(
        operator.complex_dtype,
        copy=False,
    )


def direct_aidt_forward(
    operator: PreparedAidtAcfoOperator,
    potential_real: "ArrayLike",
    potential_imag: "ArrayLike",
) -> "NDArray[np.complexfloating]":
    """Direct-DFT reference for the full PTF/ATF forward map."""

    real_channel = direct_lateral_fourier(operator, potential_real)
    imag_channel = direct_lateral_fourier(operator, potential_imag)
    output = np.einsum(
        "sqpz,qpz->sqp",
        operator.ptf,
        real_channel,
        optimize=True,
    )
    output += np.einsum(
        "sqpz,qpz->sqp",
        operator.atf,
        imag_channel,
        optimize=True,
    )
    return output.astype(operator.complex_dtype, copy=False)


def direct_aidt_adjoint(
    operator: PreparedAidtAcfoOperator,
    data_values: "ArrayLike",
) -> tuple[
    "NDArray[np.complexfloating]",
    "NDArray[np.complexfloating]",
]:
    """Direct-DFT reference for the full PTF/ATF adjoint map."""

    data = operator._data_array(data_values)
    spectrum_real = np.einsum(
        "sqpz,sqp->qpz",
        np.conj(operator.ptf),
        data,
        optimize=True,
    )
    spectrum_imag = np.einsum(
        "sqpz,sqp->qpz",
        np.conj(operator.atf),
        data,
        optimize=True,
    )
    return (
        direct_lateral_adjoint(operator, spectrum_real),
        direct_lateral_adjoint(operator, spectrum_imag),
    )
