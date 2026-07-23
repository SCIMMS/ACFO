"""Double-manifold harmonic operator for small ODT validation problems."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.special import jv

from .axisymmetric_manifold import AxisymmetricManifold
from .axisymmetric_operator import normalized_adjoint_error

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def _finite_vector(values: "ArrayLike", *, name: str) -> "NDArray[np.float64]":
    array = np.array(values, dtype=np.float64, copy=True)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    array.setflags(write=False)
    return array


def double_manifold_nodes(
    outgoing: AxisymmetricManifold,
    incident: AxisymmetricManifold,
    outgoing_phi: "ArrayLike",
    incident_phi: "ArrayLike",
) -> "NDArray[np.float64]":
    """Return ``Gamma_out - Gamma_in`` nodes with shape ``(uo, ui, po, pi, 3)``."""

    phi_out = _finite_vector(outgoing_phi, name="outgoing_phi")
    phi_in = _finite_vector(incident_phi, name="incident_phi")
    nodes_out = outgoing.target_nodes(phi_out)
    nodes_in = incident.target_nodes(phi_in)
    return (
        nodes_out[:, None, :, None, :]
        - nodes_in[None, :, None, :, :]
    )


def cylindrical_coordinates(
    r: "ArrayLike",
    z: "ArrayLike",
    beta: "ArrayLike",
) -> "NDArray[np.float64]":
    """Expand a cylindrical tensor grid in C-order."""

    r_array = _finite_vector(r, name="r")
    z_array = _finite_vector(z, name="z")
    beta_array = _finite_vector(beta, name="beta")
    rr, zz, bb = np.meshgrid(r_array, z_array, beta_array, indexing="ij")
    return np.column_stack(
        (
            (rr * np.cos(bb)).ravel(),
            (rr * np.sin(bb)).ravel(),
            zz.ravel(),
        )
    )


def direct_double_manifold_forward(
    object_values: "ArrayLike",
    r: "ArrayLike",
    z: "ArrayLike",
    beta: "ArrayLike",
    outgoing: AxisymmetricManifold,
    incident: AxisymmetricManifold,
    outgoing_phi: "ArrayLike",
    incident_phi: "ArrayLike",
    *,
    chunk_nodes: int = 256,
) -> "NDArray[np.complex128]":
    """Evaluate the independent Cartesian type-3 NUDFT reference."""

    r_array = _finite_vector(r, name="r")
    z_array = _finite_vector(z, name="z")
    beta_array = _finite_vector(beta, name="beta")
    expected_shape = (r_array.size, z_array.size, beta_array.size)
    values = np.asarray(object_values, dtype=np.complex128)
    if values.shape != expected_shape or not np.all(np.isfinite(values)):
        raise ValueError(f"object_values must be finite with shape {expected_shape}")
    chunk_nodes = int(chunk_nodes)
    if chunk_nodes <= 0:
        raise ValueError("chunk_nodes must be positive")
    nodes = double_manifold_nodes(
        outgoing,
        incident,
        outgoing_phi,
        incident_phi,
    )
    flat_nodes = nodes.reshape(-1, 3)
    coords = cylindrical_coordinates(r_array, z_array, beta_array)
    output = np.empty(flat_nodes.shape[0], dtype=np.complex128)
    for start in range(0, flat_nodes.shape[0], chunk_nodes):
        stop = min(start + chunk_nodes, flat_nodes.shape[0])
        output[start:stop] = np.exp(1j * (flat_nodes[start:stop] @ coords.T)) @ values.ravel()
    return output.reshape(nodes.shape[:-1])


def direct_double_manifold_adjoint(
    data_values: "ArrayLike",
    r: "ArrayLike",
    z: "ArrayLike",
    beta: "ArrayLike",
    outgoing: AxisymmetricManifold,
    incident: AxisymmetricManifold,
    outgoing_phi: "ArrayLike",
    incident_phi: "ArrayLike",
    *,
    chunk_nodes: int = 256,
) -> "NDArray[np.complex128]":
    """Apply the independent conjugate Cartesian exponent sum."""

    r_array = _finite_vector(r, name="r")
    z_array = _finite_vector(z, name="z")
    beta_array = _finite_vector(beta, name="beta")
    nodes = double_manifold_nodes(
        outgoing,
        incident,
        outgoing_phi,
        incident_phi,
    )
    data = np.asarray(data_values, dtype=np.complex128)
    if data.shape != nodes.shape[:-1] or not np.all(np.isfinite(data)):
        raise ValueError(f"data_values must be finite with shape {nodes.shape[:-1]}")
    chunk_nodes = int(chunk_nodes)
    if chunk_nodes <= 0:
        raise ValueError("chunk_nodes must be positive")
    flat_nodes = nodes.reshape(-1, 3)
    flat_data = data.ravel()
    coords = cylindrical_coordinates(r_array, z_array, beta_array)
    output = np.zeros(coords.shape[0], dtype=np.complex128)
    for start in range(0, flat_nodes.shape[0], chunk_nodes):
        stop = min(start + chunk_nodes, flat_nodes.shape[0])
        output += np.exp(-1j * (flat_nodes[start:stop] @ coords.T)).T @ flat_data[
            start:stop
        ]
    return output.reshape((r_array.size, z_array.size, beta_array.size))


class PreparedDoubleManifoldOperator:
    """Prepared truncated double-harmonic forward-adjoint pair.

    The supplied outgoing and incident manifolds represent absolute wavevector
    branches. Sampling nodes are their difference, not two already-shifted
    scattering-vector curves.
    """

    def __init__(
        self,
        r: "ArrayLike",
        z: "ArrayLike",
        beta: "ArrayLike",
        outgoing: AxisymmetricManifold,
        incident: AxisymmetricManifold,
        outgoing_phi: "ArrayLike",
        incident_phi: "ArrayLike",
        *,
        harmonic_cutoff: int,
    ) -> None:
        self.r = _finite_vector(r, name="r")
        self.z = _finite_vector(z, name="z")
        self.beta = _finite_vector(beta, name="beta")
        self.outgoing = outgoing
        self.incident = incident
        self.outgoing_phi = _finite_vector(outgoing_phi, name="outgoing_phi")
        self.incident_phi = _finite_vector(incident_phi, name="incident_phi")
        harmonic_cutoff = int(harmonic_cutoff)
        if harmonic_cutoff < 0:
            raise ValueError("harmonic_cutoff must be non-negative")
        self.harmonic_cutoff = harmonic_cutoff
        self.modes = np.arange(-harmonic_cutoff, harmonic_cutoff + 1, dtype=np.int64)
        self.sum_modes = np.arange(-2 * harmonic_cutoff, 2 * harmonic_cutoff + 1, dtype=np.int64)
        self.object_shape = (self.r.size, self.z.size, self.beta.size)
        self.data_shape = (
            outgoing.n_u,
            incident.n_u,
            self.outgoing_phi.size,
            self.incident_phi.size,
        )

        self.beta_transform = np.exp(-1j * self.sum_modes[:, None] * self.beta[None, :])
        self.outgoing_angular = np.exp(
            1j * self.modes[:, None] * self.outgoing_phi[None, :]
        )
        self.incident_angular = np.exp(
            1j * self.modes[:, None] * self.incident_phi[None, :]
        )
        self.outgoing_bessel = jv(
            self.modes[None, :, None],
            outgoing.q_perp[:, None, None] * self.r[None, None, :],
        )
        self.incident_bessel = jv(
            self.modes[None, :, None],
            incident.q_perp[:, None, None] * self.r[None, None, :],
        )
        self.axial = np.exp(
            1j
            * (outgoing.q_z[:, None, None] - incident.q_z[None, :, None])
            * self.z[None, None, :]
        )
        self.outgoing_mode_phase = np.power(1j, self.modes)
        self.incident_mode_phase = np.power(-1j, self.modes)
        self._sum_mode_offset = 2 * harmonic_cutoff
        self.sum_mode_indices = (
            self.modes[:, None] + self.modes[None, :] + self._sum_mode_offset
        ).astype(np.int64)
        radial_pairs = np.einsum(
            "omr,inr->oimnr",
            self.outgoing_bessel,
            self.incident_bessel,
            optimize=True,
        )
        mode_pair_phase = (
            self.outgoing_mode_phase[:, None] * self.incident_mode_phase[None, :]
        )
        self.pair_kernel = (
            radial_pairs[..., None]
            * self.axial[:, :, None, None, None, :]
            * mode_pair_phase[None, None, :, :, None, None]
        )

    @property
    def prepared_bytes(self) -> int:
        arrays = (
            self.r,
            self.z,
            self.beta,
            self.outgoing_phi,
            self.incident_phi,
            self.modes,
            self.sum_modes,
            self.beta_transform,
            self.outgoing_angular,
            self.incident_angular,
            self.outgoing_bessel,
            self.incident_bessel,
            self.axial,
            self.outgoing_mode_phase,
            self.incident_mode_phase,
            self.sum_mode_indices,
            self.pair_kernel,
        )
        return int(sum(array.nbytes for array in arrays))

    def _object_array(self, values: "ArrayLike") -> "NDArray[np.complex128]":
        array = np.asarray(values, dtype=np.complex128)
        if array.shape != self.object_shape or not np.all(np.isfinite(array)):
            raise ValueError(f"object values must be finite with shape {self.object_shape}")
        return array

    def _data_array(self, values: "ArrayLike") -> "NDArray[np.complex128]":
        array = np.asarray(values, dtype=np.complex128)
        if array.shape != self.data_shape or not np.all(np.isfinite(array)):
            raise ValueError(f"data values must be finite with shape {self.data_shape}")
        return array

    def forward(self, object_values: "ArrayLike") -> "NDArray[np.complex128]":
        """Apply the truncated double-harmonic factorization."""

        values = self._object_array(object_values)
        object_modes = np.einsum(
            "rzb,lb->rzl",
            values,
            self.beta_transform,
            optimize=True,
        )
        selected_object_modes = object_modes[:, :, self.sum_mode_indices]
        coefficients = np.einsum(
            "oimnrz,rzmn->oimn",
            self.pair_kernel,
            selected_object_modes,
            optimize=True,
        )
        return np.einsum(
            "oimn,mp,nq->oipq",
            coefficients,
            self.outgoing_angular,
            self.incident_angular,
            optimize=True,
        )

    def adjoint_euclidean(
        self,
        data_values: "ArrayLike",
    ) -> "NDArray[np.complex128]":
        """Apply the exact Euclidean adjoint of the truncated forward."""

        data = self._data_array(data_values)
        coefficient_adjoint = np.einsum(
            "oipq,mp,nq->oimn",
            data,
            np.conj(self.outgoing_angular),
            np.conj(self.incident_angular),
            optimize=True,
        )
        object_mode_adjoint = np.zeros(
            (self.r.size, self.z.size, self.sum_modes.size),
            dtype=np.complex128,
        )
        pair_adjoint = np.einsum(
            "oimnrz,oimn->rzmn",
            np.conj(self.pair_kernel),
            coefficient_adjoint,
            optimize=True,
        )
        for m_index in range(self.modes.size):
            for n_index in range(self.modes.size):
                object_mode_adjoint[
                    :,
                    :,
                    self.sum_mode_indices[m_index, n_index],
                ] += pair_adjoint[:, :, m_index, n_index]
        return np.einsum(
            "rzl,lb->rzb",
            object_mode_adjoint,
            np.conj(self.beta_transform),
            optimize=True,
        )

    def adjoint_test(
        self,
        object_values: "ArrayLike",
        data_values: "ArrayLike",
    ) -> float:
        """Return the normalized complex dot-product mismatch."""

        values = self._object_array(object_values)
        data = self._data_array(data_values)
        left = complex(np.vdot(self.forward(values), data))
        right = complex(np.vdot(values, self.adjoint_euclidean(data)))
        return normalized_adjoint_error(left, right)
