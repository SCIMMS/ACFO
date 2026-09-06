"""Independent sampled-quadrature preparation for the fixed-channel composite.

No ACFO plan, harmonic transform, Bessel recurrence or ACFO-produced matrix is
used. The physical Green mixing and final finite-action evaluator are shared.
All channels and the two analytic translation derivatives are batched through
one quadrature matrix. Retaining this matrix is an explicitly priced refresh
policy, distinct from compact deployment and from measured process peak RSS.
"""
from dataclasses import dataclass, replace

import numpy as np

from .composite_wave_operator import (
    CompositeMaterializationContract, MaterializedSO2ChannelOperator,
    layered_reflected_jones_green_mixing,
)


def _vector(name, value):
    a = np.array(value, dtype=np.float64, copy=True)
    if a.ndim != 1 or not a.size or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a nonempty finite vector")
    return a


def _readonly(a):
    a = np.ascontiguousarray(a)
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class DirectPreparedComposite:
    action: MaterializedSO2ChannelOperator
    quadrature: np.ndarray
    effective_rhs: np.ndarray
    lateral_phase_basis: np.ndarray

    @classmethod
    def build(cls, *, theta, theta_weights, phi, rho_axis, psi_axis, z_axis,
              upper_wavenumber, lower_wavenumber, damping, source_height,
              source_basis, lateral_displacement=(0.0, 0.0), channel_labels=None,
              max_kernel_bytes=128*1024**2):
        theta = _vector("theta", theta)
        weights = _vector("theta_weights", theta_weights)
        phi = _vector("phi", phi)
        rho, psi, z = [_vector(n, v) for n, v in
                       (("rho", rho_axis), ("psi", psi_axis), ("z", z_axis))]
        if weights.shape != theta.shape or np.any(weights <= 0) or np.any(rho < 0):
            raise ValueError("incompatible weights or negative radii")
        if phi.size < 3 or not np.allclose(np.diff(phi), 2*np.pi/phi.size, rtol=1e-12, atol=1e-14):
            raise ValueError("phi must sample a complete uniform orbit")
        basis = np.asarray(source_basis, dtype=np.complex128)
        if basis.ndim != 4 or basis.shape[1:] != (2, theta.size, phi.size) or not basis.shape[0] or not np.all(np.isfinite(basis)):
            raise ValueError("invalid finite incident-channel basis")
        labels = tuple(channel_labels or (f"channel_{j}" for j in range(basis.shape[0])))
        if len(labels) != basis.shape[0]:
            raise ValueError("channel labels must match basis")
        size = rho.size*psi.size*z.size*theta.size*phi.size*16
        if isinstance(max_kernel_bytes, bool) or int(max_kernel_bytes) != max_kernel_bytes or max_kernel_bytes < 1:
            raise ValueError("max_kernel_bytes must be a positive integer")
        if size > max_kernel_bytes:
            raise MemoryError(f"quadrature matrix needs {size} bytes; budget {max_kernel_bytes}")
        mixing = layered_reflected_jones_green_mixing(
            theta, phi, upper_wavenumber=upper_wavenumber,
            lower_wavenumber=lower_wavenumber, damping=damping,
            source_height=source_height, lateral_displacement=lateral_displacement)
        rr, pp, zz = np.meshgrid(rho, psi, z, indexing="ij")
        radial = float(upper_wavenumber)*np.sin(theta)[:, None]
        qx = radial*np.cos(phi)[None, :]
        qy = radial*np.sin(phi)[None, :]
        qz = np.broadcast_to(float(upper_wavenumber)*np.cos(theta)[:, None], qx.shape)
        xyz = np.column_stack(((rr*np.cos(pp)).ravel(), (rr*np.sin(pp)).ravel(), zz.ravel()))
        q = np.stack((qx, qy, qz)).reshape(3, -1)
        kernel = np.exp(1j*(xyz@q))
        kernel *= np.repeat(weights, phi.size)[None, :]*(2*np.pi/phi.size)
        # (source node, field component, incident channel): all RHS share K.
        rhs = np.einsum("cjtp,bjtp->tpcb", mixing, basis, optimize=True).reshape(theta.size*phi.size, -1)
        phase = np.stack((qx.ravel(), qy.ravel()))
        contract = CompositeMaterializationContract(
            channel_count=basis.shape[0], channel_labels=labels,
            source_height=float(source_height), lateral_displacement=tuple(map(float, lateral_displacement)),
            upper_wavenumber=float(upper_wavenumber), lower_wavenumber=float(lower_wavenumber),
            damping=float(damping), channel_data_cutoff=-1, active_harmonic_cutoff=-1,
            mode_padding=0, miller_margin=0)
        # -1 identifies a non-harmonic comparator, never an inferred cutoff.
        action = cls._materialize(kernel, rhs, phase, contract)
        return cls(action, _readonly(kernel), _readonly(rhs), _readonly(phase))

    @staticmethod
    def _materialize(kernel, rhs, phase, contract):
        packed = np.concatenate((rhs, 1j*phase[0, :, None]*rhs, 1j*phase[1, :, None]*rhs), axis=1)
        result = (kernel@packed).reshape(kernel.shape[0], 3, 3, contract.channel_count)
        b = _readonly(result[:, 0].transpose(1, 0, 2).copy())
        db = _readonly(result[:, 1:].transpose(1, 2, 0, 3).copy())
        return MaterializedSO2ChannelOperator(b, db, contract)

    def compact(self):
        return self.action

    @property
    def refresh_retained_bytes(self):
        return int(self.quadrature.nbytes+self.effective_rhs.nbytes+self.lateral_phase_basis.nbytes+self.action.cache_bytes)

    def materialize_lateral_update(self, displacement_xy):
        delta = np.asarray(displacement_xy, dtype=float)
        if delta.shape != (2,) or not np.all(np.isfinite(delta)):
            raise ValueError("displacement must contain two finite values")
        rhs = self.effective_rhs*np.exp(1j*(delta@self.lateral_phase_basis))[:, None]
        contract = replace(self.action.contract, lateral_displacement=tuple(np.asarray(self.action.contract.lateral_displacement)+delta))
        return self._materialize(self.quadrature, rhs, self.lateral_phase_basis, contract)
