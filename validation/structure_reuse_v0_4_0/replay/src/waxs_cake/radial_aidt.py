"""Selected-mode radial compression composed with the existing physical aIDT model.

The input is a complex-linear extension of two potential-density channels,
represented in an orthonormal angular subspace. The radial projector acts on
source strengths AFTER finite-volume area weighting. A physical transfer can
couple all retained modes, depths and channels; the q-local normal keeps those
couplings explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .radial_cpswf import PreparedRadialProjection


@dataclass(frozen=True)
class ModalRadialMap:
    modes: tuple[int, ...]
    area_scale: np.ndarray
    blocks: dict[int, np.ndarray | PreparedRadialProjection]
    n_q: int
    n_z: int

    @classmethod
    def build(cls, modes, area_scale, blocks, n_z):
        modes = tuple(modes)
        area = np.array(area_scale, dtype=float, copy=True)
        if (not modes or len(set(modes)) != len(modes)
                or any(int(m) != m for m in modes)
                or area.ndim != 1 or np.any(area <= 0)
                or not np.all(np.isfinite(area)) or n_z < 1):
            raise ValueError("invalid modes, positive area scale or depth count")
        copied = {}
        shapes = set()
        for m in set(abs(m) for m in modes):
            value = blocks[m]
            if isinstance(value, PreparedRadialProjection):
                shape = (value.reduced.shape[0], value.basis.shape[0])
            else:
                value = np.array(value, dtype=complex, copy=True)
                if value.ndim != 2 or not np.all(np.isfinite(value)) or np.any(value.imag):
                    raise ValueError("invalid radial matrix")
                value.setflags(write=False)
                shape = value.shape
            copied[m] = value
            shapes.add(shape)
        if len(shapes) != 1 or next(iter(shapes))[1] != area.size:
            raise ValueError("inconsistent radial dimensions")
        area.setflags(write=False)
        return cls(modes, area, copied, next(iter(shapes))[0], int(n_z))

    @property
    def input_shape(self):
        return (len(self.modes), self.area_scale.size, self.n_z, 2)

    @property
    def modal_shape(self):
        return (self.n_q, len(self.modes), self.n_z, 2)

    @property
    def cache_bytes(self):
        return self.area_scale.nbytes + sum(
            b.forward_bytes if isinstance(b, PreparedRadialProjection) else b.nbytes
            for b in self.blocks.values())

    def forward(self, values):
        x = np.asarray(values, dtype=complex)
        if x.shape != self.input_shape:
            raise ValueError("unexpected modal potential shape")
        result = np.empty(self.modal_shape, dtype=complex)
        for j, mode in enumerate(self.modes):
            # N_phi * cell_area = 2*pi*annular_area. For signed integer m,
            # (-i)^m J_m = (-i)^abs(m) J_abs(m).
            rhs = (self.area_scale[:, None, None]*x[j]).reshape(x.shape[1], -1)
            block = self.blocks[abs(mode)]
            y = block.forward(rhs) if isinstance(block, PreparedRadialProjection) else block@rhs
            result[:, j] = ((-1j)**abs(mode)*y).reshape(self.n_q, self.n_z, 2)
        return result

    def adjoint(self, values):
        y = np.asarray(values, dtype=complex)
        if y.shape != self.modal_shape:
            raise ValueError("unexpected radial-output shape")
        result = np.empty(self.input_shape, dtype=complex)
        for j, mode in enumerate(self.modes):
            rhs = ((1j)**abs(mode)*y[:, j]).reshape(self.n_q, -1)
            block = self.blocks[abs(mode)]
            x = block.adjoint(rhs) if isinstance(block, PreparedRadialProjection) else block.T@rhs
            result[j] = self.area_scale[:, None, None]*x.reshape(self.area_scale.size, self.n_z, 2)
        return result


@dataclass(frozen=True)
class PhysicalModalTransfer:
    modes: tuple[int, ...]
    angular: np.ndarray
    ptf: np.ndarray
    atf: np.ndarray
    q_chunk: int

    @classmethod
    def build(cls, modes, phi, ptf, atf, q_chunk=25):
        modes = tuple(modes)
        phi = np.asarray(phi, dtype=float)
        if len(set(modes)) != len(modes) or any(int(m) != m for m in modes):
            raise ValueError("unique integer modes required")
        # Reject aliased or nonuniform angular bases, including Nyquist duplicates.
        e = np.exp(1j*phi[:, None]*np.asarray(modes)[None, :])/np.sqrt(phi.size)
        if not np.allclose(e.conj().T@e, np.eye(len(modes)), atol=1e-12, rtol=1e-12):
            raise ValueError("angular basis is under-resolved or nonorthogonal")
        h, g = np.asarray(ptf, dtype=complex), np.asarray(atf, dtype=complex)
        if h.ndim != 4 or h.shape != g.shape or h.shape[2] != phi.size or q_chunk < 1:
            raise ValueError("PTF/ATF shape must be (source,q,phi,z)")
        if not np.all(np.isfinite(h)) or not np.all(np.isfinite(g)):
            raise ValueError("nonfinite transfer")
        for arr in (e, h, g):
            arr.setflags(write=False)
        return cls(modes, e, h, g, int(q_chunk))

    @property
    def modal_shape(self):
        return (self.ptf.shape[1], len(self.modes), self.ptf.shape[3], 2)

    @property
    def data_shape(self):
        return self.ptf.shape[:3]

    @property
    def cache_bytes(self):
        return self.angular.nbytes+self.ptf.nbytes+self.atf.nbytes

    def _slices(self):
        return (slice(j, min(j+self.q_chunk, self.ptf.shape[1]))
                for j in range(0, self.ptf.shape[1], self.q_chunk))

    def forward(self, modal):
        b = np.asarray(modal, dtype=complex)
        if b.shape != self.modal_shape:
            raise ValueError("unexpected modal spectrum shape")
        y = np.empty(self.data_shape, dtype=complex)
        nm, nz, np_ = len(self.modes), self.ptf.shape[3], self.ptf.shape[2]
        for sl in self._slices():
            chunk = b[sl]
            lateral = (chunk.transpose(0, 2, 3, 1).reshape(-1, nm)@self.angular.T)
            lateral = lateral.reshape(chunk.shape[0], nz, 2, np_).transpose(0, 3, 1, 2)
            y[:, sl] = np.einsum("sqpz,qpz->sqp", self.ptf[:, sl], lateral[..., 0])
            y[:, sl] += np.einsum("sqpz,qpz->sqp", self.atf[:, sl], lateral[..., 1])
        return y

    def adjoint(self, data):
        y = np.asarray(data, dtype=complex)
        if y.shape != self.data_shape:
            raise ValueError("unexpected data shape")
        b = np.empty(self.modal_shape, dtype=complex)
        nz, nm = self.ptf.shape[3], len(self.modes)
        for sl in self._slices():
            # Conjugate the smaller arrays, not the full persistent transfer cache.
            yc = y[:, sl].conj()
            l0 = np.einsum("sqpz,sqp->qpz", self.ptf[:, sl], yc).conj()
            l1 = np.einsum("sqpz,sqp->qpz", self.atf[:, sl], yc).conj()
            lateral = np.stack((l0, l1), axis=3)
            reduced = lateral.transpose(0, 2, 3, 1).reshape(-1, self.angular.shape[0])@self.angular.conj()
            b[sl] = reduced.reshape(lateral.shape[0], nz, 2, nm).transpose(0, 3, 1, 2)
        return b

    def compile_normal(self, weights=None):
        w = np.ones(self.data_shape) if weights is None else np.asarray(weights, dtype=float)
        if w.shape != self.data_shape or np.any(w < 0) or not np.all(np.isfinite(w)):
            raise ValueError("normal weights must be finite, nonnegative and data-shaped")
        nq, nm, nz, nc = self.modal_shape
        d = nm*nz*nc
        gram = np.empty((nq, d, d), dtype=complex)
        for q in range(nq):
            h = np.stack((self.ptf[:, q], self.atf[:, q]), axis=-1)
            # F[s,phi;mode,z,channel]. No diagonal-mode approximation.
            lift = (h[:, :, None, :, :]*self.angular[None, :, :, None, None]).reshape(-1, d)
            lift *= np.sqrt(w[:, q]).reshape(-1, 1)
            gram[q] = lift.conj().T@lift
        gram.setflags(write=False)
        return PreparedModalNormal(self.modal_shape, gram)


@dataclass(frozen=True)
class PreparedModalNormal:
    modal_shape: tuple[int, ...]
    gram: np.ndarray

    def apply(self, modal):
        b = np.asarray(modal, dtype=complex)
        if b.shape != self.modal_shape:
            raise ValueError("unexpected modal spectrum shape")
        return (self.gram@b.reshape(b.shape[0], -1, 1)).reshape(self.modal_shape)

    @property
    def cache_bytes(self):
        return self.gram.nbytes


@dataclass(frozen=True)
class PreparedRadialAidt:
    radial: ModalRadialMap
    transfer: PhysicalModalTransfer

    def __post_init__(self):
        if self.radial.modes != self.transfer.modes or self.radial.modal_shape != self.transfer.modal_shape:
            raise ValueError("radial and physical transfer geometries differ")

    def forward(self, potential):
        return self.transfer.forward(self.radial.forward(potential))

    def adjoint(self, data):
        return self.radial.adjoint(self.transfer.adjoint(data))

    def normal(self, potential, weights=None):
        data = self.forward(potential)
        if weights is not None:
            data *= weights
        return self.adjoint(data)

    def compiled_normal(self, potential, normal):
        return self.radial.adjoint(normal.apply(self.radial.forward(potential)))
