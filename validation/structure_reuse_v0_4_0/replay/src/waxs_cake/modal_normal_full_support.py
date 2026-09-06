"""FFT-composed comparator and allocation-free support accounting for modal G."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import fft
from .modal_normal_structure import analyze_modal_normal, _weights, _readonly


def full_support_contract(q_max, radius, n_phi, data_bandlimit, padding):
    """Distinguish declared input band from the kernel-envelope stress domain."""
    if not np.isfinite(q_max) or not np.isfinite(radius) or q_max < 0 or radius <= 0:
        raise ValueError("finite nonnegative q and positive radius required")
    if any(int(v) != v or v < 0 for v in (data_bandlimit, padding)):
        raise ValueError("nonnegative integer bandlimit and padding required")
    kernel = int(np.ceil(q_max*radius))
    required = max(kernel, int(data_bandlimit))+int(padding)
    if int(n_phi) != n_phi or n_phi <= 2*required:
        raise ValueError("angular sampling fails the declared kernel envelope")
    return {"q_max_R":float(q_max*radius),"H_kernel":kernel,"H_data":int(data_bandlimit),
            "padding":int(padding),"H_req":required,"mode_count":2*required+1,
            "n_phi":int(n_phi),"nyquist_check_passed":True,
            "scope":"G on all kernel-envelope modes; distinct from the declared input band and continuous-angle convergence"}


@dataclass(frozen=True)
class PreparedFftComposedNormal:
    """E* H* W H E with FFT angular lift and no precomputed channel Gram."""
    modal_shape: tuple
    ptf: np.ndarray
    atf: np.ndarray
    weights: np.ndarray
    bins: np.ndarray
    phases: np.ndarray
    q_chunk: int

    @property
    def cache_bytes(self):
        return sum(x.nbytes for x in (self.ptf,self.atf,self.weights,self.bins,self.phases))

    @property
    def incremental_array_bytes(self):
        return sum(x.nbytes for x in (self.weights,self.bins,self.phases))

    def apply(self, modal):
        b = np.asarray(modal,dtype=complex)
        if b.shape != self.modal_shape:
            raise ValueError("unexpected modal shape")
        nq,nm,nz,nc = self.modal_shape
        nphi = self.ptf.shape[2]
        result = np.empty_like(b)
        for start in range(0,nq,self.q_chunk):
            sl = slice(start,min(start+self.q_chunk,nq))
            x = np.zeros((sl.stop-sl.start,nphi,nz,nc),dtype=complex)
            x[:,self.bins] = b[sl]*self.phases[None,:,None,None]
            lateral = fft.ifft(x,axis=1,norm="ortho")
            data = np.einsum("sqpz,qpz->sqp",self.ptf[:,sl],lateral[...,0])
            data += np.einsum("sqpz,qpz->sqp",self.atf[:,sl],lateral[...,1])
            data *= self.weights[:,sl]
            conjugate = data.conj()
            l0 = np.einsum("sqpz,sqp->qpz",self.ptf[:,sl],conjugate).conj()
            l1 = np.einsum("sqpz,sqp->qpz",self.atf[:,sl],conjugate).conj()
            y = fft.fft(np.stack((l0,l1),axis=-1),axis=1,norm="ortho")
            result[sl] = y[:,self.bins]*self.phases.conj()[None,:,None,None]
        return result


def prepare_fft_composed_normal(transfer, phi, weights=None):
    info = analyze_modal_normal(transfer,phi)
    modes = np.asarray(info["modes"],dtype=np.int64)
    return PreparedFftComposedNormal(tuple(transfer.modal_shape),transfer.ptf,transfer.atf,
        _readonly(_weights(transfer,weights).copy()),_readonly(modes % info["n_phi"]),
        _readonly(np.exp(1j*modes*info["phi_offset"])),transfer.q_chunk)
