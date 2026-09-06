"""Exact sampled angular-lag structure of PhysicalModalTransfer normals.

This is the q-local transfer normal G, not the complete R* G R operator.
No mode, transfer spectrum, lag, or physical coupling is truncated.  Uniform
angular samples (including their offset) and diagonal data weights are required.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import fft


def _readonly(a):
    a.setflags(write=False)
    return a


def analyze_modal_normal(transfer, phi):
    """Recognize the existing mode-independent transfer times angular lift.

    This is a model-specific structural rule, not symmetry discovery from a
    dense matrix.  A nonuniform or differently phased lift is rejected.
    """
    phi = np.asarray(phi, dtype=float)
    modes = np.asarray(transfer.modes, dtype=np.int64)
    nq, nm, nz, nc = transfer.modal_shape
    np_ = transfer.angular.shape[0]
    if phi.shape != (np_,) or not np.all(np.isfinite(phi)):
        raise ValueError("finite angular coordinates matching the lift required")
    expected_phi = phi[0]+2*np.pi*np.arange(np_)/np_
    if not np.allclose(phi, expected_phi, atol=2e-13, rtol=0):
        raise ValueError("uniform ordered full-circle angular samples required")
    expected = np.exp(1j*phi[:, None]*modes)/np.sqrt(np_)
    if not np.allclose(transfer.angular, expected, atol=2e-13, rtol=0):
        raise ValueError("angular lift and its declared offset disagree")
    if len(np.unique(modes % np_)) != nm:
        raise ValueError("aliased angular modes")
    # Difference cardinality without allocating an M by M retained index map.
    lags = np.asarray(sorted({int(n-m) for m in modes for n in modes}), dtype=np.int64)
    span = int(modes.max()-modes.min()+1)
    nfft = fft.next_fast_len(2*span-1)
    k = nz*nc
    return {"structure": "sampled mode-difference block kernel",
            "modes": modes.tolist(), "lags": lags.tolist(), "mode_span": span,
            "n_phi": np_, "phi_offset": float(phi[0]), "fft_length": nfft,
            "contiguous_modes": span == nm,
            "shape": list(transfer.modal_shape),
            "core_bytes": {"dense": 16*nq*(nm*k)**2,
                           "lag_direct": 16*nq*len(lags)*k*k,
                           "lag_fft": 16*nq*nfft*k*k,
                           "angular_fft": 16*nq*np_*k*k},
            "exactness": "same discrete quadrature; no spectral truncation",
            "scope": "G only; radial maps depend on m and remain outside"}


def _weights(transfer, weights):
    if weights is None:
        return np.ones(transfer.data_shape)
    if np.iscomplexobj(weights):
        raise ValueError("real nonnegative diagonal data weights required")
    w = np.asarray(weights, dtype=float)
    if w.shape != transfer.data_shape or not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("finite nonnegative data-shaped weights required")
    return w


def _angular_gram_chunk(transfer, w, sl):
    # C[q,p,a,b] = sum_s conj(h[s,q,p,a])*w[s,q,p]*h[s,q,p,b].
    # Source loop avoids a full source x q x phi x K x K temporary.
    nq = sl.stop-sl.start
    np_, nz = transfer.ptf.shape[2:]
    k = 2*nz
    c = np.zeros((nq, np_, k, k), dtype=complex)
    for s in range(transfer.ptf.shape[0]):
        h = np.stack((transfer.ptf[s, sl], transfer.atf[s, sl]), axis=-1).reshape(nq, np_, k)
        c += (h.conj()*w[s, sl, :, None])[..., :, None]*h[..., None, :]
    return c


@dataclass(frozen=True)
class StructuredModalNormal:
    modal_shape: tuple[int, ...]
    modes: np.ndarray
    core: np.ndarray
    representation: str
    n_phi: int
    phi_offset: float
    q_chunk: int
    lags: np.ndarray
    fft_length: int

    @property
    def cache_bytes(self):
        return self.core.nbytes+self.modes.nbytes+self.lags.nbytes

    def apply(self, modal):
        b = np.asarray(modal, dtype=complex)
        if b.shape != self.modal_shape:
            raise ValueError("unexpected modal shape")
        nq, nm, nz, nc = b.shape
        k = nz*nc
        b = b.reshape(nq, nm, k)
        y = np.empty_like(b)
        for start in range(0, nq, self.q_chunk):
            sl = slice(start, min(nq, start+self.q_chunk))
            v = b[sl]
            if self.representation == "lag_direct":
                for j, m in enumerate(self.modes):
                    indices = np.searchsorted(self.lags, self.modes-m)
                    y[sl, j] = np.einsum("qnab,qnb->qa", self.core[sl][:, indices], v, optimize=False)
            elif self.representation == "lag_fft":
                positions = self.modes-self.modes.min()
                x = np.zeros((v.shape[0], self.fft_length, k), dtype=complex)
                x[:, positions] = v
                x = fft.fft(x, axis=1)
                z = np.einsum("qfab,qfb->qfa", self.core[sl], x, optimize=False)
                y[sl] = fft.ifft(z, axis=1)[:, positions]
            else:
                phases = np.exp(1j*self.modes*self.phi_offset)
                bins = self.modes % self.n_phi
                x = np.zeros((v.shape[0], self.n_phi, k), dtype=complex)
                x[:, bins] = v*phases[None, :, None]
                x = fft.ifft(x, axis=1, norm="ortho")
                z = np.einsum("qpab,qpb->qpa", self.core[sl], x, optimize=False)
                y[sl] = fft.fft(z, axis=1, norm="ortho")[:, bins]*phases.conj()[None, :, None]
        return y.reshape(self.modal_shape)


def prepare_structured_normal(transfer, phi, weights=None, *, representation="lag_direct",
                              max_core_bytes=768*2**20):
    """Prepare one explicitly requested exact representation from the same lift.

    lag_direct: retain only distinct requested differences; gather contraction.
    lag_fft: exact zero-padded linear convolution over the mode span.
    angular_fft: unitary angular FFT plus pointwise channel Gram comparator.
    Setup includes C construction and all transform preparation.  Retained
    objects do not own the original transfer or weights.
    """
    if representation not in ("lag_direct", "lag_fft", "angular_fft"):
        raise ValueError("unknown structured representation")
    info = analyze_modal_normal(transfer, phi)
    w = _weights(transfer, weights)
    if info["core_bytes"][representation] > max_core_bytes:
        raise MemoryError("requested exact core exceeds budget; modes unchanged")
    modes = np.asarray(info["modes"], dtype=np.int64)
    lags = np.asarray(info["lags"] if representation == "lag_direct" else [], dtype=np.int64)
    nq, nm, nz, nc = transfer.modal_shape
    np_, k = info["n_phi"], nz*nc
    length = {"lag_direct": len(lags), "lag_fft": info["fft_length"], "angular_fft": np_}[representation]
    core = np.empty((nq, length, k, k), dtype=complex)
    for start in range(0, nq, transfer.q_chunk):
        sl = slice(start, min(nq, start+transfer.q_chunk))
        c = _angular_gram_chunk(transfer, w, sl)
        if representation == "angular_fft":
            core[sl] = c
            continue
        # ifft gives mean(C exp(+2*pi*i*d*p/N)); preserve actual phi offset.
        coeff = fft.ifft(c, axis=1)
        if representation == "lag_direct":
            core[sl] = coeff[:, lags % np_]*np.exp(1j*lags*info["phi_offset"])[None, :, None, None]
        else:
            span = info["mode_span"]
            shifts = np.arange(1-span, span)
            d = -shifts  # y_m=sum_n H[n-m]x_n is convolution with H[-shift].
            embedded = np.zeros((sl.stop-sl.start, length, k, k), dtype=complex)
            embedded[:, shifts % length] = coeff[:, d % np_]*np.exp(1j*d*info["phi_offset"])[None, :, None, None]
            core[sl] = fft.fft(embedded, axis=1)
    return StructuredModalNormal(tuple(transfer.modal_shape), _readonly(modes), _readonly(core),
                                 representation, np_, info["phi_offset"], transfer.q_chunk,
                                 _readonly(lags), info["fft_length"])
