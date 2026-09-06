"""Explicitly bind a prepared transfer normal G to a compatible radial R.

The density-space action is R* G R. Mode ordering and modal dimensions are
checked; matching physical q coordinates remains the caller's geometry contract.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from .radial_aidt import ModalRadialMap, PhysicalModalTransfer
from .modal_normal_structure import analyze_modal_normal, prepare_structured_normal, _weights, _readonly
from .modal_normal_full_support import prepare_fft_composed_normal


@dataclass(frozen=True)
class ExplicitTransferNormal:
    transfer: PhysicalModalTransfer
    weights: np.ndarray

    @property
    def modal_shape(self):
        return self.transfer.modal_shape

    @property
    def cache_bytes(self):
        return self.transfer.cache_bytes+self.weights.nbytes

    def apply(self, values):
        return self.transfer.adjoint(self.weights*self.transfer.forward(values))


@dataclass(frozen=True)
class PreparedTransferNormal:
    modes: tuple[int, ...]
    action: Any
    representation: str
    incremental_array_bytes_given_transfer: int

    @property
    def modal_shape(self):
        return self.action.modal_shape

    @property
    def cache_bytes(self):
        return self.action.cache_bytes

    def bind_radial(self, radial: ModalRadialMap):
        if radial.modes != self.modes or radial.modal_shape != self.modal_shape:
            raise ValueError("radial mode ordering or modal dimensions differ from G")
        return PreparedRadialNormal(radial,self)


@dataclass(frozen=True)
class PreparedRadialNormal:
    radial: ModalRadialMap
    transfer_normal: PreparedTransferNormal

    def __post_init__(self):
        if self.radial.modes != self.transfer_normal.modes or self.radial.modal_shape != self.transfer_normal.modal_shape:
            raise ValueError("incompatible radial/transfer normal")

    @property
    def input_shape(self):
        return self.radial.input_shape

    @property
    def cache_bytes(self):
        return self.radial.cache_bytes+self.transfer_normal.cache_bytes

    def apply(self, values):
        return self.radial.adjoint(self.transfer_normal.action.apply(self.radial.forward(values)))


def prepare_transfer_normal(transfer,phi,weights=None,*,representation="fft_composed",max_core_bytes=768*2**20):
    """Prepare one named representation; reuse it across compatible radial maps."""
    if representation not in ("explicit_composed","dense_gram","fft_composed","angular_gram"):
        raise ValueError("unknown transfer-normal representation")
    info=analyze_modal_normal(transfer,phi)
    w=_weights(transfer,weights)
    if representation=="explicit_composed":
        action=ExplicitTransferNormal(transfer,_readonly(w.copy()))
        extra=action.weights.nbytes
    elif representation=="dense_gram":
        if info["core_bytes"]["dense"]>max_core_bytes:
            raise MemoryError("dense Gram exceeds core budget; support unchanged")
        action=transfer.compile_normal(w)
        extra=action.cache_bytes
    elif representation=="fft_composed":
        action=prepare_fft_composed_normal(transfer,phi,w)
        extra=action.incremental_array_bytes
    else:
        action=prepare_structured_normal(transfer,phi,w,representation="angular_fft",max_core_bytes=max_core_bytes)
        extra=action.cache_bytes
    return PreparedTransferNormal(tuple(transfer.modes),action,representation,extra)
