from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np
from scipy import special

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def gauss_theta_grid(ntheta: int, theta_max: float) -> tuple[np.ndarray, np.ndarray]:
    """Return theta nodes and weights for the sin(theta) dtheta measure."""
    if ntheta <= 0:
        raise ValueError("ntheta must be positive")
    cos_max = float(np.cos(theta_max))
    x, w = np.polynomial.legendre.leggauss(ntheta)
    half_width = 0.5 * (1.0 - cos_max)
    midpoint = 0.5 * (1.0 + cos_max)
    cos_theta = midpoint + half_width * x
    theta = np.arccos(cos_theta)
    weights = half_width * w
    return theta, weights


def focal_targets(
    *,
    nrho: int,
    npsi: int,
    nz: int,
    rho_max: float,
    z_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho, psi, z = focal_axes(
        nrho=nrho,
        npsi=npsi,
        nz=nz,
        rho_max=rho_max,
        z_max=z_max,
    )
    return flatten_focal_axes(rho, psi, z)


def focal_axes(
    *,
    nrho: int,
    npsi: int,
    nz: int,
    rho_max: float,
    z_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if nrho <= 0 or npsi <= 0 or nz <= 0:
        raise ValueError("nrho, npsi, and nz must be positive")
    rho = np.linspace(0.0, rho_max, nrho, dtype=float)
    psi = np.linspace(0.0, 2.0 * np.pi, npsi, endpoint=False, dtype=float)
    if nz == 1:
        z = np.array([0.0], dtype=float)
    else:
        z = np.linspace(-z_max, z_max, nz, dtype=float)
    return rho, psi, z


def flatten_focal_axes(
    rho: np.ndarray,
    psi: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rr, pp, zz = np.meshgrid(rho, psi, z, indexing="ij")
    return rr.ravel(), pp.ravel(), zz.ravel()


def significant_pupil_h_abs(
    pupils: np.ndarray | list[np.ndarray],
    *,
    relative_threshold: float = 1e-10,
    absolute_threshold: float = 0.0,
) -> np.ndarray:
    """Return positive azimuthal harmonic indices with non-negligible pupil power."""
    if relative_threshold < 0.0:
        raise ValueError("relative_threshold must be non-negative")
    if absolute_threshold < 0.0:
        raise ValueError("absolute_threshold must be non-negative")
    if isinstance(pupils, np.ndarray) and pupils.ndim == 2:
        pupil_stack = pupils[None, :, :]
    else:
        pupil_stack = np.asarray(pupils)
    if pupil_stack.ndim != 3:
        raise ValueError("pupils must have shape (ntheta, nphi) or (batch, ntheta, nphi)")
    nphi = int(pupil_stack.shape[2])
    coeff = np.fft.fft(pupil_stack, axis=2) / float(nphi)
    amplitudes = np.max(np.abs(coeff), axis=(0, 1))
    max_amplitude = float(np.max(amplitudes))
    if max_amplitude == 0.0:
        return np.array([0], dtype=np.int64)
    threshold = max(absolute_threshold, relative_threshold * max_amplitude)
    h_values = np.fft.fftfreq(nphi, d=1.0 / nphi).astype(int)
    h_abs = np.abs(h_values)
    significant = h_abs[amplitudes >= threshold]
    return np.ascontiguousarray(np.unique(significant).astype(np.int64, copy=False))


def extra_pupil_h_abs(
    pupils: np.ndarray | list[np.ndarray],
    *,
    geometric_h_cutoff: int,
    relative_threshold: float = 1e-6,
    absolute_threshold: float = 0.0,
) -> np.ndarray:
    """Return significant pupil harmonics not already covered by the geometry cutoff."""
    if geometric_h_cutoff < 0:
        raise ValueError("geometric_h_cutoff must be non-negative")
    h_abs = significant_pupil_h_abs(
        pupils,
        relative_threshold=relative_threshold,
        absolute_threshold=absolute_threshold,
    )
    return np.ascontiguousarray(h_abs[h_abs > geometric_h_cutoff])


def resolve_pupil_spectrum_required_h_abs(
    required_h_abs: np.ndarray | list[int] | None,
    *,
    nphi: int,
    geometric_h_cutoff: int,
    pupil_spectrum: str = "off",
    pupil_spectrum_pupils: np.ndarray | list[np.ndarray] | None = None,
    relative_threshold: float = 1e-6,
    absolute_threshold: float = 0.0,
    max_available_h_abs: int | None = None,
) -> np.ndarray:
    """Resolve manual and pupil-spectrum harmonic requirements for a plan."""
    if pupil_spectrum not in {"off", "warn", "adaptive"}:
        raise ValueError("pupil_spectrum must be 'off', 'warn', or 'adaptive'")
    if nphi <= 0:
        raise ValueError("nphi must be positive")
    n_half = nphi // 2
    if geometric_h_cutoff < 0:
        raise ValueError("geometric_h_cutoff must be non-negative")
    geometric_h_cutoff = min(int(geometric_h_cutoff), n_half)
    manual = _sanitize_required_h_abs(required_h_abs, n_half=n_half)
    if pupil_spectrum == "off":
        return manual
    if pupil_spectrum_pupils is None:
        raise ValueError(
            "pupil_spectrum_pupils is required when pupil_spectrum is 'warn' or 'adaptive'"
        )

    extra = extra_pupil_h_abs(
        pupil_spectrum_pupils,
        geometric_h_cutoff=geometric_h_cutoff,
        relative_threshold=relative_threshold,
        absolute_threshold=absolute_threshold,
    )
    extra = _sanitize_required_h_abs(extra, n_half=n_half)
    if max_available_h_abs is not None:
        unavailable = extra[extra > int(max_available_h_abs)]
        if unavailable.size:
            values = " ".join(str(int(value)) for value in unavailable[:16])
            raise ValueError(
                "pupil spectrum requires harmonics not present in this basis cache: "
                f"{values}"
            )
    if extra.size:
        values = " ".join(str(int(value)) for value in extra[:16])
        if pupil_spectrum == "warn":
            warnings.warn(
                "pupil spectrum has significant harmonics beyond the geometric "
                f"cutoff {geometric_h_cutoff}: {values}",
                RuntimeWarning,
                stacklevel=2,
            )
            return manual
        return np.ascontiguousarray(np.union1d(manual, extra).astype(np.int64, copy=False))
    return manual


def _sanitize_required_h_abs(
    required_h_abs: np.ndarray | list[int] | None,
    *,
    n_half: int,
) -> np.ndarray:
    if required_h_abs is None:
        return np.empty(0, dtype=np.int64)
    values = np.asarray(required_h_abs, dtype=np.int64).ravel()
    if values.size == 0:
        return np.empty(0, dtype=np.int64)
    values = np.unique(np.abs(values))
    return np.ascontiguousarray(values[values <= n_half])


def _union_h_positions(
    cutoff: int,
    required_h_abs: np.ndarray,
    *,
    max_cutoff: int,
) -> np.ndarray:
    base = np.arange(cutoff + 1, dtype=np.int64)
    if required_h_abs.size == 0:
        return base
    required = required_h_abs[required_h_abs <= max_cutoff]
    if required.size == 0:
        return base
    return np.ascontiguousarray(np.union1d(base, required).astype(np.int64, copy=False))


def pupil_field(
    case: str,
    theta: np.ndarray,
    phi: np.ndarray,
    *,
    theta_max: float,
    strength: float,
    vortex_charge: int,
    apodization: str,
) -> np.ndarray:
    theta_2d = theta[:, None]
    phi_2d = phi[None, :]
    radial = np.sin(theta_2d) / max(np.sin(theta_max), np.finfo(float).eps)

    if case == "clear":
        pupil = np.ones((theta.size, phi.size), dtype=np.complex128)
    elif case == "astigmatism":
        phase = strength * radial**2 * np.cos(2.0 * phi_2d)
        pupil = np.exp(1j * phase)
    elif case == "coma":
        phase = strength * radial**3 * np.cos(phi_2d)
        pupil = np.exp(1j * phase)
    elif case == "vortex":
        pupil = radial ** abs(vortex_charge) * np.exp(1j * vortex_charge * phi_2d)
    elif case == "mixed":
        phase = strength * (
            0.65 * radial**2 * np.cos(2.0 * phi_2d)
            + 0.45 * radial**3 * np.cos(phi_2d)
            + 0.25 * radial**4 * np.sin(3.0 * phi_2d)
        )
        amplitude = 1.0 + 0.12 * radial * np.cos(3.0 * phi_2d)
        pupil = amplitude * np.exp(1j * phase)
    else:
        raise ValueError(f"unknown pupil case: {case}")

    if apodization == "none":
        return pupil
    if apodization == "sqrt-cos":
        return pupil * np.sqrt(np.cos(theta_2d))
    raise ValueError(f"unknown apodization: {apodization}")


def direct_debye_wolf(
    pupil: np.ndarray,
    theta: np.ndarray,
    theta_weights: np.ndarray,
    phi: np.ndarray,
    rho: np.ndarray,
    psi: np.ndarray,
    z: np.ndarray,
    *,
    k: float,
) -> np.ndarray:
    """Direct scalar Debye-Wolf quadrature on the sampled pupil grid."""
    if pupil.shape != (theta.size, phi.size):
        raise ValueError("pupil shape does not match theta/phi grid")

    dphi = 2.0 * np.pi / float(phi.size)
    out = np.zeros(rho.size, dtype=np.complex128)
    phi_col = phi[:, None]
    psi_row = psi[None, :]
    rho_row = rho[None, :]

    for it, theta_i in enumerate(theta):
        sin_theta = float(np.sin(theta_i))
        cos_theta = float(np.cos(theta_i))
        transverse_phase = np.exp(
            1j * k * sin_theta * rho_row * np.cos(phi_col - psi_row)
        )
        defocus_phase = np.exp(1j * k * z * cos_theta)
        out += (
            theta_weights[it]
            * dphi
            * defocus_phase
            * np.sum(pupil[it, :, None] * transverse_phase, axis=0)
        )
    return out


def harmonic_debye_wolf(
    pupil: np.ndarray,
    theta: np.ndarray,
    theta_weights: np.ndarray,
    rho: np.ndarray,
    psi: np.ndarray,
    z: np.ndarray,
    *,
    k: float,
    h_cutoff: int | None,
) -> tuple[np.ndarray, int]:
    """Scalar Debye-Wolf solver using pupil azimuth Fourier modes."""
    nphi = pupil.shape[1]
    h_values = np.fft.fftfreq(nphi, d=1.0 / nphi).astype(int)
    if h_cutoff is None:
        mask = np.ones(h_values.shape, dtype=bool)
    else:
        mask = np.abs(h_values) <= h_cutoff

    h = h_values[mask]
    coeff = np.fft.fft(pupil, axis=1)[:, mask] / float(nphi)
    abs_h = np.abs(h)
    unique_abs_h, inverse_abs_h = np.unique(abs_h, return_inverse=True)
    i_pow = np.power(1j, abs_h)
    angular_rebuild = np.exp(1j * h[:, None] * psi[None, :])

    out = np.zeros(rho.size, dtype=np.complex128)
    for it, theta_i in enumerate(theta):
        sin_theta = float(np.sin(theta_i))
        cos_theta = float(np.cos(theta_i))
        arg = k * rho * sin_theta
        radial_kernel = special.jv(unique_abs_h[:, None], arg[None, :])[
            inverse_abs_h
        ]
        mode_sum = np.sum(
            coeff[it, :, None]
            * i_pow[:, None]
            * radial_kernel
            * angular_rebuild,
            axis=0,
        )
        out += (
            2.0
            * np.pi
            * theta_weights[it]
            * np.exp(1j * k * z * cos_theta)
            * mode_sum
        )
    return out, int(h.size)


@dataclass
class PreparedHarmonicDebyeWolfPlan:
    """Reusable scalar Debye-Wolf basis for fixed geometry and h support."""

    h: np.ndarray
    mask: np.ndarray
    nphi: int
    basis: np.ndarray

    @classmethod
    def build(
        cls,
        nphi: int,
        theta: np.ndarray,
        theta_weights: np.ndarray,
        rho: np.ndarray,
        psi: np.ndarray,
        z: np.ndarray,
        *,
        k: float,
        h_cutoff: int | None,
    ) -> "PreparedHarmonicDebyeWolfPlan":
        h_values = np.fft.fftfreq(nphi, d=1.0 / nphi).astype(int)
        if h_cutoff is None:
            mask = np.ones(h_values.shape, dtype=bool)
        else:
            mask = np.abs(h_values) <= h_cutoff

        h = h_values[mask]
        abs_h = np.abs(h)
        unique_abs_h, inverse_abs_h = np.unique(abs_h, return_inverse=True)
        i_pow = np.power(1j, abs_h)
        angular_rebuild = np.exp(1j * h[:, None] * psi[None, :])
        basis = np.empty((theta.size, h.size, rho.size), dtype=np.complex128)

        for it, theta_i in enumerate(theta):
            sin_theta = float(np.sin(theta_i))
            cos_theta = float(np.cos(theta_i))
            arg = k * rho * sin_theta
            radial_kernel = special.jv(unique_abs_h[:, None], arg[None, :])[
                inverse_abs_h
            ]
            defocus_phase = np.exp(1j * k * z * cos_theta)
            basis[it] = (
                2.0
                * np.pi
                * theta_weights[it]
                * i_pow[:, None]
                * radial_kernel
                * angular_rebuild
                * defocus_phase[None, :]
            )

        return cls(h=h, mask=mask, nphi=nphi, basis=basis)

    @property
    def used_modes(self) -> int:
        return int(self.h.size)

    @property
    def basis_mib(self) -> float:
        return float(self.basis.nbytes / (1024.0 * 1024.0))

    def evaluate(self, pupil: np.ndarray) -> np.ndarray:
        if pupil.shape[1] != self.nphi:
            raise ValueError("pupil nphi does not match prepared plan")
        coeff = np.fft.fft(pupil, axis=1)[:, self.mask] / float(self.nphi)
        return np.einsum("th,thp->p", coeff, self.basis, optimize=True)

    def evaluate_many(self, pupils: list[np.ndarray]) -> np.ndarray:
        coeff = np.stack(
            [
                np.fft.fft(pupil, axis=1)[:, self.mask] / float(self.nphi)
                for pupil in pupils
            ],
            axis=0,
        )
        return np.einsum("bth,thp->bp", coeff, self.basis, optimize=True)


@dataclass
class PreparedSeparableHarmonicDebyeWolfPlan:
    """Memory-light harmonic plan for tensor-product focal grids."""

    h: np.ndarray
    mask: np.ndarray
    nphi: int
    radial: np.ndarray
    angular: np.ndarray
    defocus: np.ndarray
    backend: str = "auto"
    cpp_threads: int = 0

    @classmethod
    def build(
        cls,
        nphi: int,
        theta: np.ndarray,
        theta_weights: np.ndarray,
        rho_axis: np.ndarray,
        psi_axis: np.ndarray,
        z_axis: np.ndarray,
        *,
        k: float,
        h_cutoff: int | None,
        backend: str = "auto",
        cpp_threads: int = 0,
    ) -> "PreparedSeparableHarmonicDebyeWolfPlan":
        if backend not in {"auto", "numpy", "cpp"}:
            raise ValueError("backend must be 'auto', 'numpy', or 'cpp'")
        h_values = np.fft.fftfreq(nphi, d=1.0 / nphi).astype(int)
        if h_cutoff is None:
            mask = np.ones(h_values.shape, dtype=bool)
        else:
            mask = np.abs(h_values) <= h_cutoff

        h = h_values[mask]
        abs_h = np.abs(h)
        unique_abs_h, inverse_abs_h = np.unique(abs_h, return_inverse=True)
        i_pow = np.power(1j, abs_h)
        arg = (
            k
            * np.sin(theta)[:, None, None]
            * rho_axis[None, None, :]
        )
        radial_kernel = special.jv(
            unique_abs_h[None, :, None],
            arg,
        )[:, inverse_abs_h, :]
        radial = (
            2.0
            * np.pi
            * theta_weights[:, None, None]
            * i_pow[None, :, None]
            * radial_kernel
        )
        angular = np.exp(1j * h[:, None] * psi_axis[None, :])
        defocus = np.exp(1j * k * np.cos(theta)[:, None] * z_axis[None, :])
        return cls(
            h=h,
            mask=mask,
            nphi=nphi,
            radial=np.ascontiguousarray(radial),
            angular=np.ascontiguousarray(angular),
            defocus=np.ascontiguousarray(defocus),
            backend=backend,
            cpp_threads=cpp_threads,
        )

    @property
    def used_modes(self) -> int:
        return int(self.h.size)

    @property
    def basis_mib(self) -> float:
        bytes_total = self.radial.nbytes + self.angular.nbytes + self.defocus.nbytes
        return float(bytes_total / (1024.0 * 1024.0))

    def evaluate(self, pupil: np.ndarray) -> np.ndarray:
        return self.evaluate_many([pupil])[0]

    def _evaluate_numpy_coeff(self, coeff: np.ndarray) -> np.ndarray:
        radial_sum = np.einsum("th,thr->rth", coeff, self.radial, optimize=True)
        angular_sum = np.einsum(
            "rth,hp->rtp",
            radial_sum,
            self.angular,
            optimize=True,
        )
        out = np.einsum("rtp,tz->rpz", angular_sum, self.defocus, optimize=True)
        return out.ravel()

    def evaluate_many(self, pupils: list[np.ndarray]) -> np.ndarray:
        for pupil in pupils:
            if pupil.shape[1] != self.nphi:
                raise ValueError("pupil nphi does not match separable plan")
        coeff = np.stack(
            [
                np.fft.fft(pupil, axis=1)[:, self.mask] / float(self.nphi)
                for pupil in pupils
            ],
            axis=0,
        )
        if self.backend in {"auto", "cpp"}:
            try:
                from waxs_cake import _cpp_high_na

                contract = getattr(
                    _cpp_high_na,
                    "separable_contract_many_fused",
                    _cpp_high_na.separable_contract_many,
                )
                return contract(
                    np.ascontiguousarray(coeff),
                    self.radial,
                    self.angular,
                    self.defocus,
                    self.cpp_threads,
                )
            except ImportError:
                if self.backend == "cpp":
                    raise
        radial_sum = np.einsum(
            "bth,thr->brth",
            coeff,
            self.radial,
            optimize=True,
        )
        angular_sum = np.einsum(
            "brth,hp->brtp",
            radial_sum,
            self.angular,
            optimize=True,
        )
        out = np.einsum(
            "brtp,tz->brpz",
            angular_sum,
            self.defocus,
            optimize=True,
        )
        return out.reshape(len(pupils), -1)


@dataclass
class PreparedPositiveSeparableHarmonicDebyeWolfPlan:
    """Separable plan storing each |h| radial basis once."""

    h_abs: np.ndarray
    positive_indices: np.ndarray
    negative_indices: np.ndarray
    nphi: int
    radial: np.ndarray
    angular_positive: np.ndarray
    angular_negative: np.ndarray
    defocus: np.ndarray
    backend: str = "auto"
    cpp_threads: int = 0

    @classmethod
    def build(
        cls,
        nphi: int,
        theta: np.ndarray,
        theta_weights: np.ndarray,
        rho_axis: np.ndarray,
        psi_axis: np.ndarray,
        z_axis: np.ndarray,
        *,
        k: float,
        h_cutoff: int | None,
        backend: str = "auto",
        cpp_threads: int = 0,
    ) -> "PreparedPositiveSeparableHarmonicDebyeWolfPlan":
        if backend not in {"auto", "numpy", "cpp"}:
            raise ValueError("backend must be 'auto', 'numpy', or 'cpp'")
        if cpp_threads < 0:
            raise ValueError("cpp_threads must be non-negative")

        n_half = nphi // 2
        max_cutoff = n_half if h_cutoff is None else min(int(h_cutoff), n_half)
        if max_cutoff < 0:
            raise ValueError("h_cutoff must be non-negative or None")

        h_values = np.fft.fftfreq(nphi, d=1.0 / nphi).astype(int)
        h_to_index = {int(value): idx for idx, value in enumerate(h_values)}
        h_abs = np.arange(max_cutoff + 1, dtype=np.int64)
        positive_indices = np.full(h_abs.shape, -1, dtype=np.int64)
        negative_indices = np.full(h_abs.shape, -1, dtype=np.int64)
        for ih, h_value in enumerate(h_abs):
            h_int = int(h_value)
            positive_indices[ih] = h_to_index.get(h_int, -1)
            if h_int > 0:
                negative_indices[ih] = h_to_index.get(-h_int, -1)

        i_pow = np.power(1j, h_abs)
        arg = (
            k
            * np.sin(theta)[:, None, None]
            * rho_axis[None, None, :]
        )
        radial = (
            2.0
            * np.pi
            * theta_weights[:, None, None]
            * i_pow[None, :, None]
            * special.jv(h_abs[None, :, None], arg)
        )
        angular_positive = np.exp(1j * h_abs[:, None] * psi_axis[None, :])
        angular_negative = np.conjugate(angular_positive)
        defocus = np.exp(1j * k * np.cos(theta)[:, None] * z_axis[None, :])
        return cls(
            h_abs=np.ascontiguousarray(h_abs),
            positive_indices=np.ascontiguousarray(positive_indices),
            negative_indices=np.ascontiguousarray(negative_indices),
            nphi=nphi,
            radial=np.ascontiguousarray(radial),
            angular_positive=np.ascontiguousarray(angular_positive),
            angular_negative=np.ascontiguousarray(angular_negative),
            defocus=np.ascontiguousarray(defocus),
            backend=backend,
            cpp_threads=int(cpp_threads),
        )

    @property
    def used_modes(self) -> int:
        signed = int(np.count_nonzero(self.positive_indices >= 0))
        signed += int(np.count_nonzero(self.negative_indices >= 0))
        return signed

    @property
    def stored_abs_modes(self) -> int:
        return int(self.h_abs.size)

    @property
    def basis_mib(self) -> float:
        bytes_total = (
            self.radial.nbytes
            + self.angular_positive.nbytes
            + self.angular_negative.nbytes
            + self.defocus.nbytes
        )
        return float(bytes_total / (1024.0 * 1024.0))

    def evaluate(self, pupil: np.ndarray) -> np.ndarray:
        return self.evaluate_many([pupil])[0]

    def _coefficients(self, pupils: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        transforms = np.stack(
            [np.fft.fft(pupil, axis=1) / float(self.nphi) for pupil in pupils],
            axis=0,
        )
        shape = (len(pupils), transforms.shape[1], self.h_abs.size)
        coeff_positive = np.zeros(shape, dtype=np.complex128)
        coeff_negative = np.zeros(shape, dtype=np.complex128)
        positive_valid = self.positive_indices >= 0
        negative_valid = self.negative_indices >= 0
        if np.any(positive_valid):
            coeff_positive[:, :, positive_valid] = transforms[
                :,
                :,
                self.positive_indices[positive_valid],
            ]
        if np.any(negative_valid):
            coeff_negative[:, :, negative_valid] = transforms[
                :,
                :,
                self.negative_indices[negative_valid],
            ]
        return np.ascontiguousarray(coeff_positive), np.ascontiguousarray(
            coeff_negative
        )

    def evaluate_many(self, pupils: list[np.ndarray]) -> np.ndarray:
        for pupil in pupils:
            if pupil.shape[1] != self.nphi:
                raise ValueError("pupil nphi does not match positive separable plan")
        coeff_positive, coeff_negative = self._coefficients(pupils)
        if self.backend in {"auto", "cpp"}:
            try:
                from waxs_cake import _cpp_high_na

                return _cpp_high_na.positive_separable_contract_many_fused(
                    coeff_positive,
                    coeff_negative,
                    self.radial,
                    self.angular_positive,
                    self.angular_negative,
                    self.defocus,
                    self.cpp_threads,
                )
            except (AttributeError, ImportError):
                if self.backend == "cpp":
                    raise

        radial_positive = np.einsum(
            "bth,thr->brth",
            coeff_positive,
            self.radial,
            optimize=True,
        )
        radial_negative = np.einsum(
            "bth,thr->brth",
            coeff_negative,
            self.radial,
            optimize=True,
        )
        angular_sum = np.einsum(
            "brth,hp->brtp",
            radial_positive,
            self.angular_positive,
            optimize=True,
        )
        angular_sum += np.einsum(
            "brth,hp->brtp",
            radial_negative,
            self.angular_negative,
            optimize=True,
        )
        out = np.einsum(
            "brtp,tz->brpz",
            angular_sum,
            self.defocus,
            optimize=True,
        )
        return out.reshape(len(pupils), -1)


@dataclass
class RhoDependentHarmonicGroup:
    cutoff: int
    r_indices: np.ndarray
    h_positions: np.ndarray
    h: np.ndarray
    radial: np.ndarray
    angular: np.ndarray


@dataclass
class PositiveRhoDependentHarmonicGroup:
    cutoff: int
    r_indices: np.ndarray
    h_positions: np.ndarray
    h_abs: np.ndarray
    radial: np.ndarray
    angular_positive: np.ndarray
    angular_negative: np.ndarray


@dataclass
class CachedPositiveRhoDependentHarmonicGroup:
    cutoff: int
    r_indices: np.ndarray
    h_positions: np.ndarray


@dataclass
class PositiveRhoDependentBasisCache:
    """Reusable positive-mode basis table for fixed high-NA geometry."""

    h_abs: np.ndarray
    positive_indices: np.ndarray
    negative_indices: np.ndarray
    rho_axis: np.ndarray
    radial: np.ndarray
    angular_positive: np.ndarray
    angular_negative: np.ndarray
    defocus: np.ndarray
    nphi: int
    nrho: int
    npsi: int
    nz: int
    k: float
    sin_theta_max: float
    geometric_h_cutoff: int
    max_cutoff: int
    required_h_abs: np.ndarray

    @classmethod
    def build(
        cls,
        nphi: int,
        theta: np.ndarray,
        theta_weights: np.ndarray,
        rho_axis: np.ndarray,
        psi_axis: np.ndarray,
        z_axis: np.ndarray,
        *,
        k: float,
        h_cutoff: int | None,
        required_h_abs: np.ndarray | list[int] | None = None,
        pupil_spectrum: str = "off",
        pupil_spectrum_pupils: np.ndarray | list[np.ndarray] | None = None,
        pupil_spectrum_relative_threshold: float = 1e-6,
        pupil_spectrum_absolute_threshold: float = 0.0,
        sin_theta_max: float | None = None,
    ) -> "PositiveRhoDependentBasisCache":
        n_half = nphi // 2
        geometric_h_cutoff = n_half if h_cutoff is None else min(int(h_cutoff), n_half)
        required_h_abs_array = resolve_pupil_spectrum_required_h_abs(
            required_h_abs,
            nphi=nphi,
            geometric_h_cutoff=geometric_h_cutoff,
            pupil_spectrum=pupil_spectrum,
            pupil_spectrum_pupils=pupil_spectrum_pupils,
            relative_threshold=pupil_spectrum_relative_threshold,
            absolute_threshold=pupil_spectrum_absolute_threshold,
        )
        max_cutoff = geometric_h_cutoff
        if required_h_abs_array.size:
            max_cutoff = max(max_cutoff, int(required_h_abs_array[-1]))
        if max_cutoff < 0:
            raise ValueError("h_cutoff must be non-negative or None")
        if sin_theta_max is None:
            sin_theta_max = float(np.max(np.sin(theta)))
        if sin_theta_max < 0.0:
            raise ValueError("sin_theta_max must be non-negative")

        h_values = np.fft.fftfreq(nphi, d=1.0 / nphi).astype(int)
        h_to_index = {int(value): idx for idx, value in enumerate(h_values)}
        h_abs = np.arange(max_cutoff + 1, dtype=np.int64)
        positive_indices = np.full(h_abs.shape, -1, dtype=np.int64)
        negative_indices = np.full(h_abs.shape, -1, dtype=np.int64)
        for ih, h_value in enumerate(h_abs):
            h_int = int(h_value)
            positive_indices[ih] = h_to_index.get(h_int, -1)
            if h_int > 0:
                negative_indices[ih] = h_to_index.get(-h_int, -1)

        arg = (
            k
            * np.sin(theta)[:, None, None]
            * rho_axis[None, None, :]
        )
        radial = (
            2.0
            * np.pi
            * theta_weights[:, None, None]
            * np.power(1j, h_abs)[None, :, None]
            * special.jv(h_abs[None, :, None], arg)
        )
        angular_positive = np.exp(1j * h_abs[:, None] * psi_axis[None, :])
        angular_negative = np.conjugate(angular_positive)
        defocus = np.exp(1j * k * np.cos(theta)[:, None] * z_axis[None, :])
        return cls(
            h_abs=np.ascontiguousarray(h_abs),
            positive_indices=np.ascontiguousarray(positive_indices),
            negative_indices=np.ascontiguousarray(negative_indices),
            rho_axis=np.ascontiguousarray(rho_axis),
            radial=np.ascontiguousarray(radial),
            angular_positive=np.ascontiguousarray(angular_positive),
            angular_negative=np.ascontiguousarray(angular_negative),
            defocus=np.ascontiguousarray(defocus),
            nphi=nphi,
            nrho=int(rho_axis.size),
            npsi=int(psi_axis.size),
            nz=int(z_axis.size),
            k=float(k),
            sin_theta_max=float(sin_theta_max),
            geometric_h_cutoff=geometric_h_cutoff,
            max_cutoff=max_cutoff,
            required_h_abs=np.ascontiguousarray(required_h_abs_array),
        )

    @property
    def basis_mib(self) -> float:
        bytes_total = (
            self.radial.nbytes
            + self.angular_positive.nbytes
            + self.angular_negative.nbytes
            + self.defocus.nbytes
        )
        return float(bytes_total / (1024.0 * 1024.0))

    def cutoffs(self, *, margin: int, cutoff_bin_size: int) -> np.ndarray:
        if margin < 0:
            raise ValueError("margin must be non-negative")
        if cutoff_bin_size <= 0:
            raise ValueError("cutoff_bin_size must be positive")
        cutoffs = np.ceil(
            self.k * self.rho_axis * self.sin_theta_max + margin
        ).astype(np.int64, copy=False)
        np.clip(cutoffs, 0, self.max_cutoff, out=cutoffs)
        if cutoff_bin_size > 1:
            cutoffs = (
                ((cutoffs + cutoff_bin_size - 1) // cutoff_bin_size)
                * cutoff_bin_size
            )
            np.clip(cutoffs, 0, self.max_cutoff, out=cutoffs)
        return np.ascontiguousarray(cutoffs)


@dataclass
class PreparedCachedPositiveRhoDependentHarmonicDebyeWolfPlan:
    """Positive rho-dependent plan backed by a reusable full basis cache."""

    cache: PositiveRhoDependentBasisCache
    cutoffs: np.ndarray
    groups: list[CachedPositiveRhoDependentHarmonicGroup]
    margin: int
    cutoff_bin_size: int
    required_h_abs: np.ndarray
    backend: str = "auto"
    cpp_threads: int = 0

    @classmethod
    def build(
        cls,
        cache: PositiveRhoDependentBasisCache,
        *,
        margin: int,
        cutoff_bin_size: int,
        required_h_abs: np.ndarray | list[int] | None = None,
        pupil_spectrum: str = "off",
        pupil_spectrum_pupils: np.ndarray | list[np.ndarray] | None = None,
        pupil_spectrum_relative_threshold: float = 1e-6,
        pupil_spectrum_absolute_threshold: float = 0.0,
        backend: str = "auto",
        cpp_threads: int = 0,
    ) -> "PreparedCachedPositiveRhoDependentHarmonicDebyeWolfPlan":
        if backend not in {"auto", "numpy", "cpp"}:
            raise ValueError("backend must be 'auto', 'numpy', or 'cpp'")
        if cpp_threads < 0:
            raise ValueError("cpp_threads must be non-negative")
        cutoffs = cache.cutoffs(margin=margin, cutoff_bin_size=cutoff_bin_size)
        required_h_abs_array = resolve_pupil_spectrum_required_h_abs(
            required_h_abs,
            nphi=cache.nphi,
            geometric_h_cutoff=cache.geometric_h_cutoff,
            pupil_spectrum=pupil_spectrum,
            pupil_spectrum_pupils=pupil_spectrum_pupils,
            relative_threshold=pupil_spectrum_relative_threshold,
            absolute_threshold=pupil_spectrum_absolute_threshold,
            max_available_h_abs=cache.max_cutoff,
        )
        groups: list[CachedPositiveRhoDependentHarmonicGroup] = []
        for cutoff_value in np.unique(cutoffs):
            cutoff = int(cutoff_value)
            r_indices = np.flatnonzero(cutoffs == cutoff).astype(np.int64, copy=False)
            h_positions = _union_h_positions(
                cutoff,
                required_h_abs_array,
                max_cutoff=cache.max_cutoff,
            )
            groups.append(
                CachedPositiveRhoDependentHarmonicGroup(
                    cutoff=cutoff,
                    r_indices=np.ascontiguousarray(r_indices),
                    h_positions=np.ascontiguousarray(h_positions),
                )
            )
        return cls(
            cache=cache,
            cutoffs=cutoffs,
            groups=groups,
            margin=int(margin),
            cutoff_bin_size=int(cutoff_bin_size),
            required_h_abs=np.ascontiguousarray(required_h_abs_array),
            backend=backend,
            cpp_threads=int(cpp_threads),
        )

    @property
    def group_count(self) -> int:
        return int(len(self.groups))

    @property
    def used_modes(self) -> int:
        return int(max((self._signed_count(group.h_positions) for group in self.groups), default=0))

    @property
    def mean_used_modes(self) -> float:
        if self.cache.nrho == 0:
            return 0.0
        total = sum(
            self._signed_count(group.h_positions) * group.r_indices.size
            for group in self.groups
        )
        return float(total / self.cache.nrho)

    @property
    def mean_stored_abs_modes(self) -> float:
        if self.cache.nrho == 0:
            return 0.0
        total = sum(group.h_positions.size * group.r_indices.size for group in self.groups)
        return float(total / self.cache.nrho)

    @property
    def mode_rho_work(self) -> int:
        return int(sum(group.h_positions.size * group.r_indices.size for group in self.groups))

    @property
    def signed_mode_rho_work(self) -> int:
        return int(
            sum(
                self._signed_count(group.h_positions) * group.r_indices.size
                for group in self.groups
            )
        )

    @property
    def basis_mib(self) -> float:
        return self.cache.basis_mib

    @property
    def nphi(self) -> int:
        return self.cache.nphi

    @property
    def nrho(self) -> int:
        return self.cache.nrho

    @property
    def npsi(self) -> int:
        return self.cache.npsi

    @property
    def nz(self) -> int:
        return self.cache.nz

    def _signed_count(self, h_positions: np.ndarray) -> int:
        positive = self.cache.positive_indices[h_positions] >= 0
        negative = self.cache.negative_indices[h_positions] >= 0
        return int(np.count_nonzero(positive) + np.count_nonzero(negative))

    def evaluate(self, pupil: np.ndarray) -> np.ndarray:
        return self.evaluate_many([pupil])[0]

    def evaluate_many(self, pupils: list[np.ndarray]) -> np.ndarray:
        for pupil in pupils:
            if pupil.shape[1] != self.cache.nphi:
                raise ValueError("pupil nphi does not match cached positive plan")
        if self.backend not in {"auto", "cpp"}:
            raise ValueError("cached positive-rho plan currently requires C++ backend")
        transforms = np.ascontiguousarray(
            np.stack(
                [np.fft.fft(pupil, axis=1) / float(self.cache.nphi) for pupil in pupils],
                axis=0,
            )
        )
        from waxs_cake import _cpp_high_na

        return _cpp_high_na.positive_rho_dependent_contract_many_cached_nocopy(
            transforms,
            self.cache.positive_indices,
            self.cache.negative_indices,
            [group.r_indices for group in self.groups],
            [group.h_positions for group in self.groups],
            self.cache.radial,
            self.cache.angular_positive,
            self.cache.angular_negative,
            self.cache.defocus,
            self.cache.nrho,
            self.cpp_threads,
        )


@dataclass
class PreparedRhoDependentHarmonicDebyeWolfPlan:
    """Separable harmonic plan with WAXS-style rho-dependent mode cutoffs."""

    h: np.ndarray
    max_mask: np.ndarray
    cutoffs: np.ndarray
    nphi: int
    nrho: int
    npsi: int
    nz: int
    defocus: np.ndarray
    groups: list[RhoDependentHarmonicGroup]
    margin: int
    cutoff_bin_size: int
    max_cutoff: int
    backend: str = "auto"
    cpp_threads: int = 0

    @classmethod
    def build(
        cls,
        nphi: int,
        theta: np.ndarray,
        theta_weights: np.ndarray,
        rho_axis: np.ndarray,
        psi_axis: np.ndarray,
        z_axis: np.ndarray,
        *,
        k: float,
        h_cutoff: int | None,
        margin: int,
        cutoff_bin_size: int,
        sin_theta_max: float | None = None,
        backend: str = "auto",
        cpp_threads: int = 0,
    ) -> "PreparedRhoDependentHarmonicDebyeWolfPlan":
        if backend not in {"auto", "numpy", "cpp"}:
            raise ValueError("backend must be 'auto', 'numpy', or 'cpp'")
        if margin < 0:
            raise ValueError("margin must be non-negative")
        if cutoff_bin_size <= 0:
            raise ValueError("cutoff_bin_size must be positive")
        if cpp_threads < 0:
            raise ValueError("cpp_threads must be non-negative")

        n_half = nphi // 2
        max_cutoff = n_half if h_cutoff is None else min(int(h_cutoff), n_half)
        if max_cutoff < 0:
            raise ValueError("h_cutoff must be non-negative or None")
        if sin_theta_max is None:
            sin_theta_max = float(np.max(np.sin(theta)))
        if sin_theta_max < 0.0:
            raise ValueError("sin_theta_max must be non-negative")

        h_values = np.fft.fftfreq(nphi, d=1.0 / nphi).astype(int)
        abs_h = np.abs(h_values)
        max_mask = abs_h <= max_cutoff
        h = h_values[max_mask]
        abs_h_kept = np.abs(h)

        cutoffs = np.ceil(k * rho_axis * sin_theta_max + margin).astype(
            np.int64,
            copy=False,
        )
        np.clip(cutoffs, 0, max_cutoff, out=cutoffs)
        if cutoff_bin_size > 1:
            cutoffs = (
                ((cutoffs + cutoff_bin_size - 1) // cutoff_bin_size)
                * cutoff_bin_size
            )
            np.clip(cutoffs, 0, max_cutoff, out=cutoffs)

        sin_theta = np.sin(theta)
        defocus = np.exp(1j * k * np.cos(theta)[:, None] * z_axis[None, :])
        groups: list[RhoDependentHarmonicGroup] = []
        for cutoff_value in np.unique(cutoffs):
            cutoff = int(cutoff_value)
            r_indices = np.flatnonzero(cutoffs == cutoff).astype(np.intp, copy=False)
            h_positions = np.flatnonzero(abs_h_kept <= cutoff).astype(
                np.intp,
                copy=False,
            )
            h_group = h[h_positions]
            abs_h_group = np.abs(h_group)
            unique_abs_h, inverse_abs_h = np.unique(
                abs_h_group,
                return_inverse=True,
            )
            i_pow = np.power(1j, abs_h_group)
            arg = (
                k
                * sin_theta[:, None, None]
                * rho_axis[r_indices][None, None, :]
            )
            radial_kernel = special.jv(
                unique_abs_h[None, :, None],
                arg,
            )[:, inverse_abs_h, :]
            radial = (
                2.0
                * np.pi
                * theta_weights[:, None, None]
                * i_pow[None, :, None]
                * radial_kernel
            )
            angular = np.exp(1j * h_group[:, None] * psi_axis[None, :])
            groups.append(
                RhoDependentHarmonicGroup(
                    cutoff=cutoff,
                    r_indices=np.ascontiguousarray(r_indices),
                    h_positions=np.ascontiguousarray(h_positions),
                    h=np.ascontiguousarray(h_group),
                    radial=np.ascontiguousarray(radial),
                    angular=np.ascontiguousarray(angular),
                )
            )

        return cls(
            h=np.ascontiguousarray(h),
            max_mask=np.ascontiguousarray(max_mask),
            cutoffs=np.ascontiguousarray(cutoffs),
            nphi=nphi,
            nrho=int(rho_axis.size),
            npsi=int(psi_axis.size),
            nz=int(z_axis.size),
            defocus=np.ascontiguousarray(defocus),
            groups=groups,
            margin=int(margin),
            cutoff_bin_size=int(cutoff_bin_size),
            max_cutoff=max_cutoff,
            backend=backend,
            cpp_threads=int(cpp_threads),
        )

    @property
    def used_modes(self) -> int:
        return int(max((group.h.size for group in self.groups), default=0))

    @property
    def mean_used_modes(self) -> float:
        if self.nrho == 0:
            return 0.0
        total = sum(group.h.size * group.r_indices.size for group in self.groups)
        return float(total / self.nrho)

    @property
    def mode_rho_work(self) -> int:
        return int(sum(group.h.size * group.r_indices.size for group in self.groups))

    @property
    def basis_mib(self) -> float:
        bytes_total = self.defocus.nbytes
        for group in self.groups:
            bytes_total += group.radial.nbytes + group.angular.nbytes
        return float(bytes_total / (1024.0 * 1024.0))

    @property
    def group_count(self) -> int:
        return int(len(self.groups))

    def evaluate(self, pupil: np.ndarray) -> np.ndarray:
        return self.evaluate_many([pupil])[0]

    def evaluate_many(self, pupils: list[np.ndarray]) -> np.ndarray:
        for pupil in pupils:
            if pupil.shape[1] != self.nphi:
                raise ValueError("pupil nphi does not match rho-dependent plan")
        coeff = np.stack(
            [
                np.fft.fft(pupil, axis=1)[:, self.max_mask] / float(self.nphi)
                for pupil in pupils
            ],
            axis=0,
        )
        if self.backend in {"auto", "cpp"}:
            try:
                from waxs_cake import _cpp_high_na

                return _cpp_high_na.rho_dependent_contract_many_fused(
                    np.ascontiguousarray(coeff),
                    [group.r_indices for group in self.groups],
                    [group.h_positions for group in self.groups],
                    [group.radial for group in self.groups],
                    [group.angular for group in self.groups],
                    self.defocus,
                    self.nrho,
                    self.cpp_threads,
                )
            except (AttributeError, ImportError):
                if self.backend == "cpp":
                    raise
        out = np.empty(
            (len(pupils), self.nrho, self.npsi, self.nz),
            dtype=np.complex128,
        )
        for group in self.groups:
            coeff_group = coeff[:, :, group.h_positions]
            radial_sum = np.einsum(
                "bth,thr->brth",
                coeff_group,
                group.radial,
                optimize=True,
            )
            angular_sum = np.einsum(
                "brth,hp->brtp",
                radial_sum,
                group.angular,
                optimize=True,
            )
            out[:, group.r_indices, :, :] = np.einsum(
                "brtp,tz->brpz",
                angular_sum,
                self.defocus,
                optimize=True,
            )
        return out.reshape(len(pupils), -1)


@dataclass
class PreparedPositiveRhoDependentHarmonicDebyeWolfPlan:
    """Rho-dependent harmonic plan storing radial basis by |h| only."""

    h_abs: np.ndarray
    positive_indices: np.ndarray
    negative_indices: np.ndarray
    cutoffs: np.ndarray
    nphi: int
    nrho: int
    npsi: int
    nz: int
    defocus: np.ndarray
    groups: list[PositiveRhoDependentHarmonicGroup]
    margin: int
    cutoff_bin_size: int
    geometric_h_cutoff: int
    max_cutoff: int
    required_h_abs: np.ndarray
    backend: str = "auto"
    cpp_threads: int = 0
    no_copy_coefficients: bool = False

    @classmethod
    def build(
        cls,
        nphi: int,
        theta: np.ndarray,
        theta_weights: np.ndarray,
        rho_axis: np.ndarray,
        psi_axis: np.ndarray,
        z_axis: np.ndarray,
        *,
        k: float,
        h_cutoff: int | None,
        margin: int,
        cutoff_bin_size: int,
        required_h_abs: np.ndarray | list[int] | None = None,
        pupil_spectrum: str = "off",
        pupil_spectrum_pupils: np.ndarray | list[np.ndarray] | None = None,
        pupil_spectrum_relative_threshold: float = 1e-6,
        pupil_spectrum_absolute_threshold: float = 0.0,
        sin_theta_max: float | None = None,
        backend: str = "auto",
        cpp_threads: int = 0,
        no_copy_coefficients: bool = False,
    ) -> "PreparedPositiveRhoDependentHarmonicDebyeWolfPlan":
        if backend not in {"auto", "numpy", "cpp"}:
            raise ValueError("backend must be 'auto', 'numpy', or 'cpp'")
        if margin < 0:
            raise ValueError("margin must be non-negative")
        if cutoff_bin_size <= 0:
            raise ValueError("cutoff_bin_size must be positive")
        if cpp_threads < 0:
            raise ValueError("cpp_threads must be non-negative")

        n_half = nphi // 2
        geometric_h_cutoff = n_half if h_cutoff is None else min(int(h_cutoff), n_half)
        required_h_abs_array = resolve_pupil_spectrum_required_h_abs(
            required_h_abs,
            nphi=nphi,
            geometric_h_cutoff=geometric_h_cutoff,
            pupil_spectrum=pupil_spectrum,
            pupil_spectrum_pupils=pupil_spectrum_pupils,
            relative_threshold=pupil_spectrum_relative_threshold,
            absolute_threshold=pupil_spectrum_absolute_threshold,
        )
        max_cutoff = geometric_h_cutoff
        if required_h_abs_array.size:
            max_cutoff = max(max_cutoff, int(required_h_abs_array[-1]))
        if max_cutoff < 0:
            raise ValueError("h_cutoff must be non-negative or None")
        if sin_theta_max is None:
            sin_theta_max = float(np.max(np.sin(theta)))
        if sin_theta_max < 0.0:
            raise ValueError("sin_theta_max must be non-negative")

        h_values = np.fft.fftfreq(nphi, d=1.0 / nphi).astype(int)
        h_to_index = {int(value): idx for idx, value in enumerate(h_values)}
        h_abs = np.arange(max_cutoff + 1, dtype=np.int64)
        positive_indices = np.full(h_abs.shape, -1, dtype=np.int64)
        negative_indices = np.full(h_abs.shape, -1, dtype=np.int64)
        for ih, h_value in enumerate(h_abs):
            h_int = int(h_value)
            positive_indices[ih] = h_to_index.get(h_int, -1)
            if h_int > 0:
                negative_indices[ih] = h_to_index.get(-h_int, -1)

        cutoffs = np.ceil(k * rho_axis * sin_theta_max + margin).astype(
            np.int64,
            copy=False,
        )
        np.clip(cutoffs, 0, max_cutoff, out=cutoffs)
        if cutoff_bin_size > 1:
            cutoffs = (
                ((cutoffs + cutoff_bin_size - 1) // cutoff_bin_size)
                * cutoff_bin_size
            )
            np.clip(cutoffs, 0, max_cutoff, out=cutoffs)

        sin_theta = np.sin(theta)
        defocus = np.exp(1j * k * np.cos(theta)[:, None] * z_axis[None, :])
        groups: list[PositiveRhoDependentHarmonicGroup] = []
        for cutoff_value in np.unique(cutoffs):
            cutoff = int(cutoff_value)
            r_indices = np.flatnonzero(cutoffs == cutoff).astype(np.int64, copy=False)
            h_positions = _union_h_positions(
                cutoff,
                required_h_abs_array,
                max_cutoff=max_cutoff,
            )
            h_group = h_abs[h_positions]
            i_pow = np.power(1j, h_group)
            arg = (
                k
                * sin_theta[:, None, None]
                * rho_axis[r_indices][None, None, :]
            )
            radial = (
                2.0
                * np.pi
                * theta_weights[:, None, None]
                * i_pow[None, :, None]
                * special.jv(h_group[None, :, None], arg)
            )
            angular_positive = np.exp(1j * h_group[:, None] * psi_axis[None, :])
            angular_negative = np.conjugate(angular_positive)
            groups.append(
                PositiveRhoDependentHarmonicGroup(
                    cutoff=cutoff,
                    r_indices=np.ascontiguousarray(r_indices),
                    h_positions=np.ascontiguousarray(h_positions),
                    h_abs=np.ascontiguousarray(h_group),
                    radial=np.ascontiguousarray(radial),
                    angular_positive=np.ascontiguousarray(angular_positive),
                    angular_negative=np.ascontiguousarray(angular_negative),
                )
            )

        return cls(
            h_abs=np.ascontiguousarray(h_abs),
            positive_indices=np.ascontiguousarray(positive_indices),
            negative_indices=np.ascontiguousarray(negative_indices),
            cutoffs=np.ascontiguousarray(cutoffs),
            nphi=nphi,
            nrho=int(rho_axis.size),
            npsi=int(psi_axis.size),
            nz=int(z_axis.size),
            defocus=np.ascontiguousarray(defocus),
            groups=groups,
            margin=int(margin),
            cutoff_bin_size=int(cutoff_bin_size),
            geometric_h_cutoff=geometric_h_cutoff,
            max_cutoff=max_cutoff,
            required_h_abs=np.ascontiguousarray(required_h_abs_array),
            backend=backend,
            cpp_threads=int(cpp_threads),
            no_copy_coefficients=bool(no_copy_coefficients),
        )

    @property
    def used_modes(self) -> int:
        return int(max((self._signed_count(group.h_positions) for group in self.groups), default=0))

    @property
    def stored_abs_modes(self) -> int:
        return int(max((group.h_abs.size for group in self.groups), default=0))

    @property
    def mean_used_modes(self) -> float:
        if self.nrho == 0:
            return 0.0
        total = sum(
            self._signed_count(group.h_positions) * group.r_indices.size
            for group in self.groups
        )
        return float(total / self.nrho)

    @property
    def mean_stored_abs_modes(self) -> float:
        if self.nrho == 0:
            return 0.0
        total = sum(group.h_abs.size * group.r_indices.size for group in self.groups)
        return float(total / self.nrho)

    @property
    def mode_rho_work(self) -> int:
        return int(sum(group.h_abs.size * group.r_indices.size for group in self.groups))

    @property
    def signed_mode_rho_work(self) -> int:
        return int(
            sum(
                self._signed_count(group.h_positions) * group.r_indices.size
                for group in self.groups
            )
        )

    @property
    def basis_mib(self) -> float:
        bytes_total = self.defocus.nbytes
        for group in self.groups:
            bytes_total += (
                group.radial.nbytes
                + group.angular_positive.nbytes
                + group.angular_negative.nbytes
            )
        return float(bytes_total / (1024.0 * 1024.0))

    @property
    def group_count(self) -> int:
        return int(len(self.groups))

    def _signed_count(self, h_positions: np.ndarray) -> int:
        positive = self.positive_indices[h_positions] >= 0
        negative = self.negative_indices[h_positions] >= 0
        return int(np.count_nonzero(positive) + np.count_nonzero(negative))

    def evaluate(self, pupil: np.ndarray) -> np.ndarray:
        return self.evaluate_many([pupil])[0]

    def _coefficients(self, pupils: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        transforms = np.stack(
            [np.fft.fft(pupil, axis=1) / float(self.nphi) for pupil in pupils],
            axis=0,
        )
        shape = (len(pupils), transforms.shape[1], self.h_abs.size)
        coeff_positive = np.zeros(shape, dtype=np.complex128)
        coeff_negative = np.zeros(shape, dtype=np.complex128)
        positive_valid = self.positive_indices >= 0
        negative_valid = self.negative_indices >= 0
        if np.any(positive_valid):
            coeff_positive[:, :, positive_valid] = transforms[
                :,
                :,
                self.positive_indices[positive_valid],
            ]
        if np.any(negative_valid):
            coeff_negative[:, :, negative_valid] = transforms[
                :,
                :,
                self.negative_indices[negative_valid],
            ]
        return np.ascontiguousarray(coeff_positive), np.ascontiguousarray(
            coeff_negative
        )

    def evaluate_many(self, pupils: list[np.ndarray]) -> np.ndarray:
        for pupil in pupils:
            if pupil.shape[1] != self.nphi:
                raise ValueError("pupil nphi does not match positive rho-dependent plan")
        if self.no_copy_coefficients and self.backend in {"auto", "cpp"}:
            transforms = np.ascontiguousarray(
                np.stack(
                    [
                        np.fft.fft(pupil, axis=1) / float(self.nphi)
                        for pupil in pupils
                    ],
                    axis=0,
                )
            )
            try:
                from waxs_cake import _cpp_high_na

                return _cpp_high_na.positive_rho_dependent_contract_many_nocopy(
                    transforms,
                    self.positive_indices,
                    self.negative_indices,
                    [group.r_indices for group in self.groups],
                    [group.h_positions for group in self.groups],
                    [group.radial for group in self.groups],
                    [group.angular_positive for group in self.groups],
                    [group.angular_negative for group in self.groups],
                    self.defocus,
                    self.nrho,
                    self.cpp_threads,
                )
            except (AttributeError, ImportError):
                if self.backend == "cpp":
                    raise

        coeff_positive, coeff_negative = self._coefficients(pupils)
        if self.backend in {"auto", "cpp"}:
            try:
                from waxs_cake import _cpp_high_na

                return _cpp_high_na.positive_rho_dependent_contract_many_fused(
                    coeff_positive,
                    coeff_negative,
                    [group.r_indices for group in self.groups],
                    [group.h_positions for group in self.groups],
                    [group.radial for group in self.groups],
                    [group.angular_positive for group in self.groups],
                    [group.angular_negative for group in self.groups],
                    self.defocus,
                    self.nrho,
                    self.cpp_threads,
                )
            except (AttributeError, ImportError):
                if self.backend == "cpp":
                    raise

        out = np.empty(
            (len(pupils), self.nrho, self.npsi, self.nz),
            dtype=np.complex128,
        )
        for group in self.groups:
            coeff_positive_group = coeff_positive[:, :, group.h_positions]
            coeff_negative_group = coeff_negative[:, :, group.h_positions]
            radial_positive = np.einsum(
                "bth,thr->brth",
                coeff_positive_group,
                group.radial,
                optimize=True,
            )
            radial_negative = np.einsum(
                "bth,thr->brth",
                coeff_negative_group,
                group.radial,
                optimize=True,
            )
            angular_sum = np.einsum(
                "brth,hp->brtp",
                radial_positive,
                group.angular_positive,
                optimize=True,
            )
            angular_sum += np.einsum(
                "brth,hp->brtp",
                radial_negative,
                group.angular_negative,
                optimize=True,
            )
            out[:, group.r_indices, :, :] = np.einsum(
                "brtp,tz->brpz",
                angular_sum,
                self.defocus,
                optimize=True,
            )
        return out.reshape(len(pupils), -1)


@dataclass
class PreparedFinufftDebyeWolfPlan:
    """FINUFFT type-3 baseline for the same scalar Debye-Wolf quadrature."""

    source_x: np.ndarray
    source_y: np.ndarray
    source_z: np.ndarray
    target_s: np.ndarray
    target_t: np.ndarray
    target_u: np.ndarray
    source_weight: np.ndarray
    ntheta: int
    nphi: int
    coord_scale: float

    @classmethod
    def build(
        cls,
        theta: np.ndarray,
        theta_weights: np.ndarray,
        phi: np.ndarray,
        rho: np.ndarray,
        psi: np.ndarray,
        z: np.ndarray,
        *,
        k: float,
    ) -> "PreparedFinufftDebyeWolfPlan":
        theta_2d = theta[:, None]
        phi_2d = phi[None, :]
        xi_x = k * np.sin(theta_2d) * np.cos(phi_2d)
        xi_y = k * np.sin(theta_2d) * np.sin(phi_2d)
        xi_z = np.broadcast_to(k * np.cos(theta_2d), xi_x.shape)

        max_abs_xi = float(
            max(
                np.max(np.abs(xi_x)),
                np.max(np.abs(xi_y)),
                np.max(np.abs(xi_z)),
            )
        )
        coord_scale = max(max_abs_xi / (0.95 * np.pi), 1.0)

        dphi = 2.0 * np.pi / float(phi.size)
        source_weight = np.broadcast_to(
            theta_weights[:, None] * dphi,
            (theta.size, phi.size),
        ).copy()

        return cls(
            source_x=np.ascontiguousarray((xi_x / coord_scale).ravel()),
            source_y=np.ascontiguousarray((xi_y / coord_scale).ravel()),
            source_z=np.ascontiguousarray((xi_z / coord_scale).ravel()),
            target_s=np.ascontiguousarray(rho * np.cos(psi) * coord_scale),
            target_t=np.ascontiguousarray(rho * np.sin(psi) * coord_scale),
            target_u=np.ascontiguousarray(z * coord_scale),
            source_weight=source_weight,
            ntheta=theta.size,
            nphi=phi.size,
            coord_scale=float(coord_scale),
        )

    @property
    def sources(self) -> int:
        return int(self.source_x.size)

    @property
    def targets(self) -> int:
        return int(self.target_s.size)

    @property
    def coordinate_mib(self) -> float:
        bytes_total = (
            self.source_x.nbytes
            + self.source_y.nbytes
            + self.source_z.nbytes
            + self.target_s.nbytes
            + self.target_t.nbytes
            + self.target_u.nbytes
            + self.source_weight.nbytes
        )
        return float(bytes_total / (1024.0 * 1024.0))

    def strengths(self, pupil: np.ndarray) -> np.ndarray:
        if pupil.shape != (self.ntheta, self.nphi):
            raise ValueError("pupil shape does not match FINUFFT plan")
        return np.ascontiguousarray((pupil * self.source_weight).ravel())

    def evaluate(self, pupil: np.ndarray, *, eps: float) -> np.ndarray:
        try:
            import finufft
        except ImportError as exc:
            raise RuntimeError("finufft is not installed") from exc

        return finufft.nufft3d3(
            self.source_x,
            self.source_y,
            self.source_z,
            self.strengths(pupil),
            self.target_s,
            self.target_t,
            self.target_u,
            eps=eps,
            isign=1,
        )

    def evaluate_many(self, pupils: list[np.ndarray], *, eps: float) -> np.ndarray:
        try:
            import finufft
        except ImportError as exc:
            raise RuntimeError("finufft is not installed") from exc

        strengths = np.ascontiguousarray(
            np.stack([self.strengths(pupil) for pupil in pupils], axis=0)
        )
        return finufft.nufft3d3(
            self.source_x,
            self.source_y,
            self.source_z,
            strengths,
            self.target_s,
            self.target_t,
            self.target_u,
            eps=eps,
            isign=1,
        )


def evaluate_direct_sequence(
    pupils: list[np.ndarray],
    theta: np.ndarray,
    theta_weights: np.ndarray,
    phi: np.ndarray,
    rho: np.ndarray,
    psi: np.ndarray,
    z: np.ndarray,
    *,
    k: float,
) -> np.ndarray:
    return np.stack(
        [
            direct_debye_wolf(
                pupil,
                theta,
                theta_weights,
                phi,
                rho,
                psi,
                z,
                k=k,
            )
            for pupil in pupils
        ],
        axis=0,
    )


def evaluate_plan_sequence(
    plan: PreparedHarmonicDebyeWolfPlan,
    pupils: list[np.ndarray],
) -> np.ndarray:
    if hasattr(plan, "evaluate_many"):
        return plan.evaluate_many(pupils)
    return np.stack([plan.evaluate(pupil) for pupil in pupils], axis=0)


def relative_l2(got: np.ndarray, ref: np.ndarray) -> float:
    denom = float(np.linalg.norm(ref))
    if denom == 0.0:
        return float(np.linalg.norm(got - ref))
    return float(np.linalg.norm(got - ref) / denom)


def max_abs_over_ref(got: np.ndarray, ref: np.ndarray) -> float:
    denom = float(np.max(np.abs(ref)))
    if denom == 0.0:
        return float(np.max(np.abs(got - ref)))
    return float(np.max(np.abs(got - ref)) / denom)


def median_time(func, repeats: int) -> tuple[object, float, list[float]]:
    value = None
    times: list[float] = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        value = func()
        times.append(time.perf_counter() - start)
    if value is None:
        raise RuntimeError("timed function did not run")
    return value, float(median(times)), times


def parse_cases(value: str) -> list[str]:
    cases = [part.strip() for part in value.split(",") if part.strip()]
    if not cases:
        raise ValueError("at least one pupil case is required")
    return cases


def parse_h_cutoffs(value: str) -> list[int | None]:
    cutoffs: list[int | None] = []
    for part in value.split(","):
        token = part.strip().lower()
        if not token:
            continue
        if token == "full":
            cutoffs.append(None)
        else:
            cutoff = int(token)
            if cutoff < 0:
                raise ValueError("h cutoffs must be non-negative or 'full'")
            cutoffs.append(cutoff)
    if not cutoffs:
        raise ValueError("at least one h cutoff is required")
    return cutoffs


def cutoff_label(cutoff: int | None) -> str:
    return "full" if cutoff is None else str(cutoff)


def sweep_strengths(base_strength: float, sweeps: int) -> list[float]:
    if sweeps <= 0:
        raise ValueError("sweeps must be positive")
    if sweeps == 1:
        return [base_strength]
    lo = 0.5 * base_strength
    hi = 1.5 * base_strength
    return [float(value) for value in np.linspace(lo, hi, sweeps)]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare direct scalar Debye-Wolf quadrature against a pupil-azimuth "
            "harmonic solver."
        )
    )
    parser.add_argument("--cases", default="clear,astigmatism,coma,vortex,mixed")
    parser.add_argument("--ntheta", type=int, default=32)
    parser.add_argument("--nphi", type=int, default=96)
    parser.add_argument("--nrho", type=int, default=7)
    parser.add_argument("--npsi", type=int, default=32)
    parser.add_argument("--nz", type=int, default=3)
    parser.add_argument("--rho-max", type=float, default=0.75)
    parser.add_argument("--z-max", type=float, default=0.8)
    parser.add_argument("--wavelength", type=float, default=0.532)
    parser.add_argument("--na", type=float, default=0.8)
    parser.add_argument("--n-medium", type=float, default=1.0)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--vortex-charge", type=int, default=1)
    parser.add_argument("--apodization", choices=["none", "sqrt-cos"], default="none")
    parser.add_argument("--h-cutoffs", default="full,8,12,16")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--finufft-eps", type=float, default=1e-9)
    parser.add_argument("--skip-finufft", action="store_true")
    parser.add_argument(
        "--harmonic-backend",
        choices=["auto", "numpy", "cpp"],
        default="auto",
    )
    parser.add_argument("--cpp-threads", type=int, default=0)
    parser.add_argument(
        "--sweeps",
        type=int,
        default=6,
        help="number of repeated pupil updates for the prepared-plan hot loop",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmark_results/high_na_debye_wolf_scalar_smoke.json"),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("benchmark_results/high_na_debye_wolf_scalar_smoke.csv"),
    )
    args = parser.parse_args()

    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    if args.finufft_eps <= 0.0:
        raise ValueError("finufft-eps must be positive")
    if args.sweeps <= 0:
        raise ValueError("sweeps must be positive")
    if args.cpp_threads < 0:
        raise ValueError("cpp-threads must be non-negative")
    if args.wavelength <= 0.0:
        raise ValueError("wavelength must be positive")
    if args.n_medium <= 0.0:
        raise ValueError("n-medium must be positive")
    if not (0.0 < args.na <= args.n_medium):
        raise ValueError("NA must satisfy 0 < NA <= n-medium")

    cases = parse_cases(args.cases)
    h_cutoffs = parse_h_cutoffs(args.h_cutoffs)

    theta_max = float(np.arcsin(args.na / args.n_medium))
    theta, theta_weights = gauss_theta_grid(args.ntheta, theta_max)
    phi = np.linspace(0.0, 2.0 * np.pi, args.nphi, endpoint=False, dtype=float)
    rho_axis, psi_axis, z_axis = focal_axes(
        nrho=args.nrho,
        npsi=args.npsi,
        nz=args.nz,
        rho_max=args.rho_max,
        z_max=args.z_max,
    )
    rho, psi, z = flatten_focal_axes(rho_axis, psi_axis, z_axis)
    k = 2.0 * np.pi * args.n_medium / args.wavelength
    if args.skip_finufft:
        finufft_plan = None
        finufft_build_time = None
        finufft_build_times = []
    else:
        finufft_plan, finufft_build_time, finufft_build_times = median_time(
            lambda: PreparedFinufftDebyeWolfPlan.build(
                theta,
                theta_weights,
                phi,
                rho,
                psi,
                z,
                k=k,
            ),
            args.repeats,
        )

    rows: list[dict[str, object]] = []
    for case in cases:
        strength_values = sweep_strengths(args.strength, args.sweeps)
        pupil_sequence = [
            pupil_field(
                case,
                theta,
                phi,
                theta_max=theta_max,
                strength=strength,
                vortex_charge=args.vortex_charge,
                apodization=args.apodization,
            )
            for strength in strength_values
        ]
        pupil = pupil_field(
            case,
            theta,
            phi,
            theta_max=theta_max,
            strength=args.strength,
            vortex_charge=args.vortex_charge,
            apodization=args.apodization,
        )

        direct, direct_time, direct_times = median_time(
            lambda: direct_debye_wolf(
                pupil,
                theta,
                theta_weights,
                phi,
                rho,
                psi,
                z,
                k=k,
            ),
            args.repeats,
        )
        direct_intensity = np.abs(direct) ** 2
        direct_sequence, direct_sequence_time, direct_sequence_times = median_time(
            lambda: evaluate_direct_sequence(
                pupil_sequence,
                theta,
                theta_weights,
                phi,
                rho,
                psi,
                z,
                k=k,
            ),
            args.repeats,
        )
        direct_sequence_intensity = np.abs(direct_sequence) ** 2
        if finufft_plan is None:
            finufft = None
            finufft_time = None
            finufft_times = []
            finufft_intensity = None
            finufft_sequence = None
            finufft_sequence_time = None
            finufft_sequence_times = []
            finufft_sequence_intensity = None
        else:
            finufft, finufft_time, finufft_times = median_time(
                lambda: finufft_plan.evaluate(pupil, eps=args.finufft_eps),
                args.repeats,
            )
            finufft_intensity = np.abs(finufft) ** 2
            finufft_sequence, finufft_sequence_time, finufft_sequence_times = (
                median_time(
                    lambda: finufft_plan.evaluate_many(
                        pupil_sequence,
                        eps=args.finufft_eps,
                    ),
                    args.repeats,
                )
            )
            finufft_sequence_intensity = np.abs(finufft_sequence) ** 2

        for h_cutoff in h_cutoffs:
            harmonic_result, harmonic_time, harmonic_times = median_time(
                lambda: harmonic_debye_wolf(
                    pupil,
                    theta,
                    theta_weights,
                    rho,
                    psi,
                    z,
                    k=k,
                    h_cutoff=h_cutoff,
                ),
                args.repeats,
            )
            harmonic, used_modes = harmonic_result
            harmonic_intensity = np.abs(harmonic) ** 2
            plan, plan_build_time, plan_build_times = median_time(
                lambda: PreparedHarmonicDebyeWolfPlan.build(
                    args.nphi,
                    theta,
                    theta_weights,
                    rho,
                    psi,
                    z,
                    k=k,
                    h_cutoff=h_cutoff,
                ),
                args.repeats,
            )
            prepared, prepared_time, prepared_times = median_time(
                lambda: plan.evaluate(pupil),
                args.repeats,
            )
            prepared_intensity = np.abs(prepared) ** 2
            prepared_sequence, prepared_sequence_time, prepared_sequence_times = (
                median_time(
                    lambda: evaluate_plan_sequence(plan, pupil_sequence),
                    args.repeats,
                )
            )
            prepared_sequence_intensity = np.abs(prepared_sequence) ** 2
            separable_plan, separable_build_time, separable_build_times = median_time(
                lambda: PreparedSeparableHarmonicDebyeWolfPlan.build(
                    args.nphi,
                    theta,
                    theta_weights,
                    rho_axis,
                    psi_axis,
                    z_axis,
                    k=k,
                    h_cutoff=h_cutoff,
                    backend=args.harmonic_backend,
                    cpp_threads=args.cpp_threads,
                ),
                args.repeats,
            )
            separable, separable_time, separable_times = median_time(
                lambda: separable_plan.evaluate(pupil),
                args.repeats,
            )
            separable_intensity = np.abs(separable) ** 2
            separable_sequence, separable_sequence_time, separable_sequence_times = (
                median_time(
                    lambda: evaluate_plan_sequence(separable_plan, pupil_sequence),
                    args.repeats,
                )
            )
            separable_sequence_intensity = np.abs(separable_sequence) ** 2
            rows.append(
                {
                    "case": case,
                    "h_cutoff": cutoff_label(h_cutoff),
                    "used_modes": used_modes,
                    "ntheta": args.ntheta,
                    "nphi": args.nphi,
                    "targets": rho.size,
                    "nrho": args.nrho,
                    "npsi": args.npsi,
                    "nz": args.nz,
                    "na": args.na,
                    "n_medium": args.n_medium,
                    "wavelength": args.wavelength,
                    "rho_max": args.rho_max,
                    "z_max": args.z_max,
                    "strength": args.strength,
                    "sweeps": args.sweeps,
                    "apodization": args.apodization,
                    "basis_mib": plan.basis_mib,
                    "separable_basis_mib": separable_plan.basis_mib,
                    "harmonic_backend": args.harmonic_backend,
                    "cpp_threads": args.cpp_threads,
                    "finufft_eps": None if finufft_plan is None else args.finufft_eps,
                    "finufft_sources": None
                    if finufft_plan is None
                    else finufft_plan.sources,
                    "finufft_targets": None
                    if finufft_plan is None
                    else finufft_plan.targets,
                    "finufft_coord_scale": None
                    if finufft_plan is None
                    else finufft_plan.coord_scale,
                    "finufft_coordinate_mib": None
                    if finufft_plan is None
                    else finufft_plan.coordinate_mib,
                    "finufft_coordinate_build_time_s": finufft_build_time,
                    "direct_time_s": direct_time,
                    "harmonic_time_s": harmonic_time,
                    "prepared_build_time_s": plan_build_time,
                    "prepared_time_s": prepared_time,
                    "separable_build_time_s": separable_build_time,
                    "separable_time_s": separable_time,
                    "finufft_time_s": finufft_time,
                    "direct_hot_loop_time_s": direct_sequence_time,
                    "direct_hot_per_mask_s": direct_sequence_time / args.sweeps,
                    "prepared_hot_loop_time_s": prepared_sequence_time,
                    "prepared_hot_per_mask_s": prepared_sequence_time / args.sweeps,
                    "separable_hot_loop_time_s": separable_sequence_time,
                    "separable_hot_per_mask_s": separable_sequence_time / args.sweeps,
                    "finufft_hot_loop_time_s": finufft_sequence_time,
                    "finufft_hot_per_mask_s": None
                    if finufft_sequence_time is None
                    else finufft_sequence_time / args.sweeps,
                    "speedup_vs_direct": direct_time / harmonic_time
                    if harmonic_time > 0.0
                    else None,
                    "prepared_speedup_vs_direct": direct_time / prepared_time
                    if prepared_time > 0.0
                    else None,
                    "hot_speedup_vs_direct_excl_build": direct_sequence_time
                    / prepared_sequence_time
                    if prepared_sequence_time > 0.0
                    else None,
                    "hot_speedup_vs_direct_incl_build": direct_sequence_time
                    / (prepared_sequence_time + plan_build_time)
                    if prepared_sequence_time + plan_build_time > 0.0
                    else None,
                    "finufft_speedup_vs_direct": None
                    if finufft_time in (None, 0.0)
                    else direct_time / finufft_time,
                    "finufft_hot_speedup_vs_direct": None
                    if finufft_sequence_time in (None, 0.0)
                    else direct_sequence_time / finufft_sequence_time,
                    "prepared_speedup_vs_finufft": None
                    if finufft_time in (None, 0.0) or prepared_time <= 0.0
                    else finufft_time / prepared_time,
                    "separable_speedup_vs_finufft": None
                    if finufft_time in (None, 0.0) or separable_time <= 0.0
                    else finufft_time / separable_time,
                    "hot_prepared_speedup_vs_finufft_excl_build": None
                    if finufft_sequence_time in (None, 0.0)
                    or prepared_sequence_time <= 0.0
                    else finufft_sequence_time / prepared_sequence_time,
                    "hot_separable_speedup_vs_finufft_excl_build": None
                    if finufft_sequence_time in (None, 0.0)
                    or separable_sequence_time <= 0.0
                    else finufft_sequence_time / separable_sequence_time,
                    "hot_prepared_speedup_vs_finufft_incl_build": None
                    if finufft_sequence_time in (None, 0.0)
                    or prepared_sequence_time + plan_build_time <= 0.0
                    else finufft_sequence_time
                    / (prepared_sequence_time + plan_build_time),
                    "hot_separable_speedup_vs_finufft_incl_build": None
                    if finufft_sequence_time in (None, 0.0)
                    or separable_sequence_time + separable_build_time <= 0.0
                    else finufft_sequence_time
                    / (separable_sequence_time + separable_build_time),
                    "field_relative_l2": relative_l2(harmonic, direct),
                    "field_max_abs_over_ref": max_abs_over_ref(harmonic, direct),
                    "intensity_relative_l2": relative_l2(
                        harmonic_intensity,
                        direct_intensity,
                    ),
                    "intensity_max_abs_over_ref": max_abs_over_ref(
                        harmonic_intensity,
                        direct_intensity,
                    ),
                    "prepared_field_relative_l2": relative_l2(prepared, direct),
                    "prepared_field_max_abs_over_ref": max_abs_over_ref(
                        prepared,
                        direct,
                    ),
                    "prepared_intensity_relative_l2": relative_l2(
                        prepared_intensity,
                        direct_intensity,
                    ),
                    "prepared_intensity_max_abs_over_ref": max_abs_over_ref(
                        prepared_intensity,
                        direct_intensity,
                    ),
                    "separable_field_relative_l2": relative_l2(separable, direct),
                    "separable_field_max_abs_over_ref": max_abs_over_ref(
                        separable,
                        direct,
                    ),
                    "separable_intensity_relative_l2": relative_l2(
                        separable_intensity,
                        direct_intensity,
                    ),
                    "separable_intensity_max_abs_over_ref": max_abs_over_ref(
                        separable_intensity,
                        direct_intensity,
                    ),
                    "hot_field_relative_l2": relative_l2(
                        prepared_sequence,
                        direct_sequence,
                    ),
                    "hot_field_max_abs_over_ref": max_abs_over_ref(
                        prepared_sequence,
                        direct_sequence,
                    ),
                    "hot_intensity_relative_l2": relative_l2(
                        prepared_sequence_intensity,
                        direct_sequence_intensity,
                    ),
                    "hot_intensity_max_abs_over_ref": max_abs_over_ref(
                        prepared_sequence_intensity,
                        direct_sequence_intensity,
                    ),
                    "hot_separable_field_relative_l2": relative_l2(
                        separable_sequence,
                        direct_sequence,
                    ),
                    "hot_separable_field_max_abs_over_ref": max_abs_over_ref(
                        separable_sequence,
                        direct_sequence,
                    ),
                    "hot_separable_intensity_relative_l2": relative_l2(
                        separable_sequence_intensity,
                        direct_sequence_intensity,
                    ),
                    "hot_separable_intensity_max_abs_over_ref": max_abs_over_ref(
                        separable_sequence_intensity,
                        direct_sequence_intensity,
                    ),
                    "finufft_field_relative_l2": None
                    if finufft is None
                    else relative_l2(finufft, direct),
                    "finufft_field_max_abs_over_ref": None
                    if finufft is None
                    else max_abs_over_ref(finufft, direct),
                    "finufft_intensity_relative_l2": None
                    if finufft_intensity is None
                    else relative_l2(finufft_intensity, direct_intensity),
                    "finufft_intensity_max_abs_over_ref": None
                    if finufft_intensity is None
                    else max_abs_over_ref(finufft_intensity, direct_intensity),
                    "finufft_hot_field_relative_l2": None
                    if finufft_sequence is None
                    else relative_l2(finufft_sequence, direct_sequence),
                    "finufft_hot_field_max_abs_over_ref": None
                    if finufft_sequence is None
                    else max_abs_over_ref(finufft_sequence, direct_sequence),
                    "finufft_hot_intensity_relative_l2": None
                    if finufft_sequence_intensity is None
                    else relative_l2(
                        finufft_sequence_intensity,
                        direct_sequence_intensity,
                    ),
                    "finufft_hot_intensity_max_abs_over_ref": None
                    if finufft_sequence_intensity is None
                    else max_abs_over_ref(
                        finufft_sequence_intensity,
                        direct_sequence_intensity,
                    ),
                    "prepared_vs_finufft_field_relative_l2": None
                    if finufft is None
                    else relative_l2(prepared, finufft),
                    "separable_vs_finufft_field_relative_l2": None
                    if finufft is None
                    else relative_l2(separable, finufft),
                    "hot_prepared_vs_finufft_field_relative_l2": None
                    if finufft_sequence is None
                    else relative_l2(prepared_sequence, finufft_sequence),
                    "hot_separable_vs_finufft_field_relative_l2": None
                    if finufft_sequence is None
                    else relative_l2(separable_sequence, finufft_sequence),
                    "direct_times_s": direct_times,
                    "harmonic_times_s": harmonic_times,
                    "prepared_build_times_s": plan_build_times,
                    "prepared_times_s": prepared_times,
                    "separable_build_times_s": separable_build_times,
                    "separable_times_s": separable_times,
                    "finufft_coordinate_build_times_s": finufft_build_times,
                    "finufft_times_s": finufft_times,
                    "direct_hot_loop_times_s": direct_sequence_times,
                    "prepared_hot_loop_times_s": prepared_sequence_times,
                    "separable_hot_loop_times_s": separable_sequence_times,
                    "finufft_hot_loop_times_s": finufft_sequence_times,
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "cases": cases,
            "h_cutoffs": [cutoff_label(cutoff) for cutoff in h_cutoffs],
            "ntheta": args.ntheta,
            "nphi": args.nphi,
            "nrho": args.nrho,
            "npsi": args.npsi,
            "nz": args.nz,
            "rho_max": args.rho_max,
            "z_max": args.z_max,
            "wavelength": args.wavelength,
            "na": args.na,
            "n_medium": args.n_medium,
            "theta_max": theta_max,
            "strength": args.strength,
            "vortex_charge": args.vortex_charge,
            "apodization": args.apodization,
            "repeats": args.repeats,
            "sweeps": args.sweeps,
            "finufft_eps": args.finufft_eps,
            "skip_finufft": args.skip_finufft,
            "harmonic_backend": args.harmonic_backend,
            "cpp_threads": args.cpp_threads,
        },
        "rows": rows,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(args.csv, rows)

    print(f"wrote {args.out}")
    print(f"wrote {args.csv}")
    for row in rows:
        print(
            "{case:12s} h={h_cutoff:>4s} modes={used_modes:3d} "
            "field_l2={field_relative_l2:.3e} "
            "prep_l2={prepared_field_relative_l2:.3e} "
            "hot_l2={hot_field_relative_l2:.3e} "
            "finufft_l2={finufft_field_relative_l2} "
            "direct={direct_time_s:.4f}s harmonic={harmonic_time_s:.4f}s "
            "prepared={prepared_time_s:.4f}s "
            "separable={separable_time_s:.4f}s "
            "finufft={finufft_time_s}s "
            "hot_per_mask={prepared_hot_per_mask_s:.4f}s "
            "sep_hot_vs_finufft={hot_separable_speedup_vs_finufft_excl_build}x"
            .format(**row)
        )


if __name__ == "__main__":
    main()
