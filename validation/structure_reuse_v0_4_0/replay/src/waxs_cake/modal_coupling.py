"""Finite-band angular couplings on axisymmetric target orbits."""

from __future__ import annotations

import numpy as np


class PreparedAngularModeCoupling:
    """Prepare a pointwise linear coupling as a small mode-shift operator.

    ``coupling`` has shape ``(n_orbit, n_phi, n_output, n_input)``.  The
    prepared coefficients are Fourier-series coefficients, so applying them
    to a standard DFT spectrum or to compact Fourier-series coefficients uses
    the same circular convolution.  Only the final synthesis normalization
    differs between those two representations.
    """

    def __init__(self, coupling: np.ndarray, *, tolerance: float = 1e-12) -> None:
        values = np.asarray(coupling, dtype=np.complex128)
        if values.ndim != 4 or values.shape[0] == 0 or values.shape[1] < 3:
            raise ValueError(
                "coupling must have shape (n_orbit, n_phi, n_output, n_input)"
            )
        if values.shape[2] == 0 or values.shape[3] == 0:
            raise ValueError("coupling input and output dimensions must be nonzero")
        if not np.all(np.isfinite(values)):
            raise ValueError("coupling must contain only finite values")
        tolerance = float(tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive")

        n_phi = values.shape[1]
        spectrum = np.fft.fft(values, axis=1) / n_phi
        norms = np.sqrt(np.sum(np.abs(spectrum) ** 2, axis=(0, 2, 3)))
        scale = float(norms.max(initial=0.0))
        signed = np.rint(np.fft.fftfreq(n_phi) * n_phi).astype(np.int64)
        active = norms > tolerance * scale if scale > 0.0 else np.zeros(n_phi, bool)

        self.n_orbit = int(values.shape[0])
        self.n_phi = int(n_phi)
        self.n_output = int(values.shape[2])
        self.n_input = int(values.shape[3])
        self.tolerance = tolerance
        self.shifts = np.ascontiguousarray(signed[active])
        self.coefficients = np.ascontiguousarray(
            np.stack(
                [spectrum[:, shift % n_phi] for shift in self.shifts], axis=1
            )
            if np.any(active)
            else np.empty(
                (self.n_orbit, 0, self.n_output, self.n_input),
                dtype=np.complex128,
            )
        )
        self.shifts.setflags(write=False)
        self.coefficients.setflags(write=False)

    @classmethod
    def from_mode_coefficients(
        cls,
        coefficients: np.ndarray,
        shifts: np.ndarray,
        *,
        n_phi: int,
        tolerance: float = 1e-12,
    ) -> "PreparedAngularModeCoupling":
        """Prepare directly from finite Fourier-series coefficients.

        ``coefficients`` has shape ``(n_orbit, n_shift, n_output, n_input)``
        and represents ``sum_s coefficients[:, s] * exp(1j*s*theta)`` on a
        detector grid whose angular index has length ``n_phi``.  Shifts that
        alias to the same DFT bin are combined before insignificant bins are
        discarded.  This constructor avoids materializing a full pointwise
        coupling merely to rediscover a known finite angular bandwidth.
        """

        values = np.asarray(coefficients, dtype=np.complex128)
        shift_values = np.asarray(shifts, dtype=np.int64)
        n_phi = int(n_phi)
        tolerance = float(tolerance)
        if values.ndim != 4 or values.shape[0] == 0:
            raise ValueError(
                "coefficients must have shape "
                "(n_orbit, n_shift, n_output, n_input)"
            )
        if values.shape[1] != shift_values.size or shift_values.ndim != 1:
            raise ValueError("shifts must have shape (n_shift,)")
        if values.shape[2] == 0 or values.shape[3] == 0:
            raise ValueError("coupling input and output dimensions must be nonzero")
        if n_phi < 3:
            raise ValueError("n_phi must be at least 3")
        if not np.all(np.isfinite(values)):
            raise ValueError("coefficients must contain only finite values")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive")

        residues = shift_values % n_phi
        unique_residues = np.unique(residues)
        combined = np.stack(
            [np.sum(values[:, residues == residue], axis=1) for residue in unique_residues],
            axis=1,
        )
        norms = np.sqrt(np.sum(np.abs(combined) ** 2, axis=(0, 2, 3)))
        scale = float(norms.max(initial=0.0))
        active = norms > tolerance * scale if scale > 0.0 else np.zeros(norms.size, bool)
        signed_bins = np.rint(np.fft.fftfreq(n_phi) * n_phi).astype(np.int64)

        prepared = cls.__new__(cls)
        prepared.n_orbit = int(values.shape[0])
        prepared.n_phi = n_phi
        prepared.n_output = int(values.shape[2])
        prepared.n_input = int(values.shape[3])
        prepared.tolerance = tolerance
        prepared.shifts = np.ascontiguousarray(signed_bins[unique_residues[active]])
        prepared.coefficients = np.ascontiguousarray(combined[:, active])
        prepared.shifts.setflags(write=False)
        prepared.coefficients.setflags(write=False)
        return prepared

    @classmethod
    def from_uniform_samples(
        cls,
        coupling: np.ndarray,
        *,
        n_phi: int,
        tolerance: float = 1e-12,
    ) -> "PreparedAngularModeCoupling":
        """Prepare a known finite-band coupling from a minimal odd sample grid.

        The samples must span one period uniformly, starting at the same
        angular origin as the target detector grid.  For a coupling known to
        have support ``|s| <= S``, ``2*S + 1`` samples recover its coefficients
        exactly (up to floating-point roundoff) without allocating the full
        ``n_orbit * n_phi`` pointwise tensor.
        """

        values = np.asarray(coupling, dtype=np.complex128)
        if values.ndim != 4 or values.shape[1] < 3 or values.shape[1] % 2 == 0:
            raise ValueError(
                "coupling must have shape "
                "(n_orbit, odd_n_samples, n_output, n_input)"
            )
        n_samples = values.shape[1]
        coefficients = np.fft.fft(values, axis=1) / n_samples
        shifts = np.rint(np.fft.fftfreq(n_samples) * n_samples).astype(np.int64)
        return cls.from_mode_coefficients(
            coefficients,
            shifts,
            n_phi=n_phi,
            tolerance=tolerance,
        )

    @property
    def prepared_bytes(self) -> int:
        return int(self.shifts.nbytes + self.coefficients.nbytes)

    def apply_spectrum(self, input_spectrum: np.ndarray) -> np.ndarray:
        """Apply to standard DFT spectra with shape ``(input, orbit, phi)``."""

        values = np.asarray(input_spectrum, dtype=np.complex128)
        expected = (self.n_input, self.n_orbit, self.n_phi)
        if values.shape != expected:
            raise ValueError(f"input_spectrum must have shape {expected}")
        output = np.zeros(
            (self.n_output, self.n_orbit, self.n_phi), dtype=np.complex128
        )
        for shift_index, shift in enumerate(self.shifts):
            output += np.einsum(
                "roc,crp->orp",
                self.coefficients[:, shift_index],
                np.roll(values, int(shift), axis=-1),
                optimize=True,
            )
        return output

    def apply_compact(
        self,
        coefficients: np.ndarray,
        modes: np.ndarray,
    ) -> np.ndarray:
        """Apply to compact Fourier-series coefficients and return full modes."""

        values = np.asarray(coefficients, dtype=np.complex128)
        mode_values = np.asarray(modes, dtype=np.int64)
        if mode_values.ndim != 1 or values.shape != (
            self.n_input,
            self.n_orbit,
            mode_values.size,
        ):
            raise ValueError(
                "coefficients must have shape (n_input, n_orbit, n_modes)"
            )
        if np.unique(mode_values % self.n_phi).size != mode_values.size:
            raise ValueError("modes must map to distinct angular spectrum bins")
        output = np.zeros(
            (self.n_output, self.n_orbit, self.n_phi), dtype=np.complex128
        )
        for shift_index, shift in enumerate(self.shifts):
            destination = (mode_values + int(shift)) % self.n_phi
            output[..., destination] += np.einsum(
                "roc,crm->orm",
                self.coefficients[:, shift_index],
                values,
                optimize=True,
            )
        return output

    def synthesize_spectrum(self, input_spectrum: np.ndarray) -> np.ndarray:
        """Apply a standard DFT spectrum and synthesize ``(orbit, phi, output)``."""

        output = np.fft.ifft(self.apply_spectrum(input_spectrum), axis=-1)
        return np.moveaxis(output, 0, -1)

    def synthesize_compact(
        self,
        coefficients: np.ndarray,
        modes: np.ndarray,
    ) -> np.ndarray:
        """Apply compact Fourier-series coefficients and synthesize point values."""

        output = np.fft.ifft(
            self.apply_compact(coefficients, modes), axis=-1
        ) * self.n_phi
        return np.moveaxis(output, 0, -1)

    def synthesize_compact_spectrum(
        self,
        coefficients: np.ndarray,
        modes: np.ndarray,
    ) -> np.ndarray:
        """Apply compact standard-DFT bins and synthesize point values."""

        output = np.fft.ifft(self.apply_compact(coefficients, modes), axis=-1)
        return np.moveaxis(output, 0, -1)
