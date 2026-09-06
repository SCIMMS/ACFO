"""Torch/CUDA execution plan for the physical ACFO-aIDT operator."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from .aidt_acfo import PreparedAidtAcfoOperator


def _block_diagonal_kernel_csr(
    kernel_fft: np.ndarray,
    *,
    relative_threshold: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, float | int],
]:
    """Pack ``kernel_fft[q, r, h]`` as an h-block-diagonal CSR matrix."""

    kernel = np.asarray(kernel_fft)
    if kernel.ndim != 3:
        raise ValueError("kernel_fft must have shape (q, r, h)")
    n_q, n_r, n_h = (int(value) for value in kernel.shape)
    relative_threshold = float(relative_threshold)
    if not 0.0 < relative_threshold < 1.0:
        raise ValueError(
            "relative_threshold must lie strictly between zero and one"
        )
    absolute_threshold = relative_threshold * float(n_h)
    row_counts = np.empty((n_h, n_q), dtype=np.int32)
    nnz = 0
    discarded_energy = 0.0
    total_energy = 0.0
    for h_index in range(n_h):
        block = kernel[:, :, h_index]
        magnitude = np.abs(block)
        active = magnitude >= absolute_threshold
        counts = np.count_nonzero(active, axis=1).astype(np.int32, copy=False)
        row_counts[h_index] = counts
        nnz += int(np.sum(counts, dtype=np.int64))
        squared = magnitude.astype(np.float64, copy=False) ** 2
        total_energy += float(np.sum(squared))
        discarded_energy += float(np.sum(squared[~active]))

    crow = np.empty(n_h * n_q + 1, dtype=np.int32)
    crow[0] = 0
    np.cumsum(row_counts.reshape(-1), dtype=np.int64, out=crow[1:])
    if int(crow[-1]) != nnz:
        raise RuntimeError("CSR row pointer does not match packed nnz")
    columns = np.empty(nnz, dtype=np.int32)
    values = np.empty(nnz, dtype=kernel.dtype)
    offset = 0
    for h_index in range(n_h):
        block = kernel[:, :, h_index]
        active = np.abs(block) >= absolute_threshold
        rows, local_columns = np.nonzero(active)
        count = int(local_columns.size)
        if count:
            columns[offset : offset + count] = (
                h_index * n_r + local_columns
            )
            values[offset : offset + count] = block[rows, local_columns]
        offset += count
    if offset != nnz:
        raise RuntimeError("CSR packing count changed between passes")
    stats: dict[str, float | int] = {
        "nnz": nnz,
        "dense_entries": int(n_h * n_q * n_r),
        "active_fraction": float(nnz / max(n_h * n_q * n_r, 1)),
        "relative_threshold": relative_threshold,
        "absolute_threshold": absolute_threshold,
        "relative_frobenius_tail": float(
            np.sqrt(discarded_energy / max(total_energy, np.finfo(float).tiny))
        ),
    }
    # The transpose view is CSC.  cuSPARSE handles this path substantially
    # more slowly for the present complex tall-RHS multiply, so prepare a
    # second CSR layout for the Euclidean adjoint exactly as the ODT backend
    # prepares separate forward and adjoint packed paths.
    from scipy.sparse import csr_matrix

    forward_cpu = csr_matrix(
        (values, columns, crow),
        shape=(n_h * n_q, n_h * n_r),
    )
    adjoint_cpu = forward_cpu.getH().tocsr()
    adjoint_crow = np.asarray(adjoint_cpu.indptr, dtype=np.int32)
    adjoint_columns = np.asarray(adjoint_cpu.indices, dtype=np.int32)
    adjoint_values = np.asarray(adjoint_cpu.data, dtype=kernel.dtype)
    return (
        crow,
        columns,
        values,
        adjoint_crow,
        adjoint_columns,
        adjoint_values,
        stats,
    )


def _packed_transfer_branches(
    ptf: np.ndarray,
    atf: np.ndarray,
    *,
    relative_threshold: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, float | int],
]:
    """Pack the exact two shifted-pupil branches behind PTF and ATF.

    With ``E1`` and ``E2`` denoting the forward and conjugate shifted-pupil
    branches, the public transfer functions satisfy

    ``PTF = i(E1-E2)`` and ``ATF = -(E1+E2)``.

    Each branch has an illumination/q/phi support that is independent of
    depth, so only active rows and their axial vectors are retained.
    """

    ptf_array = np.asarray(ptf)
    atf_array = np.asarray(atf)
    if ptf_array.shape != atf_array.shape or ptf_array.ndim != 4:
        raise ValueError("ptf and atf must have matching (s, q, phi, z) shapes")
    relative_threshold = float(relative_threshold)
    if not 0.0 <= relative_threshold < 1.0:
        raise ValueError(
            "transfer relative_threshold must lie in [0, 1)"
        )
    n_illumination, n_q, n_phi, n_z = (
        int(value) for value in ptf_array.shape
    )
    n_qphi = n_q * n_phi
    maximum = max(
        float(np.max(np.abs(ptf_array))),
        float(np.max(np.abs(atf_array))),
        np.finfo(float).tiny,
    )
    absolute_threshold = relative_threshold * maximum

    packed: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    active_counts = []
    discarded_energy = 0.0
    total_energy = 0.0
    for branch_index in range(2):
        global_parts: list[np.ndarray] = []
        qphi_parts: list[np.ndarray] = []
        value_parts: list[np.ndarray] = []
        for illumination in range(n_illumination):
            if branch_index == 0:
                branch = (
                    -atf_array[illumination]
                    - 1j * ptf_array[illumination]
                ) * 0.5
            else:
                branch = (
                    -atf_array[illumination]
                    + 1j * ptf_array[illumination]
                ) * 0.5
            rows = branch.astype(
                ptf_array.dtype,
                copy=False,
            ).reshape(n_qphi, n_z)
            row_maximum = np.max(np.abs(rows), axis=1)
            active = row_maximum > absolute_threshold
            local_rows = np.flatnonzero(active).astype(
                np.int64,
                copy=False,
            )
            global_parts.append(
                illumination * n_qphi + local_rows
            )
            qphi_parts.append(local_rows)
            value_parts.append(
                np.array(
                    rows[active],
                    dtype=ptf_array.dtype,
                    copy=True,
                )
            )
            if relative_threshold:
                total_energy += float(np.vdot(rows, rows).real)
                inactive_values = rows[~active]
                discarded_energy += float(
                    np.vdot(inactive_values, inactive_values).real
                )
            del branch, rows, row_maximum, active, local_rows
        global_rows = np.concatenate(global_parts)
        qphi_rows = np.concatenate(qphi_parts)
        values = np.concatenate(value_parts, axis=0)
        packed.append((global_rows, qphi_rows, values))
        active_counts.append(int(global_rows.size))
        del global_parts, qphi_parts, value_parts

    dense_transfer_bytes = int(ptf_array.nbytes + atf_array.nbytes)
    packed_value_bytes = int(sum(values.nbytes for _, _, values in packed))
    packed_index_bytes = int(
        sum(
            global_rows.nbytes + qphi_rows.nbytes
            for global_rows, qphi_rows, _ in packed
        )
    )
    stats: dict[str, float | int] = {
        "branch1_active_rows": active_counts[0],
        "branch2_active_rows": active_counts[1],
        "total_rows_per_branch": int(n_illumination * n_qphi),
        "branch1_active_fraction": float(
            active_counts[0] / max(n_illumination * n_qphi, 1)
        ),
        "branch2_active_fraction": float(
            active_counts[1] / max(n_illumination * n_qphi, 1)
        ),
        "relative_threshold": relative_threshold,
        "absolute_threshold": absolute_threshold,
        "relative_frobenius_tail": (
            float(
                np.sqrt(
                    discarded_energy
                    / max(total_energy, np.finfo(float).tiny)
                )
            )
            if relative_threshold
            else 0.0
        ),
        "dense_transfer_bytes": dense_transfer_bytes,
        "packed_value_bytes": packed_value_bytes,
        "packed_index_bytes": packed_index_bytes,
        "packed_total_bytes": packed_value_bytes + packed_index_bytes,
        "packed_to_dense_storage_fraction": float(
            (packed_value_bytes + packed_index_bytes)
            / max(dense_transfer_bytes, 1)
        ),
    }
    return (
        packed[0][0],
        packed[0][1],
        packed[0][2],
        packed[1][0],
        packed[1][1],
        packed[1][2],
        stats,
    )


class TorchPreparedAidtAcfoOperator:
    """GPU-resident counterpart of :class:`PreparedAidtAcfoOperator`.

    ``torch`` is supplied by the caller so importing the NumPy package does not
    make PyTorch a mandatory dependency.
    """

    def __init__(
        self,
        operator: PreparedAidtAcfoOperator,
        *,
        torch: Any,
        device: Any,
        contraction_backend: str = "bmm",
        kernel_relative_threshold: float = 0.0,
        transfer_backend: str = "dense",
        transfer_relative_threshold: float = 0.0,
    ) -> None:
        self.torch = torch
        self.device = device
        self.numpy_operator = operator
        self.complex_dtype = (
            torch.complex64
            if operator.complex_dtype == np.dtype(np.complex64)
            else torch.complex128
        )
        self.real_dtype = (
            torch.float32 if self.complex_dtype == torch.complex64 else torch.float64
        )
        if contraction_backend not in {"bmm", "einsum", "csr"}:
            raise ValueError(
                "contraction_backend must be bmm, einsum or csr"
            )
        self.contraction_backend = contraction_backend
        self.kernel_relative_threshold = float(kernel_relative_threshold)
        if transfer_backend not in {"dense", "branch_packed"}:
            raise ValueError(
                "transfer_backend must be dense or branch_packed"
            )
        self.transfer_backend = transfer_backend
        self.transfer_relative_threshold = float(
            transfer_relative_threshold
        )
        self.kernel_fft = None
        self.kernel_batched = None
        self.kernel_sparse = None
        self.kernel_sparse_adjoint = None
        self.kernel_sparse_stats: dict[str, float | int] = {
            "nnz": int(operator.n_phi * operator.n_frequency * operator.n_r),
            "dense_entries": int(
                operator.n_phi * operator.n_frequency * operator.n_r
            ),
            "active_fraction": 1.0,
            "relative_threshold": 0.0,
            "absolute_threshold": 0.0,
            "relative_frobenius_tail": 0.0,
        }
        if contraction_backend == "csr":
            (
                crow,
                columns,
                values,
                adjoint_crow,
                adjoint_columns,
                adjoint_values,
                stats,
            ) = _block_diagonal_kernel_csr(
                operator.lateral_operator.kernel_fft,
                relative_threshold=self.kernel_relative_threshold,
            )
            sparse_checks_were_enabled = (
                torch.sparse.check_sparse_tensor_invariants.is_enabled()
            )
            torch.sparse.check_sparse_tensor_invariants.enable()
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="Sparse CSR tensor support is in beta state.*",
                        category=UserWarning,
                    )
                    self.kernel_sparse = torch.sparse_csr_tensor(
                        torch.as_tensor(
                            crow,
                            dtype=torch.int32,
                            device=device,
                        ),
                        torch.as_tensor(
                            columns,
                            dtype=torch.int32,
                            device=device,
                        ),
                        torch.as_tensor(
                            values,
                            dtype=self.complex_dtype,
                            device=device,
                        ),
                        size=(
                            operator.n_phi * operator.n_frequency,
                            operator.n_phi * operator.n_r,
                        ),
                        dtype=self.complex_dtype,
                        device=device,
                    )
                    self.kernel_sparse_adjoint = torch.sparse_csr_tensor(
                        torch.as_tensor(
                            adjoint_crow,
                            dtype=torch.int32,
                            device=device,
                        ),
                        torch.as_tensor(
                            adjoint_columns,
                            dtype=torch.int32,
                            device=device,
                        ),
                        torch.as_tensor(
                            adjoint_values,
                            dtype=self.complex_dtype,
                            device=device,
                        ),
                        size=(
                            operator.n_phi * operator.n_r,
                            operator.n_phi * operator.n_frequency,
                        ),
                        dtype=self.complex_dtype,
                        device=device,
                    )
            finally:
                if not sparse_checks_were_enabled:
                    torch.sparse.check_sparse_tensor_invariants.disable()
            self.kernel_sparse_stats = stats
        else:
            if self.kernel_relative_threshold != 0.0:
                raise ValueError(
                    "kernel_relative_threshold is only valid for csr"
                )
            self.kernel_fft = torch.as_tensor(
                np.array(operator.lateral_operator.kernel_fft, copy=True),
                dtype=self.complex_dtype,
                device=device,
            )
            # A stride-only harmonic-major view allows cuBLAS batched matrix
            # products without storing a second copy of the prepared kernel.
            self.kernel_batched = self.kernel_fft.permute(2, 0, 1)
        self.cell_area = torch.as_tensor(
            np.array(operator.cell_area_um2, copy=True),
            dtype=self.real_dtype,
            device=device,
        )
        self.ptf = None
        self.atf = None
        self.branch1_global_rows = None
        self.branch1_qphi_rows = None
        self.branch1_values = None
        self.branch2_global_rows = None
        self.branch2_qphi_rows = None
        self.branch2_values = None
        self.transfer_sparse_stats: dict[str, float | int] = {
            "relative_threshold": 0.0,
            "relative_frobenius_tail": 0.0,
            "dense_transfer_bytes": int(
                operator.ptf.nbytes + operator.atf.nbytes
            ),
            "packed_total_bytes": int(
                operator.ptf.nbytes + operator.atf.nbytes
            ),
            "packed_to_dense_storage_fraction": 1.0,
        }
        if transfer_backend == "dense":
            if self.transfer_relative_threshold != 0.0:
                raise ValueError(
                    "transfer_relative_threshold is only valid for "
                    "branch_packed"
                )
            self.ptf = torch.as_tensor(
                np.array(operator.ptf, copy=True),
                dtype=self.complex_dtype,
                device=device,
            )
            self.atf = torch.as_tensor(
                np.array(operator.atf, copy=True),
                dtype=self.complex_dtype,
                device=device,
            )
        else:
            (
                branch1_global_rows,
                branch1_qphi_rows,
                branch1_values,
                branch2_global_rows,
                branch2_qphi_rows,
                branch2_values,
                transfer_stats,
            ) = _packed_transfer_branches(
                operator.ptf,
                operator.atf,
                relative_threshold=self.transfer_relative_threshold,
            )
            self.branch1_global_rows = torch.as_tensor(
                branch1_global_rows,
                dtype=torch.int64,
                device=device,
            )
            self.branch1_qphi_rows = torch.as_tensor(
                branch1_qphi_rows,
                dtype=torch.int64,
                device=device,
            )
            self.branch1_values = torch.as_tensor(
                branch1_values,
                dtype=self.complex_dtype,
                device=device,
            )
            self.branch2_global_rows = torch.as_tensor(
                branch2_global_rows,
                dtype=torch.int64,
                device=device,
            )
            self.branch2_qphi_rows = torch.as_tensor(
                branch2_qphi_rows,
                dtype=torch.int64,
                device=device,
            )
            self.branch2_values = torch.as_tensor(
                branch2_values,
                dtype=self.complex_dtype,
                device=device,
            )
            self.transfer_sparse_stats = transfer_stats
        self.n_r = operator.n_r
        self.n_phi = operator.n_phi
        self.n_z = operator.n_z
        self.n_frequency = operator.n_frequency
        self.n_illumination = operator.n_illumination
        self.object_shape = operator.object_shape
        self.data_shape = operator.data_shape

    @property
    def resident_bytes(self) -> int:
        tensors = [
            value
            for value in (
                self.cell_area,
                self.ptf,
                self.atf,
                self.branch1_global_rows,
                self.branch1_qphi_rows,
                self.branch1_values,
                self.branch2_global_rows,
                self.branch2_qphi_rows,
                self.branch2_values,
            )
            if value is not None
        ]
        result = int(
            sum(value.numel() * value.element_size() for value in tensors)
        )
        if self.kernel_sparse is not None:
            for sparse in (
                self.kernel_sparse,
                self.kernel_sparse_adjoint,
            ):
                result += int(
                    sparse.values().numel()
                    * sparse.values().element_size()
                    + sparse.col_indices().numel()
                    * sparse.col_indices().element_size()
                    + sparse.crow_indices().numel()
                    * sparse.crow_indices().element_size()
                )
        elif self.kernel_fft is not None:
            result += int(
                self.kernel_fft.numel() * self.kernel_fft.element_size()
            )
        return result

    def _object_tensor(self, values: Any, *, name: str) -> Any:
        tensor = torch_as_complex(
            self.torch,
            values,
            dtype=self.complex_dtype,
            device=self.device,
        )
        if tuple(tensor.shape) != self.object_shape:
            raise ValueError(f"{name} must have shape {self.object_shape}")
        return tensor

    def _data_tensor(self, values: Any) -> Any:
        tensor = torch_as_complex(
            self.torch,
            values,
            dtype=self.complex_dtype,
            device=self.device,
        )
        if tuple(tensor.shape) != self.data_shape:
            raise ValueError(f"data values must have shape {self.data_shape}")
        return tensor

    def lateral_forward(self, object_values: Any) -> Any:
        """Return negative-sign polar lateral Fourier samples."""

        values = self._object_tensor(object_values, name="object_values")
        coefficient_fft = self.torch.fft.fft(
            values * self.cell_area[:, :, None],
            dim=1,
        )
        if self.contraction_backend == "csr":
            coefficient_matrix = coefficient_fft.permute(1, 0, 2).reshape(
                self.n_phi * self.n_r,
                self.n_z,
            )
            data_fft = self.torch.sparse.mm(
                self.kernel_sparse,
                coefficient_matrix,
            ).reshape(
                self.n_phi,
                self.n_frequency,
                self.n_z,
            ).permute(1, 0, 2)
        elif self.contraction_backend == "bmm":
            data_fft = self.torch.bmm(
                self.kernel_batched,
                coefficient_fft.permute(1, 0, 2),
            ).permute(1, 0, 2)
        else:
            data_fft = self.torch.einsum(
                "qrh,rhz->qhz",
                self.kernel_fft,
                coefficient_fft,
            )
        positive = self.torch.fft.ifft(data_fft, dim=1)
        return self.torch.roll(positive, shifts=-(self.n_phi // 2), dims=1)

    def lateral_adjoint(self, spectrum_values: Any) -> Any:
        """Apply the Euclidean adjoint of :meth:`lateral_forward`."""

        spectrum = torch_as_complex(
            self.torch,
            spectrum_values,
            dtype=self.complex_dtype,
            device=self.device,
        )
        expected = (self.n_frequency, self.n_phi, self.n_z)
        if tuple(spectrum.shape) != expected:
            raise ValueError(f"spectrum_values must have shape {expected}")
        positive_data = self.torch.roll(
            spectrum,
            shifts=self.n_phi // 2,
            dims=1,
        )
        data_fft = self.torch.fft.fft(positive_data, dim=1)
        if self.contraction_backend == "csr":
            data_matrix = data_fft.permute(1, 0, 2).reshape(
                self.n_phi * self.n_frequency,
                self.n_z,
            )
            coefficient_fft = self.torch.sparse.mm(
                self.kernel_sparse_adjoint,
                data_matrix,
            ).reshape(
                self.n_phi,
                self.n_r,
                self.n_z,
            ).permute(1, 0, 2)
        elif self.contraction_backend == "bmm":
            coefficient_fft = self.torch.bmm(
                self.torch.conj(self.kernel_batched).transpose(1, 2),
                data_fft.permute(1, 0, 2),
            ).permute(1, 0, 2)
        else:
            coefficient_fft = self.torch.einsum(
                "qrh,qhz->rhz",
                self.torch.conj(self.kernel_fft),
                data_fft,
            )
        coefficients = self.torch.fft.ifft(coefficient_fft, dim=1)
        return coefficients * self.cell_area[:, :, None]

    def forward(self, potential_real: Any, potential_imag: Any) -> Any:
        real_spectrum = self.lateral_forward(
            self._object_tensor(potential_real, name="potential_real")
        )
        imag_spectrum = self.lateral_forward(
            self._object_tensor(potential_imag, name="potential_imag")
        )
        if self.transfer_backend == "branch_packed":
            real_rows = real_spectrum.reshape(
                self.n_frequency * self.n_phi,
                self.n_z,
            )
            imag_rows = imag_spectrum.reshape(
                self.n_frequency * self.n_phi,
                self.n_z,
            )
            output = self.torch.zeros(
                self.n_illumination * self.n_frequency * self.n_phi,
                dtype=self.complex_dtype,
                device=self.device,
            )
            branch1_source = (
                1j * real_rows - imag_rows
            ).index_select(0, self.branch1_qphi_rows)
            branch1_output = self.torch.sum(
                self.branch1_values * branch1_source,
                dim=1,
            )
            output.index_add_(
                0,
                self.branch1_global_rows,
                branch1_output,
            )
            del branch1_source, branch1_output
            branch2_source = (
                -1j * real_rows - imag_rows
            ).index_select(0, self.branch2_qphi_rows)
            branch2_output = self.torch.sum(
                self.branch2_values * branch2_source,
                dim=1,
            )
            output.index_add_(
                0,
                self.branch2_global_rows,
                branch2_output,
            )
            return output.reshape(self.data_shape)
        return self.torch.einsum(
            "sqpz,qpz->sqp",
            self.ptf,
            real_spectrum,
        ) + self.torch.einsum(
            "sqpz,qpz->sqp",
            self.atf,
            imag_spectrum,
        )

    def adjoint(self, data_values: Any) -> tuple[Any, Any]:
        data = self._data_tensor(data_values)
        if self.transfer_backend == "branch_packed":
            data_rows = data.reshape(-1)
            n_qphi = self.n_frequency * self.n_phi

            branch1_data = data_rows.index_select(
                0,
                self.branch1_global_rows,
            )
            branch1_weighted = (
                self.torch.conj(self.branch1_values)
                * branch1_data[:, None]
            )
            branch1_spectrum = self.torch.zeros(
                (n_qphi, self.n_z),
                dtype=self.complex_dtype,
                device=self.device,
            )
            branch1_spectrum.index_add_(
                0,
                self.branch1_qphi_rows,
                branch1_weighted,
            )
            del branch1_data, branch1_weighted

            branch2_data = data_rows.index_select(
                0,
                self.branch2_global_rows,
            )
            branch2_weighted = (
                self.torch.conj(self.branch2_values)
                * branch2_data[:, None]
            )
            branch2_spectrum = self.torch.zeros(
                (n_qphi, self.n_z),
                dtype=self.complex_dtype,
                device=self.device,
            )
            branch2_spectrum.index_add_(
                0,
                self.branch2_qphi_rows,
                branch2_weighted,
            )
            del branch2_data, branch2_weighted

            spectrum_real = (
                1j * (branch2_spectrum - branch1_spectrum)
            ).reshape(
                self.n_frequency,
                self.n_phi,
                self.n_z,
            )
            spectrum_imag = (
                -(branch1_spectrum + branch2_spectrum)
            ).reshape(
                self.n_frequency,
                self.n_phi,
                self.n_z,
            )
            return self.lateral_adjoint(
                spectrum_real
            ), self.lateral_adjoint(spectrum_imag)
        spectrum_real = self.torch.einsum(
            "sqpz,sqp->qpz",
            self.torch.conj(self.ptf),
            data,
        )
        spectrum_imag = self.torch.einsum(
            "sqpz,sqp->qpz",
            self.torch.conj(self.atf),
            data,
        )
        return self.lateral_adjoint(spectrum_real), self.lateral_adjoint(
            spectrum_imag
        )

    def real_adjoint(self, data_values: Any) -> tuple[Any, Any]:
        real_channel, imag_channel = self.adjoint(data_values)
        return self.torch.real(real_channel), self.torch.real(imag_channel)

    def dot_test(
        self,
        potential_real: Any,
        potential_imag: Any,
        data_values: Any,
    ) -> dict[str, float | complex]:
        real_channel = self._object_tensor(potential_real, name="potential_real")
        imag_channel = self._object_tensor(potential_imag, name="potential_imag")
        data = self._data_tensor(data_values)
        left_tensor = self.torch.vdot(
            self.forward(real_channel, imag_channel).reshape(-1),
            data.reshape(-1),
        )
        adjoint_real, adjoint_imag = self.adjoint(data)
        right_tensor = self.torch.vdot(
            real_channel.reshape(-1),
            adjoint_real.reshape(-1),
        ) + self.torch.vdot(
            imag_channel.reshape(-1),
            adjoint_imag.reshape(-1),
        )
        left = complex(left_tensor.detach().cpu().item())
        right = complex(right_tensor.detach().cpu().item())
        denominator = abs(left) + abs(right)
        error = (
            0.0
            if denominator == 0.0 and left == right
            else float(abs(left - right) / max(denominator, 1e-300))
        )
        return {"left": left, "right": right, "normalized_error": error}


def torch_as_complex(
    torch: Any,
    values: Any,
    *,
    dtype: Any,
    device: Any,
) -> Any:
    """Convert NumPy or Torch input without an avoidable device round trip."""

    if torch.is_tensor(values):
        return values.to(device=device, dtype=dtype)
    return torch.as_tensor(values, dtype=dtype, device=device)
