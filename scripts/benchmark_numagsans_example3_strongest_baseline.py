"""Strongest-eligible-baseline closure for NuMagSANS Example 3.

The old projected-type-3-only result is intentionally not overwritten.  This
prospective benchmark recognizes the public particle as a rigidly rotated
20^3 Cartesian lattice, accuracy-qualifies affine type-2, projected type-3,
and a dense periodic FFT pad frontier, performs an RTX-3090-only FFT screen,
then runs an independent 10-warmup/30-pair ABBA confirmation before loading
orientation 2.  Only after that confirmation is the full 800-orientation,
five-hierarchy comparator frozen.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_magnetic_sans_phase_b_gpu import (  # noqa: E402
    detector_q_grid,
    direct_fourier_channel_subset,
    numpy_cross_sections_flat,
    q_cloud_in_original_grid_frame,
    rotate_points_about_z,
    sync,
)
from scripts.benchmark_numagsans_example3_ensemble_dispatch import (  # noqa: E402
    OrientationSource,
    _archive_errors,
    _channels_from_torch_amplitudes,
    _hierarchy_errors,
    _hierarchy_outputs_torch,
    _primitive_torch,
    bootstrap_total_speedup,
    load_centers,
    md5,
    physical_scales,
    stream_archived_rings,
    validate_source_archive,
)
from waxs_cake.axisymmetric_manifold import AxisymmetricManifold  # noqa: E402
from waxs_cake.magnetic_sans import PreparedBandlimitedMagneticSansOperator  # noqa: E402
from waxs_cake.magnetic_sans_torch import (  # noqa: E402
    TorchFlatDetectorBandlimitedMagneticSansOperator,
)
from waxs_cake.numagsans_ensemble import (  # noqa: E402
    OUTPUT_NAMES,
    PRIMITIVE_OUTPUT_NAMES,
    affine_lattice_channels,
    affine_lattice_type2_contract,
    certify_affine_cartesian_lattice,
    compare_affine_lattice_certificates,
    cumulative_crossover,
    relative_l2,
)
from waxs_cake.voxel_fft_torch import (  # noqa: E402
    TorchPreparedPeriodicCoefficientFFT3D,
)


_DLL_HANDLES: list[Any] = []


def closure_decision(
    *,
    full_run: bool,
    qualification_pass: bool,
    archive_pass: bool,
    machine_pass: bool,
    speed_lower_95: float,
    required_speed_lower_95: float,
) -> dict[str, Any]:
    """Separate scientific-contract closure from the signed speed claim."""

    contract_pass = bool(
        full_run and qualification_pass and archive_pass and machine_pass
    )
    acfo_positive_claim_eligible = bool(
        contract_pass and speed_lower_95 >= required_speed_lower_95
    )
    if not full_run:
        verdict = "SMOKE_ONLY_NOT_PUBLICATION_EVIDENCE"
    elif not contract_pass:
        verdict = "FAIL_PRESERVE_STRONGEST_BASELINE_RESULT"
    elif acfo_positive_claim_eligible:
        verdict = "PASS_STRONGEST_BASELINE_CLOSURE"
    else:
        verdict = "PASS_STRONGEST_BASELINE_CLOSURE_NO_GO"
    return {
        "contract_pass": contract_pass,
        "acfo_positive_claim_eligible": acfo_positive_claim_eligible,
        "verdict": verdict,
    }


def _prepare_windows_cuda_dll_search() -> None:
    """Make the wheel-local cuFINUFFT and PyTorch CUDA DLLs discoverable."""

    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    for candidate in (
        ROOT / ".venv/Lib/site-packages/torch/lib",
        ROOT / ".venv/Lib/site-packages/cufinufft",
    ):
        if candidate.is_dir():
            _DLL_HANDLES.append(os.add_dll_directory(str(candidate)))


def _certificate_summary(certificate: dict[str, Any]) -> dict[str, Any]:
    return {
        "basis": np.asarray(certificate["basis"]).tolist(),
        "gram": np.asarray(certificate["gram"]).tolist(),
        "spacing": np.asarray(certificate["spacing"]).tolist(),
        "determinant": float(certificate["determinant"]),
        "integer_lower": np.asarray(certificate["integer_lower"]).tolist(),
        "integer_upper": np.asarray(certificate["integer_upper"]).tolist(),
        "shape": np.asarray(certificate["shape"]).tolist(),
        "origin": np.asarray(certificate["origin"]).tolist(),
        "center": np.asarray(certificate["center"]).tolist(),
        "active_sites": int(certificate["active_sites"]),
        "dense_sites": int(certificate["dense_sites"]),
        "maximum_integer_residual": float(
            certificate["maximum_integer_residual"]
        ),
        "maximum_coordinate_residual": float(
            certificate["maximum_coordinate_residual"]
        ),
        "maximum_normalized_gram_residual": float(
            certificate["maximum_normalized_gram_residual"]
        ),
    }


def _load_scaled(
    source: OrientationSource, index: int, scale: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points, nuclear, magnetization = source.load(index)
    if source.bundle is None:
        points = np.ascontiguousarray(points * float(scale))
    return points, nuclear, magnetization


def _weights(nuclear: np.ndarray, magnetization: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(
        np.column_stack([nuclear, magnetization]).T, dtype=np.float32
    )


def _error_row(
    actual: np.ndarray,
    direct: np.ndarray,
    q_hat: np.ndarray,
    polarization: np.ndarray,
) -> dict[str, Any]:
    reference_outputs = numpy_cross_sections_flat(direct, q_hat, polarization)
    actual_outputs = numpy_cross_sections_flat(actual, q_hat, polarization)
    output_errors = {
        name: relative_l2(actual_outputs[name], reference_outputs[name])
        for name in OUTPUT_NAMES
    }
    return {
        "amplitude_relative_l2": relative_l2(actual, direct),
        "output_relative_l2": output_errors,
        "worst_output_relative_l2": max(output_errors.values()),
    }


def _passes_accuracy(row: dict[str, Any], protocol: dict[str, Any]) -> bool:
    gate = protocol["accuracy_qualification"]
    return bool(
        row["amplitude_relative_l2"] <= gate["amplitude_relative_l2_max"]
        and row["worst_output_relative_l2"]
        <= gate["worst_twelve_output_relative_l2_max"]
    )


def _type2_subset(
    cp: Any,
    cufinufft: Any,
    certificate: dict[str, Any],
    channels: np.ndarray,
    q_subset: np.ndarray,
    *,
    eps: float,
    dtype: str,
) -> np.ndarray:
    contract = affine_lattice_type2_contract(certificate, q_subset)
    targets = np.asarray(contract["scaled_targets"])
    plan = cufinufft.Plan(
        2,
        tuple(int(v) for v in certificate["shape"]),
        n_trans=4,
        eps=float(eps),
        isign=-1,
        dtype=dtype,
        modeord=0,
    )
    plan.setpts(
        cp.asarray(targets[:, 0], dtype=cp.float32),
        cp.asarray(targets[:, 1], dtype=cp.float32),
        cp.asarray(targets[:, 2], dtype=cp.float32),
    )
    out = plan.execute(cp.asarray(channels, dtype=cp.complex64))
    corrected = out * cp.asarray(
        contract["center_phase"], dtype=cp.complex64
    )[None, :]
    cp.cuda.runtime.deviceSynchronize()
    return cp.asnumpy(corrected)


def _type3_subset(
    cp: Any,
    cufinufft: Any,
    rotated: np.ndarray,
    weights: np.ndarray,
    q_subset: np.ndarray,
    *,
    eps: float,
    dtype: str,
) -> np.ndarray:
    plan = cufinufft.Plan(
        3,
        2,
        n_trans=4,
        eps=float(eps),
        isign=-1,
        dtype=dtype,
    )
    plan.setpts(
        cp.asarray(rotated[:, 0], dtype=cp.float32),
        cp.asarray(rotated[:, 1], dtype=cp.float32),
        s=cp.asarray(q_subset[:, 0], dtype=cp.float32),
        t=cp.asarray(q_subset[:, 1], dtype=cp.float32),
    )
    out = plan.execute(cp.asarray(weights, dtype=cp.complex64))
    cp.cuda.runtime.deviceSynchronize()
    return cp.asnumpy(out)


def _fft_subset(
    torch: Any,
    device: Any,
    certificate: dict[str, Any],
    channels: np.ndarray,
    q_subset: np.ndarray,
    *,
    pad_factor: int,
    dtype: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    contract = affine_lattice_type2_contract(certificate, q_subset)
    interpolator = TorchPreparedPeriodicCoefficientFFT3D(
        tuple(int(v) for v in certificate["shape"]),
        contract["scaled_targets"],
        torch=torch,
        device=device,
        dtype=dtype,
        phase_sign=-1,
        pad_factor=int(pad_factor),
    )
    phase = torch.as_tensor(
        np.exp(-1j * (q_subset @ np.asarray(certificate["origin"]))),
        dtype=torch.complex64,
        device=device,
    )
    values = interpolator.forward(channels) * phase[None, :]
    sync(torch, device)
    result = values.detach().cpu().numpy()
    memory = {
        "four_channel_transformed_grid_bytes": (
            interpolator.transformed_grid_bytes
        ),
        "interpolation_state_bytes": interpolator.resident_bytes,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    del values, phase, interpolator
    torch.cuda.empty_cache()
    return result, memory


def accuracy_qualification(
    *,
    protocol: dict[str, Any],
    source: OrientationSource,
    source_rotation: float,
    flat_q: np.ndarray,
    q_hat_flat: np.ndarray,
    polarization: np.ndarray,
    reference_certificate: dict[str, Any],
    cp: Any,
    cufinufft: Any,
    torch: Any,
    device: Any,
    maximum_fft_pad_factor: int | None,
) -> dict[str, Any]:
    gate = protocol["accuracy_qualification"]
    selection_indices = [
        int(v)
        for v in gate["orientation_indices"]
        if source.bundle is not None or int(v) == 1
    ]
    target_indices = np.unique(
        np.linspace(
            0,
            flat_q.shape[0] - 1,
            int(gate["direct_targets"]),
            dtype=np.int64,
        )
    )
    q_subset = np.ascontiguousarray(flat_q[target_indices])
    q_hat_subset = np.ascontiguousarray(q_hat_flat[target_indices])
    rows: dict[str, list[dict[str, Any]]] = {
        "affine_type2": [],
        "projected_type3": [],
        "dense_periodic_fft": [],
    }
    orientation_payloads = []
    scale = float(protocol["dataset"]["source_coordinate_scale"])
    for orientation in selection_indices:
        points, nuclear, magnetization = _load_scaled(source, orientation, scale)
        rotated = rotate_points_about_z(points, source_rotation)
        certificate = certify_affine_cartesian_lattice(
            rotated,
            basis_anchor_rows=tuple(
                protocol["affine_lattice_contract"][
                    "basis_anchor_rows_zero_based"
                ]
            ),
            expected_shape=tuple(
                protocol["affine_lattice_contract"]["expected_shape"]
            ),
            coordinate_tolerance=protocol["affine_lattice_contract"][
                "coordinate_reconstruction_tolerance_nm"
            ],
            orthogonality_tolerance=protocol["affine_lattice_contract"][
                "normalized_gram_tolerance"
            ],
        )
        invariance = compare_affine_lattice_certificates(
            certificate,
            reference_certificate,
            gram_tolerance=protocol["affine_lattice_contract"][
                "normalized_gram_tolerance"
            ],
        )
        if not invariance["pass"]:
            raise RuntimeError(
                f"orientation {orientation} violates the affine lattice contract"
            )
        weights = _weights(nuclear, magnetization)
        channels = affine_lattice_channels(
            certificate, nuclear, magnetization, dtype=np.float32
        )
        direct = direct_fourier_channel_subset(
            rotated, weights, q_subset, np.arange(q_subset.shape[0])
        )
        orientation_payloads.append(
            (
                orientation,
                rotated,
                weights,
                certificate,
                channels,
                direct,
                invariance,
            )
        )

    dtype = protocol["workload"]["dtype"]
    for eps in protocol["baseline_candidates"]["affine_type2"][
        "eps_candidates"
    ]:
        per_orientation = []
        for payload in orientation_payloads:
            orientation, _, _, certificate, channels, direct, _ = payload
            actual = _type2_subset(
                cp,
                cufinufft,
                certificate,
                channels,
                q_subset,
                eps=eps,
                dtype=dtype,
            )
            row = _error_row(actual, direct, q_hat_subset, polarization)
            row["orientation"] = orientation
            per_orientation.append(row)
        summary = {
            "eps": float(eps),
            "per_orientation": per_orientation,
            "worst_amplitude_relative_l2": max(
                row["amplitude_relative_l2"] for row in per_orientation
            ),
            "worst_output_relative_l2": max(
                row["worst_output_relative_l2"] for row in per_orientation
            ),
        }
        summary["pass"] = bool(
            summary["worst_amplitude_relative_l2"]
            <= gate["amplitude_relative_l2_max"]
            and summary["worst_output_relative_l2"]
            <= gate["worst_twelve_output_relative_l2_max"]
        )
        rows["affine_type2"].append(summary)

    for eps in protocol["baseline_candidates"]["projected_type3_control"][
        "eps_candidates"
    ]:
        per_orientation = []
        for payload in orientation_payloads:
            orientation, rotated, weights, _, _, direct, _ = payload
            actual = _type3_subset(
                cp,
                cufinufft,
                rotated,
                weights,
                q_subset,
                eps=eps,
                dtype=dtype,
            )
            row = _error_row(actual, direct, q_hat_subset, polarization)
            row["orientation"] = orientation
            per_orientation.append(row)
        summary = {
            "eps": float(eps),
            "per_orientation": per_orientation,
            "worst_amplitude_relative_l2": max(
                row["amplitude_relative_l2"] for row in per_orientation
            ),
            "worst_output_relative_l2": max(
                row["worst_output_relative_l2"] for row in per_orientation
            ),
        }
        summary["pass"] = bool(
            summary["worst_amplitude_relative_l2"]
            <= gate["amplitude_relative_l2_max"]
            and summary["worst_output_relative_l2"]
            <= gate["worst_twelve_output_relative_l2_max"]
        )
        rows["projected_type3"].append(summary)

    declared_pads = protocol["baseline_candidates"]["dense_periodic_fft"][
        "pad_factors"
    ]
    tested_pads = [
        int(pad)
        for pad in declared_pads
        if maximum_fft_pad_factor is None
        or int(pad) <= int(maximum_fft_pad_factor)
    ]
    for pad in tested_pads:
        per_orientation = []
        memory: dict[str, Any] = {}
        error: str | None = None
        try:
            torch.cuda.reset_peak_memory_stats(device)
            for payload in orientation_payloads:
                orientation, _, _, certificate, channels, direct, _ = payload
                actual, memory = _fft_subset(
                    torch,
                    device,
                    certificate,
                    channels,
                    q_subset,
                    pad_factor=pad,
                    dtype=dtype,
                )
                row = _error_row(actual, direct, q_hat_subset, polarization)
                row["orientation"] = orientation
                per_orientation.append(row)
        except (RuntimeError, MemoryError) as exc:
            error = repr(exc)
            torch.cuda.empty_cache()
        summary = {
            "pad_factor": pad,
            "per_orientation": per_orientation,
            "memory": memory,
            "error": error,
        }
        if per_orientation and error is None:
            summary["worst_amplitude_relative_l2"] = max(
                row["amplitude_relative_l2"] for row in per_orientation
            )
            summary["worst_output_relative_l2"] = max(
                row["worst_output_relative_l2"] for row in per_orientation
            )
            summary["pass"] = bool(
                summary["worst_amplitude_relative_l2"]
                <= gate["amplitude_relative_l2_max"]
                and summary["worst_output_relative_l2"]
                <= gate["worst_twelve_output_relative_l2_max"]
                and memory["peak_allocated_bytes"]
                <= protocol["workload"]["maximum_allowed_resident_bytes"]
            )
        else:
            summary["pass"] = False
        rows["dense_periodic_fft"].append(summary)

    selected_type2 = next(
        (row["eps"] for row in rows["affine_type2"] if row["pass"]), None
    )
    selected_type3 = next(
        (row["eps"] for row in rows["projected_type3"] if row["pass"]), None
    )
    qualified_fft = [
        row["pad_factor"]
        for row in rows["dense_periodic_fft"]
        if row["pass"]
    ]
    return {
        "selection_completed_before_timing": True,
        "selection_orientation_indices": selection_indices,
        "target_indices": target_indices.tolist(),
        "rows": rows,
        "selected_type2_eps": selected_type2,
        "selected_type3_eps": selected_type3,
        "qualified_fft_pad_factors": qualified_fft,
        "declared_fft_pad_factors": declared_pads,
        "tested_fft_pad_factors": tested_pads,
        "resource_limited_frontier": tested_pads != declared_pads,
        "orientation_lattice_invariance": [
            {"orientation": p[0], **p[-1]} for p in orientation_payloads
        ],
    }


class FourierArms:
    def __init__(
        self,
        *,
        protocol: dict[str, Any],
        torch: Any,
        cp: Any,
        cufinufft: Any,
        device: Any,
        prepared: Any,
        acfo: Any,
        flat_q: np.ndarray,
        reference_certificate: dict[str, Any],
        type2_eps: float,
        type3_eps: float,
        selected_fft_pad: int | None,
    ) -> None:
        self.protocol = protocol
        self.torch = torch
        self.cp = cp
        self.device = device
        self.prepared = prepared
        self.acfo = acfo
        self.flat_q = flat_q
        self.reference_certificate = reference_certificate
        self.type2_eps = float(type2_eps)
        self.type3_eps = float(type3_eps)
        self.selected_fft_pad = selected_fft_pad
        self.dtype = protocol["workload"]["dtype"]
        self.shape = tuple(int(v) for v in reference_certificate["shape"])
        self.target_x = cp.asarray(flat_q[:, 0], dtype=cp.float32)
        self.target_y = cp.asarray(flat_q[:, 1], dtype=cp.float32)

        start = perf_counter()
        self.acfo_weight_tensor = torch.empty(
            (4, reference_certificate["active_sites"]),
            dtype=torch.float32,
            device=device,
        )
        sync(torch, device)
        self.acfo_buffer_setup_seconds = perf_counter() - start

        start = perf_counter()
        self.type2_plan = cufinufft.Plan(
            2,
            self.shape,
            n_trans=4,
            eps=self.type2_eps,
            isign=-1,
            dtype=self.dtype,
            modeord=0,
        )
        self.type2_q_transposed = cp.asarray(
            np.ascontiguousarray(flat_q.T, dtype=np.float32)
        )
        self.type2_basis = cp.empty((3, 3), dtype=cp.float32)
        self.type2_center = cp.empty(3, dtype=cp.float32)
        self.type2_targets = cp.empty(
            (3, flat_q.shape[0]), dtype=cp.float32
        )
        self.type2_phase_angle = cp.empty(flat_q.shape[0], dtype=cp.float32)
        self.type2_coefficients = cp.empty((4, *self.shape), dtype=cp.complex64)
        self.type2_phase = cp.empty(flat_q.shape[0], dtype=cp.complex64)
        self.type2_out = cp.empty((4, flat_q.shape[0]), dtype=cp.complex64)
        self.type2_corrected = cp.empty_like(self.type2_out)
        cp.cuda.runtime.deviceSynchronize()
        self.type2_setup_seconds = perf_counter() - start

        start = perf_counter()
        self.type3_plan = cufinufft.Plan(
            3,
            2,
            n_trans=4,
            eps=self.type3_eps,
            isign=-1,
            dtype=self.dtype,
        )
        self.type3_source_x = cp.empty(
            reference_certificate["active_sites"], dtype=cp.float32
        )
        self.type3_source_y = cp.empty(
            reference_certificate["active_sites"], dtype=cp.float32
        )
        self.type3_strengths = cp.empty(
            (4, reference_certificate["active_sites"]), dtype=cp.complex64
        )
        self.type3_out = cp.empty((4, flat_q.shape[0]), dtype=cp.complex64)
        cp.cuda.runtime.deviceSynchronize()
        self.type3_setup_seconds = perf_counter() - start

        start = perf_counter()
        self.fft_coefficients = torch.empty(
            (4, *self.shape), dtype=torch.complex64, device=device
        )
        self.fft_phase = torch.empty(
            flat_q.shape[0], dtype=torch.complex64, device=device
        )
        sync(torch, device)
        self.fft_buffer_setup_seconds = perf_counter() - start

    def _certificate(self, rotated: np.ndarray) -> dict[str, Any]:
        spec = self.protocol["affine_lattice_contract"]
        certificate = certify_affine_cartesian_lattice(
            rotated,
            basis_anchor_rows=tuple(spec["basis_anchor_rows_zero_based"]),
            expected_shape=tuple(spec["expected_shape"]),
            coordinate_tolerance=spec["coordinate_reconstruction_tolerance_nm"],
            orthogonality_tolerance=spec["normalized_gram_tolerance"],
        )
        comparison = compare_affine_lattice_certificates(
            certificate,
            self.reference_certificate,
            gram_tolerance=spec["normalized_gram_tolerance"],
        )
        if not comparison["pass"]:
            raise RuntimeError("orientation violates the frozen lattice invariance")
        return certificate

    def acfo_eval(
        self, rotated: np.ndarray, weights: np.ndarray
    ) -> tuple[Any, float]:
        sync(self.torch, self.device)
        start = perf_counter()
        self.acfo.configure_source_updates(rotated)
        self.acfo_weight_tensor.copy_(
            self.torch.from_numpy(np.ascontiguousarray(weights))
        )
        amplitudes = self.acfo.amplitudes_from_weights(
            self.acfo_weight_tensor
        )
        sync(self.torch, self.device)
        elapsed = perf_counter() - start
        return _channels_from_torch_amplitudes(self.torch, amplitudes), elapsed

    def type2_eval(
        self,
        rotated: np.ndarray,
        nuclear: np.ndarray,
        magnetization: np.ndarray,
    ) -> tuple[Any, float]:
        self.cp.cuda.runtime.deviceSynchronize()
        start = perf_counter()
        certificate = self._certificate(rotated)
        channels = affine_lattice_channels(
            certificate, nuclear, magnetization, dtype=np.float32
        )
        self.type2_basis.set(
            np.ascontiguousarray(certificate["basis"], dtype=np.float32)
        )
        self.type2_center.set(
            np.ascontiguousarray(certificate["center"], dtype=np.float32)
        )
        self.cp.matmul(
            self.type2_basis.T,
            self.type2_q_transposed,
            out=self.type2_targets,
        )
        self.type2_targets += np.float32(np.pi)
        self.cp.remainder(
            self.type2_targets,
            np.float32(2.0 * np.pi),
            out=self.type2_targets,
        )
        self.type2_targets -= np.float32(np.pi)
        self.cp.matmul(
            self.type2_center,
            self.type2_q_transposed,
            out=self.type2_phase_angle,
        )
        self.type2_coefficients.set(
            np.ascontiguousarray(channels, dtype=np.complex64)
        )
        self.type2_phase.real = self.cp.cos(self.type2_phase_angle)
        self.type2_phase.imag = -self.cp.sin(self.type2_phase_angle)
        self.type2_plan.setpts(
            self.type2_targets[0],
            self.type2_targets[1],
            self.type2_targets[2],
        )
        self.type2_plan.execute(self.type2_coefficients, out=self.type2_out)
        self.cp.multiply(
            self.type2_out,
            self.type2_phase[None, :],
            out=self.type2_corrected,
        )
        self.cp.cuda.runtime.deviceSynchronize()
        elapsed = perf_counter() - start
        return self.torch.from_dlpack(self.type2_corrected).reshape(
            4, self.prepared.manifold.q_perp.size, self.prepared.phi.size
        ), elapsed

    def type3_eval(
        self, rotated: np.ndarray, weights: np.ndarray
    ) -> tuple[Any, float]:
        self.cp.cuda.runtime.deviceSynchronize()
        start = perf_counter()
        self.type3_source_x.set(
            np.ascontiguousarray(rotated[:, 0], dtype=np.float32)
        )
        self.type3_source_y.set(
            np.ascontiguousarray(rotated[:, 1], dtype=np.float32)
        )
        self.type3_strengths.set(
            np.ascontiguousarray(weights, dtype=np.complex64)
        )
        self.type3_plan.setpts(
            self.type3_source_x,
            self.type3_source_y,
            s=self.target_x,
            t=self.target_y,
        )
        self.type3_plan.execute(self.type3_strengths, out=self.type3_out)
        self.cp.cuda.runtime.deviceSynchronize()
        elapsed = perf_counter() - start
        return self.torch.from_dlpack(self.type3_out).reshape(
            4, self.prepared.manifold.q_perp.size, self.prepared.phi.size
        ), elapsed

    def fft_eval(
        self,
        rotated: np.ndarray,
        nuclear: np.ndarray,
        magnetization: np.ndarray,
        *,
        pad_factor: int | None = None,
    ) -> tuple[Any, float]:
        pad = self.selected_fft_pad if pad_factor is None else int(pad_factor)
        if pad is None:
            raise RuntimeError("no FFT pad factor was selected")
        sync(self.torch, self.device)
        start = perf_counter()
        certificate = self._certificate(rotated)
        contract = affine_lattice_type2_contract(certificate, self.flat_q)
        channels = affine_lattice_channels(
            certificate, nuclear, magnetization, dtype=np.float32
        )
        interpolator = TorchPreparedPeriodicCoefficientFFT3D(
            self.shape,
            contract["scaled_targets"],
            torch=self.torch,
            device=self.device,
            dtype=self.dtype,
            phase_sign=-1,
            pad_factor=pad,
        )
        self.fft_coefficients.copy_(
            self.torch.from_numpy(
                np.ascontiguousarray(channels, dtype=np.complex64)
            )
        )
        self.fft_phase.copy_(
            self.torch.from_numpy(
                np.ascontiguousarray(
                    np.exp(
                        -1j
                        * (
                            self.flat_q
                            @ np.asarray(certificate["origin"])
                        )
                    ),
                    dtype=np.complex64,
                )
            )
        )
        values = interpolator.forward(self.fft_coefficients) * self.fft_phase[
            None, :
        ]
        sync(self.torch, self.device)
        elapsed = perf_counter() - start
        del interpolator
        return values.reshape(
            4, self.prepared.manifold.q_perp.size, self.prepared.phi.size
        ), elapsed


def fft_timing_screen(
    arms: FourierArms,
    *,
    rotated: np.ndarray,
    nuclear: np.ndarray,
    magnetization: np.ndarray,
    qualified_pads: list[int],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    screen = protocol["timing_screen"]
    rows = []
    # Allocation-order prewarm is explicitly excluded from every sample.
    for pad in reversed(qualified_pads):
        value, _ = arms.fft_eval(
            rotated, nuclear, magnetization, pad_factor=pad
        )
        del value
        arms.torch.cuda.empty_cache()
    for pad in qualified_pads:
        for _ in range(int(screen["warmups_per_fft_candidate"])):
            value, _ = arms.fft_eval(
                rotated, nuclear, magnetization, pad_factor=pad
            )
            del value
        samples = []
        for sample in range(int(screen["samples_per_fft_candidate"])):
            value, elapsed = arms.fft_eval(
                rotated, nuclear, magnetization, pad_factor=pad
            )
            del value
            samples.append(elapsed)
        rows.append(
            {
                "pad_factor": pad,
                "sample_ids": [f"screen-fft-p{pad}-{i}" for i in range(len(samples))],
                "samples_seconds": samples,
                "median_seconds": float(np.median(samples)),
            }
        )
    selected = (
        min(rows, key=lambda row: row["median_seconds"])["pad_factor"]
        if rows
        else None
    )
    return {
        "machine_eligible": "RTX 3090" in arms.torch.cuda.get_device_name(
            arms.device
        ),
        "rows": rows,
        "selected_fft_pad_factor": selected,
        "screen_sample_ids": [item for row in rows for item in row["sample_ids"]],
    }


def _bootstrap_ratio(
    acfo: list[float], baseline: list[float], *, seed: int, resamples: int
) -> dict[str, float]:
    a = np.asarray(acfo, dtype=np.float64)
    b = np.asarray(baseline, dtype=np.float64)
    rng = np.random.default_rng(seed)
    ratios = np.empty(int(resamples), dtype=np.float64)
    for index in range(int(resamples)):
        selected = rng.integers(0, a.size, size=a.size)
        ratios[index] = float(np.sum(b[selected]) / np.sum(a[selected]))
    return {
        "point": float(np.sum(b) / np.sum(a)),
        "lower_95": float(np.quantile(ratios, 0.025)),
        "upper_95": float(np.quantile(ratios, 0.975)),
    }


def pairwise_abba_confirmation(
    arms: FourierArms,
    *,
    rotated: np.ndarray,
    nuclear: np.ndarray,
    magnetization: np.ndarray,
    weights: np.ndarray,
    include_fft: bool,
    protocol: dict[str, Any],
    screen_sample_ids: list[str],
) -> dict[str, Any]:
    spec = protocol["independent_confirmation"]
    baselines = ["affine_type2", "projected_type3"]
    if include_fft:
        baselines.append("dense_periodic_fft")

    def evaluate(name: str) -> tuple[Any, float]:
        if name == "acfo":
            return arms.acfo_eval(rotated, weights)
        if name == "affine_type2":
            return arms.type2_eval(rotated, nuclear, magnetization)
        if name == "projected_type3":
            return arms.type3_eval(rotated, weights)
        return arms.fft_eval(rotated, nuclear, magnetization)

    comparisons: dict[str, Any] = {}
    confirmation_ids: list[str] = []
    for baseline_index, baseline in enumerate(baselines):
        for warmup in range(int(spec["warmups_per_arm_per_pair"])):
            order = ("acfo", baseline) if warmup % 2 == 0 else (baseline, "acfo")
            for arm in order:
                value, _ = evaluate(arm)
                del value
        samples = {"acfo": [], baseline: []}
        sample_ids = {"acfo": [], baseline: []}
        for sample in range(int(spec["paired_samples_per_arm"])):
            order = (
                ("acfo", baseline)
                if sample % 4 in (0, 3)
                else (baseline, "acfo")
            )
            for arm in order:
                value, elapsed = evaluate(arm)
                del value
                samples[arm].append(elapsed)
                identifier = f"confirm-{baseline}-{arm}-{sample}"
                sample_ids[arm].append(identifier)
                confirmation_ids.append(identifier)
        comparisons[baseline] = {
            "order_contract": "ABBA",
            "warmups_per_arm": int(spec["warmups_per_arm_per_pair"]),
            "samples_per_arm": int(spec["paired_samples_per_arm"]),
            "samples_seconds": samples,
            "sample_ids": sample_ids,
            "median_seconds": {
                arm: float(np.median(values)) for arm, values in samples.items()
            },
            "baseline_over_acfo_speed_ratio": _bootstrap_ratio(
                samples["acfo"],
                samples[baseline],
                seed=int(spec["bootstrap_seed"]) + baseline_index,
                resamples=int(spec["bootstrap_resamples"]),
            ),
        }
    overlap = sorted(set(screen_sample_ids) & set(confirmation_ids))
    strongest = min(
        baselines,
        key=lambda name: comparisons[name]["median_seconds"][name],
    )
    return {
        "comparisons": comparisons,
        "confirmation_sample_ids": confirmation_ids,
        "screen_confirmation_sample_overlap": overlap,
        "screen_sample_reuse_count": len(overlap),
        "strongest_baseline_selected_before_orientation_2": strongest,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    _prepare_windows_cuda_dll_search()
    import torch
    import cupy as cp
    import cufinufft

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    requested = (
        int(protocol["dataset"]["orientation_count"])
        if args.max_orientations is None
        else int(args.max_orientations)
    )
    if requested <= 0 or requested > int(protocol["dataset"]["orientation_count"]):
        raise ValueError("max orientations is outside 1..800")
    full_run = requested == int(protocol["dataset"]["orientation_count"])
    if full_run and args.source_archive is None:
        raise ValueError("the full run requires --source-archive")
    if full_run and not args.skip_archive_oracle and args.output_archive is None:
        raise ValueError("the full oracle run requires --output-archive")
    if full_run and args.maximum_fft_pad_factor is not None:
        raise ValueError("the full run may not truncate the frozen FFT frontier")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(device)
    gpu_name = torch.cuda.get_device_name(device)
    if full_run and "RTX 3090" not in gpu_name:
        raise RuntimeError("the prospective full run is frozen to an RTX 3090")
    source = OrientationSource(
        source_archive=args.source_archive,
        reduced_dir=args.reduced_dir,
        protocol=protocol,
    )
    try:
        topology = (
            validate_source_archive(source.bundle, protocol)
            if source.bundle is not None
            else {"pass": True, "role": "orientation-1 reduced smoke"}
        )
        if not topology["pass"]:
            raise RuntimeError("source archive topology is incomplete")
        packed_ids, centers = load_centers(protocol, source)
        scale = float(protocol["dataset"]["source_coordinate_scale"])
        first_points, first_nuclear, first_magnetization = _load_scaled(
            source, 1, scale
        )
        if first_points.shape[0] != int(protocol["dataset"]["sources_per_orientation"]):
            raise ValueError("unexpected source count")

        q_nodes = (
            int(protocol["workload"]["q_nodes"])
            if args.smoke_q_nodes is None
            else int(args.smoke_q_nodes)
        )
        phi_count = (
            int(protocol["workload"]["unique_theta"])
            if args.smoke_unique_theta is None
            else int(args.smoke_unique_theta)
        )
        if full_run and (
            q_nodes != int(protocol["workload"]["q_nodes"])
            or phi_count != int(protocol["workload"]["unique_theta"])
        ):
            raise ValueError("the full run may not override detector dimensions")
        phi_step = 2.0 * np.pi / phi_count
        source_rotation = 0.5 * phi_step - 0.5 * np.pi
        first_rotated = rotate_points_about_z(first_points, source_rotation)
        rotated_centers = np.stack(
            [rotate_points_about_z(table, source_rotation) for table in centers]
        )
        reference_certificate = certify_affine_cartesian_lattice(
            first_rotated,
            basis_anchor_rows=tuple(
                protocol["affine_lattice_contract"]["basis_anchor_rows_zero_based"]
            ),
            expected_shape=tuple(
                protocol["affine_lattice_contract"]["expected_shape"]
            ),
            coordinate_tolerance=protocol["affine_lattice_contract"][
                "coordinate_reconstruction_tolerance_nm"
            ],
            orthogonality_tolerance=protocol["affine_lattice_contract"][
                "normalized_gram_tolerance"
            ],
        )
        q_radius = np.linspace(
            0.0, float(protocol["workload"]["q_max_inverse_nm"]), q_nodes
        )
        manifold = AxisymmetricManifold(
            u=np.arange(q_nodes, dtype=float),
            q_perp=q_radius,
            q_z=np.zeros_like(q_radius),
            name="numagsans-example3-strongest-baseline",
            interpretation="sampling",
        )
        from waxs_cake.gpu_miller import warm_gpu_miller_kernel

        gpu_miller_jit_seconds = warm_gpu_miller_kernel()
        acfo_setup_start = perf_counter()
        radial_extent = float(np.max(np.linalg.norm(first_points, axis=1)))
        prepared = PreparedBandlimitedMagneticSansOperator(
            first_rotated,
            first_nuclear,
            first_magnetization,
            manifold,
            n_r=int(protocol["workload"]["acfo_n_r"]),
            n_z=int(protocol["workload"]["acfo_n_z"]),
            n_phi=phi_count,
            r_max=radial_extent + 0.5,
            z_range=(-radial_extent - 0.5, radial_extent + 0.5),
            phase_sign=-1,
            hist_backend="numpy",
            circular_backend="auto",
            complex_dtype=np.complex64,
            margin=int(protocol["harmonic_support"]["mode_padding"]),
        )
        phi = np.asarray(prepared.phi)
        emitted_q_hat = np.stack(
            [np.cos(phi), np.sin(phi), np.zeros_like(phi)], axis=-1
        )
        original_q_hat = q_cloud_in_original_grid_frame(
            emitted_q_hat, source_rotation
        )
        acfo = TorchFlatDetectorBandlimitedMagneticSansOperator(
            prepared,
            torch=torch,
            device=device,
            dtype=protocol["workload"]["dtype"],
            detector_directions=original_q_hat,
            source_deposition=protocol["workload"]["source_deposition"],
            kernel_backend="gpu_miller",
            active_r_indices_override=np.arange(
                int(protocol["workload"]["acfo_n_r"])
            ),
        )
        sync(torch, device)
        acfo_setup_seconds = perf_counter() - acfo_setup_start
        if acfo.kernel_backend != protocol["harmonic_support"]["required_kernel_backend"]:
            raise RuntimeError("GPU Miller backend was not selected")
        if acfo.miller_recurrence_margin != int(
            protocol["harmonic_support"]["miller_recurrence_margin"]
        ):
            raise RuntimeError("Miller recurrence margin differs from protocol")

        q_xyz = detector_q_grid(q_radius, phi)
        flat_q = np.ascontiguousarray(q_xyz.reshape(-1, 3))
        q_hat_flat = np.broadcast_to(
            original_q_hat[None, :, :], (q_nodes, phi_count, 3)
        ).reshape(-1, 3)
        polarization = np.asarray(
            protocol["workload"]["polarization_acfo"], dtype=np.float64
        )
        qualification = accuracy_qualification(
            protocol=protocol,
            source=source,
            source_rotation=source_rotation,
            flat_q=flat_q,
            q_hat_flat=q_hat_flat,
            polarization=polarization,
            reference_certificate=reference_certificate,
            cp=cp,
            cufinufft=cufinufft,
            torch=torch,
            device=device,
            maximum_fft_pad_factor=args.maximum_fft_pad_factor,
        )
        if qualification["selected_type2_eps"] is None:
            raise RuntimeError("no affine type-2 epsilon passed accuracy")
        if qualification["selected_type3_eps"] is None:
            raise RuntimeError("no projected type-3 epsilon passed accuracy")

        arms = FourierArms(
            protocol=protocol,
            torch=torch,
            cp=cp,
            cufinufft=cufinufft,
            device=device,
            prepared=prepared,
            acfo=acfo,
            flat_q=flat_q,
            reference_certificate=reference_certificate,
            type2_eps=qualification["selected_type2_eps"],
            type3_eps=qualification["selected_type3_eps"],
            selected_fft_pad=None,
        )
        production_target_indices = np.asarray(
            qualification["target_indices"], dtype=np.int64
        )
        first_weights = _weights(first_nuclear, first_magnetization)
        production_direct = direct_fourier_channel_subset(
            first_rotated, first_weights, flat_q, production_target_indices
        )
        production_accuracy: dict[str, Any] = {}
        for production_arm in ("acfo", "affine_type2", "projected_type3"):
            if production_arm == "acfo":
                production_value, _ = arms.acfo_eval(
                    first_rotated, first_weights
                )
            elif production_arm == "affine_type2":
                production_value, _ = arms.type2_eval(
                    first_rotated, first_nuclear, first_magnetization
                )
            else:
                production_value, _ = arms.type3_eval(
                    first_rotated, first_weights
                )
            selected_value = (
                production_value.reshape(4, -1)[:, production_target_indices]
                .detach()
                .cpu()
                .numpy()
            )
            row = _error_row(
                selected_value,
                production_direct,
                q_hat_flat[production_target_indices],
                polarization,
            )
            row["pass"] = _passes_accuracy(row, protocol)
            production_accuracy[production_arm] = row
            del production_value
        if not all(row["pass"] for row in production_accuracy.values()):
            raise RuntimeError(
                "a production GPU arm failed the frozen pre-timing accuracy gate"
            )
        screen = fft_timing_screen(
            arms,
            rotated=first_rotated,
            nuclear=first_nuclear,
            magnetization=first_magnetization,
            qualified_pads=qualification["qualified_fft_pad_factors"],
            protocol=protocol,
        )
        arms.selected_fft_pad = screen["selected_fft_pad_factor"]
        confirmation = pairwise_abba_confirmation(
            arms,
            rotated=first_rotated,
            nuclear=first_nuclear,
            magnetization=first_magnetization,
            weights=_weights(first_nuclear, first_magnetization),
            include_fft=screen["selected_fft_pad_factor"] is not None,
            protocol=protocol,
            screen_sample_ids=screen["screen_sample_ids"],
        )
        if confirmation["screen_sample_reuse_count"] != 0:
            raise RuntimeError("screen samples leaked into confirmation")
        strongest = confirmation[
            "strongest_baseline_selected_before_orientation_2"
        ]
        acfo_total_setup_seconds = (
            acfo_setup_seconds + arms.acfo_buffer_setup_seconds
        )

        method_names = ("acfo", strongest)
        real_dtype = torch.float32
        complex_dtype = torch.complex64
        q_hat_torch = torch.as_tensor(
            original_q_hat, dtype=real_dtype, device=device
        )
        polarization_torch = torch.as_tensor(
            polarization, dtype=real_dtype, device=device
        )
        q_xy_torch = torch.as_tensor(
            flat_q[:, :2], dtype=real_dtype, device=device
        )
        dilute = {
            arm: {
                name: torch.zeros(
                    (q_nodes, phi_count), dtype=real_dtype, device=device
                )
                for name in PRIMITIVE_OUTPUT_NAMES
            }
            for arm in method_names
        }
        packed = {
            arm: torch.zeros(
                (len(packed_ids), 4, q_nodes, phi_count),
                dtype=complex_dtype,
                device=device,
            )
            for arm in method_names
        }
        timings = {
            arm: [] for arm in method_names
        }
        reductions = {arm: [] for arm in method_names}
        phase_times: list[float] = []
        load_times: list[float] = []
        heldout_rows: list[dict[str, Any]] = []
        heldout = set(protocol["accuracy_qualification"]["held_out_orientation_indices"])
        direct_indices = np.unique(
            np.linspace(
                0,
                flat_q.shape[0] - 1,
                int(protocol["accuracy_qualification"]["direct_targets"]),
                dtype=np.int64,
            )
        )
        prefixes = []
        prefix_set = {
            int(v)
            for v in protocol["workload"]["prefix_orientation_counts"]
            if int(v) <= requested
        }

        def evaluate_selected(
            rotated: np.ndarray,
            nuclear: np.ndarray,
            magnetization: np.ndarray,
            weights: np.ndarray,
        ) -> tuple[Any, float]:
            if strongest == "affine_type2":
                return arms.type2_eval(rotated, nuclear, magnetization)
            if strongest == "projected_type3":
                return arms.type3_eval(rotated, weights)
            return arms.fft_eval(rotated, nuclear, magnetization)

        baseline_setup_seconds = {
            "affine_type2": arms.type2_setup_seconds,
            "projected_type3": arms.type3_setup_seconds,
            "dense_periodic_fft": arms.fft_buffer_setup_seconds,
        }[strongest]
        for orientation in range(1, requested + 1):
            load_start = perf_counter()
            points, nuclear, magnetization = _load_scaled(
                source, orientation, scale
            )
            load_times.append(perf_counter() - load_start)
            rotated = rotate_points_about_z(points, source_rotation)
            weights = _weights(nuclear, magnetization)
            values: dict[str, Any] = {}
            order = method_names if orientation % 2 else method_names[::-1]
            for arm in order:
                if arm == "acfo":
                    value, elapsed = arms.acfo_eval(rotated, weights)
                else:
                    value, elapsed = evaluate_selected(
                        rotated, nuclear, magnetization, weights
                    )
                values[arm] = value
                timings[arm].append(elapsed)

            phase_start = perf_counter()
            center_xy = torch.as_tensor(
                rotated_centers[:, orientation - 1, :2],
                dtype=real_dtype,
                device=device,
            )
            phase = torch.exp(-1j * (center_xy @ q_xy_torch.T)).reshape(
                len(packed_ids), q_nodes, phi_count
            )
            sync(torch, device)
            phase_times.append(perf_counter() - phase_start)
            for arm in method_names:
                reduction_start = perf_counter()
                primitive = _primitive_torch(
                    torch, values[arm], q_hat_torch, polarization_torch
                )
                for name in PRIMITIVE_OUTPUT_NAMES:
                    dilute[arm][name].add_(primitive[name])
                packed[arm].add_(phase[:, None, ...] * values[arm][None, ...])
                sync(torch, device)
                reductions[arm].append(perf_counter() - reduction_start)

            if orientation in heldout and source.bundle is not None:
                direct = direct_fourier_channel_subset(
                    rotated, weights, flat_q, direct_indices
                )
                row = {"orientation": orientation, "methods": {}}
                for arm in method_names:
                    selected = (
                        values[arm].reshape(4, -1)[:, direct_indices]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    row["methods"][arm] = _error_row(
                        selected,
                        direct,
                        q_hat_flat[direct_indices],
                        polarization,
                    )
                heldout_rows.append(row)

            if orientation in prefix_set:
                outputs = {
                    arm: _hierarchy_outputs_torch(
                        torch,
                        dilute[arm],
                        packed[arm],
                        q_hat_torch,
                        polarization_torch,
                        packed_ids,
                    )
                    for arm in method_names
                }
                prefixes.append(
                    {
                        "orientations": orientation,
                        "acfo_fourier_total_seconds": gpu_miller_jit_seconds
                        + acfo_total_setup_seconds
                        + float(np.sum(timings["acfo"])),
                        "selected_baseline_fourier_total_seconds": baseline_setup_seconds
                        + float(np.sum(timings[strongest])),
                        "selected_baseline": strongest,
                        "measured_favorable": (
                            "acfo"
                            if gpu_miller_jit_seconds
                            + acfo_total_setup_seconds
                            + float(np.sum(timings["acfo"]))
                            < baseline_setup_seconds
                            + float(np.sum(timings[strongest]))
                            else strongest
                        ),
                        "ensemble_accuracy": _hierarchy_errors(
                            torch, outputs["acfo"], outputs[strongest]
                        ),
                    }
                )
            if orientation == 1 or orientation % 25 == 0 or orientation == requested:
                print(
                    f"orientation {orientation}/{requested}: "
                    f"ACFO {np.sum(timings['acfo']):.3f}s, "
                    f"{strongest} {np.sum(timings[strongest]):.3f}s",
                    flush=True,
                )

        final_outputs = {
            arm: _hierarchy_outputs_torch(
                torch,
                dilute[arm],
                packed[arm],
                q_hat_torch,
                polarization_torch,
                packed_ids,
            )
            for arm in method_names
        }
        final_errors = _hierarchy_errors(
            torch, final_outputs["acfo"], final_outputs[strongest]
        )
        archive_validation: dict[str, Any] = {"performed": False}
        if not args.skip_archive_oracle and args.output_archive is not None:
            archive_protocol = dict(protocol)
            archive_protocol["accuracy"] = protocol["accuracy_qualification"]
            archived, stream_seconds = stream_archived_rings(
                args.output_archive, archive_protocol
            )
            scales = physical_scales()
            physical_outputs = {
                arm: _hierarchy_outputs_torch(
                    torch,
                    dilute[arm],
                    packed[arm],
                    q_hat_torch,
                    polarization_torch,
                    packed_ids,
                    scales=scales,
                )
                for arm in method_names
            }
            radial = protocol["accuracy_qualification"][
                "archived_output_radial_indices"
            ]
            archive_validation = {
                "performed": True,
                "stream_seconds": stream_seconds,
                "radial_indices": radial,
                **{
                    arm: _archive_errors(physical_outputs[arm], archived, radial)
                    for arm in method_names
                },
            }

        acfo_samples = np.asarray(timings["acfo"], dtype=np.float64)
        baseline_samples = np.asarray(timings[strongest], dtype=np.float64)
        crossover = cumulative_crossover(
            acfo_samples,
            baseline_samples,
            candidate_setup=acfo_total_setup_seconds,
            baseline_setup=baseline_setup_seconds,
            candidate_cold_start=gpu_miller_jit_seconds,
        )
        crossover_json = {
            key: value
            for key, value in crossover.items()
            if not isinstance(value, np.ndarray)
        }
        total_speed = bootstrap_total_speedup(
            acfo_samples,
            baseline_samples,
            candidate_setup=acfo_total_setup_seconds + gpu_miller_jit_seconds,
            baseline_setup=baseline_setup_seconds,
            seed=int(protocol["independent_confirmation"]["bootstrap_seed"]),
            resamples=(
                int(protocol["independent_confirmation"]["bootstrap_resamples"])
                if full_run
                else 1000
            ),
        )
        heldout_worst_amplitude = max(
            (
                value["amplitude_relative_l2"]
                for row in heldout_rows
                for value in row["methods"].values()
            ),
            default=0.0,
        )
        heldout_worst_output = max(
            (
                value["worst_output_relative_l2"]
                for row in heldout_rows
                for value in row["methods"].values()
            ),
            default=0.0,
        )
        gate = protocol["accuracy_qualification"]
        qualification_pass = bool(
            not qualification["resource_limited_frontier"]
            and qualification["selected_type2_eps"] is not None
            and qualification["selected_type3_eps"] is not None
            and confirmation["screen_sample_reuse_count"] == 0
            and heldout_worst_amplitude <= gate["amplitude_relative_l2_max"]
            and heldout_worst_output <= gate["worst_twelve_output_relative_l2_max"]
            and final_errors["worst_relative_l2"]
            <= gate["full_ensemble_selected_baseline_worst_output_relative_l2_max"]
        )
        archive_pass = bool(
            not archive_validation["performed"]
            or all(
                archive_validation[arm]["worst_relative_l2"]
                <= gate["archived_output_worst_relative_l2_max"]
                for arm in method_names
            )
        )
        machine_pass = bool(not full_run or "RTX 3090" in gpu_name)
        decision = closure_decision(
            full_run=full_run,
            qualification_pass=qualification_pass,
            archive_pass=archive_pass,
            machine_pass=machine_pass,
            speed_lower_95=float(total_speed["lower_95"]),
            required_speed_lower_95=float(
                protocol["full_ensemble"]["required_lower_95_speed_ratio"]
            ),
        )
        return {
            "schema": "numagsans-example3-strongest-baseline-result-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "verdict": decision["verdict"],
            "mode": "full" if full_run else "smoke",
            "protocol_status": protocol["status"],
            "environment": {
                "gpu": gpu_name,
                "torch": torch.__version__,
                "cupy": cp.__version__,
                "cufinufft": getattr(cufinufft, "__version__", "unknown"),
                "device": str(device),
                "dtype": protocol["workload"]["dtype"],
            },
            "provenance": {
                "protocol": str(args.protocol),
                "source_archive": str(args.source_archive) if args.source_archive else None,
                "source_archive_md5": md5(args.source_archive) if args.source_archive else None,
                "output_archive": str(args.output_archive) if args.output_archive else None,
                "output_archive_md5": md5(args.output_archive) if args.output_archive else None,
                "source_topology": topology,
            },
            "workload": {
                "orientations": requested,
                "packing_cases": 5,
                "packed_ids": packed_ids,
                "sources_per_orientation": int(first_points.shape[0]),
                "q_nodes": q_nodes,
                "unique_theta": phi_count,
                "targets": int(flat_q.shape[0]),
                "streaming": True,
            },
            "affine_lattice_certificate_orientation_1": _certificate_summary(
                reference_certificate
            ),
            "harmonic_support": {
                "max_h": acfo.max_h,
                "tier_cutoffs": acfo.optimization_stats["harmonic_tiers"],
                "kernel_backend": acfo.kernel_backend,
                "miller_recurrence_margin": acfo.miller_recurrence_margin,
                "packing_centers_entered_qR": False,
            },
            "accuracy_only_qualification": qualification,
            "production_gpu_implementation_accuracy_before_timing": (
                production_accuracy
            ),
            "fft_rtx3090_timing_screen": screen,
            "independent_confirmation": confirmation,
            "strongest_baseline_frozen_before_orientation_2": strongest,
            "setup_seconds": {
                "gpu_miller_jit_cold_start": gpu_miller_jit_seconds,
                "acfo_shared_kernel_and_plan": acfo_setup_seconds,
                "acfo_weight_buffer": arms.acfo_buffer_setup_seconds,
                "acfo_total_excluding_jit": acfo_total_setup_seconds,
                "affine_type2_plan": arms.type2_setup_seconds,
                "projected_type3_plan": arms.type3_setup_seconds,
                "dense_periodic_fft_buffers": arms.fft_buffer_setup_seconds,
                "selected_baseline": baseline_setup_seconds,
            },
            "orientation_fourier_samples_seconds": {
                "acfo": timings["acfo"],
                strongest: timings[strongest],
            },
            "descriptive_common_reduction_seconds": {
                "phase": phase_times,
                "acfo": reductions["acfo"],
                strongest: reductions[strongest],
                "data_load_excluded": load_times,
            },
            "cold_total_speedup_selected_baseline_over_acfo": total_speed,
            "acfo_positive_claim_eligible": decision[
                "acfo_positive_claim_eligible"
            ],
            "measured_crossover": crossover_json,
            "prefix_results": prefixes,
            "heldout_accuracy": {
                "rows": heldout_rows,
                "worst_amplitude_relative_l2": heldout_worst_amplitude,
                "worst_output_relative_l2": heldout_worst_output,
            },
            "full_ensemble_acfo_vs_selected_baseline": final_errors,
            "archived_output_validation": archive_validation,
            "gates": {
                "complete_frozen_fft_frontier": not qualification[
                    "resource_limited_frontier"
                ],
                "screen_confirmation_independent": confirmation[
                    "screen_sample_reuse_count"
                ]
                == 0,
                "qualification_pass": qualification_pass,
                "archive_pass": archive_pass,
                "machine_pass": machine_pass,
                "contract_pass": decision["contract_pass"],
            },
        }
    finally:
        source.close()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "NUMAGSANS_EXAMPLE3_STRONGEST_BASELINE_PROTOCOL.json",
    )
    p.add_argument("--source-archive", type=Path)
    p.add_argument("--output-archive", type=Path)
    p.add_argument(
        "--reduced-dir",
        type=Path,
        default=ROOT / "inputs/numagsans_example3_reduced",
    )
    p.add_argument("--max-orientations", type=int)
    p.add_argument("--smoke-q-nodes", type=int)
    p.add_argument("--smoke-unique-theta", type=int)
    p.add_argument("--maximum-fft-pad-factor", type=int)
    p.add_argument("--skip-archive-oracle", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "benchmark_results/numagsans_example3_strongest_baseline.json",
    )
    return p


def main() -> int:
    args = parser().parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["verdict"].startswith(("PASS", "SMOKE")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
