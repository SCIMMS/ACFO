"""Phase-B GPU benchmark for fused ACFO magnetic SANS.

The workload is the archived NuMagSANS Example 2 detector (1000 radial rows,
999 unique periodic angles).  ACFO, cuFINUFFT and a regular-grid GPU FFT all
include the twelve detector-local polarization contractions.  An optional
patched upstream NuMagSANS process provides an interactive CUDA-kernel baseline
without charging file I/O or CSV export to the direct method.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
from time import perf_counter
from typing import Any, Callable

import numpy as np
from scipy.fft import next_fast_len


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from waxs_cake.axisymmetric_manifold import AxisymmetricManifold  # noqa: E402
from waxs_cake.magnetic_sans import (  # noqa: E402
    PreparedBandlimitedMagneticSansOperator,
    load_numagsans_fourier_sources,
)
from waxs_cake.magnetic_sans_torch import (  # noqa: E402
    TorchFlatDetectorBandlimitedMagneticSansOperator,
)
from waxs_cake.voxel_fft_torch import (  # noqa: E402
    TorchPreparedPlanarFFTInterpolator,
    TorchPreparedVoxelFFTInterpolator,
)


COMMON_OUTPUT_NAMES = (
    "S_N",
    "S_M",
    "S_NM",
    "S_P",
    "S_chi",
    "S_sf",
    "S_pm",
    "S_mp",
    "S_pp",
    "S_mm",
    "S_p",
    "S_m",
)


def timing_sample_provenance_from_environment(
    sample_count: int,
) -> dict[str, Any] | None:
    phase = os.environ.get("ACFO_TIMING_PHASE")
    invocation_id = os.environ.get(
        "ACFO_TIMING_SUBPROCESS_INVOCATION_ID"
    )
    if not phase or not invocation_id:
        return None
    sample_set_id = f"{phase}:{invocation_id}"
    return {
        "schema": "acfo-timing-sample-provenance-v1",
        "phase": phase,
        "execution_model": "separate_subprocess",
        "subprocess_invocation_id": invocation_id,
        "sample_set_id": sample_set_id,
        "sample_ids": [
            f"{sample_set_id}:abba_pair:{index:03d}"
            for index in range(sample_count)
        ],
        "samples_per_arm": sample_count,
    }


def rotate_points_about_z(points: np.ndarray, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    result = np.array(points, copy=True)
    result[:, 0] = c * points[:, 0] - s * points[:, 1]
    result[:, 1] = s * points[:, 0] + c * points[:, 1]
    return result


def detector_q_grid(q_radius: np.ndarray, phi: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            q_radius[:, None] * np.cos(phi)[None, :],
            q_radius[:, None] * np.sin(phi)[None, :],
            np.zeros((q_radius.size, phi.size)),
        ],
        axis=-1,
    )


def regular_grid_geometry(points: np.ndarray) -> dict[str, Any]:
    """Recover the exact Cartesian lattice contract of the active source sites."""

    points_arr = np.asarray(points, dtype=np.float64)
    if points_arr.ndim != 2 or points_arr.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    lower = np.min(points_arr, axis=0)
    upper = np.max(points_arr, axis=0)
    unique_axes = tuple(np.unique(points_arr[:, axis]) for axis in range(3))
    if any(axis.size < 2 for axis in unique_axes):
        raise ValueError("each regular-grid axis must contain at least two coordinates")
    spacing = np.asarray([np.median(np.diff(axis)) for axis in unique_axes])
    if np.any(spacing <= 0.0):
        raise ValueError("regular-grid spacing must be positive")
    shape = np.rint((upper - lower) / spacing).astype(int) + 1
    dense_sites = int(np.prod(shape, dtype=np.int64))
    maximum_dense_sites = max(1_000_000, 64 * int(points_arr.shape[0]))
    if dense_sites > maximum_dense_sites:
        raise ValueError(
            "inferred Cartesian lattice is pathologically sparse; use a "
            "nonuniform baseline instead: "
            f"dense_sites={dense_sites}, active_sites={points_arr.shape[0]}"
        )
    indices = np.rint((points_arr - lower) / spacing).astype(int)
    reconstructed = lower[None, :] + indices * spacing[None, :]
    residual = float(np.max(np.abs(reconstructed - points_arr)))
    tolerance = 1e-9 * max(1.0, float(np.max(np.abs(points_arr))))
    if residual > tolerance:
        raise ValueError(
            f"source coordinates are not an exact Cartesian lattice: residual={residual}"
        )
    if np.any(indices < 0) or np.any(indices >= shape[None, :]):
        raise ValueError("regular-grid indices escaped the inferred shape")
    center_index = shape // 2
    center = lower + center_index * spacing
    bounds = tuple(
        (
            float(lower[axis] - 0.5 * spacing[axis]),
            float(upper[axis] + 0.5 * spacing[axis]),
        )
        for axis in range(3)
    )
    return {
        "lower": lower,
        "upper": upper,
        "spacing": spacing,
        "shape": shape,
        "indices": indices,
        "center_index": center_index,
        "center": center,
        "bounds": bounds,
        "cell_volume": float(np.prod(spacing)),
        "maximum_coordinate_residual": residual,
    }


def regular_grid_channels(
    points: np.ndarray, nuclear: np.ndarray, magnetization: np.ndarray
) -> tuple[np.ndarray, tuple[tuple[float, float], ...], float]:
    geometry = regular_grid_geometry(points)
    shape = tuple(int(value) for value in geometry["shape"])
    indices = np.asarray(geometry["indices"])
    channels = np.zeros((4, *shape), dtype=np.float64)
    channels[0, indices[:, 0], indices[:, 1], indices[:, 2]] = nuclear
    for channel in range(3):
        channels[channel + 1, indices[:, 0], indices[:, 1], indices[:, 2]] = magnetization[:, channel]
    return channels, geometry["bounds"], float(geometry["cell_volume"])


def projected_type2_contract(
    channels: np.ndarray,
    geometry: dict[str, Any],
    q_xyz: np.ndarray,
    *,
    eliminated_axis: int = 2,
    zero_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Exactly eliminate a source-grid axis that the detector never probes.

    If every target satisfies ``q[eliminated_axis] == 0``, the Fourier phase is
    independent of that source coordinate.  Summing the regular-grid
    coefficients along the matching axis therefore produces an exact lower-
    dimensional type-2 problem, not an approximation or interpolation.
    """

    channel_arr = np.asarray(channels)
    target_arr = np.asarray(q_xyz, dtype=np.float64).reshape(-1, 3)
    if channel_arr.ndim != 4:
        raise ValueError("channels must have shape (C, nx, ny, nz)")
    axis = int(eliminated_axis)
    if axis not in (0, 1, 2):
        raise ValueError("eliminated_axis must be 0, 1, or 2")
    maximum_eliminated_q = float(np.max(np.abs(target_arr[:, axis])))
    if maximum_eliminated_q > float(zero_tolerance):
        raise ValueError(
            "detector targets are not confined to the projected Fourier plane: "
            f"max |q_axis|={maximum_eliminated_q}"
        )

    kept_axes = tuple(index for index in range(3) if index != axis)
    projected = np.sum(channel_arr, axis=axis + 1)
    spacing = np.asarray(geometry["spacing"], dtype=np.float64)
    center = np.asarray(geometry["center"], dtype=np.float64)
    shape = np.asarray(geometry["shape"], dtype=np.int64)
    scaled_targets = target_arr[:, kept_axes] * spacing[list(kept_axes)][None, :]
    center_phase = np.exp(-1j * (target_arr @ center))
    return {
        "coefficients": np.ascontiguousarray(projected),
        "kept_axes": kept_axes,
        "eliminated_axis": axis,
        "eliminated_grid_size": int(shape[axis]),
        "shape": tuple(int(shape[index]) for index in kept_axes),
        "spacing": spacing[list(kept_axes)],
        "scaled_targets": np.ascontiguousarray(scaled_targets),
        "center_phase": np.ascontiguousarray(center_phase),
        "maximum_eliminated_q": maximum_eliminated_q,
    }


def positive_int_csv(value: str) -> tuple[int, ...]:
    """Parse and de-duplicate a positive integer frontier in declared order."""

    result: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parsed = int(item)
        if parsed <= 0:
            raise argparse.ArgumentTypeError(
                "projected FFT pad factors must be positive"
            )
        if parsed not in result:
            result.append(parsed)
    if not result:
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of positive pad factors"
        )
    return tuple(result)


def select_projected_fft_frontier(
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply the frozen accuracy-first, timing-second FFT selection rule."""

    qualified = [
        row
        for row in candidates
        if row.get("available") is True
        and row.get("accuracy_qualified") is True
        and row.get("protocol_pass") is True
    ]
    selected = (
        min(
            qualified,
            key=lambda row: (
                float(row["baseline_median_seconds"]),
                int(row["pad_factor"]),
            ),
        )
        if qualified
        else None
    )
    return {
        "qualification_rule": (
            "direct complex-amplitude relative L2 <= amplitude gate AND "
            "worst of twelve direct-output relative L2 values <= output gate"
        ),
        "ranking_rule": (
            "minimum baseline hot median among accuracy-qualified candidates; "
            "pad factor breaks exact timing ties"
        ),
        "accuracy_filter_precedes_timing_rank": True,
        "qualified_pad_factors": [
            int(row["pad_factor"]) for row in qualified
        ],
        "selected_pad_factor": (
            int(selected["pad_factor"]) if selected is not None else None
        ),
        "selected_candidate": selected,
    }


def q_cloud_in_original_grid_frame(q_xyz: np.ndarray, source_rotation: float) -> np.ndarray:
    """Undo the ACFO source-coordinate rotation while preserving vector channels."""

    q_original = np.array(q_xyz, dtype=np.float64, copy=True)
    angle = -float(source_rotation)
    c, s = np.cos(angle), np.sin(angle)
    qx = c * q_xyz[..., 0] - s * q_xyz[..., 1]
    qy = s * q_xyz[..., 0] + c * q_xyz[..., 1]
    q_original[..., 0] = qx
    q_original[..., 1] = qy
    return q_original


def direct_fourier_channel_subset(
    points: np.ndarray,
    weights: np.ndarray,
    q_xyz: np.ndarray,
    target_indices: np.ndarray,
) -> np.ndarray:
    """Independent complex128 exponent sums for a small target subset."""

    selected_q = np.asarray(q_xyz, dtype=np.float64).reshape(-1, 3)[target_indices]
    phase = np.exp(-1j * (selected_q @ np.asarray(points, dtype=np.float64).T))
    return np.asarray(weights, dtype=np.complex128) @ phase.T


def amortized_speedup_table(
    *,
    acfo_setup_s: float,
    baseline_setup_s: float,
    acfo_hot_s: float,
    baseline_hot_s: float,
    reuse_counts: tuple[int, ...] = (1, 10, 100, 1000),
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for count in reuse_counts:
        acfo_total = float(acfo_setup_s) + int(count) * float(acfo_hot_s)
        baseline_total = float(baseline_setup_s) + int(count) * float(baseline_hot_s)
        rows.append(
            {
                "updates": int(count),
                "acfo_total_s": acfo_total,
                "baseline_total_s": baseline_total,
                "baseline_over_acfo_speedup": baseline_total / acfo_total,
            }
        )
    return rows


def relative_l2(actual: Any, reference: Any) -> float:
    a = np.asarray(actual)
    b = np.asarray(reference)
    return float(np.linalg.norm(a - b) / max(float(np.linalg.norm(b)), 1e-300))


def torch_magnetic_amplitudes_to_channel_numpy(amplitudes: Any) -> np.ndarray:
    """Return the torch amplitude container as (N, Mx, My, Mz) channels."""

    nuclear = amplitudes.nuclear
    magnetization = amplitudes.magnetization
    if nuclear.ndim != 2 or magnetization.ndim != 3:
        raise ValueError("magnetic amplitude tensors have the wrong rank")
    if tuple(magnetization.shape[:2]) != tuple(nuclear.shape):
        raise ValueError("nuclear and magnetic detector shapes do not match")
    if int(magnetization.shape[-1]) != 3:
        raise ValueError("magnetization must have three Cartesian components")
    channels = nuclear.new_empty(
        (4, *nuclear.shape), dtype=nuclear.dtype, device=nuclear.device
    )
    channels[0] = nuclear
    channels[1:] = magnetization.permute(2, 0, 1)
    return channels.detach().cpu().numpy()


def sync(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def torch_cross_sections(torch: Any, amplitudes: Any, q_hat: Any, polarization: Any) -> dict[str, Any]:
    nuclear_amp = amplitudes[0]
    magnetization = amplitudes[1:].permute(1, 2, 0)
    q_hat = q_hat.unsqueeze(0)
    interaction = q_hat * torch.sum(q_hat * magnetization, dim=-1, keepdim=True) - magnetization
    p = polarization.to(amplitudes.dtype)
    projected_amp = torch.einsum("qpi,i->qp", interaction, p)
    nuclear = torch.abs(nuclear_amp) ** 2
    magnetic = torch.sum(torch.abs(interaction) ** 2, dim=-1)
    projected = torch.abs(projected_amp) ** 2
    nuclear_magnetic = 2.0 * torch.real(nuclear_amp * torch.conj(projected_amp))
    spin_flip = magnetic - projected
    chiral = torch.real(
        -1j
        * torch.einsum(
            "i,qpi->qp", p, torch.linalg.cross(interaction, torch.conj(interaction))
        )
    )
    pm = spin_flip + chiral
    mp = spin_flip - chiral
    pp = nuclear + nuclear_magnetic + projected
    mm = nuclear - nuclear_magnetic + projected
    return {
        "S_N": nuclear,
        "S_M": magnetic,
        "S_NM": nuclear_magnetic,
        "S_P": projected,
        "S_chi": chiral,
        "S_sf": spin_flip,
        "S_pm": pm,
        "S_mp": mp,
        "S_pp": pp,
        "S_mm": mm,
        "S_p": pp + pm,
        "S_m": mm + mp,
    }


def cupy_cross_sections(cp: Any, amplitudes: Any, q_hat: Any, polarization: Any) -> dict[str, Any]:
    nuclear_amp = amplitudes[0]
    magnetization = cp.moveaxis(amplitudes[1:], 0, -1)
    interaction = q_hat[None, :, :] * cp.sum(q_hat[None, :, :] * magnetization, axis=-1, keepdims=True) - magnetization
    projected_amp = cp.einsum("qpi,i->qp", interaction, polarization)
    nuclear = cp.abs(nuclear_amp) ** 2
    magnetic = cp.sum(cp.abs(interaction) ** 2, axis=-1)
    projected = cp.abs(projected_amp) ** 2
    nuclear_magnetic = 2.0 * cp.real(nuclear_amp * cp.conj(projected_amp))
    spin_flip = magnetic - projected
    chiral = cp.real(-1j * cp.einsum("i,qpi->qp", polarization, cp.cross(interaction, cp.conj(interaction))))
    pm = spin_flip + chiral
    mp = spin_flip - chiral
    pp = nuclear + nuclear_magnetic + projected
    mm = nuclear - nuclear_magnetic + projected
    return {
        "S_N": nuclear,
        "S_M": magnetic,
        "S_NM": nuclear_magnetic,
        "S_P": projected,
        "S_chi": chiral,
        "S_sf": spin_flip,
        "S_pm": pm,
        "S_mp": mp,
        "S_pp": pp,
        "S_mm": mm,
        "S_p": pp + pm,
        "S_m": mm + mp,
    }


def numpy_cross_sections_flat(
    amplitudes: np.ndarray,
    q_hat: np.ndarray,
    polarization: np.ndarray,
) -> dict[str, np.ndarray]:
    """Evaluate the common twelve outputs for a flat target subset."""

    values = np.asarray(amplitudes)
    directions = np.asarray(q_hat, dtype=np.float64)
    p = np.asarray(polarization, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 4:
        raise ValueError("amplitudes must have shape (4, targets)")
    if directions.shape != (values.shape[1], 3):
        raise ValueError("q_hat must have shape (targets, 3)")
    nuclear_amp = values[0]
    magnetization = values[1:].T
    interaction = (
        directions
        * np.sum(directions * magnetization, axis=-1, keepdims=True)
        - magnetization
    )
    projected_amp = np.einsum("qi,i->q", interaction, p)
    nuclear = np.abs(nuclear_amp) ** 2
    magnetic = np.sum(np.abs(interaction) ** 2, axis=-1)
    projected = np.abs(projected_amp) ** 2
    nuclear_magnetic = 2.0 * np.real(nuclear_amp * np.conj(projected_amp))
    spin_flip = magnetic - projected
    chiral = np.real(
        -1j * np.einsum("i,qi->q", p, np.cross(interaction, np.conj(interaction)))
    )
    pm = spin_flip + chiral
    mp = spin_flip - chiral
    pp = nuclear + nuclear_magnetic + projected
    mm = nuclear - nuclear_magnetic + projected
    return {
        "S_N": nuclear,
        "S_M": magnetic,
        "S_NM": nuclear_magnetic,
        "S_P": projected,
        "S_chi": chiral,
        "S_sf": spin_flip,
        "S_pm": pm,
        "S_mp": mp,
        "S_pp": pp,
        "S_mm": mm,
        "S_p": pp + pm,
        "S_m": mm + mp,
    }


def abba(
    torch: Any,
    device: Any,
    first: Callable[[], Any],
    second: Callable[[], Any],
    *,
    warmups: int,
    samples: int,
) -> tuple[list[float], list[float], Any, Any]:
    if samples % 2:
        raise ValueError("samples must be even")
    for index in range(warmups * 2):
        _ = (first if index % 2 == 0 else second)()
        sync(torch, device)
    a_samples: list[float] = []
    b_samples: list[float] = []
    a_value = b_value = None
    for _ in range(samples // 2):
        for label, func in (("A", first), ("B", second), ("B", second), ("A", first)):
            sync(torch, device)
            start = perf_counter()
            value = func()
            sync(torch, device)
            elapsed = perf_counter() - start
            if label == "A":
                a_samples.append(elapsed)
                a_value = value
            else:
                b_samples.append(elapsed)
                b_value = value
    return a_samples, b_samples, a_value, b_value


def speedup_interval(candidate: list[float], baseline: list[float], seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    a = np.asarray(candidate)
    b = np.asarray(baseline)
    values = np.empty(4000)
    for index in range(values.size):
        values[index] = np.median(rng.choice(b, b.size, replace=True)) / np.median(
            rng.choice(a, a.size, replace=True)
        )
    return {
        "point": float(np.median(b) / np.median(a)),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


class NuMagSANSBenchmarkServer:
    def __init__(
        self,
        executable: Path,
        config: Path,
        *,
        common_output: bool = False,
    ) -> None:
        startup_start = perf_counter()
        self.common_output = bool(common_output)
        env = dict(os.environ)
        env["NUMAGSANS_BENCHMARK_SERVER"] = "1"
        self.process = subprocess.Popen(
            [str(executable.resolve()), str(config.resolve())],
            cwd=str(config.resolve().parent),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        self._lines: queue.Queue[str | None] = queue.Queue()
        self.transcript: list[str] = []
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()
        try:
            self._wait_until_ready()
        except Exception:
            self.close()
            raise
        self.startup_seconds = perf_counter() - startup_start

    def _read_stdout(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self._lines.put(line.rstrip("\n"))
        self._lines.put(None)

    def _next_line(self, timeout: float) -> str:
        try:
            line = self._lines.get(timeout=timeout)
        except queue.Empty as exc:
            raise RuntimeError(
                "timed out waiting for NuMagSANS benchmark-server output; "
                f"returncode={self.process.poll()}; transcript_tail={self.transcript[-40:]}"
            ) from exc
        if line is None:
            raise RuntimeError(
                "NuMagSANS benchmark server terminated before the expected marker; "
                f"returncode={self.process.poll()}; transcript_tail={self.transcript[-40:]}"
            )
        self.transcript.append(line)
        return line

    def _wait_until_ready(self) -> None:
        marker = (
            "NUMAGSANS_COMMON_OUTPUT_READY"
            if self.common_output
            else "NUMAGSANS_BENCHMARK_READY"
        )
        while True:
            line = self._next_line(180.0)
            if marker in line:
                return

    def sample(self) -> float:
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("NuMagSANS server pipes are unavailable")
        try:
            command = "RUN_COMMON" if self.common_output else "RUN"
            self.process.stdin.write(command + "\n")
            self.process.stdin.flush()
        except BrokenPipeError as exc:
            raise RuntimeError(
                "NuMagSANS benchmark server closed its command pipe; "
                f"returncode={self.process.poll()}; transcript_tail={self.transcript[-40:]}"
            ) from exc
        while True:
            line = self._next_line(10800.0)
            marker = (
                "NUMAGSANS_COMMON_SECONDS="
                if self.common_output
                else "NUMAGSANS_BENCHMARK_SECONDS="
            )
            if marker in line:
                return float(line.split(marker, 1)[1].strip())

    def sample_upstream(self) -> float:
        if not self.common_output:
            raise RuntimeError("upstream diagnostic requires the common-output server")
        if self.process.stdin is None:
            raise RuntimeError("NuMagSANS server command pipe is unavailable")
        self.process.stdin.write("RUN_UPSTREAM\n")
        self.process.stdin.flush()
        while True:
            line = self._next_line(10800.0)
            marker = "NUMAGSANS_UPSTREAM_SECONDS="
            if marker in line:
                return float(line.split(marker, 1)[1].strip())

    def probe(self, indices: np.ndarray) -> dict[str, Any]:
        if not self.common_output:
            raise RuntimeError("common-output probe requires the common-output server")
        if self.process.stdin is None:
            raise RuntimeError("NuMagSANS server command pipe is unavailable")
        requested = np.asarray(indices, dtype=np.int64).reshape(-1)
        if requested.size == 0 or np.any(requested < 0):
            raise ValueError("probe indices must be a non-empty nonnegative array")
        command = "PROBE_COMMON " + " ".join(str(int(value)) for value in requested)
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()
        normalization: float | None = None
        output_names: tuple[str, ...] | None = None
        target_count: int | None = None
        rows: dict[int, list[float]] = {}
        while True:
            line = self._next_line(10800.0)
            if line.startswith("NUMAGSANS_COMMON_NORMALIZATION="):
                normalization = float(line.split("=", 1)[1])
            elif line.startswith("NUMAGSANS_COMMON_OUTPUTS="):
                output_names = tuple(line.split("=", 1)[1].split(","))
            elif line.startswith("NUMAGSANS_COMMON_TARGETS="):
                target_count = int(line.split("=", 1)[1])
            elif line.startswith("NUMAGSANS_COMMON_PROBE="):
                fields = line.split("=", 1)[1].split(",")
                rows[int(fields[0])] = [float(value) for value in fields[1:]]
            elif line.startswith("NUMAGSANS_COMMON_PROBE_ERROR="):
                raise RuntimeError(line)
            elif line.startswith("NUMAGSANS_COMMON_PROBE_DONE="):
                break
        if normalization is None or not np.isfinite(normalization) or normalization <= 0.0:
            raise RuntimeError("NuMagSANS common-output normalization was not reported")
        if output_names != COMMON_OUTPUT_NAMES:
            raise RuntimeError(
                f"unexpected NuMagSANS output contract: {output_names!r}"
            )
        if target_count is None or target_count <= 0:
            raise RuntimeError("NuMagSANS common-output target count was not reported")
        if set(rows) != {int(value) for value in requested}:
            raise RuntimeError("NuMagSANS common-output probe is incomplete")
        values = np.asarray([rows[int(index)] for index in requested], dtype=np.float64)
        if values.shape != (requested.size, len(COMMON_OUTPUT_NAMES)):
            raise RuntimeError(f"unexpected common-output probe shape: {values.shape}")
        return {
            "indices": requested,
            "normalization": normalization,
            "output_names": output_names,
            "target_count": target_count,
            "values": values,
        }

    def close(self) -> None:
        if self.process.stdin is not None and self.process.poll() is None:
            try:
                self.process.stdin.write("QUIT\n")
                self.process.stdin.flush()
            except BrokenPipeError:
                pass
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()


def numagsans_abba(
    torch: Any,
    device: Any,
    acfo: Callable[[], Any],
    server: NuMagSANSBenchmarkServer,
    *,
    warmups: int,
    samples: int,
) -> tuple[list[float], list[float]]:
    for index in range(warmups * 2):
        if index % 2 == 0:
            _ = acfo()
            sync(torch, device)
        else:
            _ = server.sample()
    a: list[float] = []
    b: list[float] = []
    for _ in range(samples // 2):
        for label in ("A", "B", "B", "A"):
            if label == "A":
                sync(torch, device)
                start = perf_counter()
                _ = acfo()
                sync(torch, device)
                a.append(perf_counter() - start)
            else:
                b.append(server.sample())
    return a, b


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if args.warmups < 0 or args.samples <= 0 or args.samples % 2:
        raise ValueError("warmups must be non-negative and samples must be positive/even")
    if args.direct_target_count <= 0:
        raise ValueError("direct_target_count must be positive")
    if args.type2_direct_tolerance <= 0.0 or args.accuracy_tolerance <= 0.0:
        raise ValueError("accuracy tolerances must be positive")
    if args.numagsans_common_direct_tolerance <= 0.0:
        raise ValueError("NuMagSANS common-output direct tolerance must be positive")
    if args.numagsans_common_output and args.numagsans_executable is None:
        raise ValueError(
            "--numagsans-common-output requires --numagsans-executable"
        )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.cuda.set_device(device)
    points, nuclear, magnetization = load_numagsans_fourier_sources(args.magnetic, args.nuclear)
    points = np.ascontiguousarray(points * float(args.source_coordinate_scale))
    unique_phi = args.theta_nodes - 1
    phi_step = 2.0 * np.pi / unique_phi
    source_rotation = 0.5 * phi_step - 0.5 * np.pi
    rotated_points = rotate_points_about_z(points, source_rotation)
    q_radius = np.linspace(0.0, args.q_max, args.q_nodes)
    manifold = AxisymmetricManifold(
        u=np.arange(args.q_nodes, dtype=float),
        q_perp=q_radius,
        q_z=np.zeros_like(q_radius),
        name="numagsans-example2-phase-b",
        interpretation="sampling",
    )
    radial_extent = float(np.max(np.hypot(points[:, 0], points[:, 1])))
    z_range = (float(np.min(points[:, 2]) - 0.5), float(np.max(points[:, 2]) + 0.5))
    gpu_miller_jit_seconds = None
    if args.acfo_kernel_backend == "gpu_miller":
        from waxs_cake.gpu_miller import warm_gpu_miller_kernel

        gpu_miller_jit_seconds = warm_gpu_miller_kernel()
    setup_start = perf_counter()
    cpu_prepared = PreparedBandlimitedMagneticSansOperator(
        rotated_points,
        nuclear,
        magnetization,
        manifold,
        n_r=args.acfo_n_r,
        n_z=args.acfo_n_z,
        n_phi=unique_phi,
        r_max=radial_extent + 0.5,
        z_range=z_range,
        phase_sign=-1,
        hist_backend="numpy",
        circular_backend="auto",
        complex_dtype=np.complex64 if args.dtype == "complex64" else np.complex128,
        margin=args.harmonic_margin,
    )
    cpu_histogram_and_plan_setup_seconds = perf_counter() - setup_start
    phi = np.asarray(cpu_prepared.phi)
    q_xyz = detector_q_grid(q_radius, phi)
    emitted_q_hat_np = np.stack(
        [np.cos(phi), np.sin(phi), np.zeros_like(phi)], axis=-1
    )
    # The angular source shift aligns bin-centred ACFO samples with upstream
    # theta nodes but does not rotate magnetization components.  Contract the
    # vector amplitudes in their original, permuted NuMagSANS component frame.
    q_hat_np = q_cloud_in_original_grid_frame(
        emitted_q_hat_np, source_rotation
    )
    torch_kernel_setup_start = perf_counter()
    acfo = TorchFlatDetectorBandlimitedMagneticSansOperator(
        cpu_prepared,
        torch=torch,
        device=device,
        dtype=args.dtype,
        detector_directions=q_hat_np,
        source_deposition=args.source_deposition,
        kernel_backend=args.acfo_kernel_backend,
    )
    sync(torch, device)
    torch_kernel_and_pack_setup_seconds = (
        perf_counter() - torch_kernel_setup_start
    )
    if (
        args.require_miller_kernel
        and acfo.kernel_backend
        not in {
            "cpp_miller_downward_recurrence",
            "gpu_miller_downward_recurrence",
        }
    ):
        raise RuntimeError(
            "production Miller kernel is required, but the selected backend is "
            f"{acfo.kernel_backend}"
        )
    source_update_map_setup_start = perf_counter()
    acfo.configure_source_updates(rotated_points)
    sync(torch, device)
    source_update_map_setup_seconds = (
        perf_counter() - source_update_map_setup_start
    )
    acfo_setup_seconds = perf_counter() - setup_start
    gpu_miller_parity = None
    if args.acfo_kernel_backend == "gpu_miller":
        from waxs_cake.gpu_miller import gpu_miller_kernel64

        template = cpu_prepared.channel_plans[0]
        q_probe = np.unique(
            np.linspace(0, acfo.n_q - 1, 16, dtype=np.int64)
        )
        r_probe_local = np.unique(
            np.linspace(0, acfo.n_active_r - 1, 16, dtype=np.int64)
        )
        r_probe = acfo.active_r_indices[r_probe_local]
        cpu_probe = template._analytic_kernel_hat_modes_r(
            q_probe,
            r_probe,
            acfo.max_h,
        )
        gpu_probe = gpu_miller_kernel64(
            template.q_perp[q_probe],
            template.binned.r_centers[r_probe],
            n_phi=acfo.n_phi,
            max_cutoff=acfo.max_h,
            extra_order=64,
            torch=torch,
        )
        sync(torch, device)
        gpu_probe_np = gpu_probe.detach().cpu().numpy()
        gpu_miller_parity = {
            "q_count": int(q_probe.size),
            "radius_count": int(r_probe.size),
            "coefficient_count": int(cpu_probe.size),
            "relative_l2_vs_compiled_cpu_miller": relative_l2(
                gpu_probe_np.astype(np.complex128, copy=False),
                cpu_probe.astype(np.complex128, copy=False),
            ),
            "maximum_absolute_error_vs_compiled_cpu_miller": float(
                np.max(np.abs(gpu_probe_np - cpu_probe))
            ),
        }
    real_dtype = torch.float32 if args.dtype == "complex64" else torch.float64
    weights_np = np.ascontiguousarray(np.column_stack([nuclear, magnetization]).T)
    weights_torch = torch.as_tensor(weights_np, dtype=real_dtype, device=device)
    polarization_torch = torch.as_tensor([0.0, 1.0, 0.0], dtype=real_dtype, device=device)
    acfo_func = lambda: acfo.cross_sections_from_weights(weights_torch, polarization_torch)

    sync(torch, device)
    acfo_first_start = perf_counter()
    _ = acfo_func()
    sync(torch, device)
    acfo_first_seconds = perf_counter() - acfo_first_start

    for _ in range(args.warmups):
        _ = acfo_func()
    sync(torch, device)
    acfo_standalone_samples = []
    for _ in range(args.samples):
        sync(torch, device)
        acfo_sample_start = perf_counter()
        _ = acfo_func()
        sync(torch, device)
        acfo_standalone_samples.append(perf_counter() - acfo_sample_start)

    fixed_source = acfo.source_spectrum_from_weights(weights_torch)
    fixed_amplitudes = acfo._amplitudes_for_source(fixed_source)
    stage_functions = {
        "source_update": lambda: acfo.source_spectrum_from_weights(weights_torch),
        "fourier_and_ifft": lambda: acfo._amplitudes_for_source(fixed_source),
        "polarization_contraction": lambda: acfo._cross_sections_from_amplitudes(
            fixed_amplitudes, polarization_torch
        ),
    }
    stage_medians = {}
    for stage_name, stage_func in stage_functions.items():
        for _ in range(args.warmups):
            _ = stage_func()
        sync(torch, device)
        stage_samples = []
        for _ in range(args.samples):
            sync(torch, device)
            stage_start = perf_counter()
            _ = stage_func()
            sync(torch, device)
            stage_samples.append(perf_counter() - stage_start)
        stage_medians[stage_name] = float(np.median(stage_samples))

    q_hat_torch = torch.as_tensor(q_hat_np, dtype=real_dtype, device=device)
    q_original = q_cloud_in_original_grid_frame(q_xyz, source_rotation)
    grid_geometry = None
    channels = None
    bounds = None
    cell_volume = None
    regular_grid_error = None
    try:
        grid_geometry = regular_grid_geometry(points)
        channels, bounds, cell_volume = regular_grid_channels(
            points, nuclear, magnetization
        )
    except ValueError as exc:
        regular_grid_error = str(exc)
    direct_indices = np.unique(
        np.linspace(
            0,
            q_original.reshape(-1, 3).shape[0] - 1,
            min(int(args.direct_target_count), q_original.reshape(-1, 3).shape[0]),
            dtype=np.int64,
        )
    )
    direct_amplitudes = direct_fourier_channel_subset(
        points,
        weights_np,
        q_original,
        direct_indices,
    )
    q_hat_flat = np.broadcast_to(
        q_hat_np[None, :, :],
        (args.q_nodes, unique_phi, 3),
    ).reshape(-1, 3)
    direct_observables = numpy_cross_sections_flat(
        direct_amplitudes,
        q_hat_flat[direct_indices],
        np.asarray([0.0, 1.0, 0.0]),
    )
    acfo_direct_amplitudes = torch_magnetic_amplitudes_to_channel_numpy(
        fixed_amplitudes
    ).reshape(4, -1)[:, direct_indices]
    acfo_direct_amplitude_error = relative_l2(
        acfo_direct_amplitudes.astype(np.complex128, copy=False),
        direct_amplitudes,
    )
    acfo_direct_observables = numpy_cross_sections_flat(
        acfo_direct_amplitudes,
        q_hat_flat[direct_indices],
        np.asarray([0.0, 1.0, 0.0]),
    )
    acfo_direct_output_errors = {
        name: relative_l2(acfo_direct_observables[name], direct_observables[name])
        for name in COMMON_OUTPUT_NAMES
    }

    # cuFINUFFT is optional locally but mandatory in the frozen external run.
    # Type-3 is retained as a diagnostic oracle; native type-2 is the eligible
    # regular-grid production baseline for this archived Example 2 source.
    cufinufft_type3_result = None
    cufinufft_type3_projected_result = None
    cufinufft_type2_result = None
    cufinufft_type2_projected_result = None
    try:
        if hasattr(os, "add_dll_directory"):
            site_packages = Path(sys.prefix) / "Lib" / "site-packages"
            for candidate in (
                site_packages / "torch" / "lib",
                site_packages / "cufinufft",
                site_packages / "cufinufft.libs",
            ):
                if candidate.exists():
                    os.add_dll_directory(str(candidate))
        import cupy as cp
        import cufinufft
        cp_complex = cp.complex64 if args.dtype == "complex64" else cp.complex128
        cp_real = cp.float32 if args.dtype == "complex64" else cp.float64
        q_hat_cp = cp.asarray(q_hat_np, dtype=cp_real)
        polarization_cp = cp.asarray([0.0, 1.0, 0.0], dtype=cp_complex)
        flat_q = q_xyz.reshape(-1, 3)
        flat_q_original = q_original.reshape(-1, 3)

        try:
            if getattr(args, "skip_type3", False):
                raise RuntimeError("type-3 diagnostic skipped by explicit contract")
            cuf3_setup_start = perf_counter()
            type3_plan = cufinufft.Plan(
                3,
                3,
                n_trans=4,
                eps=args.cufinufft_eps,
                isign=-1,
                dtype=args.dtype,
            )
            x = cp.asarray(rotated_points[:, 0], dtype=cp_real)
            y = cp.asarray(rotated_points[:, 1], dtype=cp_real)
            z = cp.asarray(rotated_points[:, 2], dtype=cp_real)
            s = cp.asarray(flat_q[:, 0], dtype=cp_real)
            t = cp.asarray(flat_q[:, 1], dtype=cp_real)
            u = cp.asarray(flat_q[:, 2], dtype=cp_real)
            type3_plan.setpts(x, y, z, s, t, u)
            strengths = cp.asarray(weights_np, dtype=cp_complex)
            type3_out = cp.empty((4, flat_q.shape[0]), dtype=cp_complex)
            cp.cuda.runtime.deviceSynchronize()
            type3_setup_seconds = perf_counter() - cuf3_setup_start

            def cufinufft_type3_func() -> dict[str, Any]:
                type3_plan.execute(strengths, out=type3_out)
                values = type3_out.reshape(4, args.q_nodes, unique_phi)
                return cupy_cross_sections(cp, values, q_hat_cp, polarization_cp)

            def cuf3_wrapped() -> Any:
                value = cufinufft_type3_func()
                cp.cuda.runtime.deviceSynchronize()
                return value

            cp.cuda.runtime.deviceSynchronize()
            type3_first_start = perf_counter()
            _ = cuf3_wrapped()
            type3_first_seconds = perf_counter() - type3_first_start
            acfo_samples, type3_samples, acfo_value, type3_value = abba(
                torch,
                device,
                acfo_func,
                cuf3_wrapped,
                warmups=args.warmups,
                samples=args.samples,
            )
            type3_cross_errors = {}
            for name, reference in type3_value.items():
                actual = acfo_value[name].detach().cpu().numpy()
                type3_cross_errors[name] = relative_l2(actual, cp.asnumpy(reference))
            type3_direct = cp.asnumpy(type3_out[:, direct_indices])
            type3_hot = float(np.median(type3_samples))
            type3_acfo_hot = float(np.median(acfo_samples))
            cufinufft_type3_result = {
                "available": True,
                "eligible_production_baseline": False,
                "role": "nonuniform-to-nonuniform diagnostic oracle",
                "plan_type": 3,
                "n_trans": 4,
                "setup_seconds": type3_setup_seconds,
                "first_solve_seconds": type3_first_seconds,
                "acfo_samples_seconds": acfo_samples,
                "baseline_samples_seconds": type3_samples,
                "acfo_median_seconds": type3_acfo_hot,
                "baseline_median_seconds": type3_hot,
                "baseline_over_acfo_speedup": speedup_interval(
                    acfo_samples, type3_samples, 20260816
                ),
                "cross_section_relative_l2": type3_cross_errors,
                "worst_cross_section_relative_l2": max(type3_cross_errors.values()),
                "direct_subset_target_count": int(direct_indices.size),
                "amplitude_relative_l2_vs_direct_subset": relative_l2(
                    type3_direct.astype(np.complex128, copy=False), direct_amplitudes
                ),
                "eps": args.cufinufft_eps,
                "amortized": amortized_speedup_table(
                    acfo_setup_s=acfo_setup_seconds,
                    baseline_setup_s=type3_setup_seconds,
                    acfo_hot_s=type3_acfo_hot,
                    baseline_hot_s=type3_hot,
                ),
            }
        except Exception as exc:
            cufinufft_type3_result = {"available": False, "error": repr(exc)}

        try:
            if getattr(args, "skip_type3", False):
                raise RuntimeError(
                    "projected type-3 diagnostic skipped by explicit contract"
                )
            projected_type3_setup_start = perf_counter()
            projected_type3_plan = cufinufft.Plan(
                3,
                2,
                n_trans=4,
                eps=args.cufinufft_eps,
                isign=-1,
                dtype=args.dtype,
            )
            projected_source_x = cp.asarray(
                rotated_points[:, 0], dtype=cp_real
            )
            projected_source_y = cp.asarray(
                rotated_points[:, 1], dtype=cp_real
            )
            projected_target_x = cp.asarray(flat_q[:, 0], dtype=cp_real)
            projected_target_y = cp.asarray(flat_q[:, 1], dtype=cp_real)
            projected_type3_plan.setpts(
                projected_source_x,
                projected_source_y,
                s=projected_target_x,
                t=projected_target_y,
            )
            projected_type3_strengths = cp.asarray(
                weights_np, dtype=cp_complex
            )
            projected_type3_out = cp.empty(
                (4, flat_q.shape[0]), dtype=cp_complex
            )
            cp.cuda.runtime.deviceSynchronize()
            projected_type3_setup_seconds = (
                perf_counter() - projected_type3_setup_start
            )

            def cufinufft_type3_projected_func() -> dict[str, Any]:
                projected_type3_plan.execute(
                    projected_type3_strengths, out=projected_type3_out
                )
                values = projected_type3_out.reshape(
                    4, args.q_nodes, unique_phi
                )
                return cupy_cross_sections(
                    cp, values, q_hat_cp, polarization_cp
                )

            def projected_type3_wrapped() -> Any:
                value = cufinufft_type3_projected_func()
                cp.cuda.runtime.deviceSynchronize()
                return value

            cp.cuda.runtime.deviceSynchronize()
            projected_type3_first_start = perf_counter()
            _ = projected_type3_wrapped()
            projected_type3_first_seconds = (
                perf_counter() - projected_type3_first_start
            )
            (
                projected_type3_acfo_samples,
                projected_type3_samples,
                projected_type3_acfo_value,
                projected_type3_value,
            ) = abba(
                torch,
                device,
                acfo_func,
                projected_type3_wrapped,
                warmups=args.warmups,
                samples=args.samples,
            )
            projected_type3_cross_errors = {}
            for name, reference in projected_type3_value.items():
                actual = projected_type3_acfo_value[name].detach().cpu().numpy()
                projected_type3_cross_errors[name] = relative_l2(
                    actual, cp.asnumpy(reference)
                )
            projected_type3_direct = cp.asnumpy(
                projected_type3_out[:, direct_indices]
            )
            projected_type3_hot = float(np.median(projected_type3_samples))
            projected_type3_acfo_hot = float(
                np.median(projected_type3_acfo_samples)
            )
            projected_type3_direct_error = relative_l2(
                projected_type3_direct.astype(np.complex128, copy=False),
                direct_amplitudes,
            )
            projected_type3_direct_observables = numpy_cross_sections_flat(
                projected_type3_direct,
                q_hat_flat[direct_indices],
                np.asarray([0.0, 1.0, 0.0]),
            )
            projected_type3_output_errors = {
                name: relative_l2(
                    projected_type3_direct_observables[name],
                    direct_observables[name],
                )
                for name in COMMON_OUTPUT_NAMES
            }
            cufinufft_type3_projected_result = {
                "available": True,
                "eligible_production_baseline": True,
                "role": (
                    "exact flat-detector projection followed by 2-D "
                    "nonuniform-to-nonuniform type-3"
                ),
                "plan_type": 3,
                "dimension": 2,
                "n_trans": 4,
                "exact_contraction": True,
                "maximum_eliminated_q": float(
                    np.max(np.abs(flat_q[:, 2]))
                ),
                "setup_seconds": projected_type3_setup_seconds,
                "first_solve_seconds": projected_type3_first_seconds,
                "acfo_samples_seconds": projected_type3_acfo_samples,
                "baseline_samples_seconds": projected_type3_samples,
                "acfo_median_seconds": projected_type3_acfo_hot,
                "baseline_median_seconds": projected_type3_hot,
                "baseline_over_acfo_speedup": speedup_interval(
                    projected_type3_acfo_samples,
                    projected_type3_samples,
                    20260824,
                ),
                "cross_section_relative_l2": (
                    projected_type3_cross_errors
                ),
                "worst_cross_section_relative_l2": max(
                    projected_type3_cross_errors.values()
                ),
                "direct_subset_target_count": int(direct_indices.size),
                "amplitude_relative_l2_vs_direct_subset": (
                    projected_type3_direct_error
                ),
                "direct_subset_output_relative_l2": (
                    projected_type3_output_errors
                ),
                "worst_direct_subset_output_relative_l2": max(
                    projected_type3_output_errors.values()
                ),
                "direct_subset_tolerance": args.type2_direct_tolerance,
                "direct_subset_accuracy_pass": bool(
                    projected_type3_direct_error
                    <= args.type2_direct_tolerance
                ),
                "eps": args.cufinufft_eps,
                "amortized": amortized_speedup_table(
                    acfo_setup_s=acfo_setup_seconds,
                    baseline_setup_s=projected_type3_setup_seconds,
                    acfo_hot_s=projected_type3_acfo_hot,
                    baseline_hot_s=projected_type3_hot,
                ),
            }
        except Exception as exc:
            cufinufft_type3_projected_result = {
                "available": False,
                "error": repr(exc),
            }

        try:
            if grid_geometry is None or channels is None:
                raise RuntimeError(
                    "native type-2 unavailable for non-Cartesian source: "
                    f"{regular_grid_error}"
                )
            type2_shape = tuple(int(value) for value in grid_geometry["shape"])
            type2_setup_start = perf_counter()
            type2_plan = cufinufft.Plan(
                2,
                type2_shape,
                n_trans=4,
                eps=args.cufinufft_eps,
                isign=-1,
                dtype=args.dtype,
                modeord=0,
            )
            spacing = np.asarray(grid_geometry["spacing"], dtype=np.float64)
            scaled_q = flat_q_original * spacing[None, :]
            type2_x = cp.asarray(scaled_q[:, 0], dtype=cp_real)
            type2_y = cp.asarray(scaled_q[:, 1], dtype=cp_real)
            type2_z = cp.asarray(scaled_q[:, 2], dtype=cp_real)
            type2_plan.setpts(type2_x, type2_y, type2_z)
            type2_coefficients = cp.asarray(channels, dtype=cp_complex)
            type2_out = cp.empty((4, flat_q_original.shape[0]), dtype=cp_complex)
            center = np.asarray(grid_geometry["center"], dtype=np.float64)
            center_phase = cp.exp(
                -1j * cp.asarray(flat_q_original @ center, dtype=cp_real)
            ).astype(cp_complex, copy=False)
            cp.cuda.runtime.deviceSynchronize()
            type2_setup_seconds = perf_counter() - type2_setup_start

            def cufinufft_type2_func() -> dict[str, Any]:
                type2_plan.execute(type2_coefficients, out=type2_out)
                corrected = type2_out * center_phase[None, :]
                values = corrected.reshape(4, args.q_nodes, unique_phi)
                return cupy_cross_sections(cp, values, q_hat_cp, polarization_cp)

            def cuf2_wrapped() -> Any:
                value = cufinufft_type2_func()
                cp.cuda.runtime.deviceSynchronize()
                return value

            cp.cuda.runtime.deviceSynchronize()
            type2_first_start = perf_counter()
            _ = cuf2_wrapped()
            type2_first_seconds = perf_counter() - type2_first_start
            type2_acfo_samples, type2_samples, type2_acfo_value, type2_value = abba(
                torch,
                device,
                acfo_func,
                cuf2_wrapped,
                warmups=args.warmups,
                samples=args.samples,
            )
            type2_cross_errors = {}
            for name, reference in type2_value.items():
                actual = type2_acfo_value[name].detach().cpu().numpy()
                type2_cross_errors[name] = relative_l2(actual, cp.asnumpy(reference))
            corrected_type2 = type2_out * center_phase[None, :]
            type2_direct = cp.asnumpy(corrected_type2[:, direct_indices])
            type2_hot = float(np.median(type2_samples))
            type2_acfo_hot = float(np.median(type2_acfo_samples))
            type2_direct_error = relative_l2(
                type2_direct.astype(np.complex128, copy=False), direct_amplitudes
            )
            type2_direct_observables = numpy_cross_sections_flat(
                type2_direct,
                q_hat_flat[direct_indices],
                np.asarray([0.0, 1.0, 0.0]),
            )
            type2_direct_output_errors = {
                name: relative_l2(
                    type2_direct_observables[name], direct_observables[name]
                )
                for name in COMMON_OUTPUT_NAMES
            }
            cufinufft_type2_result = {
                "available": True,
                "eligible_production_baseline": True,
                "role": "native regular-grid coefficients to nonuniform detector targets",
                "plan_type": 2,
                "modeord": 0,
                "n_trans": 4,
                "grid_shape": list(type2_shape),
                "grid_spacing": spacing.tolist(),
                "grid_center": center.tolist(),
                "scaled_target_coordinate_max_abs": float(np.max(np.abs(scaled_q))),
                "setup_seconds": type2_setup_seconds,
                "first_solve_seconds": type2_first_seconds,
                "acfo_samples_seconds": type2_acfo_samples,
                "baseline_samples_seconds": type2_samples,
                "acfo_median_seconds": type2_acfo_hot,
                "baseline_median_seconds": type2_hot,
                "baseline_over_acfo_speedup": speedup_interval(
                    type2_acfo_samples, type2_samples, 20260820
                ),
                "cross_section_relative_l2": type2_cross_errors,
                "worst_cross_section_relative_l2": max(type2_cross_errors.values()),
                "direct_subset_target_count": int(direct_indices.size),
                "amplitude_relative_l2_vs_direct_subset": type2_direct_error,
                "direct_subset_output_relative_l2": type2_direct_output_errors,
                "worst_direct_subset_output_relative_l2": max(
                    type2_direct_output_errors.values()
                ),
                "direct_subset_tolerance": args.type2_direct_tolerance,
                "direct_subset_accuracy_pass": bool(
                    type2_direct_error <= args.type2_direct_tolerance
                ),
                "eps": args.cufinufft_eps,
                "amortized": amortized_speedup_table(
                    acfo_setup_s=acfo_setup_seconds,
                    baseline_setup_s=type2_setup_seconds,
                    acfo_hot_s=type2_acfo_hot,
                    baseline_hot_s=type2_hot,
                ),
            }
        except Exception as exc:
            cufinufft_type2_result = {"available": False, "error": repr(exc)}

        try:
            if grid_geometry is None or channels is None:
                raise RuntimeError(
                    "projected type-2 unavailable for non-Cartesian source: "
                    f"{regular_grid_error}"
                )
            projected_contract = projected_type2_contract(
                channels,
                grid_geometry,
                flat_q_original,
                eliminated_axis=2,
            )
            projected_shape = tuple(projected_contract["shape"])
            projected_setup_start = perf_counter()
            projected_plan = cufinufft.Plan(
                2,
                projected_shape,
                n_trans=4,
                eps=args.cufinufft_eps,
                isign=-1,
                dtype=args.dtype,
                modeord=0,
            )
            projected_scaled_q = np.asarray(
                projected_contract["scaled_targets"], dtype=np.float64
            )
            projected_x = cp.asarray(projected_scaled_q[:, 0], dtype=cp_real)
            projected_y = cp.asarray(projected_scaled_q[:, 1], dtype=cp_real)
            projected_plan.setpts(projected_x, projected_y)
            projected_coefficients = cp.asarray(
                projected_contract["coefficients"], dtype=cp_complex
            )
            projected_out = cp.empty(
                (4, flat_q_original.shape[0]), dtype=cp_complex
            )
            projected_center_phase = cp.asarray(
                projected_contract["center_phase"], dtype=cp_complex
            )
            cp.cuda.runtime.deviceSynchronize()
            projected_setup_seconds = perf_counter() - projected_setup_start

            def cufinufft_type2_projected_func() -> dict[str, Any]:
                projected_plan.execute(projected_coefficients, out=projected_out)
                corrected = projected_out * projected_center_phase[None, :]
                values = corrected.reshape(4, args.q_nodes, unique_phi)
                return cupy_cross_sections(
                    cp, values, q_hat_cp, polarization_cp
                )

            def projected_wrapped() -> Any:
                value = cufinufft_type2_projected_func()
                cp.cuda.runtime.deviceSynchronize()
                return value

            cp.cuda.runtime.deviceSynchronize()
            projected_first_start = perf_counter()
            _ = projected_wrapped()
            projected_first_seconds = perf_counter() - projected_first_start
            (
                projected_acfo_samples,
                projected_samples,
                projected_acfo_value,
                projected_value,
            ) = abba(
                torch,
                device,
                acfo_func,
                projected_wrapped,
                warmups=args.warmups,
                samples=args.samples,
            )
            projected_cross_errors = {}
            for name, reference in projected_value.items():
                actual = projected_acfo_value[name].detach().cpu().numpy()
                projected_cross_errors[name] = relative_l2(
                    actual, cp.asnumpy(reference)
                )
            corrected_projected = (
                projected_out * projected_center_phase[None, :]
            )
            projected_direct = cp.asnumpy(
                corrected_projected[:, direct_indices]
            )
            projected_hot = float(np.median(projected_samples))
            projected_acfo_hot = float(np.median(projected_acfo_samples))
            projected_direct_error = relative_l2(
                projected_direct.astype(np.complex128, copy=False),
                direct_amplitudes,
            )
            projected_direct_observables = numpy_cross_sections_flat(
                projected_direct,
                q_hat_flat[direct_indices],
                np.asarray([0.0, 1.0, 0.0]),
            )
            projected_direct_output_errors = {
                name: relative_l2(
                    projected_direct_observables[name], direct_observables[name]
                )
                for name in COMMON_OUTPUT_NAMES
            }
            cufinufft_type2_projected_result = {
                "available": True,
                "eligible_production_baseline": True,
                "role": (
                    "exact flat-detector projection followed by native 2-D "
                    "regular-grid type-2"
                ),
                "plan_type": 2,
                "dimension": 2,
                "modeord": 0,
                "n_trans": 4,
                "grid_shape": list(projected_shape),
                "eliminated_axis": int(
                    projected_contract["eliminated_axis"]
                ),
                "eliminated_grid_size": int(
                    projected_contract["eliminated_grid_size"]
                ),
                "maximum_eliminated_q": float(
                    projected_contract["maximum_eliminated_q"]
                ),
                "exact_contraction": True,
                "scaled_target_coordinate_max_abs": float(
                    np.max(np.abs(projected_scaled_q))
                ),
                "setup_seconds": projected_setup_seconds,
                "first_solve_seconds": projected_first_seconds,
                "acfo_samples_seconds": projected_acfo_samples,
                "baseline_samples_seconds": projected_samples,
                "acfo_median_seconds": projected_acfo_hot,
                "baseline_median_seconds": projected_hot,
                "baseline_over_acfo_speedup": speedup_interval(
                    projected_acfo_samples,
                    projected_samples,
                    20260824,
                ),
                "cross_section_relative_l2": projected_cross_errors,
                "worst_cross_section_relative_l2": max(
                    projected_cross_errors.values()
                ),
                "direct_subset_target_count": int(direct_indices.size),
                "amplitude_relative_l2_vs_direct_subset": (
                    projected_direct_error
                ),
                "direct_subset_output_relative_l2": (
                    projected_direct_output_errors
                ),
                "worst_direct_subset_output_relative_l2": max(
                    projected_direct_output_errors.values()
                ),
                "direct_subset_tolerance": args.type2_direct_tolerance,
                "direct_subset_accuracy_pass": bool(
                    projected_direct_error <= args.type2_direct_tolerance
                ),
                "eps": args.cufinufft_eps,
                "amortized": amortized_speedup_table(
                    acfo_setup_s=acfo_setup_seconds,
                    baseline_setup_s=projected_setup_seconds,
                    acfo_hot_s=projected_acfo_hot,
                    baseline_hot_s=projected_hot,
                ),
            }
        except Exception as exc:
            cufinufft_type2_projected_result = {
                "available": False,
                "error": repr(exc),
            }
    except Exception as exc:  # recorded, then made fatal by the external verifier
        cufinufft_type3_result = {"available": False, "error": repr(exc)}
        cufinufft_type3_projected_result = {
            "available": False,
            "error": repr(exc),
        }
        cufinufft_type2_result = {"available": False, "error": repr(exc)}
        cufinufft_type2_projected_result = {
            "available": False,
            "error": repr(exc),
        }

    fft_result = None
    if (
        args.fft_pad_factor > 0
        and grid_geometry is not None
        and channels is not None
        and bounds is not None
        and cell_volume is not None
    ):
        # The regular grid is kept in its original orientation and evaluated on
        # the corresponding rotated q cloud.  This is mathematically identical
        # to rotating source coordinates but preserves the Cartesian lattice.
        fft_setup_start = perf_counter()
        fft_plan = TorchPreparedVoxelFFTInterpolator(
            tuple(channels.shape[1:]),
            bounds,
            q_original,
            torch=torch,
            device=device,
            dtype=args.dtype,
            phase_sign=-1,
            pad_factor=args.fft_pad_factor,
            continuous_voxels=False,
        )
        density = torch.as_tensor(
            channels / cell_volume, dtype=real_dtype, device=device
        )
        fft_setup_seconds = perf_counter() - fft_setup_start

        def fft_func() -> dict[str, Any]:
            transformed = fft_plan.forward(density)
            return torch_cross_sections(
                torch, transformed, q_hat_torch, polarization_torch
            )

        sync(torch, device)
        fft_first_start = perf_counter()
        _ = fft_func()
        sync(torch, device)
        fft_first_seconds = perf_counter() - fft_first_start
        fft_acfo_samples, fft_samples, fft_acfo_value, fft_value = abba(
            torch,
            device,
            acfo_func,
            fft_func,
            warmups=args.warmups,
            samples=args.samples,
        )
        errors = {
            name: relative_l2(
                fft_value[name].detach().cpu().numpy(),
                fft_acfo_value[name].detach().cpu().numpy(),
            )
            for name in fft_value
        }
        fft_result = {
            "pad_factor": args.fft_pad_factor,
            "channel_execution": "single batched four-channel FFT/interpolation call",
            "padded_shape": list(fft_plan.padded_shape),
            "setup_seconds": fft_setup_seconds,
            "first_solve_seconds": fft_first_seconds,
            "acfo_samples_seconds": fft_acfo_samples,
            "baseline_samples_seconds": fft_samples,
            "acfo_median_seconds": float(np.median(fft_acfo_samples)),
            "baseline_median_seconds": float(np.median(fft_samples)),
            "baseline_over_acfo_speedup": speedup_interval(fft_acfo_samples, fft_samples, 20260817),
            "relative_l2_vs_acfo": errors,
            "worst_relative_l2_vs_acfo": max(errors.values()),
            "accuracy_qualified": max(errors.values()) <= args.accuracy_tolerance,
            "plan_resident_bytes": fft_plan.resident_bytes,
            "amortized": amortized_speedup_table(
                acfo_setup_s=acfo_setup_seconds,
                baseline_setup_s=fft_setup_seconds,
                acfo_hot_s=float(np.median(fft_acfo_samples)),
                baseline_hot_s=float(np.median(fft_samples)),
            ),
        }
    elif args.fft_pad_factor > 0:
        fft_result = {
            "available": False,
            "reason": (
                "regular-grid FFT unavailable for non-Cartesian source: "
                f"{regular_grid_error}"
            ),
        }

    projected_fft_frontier_result = None
    projected_fft_pad_factors = tuple(
        int(value)
        for value in getattr(args, "projected_fft_pad_factors", ())
    )
    if projected_fft_pad_factors:
        projected_fft_candidates: list[dict[str, Any]] = []
        if grid_geometry is None or channels is None or bounds is None:
            projected_fft_frontier_result = {
                "available": False,
                "frontier_complete": False,
                "requested_pad_factors": list(projected_fft_pad_factors),
                "reason": (
                    "projected regular-grid FFT unavailable for non-Cartesian "
                    f"source: {regular_grid_error}"
                ),
                "candidates": [],
            }
        else:
            try:
                projected_fft_contract = projected_type2_contract(
                    channels,
                    grid_geometry,
                    q_original,
                    eliminated_axis=2,
                )
                projected_kept_axes = tuple(
                    int(value)
                    for value in projected_fft_contract["kept_axes"]
                )
                projected_bounds = tuple(
                    bounds[axis] for axis in projected_kept_axes
                )
                projected_q_xy = np.ascontiguousarray(
                    q_original[..., list(projected_kept_axes)]
                )
                projected_coefficients_torch = torch.as_tensor(
                    projected_fft_contract["coefficients"],
                    dtype=real_dtype,
                    device=device,
                )
                projected_coefficient_bytes = int(
                    projected_coefficients_torch.nelement()
                    * projected_coefficients_torch.element_size()
                )
                for pad_factor in projected_fft_pad_factors:
                    projected_fft_plan = None
                    projected_fft_func = None
                    projected_fft_amplitudes = None
                    projected_fft_value = None
                    projected_fft_acfo_value = None
                    try:
                        projected_fft_setup_start = perf_counter()
                        projected_fft_plan = TorchPreparedPlanarFFTInterpolator(
                            tuple(projected_fft_contract["shape"]),
                            projected_bounds,
                            projected_q_xy,
                            torch=torch,
                            device=device,
                            dtype=args.dtype,
                            phase_sign=-1,
                            pad_factor=pad_factor,
                            padded_shape=tuple(
                                next_fast_len(
                                    int(pad_factor) * int(axis_size)
                                )
                                for axis_size in projected_fft_contract[
                                    "shape"
                                ]
                            ),
                        )
                        sync(torch, device)
                        projected_fft_setup_seconds = (
                            perf_counter() - projected_fft_setup_start
                        )

                        def projected_fft_func() -> dict[str, Any]:
                            amplitudes = projected_fft_plan.forward(
                                projected_coefficients_torch
                            )
                            return torch_cross_sections(
                                torch,
                                amplitudes,
                                q_hat_torch,
                                polarization_torch,
                            )

                        sync(torch, device)
                        projected_fft_first_start = perf_counter()
                        _ = projected_fft_func()
                        sync(torch, device)
                        projected_fft_first_seconds = (
                            perf_counter() - projected_fft_first_start
                        )
                        (
                            projected_fft_acfo_samples,
                            projected_fft_samples,
                            projected_fft_acfo_value,
                            projected_fft_value,
                        ) = abba(
                            torch,
                            device,
                            acfo_func,
                            projected_fft_func,
                            warmups=args.warmups,
                            samples=args.samples,
                        )
                        projected_fft_amplitudes = projected_fft_plan.forward(
                            projected_coefficients_torch
                        )
                        sync(torch, device)
                        projected_fft_direct = (
                            projected_fft_amplitudes.reshape(4, -1)[
                                :, direct_indices
                            ]
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        projected_fft_direct_error = relative_l2(
                            projected_fft_direct.astype(
                                np.complex128, copy=False
                            ),
                            direct_amplitudes,
                        )
                        projected_fft_direct_observables = (
                            numpy_cross_sections_flat(
                                projected_fft_direct,
                                q_hat_flat[direct_indices],
                                np.asarray([0.0, 1.0, 0.0]),
                            )
                        )
                        projected_fft_direct_output_errors = {
                            name: relative_l2(
                                projected_fft_direct_observables[name],
                                direct_observables[name],
                            )
                            for name in COMMON_OUTPUT_NAMES
                        }
                        projected_fft_cross_errors = {
                            name: relative_l2(
                                projected_fft_value[name]
                                .detach()
                                .cpu()
                                .numpy(),
                                projected_fft_acfo_value[name]
                                .detach()
                                .cpu()
                                .numpy(),
                            )
                            for name in COMMON_OUTPUT_NAMES
                        }
                        amplitude_pass = bool(
                            projected_fft_direct_error
                            <= args.type2_direct_tolerance
                        )
                        output_pass = bool(
                            max(projected_fft_direct_output_errors.values())
                            <= args.accuracy_tolerance
                        )
                        protocol_pass = bool(
                            projected_fft_contract["maximum_eliminated_q"]
                            <= 1e-12
                            and len(projected_fft_acfo_samples)
                            == args.samples
                            and len(projected_fft_samples) == args.samples
                        )
                        padded_shape = tuple(
                            int(value)
                            for value in projected_fft_plan.padded_shape
                        )
                        complex_itemsize = (
                            np.dtype(np.complex64).itemsize
                            if args.dtype == "complex64"
                            else np.dtype(np.complex128).itemsize
                        )
                        candidate = {
                            "available": True,
                            "eligible_production_baseline": True,
                            "role": (
                                "exact flat-detector axial contraction followed "
                                "by 2-D padded FFT and bilinear interpolation"
                            ),
                            "dimension": 2,
                            "pad_factor": int(pad_factor),
                            "grid_shape": list(
                                projected_fft_contract["shape"]
                            ),
                            "padded_shape": list(padded_shape),
                            "padded_shape_rule": (
                                "per-axis scipy.fft.next_fast_len("
                                "pad_factor * source_grid_size)"
                            ),
                            "eliminated_axis": int(
                                projected_fft_contract["eliminated_axis"]
                            ),
                            "eliminated_grid_size": int(
                                projected_fft_contract[
                                    "eliminated_grid_size"
                                ]
                            ),
                            "maximum_eliminated_q": float(
                                projected_fft_contract[
                                    "maximum_eliminated_q"
                                ]
                            ),
                            "exact_axial_contraction": True,
                            "interpolation": "bilinear",
                            "channel_execution": (
                                "single batched four-channel 2-D "
                                "FFT/interpolation call"
                            ),
                            "setup_seconds": projected_fft_setup_seconds,
                            "first_solve_seconds": (
                                projected_fft_first_seconds
                            ),
                            "acfo_samples_seconds": (
                                projected_fft_acfo_samples
                            ),
                            "baseline_samples_seconds": projected_fft_samples,
                            "acfo_median_seconds": float(
                                np.median(projected_fft_acfo_samples)
                            ),
                            "baseline_median_seconds": float(
                                np.median(projected_fft_samples)
                            ),
                            "baseline_over_acfo_speedup": speedup_interval(
                                projected_fft_acfo_samples,
                                projected_fft_samples,
                                20260825 + int(pad_factor),
                            ),
                            "relative_l2_vs_acfo": (
                                projected_fft_cross_errors
                            ),
                            "worst_relative_l2_vs_acfo": max(
                                projected_fft_cross_errors.values()
                            ),
                            "direct_subset_target_count": int(
                                direct_indices.size
                            ),
                            "amplitude_relative_l2_vs_direct_subset": (
                                projected_fft_direct_error
                            ),
                            "direct_subset_output_relative_l2": (
                                projected_fft_direct_output_errors
                            ),
                            "worst_direct_subset_output_relative_l2": max(
                                projected_fft_direct_output_errors.values()
                            ),
                            "amplitude_tolerance": (
                                args.type2_direct_tolerance
                            ),
                            "worst_output_tolerance": (
                                args.accuracy_tolerance
                            ),
                            "amplitude_accuracy_pass": amplitude_pass,
                            "output_accuracy_pass": output_pass,
                            "accuracy_qualified": bool(
                                amplitude_pass and output_pass
                            ),
                            "protocol_pass": protocol_pass,
                            "plan_resident_bytes": int(
                                projected_fft_plan.resident_bytes
                            ),
                            "coefficient_resident_bytes": (
                                projected_coefficient_bytes
                            ),
                            "padded_four_channel_workspace_bytes": int(
                                4
                                * np.prod(padded_shape, dtype=np.int64)
                                * complex_itemsize
                            ),
                            "amortized": amortized_speedup_table(
                                acfo_setup_s=acfo_setup_seconds,
                                baseline_setup_s=(
                                    projected_fft_setup_seconds
                                ),
                                acfo_hot_s=float(
                                    np.median(projected_fft_acfo_samples)
                                ),
                                baseline_hot_s=float(
                                    np.median(projected_fft_samples)
                                ),
                            ),
                        }
                    except Exception as exc:
                        candidate = {
                            "available": False,
                            "pad_factor": int(pad_factor),
                            "accuracy_qualified": False,
                            "protocol_pass": False,
                            "error": repr(exc),
                        }
                    projected_fft_candidates.append(candidate)
                    del projected_fft_value
                    del projected_fft_acfo_value
                    del projected_fft_amplitudes
                    del projected_fft_func
                    del projected_fft_plan
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                projected_fft_selection = select_projected_fft_frontier(
                    projected_fft_candidates
                )
                projected_fft_frontier_complete = bool(
                    len(projected_fft_candidates)
                    == len(projected_fft_pad_factors)
                    and all(
                        row.get("protocol_pass") is True
                        for row in projected_fft_candidates
                    )
                )
                projected_fft_frontier_result = {
                    "available": True,
                    "frontier_complete": projected_fft_frontier_complete,
                    "requested_pad_factors": list(
                        projected_fft_pad_factors
                    ),
                    "accuracy_gates": {
                        "amplitude_relative_l2_vs_direct_subset": (
                            args.type2_direct_tolerance
                        ),
                        "worst_twelve_output_relative_l2_vs_direct_subset": (
                            args.accuracy_tolerance
                        ),
                    },
                    "selection_contract": projected_fft_selection,
                    "candidates": projected_fft_candidates,
                    "performance_sign_is_integrity_gate": False,
                }
            except Exception as exc:
                projected_fft_frontier_result = {
                    "available": False,
                    "frontier_complete": False,
                    "requested_pad_factors": list(
                        projected_fft_pad_factors
                    ),
                    "error": repr(exc),
                    "candidates": projected_fft_candidates,
                }

    if device.type == "cuda":
        torch.cuda.empty_cache()

    numagsans_result = None
    if args.numagsans_executable is not None:
        if args.numagsans_config is None:
            raise ValueError("--numagsans-config is required with --numagsans-executable")
        server = None
        try:
            server = NuMagSANSBenchmarkServer(
                args.numagsans_executable,
                args.numagsans_config,
                common_output=args.numagsans_common_output,
            )
            first_solve_seconds = server.sample()
            a, b = numagsans_abba(
                torch,
                device,
                acfo_func,
                server,
                warmups=args.warmups,
                samples=args.samples,
            )
            if args.numagsans_common_output:
                q_indices = direct_indices // unique_phi
                phi_indices = direct_indices % unique_phi
                theta_indices = (-phi_indices) % unique_phi
                upstream_indices = q_indices * args.theta_nodes + theta_indices
                probe = server.probe(upstream_indices)
                normalization = float(probe["normalization"])
                common_values = np.asarray(probe["values"]) / normalization
                common_cross = {
                    name: common_values[:, index]
                    for index, name in enumerate(COMMON_OUTPUT_NAMES)
                }
                direct_cross = numpy_cross_sections_flat(
                    direct_amplitudes,
                    q_hat_np[phi_indices],
                    np.asarray([0.0, 1.0, 0.0]),
                )
                acfo_probe_torch = acfo_func()
                sync(torch, device)
                acfo_probe = {
                    name: np.asarray(acfo_probe_torch[name].detach().cpu()).reshape(-1)[
                        direct_indices
                    ]
                    for name in COMMON_OUTPUT_NAMES
                }
                direct_errors = {
                    name: relative_l2(common_cross[name], direct_cross[name])
                    for name in COMMON_OUTPUT_NAMES
                }
                acfo_errors = {
                    name: relative_l2(common_cross[name], acfo_probe[name])
                    for name in COMMON_OUTPUT_NAMES
                }
                worst_direct = max(direct_errors.values())
                worst_acfo = max(acfo_errors.values())
                accuracy_pass = bool(
                    worst_direct <= args.numagsans_common_direct_tolerance
                    and worst_acfo <= args.accuracy_tolerance
                )
                for _ in range(args.warmups):
                    server.sample_upstream()
                upstream_diagnostic_samples = [
                    server.sample_upstream() for _ in range(args.samples)
                ]
                numagsans_result = {
                    "available": True,
                    "eligible_common_output_baseline": accuracy_pass,
                    "role": "patched upstream direct-sum kernel restricted to the common twelve dimensionless 2-D observables",
                    "common_output_schema": "acfo-numagsans-common-output-v1",
                    "common_output_names": list(COMMON_OUTPUT_NAMES),
                    "excluded_work": [
                        "18 G_ij correlation-matrix arrays",
                        "1-D averages",
                        "correlation and pair-distribution transforms",
                        "angular spectra",
                        "host export and physical unit scaling",
                    ],
                    "protocol": f"{args.warmups} warmups per arm; {args.samples} samples per arm; ABBA; NuMagSANS CUDA-event timing",
                    "startup_seconds": server.startup_seconds,
                    "first_solve_seconds": first_solve_seconds,
                    "acfo_samples_seconds": a,
                    "baseline_samples_seconds": b,
                    "acfo_median_seconds": float(np.median(a)),
                    "baseline_median_seconds": float(np.median(b)),
                    "baseline_over_acfo_speedup": speedup_interval(a, b, 20260818),
                    "probe": {
                        "target_count": int(direct_indices.size),
                        "common_workload_target_count": int(probe["target_count"]),
                        "acfo_flat_indices": direct_indices.tolist(),
                        "upstream_flat_indices": upstream_indices.tolist(),
                        "normalization": normalization,
                        "relative_l2_vs_direct": direct_errors,
                        "worst_relative_l2_vs_direct": worst_direct,
                        "relative_l2_vs_acfo": acfo_errors,
                        "worst_relative_l2_vs_acfo": worst_acfo,
                        "direct_tolerance": args.numagsans_common_direct_tolerance,
                        "acfo_tolerance": args.accuracy_tolerance,
                        "accuracy_pass": accuracy_pass,
                    },
                    "upstream_extra_output_diagnostic": {
                        "eligible_performance_baseline": False,
                        "samples_seconds": upstream_diagnostic_samples,
                        "median_seconds": float(np.median(upstream_diagnostic_samples)),
                        "reason": "computes G_ij outputs beyond the common contract and does not form all twelve common derived outputs in the timed kernel",
                    },
                    "amortized": amortized_speedup_table(
                        acfo_setup_s=acfo_setup_seconds,
                        baseline_setup_s=server.startup_seconds,
                        acfo_hot_s=float(np.median(a)),
                        baseline_hot_s=float(np.median(b)),
                    ),
                    "startup_transcript_tail": server.transcript[-80:],
                }
            else:
                numagsans_result = {
                    "available": True,
                    "eligible_common_output_baseline": False,
                    "eligibility_blocker": "upstream combined kernel computes outputs beyond the common 12-observable contract",
                    "protocol": f"{args.warmups} warmups per arm; {args.samples} samples per arm; ABBA; upstream CUDA event timing",
                    "startup_seconds": server.startup_seconds,
                    "first_solve_seconds": first_solve_seconds,
                    "acfo_samples_seconds": a,
                    "baseline_samples_seconds": b,
                    "acfo_median_seconds": float(np.median(a)),
                    "baseline_median_seconds": float(np.median(b)),
                    "baseline_over_acfo_speedup": speedup_interval(a, b, 20260818),
                    "startup_transcript_tail": server.transcript[-40:],
                }
        except Exception as exc:
            numagsans_result = {
                "available": False,
                "error": repr(exc),
                "startup_transcript_tail": server.transcript[-40:] if server is not None else [],
            }
        finally:
            if server is not None:
                server.close()
    else:
        numagsans_result = {"available": False, "reason": "no patched upstream executable supplied"}

    acfo_direct_accuracy_pass = bool(
        acfo_direct_amplitude_error <= args.type2_direct_tolerance
        and max(acfo_direct_output_errors.values())
        <= args.accuracy_tolerance
    )
    type3_accuracy_pass = bool(
        cufinufft_type3_result.get("available")
        and cufinufft_type3_result.get("worst_cross_section_relative_l2", np.inf)
        <= args.accuracy_tolerance
        and cufinufft_type3_result.get(
            "amplitude_relative_l2_vs_direct_subset", np.inf
        )
        <= args.type2_direct_tolerance
    )
    projected_type3_accuracy_pass = bool(
        cufinufft_type3_projected_result.get("available")
        and cufinufft_type3_projected_result.get(
            "worst_cross_section_relative_l2", np.inf
        )
        <= args.accuracy_tolerance
        and cufinufft_type3_projected_result.get(
            "direct_subset_accuracy_pass"
        )
        is True
    )
    projected_type3_protocol_pass = bool(
        cufinufft_type3_projected_result.get("plan_type") == 3
        and cufinufft_type3_projected_result.get("dimension") == 2
        and cufinufft_type3_projected_result.get("n_trans") == 4
        and cufinufft_type3_projected_result.get("exact_contraction") is True
        and cufinufft_type3_projected_result.get(
            "maximum_eliminated_q", np.inf
        )
        <= 1e-12
        and len(
            cufinufft_type3_projected_result.get(
                "acfo_samples_seconds", []
            )
        )
        == args.samples
        and len(
            cufinufft_type3_projected_result.get(
                "baseline_samples_seconds", []
            )
        )
        == args.samples
    )
    projected_type3_closure_complete = bool(
        projected_type3_accuracy_pass and projected_type3_protocol_pass
    )
    projected_type3_lower_95 = cufinufft_type3_projected_result.get(
        "baseline_over_acfo_speedup", {}
    ).get("lower_95")
    projected_type3_acfo_go = bool(
        projected_type3_closure_complete
        and projected_type3_lower_95 is not None
        and float(projected_type3_lower_95)
        >= args.required_lower_95_speedup
    )
    type2_accuracy_pass = bool(
        cufinufft_type2_result.get("available")
        and cufinufft_type2_result.get("worst_cross_section_relative_l2", np.inf)
        <= args.accuracy_tolerance
        and cufinufft_type2_result.get("direct_subset_accuracy_pass") is True
        and cufinufft_type2_result.get(
            "worst_direct_subset_output_relative_l2", np.inf
        )
        <= args.accuracy_tolerance
    )
    type2_protocol_pass = bool(
        cufinufft_type2_result.get("plan_type") == 2
        and cufinufft_type2_result.get("n_trans") == 4
        and len(cufinufft_type2_result.get("acfo_samples_seconds", [])) == args.samples
        and len(cufinufft_type2_result.get("baseline_samples_seconds", []))
        == args.samples
        and cufinufft_type2_result.get("scaled_target_coordinate_max_abs", np.inf)
        < np.pi
    )
    native_type2_closure_complete = bool(type2_accuracy_pass and type2_protocol_pass)
    type2_lower_95 = cufinufft_type2_result.get(
        "baseline_over_acfo_speedup", {}
    ).get("lower_95")
    native_type2_acfo_go = bool(
        native_type2_closure_complete
        and type2_lower_95 is not None
        and float(type2_lower_95) >= args.required_lower_95_speedup
    )
    projected_type2_accuracy_pass = bool(
        cufinufft_type2_projected_result.get("available")
        and cufinufft_type2_projected_result.get(
            "worst_cross_section_relative_l2", np.inf
        )
        <= args.accuracy_tolerance
        and cufinufft_type2_projected_result.get(
            "direct_subset_accuracy_pass"
        )
        is True
        and cufinufft_type2_projected_result.get(
            "worst_direct_subset_output_relative_l2", np.inf
        )
        <= args.accuracy_tolerance
    )
    projected_type2_protocol_pass = bool(
        cufinufft_type2_projected_result.get("plan_type") == 2
        and cufinufft_type2_projected_result.get("dimension") == 2
        and cufinufft_type2_projected_result.get("n_trans") == 4
        and cufinufft_type2_projected_result.get("exact_contraction") is True
        and cufinufft_type2_projected_result.get(
            "maximum_eliminated_q", np.inf
        )
        <= 1e-12
        and len(
            cufinufft_type2_projected_result.get(
                "acfo_samples_seconds", []
            )
        )
        == args.samples
        and len(
            cufinufft_type2_projected_result.get(
                "baseline_samples_seconds", []
            )
        )
        == args.samples
        and cufinufft_type2_projected_result.get(
            "scaled_target_coordinate_max_abs", np.inf
        )
        < np.pi
    )
    projected_type2_closure_complete = bool(
        projected_type2_accuracy_pass and projected_type2_protocol_pass
    )
    projected_type2_lower_95 = cufinufft_type2_projected_result.get(
        "baseline_over_acfo_speedup", {}
    ).get("lower_95")
    projected_type2_acfo_go = bool(
        projected_type2_closure_complete
        and projected_type2_lower_95 is not None
        and float(projected_type2_lower_95)
        >= args.required_lower_95_speedup
    )
    projected_fft_frontier = projected_fft_frontier_result or {}
    projected_fft_selection = projected_fft_frontier.get(
        "selection_contract", {}
    )
    projected_fft_selected = projected_fft_selection.get(
        "selected_candidate"
    )
    projected_fft_frontier_protocol_pass = bool(
        projected_fft_pad_factors
        and projected_fft_frontier.get("frontier_complete") is True
        and tuple(projected_fft_frontier.get("requested_pad_factors", ()))
        == projected_fft_pad_factors
        and projected_fft_selection.get(
            "accuracy_filter_precedes_timing_rank"
        )
        is True
    )
    projected_fft_has_accuracy_qualified_candidate = bool(
        projected_fft_selected is not None
    )
    projected_fft_selected_lower_95 = (
        projected_fft_selected.get("baseline_over_acfo_speedup", {}).get(
            "lower_95"
        )
        if projected_fft_selected is not None
        else None
    )
    projected_fft_selected_acfo_go = bool(
        projected_fft_frontier_protocol_pass
        and projected_fft_selected_lower_95 is not None
        and float(projected_fft_selected_lower_95)
        >= args.required_lower_95_speedup
    )
    fft_accuracy_pass = bool(fft_result and fft_result.get("accuracy_qualified"))
    fft_protocol_pass = bool(
        fft_result
        and fft_result.get("channel_execution", "").startswith("single batched")
        and len(fft_result.get("acfo_samples_seconds", [])) == args.samples
        and len(fft_result.get("baseline_samples_seconds", [])) == args.samples
    )
    generic_lower_bounds = [
        float(value)
        for value in (
            type2_lower_95,
            (
                fft_result.get("baseline_over_acfo_speedup", {}).get("lower_95")
                if fft_result and fft_accuracy_pass
                else None
            ),
        )
        if value is not None
    ]
    generic_baselines_closed = bool(
        native_type2_closure_complete and fft_accuracy_pass and fft_protocol_pass
    )
    generic_acfo_go = bool(
        generic_baselines_closed
        and len(generic_lower_bounds) == 2
        and min(generic_lower_bounds) >= args.required_lower_95_speedup
    )
    domain_protocol_pass = bool(
        args.numagsans_common_output
        and numagsans_result.get("available") is True
        and numagsans_result.get("eligible_common_output_baseline") is True
        and numagsans_result.get("common_output_schema")
        == "acfo-numagsans-common-output-v1"
        and tuple(numagsans_result.get("common_output_names", ()))
        == COMMON_OUTPUT_NAMES
        and len(numagsans_result.get("acfo_samples_seconds", [])) == args.samples
        and len(numagsans_result.get("baseline_samples_seconds", []))
        == args.samples
        and numagsans_result.get("probe", {}).get(
            "common_workload_target_count"
        )
        == args.q_nodes * unique_phi
        and numagsans_result.get("probe", {}).get("accuracy_pass") is True
    )
    domain_common_output_closed = domain_protocol_pass
    domain_lower_95 = numagsans_result.get("baseline_over_acfo_speedup", {}).get(
        "lower_95"
    )
    domain_acfo_go = bool(
        domain_common_output_closed
        and domain_lower_95 is not None
        and float(domain_lower_95) >= args.required_lower_95_speedup
    )
    manuscript_performance_claim_eligible = bool(
        generic_acfo_go and domain_common_output_closed
    )
    geometry_aware_qualified_candidates: list[dict[str, Any]] = []
    if projected_type2_closure_complete:
        geometry_aware_qualified_candidates.append(
            {
                "baseline": "projected_2d_type2",
                "baseline_median_seconds": (
                    cufinufft_type2_projected_result[
                        "baseline_median_seconds"
                    ]
                ),
                "baseline_over_acfo_speedup": (
                    cufinufft_type2_projected_result[
                        "baseline_over_acfo_speedup"
                    ]
                ),
            }
        )
    if (
        projected_fft_frontier_protocol_pass
        and projected_fft_selected is not None
    ):
        geometry_aware_qualified_candidates.append(
            {
                "baseline": "projected_2d_padded_fft",
                "pad_factor": int(projected_fft_selected["pad_factor"]),
                "baseline_median_seconds": projected_fft_selected[
                    "baseline_median_seconds"
                ],
                "baseline_over_acfo_speedup": projected_fft_selected[
                    "baseline_over_acfo_speedup"
                ],
            }
        )
    strongest_geometry_candidate = (
        min(
            geometry_aware_qualified_candidates,
            key=lambda row: (
                float(row["baseline_median_seconds"]),
                str(row["baseline"]),
            ),
        )
        if geometry_aware_qualified_candidates
        else None
    )
    if grid_geometry is not None and projected_fft_pad_factors:
        geometry_aware_baseline = (
            strongest_geometry_candidate["baseline"]
            if strongest_geometry_candidate is not None
            else "none_accuracy_qualified"
        )
        geometry_aware_closure_complete = bool(
            acfo_direct_accuracy_pass
            and projected_type2_closure_complete
            and projected_fft_frontier_protocol_pass
            and strongest_geometry_candidate is not None
        )
        strongest_lower_95 = (
            strongest_geometry_candidate[
                "baseline_over_acfo_speedup"
            ].get("lower_95")
            if strongest_geometry_candidate is not None
            else None
        )
        geometry_aware_acfo_go = bool(
            geometry_aware_closure_complete
            and strongest_lower_95 is not None
            and float(strongest_lower_95)
            >= args.required_lower_95_speedup
        )
    elif grid_geometry is not None:
        geometry_aware_baseline = "projected_2d_type2"
        geometry_aware_closure_complete = projected_type2_closure_complete
        geometry_aware_acfo_go = projected_type2_acfo_go
    else:
        geometry_aware_baseline = "projected_2d_type3"
        geometry_aware_closure_complete = projected_type3_closure_complete
        geometry_aware_acfo_go = projected_type3_acfo_go
    geometry_aware_manuscript_claim_eligible = bool(
        geometry_aware_acfo_go
    )
    geometry_aware_domain_claim_eligible = bool(
        geometry_aware_acfo_go and domain_common_output_closed
    )
    if grid_geometry is not None and projected_fft_pad_factors:
        verdict = (
            "STRONGEST_GEOMETRY_BASELINE_FRONTIER_COMPLETE_ACFO_GO"
            if geometry_aware_closure_complete and geometry_aware_acfo_go
            else (
                "STRONGEST_GEOMETRY_BASELINE_FRONTIER_COMPLETE_ACFO_NO_GO"
                if geometry_aware_closure_complete
                else "STRONGEST_GEOMETRY_BASELINE_FRONTIER_INCOMPLETE"
            )
        )
    elif grid_geometry is None:
        verdict = (
            "PROJECTED_TYPE3_CLOSURE_COMPLETE_ACFO_GO"
            if projected_type3_closure_complete and projected_type3_acfo_go
            else (
                "PROJECTED_TYPE3_CLOSURE_COMPLETE_ACFO_NO_GO"
                if projected_type3_closure_complete
                else "PROJECTED_TYPE3_CLOSURE_INCOMPLETE"
            )
        )
    elif native_type2_closure_complete:
        verdict = (
            "NATIVE_TYPE2_CLOSURE_COMPLETE_ACFO_GO"
            if native_type2_acfo_go
            else "NATIVE_TYPE2_CLOSURE_COMPLETE_ACFO_NO_GO"
        )
    else:
        verdict = "NATIVE_TYPE2_CLOSURE_INCOMPLETE"
    return {
        "schema": "acfo-magnetic-sans-native-type2-closure-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "timing_sample_provenance": (
            timing_sample_provenance_from_environment(args.samples)
        ),
        "verdict": verdict,
        "environment": {
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "dtype": args.dtype,
        },
        "workload": {
            "sources": int(points.shape[0]),
            "source_coordinate_scale": float(args.source_coordinate_scale),
            "q_nodes": args.q_nodes,
            "theta_nodes_upstream": args.theta_nodes,
            "unique_theta": unique_phi,
            "targets": args.q_nodes * unique_phi,
            "acfo_histogram": [args.acfo_n_r, args.acfo_n_z, unique_phi],
            "acfo_harmonic_margin_requested": args.harmonic_margin,
            "max_h": acfo.max_h,
            "cufinufft_eps": args.cufinufft_eps,
            "warmups_per_arm": args.warmups,
            "samples_per_arm": args.samples,
            "projected_fft_pad_factors": list(
                projected_fft_pad_factors
            ),
            "regular_cartesian_grid": grid_geometry is not None,
            "regular_grid_rejection": regular_grid_error,
            "regular_grid_shape": (
                [int(value) for value in grid_geometry["shape"]]
                if grid_geometry is not None
                else None
            ),
            "regular_grid_spacing": [
                float(value) for value in grid_geometry["spacing"]
            ] if grid_geometry is not None else None,
            "regular_grid_center": (
                [float(value) for value in grid_geometry["center"]]
                if grid_geometry is not None
                else None
            ),
            "regular_grid_maximum_coordinate_residual": (
                float(grid_geometry["maximum_coordinate_residual"])
                if grid_geometry is not None
                else None
            ),
            "vector_component_frame": "permuted upstream NuMagSANS (y,z,x)",
            "source_deposition": args.source_deposition,
            "detector_direction_contract": (
                "inverse-rotate emitted ACFO directions by the source angular shift "
                "before Halpern-Johnson/polarization contraction"
            ),
        },
        "acfo": {
            "setup_seconds": acfo_setup_seconds,
            "setup_breakdown_seconds": {
                "cpu_histogram_and_plan": cpu_histogram_and_plan_setup_seconds,
                "torch_kernel_and_pack": torch_kernel_and_pack_setup_seconds,
                "source_update_map": source_update_map_setup_seconds,
            },
            "kernel_backend": acfo.kernel_backend,
            "miller_recurrence_margin": acfo.miller_recurrence_margin,
            "gpu_miller_jit_seconds_excluded_from_operator_setup": gpu_miller_jit_seconds,
            "gpu_miller_parity": gpu_miller_parity,
            "first_solve_seconds": acfo_first_seconds,
            "hot_samples_seconds": acfo_standalone_samples,
            "hot_median_seconds": float(np.median(acfo_standalone_samples)),
            "descriptive_stage_medians_seconds": stage_medians,
            "resident_bytes": acfo.resident_bytes,
            "optimization_stats": acfo.optimization_stats,
            "field_update_included": True,
            "fourier_channels": 4,
            "detector_observables": 12,
            "direct_subset_target_count": int(direct_indices.size),
            "amplitude_relative_l2_vs_direct_subset": float(
                acfo_direct_amplitude_error
            ),
            "direct_subset_output_relative_l2": acfo_direct_output_errors,
            "worst_direct_subset_output_relative_l2": float(
                max(acfo_direct_output_errors.values())
            ),
        },
        "cufinufft": cufinufft_type3_result,
        "cufinufft_type3_diagnostic": cufinufft_type3_result,
        "cufinufft_type3_projected": cufinufft_type3_projected_result,
        "cufinufft_type2_native": cufinufft_type2_result,
        "cufinufft_type2_projected": cufinufft_type2_projected_result,
        "regular_grid_fft": fft_result,
        "projected_2d_fft_frontier": projected_fft_frontier_result,
        "numagsans_cuda": numagsans_result,
        "decision": {
            "acfo_direct_accuracy_pass": acfo_direct_accuracy_pass,
            "type3_diagnostic_accuracy_pass": type3_accuracy_pass,
            "projected_type3_accuracy_pass": projected_type3_accuracy_pass,
            "projected_type3_protocol_pass": projected_type3_protocol_pass,
            "projected_type3_closure_complete": (
                projected_type3_closure_complete
            ),
            "projected_type3_acfo_go": projected_type3_acfo_go,
            "native_type2_accuracy_pass": type2_accuracy_pass,
            "native_type2_protocol_pass": type2_protocol_pass,
            "native_type2_closure_complete": native_type2_closure_complete,
            "native_type2_acfo_go": native_type2_acfo_go,
            "projected_type2_accuracy_pass": projected_type2_accuracy_pass,
            "projected_type2_protocol_pass": projected_type2_protocol_pass,
            "projected_type2_closure_complete": (
                projected_type2_closure_complete
            ),
            "projected_type2_acfo_go": projected_type2_acfo_go,
            "projected_fft_frontier_protocol_pass": (
                projected_fft_frontier_protocol_pass
            ),
            "projected_fft_has_accuracy_qualified_candidate": (
                projected_fft_has_accuracy_qualified_candidate
            ),
            "projected_fft_selected_pad_factor": (
                int(projected_fft_selected["pad_factor"])
                if projected_fft_selected is not None
                else None
            ),
            "projected_fft_selected_acfo_go": (
                projected_fft_selected_acfo_go
            ),
            "batched_fft_accuracy_pass": fft_accuracy_pass,
            "batched_fft_protocol_pass": fft_protocol_pass,
            "generic_baselines_closed": generic_baselines_closed,
            "generic_acfo_go": generic_acfo_go,
            "domain_common_output_protocol_pass": domain_protocol_pass,
            "domain_common_output_closed": domain_common_output_closed,
            "domain_acfo_go": domain_acfo_go,
            "manuscript_performance_claim_eligible": manuscript_performance_claim_eligible,
            "geometry_aware_manuscript_claim_eligible": (
                geometry_aware_manuscript_claim_eligible
            ),
            "geometry_aware_domain_claim_eligible": (
                geometry_aware_domain_claim_eligible
            ),
            "geometry_aware_baseline": geometry_aware_baseline,
            "geometry_aware_qualified_candidates": (
                geometry_aware_qualified_candidates
            ),
            "geometry_aware_selection_rule": (
                "accuracy qualification first; fastest hot median among "
                "qualified projected type-2 and projected padded-FFT arms"
            ),
            "geometry_aware_selection_uses_timing_before_accuracy": False,
            "geometry_aware_closure_complete": (
                geometry_aware_closure_complete
            ),
            "geometry_aware_acfo_go": geometry_aware_acfo_go,
            "accuracy_tolerance": args.accuracy_tolerance,
            "type2_direct_tolerance": args.type2_direct_tolerance,
            "required_lower_95_speedup": args.required_lower_95_speedup,
            "unfavorable_speed_result_is_valid_closure": True,
        },
        "claim_boundary": [
            "The native type-2 sign is retained even if ACFO loses.",
            (
                "The exact projected 2-D type-2 is the strongest flat-detector "
                "NUFFT baseline; its sign is retained even if ACFO loses."
            ),
            (
                "The exact axial contraction plus projected 2-D padded-FFT "
                "frontier is accuracy-filtered before timing rank; the fastest "
                "qualified projected FFT or type-2 arm defines the strongest "
                "geometry-aware comparator."
            ),
            (
                "For a generally rotated non-Cartesian source, exact planar "
                "2-D type-3 replaces type-2 as the eligible strongest baseline."
            ),
            "Type-3 is a diagnostic oracle, not the eligible regular-grid performance baseline.",
            "The regular-grid FFT executes all four channels in one batched call.",
            (
                "The patched NuMagSANS common-output kernel is performance-eligible; the original extra-output kernel remains diagnostic."
                if domain_common_output_closed
                else "The upstream NuMagSANS timing remains descriptive until a common-output kernel or explicit extra-output accounting is available."
            ),
        ],
    }


def parser() -> argparse.ArgumentParser:
    base = ROOT / "inputs" / "numagsans_example2"
    p = argparse.ArgumentParser()
    p.add_argument("--magnetic", type=Path, default=base / "m_1.csv")
    p.add_argument("--nuclear", type=Path, default=base / "n_1.csv")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=("complex64", "complex128"), default="complex64")
    p.add_argument("--q-nodes", type=int, default=1000)
    p.add_argument("--theta-nodes", type=int, default=1000)
    p.add_argument("--q-max", type=float, default=2.0)
    p.add_argument(
        "--source-coordinate-scale",
        type=float,
        default=1.0,
        help="Multiply source coordinates before applying detector q units.",
    )
    p.add_argument("--acfo-n-r", type=int, default=384)
    p.add_argument("--acfo-n-z", type=int, default=41)
    p.add_argument("--harmonic-margin", type=int, default=16)
    p.add_argument(
        "--acfo-kernel-backend",
        choices=("auto", "cpu_miller", "gpu_miller"),
        default="auto",
    )
    p.add_argument(
        "--require-miller-kernel",
        action="store_true",
        help="Fail setup unless the compiled Miller downward-recurrence kernel is active.",
    )
    p.add_argument(
        "--source-deposition",
        choices=("nearest", "linear"),
        default="nearest",
        help="Point-to-cylindrical-grid deposition used by repeated source updates.",
    )
    p.add_argument("--cufinufft-eps", type=float, default=1e-5)
    p.add_argument(
        "--skip-type3",
        action="store_true",
        help="Skip the non-production type-3 diagnostic during frontier sweeps.",
    )
    p.add_argument("--fft-pad-factor", type=int, default=12)
    p.add_argument(
        "--projected-fft-pad-factors",
        type=positive_int_csv,
        default=(),
        help=(
            "Comma-separated exact-projection 2-D FFT padding frontier. "
            "Candidates are accuracy-filtered before timing rank."
        ),
    )
    p.add_argument("--warmups", type=int, default=10)
    p.add_argument("--samples", type=int, default=30)
    p.add_argument("--accuracy-tolerance", type=float, default=1e-2)
    p.add_argument("--direct-target-count", type=int, default=32)
    p.add_argument("--type2-direct-tolerance", type=float, default=1e-3)
    p.add_argument("--required-lower-95-speedup", type=float, default=1.5)
    p.add_argument("--numagsans-executable", type=Path)
    p.add_argument("--numagsans-config", type=Path)
    p.add_argument("--numagsans-common-output", action="store_true")
    p.add_argument("--numagsans-common-direct-tolerance", type=float, default=1e-3)
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "magnetic_sans_phase_b_gpu.json",
    )
    return p


def main() -> None:
    args = parser().parse_args()
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
