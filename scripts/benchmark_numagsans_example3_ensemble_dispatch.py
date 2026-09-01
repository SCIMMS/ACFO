"""Prospective 800-orientation x five-case NuMagSANS Example 3 benchmark.

The benchmark streams orientations and never materializes the 25.6 GB complex
amplitude library.  ACFO and the exact projected 2-D type-3 baseline share the
same dilute/packed reducer, so packing does not alter backend eligibility.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any
import zipfile

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
    torch_cross_sections,
)
from waxs_cake.axisymmetric_manifold import AxisymmetricManifold  # noqa: E402
from waxs_cake.magnetic_sans import (  # noqa: E402
    PreparedBandlimitedMagneticSansOperator,
    load_numagsans_fourier_sources,
)
from waxs_cake.magnetic_sans_torch import (  # noqa: E402
    TorchFlatDetectorBandlimitedMagneticSansOperator,
)
from waxs_cake.numagsans_ensemble import (  # noqa: E402
    OUTPUT_NAMES,
    PRIMITIVE_OUTPUT_NAMES,
    cumulative_crossover,
    frozen_dispatch_decision,
    load_structure_positions,
    relative_l2,
)


ARCHIVED_COLUMNS = ("qz", "qy", "q", "theta", *OUTPUT_NAMES)


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_text_pair(
    magnetic: Any, nuclear: Any, *, coordinate_scale: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    magnetic_values = np.loadtxt(magnetic, dtype=np.float64)
    nuclear_values = np.loadtxt(nuclear, dtype=np.float64)
    if magnetic_values.shape[0] != nuclear_values.shape[0]:
        raise ValueError("magnetic and nuclear source counts differ")
    magnetic_points = magnetic_values[:, :3]
    nuclear_points = nuclear_values[:, :3]
    if not np.array_equal(magnetic_points, nuclear_points):
        raise ValueError("magnetic and nuclear coordinates differ")
    permutation = np.asarray([1, 2, 0], dtype=np.intp)
    points = magnetic_points[:, permutation] * float(coordinate_scale)
    magnetization = magnetic_values[:, 3:6][:, permutation]
    nuclear_weights = nuclear_values[:, 3]
    return (
        np.ascontiguousarray(points),
        np.ascontiguousarray(nuclear_weights),
        np.ascontiguousarray(magnetization),
    )


class OrientationSource:
    def __init__(
        self,
        *,
        source_archive: Path | None,
        reduced_dir: Path | None,
        protocol: dict[str, Any],
    ) -> None:
        self.source_archive = source_archive
        self.reduced_dir = reduced_dir
        self.protocol = protocol
        self.bundle = zipfile.ZipFile(source_archive) if source_archive else None

    def close(self) -> None:
        if self.bundle is not None:
            self.bundle.close()

    def load(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scale = self.protocol["dataset"]["source_coordinate_scale"]
        if self.bundle is None:
            if self.reduced_dir is None:
                raise RuntimeError("no source input configured")
            return load_numagsans_fourier_sources(
                self.reduced_dir / "m_1.txt",
                self.reduced_dir / "n_1.txt",
            )
        magnetic_name = self.protocol["dataset"]["source_archive"][
            "magnetic_member_template"
        ].format(index=index)
        nuclear_name = self.protocol["dataset"]["source_archive"][
            "nuclear_member_template"
        ].format(index=index)
        with self.bundle.open(magnetic_name) as magnetic_raw, self.bundle.open(
            nuclear_name
        ) as nuclear_raw:
            return _load_text_pair(
                io.TextIOWrapper(magnetic_raw, encoding="utf-8"),
                io.TextIOWrapper(nuclear_raw, encoding="utf-8"),
                coordinate_scale=scale,
            )


def validate_source_archive(bundle: zipfile.ZipFile, protocol: dict[str, Any]) -> dict[str, Any]:
    names = set(bundle.namelist())
    missing: list[str] = []
    source = protocol["dataset"]["source_archive"]
    for index in range(1, protocol["dataset"]["orientation_count"] + 1):
        for key in ("magnetic_member_template", "nuclear_member_template"):
            name = source[key].format(index=index)
            if name not in names:
                missing.append(name)
    for item in protocol["packing_hierarchy"]:
        member = item.get("structure_member")
        if member is not None and member not in names:
            missing.append(member)
    return {
        "members": len(names),
        "required_orientation_pairs": protocol["dataset"]["orientation_count"],
        "missing_count": len(missing),
        "missing_first_ten": missing[:10],
        "pass": not missing,
    }


def load_centers(
    protocol: dict[str, Any], source: OrientationSource
) -> tuple[list[str], np.ndarray]:
    ids: list[str] = []
    tables: list[np.ndarray] = []
    for item in protocol["packing_hierarchy"]:
        member = item.get("structure_member")
        if member is None:
            continue
        ids.append(item["id"])
        if source.bundle is None:
            values = load_structure_positions(
                ROOT / item["local_structure_file"],
                coordinate_scale=protocol["dataset"]["source_coordinate_scale"],
            )
        else:
            with source.bundle.open(member) as raw:
                table = np.loadtxt(io.TextIOWrapper(raw, encoding="utf-8"))
            permutation = np.asarray([1, 2, 0], dtype=np.intp)
            values = (
                np.asarray(table[:, :3], dtype=np.float64)[:, permutation]
                * protocol["dataset"]["source_coordinate_scale"]
            )
        tables.append(np.ascontiguousarray(values))
    return ids, np.stack(tables)


def _channels_from_torch_amplitudes(torch: Any, amplitudes: Any) -> Any:
    result = torch.empty(
        (4, *amplitudes.nuclear.shape),
        dtype=amplitudes.nuclear.dtype,
        device=amplitudes.nuclear.device,
    )
    result[0] = amplitudes.nuclear
    result[1:] = amplitudes.magnetization.permute(2, 0, 1)
    return result


def _primitive_torch(torch: Any, channels: Any, q_hat: Any, polarization: Any) -> dict[str, Any]:
    all_outputs = torch_cross_sections(torch, channels, q_hat, polarization)
    return {name: all_outputs[name] for name in PRIMITIVE_OUTPUT_NAMES}


def _derive_twelve_torch(
    primitive: dict[str, Any],
    *,
    nuclear_scale: float = 1.0,
    magnetic_scale: float = 1.0,
    nuclear_magnetic_scale: float = 1.0,
) -> dict[str, Any]:
    result = {
        "S_N": primitive["S_N"] * float(nuclear_scale),
        "S_M": primitive["S_M"] * float(magnetic_scale),
        "S_NM": primitive["S_NM"] * float(nuclear_magnetic_scale),
        "S_P": primitive["S_P"] * float(magnetic_scale),
        "S_chi": primitive["S_chi"] * float(magnetic_scale),
    }
    result["S_sf"] = result["S_M"] - result["S_P"]
    result["S_pm"] = result["S_sf"] + result["S_chi"]
    result["S_mp"] = result["S_sf"] - result["S_chi"]
    result["S_pp"] = result["S_N"] + result["S_NM"] + result["S_P"]
    result["S_mm"] = result["S_N"] - result["S_NM"] + result["S_P"]
    result["S_p"] = result["S_pp"] + result["S_pm"]
    result["S_m"] = result["S_mm"] + result["S_mp"]
    return result


def _hierarchy_outputs_torch(
    torch: Any,
    dilute: dict[str, Any],
    packed: Any,
    q_hat: Any,
    polarization: Any,
    packed_ids: list[str],
    *,
    scales: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, dict[str, Any]]:
    fn, fm, fnm = scales
    result = {
        "dilute": _derive_twelve_torch(
            dilute,
            nuclear_scale=fn,
            magnetic_scale=fm,
            nuclear_magnetic_scale=fnm,
        )
    }
    for local, name in enumerate(packed_ids):
        primitive = _primitive_torch(torch, packed[local], q_hat, polarization)
        result[name] = _derive_twelve_torch(
            primitive,
            nuclear_scale=fn,
            magnetic_scale=fm,
            nuclear_magnetic_scale=fnm,
        )
    return result


def _hierarchy_errors(torch: Any, actual: dict[str, dict[str, Any]], reference: dict[str, dict[str, Any]]) -> dict[str, Any]:
    errors: dict[str, dict[str, float]] = {}
    for case in reference:
        errors[case] = {}
        for name in OUTPUT_NAMES:
            numerator = torch.linalg.vector_norm(actual[case][name] - reference[case][name])
            denominator = torch.clamp(
                torch.linalg.vector_norm(reference[case][name]), min=1e-30
            )
            errors[case][name] = float((numerator / denominator).item())
    worst = max(value for case in errors.values() for value in case.values())
    return {"by_case": errors, "worst_relative_l2": worst}


def physical_scales() -> tuple[float, float, float]:
    volume = 2.70336e-20
    cell_volume = 2.0 * 2.0 * 2.0 * 1e-27
    nuclear_length = 8e14 * cell_volume
    magnetic_moment = 1700e3 * cell_volume
    b_h = 2.91e8
    common = 1e-2 / volume
    return (
        common * nuclear_length**2,
        common * (magnetic_moment * b_h) ** 2,
        common * nuclear_length * magnetic_moment * b_h,
    )


def stream_archived_rings(
    archive: Path,
    protocol: dict[str, Any],
) -> tuple[dict[str, dict[str, np.ndarray]], float]:
    selected = set(protocol["accuracy"]["archived_output_radial_indices"])
    maximum = max(selected)
    n_theta = protocol["workload"]["theta_nodes_upstream"]
    case_result: dict[str, dict[str, np.ndarray]] = {}
    start = perf_counter()
    with zipfile.ZipFile(archive) as bundle:
        for item in protocol["packing_hierarchy"]:
            member = protocol["dataset"]["output_archive"][
                "sans2d_member_template"
            ].format(archive_index=item["archive_index"])
            rows: dict[int, list[list[float]]] = {index: [] for index in selected}
            with bundle.open(member) as raw:
                reader = csv.reader(
                    line.decode("utf-8") for line in raw
                )
                header = next(reader)
                indices = {name: header.index(name) for name in ARCHIVED_COLUMNS}
                for flat_index, row in enumerate(reader):
                    radial, theta = divmod(flat_index, n_theta)
                    if radial in selected and theta < n_theta - 1:
                        rows[radial].append(
                            [float(row[indices[name]]) for name in ARCHIVED_COLUMNS]
                        )
                    if radial > maximum:
                        break
            ordered = sorted(selected)
            if any(len(rows[index]) != n_theta - 1 for index in ordered):
                raise RuntimeError(f"incomplete archived ring in {member}")
            stacked = np.stack([np.asarray(rows[index]) for index in ordered])
            case_result[item["id"]] = {
                name: stacked[..., offset]
                for offset, name in enumerate(ARCHIVED_COLUMNS)
            }
    return case_result, perf_counter() - start


def _archive_errors(
    predictions: dict[str, dict[str, Any]],
    archived: dict[str, dict[str, np.ndarray]],
    radial_indices: list[int],
) -> dict[str, Any]:
    upstream_order = (-np.arange(next(iter(archived.values()))["q"].shape[1])) % next(iter(archived.values()))["q"].shape[1]
    by_case: dict[str, dict[str, float]] = {}
    for case, reference in archived.items():
        by_case[case] = {}
        for name in OUTPUT_NAMES:
            values = predictions[case][name].detach().cpu().numpy()
            selected = values[np.asarray(radial_indices)][:, upstream_order]
            by_case[case][name] = relative_l2(selected, reference[name])
    worst = max(value for case in by_case.values() for value in case.values())
    return {"by_case": by_case, "worst_relative_l2": worst}


def bootstrap_total_speedup(
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    candidate_setup: float,
    baseline_setup: float,
    seed: int,
    resamples: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = candidate.size
    ratios = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sample = rng.integers(0, n, size=n)
        ratios[index] = (
            float(baseline_setup) + float(np.sum(baseline[sample]))
        ) / (
            float(candidate_setup) + float(np.sum(candidate[sample]))
        )
    point = (float(baseline_setup) + float(np.sum(baseline))) / (
        float(candidate_setup) + float(np.sum(candidate))
    )
    return {
        "point": point,
        "lower_95": float(np.quantile(ratios, 0.025)),
        "upper_95": float(np.quantile(ratios, 0.975)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import cupy as cp
    import cufinufft
    import torch

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    requested = protocol["dataset"]["orientation_count"] if args.max_orientations is None else int(args.max_orientations)
    if requested <= 0 or requested > protocol["dataset"]["orientation_count"]:
        raise ValueError("max orientations is outside 1..800")
    full_run = requested == protocol["dataset"]["orientation_count"]
    if full_run and args.source_archive is None:
        raise ValueError("the full run requires --source-archive")
    if full_run and not args.skip_archive_oracle and args.output_archive is None:
        raise ValueError("the full oracle run requires --output-archive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the prospective ensemble benchmark requires CUDA")
    torch.cuda.set_device(device)
    real_dtype = torch.float32
    complex_dtype = torch.complex64
    source = OrientationSource(
        source_archive=args.source_archive,
        reduced_dir=args.reduced_dir,
        protocol=protocol,
    )
    try:
        topology = (
            validate_source_archive(source.bundle, protocol)
            if source.bundle is not None
            else {"pass": True, "role": "reduced repeated-orientation smoke"}
        )
        if not topology["pass"]:
            raise RuntimeError(
                "source archive topology is incomplete: "
                + json.dumps(topology, sort_keys=True)
            )
        packed_ids, centers = load_centers(protocol, source)
        first_points, first_nuclear, first_magnetization = source.load(1)
        scale = protocol["dataset"]["source_coordinate_scale"]
        if source.bundle is None:
            first_points = np.ascontiguousarray(first_points * scale)
        if first_points.shape[0] != protocol["dataset"]["sources_per_orientation"]:
            raise ValueError("unexpected source count")
        phi_count = protocol["workload"]["unique_theta"]
        phi_step = 2.0 * np.pi / phi_count
        source_rotation = 0.5 * phi_step - 0.5 * np.pi
        first_rotated = rotate_points_about_z(first_points, source_rotation)
        rotated_centers = np.stack(
            [rotate_points_about_z(table, source_rotation) for table in centers]
        )
        q_radius = np.linspace(
            0.0,
            protocol["workload"]["q_max_inverse_nm"],
            protocol["workload"]["q_nodes"],
        )
        manifold = AxisymmetricManifold(
            u=np.arange(q_radius.size, dtype=float),
            q_perp=q_radius,
            q_z=np.zeros_like(q_radius),
            name="numagsans-example3-ensemble",
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
            n_r=protocol["workload"]["acfo_n_r"],
            n_z=protocol["workload"]["acfo_n_z"],
            n_phi=phi_count,
            r_max=radial_extent + 0.5,
            z_range=(-radial_extent - 0.5, radial_extent + 0.5),
            phase_sign=-1,
            hist_backend="numpy",
            circular_backend="auto",
            complex_dtype=np.complex64,
            margin=protocol["harmonic_support"]["mode_padding"],
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
            dtype="complex64",
            detector_directions=original_q_hat,
            source_deposition=protocol["workload"]["source_deposition"],
            kernel_backend="gpu_miller",
            active_r_indices_override=np.arange(protocol["workload"]["acfo_n_r"]),
        )
        sync(torch, device)
        acfo_setup_seconds = perf_counter() - acfo_setup_start
        if acfo.kernel_backend != protocol["harmonic_support"]["required_kernel_backend"]:
            raise RuntimeError("GPU Miller backend was not selected")
        if acfo.miller_recurrence_margin != protocol["harmonic_support"]["miller_recurrence_margin"]:
            raise RuntimeError("Miller recurrence margin differs from the frozen value")

        q_xyz = detector_q_grid(q_radius, phi)
        q_original = q_cloud_in_original_grid_frame(q_xyz, source_rotation)
        flat_q = q_xyz.reshape(-1, 3)
        flat_q_original = q_original.reshape(-1, 3)
        q_hat_torch = torch.as_tensor(
            original_q_hat, dtype=real_dtype, device=device
        )
        polarization_torch = torch.as_tensor(
            protocol["workload"]["polarization_acfo"],
            dtype=real_dtype,
            device=device,
        )
        q_xy_torch = torch.as_tensor(
            flat_q[:, :2], dtype=real_dtype, device=device
        )

        cp_real = cp.float32
        cp_complex = cp.complex64
        target_x = cp.asarray(flat_q[:, 0], dtype=cp_real)
        target_y = cp.asarray(flat_q[:, 1], dtype=cp_real)
        baseline_setup_start = perf_counter()
        type3_plan = cufinufft.Plan(
            3,
            2,
            n_trans=protocol["baseline_dispatch"]["n_trans"],
            eps=protocol["baseline_dispatch"]["eps"],
            isign=-1,
            dtype=protocol["workload"]["dtype"],
        )
        baseline_out = cp.empty(
            (4, protocol["workload"]["targets"]), dtype=cp_complex
        )
        cp.cuda.runtime.deviceSynchronize()
        baseline_setup_seconds = perf_counter() - baseline_setup_start

        warmup_count = int(
            protocol["timing_and_statistics"]["pre_timing_warmups_per_arm"]
        )
        warmup_weights = np.ascontiguousarray(
            np.column_stack([first_nuclear, first_magnetization]).T,
            dtype=np.float32,
        )
        warmup_weight_tensor = torch.as_tensor(
            warmup_weights, dtype=real_dtype, device=device
        )
        warmup_source_x = cp.asarray(first_rotated[:, 0], dtype=cp_real)
        warmup_source_y = cp.asarray(first_rotated[:, 1], dtype=cp_real)
        warmup_strengths = cp.asarray(warmup_weights, dtype=cp_complex)
        pre_timing_warmup_start = perf_counter()
        for warmup_index in range(warmup_count * 2):
            if warmup_index % 2 == 0:
                acfo.configure_source_updates(first_rotated)
                _ = acfo.amplitudes_from_weights(warmup_weight_tensor)
                sync(torch, device)
            else:
                type3_plan.setpts(
                    warmup_source_x,
                    warmup_source_y,
                    s=target_x,
                    t=target_y,
                )
                type3_plan.execute(warmup_strengths, out=baseline_out)
                cp.cuda.runtime.deviceSynchronize()
        pre_timing_warmup_seconds = perf_counter() - pre_timing_warmup_start

        dilute = {
            arm: {
                name: torch.zeros(
                    (q_radius.size, phi_count), dtype=real_dtype, device=device
                )
                for name in PRIMITIVE_OUTPUT_NAMES
            }
            for arm in ("acfo", "projected_type3")
        }
        packed = {
            arm: torch.zeros(
                (len(packed_ids), 4, q_radius.size, phi_count),
                dtype=complex_dtype,
                device=device,
            )
            for arm in ("acfo", "projected_type3")
        }
        timings = {
            "acfo": [],
            "projected_type3": [],
            "phase": [],
            "acfo_reduction": [],
            "projected_type3_reduction": [],
            "data_load": [],
        }
        accuracy_rows: list[dict[str, Any]] = []
        heldout = set(protocol["accuracy"]["held_out_orientation_indices"])
        direct_indices = np.unique(
            np.linspace(
                0,
                protocol["workload"]["targets"] - 1,
                protocol["accuracy"]["direct_targets_per_held_out_orientation"],
                dtype=np.int64,
            )
        )
        q_hat_flat = np.broadcast_to(
            original_q_hat[None, :, :],
            (q_radius.size, phi_count, 3),
        ).reshape(-1, 3)
        prefixes: list[dict[str, Any]] = []
        prefix_set = {
            value
            for value in protocol["workload"]["prefix_orientation_counts"]
            if value <= requested
        }

        for orientation in range(1, requested + 1):
            load_start = perf_counter()
            points, nuclear, magnetization = source.load(orientation)
            if source.bundle is None:
                points = np.ascontiguousarray(points * scale)
            timings["data_load"].append(perf_counter() - load_start)
            rotated = rotate_points_about_z(points, source_rotation)
            weights = np.ascontiguousarray(
                np.column_stack([nuclear, magnetization]).T, dtype=np.float32
            )
            arm_channels: dict[str, Any] = {}

            def evaluate_acfo() -> Any:
                sync(torch, device)
                start = perf_counter()
                acfo.configure_source_updates(rotated)
                weight_tensor = torch.as_tensor(
                    weights, dtype=real_dtype, device=device
                )
                amplitudes = acfo.amplitudes_from_weights(weight_tensor)
                sync(torch, device)
                timings["acfo"].append(perf_counter() - start)
                return _channels_from_torch_amplitudes(torch, amplitudes)

            def evaluate_type3() -> Any:
                cp.cuda.runtime.deviceSynchronize()
                start = perf_counter()
                source_x = cp.asarray(rotated[:, 0], dtype=cp_real)
                source_y = cp.asarray(rotated[:, 1], dtype=cp_real)
                strengths = cp.asarray(weights, dtype=cp_complex)
                type3_plan.setpts(
                    source_x, source_y, s=target_x, t=target_y
                )
                type3_plan.execute(strengths, out=baseline_out)
                cp.cuda.runtime.deviceSynchronize()
                timings["projected_type3"].append(perf_counter() - start)
                return torch.from_dlpack(baseline_out).reshape(
                    4, q_radius.size, phi_count
                )

            order = ("acfo", "projected_type3") if orientation % 2 else ("projected_type3", "acfo")
            for arm in order:
                arm_channels[arm] = evaluate_acfo() if arm == "acfo" else evaluate_type3()

            phase_start = perf_counter()
            center_xy = torch.as_tensor(
                rotated_centers[:, orientation - 1, :2],
                dtype=real_dtype,
                device=device,
            )
            phase = torch.exp(
                -1j * (center_xy @ q_xy_torch.T)
            ).reshape(len(packed_ids), q_radius.size, phi_count)
            sync(torch, device)
            timings["phase"].append(perf_counter() - phase_start)
            for arm in ("acfo", "projected_type3"):
                reduction_start = perf_counter()
                primitive = _primitive_torch(
                    torch, arm_channels[arm], q_hat_torch, polarization_torch
                )
                for name in PRIMITIVE_OUTPUT_NAMES:
                    dilute[arm][name].add_(primitive[name])
                packed[arm].add_(phase[:, None, ...] * arm_channels[arm][None, ...])
                sync(torch, device)
                timings[f"{arm}_reduction"].append(
                    perf_counter() - reduction_start
                )

            if orientation in heldout and source.bundle is not None:
                direct = direct_fourier_channel_subset(
                    points, weights, q_original, direct_indices
                )
                direct_outputs = numpy_cross_sections_flat(
                    direct,
                    q_hat_flat[direct_indices],
                    np.asarray(protocol["workload"]["polarization_acfo"]),
                )
                row: dict[str, Any] = {"orientation": orientation, "methods": {}}
                for arm in ("acfo", "projected_type3"):
                    selected = (
                        arm_channels[arm].reshape(4, -1)[:, direct_indices]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    outputs = numpy_cross_sections_flat(
                        selected,
                        q_hat_flat[direct_indices],
                        np.asarray(protocol["workload"]["polarization_acfo"]),
                    )
                    output_errors = {
                        name: relative_l2(outputs[name], direct_outputs[name])
                        for name in OUTPUT_NAMES
                    }
                    row["methods"][arm] = {
                        "amplitude_relative_l2": relative_l2(selected, direct),
                        "output_relative_l2": output_errors,
                        "worst_output_relative_l2": max(output_errors.values()),
                    }
                accuracy_rows.append(row)

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
                    for arm in ("acfo", "projected_type3")
                }
                errors = _hierarchy_errors(
                    torch, outputs["acfo"], outputs["projected_type3"]
                )
                prefixes.append(
                    {
                        "orientations": orientation,
                        "frozen_cold_dispatch": frozen_dispatch_decision(
                            orientation, qualification_pass=True, cold_process=True
                        ),
                        "acfo_cold_fourier_total_seconds": gpu_miller_jit_seconds
                        + acfo_setup_seconds
                        + float(np.sum(timings["acfo"])),
                        "projected_type3_fourier_total_seconds": baseline_setup_seconds
                        + float(np.sum(timings["projected_type3"])),
                        "measured_favorable": (
                            "ACFO"
                            if gpu_miller_jit_seconds
                            + acfo_setup_seconds
                            + float(np.sum(timings["acfo"]))
                            < baseline_setup_seconds
                            + float(np.sum(timings["projected_type3"]))
                            else "PROJECTED_TYPE3"
                        ),
                        "ensemble_accuracy": errors,
                    }
                )
            if orientation == 1 or orientation % 25 == 0 or orientation == requested:
                print(
                    f"orientation {orientation}/{requested}: "
                    f"ACFO Fourier {np.sum(timings['acfo']):.3f} s, "
                    f"projected type-3 {np.sum(timings['projected_type3']):.3f} s",
                    flush=True,
                )

        sync(torch, device)
        candidate_samples = np.asarray(timings["acfo"])
        baseline_samples = np.asarray(timings["projected_type3"])
        crossover = cumulative_crossover(
            candidate_samples,
            baseline_samples,
            candidate_setup=acfo_setup_seconds,
            baseline_setup=baseline_setup_seconds,
            candidate_cold_start=gpu_miller_jit_seconds,
        )
        crossover_json = {
            key: value
            for key, value in crossover.items()
            if not isinstance(value, np.ndarray)
        }
        final_outputs = {
            arm: _hierarchy_outputs_torch(
                torch,
                dilute[arm],
                packed[arm],
                q_hat_torch,
                polarization_torch,
                packed_ids,
            )
            for arm in ("acfo", "projected_type3")
        }
        final_errors = _hierarchy_errors(
            torch,
            final_outputs["acfo"],
            final_outputs["projected_type3"],
        )
        archive_validation: dict[str, Any] = {"performed": False}
        if not args.skip_archive_oracle and args.output_archive is not None:
            archived, stream_seconds = stream_archived_rings(
                args.output_archive, protocol
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
                for arm in ("acfo", "projected_type3")
            }
            archive_validation = {
                "performed": True,
                "stream_seconds": stream_seconds,
                "radial_indices": protocol["accuracy"]["archived_output_radial_indices"],
                "acfo": _archive_errors(
                    physical_outputs["acfo"],
                    archived,
                    protocol["accuracy"]["archived_output_radial_indices"],
                ),
                "projected_type3": _archive_errors(
                    physical_outputs["projected_type3"],
                    archived,
                    protocol["accuracy"]["archived_output_radial_indices"],
                ),
            }

        heldout_worst_amplitude = max(
            (
                method["amplitude_relative_l2"]
                for row in accuracy_rows
                for method in row["methods"].values()
            ),
            default=0.0,
        )
        heldout_worst_output = max(
            (
                method["worst_output_relative_l2"]
                for row in accuracy_rows
                for method in row["methods"].values()
            ),
            default=0.0,
        )
        qualification_pass = bool(
            heldout_worst_amplitude <= protocol["accuracy"]["amplitude_relative_l2_max"]
            and heldout_worst_output <= protocol["accuracy"]["worst_twelve_output_relative_l2_max"]
            and final_errors["worst_relative_l2"]
            <= protocol["accuracy"]["acfo_vs_projected_type3_full_ensemble_worst_output_relative_l2_max"]
        )
        prefix_scored = [row for row in prefixes if row["orientations"] >= 50 or row["orientations"] < 50]
        correct = sum(
            row["frozen_cold_dispatch"] == row["measured_favorable"]
            for row in prefix_scored
        )
        false_go = sum(
            row["frozen_cold_dispatch"] == "ACFO"
            and row["measured_favorable"] != "ACFO"
            for row in prefix_scored
        )
        cold_speedup = bootstrap_total_speedup(
            candidate_samples,
            baseline_samples,
            candidate_setup=acfo_setup_seconds + gpu_miller_jit_seconds,
            baseline_setup=baseline_setup_seconds,
            seed=protocol["timing_and_statistics"]["bootstrap_seed"],
            resamples=(
                protocol["timing_and_statistics"]["bootstrap_resamples"]
                if full_run
                else min(1000, protocol["timing_and_statistics"]["bootstrap_resamples"])
            ),
        )
        timing_pass = bool(
            not full_run
            or cold_speedup["lower_95"]
            >= protocol["timing_and_statistics"]["required_full_800_lower_95_speedup"]
        )
        archive_pass = bool(
            not archive_validation["performed"]
            or (
                archive_validation["acfo"]["worst_relative_l2"]
                <= protocol["accuracy"]["archived_output_worst_relative_l2_max"]
                and archive_validation["projected_type3"]["worst_relative_l2"]
                <= protocol["accuracy"]["archived_output_worst_relative_l2_max"]
            )
        )
        dispatch_pass = bool(
            not full_run
            or (
                false_go == protocol["timing_and_statistics"]["false_go_count_must_equal"]
                and correct / max(len(prefix_scored), 1)
                >= protocol["timing_and_statistics"]["required_dispatch_score"]
                and prefixes[-1]["frozen_cold_dispatch"] == prefixes[-1]["measured_favorable"]
            )
        )
        passed = qualification_pass and timing_pass and archive_pass and dispatch_pass
        return {
            "schema": "numagsans-example3-ensemble-dispatch-result-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "verdict": (
                "PASS_FROZEN_DISPATCH_AND_CROSSOVER"
                if passed
                else "FAIL_PRESERVE_FROZEN_RULE"
            ),
            "mode": "full" if full_run else "smoke",
            "protocol_status": protocol["status"],
            "environment": {
                "gpu": torch.cuda.get_device_name(device),
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
                "targets": protocol["workload"]["targets"],
                "streaming": True,
            },
            "harmonic_support": {
                "max_h": acfo.max_h,
                "tier_cutoffs": acfo.optimization_stats["harmonic_tiers"],
                "active_radii": acfo.optimization_stats["active_radii"],
                "active_radii_overridden": acfo.optimization_stats["active_r_indices_overridden"],
                "kernel_backend": acfo.kernel_backend,
                "miller_recurrence_margin": acfo.miller_recurrence_margin,
                "packing_centers_entered_qR": False,
            },
            "setup_seconds": {
                "gpu_miller_jit_cold_start": gpu_miller_jit_seconds,
                "acfo_shared_kernel_and_plan": acfo_setup_seconds,
                "projected_type3_plan": baseline_setup_seconds,
                "pre_timing_warmups_per_arm_excluded": warmup_count,
                "pre_timing_warmup_seconds_excluded": pre_timing_warmup_seconds,
            },
            "orientation_fourier_samples_seconds": {
                "acfo": timings["acfo"],
                "projected_type3": timings["projected_type3"],
            },
            "descriptive_common_reduction_seconds": {
                "phase": timings["phase"],
                "acfo": timings["acfo_reduction"],
                "projected_type3": timings["projected_type3_reduction"],
                "data_load_excluded": timings["data_load"],
            },
            "cold_total_speedup_projected_type3_over_acfo": cold_speedup,
            "measured_crossover": crossover_json,
            "frozen_expected_crossover": protocol["crossover"]["expected_from_prior_pilot"],
            "prefix_dispatch": prefixes,
            "dispatch_score": {
                "correct": correct,
                "total": len(prefix_scored),
                "fraction": correct / max(len(prefix_scored), 1),
                "false_go_count": false_go,
            },
            "heldout_accuracy": {
                "rows": accuracy_rows,
                "worst_amplitude_relative_l2": heldout_worst_amplitude,
                "worst_output_relative_l2": heldout_worst_output,
            },
            "full_ensemble_acfo_vs_projected_type3": final_errors,
            "archived_output_validation": archive_validation,
            "gates": {
                "qualification_pass": qualification_pass,
                "timing_pass": timing_pass,
                "archive_pass": archive_pass,
                "dispatch_pass": dispatch_pass,
            },
        }
    finally:
        source.close()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "NUMAGSANS_EXAMPLE3_ENSEMBLE_DISPATCH_PROTOCOL.json",
    )
    p.add_argument("--source-archive", type=Path)
    p.add_argument("--output-archive", type=Path)
    p.add_argument(
        "--reduced-dir",
        type=Path,
        default=ROOT / "inputs/numagsans_example3_reduced",
    )
    p.add_argument("--max-orientations", type=int)
    p.add_argument("--skip-archive-oracle", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmark_results/numagsans_example3_ensemble_dispatch.json",
    )
    return p


def main() -> int:
    args = parser().parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["verdict"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
