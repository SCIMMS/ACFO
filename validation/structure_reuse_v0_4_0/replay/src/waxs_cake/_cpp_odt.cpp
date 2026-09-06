#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

namespace py = pybind11;

namespace {

using Complex128 = std::complex<double>;
using ComplexArray = py::array_t<Complex128, py::array::c_style | py::array::forcecast>;
using FloatArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Int64Array = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;

unsigned int choose_thread_count_capped(
    std::int64_t tasks,
    std::int64_t work_per_task,
    unsigned int auto_cap
) {
    if (tasks <= 1 || tasks * std::max<std::int64_t>(work_per_task, 1) < 100000) {
        return 1;
    }
    unsigned int hardware = std::thread::hardware_concurrency();
    if (hardware == 0) {
        hardware = 1;
    }
    const unsigned int capped_hardware = std::min(hardware, auto_cap);
    return std::max(1u, std::min<unsigned int>(capped_hardware, static_cast<unsigned int>(tasks)));
}

unsigned int choose_thread_count(std::int64_t tasks, std::int64_t work_per_task) {
    return choose_thread_count_capped(tasks, work_per_task, 16);
}

template <typename Func>
void parallel_for(std::int64_t tasks, unsigned int threads, Func func) {
    if (threads <= 1 || tasks <= 1) {
        func(0, tasks);
        return;
    }

    std::vector<std::thread> workers;
    workers.reserve(threads);
    const std::int64_t block = (tasks + static_cast<std::int64_t>(threads) - 1) /
                               static_cast<std::int64_t>(threads);
    for (unsigned int thread = 0; thread < threads; ++thread) {
        const std::int64_t start = static_cast<std::int64_t>(thread) * block;
        const std::int64_t stop = std::min(tasks, start + block);
        if (start >= stop) {
            break;
        }
        workers.emplace_back([=, &func]() { func(start, stop); });
    }
    for (auto& worker : workers) {
        worker.join();
    }
}

void validate_forward_shapes(
    const py::buffer_info& coeff_h,
    const py::buffer_info& radial,
    const py::buffer_info& axial,
    const py::buffer_info& angular
) {
    if (coeff_h.ndim != 3) {
        throw std::invalid_argument("coeff_h must have shape (nr, nz, nh)");
    }
    if (radial.ndim != 3) {
        throw std::invalid_argument("radial must have shape (nh, nq, nr)");
    }
    if (axial.ndim != 2) {
        throw std::invalid_argument("axial must have shape (nq, nz)");
    }
    if (angular.ndim != 2) {
        throw std::invalid_argument("angular must have shape (nh, nq)");
    }
    if (coeff_h.shape[0] != radial.shape[2] || coeff_h.shape[1] != axial.shape[1] ||
        coeff_h.shape[2] != radial.shape[0] || coeff_h.shape[2] != angular.shape[0] ||
        radial.shape[1] != axial.shape[0] || radial.shape[1] != angular.shape[1]) {
        throw std::invalid_argument("inconsistent ODT structured forward shapes");
    }
}

void validate_adjoint_shapes(
    const py::buffer_info& residual,
    const py::buffer_info& radial,
    const py::buffer_info& axial,
    const py::buffer_info& angular
) {
    if (residual.ndim != 1) {
        throw std::invalid_argument("residual must have shape (nq,)");
    }
    if (radial.ndim != 3) {
        throw std::invalid_argument("radial must have shape (nh, nq, nr)");
    }
    if (axial.ndim != 2) {
        throw std::invalid_argument("axial must have shape (nq, nz)");
    }
    if (angular.ndim != 2) {
        throw std::invalid_argument("angular must have shape (nh, nq)");
    }
    if (residual.shape[0] != radial.shape[1] || residual.shape[0] != axial.shape[0] ||
        residual.shape[0] != angular.shape[1] || radial.shape[0] != angular.shape[0]) {
        throw std::invalid_argument("inconsistent ODT structured adjoint shapes");
    }
}

void validate_axis_grid_forward_shapes(
    const py::buffer_info& coeff_h,
    const py::buffer_info& radial,
    const py::buffer_info& axial,
    const py::buffer_info& mode_phase,
    const py::buffer_info& h_slots,
    std::int64_t cap_phi
) {
    if (coeff_h.ndim != 4) {
        throw std::invalid_argument("coeff_h must have shape (n_illum, nr, nz, nh)");
    }
    if (radial.ndim != 3) {
        throw std::invalid_argument("radial must have shape (nh, cap_radial, nr)");
    }
    if (axial.ndim != 2) {
        throw std::invalid_argument("axial must have shape (cap_radial, nz)");
    }
    if (mode_phase.ndim != 1 || h_slots.ndim != 1) {
        throw std::invalid_argument("mode_phase and h_slots must have shape (nh,)");
    }
    if (cap_phi <= 0) {
        throw std::invalid_argument("cap_phi must be positive");
    }
    if (coeff_h.shape[1] != radial.shape[2] || coeff_h.shape[2] != axial.shape[1] ||
        coeff_h.shape[3] != radial.shape[0] || coeff_h.shape[3] != mode_phase.shape[0] ||
        coeff_h.shape[3] != h_slots.shape[0] || radial.shape[1] != axial.shape[0]) {
        throw std::invalid_argument("inconsistent ODT axis-grid forward shapes");
    }
}

void validate_axis_grid_adjoint_shapes(
    const py::buffer_info& residual_modes,
    const py::buffer_info& radial,
    const py::buffer_info& axial,
    const py::buffer_info& mode_phase,
    const py::buffer_info& h_slots
) {
    if (residual_modes.ndim != 3) {
        throw std::invalid_argument("residual_modes must have shape (n_illum, cap_radial, cap_phi)");
    }
    if (radial.ndim != 3) {
        throw std::invalid_argument("radial must have shape (nh, cap_radial, nr)");
    }
    if (axial.ndim != 2) {
        throw std::invalid_argument("axial must have shape (cap_radial, nz)");
    }
    if (mode_phase.ndim != 1 || h_slots.ndim != 1) {
        throw std::invalid_argument("mode_phase and h_slots must have shape (nh,)");
    }
    if (radial.shape[0] != mode_phase.shape[0] || radial.shape[0] != h_slots.shape[0] ||
        residual_modes.shape[1] != radial.shape[1] || axial.shape[0] != radial.shape[1]) {
        throw std::invalid_argument("inconsistent ODT axis-grid adjoint shapes");
    }
}

void validate_phase_selected_dft_shapes(
    const py::buffer_info& coeff,
    const py::buffer_info& phase,
    const py::buffer_info& twiddle
) {
    if (coeff.ndim != 3) {
        throw std::invalid_argument("coeff must have shape (nr, nz, n_beta)");
    }
    if (phase.ndim != 4) {
        throw std::invalid_argument("phase must have shape (n_illum, nr, nz, n_beta)");
    }
    if (twiddle.ndim != 2) {
        throw std::invalid_argument("twiddle must have shape (nh, n_beta)");
    }
    if (phase.shape[1] != coeff.shape[0] || phase.shape[2] != coeff.shape[1] ||
        phase.shape[3] != coeff.shape[2] || twiddle.shape[1] != coeff.shape[2]) {
        throw std::invalid_argument("inconsistent ODT phase-selected DFT shapes");
    }
}

void validate_phase_selected_idft_shapes(
    const py::buffer_info& compact,
    const py::buffer_info& phase,
    const py::buffer_info& twiddle
) {
    if (compact.ndim != 4) {
        throw std::invalid_argument("compact must have shape (n_illum, nr, nz, nh)");
    }
    if (phase.ndim != 4) {
        throw std::invalid_argument("phase must have shape (n_illum, nr, nz, n_beta)");
    }
    if (twiddle.ndim != 2) {
        throw std::invalid_argument("twiddle must have shape (nh, n_beta)");
    }
    if (compact.shape[0] != phase.shape[0] || compact.shape[1] != phase.shape[1] ||
        compact.shape[2] != phase.shape[2] || compact.shape[3] != twiddle.shape[0] ||
        phase.shape[3] != twiddle.shape[1]) {
        throw std::invalid_argument("inconsistent ODT phase-selected adjoint DFT shapes");
    }
}

void validate_source_slots(
    const std::int64_t* source_slots,
    std::int64_t count,
    std::int64_t n_beta
) {
    if (n_beta <= 0) {
        throw std::invalid_argument("n_beta must be positive");
    }
    for (std::int64_t index = 0; index < count; ++index) {
        const std::int64_t slot = source_slots[index];
        if (slot < 0 || slot >= n_beta) {
            throw std::invalid_argument("source_slots contains an out-of-range value");
        }
    }
}

void validate_cone_axis_forward_shapes(
    const py::buffer_info& coeff,
    const py::buffer_info& transverse,
    const py::buffer_info& psi,
    const py::buffer_info& axial_phase,
    const py::buffer_info& source_slots
) {
    if (coeff.ndim != 3) {
        throw std::invalid_argument("coeff_h_full must have shape (nr, nz, n_beta)");
    }
    if (transverse.ndim != 2) {
        throw std::invalid_argument("transverse_coeff must have shape (nl, nr)");
    }
    if (psi.ndim != 2) {
        throw std::invalid_argument("psi_phase must have shape (n_illum, nl)");
    }
    if (axial_phase.ndim != 1) {
        throw std::invalid_argument("axial_phase must have shape (nz,)");
    }
    if (source_slots.ndim != 2) {
        throw std::invalid_argument("source_slots must have shape (nh, nl)");
    }
    if (transverse.shape[1] != coeff.shape[0] || axial_phase.shape[0] != coeff.shape[1] ||
        psi.shape[1] != transverse.shape[0] || source_slots.shape[1] != transverse.shape[0]) {
        throw std::invalid_argument("inconsistent cone-axis forward decomposition shapes");
    }
}

void validate_cone_axis_forward_fold_shapes(
    const py::buffer_info& coeff,
    const py::buffer_info& radial,
    const py::buffer_info& axial,
    const py::buffer_info& mode_phase,
    const py::buffer_info& h_slots,
    const py::buffer_info& transverse,
    const py::buffer_info& psi,
    const py::buffer_info& axial_phase,
    const py::buffer_info& source_slots,
    std::int64_t cap_phi
) {
    validate_cone_axis_forward_shapes(coeff, transverse, psi, axial_phase, source_slots);
    if (radial.ndim != 3) {
        throw std::invalid_argument("radial must have shape (nh, cap_radial, nr)");
    }
    if (axial.ndim != 2) {
        throw std::invalid_argument("axial must have shape (cap_radial, nz)");
    }
    if (mode_phase.ndim != 1 || h_slots.ndim != 1) {
        throw std::invalid_argument("mode_phase and h_slots must have shape (nh,)");
    }
    if (cap_phi <= 0) {
        throw std::invalid_argument("cap_phi must be positive");
    }
    if (radial.shape[0] != source_slots.shape[0] || radial.shape[2] != coeff.shape[0] ||
        axial.shape[0] != radial.shape[1] || axial.shape[1] != coeff.shape[1] ||
        mode_phase.shape[0] != source_slots.shape[0] ||
        h_slots.shape[0] != source_slots.shape[0]) {
        throw std::invalid_argument("inconsistent cone-axis fused forward-fold shapes");
    }
}

void validate_cone_axis_adjoint_shapes(
    const py::buffer_info& compact,
    const py::buffer_info& transverse,
    const py::buffer_info& psi,
    const py::buffer_info& axial_phase,
    const py::buffer_info& source_slots,
    std::int64_t n_beta
) {
    if (compact.ndim != 4) {
        throw std::invalid_argument("compact must have shape (n_illum, nr, nz, nh)");
    }
    if (transverse.ndim != 2) {
        throw std::invalid_argument("transverse_coeff must have shape (nl, nr)");
    }
    if (psi.ndim != 2) {
        throw std::invalid_argument("psi_phase must have shape (n_illum, nl)");
    }
    if (axial_phase.ndim != 1) {
        throw std::invalid_argument("axial_phase must have shape (nz,)");
    }
    if (source_slots.ndim != 2) {
        throw std::invalid_argument("source_slots must have shape (nh, nl)");
    }
    if (transverse.shape[1] != compact.shape[1] || axial_phase.shape[0] != compact.shape[2] ||
        psi.shape[0] != compact.shape[0] || psi.shape[1] != transverse.shape[0] ||
        source_slots.shape[0] != compact.shape[3] || source_slots.shape[1] != transverse.shape[0]) {
        throw std::invalid_argument("inconsistent cone-axis adjoint decomposition shapes");
    }
    if (n_beta <= 0) {
        throw std::invalid_argument("n_beta must be positive");
    }
}

void validate_cone_axis_adjoint_unfold_scatter_shapes(
    const py::buffer_info& residual_modes,
    const py::buffer_info& radial,
    const py::buffer_info& axial,
    const py::buffer_info& mode_phase,
    const py::buffer_info& h_slots,
    const py::buffer_info& transverse,
    const py::buffer_info& psi,
    const py::buffer_info& axial_phase,
    const py::buffer_info& source_slots,
    std::int64_t n_beta
) {
    validate_axis_grid_adjoint_shapes(residual_modes, radial, axial, mode_phase, h_slots);
    if (transverse.ndim != 2) {
        throw std::invalid_argument("transverse_coeff must have shape (nl, nr)");
    }
    if (psi.ndim != 2) {
        throw std::invalid_argument("psi_phase must have shape (n_illum, nl)");
    }
    if (axial_phase.ndim != 1) {
        throw std::invalid_argument("axial_phase must have shape (nz,)");
    }
    if (source_slots.ndim != 2) {
        throw std::invalid_argument("source_slots must have shape (nh, nl)");
    }
    if (radial.shape[2] != transverse.shape[1] || axial.shape[1] != axial_phase.shape[0] ||
        residual_modes.shape[0] != psi.shape[0] || psi.shape[1] != transverse.shape[0] ||
        source_slots.shape[0] != radial.shape[0] || source_slots.shape[1] != transverse.shape[0]) {
        throw std::invalid_argument("inconsistent cone-axis fused adjoint-unfold shapes");
    }
    if (n_beta <= 0) {
        throw std::invalid_argument("n_beta must be positive");
    }
}

void validate_cone_axis_l_pruning_shapes(
    const py::buffer_info& l_offsets,
    const py::buffer_info& l_indices,
    std::int64_t nr,
    std::int64_t nl
) {
    if (l_offsets.ndim != 1) {
        throw std::invalid_argument("l_offsets must have shape (nr + 1,)");
    }
    if (l_indices.ndim != 1) {
        throw std::invalid_argument("l_indices must have shape (n_active_l,)");
    }
    if (l_offsets.shape[0] != nr + 1) {
        throw std::invalid_argument("l_offsets must have length nr + 1");
    }
    const auto* offsets = static_cast<const std::int64_t*>(l_offsets.ptr);
    const auto* indices = static_cast<const std::int64_t*>(l_indices.ptr);
    if (offsets[0] != 0 || offsets[nr] != l_indices.shape[0]) {
        throw std::invalid_argument("l_offsets endpoints do not match l_indices");
    }
    for (std::int64_t r = 0; r < nr; ++r) {
        if (offsets[r] > offsets[r + 1]) {
            throw std::invalid_argument("l_offsets must be nondecreasing");
        }
    }
    for (std::int64_t index = 0; index < l_indices.shape[0]; ++index) {
        const std::int64_t l = indices[index];
        if (l < 0 || l >= nl) {
            throw std::invalid_argument("l_indices contains an out-of-range value");
        }
    }
}

void validate_cone_axis_prepared_l_offsets(
    const py::buffer_info& l_offsets,
    std::int64_t nr,
    std::int64_t active_l_total
) {
    if (l_offsets.ndim != 1) {
        throw std::invalid_argument("l_offsets must have shape (nr + 1,)");
    }
    if (l_offsets.shape[0] != nr + 1) {
        throw std::invalid_argument("l_offsets must have length nr + 1");
    }
    const auto* offsets = static_cast<const std::int64_t*>(l_offsets.ptr);
    if (offsets[0] != 0 || offsets[nr] != active_l_total) {
        throw std::invalid_argument("l_offsets endpoints do not match prepared active-l arrays");
    }
    for (std::int64_t r = 0; r < nr; ++r) {
        if (offsets[r] > offsets[r + 1]) {
            throw std::invalid_argument("l_offsets must be nondecreasing");
        }
    }
}

void validate_same_direction_forward_fold_shapes(
    const py::buffer_info& coeff,
    const py::buffer_info& radial,
    const py::buffer_info& axial,
    const py::buffer_info& mode_phase,
    const py::buffer_info& h_slots,
    const py::buffer_info& transverse,
    const py::buffer_info& axial_phase,
    const py::buffer_info& source_slots,
    std::int64_t cap_phi
) {
    if (coeff.ndim != 3) {
        throw std::invalid_argument("coeff_h_full must have shape (nr, nz, n_beta)");
    }
    if (transverse.ndim != 3) {
        throw std::invalid_argument("transverse_by_mag must have shape (n_mag, nl, nr)");
    }
    if (axial_phase.ndim != 2) {
        throw std::invalid_argument("axial_phase_by_mag must have shape (n_mag, nz)");
    }
    if (source_slots.ndim != 2) {
        throw std::invalid_argument("source_slots must have shape (nh, nl)");
    }
    if (radial.ndim != 3) {
        throw std::invalid_argument("radial must have shape (nh, cap_radial, nr)");
    }
    if (axial.ndim != 2) {
        throw std::invalid_argument("axial must have shape (cap_radial, nz)");
    }
    if (mode_phase.ndim != 1 || h_slots.ndim != 1) {
        throw std::invalid_argument("mode_phase and h_slots must have shape (nh,)");
    }
    if (cap_phi <= 0) {
        throw std::invalid_argument("cap_phi must be positive");
    }
    if (transverse.shape[0] != axial_phase.shape[0] ||
        transverse.shape[1] != source_slots.shape[1] ||
        transverse.shape[2] != coeff.shape[0] ||
        axial_phase.shape[1] != coeff.shape[1] ||
        radial.shape[0] != source_slots.shape[0] ||
        radial.shape[2] != coeff.shape[0] ||
        axial.shape[0] != radial.shape[1] ||
        axial.shape[1] != coeff.shape[1] ||
        mode_phase.shape[0] != source_slots.shape[0] ||
        h_slots.shape[0] != source_slots.shape[0]) {
        throw std::invalid_argument("inconsistent same-direction forward-fold shapes");
    }
}

void validate_same_direction_adjoint_unfold_scatter_shapes(
    const py::buffer_info& residual_modes,
    const py::buffer_info& radial,
    const py::buffer_info& axial,
    const py::buffer_info& mode_phase,
    const py::buffer_info& h_slots,
    const py::buffer_info& transverse,
    const py::buffer_info& axial_phase,
    const py::buffer_info& source_slots,
    std::int64_t n_beta
) {
    validate_axis_grid_adjoint_shapes(residual_modes, radial, axial, mode_phase, h_slots);
    if (transverse.ndim != 3) {
        throw std::invalid_argument("transverse_by_mag must have shape (n_mag, nl, nr)");
    }
    if (axial_phase.ndim != 2) {
        throw std::invalid_argument("axial_phase_by_mag must have shape (n_mag, nz)");
    }
    if (source_slots.ndim != 2) {
        throw std::invalid_argument("source_slots must have shape (nh, nl)");
    }
    if (transverse.shape[0] != residual_modes.shape[0] ||
        axial_phase.shape[0] != residual_modes.shape[0] ||
        transverse.shape[1] != source_slots.shape[1] ||
        transverse.shape[2] != radial.shape[2] ||
        axial_phase.shape[1] != axial.shape[1] ||
        source_slots.shape[0] != radial.shape[0]) {
        throw std::invalid_argument("inconsistent same-direction adjoint-unfold shapes");
    }
    if (n_beta <= 0) {
        throw std::invalid_argument("n_beta must be positive");
    }
}

void validate_svd_rank_forward_fold_shapes(
    const py::buffer_info& coeff,
    const py::buffer_info& radial,
    const py::buffer_info& axial,
    const py::buffer_info& mode_phase,
    const py::buffer_info& h_slots,
    const py::buffer_info& weights,
    const py::buffer_info& source_slots,
    std::int64_t cap_phi
) {
    if (coeff.ndim != 3) {
        throw std::invalid_argument("coeff_h_full must have shape (nr, nz, n_beta)");
    }
    if (weights.ndim != 4) {
        throw std::invalid_argument("svd weights must have shape (rank, nl, nr, nz)");
    }
    if (source_slots.ndim != 2) {
        throw std::invalid_argument("source_slots must have shape (nh, nl)");
    }
    if (radial.ndim != 3) {
        throw std::invalid_argument("radial must have shape (nh, cap_radial, nr)");
    }
    if (axial.ndim != 2) {
        throw std::invalid_argument("axial must have shape (cap_radial, nz)");
    }
    if (mode_phase.ndim != 1 || h_slots.ndim != 1) {
        throw std::invalid_argument("mode_phase and h_slots must have shape (nh,)");
    }
    if (cap_phi <= 0) {
        throw std::invalid_argument("cap_phi must be positive");
    }
    if (weights.shape[0] <= 0 ||
        weights.shape[1] != source_slots.shape[1] ||
        weights.shape[2] != coeff.shape[0] ||
        weights.shape[3] != coeff.shape[1] ||
        radial.shape[0] != source_slots.shape[0] ||
        radial.shape[2] != coeff.shape[0] ||
        axial.shape[0] != radial.shape[1] ||
        axial.shape[1] != coeff.shape[1] ||
        mode_phase.shape[0] != source_slots.shape[0] ||
        h_slots.shape[0] != source_slots.shape[0]) {
        throw std::invalid_argument("inconsistent SVD rank forward-fold shapes");
    }
}

void validate_svd_rank_adjoint_unfold_scatter_shapes(
    const py::buffer_info& residual_modes,
    const py::buffer_info& radial,
    const py::buffer_info& axial,
    const py::buffer_info& mode_phase,
    const py::buffer_info& h_slots,
    const py::buffer_info& weights,
    const py::buffer_info& source_slots,
    std::int64_t n_beta
) {
    validate_axis_grid_adjoint_shapes(residual_modes, radial, axial, mode_phase, h_slots);
    if (weights.ndim != 4) {
        throw std::invalid_argument("svd weights must have shape (rank, nl, nr, nz)");
    }
    if (source_slots.ndim != 2) {
        throw std::invalid_argument("source_slots must have shape (nh, nl)");
    }
    if (weights.shape[0] != residual_modes.shape[0] ||
        weights.shape[1] != source_slots.shape[1] ||
        weights.shape[2] != radial.shape[2] ||
        weights.shape[3] != axial.shape[1] ||
        source_slots.shape[0] != radial.shape[0]) {
        throw std::invalid_argument("inconsistent SVD rank adjoint-unfold shapes");
    }
    if (n_beta <= 0) {
        throw std::invalid_argument("n_beta must be positive");
    }
}

void validate_resample4_shapes(
    const py::buffer_info& values,
    const py::buffer_info& indices,
    const py::buffer_info& weights
) {
    if (values.ndim != 1) {
        throw std::invalid_argument("values must have shape (n_source,)");
    }
    if (indices.ndim != 2 || weights.ndim != 2) {
        throw std::invalid_argument("indices and weights must have shape (n_target, 4)");
    }
    if (indices.shape[1] != 4 || weights.shape[1] != 4 || indices.shape[0] != weights.shape[0]) {
        throw std::invalid_argument("inconsistent 4-point resampling shapes");
    }
}

void validate_resample4_indices(
    const std::int64_t* indices,
    std::int64_t count,
    std::int64_t source_size
) {
    if (source_size < 0) {
        throw std::invalid_argument("source_size must be non-negative");
    }
    for (std::int64_t item = 0; item < count; ++item) {
        const std::int64_t index = indices[item];
        if (index < 0 || index >= source_size) {
            throw std::invalid_argument("resampling index out of range");
        }
    }
}

ComplexArray forward_contract(
    ComplexArray coeff_h_array,
    FloatArray radial_array,
    ComplexArray axial_array,
    ComplexArray angular_array,
    std::int64_t requested_threads
) {
    const py::buffer_info coeff_info = coeff_h_array.request();
    const py::buffer_info radial_info = radial_array.request();
    const py::buffer_info axial_info = axial_array.request();
    const py::buffer_info angular_info = angular_array.request();
    validate_forward_shapes(coeff_info, radial_info, axial_info, angular_info);

    const std::int64_t nr = static_cast<std::int64_t>(coeff_info.shape[0]);
    const std::int64_t nz = static_cast<std::int64_t>(coeff_info.shape[1]);
    const std::int64_t nh = static_cast<std::int64_t>(coeff_info.shape[2]);
    const std::int64_t nq = static_cast<std::int64_t>(radial_info.shape[1]);

    auto out = ComplexArray({nq});
    py::buffer_info out_info = out.request();

    const auto* coeff = static_cast<const Complex128*>(coeff_info.ptr);
    const auto* radial = static_cast<const double*>(radial_info.ptr);
    const auto* axial = static_cast<const Complex128*>(axial_info.ptr);
    const auto* angular = static_cast<const Complex128*>(angular_info.ptr);
    auto* out_ptr = static_cast<Complex128*>(out_info.ptr);

    unsigned int threads = 1;
    if (requested_threads > 0) {
        threads = static_cast<unsigned int>(requested_threads);
    } else {
        threads = choose_thread_count(nq, nh * nr * nz);
    }

    {
        py::gil_scoped_release release;
        parallel_for(nq, threads, [&](std::int64_t start, std::int64_t stop) {
            for (std::int64_t m = start; m < stop; ++m) {
                double out_re = 0.0;
                double out_im = 0.0;
                const Complex128* axial_row = axial + m * nz;
                for (std::int64_t h = 0; h < nh; ++h) {
                    double inner_re = 0.0;
                    double inner_im = 0.0;
                    const double* radial_row = radial + (h * nq + m) * nr;
                    for (std::int64_t r = 0; r < nr; ++r) {
                        const double rad = radial_row[r];
                        const Complex128* coeff_base = coeff + (r * nz * nh + h);
                        for (std::int64_t z = 0; z < nz; ++z) {
                            const Complex128 c = coeff_base[z * nh];
                            const Complex128 a = axial_row[z];
                            inner_re += rad * (c.real() * a.real() - c.imag() * a.imag());
                            inner_im += rad * (c.real() * a.imag() + c.imag() * a.real());
                        }
                    }
                    const Complex128 ang = angular[h * nq + m];
                    out_re += ang.real() * inner_re - ang.imag() * inner_im;
                    out_im += ang.real() * inner_im + ang.imag() * inner_re;
                }
                out_ptr[m] = Complex128{out_re, out_im};
            }
        });
    }

    return out;
}

ComplexArray resample4_interpolate(
    ComplexArray values_array,
    Int64Array indices_array,
    FloatArray weights_array,
    std::int64_t requested_threads
) {
    const py::buffer_info values_info = values_array.request();
    const py::buffer_info indices_info = indices_array.request();
    const py::buffer_info weights_info = weights_array.request();
    validate_resample4_shapes(values_info, indices_info, weights_info);

    const std::int64_t source_size = static_cast<std::int64_t>(values_info.shape[0]);
    const std::int64_t target_size = static_cast<std::int64_t>(indices_info.shape[0]);

    const auto* values = static_cast<const Complex128*>(values_info.ptr);
    const auto* indices = static_cast<const std::int64_t*>(indices_info.ptr);
    const auto* weights = static_cast<const double*>(weights_info.ptr);
    validate_resample4_indices(indices, target_size * 4, source_size);

    auto out = ComplexArray({target_size});
    py::buffer_info out_info = out.request();
    auto* out_ptr = static_cast<Complex128*>(out_info.ptr);

    unsigned int threads = 1;
    if (requested_threads > 0) {
        threads = static_cast<unsigned int>(requested_threads);
    } else {
        threads = choose_thread_count(target_size, 4);
    }

    {
        py::gil_scoped_release release;
        parallel_for(target_size, threads, [&](std::int64_t start, std::int64_t stop) {
            for (std::int64_t row = start; row < stop; ++row) {
                const std::int64_t* idx = indices + row * 4;
                const double* w = weights + row * 4;
                double sum_re = 0.0;
                double sum_im = 0.0;
                for (std::int64_t item = 0; item < 4; ++item) {
                    const Complex128 value = values[idx[item]];
                    sum_re += w[item] * value.real();
                    sum_im += w[item] * value.imag();
                }
                out_ptr[row] = Complex128{sum_re, sum_im};
            }
        });
    }

    return out;
}

ComplexArray resample4_scatter_adjoint(
    ComplexArray residual_array,
    Int64Array indices_array,
    FloatArray weights_array,
    std::int64_t source_size
) {
    const py::buffer_info residual_info = residual_array.request();
    const py::buffer_info indices_info = indices_array.request();
    const py::buffer_info weights_info = weights_array.request();
    if (residual_info.ndim != 1) {
        throw std::invalid_argument("residual must have shape (n_target,)");
    }
    if (indices_info.ndim != 2 || weights_info.ndim != 2 || indices_info.shape[1] != 4 ||
        weights_info.shape[1] != 4 || indices_info.shape[0] != residual_info.shape[0] ||
        weights_info.shape[0] != residual_info.shape[0]) {
        throw std::invalid_argument("inconsistent 4-point scatter-adjoint shapes");
    }
    if (source_size < 0) {
        throw std::invalid_argument("source_size must be non-negative");
    }

    const std::int64_t target_size = static_cast<std::int64_t>(residual_info.shape[0]);
    const auto* residual = static_cast<const Complex128*>(residual_info.ptr);
    const auto* indices = static_cast<const std::int64_t*>(indices_info.ptr);
    const auto* weights = static_cast<const double*>(weights_info.ptr);
    validate_resample4_indices(indices, target_size * 4, source_size);

    auto out = ComplexArray({source_size});
    py::buffer_info out_info = out.request();
    auto* out_ptr = static_cast<Complex128*>(out_info.ptr);
    std::fill(out_ptr, out_ptr + source_size, Complex128{0.0, 0.0});

    {
        py::gil_scoped_release release;
        for (std::int64_t row = 0; row < target_size; ++row) {
            const Complex128 res = residual[row];
            const std::int64_t* idx = indices + row * 4;
            const double* w = weights + row * 4;
            for (std::int64_t item = 0; item < 4; ++item) {
                out_ptr[idx[item]] += Complex128{w[item] * res.real(), w[item] * res.imag()};
            }
        }
    }

    return out;
}

ComplexArray axis_grid_forward_fold(
    ComplexArray coeff_h_array,
    FloatArray radial_array,
    ComplexArray axial_array,
    ComplexArray mode_phase_array,
    Int64Array h_slots_array,
    std::int64_t cap_phi,
    std::int64_t requested_threads
) {
    const py::buffer_info coeff_info = coeff_h_array.request();
    const py::buffer_info radial_info = radial_array.request();
    const py::buffer_info axial_info = axial_array.request();
    const py::buffer_info mode_phase_info = mode_phase_array.request();
    const py::buffer_info h_slots_info = h_slots_array.request();
    validate_axis_grid_forward_shapes(
        coeff_info,
        radial_info,
        axial_info,
        mode_phase_info,
        h_slots_info,
        cap_phi
    );

    const std::int64_t n_illum = static_cast<std::int64_t>(coeff_info.shape[0]);
    const std::int64_t nr = static_cast<std::int64_t>(coeff_info.shape[1]);
    const std::int64_t nz = static_cast<std::int64_t>(coeff_info.shape[2]);
    const std::int64_t nh = static_cast<std::int64_t>(coeff_info.shape[3]);
    const std::int64_t cap_radial = static_cast<std::int64_t>(radial_info.shape[1]);

    auto folded = ComplexArray({n_illum, cap_radial, cap_phi});
    py::buffer_info folded_info = folded.request();

    const auto* coeff = static_cast<const Complex128*>(coeff_info.ptr);
    const auto* radial = static_cast<const double*>(radial_info.ptr);
    const auto* axial = static_cast<const Complex128*>(axial_info.ptr);
    const auto* mode_phase = static_cast<const Complex128*>(mode_phase_info.ptr);
    const auto* h_slots = static_cast<const std::int64_t*>(h_slots_info.ptr);
    auto* folded_ptr = static_cast<Complex128*>(folded_info.ptr);
    std::fill(folded_ptr, folded_ptr + n_illum * cap_radial * cap_phi, Complex128{0.0, 0.0});

    const std::int64_t tasks = n_illum * cap_radial;
    unsigned int threads = 1;
    if (requested_threads > 0) {
        threads = static_cast<unsigned int>(requested_threads);
    } else {
        threads = choose_thread_count(tasks, nh * nr * nz);
    }

    {
        py::gil_scoped_release release;
        parallel_for(tasks, threads, [&](std::int64_t start, std::int64_t stop) {
            for (std::int64_t task = start; task < stop; ++task) {
                const std::int64_t illum = task / cap_radial;
                const std::int64_t u = task - illum * cap_radial;
                Complex128* folded_base = folded_ptr + (illum * cap_radial + u) * cap_phi;
                for (std::int64_t h = 0; h < nh; ++h) {
                    const std::int64_t slot = h_slots[h];
                    if (slot < 0 || slot >= cap_phi) {
                        throw std::invalid_argument("h_slots contains an out-of-range value");
                    }
                    double inner_re = 0.0;
                    double inner_im = 0.0;
                    const double* radial_row = radial + (h * cap_radial + u) * nr;
                    const Complex128* axial_row = axial + u * nz;
                    for (std::int64_t r = 0; r < nr; ++r) {
                        const double rad = radial_row[r];
                        const Complex128* coeff_base =
                            coeff + ((illum * nr + r) * nz * nh + h);
                        for (std::int64_t z = 0; z < nz; ++z) {
                            const Complex128 c = coeff_base[z * nh];
                            const Complex128 a = axial_row[z];
                            inner_re += rad * (c.real() * a.real() - c.imag() * a.imag());
                            inner_im += rad * (c.real() * a.imag() + c.imag() * a.real());
                        }
                    }
                    const Complex128 phase = mode_phase[h];
                    folded_base[slot] += Complex128{
                        phase.real() * inner_re - phase.imag() * inner_im,
                        phase.real() * inner_im + phase.imag() * inner_re,
                    };
                }
            }
        });
    }

    return folded;
}

ComplexArray phase_selected_dft(
    ComplexArray coeff_array,
    ComplexArray phase_array,
    ComplexArray twiddle_array,
    std::int64_t requested_threads
) {
    const py::buffer_info coeff_info = coeff_array.request();
    const py::buffer_info phase_info = phase_array.request();
    const py::buffer_info twiddle_info = twiddle_array.request();
    validate_phase_selected_dft_shapes(coeff_info, phase_info, twiddle_info);

    const std::int64_t nr = static_cast<std::int64_t>(coeff_info.shape[0]);
    const std::int64_t nz = static_cast<std::int64_t>(coeff_info.shape[1]);
    const std::int64_t n_beta = static_cast<std::int64_t>(coeff_info.shape[2]);
    const std::int64_t n_illum = static_cast<std::int64_t>(phase_info.shape[0]);
    const std::int64_t nh = static_cast<std::int64_t>(twiddle_info.shape[0]);

    auto out = ComplexArray({n_illum, nr, nz, nh});
    py::buffer_info out_info = out.request();

    const auto* coeff = static_cast<const Complex128*>(coeff_info.ptr);
    const auto* phase = static_cast<const Complex128*>(phase_info.ptr);
    const auto* twiddle = static_cast<const Complex128*>(twiddle_info.ptr);
    auto* out_ptr = static_cast<Complex128*>(out_info.ptr);

    const std::int64_t tasks = n_illum * nr * nz;
    unsigned int threads = 1;
    if (requested_threads > 0) {
        threads = static_cast<unsigned int>(requested_threads);
    } else {
        threads = choose_thread_count(tasks, nh * n_beta);
    }

    {
        py::gil_scoped_release release;
        parallel_for(tasks, threads, [&](std::int64_t start, std::int64_t stop) {
            for (std::int64_t task = start; task < stop; ++task) {
                const std::int64_t z = task % nz;
                const std::int64_t tmp = task / nz;
                const std::int64_t r = tmp % nr;
                const std::int64_t illum = tmp / nr;
                const Complex128* coeff_row = coeff + (r * nz + z) * n_beta;
                const Complex128* phase_row = phase + ((illum * nr + r) * nz + z) * n_beta;
                Complex128* out_row = out_ptr + ((illum * nr + r) * nz + z) * nh;
                for (std::int64_t h = 0; h < nh; ++h) {
                    double sum_re = 0.0;
                    double sum_im = 0.0;
                    const Complex128* twiddle_row = twiddle + h * n_beta;
                    for (std::int64_t b = 0; b < n_beta; ++b) {
                        const Complex128 pc = phase_row[b] * coeff_row[b];
                        const Complex128 tw = twiddle_row[b];
                        sum_re += pc.real() * tw.real() - pc.imag() * tw.imag();
                        sum_im += pc.real() * tw.imag() + pc.imag() * tw.real();
                    }
                    out_row[h] = Complex128{sum_re, sum_im};
                }
            }
        });
    }

    return out;
}

ComplexArray adjoint_contract_compact(
    ComplexArray residual_array,
    FloatArray radial_array,
    ComplexArray axial_array,
    ComplexArray angular_array,
    std::int64_t requested_threads
) {
    const py::buffer_info residual_info = residual_array.request();
    const py::buffer_info radial_info = radial_array.request();
    const py::buffer_info axial_info = axial_array.request();
    const py::buffer_info angular_info = angular_array.request();
    validate_adjoint_shapes(residual_info, radial_info, axial_info, angular_info);

    const std::int64_t nh = static_cast<std::int64_t>(radial_info.shape[0]);
    const std::int64_t nq = static_cast<std::int64_t>(radial_info.shape[1]);
    const std::int64_t nr = static_cast<std::int64_t>(radial_info.shape[2]);
    const std::int64_t nz = static_cast<std::int64_t>(axial_info.shape[1]);

    auto out = ComplexArray({nr, nz, nh});
    py::buffer_info out_info = out.request();

    const auto* residual = static_cast<const Complex128*>(residual_info.ptr);
    const auto* radial = static_cast<const double*>(radial_info.ptr);
    const auto* axial = static_cast<const Complex128*>(axial_info.ptr);
    const auto* angular = static_cast<const Complex128*>(angular_info.ptr);
    auto* out_ptr = static_cast<Complex128*>(out_info.ptr);

    const std::int64_t tasks = nh * nr;
    unsigned int threads = 1;
    if (requested_threads > 0) {
        threads = static_cast<unsigned int>(requested_threads);
    } else {
        threads = choose_thread_count(tasks, nq * nz);
    }

    {
        py::gil_scoped_release release;
        parallel_for(tasks, threads, [&](std::int64_t start, std::int64_t stop) {
            std::vector<double> accum_re(static_cast<std::size_t>(nz));
            std::vector<double> accum_im(static_cast<std::size_t>(nz));
            for (std::int64_t task = start; task < stop; ++task) {
                const std::int64_t h = task / nr;
                const std::int64_t r = task - h * nr;
                std::fill(accum_re.begin(), accum_re.end(), 0.0);
                std::fill(accum_im.begin(), accum_im.end(), 0.0);
                for (std::int64_t m = 0; m < nq; ++m) {
                    const double rad = radial[(h * nq + m) * nr + r];
                    const Complex128 res = residual[m];
                    const Complex128 ang = angular[h * nq + m];
                    const double tmp_re = res.real() * ang.real() + res.imag() * ang.imag();
                    const double tmp_im = res.imag() * ang.real() - res.real() * ang.imag();
                    const double scaled_re = rad * tmp_re;
                    const double scaled_im = rad * tmp_im;
                    const Complex128* axial_row = axial + m * nz;
                    for (std::int64_t z = 0; z < nz; ++z) {
                        const Complex128 ax = axial_row[z];
                        accum_re[static_cast<std::size_t>(z)] +=
                            scaled_re * ax.real() + scaled_im * ax.imag();
                        accum_im[static_cast<std::size_t>(z)] +=
                            scaled_im * ax.real() - scaled_re * ax.imag();
                    }
                }
                for (std::int64_t z = 0; z < nz; ++z) {
                    const std::size_t zi = static_cast<std::size_t>(z);
                    out_ptr[(r * nz + z) * nh + h] =
                        Complex128{accum_re[zi], accum_im[zi]};
                }
            }
        });
    }

    return out;
}

ComplexArray axis_grid_adjoint_unfold(
    ComplexArray residual_modes_array,
    FloatArray radial_array,
    ComplexArray axial_array,
    ComplexArray mode_phase_array,
    Int64Array h_slots_array,
    std::int64_t requested_threads
) {
    const py::buffer_info residual_info = residual_modes_array.request();
    const py::buffer_info radial_info = radial_array.request();
    const py::buffer_info axial_info = axial_array.request();
    const py::buffer_info mode_phase_info = mode_phase_array.request();
    const py::buffer_info h_slots_info = h_slots_array.request();
    validate_axis_grid_adjoint_shapes(
        residual_info,
        radial_info,
        axial_info,
        mode_phase_info,
        h_slots_info
    );

    const std::int64_t n_illum = static_cast<std::int64_t>(residual_info.shape[0]);
    const std::int64_t cap_radial = static_cast<std::int64_t>(residual_info.shape[1]);
    const std::int64_t cap_phi = static_cast<std::int64_t>(residual_info.shape[2]);
    const std::int64_t nh = static_cast<std::int64_t>(radial_info.shape[0]);
    const std::int64_t nr = static_cast<std::int64_t>(radial_info.shape[2]);
    const std::int64_t nz = static_cast<std::int64_t>(axial_info.shape[1]);

    auto compact = ComplexArray({n_illum, nr, nz, nh});
    py::buffer_info compact_info = compact.request();

    const auto* residual = static_cast<const Complex128*>(residual_info.ptr);
    const auto* radial = static_cast<const double*>(radial_info.ptr);
    const auto* axial = static_cast<const Complex128*>(axial_info.ptr);
    const auto* mode_phase = static_cast<const Complex128*>(mode_phase_info.ptr);
    const auto* h_slots = static_cast<const std::int64_t*>(h_slots_info.ptr);
    auto* compact_ptr = static_cast<Complex128*>(compact_info.ptr);

    const std::int64_t tasks = n_illum * nr * nh;
    unsigned int threads = 1;
    if (requested_threads > 0) {
        threads = static_cast<unsigned int>(requested_threads);
    } else {
        threads = choose_thread_count(tasks, cap_radial * nz);
    }

    {
        py::gil_scoped_release release;
        parallel_for(tasks, threads, [&](std::int64_t start, std::int64_t stop) {
            std::vector<double> accum_re(static_cast<std::size_t>(nz));
            std::vector<double> accum_im(static_cast<std::size_t>(nz));
            for (std::int64_t task = start; task < stop; ++task) {
                const std::int64_t tmp = task / nh;
                const std::int64_t h = task - tmp * nh;
                const std::int64_t illum = tmp / nr;
                const std::int64_t r = tmp - illum * nr;
                const std::int64_t slot = h_slots[h];
                if (slot < 0 || slot >= cap_phi) {
                    throw std::invalid_argument("h_slots contains an out-of-range value");
                }
                std::fill(accum_re.begin(), accum_re.end(), 0.0);
                std::fill(accum_im.begin(), accum_im.end(), 0.0);
                const Complex128 phase = std::conj(mode_phase[h]);
                for (std::int64_t u = 0; u < cap_radial; ++u) {
                    const Complex128 res = residual[(illum * cap_radial + u) * cap_phi + slot];
                    const double rad = radial[(h * cap_radial + u) * nr + r];
                    const double tmp_re = res.real() * phase.real() - res.imag() * phase.imag();
                    const double tmp_im = res.real() * phase.imag() + res.imag() * phase.real();
                    const double scaled_re = rad * tmp_re;
                    const double scaled_im = rad * tmp_im;
                    const Complex128* axial_row = axial + u * nz;
                    for (std::int64_t z = 0; z < nz; ++z) {
                        const Complex128 ax = axial_row[z];
                        accum_re[static_cast<std::size_t>(z)] +=
                            scaled_re * ax.real() + scaled_im * ax.imag();
                        accum_im[static_cast<std::size_t>(z)] +=
                            scaled_im * ax.real() - scaled_re * ax.imag();
                    }
                }
                for (std::int64_t z = 0; z < nz; ++z) {
                    const std::size_t zi = static_cast<std::size_t>(z);
                    compact_ptr[((illum * nr + r) * nz + z) * nh + h] =
                        Complex128{accum_re[zi], accum_im[zi]};
                }
            }
        });
    }

    return compact;
}

ComplexArray phase_selected_idft_adjoint(
    ComplexArray compact_array,
    ComplexArray phase_array,
    ComplexArray twiddle_array,
    std::int64_t requested_threads
) {
    const py::buffer_info compact_info = compact_array.request();
    const py::buffer_info phase_info = phase_array.request();
    const py::buffer_info twiddle_info = twiddle_array.request();
    validate_phase_selected_idft_shapes(compact_info, phase_info, twiddle_info);

    const std::int64_t n_illum = static_cast<std::int64_t>(compact_info.shape[0]);
    const std::int64_t nr = static_cast<std::int64_t>(compact_info.shape[1]);
    const std::int64_t nz = static_cast<std::int64_t>(compact_info.shape[2]);
    const std::int64_t nh = static_cast<std::int64_t>(compact_info.shape[3]);
    const std::int64_t n_beta = static_cast<std::int64_t>(phase_info.shape[3]);

    auto out = ComplexArray({nr, nz, n_beta});
    py::buffer_info out_info = out.request();

    const auto* compact = static_cast<const Complex128*>(compact_info.ptr);
    const auto* phase = static_cast<const Complex128*>(phase_info.ptr);
    const auto* twiddle = static_cast<const Complex128*>(twiddle_info.ptr);
    auto* out_ptr = static_cast<Complex128*>(out_info.ptr);

    const std::int64_t tasks = nr * nz * n_beta;
    unsigned int threads = 1;
    if (requested_threads > 0) {
        threads = static_cast<unsigned int>(requested_threads);
    } else {
        threads = choose_thread_count(tasks, n_illum * nh);
    }

    {
        py::gil_scoped_release release;
        parallel_for(tasks, threads, [&](std::int64_t start, std::int64_t stop) {
            for (std::int64_t task = start; task < stop; ++task) {
                const std::int64_t b = task % n_beta;
                const std::int64_t tmp = task / n_beta;
                const std::int64_t z = tmp % nz;
                const std::int64_t r = tmp / nz;
                double sum_re = 0.0;
                double sum_im = 0.0;
                for (std::int64_t illum = 0; illum < n_illum; ++illum) {
                    const Complex128 phase_conj =
                        std::conj(phase[((illum * nr + r) * nz + z) * n_beta + b]);
                    const Complex128* compact_row =
                        compact + ((illum * nr + r) * nz + z) * nh;
                    for (std::int64_t h = 0; h < nh; ++h) {
                        const Complex128 tw_conj = std::conj(twiddle[h * n_beta + b]);
                        const Complex128 ct = compact_row[h] * tw_conj;
                        sum_re += ct.real() * phase_conj.real() - ct.imag() * phase_conj.imag();
                        sum_im += ct.real() * phase_conj.imag() + ct.imag() * phase_conj.real();
                    }
                }
                out_ptr[(r * nz + z) * n_beta + b] = Complex128{sum_re, sum_im};
            }
        });
    }

    return out;
}

ComplexArray cone_axis_decompose_forward(
    ComplexArray coeff_h_full_array,
    ComplexArray transverse_coeff_array,
    ComplexArray psi_phase_array,
    ComplexArray axial_phase_array,
    Int64Array source_slots_array,
    std::int64_t requested_threads
) {
    const py::buffer_info coeff_info = coeff_h_full_array.request();
    const py::buffer_info transverse_info = transverse_coeff_array.request();
    const py::buffer_info psi_info = psi_phase_array.request();
    const py::buffer_info axial_info = axial_phase_array.request();
    const py::buffer_info slots_info = source_slots_array.request();
    validate_cone_axis_forward_shapes(
        coeff_info,
        transverse_info,
        psi_info,
        axial_info,
        slots_info
    );

    const std::int64_t nr = static_cast<std::int64_t>(coeff_info.shape[0]);
    const std::int64_t nz = static_cast<std::int64_t>(coeff_info.shape[1]);
    const std::int64_t n_beta = static_cast<std::int64_t>(coeff_info.shape[2]);
    const std::int64_t n_illum = static_cast<std::int64_t>(psi_info.shape[0]);
    const std::int64_t nl = static_cast<std::int64_t>(transverse_info.shape[0]);
    const std::int64_t nh = static_cast<std::int64_t>(slots_info.shape[0]);

    const auto* source_slots = static_cast<const std::int64_t*>(slots_info.ptr);
    validate_source_slots(source_slots, nh * nl, n_beta);

    auto out = ComplexArray({n_illum, nr, nz, nh});
    py::buffer_info out_info = out.request();

    const auto* coeff = static_cast<const Complex128*>(coeff_info.ptr);
    const auto* transverse = static_cast<const Complex128*>(transverse_info.ptr);
    const auto* psi = static_cast<const Complex128*>(psi_info.ptr);
    const auto* axial = static_cast<const Complex128*>(axial_info.ptr);
    auto* out_ptr = static_cast<Complex128*>(out_info.ptr);

    const std::int64_t tasks = n_illum * nr * nz * nh;
    unsigned int threads = 1;
    if (requested_threads > 0) {
        threads = static_cast<unsigned int>(requested_threads);
    } else {
        threads = choose_thread_count(tasks, nl);
    }

    {
        py::gil_scoped_release release;
        parallel_for(tasks, threads, [&](std::int64_t start, std::int64_t stop) {
            for (std::int64_t task = start; task < stop; ++task) {
                const std::int64_t h = task % nh;
                const std::int64_t tmp0 = task / nh;
                const std::int64_t z = tmp0 % nz;
                const std::int64_t tmp1 = tmp0 / nz;
                const std::int64_t r = tmp1 % nr;
                const std::int64_t illum = tmp1 / nr;

                Complex128 sum{0.0, 0.0};
                const Complex128* psi_row = psi + illum * nl;
                const Complex128* transverse_col = transverse + r;
                const std::int64_t* slot_row = source_slots + h * nl;
                const Complex128* coeff_row = coeff + (r * nz + z) * n_beta;
                for (std::int64_t l = 0; l < nl; ++l) {
                    const std::int64_t slot = slot_row[l];
                    const Complex128 scale = transverse_col[l * nr] * psi_row[l];
                    sum += coeff_row[slot] * scale;
                }
                out_ptr[((illum * nr + r) * nz + z) * nh + h] = sum * axial[z];
            }
        });
    }

    return out;
}

ComplexArray cone_axis_forward_fold(
    ComplexArray coeff_h_full_array,
    FloatArray radial_array,
    ComplexArray axial_array,
    ComplexArray mode_phase_array,
    Int64Array h_slots_array,
    ComplexArray transverse_coeff_array,
    ComplexArray psi_phase_array,
    ComplexArray axial_phase_array,
    Int64Array source_slots_array,
    std::int64_t cap_phi,
    std::int64_t requested_threads
) {
    const py::buffer_info coeff_info = coeff_h_full_array.request();
    const py::buffer_info radial_info = radial_array.request();
    const py::buffer_info axial_info = axial_array.request();
    const py::buffer_info mode_phase_info = mode_phase_array.request();
    const py::buffer_info h_slots_info = h_slots_array.request();
    const py::buffer_info transverse_info = transverse_coeff_array.request();
    const py::buffer_info psi_info = psi_phase_array.request();
    const py::buffer_info axial_phase_info = axial_phase_array.request();
    const py::buffer_info source_slots_info = source_slots_array.request();
    validate_cone_axis_forward_fold_shapes(
        coeff_info,
        radial_info,
        axial_info,
        mode_phase_info,
        h_slots_info,
        transverse_info,
        psi_info,
        axial_phase_info,
        source_slots_info,
        cap_phi
    );

    const std::int64_t nr = static_cast<std::int64_t>(coeff_info.shape[0]);
    const std::int64_t nz = static_cast<std::int64_t>(coeff_info.shape[1]);
    const std::int64_t n_beta = static_cast<std::int64_t>(coeff_info.shape[2]);
    const std::int64_t nh = static_cast<std::int64_t>(source_slots_info.shape[0]);
    const std::int64_t nl = static_cast<std::int64_t>(source_slots_info.shape[1]);
    const std::int64_t n_illum = static_cast<std::int64_t>(psi_info.shape[0]);
    const std::int64_t cap_radial = static_cast<std::int64_t>(radial_info.shape[1]);

    const auto* source_slots = static_cast<const std::int64_t*>(source_slots_info.ptr);
    const auto* h_slots = static_cast<const std::int64_t*>(h_slots_info.ptr);
    validate_source_slots(source_slots, nh * nl, n_beta);
    validate_source_slots(h_slots, nh, cap_phi);

    auto folded = ComplexArray({n_illum, cap_radial, cap_phi});
    py::buffer_info folded_info = folded.request();
    auto* folded_ptr = static_cast<Complex128*>(folded_info.ptr);
    std::fill(folded_ptr, folded_ptr + n_illum * cap_radial * cap_phi, Complex128{0.0, 0.0});

    const auto* coeff = static_cast<const Complex128*>(coeff_info.ptr);
    const auto* radial = static_cast<const double*>(radial_info.ptr);
    const auto* axial = static_cast<const Complex128*>(axial_info.ptr);
    const auto* mode_phase = static_cast<const Complex128*>(mode_phase_info.ptr);
    const auto* transverse = static_cast<const Complex128*>(transverse_info.ptr);
    const auto* psi = static_cast<const Complex128*>(psi_info.ptr);
    const auto* axial_phase = static_cast<const Complex128*>(axial_phase_info.ptr);

    const std::int64_t tasks = n_illum * nh * nr * nz;
    unsigned int threads = 1;
    if (requested_threads > 0) {
        threads = std::min<unsigned int>(
            static_cast<unsigned int>(requested_threads),
            static_cast<unsigned int>(std::max<std::int64_t>(tasks, 1))
        );
    } else {
        threads = choose_thread_count(tasks, nl + cap_radial);
    }

    const std::int64_t folded_size = n_illum * cap_radial * cap_phi;
    const std::int64_t compact_size = n_illum * cap_radial * nh;
    std::vector<std::vector<Complex128>> local_buffers;
    local_buffers.reserve(threads);
    for (unsigned int thread = 0; thread < threads; ++thread) {
        local_buffers.emplace_back(
            static_cast<std::size_t>(compact_size),
            Complex128{0.0, 0.0}
        );
    }

    {
        py::gil_scoped_release release;
        const std::int64_t block = (tasks + static_cast<std::int64_t>(threads) - 1) /
                                   static_cast<std::int64_t>(threads);
        std::vector<std::thread> workers;
        workers.reserve(threads);
        for (unsigned int thread = 0; thread < threads; ++thread) {
            const std::int64_t start = static_cast<std::int64_t>(thread) * block;
            const std::int64_t stop = std::min(tasks, start + block);
            if (start >= stop) {
                break;
            }
            workers.emplace_back([=, &local_buffers]() {
                auto& local = local_buffers[thread];
                for (std::int64_t task = start; task < stop; ++task) {
                    const std::int64_t z = task % nz;
                    const std::int64_t tmp0 = task / nz;
                    const std::int64_t r = tmp0 % nr;
                    const std::int64_t tmp1 = tmp0 / nr;
                    const std::int64_t h = tmp1 % nh;
                    const std::int64_t illum = tmp1 / nh;

                    Complex128 source_sum{0.0, 0.0};
                    const Complex128* coeff_row = coeff + (r * nz + z) * n_beta;
                    const Complex128* psi_row = psi + illum * nl;
                    const std::int64_t* slot_row = source_slots + h * nl;
                    for (std::int64_t l = 0; l < nl; ++l) {
                        source_sum += coeff_row[slot_row[l]] *
                                      transverse[l * nr + r] *
                                      psi_row[l];
                    }

                    const Complex128 source_value = source_sum * axial_phase[z] * mode_phase[h];
                    for (std::int64_t u = 0; u < cap_radial; ++u) {
                        const double rad = radial[(h * cap_radial + u) * nr + r];
                        const Complex128 ax = axial[u * nz + z];
                        local[(illum * cap_radial + u) * nh + h] +=
                            rad * source_value * ax;
                    }
                }
            });
        }
        for (auto& worker : workers) {
            worker.join();
        }

        for (const auto& local : local_buffers) {
            for (std::int64_t illum = 0; illum < n_illum; ++illum) {
                for (std::int64_t u = 0; u < cap_radial; ++u) {
                    const Complex128* local_row =
                        local.data() + (illum * cap_radial + u) * nh;
                    Complex128* folded_row =
                        folded_ptr + (illum * cap_radial + u) * cap_phi;
                    for (std::int64_t h = 0; h < nh; ++h) {
                        folded_row[h_slots[h]] += local_row[h];
                    }
                }
            }
        }
    }

    return folded;
}

ComplexArray cone_axis_forward_fold_pruned(
    ComplexArray coeff_h_full_array,
    FloatArray radial_array,
    ComplexArray axial_array,
    ComplexArray mode_phase_array,
    Int64Array h_slots_array,
    ComplexArray transverse_coeff_array,
    ComplexArray psi_phase_array,
    ComplexArray axial_phase_array,
    Int64Array source_slots_array,
    Int64Array l_offsets_array,
    Int64Array l_indices_array,
    std::int64_t cap_phi,
    std::int64_t requested_threads
) {
    const py::buffer_info coeff_info = coeff_h_full_array.request();
    const py::buffer_info radial_info = radial_array.request();
    const py::buffer_info axial_info = axial_array.request();
    const py::buffer_info mode_phase_info = mode_phase_array.request();
    const py::buffer_info h_slots_info = h_slots_array.request();
    const py::buffer_info transverse_info = transverse_coeff_array.request();
    const py::buffer_info psi_info = psi_phase_array.request();
    const py::buffer_info axial_phase_info = axial_phase_array.request();
    const py::buffer_info source_slots_info = source_slots_array.request();
    const py::buffer_info l_offsets_info = l_offsets_array.request();
    const py::buffer_info l_indices_info = l_indices_array.request();
    validate_cone_axis_forward_fold_shapes(
        coeff_info,
        radial_info,
        axial_info,
        mode_phase_info,
        h_slots_info,
        transverse_info,
        psi_info,
        axial_phase_info,
        source_slots_info,
        cap_phi
    );

    const std::int64_t nr = static_cast<std::int64_t>(coeff_info.shape[0]);
    const std::int64_t nz = static_cast<std::int64_t>(coeff_info.shape[1]);
    const std::int64_t n_beta = static_cast<std::int64_t>(coeff_info.shape[2]);
    const std::int64_t nh = static_cast<std::int64_t>(source_slots_info.shape[0]);
    const std::int64_t nl = static_cast<std::int64_t>(source_slots_info.shape[1]);
    const std::int64_t n_illum = static_cast<std::int64_t>(psi_info.shape[0]);
    const std::int64_t cap_radial = static_cast<std::int64_t>(radial_info.shape[1]);

    validate_cone_axis_l_pruning_shapes(l_offsets_info, l_indices_info, nr, nl);
    const std::int64_t active_l_total = static_cast<std::int64_t>(l_indices_info.shape[0]);

    const auto* source_slots = static_cast<const std::int64_t*>(source_slots_info.ptr);
    const auto* h_slots = static_cast<const std::int64_t*>(h_slots_info.ptr);
    validate_source_slots(source_slots, nh * nl, n_beta);
    validate_source_slots(h_slots, nh, cap_phi);

    auto folded = ComplexArray({n_illum, cap_radial, cap_phi});
    py::buffer_info folded_info = folded.request();
    auto* folded_ptr = static_cast<Complex128*>(folded_info.ptr);
    std::fill(folded_ptr, folded_ptr + n_illum * cap_radial * cap_phi, Complex128{0.0, 0.0});

    const auto* coeff = static_cast<const Complex128*>(coeff_info.ptr);
    const auto* radial = static_cast<const double*>(radial_info.ptr);
    const auto* axial = static_cast<const Complex128*>(axial_info.ptr);
    const auto* mode_phase = static_cast<const Complex128*>(mode_phase_info.ptr);
    const auto* transverse = static_cast<const Complex128*>(transverse_info.ptr);
    const auto* psi = static_cast<const Complex128*>(psi_info.ptr);
    const auto* axial_phase = static_cast<const Complex128*>(axial_phase_info.ptr);
    const auto* l_offsets = static_cast<const std::int64_t*>(l_offsets_info.ptr);
    const auto* l_indices = static_cast<const std::int64_t*>(l_indices_info.ptr);

    std::vector<Complex128> source_weight(
        static_cast<std::size_t>(n_illum * nr * nl),
        Complex128{0.0, 0.0}
    );
    for (std::int64_t illum = 0; illum < n_illum; ++illum) {
        for (std::int64_t r = 0; r < nr; ++r) {
            Complex128* weight_row =
                source_weight.data() + (illum * nr + r) * nl;
            const Complex128* psi_row = psi + illum * nl;
            for (std::int64_t l = 0; l < nl; ++l) {
                weight_row[l] = transverse[l * nr + r] * psi_row[l];
            }
        }
    }
    std::vector<Complex128> source_phase(static_cast<std::size_t>(nh * nz));
    for (std::int64_t h = 0; h < nh; ++h) {
        for (std::int64_t z = 0; z < nz; ++z) {
            source_phase[static_cast<std::size_t>(h * nz + z)] =
                mode_phase[h] * axial_phase[z];
        }
    }
    const std::int64_t active_l_avg = std::max<std::int64_t>(1, active_l_total / std::max<std::int64_t>(nr, 1));
    const std::int64_t tasks = n_illum * nh * nr * nz;
    unsigned int threads = 1;
    if (requested_threads > 0) {
        threads = std::min<unsigned int>(
            static_cast<unsigned int>(requested_threads),
            static_cast<unsigned int>(std::max<std::int64_t>(tasks, 1))
        );
    } else {
        threads = choose_thread_count(tasks, active_l_avg + cap_radial);
    }

    const std::int64_t folded_size = n_illum * cap_radial * cap_phi;
    const std::int64_t compact_size = n_illum * cap_radial * nh;
    std::vector<std::vector<Complex128>> local_buffers;
    local_buffers.reserve(threads);
    for (unsigned int thread = 0; thread < threads; ++thread) {
        local_buffers.emplace_back(
            static_cast<std::size_t>(compact_size),
            Complex128{0.0, 0.0}
        );
    }

    {
        py::gil_scoped_release release;
        const std::int64_t block = (tasks + static_cast<std::int64_t>(threads) - 1) /
                                   static_cast<std::int64_t>(threads);
        std::vector<std::thread> workers;
        workers.reserve(threads);
        for (unsigned int thread = 0; thread < threads; ++thread) {
            const std::int64_t start = static_cast<std::int64_t>(thread) * block;
            const std::int64_t stop = std::min(tasks, start + block);
            if (start >= stop) {
                break;
            }
            workers.emplace_back([=, &local_buffers]() {
                auto& local = local_buffers[thread];
                for (std::int64_t task = start; task < stop; ++task) {
                    const std::int64_t z = task % nz;
                    const std::int64_t tmp0 = task / nz;
                    const std::int64_t r = tmp0 % nr;
                    const std::int64_t tmp1 = tmp0 / nr;
                    const std::int64_t h = tmp1 % nh;
                    const std::int64_t illum = tmp1 / nh;

                    Complex128 source_sum{0.0, 0.0};
                    const Complex128* coeff_row = coeff + (r * nz + z) * n_beta;
                    const Complex128* weight_row =
                        source_weight.data() + (illum * nr + r) * nl;
                    const std::int64_t* slot_row = source_slots + h * nl;
                    const std::int64_t l_start = l_offsets[r];
                    const std::int64_t l_stop = l_offsets[r + 1];
                    for (std::int64_t cursor = l_start; cursor < l_stop; ++cursor) {
                        const std::int64_t l = l_indices[cursor];
                        source_sum += coeff_row[slot_row[l]] *
                                      weight_row[l];
                    }

                    const Complex128 source_value =
                        source_sum * source_phase[static_cast<std::size_t>(h * nz + z)];
                    for (std::int64_t u = 0; u < cap_radial; ++u) {
                        const double rad = radial[(h * cap_radial + u) * nr + r];
                        const Complex128 ax = axial[u * nz + z];
                        local[(illum * cap_radial + u) * nh + h] +=
                            rad * source_value * ax;
                    }
                }
            });
        }
        for (auto& worker : workers) {
            worker.join();
        }

        for (const auto& local : local_buffers) {
            for (std::int64_t illum = 0; illum < n_illum; ++illum) {
                for (std::int64_t u = 0; u < cap_radial; ++u) {
                    const Complex128* local_row =
                        local.data() + (illum * cap_radial + u) * nh;
                    Complex128* folded_row =
                        folded_ptr + (illum * cap_radial + u) * cap_phi;
                    for (std::int64_t h = 0; h < nh; ++h) {
                        folded_row[h_slots[h]] += local_row[h];
                    }
                }
            }
        }
    }

    return folded;
}

ComplexArray cone_axis_forward_fold_pruned_partitioned(
    ComplexArray coeff_h_full_array,
    FloatArray radial_array,
    ComplexArray axial_array,
    ComplexArray mode_phase_array,
    Int64Array h_slots_array,
    ComplexArray transverse_coeff_array,
    ComplexArray psi_phase_array,
    ComplexArray axial_phase_array,
    Int64Array source_slots_array,
    Int64Array l_offsets_array,
    Int64Array l_indices_array,
    std::int64_t cap_phi,
    std::int64_t requested_threads
) {
    const py::buffer_info coeff_info = coeff_h_full_array.request();
    const py::buffer_info radial_info = radial_array.request();
    const py::buffer_info axial_info = axial_array.request();
    const py::buffer_info mode_phase_info = mode_phase_array.request();
    const py::buffer_info h_slots_info = h_slots_array.request();
    const py::buffer_info transverse_info = transverse_coeff_array.request();
    const py::buffer_info psi_info = psi_phase_array.request();
    const py::buffer_info axial_phase_info = axial_phase_array.request();
    const py::buffer_info source_slots_info = source_slots_array.request();
    const py::buffer_info l_offsets_info = l_offsets_array.request();
    const py::buffer_info l_indices_info = l_indices_array.request();
    validate_cone_axis_forward_fold_shapes(
        coeff_info,
        radial_info,
        axial_info,
        mode_phase_info,
        h_slots_info,
        transverse_info,
        psi_info,
        axial_phase_info,
        source_slots_info,
        cap_phi
    );

    const std::int64_t nr = static_cast<std::int64_t>(coeff_info.shape[0]);
    const std::int64_t nz = static_cast<std::int64_t>(coeff_info.shape[1]);
    const std::int64_t n_beta = static_cast<std::int64_t>(coeff_info.shape[2]);
    const std::int64_t nh = static_cast<std::int64_t>(source_slots_info.shape[0]);
    const std::int64_t nl = static_cast<std::int64_t>(source_slots_info.shape[1]);
    const std::int64_t n_illum = static_cast<std::int64_t>(psi_info.shape[0]);
    const std::int64_t cap_radial = static_cast<std::int64_t>(radial_info.shape[1]);

    validate_cone_axis_l_pruning_shapes(l_offsets_info, l_indices_info, nr, nl);
    const std::int64_t active_l_total = static_cast<std::int64_t>(l_indices_info.shape[0]);

    const auto* source_slots = static_cast<const std::int64_t*>(source_slots_info.ptr);
    const auto* h_slots = static_cast<const std::int64_t*>(h_slots_info.ptr);
    validate_source_slots(source_slots, nh * nl, n_beta);
    validate_source_slots(h_slots, nh, cap_phi);

    std::vector<unsigned char> slot_seen(static_cast<std::size_t>(cap_phi), 0);
    for (std::int64_t h = 0; h < nh; ++h) {
        const std::int64_t slot = h_slots[h];
        if (slot_seen[static_cast<std::size_t>(slot)] != 0) {
            throw std::invalid_argument(
                "partitioned cone-axis forward requires unique h_slots"
            );
        }
        slot_seen[static_cast<std::size_t>(slot)] = 1;
    }

    auto folded = ComplexArray({n_illum, cap_radial, cap_phi});
    py::buffer_info folded_info = folded.request();
    auto* folded_ptr = static_cast<Complex128*>(folded_info.ptr);
    std::fill(folded_ptr, folded_ptr + n_illum * cap_radial * cap_phi, Complex128{0.0, 0.0});

    const auto* coeff = static_cast<const Complex128*>(coeff_info.ptr);
    const auto* radial = static_cast<const double*>(radial_info.ptr);
    const auto* axial = static_cast<const Complex128*>(axial_info.ptr);
    const auto* mode_phase = static_cast<const Complex128*>(mode_phase_info.ptr);
    const auto* transverse = static_cast<const Complex128*>(transverse_info.ptr);
    const auto* psi = static_cast<const Complex128*>(psi_info.ptr);
    const auto* axial_phase = static_cast<const Complex128*>(axial_phase_info.ptr);
    const auto* l_offsets = static_cast<const std::int64_t*>(l_offsets_info.ptr);
    const auto* l_indices = static_cast<const std::int64_t*>(l_indices_info.ptr);

    std::vector<Complex128> source_weight(
        static_cast<std::size_t>(n_illum * nr * nl),
        Complex128{0.0, 0.0}
    );
    for (std::int64_t illum = 0; illum < n_illum; ++illum) {
        for (std::int64_t r = 0; r < nr; ++r) {
            Complex128* weight_row =
                source_weight.data() + (illum * nr + r) * nl;
            const Complex128* psi_row = psi + illum * nl;
            for (std::int64_t l = 0; l < nl; ++l) {
                weight_row[l] = transverse[l * nr + r] * psi_row[l];
            }
        }
    }
    std::vector<Complex128> source_phase(static_cast<std::size_t>(nh * nz));
    for (std::int64_t h = 0; h < nh; ++h) {
        for (std::int64_t z = 0; z < nz; ++z) {
            source_phase[static_cast<std::size_t>(h * nz + z)] =
                mode_phase[h] * axial_phase[z];
        }
    }

    const std::int64_t source_tasks = n_illum * nh * nr * nz;
    const std::int64_t output_tasks = n_illum * nh * cap_radial;
    const std::int64_t active_l_avg = std::max<std::int64_t>(
        1,
        active_l_total / std::max<std::int64_t>(nr, 1)
    );
    unsigned int threads = 1;
    if (requested_threads > 0) {
        threads = std::min<unsigned int>(
            static_cast<unsigned int>(requested_threads),
            static_cast<unsigned int>(std::max<std::int64_t>(
                std::max(source_tasks, output_tasks),
                1
            ))
        );
    } else {
        threads = choose_thread_count(
            std::max(source_tasks, output_tasks),
            active_l_avg + nr * nz
        );
    }

    std::vector<Complex128> source_values(
        static_cast<std::size_t>(source_tasks),
        Complex128{0.0, 0.0}
    );

    {
        py::gil_scoped_release release;
        parallel_for(source_tasks, threads, [&](std::int64_t start, std::int64_t stop) {
            for (std::int64_t task = start; task < stop; ++task) {
                const std::int64_t z = task % nz;
                const std::int64_t tmp0 = task / nz;
                const std::int64_t r = tmp0 % nr;
                const std::int64_t tmp1 = tmp0 / nr;
                const std::int64_t h = tmp1 % nh;
                const std::int64_t illum = tmp1 / nh;

                Complex128 source_sum{0.0, 0.0};
                const Complex128* coeff_row = coeff + (r * nz + z) * n_beta;
                const Complex128* weight_row =
                    source_weight.data() + (illum * nr + r) * nl;
                const std::int64_t* slot_row = source_slots + h * nl;
                const std::int64_t l_start = l_offsets[r];
                const std::int64_t l_stop = l_offsets[r + 1];
                for (std::int64_t cursor = l_start; cursor < l_stop; ++cursor) {
                    const std::int64_t l = l_indices[cursor];
                    source_sum += coeff_row[slot_row[l]] * weight_row[l];
                }

                source_values[static_cast<std::size_t>(task)] =
                    source_sum * source_phase[static_cast<std::size_t>(h * nz + z)];
            }
        });

        parallel_for(output_tasks, threads, [&](std::int64_t start, std::int64_t stop) {
            for (std::int64_t task = start; task < stop; ++task) {
                const std::int64_t u = task % cap_radial;
                const std::int64_t tmp0 = task / cap_radial;
                const std::int64_t h = tmp0 % nh;
                const std::int64_t illum = tmp0 / nh;

                double out_re = 0.0;
                double out_im = 0.0;
                for (std::int64_t r = 0; r < nr; ++r) {
                    const double rad = radial[(h * cap_radial + u) * nr + r];
                    const Complex128* source_row =
                        source_values.data() + (((illum * nh + h) * nr + r) * nz);
                    const Complex128* axial_row = axial + u * nz;
                    for (std::int64_t z = 0; z < nz; ++z) {
                        const Complex128 sv = source_row[z];
                        const Complex128 ax = axial_row[z];
                        out_re += rad * (sv.real() * ax.real() - sv.imag() * ax.imag());
                        out_im += rad * (sv.real() * ax.imag() + sv.imag() * ax.real());
                    }
                }
                folded_ptr[(illum * cap_radial + u) * cap_phi + h_slots[h]] =
                    Complex128{out_re, out_im};
            }
        });
    }

    return folded;
}

ComplexArray same_direction_forward_fold(
    ComplexArray coeff_h_full_array,
    FloatArray radial_array,
    ComplexArray axial_array,
    ComplexArray mode_phase_array,
    Int64Array h_slots_array,
    ComplexArray transverse_by_mag_array,
    ComplexArray axial_phase_by_mag_array,
    Int64Array source_slots_array,
    std::int64_t cap_phi,
    std::int64_t requested_threads
) {
    const py::buffer_info coeff_info = coeff_h_full_array.request();
    const py::buffer_info radial_info = radial_array.request();
    const py::buffer_info axial_info = axial_array.request();
    const py::buffer_info mode_phase_info = mode_phase_array.request();
    const py::buffer_info h_slots_info = h_slots_array.request();
    const py::buffer_info transverse_info = transverse_by_mag_array.request();
    const py::buffer_info axial_phase_info = axial_phase_by_mag_array.request();
    const py::buffer_info source_slots_info = source_slots_array.request();
    validate_same_direction_forward_fold_shapes(
        coeff_info,
        radial_info,
        axial_info,
        mode_phase_info,
        h_slots_info,
        transverse_info,
        axial_phase_info,
        source_slots_info,
        cap_phi
    );

    const std::int64_t nr = static_cast<std::int64_t>(coeff_info.shape[0]);
    const std::int64_t nz = static_cast<std::int64_t>(coeff_info.shape[1]);
    const std::int64_t n_beta = static_cast<std::int64_t>(coeff_info.shape[2]);
    const std::int64_t n_mag = static_cast<std::int64_t>(transverse_info.shape[0]);
    const std::int64_t nl = static_cast<std::int64_t>(transverse_info.shape[1]);
    const std::int64_t nh = static_cast<std::int64_t>(source_slots_info.shape[0]);
    const std::int64_t cap_radial = static_cast<std::int64_t>(radial_info.shape[1]);

    const auto* source_slots = static_cast<const std::int64_t*>(source_slots_info.ptr);
    const auto* h_slots = static_cast<const std::int64_t*>(h_slots_info.ptr);
    validate_source_slots(source_slots, nh * nl, n_beta);
    validate_source_slots(h_slots, nh, cap_phi);

    auto folded = ComplexArray({n_mag, cap_radial, cap_phi});
    py::buffer_info folded_info = folded.request();
    auto* folded_ptr = static_cast<Complex128*>(folded_info.ptr);
    std::fill(folded_ptr, folded_ptr + n_mag * cap_radial * cap_phi, Complex128{0.0, 0.0});

    const auto* coeff = static_cast<const Complex128*>(coeff_info.ptr);
    const auto* radial = static_cast<const double*>(radial_info.ptr);
    const auto* axial = static_cast<const Complex128*>(axial_info.ptr);
    const auto* mode_phase = static_cast<const Complex128*>(mode_phase_info.ptr);
    const auto* transverse = static_cast<const Complex128*>(transverse_info.ptr);
    const auto* axial_phase = static_cast<const Complex128*>(axial_phase_info.ptr);

    const std::int64_t tasks = nh * nr * nz;
    unsigned int threads = 1;
    if (requested_threads > 0) {
        threads = std::min<unsigned int>(
            static_cast<unsigned int>(requested_threads),
            static_cast<unsigned int>(std::max<std::int64_t>(tasks, 1))
        );
    } else {
        threads = choose_thread_count(tasks, n_mag * (nl + cap_radial));
    }

    const std::int64_t folded_size = n_mag * cap_radial * cap_phi;
    std::vector<std::vector<Complex128>> local_buffers;
    local_buffers.reserve(threads);
    for (unsigned int thread = 0; thread < threads; ++thread) {
        local_buffers.emplace_back(
            static_cast<std::size_t>(folded_size),
            Complex128{0.0, 0.0}
        );
    }

    {
        py::gil_scoped_release release;
        const std::int64_t block = (tasks + static_cast<std::int64_t>(threads) - 1) /
                                   static_cast<std::int64_t>(threads);
        std::vector<std::thread> workers;
        workers.reserve(threads);
        for (unsigned int thread = 0; thread < threads; ++thread) {
            const std::int64_t start = static_cast<std::int64_t>(thread) * block;
            const std::int64_t stop = std::min(tasks, start + block);
            if (start >= stop) {
                break;
            }
            workers.emplace_back([=, &local_buffers]() {
                auto& local = local_buffers[thread];
                std::vector<Complex128> source_sum(static_cast<std::size_t>(n_mag));
                for (std::int64_t task = start; task < stop; ++task) {
                    const std::int64_t z = task % nz;
                    const std::int64_t tmp0 = task / nz;
                    const std::int64_t r = tmp0 % nr;
                    const std::int64_t h = tmp0 / nr;

                    std::fill(source_sum.begin(), source_sum.end(), Complex128{0.0, 0.0});
                    const Complex128* coeff_row = coeff + (r * nz + z) * n_beta;
                    const std::int64_t* slot_row = source_slots + h * nl;
                    for (std::int64_t l = 0; l < nl; ++l) {
                        const Complex128 c = coeff_row[slot_row[l]];
                        const std::int64_t lr = l * nr + r;
                        for (std::int64_t mag = 0; mag < n_mag; ++mag) {
                            source_sum[static_cast<std::size_t>(mag)] +=
                                c * transverse[(mag * nl * nr) + lr];
                        }
                    }

                    const Complex128 mode = mode_phase[h];
                    const std::int64_t phi_slot = h_slots[h];
                    for (std::int64_t mag = 0; mag < n_mag; ++mag) {
                        const Complex128 source_value =
                            source_sum[static_cast<std::size_t>(mag)] *
                            axial_phase[mag * nz + z] *
                            mode;
                        for (std::int64_t u = 0; u < cap_radial; ++u) {
                            const double rad = radial[(h * cap_radial + u) * nr + r];
                            const Complex128 ax = axial[u * nz + z];
                            local[(mag * cap_radial + u) * cap_phi + phi_slot] +=
                                rad * source_value * ax;
                        }
                    }
                }
            });
        }
        for (auto& worker : workers) {
            worker.join();
        }

        for (const auto& local : local_buffers) {
            for (std::int64_t index = 0; index < folded_size; ++index) {
                folded_ptr[index] += local[static_cast<std::size_t>(index)];
            }
        }
    }

    return folded;
}

ComplexArray svd_rank_forward_fold(
    ComplexArray coeff_h_full_array,
    FloatArray radial_array,
    ComplexArray axial_array,
    ComplexArray mode_phase_array,
    Int64Array h_slots_array,
    ComplexArray weights_array,
    Int64Array source_slots_array,
    std::int64_t cap_phi,
    std::int64_t requested_threads
) {
    const py::buffer_info coeff_info = coeff_h_full_array.request();
    const py::buffer_info radial_info = radial_array.request();
    const py::buffer_info axial_info = axial_array.request();
    const py::buffer_info mode_phase_info = mode_phase_array.request();
    const py::buffer_info h_slots_info = h_slots_array.request();
    const py::buffer_info weights_info = weights_array.request();
    const py::buffer_info source_slots_info = source_slots_array.request();
    validate_svd_rank_forward_fold_shapes(
        coeff_info,
        radial_info,
        axial_info,
        mode_phase_info,
        h_slots_info,
        weights_info,
        source_slots_info,
        cap_phi
    );

    const std::int64_t nr = static_cast<std::int64_t>(coeff_info.shape[0]);
    const std::int64_t nz = static_cast<std::int64_t>(coeff_info.shape[1]);
    const std::int64_t n_beta = static_cast<std::int64_t>(coeff_info.shape[2]);
    const std::int64_t rank = static_cast<std::int64_t>(weights_info.shape[0]);
    const std::int64_t nl = static_cast<std::int64_t>(weights_info.shape[1]);
    const std::int64_t nh = static_cast<std::int64_t>(source_slots_info.shape[0]);
    const std::int64_t cap_radial = static_cast<std::int64_t>(radial_info.shape[1]);

    const auto* source_slots = static_cast<const std::int64_t*>(source_slots_info.ptr);
    const auto* h_slots = static_cast<const std::int64_t*>(h_slots_info.ptr);
    validate_source_slots(source_slots, nh * nl, n_beta);
    validate_source_slots(h_slots, nh, cap_phi);

    auto folded = ComplexArray({rank, cap_radial, cap_phi});
    py::buffer_info folded_info = folded.request();
    auto* folded_ptr = static_cast<Complex128*>(folded_info.ptr);
    std::fill(folded_ptr, folded_ptr + rank * cap_radial * cap_phi, Complex128{0.0, 0.0});

    const auto* coeff = static_cast<const Complex128*>(coeff_info.ptr);
    const auto* radial = static_cast<const double*>(radial_info.ptr);
    const auto* axial = static_cast<const Complex128*>(axial_info.ptr);
    const auto* mode_phase = static_cast<const Complex128*>(mode_phase_info.ptr);
    const auto* weights = static_cast<const Complex128*>(weights_info.ptr);

    const std::int64_t tasks = nh * nr * nz;
    unsigned int threads = 1;
    if (requested_threads > 0) {
        threads = std::min<unsigned int>(
            static_cast<unsigned int>(requested_threads),
            static_cast<unsigned int>(std::max<std::int64_t>(tasks, 1))
        );
    } else {
        threads = choose_thread_count_capped(tasks, rank * (nl + cap_radial), 8);
    }

    const std::int64_t folded_size = rank * cap_radial * cap_phi;
    std::vector<std::vector<Complex128>> local_buffers;
    local_buffers.reserve(threads);
    for (unsigned int thread = 0; thread < threads; ++thread) {
        local_buffers.emplace_back(
            static_cast<std::size_t>(folded_size),
            Complex128{0.0, 0.0}
        );
    }

    {
        py::gil_scoped_release release;
        const std::int64_t block = (tasks + static_cast<std::int64_t>(threads) - 1) /
                                   static_cast<std::int64_t>(threads);
        std::vector<std::thread> workers;
        workers.reserve(threads);
        for (unsigned int thread = 0; thread < threads; ++thread) {
            const std::int64_t start = static_cast<std::int64_t>(thread) * block;
            const std::int64_t stop = std::min(tasks, start + block);
            if (start >= stop) {
                break;
            }
            workers.emplace_back([=, &local_buffers]() {
                auto& local = local_buffers[thread];
                std::vector<Complex128> source_sum(static_cast<std::size_t>(rank));
                for (std::int64_t task = start; task < stop; ++task) {
                    const std::int64_t z = task % nz;
                    const std::int64_t tmp0 = task / nz;
                    const std::int64_t r = tmp0 % nr;
                    const std::int64_t h = tmp0 / nr;

                    std::fill(source_sum.begin(), source_sum.end(), Complex128{0.0, 0.0});
                    const Complex128* coeff_row = coeff + (r * nz + z) * n_beta;
                    const std::int64_t* slot_row = source_slots + h * nl;
                    for (std::int64_t l = 0; l < nl; ++l) {
                        const Complex128 c = coeff_row[slot_row[l]];
                        for (std::int64_t s = 0; s < rank; ++s) {
                            source_sum[static_cast<std::size_t>(s)] +=
                                c * weights[((s * nl + l) * nr + r) * nz + z];
                        }
                    }

                    const Complex128 mode = mode_phase[h];
                    const std::int64_t phi_slot = h_slots[h];
                    for (std::int64_t s = 0; s < rank; ++s) {
                        const Complex128 source_value =
                            source_sum[static_cast<std::size_t>(s)] * mode;
                        for (std::int64_t u = 0; u < cap_radial; ++u) {
                            const double rad = radial[(h * cap_radial + u) * nr + r];
                            const Complex128 ax = axial[u * nz + z];
                            local[(s * cap_radial + u) * cap_phi + phi_slot] +=
                                rad * source_value * ax;
                        }
                    }
                }
            });
        }
        for (auto& worker : workers) {
            worker.join();
        }

        for (const auto& local : local_buffers) {
            for (std::int64_t index = 0; index < folded_size; ++index) {
                folded_ptr[index] += local[static_cast<std::size_t>(index)];
            }
        }
    }

    return folded;
}

ComplexArray cone_axis_decompose_adjoint(
    ComplexArray compact_array,
    ComplexArray transverse_coeff_array,
    ComplexArray psi_phase_array,
    ComplexArray axial_phase_array,
    Int64Array source_slots_array,
    std::int64_t n_beta,
    std::int64_t requested_threads
) {
    const py::buffer_info compact_info = compact_array.request();
    const py::buffer_info transverse_info = transverse_coeff_array.request();
    const py::buffer_info psi_info = psi_phase_array.request();
    const py::buffer_info axial_info = axial_phase_array.request();
    const py::buffer_info slots_info = source_slots_array.request();
    validate_cone_axis_adjoint_shapes(
        compact_info,
        transverse_info,
        psi_info,
        axial_info,
        slots_info,
        n_beta
    );

    const std::int64_t n_illum = static_cast<std::int64_t>(compact_info.shape[0]);
    const std::int64_t nr = static_cast<std::int64_t>(compact_info.shape[1]);
    const std::int64_t nz = static_cast<std::int64_t>(compact_info.shape[2]);
    const std::int64_t nh = static_cast<std::int64_t>(compact_info.shape[3]);
    const std::int64_t nl = static_cast<std::int64_t>(transverse_info.shape[0]);

    const auto* source_slots = static_cast<const std::int64_t*>(slots_info.ptr);
    validate_source_slots(source_slots, nh * nl, n_beta);

    auto out = ComplexArray({nr, nz, n_beta});
    py::buffer_info out_info = out.request();

    const auto* compact = static_cast<const Complex128*>(compact_info.ptr);
    const auto* transverse = static_cast<const Complex128*>(transverse_info.ptr);
    const auto* psi = static_cast<const Complex128*>(psi_info.ptr);
    const auto* axial = static_cast<const Complex128*>(axial_info.ptr);
    auto* out_ptr = static_cast<Complex128*>(out_info.ptr);
    std::fill(out_ptr, out_ptr + nr * nz * n_beta, Complex128{0.0, 0.0});

    const std::int64_t tasks = nr * nz;
    unsigned int threads = 1;
    if (requested_threads > 0) {
        threads = static_cast<unsigned int>(requested_threads);
    } else {
        threads = choose_thread_count(tasks, n_illum * nh * nl);
    }

    {
        py::gil_scoped_release release;
        parallel_for(tasks, threads, [&](std::int64_t start, std::int64_t stop) {
            std::vector<Complex128> local(static_cast<std::size_t>(n_beta));
            for (std::int64_t task = start; task < stop; ++task) {
                const std::int64_t z = task % nz;
                const std::int64_t r = task / nz;
                std::fill(local.begin(), local.end(), Complex128{0.0, 0.0});
                const Complex128 axial_conj = std::conj(axial[z]);
                for (std::int64_t h = 0; h < nh; ++h) {
                    const std::int64_t* slot_row = source_slots + h * nl;
                    for (std::int64_t l = 0; l < nl; ++l) {
                        Complex128 illum_sum{0.0, 0.0};
                        for (std::int64_t illum = 0; illum < n_illum; ++illum) {
                            const Complex128 c =
                                compact[((illum * nr + r) * nz + z) * nh + h];
                            illum_sum += c * std::conj(psi[illum * nl + l]);
                        }
                        const Complex128 scale =
                            std::conj(transverse[l * nr + r]) * axial_conj;
                        local[static_cast<std::size_t>(slot_row[l])] += illum_sum * scale;
                    }
                }
                Complex128* out_row = out_ptr + (r * nz + z) * n_beta;
                std::copy(local.begin(), local.end(), out_row);
            }
        });
    }

    return out;
}

ComplexArray cone_axis_adjoint_unfold_scatter(
    ComplexArray residual_modes_array,
    FloatArray radial_array,
    ComplexArray axial_array,
    ComplexArray mode_phase_array,
    Int64Array h_slots_array,
    ComplexArray transverse_coeff_array,
    ComplexArray psi_phase_array,
    ComplexArray axial_phase_array,
    Int64Array source_slots_array,
    std::int64_t n_beta,
    std::int64_t requested_threads
) {
    const py::buffer_info residual_info = residual_modes_array.request();
    const py::buffer_info radial_info = radial_array.request();
    const py::buffer_info axis_axial_info = axial_array.request();
    const py::buffer_info mode_phase_info = mode_phase_array.request();
    const py::buffer_info h_slots_info = h_slots_array.request();
    const py::buffer_info transverse_info = transverse_coeff_array.request();
    const py::buffer_info psi_info = psi_phase_array.request();
    const py::buffer_info source_axial_info = axial_phase_array.request();
    const py::buffer_info source_slots_info = source_slots_array.request();
    validate_cone_axis_adjoint_unfold_scatter_shapes(
        residual_info,
        radial_info,
        axis_axial_info,
        mode_phase_info,
        h_slots_info,
        transverse_info,
        psi_info,
        source_axial_info,
        source_slots_info,
        n_beta
    );

    const std::int64_t n_illum = static_cast<std::int64_t>(residual_info.shape[0]);
    const std::int64_t cap_radial = static_cast<std::int64_t>(residual_info.shape[1]);
    const std::int64_t cap_phi = static_cast<std::int64_t>(residual_info.shape[2]);
    const std::int64_t nh = static_cast<std::int64_t>(radial_info.shape[0]);
    const std::int64_t nr = static_cast<std::int64_t>(radial_info.shape[2]);
    const std::int64_t nz = static_cast<std::int64_t>(axis_axial_info.shape[1]);
    const std::int64_t nl = static_cast<std::int64_t>(transverse_info.shape[0]);

    const auto* source_slots = static_cast<const std::int64_t*>(source_slots_info.ptr);
    const auto* h_slots = static_cast<const std::int64_t*>(h_slots_info.ptr);
    validate_source_slots(source_slots, nh * nl, n_beta);
    validate_source_slots(h_slots, nh, cap_phi);

    auto out = ComplexArray({nr, nz, n_beta});
    py::buffer_info out_info = out.request();

    const auto* residual = static_cast<const Complex128*>(residual_info.ptr);
    const auto* radial = static_cast<const double*>(radial_info.ptr);
    const auto* axis_axial = static_cast<const Complex128*>(axis_axial_info.ptr);
    const auto* mode_phase = static_cast<const Complex128*>(mode_phase_info.ptr);
    const auto* transverse = static_cast<const Complex128*>(transverse_info.ptr);
    const auto* psi = static_cast<const Complex128*>(psi_info.ptr);
    const auto* source_axial = static_cast<const Complex128*>(source_axial_info.ptr);
    auto* out_ptr = static_cast<Complex128*>(out_info.ptr);
    std::fill(out_ptr, out_ptr + nr * nz * n_beta, Complex128{0.0, 0.0});

    std::vector<Complex128> axis_axial_conj(static_cast<std::size_t>(cap_radial * nz));
    for (std::int64_t index = 0; index < cap_radial * nz; ++index) {
        axis_axial_conj[static_cast<std::size_t>(index)] = std::conj(axis_axial[index]);
    }
    std::vector<Complex128> mode_phase_conj(static_cast<std::size_t>(nh));
    for (std::int64_t h = 0; h < nh; ++h) {
        mode_phase_conj[static_cast<std::size_t>(h)] = std::conj(mode_phase[h]);
    }
    std::vector<Complex128> psi_conj(static_cast<std::size_t>(n_illum * nl));
    for (std::int64_t index = 0; index < n_illum * nl; ++index) {
        psi_conj[static_cast<std::size_t>(index)] = std::conj(psi[index]);
    }
    std::vector<Complex128> source_axial_conj(static_cast<std::size_t>(nz));
    for (std::int64_t z = 0; z < nz; ++z) {
        source_axial_conj[static_cast<std::size_t>(z)] = std::conj(source_axial[z]);
    }
    std::vector<Complex128> transverse_conj(static_cast<std::size_t>(nl * nr));
    for (std::int64_t index = 0; index < nl * nr; ++index) {
        transverse_conj[static_cast<std::size_t>(index)] = std::conj(transverse[index]);
    }

    const std::int64_t tasks = nr;
    unsigned int threads = 1;
    if (requested_threads > 0) {
        threads = static_cast<unsigned int>(requested_threads);
    } else {
        threads = choose_thread_count(tasks, nh * n_illum * nz * (cap_radial + nl));
    }

    {
        py::gil_scoped_release release;
        parallel_for(tasks, threads, [&](std::int64_t start, std::int64_t stop) {
            std::vector<Complex128> accum_z(static_cast<std::size_t>(nz));
            std::vector<Complex128> compact_cache(static_cast<std::size_t>(n_illum * nz));
            for (std::int64_t r = start; r < stop; ++r) {
                Complex128* out_r = out_ptr + r * nz * n_beta;
                for (std::int64_t h = 0; h < nh; ++h) {
                    const std::int64_t phi_slot = h_slots[h];
                    const std::int64_t* slot_row = source_slots + h * nl;
                    for (std::int64_t illum = 0; illum < n_illum; ++illum) {
                        std::fill(accum_z.begin(), accum_z.end(), Complex128{0.0, 0.0});
                        const Complex128* residual_row =
                            residual + illum * cap_radial * cap_phi + phi_slot;
                        for (std::int64_t u = 0; u < cap_radial; ++u) {
                            const Complex128 scaled =
                                residual_row[u * cap_phi] *
                                radial[(h * cap_radial + u) * nr + r];
                            const Complex128* axis_row =
                                axis_axial_conj.data() + u * nz;
                            for (std::int64_t z = 0; z < nz; ++z) {
                                accum_z[static_cast<std::size_t>(z)] += scaled * axis_row[z];
                            }
                        }
                        const Complex128 phase = mode_phase_conj[static_cast<std::size_t>(h)];
                        Complex128* compact_row = compact_cache.data() + illum * nz;
                        for (std::int64_t z = 0; z < nz; ++z) {
                            compact_row[z] =
                                accum_z[static_cast<std::size_t>(z)] *
                                phase *
                                source_axial_conj[static_cast<std::size_t>(z)];
                        }
                    }

                    for (std::int64_t l = 0; l < nl; ++l) {
                        const Complex128 scale = transverse_conj[static_cast<std::size_t>(l * nr + r)];
                        const std::int64_t slot = slot_row[l];
                        for (std::int64_t z = 0; z < nz; ++z) {
                            Complex128 illum_sum{0.0, 0.0};
                            for (std::int64_t illum = 0; illum < n_illum; ++illum) {
                                illum_sum += compact_cache[static_cast<std::size_t>(illum * nz + z)] *
                                             psi_conj[static_cast<std::size_t>(illum * nl + l)];
                            }
                            out_r[z * n_beta + slot] += illum_sum * scale;
                        }
                    }
                }
            }
        });
    }

    return out;
}

ComplexArray cone_axis_adjoint_unfold_scatter_pruned(
    ComplexArray residual_modes_array,
    FloatArray radial_array,
    ComplexArray axial_array,
    ComplexArray mode_phase_array,
    Int64Array h_slots_array,
    ComplexArray transverse_coeff_array,
    ComplexArray psi_phase_array,
    ComplexArray axial_phase_array,
    Int64Array source_slots_array,
    Int64Array l_offsets_array,
    Int64Array l_indices_array,
    std::int64_t n_beta,
    std::int64_t requested_threads
) {
    const py::buffer_info residual_info = residual_modes_array.request();
    const py::buffer_info radial_info = radial_array.request();
    const py::buffer_info axis_axial_info = axial_array.request();
    const py::buffer_info mode_phase_info = mode_phase_array.request();
    const py::buffer_info h_slots_info = h_slots_array.request();
    const py::buffer_info transverse_info = transverse_coeff_array.request();
    const py::buffer_info psi_info = psi_phase_array.request();
    const py::buffer_info source_axial_info = axial_phase_array.request();
    const py::buffer_info source_slots_info = source_slots_array.request();
    const py::buffer_info l_offsets_info = l_offsets_array.request();
    const py::buffer_info l_indices_info = l_indices_array.request();
    validate_cone_axis_adjoint_unfold_scatter_shapes(
        residual_info,
        radial_info,
        axis_axial_info,
        mode_phase_info,
        h_slots_info,
        transverse_info,
        psi_info,
        source_axial_info,
        source_slots_info,
        n_beta
    );

    const std::int64_t n_illum = static_cast<std::int64_t>(residual_info.shape[0]);
    const std::int64_t cap_radial = static_cast<std::int64_t>(residual_info.shape[1]);
    const std::int64_t cap_phi = static_cast<std::int64_t>(residual_info.shape[2]);
    const std::int64_t nh = static_cast<std::int64_t>(radial_info.shape[0]);
    const std::int64_t nr = static_cast<std::int64_t>(radial_info.shape[2]);
    const std::int64_t nz = static_cast<std::int64_t>(axis_axial_info.shape[1]);
    const std::int64_t nl = static_cast<std::int64_t>(transverse_info.shape[0]);

    validate_cone_axis_l_pruning_shapes(l_offsets_info, l_indices_info, nr, nl);
    const std::int64_t active_l_total = static_cast<std::int64_t>(l_indices_info.shape[0]);

    const auto* source_slots = static_cast<const std::int64_t*>(source_slots_info.ptr);
    const auto* h_slots = static_cast<const std::int64_t*>(h_slots_info.ptr);
    validate_source_slots(source_slots, nh * nl, n_beta);
    validate_source_slots(h_slots, nh, cap_phi);

    auto out = ComplexArray({nr, nz, n_beta});
    py::buffer_info out_info = out.request();

    const auto* residual = static_cast<const Complex128*>(residual_info.ptr);
    const auto* radial = static_cast<const double*>(radial_info.ptr);
    const auto* axis_axial = static_cast<const Complex128*>(axis_axial_info.ptr);
    const auto* mode_phase = static_cast<const Complex128*>(mode_phase_info.ptr);
    const auto* transverse = static_cast<const Complex128*>(transverse_info.ptr);
    const auto* psi = static_cast<const Complex128*>(psi_info.ptr);
    const auto* source_axial = static_cast<const Complex128*>(source_axial_info.ptr);
    const auto* l_offsets = static_cast<const std::int64_t*>(l_offsets_info.ptr);
    const auto* l_indices = static_cast<const std::int64_t*>(l_indices_info.ptr);
    auto* out_ptr = static_cast<Complex128*>(out_info.ptr);
    std::fill(out_ptr, out_ptr + nr * nz * n_beta, Complex128{0.0, 0.0});

    std::vector<Complex128> axis_axial_conj(static_cast<std::size_t>(cap_radial * nz));
    for (std::int64_t index = 0; index < cap_radial * nz; ++index) {
        axis_axial_conj[static_cast<std::size_t>(index)] = std::conj(axis_axial[index]);
    }
    std::vector<Complex128> mode_phase_conj(static_cast<std::size_t>(nh));
    for (std::int64_t h = 0; h < nh; ++h) {
        mode_phase_conj[static_cast<std::size_t>(h)] = std::conj(mode_phase[h]);
    }
    std::vector<Complex128> psi_conj(static_cast<std::size_t>(n_illum * nl));
    for (std::int64_t index = 0; index < n_illum * nl; ++index) {
        psi_conj[static_cast<std::size_t>(index)] = std::conj(psi[index]);
    }
    std::vector<Complex128> source_axial_conj(static_cast<std::size_t>(nz));
    for (std::int64_t z = 0; z < nz; ++z) {
        source_axial_conj[static_cast<std::size_t>(z)] = std::conj(source_axial[z]);
    }
    std::vector<Complex128> transverse_conj(static_cast<std::size_t>(nl * nr));
    for (std::int64_t index = 0; index < nl * nr; ++index) {
        transverse_conj[static_cast<std::size_t>(index)] = std::conj(transverse[index]);
    }
    std::vector<Complex128> active_transverse_conj(static_cast<std::size_t>(active_l_total));
    std::vector<Complex128> active_psi_conj(
        static_cast<std::size_t>(n_illum * active_l_total)
    );
    std::vector<std::int64_t> active_source_slots(
        static_cast<std::size_t>(nh * active_l_total)
    );
    for (std::int64_t r = 0; r < nr; ++r) {
        for (std::int64_t cursor = l_offsets[r]; cursor < l_offsets[r + 1]; ++cursor) {
            const std::int64_t l = l_indices[cursor];
            active_transverse_conj[static_cast<std::size_t>(cursor)] =
                transverse_conj[static_cast<std::size_t>(l * nr + r)];
            for (std::int64_t illum = 0; illum < n_illum; ++illum) {
                active_psi_conj[static_cast<std::size_t>(illum * active_l_total + cursor)] =
                    psi_conj[static_cast<std::size_t>(illum * nl + l)];
            }
            for (std::int64_t h = 0; h < nh; ++h) {
                active_source_slots[static_cast<std::size_t>(h * active_l_total + cursor)] =
                    source_slots[h * nl + l];
            }
        }
    }
    std::vector<Complex128> source_phase_conj(static_cast<std::size_t>(nh * nz));
    for (std::int64_t h = 0; h < nh; ++h) {
        for (std::int64_t z = 0; z < nz; ++z) {
            source_phase_conj[static_cast<std::size_t>(h * nz + z)] =
                mode_phase_conj[static_cast<std::size_t>(h)] *
                source_axial_conj[static_cast<std::size_t>(z)];
        }
    }
    std::vector<Complex128> radial_axis_conj(
        static_cast<std::size_t>(nh * nr * cap_radial * nz)
    );
    for (std::int64_t h = 0; h < nh; ++h) {
        for (std::int64_t r = 0; r < nr; ++r) {
            for (std::int64_t u = 0; u < cap_radial; ++u) {
                const double rad = radial[(h * cap_radial + u) * nr + r];
                const Complex128* axis_row =
                    axis_axial_conj.data() + u * nz;
                Complex128* pre_row =
                    radial_axis_conj.data() + ((h * nr + r) * cap_radial + u) * nz;
                for (std::int64_t z = 0; z < nz; ++z) {
                    pre_row[z] = rad * axis_row[z];
                }
            }
        }
    }

    const std::int64_t active_l_avg = std::max<std::int64_t>(1, active_l_total / std::max<std::int64_t>(nr, 1));
    const std::int64_t tasks = nr;
    unsigned int threads = 1;
    if (requested_threads > 0) {
        threads = static_cast<unsigned int>(requested_threads);
    } else {
        threads = choose_thread_count(tasks, nh * n_illum * nz * (cap_radial + active_l_avg));
    }

    {
        py::gil_scoped_release release;
        parallel_for(tasks, threads, [&](std::int64_t start, std::int64_t stop) {
            std::vector<Complex128> accum_z(static_cast<std::size_t>(nz));
            std::vector<Complex128> source_z(static_cast<std::size_t>(nz));
            std::vector<Complex128> l_accum;
            for (std::int64_t r = start; r < stop; ++r) {
                Complex128* out_r = out_ptr + r * nz * n_beta;
                const std::int64_t l_start = l_offsets[r];
                const std::int64_t l_stop = l_offsets[r + 1];
                const std::int64_t active_l_count = l_stop - l_start;
                const std::size_t l_accum_size =
                    static_cast<std::size_t>(active_l_count * nz);
                if (l_accum.size() < l_accum_size) {
                    l_accum.resize(l_accum_size);
                }
                if (active_l_count <= 0 || n_illum <= 0) {
                    continue;
                }
                for (std::int64_t h = 0; h < nh; ++h) {
                    const std::int64_t phi_slot = h_slots[h];
                    const Complex128* phase_row =
                        source_phase_conj.data() + h * nz;
                    const std::int64_t* active_slot_row =
                        active_source_slots.data() + h * active_l_total;
                    {
                        const std::int64_t illum = 0;
                        std::fill(accum_z.begin(), accum_z.end(), Complex128{0.0, 0.0});
                        const Complex128* residual_row =
                            residual + illum * cap_radial * cap_phi + phi_slot;
                        for (std::int64_t u = 0; u < cap_radial; ++u) {
                            const Complex128* axis_row =
                                radial_axis_conj.data() + ((h * nr + r) * cap_radial + u) * nz;
                            const Complex128 residual_value = residual_row[u * cap_phi];
                            for (std::int64_t z = 0; z < nz; ++z) {
                                accum_z[static_cast<std::size_t>(z)] += residual_value * axis_row[z];
                            }
                        }
                        for (std::int64_t z = 0; z < nz; ++z) {
                            source_z[static_cast<std::size_t>(z)] =
                                accum_z[static_cast<std::size_t>(z)] * phase_row[z];
                        }
                        const Complex128* active_psi_row =
                            active_psi_conj.data() + illum * active_l_total;
                        for (
                            std::int64_t local_l = 0, cursor = l_start;
                            cursor < l_stop;
                            ++local_l, ++cursor
                        ) {
                            const Complex128 weight =
                                active_psi_row[cursor];
                            Complex128* l_row =
                                l_accum.data() + local_l * nz;
                            for (std::int64_t z = 0; z < nz; ++z) {
                                l_row[z] =
                                    source_z[static_cast<std::size_t>(z)] * weight;
                            }
                        }
                    }
                    for (std::int64_t illum = 1; illum < n_illum; ++illum) {
                        std::fill(accum_z.begin(), accum_z.end(), Complex128{0.0, 0.0});
                        const Complex128* residual_row =
                            residual + illum * cap_radial * cap_phi + phi_slot;
                        for (std::int64_t u = 0; u < cap_radial; ++u) {
                            const Complex128* axis_row =
                                radial_axis_conj.data() + ((h * nr + r) * cap_radial + u) * nz;
                            const Complex128 residual_value = residual_row[u * cap_phi];
                            for (std::int64_t z = 0; z < nz; ++z) {
                                accum_z[static_cast<std::size_t>(z)] += residual_value * axis_row[z];
                            }
                        }
                        for (std::int64_t z = 0; z < nz; ++z) {
                            source_z[static_cast<std::size_t>(z)] =
                                accum_z[static_cast<std::size_t>(z)] * phase_row[z];
                        }
                        const Complex128* active_psi_row =
                            active_psi_conj.data() + illum * active_l_total;
                        for (
                            std::int64_t local_l = 0, cursor = l_start;
                            cursor < l_stop;
                            ++local_l, ++cursor
                        ) {
                            const Complex128 weight =
                                active_psi_row[cursor];
                            Complex128* l_row =
                                l_accum.data() + local_l * nz;
                            for (std::int64_t z = 0; z < nz; ++z) {
                                l_row[z] +=
                                    source_z[static_cast<std::size_t>(z)] * weight;
                            }
                        }
                    }

                    for (
                        std::int64_t local_l = 0, cursor = l_start;
                        cursor < l_stop;
                        ++local_l, ++cursor
                    ) {
                        const Complex128 scale =
                            active_transverse_conj[static_cast<std::size_t>(cursor)];
                        const std::int64_t slot = active_slot_row[cursor];
                        const Complex128* l_row =
                            l_accum.data() + local_l * nz;
                        for (std::int64_t z = 0; z < nz; ++z) {
                            out_r[z * n_beta + slot] += l_row[z] * scale;
                        }
                    }
                }
            }
        });
    }

    return out;
}

py::tuple cone_axis_prepare_adjoint_pruned(
    FloatArray radial_array,
    ComplexArray axial_array,
    ComplexArray mode_phase_array,
    ComplexArray transverse_coeff_array,
    ComplexArray psi_phase_array,
    ComplexArray axial_phase_array,
    Int64Array source_slots_array,
    Int64Array l_offsets_array,
    Int64Array l_indices_array
) {
    const py::buffer_info radial_info = radial_array.request();
    const py::buffer_info axis_axial_info = axial_array.request();
    const py::buffer_info mode_phase_info = mode_phase_array.request();
    const py::buffer_info transverse_info = transverse_coeff_array.request();
    const py::buffer_info psi_info = psi_phase_array.request();
    const py::buffer_info source_axial_info = axial_phase_array.request();
    const py::buffer_info source_slots_info = source_slots_array.request();
    const py::buffer_info l_offsets_info = l_offsets_array.request();
    const py::buffer_info l_indices_info = l_indices_array.request();

    if (radial_info.ndim != 3) {
        throw std::invalid_argument("radial must have shape (nh, cap_radial, nr)");
    }
    if (axis_axial_info.ndim != 2) {
        throw std::invalid_argument("axial must have shape (cap_radial, nz)");
    }
    if (mode_phase_info.ndim != 1) {
        throw std::invalid_argument("mode_phase must have shape (nh,)");
    }
    if (transverse_info.ndim != 2) {
        throw std::invalid_argument("transverse_coeff must have shape (nl, nr)");
    }
    if (psi_info.ndim != 2) {
        throw std::invalid_argument("psi_phase must have shape (n_illum, nl)");
    }
    if (source_axial_info.ndim != 1) {
        throw std::invalid_argument("axial_phase must have shape (nz,)");
    }
    if (source_slots_info.ndim != 2) {
        throw std::invalid_argument("source_slots must have shape (nh, nl)");
    }
    const std::int64_t nh = static_cast<std::int64_t>(radial_info.shape[0]);
    const std::int64_t cap_radial = static_cast<std::int64_t>(radial_info.shape[1]);
    const std::int64_t nr = static_cast<std::int64_t>(radial_info.shape[2]);
    const std::int64_t nz = static_cast<std::int64_t>(axis_axial_info.shape[1]);
    const std::int64_t nl = static_cast<std::int64_t>(transverse_info.shape[0]);
    const std::int64_t n_illum = static_cast<std::int64_t>(psi_info.shape[0]);
    if (axis_axial_info.shape[0] != cap_radial ||
        mode_phase_info.shape[0] != nh ||
        transverse_info.shape[1] != nr ||
        psi_info.shape[1] != nl ||
        source_axial_info.shape[0] != nz ||
        source_slots_info.shape[0] != nh ||
        source_slots_info.shape[1] != nl) {
        throw std::invalid_argument("inconsistent cone-axis prepared adjoint shapes");
    }
    validate_cone_axis_l_pruning_shapes(l_offsets_info, l_indices_info, nr, nl);
    const std::int64_t active_l_total = static_cast<std::int64_t>(l_indices_info.shape[0]);

    auto active_transverse_conj_array = ComplexArray({active_l_total});
    auto active_psi_conj_array = ComplexArray({n_illum, active_l_total});
    auto active_source_slots_array = Int64Array({nh, active_l_total});
    auto source_phase_conj_array = ComplexArray({nh, nz});
    auto radial_axis_conj_array = ComplexArray({nh, nr, cap_radial, nz});

    py::buffer_info active_transverse_info = active_transverse_conj_array.request();
    py::buffer_info active_psi_info = active_psi_conj_array.request();
    py::buffer_info active_source_slots_info = active_source_slots_array.request();
    py::buffer_info source_phase_info = source_phase_conj_array.request();
    py::buffer_info radial_axis_info = radial_axis_conj_array.request();

    const auto* radial = static_cast<const double*>(radial_info.ptr);
    const auto* axis_axial = static_cast<const Complex128*>(axis_axial_info.ptr);
    const auto* mode_phase = static_cast<const Complex128*>(mode_phase_info.ptr);
    const auto* transverse = static_cast<const Complex128*>(transverse_info.ptr);
    const auto* psi = static_cast<const Complex128*>(psi_info.ptr);
    const auto* source_axial = static_cast<const Complex128*>(source_axial_info.ptr);
    const auto* source_slots = static_cast<const std::int64_t*>(source_slots_info.ptr);
    const auto* l_offsets = static_cast<const std::int64_t*>(l_offsets_info.ptr);
    const auto* l_indices = static_cast<const std::int64_t*>(l_indices_info.ptr);
    auto* active_transverse = static_cast<Complex128*>(active_transverse_info.ptr);
    auto* active_psi = static_cast<Complex128*>(active_psi_info.ptr);
    auto* active_source_slots = static_cast<std::int64_t*>(active_source_slots_info.ptr);
    auto* source_phase = static_cast<Complex128*>(source_phase_info.ptr);
    auto* radial_axis = static_cast<Complex128*>(radial_axis_info.ptr);

    {
        py::gil_scoped_release release;
        std::vector<Complex128> axis_axial_conj(static_cast<std::size_t>(cap_radial * nz));
        for (std::int64_t index = 0; index < cap_radial * nz; ++index) {
            axis_axial_conj[static_cast<std::size_t>(index)] = std::conj(axis_axial[index]);
        }
        std::vector<Complex128> mode_phase_conj(static_cast<std::size_t>(nh));
        for (std::int64_t h = 0; h < nh; ++h) {
            mode_phase_conj[static_cast<std::size_t>(h)] = std::conj(mode_phase[h]);
        }
        std::vector<Complex128> psi_conj(static_cast<std::size_t>(n_illum * nl));
        for (std::int64_t index = 0; index < n_illum * nl; ++index) {
            psi_conj[static_cast<std::size_t>(index)] = std::conj(psi[index]);
        }
        std::vector<Complex128> source_axial_conj(static_cast<std::size_t>(nz));
        for (std::int64_t z = 0; z < nz; ++z) {
            source_axial_conj[static_cast<std::size_t>(z)] = std::conj(source_axial[z]);
        }
        std::vector<Complex128> transverse_conj(static_cast<std::size_t>(nl * nr));
        for (std::int64_t index = 0; index < nl * nr; ++index) {
            transverse_conj[static_cast<std::size_t>(index)] = std::conj(transverse[index]);
        }
        for (std::int64_t r = 0; r < nr; ++r) {
            for (std::int64_t cursor = l_offsets[r]; cursor < l_offsets[r + 1]; ++cursor) {
                const std::int64_t l = l_indices[cursor];
                active_transverse[cursor] =
                    transverse_conj[static_cast<std::size_t>(l * nr + r)];
                for (std::int64_t illum = 0; illum < n_illum; ++illum) {
                    active_psi[illum * active_l_total + cursor] =
                        psi_conj[static_cast<std::size_t>(illum * nl + l)];
                }
                for (std::int64_t h = 0; h < nh; ++h) {
                    active_source_slots[h * active_l_total + cursor] =
                        source_slots[h * nl + l];
                }
            }
        }
        for (std::int64_t h = 0; h < nh; ++h) {
            for (std::int64_t z = 0; z < nz; ++z) {
                source_phase[h * nz + z] =
                    mode_phase_conj[static_cast<std::size_t>(h)] *
                    source_axial_conj[static_cast<std::size_t>(z)];
            }
        }
        for (std::int64_t h = 0; h < nh; ++h) {
            for (std::int64_t r = 0; r < nr; ++r) {
                for (std::int64_t u = 0; u < cap_radial; ++u) {
                    const double rad = radial[(h * cap_radial + u) * nr + r];
                    const Complex128* axis_row =
                        axis_axial_conj.data() + u * nz;
                    Complex128* pre_row =
                        radial_axis + ((h * nr + r) * cap_radial + u) * nz;
                    for (std::int64_t z = 0; z < nz; ++z) {
                        pre_row[z] = rad * axis_row[z];
                    }
                }
            }
        }
    }

    return py::make_tuple(
        active_transverse_conj_array,
        active_psi_conj_array,
        active_source_slots_array,
        source_phase_conj_array,
        radial_axis_conj_array
    );
}

ComplexArray run_cone_axis_adjoint_prepared_core(
    const Complex128* residual,
    const std::int64_t* h_slots,
    const std::int64_t* l_offsets,
    const Complex128* active_transverse,
    const Complex128* active_psi,
    const std::int64_t* active_source_slots,
    const Complex128* source_phase,
    const Complex128* radial_axis,
    std::int64_t n_illum,
    std::int64_t cap_radial,
    std::int64_t cap_phi,
    std::int64_t active_l_total,
    std::int64_t nh,
    std::int64_t nz,
    std::int64_t nr,
    std::int64_t n_beta,
    std::int64_t requested_threads
) {
    auto out = ComplexArray({nr, nz, n_beta});
    py::buffer_info out_info = out.request();
    auto* out_ptr = static_cast<Complex128*>(out_info.ptr);
    std::fill(out_ptr, out_ptr + nr * nz * n_beta, Complex128{0.0, 0.0});

    const std::int64_t active_l_avg =
        std::max<std::int64_t>(1, active_l_total / std::max<std::int64_t>(nr, 1));
    const std::int64_t tasks = nr;
    unsigned int threads = 1;
    if (requested_threads > 0) {
        threads = static_cast<unsigned int>(requested_threads);
    } else {
        threads = choose_thread_count(
            tasks,
            nh * n_illum * nz * (cap_radial + active_l_avg)
        );
    }

    {
        py::gil_scoped_release release;
        parallel_for(tasks, threads, [&](std::int64_t start, std::int64_t stop) {
            std::vector<Complex128> accum_z(static_cast<std::size_t>(nz));
            std::vector<Complex128> source_z(static_cast<std::size_t>(nz));
            std::vector<Complex128> l_accum;
            for (std::int64_t r = start; r < stop; ++r) {
                Complex128* out_r = out_ptr + r * nz * n_beta;
                const std::int64_t l_start = l_offsets[r];
                const std::int64_t l_stop = l_offsets[r + 1];
                const std::int64_t active_l_count = l_stop - l_start;
                const std::size_t l_accum_size =
                    static_cast<std::size_t>(active_l_count * nz);
                if (l_accum.size() < l_accum_size) {
                    l_accum.resize(l_accum_size);
                }
                if (active_l_count <= 0 || n_illum <= 0) {
                    continue;
                }
                for (std::int64_t h = 0; h < nh; ++h) {
                    const std::int64_t phi_slot = h_slots[h];
                    const Complex128* phase_row = source_phase + h * nz;
                    const std::int64_t* active_slot_row =
                        active_source_slots + h * active_l_total;
                    {
                        const std::int64_t illum = 0;
                        std::fill(accum_z.begin(), accum_z.end(), Complex128{0.0, 0.0});
                        const Complex128* residual_row =
                            residual + illum * cap_radial * cap_phi + phi_slot;
                        for (std::int64_t u = 0; u < cap_radial; ++u) {
                            const Complex128* axis_row =
                                radial_axis + ((h * nr + r) * cap_radial + u) * nz;
                            const Complex128 residual_value = residual_row[u * cap_phi];
                            for (std::int64_t z = 0; z < nz; ++z) {
                                accum_z[static_cast<std::size_t>(z)] +=
                                    residual_value * axis_row[z];
                            }
                        }
                        for (std::int64_t z = 0; z < nz; ++z) {
                            source_z[static_cast<std::size_t>(z)] =
                                accum_z[static_cast<std::size_t>(z)] * phase_row[z];
                        }
                        const Complex128* active_psi_row =
                            active_psi + illum * active_l_total;
                        for (
                            std::int64_t local_l = 0, cursor = l_start;
                            cursor < l_stop;
                            ++local_l, ++cursor
                        ) {
                            const Complex128 weight = active_psi_row[cursor];
                            Complex128* l_row = l_accum.data() + local_l * nz;
                            for (std::int64_t z = 0; z < nz; ++z) {
                                l_row[z] =
                                    source_z[static_cast<std::size_t>(z)] * weight;
                            }
                        }
                    }
                    for (std::int64_t illum = 1; illum < n_illum; ++illum) {
                        std::fill(accum_z.begin(), accum_z.end(), Complex128{0.0, 0.0});
                        const Complex128* residual_row =
                            residual + illum * cap_radial * cap_phi + phi_slot;
                        for (std::int64_t u = 0; u < cap_radial; ++u) {
                            const Complex128* axis_row =
                                radial_axis + ((h * nr + r) * cap_radial + u) * nz;
                            const Complex128 residual_value = residual_row[u * cap_phi];
                            for (std::int64_t z = 0; z < nz; ++z) {
                                accum_z[static_cast<std::size_t>(z)] +=
                                    residual_value * axis_row[z];
                            }
                        }
                        for (std::int64_t z = 0; z < nz; ++z) {
                            source_z[static_cast<std::size_t>(z)] =
                                accum_z[static_cast<std::size_t>(z)] * phase_row[z];
                        }
                        const Complex128* active_psi_row =
                            active_psi + illum * active_l_total;
                        for (
                            std::int64_t local_l = 0, cursor = l_start;
                            cursor < l_stop;
                            ++local_l, ++cursor
                        ) {
                            const Complex128 weight = active_psi_row[cursor];
                            Complex128* l_row = l_accum.data() + local_l * nz;
                            for (std::int64_t z = 0; z < nz; ++z) {
                                l_row[z] +=
                                    source_z[static_cast<std::size_t>(z)] * weight;
                            }
                        }
                    }
                    for (
                        std::int64_t local_l = 0, cursor = l_start;
                        cursor < l_stop;
                        ++local_l, ++cursor
                    ) {
                        const Complex128 scale = active_transverse[cursor];
                        const std::int64_t slot = active_slot_row[cursor];
                        const Complex128* l_row = l_accum.data() + local_l * nz;
                        for (std::int64_t z = 0; z < nz; ++z) {
                            out_r[z * n_beta + slot] += l_row[z] * scale;
                        }
                    }
                }
            }
        });
    }

    return out;
}

ComplexArray run_cone_axis_adjoint_prepared_gathered_core(
    const Complex128* residual_gathered,
    const std::int64_t* l_offsets,
    const Complex128* active_transverse,
    const Complex128* active_psi,
    const std::int64_t* active_source_slots,
    const Complex128* source_phase,
    const Complex128* radial_axis,
    std::int64_t n_illum,
    std::int64_t cap_radial,
    std::int64_t active_l_total,
    std::int64_t nh,
    std::int64_t nz,
    std::int64_t nr,
    std::int64_t n_beta,
    std::int64_t requested_threads
) {
    auto out = ComplexArray({nr, nz, n_beta});
    py::buffer_info out_info = out.request();
    auto* out_ptr = static_cast<Complex128*>(out_info.ptr);
    std::fill(out_ptr, out_ptr + nr * nz * n_beta, Complex128{0.0, 0.0});

    const std::int64_t active_l_avg =
        std::max<std::int64_t>(1, active_l_total / std::max<std::int64_t>(nr, 1));
    const std::int64_t tasks = nr;
    unsigned int threads = 1;
    if (requested_threads > 0) {
        threads = static_cast<unsigned int>(requested_threads);
    } else {
        threads = choose_thread_count(
            tasks,
            nh * n_illum * nz * (cap_radial + active_l_avg)
        );
    }

    {
        py::gil_scoped_release release;
        parallel_for(tasks, threads, [&](std::int64_t start, std::int64_t stop) {
            std::vector<Complex128> accum_z(static_cast<std::size_t>(nz));
            std::vector<Complex128> source_z(static_cast<std::size_t>(nz));
            std::vector<Complex128> l_accum;
            for (std::int64_t r = start; r < stop; ++r) {
                Complex128* out_r = out_ptr + r * nz * n_beta;
                const std::int64_t l_start = l_offsets[r];
                const std::int64_t l_stop = l_offsets[r + 1];
                const std::int64_t active_l_count = l_stop - l_start;
                const std::size_t l_accum_size =
                    static_cast<std::size_t>(active_l_count * nz);
                if (l_accum.size() < l_accum_size) {
                    l_accum.resize(l_accum_size);
                }
                if (active_l_count <= 0 || n_illum <= 0) {
                    continue;
                }
                for (std::int64_t h = 0; h < nh; ++h) {
                    const Complex128* phase_row = source_phase + h * nz;
                    const Complex128* residual_h =
                        residual_gathered + h * n_illum * cap_radial;
                    const std::int64_t* active_slot_row =
                        active_source_slots + h * active_l_total;
                    {
                        const std::int64_t illum = 0;
                        std::fill(accum_z.begin(), accum_z.end(), Complex128{0.0, 0.0});
                        const Complex128* residual_row = residual_h;
                        for (std::int64_t u = 0; u < cap_radial; ++u) {
                            const Complex128* axis_row =
                                radial_axis + ((h * nr + r) * cap_radial + u) * nz;
                            const Complex128 residual_value = residual_row[u];
                            for (std::int64_t z = 0; z < nz; ++z) {
                                accum_z[static_cast<std::size_t>(z)] +=
                                    residual_value * axis_row[z];
                            }
                        }
                        for (std::int64_t z = 0; z < nz; ++z) {
                            source_z[static_cast<std::size_t>(z)] =
                                accum_z[static_cast<std::size_t>(z)] * phase_row[z];
                        }
                        const Complex128* active_psi_row =
                            active_psi + illum * active_l_total;
                        for (
                            std::int64_t local_l = 0, cursor = l_start;
                            cursor < l_stop;
                            ++local_l, ++cursor
                        ) {
                            const Complex128 weight = active_psi_row[cursor];
                            Complex128* l_row = l_accum.data() + local_l * nz;
                            for (std::int64_t z = 0; z < nz; ++z) {
                                l_row[z] =
                                    source_z[static_cast<std::size_t>(z)] * weight;
                            }
                        }
                    }
                    for (std::int64_t illum = 1; illum < n_illum; ++illum) {
                        std::fill(accum_z.begin(), accum_z.end(), Complex128{0.0, 0.0});
                        const Complex128* residual_row =
                            residual_h + illum * cap_radial;
                        for (std::int64_t u = 0; u < cap_radial; ++u) {
                            const Complex128* axis_row =
                                radial_axis + ((h * nr + r) * cap_radial + u) * nz;
                            const Complex128 residual_value = residual_row[u];
                            for (std::int64_t z = 0; z < nz; ++z) {
                                accum_z[static_cast<std::size_t>(z)] +=
                                    residual_value * axis_row[z];
                            }
                        }
                        for (std::int64_t z = 0; z < nz; ++z) {
                            source_z[static_cast<std::size_t>(z)] =
                                accum_z[static_cast<std::size_t>(z)] * phase_row[z];
                        }
                        const Complex128* active_psi_row =
                            active_psi + illum * active_l_total;
                        for (
                            std::int64_t local_l = 0, cursor = l_start;
                            cursor < l_stop;
                            ++local_l, ++cursor
                        ) {
                            const Complex128 weight = active_psi_row[cursor];
                            Complex128* l_row = l_accum.data() + local_l * nz;
                            for (std::int64_t z = 0; z < nz; ++z) {
                                l_row[z] +=
                                    source_z[static_cast<std::size_t>(z)] * weight;
                            }
                        }
                    }
                    for (
                        std::int64_t local_l = 0, cursor = l_start;
                        cursor < l_stop;
                        ++local_l, ++cursor
                    ) {
                        const Complex128 scale = active_transverse[cursor];
                        const std::int64_t slot = active_slot_row[cursor];
                        const Complex128* l_row = l_accum.data() + local_l * nz;
                        for (std::int64_t z = 0; z < nz; ++z) {
                            out_r[z * n_beta + slot] += l_row[z] * scale;
                        }
                    }
                }
            }
        });
    }

    return out;
}

ComplexArray run_cone_axis_adjoint_prepared_gathered_zmajor_core(
    const Complex128* residual_gathered,
    const std::int64_t* l_offsets,
    const Complex128* active_transverse,
    const Complex128* active_psi,
    const std::int64_t* active_source_slots,
    const Complex128* source_phase,
    const Complex128* radial_axis,
    std::int64_t n_illum,
    std::int64_t cap_radial,
    std::int64_t active_l_total,
    std::int64_t nh,
    std::int64_t nz,
    std::int64_t nr,
    std::int64_t n_beta,
    std::int64_t requested_threads
) {
    auto out = ComplexArray({nr, nz, n_beta});
    py::buffer_info out_info = out.request();
    auto* out_ptr = static_cast<Complex128*>(out_info.ptr);
    std::fill(out_ptr, out_ptr + nr * nz * n_beta, Complex128{0.0, 0.0});

    const std::int64_t active_l_avg =
        std::max<std::int64_t>(1, active_l_total / std::max<std::int64_t>(nr, 1));
    const std::int64_t tasks = nr;
    unsigned int threads = 1;
    if (requested_threads > 0) {
        threads = static_cast<unsigned int>(requested_threads);
    } else {
        threads = choose_thread_count(
            tasks,
            nh * n_illum * nz * (cap_radial + active_l_avg)
        );
    }

    {
        py::gil_scoped_release release;
        parallel_for(tasks, threads, [&](std::int64_t start, std::int64_t stop) {
            std::vector<Complex128> accum_z(static_cast<std::size_t>(nz));
            std::vector<Complex128> source_z(static_cast<std::size_t>(nz));
            std::vector<Complex128> l_accum;
            for (std::int64_t r = start; r < stop; ++r) {
                Complex128* out_r = out_ptr + r * nz * n_beta;
                const std::int64_t l_start = l_offsets[r];
                const std::int64_t l_stop = l_offsets[r + 1];
                const std::int64_t active_l_count = l_stop - l_start;
                const std::size_t l_accum_size =
                    static_cast<std::size_t>(nz * active_l_count);
                if (l_accum.size() < l_accum_size) {
                    l_accum.resize(l_accum_size);
                }
                if (active_l_count <= 0 || n_illum <= 0) {
                    continue;
                }
                for (std::int64_t h = 0; h < nh; ++h) {
                    const Complex128* phase_row = source_phase + h * nz;
                    const Complex128* residual_h =
                        residual_gathered + h * n_illum * cap_radial;
                    const std::int64_t* active_slot_row =
                        active_source_slots + h * active_l_total;
                    {
                        const std::int64_t illum = 0;
                        std::fill(accum_z.begin(), accum_z.end(), Complex128{0.0, 0.0});
                        const Complex128* residual_row = residual_h;
                        for (std::int64_t u = 0; u < cap_radial; ++u) {
                            const Complex128* axis_row =
                                radial_axis + ((h * nr + r) * cap_radial + u) * nz;
                            const Complex128 residual_value = residual_row[u];
                            for (std::int64_t z = 0; z < nz; ++z) {
                                accum_z[static_cast<std::size_t>(z)] +=
                                    residual_value * axis_row[z];
                            }
                        }
                        for (std::int64_t z = 0; z < nz; ++z) {
                            source_z[static_cast<std::size_t>(z)] =
                                accum_z[static_cast<std::size_t>(z)] * phase_row[z];
                        }
                        const Complex128* active_psi_row =
                            active_psi + illum * active_l_total;
                        for (std::int64_t z = 0; z < nz; ++z) {
                            const Complex128 source_value =
                                source_z[static_cast<std::size_t>(z)];
                            Complex128* l_row =
                                l_accum.data() + z * active_l_count;
                            for (
                                std::int64_t local_l = 0, cursor = l_start;
                                cursor < l_stop;
                                ++local_l, ++cursor
                            ) {
                                l_row[local_l] =
                                    source_value * active_psi_row[cursor];
                            }
                        }
                    }
                    for (std::int64_t illum = 1; illum < n_illum; ++illum) {
                        std::fill(accum_z.begin(), accum_z.end(), Complex128{0.0, 0.0});
                        const Complex128* residual_row =
                            residual_h + illum * cap_radial;
                        for (std::int64_t u = 0; u < cap_radial; ++u) {
                            const Complex128* axis_row =
                                radial_axis + ((h * nr + r) * cap_radial + u) * nz;
                            const Complex128 residual_value = residual_row[u];
                            for (std::int64_t z = 0; z < nz; ++z) {
                                accum_z[static_cast<std::size_t>(z)] +=
                                    residual_value * axis_row[z];
                            }
                        }
                        for (std::int64_t z = 0; z < nz; ++z) {
                            source_z[static_cast<std::size_t>(z)] =
                                accum_z[static_cast<std::size_t>(z)] * phase_row[z];
                        }
                        const Complex128* active_psi_row =
                            active_psi + illum * active_l_total;
                        for (std::int64_t z = 0; z < nz; ++z) {
                            const Complex128 source_value =
                                source_z[static_cast<std::size_t>(z)];
                            Complex128* l_row =
                                l_accum.data() + z * active_l_count;
                            for (
                                std::int64_t local_l = 0, cursor = l_start;
                                cursor < l_stop;
                                ++local_l, ++cursor
                            ) {
                                l_row[local_l] +=
                                    source_value * active_psi_row[cursor];
                            }
                        }
                    }
                    for (std::int64_t z = 0; z < nz; ++z) {
                        Complex128* out_z = out_r + z * n_beta;
                        const Complex128* l_row =
                            l_accum.data() + z * active_l_count;
                        for (
                            std::int64_t local_l = 0, cursor = l_start;
                            cursor < l_stop;
                            ++local_l, ++cursor
                        ) {
                            const Complex128 scale = active_transverse[cursor];
                            const std::int64_t slot = active_slot_row[cursor];
                            out_z[slot] += l_row[local_l] * scale;
                        }
                    }
                }
            }
        });
    }

    return out;
}

ComplexArray cone_axis_adjoint_unfold_scatter_pruned_prepared(
    ComplexArray residual_modes_array,
    Int64Array h_slots_array,
    Int64Array l_offsets_array,
    ComplexArray active_transverse_conj_array,
    ComplexArray active_psi_conj_array,
    Int64Array active_source_slots_array,
    ComplexArray source_phase_conj_array,
    ComplexArray radial_axis_conj_array,
    std::int64_t n_beta,
    std::int64_t requested_threads
) {
    const py::buffer_info residual_info = residual_modes_array.request();
    const py::buffer_info h_slots_info = h_slots_array.request();
    const py::buffer_info l_offsets_info = l_offsets_array.request();
    const py::buffer_info active_transverse_info = active_transverse_conj_array.request();
    const py::buffer_info active_psi_info = active_psi_conj_array.request();
    const py::buffer_info active_source_slots_info = active_source_slots_array.request();
    const py::buffer_info source_phase_info = source_phase_conj_array.request();
    const py::buffer_info radial_axis_info = radial_axis_conj_array.request();

    if (residual_info.ndim != 3) {
        throw std::invalid_argument("residual_modes must have shape (n_illum, cap_radial, cap_phi)");
    }
    if (h_slots_info.ndim != 1) {
        throw std::invalid_argument("h_slots must have shape (nh,)");
    }
    if (active_transverse_info.ndim != 1) {
        throw std::invalid_argument("active_transverse_conj must have shape (n_active_l,)");
    }
    if (active_psi_info.ndim != 2) {
        throw std::invalid_argument("active_psi_conj must have shape (n_illum, n_active_l)");
    }
    if (active_source_slots_info.ndim != 2) {
        throw std::invalid_argument("active_source_slots must have shape (nh, n_active_l)");
    }
    if (source_phase_info.ndim != 2) {
        throw std::invalid_argument("source_phase_conj must have shape (nh, nz)");
    }
    if (radial_axis_info.ndim != 4) {
        throw std::invalid_argument("radial_axis_conj must have shape (nh, nr, cap_radial, nz)");
    }
    const std::int64_t n_illum = static_cast<std::int64_t>(residual_info.shape[0]);
    const std::int64_t cap_radial = static_cast<std::int64_t>(residual_info.shape[1]);
    const std::int64_t cap_phi = static_cast<std::int64_t>(residual_info.shape[2]);
    const std::int64_t active_l_total = static_cast<std::int64_t>(active_transverse_info.shape[0]);
    const std::int64_t nh = static_cast<std::int64_t>(source_phase_info.shape[0]);
    const std::int64_t nz = static_cast<std::int64_t>(source_phase_info.shape[1]);
    const std::int64_t nr = static_cast<std::int64_t>(radial_axis_info.shape[1]);
    if (n_beta <= 0) {
        throw std::invalid_argument("n_beta must be positive");
    }
    if (h_slots_info.shape[0] != nh ||
        active_psi_info.shape[0] != n_illum ||
        active_psi_info.shape[1] != active_l_total ||
        active_source_slots_info.shape[0] != nh ||
        active_source_slots_info.shape[1] != active_l_total ||
        radial_axis_info.shape[0] != nh ||
        radial_axis_info.shape[2] != cap_radial ||
        radial_axis_info.shape[3] != nz) {
        throw std::invalid_argument("inconsistent prepared cone-axis adjoint shapes");
    }
    validate_cone_axis_prepared_l_offsets(l_offsets_info, nr, active_l_total);

    const auto* h_slots = static_cast<const std::int64_t*>(h_slots_info.ptr);
    validate_source_slots(h_slots, nh, cap_phi);
    const auto* active_source_slots = static_cast<const std::int64_t*>(active_source_slots_info.ptr);
    validate_source_slots(active_source_slots, nh * active_l_total, n_beta);

    const auto* residual = static_cast<const Complex128*>(residual_info.ptr);
    const auto* l_offsets = static_cast<const std::int64_t*>(l_offsets_info.ptr);
    const auto* active_transverse = static_cast<const Complex128*>(active_transverse_info.ptr);
    const auto* active_psi = static_cast<const Complex128*>(active_psi_info.ptr);
    const auto* source_phase = static_cast<const Complex128*>(source_phase_info.ptr);
    const auto* radial_axis = static_cast<const Complex128*>(radial_axis_info.ptr);
    return run_cone_axis_adjoint_prepared_core(
        residual,
        h_slots,
        l_offsets,
        active_transverse,
        active_psi,
        active_source_slots,
        source_phase,
        radial_axis,
        n_illum,
        cap_radial,
        cap_phi,
        active_l_total,
        nh,
        nz,
        nr,
        n_beta,
        requested_threads
    );
}

class ConeAxisPreparedAdjointPlan {
public:
    ConeAxisPreparedAdjointPlan(
        Int64Array h_slots_array,
        Int64Array l_offsets_array,
        ComplexArray active_transverse_conj_array,
        ComplexArray active_psi_conj_array,
        Int64Array active_source_slots_array,
        ComplexArray source_phase_conj_array,
        ComplexArray radial_axis_conj_array,
        std::int64_t n_beta
    )
        : h_slots_array_(h_slots_array),
          l_offsets_array_(l_offsets_array),
          active_transverse_conj_array_(active_transverse_conj_array),
          active_psi_conj_array_(active_psi_conj_array),
          active_source_slots_array_(active_source_slots_array),
          source_phase_conj_array_(source_phase_conj_array),
          radial_axis_conj_array_(radial_axis_conj_array),
          n_beta_(n_beta) {
        const py::buffer_info h_slots_info = h_slots_array_.request();
        const py::buffer_info l_offsets_info = l_offsets_array_.request();
        const py::buffer_info active_transverse_info = active_transverse_conj_array_.request();
        const py::buffer_info active_psi_info = active_psi_conj_array_.request();
        const py::buffer_info active_source_slots_info = active_source_slots_array_.request();
        const py::buffer_info source_phase_info = source_phase_conj_array_.request();
        const py::buffer_info radial_axis_info = radial_axis_conj_array_.request();

        if (n_beta_ <= 0) {
            throw std::invalid_argument("n_beta must be positive");
        }
        if (h_slots_info.ndim != 1) {
            throw std::invalid_argument("h_slots must have shape (nh,)");
        }
        if (active_transverse_info.ndim != 1) {
            throw std::invalid_argument("active_transverse_conj must have shape (n_active_l,)");
        }
        if (active_psi_info.ndim != 2) {
            throw std::invalid_argument("active_psi_conj must have shape (n_illum, n_active_l)");
        }
        if (active_source_slots_info.ndim != 2) {
            throw std::invalid_argument("active_source_slots must have shape (nh, n_active_l)");
        }
        if (source_phase_info.ndim != 2) {
            throw std::invalid_argument("source_phase_conj must have shape (nh, nz)");
        }
        if (radial_axis_info.ndim != 4) {
            throw std::invalid_argument("radial_axis_conj must have shape (nh, nr, cap_radial, nz)");
        }

        n_illum_ = static_cast<std::int64_t>(active_psi_info.shape[0]);
        active_l_total_ = static_cast<std::int64_t>(active_transverse_info.shape[0]);
        nh_ = static_cast<std::int64_t>(source_phase_info.shape[0]);
        nz_ = static_cast<std::int64_t>(source_phase_info.shape[1]);
        nr_ = static_cast<std::int64_t>(radial_axis_info.shape[1]);
        cap_radial_ = static_cast<std::int64_t>(radial_axis_info.shape[2]);
        if (h_slots_info.shape[0] != nh_ ||
            active_psi_info.shape[1] != active_l_total_ ||
            active_source_slots_info.shape[0] != nh_ ||
            active_source_slots_info.shape[1] != active_l_total_ ||
            radial_axis_info.shape[0] != nh_ ||
            radial_axis_info.shape[3] != nz_) {
            throw std::invalid_argument("inconsistent prepared cone-axis adjoint plan shapes");
        }
        validate_cone_axis_prepared_l_offsets(l_offsets_info, nr_, active_l_total_);

        h_slots_ = static_cast<const std::int64_t*>(h_slots_info.ptr);
        l_offsets_ = static_cast<const std::int64_t*>(l_offsets_info.ptr);
        active_transverse_ = static_cast<const Complex128*>(active_transverse_info.ptr);
        active_psi_ = static_cast<const Complex128*>(active_psi_info.ptr);
        active_source_slots_ = static_cast<const std::int64_t*>(active_source_slots_info.ptr);
        source_phase_ = static_cast<const Complex128*>(source_phase_info.ptr);
        radial_axis_ = static_cast<const Complex128*>(radial_axis_info.ptr);
        validate_source_slots(active_source_slots_, nh_ * active_l_total_, n_beta_);
    }

    ComplexArray execute(
        ComplexArray residual_modes_array,
        std::int64_t requested_threads
    ) const {
        const py::buffer_info residual_info = residual_modes_array.request();
        if (residual_info.ndim != 3) {
            throw std::invalid_argument("residual_modes must have shape (n_illum, cap_radial, cap_phi)");
        }
        if (residual_info.shape[0] != n_illum_ ||
            residual_info.shape[1] != cap_radial_) {
            throw std::invalid_argument("residual_modes shape does not match prepared cone-axis plan");
        }
        const std::int64_t cap_phi = static_cast<std::int64_t>(residual_info.shape[2]);
        const auto* residual = static_cast<const Complex128*>(residual_info.ptr);
        validate_source_slots(h_slots_, nh_, cap_phi);
        return run_cone_axis_adjoint_prepared_core(
            residual,
            h_slots_,
            l_offsets_,
            active_transverse_,
            active_psi_,
            active_source_slots_,
            source_phase_,
            radial_axis_,
            n_illum_,
            cap_radial_,
            cap_phi,
            active_l_total_,
            nh_,
            nz_,
            nr_,
            n_beta_,
            requested_threads
        );
    }

    ComplexArray execute_gathered(
        ComplexArray residual_modes_array,
        std::int64_t requested_threads
    ) const {
        const py::buffer_info residual_info = residual_modes_array.request();
        if (residual_info.ndim != 3) {
            throw std::invalid_argument("residual_modes must have shape (n_illum, cap_radial, cap_phi)");
        }
        if (residual_info.shape[0] != n_illum_ ||
            residual_info.shape[1] != cap_radial_) {
            throw std::invalid_argument("residual_modes shape does not match prepared cone-axis plan");
        }
        const std::int64_t cap_phi = static_cast<std::int64_t>(residual_info.shape[2]);
        const auto* residual = static_cast<const Complex128*>(residual_info.ptr);
        validate_source_slots(h_slots_, nh_, cap_phi);

        std::vector<Complex128> residual_gathered(
            static_cast<std::size_t>(nh_ * n_illum_ * cap_radial_)
        );
        for (std::int64_t h = 0; h < nh_; ++h) {
            const std::int64_t phi_slot = h_slots_[h];
            Complex128* gathered_h =
                residual_gathered.data() + h * n_illum_ * cap_radial_;
            for (std::int64_t illum = 0; illum < n_illum_; ++illum) {
                const Complex128* residual_row =
                    residual + illum * cap_radial_ * cap_phi + phi_slot;
                Complex128* gathered_row = gathered_h + illum * cap_radial_;
                for (std::int64_t u = 0; u < cap_radial_; ++u) {
                    gathered_row[u] = residual_row[u * cap_phi];
                }
            }
        }

        return run_cone_axis_adjoint_prepared_gathered_core(
            residual_gathered.data(),
            l_offsets_,
            active_transverse_,
            active_psi_,
            active_source_slots_,
            source_phase_,
            radial_axis_,
            n_illum_,
            cap_radial_,
            active_l_total_,
            nh_,
            nz_,
            nr_,
            n_beta_,
            requested_threads
        );
    }

    ComplexArray execute_gathered_zmajor(
        ComplexArray residual_modes_array,
        std::int64_t requested_threads
    ) const {
        const py::buffer_info residual_info = residual_modes_array.request();
        if (residual_info.ndim != 3) {
            throw std::invalid_argument("residual_modes must have shape (n_illum, cap_radial, cap_phi)");
        }
        if (residual_info.shape[0] != n_illum_ ||
            residual_info.shape[1] != cap_radial_) {
            throw std::invalid_argument("residual_modes shape does not match prepared cone-axis plan");
        }
        const std::int64_t cap_phi = static_cast<std::int64_t>(residual_info.shape[2]);
        const auto* residual = static_cast<const Complex128*>(residual_info.ptr);
        validate_source_slots(h_slots_, nh_, cap_phi);

        std::vector<Complex128> residual_gathered(
            static_cast<std::size_t>(nh_ * n_illum_ * cap_radial_)
        );
        for (std::int64_t h = 0; h < nh_; ++h) {
            const std::int64_t phi_slot = h_slots_[h];
            Complex128* gathered_h =
                residual_gathered.data() + h * n_illum_ * cap_radial_;
            for (std::int64_t illum = 0; illum < n_illum_; ++illum) {
                const Complex128* residual_row =
                    residual + illum * cap_radial_ * cap_phi + phi_slot;
                Complex128* gathered_row = gathered_h + illum * cap_radial_;
                for (std::int64_t u = 0; u < cap_radial_; ++u) {
                    gathered_row[u] = residual_row[u * cap_phi];
                }
            }
        }

        return run_cone_axis_adjoint_prepared_gathered_zmajor_core(
            residual_gathered.data(),
            l_offsets_,
            active_transverse_,
            active_psi_,
            active_source_slots_,
            source_phase_,
            radial_axis_,
            n_illum_,
            cap_radial_,
            active_l_total_,
            nh_,
            nz_,
            nr_,
            n_beta_,
            requested_threads
        );
    }

    std::int64_t n_beta() const { return n_beta_; }
    std::int64_t n_illum() const { return n_illum_; }
    std::int64_t cap_radial() const { return cap_radial_; }
    std::int64_t n_r() const { return nr_; }
    std::int64_t n_z() const { return nz_; }
    std::int64_t n_h() const { return nh_; }
    std::int64_t active_l_total() const { return active_l_total_; }

private:
    Int64Array h_slots_array_;
    Int64Array l_offsets_array_;
    ComplexArray active_transverse_conj_array_;
    ComplexArray active_psi_conj_array_;
    Int64Array active_source_slots_array_;
    ComplexArray source_phase_conj_array_;
    ComplexArray radial_axis_conj_array_;
    std::int64_t n_beta_ = 0;
    std::int64_t n_illum_ = 0;
    std::int64_t cap_radial_ = 0;
    std::int64_t active_l_total_ = 0;
    std::int64_t nh_ = 0;
    std::int64_t nz_ = 0;
    std::int64_t nr_ = 0;
    const std::int64_t* h_slots_ = nullptr;
    const std::int64_t* l_offsets_ = nullptr;
    const Complex128* active_transverse_ = nullptr;
    const Complex128* active_psi_ = nullptr;
    const std::int64_t* active_source_slots_ = nullptr;
    const Complex128* source_phase_ = nullptr;
    const Complex128* radial_axis_ = nullptr;
};

ComplexArray cone_axis_adjoint_unfold_scatter_pruned_prepared_batch(
    ComplexArray residual_modes_batch_array,
    Int64Array h_slots_array,
    Int64Array l_offsets_array,
    ComplexArray active_transverse_conj_array,
    ComplexArray active_psi_conj_array,
    Int64Array active_source_slots_array,
    ComplexArray source_phase_conj_array,
    ComplexArray radial_axis_conj_array,
    std::int64_t n_beta,
    std::int64_t requested_threads
) {
    const py::buffer_info residual_info = residual_modes_batch_array.request();
    const py::buffer_info h_slots_info = h_slots_array.request();
    const py::buffer_info l_offsets_info = l_offsets_array.request();
    const py::buffer_info active_transverse_info = active_transverse_conj_array.request();
    const py::buffer_info active_psi_info = active_psi_conj_array.request();
    const py::buffer_info active_source_slots_info = active_source_slots_array.request();
    const py::buffer_info source_phase_info = source_phase_conj_array.request();
    const py::buffer_info radial_axis_info = radial_axis_conj_array.request();

    if (residual_info.ndim != 4) {
        throw std::invalid_argument(
            "residual_modes_batch must have shape (batch, n_illum, cap_radial, cap_phi)"
        );
    }
    if (h_slots_info.ndim != 1) {
        throw std::invalid_argument("h_slots must have shape (nh,)");
    }
    if (active_transverse_info.ndim != 1) {
        throw std::invalid_argument("active_transverse_conj must have shape (n_active_l,)");
    }
    if (active_psi_info.ndim != 2) {
        throw std::invalid_argument("active_psi_conj must have shape (n_illum, n_active_l)");
    }
    if (active_source_slots_info.ndim != 2) {
        throw std::invalid_argument("active_source_slots must have shape (nh, n_active_l)");
    }
    if (source_phase_info.ndim != 2) {
        throw std::invalid_argument("source_phase_conj must have shape (nh, nz)");
    }
    if (radial_axis_info.ndim != 4) {
        throw std::invalid_argument("radial_axis_conj must have shape (nh, nr, cap_radial, nz)");
    }

    const std::int64_t batch = static_cast<std::int64_t>(residual_info.shape[0]);
    const std::int64_t n_illum = static_cast<std::int64_t>(residual_info.shape[1]);
    const std::int64_t cap_radial = static_cast<std::int64_t>(residual_info.shape[2]);
    const std::int64_t cap_phi = static_cast<std::int64_t>(residual_info.shape[3]);
    const std::int64_t active_l_total = static_cast<std::int64_t>(active_transverse_info.shape[0]);
    const std::int64_t nh = static_cast<std::int64_t>(source_phase_info.shape[0]);
    const std::int64_t nz = static_cast<std::int64_t>(source_phase_info.shape[1]);
    const std::int64_t nr = static_cast<std::int64_t>(radial_axis_info.shape[1]);
    if (batch <= 0) {
        throw std::invalid_argument("batch must be positive");
    }
    if (n_beta <= 0) {
        throw std::invalid_argument("n_beta must be positive");
    }
    if (h_slots_info.shape[0] != nh ||
        active_psi_info.shape[0] != n_illum ||
        active_psi_info.shape[1] != active_l_total ||
        active_source_slots_info.shape[0] != nh ||
        active_source_slots_info.shape[1] != active_l_total ||
        radial_axis_info.shape[0] != nh ||
        radial_axis_info.shape[2] != cap_radial ||
        radial_axis_info.shape[3] != nz) {
        throw std::invalid_argument("inconsistent prepared cone-axis batch adjoint shapes");
    }
    validate_cone_axis_prepared_l_offsets(l_offsets_info, nr, active_l_total);

    const auto* h_slots = static_cast<const std::int64_t*>(h_slots_info.ptr);
    validate_source_slots(h_slots, nh, cap_phi);
    const auto* active_source_slots = static_cast<const std::int64_t*>(active_source_slots_info.ptr);
    validate_source_slots(active_source_slots, nh * active_l_total, n_beta);

    auto out = ComplexArray({batch, nr, nz, n_beta});
    py::buffer_info out_info = out.request();
    const auto* residual = static_cast<const Complex128*>(residual_info.ptr);
    const auto* l_offsets = static_cast<const std::int64_t*>(l_offsets_info.ptr);
    const auto* active_transverse = static_cast<const Complex128*>(active_transverse_info.ptr);
    const auto* active_psi = static_cast<const Complex128*>(active_psi_info.ptr);
    const auto* source_phase = static_cast<const Complex128*>(source_phase_info.ptr);
    const auto* radial_axis = static_cast<const Complex128*>(radial_axis_info.ptr);
    auto* out_ptr = static_cast<Complex128*>(out_info.ptr);
    std::fill(out_ptr, out_ptr + batch * nr * nz * n_beta, Complex128{0.0, 0.0});

    const std::int64_t active_l_avg = std::max<std::int64_t>(
        1,
        active_l_total / std::max<std::int64_t>(nr, 1)
    );
    const std::int64_t tasks = nr * batch;
    unsigned int threads = 1;
    if (requested_threads > 0) {
        threads = static_cast<unsigned int>(requested_threads);
    } else {
        threads = choose_thread_count(
            tasks,
            nh * n_illum * nz * (cap_radial + active_l_avg)
        );
    }

    {
        py::gil_scoped_release release;
        parallel_for(tasks, threads, [&](std::int64_t start, std::int64_t stop) {
            std::vector<Complex128> accum_z(static_cast<std::size_t>(nz));
            std::vector<Complex128> source_z(static_cast<std::size_t>(nz));
            std::vector<Complex128> l_accum;
            for (std::int64_t task = start; task < stop; ++task) {
                const std::int64_t r = task / batch;
                const std::int64_t b = task - r * batch;
                Complex128* out_r =
                    out_ptr + ((b * nr + r) * nz * n_beta);
                const Complex128* residual_b =
                    residual + b * n_illum * cap_radial * cap_phi;
                const std::int64_t l_start = l_offsets[r];
                const std::int64_t l_stop = l_offsets[r + 1];
                const std::int64_t active_l_count = l_stop - l_start;
                const std::size_t l_accum_size =
                    static_cast<std::size_t>(active_l_count * nz);
                if (l_accum.size() < l_accum_size) {
                    l_accum.resize(l_accum_size);
                }
                if (active_l_count <= 0 || n_illum <= 0) {
                    continue;
                }
                for (std::int64_t h = 0; h < nh; ++h) {
                    const std::int64_t phi_slot = h_slots[h];
                    const Complex128* phase_row = source_phase + h * nz;
                    const std::int64_t* active_slot_row =
                        active_source_slots + h * active_l_total;
                    {
                        const std::int64_t illum = 0;
                        std::fill(accum_z.begin(), accum_z.end(), Complex128{0.0, 0.0});
                        const Complex128* residual_row =
                            residual_b + illum * cap_radial * cap_phi + phi_slot;
                        for (std::int64_t u = 0; u < cap_radial; ++u) {
                            const Complex128* axis_row =
                                radial_axis + ((h * nr + r) * cap_radial + u) * nz;
                            const Complex128 residual_value = residual_row[u * cap_phi];
                            for (std::int64_t z = 0; z < nz; ++z) {
                                accum_z[static_cast<std::size_t>(z)] +=
                                    residual_value * axis_row[z];
                            }
                        }
                        for (std::int64_t z = 0; z < nz; ++z) {
                            source_z[static_cast<std::size_t>(z)] =
                                accum_z[static_cast<std::size_t>(z)] * phase_row[z];
                        }
                        const Complex128* active_psi_row =
                            active_psi + illum * active_l_total;
                        for (
                            std::int64_t local_l = 0, cursor = l_start;
                            cursor < l_stop;
                            ++local_l, ++cursor
                        ) {
                            const Complex128 weight = active_psi_row[cursor];
                            Complex128* l_row = l_accum.data() + local_l * nz;
                            for (std::int64_t z = 0; z < nz; ++z) {
                                l_row[z] =
                                    source_z[static_cast<std::size_t>(z)] * weight;
                            }
                        }
                    }
                    for (std::int64_t illum = 1; illum < n_illum; ++illum) {
                        std::fill(accum_z.begin(), accum_z.end(), Complex128{0.0, 0.0});
                        const Complex128* residual_row =
                            residual_b + illum * cap_radial * cap_phi + phi_slot;
                        for (std::int64_t u = 0; u < cap_radial; ++u) {
                            const Complex128* axis_row =
                                radial_axis + ((h * nr + r) * cap_radial + u) * nz;
                            const Complex128 residual_value = residual_row[u * cap_phi];
                            for (std::int64_t z = 0; z < nz; ++z) {
                                accum_z[static_cast<std::size_t>(z)] +=
                                    residual_value * axis_row[z];
                            }
                        }
                        for (std::int64_t z = 0; z < nz; ++z) {
                            source_z[static_cast<std::size_t>(z)] =
                                accum_z[static_cast<std::size_t>(z)] * phase_row[z];
                        }
                        const Complex128* active_psi_row =
                            active_psi + illum * active_l_total;
                        for (
                            std::int64_t local_l = 0, cursor = l_start;
                            cursor < l_stop;
                            ++local_l, ++cursor
                        ) {
                            const Complex128 weight = active_psi_row[cursor];
                            Complex128* l_row = l_accum.data() + local_l * nz;
                            for (std::int64_t z = 0; z < nz; ++z) {
                                l_row[z] +=
                                    source_z[static_cast<std::size_t>(z)] * weight;
                            }
                        }
                    }
                    for (
                        std::int64_t local_l = 0, cursor = l_start;
                        cursor < l_stop;
                        ++local_l, ++cursor
                    ) {
                        const Complex128 scale = active_transverse[cursor];
                        const std::int64_t slot = active_slot_row[cursor];
                        const Complex128* l_row = l_accum.data() + local_l * nz;
                        for (std::int64_t z = 0; z < nz; ++z) {
                            out_r[z * n_beta + slot] += l_row[z] * scale;
                        }
                    }
                }
            }
        });
    }

    return out;
}

ComplexArray same_direction_adjoint_unfold_scatter(
    ComplexArray residual_modes_array,
    FloatArray radial_array,
    ComplexArray axial_array,
    ComplexArray mode_phase_array,
    Int64Array h_slots_array,
    ComplexArray transverse_by_mag_array,
    ComplexArray axial_phase_by_mag_array,
    Int64Array source_slots_array,
    std::int64_t n_beta,
    std::int64_t requested_threads
) {
    const py::buffer_info residual_info = residual_modes_array.request();
    const py::buffer_info radial_info = radial_array.request();
    const py::buffer_info axis_axial_info = axial_array.request();
    const py::buffer_info mode_phase_info = mode_phase_array.request();
    const py::buffer_info h_slots_info = h_slots_array.request();
    const py::buffer_info transverse_info = transverse_by_mag_array.request();
    const py::buffer_info source_axial_info = axial_phase_by_mag_array.request();
    const py::buffer_info source_slots_info = source_slots_array.request();
    validate_same_direction_adjoint_unfold_scatter_shapes(
        residual_info,
        radial_info,
        axis_axial_info,
        mode_phase_info,
        h_slots_info,
        transverse_info,
        source_axial_info,
        source_slots_info,
        n_beta
    );

    const std::int64_t n_mag = static_cast<std::int64_t>(residual_info.shape[0]);
    const std::int64_t cap_radial = static_cast<std::int64_t>(residual_info.shape[1]);
    const std::int64_t cap_phi = static_cast<std::int64_t>(residual_info.shape[2]);
    const std::int64_t nh = static_cast<std::int64_t>(radial_info.shape[0]);
    const std::int64_t nr = static_cast<std::int64_t>(radial_info.shape[2]);
    const std::int64_t nz = static_cast<std::int64_t>(axis_axial_info.shape[1]);
    const std::int64_t nl = static_cast<std::int64_t>(transverse_info.shape[1]);

    const auto* source_slots = static_cast<const std::int64_t*>(source_slots_info.ptr);
    const auto* h_slots = static_cast<const std::int64_t*>(h_slots_info.ptr);
    validate_source_slots(source_slots, nh * nl, n_beta);
    validate_source_slots(h_slots, nh, cap_phi);

    auto out = ComplexArray({nr, nz, n_beta});
    py::buffer_info out_info = out.request();

    const auto* residual = static_cast<const Complex128*>(residual_info.ptr);
    const auto* radial = static_cast<const double*>(radial_info.ptr);
    const auto* axis_axial = static_cast<const Complex128*>(axis_axial_info.ptr);
    const auto* mode_phase = static_cast<const Complex128*>(mode_phase_info.ptr);
    const auto* transverse = static_cast<const Complex128*>(transverse_info.ptr);
    const auto* source_axial = static_cast<const Complex128*>(source_axial_info.ptr);
    auto* out_ptr = static_cast<Complex128*>(out_info.ptr);
    std::fill(out_ptr, out_ptr + nr * nz * n_beta, Complex128{0.0, 0.0});

    std::vector<Complex128> axis_axial_conj(static_cast<std::size_t>(cap_radial * nz));
    for (std::int64_t index = 0; index < cap_radial * nz; ++index) {
        axis_axial_conj[static_cast<std::size_t>(index)] = std::conj(axis_axial[index]);
    }
    std::vector<Complex128> mode_phase_conj(static_cast<std::size_t>(nh));
    for (std::int64_t h = 0; h < nh; ++h) {
        mode_phase_conj[static_cast<std::size_t>(h)] = std::conj(mode_phase[h]);
    }
    std::vector<Complex128> source_axial_conj(static_cast<std::size_t>(n_mag * nz));
    for (std::int64_t index = 0; index < n_mag * nz; ++index) {
        source_axial_conj[static_cast<std::size_t>(index)] = std::conj(source_axial[index]);
    }
    std::vector<Complex128> transverse_conj(static_cast<std::size_t>(n_mag * nl * nr));
    for (std::int64_t index = 0; index < n_mag * nl * nr; ++index) {
        transverse_conj[static_cast<std::size_t>(index)] = std::conj(transverse[index]);
    }

    const std::int64_t tasks = nr;
    unsigned int threads = 1;
    if (requested_threads > 0) {
        threads = static_cast<unsigned int>(requested_threads);
    } else {
        threads = choose_thread_count(tasks, nh * n_mag * nz * (cap_radial + nl));
    }

    {
        py::gil_scoped_release release;
        parallel_for(tasks, threads, [&](std::int64_t start, std::int64_t stop) {
            std::vector<Complex128> accum_z(static_cast<std::size_t>(nz));
            std::vector<Complex128> compact_cache(static_cast<std::size_t>(n_mag * nz));
            for (std::int64_t r = start; r < stop; ++r) {
                Complex128* out_r = out_ptr + r * nz * n_beta;
                for (std::int64_t h = 0; h < nh; ++h) {
                    const std::int64_t phi_slot = h_slots[h];
                    const std::int64_t* slot_row = source_slots + h * nl;
                    for (std::int64_t mag = 0; mag < n_mag; ++mag) {
                        std::fill(accum_z.begin(), accum_z.end(), Complex128{0.0, 0.0});
                        const Complex128* residual_row =
                            residual + mag * cap_radial * cap_phi + phi_slot;
                        for (std::int64_t u = 0; u < cap_radial; ++u) {
                            const Complex128 scaled =
                                residual_row[u * cap_phi] *
                                radial[(h * cap_radial + u) * nr + r];
                            const Complex128* axis_row =
                                axis_axial_conj.data() + u * nz;
                            for (std::int64_t z = 0; z < nz; ++z) {
                                accum_z[static_cast<std::size_t>(z)] += scaled * axis_row[z];
                            }
                        }
                        const Complex128 phase = mode_phase_conj[static_cast<std::size_t>(h)];
                        Complex128* compact_row = compact_cache.data() + mag * nz;
                        for (std::int64_t z = 0; z < nz; ++z) {
                            compact_row[z] =
                                accum_z[static_cast<std::size_t>(z)] *
                                phase *
                                source_axial_conj[static_cast<std::size_t>(mag * nz + z)];
                        }
                    }

                    for (std::int64_t l = 0; l < nl; ++l) {
                        const std::int64_t slot = slot_row[l];
                        for (
                            std::int64_t z = 0; z < nz; ++z
                        ) {
                            Complex128 mag_sum{0.0, 0.0};
                            for (std::int64_t mag = 0; mag < n_mag; ++mag) {
                                mag_sum += compact_cache[static_cast<std::size_t>(mag * nz + z)] *
                                           transverse_conj[
                                               static_cast<std::size_t>((mag * nl + l) * nr + r)
                                           ];
                            }
                            out_r[z * n_beta + slot] += mag_sum;
                        }
                    }
                }
            }
        });
    }

    return out;
}

ComplexArray svd_rank_adjoint_unfold_scatter(
    ComplexArray residual_modes_array,
    FloatArray radial_array,
    ComplexArray axial_array,
    ComplexArray mode_phase_array,
    Int64Array h_slots_array,
    ComplexArray weights_array,
    Int64Array source_slots_array,
    std::int64_t n_beta,
    std::int64_t requested_threads,
    bool weights_are_conjugated
) {
    const py::buffer_info residual_info = residual_modes_array.request();
    const py::buffer_info radial_info = radial_array.request();
    const py::buffer_info axis_axial_info = axial_array.request();
    const py::buffer_info mode_phase_info = mode_phase_array.request();
    const py::buffer_info h_slots_info = h_slots_array.request();
    const py::buffer_info weights_info = weights_array.request();
    const py::buffer_info source_slots_info = source_slots_array.request();
    validate_svd_rank_adjoint_unfold_scatter_shapes(
        residual_info,
        radial_info,
        axis_axial_info,
        mode_phase_info,
        h_slots_info,
        weights_info,
        source_slots_info,
        n_beta
    );

    const std::int64_t rank = static_cast<std::int64_t>(residual_info.shape[0]);
    const std::int64_t cap_radial = static_cast<std::int64_t>(residual_info.shape[1]);
    const std::int64_t cap_phi = static_cast<std::int64_t>(residual_info.shape[2]);
    const std::int64_t nh = static_cast<std::int64_t>(radial_info.shape[0]);
    const std::int64_t nr = static_cast<std::int64_t>(radial_info.shape[2]);
    const std::int64_t nz = static_cast<std::int64_t>(axis_axial_info.shape[1]);
    const std::int64_t nl = static_cast<std::int64_t>(weights_info.shape[1]);

    const auto* source_slots = static_cast<const std::int64_t*>(source_slots_info.ptr);
    const auto* h_slots = static_cast<const std::int64_t*>(h_slots_info.ptr);
    validate_source_slots(source_slots, nh * nl, n_beta);
    validate_source_slots(h_slots, nh, cap_phi);

    auto out = ComplexArray({nr, nz, n_beta});
    py::buffer_info out_info = out.request();

    const auto* residual = static_cast<const Complex128*>(residual_info.ptr);
    const auto* radial = static_cast<const double*>(radial_info.ptr);
    const auto* axis_axial = static_cast<const Complex128*>(axis_axial_info.ptr);
    const auto* mode_phase = static_cast<const Complex128*>(mode_phase_info.ptr);
    const auto* weights = static_cast<const Complex128*>(weights_info.ptr);
    auto* out_ptr = static_cast<Complex128*>(out_info.ptr);
    std::fill(out_ptr, out_ptr + nr * nz * n_beta, Complex128{0.0, 0.0});

    std::vector<Complex128> axis_axial_conj(static_cast<std::size_t>(cap_radial * nz));
    for (std::int64_t index = 0; index < cap_radial * nz; ++index) {
        axis_axial_conj[static_cast<std::size_t>(index)] = std::conj(axis_axial[index]);
    }
    std::vector<Complex128> mode_phase_conj(static_cast<std::size_t>(nh));
    for (std::int64_t h = 0; h < nh; ++h) {
        mode_phase_conj[static_cast<std::size_t>(h)] = std::conj(mode_phase[h]);
    }
    const Complex128* adjoint_weights = weights;
    std::vector<Complex128> weights_conj;
    if (!weights_are_conjugated) {
        weights_conj.resize(static_cast<std::size_t>(rank * nl * nr * nz));
        for (std::int64_t index = 0; index < rank * nl * nr * nz; ++index) {
            weights_conj[static_cast<std::size_t>(index)] = std::conj(weights[index]);
        }
        adjoint_weights = weights_conj.data();
    }

    const std::int64_t tasks = nr;
    unsigned int threads = 1;
    if (requested_threads > 0) {
        threads = static_cast<unsigned int>(requested_threads);
    } else {
        threads = choose_thread_count_capped(tasks, nh * rank * nz * (cap_radial + nl), 8);
    }

    {
        py::gil_scoped_release release;
        parallel_for(tasks, threads, [&](std::int64_t start, std::int64_t stop) {
            std::vector<Complex128> accum_z(static_cast<std::size_t>(nz));
            std::vector<Complex128> compact_cache(static_cast<std::size_t>(rank * nz));
            for (std::int64_t r = start; r < stop; ++r) {
                Complex128* out_r = out_ptr + r * nz * n_beta;
                for (std::int64_t h = 0; h < nh; ++h) {
                    const std::int64_t phi_slot = h_slots[h];
                    const std::int64_t* slot_row = source_slots + h * nl;
                    for (std::int64_t s = 0; s < rank; ++s) {
                        std::fill(accum_z.begin(), accum_z.end(), Complex128{0.0, 0.0});
                        const Complex128* residual_row =
                            residual + s * cap_radial * cap_phi + phi_slot;
                        for (std::int64_t u = 0; u < cap_radial; ++u) {
                            const Complex128 scaled =
                                residual_row[u * cap_phi] *
                                radial[(h * cap_radial + u) * nr + r];
                            const Complex128* axis_row =
                                axis_axial_conj.data() + u * nz;
                            for (std::int64_t z = 0; z < nz; ++z) {
                                accum_z[static_cast<std::size_t>(z)] += scaled * axis_row[z];
                            }
                        }
                        const Complex128 phase = mode_phase_conj[static_cast<std::size_t>(h)];
                        Complex128* compact_row = compact_cache.data() + s * nz;
                        for (std::int64_t z = 0; z < nz; ++z) {
                            compact_row[z] = accum_z[static_cast<std::size_t>(z)] * phase;
                        }
                    }

                    for (std::int64_t l = 0; l < nl; ++l) {
                        const std::int64_t slot = slot_row[l];
                        for (std::int64_t z = 0; z < nz; ++z) {
                            Complex128 rank_sum{0.0, 0.0};
                            for (std::int64_t s = 0; s < rank; ++s) {
                                rank_sum += compact_cache[static_cast<std::size_t>(s * nz + z)] *
                                            adjoint_weights[
                                                static_cast<std::size_t>(((s * nl + l) * nr + r) * nz + z)
                                            ];
                            }
                            out_r[z * n_beta + slot] += rank_sum;
                        }
                    }
                }
            }
        });
    }

    return out;
}

}  // namespace

PYBIND11_MODULE(_cpp_odt, m) {
    m.doc() = "C++ kernels for experimental ODT Ewald-cap operator benchmarks.";
    m.def(
        "forward_contract",
        &forward_contract,
        py::arg("coeff_h"),
        py::arg("radial"),
        py::arg("axial"),
        py::arg("angular"),
        py::arg("threads") = 0,
        "Evaluate cached structured ODT Ewald-cap forward contractions."
    );
    m.def(
        "adjoint_contract_compact",
        &adjoint_contract_compact,
        py::arg("residual"),
        py::arg("radial"),
        py::arg("axial"),
        py::arg("angular"),
        py::arg("threads") = 0,
        "Evaluate cached structured ODT Ewald-cap adjoint contractions into compact harmonic coefficients."
    );
    m.def(
        "resample4_interpolate",
        &resample4_interpolate,
        py::arg("values"),
        py::arg("indices"),
        py::arg("weights"),
        py::arg("threads") = 0,
        "Apply cached four-point complex interpolation for detector resampling."
    );
    m.def(
        "resample4_scatter_adjoint",
        &resample4_scatter_adjoint,
        py::arg("residual"),
        py::arg("indices"),
        py::arg("weights"),
        py::arg("source_size"),
        "Apply the transpose of cached four-point detector interpolation."
    );
    m.def(
        "axis_grid_forward_fold",
        &axis_grid_forward_fold,
        py::arg("coeff_h"),
        py::arg("radial"),
        py::arg("axial"),
        py::arg("mode_phase"),
        py::arg("h_slots"),
        py::arg("cap_phi"),
        py::arg("threads") = 0,
        "Fold axis-grid ODT harmonic contractions before detector-phi FFT."
    );
    m.def(
        "axis_grid_adjoint_unfold",
        &axis_grid_adjoint_unfold,
        py::arg("residual_modes"),
        py::arg("radial"),
        py::arg("axial"),
        py::arg("mode_phase"),
        py::arg("h_slots"),
        py::arg("threads") = 0,
        "Unfold detector-phi FFT residual modes into compact axis-grid ODT harmonics."
    );
    m.def(
        "phase_selected_dft",
        &phase_selected_dft,
        py::arg("coeff"),
        py::arg("phase"),
        py::arg("twiddle"),
        py::arg("threads") = 0,
        "Compute selected beta harmonics of phase-modulated ODT coefficients."
    );
    m.def(
        "phase_selected_idft_adjoint",
        &phase_selected_idft_adjoint,
        py::arg("compact"),
        py::arg("phase"),
        py::arg("twiddle"),
        py::arg("threads") = 0,
        "Apply the adjoint of selected phase-modulated beta harmonics."
    );
    m.def(
        "cone_axis_decompose_forward",
        &cone_axis_decompose_forward,
        py::arg("coeff_h_full"),
        py::arg("transverse_coeff"),
        py::arg("psi_phase"),
        py::arg("axial_phase"),
        py::arg("source_slots"),
        py::arg("threads") = 0,
        "Apply full-rank cone z0/zperp source-harmonic decomposition."
    );
    m.def(
        "cone_axis_forward_fold",
        &cone_axis_forward_fold,
        py::arg("coeff_h_full"),
        py::arg("radial"),
        py::arg("axial"),
        py::arg("mode_phase"),
        py::arg("h_slots"),
        py::arg("transverse_coeff"),
        py::arg("psi_phase"),
        py::arg("axial_phase"),
        py::arg("source_slots"),
        py::arg("cap_phi"),
        py::arg("threads") = 0,
        "Fuse full-rank cone z0/zperp source decomposition with axis-grid harmonic fold."
    );
    m.def(
        "cone_axis_forward_fold_pruned",
        &cone_axis_forward_fold_pruned,
        py::arg("coeff_h_full"),
        py::arg("radial"),
        py::arg("axial"),
        py::arg("mode_phase"),
        py::arg("h_slots"),
        py::arg("transverse_coeff"),
        py::arg("psi_phase"),
        py::arg("axial_phase"),
        py::arg("source_slots"),
        py::arg("l_offsets"),
        py::arg("l_indices"),
        py::arg("cap_phi"),
        py::arg("threads") = 0,
        "Fuse cone source decomposition with axis-grid fold using radial adaptive l pruning."
    );
    m.def(
        "cone_axis_forward_fold_pruned_partitioned",
        &cone_axis_forward_fold_pruned_partitioned,
        py::arg("coeff_h_full"),
        py::arg("radial"),
        py::arg("axial"),
        py::arg("mode_phase"),
        py::arg("h_slots"),
        py::arg("transverse_coeff"),
        py::arg("psi_phase"),
        py::arg("axial_phase"),
        py::arg("source_slots"),
        py::arg("l_offsets"),
        py::arg("l_indices"),
        py::arg("cap_phi"),
        py::arg("threads") = 0,
        "Evaluate pruned cone-axis forward by precomputing source values and partitioning output slots."
    );
    m.def(
        "cone_axis_decompose_adjoint",
        &cone_axis_decompose_adjoint,
        py::arg("compact"),
        py::arg("transverse_coeff"),
        py::arg("psi_phase"),
        py::arg("axial_phase"),
        py::arg("source_slots"),
        py::arg("n_beta"),
        py::arg("threads") = 0,
        "Apply the adjoint of full-rank cone z0/zperp source-harmonic decomposition."
    );
    m.def(
        "cone_axis_adjoint_unfold_scatter",
        &cone_axis_adjoint_unfold_scatter,
        py::arg("residual_modes"),
        py::arg("radial"),
        py::arg("axial"),
        py::arg("mode_phase"),
        py::arg("h_slots"),
        py::arg("transverse_coeff"),
        py::arg("psi_phase"),
        py::arg("axial_phase"),
        py::arg("source_slots"),
        py::arg("n_beta"),
        py::arg("threads") = 0,
        "Fuse axis-grid adjoint unfold with full-rank cone source-harmonic adjoint scatter."
    );
    m.def(
        "cone_axis_adjoint_unfold_scatter_pruned",
        &cone_axis_adjoint_unfold_scatter_pruned,
        py::arg("residual_modes"),
        py::arg("radial"),
        py::arg("axial"),
        py::arg("mode_phase"),
        py::arg("h_slots"),
        py::arg("transverse_coeff"),
        py::arg("psi_phase"),
        py::arg("axial_phase"),
        py::arg("source_slots"),
        py::arg("l_offsets"),
        py::arg("l_indices"),
        py::arg("n_beta"),
        py::arg("threads") = 0,
        "Fuse axis-grid adjoint unfold with cone source adjoint scatter using radial adaptive l pruning."
    );
    m.def(
        "cone_axis_prepare_adjoint_pruned",
        &cone_axis_prepare_adjoint_pruned,
        py::arg("radial"),
        py::arg("axial"),
        py::arg("mode_phase"),
        py::arg("transverse_coeff"),
        py::arg("psi_phase"),
        py::arg("axial_phase"),
        py::arg("source_slots"),
        py::arg("l_offsets"),
        py::arg("l_indices"),
        "Prepare reusable cone-axis pruned adjoint tables for repeated residuals."
    );
    m.def(
        "cone_axis_adjoint_unfold_scatter_pruned_prepared",
        &cone_axis_adjoint_unfold_scatter_pruned_prepared,
        py::arg("residual_modes"),
        py::arg("h_slots"),
        py::arg("l_offsets"),
        py::arg("active_transverse_conj"),
        py::arg("active_psi_conj"),
        py::arg("active_source_slots"),
        py::arg("source_phase_conj"),
        py::arg("radial_axis_conj"),
        py::arg("n_beta"),
        py::arg("threads") = 0,
        "Run pruned cone-axis adjoint with reusable prepared tables."
    );
    py::class_<ConeAxisPreparedAdjointPlan>(m, "ConeAxisPreparedAdjointPlan")
        .def(
            py::init<
                Int64Array,
                Int64Array,
                ComplexArray,
                ComplexArray,
                Int64Array,
                ComplexArray,
                ComplexArray,
                std::int64_t
            >(),
            py::arg("h_slots"),
            py::arg("l_offsets"),
            py::arg("active_transverse_conj"),
            py::arg("active_psi_conj"),
            py::arg("active_source_slots"),
            py::arg("source_phase_conj"),
            py::arg("radial_axis_conj"),
            py::arg("n_beta")
        )
        .def(
            "execute",
            &ConeAxisPreparedAdjointPlan::execute,
            py::arg("residual_modes"),
            py::arg("threads") = 0,
            "Run the prepared cone-axis adjoint for one residual mode stack."
        )
        .def(
            "execute_gathered",
            &ConeAxisPreparedAdjointPlan::execute_gathered,
            py::arg("residual_modes"),
            py::arg("threads") = 0,
            "Run the prepared cone-axis adjoint after gathering h-slot residuals."
        )
        .def(
            "execute_gathered_zmajor",
            &ConeAxisPreparedAdjointPlan::execute_gathered_zmajor,
            py::arg("residual_modes"),
            py::arg("threads") = 0,
            "Run the gathered prepared cone-axis adjoint with z-major l accumulation."
        )
        .def_property_readonly("n_beta", &ConeAxisPreparedAdjointPlan::n_beta)
        .def_property_readonly("n_illum", &ConeAxisPreparedAdjointPlan::n_illum)
        .def_property_readonly("cap_radial", &ConeAxisPreparedAdjointPlan::cap_radial)
        .def_property_readonly("n_r", &ConeAxisPreparedAdjointPlan::n_r)
        .def_property_readonly("n_z", &ConeAxisPreparedAdjointPlan::n_z)
        .def_property_readonly("n_h", &ConeAxisPreparedAdjointPlan::n_h)
        .def_property_readonly("active_l_total", &ConeAxisPreparedAdjointPlan::active_l_total);
    m.def(
        "cone_axis_adjoint_unfold_scatter_pruned_prepared_batch",
        &cone_axis_adjoint_unfold_scatter_pruned_prepared_batch,
        py::arg("residual_modes_batch"),
        py::arg("h_slots"),
        py::arg("l_offsets"),
        py::arg("active_transverse_conj"),
        py::arg("active_psi_conj"),
        py::arg("active_source_slots"),
        py::arg("source_phase_conj"),
        py::arg("radial_axis_conj"),
        py::arg("n_beta"),
        py::arg("threads") = 0,
        "Run batched pruned cone-axis adjoint with reusable prepared tables."
    );
    m.def(
        "same_direction_forward_fold",
        &same_direction_forward_fold,
        py::arg("coeff_h_full"),
        py::arg("radial"),
        py::arg("axial"),
        py::arg("mode_phase"),
        py::arg("h_slots"),
        py::arg("transverse_by_mag"),
        py::arg("axial_phase_by_mag"),
        py::arg("source_slots"),
        py::arg("cap_phi"),
        py::arg("threads") = 0,
        "Fuse same-direction z_perp magnitude decomposition with axis-grid harmonic fold."
    );
    m.def(
        "same_direction_adjoint_unfold_scatter",
        &same_direction_adjoint_unfold_scatter,
        py::arg("residual_modes"),
        py::arg("radial"),
        py::arg("axial"),
        py::arg("mode_phase"),
        py::arg("h_slots"),
        py::arg("transverse_by_mag"),
        py::arg("axial_phase_by_mag"),
        py::arg("source_slots"),
        py::arg("n_beta"),
        py::arg("threads") = 0,
        "Fuse axis-grid adjoint unfold with same-direction z_perp magnitude scatter."
    );
    m.def(
        "svd_rank_forward_fold",
        &svd_rank_forward_fold,
        py::arg("coeff_h_full"),
        py::arg("radial"),
        py::arg("axial"),
        py::arg("mode_phase"),
        py::arg("h_slots"),
        py::arg("weights"),
        py::arg("source_slots"),
        py::arg("cap_phi"),
        py::arg("threads") = 0,
        "Fuse SVD-compressed z_perp magnitude rank modes with axis-grid harmonic fold."
    );
    m.def(
        "svd_rank_adjoint_unfold_scatter",
        &svd_rank_adjoint_unfold_scatter,
        py::arg("residual_modes"),
        py::arg("radial"),
        py::arg("axial"),
        py::arg("mode_phase"),
        py::arg("h_slots"),
        py::arg("weights"),
        py::arg("source_slots"),
        py::arg("n_beta"),
        py::arg("threads") = 0,
        py::arg("weights_are_conjugated") = false,
        "Fuse axis-grid adjoint unfold with SVD-compressed z_perp magnitude rank scatter."
    );
}
