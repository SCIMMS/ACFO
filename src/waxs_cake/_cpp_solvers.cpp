#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

namespace py = pybind11;

namespace {

using Complex128 = std::complex<double>;
using Complex64 = std::complex<float>;

#if defined(_MSC_VER)
#define WAXS_IVDEP __pragma(loop(ivdep))
#elif defined(__GNUC__)
#define WAXS_IVDEP _Pragma("GCC ivdep")
#else
#define WAXS_IVDEP
#endif

template <typename ComplexT>
using TypedComplexArray =
    py::array_t<ComplexT, py::array::c_style | py::array::forcecast>;

using ComplexArray = TypedComplexArray<Complex128>;
using Complex64Array = TypedComplexArray<Complex64>;

template <typename ComplexT>
ComplexT i_power(py::ssize_t n) {
    using ScalarT = typename ComplexT::value_type;
    switch (static_cast<int>(n & 3)) {
        case 0:
            return ComplexT{static_cast<ScalarT>(1), static_cast<ScalarT>(0)};
        case 1:
            return ComplexT{static_cast<ScalarT>(0), static_cast<ScalarT>(1)};
        case 2:
            return ComplexT{static_cast<ScalarT>(-1), static_cast<ScalarT>(0)};
        default:
            return ComplexT{static_cast<ScalarT>(0), static_cast<ScalarT>(-1)};
    }
}

unsigned int choose_thread_count(py::ssize_t n_items, py::ssize_t work_per_item) {
    if (n_items <= 1 || work_per_item < 20000) {
        return 1;
    }
    unsigned int hardware = std::thread::hardware_concurrency();
    if (hardware == 0) {
        hardware = 1;
    }
    return std::max(1u, std::min<unsigned int>(hardware, static_cast<unsigned int>(n_items)));
}

void validate_fused_shapes(
    const py::buffer_info& hhat,
    const py::buffer_info& z_phase,
    const py::buffer_info& khat,
    const py::buffer_info& form_factors
) {
    if (hhat.ndim != 4) {
        throw std::invalid_argument("hhat must have shape (n_elements, n_r, n_z, n_phi)");
    }
    if (z_phase.ndim != 2) {
        throw std::invalid_argument("z_phase must have shape (n_q, n_z)");
    }
    if (khat.ndim != 3) {
        throw std::invalid_argument("khat must have shape (n_q, n_r, n_phi)");
    }
    if (form_factors.ndim != 2) {
        throw std::invalid_argument("form_factors must have shape (n_elements, n_q)");
    }
    if (hhat.shape[0] != form_factors.shape[0] || hhat.shape[1] != khat.shape[1] ||
        hhat.shape[2] != z_phase.shape[1] || hhat.shape[3] != khat.shape[2] ||
        z_phase.shape[0] != khat.shape[0] || z_phase.shape[0] != form_factors.shape[1]) {
        throw std::invalid_argument("inconsistent fused circular contraction shapes");
    }
}

void validate_z_reduced_shapes(
    const py::buffer_info& z_reduced,
    const py::buffer_info& khat,
    const py::buffer_info& form_factors
) {
    if (z_reduced.ndim != 4) {
        throw std::invalid_argument("z_reduced must have shape (n_q, n_elements, n_r, n_phi)");
    }
    if (khat.ndim != 3) {
        throw std::invalid_argument("khat must have shape (n_q, n_r, n_phi)");
    }
    if (form_factors.ndim != 2) {
        throw std::invalid_argument("form_factors must have shape (n_elements, n_q)");
    }
    if (z_reduced.shape[0] != khat.shape[0] || z_reduced.shape[0] != form_factors.shape[1] ||
        z_reduced.shape[1] != form_factors.shape[0] || z_reduced.shape[2] != khat.shape[1] ||
        z_reduced.shape[3] != khat.shape[2]) {
        throw std::invalid_argument("inconsistent z-reduced circular contraction shapes");
    }
}

void validate_ring_cutoff_shapes(
    const py::buffer_info& hhat,
    const py::buffer_info& z_phase,
    const py::buffer_info& khat,
    const py::buffer_info& form_factors,
    const py::buffer_info& cutoffs
) {
    validate_fused_shapes(hhat, z_phase, khat, form_factors);
    if (cutoffs.ndim != 2) {
        throw std::invalid_argument("cutoffs must have shape (n_q, n_r)");
    }
    if (cutoffs.shape[0] != z_phase.shape[0] || cutoffs.shape[1] != hhat.shape[1]) {
        throw std::invalid_argument("cutoffs must have shape (n_q, n_r)");
    }
}

void validate_r_dependent_modes_shapes(
    const py::buffer_info& hhat,
    const py::buffer_info& z_phase,
    const py::buffer_info& khat,
    const py::buffer_info& form_factors,
    const py::buffer_info& cutoffs,
    std::int64_t max_cutoff
) {
    if (hhat.ndim != 4) {
        throw std::invalid_argument("hhat must have shape (n_elements, n_r, n_z, n_phi)");
    }
    if (z_phase.ndim != 2) {
        throw std::invalid_argument("z_phase must have shape (n_q, n_z)");
    }
    if (khat.ndim != 3) {
        throw std::invalid_argument("compact khat must have shape (n_q, n_r, n_h)");
    }
    if (form_factors.ndim != 2) {
        throw std::invalid_argument("form_factors must have shape (n_elements, n_q)");
    }
    if (cutoffs.ndim != 2) {
        throw std::invalid_argument("cutoffs must have shape (n_q, n_r)");
    }
    if (max_cutoff < 0) {
        throw std::invalid_argument("max_cutoff must be non-negative");
    }
    const py::ssize_t expected_n_h = static_cast<py::ssize_t>(max_cutoff) + 1;
    if (hhat.shape[0] != form_factors.shape[0] || hhat.shape[1] != khat.shape[1] ||
        hhat.shape[2] != z_phase.shape[1] || z_phase.shape[0] != khat.shape[0] ||
        z_phase.shape[0] != form_factors.shape[1] || cutoffs.shape[0] != z_phase.shape[0] ||
        cutoffs.shape[1] != hhat.shape[1] || khat.shape[2] != expected_n_h) {
        throw std::invalid_argument("inconsistent compact R-dependent contraction shapes");
    }
    if (max_cutoff >= hhat.shape[3] / 2) {
        throw std::invalid_argument("compact R-dependent contraction requires max_cutoff < n_phi / 2");
    }
}

void validate_r_dependent_half_modes_shapes(
    const py::buffer_info& hhat,
    const py::buffer_info& z_phase,
    const py::buffer_info& khat,
    const py::buffer_info& form_factors,
    const py::buffer_info& cutoffs,
    std::int64_t n_phi,
    std::int64_t max_cutoff
) {
    if (hhat.ndim != 4) {
        throw std::invalid_argument("hhat must have shape (n_elements, n_r, n_z, n_modes)");
    }
    if (z_phase.ndim != 2) {
        throw std::invalid_argument("z_phase must have shape (n_q, n_z)");
    }
    if (khat.ndim != 3) {
        throw std::invalid_argument("compact khat must have shape (n_q, n_r, n_h)");
    }
    if (form_factors.ndim != 2) {
        throw std::invalid_argument("form_factors must have shape (n_elements, n_q)");
    }
    if (cutoffs.ndim != 2) {
        throw std::invalid_argument("cutoffs must have shape (n_q, n_r)");
    }
    if (n_phi <= 0) {
        throw std::invalid_argument("n_phi must be positive");
    }
    if (max_cutoff < 0) {
        throw std::invalid_argument("max_cutoff must be non-negative");
    }
    const py::ssize_t n_phi_ss = static_cast<py::ssize_t>(n_phi);
    const py::ssize_t expected_n_h = static_cast<py::ssize_t>(max_cutoff) + 1;
    if (hhat.shape[0] != form_factors.shape[0] || hhat.shape[1] != khat.shape[1] ||
        hhat.shape[2] != z_phase.shape[1] || hhat.shape[3] < expected_n_h ||
        hhat.shape[3] > n_phi_ss ||
        z_phase.shape[0] != khat.shape[0] || z_phase.shape[0] != form_factors.shape[1] ||
        cutoffs.shape[0] != z_phase.shape[0] || cutoffs.shape[1] != hhat.shape[1] ||
        khat.shape[2] != expected_n_h) {
        throw std::invalid_argument("inconsistent half-spectrum compact R-dependent contraction shapes");
    }
    if (max_cutoff >= n_phi_ss / 2) {
        throw std::invalid_argument("half-spectrum compact R-dependent contraction requires max_cutoff < n_phi / 2");
    }
}

void validate_r_dependent_half_miller_shapes(
    const py::buffer_info& hhat,
    const py::buffer_info& z_phase,
    const py::buffer_info& q_perp,
    const py::buffer_info& r_centers,
    const py::buffer_info& form_factors,
    const py::buffer_info& cutoffs,
    std::int64_t n_phi,
    std::int64_t max_cutoff
) {
    if (hhat.ndim != 4) {
        throw std::invalid_argument("hhat must have shape (n_elements, n_r, n_z, n_modes)");
    }
    if (z_phase.ndim != 2) {
        throw std::invalid_argument("z_phase must have shape (n_q, n_z)");
    }
    if (q_perp.ndim != 1) {
        throw std::invalid_argument("q_perp must have shape (n_q,)");
    }
    if (r_centers.ndim != 1) {
        throw std::invalid_argument("r_centers must have shape (n_r,)");
    }
    if (form_factors.ndim != 2) {
        throw std::invalid_argument("form_factors must have shape (n_elements, n_q)");
    }
    if (cutoffs.ndim != 2) {
        throw std::invalid_argument("cutoffs must have shape (n_q, n_r)");
    }
    if (n_phi <= 0) {
        throw std::invalid_argument("n_phi must be positive");
    }
    if (max_cutoff < 0) {
        throw std::invalid_argument("max_cutoff must be non-negative");
    }
    const py::ssize_t n_phi_ss = static_cast<py::ssize_t>(n_phi);
    const py::ssize_t expected_n_h = static_cast<py::ssize_t>(max_cutoff) + 1;
    if (hhat.shape[0] != form_factors.shape[0] || hhat.shape[1] != r_centers.shape[0] ||
        hhat.shape[2] != z_phase.shape[1] || hhat.shape[3] < expected_n_h ||
        hhat.shape[3] > n_phi_ss || z_phase.shape[0] != q_perp.shape[0] ||
        z_phase.shape[0] != form_factors.shape[1] || cutoffs.shape[0] != z_phase.shape[0] ||
        cutoffs.shape[1] != hhat.shape[1]) {
        throw std::invalid_argument("inconsistent fused Miller R-dependent contraction shapes");
    }
    if (max_cutoff >= n_phi_ss / 2) {
        throw std::invalid_argument("fused Miller R-dependent contraction requires max_cutoff < n_phi / 2");
    }
}

void validate_r_dependent_half_z_reduced_shapes(
    const py::buffer_info& z_pos,
    const py::buffer_info& z_neg,
    const py::buffer_info& khat,
    const py::buffer_info& form_factors,
    const py::buffer_info& cutoffs,
    std::int64_t n_phi,
    std::int64_t max_cutoff
) {
    if (z_pos.ndim != 4 || z_neg.ndim != 4) {
        throw std::invalid_argument("z_pos and z_neg must have shape (n_q, n_elements, n_r, n_h)");
    }
    if (khat.ndim != 3) {
        throw std::invalid_argument("compact khat must have shape (n_q, n_r, n_h)");
    }
    if (form_factors.ndim != 2) {
        throw std::invalid_argument("form_factors must have shape (n_elements, n_q)");
    }
    if (cutoffs.ndim != 2) {
        throw std::invalid_argument("cutoffs must have shape (n_q, n_r)");
    }
    if (n_phi <= 0) {
        throw std::invalid_argument("n_phi must be positive");
    }
    if (max_cutoff < 0) {
        throw std::invalid_argument("max_cutoff must be non-negative");
    }
    const py::ssize_t expected_n_h = static_cast<py::ssize_t>(max_cutoff) + 1;
    const py::ssize_t n_phi_ss = static_cast<py::ssize_t>(n_phi);
    if (z_pos.shape[0] != z_neg.shape[0] || z_pos.shape[1] != z_neg.shape[1] ||
        z_pos.shape[2] != z_neg.shape[2] || z_pos.shape[3] != z_neg.shape[3] ||
        z_pos.shape[0] != khat.shape[0] || z_pos.shape[1] != form_factors.shape[0] ||
        z_pos.shape[2] != khat.shape[1] || z_pos.shape[3] != expected_n_h ||
        z_pos.shape[0] != form_factors.shape[1] || cutoffs.shape[0] != z_pos.shape[0] ||
        cutoffs.shape[1] != z_pos.shape[2] || khat.shape[2] != expected_n_h) {
        throw std::invalid_argument("inconsistent half-spectrum z-reduced R-dependent contraction shapes");
    }
    if (max_cutoff >= n_phi_ss / 2) {
        throw std::invalid_argument("half-spectrum z-reduced contraction requires max_cutoff < n_phi / 2");
    }
}

void validate_sparse_rz_shapes(
    const py::buffer_info& active_e,
    const py::buffer_info& active_r,
    const py::buffer_info& active_z,
    const py::buffer_info& active_hhat,
    const py::buffer_info& z_phase,
    const py::buffer_info& khat,
    const py::buffer_info& form_factors
) {
    if (active_e.ndim != 1 || active_r.ndim != 1 || active_z.ndim != 1) {
        throw std::invalid_argument("active index arrays must be one-dimensional");
    }
    if (active_hhat.ndim != 2) {
        throw std::invalid_argument("active_hhat must have shape (n_active, n_h)");
    }
    if (z_phase.ndim != 2) {
        throw std::invalid_argument("z_phase must have shape (n_q, n_z)");
    }
    if (khat.ndim != 3) {
        throw std::invalid_argument("khat must have shape (n_q, n_r, n_h)");
    }
    if (form_factors.ndim != 2) {
        throw std::invalid_argument("form_factors must have shape (n_elements, n_q)");
    }
    if (active_e.shape[0] != active_r.shape[0] || active_e.shape[0] != active_z.shape[0] ||
        active_e.shape[0] != active_hhat.shape[0]) {
        throw std::invalid_argument("active index arrays and active_hhat length differ");
    }
    if (z_phase.shape[0] != khat.shape[0] || z_phase.shape[0] != form_factors.shape[1] ||
        active_hhat.shape[1] != khat.shape[2]) {
        throw std::invalid_argument("inconsistent sparse circular contraction shapes");
    }
}

void validate_sparse_flat_shapes(
    const py::buffer_info& active_e,
    const py::buffer_info& active_r,
    const py::buffer_info& active_z,
    const py::buffer_info& active_beta,
    const py::buffer_info& active_values,
    const py::buffer_info& twiddle,
    const py::buffer_info& z_phase,
    const py::buffer_info& khat,
    const py::buffer_info& form_factors
) {
    if (active_e.ndim != 1 || active_r.ndim != 1 || active_z.ndim != 1 ||
        active_beta.ndim != 1 || active_values.ndim != 1) {
        throw std::invalid_argument("active sparse-flat arrays must be one-dimensional");
    }
    if (twiddle.ndim != 2) {
        throw std::invalid_argument("twiddle must have shape (n_phi, n_h)");
    }
    if (z_phase.ndim != 2) {
        throw std::invalid_argument("z_phase must have shape (n_q, n_z)");
    }
    if (khat.ndim != 3) {
        throw std::invalid_argument("khat must have shape (n_q, n_r, n_h)");
    }
    if (form_factors.ndim != 2) {
        throw std::invalid_argument("form_factors must have shape (n_elements, n_q)");
    }
    if (active_e.shape[0] != active_r.shape[0] || active_e.shape[0] != active_z.shape[0] ||
        active_e.shape[0] != active_beta.shape[0] ||
        active_e.shape[0] != active_values.shape[0]) {
        throw std::invalid_argument("active sparse-flat arrays have different lengths");
    }
    if (z_phase.shape[0] != khat.shape[0] || z_phase.shape[0] != form_factors.shape[1] ||
        twiddle.shape[1] != khat.shape[2]) {
        throw std::invalid_argument("inconsistent sparse-flat circular contraction shapes");
    }
}

void validate_sparse_profile_shapes(
    const py::buffer_info& profile_e,
    const py::buffer_info& profile_r,
    const py::buffer_info& profile_z,
    const py::buffer_info& profile_starts,
    const py::buffer_info& profile_counts,
    const py::buffer_info& active_beta,
    const py::buffer_info& active_values,
    const py::buffer_info& twiddle,
    const py::buffer_info& z_phase,
    const py::buffer_info& khat,
    const py::buffer_info& form_factors
) {
    if (profile_e.ndim != 1 || profile_r.ndim != 1 || profile_z.ndim != 1 ||
        profile_starts.ndim != 1 || profile_counts.ndim != 1 ||
        active_beta.ndim != 1 || active_values.ndim != 1) {
        throw std::invalid_argument("sparse profile arrays must be one-dimensional");
    }
    if (twiddle.ndim != 2) {
        throw std::invalid_argument("twiddle must have shape (n_phi, n_h)");
    }
    if (z_phase.ndim != 2) {
        throw std::invalid_argument("z_phase must have shape (n_q, n_z)");
    }
    if (khat.ndim != 3) {
        throw std::invalid_argument("khat must have shape (n_q, n_r, n_h)");
    }
    if (form_factors.ndim != 2) {
        throw std::invalid_argument("form_factors must have shape (n_elements, n_q)");
    }
    if (profile_e.shape[0] != profile_r.shape[0] ||
        profile_e.shape[0] != profile_z.shape[0] ||
        profile_e.shape[0] != profile_starts.shape[0] ||
        profile_e.shape[0] != profile_counts.shape[0]) {
        throw std::invalid_argument("sparse profile arrays have different lengths");
    }
    if (active_beta.shape[0] != active_values.shape[0]) {
        throw std::invalid_argument("active beta and value arrays have different lengths");
    }
    if (z_phase.shape[0] != khat.shape[0] || z_phase.shape[0] != form_factors.shape[1] ||
        twiddle.shape[1] != khat.shape[2]) {
        throw std::invalid_argument("inconsistent sparse-profile circular contraction shapes");
    }

    const auto* starts = static_cast<const std::int64_t*>(profile_starts.ptr);
    const auto* counts = static_cast<const std::int64_t*>(profile_counts.ptr);
    const py::ssize_t n_profiles = profile_starts.shape[0];
    const py::ssize_t n_active = active_values.shape[0];
    for (py::ssize_t p = 0; p < n_profiles; ++p) {
        if (starts[p] < 0 || counts[p] < 0 || starts[p] + counts[p] > n_active) {
            throw std::invalid_argument("sparse profile start/count is out of range");
        }
    }
}

void validate_sparse_source_projection_shapes(
    const py::buffer_info& profile_starts,
    const py::buffer_info& profile_counts,
    const py::buffer_info& active_z,
    const py::buffer_info& active_beta,
    const py::buffer_info& active_values,
    const py::buffer_info& z_phase,
    std::int64_t n_phi
) {
    if (profile_starts.ndim != 1 || profile_counts.ndim != 1 ||
        active_z.ndim != 1 || active_beta.ndim != 1 || active_values.ndim != 1) {
        throw std::invalid_argument("sparse source-projection arrays must be one-dimensional");
    }
    if (z_phase.ndim != 2) {
        throw std::invalid_argument("z_phase must have shape (n_q, n_z)");
    }
    if (n_phi <= 0) {
        throw std::invalid_argument("n_phi must be positive");
    }
    if (profile_starts.shape[0] != profile_counts.shape[0]) {
        throw std::invalid_argument("profile starts and counts have different lengths");
    }
    if (active_z.shape[0] != active_beta.shape[0] ||
        active_z.shape[0] != active_values.shape[0]) {
        throw std::invalid_argument("active z, beta, and value arrays have different lengths");
    }

    const auto* starts = static_cast<const std::int64_t*>(profile_starts.ptr);
    const auto* counts = static_cast<const std::int64_t*>(profile_counts.ptr);
    const auto* z = static_cast<const std::int64_t*>(active_z.ptr);
    const auto* beta = static_cast<const std::int64_t*>(active_beta.ptr);
    const py::ssize_t n_profiles = profile_starts.shape[0];
    const py::ssize_t n_active = active_values.shape[0];
    const py::ssize_t n_z = z_phase.shape[1];
    const py::ssize_t n_phi_ss = static_cast<py::ssize_t>(n_phi);
    for (py::ssize_t p = 0; p < n_profiles; ++p) {
        if (starts[p] < 0 || counts[p] < 0 || starts[p] + counts[p] > n_active) {
            throw std::invalid_argument("sparse source-projection start/count is out of range");
        }
        for (py::ssize_t j = starts[p]; j < starts[p] + counts[p]; ++j) {
            if (z[j] < 0 || z[j] >= n_z) {
                throw std::invalid_argument("active z index is out of range");
            }
            if (beta[j] < 0 || beta[j] >= n_phi_ss) {
                throw std::invalid_argument("active beta index is out of range");
            }
        }
    }
}

template <typename ComplexT>
void fused_worker(
    const ComplexT* hhat,
    const ComplexT* z_phase,
    const ComplexT* khat,
    const ComplexT* form_factors,
    ComplexT* out,
    py::ssize_t n_elements,
    py::ssize_t n_q,
    py::ssize_t n_r,
    py::ssize_t n_z,
    py::ssize_t n_phi,
    py::ssize_t begin,
    py::ssize_t end
) {
    std::vector<ComplexT> zsum(static_cast<std::size_t>(n_phi));

    for (py::ssize_t b = begin; b < end; ++b) {
        ComplexT* out_row = out + b * n_phi;
        std::fill(out_row, out_row + n_phi, ComplexT{});

        for (py::ssize_t e = 0; e < n_elements; ++e) {
            const ComplexT ff = form_factors[e * n_q + b];
            if (ff == ComplexT{}) {
                continue;
            }
            for (py::ssize_t r = 0; r < n_r; ++r) {
                std::fill(zsum.begin(), zsum.end(), ComplexT{});

                for (py::ssize_t z = 0; z < n_z; ++z) {
                    const ComplexT phase = z_phase[b * n_z + z];
                    const ComplexT* hhat_row =
                        hhat + (((e * n_r + r) * n_z + z) * n_phi);
                    for (py::ssize_t h = 0; h < n_phi; ++h) {
                        zsum[static_cast<std::size_t>(h)] += phase * hhat_row[h];
                    }
                }

                const ComplexT* khat_row = khat + ((b * n_r + r) * n_phi);
                const ComplexT coeff = ff;
                for (py::ssize_t h = 0; h < n_phi; ++h) {
                    out_row[h] += coeff * zsum[static_cast<std::size_t>(h)] * khat_row[h];
                }
            }
        }
    }
}

template <typename ComplexT>
void z_reduced_worker(
    const ComplexT* z_reduced,
    const ComplexT* khat,
    const ComplexT* form_factors,
    ComplexT* out,
    py::ssize_t n_q,
    py::ssize_t n_elements,
    py::ssize_t n_r,
    py::ssize_t n_phi,
    py::ssize_t begin,
    py::ssize_t end
) {
    for (py::ssize_t b = begin; b < end; ++b) {
        ComplexT* out_row = out + b * n_phi;
        std::fill(out_row, out_row + n_phi, ComplexT{});

        for (py::ssize_t e = 0; e < n_elements; ++e) {
            const ComplexT ff = form_factors[e * n_q + b];
            if (ff == ComplexT{}) {
                continue;
            }
            for (py::ssize_t r = 0; r < n_r; ++r) {
                const ComplexT* z_row =
                    z_reduced + (((b * n_elements + e) * n_r + r) * n_phi);
                const ComplexT* khat_row = khat + ((b * n_r + r) * n_phi);
                for (py::ssize_t h = 0; h < n_phi; ++h) {
                    out_row[h] += ff * z_row[h] * khat_row[h];
                }
            }
        }
    }
}

template <typename ComplexT>
void ring_average_fused_worker(
    const ComplexT* hhat,
    const ComplexT* z_phase,
    const ComplexT* khat,
    const ComplexT* form_factors,
    double* out,
    py::ssize_t n_elements,
    py::ssize_t n_q,
    py::ssize_t n_r,
    py::ssize_t n_z,
    py::ssize_t n_phi,
    py::ssize_t begin,
    py::ssize_t end
) {
    std::vector<ComplexT> ahat(static_cast<std::size_t>(n_phi));
    std::vector<ComplexT> zsum(static_cast<std::size_t>(n_phi));
    const double norm = static_cast<double>(n_phi) * static_cast<double>(n_phi);

    for (py::ssize_t b = begin; b < end; ++b) {
        std::fill(ahat.begin(), ahat.end(), ComplexT{});

        for (py::ssize_t e = 0; e < n_elements; ++e) {
            const ComplexT ff = form_factors[e * n_q + b];
            if (ff == ComplexT{}) {
                continue;
            }
            for (py::ssize_t r = 0; r < n_r; ++r) {
                std::fill(zsum.begin(), zsum.end(), ComplexT{});

                for (py::ssize_t z = 0; z < n_z; ++z) {
                    const ComplexT phase = z_phase[b * n_z + z];
                    const ComplexT* hhat_row =
                        hhat + (((e * n_r + r) * n_z + z) * n_phi);
                    for (py::ssize_t h = 0; h < n_phi; ++h) {
                        zsum[static_cast<std::size_t>(h)] += phase * hhat_row[h];
                    }
                }

                const ComplexT* khat_row = khat + ((b * n_r + r) * n_phi);
                for (py::ssize_t h = 0; h < n_phi; ++h) {
                    ahat[static_cast<std::size_t>(h)] +=
                        ff * zsum[static_cast<std::size_t>(h)] * khat_row[h];
                }
            }
        }

        double total = 0.0;
        for (py::ssize_t h = 0; h < n_phi; ++h) {
            total += static_cast<double>(std::norm(ahat[static_cast<std::size_t>(h)]));
        }
        out[b] = total / norm;
    }
}

template <typename ComplexT>
void ring_average_r_dependent_worker(
    const ComplexT* hhat,
    const ComplexT* z_phase,
    const ComplexT* khat,
    const ComplexT* form_factors,
    const std::int64_t* cutoffs,
    double* out,
    py::ssize_t n_elements,
    py::ssize_t n_q,
    py::ssize_t n_r,
    py::ssize_t n_z,
    py::ssize_t n_phi,
    py::ssize_t begin,
    py::ssize_t end
) {
    std::vector<ComplexT> ahat(static_cast<std::size_t>(n_phi));
    std::vector<ComplexT> zsum(static_cast<std::size_t>(n_phi));
    const double norm = static_cast<double>(n_phi) * static_cast<double>(n_phi);
    const py::ssize_t n_half = n_phi / 2;

    for (py::ssize_t b = begin; b < end; ++b) {
        std::fill(ahat.begin(), ahat.end(), ComplexT{});

        for (py::ssize_t e = 0; e < n_elements; ++e) {
            const ComplexT ff = form_factors[e * n_q + b];
            if (ff == ComplexT{}) {
                continue;
            }
            for (py::ssize_t r = 0; r < n_r; ++r) {
                py::ssize_t cutoff = static_cast<py::ssize_t>(cutoffs[b * n_r + r]);
                if (cutoff < 0) {
                    continue;
                }
                if (cutoff >= n_half) {
                    cutoff = n_half;
                }

                const ComplexT* khat_row = khat + ((b * n_r + r) * n_phi);

                if (cutoff >= n_half) {
                    std::fill(zsum.begin(), zsum.end(), ComplexT{});
                    for (py::ssize_t z = 0; z < n_z; ++z) {
                        const ComplexT phase = z_phase[b * n_z + z];
                        const ComplexT* hhat_row =
                            hhat + (((e * n_r + r) * n_z + z) * n_phi);
                        for (py::ssize_t h = 0; h < n_phi; ++h) {
                            zsum[static_cast<std::size_t>(h)] += phase * hhat_row[h];
                        }
                    }
                    for (py::ssize_t h = 0; h < n_phi; ++h) {
                        ahat[static_cast<std::size_t>(h)] +=
                            ff * zsum[static_cast<std::size_t>(h)] * khat_row[h];
                    }
                } else {
                    for (py::ssize_t h = 0; h <= cutoff; ++h) {
                        zsum[static_cast<std::size_t>(h)] = ComplexT{};
                    }
                    for (py::ssize_t h = n_phi - cutoff; h < n_phi; ++h) {
                        zsum[static_cast<std::size_t>(h)] = ComplexT{};
                    }
                    for (py::ssize_t z = 0; z < n_z; ++z) {
                        const ComplexT phase = z_phase[b * n_z + z];
                        const ComplexT* hhat_row =
                            hhat + (((e * n_r + r) * n_z + z) * n_phi);
                        for (py::ssize_t h = 0; h <= cutoff; ++h) {
                            zsum[static_cast<std::size_t>(h)] += phase * hhat_row[h];
                        }
                        for (py::ssize_t h = n_phi - cutoff; h < n_phi; ++h) {
                            zsum[static_cast<std::size_t>(h)] += phase * hhat_row[h];
                        }
                    }
                    for (py::ssize_t h = 0; h <= cutoff; ++h) {
                        ahat[static_cast<std::size_t>(h)] +=
                            ff * zsum[static_cast<std::size_t>(h)] * khat_row[h];
                    }
                    for (py::ssize_t h = n_phi - cutoff; h < n_phi; ++h) {
                        ahat[static_cast<std::size_t>(h)] +=
                            ff * zsum[static_cast<std::size_t>(h)] * khat_row[h];
                    }
                }
            }
        }

        double total = 0.0;
        for (py::ssize_t h = 0; h < n_phi; ++h) {
            total += static_cast<double>(std::norm(ahat[static_cast<std::size_t>(h)]));
        }
        out[b] = total / norm;
    }
}

template <typename ComplexT>
void r_dependent_worker(
    const ComplexT* hhat,
    const ComplexT* z_phase,
    const ComplexT* khat,
    const ComplexT* form_factors,
    const std::int64_t* cutoffs,
    ComplexT* out,
    py::ssize_t n_elements,
    py::ssize_t n_q,
    py::ssize_t n_r,
    py::ssize_t n_z,
    py::ssize_t n_phi,
    py::ssize_t begin,
    py::ssize_t end
) {
    std::vector<ComplexT> zsum(static_cast<std::size_t>(n_phi));
    const py::ssize_t n_half = n_phi / 2;

    for (py::ssize_t b = begin; b < end; ++b) {
        ComplexT* out_row = out + b * n_phi;
        std::fill(out_row, out_row + n_phi, ComplexT{});

        for (py::ssize_t e = 0; e < n_elements; ++e) {
            const ComplexT ff = form_factors[e * n_q + b];
            if (ff == ComplexT{}) {
                continue;
            }
            for (py::ssize_t r = 0; r < n_r; ++r) {
                py::ssize_t cutoff = static_cast<py::ssize_t>(cutoffs[b * n_r + r]);
                if (cutoff < 0) {
                    continue;
                }
                if (cutoff >= n_half) {
                    cutoff = n_half;
                }

                const ComplexT* khat_row = khat + ((b * n_r + r) * n_phi);

                if (cutoff >= n_half) {
                    std::fill(zsum.begin(), zsum.end(), ComplexT{});
                    for (py::ssize_t z = 0; z < n_z; ++z) {
                        const ComplexT phase = z_phase[b * n_z + z];
                        const ComplexT* hhat_row =
                            hhat + (((e * n_r + r) * n_z + z) * n_phi);
                        for (py::ssize_t h = 0; h < n_phi; ++h) {
                            zsum[static_cast<std::size_t>(h)] += phase * hhat_row[h];
                        }
                    }
                    for (py::ssize_t h = 0; h < n_phi; ++h) {
                        out_row[h] +=
                            ff * zsum[static_cast<std::size_t>(h)] * khat_row[h];
                    }
                } else {
                    for (py::ssize_t h = 0; h <= cutoff; ++h) {
                        zsum[static_cast<std::size_t>(h)] = ComplexT{};
                    }
                    for (py::ssize_t h = n_phi - cutoff; h < n_phi; ++h) {
                        zsum[static_cast<std::size_t>(h)] = ComplexT{};
                    }
                    for (py::ssize_t z = 0; z < n_z; ++z) {
                        const ComplexT phase = z_phase[b * n_z + z];
                        const ComplexT* hhat_row =
                            hhat + (((e * n_r + r) * n_z + z) * n_phi);
                        for (py::ssize_t h = 0; h <= cutoff; ++h) {
                            zsum[static_cast<std::size_t>(h)] += phase * hhat_row[h];
                        }
                        for (py::ssize_t h = n_phi - cutoff; h < n_phi; ++h) {
                            zsum[static_cast<std::size_t>(h)] += phase * hhat_row[h];
                        }
                    }
                    for (py::ssize_t h = 0; h <= cutoff; ++h) {
                        out_row[h] +=
                            ff * zsum[static_cast<std::size_t>(h)] * khat_row[h];
                    }
                    for (py::ssize_t h = n_phi - cutoff; h < n_phi; ++h) {
                        out_row[h] +=
                            ff * zsum[static_cast<std::size_t>(h)] * khat_row[h];
                    }
                }
            }
        }
    }
}

template <typename ComplexT>
void r_dependent_modes_worker(
    const ComplexT* hhat,
    const ComplexT* z_phase,
    const ComplexT* khat,
    const ComplexT* form_factors,
    const std::int64_t* cutoffs,
    ComplexT* out,
    py::ssize_t n_elements,
    py::ssize_t n_q,
    py::ssize_t n_r,
    py::ssize_t n_z,
    py::ssize_t n_phi,
    py::ssize_t max_cutoff,
    py::ssize_t begin,
    py::ssize_t end
) {
    const py::ssize_t n_h = max_cutoff + 1;
    std::vector<ComplexT> zsum_pos(static_cast<std::size_t>(max_cutoff + 1));
    std::vector<ComplexT> zsum_neg(static_cast<std::size_t>(max_cutoff));

    for (py::ssize_t b = begin; b < end; ++b) {
        ComplexT* out_row = out + b * n_phi;
        std::fill(out_row, out_row + n_phi, ComplexT{});

        for (py::ssize_t e = 0; e < n_elements; ++e) {
            const ComplexT ff = form_factors[e * n_q + b];
            if (ff == ComplexT{}) {
                continue;
            }
            for (py::ssize_t r = 0; r < n_r; ++r) {
                py::ssize_t cutoff = static_cast<py::ssize_t>(cutoffs[b * n_r + r]);
                if (cutoff < 0) {
                    continue;
                }
                if (cutoff > max_cutoff) {
                    cutoff = max_cutoff;
                }

                const ComplexT* khat_row = khat + ((b * n_r + r) * n_h);
                for (py::ssize_t h = 0; h <= cutoff; ++h) {
                    zsum_pos[static_cast<std::size_t>(h)] = ComplexT{};
                }
                for (py::ssize_t t = 0; t < cutoff; ++t) {
                    zsum_neg[static_cast<std::size_t>(t)] = ComplexT{};
                }

                for (py::ssize_t z = 0; z < n_z; ++z) {
                    const ComplexT phase = z_phase[b * n_z + z];
                    const ComplexT* hhat_row =
                        hhat + (((e * n_r + r) * n_z + z) * n_phi);
                    for (py::ssize_t h = 0; h <= cutoff; ++h) {
                        zsum_pos[static_cast<std::size_t>(h)] += phase * hhat_row[h];
                    }
                    for (py::ssize_t t = 0; t < cutoff; ++t) {
                        const py::ssize_t full_h = n_phi - cutoff + t;
                        zsum_neg[static_cast<std::size_t>(t)] +=
                            phase * hhat_row[full_h];
                    }
                }

                for (py::ssize_t h = 0; h <= cutoff; ++h) {
                    out_row[h] +=
                        ff * zsum_pos[static_cast<std::size_t>(h)] * khat_row[h];
                }
                for (py::ssize_t t = 0; t < cutoff; ++t) {
                    const py::ssize_t full_h = n_phi - cutoff + t;
                    out_row[full_h] +=
                        ff * zsum_neg[static_cast<std::size_t>(t)] *
                        khat_row[cutoff - t];
                }
            }
        }
    }
}

template <typename ComplexT>
void fill_miller_kernel_row(
    double x,
    py::ssize_t n_phi,
    py::ssize_t max_cutoff,
    py::ssize_t extra_order,
    ComplexT* row,
    std::vector<double>& values
) {
    using ScalarT = typename ComplexT::value_type;
    const py::ssize_t n_h = max_cutoff + 1;
    constexpr double tiny = 1.0e-300;
    constexpr double threshold = 1.0e100;
    constexpr double inv_threshold = 1.0e-100;

    if (std::abs(x) < tiny) {
        std::fill(row, row + n_h, ComplexT{});
        row[0] = ComplexT{static_cast<ScalarT>(n_phi), static_cast<ScalarT>(0)};
        return;
    }

    py::ssize_t m = static_cast<py::ssize_t>(std::ceil(std::abs(x))) + extra_order;
    m = std::max(m, max_cutoff + extra_order);
    if (m < max_cutoff) {
        m = max_cutoff;
    }
    values.assign(static_cast<std::size_t>(m + 1), 0.0);

    double b_next = 0.0;
    double b_curr = 1.0;
    values[static_cast<std::size_t>(m)] = b_curr;
    for (py::ssize_t n = m; n > 0; --n) {
        const double b_prev = (2.0 * static_cast<double>(n) / x) * b_curr - b_next;
        values[static_cast<std::size_t>(n - 1)] = b_prev;
        b_next = b_curr;
        b_curr = b_prev;

        if (std::abs(b_curr) > threshold || std::abs(b_next) > threshold) {
            for (py::ssize_t k = n - 1; k <= m; ++k) {
                values[static_cast<std::size_t>(k)] *= inv_threshold;
            }
            b_curr *= inv_threshold;
            b_next *= inv_threshold;
        }
    }

    double denom = values[0];
    for (py::ssize_t n = 2; n <= m; n += 2) {
        denom += 2.0 * values[static_cast<std::size_t>(n)];
    }
    if (denom == 0.0 || !std::isfinite(denom)) {
        std::fill(row, row + n_h, ComplexT{});
        return;
    }
    const double scale = static_cast<double>(n_phi) / denom;

    for (py::ssize_t n = 0; n <= max_cutoff; ++n) {
        const double jn = values[static_cast<std::size_t>(n)] * scale;
        row[n] = static_cast<ScalarT>(jn) * i_power<ComplexT>(n);
    }
}

template <typename ComplexT>
void r_dependent_half_modes_worker(
    const ComplexT* hhat,
    const ComplexT* z_phase,
    const ComplexT* khat,
    const ComplexT* form_factors,
    const std::int64_t* cutoffs,
    ComplexT* out,
    py::ssize_t n_elements,
    py::ssize_t n_q,
    py::ssize_t n_r,
    py::ssize_t n_z,
    py::ssize_t n_phi,
    py::ssize_t n_hhat,
    py::ssize_t max_cutoff,
    py::ssize_t begin,
    py::ssize_t end
) {
    using ScalarT = typename ComplexT::value_type;
    const py::ssize_t n_h = max_cutoff + 1;
    std::vector<ScalarT> zsum_pos_re(static_cast<std::size_t>(max_cutoff + 1));
    std::vector<ScalarT> zsum_pos_im(static_cast<std::size_t>(max_cutoff + 1));
    std::vector<ScalarT> zsum_neg_re(static_cast<std::size_t>(max_cutoff + 1));
    std::vector<ScalarT> zsum_neg_im(static_cast<std::size_t>(max_cutoff + 1));
    ScalarT* const pos_re = zsum_pos_re.data();
    ScalarT* const pos_im = zsum_pos_im.data();
    ScalarT* const neg_re = zsum_neg_re.data();
    ScalarT* const neg_im = zsum_neg_im.data();

    for (py::ssize_t b = begin; b < end; ++b) {
        ComplexT* out_row = out + b * n_phi;
        std::fill(out_row, out_row + n_phi, ComplexT{});
        const ComplexT* phase_row = z_phase + b * n_z;

        for (py::ssize_t e = 0; e < n_elements; ++e) {
            const ComplexT ff = form_factors[e * n_q + b];
            if (ff == ComplexT{}) {
                continue;
            }
            for (py::ssize_t r = 0; r < n_r; ++r) {
                py::ssize_t cutoff = static_cast<py::ssize_t>(cutoffs[b * n_r + r]);
                if (cutoff < 0) {
                    continue;
                }
                if (cutoff > max_cutoff) {
                    cutoff = max_cutoff;
                }

                const ComplexT* khat_row = khat + ((b * n_r + r) * n_h);
                const std::size_t n_active_h = static_cast<std::size_t>(cutoff + 1);
                std::fill_n(pos_re, n_active_h, ScalarT{});
                std::fill_n(pos_im, n_active_h, ScalarT{});
                std::fill_n(neg_re, n_active_h, ScalarT{});
                std::fill_n(neg_im, n_active_h, ScalarT{});

                for (py::ssize_t z = 0; z < n_z; ++z) {
                    const ComplexT phase = phase_row[z];
                    const ScalarT pr = phase.real();
                    const ScalarT pi = phase.imag();
                    const ComplexT* hhat_row =
                        hhat + (((e * n_r + r) * n_z + z) * n_hhat);
                    const ComplexT h0 = hhat_row[0];
                    pos_re[0] += pr * h0.real() - pi * h0.imag();
                    pos_im[0] += pr * h0.imag() + pi * h0.real();
                    WAXS_IVDEP
                    for (py::ssize_t h = 1; h <= cutoff; ++h) {
                        const ComplexT value = hhat_row[h];
                        const ScalarT vr = value.real();
                        const ScalarT vi = value.imag();
                        const ScalarT pr_vr = pr * vr;
                        const ScalarT pi_vi = pi * vi;
                        const ScalarT pr_vi = pr * vi;
                        const ScalarT pi_vr = pi * vr;
                        pos_re[h] += pr_vr - pi_vi;
                        pos_im[h] += pr_vi + pi_vr;
                        neg_re[h] += pr_vr + pi_vi;
                        neg_im[h] += pi_vr - pr_vi;
                    }
                }

                out_row[0] += ff * ComplexT{pos_re[0], pos_im[0]} * khat_row[0];
                WAXS_IVDEP
                for (py::ssize_t h = 1; h <= cutoff; ++h) {
                    const ComplexT coeff = ff * khat_row[h];
                    out_row[h] += coeff * ComplexT{pos_re[h], pos_im[h]};
                    out_row[n_phi - h] += coeff * ComplexT{neg_re[h], neg_im[h]};
                }
            }
        }
    }
}

template <typename ComplexT>
void r_dependent_half_modes_miller_worker(
    const ComplexT* hhat,
    const ComplexT* z_phase,
    const double* q_perp,
    const double* r_centers,
    const ComplexT* form_factors,
    const std::int64_t* cutoffs,
    ComplexT* out,
    py::ssize_t n_elements,
    py::ssize_t n_q,
    py::ssize_t n_r,
    py::ssize_t n_z,
    py::ssize_t n_phi,
    py::ssize_t n_hhat,
    py::ssize_t max_cutoff,
    py::ssize_t extra_order,
    py::ssize_t begin,
    py::ssize_t end
) {
    using ScalarT = typename ComplexT::value_type;
    std::vector<ComplexT> kernel(static_cast<std::size_t>(max_cutoff + 1));
    std::vector<double> miller_values;
    std::vector<ScalarT> zsum_pos_re(static_cast<std::size_t>(max_cutoff + 1));
    std::vector<ScalarT> zsum_pos_im(static_cast<std::size_t>(max_cutoff + 1));
    std::vector<ScalarT> zsum_neg_re(static_cast<std::size_t>(max_cutoff + 1));
    std::vector<ScalarT> zsum_neg_im(static_cast<std::size_t>(max_cutoff + 1));
    ScalarT* const pos_re = zsum_pos_re.data();
    ScalarT* const pos_im = zsum_pos_im.data();
    ScalarT* const neg_re = zsum_neg_re.data();
    ScalarT* const neg_im = zsum_neg_im.data();

    for (py::ssize_t b = begin; b < end; ++b) {
        ComplexT* out_row = out + b * n_phi;
        std::fill(out_row, out_row + n_phi, ComplexT{});
        const ComplexT* phase_row = z_phase + b * n_z;

        for (py::ssize_t r = 0; r < n_r; ++r) {
            py::ssize_t cutoff = static_cast<py::ssize_t>(cutoffs[b * n_r + r]);
            if (cutoff < 0) {
                continue;
            }
            if (cutoff > max_cutoff) {
                cutoff = max_cutoff;
            }

            fill_miller_kernel_row<ComplexT>(
                q_perp[b] * r_centers[r],
                n_phi,
                max_cutoff,
                extra_order,
                kernel.data(),
                miller_values
            );
            const std::size_t n_active_h = static_cast<std::size_t>(cutoff + 1);

            for (py::ssize_t e = 0; e < n_elements; ++e) {
                const ComplexT ff = form_factors[e * n_q + b];
                if (ff == ComplexT{}) {
                    continue;
                }

                std::fill_n(pos_re, n_active_h, ScalarT{});
                std::fill_n(pos_im, n_active_h, ScalarT{});
                std::fill_n(neg_re, n_active_h, ScalarT{});
                std::fill_n(neg_im, n_active_h, ScalarT{});

                for (py::ssize_t z = 0; z < n_z; ++z) {
                    const ComplexT phase = phase_row[z];
                    const ScalarT pr = phase.real();
                    const ScalarT pi = phase.imag();
                    const ComplexT* hhat_row =
                        hhat + (((e * n_r + r) * n_z + z) * n_hhat);
                    const ComplexT h0 = hhat_row[0];
                    pos_re[0] += pr * h0.real() - pi * h0.imag();
                    pos_im[0] += pr * h0.imag() + pi * h0.real();
                    WAXS_IVDEP
                    for (py::ssize_t h = 1; h <= cutoff; ++h) {
                        const ComplexT value = hhat_row[h];
                        const ScalarT vr = value.real();
                        const ScalarT vi = value.imag();
                        const ScalarT pr_vr = pr * vr;
                        const ScalarT pi_vi = pi * vi;
                        const ScalarT pr_vi = pr * vi;
                        const ScalarT pi_vr = pi * vr;
                        pos_re[h] += pr_vr - pi_vi;
                        pos_im[h] += pr_vi + pi_vr;
                        neg_re[h] += pr_vr + pi_vi;
                        neg_im[h] += pi_vr - pr_vi;
                    }
                }

                out_row[0] += ff * ComplexT{pos_re[0], pos_im[0]} * kernel[0];
                WAXS_IVDEP
                for (py::ssize_t h = 1; h <= cutoff; ++h) {
                    const ComplexT coeff = ff * kernel[h];
                    out_row[h] += coeff * ComplexT{pos_re[h], pos_im[h]};
                    out_row[n_phi - h] += coeff * ComplexT{neg_re[h], neg_im[h]};
                }
            }
        }
    }
}

template <typename ComplexT>
void r_dependent_half_z_reduced_worker(
    const ComplexT* z_pos,
    const ComplexT* z_neg,
    const ComplexT* khat,
    const ComplexT* form_factors,
    const std::int64_t* cutoffs,
    ComplexT* out,
    py::ssize_t n_elements,
    py::ssize_t n_q,
    py::ssize_t n_r,
    py::ssize_t n_h,
    py::ssize_t n_phi,
    py::ssize_t max_cutoff,
    py::ssize_t begin,
    py::ssize_t end
) {
    for (py::ssize_t b = begin; b < end; ++b) {
        ComplexT* out_row = out + b * n_phi;
        std::fill(out_row, out_row + n_phi, ComplexT{});

        for (py::ssize_t e = 0; e < n_elements; ++e) {
            const ComplexT ff = form_factors[e * n_q + b];
            if (ff == ComplexT{}) {
                continue;
            }
            for (py::ssize_t r = 0; r < n_r; ++r) {
                py::ssize_t cutoff = static_cast<py::ssize_t>(cutoffs[b * n_r + r]);
                if (cutoff < 0) {
                    continue;
                }
                if (cutoff > max_cutoff) {
                    cutoff = max_cutoff;
                }

                const ComplexT* z_pos_row =
                    z_pos + (((b * n_elements + e) * n_r + r) * n_h);
                const ComplexT* z_neg_row =
                    z_neg + (((b * n_elements + e) * n_r + r) * n_h);
                const ComplexT* khat_row = khat + ((b * n_r + r) * n_h);
                out_row[0] += ff * z_pos_row[0] * khat_row[0];
                for (py::ssize_t h = 1; h <= cutoff; ++h) {
                    const ComplexT coeff = ff * khat_row[h];
                    out_row[h] += coeff * z_pos_row[h];
                    out_row[n_phi - h] += coeff * z_neg_row[h];
                }
            }
        }
    }
}

template <typename ComplexT>
void miller_kernel_worker(
    const double* q_perp,
    const double* r_centers,
    ComplexT* out,
    py::ssize_t n_q,
    py::ssize_t n_r,
    py::ssize_t n_phi,
    py::ssize_t max_cutoff,
    py::ssize_t extra_order,
    py::ssize_t begin,
    py::ssize_t end
) {
    using ScalarT = typename ComplexT::value_type;
    const py::ssize_t n_h = max_cutoff + 1;
    constexpr double tiny = 1.0e-300;
    constexpr double threshold = 1.0e100;
    constexpr double inv_threshold = 1.0e-100;

    std::vector<double> values;

    for (py::ssize_t item = begin; item < end; ++item) {
        const py::ssize_t b = item / n_r;
        const py::ssize_t r = item - b * n_r;
        const double x = q_perp[b] * r_centers[r];
        ComplexT* row = out + (b * n_r + r) * n_h;

        if (std::abs(x) < tiny) {
            std::fill(row, row + n_h, ComplexT{});
            row[0] = ComplexT{static_cast<ScalarT>(n_phi), static_cast<ScalarT>(0)};
            continue;
        }

        py::ssize_t m = static_cast<py::ssize_t>(std::ceil(std::abs(x))) + extra_order;
        m = std::max(m, max_cutoff + extra_order);
        if (m < max_cutoff) {
            m = max_cutoff;
        }
        values.assign(static_cast<std::size_t>(m + 1), 0.0);

        double b_next = 0.0;
        double b_curr = 1.0;
        values[static_cast<std::size_t>(m)] = b_curr;
        for (py::ssize_t n = m; n > 0; --n) {
            const double b_prev = (2.0 * static_cast<double>(n) / x) * b_curr - b_next;
            values[static_cast<std::size_t>(n - 1)] = b_prev;
            b_next = b_curr;
            b_curr = b_prev;

            if (std::abs(b_curr) > threshold || std::abs(b_next) > threshold) {
                for (py::ssize_t k = n - 1; k <= m; ++k) {
                    values[static_cast<std::size_t>(k)] *= inv_threshold;
                }
                b_curr *= inv_threshold;
                b_next *= inv_threshold;
            }
        }

        double denom = values[0];
        for (py::ssize_t n = 2; n <= m; n += 2) {
            denom += 2.0 * values[static_cast<std::size_t>(n)];
        }
        if (denom == 0.0 || !std::isfinite(denom)) {
            std::fill(row, row + n_h, ComplexT{});
            continue;
        }
        const double scale = static_cast<double>(n_phi) / denom;

        for (py::ssize_t n = 0; n <= max_cutoff; ++n) {
            const double jn = values[static_cast<std::size_t>(n)] * scale;
            const ComplexT coeff =
                static_cast<ScalarT>(jn) * i_power<ComplexT>(n);
            row[n] = coeff;
        }
    }
}

template <typename ComplexT>
void table_kernel_worker(
    const double* q_perp,
    const double* r_centers,
    const double* table,
    ComplexT* out,
    py::ssize_t n_q,
    py::ssize_t n_r,
    py::ssize_t n_phi,
    py::ssize_t max_cutoff,
    py::ssize_t n_x,
    double dx,
    py::ssize_t begin,
    py::ssize_t end
) {
    using ScalarT = typename ComplexT::value_type;
    const py::ssize_t n_h = max_cutoff + 1;
    const double last_scaled = static_cast<double>(n_x - 1);
    const double dx_scale = dx;

    for (py::ssize_t item = begin; item < end; ++item) {
        const py::ssize_t b = item / n_r;
        const py::ssize_t r = item - b * n_r;
        const double x = q_perp[b] * r_centers[r];
        ComplexT* row = out + (b * n_r + r) * n_h;

        double scaled = x / dx;
        if (scaled <= 0.0) {
            scaled = 0.0;
        } else if (scaled >= last_scaled) {
            scaled = last_scaled;
        }
        py::ssize_t ix = static_cast<py::ssize_t>(std::floor(scaled));
        double t = scaled - static_cast<double>(ix);
        if (ix >= n_x - 1) {
            ix = n_x - 2;
            t = 1.0;
        }

        const double t2 = t * t;
        const double t3 = t2 * t;
        const double h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        const double h10 = t3 - 2.0 * t2 + t;
        const double h01 = -2.0 * t3 + 3.0 * t2;
        const double h11 = t3 - t2;

        for (py::ssize_t h = 0; h <= max_cutoff; ++h) {
            const double y0 = table[h * n_x + ix];
            const double y1 = table[h * n_x + ix + 1];
            const double d0 =
                (h == 0)
                    ? -table[n_x + ix]
                    : 0.5 * (table[(h - 1) * n_x + ix] - table[(h + 1) * n_x + ix]);
            const double d1 =
                (h == 0)
                    ? -table[n_x + ix + 1]
                    : 0.5 * (table[(h - 1) * n_x + ix + 1] -
                             table[(h + 1) * n_x + ix + 1]);
            const double jh = h00 * y0 + h10 * dx_scale * d0 + h01 * y1 +
                              h11 * dx_scale * d1;
            const ComplexT coeff =
                static_cast<ScalarT>(static_cast<double>(n_phi) * jh) *
                i_power<ComplexT>(h);
            row[h] = coeff;
        }
    }
}

template <typename ComplexT>
void sparse_profile_worker(
    const std::int64_t* profile_e,
    const std::int64_t* profile_r,
    const std::int64_t* profile_z,
    const std::int64_t* profile_starts,
    const std::int64_t* profile_counts,
    const std::int64_t* active_beta,
    const ComplexT* active_values,
    const ComplexT* twiddle,
    const ComplexT* z_phase,
    const ComplexT* khat,
    const ComplexT* form_factors,
    ComplexT* out,
    py::ssize_t n_q,
    py::ssize_t n_profiles,
    py::ssize_t n_z,
    py::ssize_t n_r,
    py::ssize_t n_h,
    py::ssize_t begin,
    py::ssize_t end
) {
    std::vector<ComplexT> angular_sum(static_cast<std::size_t>(n_h));

    for (py::ssize_t b = begin; b < end; ++b) {
        ComplexT* out_row = out + b * n_h;
        std::fill(out_row, out_row + n_h, ComplexT{});

        for (py::ssize_t p = 0; p < n_profiles; ++p) {
            const auto e = static_cast<py::ssize_t>(profile_e[p]);
            const auto r = static_cast<py::ssize_t>(profile_r[p]);
            const auto z = static_cast<py::ssize_t>(profile_z[p]);
            const ComplexT coeff = z_phase[b * n_z + z] * form_factors[e * n_q + b];
            if (coeff == ComplexT{}) {
                continue;
            }

            const auto start = static_cast<py::ssize_t>(profile_starts[p]);
            const auto count = static_cast<py::ssize_t>(profile_counts[p]);
            const ComplexT* khat_row = khat + ((b * n_r + r) * n_h);
            if (count == 1) {
                const auto beta = static_cast<py::ssize_t>(active_beta[start]);
                const ComplexT value_coeff = coeff * active_values[start];
                const ComplexT* twiddle_row = twiddle + beta * n_h;
                for (py::ssize_t h = 0; h < n_h; ++h) {
                    out_row[h] += value_coeff * twiddle_row[h] * khat_row[h];
                }
                continue;
            }

            std::fill(angular_sum.begin(), angular_sum.end(), ComplexT{});
            for (py::ssize_t j = start; j < start + count; ++j) {
                const auto beta = static_cast<py::ssize_t>(active_beta[j]);
                const ComplexT value = active_values[j];
                const ComplexT* twiddle_row = twiddle + beta * n_h;
                for (py::ssize_t h = 0; h < n_h; ++h) {
                    angular_sum[static_cast<std::size_t>(h)] += value * twiddle_row[h];
                }
            }
            for (py::ssize_t h = 0; h < n_h; ++h) {
                out_row[h] +=
                    coeff * angular_sum[static_cast<std::size_t>(h)] * khat_row[h];
            }
        }
    }
}

template <typename ComplexT>
void sparse_profile_hhat_worker(
    const std::int64_t* profile_starts,
    const std::int64_t* profile_counts,
    const std::int64_t* active_beta,
    const ComplexT* active_values,
    const ComplexT* twiddle,
    ComplexT* out,
    py::ssize_t n_profiles,
    py::ssize_t n_h,
    py::ssize_t begin,
    py::ssize_t end
) {
    for (py::ssize_t p = begin; p < end; ++p) {
        ComplexT* out_row = out + p * n_h;
        std::fill(out_row, out_row + n_h, ComplexT{});
        const auto start = static_cast<py::ssize_t>(profile_starts[p]);
        const auto count = static_cast<py::ssize_t>(profile_counts[p]);
        for (py::ssize_t j = start; j < start + count; ++j) {
            const auto beta = static_cast<py::ssize_t>(active_beta[j]);
            const ComplexT value = active_values[j];
            const ComplexT* twiddle_row = twiddle + beta * n_h;
            for (py::ssize_t h = 0; h < n_h; ++h) {
                out_row[h] += value * twiddle_row[h];
            }
        }
    }
}

template <typename ComplexT>
void sparse_source_projection_worker(
    const std::int64_t* profile_starts,
    const std::int64_t* profile_counts,
    const std::int64_t* active_z,
    const std::int64_t* active_beta,
    const ComplexT* active_values,
    const ComplexT* z_phase,
    ComplexT* out,
    py::ssize_t n_q,
    py::ssize_t n_profiles,
    py::ssize_t n_z,
    py::ssize_t n_phi,
    py::ssize_t begin,
    py::ssize_t end
) {
    for (py::ssize_t b = begin; b < end; ++b) {
        for (py::ssize_t p = 0; p < n_profiles; ++p) {
            ComplexT* out_row = out + ((b * n_profiles + p) * n_phi);
            std::fill(out_row, out_row + n_phi, ComplexT{});

            const auto start = static_cast<py::ssize_t>(profile_starts[p]);
            const auto count = static_cast<py::ssize_t>(profile_counts[p]);
            for (py::ssize_t j = start; j < start + count; ++j) {
                const auto z = static_cast<py::ssize_t>(active_z[j]);
                const auto beta = static_cast<py::ssize_t>(active_beta[j]);
                out_row[beta] += active_values[j] * z_phase[b * n_z + z];
            }
        }
    }
}

template <typename ComplexT>
void sparse_source_r_dependent_contract_worker(
    const ComplexT* projected_hhat,
    const ComplexT* khat_unique,
    const ComplexT* form_factors,
    const std::int64_t* cutoffs,
    const std::int64_t* profile_r_inverse,
    const std::int64_t* h_abs,
    ComplexT* out,
    py::ssize_t n_q,
    py::ssize_t n_profiles,
    py::ssize_t n_unique_r,
    py::ssize_t n_h,
    py::ssize_t begin,
    py::ssize_t end
) {
    for (py::ssize_t b = begin; b < end; ++b) {
        ComplexT* out_row = out + b * n_h;
        std::fill(out_row, out_row + n_h, ComplexT{});

        for (py::ssize_t p = 0; p < n_profiles; ++p) {
            const ComplexT ff = form_factors[b * n_profiles + p];
            if (ff == ComplexT{}) {
                continue;
            }
            const auto cutoff = cutoffs[b * n_profiles + p];
            const ComplexT* projected_row =
                projected_hhat + ((b * n_profiles + p) * n_h);
            const auto r = static_cast<py::ssize_t>(profile_r_inverse[p]);
            const ComplexT* khat_row = khat_unique + ((b * n_unique_r + r) * n_h);
            for (py::ssize_t h = 0; h < n_h; ++h) {
                if (h_abs[h] <= cutoff) {
                    out_row[h] += ff * projected_row[h] * khat_row[h];
                }
            }
        }
    }
}

template <typename ComplexT>
void sparse_flat_worker(
    const std::int64_t* active_e,
    const std::int64_t* active_r,
    const std::int64_t* active_z,
    const std::int64_t* active_beta,
    const ComplexT* active_values,
    const ComplexT* twiddle,
    const ComplexT* z_phase,
    const ComplexT* khat,
    const ComplexT* form_factors,
    ComplexT* out,
    py::ssize_t n_q,
    py::ssize_t n_active,
    py::ssize_t n_z,
    py::ssize_t n_r,
    py::ssize_t n_h,
    py::ssize_t begin,
    py::ssize_t end
) {
    for (py::ssize_t b = begin; b < end; ++b) {
        ComplexT* out_row = out + b * n_h;
        std::fill(out_row, out_row + n_h, ComplexT{});

        for (py::ssize_t c = 0; c < n_active; ++c) {
            const auto e = static_cast<py::ssize_t>(active_e[c]);
            const auto r = static_cast<py::ssize_t>(active_r[c]);
            const auto z = static_cast<py::ssize_t>(active_z[c]);
            const auto beta = static_cast<py::ssize_t>(active_beta[c]);
            const ComplexT coeff =
                active_values[c] * z_phase[b * n_z + z] * form_factors[e * n_q + b];
            if (coeff == ComplexT{}) {
                continue;
            }
            const ComplexT* khat_row = khat + ((b * n_r + r) * n_h);
            const ComplexT* twiddle_row = twiddle + beta * n_h;
            for (py::ssize_t h = 0; h < n_h; ++h) {
                out_row[h] += coeff * twiddle_row[h] * khat_row[h];
            }
        }
    }
}

template <typename ComplexT>
void sparse_rz_worker(
    const std::int64_t* active_e,
    const std::int64_t* active_r,
    const std::int64_t* active_z,
    const ComplexT* active_hhat,
    const ComplexT* z_phase,
    const ComplexT* khat,
    const ComplexT* form_factors,
    ComplexT* out,
    py::ssize_t n_q,
    py::ssize_t n_active,
    py::ssize_t n_z,
    py::ssize_t n_r,
    py::ssize_t n_h,
    py::ssize_t begin,
    py::ssize_t end
) {
    for (py::ssize_t b = begin; b < end; ++b) {
        ComplexT* out_row = out + b * n_h;
        std::fill(out_row, out_row + n_h, ComplexT{});

        for (py::ssize_t c = 0; c < n_active; ++c) {
            const auto e = static_cast<py::ssize_t>(active_e[c]);
            const auto r = static_cast<py::ssize_t>(active_r[c]);
            const auto z = static_cast<py::ssize_t>(active_z[c]);
            const ComplexT coeff = z_phase[b * n_z + z] * form_factors[e * n_q + b];
            if (coeff == ComplexT{}) {
                continue;
            }
            const ComplexT* hhat_row = active_hhat + c * n_h;
            const ComplexT* khat_row = khat + ((b * n_r + r) * n_h);
            for (py::ssize_t h = 0; h < n_h; ++h) {
                out_row[h] += coeff * hhat_row[h] * khat_row[h];
            }
        }
    }
}

template <typename Worker>
void run_parallel(py::ssize_t n_q, py::ssize_t work_per_q, Worker worker) {
    const unsigned int n_threads = choose_thread_count(n_q, work_per_q);
    if (n_threads == 1) {
        worker(0, n_q);
        return;
    }

    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (unsigned int thread_id = 0; thread_id < n_threads; ++thread_id) {
        const py::ssize_t begin = (n_q * thread_id) / n_threads;
        const py::ssize_t end = (n_q * (thread_id + 1)) / n_threads;
        threads.emplace_back([=, &worker]() { worker(begin, end); });
    }
    for (auto& thread : threads) {
        thread.join();
    }
}

template <typename Worker>
void run_parallel_dynamic(py::ssize_t n_items, py::ssize_t work_per_item, Worker worker) {
    const unsigned int n_threads = choose_thread_count(n_items, work_per_item);
    if (n_threads == 1) {
        worker(0, n_items);
        return;
    }

    std::atomic<py::ssize_t> next{0};
    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (unsigned int thread_id = 0; thread_id < n_threads; ++thread_id) {
        threads.emplace_back([&]() {
            while (true) {
                const py::ssize_t begin = next.fetch_add(1);
                if (begin >= n_items) {
                    break;
                }
                worker(begin, begin + 1);
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
}

template <typename ComplexT>
TypedComplexArray<ComplexT> circular_contract_fused_impl(
    TypedComplexArray<ComplexT> hhat,
    TypedComplexArray<ComplexT> z_phase,
    TypedComplexArray<ComplexT> khat,
    TypedComplexArray<ComplexT> form_factors
) {
    const py::buffer_info hhat_info = hhat.request();
    const py::buffer_info z_phase_info = z_phase.request();
    const py::buffer_info khat_info = khat.request();
    const py::buffer_info ff_info = form_factors.request();
    validate_fused_shapes(hhat_info, z_phase_info, khat_info, ff_info);

    const py::ssize_t n_elements = hhat_info.shape[0];
    const py::ssize_t n_r = hhat_info.shape[1];
    const py::ssize_t n_z = hhat_info.shape[2];
    const py::ssize_t n_phi = hhat_info.shape[3];
    const py::ssize_t n_q = z_phase_info.shape[0];
    TypedComplexArray<ComplexT> out({n_q, n_phi});
    ComplexT* out_ptr = out.mutable_data();

    const auto work_per_q = n_elements * n_r * n_z * n_phi;
    {
        py::gil_scoped_release release;
        run_parallel(n_q, work_per_q, [&](py::ssize_t begin, py::ssize_t end) {
            fused_worker<ComplexT>(
                static_cast<const ComplexT*>(hhat_info.ptr),
                static_cast<const ComplexT*>(z_phase_info.ptr),
                static_cast<const ComplexT*>(khat_info.ptr),
                static_cast<const ComplexT*>(ff_info.ptr),
                out_ptr,
                n_elements,
                n_q,
                n_r,
                n_z,
                n_phi,
                begin,
                end
            );
        });
    }
    return out;
}

ComplexArray circular_contract_fused(
    ComplexArray hhat,
    ComplexArray z_phase,
    ComplexArray khat,
    ComplexArray form_factors
) {
    return circular_contract_fused_impl<Complex128>(hhat, z_phase, khat, form_factors);
}

Complex64Array circular_contract_fused64(
    Complex64Array hhat,
    Complex64Array z_phase,
    Complex64Array khat,
    Complex64Array form_factors
) {
    return circular_contract_fused_impl<Complex64>(hhat, z_phase, khat, form_factors);
}

template <typename ComplexT>
py::array_t<double> circular_ring_average_fused_impl(
    TypedComplexArray<ComplexT> hhat,
    TypedComplexArray<ComplexT> z_phase,
    TypedComplexArray<ComplexT> khat,
    TypedComplexArray<ComplexT> form_factors
) {
    const py::buffer_info hhat_info = hhat.request();
    const py::buffer_info z_phase_info = z_phase.request();
    const py::buffer_info khat_info = khat.request();
    const py::buffer_info ff_info = form_factors.request();
    validate_fused_shapes(hhat_info, z_phase_info, khat_info, ff_info);

    const py::ssize_t n_elements = hhat_info.shape[0];
    const py::ssize_t n_r = hhat_info.shape[1];
    const py::ssize_t n_z = hhat_info.shape[2];
    const py::ssize_t n_phi = hhat_info.shape[3];
    const py::ssize_t n_q = z_phase_info.shape[0];
    py::array_t<double, py::array::c_style> out({n_q});
    double* out_ptr = out.mutable_data();

    const auto work_per_q = n_elements * n_r * n_z * n_phi;
    {
        py::gil_scoped_release release;
        run_parallel(n_q, work_per_q, [&](py::ssize_t begin, py::ssize_t end) {
            ring_average_fused_worker<ComplexT>(
                static_cast<const ComplexT*>(hhat_info.ptr),
                static_cast<const ComplexT*>(z_phase_info.ptr),
                static_cast<const ComplexT*>(khat_info.ptr),
                static_cast<const ComplexT*>(ff_info.ptr),
                out_ptr,
                n_elements,
                n_q,
                n_r,
                n_z,
                n_phi,
                begin,
                end
            );
        });
    }
    return out;
}

py::array_t<double> circular_ring_average_fused(
    ComplexArray hhat,
    ComplexArray z_phase,
    ComplexArray khat,
    ComplexArray form_factors
) {
    return circular_ring_average_fused_impl<Complex128>(hhat, z_phase, khat, form_factors);
}

py::array_t<double> circular_ring_average_fused64(
    Complex64Array hhat,
    Complex64Array z_phase,
    Complex64Array khat,
    Complex64Array form_factors
) {
    return circular_ring_average_fused_impl<Complex64>(hhat, z_phase, khat, form_factors);
}

template <typename ComplexT>
py::array_t<double> circular_ring_average_r_dependent_impl(
    TypedComplexArray<ComplexT> hhat,
    TypedComplexArray<ComplexT> z_phase,
    TypedComplexArray<ComplexT> khat,
    TypedComplexArray<ComplexT> form_factors,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> cutoffs
) {
    const py::buffer_info hhat_info = hhat.request();
    const py::buffer_info z_phase_info = z_phase.request();
    const py::buffer_info khat_info = khat.request();
    const py::buffer_info ff_info = form_factors.request();
    const py::buffer_info cutoff_info = cutoffs.request();
    validate_ring_cutoff_shapes(
        hhat_info,
        z_phase_info,
        khat_info,
        ff_info,
        cutoff_info
    );

    const py::ssize_t n_elements = hhat_info.shape[0];
    const py::ssize_t n_r = hhat_info.shape[1];
    const py::ssize_t n_z = hhat_info.shape[2];
    const py::ssize_t n_phi = hhat_info.shape[3];
    const py::ssize_t n_q = z_phase_info.shape[0];
    py::array_t<double, py::array::c_style> out({n_q});
    double* out_ptr = out.mutable_data();

    const auto work_per_q = n_elements * n_r * n_z * n_phi;
    {
        py::gil_scoped_release release;
        run_parallel(n_q, work_per_q, [&](py::ssize_t begin, py::ssize_t end) {
            ring_average_r_dependent_worker<ComplexT>(
                static_cast<const ComplexT*>(hhat_info.ptr),
                static_cast<const ComplexT*>(z_phase_info.ptr),
                static_cast<const ComplexT*>(khat_info.ptr),
                static_cast<const ComplexT*>(ff_info.ptr),
                static_cast<const std::int64_t*>(cutoff_info.ptr),
                out_ptr,
                n_elements,
                n_q,
                n_r,
                n_z,
                n_phi,
                begin,
                end
            );
        });
    }
    return out;
}

py::array_t<double> circular_ring_average_r_dependent(
    ComplexArray hhat,
    ComplexArray z_phase,
    ComplexArray khat,
    ComplexArray form_factors,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> cutoffs
) {
    return circular_ring_average_r_dependent_impl<Complex128>(
        hhat,
        z_phase,
        khat,
        form_factors,
        cutoffs
    );
}

py::array_t<double> circular_ring_average_r_dependent64(
    Complex64Array hhat,
    Complex64Array z_phase,
    Complex64Array khat,
    Complex64Array form_factors,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> cutoffs
) {
    return circular_ring_average_r_dependent_impl<Complex64>(
        hhat,
        z_phase,
        khat,
        form_factors,
        cutoffs
    );
}

template <typename ComplexT>
TypedComplexArray<ComplexT> circular_contract_r_dependent_impl(
    TypedComplexArray<ComplexT> hhat,
    TypedComplexArray<ComplexT> z_phase,
    TypedComplexArray<ComplexT> khat,
    TypedComplexArray<ComplexT> form_factors,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> cutoffs
) {
    const py::buffer_info hhat_info = hhat.request();
    const py::buffer_info z_phase_info = z_phase.request();
    const py::buffer_info khat_info = khat.request();
    const py::buffer_info ff_info = form_factors.request();
    const py::buffer_info cutoff_info = cutoffs.request();
    validate_ring_cutoff_shapes(
        hhat_info,
        z_phase_info,
        khat_info,
        ff_info,
        cutoff_info
    );

    const py::ssize_t n_elements = hhat_info.shape[0];
    const py::ssize_t n_r = hhat_info.shape[1];
    const py::ssize_t n_z = hhat_info.shape[2];
    const py::ssize_t n_phi = hhat_info.shape[3];
    const py::ssize_t n_q = z_phase_info.shape[0];
    TypedComplexArray<ComplexT> out({n_q, n_phi});
    ComplexT* out_ptr = out.mutable_data();

    const auto work_per_q = n_elements * n_r * n_z * n_phi;
    {
        py::gil_scoped_release release;
        run_parallel(n_q, work_per_q, [&](py::ssize_t begin, py::ssize_t end) {
            r_dependent_worker<ComplexT>(
                static_cast<const ComplexT*>(hhat_info.ptr),
                static_cast<const ComplexT*>(z_phase_info.ptr),
                static_cast<const ComplexT*>(khat_info.ptr),
                static_cast<const ComplexT*>(ff_info.ptr),
                static_cast<const std::int64_t*>(cutoff_info.ptr),
                out_ptr,
                n_elements,
                n_q,
                n_r,
                n_z,
                n_phi,
                begin,
                end
            );
        });
    }
    return out;
}

ComplexArray circular_contract_r_dependent(
    ComplexArray hhat,
    ComplexArray z_phase,
    ComplexArray khat,
    ComplexArray form_factors,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> cutoffs
) {
    return circular_contract_r_dependent_impl<Complex128>(
        hhat,
        z_phase,
        khat,
        form_factors,
        cutoffs
    );
}

Complex64Array circular_contract_r_dependent64(
    Complex64Array hhat,
    Complex64Array z_phase,
    Complex64Array khat,
    Complex64Array form_factors,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> cutoffs
) {
    return circular_contract_r_dependent_impl<Complex64>(
        hhat,
        z_phase,
        khat,
        form_factors,
        cutoffs
    );
}

template <typename ComplexT>
TypedComplexArray<ComplexT> circular_contract_r_dependent_modes_impl(
    TypedComplexArray<ComplexT> hhat,
    TypedComplexArray<ComplexT> z_phase,
    TypedComplexArray<ComplexT> khat,
    TypedComplexArray<ComplexT> form_factors,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> cutoffs,
    std::int64_t max_cutoff
) {
    const py::buffer_info hhat_info = hhat.request();
    const py::buffer_info z_phase_info = z_phase.request();
    const py::buffer_info khat_info = khat.request();
    const py::buffer_info ff_info = form_factors.request();
    const py::buffer_info cutoff_info = cutoffs.request();
    validate_r_dependent_modes_shapes(
        hhat_info,
        z_phase_info,
        khat_info,
        ff_info,
        cutoff_info,
        max_cutoff
    );

    const py::ssize_t n_elements = hhat_info.shape[0];
    const py::ssize_t n_r = hhat_info.shape[1];
    const py::ssize_t n_z = hhat_info.shape[2];
    const py::ssize_t n_phi = hhat_info.shape[3];
    const py::ssize_t n_q = z_phase_info.shape[0];
    TypedComplexArray<ComplexT> out({n_q, n_phi});
    ComplexT* out_ptr = out.mutable_data();

    const auto work_per_q =
        n_elements * n_r * n_z * (2 * static_cast<py::ssize_t>(max_cutoff) + 1);
    {
        py::gil_scoped_release release;
        run_parallel_dynamic(n_q, work_per_q, [&](py::ssize_t begin, py::ssize_t end) {
            r_dependent_modes_worker<ComplexT>(
                static_cast<const ComplexT*>(hhat_info.ptr),
                static_cast<const ComplexT*>(z_phase_info.ptr),
                static_cast<const ComplexT*>(khat_info.ptr),
                static_cast<const ComplexT*>(ff_info.ptr),
                static_cast<const std::int64_t*>(cutoff_info.ptr),
                out_ptr,
                n_elements,
                n_q,
                n_r,
                n_z,
                n_phi,
                static_cast<py::ssize_t>(max_cutoff),
                begin,
                end
            );
        });
    }
    return out;
}

ComplexArray circular_contract_r_dependent_modes(
    ComplexArray hhat,
    ComplexArray z_phase,
    ComplexArray khat,
    ComplexArray form_factors,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> cutoffs,
    std::int64_t max_cutoff
) {
    return circular_contract_r_dependent_modes_impl<Complex128>(
        hhat,
        z_phase,
        khat,
        form_factors,
        cutoffs,
        max_cutoff
    );
}

Complex64Array circular_contract_r_dependent_modes64(
    Complex64Array hhat,
    Complex64Array z_phase,
    Complex64Array khat,
    Complex64Array form_factors,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> cutoffs,
    std::int64_t max_cutoff
) {
    return circular_contract_r_dependent_modes_impl<Complex64>(
        hhat,
        z_phase,
        khat,
        form_factors,
        cutoffs,
        max_cutoff
    );
}

template <typename ComplexT>
TypedComplexArray<ComplexT> circular_contract_r_dependent_half_modes_impl(
    TypedComplexArray<ComplexT> hhat,
    TypedComplexArray<ComplexT> z_phase,
    TypedComplexArray<ComplexT> khat,
    TypedComplexArray<ComplexT> form_factors,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> cutoffs,
    std::int64_t n_phi,
    std::int64_t max_cutoff
) {
    const py::buffer_info hhat_info = hhat.request();
    const py::buffer_info z_phase_info = z_phase.request();
    const py::buffer_info khat_info = khat.request();
    const py::buffer_info ff_info = form_factors.request();
    const py::buffer_info cutoff_info = cutoffs.request();
    validate_r_dependent_half_modes_shapes(
        hhat_info,
        z_phase_info,
        khat_info,
        ff_info,
        cutoff_info,
        n_phi,
        max_cutoff
    );

    const py::ssize_t n_elements = hhat_info.shape[0];
    const py::ssize_t n_r = hhat_info.shape[1];
    const py::ssize_t n_z = hhat_info.shape[2];
    const py::ssize_t n_hhat = hhat_info.shape[3];
    const py::ssize_t n_q = z_phase_info.shape[0];
    const py::ssize_t n_phi_ss = static_cast<py::ssize_t>(n_phi);
    TypedComplexArray<ComplexT> out({n_q, n_phi_ss});
    ComplexT* out_ptr = out.mutable_data();

    const auto work_per_q =
        n_elements * n_r * n_z * (2 * static_cast<py::ssize_t>(max_cutoff) + 1);
    {
        py::gil_scoped_release release;
        run_parallel_dynamic(n_q, work_per_q, [&](py::ssize_t begin, py::ssize_t end) {
            r_dependent_half_modes_worker<ComplexT>(
                static_cast<const ComplexT*>(hhat_info.ptr),
                static_cast<const ComplexT*>(z_phase_info.ptr),
                static_cast<const ComplexT*>(khat_info.ptr),
                static_cast<const ComplexT*>(ff_info.ptr),
                static_cast<const std::int64_t*>(cutoff_info.ptr),
                out_ptr,
                n_elements,
                n_q,
                n_r,
                n_z,
                n_phi_ss,
                n_hhat,
                static_cast<py::ssize_t>(max_cutoff),
                begin,
                end
            );
        });
    }
    return out;
}

ComplexArray circular_contract_r_dependent_half_modes(
    ComplexArray hhat,
    ComplexArray z_phase,
    ComplexArray khat,
    ComplexArray form_factors,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> cutoffs,
    std::int64_t n_phi,
    std::int64_t max_cutoff
) {
    return circular_contract_r_dependent_half_modes_impl<Complex128>(
        hhat,
        z_phase,
        khat,
        form_factors,
        cutoffs,
        n_phi,
        max_cutoff
    );
}

Complex64Array circular_contract_r_dependent_half_modes64(
    Complex64Array hhat,
    Complex64Array z_phase,
    Complex64Array khat,
    Complex64Array form_factors,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> cutoffs,
    std::int64_t n_phi,
    std::int64_t max_cutoff
) {
    return circular_contract_r_dependent_half_modes_impl<Complex64>(
        hhat,
        z_phase,
        khat,
        form_factors,
        cutoffs,
        n_phi,
        max_cutoff
    );
}

template <typename ComplexT>
TypedComplexArray<ComplexT> circular_contract_r_dependent_half_modes_miller_impl(
    TypedComplexArray<ComplexT> hhat,
    TypedComplexArray<ComplexT> z_phase,
    py::array_t<double, py::array::c_style | py::array::forcecast> q_perp,
    py::array_t<double, py::array::c_style | py::array::forcecast> r_centers,
    TypedComplexArray<ComplexT> form_factors,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> cutoffs,
    std::int64_t n_phi,
    std::int64_t max_cutoff,
    std::int64_t extra_order
) {
    const py::buffer_info hhat_info = hhat.request();
    const py::buffer_info z_phase_info = z_phase.request();
    const py::buffer_info q_info = q_perp.request();
    const py::buffer_info r_info = r_centers.request();
    const py::buffer_info ff_info = form_factors.request();
    const py::buffer_info cutoff_info = cutoffs.request();
    validate_r_dependent_half_miller_shapes(
        hhat_info,
        z_phase_info,
        q_info,
        r_info,
        ff_info,
        cutoff_info,
        n_phi,
        max_cutoff
    );
    if (extra_order < 0) {
        throw std::invalid_argument("extra_order must be non-negative");
    }

    const py::ssize_t n_elements = hhat_info.shape[0];
    const py::ssize_t n_r = hhat_info.shape[1];
    const py::ssize_t n_z = hhat_info.shape[2];
    const py::ssize_t n_hhat = hhat_info.shape[3];
    const py::ssize_t n_q = z_phase_info.shape[0];
    const py::ssize_t n_phi_ss = static_cast<py::ssize_t>(n_phi);
    TypedComplexArray<ComplexT> out({n_q, n_phi_ss});
    ComplexT* out_ptr = out.mutable_data();

    const auto work_per_q =
        n_r
        * (
            static_cast<py::ssize_t>(max_cutoff + extra_order + 1)
            + n_elements * n_z * (2 * static_cast<py::ssize_t>(max_cutoff) + 1)
        );
    {
        py::gil_scoped_release release;
        run_parallel_dynamic(n_q, work_per_q, [&](py::ssize_t begin, py::ssize_t end) {
            r_dependent_half_modes_miller_worker<ComplexT>(
                static_cast<const ComplexT*>(hhat_info.ptr),
                static_cast<const ComplexT*>(z_phase_info.ptr),
                static_cast<const double*>(q_info.ptr),
                static_cast<const double*>(r_info.ptr),
                static_cast<const ComplexT*>(ff_info.ptr),
                static_cast<const std::int64_t*>(cutoff_info.ptr),
                out_ptr,
                n_elements,
                n_q,
                n_r,
                n_z,
                n_phi_ss,
                n_hhat,
                static_cast<py::ssize_t>(max_cutoff),
                static_cast<py::ssize_t>(extra_order),
                begin,
                end
            );
        });
    }
    return out;
}

ComplexArray circular_contract_r_dependent_half_modes_miller(
    ComplexArray hhat,
    ComplexArray z_phase,
    py::array_t<double, py::array::c_style | py::array::forcecast> q_perp,
    py::array_t<double, py::array::c_style | py::array::forcecast> r_centers,
    ComplexArray form_factors,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> cutoffs,
    std::int64_t n_phi,
    std::int64_t max_cutoff,
    std::int64_t extra_order
) {
    return circular_contract_r_dependent_half_modes_miller_impl<Complex128>(
        hhat,
        z_phase,
        q_perp,
        r_centers,
        form_factors,
        cutoffs,
        n_phi,
        max_cutoff,
        extra_order
    );
}

Complex64Array circular_contract_r_dependent_half_modes_miller64(
    Complex64Array hhat,
    Complex64Array z_phase,
    py::array_t<double, py::array::c_style | py::array::forcecast> q_perp,
    py::array_t<double, py::array::c_style | py::array::forcecast> r_centers,
    Complex64Array form_factors,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> cutoffs,
    std::int64_t n_phi,
    std::int64_t max_cutoff,
    std::int64_t extra_order
) {
    return circular_contract_r_dependent_half_modes_miller_impl<Complex64>(
        hhat,
        z_phase,
        q_perp,
        r_centers,
        form_factors,
        cutoffs,
        n_phi,
        max_cutoff,
        extra_order
    );
}

template <typename ComplexT>
TypedComplexArray<ComplexT> circular_contract_r_dependent_half_z_reduced_impl(
    TypedComplexArray<ComplexT> z_pos,
    TypedComplexArray<ComplexT> z_neg,
    TypedComplexArray<ComplexT> khat,
    TypedComplexArray<ComplexT> form_factors,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> cutoffs,
    std::int64_t n_phi,
    std::int64_t max_cutoff
) {
    const py::buffer_info z_pos_info = z_pos.request();
    const py::buffer_info z_neg_info = z_neg.request();
    const py::buffer_info khat_info = khat.request();
    const py::buffer_info ff_info = form_factors.request();
    const py::buffer_info cutoff_info = cutoffs.request();
    validate_r_dependent_half_z_reduced_shapes(
        z_pos_info,
        z_neg_info,
        khat_info,
        ff_info,
        cutoff_info,
        n_phi,
        max_cutoff
    );

    const py::ssize_t n_q = z_pos_info.shape[0];
    const py::ssize_t n_elements = z_pos_info.shape[1];
    const py::ssize_t n_r = z_pos_info.shape[2];
    const py::ssize_t n_h = z_pos_info.shape[3];
    const py::ssize_t n_phi_ss = static_cast<py::ssize_t>(n_phi);
    TypedComplexArray<ComplexT> out({n_q, n_phi_ss});
    ComplexT* out_ptr = out.mutable_data();

    const auto work_per_q =
        n_elements * n_r * (2 * static_cast<py::ssize_t>(max_cutoff) + 1);
    {
        py::gil_scoped_release release;
        run_parallel_dynamic(n_q, work_per_q, [&](py::ssize_t begin, py::ssize_t end) {
            r_dependent_half_z_reduced_worker<ComplexT>(
                static_cast<const ComplexT*>(z_pos_info.ptr),
                static_cast<const ComplexT*>(z_neg_info.ptr),
                static_cast<const ComplexT*>(khat_info.ptr),
                static_cast<const ComplexT*>(ff_info.ptr),
                static_cast<const std::int64_t*>(cutoff_info.ptr),
                out_ptr,
                n_elements,
                n_q,
                n_r,
                n_h,
                n_phi_ss,
                static_cast<py::ssize_t>(max_cutoff),
                begin,
                end
            );
        });
    }
    return out;
}

ComplexArray circular_contract_r_dependent_half_z_reduced(
    ComplexArray z_pos,
    ComplexArray z_neg,
    ComplexArray khat,
    ComplexArray form_factors,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> cutoffs,
    std::int64_t n_phi,
    std::int64_t max_cutoff
) {
    return circular_contract_r_dependent_half_z_reduced_impl<Complex128>(
        z_pos,
        z_neg,
        khat,
        form_factors,
        cutoffs,
        n_phi,
        max_cutoff
    );
}

Complex64Array circular_contract_r_dependent_half_z_reduced64(
    Complex64Array z_pos,
    Complex64Array z_neg,
    Complex64Array khat,
    Complex64Array form_factors,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> cutoffs,
    std::int64_t n_phi,
    std::int64_t max_cutoff
) {
    return circular_contract_r_dependent_half_z_reduced_impl<Complex64>(
        z_pos,
        z_neg,
        khat,
        form_factors,
        cutoffs,
        n_phi,
        max_cutoff
    );
}

template <typename ComplexT>
TypedComplexArray<ComplexT> analytic_kernel_hat_modes_miller_impl(
    py::array_t<double, py::array::c_style | py::array::forcecast> q_perp,
    py::array_t<double, py::array::c_style | py::array::forcecast> r_centers,
    std::int64_t n_phi,
    std::int64_t max_cutoff,
    std::int64_t extra_order
) {
    const py::buffer_info q_info = q_perp.request();
    const py::buffer_info r_info = r_centers.request();
    if (q_info.ndim != 1 || r_info.ndim != 1) {
        throw std::invalid_argument("q_perp and r_centers must be one-dimensional");
    }
    if (n_phi <= 0) {
        throw std::invalid_argument("n_phi must be positive");
    }
    if (max_cutoff < 0 || max_cutoff >= n_phi / 2) {
        throw std::invalid_argument("max_cutoff must satisfy 0 <= max_cutoff < n_phi / 2");
    }
    if (extra_order < 0) {
        throw std::invalid_argument("extra_order must be non-negative");
    }

    const py::ssize_t n_q = q_info.shape[0];
    const py::ssize_t n_r = r_info.shape[0];
    const py::ssize_t n_h = static_cast<py::ssize_t>(max_cutoff) + 1;
    TypedComplexArray<ComplexT> out({n_q, n_r, n_h});
    ComplexT* out_ptr = out.mutable_data();

    const py::ssize_t n_items = n_q * n_r;
    const auto work_per_item =
        std::max<py::ssize_t>(1, static_cast<py::ssize_t>(max_cutoff + extra_order));
    {
        py::gil_scoped_release release;
        run_parallel(n_items, work_per_item, [&](py::ssize_t begin, py::ssize_t end) {
            miller_kernel_worker<ComplexT>(
                static_cast<const double*>(q_info.ptr),
                static_cast<const double*>(r_info.ptr),
                out_ptr,
                n_q,
                n_r,
                static_cast<py::ssize_t>(n_phi),
                static_cast<py::ssize_t>(max_cutoff),
                static_cast<py::ssize_t>(extra_order),
                begin,
                end
            );
        });
    }
    return out;
}

ComplexArray analytic_kernel_hat_modes_miller(
    py::array_t<double, py::array::c_style | py::array::forcecast> q_perp,
    py::array_t<double, py::array::c_style | py::array::forcecast> r_centers,
    std::int64_t n_phi,
    std::int64_t max_cutoff,
    std::int64_t extra_order
) {
    return analytic_kernel_hat_modes_miller_impl<Complex128>(
        q_perp,
        r_centers,
        n_phi,
        max_cutoff,
        extra_order
    );
}

Complex64Array analytic_kernel_hat_modes_miller64(
    py::array_t<double, py::array::c_style | py::array::forcecast> q_perp,
    py::array_t<double, py::array::c_style | py::array::forcecast> r_centers,
    std::int64_t n_phi,
    std::int64_t max_cutoff,
    std::int64_t extra_order
) {
    return analytic_kernel_hat_modes_miller_impl<Complex64>(
        q_perp,
        r_centers,
        n_phi,
        max_cutoff,
        extra_order
    );
}

template <typename ComplexT>
TypedComplexArray<ComplexT> analytic_kernel_hat_modes_table_impl(
    py::array_t<double, py::array::c_style | py::array::forcecast> q_perp,
    py::array_t<double, py::array::c_style | py::array::forcecast> r_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> bessel_table,
    std::int64_t n_phi,
    std::int64_t max_cutoff,
    double dx
) {
    const py::buffer_info q_info = q_perp.request();
    const py::buffer_info r_info = r_centers.request();
    const py::buffer_info table_info = bessel_table.request();
    if (q_info.ndim != 1 || r_info.ndim != 1) {
        throw std::invalid_argument("q_perp and r_centers must be one-dimensional");
    }
    if (table_info.ndim != 2) {
        throw std::invalid_argument("bessel_table must have shape (n_orders, n_x)");
    }
    if (n_phi <= 0) {
        throw std::invalid_argument("n_phi must be positive");
    }
    if (max_cutoff < 0 || max_cutoff >= n_phi / 2) {
        throw std::invalid_argument("max_cutoff must satisfy 0 <= max_cutoff < n_phi / 2");
    }
    if (dx <= 0.0 || !std::isfinite(dx)) {
        throw std::invalid_argument("dx must be positive and finite");
    }

    const py::ssize_t n_orders = table_info.shape[0];
    const py::ssize_t n_x = table_info.shape[1];
    if (n_orders < static_cast<py::ssize_t>(max_cutoff) + 2) {
        throw std::invalid_argument("bessel_table must include orders 0..max_cutoff+1");
    }
    if (n_x < 2) {
        throw std::invalid_argument("bessel_table must contain at least two x samples");
    }

    const py::ssize_t n_q = q_info.shape[0];
    const py::ssize_t n_r = r_info.shape[0];
    const py::ssize_t n_h = static_cast<py::ssize_t>(max_cutoff) + 1;
    TypedComplexArray<ComplexT> out({n_q, n_r, n_h});
    ComplexT* out_ptr = out.mutable_data();

    const py::ssize_t n_items = n_q * n_r;
    const auto work_per_item =
        std::max<py::ssize_t>(1, static_cast<py::ssize_t>(max_cutoff));
    {
        py::gil_scoped_release release;
        run_parallel(n_items, work_per_item, [&](py::ssize_t begin, py::ssize_t end) {
            table_kernel_worker<ComplexT>(
                static_cast<const double*>(q_info.ptr),
                static_cast<const double*>(r_info.ptr),
                static_cast<const double*>(table_info.ptr),
                out_ptr,
                n_q,
                n_r,
                static_cast<py::ssize_t>(n_phi),
                static_cast<py::ssize_t>(max_cutoff),
                n_x,
                dx,
                begin,
                end
            );
        });
    }
    return out;
}

ComplexArray analytic_kernel_hat_modes_table(
    py::array_t<double, py::array::c_style | py::array::forcecast> q_perp,
    py::array_t<double, py::array::c_style | py::array::forcecast> r_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> bessel_table,
    std::int64_t n_phi,
    std::int64_t max_cutoff,
    double dx
) {
    return analytic_kernel_hat_modes_table_impl<Complex128>(
        q_perp,
        r_centers,
        bessel_table,
        n_phi,
        max_cutoff,
        dx
    );
}

Complex64Array analytic_kernel_hat_modes_table64(
    py::array_t<double, py::array::c_style | py::array::forcecast> q_perp,
    py::array_t<double, py::array::c_style | py::array::forcecast> r_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> bessel_table,
    std::int64_t n_phi,
    std::int64_t max_cutoff,
    double dx
) {
    return analytic_kernel_hat_modes_table_impl<Complex64>(
        q_perp,
        r_centers,
        bessel_table,
        n_phi,
        max_cutoff,
        dx
    );
}

template <typename ComplexT>
TypedComplexArray<ComplexT> circular_contract_z_reduced_impl(
    TypedComplexArray<ComplexT> z_reduced,
    TypedComplexArray<ComplexT> khat,
    TypedComplexArray<ComplexT> form_factors
) {
    const py::buffer_info z_reduced_info = z_reduced.request();
    const py::buffer_info khat_info = khat.request();
    const py::buffer_info ff_info = form_factors.request();
    validate_z_reduced_shapes(z_reduced_info, khat_info, ff_info);

    const py::ssize_t n_q = z_reduced_info.shape[0];
    const py::ssize_t n_elements = z_reduced_info.shape[1];
    const py::ssize_t n_r = z_reduced_info.shape[2];
    const py::ssize_t n_phi = z_reduced_info.shape[3];
    TypedComplexArray<ComplexT> out({n_q, n_phi});
    ComplexT* out_ptr = out.mutable_data();

    const auto work_per_q = n_elements * n_r * n_phi;
    {
        py::gil_scoped_release release;
        run_parallel(n_q, work_per_q, [&](py::ssize_t begin, py::ssize_t end) {
            z_reduced_worker<ComplexT>(
                static_cast<const ComplexT*>(z_reduced_info.ptr),
                static_cast<const ComplexT*>(khat_info.ptr),
                static_cast<const ComplexT*>(ff_info.ptr),
                out_ptr,
                n_q,
                n_elements,
                n_r,
                n_phi,
                begin,
                end
            );
        });
    }
    return out;
}

ComplexArray circular_contract_z_reduced(
    ComplexArray z_reduced,
    ComplexArray khat,
    ComplexArray form_factors
) {
    return circular_contract_z_reduced_impl<Complex128>(
        z_reduced,
        khat,
        form_factors
    );
}

Complex64Array circular_contract_z_reduced64(
    Complex64Array z_reduced,
    Complex64Array khat,
    Complex64Array form_factors
) {
    return circular_contract_z_reduced_impl<Complex64>(z_reduced, khat, form_factors);
}

template <typename ComplexT>
TypedComplexArray<ComplexT> circular_contract_sparse_rz_impl(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> active_e,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> active_r,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> active_z,
    TypedComplexArray<ComplexT> active_hhat,
    TypedComplexArray<ComplexT> z_phase,
    TypedComplexArray<ComplexT> khat,
    TypedComplexArray<ComplexT> form_factors
) {
    const py::buffer_info e_info = active_e.request();
    const py::buffer_info r_info = active_r.request();
    const py::buffer_info z_info = active_z.request();
    const py::buffer_info hhat_info = active_hhat.request();
    const py::buffer_info z_phase_info = z_phase.request();
    const py::buffer_info khat_info = khat.request();
    const py::buffer_info ff_info = form_factors.request();
    validate_sparse_rz_shapes(
        e_info,
        r_info,
        z_info,
        hhat_info,
        z_phase_info,
        khat_info,
        ff_info
    );

    const py::ssize_t n_active = hhat_info.shape[0];
    const py::ssize_t n_h = hhat_info.shape[1];
    const py::ssize_t n_q = z_phase_info.shape[0];
    const py::ssize_t n_z = z_phase_info.shape[1];
    const py::ssize_t n_r = khat_info.shape[1];
    TypedComplexArray<ComplexT> out({n_q, n_h});
    ComplexT* out_ptr = out.mutable_data();

    const auto work_per_q = n_active * n_h;
    {
        py::gil_scoped_release release;
        run_parallel(n_q, work_per_q, [&](py::ssize_t begin, py::ssize_t end) {
            sparse_rz_worker<ComplexT>(
                static_cast<const std::int64_t*>(e_info.ptr),
                static_cast<const std::int64_t*>(r_info.ptr),
                static_cast<const std::int64_t*>(z_info.ptr),
                static_cast<const ComplexT*>(hhat_info.ptr),
                static_cast<const ComplexT*>(z_phase_info.ptr),
                static_cast<const ComplexT*>(khat_info.ptr),
                static_cast<const ComplexT*>(ff_info.ptr),
                out_ptr,
                n_q,
                n_active,
                n_z,
                n_r,
                n_h,
                begin,
                end
            );
        });
    }
    return out;
}

ComplexArray circular_contract_sparse_rz(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> active_e,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> active_r,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> active_z,
    ComplexArray active_hhat,
    ComplexArray z_phase,
    ComplexArray khat,
    ComplexArray form_factors
) {
    return circular_contract_sparse_rz_impl<Complex128>(
        active_e,
        active_r,
        active_z,
        active_hhat,
        z_phase,
        khat,
        form_factors
    );
}

Complex64Array circular_contract_sparse_rz64(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> active_e,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> active_r,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> active_z,
    Complex64Array active_hhat,
    Complex64Array z_phase,
    Complex64Array khat,
    Complex64Array form_factors
) {
    return circular_contract_sparse_rz_impl<Complex64>(
        active_e,
        active_r,
        active_z,
        active_hhat,
        z_phase,
        khat,
        form_factors
    );
}

template <typename ComplexT>
TypedComplexArray<ComplexT> circular_contract_sparse_flat_impl(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> active_e,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> active_r,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> active_z,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> active_beta,
    TypedComplexArray<ComplexT> active_values,
    TypedComplexArray<ComplexT> twiddle,
    TypedComplexArray<ComplexT> z_phase,
    TypedComplexArray<ComplexT> khat,
    TypedComplexArray<ComplexT> form_factors
) {
    const py::buffer_info e_info = active_e.request();
    const py::buffer_info r_info = active_r.request();
    const py::buffer_info z_info = active_z.request();
    const py::buffer_info beta_info = active_beta.request();
    const py::buffer_info value_info = active_values.request();
    const py::buffer_info twiddle_info = twiddle.request();
    const py::buffer_info z_phase_info = z_phase.request();
    const py::buffer_info khat_info = khat.request();
    const py::buffer_info ff_info = form_factors.request();
    validate_sparse_flat_shapes(
        e_info,
        r_info,
        z_info,
        beta_info,
        value_info,
        twiddle_info,
        z_phase_info,
        khat_info,
        ff_info
    );

    const py::ssize_t n_active = value_info.shape[0];
    const py::ssize_t n_h = twiddle_info.shape[1];
    const py::ssize_t n_q = z_phase_info.shape[0];
    const py::ssize_t n_z = z_phase_info.shape[1];
    const py::ssize_t n_r = khat_info.shape[1];
    TypedComplexArray<ComplexT> out({n_q, n_h});
    ComplexT* out_ptr = out.mutable_data();

    const auto work_per_q = n_active * n_h;
    {
        py::gil_scoped_release release;
        run_parallel(n_q, work_per_q, [&](py::ssize_t begin, py::ssize_t end) {
            sparse_flat_worker<ComplexT>(
                static_cast<const std::int64_t*>(e_info.ptr),
                static_cast<const std::int64_t*>(r_info.ptr),
                static_cast<const std::int64_t*>(z_info.ptr),
                static_cast<const std::int64_t*>(beta_info.ptr),
                static_cast<const ComplexT*>(value_info.ptr),
                static_cast<const ComplexT*>(twiddle_info.ptr),
                static_cast<const ComplexT*>(z_phase_info.ptr),
                static_cast<const ComplexT*>(khat_info.ptr),
                static_cast<const ComplexT*>(ff_info.ptr),
                out_ptr,
                n_q,
                n_active,
                n_z,
                n_r,
                n_h,
                begin,
                end
            );
        });
    }
    return out;
}

ComplexArray circular_contract_sparse_flat(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> active_e,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> active_r,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> active_z,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> active_beta,
    ComplexArray active_values,
    ComplexArray twiddle,
    ComplexArray z_phase,
    ComplexArray khat,
    ComplexArray form_factors
) {
    return circular_contract_sparse_flat_impl<Complex128>(
        active_e,
        active_r,
        active_z,
        active_beta,
        active_values,
        twiddle,
        z_phase,
        khat,
        form_factors
    );
}

Complex64Array circular_contract_sparse_flat64(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> active_e,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> active_r,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> active_z,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> active_beta,
    Complex64Array active_values,
    Complex64Array twiddle,
    Complex64Array z_phase,
    Complex64Array khat,
    Complex64Array form_factors
) {
    return circular_contract_sparse_flat_impl<Complex64>(
        active_e,
        active_r,
        active_z,
        active_beta,
        active_values,
        twiddle,
        z_phase,
        khat,
        form_factors
    );
}

template <typename ComplexT>
TypedComplexArray<ComplexT> circular_contract_sparse_profiles_impl(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> profile_e,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> profile_r,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> profile_z,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> profile_starts,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> profile_counts,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> active_beta,
    TypedComplexArray<ComplexT> active_values,
    TypedComplexArray<ComplexT> twiddle,
    TypedComplexArray<ComplexT> z_phase,
    TypedComplexArray<ComplexT> khat,
    TypedComplexArray<ComplexT> form_factors
) {
    const py::buffer_info profile_e_info = profile_e.request();
    const py::buffer_info profile_r_info = profile_r.request();
    const py::buffer_info profile_z_info = profile_z.request();
    const py::buffer_info starts_info = profile_starts.request();
    const py::buffer_info counts_info = profile_counts.request();
    const py::buffer_info beta_info = active_beta.request();
    const py::buffer_info value_info = active_values.request();
    const py::buffer_info twiddle_info = twiddle.request();
    const py::buffer_info z_phase_info = z_phase.request();
    const py::buffer_info khat_info = khat.request();
    const py::buffer_info ff_info = form_factors.request();
    validate_sparse_profile_shapes(
        profile_e_info,
        profile_r_info,
        profile_z_info,
        starts_info,
        counts_info,
        beta_info,
        value_info,
        twiddle_info,
        z_phase_info,
        khat_info,
        ff_info
    );

    const py::ssize_t n_profiles = profile_e_info.shape[0];
    const py::ssize_t n_active = value_info.shape[0];
    const py::ssize_t n_h = twiddle_info.shape[1];
    const py::ssize_t n_q = z_phase_info.shape[0];
    const py::ssize_t n_z = z_phase_info.shape[1];
    const py::ssize_t n_r = khat_info.shape[1];
    TypedComplexArray<ComplexT> out({n_q, n_h});
    ComplexT* out_ptr = out.mutable_data();

    const auto work_per_q = (n_active + n_profiles) * n_h;
    {
        py::gil_scoped_release release;
        run_parallel(n_q, work_per_q, [&](py::ssize_t begin, py::ssize_t end) {
            sparse_profile_worker<ComplexT>(
                static_cast<const std::int64_t*>(profile_e_info.ptr),
                static_cast<const std::int64_t*>(profile_r_info.ptr),
                static_cast<const std::int64_t*>(profile_z_info.ptr),
                static_cast<const std::int64_t*>(starts_info.ptr),
                static_cast<const std::int64_t*>(counts_info.ptr),
                static_cast<const std::int64_t*>(beta_info.ptr),
                static_cast<const ComplexT*>(value_info.ptr),
                static_cast<const ComplexT*>(twiddle_info.ptr),
                static_cast<const ComplexT*>(z_phase_info.ptr),
                static_cast<const ComplexT*>(khat_info.ptr),
                static_cast<const ComplexT*>(ff_info.ptr),
                out_ptr,
                n_q,
                n_profiles,
                n_z,
                n_r,
                n_h,
                begin,
                end
            );
        });
    }
    return out;
}

ComplexArray circular_contract_sparse_profiles(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> profile_e,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> profile_r,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> profile_z,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> profile_starts,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> profile_counts,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> active_beta,
    ComplexArray active_values,
    ComplexArray twiddle,
    ComplexArray z_phase,
    ComplexArray khat,
    ComplexArray form_factors
) {
    return circular_contract_sparse_profiles_impl<Complex128>(
        profile_e,
        profile_r,
        profile_z,
        profile_starts,
        profile_counts,
        active_beta,
        active_values,
        twiddle,
        z_phase,
        khat,
        form_factors
    );
}

Complex64Array circular_contract_sparse_profiles64(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> profile_e,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> profile_r,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> profile_z,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> profile_starts,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> profile_counts,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> active_beta,
    Complex64Array active_values,
    Complex64Array twiddle,
    Complex64Array z_phase,
    Complex64Array khat,
    Complex64Array form_factors
) {
    return circular_contract_sparse_profiles_impl<Complex64>(
        profile_e,
        profile_r,
        profile_z,
        profile_starts,
        profile_counts,
        active_beta,
        active_values,
        twiddle,
        z_phase,
        khat,
        form_factors
    );
}

template <typename ComplexT>
TypedComplexArray<ComplexT> build_sparse_source_projection_impl(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> profile_starts,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> profile_counts,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> active_z,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> active_beta,
    TypedComplexArray<ComplexT> active_values,
    TypedComplexArray<ComplexT> z_phase,
    std::int64_t n_phi
) {
    const py::buffer_info starts_info = profile_starts.request();
    const py::buffer_info counts_info = profile_counts.request();
    const py::buffer_info z_info = active_z.request();
    const py::buffer_info beta_info = active_beta.request();
    const py::buffer_info value_info = active_values.request();
    const py::buffer_info z_phase_info = z_phase.request();
    validate_sparse_source_projection_shapes(
        starts_info,
        counts_info,
        z_info,
        beta_info,
        value_info,
        z_phase_info,
        n_phi
    );

    const py::ssize_t n_profiles = starts_info.shape[0];
    const py::ssize_t n_q = z_phase_info.shape[0];
    const py::ssize_t n_z = z_phase_info.shape[1];
    const py::ssize_t n_phi_ss = static_cast<py::ssize_t>(n_phi);
    TypedComplexArray<ComplexT> out({n_q, n_profiles, n_phi_ss});
    ComplexT* out_ptr = out.mutable_data();

    const auto work_per_q = n_profiles * n_phi_ss + value_info.shape[0];
    {
        py::gil_scoped_release release;
        run_parallel(n_q, work_per_q, [&](py::ssize_t begin, py::ssize_t end) {
            sparse_source_projection_worker<ComplexT>(
                static_cast<const std::int64_t*>(starts_info.ptr),
                static_cast<const std::int64_t*>(counts_info.ptr),
                static_cast<const std::int64_t*>(z_info.ptr),
                static_cast<const std::int64_t*>(beta_info.ptr),
                static_cast<const ComplexT*>(value_info.ptr),
                static_cast<const ComplexT*>(z_phase_info.ptr),
                out_ptr,
                n_q,
                n_profiles,
                n_z,
                n_phi_ss,
                begin,
                end
            );
        });
    }
    return out;
}

ComplexArray build_sparse_source_projection(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> profile_starts,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> profile_counts,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> active_z,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> active_beta,
    ComplexArray active_values,
    ComplexArray z_phase,
    std::int64_t n_phi
) {
    return build_sparse_source_projection_impl<Complex128>(
        profile_starts,
        profile_counts,
        active_z,
        active_beta,
        active_values,
        z_phase,
        n_phi
    );
}

Complex64Array build_sparse_source_projection64(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> profile_starts,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> profile_counts,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> active_z,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> active_beta,
    Complex64Array active_values,
    Complex64Array z_phase,
    std::int64_t n_phi
) {
    return build_sparse_source_projection_impl<Complex64>(
        profile_starts,
        profile_counts,
        active_z,
        active_beta,
        active_values,
        z_phase,
        n_phi
    );
}

template <typename ComplexT>
TypedComplexArray<ComplexT> sparse_source_r_dependent_contract_impl(
    TypedComplexArray<ComplexT> projected_hhat,
    TypedComplexArray<ComplexT> khat_unique,
    TypedComplexArray<ComplexT> form_factors,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> cutoffs,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> profile_r_inverse,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> h_abs
) {
    const py::buffer_info projected_info = projected_hhat.request();
    const py::buffer_info khat_info = khat_unique.request();
    const py::buffer_info ff_info = form_factors.request();
    const py::buffer_info cutoff_info = cutoffs.request();
    const py::buffer_info inverse_info = profile_r_inverse.request();
    const py::buffer_info h_abs_info = h_abs.request();

    if (projected_info.ndim != 3 || khat_info.ndim != 3) {
        throw std::invalid_argument(
            "projected_hhat must have shape (n_q, n_profiles, n_h) and "
            "khat_unique must have shape (n_q, n_unique_r, n_h)"
        );
    }
    if (ff_info.ndim != 2 || cutoff_info.ndim != 2) {
        throw std::invalid_argument(
            "form_factors and cutoffs must have shape (n_q, n_profiles)"
        );
    }
    if (inverse_info.ndim != 1 || h_abs_info.ndim != 1) {
        throw std::invalid_argument(
            "profile_r_inverse and h_abs must be one-dimensional"
        );
    }
    const py::ssize_t n_q = projected_info.shape[0];
    const py::ssize_t n_profiles = projected_info.shape[1];
    const py::ssize_t n_h = projected_info.shape[2];
    const py::ssize_t n_unique_r = khat_info.shape[1];
    if (khat_info.shape[0] != n_q || khat_info.shape[2] != n_h ||
        ff_info.shape[0] != n_q ||
        ff_info.shape[1] != n_profiles || cutoff_info.shape[0] != n_q ||
        cutoff_info.shape[1] != n_profiles || inverse_info.shape[0] != n_profiles ||
        h_abs_info.shape[0] != n_h) {
        throw std::invalid_argument(
            "inconsistent sparse source R-dependent contraction shapes"
        );
    }
    const auto* inverse_ptr = static_cast<const std::int64_t*>(inverse_info.ptr);
    for (py::ssize_t p = 0; p < n_profiles; ++p) {
        if (inverse_ptr[p] < 0 || inverse_ptr[p] >= n_unique_r) {
            throw std::invalid_argument("profile_r_inverse is out of range");
        }
    }

    TypedComplexArray<ComplexT> out({n_q, n_h});
    ComplexT* out_ptr = out.mutable_data();
    const auto work_per_q = n_profiles * n_h;
    {
        py::gil_scoped_release release;
        run_parallel_dynamic(n_q, work_per_q, [&](py::ssize_t begin, py::ssize_t end) {
            sparse_source_r_dependent_contract_worker<ComplexT>(
                static_cast<const ComplexT*>(projected_info.ptr),
                static_cast<const ComplexT*>(khat_info.ptr),
                static_cast<const ComplexT*>(ff_info.ptr),
                static_cast<const std::int64_t*>(cutoff_info.ptr),
                static_cast<const std::int64_t*>(inverse_info.ptr),
                static_cast<const std::int64_t*>(h_abs_info.ptr),
                out_ptr,
                n_q,
                n_profiles,
                n_unique_r,
                n_h,
                begin,
                end
            );
        });
    }
    return out;
}

ComplexArray sparse_source_r_dependent_contract(
    ComplexArray projected_hhat,
    ComplexArray khat_unique,
    ComplexArray form_factors,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> cutoffs,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> profile_r_inverse,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> h_abs
) {
    return sparse_source_r_dependent_contract_impl<Complex128>(
        projected_hhat,
        khat_unique,
        form_factors,
        cutoffs,
        profile_r_inverse,
        h_abs
    );
}

Complex64Array sparse_source_r_dependent_contract64(
    Complex64Array projected_hhat,
    Complex64Array khat_unique,
    Complex64Array form_factors,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> cutoffs,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> profile_r_inverse,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> h_abs
) {
    return sparse_source_r_dependent_contract_impl<Complex64>(
        projected_hhat,
        khat_unique,
        form_factors,
        cutoffs,
        profile_r_inverse,
        h_abs
    );
}

template <typename ComplexT>
TypedComplexArray<ComplexT> build_sparse_profile_hhat_impl(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> profile_starts,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> profile_counts,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> active_beta,
    TypedComplexArray<ComplexT> active_values,
    TypedComplexArray<ComplexT> twiddle
) {
    const py::buffer_info starts_info = profile_starts.request();
    const py::buffer_info counts_info = profile_counts.request();
    const py::buffer_info beta_info = active_beta.request();
    const py::buffer_info value_info = active_values.request();
    const py::buffer_info twiddle_info = twiddle.request();

    if (starts_info.ndim != 1 || counts_info.ndim != 1 ||
        beta_info.ndim != 1 || value_info.ndim != 1) {
        throw std::invalid_argument("sparse profile hhat inputs must be one-dimensional");
    }
    if (twiddle_info.ndim != 2) {
        throw std::invalid_argument("twiddle must have shape (n_phi, n_h)");
    }
    if (starts_info.shape[0] != counts_info.shape[0]) {
        throw std::invalid_argument("profile starts and counts have different lengths");
    }
    if (beta_info.shape[0] != value_info.shape[0]) {
        throw std::invalid_argument("active beta and value arrays have different lengths");
    }

    const auto* starts = static_cast<const std::int64_t*>(starts_info.ptr);
    const auto* counts = static_cast<const std::int64_t*>(counts_info.ptr);
    const py::ssize_t n_profiles = starts_info.shape[0];
    const py::ssize_t n_active = value_info.shape[0];
    for (py::ssize_t p = 0; p < n_profiles; ++p) {
        if (starts[p] < 0 || counts[p] < 0 || starts[p] + counts[p] > n_active) {
            throw std::invalid_argument("sparse profile start/count is out of range");
        }
    }

    const py::ssize_t n_h = twiddle_info.shape[1];
    TypedComplexArray<ComplexT> out({n_profiles, n_h});
    ComplexT* out_ptr = out.mutable_data();
    const py::ssize_t avg_count = n_profiles == 0 ? 0 : (n_active + n_profiles - 1) / n_profiles;
    const auto work_per_profile = std::max<py::ssize_t>(1, avg_count) * n_h;
    {
        py::gil_scoped_release release;
        run_parallel(n_profiles, work_per_profile, [&](py::ssize_t begin, py::ssize_t end) {
            sparse_profile_hhat_worker<ComplexT>(
                static_cast<const std::int64_t*>(starts_info.ptr),
                static_cast<const std::int64_t*>(counts_info.ptr),
                static_cast<const std::int64_t*>(beta_info.ptr),
                static_cast<const ComplexT*>(value_info.ptr),
                static_cast<const ComplexT*>(twiddle_info.ptr),
                out_ptr,
                n_profiles,
                n_h,
                begin,
                end
            );
        });
    }
    return out;
}

ComplexArray build_sparse_profile_hhat(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> profile_starts,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> profile_counts,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> active_beta,
    ComplexArray active_values,
    ComplexArray twiddle
) {
    return build_sparse_profile_hhat_impl<Complex128>(
        profile_starts,
        profile_counts,
        active_beta,
        active_values,
        twiddle
    );
}

Complex64Array build_sparse_profile_hhat64(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> profile_starts,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> profile_counts,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> active_beta,
    Complex64Array active_values,
    Complex64Array twiddle
) {
    return build_sparse_profile_hhat_impl<Complex64>(
        profile_starts,
        profile_counts,
        active_beta,
        active_values,
        twiddle
    );
}

}  // namespace

PYBIND11_MODULE(_cpp_solvers, m) {
    m.doc() = "C++ circular contraction kernels for waxs_cake";
    m.def(
        "circular_contract_fused",
        &circular_contract_fused,
        py::arg("hhat"),
        py::arg("z_phase"),
        py::arg("khat"),
        py::arg("form_factors")
    );
    m.def(
        "circular_contract_fused64",
        &circular_contract_fused64,
        py::arg("hhat"),
        py::arg("z_phase"),
        py::arg("khat"),
        py::arg("form_factors")
    );
    m.def(
        "circular_ring_average_fused",
        &circular_ring_average_fused,
        py::arg("hhat"),
        py::arg("z_phase"),
        py::arg("khat"),
        py::arg("form_factors")
    );
    m.def(
        "circular_ring_average_fused64",
        &circular_ring_average_fused64,
        py::arg("hhat"),
        py::arg("z_phase"),
        py::arg("khat"),
        py::arg("form_factors")
    );
    m.def(
        "circular_ring_average_r_dependent",
        &circular_ring_average_r_dependent,
        py::arg("hhat"),
        py::arg("z_phase"),
        py::arg("khat"),
        py::arg("form_factors"),
        py::arg("cutoffs")
    );
    m.def(
        "circular_ring_average_r_dependent64",
        &circular_ring_average_r_dependent64,
        py::arg("hhat"),
        py::arg("z_phase"),
        py::arg("khat"),
        py::arg("form_factors"),
        py::arg("cutoffs")
    );
    m.def(
        "circular_contract_r_dependent",
        &circular_contract_r_dependent,
        py::arg("hhat"),
        py::arg("z_phase"),
        py::arg("khat"),
        py::arg("form_factors"),
        py::arg("cutoffs")
    );
    m.def(
        "circular_contract_r_dependent64",
        &circular_contract_r_dependent64,
        py::arg("hhat"),
        py::arg("z_phase"),
        py::arg("khat"),
        py::arg("form_factors"),
        py::arg("cutoffs")
    );
    m.def(
        "circular_contract_r_dependent_modes",
        &circular_contract_r_dependent_modes,
        py::arg("hhat"),
        py::arg("z_phase"),
        py::arg("khat"),
        py::arg("form_factors"),
        py::arg("cutoffs"),
        py::arg("max_cutoff")
    );
    m.def(
        "circular_contract_r_dependent_modes64",
        &circular_contract_r_dependent_modes64,
        py::arg("hhat"),
        py::arg("z_phase"),
        py::arg("khat"),
        py::arg("form_factors"),
        py::arg("cutoffs"),
        py::arg("max_cutoff")
    );
    m.def(
        "circular_contract_r_dependent_half_modes",
        &circular_contract_r_dependent_half_modes,
        py::arg("hhat"),
        py::arg("z_phase"),
        py::arg("khat"),
        py::arg("form_factors"),
        py::arg("cutoffs"),
        py::arg("n_phi"),
        py::arg("max_cutoff")
    );
    m.def(
        "circular_contract_r_dependent_half_modes64",
        &circular_contract_r_dependent_half_modes64,
        py::arg("hhat"),
        py::arg("z_phase"),
        py::arg("khat"),
        py::arg("form_factors"),
        py::arg("cutoffs"),
        py::arg("n_phi"),
        py::arg("max_cutoff")
    );
    m.def(
        "circular_contract_r_dependent_half_modes_miller",
        &circular_contract_r_dependent_half_modes_miller,
        py::arg("hhat"),
        py::arg("z_phase"),
        py::arg("q_perp"),
        py::arg("r_centers"),
        py::arg("form_factors"),
        py::arg("cutoffs"),
        py::arg("n_phi"),
        py::arg("max_cutoff"),
        py::arg("extra_order")
    );
    m.def(
        "circular_contract_r_dependent_half_modes_miller64",
        &circular_contract_r_dependent_half_modes_miller64,
        py::arg("hhat"),
        py::arg("z_phase"),
        py::arg("q_perp"),
        py::arg("r_centers"),
        py::arg("form_factors"),
        py::arg("cutoffs"),
        py::arg("n_phi"),
        py::arg("max_cutoff"),
        py::arg("extra_order")
    );
    m.def(
        "circular_contract_r_dependent_half_z_reduced",
        &circular_contract_r_dependent_half_z_reduced,
        py::arg("z_pos"),
        py::arg("z_neg"),
        py::arg("khat"),
        py::arg("form_factors"),
        py::arg("cutoffs"),
        py::arg("n_phi"),
        py::arg("max_cutoff")
    );
    m.def(
        "circular_contract_r_dependent_half_z_reduced64",
        &circular_contract_r_dependent_half_z_reduced64,
        py::arg("z_pos"),
        py::arg("z_neg"),
        py::arg("khat"),
        py::arg("form_factors"),
        py::arg("cutoffs"),
        py::arg("n_phi"),
        py::arg("max_cutoff")
    );
    m.def(
        "analytic_kernel_hat_modes_miller",
        &analytic_kernel_hat_modes_miller,
        py::arg("q_perp"),
        py::arg("r_centers"),
        py::arg("n_phi"),
        py::arg("max_cutoff"),
        py::arg("extra_order") = 64
    );
    m.def(
        "analytic_kernel_hat_modes_miller64",
        &analytic_kernel_hat_modes_miller64,
        py::arg("q_perp"),
        py::arg("r_centers"),
        py::arg("n_phi"),
        py::arg("max_cutoff"),
        py::arg("extra_order") = 64
    );
    m.def(
        "analytic_kernel_hat_modes_table",
        &analytic_kernel_hat_modes_table,
        py::arg("q_perp"),
        py::arg("r_centers"),
        py::arg("bessel_table"),
        py::arg("n_phi"),
        py::arg("max_cutoff"),
        py::arg("dx")
    );
    m.def(
        "analytic_kernel_hat_modes_table64",
        &analytic_kernel_hat_modes_table64,
        py::arg("q_perp"),
        py::arg("r_centers"),
        py::arg("bessel_table"),
        py::arg("n_phi"),
        py::arg("max_cutoff"),
        py::arg("dx")
    );
    m.def(
        "circular_contract_z_reduced",
        &circular_contract_z_reduced,
        py::arg("z_reduced"),
        py::arg("khat"),
        py::arg("form_factors")
    );
    m.def(
        "circular_contract_z_reduced64",
        &circular_contract_z_reduced64,
        py::arg("z_reduced"),
        py::arg("khat"),
        py::arg("form_factors")
    );
    m.def(
        "circular_contract_sparse_rz",
        &circular_contract_sparse_rz,
        py::arg("active_e"),
        py::arg("active_r"),
        py::arg("active_z"),
        py::arg("active_hhat"),
        py::arg("z_phase"),
        py::arg("khat"),
        py::arg("form_factors")
    );
    m.def(
        "circular_contract_sparse_rz64",
        &circular_contract_sparse_rz64,
        py::arg("active_e"),
        py::arg("active_r"),
        py::arg("active_z"),
        py::arg("active_hhat"),
        py::arg("z_phase"),
        py::arg("khat"),
        py::arg("form_factors")
    );
    m.def(
        "circular_contract_sparse_flat",
        &circular_contract_sparse_flat,
        py::arg("active_e"),
        py::arg("active_r"),
        py::arg("active_z"),
        py::arg("active_beta"),
        py::arg("active_values"),
        py::arg("twiddle"),
        py::arg("z_phase"),
        py::arg("khat"),
        py::arg("form_factors")
    );
    m.def(
        "circular_contract_sparse_flat64",
        &circular_contract_sparse_flat64,
        py::arg("active_e"),
        py::arg("active_r"),
        py::arg("active_z"),
        py::arg("active_beta"),
        py::arg("active_values"),
        py::arg("twiddle"),
        py::arg("z_phase"),
        py::arg("khat"),
        py::arg("form_factors")
    );
    m.def(
        "circular_contract_sparse_profiles",
        &circular_contract_sparse_profiles,
        py::arg("profile_e"),
        py::arg("profile_r"),
        py::arg("profile_z"),
        py::arg("profile_starts"),
        py::arg("profile_counts"),
        py::arg("active_beta"),
        py::arg("active_values"),
        py::arg("twiddle"),
        py::arg("z_phase"),
        py::arg("khat"),
        py::arg("form_factors")
    );
    m.def(
        "circular_contract_sparse_profiles64",
        &circular_contract_sparse_profiles64,
        py::arg("profile_e"),
        py::arg("profile_r"),
        py::arg("profile_z"),
        py::arg("profile_starts"),
        py::arg("profile_counts"),
        py::arg("active_beta"),
        py::arg("active_values"),
        py::arg("twiddle"),
        py::arg("z_phase"),
        py::arg("khat"),
        py::arg("form_factors")
    );
    m.def(
        "build_sparse_source_projection",
        &build_sparse_source_projection,
        py::arg("profile_starts"),
        py::arg("profile_counts"),
        py::arg("active_z"),
        py::arg("active_beta"),
        py::arg("active_values"),
        py::arg("z_phase"),
        py::arg("n_phi")
    );
    m.def(
        "build_sparse_source_projection64",
        &build_sparse_source_projection64,
        py::arg("profile_starts"),
        py::arg("profile_counts"),
        py::arg("active_z"),
        py::arg("active_beta"),
        py::arg("active_values"),
        py::arg("z_phase"),
        py::arg("n_phi")
    );
    m.def(
        "sparse_source_r_dependent_contract",
        &sparse_source_r_dependent_contract,
        py::arg("projected_hhat"),
        py::arg("khat_unique"),
        py::arg("form_factors"),
        py::arg("cutoffs"),
        py::arg("profile_r_inverse"),
        py::arg("h_abs")
    );
    m.def(
        "sparse_source_r_dependent_contract64",
        &sparse_source_r_dependent_contract64,
        py::arg("projected_hhat"),
        py::arg("khat_unique"),
        py::arg("form_factors"),
        py::arg("cutoffs"),
        py::arg("profile_r_inverse"),
        py::arg("h_abs")
    );
    m.def(
        "build_sparse_profile_hhat",
        &build_sparse_profile_hhat,
        py::arg("profile_starts"),
        py::arg("profile_counts"),
        py::arg("active_beta"),
        py::arg("active_values"),
        py::arg("twiddle")
    );
    m.def(
        "build_sparse_profile_hhat64",
        &build_sparse_profile_hhat64,
        py::arg("profile_starts"),
        py::arg("profile_counts"),
        py::arg("active_beta"),
        py::arg("active_values"),
        py::arg("twiddle")
    );
}
