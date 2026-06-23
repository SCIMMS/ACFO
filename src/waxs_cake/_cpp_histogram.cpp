#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace py = pybind11;

namespace {

constexpr double kTwoPi = 6.283185307179586476925286766559;
constexpr std::size_t kThreadLocalMemoryBudget = 64ull * 1024ull * 1024ull;
constexpr py::ssize_t kMinAtomsPerThread = 50000;
constexpr char kHistogramThreadsEnv[] = "WAXS_CAKE_HIST_THREADS";

struct BinSpec {
    py::ssize_t n_atoms;
    py::ssize_t n_elements;
    py::ssize_t n_r;
    py::ssize_t n_z;
    py::ssize_t n_phi;
    double r_scale;
    double z_min;
    double z_scale;
    double beta_scale;
    std::size_t n_bins;
};

enum class AngleLutMode {
    Nearest,
    Cubic,
};

struct AngleLut {
    AngleLutMode mode = AngleLutMode::Nearest;
    py::ssize_t size = 0;
    std::vector<double> table;
    std::vector<double> coeffs;

    bool enabled() const {
        return size > 0;
    }
};

BinSpec make_spec(
    py::ssize_t n_atoms,
    py::ssize_t n_elements,
    py::ssize_t n_r,
    py::ssize_t n_z,
    py::ssize_t n_phi,
    double r_max,
    double z_min,
    double z_max,
    py::ssize_t angle_lut_size = 0,
    const std::string& angle_lut_mode = "nearest"
) {
    if (n_elements <= 0 || n_r <= 0 || n_z <= 0 || n_phi <= 0) {
        throw std::invalid_argument("n_elements, n_r, n_z, and n_phi must be positive");
    }
    if (!(r_max > 0.0)) {
        throw std::invalid_argument("r_max must be positive");
    }
    if (!(z_min < z_max)) {
        throw std::invalid_argument("z range must be increasing");
    }

    const auto max_size = std::numeric_limits<std::size_t>::max();
    const auto e = static_cast<std::size_t>(n_elements);
    const auto r = static_cast<std::size_t>(n_r);
    const auto z = static_cast<std::size_t>(n_z);
    const auto p = static_cast<std::size_t>(n_phi);
    if (e > max_size / r || e * r > max_size / z || e * r * z > max_size / p) {
        throw std::overflow_error("histogram shape is too large");
    }

    return BinSpec{
        n_atoms,
        n_elements,
        n_r,
        n_z,
        n_phi,
        static_cast<double>(n_r) / r_max,
        z_min,
        static_cast<double>(n_z) / (z_max - z_min),
        static_cast<double>(n_phi) / kTwoPi,
        e * r * z * p,
    };
}

py::array ensure_coords(py::array coords) {
    py::array contiguous = py::array::ensure(coords, py::array::c_style);
    if (!contiguous) {
        throw std::invalid_argument("coords must be C-contiguous");
    }
    const py::buffer_info info = contiguous.request();
    if (info.ndim != 2 || info.shape[1] != 3) {
        throw std::invalid_argument("coords must have shape (n_atoms, 3)");
    }
    const auto f32 = py::format_descriptor<float>::format();
    const auto f64 = py::format_descriptor<double>::format();
    if (!((info.format == f32 && info.itemsize == sizeof(float)) ||
          (info.format == f64 && info.itemsize == sizeof(double)))) {
        throw std::invalid_argument("coords must have dtype float32 or float64");
    }
    return contiguous;
}

py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
ensure_elements(py::object element_indices, py::ssize_t n_atoms) {
    if (element_indices.is_none()) {
        return {};
    }
    auto elements =
        py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>::ensure(
            element_indices
        );
    if (!elements) {
        throw std::invalid_argument("element_indices must be integer array-like");
    }
    const py::buffer_info info = elements.request();
    if (info.ndim != 1 || info.shape[0] != n_atoms) {
        throw std::invalid_argument("element_indices must have one entry per atom");
    }
    return elements;
}

template <typename T>
int clipped_index(T value, py::ssize_t upper) {
    int idx = static_cast<int>(value);
    if (idx < 0) {
        return 0;
    }
    if (idx >= upper) {
        return static_cast<int>(upper - 1);
    }
    return idx;
}

int beta_index_exact(double x, double y, const BinSpec& spec) {
    double beta = std::atan2(y, x);
    if (beta < 0.0) {
        beta += kTwoPi;
    }
    return clipped_index(beta * spec.beta_scale, spec.n_phi);
}

int beta_index_lut(
    double x,
    double y,
    const BinSpec& spec,
    const AngleLut& angle_lut
) {
    if (x == 0.0 && y == 0.0) {
        return 0;
    }
    const double ax = std::abs(x);
    const double ay = std::abs(y);
    const double denom = ax + ay;
    const double t = ay / denom;
    double angle;
    if (angle_lut.mode == AngleLutMode::Cubic) {
        double scaled = t * static_cast<double>(angle_lut.size);
        auto interval = static_cast<py::ssize_t>(scaled);
        double u = scaled - static_cast<double>(interval);
        if (interval >= angle_lut.size) {
            interval = angle_lut.size - 1;
            u = 1.0;
        }
        const double* coeff = angle_lut.coeffs.data() + 4 * interval;
        angle = ((coeff[3] * u + coeff[2]) * u + coeff[1]) * u + coeff[0];
    } else {
        const int table_idx = clipped_index(
            t * static_cast<double>(angle_lut.size + 1),
            angle_lut.size + 1
        );
        angle = angle_lut.table[static_cast<std::size_t>(table_idx)];
    }

    double beta;
    if (x >= 0.0 && y >= 0.0) {
        beta = angle;
    } else if (x < 0.0 && y >= 0.0) {
        beta = 3.1415926535897932384626433832795 - angle;
    } else if (x < 0.0 && y < 0.0) {
        beta = 3.1415926535897932384626433832795 + angle;
    } else {
        beta = kTwoPi - angle;
    }
    return clipped_index(beta * spec.beta_scale, spec.n_phi);
}

double angle_lut_derivative(double t) {
    const double one_minus_t = 1.0 - t;
    return 1.0 / (one_minus_t * one_minus_t + t * t);
}

AngleLutMode parse_angle_lut_mode(const std::string& mode) {
    if (mode == "nearest") {
        return AngleLutMode::Nearest;
    }
    if (mode == "cubic") {
        return AngleLutMode::Cubic;
    }
    throw std::invalid_argument("angle_lut_mode must be 'nearest' or 'cubic'");
}

AngleLut make_angle_lut(py::ssize_t angle_lut_size, const std::string& mode) {
    AngleLut lut;
    if (angle_lut_size <= 0) {
        return lut;
    }
    lut.mode = parse_angle_lut_mode(mode);
    lut.size = angle_lut_size;
    lut.table.resize(static_cast<std::size_t>(angle_lut_size + 1));
    for (py::ssize_t i = 0; i <= angle_lut_size; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(angle_lut_size);
        lut.table[static_cast<std::size_t>(i)] = std::atan2(t, 1.0 - t);
    }
    if (lut.mode == AngleLutMode::Cubic) {
        const double h = 1.0 / static_cast<double>(angle_lut_size);
        lut.coeffs.resize(static_cast<std::size_t>(4 * angle_lut_size));
        for (py::ssize_t i = 0; i < angle_lut_size; ++i) {
            const double t0 = static_cast<double>(i) * h;
            const double t1 = static_cast<double>(i + 1) * h;
            const double y0 = lut.table[static_cast<std::size_t>(i)];
            const double y1 = lut.table[static_cast<std::size_t>(i + 1)];
            const double m0 = angle_lut_derivative(t0);
            const double m1 = angle_lut_derivative(t1);
            double* coeff = lut.coeffs.data() + 4 * i;
            coeff[0] = y0;
            coeff[1] = h * m0;
            coeff[2] = 3.0 * (y1 - y0) - h * (2.0 * m0 + m1);
            coeff[3] = 2.0 * (y0 - y1) + h * (m0 + m1);
        }
    }
    return lut;
}

template <typename CoordT, typename OutT, typename WeightT>
void bin_range(
    const CoordT* coords,
    const std::int64_t* elements,
    const WeightT* weights,
    OutT* hist,
    const BinSpec& spec,
    const AngleLut& angle_lut,
    py::ssize_t begin,
    py::ssize_t end
) {
    const bool has_elements = elements != nullptr;
    const bool has_weights = weights != nullptr;
    const bool use_lut = angle_lut.enabled();

    for (py::ssize_t i = begin; i < end; ++i) {
        const double x = static_cast<double>(coords[3 * i]);
        const double y = static_cast<double>(coords[3 * i + 1]);
        const double z = static_cast<double>(coords[3 * i + 2]);

        const int r_idx = clipped_index(std::sqrt(x * x + y * y) * spec.r_scale, spec.n_r);
        const int z_idx = clipped_index((z - spec.z_min) * spec.z_scale, spec.n_z);

        const int beta_idx =
            use_lut ? beta_index_lut(x, y, spec, angle_lut) : beta_index_exact(x, y, spec);

        const auto element = static_cast<std::size_t>(has_elements ? elements[i] : 0);
        const std::size_t flat =
            (((element * static_cast<std::size_t>(spec.n_r) + static_cast<std::size_t>(r_idx)) *
                  static_cast<std::size_t>(spec.n_z) +
              static_cast<std::size_t>(z_idx)) *
                 static_cast<std::size_t>(spec.n_phi) +
             static_cast<std::size_t>(beta_idx));

        hist[flat] += has_weights ? static_cast<OutT>(weights[i]) : OutT{1};
    }
}

template <typename CoordT, typename OutT>
void bin_range_single_unweighted(
    const CoordT* coords,
    OutT* hist,
    const BinSpec& spec,
    const AngleLut& angle_lut,
    py::ssize_t begin,
    py::ssize_t end
) {
    const bool use_lut = angle_lut.enabled();
    const auto n_z = static_cast<std::size_t>(spec.n_z);
    const auto n_phi = static_cast<std::size_t>(spec.n_phi);

    for (py::ssize_t i = begin; i < end; ++i) {
        const double x = static_cast<double>(coords[3 * i]);
        const double y = static_cast<double>(coords[3 * i + 1]);
        const double z = static_cast<double>(coords[3 * i + 2]);

        const int r_idx = clipped_index(std::sqrt(x * x + y * y) * spec.r_scale, spec.n_r);
        const int z_idx = clipped_index((z - spec.z_min) * spec.z_scale, spec.n_z);
        const int beta_idx =
            use_lut ? beta_index_lut(x, y, spec, angle_lut) : beta_index_exact(x, y, spec);

        const std::size_t flat =
            ((static_cast<std::size_t>(r_idx) * n_z + static_cast<std::size_t>(z_idx)) *
                 n_phi +
             static_cast<std::size_t>(beta_idx));
        hist[flat] += OutT{1};
    }
}

template <
    bool HasElements,
    bool HasWeights,
    typename CoordT,
    typename OutT,
    typename WeightT>
void bin_range_specialized(
    const CoordT* coords,
    const std::int64_t* elements,
    const WeightT* weights,
    OutT* hist,
    const BinSpec& spec,
    const AngleLut& angle_lut,
    py::ssize_t begin,
    py::ssize_t end
) {
    const bool use_lut = angle_lut.enabled();
    const auto n_r = static_cast<std::size_t>(spec.n_r);
    const auto n_z = static_cast<std::size_t>(spec.n_z);
    const auto n_phi = static_cast<std::size_t>(spec.n_phi);

    for (py::ssize_t i = begin; i < end; ++i) {
        const double x = static_cast<double>(coords[3 * i]);
        const double y = static_cast<double>(coords[3 * i + 1]);
        const double z = static_cast<double>(coords[3 * i + 2]);

        const int r_idx = clipped_index(std::sqrt(x * x + y * y) * spec.r_scale, spec.n_r);
        const int z_idx = clipped_index((z - spec.z_min) * spec.z_scale, spec.n_z);
        const int beta_idx =
            use_lut ? beta_index_lut(x, y, spec, angle_lut) : beta_index_exact(x, y, spec);

        std::size_t element = 0;
        if constexpr (HasElements) {
            element = static_cast<std::size_t>(elements[i]);
        }
        const std::size_t flat =
            (((element * n_r + static_cast<std::size_t>(r_idx)) * n_z +
              static_cast<std::size_t>(z_idx)) *
                 n_phi +
             static_cast<std::size_t>(beta_idx));

        if constexpr (HasWeights) {
            hist[flat] += static_cast<OutT>(weights[i]);
        } else {
            hist[flat] += OutT{1};
        }
    }
}

template <typename CoordT, typename OutT, typename WeightT>
void bin_range_dispatch(
    const CoordT* coords,
    const std::int64_t* elements,
    const WeightT* weights,
    OutT* hist,
    const BinSpec& spec,
    const AngleLut& angle_lut,
    py::ssize_t begin,
    py::ssize_t end
) {
    const bool has_elements = elements != nullptr;
    const bool has_weights = weights != nullptr;

    if (!has_elements && !has_weights) {
        bin_range_single_unweighted(
            coords,
            hist,
            spec,
            angle_lut,
            begin,
            end
        );
    } else if (has_elements && !has_weights) {
        bin_range_specialized<true, false>(
            coords,
            elements,
            weights,
            hist,
            spec,
            angle_lut,
            begin,
            end
        );
    } else if (!has_elements && has_weights) {
        bin_range_specialized<false, true>(
            coords,
            elements,
            weights,
            hist,
            spec,
            angle_lut,
            begin,
            end
        );
    } else {
        bin_range_specialized<true, true>(
            coords,
            elements,
            weights,
            hist,
            spec,
            angle_lut,
            begin,
            end
        );
    }
}

template <typename CoordT>
void flat_indices_from_coords_range(
    const CoordT* coords,
    const std::int64_t* elements,
    std::int64_t* out,
    const BinSpec& spec,
    const AngleLut& angle_lut,
    py::ssize_t begin,
    py::ssize_t end
) {
    const bool has_elements = elements != nullptr;
    const bool use_lut = angle_lut.enabled();

    for (py::ssize_t i = begin; i < end; ++i) {
        const double x = static_cast<double>(coords[3 * i]);
        const double y = static_cast<double>(coords[3 * i + 1]);
        const double z = static_cast<double>(coords[3 * i + 2]);

        const int r_idx = clipped_index(std::sqrt(x * x + y * y) * spec.r_scale, spec.n_r);
        const int z_idx = clipped_index((z - spec.z_min) * spec.z_scale, spec.n_z);
        const int beta_idx =
            use_lut ? beta_index_lut(x, y, spec, angle_lut) : beta_index_exact(x, y, spec);
        const auto element = static_cast<std::size_t>(has_elements ? elements[i] : 0);
        out[i] = static_cast<std::int64_t>(
            (((element * static_cast<std::size_t>(spec.n_r) + static_cast<std::size_t>(r_idx)) *
                  static_cast<std::size_t>(spec.n_z) +
              static_cast<std::size_t>(z_idx)) *
                 static_cast<std::size_t>(spec.n_phi) +
             static_cast<std::size_t>(beta_idx))
        );
    }
}

template <typename OutT>
unsigned int choose_thread_count(py::ssize_t n_atoms, std::size_t n_bins);

template <typename CoordT>
void run_flat_indices_from_coords(
    const CoordT* coords,
    const std::int64_t* elements,
    std::int64_t* out,
    const BinSpec& spec,
    const AngleLut& angle_lut
) {
    const unsigned int n_threads = choose_thread_count<std::int64_t>(
        spec.n_atoms,
        static_cast<std::size_t>(std::max<py::ssize_t>(1, spec.n_r * spec.n_z * spec.n_phi))
    );
    if (n_threads == 1) {
        flat_indices_from_coords_range(coords, elements, out, spec, angle_lut, 0, spec.n_atoms);
        return;
    }

    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (unsigned int thread_id = 0; thread_id < n_threads; ++thread_id) {
        const py::ssize_t begin = (spec.n_atoms * thread_id) / n_threads;
        const py::ssize_t end = (spec.n_atoms * (thread_id + 1)) / n_threads;
        threads.emplace_back([&, begin, end]() {
            flat_indices_from_coords_range(coords, elements, out, spec, angle_lut, begin, end);
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
}

template <typename OutT>
unsigned int choose_thread_count(py::ssize_t n_atoms, std::size_t n_bins) {
    if (n_atoms < 2 * kMinAtomsPerThread) {
        return 1;
    }

    unsigned int hardware = std::thread::hardware_concurrency();
    if (hardware == 0) {
        hardware = 1;
    }

    const auto atom_threads =
        static_cast<unsigned int>((n_atoms + kMinAtomsPerThread - 1) / kMinAtomsPerThread);
    const std::size_t bytes_per_hist = std::max<std::size_t>(1, n_bins * sizeof(OutT));
    const auto memory_threads = static_cast<unsigned int>(
        std::max<std::size_t>(1, kThreadLocalMemoryBudget / bytes_per_hist)
    );
    const auto atoms_cap = static_cast<unsigned int>(std::max<py::ssize_t>(1, n_atoms));
    unsigned int requested_threads = hardware;
    if (const char* env = std::getenv(kHistogramThreadsEnv)) {
        char* end = nullptr;
        const long value = std::strtol(env, &end, 10);
        if (end != env && value > 0) {
            requested_threads = static_cast<unsigned int>(value);
        }
    }

    return std::max(
        1u,
        std::min({hardware, requested_threads, atom_threads, memory_threads, atoms_cap})
    );
}

template <typename CoordT, typename OutT, typename WeightT>
void run_histogram(
    const CoordT* coords,
    const std::int64_t* elements,
    const WeightT* weights,
    OutT* hist,
    const BinSpec& spec,
    const AngleLut& angle_lut
) {
    std::fill(hist, hist + spec.n_bins, OutT{});

    const unsigned int n_threads = choose_thread_count<OutT>(spec.n_atoms, spec.n_bins);
    if (n_threads == 1) {
        bin_range_dispatch(coords, elements, weights, hist, spec, angle_lut, 0, spec.n_atoms);
        return;
    }

    std::vector<std::vector<OutT>> locals;
    locals.reserve(n_threads);
    for (unsigned int i = 0; i < n_threads; ++i) {
        locals.emplace_back(spec.n_bins, OutT{});
    }

    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (unsigned int thread_id = 0; thread_id < n_threads; ++thread_id) {
        const py::ssize_t begin = (spec.n_atoms * thread_id) / n_threads;
        const py::ssize_t end = (spec.n_atoms * (thread_id + 1)) / n_threads;
        threads.emplace_back([&, begin, end, thread_id]() {
            bin_range_dispatch(
                coords,
                elements,
                weights,
                locals[thread_id].data(),
                spec,
                angle_lut,
                begin,
                end
            );
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    for (const auto& local : locals) {
        const OutT* local_data = local.data();
        for (std::size_t i = 0; i < spec.n_bins; ++i) {
            hist[i] += local_data[i];
        }
    }
}

py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
ensure_flat_indices(py::object flat_indices) {
    auto indices =
        py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>::ensure(
            flat_indices
        );
    if (!indices) {
        throw std::invalid_argument("flat_indices must be integer array-like");
    }
    const py::buffer_info info = indices.request();
    if (info.ndim != 1) {
        throw std::invalid_argument("flat_indices must be one-dimensional");
    }
    return indices;
}

void validate_flat_indices(
    const std::int64_t* flat_indices,
    py::ssize_t n_atoms,
    std::size_t n_bins
) {
    for (py::ssize_t i = 0; i < n_atoms; ++i) {
        const std::int64_t idx = flat_indices[i];
        if (idx < 0 || static_cast<std::size_t>(idx) >= n_bins) {
            throw std::out_of_range("flat_indices contains an out-of-range bin");
        }
    }
}

template <typename OutT>
void bin_flat_unweighted_range(
    const std::int64_t* flat_indices,
    OutT* hist,
    py::ssize_t begin,
    py::ssize_t end
) {
    for (py::ssize_t i = begin; i < end; ++i) {
        hist[static_cast<std::size_t>(flat_indices[i])] += OutT{1};
    }
}

template <typename OutT, typename WeightT>
void bin_flat_weighted_range(
    const std::int64_t* flat_indices,
    const WeightT* weights,
    OutT* hist,
    py::ssize_t begin,
    py::ssize_t end
) {
    for (py::ssize_t i = begin; i < end; ++i) {
        hist[static_cast<std::size_t>(flat_indices[i])] += static_cast<OutT>(weights[i]);
    }
}

template <typename OutT, typename WeightT>
void bin_flat_dispatch(
    const std::int64_t* flat_indices,
    const WeightT* weights,
    OutT* hist,
    py::ssize_t begin,
    py::ssize_t end
) {
    if (weights == nullptr) {
        bin_flat_unweighted_range(flat_indices, hist, begin, end);
    } else {
        bin_flat_weighted_range(flat_indices, weights, hist, begin, end);
    }
}

template <typename OutT, typename WeightT>
void run_flat_histogram(
    const std::int64_t* flat_indices,
    const WeightT* weights,
    OutT* hist,
    py::ssize_t n_atoms,
    std::size_t n_bins
) {
    std::fill(hist, hist + n_bins, OutT{});

    const unsigned int n_threads = choose_thread_count<OutT>(n_atoms, n_bins);
    if (n_threads == 1) {
        bin_flat_dispatch(flat_indices, weights, hist, 0, n_atoms);
        return;
    }

    std::vector<std::vector<OutT>> locals;
    locals.reserve(n_threads);
    for (unsigned int i = 0; i < n_threads; ++i) {
        locals.emplace_back(n_bins, OutT{});
    }

    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (unsigned int thread_id = 0; thread_id < n_threads; ++thread_id) {
        const py::ssize_t begin = (n_atoms * thread_id) / n_threads;
        const py::ssize_t end = (n_atoms * (thread_id + 1)) / n_threads;
        threads.emplace_back([&, begin, end, thread_id]() {
            bin_flat_dispatch(
                flat_indices,
                weights,
                locals[thread_id].data(),
                begin,
                end
            );
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    for (const auto& local : locals) {
        const OutT* local_data = local.data();
        for (std::size_t i = 0; i < n_bins; ++i) {
            hist[i] += local_data[i];
        }
    }
}

template <typename OutT, typename WeightT>
py::array_t<OutT> make_flat_output_and_run(
    py::object flat_indices_obj,
    const WeightT* weights,
    py::ssize_t n_bins_arg,
    bool validate_indices = true
) {
    if (n_bins_arg <= 0) {
        throw std::invalid_argument("n_bins must be positive");
    }
    auto flat_indices = ensure_flat_indices(flat_indices_obj);
    const py::buffer_info index_info = flat_indices.request();
    const py::ssize_t n_atoms = index_info.shape[0];
    const auto n_bins = static_cast<std::size_t>(n_bins_arg);
    const auto* flat_ptr = static_cast<const std::int64_t*>(index_info.ptr);
    if (validate_indices) {
        validate_flat_indices(flat_ptr, n_atoms, n_bins);
    }

    py::array_t<OutT> out({n_bins_arg});
    OutT* out_ptr = out.mutable_data();
    {
        py::gil_scoped_release release;
        run_flat_histogram(flat_ptr, weights, out_ptr, n_atoms, n_bins);
    }
    return out;
}

template <typename OutT, typename WeightT>
py::array_t<OutT> make_output_and_run(
    py::array coords_obj,
    py::object element_indices,
    const WeightT* weights,
    py::ssize_t n_elements,
    py::ssize_t n_r,
    py::ssize_t n_z,
    py::ssize_t n_phi,
    double r_max,
    double z_min,
    double z_max,
    py::ssize_t angle_lut_size = 0,
    const std::string& angle_lut_mode = "nearest"
) {
    py::array coords = ensure_coords(coords_obj);
    const py::buffer_info coord_info = coords.request();
    const py::ssize_t n_atoms = coord_info.shape[0];
    const bool has_element_indices = !element_indices.is_none();
    auto elements = ensure_elements(element_indices, n_atoms);
    const std::int64_t* element_ptr = has_element_indices ? elements.data() : nullptr;

    auto spec = make_spec(n_atoms, n_elements, n_r, n_z, n_phi, r_max, z_min, z_max);
    auto angle_lut = make_angle_lut(angle_lut_size, angle_lut_mode);
    std::vector<py::ssize_t> shape{n_elements, n_r, n_z, n_phi};
    py::array_t<OutT> out(shape);
    OutT* out_ptr = out.mutable_data();

    {
        py::gil_scoped_release release;
        if (coord_info.format == py::format_descriptor<double>::format()) {
            run_histogram(
                static_cast<const double*>(coord_info.ptr),
                element_ptr,
                weights,
                out_ptr,
                spec,
                angle_lut
            );
        } else {
            run_histogram(
                static_cast<const float*>(coord_info.ptr),
                element_ptr,
                weights,
                out_ptr,
                spec,
                angle_lut
            );
        }
    }
    return out;
}

py::array_t<std::int64_t> histogram_unweighted(
    py::array coords,
    py::object element_indices,
    py::ssize_t n_elements,
    py::ssize_t n_r,
    py::ssize_t n_z,
    py::ssize_t n_phi,
    double r_max,
    double z_min,
    double z_max,
    py::ssize_t angle_lut_size = 0,
    const std::string& angle_lut_mode = "nearest"
) {
    return make_output_and_run<std::int64_t, std::int64_t>(
        coords,
        element_indices,
        nullptr,
        n_elements,
        n_r,
        n_z,
        n_phi,
        r_max,
        z_min,
        z_max,
        angle_lut_size,
        angle_lut_mode
    );
}

py::array_t<std::uint32_t> histogram_unweighted_uint32(
    py::array coords,
    py::object element_indices,
    py::ssize_t n_elements,
    py::ssize_t n_r,
    py::ssize_t n_z,
    py::ssize_t n_phi,
    double r_max,
    double z_min,
    double z_max,
    py::ssize_t angle_lut_size = 0,
    const std::string& angle_lut_mode = "nearest"
) {
    py::array checked_coords = ensure_coords(coords);
    const py::ssize_t n_atoms = checked_coords.request().shape[0];
    if (static_cast<std::uint64_t>(n_atoms) >
        static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::overflow_error("uint32 histogram cannot safely count this many atoms");
    }
    return make_output_and_run<std::uint32_t, std::uint32_t>(
        checked_coords,
        element_indices,
        nullptr,
        n_elements,
        n_r,
        n_z,
        n_phi,
        r_max,
        z_min,
        z_max,
        angle_lut_size,
        angle_lut_mode
    );
}

py::array_t<float> histogram_unweighted_float32(
    py::array coords,
    py::object element_indices,
    py::ssize_t n_elements,
    py::ssize_t n_r,
    py::ssize_t n_z,
    py::ssize_t n_phi,
    double r_max,
    double z_min,
    double z_max,
    py::ssize_t angle_lut_size = 0,
    const std::string& angle_lut_mode = "nearest"
) {
    return make_output_and_run<float, float>(
        coords,
        element_indices,
        nullptr,
        n_elements,
        n_r,
        n_z,
        n_phi,
        r_max,
        z_min,
        z_max,
        angle_lut_size,
        angle_lut_mode
    );
}

py::array_t<double> histogram_weighted_real(
    py::array coords,
    py::object element_indices,
    py::array_t<double, py::array::c_style | py::array::forcecast> weights,
    py::ssize_t n_elements,
    py::ssize_t n_r,
    py::ssize_t n_z,
    py::ssize_t n_phi,
    double r_max,
    double z_min,
    double z_max,
    py::ssize_t angle_lut_size = 0,
    const std::string& angle_lut_mode = "nearest"
) {
    const py::buffer_info weight_info = weights.request();
    py::array checked_coords = ensure_coords(coords);
    const py::ssize_t n_atoms = checked_coords.request().shape[0];
    if (weight_info.ndim != 1 || weight_info.shape[0] != n_atoms) {
        throw std::invalid_argument("weights must have one entry per atom");
    }
    return make_output_and_run<double, double>(
        checked_coords,
        element_indices,
        weights.data(),
        n_elements,
        n_r,
        n_z,
        n_phi,
        r_max,
        z_min,
        z_max,
        angle_lut_size,
        angle_lut_mode
    );
}

py::array_t<float> histogram_weighted_real_float32(
    py::array coords,
    py::object element_indices,
    py::array_t<float, py::array::c_style | py::array::forcecast> weights,
    py::ssize_t n_elements,
    py::ssize_t n_r,
    py::ssize_t n_z,
    py::ssize_t n_phi,
    double r_max,
    double z_min,
    double z_max,
    py::ssize_t angle_lut_size = 0,
    const std::string& angle_lut_mode = "nearest"
) {
    const py::buffer_info weight_info = weights.request();
    py::array checked_coords = ensure_coords(coords);
    const py::ssize_t n_atoms = checked_coords.request().shape[0];
    if (weight_info.ndim != 1 || weight_info.shape[0] != n_atoms) {
        throw std::invalid_argument("weights must have one entry per atom");
    }
    return make_output_and_run<float, float>(
        checked_coords,
        element_indices,
        weights.data(),
        n_elements,
        n_r,
        n_z,
        n_phi,
        r_max,
        z_min,
        z_max,
        angle_lut_size,
        angle_lut_mode
    );
}

py::array_t<std::complex<double>> histogram_weighted_complex(
    py::array coords,
    py::object element_indices,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> weights,
    py::ssize_t n_elements,
    py::ssize_t n_r,
    py::ssize_t n_z,
    py::ssize_t n_phi,
    double r_max,
    double z_min,
    double z_max,
    py::ssize_t angle_lut_size = 0,
    const std::string& angle_lut_mode = "nearest"
) {
    const py::buffer_info weight_info = weights.request();
    py::array checked_coords = ensure_coords(coords);
    const py::ssize_t n_atoms = checked_coords.request().shape[0];
    if (weight_info.ndim != 1 || weight_info.shape[0] != n_atoms) {
        throw std::invalid_argument("weights must have one entry per atom");
    }
    return make_output_and_run<std::complex<double>, std::complex<double>>(
        checked_coords,
        element_indices,
        weights.data(),
        n_elements,
        n_r,
        n_z,
        n_phi,
        r_max,
        z_min,
        z_max,
        angle_lut_size,
        angle_lut_mode
    );
}

py::array_t<std::complex<float>> histogram_weighted_complex64(
    py::array coords,
    py::object element_indices,
    py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> weights,
    py::ssize_t n_elements,
    py::ssize_t n_r,
    py::ssize_t n_z,
    py::ssize_t n_phi,
    double r_max,
    double z_min,
    double z_max,
    py::ssize_t angle_lut_size = 0,
    const std::string& angle_lut_mode = "nearest"
) {
    const py::buffer_info weight_info = weights.request();
    py::array checked_coords = ensure_coords(coords);
    const py::ssize_t n_atoms = checked_coords.request().shape[0];
    if (weight_info.ndim != 1 || weight_info.shape[0] != n_atoms) {
        throw std::invalid_argument("weights must have one entry per atom");
    }
    return make_output_and_run<std::complex<float>, std::complex<float>>(
        checked_coords,
        element_indices,
        weights.data(),
        n_elements,
        n_r,
        n_z,
        n_phi,
        r_max,
        z_min,
        z_max,
        angle_lut_size,
        angle_lut_mode
    );
}

py::array_t<std::int64_t> flat_indices_from_coords(
    py::array coords_obj,
    py::object element_indices,
    py::ssize_t n_elements,
    py::ssize_t n_r,
    py::ssize_t n_z,
    py::ssize_t n_phi,
    double r_max,
    double z_min,
    double z_max,
    py::ssize_t angle_lut_size,
    const std::string& angle_lut_mode = "nearest"
) {
    py::array coords = ensure_coords(coords_obj);
    const py::buffer_info coord_info = coords.request();
    const py::ssize_t n_atoms = coord_info.shape[0];
    const bool has_element_indices = !element_indices.is_none();
    auto elements = ensure_elements(element_indices, n_atoms);
    const std::int64_t* element_ptr = has_element_indices ? elements.data() : nullptr;
    auto spec = make_spec(n_atoms, n_elements, n_r, n_z, n_phi, r_max, z_min, z_max);
    auto angle_lut = make_angle_lut(angle_lut_size, angle_lut_mode);

    py::array_t<std::int64_t> out({n_atoms});
    std::int64_t* out_ptr = out.mutable_data();
    {
        py::gil_scoped_release release;
        if (coord_info.format == py::format_descriptor<double>::format()) {
            run_flat_indices_from_coords(
                static_cast<const double*>(coord_info.ptr),
                element_ptr,
                out_ptr,
                spec,
                angle_lut
            );
        } else {
            run_flat_indices_from_coords(
                static_cast<const float*>(coord_info.ptr),
                element_ptr,
                out_ptr,
                spec,
                angle_lut
            );
        }
    }
    return out;
}

py::array_t<std::int64_t> histogram_flat_unweighted(
    py::object flat_indices,
    py::ssize_t n_bins,
    bool validate_indices = true
) {
    return make_flat_output_and_run<std::int64_t, std::int64_t>(
        flat_indices,
        nullptr,
        n_bins,
        validate_indices
    );
}

py::array_t<std::uint32_t> histogram_flat_unweighted_uint32(
    py::object flat_indices,
    py::ssize_t n_bins,
    bool validate_indices = true
) {
    return make_flat_output_and_run<std::uint32_t, std::uint32_t>(
        flat_indices,
        nullptr,
        n_bins,
        validate_indices
    );
}

py::array_t<float> histogram_flat_unweighted_float32(
    py::object flat_indices,
    py::ssize_t n_bins,
    bool validate_indices = true
) {
    return make_flat_output_and_run<float, float>(
        flat_indices,
        nullptr,
        n_bins,
        validate_indices
    );
}

py::array_t<double> histogram_flat_weighted_real(
    py::object flat_indices,
    py::array_t<double, py::array::c_style | py::array::forcecast> weights,
    py::ssize_t n_bins,
    bool validate_indices = true
) {
    const py::buffer_info weight_info = weights.request();
    auto checked_indices = ensure_flat_indices(flat_indices);
    if (weight_info.ndim != 1 || weight_info.shape[0] != checked_indices.request().shape[0]) {
        throw std::invalid_argument("weights must have one entry per index");
    }
    return make_flat_output_and_run<double, double>(
        checked_indices,
        weights.data(),
        n_bins,
        validate_indices
    );
}

py::array_t<float> histogram_flat_weighted_real_float32(
    py::object flat_indices,
    py::array_t<float, py::array::c_style | py::array::forcecast> weights,
    py::ssize_t n_bins,
    bool validate_indices = true
) {
    const py::buffer_info weight_info = weights.request();
    auto checked_indices = ensure_flat_indices(flat_indices);
    if (weight_info.ndim != 1 || weight_info.shape[0] != checked_indices.request().shape[0]) {
        throw std::invalid_argument("weights must have one entry per index");
    }
    return make_flat_output_and_run<float, float>(
        checked_indices,
        weights.data(),
        n_bins,
        validate_indices
    );
}

py::array_t<std::complex<double>> histogram_flat_weighted_complex(
    py::object flat_indices,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> weights,
    py::ssize_t n_bins,
    bool validate_indices = true
) {
    const py::buffer_info weight_info = weights.request();
    auto checked_indices = ensure_flat_indices(flat_indices);
    if (weight_info.ndim != 1 || weight_info.shape[0] != checked_indices.request().shape[0]) {
        throw std::invalid_argument("weights must have one entry per index");
    }
    return make_flat_output_and_run<std::complex<double>, std::complex<double>>(
        checked_indices,
        weights.data(),
        n_bins,
        validate_indices
    );
}

py::array_t<std::complex<float>> histogram_flat_weighted_complex64(
    py::object flat_indices,
    py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> weights,
    py::ssize_t n_bins,
    bool validate_indices = true
) {
    const py::buffer_info weight_info = weights.request();
    auto checked_indices = ensure_flat_indices(flat_indices);
    if (weight_info.ndim != 1 || weight_info.shape[0] != checked_indices.request().shape[0]) {
        throw std::invalid_argument("weights must have one entry per index");
    }
    return make_flat_output_and_run<std::complex<float>, std::complex<float>>(
        checked_indices,
        weights.data(),
        n_bins,
        validate_indices
    );
}

}  // namespace

PYBIND11_MODULE(_cpp_histogram, m) {
    m.doc() = "C++ cylindrical histogram kernels for waxs_cake";
    m.def(
        "histogram_unweighted",
        &histogram_unweighted,
        py::arg("coords"),
        py::arg("element_indices"),
        py::arg("n_elements"),
        py::arg("n_r"),
        py::arg("n_z"),
        py::arg("n_phi"),
        py::arg("r_max"),
        py::arg("z_min"),
        py::arg("z_max"),
        py::arg("angle_lut_size") = 0,
        py::arg("angle_lut_mode") = "nearest"
    );
    m.def(
        "histogram_unweighted_uint32",
        &histogram_unweighted_uint32,
        py::arg("coords"),
        py::arg("element_indices"),
        py::arg("n_elements"),
        py::arg("n_r"),
        py::arg("n_z"),
        py::arg("n_phi"),
        py::arg("r_max"),
        py::arg("z_min"),
        py::arg("z_max"),
        py::arg("angle_lut_size") = 0,
        py::arg("angle_lut_mode") = "nearest"
    );
    m.def(
        "histogram_unweighted_float32",
        &histogram_unweighted_float32,
        py::arg("coords"),
        py::arg("element_indices"),
        py::arg("n_elements"),
        py::arg("n_r"),
        py::arg("n_z"),
        py::arg("n_phi"),
        py::arg("r_max"),
        py::arg("z_min"),
        py::arg("z_max"),
        py::arg("angle_lut_size") = 0,
        py::arg("angle_lut_mode") = "nearest"
    );
    m.def(
        "histogram_weighted_real",
        &histogram_weighted_real,
        py::arg("coords"),
        py::arg("element_indices"),
        py::arg("weights"),
        py::arg("n_elements"),
        py::arg("n_r"),
        py::arg("n_z"),
        py::arg("n_phi"),
        py::arg("r_max"),
        py::arg("z_min"),
        py::arg("z_max"),
        py::arg("angle_lut_size") = 0,
        py::arg("angle_lut_mode") = "nearest"
    );
    m.def(
        "histogram_weighted_real_float32",
        &histogram_weighted_real_float32,
        py::arg("coords"),
        py::arg("element_indices"),
        py::arg("weights"),
        py::arg("n_elements"),
        py::arg("n_r"),
        py::arg("n_z"),
        py::arg("n_phi"),
        py::arg("r_max"),
        py::arg("z_min"),
        py::arg("z_max"),
        py::arg("angle_lut_size") = 0,
        py::arg("angle_lut_mode") = "nearest"
    );
    m.def(
        "histogram_weighted_complex",
        &histogram_weighted_complex,
        py::arg("coords"),
        py::arg("element_indices"),
        py::arg("weights"),
        py::arg("n_elements"),
        py::arg("n_r"),
        py::arg("n_z"),
        py::arg("n_phi"),
        py::arg("r_max"),
        py::arg("z_min"),
        py::arg("z_max"),
        py::arg("angle_lut_size") = 0,
        py::arg("angle_lut_mode") = "nearest"
    );
    m.def(
        "histogram_weighted_complex64",
        &histogram_weighted_complex64,
        py::arg("coords"),
        py::arg("element_indices"),
        py::arg("weights"),
        py::arg("n_elements"),
        py::arg("n_r"),
        py::arg("n_z"),
        py::arg("n_phi"),
        py::arg("r_max"),
        py::arg("z_min"),
        py::arg("z_max"),
        py::arg("angle_lut_size") = 0,
        py::arg("angle_lut_mode") = "nearest"
    );
    m.def(
        "flat_indices_from_coords",
        &flat_indices_from_coords,
        py::arg("coords"),
        py::arg("element_indices"),
        py::arg("n_elements"),
        py::arg("n_r"),
        py::arg("n_z"),
        py::arg("n_phi"),
        py::arg("r_max"),
        py::arg("z_min"),
        py::arg("z_max"),
        py::arg("angle_lut_size") = 0,
        py::arg("angle_lut_mode") = "nearest"
    );
    m.def(
        "histogram_flat_unweighted",
        &histogram_flat_unweighted,
        py::arg("flat_indices"),
        py::arg("n_bins"),
        py::arg("validate_indices") = true
    );
    m.def(
        "histogram_flat_unweighted_uint32",
        &histogram_flat_unweighted_uint32,
        py::arg("flat_indices"),
        py::arg("n_bins"),
        py::arg("validate_indices") = true
    );
    m.def(
        "histogram_flat_unweighted_float32",
        &histogram_flat_unweighted_float32,
        py::arg("flat_indices"),
        py::arg("n_bins"),
        py::arg("validate_indices") = true
    );
    m.def(
        "histogram_flat_weighted_real",
        &histogram_flat_weighted_real,
        py::arg("flat_indices"),
        py::arg("weights"),
        py::arg("n_bins"),
        py::arg("validate_indices") = true
    );
    m.def(
        "histogram_flat_weighted_real_float32",
        &histogram_flat_weighted_real_float32,
        py::arg("flat_indices"),
        py::arg("weights"),
        py::arg("n_bins"),
        py::arg("validate_indices") = true
    );
    m.def(
        "histogram_flat_weighted_complex",
        &histogram_flat_weighted_complex,
        py::arg("flat_indices"),
        py::arg("weights"),
        py::arg("n_bins"),
        py::arg("validate_indices") = true
    );
    m.def(
        "histogram_flat_weighted_complex64",
        &histogram_flat_weighted_complex64,
        py::arg("flat_indices"),
        py::arg("weights"),
        py::arg("n_bins"),
        py::arg("validate_indices") = true
    );
}
