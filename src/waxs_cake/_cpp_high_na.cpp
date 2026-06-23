#include <pybind11/numpy.h>
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
using Int64Array = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;

struct RhoDependentGroup {
    Int64Array r_indices_array;
    Int64Array h_positions_array;
    ComplexArray radial_array;
    ComplexArray angular_array;
    py::buffer_info r_indices_info;
    py::buffer_info h_positions_info;
    py::buffer_info radial_info;
    py::buffer_info angular_info;
    const std::int64_t* r_indices;
    const std::int64_t* h_positions;
    const Complex128* radial;
    const Complex128* angular;
    std::int64_t nr;
    std::int64_t nh;
};

struct PositiveRhoDependentGroup {
    Int64Array r_indices_array;
    Int64Array h_positions_array;
    ComplexArray radial_array;
    ComplexArray angular_positive_array;
    ComplexArray angular_negative_array;
    py::buffer_info r_indices_info;
    py::buffer_info h_positions_info;
    py::buffer_info radial_info;
    py::buffer_info angular_positive_info;
    py::buffer_info angular_negative_info;
    const std::int64_t* r_indices;
    const std::int64_t* h_positions;
    const Complex128* radial;
    const Complex128* angular_positive;
    const Complex128* angular_negative;
    std::int64_t nr;
    std::int64_t nh;
};

unsigned int choose_thread_count(std::int64_t tasks, std::int64_t work_per_task) {
    if (tasks <= 1 || work_per_task < 50000) {
        return 1;
    }
    unsigned int hardware = std::thread::hardware_concurrency();
    if (hardware == 0) {
        hardware = 1;
    }
    return std::max(1u, std::min<unsigned int>(hardware, static_cast<unsigned int>(tasks)));
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

void validate_shapes(
    const py::buffer_info& coeff,
    const py::buffer_info& radial,
    const py::buffer_info& angular,
    const py::buffer_info& defocus
) {
    if (coeff.ndim != 3) {
        throw std::invalid_argument("coeff must have shape (batch, ntheta, nh)");
    }
    if (radial.ndim != 3) {
        throw std::invalid_argument("radial must have shape (ntheta, nh, nrho)");
    }
    if (angular.ndim != 2) {
        throw std::invalid_argument("angular must have shape (nh, npsi)");
    }
    if (defocus.ndim != 2) {
        throw std::invalid_argument("defocus must have shape (ntheta, nz)");
    }
    if (coeff.shape[1] != radial.shape[0] || coeff.shape[2] != radial.shape[1] ||
        coeff.shape[2] != angular.shape[0] || coeff.shape[1] != defocus.shape[0]) {
        throw std::invalid_argument("inconsistent separable high-NA contraction shapes");
    }
}

void validate_adjoint_shapes(
    const py::buffer_info& psi_contracted,
    const py::buffer_info& radial_conj,
    const py::buffer_info& defocus_conj
) {
    if (psi_contracted.ndim != 3) {
        throw std::invalid_argument("psi_contracted must have shape (nrho, nh, nz)");
    }
    if (radial_conj.ndim != 3) {
        throw std::invalid_argument("radial_conj must have shape (ntheta, nh, nrho)");
    }
    if (defocus_conj.ndim != 2) {
        throw std::invalid_argument("defocus_conj must have shape (ntheta, nz)");
    }
    if (psi_contracted.shape[0] != radial_conj.shape[2] ||
        psi_contracted.shape[1] != radial_conj.shape[1] ||
        psi_contracted.shape[2] != defocus_conj.shape[1] ||
        radial_conj.shape[0] != defocus_conj.shape[0]) {
        throw std::invalid_argument("inconsistent separable adjoint contraction shapes");
    }
}

ComplexArray separable_contract_many(
    ComplexArray coeff_array,
    ComplexArray radial_array,
    ComplexArray angular_array,
    ComplexArray defocus_array,
    std::int64_t requested_threads
) {
    const py::buffer_info coeff_info = coeff_array.request();
    const py::buffer_info radial_info = radial_array.request();
    const py::buffer_info angular_info = angular_array.request();
    const py::buffer_info defocus_info = defocus_array.request();
    validate_shapes(coeff_info, radial_info, angular_info, defocus_info);

    const std::int64_t batch = static_cast<std::int64_t>(coeff_info.shape[0]);
    const std::int64_t ntheta = static_cast<std::int64_t>(coeff_info.shape[1]);
    const std::int64_t nh = static_cast<std::int64_t>(coeff_info.shape[2]);
    const std::int64_t nrho = static_cast<std::int64_t>(radial_info.shape[2]);
    const std::int64_t npsi = static_cast<std::int64_t>(angular_info.shape[1]);
    const std::int64_t nz = static_cast<std::int64_t>(defocus_info.shape[1]);

    auto out = ComplexArray({batch, nrho * npsi * nz});
    py::buffer_info out_info = out.request();

    const auto* coeff = static_cast<const Complex128*>(coeff_info.ptr);
    const auto* radial = static_cast<const Complex128*>(radial_info.ptr);
    const auto* angular = static_cast<const Complex128*>(angular_info.ptr);
    const auto* defocus = static_cast<const Complex128*>(defocus_info.ptr);
    auto* out_ptr = static_cast<Complex128*>(out_info.ptr);

    std::vector<Complex128> temp(
        static_cast<std::size_t>(batch * nrho * ntheta * npsi),
        Complex128{0.0, 0.0}
    );

    unsigned int threads = 1;
    if (requested_threads > 0) {
        threads = static_cast<unsigned int>(requested_threads);
    } else {
        threads = choose_thread_count(batch * nrho, ntheta * nh * npsi);
    }

    {
        py::gil_scoped_release release;

        parallel_for(batch * nrho, threads, [&](std::int64_t start, std::int64_t stop) {
            for (std::int64_t task = start; task < stop; ++task) {
                const std::int64_t b = task / nrho;
                const std::int64_t r = task - b * nrho;
                for (std::int64_t t = 0; t < ntheta; ++t) {
                    Complex128* temp_slice =
                        temp.data() + (((b * nrho + r) * ntheta + t) * npsi);
                    std::fill(temp_slice, temp_slice + npsi, Complex128{0.0, 0.0});
                    for (std::int64_t h = 0; h < nh; ++h) {
                        const Complex128 scale =
                            coeff[(b * ntheta + t) * nh + h] *
                            radial[(t * nh + h) * nrho + r];
                        const Complex128* angular_row = angular + h * npsi;
                        for (std::int64_t p = 0; p < npsi; ++p) {
                            temp_slice[p] += scale * angular_row[p];
                        }
                    }
                }
            }
        });

        const unsigned int stage2_threads = requested_threads > 0
                                                ? static_cast<unsigned int>(requested_threads)
                                                : choose_thread_count(batch * nrho * npsi, ntheta * nz);
        parallel_for(
            batch * nrho * npsi,
            stage2_threads,
            [&](std::int64_t start, std::int64_t stop) {
                for (std::int64_t task = start; task < stop; ++task) {
                    const std::int64_t p = task % npsi;
                    const std::int64_t tmp = task / npsi;
                    const std::int64_t r = tmp % nrho;
                    const std::int64_t b = tmp / nrho;
                    Complex128* out_slice =
                        out_ptr + b * (nrho * npsi * nz) + (r * npsi + p) * nz;
                    std::fill(out_slice, out_slice + nz, Complex128{0.0, 0.0});
                    for (std::int64_t t = 0; t < ntheta; ++t) {
                        const Complex128 value =
                            temp[((b * nrho + r) * ntheta + t) * npsi + p];
                        const Complex128* defocus_row = defocus + t * nz;
                        for (std::int64_t z = 0; z < nz; ++z) {
                            out_slice[z] += value * defocus_row[z];
                        }
                    }
                }
            }
        );
    }

    return out;
}

ComplexArray separable_contract_many_fused(
    ComplexArray coeff_array,
    ComplexArray radial_array,
    ComplexArray angular_array,
    ComplexArray defocus_array,
    std::int64_t requested_threads
) {
    const py::buffer_info coeff_info = coeff_array.request();
    const py::buffer_info radial_info = radial_array.request();
    const py::buffer_info angular_info = angular_array.request();
    const py::buffer_info defocus_info = defocus_array.request();
    validate_shapes(coeff_info, radial_info, angular_info, defocus_info);

    const std::int64_t batch = static_cast<std::int64_t>(coeff_info.shape[0]);
    const std::int64_t ntheta = static_cast<std::int64_t>(coeff_info.shape[1]);
    const std::int64_t nh = static_cast<std::int64_t>(coeff_info.shape[2]);
    const std::int64_t nrho = static_cast<std::int64_t>(radial_info.shape[2]);
    const std::int64_t npsi = static_cast<std::int64_t>(angular_info.shape[1]);
    const std::int64_t nz = static_cast<std::int64_t>(defocus_info.shape[1]);

    auto out = ComplexArray({batch, nrho * npsi * nz});
    py::buffer_info out_info = out.request();

    const auto* coeff = static_cast<const Complex128*>(coeff_info.ptr);
    const auto* radial = static_cast<const Complex128*>(radial_info.ptr);
    const auto* angular = static_cast<const Complex128*>(angular_info.ptr);
    const auto* defocus = static_cast<const Complex128*>(defocus_info.ptr);
    auto* out_ptr = static_cast<Complex128*>(out_info.ptr);

    unsigned int threads = 1;
    if (requested_threads > 0) {
        threads = static_cast<unsigned int>(requested_threads);
    } else {
        threads = choose_thread_count(batch * nrho, ntheta * nh * npsi);
    }

    {
        py::gil_scoped_release release;
        parallel_for(batch * nrho, threads, [&](std::int64_t start, std::int64_t stop) {
            std::vector<double> angular_temp_re(static_cast<std::size_t>(npsi));
            std::vector<double> angular_temp_im(static_cast<std::size_t>(npsi));
            std::vector<double> out_re(static_cast<std::size_t>(npsi * nz));
            std::vector<double> out_im(static_cast<std::size_t>(npsi * nz));
            for (std::int64_t task = start; task < stop; ++task) {
                const std::int64_t b = task / nrho;
                const std::int64_t r = task - b * nrho;
                Complex128* out_base = out_ptr + b * (nrho * npsi * nz) + r * npsi * nz;
                std::fill(out_re.begin(), out_re.end(), 0.0);
                std::fill(out_im.begin(), out_im.end(), 0.0);

                for (std::int64_t t = 0; t < ntheta; ++t) {
                    std::fill(
                        angular_temp_re.begin(),
                        angular_temp_re.end(),
                        0.0
                    );
                    std::fill(
                        angular_temp_im.begin(),
                        angular_temp_im.end(),
                        0.0
                    );
                    for (std::int64_t h = 0; h < nh; ++h) {
                        const Complex128 c = coeff[(b * ntheta + t) * nh + h];
                        const Complex128 q = radial[(t * nh + h) * nrho + r];
                        const double scale_re = c.real() * q.real() - c.imag() * q.imag();
                        const double scale_im = c.real() * q.imag() + c.imag() * q.real();
                        const Complex128* angular_row = angular + h * npsi;
                        for (std::int64_t p = 0; p < npsi; ++p) {
                            const Complex128 a = angular_row[p];
                            angular_temp_re[static_cast<std::size_t>(p)] +=
                                scale_re * a.real() - scale_im * a.imag();
                            angular_temp_im[static_cast<std::size_t>(p)] +=
                                scale_re * a.imag() + scale_im * a.real();
                        }
                    }
                    const Complex128* defocus_row = defocus + t * nz;
                    for (std::int64_t p = 0; p < npsi; ++p) {
                        const double value_re = angular_temp_re[static_cast<std::size_t>(p)];
                        const double value_im = angular_temp_im[static_cast<std::size_t>(p)];
                        for (std::int64_t z = 0; z < nz; ++z) {
                            const Complex128 d = defocus_row[z];
                            const std::size_t out_idx = static_cast<std::size_t>(p * nz + z);
                            out_re[out_idx] += value_re * d.real() - value_im * d.imag();
                            out_im[out_idx] += value_re * d.imag() + value_im * d.real();
                        }
                    }
                }
                for (std::int64_t pz = 0; pz < npsi * nz; ++pz) {
                    const std::size_t idx = static_cast<std::size_t>(pz);
                    out_base[pz] = Complex128{out_re[idx], out_im[idx]};
                }
            }
        });
    }

    return out;
}

ComplexArray positive_separable_contract_many_fused(
    ComplexArray coeff_positive_array,
    ComplexArray coeff_negative_array,
    ComplexArray radial_array,
    ComplexArray angular_positive_array,
    ComplexArray angular_negative_array,
    ComplexArray defocus_array,
    std::int64_t requested_threads
) {
    const py::buffer_info coeff_positive_info = coeff_positive_array.request();
    const py::buffer_info coeff_negative_info = coeff_negative_array.request();
    const py::buffer_info radial_info = radial_array.request();
    const py::buffer_info angular_positive_info = angular_positive_array.request();
    const py::buffer_info angular_negative_info = angular_negative_array.request();
    const py::buffer_info defocus_info = defocus_array.request();
    validate_shapes(
        coeff_positive_info,
        radial_info,
        angular_positive_info,
        defocus_info
    );
    validate_shapes(
        coeff_negative_info,
        radial_info,
        angular_negative_info,
        defocus_info
    );

    const std::int64_t batch = static_cast<std::int64_t>(coeff_positive_info.shape[0]);
    const std::int64_t ntheta = static_cast<std::int64_t>(coeff_positive_info.shape[1]);
    const std::int64_t nh = static_cast<std::int64_t>(coeff_positive_info.shape[2]);
    const std::int64_t nrho = static_cast<std::int64_t>(radial_info.shape[2]);
    const std::int64_t npsi = static_cast<std::int64_t>(angular_positive_info.shape[1]);
    const std::int64_t nz = static_cast<std::int64_t>(defocus_info.shape[1]);

    auto out = ComplexArray({batch, nrho * npsi * nz});
    py::buffer_info out_info = out.request();

    const auto* coeff_positive =
        static_cast<const Complex128*>(coeff_positive_info.ptr);
    const auto* coeff_negative =
        static_cast<const Complex128*>(coeff_negative_info.ptr);
    const auto* radial = static_cast<const Complex128*>(radial_info.ptr);
    const auto* angular_positive =
        static_cast<const Complex128*>(angular_positive_info.ptr);
    const auto* angular_negative =
        static_cast<const Complex128*>(angular_negative_info.ptr);
    const auto* defocus = static_cast<const Complex128*>(defocus_info.ptr);
    auto* out_ptr = static_cast<Complex128*>(out_info.ptr);

    unsigned int threads = 1;
    if (requested_threads > 0) {
        threads = static_cast<unsigned int>(requested_threads);
    } else {
        threads = choose_thread_count(batch * nrho, ntheta * nh * npsi);
    }

    {
        py::gil_scoped_release release;
        parallel_for(batch * nrho, threads, [&](std::int64_t start, std::int64_t stop) {
            std::vector<double> angular_temp_re(static_cast<std::size_t>(npsi));
            std::vector<double> angular_temp_im(static_cast<std::size_t>(npsi));
            std::vector<double> out_re(static_cast<std::size_t>(npsi * nz));
            std::vector<double> out_im(static_cast<std::size_t>(npsi * nz));
            for (std::int64_t task = start; task < stop; ++task) {
                const std::int64_t b = task / nrho;
                const std::int64_t r = task - b * nrho;
                Complex128* out_base = out_ptr + b * (nrho * npsi * nz) + r * npsi * nz;
                std::fill(out_re.begin(), out_re.end(), 0.0);
                std::fill(out_im.begin(), out_im.end(), 0.0);

                for (std::int64_t t = 0; t < ntheta; ++t) {
                    std::fill(angular_temp_re.begin(), angular_temp_re.end(), 0.0);
                    std::fill(angular_temp_im.begin(), angular_temp_im.end(), 0.0);
                    for (std::int64_t h = 0; h < nh; ++h) {
                        const Complex128 q = radial[(t * nh + h) * nrho + r];
                        const Complex128 c_pos =
                            coeff_positive[(b * ntheta + t) * nh + h];
                        const Complex128 c_neg =
                            coeff_negative[(b * ntheta + t) * nh + h];
                        const double scale_pos_re =
                            c_pos.real() * q.real() - c_pos.imag() * q.imag();
                        const double scale_pos_im =
                            c_pos.real() * q.imag() + c_pos.imag() * q.real();
                        const double scale_neg_re =
                            c_neg.real() * q.real() - c_neg.imag() * q.imag();
                        const double scale_neg_im =
                            c_neg.real() * q.imag() + c_neg.imag() * q.real();
                        const Complex128* angular_pos_row = angular_positive + h * npsi;
                        const Complex128* angular_neg_row = angular_negative + h * npsi;
                        for (std::int64_t p = 0; p < npsi; ++p) {
                            const Complex128 a_pos = angular_pos_row[p];
                            const Complex128 a_neg = angular_neg_row[p];
                            angular_temp_re[static_cast<std::size_t>(p)] +=
                                scale_pos_re * a_pos.real() -
                                scale_pos_im * a_pos.imag() +
                                scale_neg_re * a_neg.real() -
                                scale_neg_im * a_neg.imag();
                            angular_temp_im[static_cast<std::size_t>(p)] +=
                                scale_pos_re * a_pos.imag() +
                                scale_pos_im * a_pos.real() +
                                scale_neg_re * a_neg.imag() +
                                scale_neg_im * a_neg.real();
                        }
                    }
                    const Complex128* defocus_row = defocus + t * nz;
                    for (std::int64_t p = 0; p < npsi; ++p) {
                        const double value_re = angular_temp_re[static_cast<std::size_t>(p)];
                        const double value_im = angular_temp_im[static_cast<std::size_t>(p)];
                        for (std::int64_t z = 0; z < nz; ++z) {
                            const Complex128 d = defocus_row[z];
                            const std::size_t out_idx = static_cast<std::size_t>(p * nz + z);
                            out_re[out_idx] += value_re * d.real() - value_im * d.imag();
                            out_im[out_idx] += value_re * d.imag() + value_im * d.real();
                        }
                    }
                }
                for (std::int64_t pz = 0; pz < npsi * nz; ++pz) {
                    const std::size_t idx = static_cast<std::size_t>(pz);
                    out_base[pz] = Complex128{out_re[idx], out_im[idx]};
                }
            }
        });
    }

    return out;
}

ComplexArray rho_dependent_contract_many_fused(
    ComplexArray coeff_array,
    py::list r_indices_arrays,
    py::list h_position_arrays,
    py::list radial_arrays,
    py::list angular_arrays,
    ComplexArray defocus_array,
    std::int64_t nrho,
    std::int64_t requested_threads
) {
    const py::buffer_info coeff_info = coeff_array.request();
    const py::buffer_info defocus_info = defocus_array.request();
    if (coeff_info.ndim != 3) {
        throw std::invalid_argument("coeff must have shape (batch, ntheta, nh_max)");
    }
    if (defocus_info.ndim != 2) {
        throw std::invalid_argument("defocus must have shape (ntheta, nz)");
    }
    if (coeff_info.shape[1] != defocus_info.shape[0]) {
        throw std::invalid_argument("coeff and defocus ntheta dimensions differ");
    }
    if (nrho <= 0) {
        throw std::invalid_argument("nrho must be positive");
    }

    const std::size_t group_count = py::len(r_indices_arrays);
    if (group_count == 0) {
        throw std::invalid_argument("at least one rho-dependent group is required");
    }
    if (py::len(h_position_arrays) != group_count ||
        py::len(radial_arrays) != group_count ||
        py::len(angular_arrays) != group_count) {
        throw std::invalid_argument("rho-dependent group lists must have equal length");
    }

    const std::int64_t batch = static_cast<std::int64_t>(coeff_info.shape[0]);
    const std::int64_t ntheta = static_cast<std::int64_t>(coeff_info.shape[1]);
    const std::int64_t nh_max = static_cast<std::int64_t>(coeff_info.shape[2]);
    const std::int64_t nz = static_cast<std::int64_t>(defocus_info.shape[1]);

    std::vector<RhoDependentGroup> groups;
    groups.reserve(group_count);
    std::int64_t npsi = -1;
    for (std::size_t ig = 0; ig < group_count; ++ig) {
        RhoDependentGroup group{
            py::cast<Int64Array>(r_indices_arrays[ig]),
            py::cast<Int64Array>(h_position_arrays[ig]),
            py::cast<ComplexArray>(radial_arrays[ig]),
            py::cast<ComplexArray>(angular_arrays[ig]),
            py::buffer_info(),
            py::buffer_info(),
            py::buffer_info(),
            py::buffer_info(),
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            0,
            0,
        };
        group.r_indices_info = group.r_indices_array.request();
        group.h_positions_info = group.h_positions_array.request();
        group.radial_info = group.radial_array.request();
        group.angular_info = group.angular_array.request();

        if (group.r_indices_info.ndim != 1 || group.h_positions_info.ndim != 1) {
            throw std::invalid_argument("r_indices and h_positions must be 1D arrays");
        }
        if (group.radial_info.ndim != 3) {
            throw std::invalid_argument("radial group must have shape (ntheta, nh, nr)");
        }
        if (group.angular_info.ndim != 2) {
            throw std::invalid_argument("angular group must have shape (nh, npsi)");
        }
        group.nr = static_cast<std::int64_t>(group.r_indices_info.shape[0]);
        group.nh = static_cast<std::int64_t>(group.h_positions_info.shape[0]);
        if (group.radial_info.shape[0] != ntheta ||
            group.radial_info.shape[1] != group.nh ||
            group.radial_info.shape[2] != group.nr ||
            group.angular_info.shape[0] != group.nh) {
            throw std::invalid_argument("inconsistent rho-dependent group shapes");
        }
        if (npsi < 0) {
            npsi = static_cast<std::int64_t>(group.angular_info.shape[1]);
        } else if (group.angular_info.shape[1] != npsi) {
            throw std::invalid_argument("all angular groups must have the same npsi");
        }

        group.r_indices = static_cast<const std::int64_t*>(group.r_indices_info.ptr);
        group.h_positions = static_cast<const std::int64_t*>(group.h_positions_info.ptr);
        group.radial = static_cast<const Complex128*>(group.radial_info.ptr);
        group.angular = static_cast<const Complex128*>(group.angular_info.ptr);
        for (std::int64_t ir = 0; ir < group.nr; ++ir) {
            if (group.r_indices[ir] < 0 || group.r_indices[ir] >= nrho) {
                throw std::invalid_argument("rho group index out of bounds");
            }
        }
        for (std::int64_t ih = 0; ih < group.nh; ++ih) {
            if (group.h_positions[ih] < 0 || group.h_positions[ih] >= nh_max) {
                throw std::invalid_argument("harmonic position out of bounds");
            }
        }
        groups.push_back(std::move(group));
    }

    auto out = ComplexArray({batch, nrho * npsi * nz});
    py::buffer_info out_info = out.request();

    const auto* coeff = static_cast<const Complex128*>(coeff_info.ptr);
    const auto* defocus = static_cast<const Complex128*>(defocus_info.ptr);
    auto* out_ptr = static_cast<Complex128*>(out_info.ptr);

    {
        py::gil_scoped_release release;
        for (const RhoDependentGroup& group : groups) {
            unsigned int threads = 1;
            if (requested_threads > 0) {
                threads = static_cast<unsigned int>(requested_threads);
            } else {
                threads = choose_thread_count(batch * group.nr, ntheta * group.nh * npsi);
            }
            parallel_for(batch * group.nr, threads, [&](std::int64_t start, std::int64_t stop) {
                std::vector<double> angular_temp_re(static_cast<std::size_t>(npsi));
                std::vector<double> angular_temp_im(static_cast<std::size_t>(npsi));
                std::vector<double> out_re(static_cast<std::size_t>(npsi * nz));
                std::vector<double> out_im(static_cast<std::size_t>(npsi * nz));
                for (std::int64_t task = start; task < stop; ++task) {
                    const std::int64_t b = task / group.nr;
                    const std::int64_t local_r = task - b * group.nr;
                    const std::int64_t r = group.r_indices[local_r];
                    Complex128* out_base =
                        out_ptr + b * (nrho * npsi * nz) + r * npsi * nz;
                    std::fill(out_re.begin(), out_re.end(), 0.0);
                    std::fill(out_im.begin(), out_im.end(), 0.0);

                    for (std::int64_t t = 0; t < ntheta; ++t) {
                        std::fill(angular_temp_re.begin(), angular_temp_re.end(), 0.0);
                        std::fill(angular_temp_im.begin(), angular_temp_im.end(), 0.0);
                        for (std::int64_t h = 0; h < group.nh; ++h) {
                            const std::int64_t hpos = group.h_positions[h];
                            const Complex128 c =
                                coeff[(b * ntheta + t) * nh_max + hpos];
                            const Complex128 q =
                                group.radial[(t * group.nh + h) * group.nr + local_r];
                            const double scale_re =
                                c.real() * q.real() - c.imag() * q.imag();
                            const double scale_im =
                                c.real() * q.imag() + c.imag() * q.real();
                            const Complex128* angular_row = group.angular + h * npsi;
                            for (std::int64_t p = 0; p < npsi; ++p) {
                                const Complex128 a = angular_row[p];
                                angular_temp_re[static_cast<std::size_t>(p)] +=
                                    scale_re * a.real() - scale_im * a.imag();
                                angular_temp_im[static_cast<std::size_t>(p)] +=
                                    scale_re * a.imag() + scale_im * a.real();
                            }
                        }
                        const Complex128* defocus_row = defocus + t * nz;
                        for (std::int64_t p = 0; p < npsi; ++p) {
                            const double value_re =
                                angular_temp_re[static_cast<std::size_t>(p)];
                            const double value_im =
                                angular_temp_im[static_cast<std::size_t>(p)];
                            for (std::int64_t z = 0; z < nz; ++z) {
                                const Complex128 d = defocus_row[z];
                                const std::size_t out_idx =
                                    static_cast<std::size_t>(p * nz + z);
                                out_re[out_idx] +=
                                    value_re * d.real() - value_im * d.imag();
                                out_im[out_idx] +=
                                    value_re * d.imag() + value_im * d.real();
                            }
                        }
                    }
                    for (std::int64_t pz = 0; pz < npsi * nz; ++pz) {
                        const std::size_t idx = static_cast<std::size_t>(pz);
                        out_base[pz] = Complex128{out_re[idx], out_im[idx]};
                    }
                }
            });
        }
    }

    return out;
}

ComplexArray positive_rho_dependent_contract_many_fused(
    ComplexArray coeff_positive_array,
    ComplexArray coeff_negative_array,
    py::list r_indices_arrays,
    py::list h_position_arrays,
    py::list radial_arrays,
    py::list angular_positive_arrays,
    py::list angular_negative_arrays,
    ComplexArray defocus_array,
    std::int64_t nrho,
    std::int64_t requested_threads
) {
    const py::buffer_info coeff_positive_info = coeff_positive_array.request();
    const py::buffer_info coeff_negative_info = coeff_negative_array.request();
    const py::buffer_info defocus_info = defocus_array.request();
    if (coeff_positive_info.ndim != 3 || coeff_negative_info.ndim != 3) {
        throw std::invalid_argument(
            "coeff_positive and coeff_negative must have shape (batch, ntheta, nh_max)"
        );
    }
    if (coeff_positive_info.shape != coeff_negative_info.shape) {
        throw std::invalid_argument("positive and negative coefficient shapes differ");
    }
    if (defocus_info.ndim != 2) {
        throw std::invalid_argument("defocus must have shape (ntheta, nz)");
    }
    if (coeff_positive_info.shape[1] != defocus_info.shape[0]) {
        throw std::invalid_argument("coeff and defocus ntheta dimensions differ");
    }
    if (nrho <= 0) {
        throw std::invalid_argument("nrho must be positive");
    }

    const std::size_t group_count = py::len(r_indices_arrays);
    if (group_count == 0) {
        throw std::invalid_argument("at least one positive rho-dependent group is required");
    }
    if (py::len(h_position_arrays) != group_count ||
        py::len(radial_arrays) != group_count ||
        py::len(angular_positive_arrays) != group_count ||
        py::len(angular_negative_arrays) != group_count) {
        throw std::invalid_argument(
            "positive rho-dependent group lists must have equal length"
        );
    }

    const std::int64_t batch =
        static_cast<std::int64_t>(coeff_positive_info.shape[0]);
    const std::int64_t ntheta =
        static_cast<std::int64_t>(coeff_positive_info.shape[1]);
    const std::int64_t nh_max =
        static_cast<std::int64_t>(coeff_positive_info.shape[2]);
    const std::int64_t nz = static_cast<std::int64_t>(defocus_info.shape[1]);

    std::vector<PositiveRhoDependentGroup> groups;
    groups.reserve(group_count);
    std::int64_t npsi = -1;
    for (std::size_t ig = 0; ig < group_count; ++ig) {
        PositiveRhoDependentGroup group{
            py::cast<Int64Array>(r_indices_arrays[ig]),
            py::cast<Int64Array>(h_position_arrays[ig]),
            py::cast<ComplexArray>(radial_arrays[ig]),
            py::cast<ComplexArray>(angular_positive_arrays[ig]),
            py::cast<ComplexArray>(angular_negative_arrays[ig]),
            py::buffer_info(),
            py::buffer_info(),
            py::buffer_info(),
            py::buffer_info(),
            py::buffer_info(),
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            0,
            0,
        };
        group.r_indices_info = group.r_indices_array.request();
        group.h_positions_info = group.h_positions_array.request();
        group.radial_info = group.radial_array.request();
        group.angular_positive_info = group.angular_positive_array.request();
        group.angular_negative_info = group.angular_negative_array.request();

        if (group.r_indices_info.ndim != 1 || group.h_positions_info.ndim != 1) {
            throw std::invalid_argument("r_indices and h_positions must be 1D arrays");
        }
        if (group.radial_info.ndim != 3) {
            throw std::invalid_argument("radial group must have shape (ntheta, nh, nr)");
        }
        if (group.angular_positive_info.ndim != 2 ||
            group.angular_negative_info.ndim != 2) {
            throw std::invalid_argument("angular groups must have shape (nh, npsi)");
        }

        group.nr = static_cast<std::int64_t>(group.r_indices_info.shape[0]);
        group.nh = static_cast<std::int64_t>(group.h_positions_info.shape[0]);
        if (group.radial_info.shape[0] != ntheta ||
            group.radial_info.shape[1] != group.nh ||
            group.radial_info.shape[2] != group.nr ||
            group.angular_positive_info.shape[0] != group.nh ||
            group.angular_negative_info.shape[0] != group.nh ||
            group.angular_positive_info.shape[1] != group.angular_negative_info.shape[1]) {
            throw std::invalid_argument(
                "inconsistent positive rho-dependent group shapes"
            );
        }
        if (npsi < 0) {
            npsi = static_cast<std::int64_t>(group.angular_positive_info.shape[1]);
        } else if (group.angular_positive_info.shape[1] != npsi) {
            throw std::invalid_argument("all angular groups must have the same npsi");
        }

        group.r_indices = static_cast<const std::int64_t*>(group.r_indices_info.ptr);
        group.h_positions =
            static_cast<const std::int64_t*>(group.h_positions_info.ptr);
        group.radial = static_cast<const Complex128*>(group.radial_info.ptr);
        group.angular_positive =
            static_cast<const Complex128*>(group.angular_positive_info.ptr);
        group.angular_negative =
            static_cast<const Complex128*>(group.angular_negative_info.ptr);
        for (std::int64_t ir = 0; ir < group.nr; ++ir) {
            if (group.r_indices[ir] < 0 || group.r_indices[ir] >= nrho) {
                throw std::invalid_argument("rho group index out of bounds");
            }
        }
        for (std::int64_t ih = 0; ih < group.nh; ++ih) {
            if (group.h_positions[ih] < 0 || group.h_positions[ih] >= nh_max) {
                throw std::invalid_argument("harmonic position out of bounds");
            }
        }
        groups.push_back(std::move(group));
    }

    auto out = ComplexArray({batch, nrho * npsi * nz});
    py::buffer_info out_info = out.request();

    const auto* coeff_positive =
        static_cast<const Complex128*>(coeff_positive_info.ptr);
    const auto* coeff_negative =
        static_cast<const Complex128*>(coeff_negative_info.ptr);
    const auto* defocus = static_cast<const Complex128*>(defocus_info.ptr);
    auto* out_ptr = static_cast<Complex128*>(out_info.ptr);

    {
        py::gil_scoped_release release;
        for (const PositiveRhoDependentGroup& group : groups) {
            unsigned int threads = 1;
            if (requested_threads > 0) {
                threads = static_cast<unsigned int>(requested_threads);
            } else {
                threads = choose_thread_count(batch * group.nr, ntheta * group.nh * npsi);
            }
            parallel_for(batch * group.nr, threads, [&](std::int64_t start, std::int64_t stop) {
                std::vector<double> angular_temp_re(static_cast<std::size_t>(npsi));
                std::vector<double> angular_temp_im(static_cast<std::size_t>(npsi));
                std::vector<double> out_re(static_cast<std::size_t>(npsi * nz));
                std::vector<double> out_im(static_cast<std::size_t>(npsi * nz));
                for (std::int64_t task = start; task < stop; ++task) {
                    const std::int64_t b = task / group.nr;
                    const std::int64_t local_r = task - b * group.nr;
                    const std::int64_t r = group.r_indices[local_r];
                    Complex128* out_base =
                        out_ptr + b * (nrho * npsi * nz) + r * npsi * nz;
                    std::fill(out_re.begin(), out_re.end(), 0.0);
                    std::fill(out_im.begin(), out_im.end(), 0.0);

                    for (std::int64_t t = 0; t < ntheta; ++t) {
                        std::fill(angular_temp_re.begin(), angular_temp_re.end(), 0.0);
                        std::fill(angular_temp_im.begin(), angular_temp_im.end(), 0.0);
                        for (std::int64_t h = 0; h < group.nh; ++h) {
                            const std::int64_t hpos = group.h_positions[h];
                            const Complex128 q =
                                group.radial[(t * group.nh + h) * group.nr + local_r];
                            const Complex128 c_pos =
                                coeff_positive[(b * ntheta + t) * nh_max + hpos];
                            const Complex128 c_neg =
                                coeff_negative[(b * ntheta + t) * nh_max + hpos];
                            const double scale_pos_re =
                                c_pos.real() * q.real() - c_pos.imag() * q.imag();
                            const double scale_pos_im =
                                c_pos.real() * q.imag() + c_pos.imag() * q.real();
                            const double scale_neg_re =
                                c_neg.real() * q.real() - c_neg.imag() * q.imag();
                            const double scale_neg_im =
                                c_neg.real() * q.imag() + c_neg.imag() * q.real();
                            const Complex128* angular_pos_row =
                                group.angular_positive + h * npsi;
                            const Complex128* angular_neg_row =
                                group.angular_negative + h * npsi;
                            for (std::int64_t p = 0; p < npsi; ++p) {
                                const Complex128 a_pos = angular_pos_row[p];
                                const Complex128 a_neg = angular_neg_row[p];
                                angular_temp_re[static_cast<std::size_t>(p)] +=
                                    scale_pos_re * a_pos.real() -
                                    scale_pos_im * a_pos.imag() +
                                    scale_neg_re * a_neg.real() -
                                    scale_neg_im * a_neg.imag();
                                angular_temp_im[static_cast<std::size_t>(p)] +=
                                    scale_pos_re * a_pos.imag() +
                                    scale_pos_im * a_pos.real() +
                                    scale_neg_re * a_neg.imag() +
                                    scale_neg_im * a_neg.real();
                            }
                        }
                        const Complex128* defocus_row = defocus + t * nz;
                        for (std::int64_t p = 0; p < npsi; ++p) {
                            const double value_re =
                                angular_temp_re[static_cast<std::size_t>(p)];
                            const double value_im =
                                angular_temp_im[static_cast<std::size_t>(p)];
                            for (std::int64_t z = 0; z < nz; ++z) {
                                const Complex128 d = defocus_row[z];
                                const std::size_t out_idx =
                                    static_cast<std::size_t>(p * nz + z);
                                out_re[out_idx] +=
                                    value_re * d.real() - value_im * d.imag();
                                out_im[out_idx] +=
                                    value_re * d.imag() + value_im * d.real();
                            }
                        }
                    }
                    for (std::int64_t pz = 0; pz < npsi * nz; ++pz) {
                        const std::size_t idx = static_cast<std::size_t>(pz);
                        out_base[pz] = Complex128{out_re[idx], out_im[idx]};
                    }
                }
            });
        }
    }

    return out;
}

ComplexArray positive_rho_dependent_contract_many_nocopy(
    ComplexArray coeff_full_array,
    Int64Array positive_indices_array,
    Int64Array negative_indices_array,
    py::list r_indices_arrays,
    py::list h_position_arrays,
    py::list radial_arrays,
    py::list angular_positive_arrays,
    py::list angular_negative_arrays,
    ComplexArray defocus_array,
    std::int64_t nrho,
    std::int64_t requested_threads
) {
    const py::buffer_info coeff_full_info = coeff_full_array.request();
    const py::buffer_info positive_indices_info = positive_indices_array.request();
    const py::buffer_info negative_indices_info = negative_indices_array.request();
    const py::buffer_info defocus_info = defocus_array.request();
    if (coeff_full_info.ndim != 3) {
        throw std::invalid_argument("coeff_full must have shape (batch, ntheta, nphi)");
    }
    if (positive_indices_info.ndim != 1 || negative_indices_info.ndim != 1) {
        throw std::invalid_argument("positive_indices and negative_indices must be 1D");
    }
    if (positive_indices_info.shape[0] != negative_indices_info.shape[0]) {
        throw std::invalid_argument("positive and negative index lengths differ");
    }
    if (defocus_info.ndim != 2) {
        throw std::invalid_argument("defocus must have shape (ntheta, nz)");
    }
    if (coeff_full_info.shape[1] != defocus_info.shape[0]) {
        throw std::invalid_argument("coeff and defocus ntheta dimensions differ");
    }
    if (nrho <= 0) {
        throw std::invalid_argument("nrho must be positive");
    }

    const std::size_t group_count = py::len(r_indices_arrays);
    if (group_count == 0) {
        throw std::invalid_argument("at least one positive rho-dependent group is required");
    }
    if (py::len(h_position_arrays) != group_count ||
        py::len(radial_arrays) != group_count ||
        py::len(angular_positive_arrays) != group_count ||
        py::len(angular_negative_arrays) != group_count) {
        throw std::invalid_argument(
            "positive rho-dependent group lists must have equal length"
        );
    }

    const std::int64_t batch = static_cast<std::int64_t>(coeff_full_info.shape[0]);
    const std::int64_t ntheta = static_cast<std::int64_t>(coeff_full_info.shape[1]);
    const std::int64_t nphi = static_cast<std::int64_t>(coeff_full_info.shape[2]);
    const std::int64_t nh_index =
        static_cast<std::int64_t>(positive_indices_info.shape[0]);
    const std::int64_t nz = static_cast<std::int64_t>(defocus_info.shape[1]);

    const auto* positive_indices =
        static_cast<const std::int64_t*>(positive_indices_info.ptr);
    const auto* negative_indices =
        static_cast<const std::int64_t*>(negative_indices_info.ptr);
    for (std::int64_t h = 0; h < nh_index; ++h) {
        const std::int64_t pos = positive_indices[h];
        const std::int64_t neg = negative_indices[h];
        if (pos < -1 || pos >= nphi || neg < -1 || neg >= nphi) {
            throw std::invalid_argument("harmonic coefficient index out of bounds");
        }
    }

    std::vector<PositiveRhoDependentGroup> groups;
    groups.reserve(group_count);
    std::int64_t npsi = -1;
    for (std::size_t ig = 0; ig < group_count; ++ig) {
        PositiveRhoDependentGroup group{
            py::cast<Int64Array>(r_indices_arrays[ig]),
            py::cast<Int64Array>(h_position_arrays[ig]),
            py::cast<ComplexArray>(radial_arrays[ig]),
            py::cast<ComplexArray>(angular_positive_arrays[ig]),
            py::cast<ComplexArray>(angular_negative_arrays[ig]),
            py::buffer_info(),
            py::buffer_info(),
            py::buffer_info(),
            py::buffer_info(),
            py::buffer_info(),
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            0,
            0,
        };
        group.r_indices_info = group.r_indices_array.request();
        group.h_positions_info = group.h_positions_array.request();
        group.radial_info = group.radial_array.request();
        group.angular_positive_info = group.angular_positive_array.request();
        group.angular_negative_info = group.angular_negative_array.request();

        if (group.r_indices_info.ndim != 1 || group.h_positions_info.ndim != 1) {
            throw std::invalid_argument("r_indices and h_positions must be 1D arrays");
        }
        if (group.radial_info.ndim != 3) {
            throw std::invalid_argument("radial group must have shape (ntheta, nh, nr)");
        }
        if (group.angular_positive_info.ndim != 2 ||
            group.angular_negative_info.ndim != 2) {
            throw std::invalid_argument("angular groups must have shape (nh, npsi)");
        }

        group.nr = static_cast<std::int64_t>(group.r_indices_info.shape[0]);
        group.nh = static_cast<std::int64_t>(group.h_positions_info.shape[0]);
        if (group.radial_info.shape[0] != ntheta ||
            group.radial_info.shape[1] != group.nh ||
            group.radial_info.shape[2] != group.nr ||
            group.angular_positive_info.shape[0] != group.nh ||
            group.angular_negative_info.shape[0] != group.nh ||
            group.angular_positive_info.shape[1] != group.angular_negative_info.shape[1]) {
            throw std::invalid_argument(
                "inconsistent positive rho-dependent group shapes"
            );
        }
        if (npsi < 0) {
            npsi = static_cast<std::int64_t>(group.angular_positive_info.shape[1]);
        } else if (group.angular_positive_info.shape[1] != npsi) {
            throw std::invalid_argument("all angular groups must have the same npsi");
        }

        group.r_indices = static_cast<const std::int64_t*>(group.r_indices_info.ptr);
        group.h_positions =
            static_cast<const std::int64_t*>(group.h_positions_info.ptr);
        group.radial = static_cast<const Complex128*>(group.radial_info.ptr);
        group.angular_positive =
            static_cast<const Complex128*>(group.angular_positive_info.ptr);
        group.angular_negative =
            static_cast<const Complex128*>(group.angular_negative_info.ptr);
        for (std::int64_t ir = 0; ir < group.nr; ++ir) {
            if (group.r_indices[ir] < 0 || group.r_indices[ir] >= nrho) {
                throw std::invalid_argument("rho group index out of bounds");
            }
        }
        for (std::int64_t ih = 0; ih < group.nh; ++ih) {
            if (group.h_positions[ih] < 0 || group.h_positions[ih] >= nh_index) {
                throw std::invalid_argument("harmonic position out of bounds");
            }
        }
        groups.push_back(std::move(group));
    }

    auto out = ComplexArray({batch, nrho * npsi * nz});
    py::buffer_info out_info = out.request();

    const auto* coeff_full = static_cast<const Complex128*>(coeff_full_info.ptr);
    const auto* defocus = static_cast<const Complex128*>(defocus_info.ptr);
    auto* out_ptr = static_cast<Complex128*>(out_info.ptr);

    std::vector<std::int64_t> group_offsets;
    group_offsets.reserve(groups.size() + 1);
    group_offsets.push_back(0);
    std::int64_t total_tasks = 0;
    std::int64_t weighted_nh = 0;
    for (const PositiveRhoDependentGroup& group : groups) {
        const std::int64_t group_tasks = batch * group.nr;
        total_tasks += group_tasks;
        weighted_nh += group_tasks * group.nh;
        group_offsets.push_back(total_tasks);
    }
    const std::int64_t avg_nh =
        total_tasks > 0 ? std::max<std::int64_t>(1, weighted_nh / total_tasks) : 1;
    unsigned int threads = 1;
    if (requested_threads > 0) {
        threads = static_cast<unsigned int>(requested_threads);
    } else {
        threads = choose_thread_count(total_tasks, ntheta * avg_nh * npsi);
    }

    {
        py::gil_scoped_release release;
        parallel_for(total_tasks, threads, [&](std::int64_t start, std::int64_t stop) {
            std::vector<double> angular_temp_re(static_cast<std::size_t>(npsi));
            std::vector<double> angular_temp_im(static_cast<std::size_t>(npsi));
            std::vector<double> out_re(static_cast<std::size_t>(npsi * nz));
            std::vector<double> out_im(static_cast<std::size_t>(npsi * nz));
            auto group_it =
                std::upper_bound(group_offsets.begin(), group_offsets.end(), start);
            std::int64_t group_index =
                static_cast<std::int64_t>(group_it - group_offsets.begin()) - 1;
            std::int64_t group_stop = group_offsets[static_cast<std::size_t>(group_index + 1)];
            for (std::int64_t task = start; task < stop; ++task) {
                while (task >= group_stop) {
                    ++group_index;
                    group_stop = group_offsets[static_cast<std::size_t>(group_index + 1)];
                }
                const PositiveRhoDependentGroup& group =
                    groups[static_cast<std::size_t>(group_index)];
                const std::int64_t local_task = task - group_offsets[group_index];
                const std::int64_t b = local_task / group.nr;
                const std::int64_t local_r = local_task - b * group.nr;
                const std::int64_t r = group.r_indices[local_r];
                Complex128* out_base =
                    out_ptr + b * (nrho * npsi * nz) + r * npsi * nz;
                std::fill(out_re.begin(), out_re.end(), 0.0);
                std::fill(out_im.begin(), out_im.end(), 0.0);

                for (std::int64_t t = 0; t < ntheta; ++t) {
                    std::fill(angular_temp_re.begin(), angular_temp_re.end(), 0.0);
                    std::fill(angular_temp_im.begin(), angular_temp_im.end(), 0.0);
                    const Complex128* coeff_row =
                        coeff_full + (b * ntheta + t) * nphi;
                    for (std::int64_t h = 0; h < group.nh; ++h) {
                        const std::int64_t hpos = group.h_positions[h];
                        const std::int64_t pos_index = positive_indices[hpos];
                        const std::int64_t neg_index = negative_indices[hpos];
                        const Complex128 q =
                            group.radial[(t * group.nh + h) * group.nr + local_r];
                        const Complex128 c_pos =
                            pos_index >= 0 ? coeff_row[pos_index] : Complex128{0.0, 0.0};
                        const Complex128 c_neg =
                            neg_index >= 0 ? coeff_row[neg_index] : Complex128{0.0, 0.0};
                        const double scale_pos_re =
                            c_pos.real() * q.real() - c_pos.imag() * q.imag();
                        const double scale_pos_im =
                            c_pos.real() * q.imag() + c_pos.imag() * q.real();
                        const double scale_neg_re =
                            c_neg.real() * q.real() - c_neg.imag() * q.imag();
                        const double scale_neg_im =
                            c_neg.real() * q.imag() + c_neg.imag() * q.real();
                        const Complex128* angular_pos_row =
                            group.angular_positive + h * npsi;
                        const Complex128* angular_neg_row =
                            group.angular_negative + h * npsi;
                        for (std::int64_t p = 0; p < npsi; ++p) {
                            const Complex128 a_pos = angular_pos_row[p];
                            const Complex128 a_neg = angular_neg_row[p];
                            angular_temp_re[static_cast<std::size_t>(p)] +=
                                scale_pos_re * a_pos.real() -
                                scale_pos_im * a_pos.imag() +
                                scale_neg_re * a_neg.real() -
                                scale_neg_im * a_neg.imag();
                            angular_temp_im[static_cast<std::size_t>(p)] +=
                                scale_pos_re * a_pos.imag() +
                                scale_pos_im * a_pos.real() +
                                scale_neg_re * a_neg.imag() +
                                scale_neg_im * a_neg.real();
                        }
                    }
                    const Complex128* defocus_row = defocus + t * nz;
                    for (std::int64_t p = 0; p < npsi; ++p) {
                        const double value_re =
                            angular_temp_re[static_cast<std::size_t>(p)];
                        const double value_im =
                            angular_temp_im[static_cast<std::size_t>(p)];
                        for (std::int64_t z = 0; z < nz; ++z) {
                            const Complex128 d = defocus_row[z];
                            const std::size_t out_idx =
                                static_cast<std::size_t>(p * nz + z);
                            out_re[out_idx] +=
                                value_re * d.real() - value_im * d.imag();
                            out_im[out_idx] +=
                                value_re * d.imag() + value_im * d.real();
                        }
                    }
                }
                for (std::int64_t pz = 0; pz < npsi * nz; ++pz) {
                    const std::size_t idx = static_cast<std::size_t>(pz);
                    out_base[pz] = Complex128{out_re[idx], out_im[idx]};
                }
            }
        });
    }

    return out;
}

ComplexArray positive_rho_dependent_contract_many_cached_nocopy(
    ComplexArray coeff_full_array,
    Int64Array positive_indices_array,
    Int64Array negative_indices_array,
    py::list r_indices_arrays,
    py::list h_position_arrays,
    ComplexArray radial_array,
    ComplexArray angular_positive_array,
    ComplexArray angular_negative_array,
    ComplexArray defocus_array,
    std::int64_t nrho,
    std::int64_t requested_threads
) {
    const py::buffer_info coeff_full_info = coeff_full_array.request();
    const py::buffer_info positive_indices_info = positive_indices_array.request();
    const py::buffer_info negative_indices_info = negative_indices_array.request();
    const py::buffer_info radial_info = radial_array.request();
    const py::buffer_info angular_positive_info = angular_positive_array.request();
    const py::buffer_info angular_negative_info = angular_negative_array.request();
    const py::buffer_info defocus_info = defocus_array.request();
    if (coeff_full_info.ndim != 3) {
        throw std::invalid_argument("coeff_full must have shape (batch, ntheta, nphi)");
    }
    if (positive_indices_info.ndim != 1 || negative_indices_info.ndim != 1) {
        throw std::invalid_argument("positive_indices and negative_indices must be 1D");
    }
    if (positive_indices_info.shape[0] != negative_indices_info.shape[0]) {
        throw std::invalid_argument("positive and negative index lengths differ");
    }
    if (radial_info.ndim != 3) {
        throw std::invalid_argument("radial cache must have shape (ntheta, nh, nrho)");
    }
    if (angular_positive_info.ndim != 2 || angular_negative_info.ndim != 2) {
        throw std::invalid_argument("angular caches must have shape (nh, npsi)");
    }
    if (defocus_info.ndim != 2) {
        throw std::invalid_argument("defocus must have shape (ntheta, nz)");
    }
    if (nrho <= 0) {
        throw std::invalid_argument("nrho must be positive");
    }

    const std::int64_t batch = static_cast<std::int64_t>(coeff_full_info.shape[0]);
    const std::int64_t ntheta = static_cast<std::int64_t>(coeff_full_info.shape[1]);
    const std::int64_t nphi = static_cast<std::int64_t>(coeff_full_info.shape[2]);
    const std::int64_t nh_max =
        static_cast<std::int64_t>(positive_indices_info.shape[0]);
    const std::int64_t npsi =
        static_cast<std::int64_t>(angular_positive_info.shape[1]);
    const std::int64_t nz = static_cast<std::int64_t>(defocus_info.shape[1]);
    if (radial_info.shape[0] != ntheta ||
        radial_info.shape[1] != nh_max ||
        radial_info.shape[2] != nrho ||
        angular_positive_info.shape[0] != nh_max ||
        angular_negative_info.shape[0] != nh_max ||
        angular_negative_info.shape[1] != npsi ||
        defocus_info.shape[0] != ntheta) {
        throw std::invalid_argument("inconsistent cached positive rho-dependent shapes");
    }

    const auto* positive_indices =
        static_cast<const std::int64_t*>(positive_indices_info.ptr);
    const auto* negative_indices =
        static_cast<const std::int64_t*>(negative_indices_info.ptr);
    for (std::int64_t h = 0; h < nh_max; ++h) {
        const std::int64_t pos = positive_indices[h];
        const std::int64_t neg = negative_indices[h];
        if (pos < -1 || pos >= nphi || neg < -1 || neg >= nphi) {
            throw std::invalid_argument("harmonic coefficient index out of bounds");
        }
    }

    const std::size_t group_count = py::len(r_indices_arrays);
    if (group_count == 0 || py::len(h_position_arrays) != group_count) {
        throw std::invalid_argument("cached positive group lists must have equal length");
    }
    std::vector<Int64Array> r_indices_arrays_keepalive;
    std::vector<Int64Array> h_positions_arrays_keepalive;
    std::vector<py::buffer_info> r_indices_infos;
    std::vector<py::buffer_info> h_positions_infos;
    std::vector<const std::int64_t*> r_indices_ptrs;
    std::vector<const std::int64_t*> h_positions_ptrs;
    std::vector<std::int64_t> group_nr;
    std::vector<std::int64_t> group_nh;
    r_indices_arrays_keepalive.reserve(group_count);
    h_positions_arrays_keepalive.reserve(group_count);
    r_indices_infos.reserve(group_count);
    h_positions_infos.reserve(group_count);
    r_indices_ptrs.reserve(group_count);
    h_positions_ptrs.reserve(group_count);
    group_nr.reserve(group_count);
    group_nh.reserve(group_count);
    for (std::size_t ig = 0; ig < group_count; ++ig) {
        r_indices_arrays_keepalive.push_back(py::cast<Int64Array>(r_indices_arrays[ig]));
        h_positions_arrays_keepalive.push_back(py::cast<Int64Array>(h_position_arrays[ig]));
        r_indices_infos.push_back(r_indices_arrays_keepalive.back().request());
        h_positions_infos.push_back(h_positions_arrays_keepalive.back().request());
        if (r_indices_infos.back().ndim != 1 || h_positions_infos.back().ndim != 1) {
            throw std::invalid_argument("r_indices and h_positions must be 1D arrays");
        }
        const std::int64_t nr =
            static_cast<std::int64_t>(r_indices_infos.back().shape[0]);
        const std::int64_t nh =
            static_cast<std::int64_t>(h_positions_infos.back().shape[0]);
        const auto* r_ptr = static_cast<const std::int64_t*>(r_indices_infos.back().ptr);
        const auto* h_ptr = static_cast<const std::int64_t*>(h_positions_infos.back().ptr);
        for (std::int64_t ir = 0; ir < nr; ++ir) {
            if (r_ptr[ir] < 0 || r_ptr[ir] >= nrho) {
                throw std::invalid_argument("rho group index out of bounds");
            }
        }
        for (std::int64_t ih = 0; ih < nh; ++ih) {
            if (h_ptr[ih] < 0 || h_ptr[ih] >= nh_max) {
                throw std::invalid_argument("harmonic position out of bounds");
            }
        }
        r_indices_ptrs.push_back(r_ptr);
        h_positions_ptrs.push_back(h_ptr);
        group_nr.push_back(nr);
        group_nh.push_back(nh);
    }

    auto out = ComplexArray({batch, nrho * npsi * nz});
    py::buffer_info out_info = out.request();

    const auto* coeff_full = static_cast<const Complex128*>(coeff_full_info.ptr);
    const auto* radial = static_cast<const Complex128*>(radial_info.ptr);
    const auto* angular_positive =
        static_cast<const Complex128*>(angular_positive_info.ptr);
    const auto* angular_negative =
        static_cast<const Complex128*>(angular_negative_info.ptr);
    const auto* defocus = static_cast<const Complex128*>(defocus_info.ptr);
    auto* out_ptr = static_cast<Complex128*>(out_info.ptr);

    std::vector<std::int64_t> group_offsets;
    group_offsets.reserve(group_count + 1);
    group_offsets.push_back(0);
    std::int64_t total_tasks = 0;
    std::int64_t weighted_nh = 0;
    for (std::size_t ig = 0; ig < group_count; ++ig) {
        const std::int64_t tasks = batch * group_nr[ig];
        total_tasks += tasks;
        weighted_nh += tasks * group_nh[ig];
        group_offsets.push_back(total_tasks);
    }
    const std::int64_t avg_nh =
        total_tasks > 0 ? std::max<std::int64_t>(1, weighted_nh / total_tasks) : 1;
    unsigned int threads = 1;
    if (requested_threads > 0) {
        threads = static_cast<unsigned int>(requested_threads);
    } else {
        threads = choose_thread_count(total_tasks, ntheta * avg_nh * npsi);
    }

    {
        py::gil_scoped_release release;
        parallel_for(total_tasks, threads, [&](std::int64_t start, std::int64_t stop) {
            std::vector<double> angular_temp_re(static_cast<std::size_t>(npsi));
            std::vector<double> angular_temp_im(static_cast<std::size_t>(npsi));
            std::vector<double> out_re(static_cast<std::size_t>(npsi * nz));
            std::vector<double> out_im(static_cast<std::size_t>(npsi * nz));
            auto group_it =
                std::upper_bound(group_offsets.begin(), group_offsets.end(), start);
            std::int64_t group_index =
                static_cast<std::int64_t>(group_it - group_offsets.begin()) - 1;
            std::int64_t group_stop =
                group_offsets[static_cast<std::size_t>(group_index + 1)];
            for (std::int64_t task = start; task < stop; ++task) {
                while (task >= group_stop) {
                    ++group_index;
                    group_stop = group_offsets[static_cast<std::size_t>(group_index + 1)];
                }
                const std::size_t ig = static_cast<std::size_t>(group_index);
                const std::int64_t local_task = task - group_offsets[ig];
                const std::int64_t b = local_task / group_nr[ig];
                const std::int64_t local_r = local_task - b * group_nr[ig];
                const std::int64_t r = r_indices_ptrs[ig][local_r];
                Complex128* out_base =
                    out_ptr + b * (nrho * npsi * nz) + r * npsi * nz;
                std::fill(out_re.begin(), out_re.end(), 0.0);
                std::fill(out_im.begin(), out_im.end(), 0.0);

                for (std::int64_t t = 0; t < ntheta; ++t) {
                    std::fill(angular_temp_re.begin(), angular_temp_re.end(), 0.0);
                    std::fill(angular_temp_im.begin(), angular_temp_im.end(), 0.0);
                    const Complex128* coeff_row =
                        coeff_full + (b * ntheta + t) * nphi;
                    for (std::int64_t local_h = 0; local_h < group_nh[ig]; ++local_h) {
                        const std::int64_t hpos = h_positions_ptrs[ig][local_h];
                        const std::int64_t pos_index = positive_indices[hpos];
                        const std::int64_t neg_index = negative_indices[hpos];
                        const Complex128 q = radial[(t * nh_max + hpos) * nrho + r];
                        const Complex128 c_pos =
                            pos_index >= 0 ? coeff_row[pos_index] : Complex128{0.0, 0.0};
                        const Complex128 c_neg =
                            neg_index >= 0 ? coeff_row[neg_index] : Complex128{0.0, 0.0};
                        const double scale_pos_re =
                            c_pos.real() * q.real() - c_pos.imag() * q.imag();
                        const double scale_pos_im =
                            c_pos.real() * q.imag() + c_pos.imag() * q.real();
                        const double scale_neg_re =
                            c_neg.real() * q.real() - c_neg.imag() * q.imag();
                        const double scale_neg_im =
                            c_neg.real() * q.imag() + c_neg.imag() * q.real();
                        const Complex128* angular_pos_row =
                            angular_positive + hpos * npsi;
                        const Complex128* angular_neg_row =
                            angular_negative + hpos * npsi;
                        for (std::int64_t p = 0; p < npsi; ++p) {
                            const Complex128 a_pos = angular_pos_row[p];
                            const Complex128 a_neg = angular_neg_row[p];
                            angular_temp_re[static_cast<std::size_t>(p)] +=
                                scale_pos_re * a_pos.real() -
                                scale_pos_im * a_pos.imag() +
                                scale_neg_re * a_neg.real() -
                                scale_neg_im * a_neg.imag();
                            angular_temp_im[static_cast<std::size_t>(p)] +=
                                scale_pos_re * a_pos.imag() +
                                scale_pos_im * a_pos.real() +
                                scale_neg_re * a_neg.imag() +
                                scale_neg_im * a_neg.real();
                        }
                    }
                    const Complex128* defocus_row = defocus + t * nz;
                    for (std::int64_t p = 0; p < npsi; ++p) {
                        const double value_re =
                            angular_temp_re[static_cast<std::size_t>(p)];
                        const double value_im =
                            angular_temp_im[static_cast<std::size_t>(p)];
                        for (std::int64_t z = 0; z < nz; ++z) {
                            const Complex128 d = defocus_row[z];
                            const std::size_t out_idx =
                                static_cast<std::size_t>(p * nz + z);
                            out_re[out_idx] +=
                                value_re * d.real() - value_im * d.imag();
                            out_im[out_idx] +=
                                value_re * d.imag() + value_im * d.real();
                        }
                    }
                }
                for (std::int64_t pz = 0; pz < npsi * nz; ++pz) {
                    const std::size_t idx = static_cast<std::size_t>(pz);
                    out_base[pz] = Complex128{out_re[idx], out_im[idx]};
                }
            }
        });
    }

    return out;
}

ComplexArray separable_adjoint_contract(
    ComplexArray psi_contracted_array,
    ComplexArray radial_conj_array,
    ComplexArray defocus_conj_array,
    std::int64_t requested_threads
) {
    const py::buffer_info psi_info = psi_contracted_array.request();
    const py::buffer_info radial_info = radial_conj_array.request();
    const py::buffer_info defocus_info = defocus_conj_array.request();
    validate_adjoint_shapes(psi_info, radial_info, defocus_info);

    const std::int64_t nrho = static_cast<std::int64_t>(psi_info.shape[0]);
    const std::int64_t nh = static_cast<std::int64_t>(psi_info.shape[1]);
    const std::int64_t nz = static_cast<std::int64_t>(psi_info.shape[2]);
    const std::int64_t ntheta = static_cast<std::int64_t>(radial_info.shape[0]);

    auto out = ComplexArray({ntheta, nh});
    py::buffer_info out_info = out.request();

    const auto* psi = static_cast<const Complex128*>(psi_info.ptr);
    const auto* radial = static_cast<const Complex128*>(radial_info.ptr);
    const auto* defocus = static_cast<const Complex128*>(defocus_info.ptr);
    auto* out_ptr = static_cast<Complex128*>(out_info.ptr);

    unsigned int threads = 1;
    if (requested_threads > 0) {
        threads = static_cast<unsigned int>(requested_threads);
    } else {
        threads = choose_thread_count(ntheta * nh, nrho * nz);
    }

    {
        py::gil_scoped_release release;
        parallel_for(ntheta * nh, threads, [&](std::int64_t start, std::int64_t stop) {
            for (std::int64_t task = start; task < stop; ++task) {
                const std::int64_t t = task / nh;
                const std::int64_t h = task - t * nh;
                double accum_re = 0.0;
                double accum_im = 0.0;
                const Complex128* radial_base = radial + (t * nh + h) * nrho;
                const Complex128* defocus_row = defocus + t * nz;
                for (std::int64_t r = 0; r < nrho; ++r) {
                    const Complex128 q = radial_base[r];
                    const Complex128* psi_base = psi + (r * nh + h) * nz;
                    for (std::int64_t z = 0; z < nz; ++z) {
                        const Complex128 a = psi_base[z];
                        const Complex128 d = defocus_row[z];
                        const double aq_re =
                            a.real() * q.real() - a.imag() * q.imag();
                        const double aq_im =
                            a.real() * q.imag() + a.imag() * q.real();
                        accum_re += aq_re * d.real() - aq_im * d.imag();
                        accum_im += aq_re * d.imag() + aq_im * d.real();
                    }
                }
                out_ptr[t * nh + h] = Complex128{accum_re, accum_im};
            }
        });
    }

    return out;
}

}  // namespace

PYBIND11_MODULE(_cpp_high_na, m) {
    m.doc() = "C++ kernels for experimental high-NA Debye-Wolf benchmarks.";
    m.def(
        "separable_contract_many",
        &separable_contract_many,
        py::arg("coeff"),
        py::arg("radial"),
        py::arg("angular"),
        py::arg("defocus"),
        py::arg("threads") = 0,
        "Evaluate batched separable harmonic Debye-Wolf contractions."
    );
    m.def(
        "separable_contract_many_fused",
        &separable_contract_many_fused,
        py::arg("coeff"),
        py::arg("radial"),
        py::arg("angular"),
        py::arg("defocus"),
        py::arg("threads") = 0,
        "Evaluate batched separable harmonic Debye-Wolf contractions with fused defocus accumulation."
    );
    m.def(
        "separable_adjoint_contract",
        &separable_adjoint_contract,
        py::arg("psi_contracted"),
        py::arg("radial_conj"),
        py::arg("defocus_conj"),
        py::arg("threads") = 0,
        "Evaluate separable harmonic Debye-Wolf adjoint contractions."
    );
    m.def(
        "positive_separable_contract_many_fused",
        &positive_separable_contract_many_fused,
        py::arg("coeff_positive"),
        py::arg("coeff_negative"),
        py::arg("radial"),
        py::arg("angular_positive"),
        py::arg("angular_negative"),
        py::arg("defocus"),
        py::arg("threads") = 0,
        "Evaluate positive-mode separable harmonic Debye-Wolf contractions."
    );
    m.def(
        "rho_dependent_contract_many_fused",
        &rho_dependent_contract_many_fused,
        py::arg("coeff"),
        py::arg("r_indices"),
        py::arg("h_positions"),
        py::arg("radial"),
        py::arg("angular"),
        py::arg("defocus"),
        py::arg("nrho"),
        py::arg("threads") = 0,
        "Evaluate rho-dependent separable harmonic Debye-Wolf contractions."
    );
    m.def(
        "positive_rho_dependent_contract_many_fused",
        &positive_rho_dependent_contract_many_fused,
        py::arg("coeff_positive"),
        py::arg("coeff_negative"),
        py::arg("r_indices"),
        py::arg("h_positions"),
        py::arg("radial"),
        py::arg("angular_positive"),
        py::arg("angular_negative"),
        py::arg("defocus"),
        py::arg("nrho"),
        py::arg("threads") = 0,
        "Evaluate positive-mode rho-dependent Debye-Wolf contractions."
    );
    m.def(
        "positive_rho_dependent_contract_many_nocopy",
        &positive_rho_dependent_contract_many_nocopy,
        py::arg("coeff_full"),
        py::arg("positive_indices"),
        py::arg("negative_indices"),
        py::arg("r_indices"),
        py::arg("h_positions"),
        py::arg("radial"),
        py::arg("angular_positive"),
        py::arg("angular_negative"),
        py::arg("defocus"),
        py::arg("nrho"),
        py::arg("threads") = 0,
        "Evaluate positive-mode rho-dependent contractions without coefficient packing."
    );
    m.def(
        "positive_rho_dependent_contract_many_cached_nocopy",
        &positive_rho_dependent_contract_many_cached_nocopy,
        py::arg("coeff_full"),
        py::arg("positive_indices"),
        py::arg("negative_indices"),
        py::arg("r_indices"),
        py::arg("h_positions"),
        py::arg("radial"),
        py::arg("angular_positive"),
        py::arg("angular_negative"),
        py::arg("defocus"),
        py::arg("nrho"),
        py::arg("threads") = 0,
        "Evaluate positive rho-dependent contractions from a reusable basis cache."
    );
}
