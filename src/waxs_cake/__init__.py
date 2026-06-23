"""Prototype solvers for atomistic WAXS cake-map generation."""

from .geometry import ewald_ring
from .histogram import (
    BinnedStructure,
    cylindrical_flat_indices,
    encode_elements,
    make_cylindrical_histogram,
    make_cylindrical_histogram_from_flat_indices,
    make_cylindrical_histogram_from_indices,
    make_cylindrical_histogram_indexed,
)
from .odt_measured_contract import (
    OdtMeasuredData,
    PreparedOperatorDescriptor,
    ValidationReport,
    build_prepared_operator_from_contract,
    load_odt_measured_contract,
    save_odt_measured_contract,
    validate_odt_measured_contract,
)
from .physical_scaling import PhysicalGrid, choose_physical_grid, water_box_side_nm
from .presets import FAST_PRESET_NAMES, fast_preset_options
from .solvers import (
    KernelInterpolationTable,
    PreparedCakePlan,
    circular_fft_amplitude,
    direct_amplitude,
    estimate_bessel_cutoff,
    hybrid_amplitude,
    jacobi_anger_amplitude,
    nufft_amplitude,
    nufft_amplitude_chunked,
)

__all__ = [
    "BinnedStructure",
    "FAST_PRESET_NAMES",
    "KernelInterpolationTable",
    "PhysicalGrid",
    "OdtMeasuredData",
    "PreparedOperatorDescriptor",
    "ValidationReport",
    "build_prepared_operator_from_contract",
    "PreparedCakePlan",
    "choose_physical_grid",
    "circular_fft_amplitude",
    "cylindrical_flat_indices",
    "direct_amplitude",
    "encode_elements",
    "estimate_bessel_cutoff",
    "ewald_ring",
    "fast_preset_options",
    "hybrid_amplitude",
    "jacobi_anger_amplitude",
    "make_cylindrical_histogram",
    "make_cylindrical_histogram_from_flat_indices",
    "make_cylindrical_histogram_from_indices",
    "make_cylindrical_histogram_indexed",
    "load_odt_measured_contract",
    "nufft_amplitude",
    "nufft_amplitude_chunked",
    "save_odt_measured_contract",
    "validate_odt_measured_contract",
    "water_box_side_nm",
]
