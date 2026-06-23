"""Physical-size helpers for resolution-aware WAXS benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, pi, sqrt

from scipy.fft import next_fast_len


WATER_MOLECULES_PER_NM3 = 33.3679
WATER_ATOMS_PER_MOLECULE = 3


@dataclass(frozen=True)
class PhysicalGrid:
    """Resolution-aware simulation box and cylindrical histogram grid."""

    n_atoms: int
    box_side_nm: float
    r_max_nm: float
    z_range_nm: tuple[float, float]
    n_r: int
    n_z: int
    n_phi: int
    bin_width_nm: float
    qmax_inv_nm: float
    atoms_per_nm3: float
    water_molecules: float
    n_phi_detector: int
    n_phi_bandlimit: int
    n_phi_arc: int

    @property
    def height_nm(self) -> float:
        return self.z_range_nm[1] - self.z_range_nm[0]

    @property
    def n_bins_per_element(self) -> int:
        return self.n_r * self.n_z * self.n_phi

    @property
    def dr_nm(self) -> float:
        return self.r_max_nm / self.n_r

    @property
    def dz_nm(self) -> float:
        return self.height_nm / self.n_z

    @property
    def outer_arc_nm(self) -> float:
        return (2.0 * pi * self.r_max_nm) / self.n_phi


def q_to_inv_nm(q_value: float, unit: str) -> float:
    """Convert a q value to nm^-1."""

    if unit == "inv_nm":
        return float(q_value)
    if unit == "inv_angstrom":
        return 10.0 * float(q_value)
    raise ValueError("unit must be 'inv_nm' or 'inv_angstrom'")


def water_box_side_nm(
    n_atoms: int,
    *,
    atoms_per_molecule: int = WATER_ATOMS_PER_MOLECULE,
    molecules_per_nm3: float = WATER_MOLECULES_PER_NM3,
) -> float:
    """Return the cube side length for water-equivalent density."""

    if n_atoms <= 0:
        raise ValueError("n_atoms must be positive")
    if atoms_per_molecule <= 0:
        raise ValueError("atoms_per_molecule must be positive")
    if molecules_per_nm3 <= 0.0:
        raise ValueError("molecules_per_nm3 must be positive")
    n_molecules = float(n_atoms) / float(atoms_per_molecule)
    volume_nm3 = n_molecules / float(molecules_per_nm3)
    return volume_nm3 ** (1.0 / 3.0)


def _ceil_even(value: float) -> int:
    out = int(ceil(value))
    return out if out % 2 == 0 else out + 1


def _next_fft_friendly_even(value: int) -> int:
    """Return an even FFT-friendly length no smaller than ``value``."""

    target = max(2, int(value))
    if target % 2:
        target += 1
    while True:
        candidate = int(next_fast_len(target, real=True))
        if candidate % 2 == 0:
            return candidate
        target = candidate + 1


def choose_physical_grid(
    n_atoms: int,
    *,
    bin_width_nm: float,
    qmax: float,
    q_unit: str = "inv_angstrom",
    n_phi_detector: int = 180,
    harmonic_margin: int = 16,
    angular_rule: str = "bandlimit",
    atoms_per_molecule: int = WATER_ATOMS_PER_MOLECULE,
    molecules_per_nm3: float = WATER_MOLECULES_PER_NM3,
) -> PhysicalGrid:
    """Choose a physically scaled cube and cylindrical histogram grid.

    ``bin_width_nm`` controls radial and axial bin sizes. The angular grid is
    never smaller than ``n_phi_detector``. With ``angular_rule='bandlimit'`` it
    also resolves Fourier modes up to roughly ``qmax * r_max``; with
    ``angular_rule='arc'`` it additionally keeps the outer azimuthal arc length
    no larger than ``bin_width_nm``.
    """

    if bin_width_nm <= 0.0:
        raise ValueError("bin_width_nm must be positive")
    if n_phi_detector <= 0:
        raise ValueError("n_phi_detector must be positive")
    if harmonic_margin < 0:
        raise ValueError("harmonic_margin must be non-negative")
    if angular_rule not in {"bandlimit", "arc"}:
        raise ValueError("angular_rule must be 'bandlimit' or 'arc'")

    box_side = water_box_side_nm(
        n_atoms,
        atoms_per_molecule=atoms_per_molecule,
        molecules_per_nm3=molecules_per_nm3,
    )
    r_max = box_side / sqrt(2.0)
    qmax_inv_nm = q_to_inv_nm(qmax, q_unit)
    atoms_per_nm3 = float(atoms_per_molecule) * float(molecules_per_nm3)

    n_r = max(1, int(ceil(r_max / bin_width_nm)))
    n_z = max(1, int(ceil(box_side / bin_width_nm)))
    n_phi_bandlimit = _ceil_even(2.0 * qmax_inv_nm * r_max + 2.0 * harmonic_margin)
    n_phi_arc = _ceil_even((2.0 * pi * r_max) / bin_width_nm)
    n_phi_min = max(n_phi_detector, n_phi_bandlimit)
    if angular_rule == "arc":
        n_phi_min = max(n_phi_min, n_phi_arc)
    n_phi = _next_fft_friendly_even(n_phi_min)

    return PhysicalGrid(
        n_atoms=int(n_atoms),
        box_side_nm=box_side,
        r_max_nm=r_max,
        z_range_nm=(-0.5 * box_side, 0.5 * box_side),
        n_r=n_r,
        n_z=n_z,
        n_phi=int(n_phi),
        bin_width_nm=float(bin_width_nm),
        qmax_inv_nm=qmax_inv_nm,
        atoms_per_nm3=atoms_per_nm3,
        water_molecules=float(n_atoms) / float(atoms_per_molecule),
        n_phi_detector=int(n_phi_detector),
        n_phi_bandlimit=n_phi_bandlimit,
        n_phi_arc=n_phi_arc,
    )
