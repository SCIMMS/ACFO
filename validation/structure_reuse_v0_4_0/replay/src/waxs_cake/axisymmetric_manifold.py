"""Contract for axisymmetric reciprocal-space sampling manifolds.

The contract deliberately describes sampling geometry, not physical validity.
Any finite meridional curve can be sampled by the prepared circular solver;
calling a curve ``dispersion-derived`` records provenance supplied by the
caller, rather than certifying a wave model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, TYPE_CHECKING

import numpy as np

from .geometry import ewald_ring

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from .histogram import BinnedStructure
    from .solvers import PreparedCakePlan


ManifoldInterpretation = Literal["sampling", "dispersion-derived"]
MeridionalCallback = Callable[["NDArray[np.float64]"], tuple["ArrayLike", "ArrayLike"]]


def _readonly_vector(value: "ArrayLike", *, field: str) -> "NDArray[np.float64]":
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{field} must be a one-dimensional array")
    if array.size == 0:
        raise ValueError(f"{field} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field} must contain only finite values")
    array.setflags(write=False)
    return array


def parameter_trapezoid_weights(u: "ArrayLike") -> "NDArray[np.float64]":
    """Return trapezoid weights for integration with respect to ``u``.

    These are parameter-measure weights only. They do not include the surface
    Jacobian of a surface of revolution.
    """

    u_array = _readonly_vector(u, field="u")
    if u_array.size < 2:
        raise ValueError("at least two u samples are required for quadrature")
    if np.any(np.diff(u_array) <= 0.0):
        raise ValueError("u must be strictly increasing")

    weights = np.empty_like(u_array)
    weights[0] = 0.5 * (u_array[1] - u_array[0])
    weights[-1] = 0.5 * (u_array[-1] - u_array[-2])
    if u_array.size > 2:
        weights[1:-1] = 0.5 * (u_array[2:] - u_array[:-2])
    weights.setflags(write=False)
    return weights


def surface_radial_weights(
    q_perp: "ArrayLike",
    dq_perp_du: "ArrayLike",
    dq_z_du: "ArrayLike",
    u_weights: "ArrayLike",
) -> "NDArray[np.float64]":
    """Return radial weights for integration over a surface of revolution.

    The returned values represent

    ``w_u * Q_perp * sqrt((dQ_perp/du)^2 + (dQ_z/du)^2)``.

    The azimuthal quadrature weight is intentionally not included.
    """

    q_perp_array = _readonly_vector(q_perp, field="q_perp")
    dq_perp_array = _readonly_vector(dq_perp_du, field="dq_perp_du")
    dq_z_array = _readonly_vector(dq_z_du, field="dq_z_du")
    u_weight_array = _readonly_vector(u_weights, field="u_weights")
    shape = q_perp_array.shape
    if any(array.shape != shape for array in (dq_perp_array, dq_z_array, u_weight_array)):
        raise ValueError("surface-weight inputs must have matching shapes")
    if np.any(q_perp_array < 0.0):
        raise ValueError("q_perp must be non-negative")
    if np.any(u_weight_array < 0.0):
        raise ValueError("u_weights must be non-negative")

    weights = u_weight_array * q_perp_array * np.hypot(dq_perp_array, dq_z_array)
    weights.setflags(write=False)
    return weights


@dataclass(frozen=True)
class AxisymmetricManifold:
    """One meridional curve and its optional discrete data-space weights.

    ``u`` labels samples and must be strictly increasing, but it need not be
    uniformly spaced and neither ``q_perp`` nor ``q_z`` must be monotone.
    Repeated reciprocal-space nodes are allowed because repeated measurements
    are a valid point-evaluation layout.

    ``data_weights`` define the radial factor in a weighted data-space inner
    product. ``None`` means the Euclidean discrete inner product. Point
    evaluation never applies a surface Jacobian implicitly.
    """

    u: "ArrayLike"
    q_perp: "ArrayLike"
    q_z: "ArrayLike"
    data_weights: "ArrayLike | None" = None
    name: str = "axisymmetric-sampling-manifold"
    interpretation: ManifoldInterpretation = "sampling"
    frequency_units: str = "inverse_length"

    def __post_init__(self) -> None:
        u = _readonly_vector(self.u, field="u")
        q_perp = _readonly_vector(self.q_perp, field="q_perp")
        q_z = _readonly_vector(self.q_z, field="q_z")
        if q_perp.shape != u.shape or q_z.shape != u.shape:
            raise ValueError("u, q_perp, and q_z must have matching shapes")
        if u.size > 1 and np.any(np.diff(u) <= 0.0):
            raise ValueError("u must be strictly increasing")
        if np.any(q_perp < 0.0):
            raise ValueError("q_perp must be non-negative")
        if self.interpretation not in {"sampling", "dispersion-derived"}:
            raise ValueError("interpretation must be 'sampling' or 'dispersion-derived'")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        if not isinstance(self.frequency_units, str) or not self.frequency_units.strip():
            raise ValueError("frequency_units must be a non-empty string")

        data_weights = None
        if self.data_weights is not None:
            data_weights = _readonly_vector(self.data_weights, field="data_weights")
            if data_weights.shape != u.shape:
                raise ValueError("data_weights must have the same shape as u")
            if np.any(data_weights < 0.0):
                raise ValueError("data_weights must be non-negative")
            if not np.any(data_weights > 0.0):
                raise ValueError("data_weights must contain at least one positive value")

        object.__setattr__(self, "u", u)
        object.__setattr__(self, "q_perp", q_perp)
        object.__setattr__(self, "q_z", q_z)
        object.__setattr__(self, "data_weights", data_weights)

    @property
    def n_u(self) -> int:
        return int(self.u.size)

    @property
    def q_norm(self) -> "NDArray[np.float64]":
        """Return ``sqrt(q_perp**2 + q_z**2)`` for form-factor evaluation."""

        result = np.hypot(self.q_perp, self.q_z)
        result.setflags(write=False)
        return result

    @property
    def resolved_data_weights(self) -> "NDArray[np.float64]":
        """Return explicit radial data weights, using ones for Euclidean data."""

        if self.data_weights is not None:
            return self.data_weights
        weights = np.ones(self.n_u, dtype=np.float64)
        weights.setflags(write=False)
        return weights

    def target_nodes(self, phi: "ArrayLike") -> "NDArray[np.float64]":
        """Return reciprocal-space nodes with shape ``(n_u, n_phi, 3)``."""

        phi_array = _readonly_vector(phi, field="phi")
        nodes = np.empty((self.n_u, phi_array.size, 3), dtype=np.float64)
        nodes[..., 0] = self.q_perp[:, None] * np.cos(phi_array)[None, :]
        nodes[..., 1] = self.q_perp[:, None] * np.sin(phi_array)[None, :]
        nodes[..., 2] = self.q_z[:, None]
        return nodes

    @staticmethod
    def uniform_phi(n_phi: int, *, center_offset: float = 0.5) -> "NDArray[np.float64]":
        """Return the periodic azimuth grid used by the circular FFT backend."""

        n_phi = int(n_phi)
        if n_phi <= 0:
            raise ValueError("n_phi must be positive")
        if not np.isfinite(center_offset):
            raise ValueError("center_offset must be finite")
        phi = (np.arange(n_phi, dtype=np.float64) + center_offset) * (2.0 * np.pi / n_phi)
        phi.setflags(write=False)
        return phi

    @classmethod
    def from_callback(
        cls,
        u: "ArrayLike",
        callback: MeridionalCallback,
        **kwargs: object,
    ) -> "AxisymmetricManifold":
        """Sample an analytic meridional callback on an explicit ``u`` grid."""

        u_array = _readonly_vector(u, field="u")
        q_perp, q_z = callback(u_array)
        return cls(u_array, q_perp, q_z, **kwargs)

    @classmethod
    def ewald_sphere(
        cls,
        q: "ArrayLike",
        wavelength: float,
        **kwargs: object,
    ) -> "AxisymmetricManifold":
        """Build the existing elastic Ewald-sphere ring family."""

        q_array = _readonly_vector(q, field="q")
        q_perp, q_z = ewald_ring(q_array, wavelength)
        kwargs.setdefault("name", "elastic-ewald-sphere")
        kwargs.setdefault("interpretation", "dispersion-derived")
        return cls(q_array, q_perp, q_z, **kwargs)


def prepare_axisymmetric_plan(
    binned: "BinnedStructure",
    manifold: AxisymmetricManifold,
    **kwargs: object,
) -> "PreparedCakePlan":
    """Adapt a manifold to the current :class:`PreparedCakePlan` backend.

    The legacy constructor still requires ``q`` and ``wavelength``. With
    explicit ``q_perp`` and ``q_z``, wavelength is not used for geometry; the
    adapter supplies ``q_norm`` as ``q`` so element form factors retain their
    physical magnitude argument.
    """

    reserved = {"q", "wavelength", "q_perp", "q_z"}.intersection(kwargs)
    if reserved:
        fields = ", ".join(sorted(reserved))
        raise TypeError(f"geometry arguments are owned by manifold: {fields}")

    from .solvers import PreparedCakePlan

    return PreparedCakePlan(
        binned,
        manifold.q_norm,
        1.0,
        q_perp=manifold.q_perp,
        q_z=manifold.q_z,
        **kwargs,
    )
