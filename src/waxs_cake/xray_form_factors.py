"""X-ray atomic form-factor helpers for WAXS validation workflows."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _periodictable_element(symbol: str):
    try:
        import periodictable as pt
    except ImportError as exc:  # pragma: no cover - exercised only without dependency
        raise ImportError(
            "xray_f0 form factors require the 'periodictable' package. "
            "Install it with `pip install periodictable`."
        ) from exc

    try:
        element = getattr(pt, symbol)
    except AttributeError as exc:
        raise ValueError(f"periodictable does not know element {symbol!r}") from exc

    if getattr(element, "xray", None) is None:
        raise ValueError(f"periodictable has no X-ray form factor for {symbol!r}")
    return element


def xray_f0(element: str, q_inv_nm: np.ndarray) -> np.ndarray:
    """Return neutral-atom elastic X-ray f0 values for solver q in 1/nm.

    ``periodictable`` evaluates Waasmaier-Kirfel f0 as a function of
    Q in 1/Angstrom. The WAXS solvers use q in 1/nm internally, so this
    helper performs the unit conversion at the validation boundary.
    """

    q_inv_nm = np.asarray(q_inv_nm, dtype=np.float64)
    q_inv_angstrom = q_inv_nm / 10.0
    values = np.asarray(_periodictable_element(str(element)).xray.f0(q_inv_angstrom))
    values = values.astype(np.float64, copy=False)
    if values.shape != q_inv_nm.shape:
        values = np.broadcast_to(values, q_inv_nm.shape).astype(np.float64, copy=False)
    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"xray_f0 produced non-finite values for {element!r}; "
            "check that q is within the periodictable f0 range"
        )
    return values


def xray_f0_form_factors(
    elements: Sequence[str] | np.ndarray,
    q_inv_nm: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build a solver-compatible element-to-f0(q) mapping."""

    unique = sorted(set(str(element) for element in elements))
    return {element: xray_f0(element, q_inv_nm) for element in unique}
