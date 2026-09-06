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


def torch_xray_f0_form_factors(
    elements: Sequence[str] | np.ndarray,
    q_inv_nm,
    *,
    torch=None,
):
    """Differentiable Waasmaier--Kirfel f0 values for real torch q tensors.

    The coefficient table is read from ``periodictable`` while evaluation is
    performed with torch operations, preserving the q-dependent chain rule.
    """

    if torch is None:
        import torch as torch_module

        torch = torch_module
    from periodictable import cromermann

    q = torch.as_tensor(q_inv_nm)
    if not q.is_floating_point() or not bool(torch.all(torch.isfinite(q)).item()):
        raise ValueError("q_inv_nm must be a finite real floating-point tensor")
    stol = q / (40.0 * torch.pi)
    if bool(torch.any(stol.detach() > 6.0).item()):
        raise ValueError("q exceeds the Waasmaier--Kirfel validity range")
    output = {}
    for element in sorted(set(str(value) for value in elements)):
        formula = cromermann.getCMformula(element)
        a = torch.as_tensor(formula.a, dtype=q.dtype, device=q.device)
        b = torch.as_tensor(formula.b, dtype=q.dtype, device=q.device)
        c = torch.as_tensor(formula.c, dtype=q.dtype, device=q.device)
        output[element] = torch.sum(
            a[:, None] * torch.exp(-b[:, None] * stol.reshape(1, -1) ** 2),
            dim=0,
        ).reshape(q.shape) + c
    return output


def torch_xray_motif_form_factor(
    elements: Sequence[str] | np.ndarray,
    fractional_coordinates,
    absolute_hkl,
    reciprocal_matrix_inv_nm,
    *,
    torch=None,
):
    """Return a differentiable atomic motif factor on arbitrary hkl targets."""

    if torch is None:
        import torch as torch_module

        torch = torch_module
    hkl = torch.as_tensor(absolute_hkl)
    coordinates = torch.as_tensor(
        fractional_coordinates, dtype=hkl.dtype, device=hkl.device
    )
    reciprocal = torch.as_tensor(
        reciprocal_matrix_inv_nm, dtype=hkl.dtype, device=hkl.device
    )
    if hkl.ndim != 2 or hkl.shape[1] != 3:
        raise ValueError("absolute_hkl must have shape (n_q, 3)")
    if coordinates.ndim != 2 or coordinates.shape != (len(elements), 3):
        raise ValueError("fractional_coordinates must have shape (n_atom, 3)")
    if reciprocal.shape != (3, 3):
        raise ValueError("reciprocal_matrix_inv_nm must have shape (3, 3)")
    q_norm = torch.linalg.vector_norm(hkl @ reciprocal, dim=1)
    per_element = torch_xray_f0_form_factors(elements, q_norm, torch=torch)
    atom_factors = torch.stack([per_element[str(element)] for element in elements])
    phase = torch.exp(2j * torch.pi * (coordinates @ hkl.transpose(0, 1)))
    return torch.sum(atom_factors.to(dtype=phase.dtype) * phase, dim=0)
