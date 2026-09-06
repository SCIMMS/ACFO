"""Multi-anchor Taylor patches for repeated local ACFO geometry sweeps."""

from __future__ import annotations

from typing import Any


class AxisymmetricTaylorAnchorBank:
    """Cache fixed-object Taylor jets at several meridional geometries.

    Each query row is assigned independently to the closest anchor using the
    dimensionless distance ``sqrt((R*dq_perp)^2 + (Z*dq_z)^2)``.  Azimuthal
    displacements are evaluated by each jet but do not affect anchor selection
    in this first meridional anchor implementation.
    """

    def __init__(
        self,
        operator: Any,
        object_values: Any,
        anchor_q_perp: Any,
        anchor_q_z: Any,
        max_total_order: int,
    ) -> None:
        self.operator = operator
        self.torch = operator.torch
        self.device = operator.device
        self.max_total_order = int(max_total_order)
        if self.max_total_order < 0:
            raise ValueError("max_total_order must be non-negative")
        qp = self.torch.as_tensor(
            anchor_q_perp, dtype=operator.real_dtype, device=self.device
        )
        qz = self.torch.as_tensor(
            anchor_q_z, dtype=operator.real_dtype, device=self.device
        )
        if qp.ndim != 2 or tuple(qp.shape) != tuple(qz.shape):
            raise ValueError("anchor_q_perp and anchor_q_z must have equal 2-D shape")
        if qp.shape[0] == 0 or qp.shape[1] != operator.n_q:
            raise ValueError(
                f"anchor arrays must have shape (n_anchor, {operator.n_q})"
            )
        if not bool(self.torch.all(self.torch.isfinite(qp)).item()) or not bool(
            self.torch.all(self.torch.isfinite(qz)).item()
        ):
            raise ValueError("anchor geometries must be finite")
        self.anchor_q_perp = qp
        self.anchor_q_z = qz
        self.n_anchor = int(qp.shape[0])
        self.radial_scale = float(
            self.torch.max(self.torch.abs(operator.r_centers)).detach().cpu()
        )
        self.axial_scale = float(
            self.torch.max(self.torch.abs(operator.z_centers)).detach().cpu()
        )
        self.jets = []
        for index in range(self.n_anchor):
            self.jets.append(
                operator.forward_jet(
                    object_values,
                    self.max_total_order,
                    qp[index],
                    qz[index],
                )
            )

    def _query(self, values: Any, name: str) -> Any:
        tensor = self.torch.as_tensor(
            values, dtype=self.operator.real_dtype, device=self.device
        )
        if tuple(tensor.shape) != (self.operator.n_q,) or not bool(
            self.torch.all(self.torch.isfinite(tensor)).item()
        ):
            raise ValueError(f"{name} must have shape ({self.operator.n_q},)")
        return tensor

    def select_anchors(self, q_perp: Any, q_z: Any) -> tuple[Any, Any]:
        """Return per-target anchor indices and dimensionless distances."""

        qp = self._query(q_perp, "q_perp")
        qz = self._query(q_z, "q_z")
        self.operator._resolve_geometry(qp, qz)
        distance = self.torch.sqrt(
            (self.radial_scale * (self.anchor_q_perp - qp[None, :])) ** 2
            + (self.axial_scale * (self.anchor_q_z - qz[None, :])) ** 2
        )
        minimum, indices = self.torch.min(distance, dim=0)
        return indices, minimum

    def evaluate(
        self,
        q_perp: Any,
        q_z: Any,
        delta_phi: Any = 0.0,
        *,
        max_total_order: int | None = None,
    ) -> tuple[Any, Any, Any]:
        """Evaluate a query and return prediction, selected anchor and distance."""

        qp = self._query(q_perp, "q_perp")
        qz = self._query(q_z, "q_z")
        anchor_indices, distances = self.select_anchors(qp, qz)
        output = self.torch.empty(
            self.operator.data_shape,
            dtype=self.operator.complex_dtype,
            device=self.device,
        )
        for anchor in range(self.n_anchor):
            selected = anchor_indices == anchor
            if not bool(self.torch.any(selected).item()):
                continue
            candidate = self.operator.evaluate_jet(
                self.jets[anchor],
                qp - self.anchor_q_perp[anchor],
                qz - self.anchor_q_z[anchor],
                delta_phi,
                max_total_order=max_total_order,
            )
            output[selected] = candidate[selected]
        return output, anchor_indices, distances

    @property
    def coefficient_count(self) -> int:
        return self.n_anchor * len(self.jets[0])
