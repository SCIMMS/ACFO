from __future__ import annotations

import meep as mp
import numpy as np


def amplitude(point: mp.Vector3) -> complex:
    return complex(np.exp(1j * (1.7 * point.x - 0.8 * point.y + 0.4 * point.z)))


resolution = 8
half_width = 0.25
source_time = mp.GaussianSource(frequency=1.0, fwidth=0.3, is_integrated=True)
sources = [
    mp.Source(
        source_time,
        component=component,
        center=mp.Vector3(),
        size=mp.Vector3(2 * half_width, 2 * half_width, 2 * half_width),
        amplitude=1.0 + 0.1j * index,
        amp_func=amplitude,
    )
    for index, component in enumerate((mp.Ex, mp.Ey, mp.Ez))
]
sim = mp.Simulation(
    cell_size=mp.Vector3(3, 3, 3),
    resolution=resolution,
    default_material=mp.Medium(epsilon=2.0),
    boundary_layers=[mp.PML(0.5)],
    sources=sources,
    force_complex_fields=True,
)
sim.init_sim()
volume = mp.Volume(center=mp.Vector3(), size=mp.Vector3(1, 1, 1))
metadata = sim.get_array_metadata(vol=volume)
print("metadata entries", len(metadata))
for index, value in enumerate(metadata):
    array = np.asarray(value)
    print("metadata", index, array.shape, array.dtype, array.ravel()[:6])
for name, component in zip(("Ex", "Ey", "Ez"), (mp.Ex, mp.Ey, mp.Ez), strict=True):
    source = np.asarray(sim.get_source(component, vol=volume))
    nonzero = np.argwhere(np.abs(source) > 0)
    print(
        name,
        "shape",
        source.shape,
        "nonzero",
        nonzero.shape[0],
        "first-index",
        nonzero[0].tolist() if nonzero.size else None,
        "last-index",
        nonzero[-1].tolist() if nonzero.size else None,
        "norm",
        float(np.linalg.norm(source)),
        "sum",
        complex(np.sum(source)),
    )
sim.reset_meep()
