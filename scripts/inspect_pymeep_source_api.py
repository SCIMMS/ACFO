from __future__ import annotations

import inspect

import meep as mp


def names(obj: object, terms: tuple[str, ...]) -> list[str]:
    return sorted(
        name for name in dir(obj) if any(term in name.lower() for term in terms)
    )


print("meep", mp.__version__)
print("module source/current names", names(mp, ("source", "current", "jx", "jy", "jz")))
print("Simulation source/array names", names(mp.Simulation, ("source", "array", "field")))
print("fields class candidates", [name for name in dir(mp) if "fields" in name.lower()])
for method_name in names(mp.Simulation, ("source", "array")):
    method = getattr(mp.Simulation, method_name, None)
    if callable(method):
        try:
            print(method_name, inspect.signature(method))
        except (TypeError, ValueError):
            pass
print("get_source doc", inspect.getdoc(mp.Simulation.get_source))
try:
    print("get_source source", inspect.getsource(mp.Simulation.get_source))
except (OSError, TypeError):
    pass
