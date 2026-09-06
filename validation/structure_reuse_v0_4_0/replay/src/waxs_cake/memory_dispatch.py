"""Deterministic memory policies for prepared and streaming ACFO operators.

The routines in this module do not inspect workload accuracy or change an
operator's mathematical support.  They only choose a block or cache policy
after the caller has fixed the representation, dtype, harmonic cutoff and
accuracy contract.  Keeping that separation makes memory tuning auditable:
the same numerical problem is evaluated with a different materialization
strategy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import numbers
import types
from typing import Any, Iterable, Mapping


MIB = 1024**2


@dataclass(frozen=True)
class BlockMemoryDecision:
    """Resolved streaming block under a fixed byte budget."""

    total_items: int
    block_size: int
    block_count: int
    fixed_bytes: int
    bytes_per_item: int
    budget_bytes: int
    usable_budget_bytes: int
    estimated_peak_bytes: int
    safety_fraction: float
    alignment: int
    clipped_by_budget: bool

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


@dataclass(frozen=True)
class CacheMemoryDecision:
    """Resolved materialize/stream decision for a reusable cache."""

    policy: str
    cache_bytes: int
    working_bytes: int
    budget_bytes: int
    usable_budget_bytes: int
    reuse_count: int
    minimum_reuse_for_cache: int
    fits_budget: bool
    reason: str

    def to_dict(self) -> dict[str, int | str | bool]:
        return asdict(self)


def choose_block_size(
    *,
    total_items: int,
    fixed_bytes: int,
    bytes_per_item: int,
    budget_bytes: int,
    safety_fraction: float = 0.80,
    minimum: int = 1,
    maximum: int | None = None,
    alignment: int = 1,
) -> BlockMemoryDecision:
    """Choose the largest aligned block whose estimate fits the budget.

    ``fixed_bytes`` includes prepared state and output storage that is present
    regardless of block size.  ``bytes_per_item`` is the incremental working
    set per streamed item.  The function fails closed when even the minimum
    block cannot fit; it never silently changes the problem size.
    """

    total_items = int(total_items)
    fixed_bytes = int(fixed_bytes)
    bytes_per_item = int(bytes_per_item)
    budget_bytes = int(budget_bytes)
    minimum = int(minimum)
    alignment = int(alignment)
    if total_items <= 0:
        raise ValueError("total_items must be positive")
    if fixed_bytes < 0 or bytes_per_item <= 0 or budget_bytes <= 0:
        raise ValueError("fixed_bytes must be nonnegative and byte counts positive")
    if not 0.0 < float(safety_fraction) <= 1.0:
        raise ValueError("safety_fraction must lie in (0, 1]")
    if minimum <= 0 or alignment <= 0:
        raise ValueError("minimum and alignment must be positive")
    if maximum is None:
        maximum = total_items
    maximum = min(int(maximum), total_items)
    if maximum < minimum:
        raise ValueError("maximum must be at least minimum")

    usable = int(budget_bytes * float(safety_fraction))
    available = usable - fixed_bytes
    if available < minimum * bytes_per_item:
        raise MemoryError(
            "the fixed state and minimum streaming block exceed the usable budget"
        )
    raw = min(maximum, available // bytes_per_item)
    if raw >= alignment:
        resolved = (raw // alignment) * alignment
    else:
        resolved = raw
    resolved = max(minimum, min(maximum, resolved))
    estimated = fixed_bytes + resolved * bytes_per_item
    if estimated > usable:
        raise MemoryError("aligned block does not fit the usable budget")
    block_count = (total_items + resolved - 1) // resolved
    return BlockMemoryDecision(
        total_items=total_items,
        block_size=resolved,
        block_count=block_count,
        fixed_bytes=fixed_bytes,
        bytes_per_item=bytes_per_item,
        budget_bytes=budget_bytes,
        usable_budget_bytes=usable,
        estimated_peak_bytes=estimated,
        safety_fraction=float(safety_fraction),
        alignment=alignment,
        clipped_by_budget=resolved < maximum,
    )


def choose_cache_policy(
    *,
    cache_bytes: int,
    working_bytes: int,
    budget_bytes: int,
    reuse_count: int,
    minimum_reuse_for_cache: int,
    safety_fraction: float = 0.80,
) -> CacheMemoryDecision:
    """Choose cache materialization only when it fits and reuse justifies it."""

    cache_bytes = int(cache_bytes)
    working_bytes = int(working_bytes)
    budget_bytes = int(budget_bytes)
    reuse_count = int(reuse_count)
    minimum_reuse_for_cache = int(minimum_reuse_for_cache)
    if min(cache_bytes, working_bytes) < 0 or budget_bytes <= 0:
        raise ValueError("byte counts must be nonnegative and budget positive")
    if reuse_count < 0 or minimum_reuse_for_cache < 0:
        raise ValueError("reuse counts must be nonnegative")
    if not 0.0 < float(safety_fraction) <= 1.0:
        raise ValueError("safety_fraction must lie in (0, 1]")
    usable = int(budget_bytes * float(safety_fraction))
    fits = working_bytes + cache_bytes <= usable
    enough_reuse = reuse_count >= minimum_reuse_for_cache
    if fits and enough_reuse:
        policy = "materialize"
        reason = "cache fits the protected budget and the reuse threshold is met"
    elif not fits:
        policy = "stream"
        reason = "cache would exceed the protected budget"
    else:
        policy = "stream"
        reason = "reuse is below the materialization threshold"
    return CacheMemoryDecision(
        policy=policy,
        cache_bytes=cache_bytes,
        working_bytes=working_bytes,
        budget_bytes=budget_bytes,
        usable_budget_bytes=usable,
        reuse_count=reuse_count,
        minimum_reuse_for_cache=minimum_reuse_for_cache,
        fits_budget=fits,
        reason=reason,
    )


def retained_nbytes(value: Any) -> int:
    """Return a conservative, de-duplicated byte count for retained arrays.

    NumPy, CuPy and Torch arrays expose either ``nbytes`` or
    ``numel()*element_size()``.  Containers and ordinary objects are traversed
    recursively.  Python object overhead and allocator/runtime plan memory are
    intentionally excluded; external RSS or device-process measurements must
    accompany this structural count in publication benchmarks.
    """

    seen_objects: set[int] = set()
    seen_storage: set[tuple[str, int]] = set()

    def visit(item: Any) -> int:
        if item is None or isinstance(
            item, (str, bytes, bytearray, int, float, bool, type, types.ModuleType)
        ) or callable(item):
            return 0
        object_id = id(item)
        if object_id in seen_objects:
            return 0
        seen_objects.add(object_id)

        nbytes = getattr(item, "nbytes", None)
        if isinstance(nbytes, numbers.Integral):
            storage_owner = item
            base = getattr(storage_owner, "base", None)
            while base is not None and base is not storage_owner:
                storage_owner = base
                base = getattr(storage_owner, "base", None)
            storage_nbytes = getattr(storage_owner, "nbytes", nbytes)
            pointer = None
            data = getattr(storage_owner, "data", None)
            if data is not None:
                pointer = getattr(data, "ptr", None)
            if pointer is None:
                array_interface = getattr(storage_owner, "__array_interface__", None)
                if isinstance(array_interface, Mapping):
                    data_field = array_interface.get("data")
                    if isinstance(data_field, tuple) and data_field:
                        pointer = data_field[0]
            key = (
                type(storage_owner).__module__,
                int(pointer) if pointer else id(storage_owner),
            )
            if key in seen_storage:
                return 0
            seen_storage.add(key)
            return int(storage_nbytes)

        numel = getattr(item, "numel", None)
        element_size = getattr(item, "element_size", None)
        if callable(numel) and callable(element_size):
            pointer = None
            data_ptr = getattr(item, "data_ptr", None)
            if callable(data_ptr):
                try:
                    pointer = int(data_ptr())
                except (TypeError, ValueError, RuntimeError):
                    pointer = None
            key = (type(item).__module__, pointer or object_id)
            if key in seen_storage:
                return 0
            seen_storage.add(key)
            return int(numel()) * int(element_size())

        if isinstance(item, Mapping):
            return sum(visit(entry) for entry in item.values())
        if isinstance(item, (list, tuple, set, frozenset)):
            return sum(visit(entry) for entry in item)
        namespace = getattr(item, "__dict__", None)
        if isinstance(namespace, Mapping):
            return sum(visit(entry) for entry in namespace.values())
        return 0

    return visit(value)


def sum_nbytes(values: Iterable[Any]) -> int:
    """Convenience structural count for an iterable of independent values."""

    return sum(retained_nbytes(value) for value in values)
