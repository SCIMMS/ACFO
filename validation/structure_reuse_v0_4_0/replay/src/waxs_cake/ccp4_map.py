"""Small, dependency-free reader for streamed CCP4/MRC mode-2 maps."""

from __future__ import annotations

import bz2
import struct
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np


@dataclass(frozen=True)
class Ccp4Header:
    shape_crs: tuple[int, int, int]
    mode: int
    start_crs: tuple[int, int, int]
    grid_xyz: tuple[int, int, int]
    cell: tuple[float, float, float, float, float, float]
    map_crs_to_xyz: tuple[int, int, int]
    data_min: float
    data_max: float
    data_mean: float
    space_group: int
    symmetry_bytes: int
    rms: float
    labels: tuple[str, ...]
    endian: str

    @property
    def section_shape(self) -> tuple[int, int]:
        columns, rows, _ = self.shape_crs
        return rows, columns

    @property
    def voxel_count(self) -> int:
        columns, rows, sections = self.shape_crs
        return columns * rows * sections


def _open(path: Path) -> BinaryIO:
    if path.suffix.lower() == ".bz2":
        return bz2.open(path, "rb")
    return path.open("rb")


def read_ccp4_header(path: str | Path) -> Ccp4Header:
    path = Path(path)
    with _open(path) as handle:
        raw = handle.read(1024)
    if len(raw) != 1024 or raw[208:212] != b"MAP ":
        raise ValueError(f"{path} is not a CCP4/MRC map with a standard header")
    endian = "<"
    integers = struct.unpack("<256i", raw)
    if not (
        all(0 < value < 1_000_000 for value in integers[:3])
        and integers[3] in {0, 1, 2, 3, 4, 6, 12, 16}
    ):
        endian = ">"
        integers = struct.unpack(">256i", raw)
    floats = struct.unpack(endian + "256f", raw)
    label_count = min(max(integers[55], 0), 10)
    labels = tuple(
        raw[224 + 80 * index : 224 + 80 * (index + 1)]
        .rstrip(b"\x00 ")
        .decode("utf-8", errors="replace")
        for index in range(label_count)
    )
    return Ccp4Header(
        shape_crs=tuple(int(v) for v in integers[:3]),
        mode=int(integers[3]),
        start_crs=tuple(int(v) for v in integers[4:7]),
        grid_xyz=tuple(int(v) for v in integers[7:10]),
        cell=tuple(float(v) for v in floats[10:16]),
        map_crs_to_xyz=tuple(int(v) for v in integers[16:19]),
        data_min=float(floats[19]),
        data_max=float(floats[20]),
        data_mean=float(floats[21]),
        space_group=int(integers[22]),
        symmetry_bytes=int(integers[23]),
        rms=float(floats[54]),
        labels=labels,
        endian=endian,
    )


@contextmanager
def iter_ccp4_sections(
    path: str | Path, header: Ccp4Header | None = None
) -> Iterator[Iterator[np.ndarray]]:
    """Yield read-only ``(row, column)`` arrays without loading the full map."""

    path = Path(path)
    resolved = read_ccp4_header(path) if header is None else header
    if resolved.mode != 2:
        raise ValueError("streaming reader currently supports only mode-2 float maps")
    columns, rows, sections = resolved.shape_crs
    values_per_section = columns * rows
    byte_count = values_per_section * 4
    dtype = np.dtype(resolved.endian + "f4")
    handle = _open(path)
    try:
        skipped = handle.read(1024 + resolved.symmetry_bytes)
        if len(skipped) != 1024 + resolved.symmetry_bytes:
            raise ValueError("truncated CCP4 header or symmetry block")

        def iterator() -> Iterator[np.ndarray]:
            for _ in range(sections):
                raw = handle.read(byte_count)
                if len(raw) != byte_count:
                    raise ValueError("truncated CCP4 map data")
                yield np.frombuffer(raw, dtype=dtype).reshape(rows, columns)
            if handle.read(1):
                raise ValueError("unexpected trailing bytes after CCP4 map data")

        yield iterator()
    finally:
        handle.close()
