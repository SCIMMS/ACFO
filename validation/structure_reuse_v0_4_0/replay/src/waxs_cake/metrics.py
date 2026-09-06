"""Error and timing helpers for method comparisons."""

from __future__ import annotations

import numpy as np


def relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    denom = np.linalg.norm(reference.ravel())
    if denom == 0:
        return float(np.linalg.norm(candidate.ravel()))
    return float(np.linalg.norm((candidate - reference).ravel()) / denom)


def max_relative_abs(candidate: np.ndarray, reference: np.ndarray) -> float:
    scale = np.max(np.abs(reference))
    if scale == 0:
        return float(np.max(np.abs(candidate)))
    return float(np.max(np.abs(candidate - reference)) / scale)


def intensity(amplitude: np.ndarray) -> np.ndarray:
    return np.abs(amplitude) ** 2
