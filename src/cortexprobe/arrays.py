"""NumPy types."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int_]


def unit_responses(activations: FloatArray, n_frames: int) -> FloatArray:
    activations = np.asarray(activations, dtype=np.float64)
    if activations.ndim != 2:
        raise ValueError("activations must have shape (frames, units)")
    if activations.shape[0] != n_frames:
        raise ValueError("activations must have one row per stimulus frame")
    return activations.T
