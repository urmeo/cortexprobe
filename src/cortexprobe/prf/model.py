"""Gaussian receptive-field overlap in pixel coordinates.

Adapted from Dumoulin & Wandell (2008), NeuroImage 39:647-660. Predictions are raw aperture
overlaps, without the haemodynamic convolution used for fMRI.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..arrays import FloatArray
from ..geometry import Grid


class ReceptiveField(ABC):
    """Spatial weights over the visual field."""

    @abstractmethod
    def weights(self, grid: Grid) -> FloatArray:
        """Return an ``(H, W)`` array of field weights on ``grid``."""


@dataclass(frozen=True)
class GaussianReceptiveField(ReceptiveField):
    """An isotropic 2D Gaussian receptive field.

    Parameters are in the pixel coordinates defined by :class:`~cortexprobe.geometry.Grid`:
    the origin is the centre of gaze, x increases rightward, y increases upward.
    """

    x0: float
    y0: float
    sigma: float

    def __post_init__(self) -> None:
        for name in ("x0", "y0", "sigma"):
            value = getattr(self, name)
            try:
                valid = (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                )
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError(f"{name} must be finite")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        variance = self.sigma * self.sigma
        if variance == 0.0 or not math.isfinite(2.0 * math.pi * variance):
            raise ValueError("sigma cannot be represented by the Gaussian model")

    @property
    def eccentricity(self) -> float:
        return float(np.hypot(self.x0, self.y0))

    @property
    def polar_angle(self) -> float:
        return float(np.degrees(np.arctan2(self.y0, self.x0)) % 360.0)

    def weights(self, grid: Grid) -> FloatArray:
        """Sample a Gaussian whose continuous integral is one.

        Finite grids truncate its mass, particularly near an edge. The fitter checks the
        centered field at the sigma ceiling; this does not guarantee mass at other centers.
        Unit volume defines the amplitude convention. A free fitted beta absorbs the scaling
        between unit-volume and unit-peak weights, so this choice does not force a size trend.
        """
        dx = grid.x - self.x0
        dy = grid.y - self.y0
        variance = self.sigma**2
        weights: FloatArray = np.exp(-(dx**2 + dy**2) / (2.0 * variance)) / (2.0 * np.pi * variance)
        return weights


def predict(weights: FloatArray, apertures: FloatArray) -> FloatArray:
    """Frame overlaps: ``r(t) = sum_xy w(x, y) * aperture_t(x, y)``."""
    if weights.ndim != 2 or apertures.ndim != 3:
        raise ValueError("weights must be 2D and apertures must have shape (frames, height, width)")
    if len(apertures) == 0:
        raise ValueError("prediction requires at least one aperture frame")
    if weights.shape != apertures.shape[1:]:
        raise ValueError(
            f"receptive field {weights.shape} does not match aperture frames {apertures.shape[1:]}"
        )
    return apertures.reshape(len(apertures), -1) @ weights.ravel()


def design_matrix(
    fields: list[GaussianReceptiveField], grid: Grid, apertures: FloatArray
) -> FloatArray:
    """Stack predicted timecourses, one column per candidate field."""
    if not fields:
        raise ValueError("design matrix requires at least one candidate field")
    return np.column_stack([predict(field.weights(grid), apertures) for field in fields])
