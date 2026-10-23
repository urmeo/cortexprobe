"""Gaussian overlaps."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..arrays import FloatArray
from ..config import _finite_float
from ..geometry import Grid


class ReceptiveField(ABC):
    """Spatial weights."""

    @abstractmethod
    def weights(self, grid: Grid) -> FloatArray:
        """Grid weights."""


@dataclass(frozen=True)
class GaussianReceptiveField(ReceptiveField):
    """Isotropic Gaussian."""

    x0: float
    y0: float
    sigma: float

    def __post_init__(self) -> None:
        for name in ("x0", "y0", "sigma"):
            _finite_float(getattr(self, name), name, ValueError)
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        sigma = float(self.sigma)
        variance = sigma * sigma
        normalization = 2.0 * math.pi * variance
        if (
            normalization == 0.0
            or not math.isfinite(normalization)
            or not math.isfinite(1.0 / normalization)
        ):
            raise ValueError("sigma cannot be represented by the Gaussian model")

    @property
    def eccentricity(self) -> float:
        return float(np.hypot(self.x0, self.y0))

    @property
    def polar_angle(self) -> float:
        return float(np.degrees(np.arctan2(self.y0, self.x0)) % 360.0)

    def weights(self, grid: Grid) -> FloatArray:
        """Normalized weights."""
        # Distant tails.
        with np.errstate(over="ignore"):
            dx = (grid.x - self.x0) / self.sigma
            dy = (grid.y - self.y0) / self.sigma
            squared_distance = dx**2 + dy**2
        weights: FloatArray = np.exp(-0.5 * squared_distance) / (2.0 * np.pi * self.sigma**2)
        return weights


def predict(weights: FloatArray, apertures: FloatArray) -> FloatArray:
    """Frame overlaps."""
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
    fields: Sequence[GaussianReceptiveField], grid: Grid, apertures: FloatArray
) -> FloatArray:
    """Candidate predictions."""
    if not fields:
        raise ValueError("design matrix requires at least one candidate field")
    return np.column_stack([predict(field.weights(grid), apertures) for field in fields])
