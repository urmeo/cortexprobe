"""Pixel coordinates shared by apertures and receptive fields."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .arrays import BoolArray, FloatArray


@dataclass(frozen=True)
class Grid:
    """Square pixel lattice centered at the origin, with x rightward and y upward.

    Array rows run downward; ``field_mask`` selects the inscribed circular field.
    """

    resolution: int

    def __post_init__(self) -> None:
        if not isinstance(self.resolution, int) or isinstance(self.resolution, bool):
            raise ValueError("grid resolution must be an integer")
        if self.resolution < 2:
            raise ValueError("grid resolution must be at least 2")

    @property
    def radius(self) -> float:
        """Radius of the circular field inscribed in the square grid."""
        return self.resolution / 2.0

    @cached_property
    def _axis(self) -> FloatArray:
        centre = (self.resolution - 1) / 2.0
        return np.arange(self.resolution, dtype=np.float64) - centre

    @cached_property
    def x(self) -> FloatArray:
        return np.tile(self._axis, (self.resolution, 1))

    @cached_property
    def y(self) -> FloatArray:
        return np.flipud(np.tile(self._axis[:, None], (1, self.resolution)))

    @cached_property
    def eccentricity(self) -> FloatArray:
        return np.hypot(self.x, self.y)

    @cached_property
    def polar_angle(self) -> FloatArray:
        """Angle in degrees, counter-clockwise from the positive x axis, in [0, 360)."""
        angle: FloatArray = np.degrees(np.arctan2(self.y, self.x)) % 360.0
        return angle

    @cached_property
    def field_mask(self) -> BoolArray:
        """Circular aperture inscribed in the square field."""
        return self.eccentricity <= self.radius

    @property
    def shape(self) -> tuple[int, int]:
        return (self.resolution, self.resolution)
