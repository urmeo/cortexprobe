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
    Coordinate arrays and the field mask are read-only views of cached data.
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
        axis = np.arange(self.resolution, dtype=np.float64) - centre
        axis.setflags(write=False)
        return axis

    @cached_property
    def _x(self) -> FloatArray:
        x = np.broadcast_to(self._axis, self.shape).copy()
        x.setflags(write=False)
        return x

    @cached_property
    def _y(self) -> FloatArray:
        y = np.broadcast_to(self._axis[::-1, None], self.shape).copy()
        y.setflags(write=False)
        return y

    @cached_property
    def _eccentricity(self) -> FloatArray:
        eccentricity = np.hypot(self.x, self.y)
        eccentricity.setflags(write=False)
        return eccentricity

    @cached_property
    def _polar_angle(self) -> FloatArray:
        angle: FloatArray = np.degrees(np.arctan2(self.y, self.x)) % 360.0
        angle.setflags(write=False)
        return angle

    @cached_property
    def _field_mask(self) -> BoolArray:
        mask = self.eccentricity <= self.radius
        mask.setflags(write=False)
        return mask

    @property
    def x(self) -> FloatArray:
        return self._x.view()

    @property
    def y(self) -> FloatArray:
        return self._y.view()

    @property
    def eccentricity(self) -> FloatArray:
        return self._eccentricity.view()

    @property
    def polar_angle(self) -> FloatArray:
        """Angle in degrees, counter-clockwise from the positive x axis, in [0, 360)."""
        return self._polar_angle.view()

    @property
    def field_mask(self) -> BoolArray:
        """Circular aperture inscribed in the square field."""
        return self._field_mask.view()

    @property
    def shape(self) -> tuple[int, int]:
        return (self.resolution, self.resolution)
