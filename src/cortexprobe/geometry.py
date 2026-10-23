"""Pixel coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .arrays import BoolArray, FloatArray


@dataclass(frozen=True)
class Grid:
    resolution: int

    def __post_init__(self) -> None:
        if not isinstance(self.resolution, int) or isinstance(self.resolution, bool):
            raise ValueError("grid resolution must be an integer")
        if self.resolution < 2:
            raise ValueError("grid resolution must be at least 2")

    @property
    def radius(self) -> float:
        return self.resolution / 2.0

    @cached_property
    def _coordinates(self) -> tuple[FloatArray, FloatArray]:
        axis = np.arange(self.resolution, dtype=np.float64) - (self.resolution - 1) / 2.0
        x, y = np.meshgrid(axis, axis[::-1])
        x.setflags(write=False)
        y.setflags(write=False)
        return x, y

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
        return self._coordinates[0].view()

    @property
    def y(self) -> FloatArray:
        return self._coordinates[1].view()

    @property
    def eccentricity(self) -> FloatArray:
        return self._eccentricity.view()

    @property
    def polar_angle(self) -> FloatArray:
        """Polar degrees."""
        return self._polar_angle.view()

    @property
    def field_mask(self) -> BoolArray:
        """Circular field."""
        return self._field_mask.view()

    @property
    def shape(self) -> tuple[int, int]:
        return (self.resolution, self.resolution)
