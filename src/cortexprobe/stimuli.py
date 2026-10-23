"""Mapping apertures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .arrays import BoolArray, FloatArray, IntArray
from .config import ConfigError, StimulusConfig
from .geometry import Grid


def frame_similarity(stack: BoolArray) -> FloatArray:
    """Cosine overlap."""
    if stack.ndim != 3 or len(stack) == 0:
        raise ValueError("apertures must be a nonempty (frames, height, width) stack")
    flat = stack.reshape(len(stack), -1).astype(np.float64)
    frame_scales = np.max(np.abs(flat), axis=1, initial=0.0)
    if not np.isfinite(frame_scales).all() or np.any(frame_scales == 0.0):
        raise ValueError("similarity requires finite, nonempty aperture frames")
    flat /= frame_scales[:, None]
    squared_norms = np.einsum("ij,ij->i", flat, flat)
    similarity: FloatArray = (flat @ flat.T) / np.sqrt(
        squared_norms[:, None] * squared_norms[None, :]
    )
    return similarity


def _prune_leaking_frames(stack: BoolArray, groups: np.ndarray, threshold: float) -> IntArray:
    similarity = frame_similarity(stack)
    violations = (similarity > threshold) & (groups[:, None] != groups[None, :])
    keep = np.ones(len(stack), dtype=bool)
    violation_counts = violations.sum(axis=1)
    while violation_counts.any():
        removed = int(np.argmax(violation_counts))
        keep[removed] = False
        violation_counts -= violations[:, removed]
        violations[removed, :] = False
        violations[:, removed] = False
        violation_counts[removed] = 0
    return np.flatnonzero(keep)


@dataclass(frozen=True)
class ApertureSequence:
    apertures: BoolArray
    grid: Grid
    kind: str
    frame_index: np.ndarray
    group: np.ndarray

    def __post_init__(self) -> None:
        if self.apertures.dtype != np.bool_:
            raise TypeError("apertures must be boolean")
        if self.apertures.ndim != 3 or len(self.apertures) == 0:
            raise ValueError("apertures must be a nonempty (frames, height, width) stack")
        if self.apertures.shape[1:] != self.grid.shape:
            raise ValueError("aperture frames must match the grid")
        if np.ndim(self.frame_index) != 1 or len(self.frame_index) != len(self.apertures):
            raise ValueError("frame_index must label every frame")
        if np.ndim(self.group) != 1 or len(self.group) != len(self.apertures):
            raise ValueError("group must label every frame")
        if not self.apertures[:, self.grid.field_mask].any(axis=1).all():
            raise ConfigError("aperture frames must expose at least one field pixel")

    @property
    def n_frames(self) -> int:
        return int(self.apertures.shape[0])

    @property
    def coverage(self) -> FloatArray:
        """Exposed fraction."""
        exposed: FloatArray = (
            self.apertures[:, self.grid.field_mask].sum(axis=1) / self.grid.field_mask.sum()
        )
        return exposed

    def as_float(self) -> FloatArray:
        return self.apertures.astype(np.float64)


class ApertureGenerator(ABC):
    """Grouped apertures."""

    kind: str

    def __init__(self, config: StimulusConfig) -> None:
        self.config = config
        self.grid = Grid(config.resolution)

    def build(self) -> ApertureSequence:
        frames, labels, groups = self._frames()
        stack = np.stack(frames) & self.grid.field_mask
        group_array = np.asarray(groups)

        if not stack.any(axis=(1, 2)).all():
            raise ConfigError(
                f"{self.kind} design contains empty apertures at "
                f"resolution={self.config.resolution}; "
                "increase the aperture width, thickness, or span, or raise the resolution"
            )

        keep = _prune_leaking_frames(stack, group_array, self.config.max_fold_similarity)
        original_group_count = len(np.unique(group_array))
        retained_group_count = len(np.unique(group_array[keep]))
        if original_group_count >= 2 and retained_group_count < 2:
            raise ConfigError(
                f"{self.kind} apertures define {original_group_count} cross-validation groups "
                f"but only {retained_group_count} survive pruning at "
                f"max_fold_similarity={self.config.max_fold_similarity}. Raise the threshold or "
                "narrow the aperture (ring_thickness_frac, wedge_span_deg, bar_width_frac). "
                "Raising n_steps does not help: smaller steps increase overlap between frames."
            )
        return ApertureSequence(
            apertures=stack[keep],
            grid=self.grid,
            kind=self.kind,
            frame_index=np.asarray(labels)[keep],
            group=group_array[keep],
        )

    @property
    def n_frames(self) -> int:
        return self.config.n_steps

    @abstractmethod
    def _frames(self) -> tuple[list[BoolArray], list[int], list[int]]:
        """Frame metadata."""


class BarSweep(ApertureGenerator):
    kind = "bar"

    @property
    def n_frames(self) -> int:
        return self.config.n_bar_frames

    def _frames(self) -> tuple[list[BoolArray], list[int], list[int]]:
        grid = self.grid
        half_width = self.config.bar_width_px / 2.0
        travel = np.linspace(-grid.radius, grid.radius, self.config.n_steps)
        # Unique identifiers.
        label_stride = max(1000, self.config.n_steps)

        frames: list[BoolArray] = []
        labels: list[int] = []
        groups: list[int] = []
        for direction in self.config.directions:
            theta = np.radians(direction % 180)
            projection = grid.x * np.cos(theta) + grid.y * np.sin(theta)
            offsets = travel if direction < 180 else travel[::-1]
            for step, offset in enumerate(offsets):
                frames.append(np.abs(projection - offset) <= half_width)
                labels.append(direction * label_stride + step)
                groups.append(direction % 180)
        return frames, labels, groups


class RotatingWedge(ApertureGenerator):
    kind = "wedge"

    def _frames(self) -> tuple[list[BoolArray], list[int], list[int]]:
        span = self.config.wedge_span_deg
        starts = np.linspace(0.0, 360.0, self.config.n_steps, endpoint=False)
        return (
            [(self.grid.polar_angle - start) % 360.0 <= span for start in starts],
            list(range(len(starts))),
            [int(start // 90.0) for start in starts],
        )


class ExpandingRing(ApertureGenerator):
    """Expanding annulus."""

    kind = "ring"

    def _frames(self) -> tuple[list[BoolArray], list[int], list[int]]:
        half_thickness = self.config.ring_thickness_px / 2.0
        centres = np.linspace(0.0, self.grid.radius, self.config.n_steps)
        block = max(1, self.config.n_steps // 4)
        return (
            [np.abs(self.grid.eccentricity - centre) <= half_thickness for centre in centres],
            list(range(len(centres))),
            [step // block for step in range(len(centres))],
        )


GENERATORS = {generator.kind: generator for generator in (BarSweep, RotatingWedge, ExpandingRing)}


def build_apertures(config: StimulusConfig, kind: str = "bar") -> ApertureSequence:
    try:
        generator = GENERATORS[kind]
    except KeyError:
        raise ValueError(
            f"unknown stimulus kind {kind!r}; expected one of {sorted(GENERATORS)}"
        ) from None
    return generator(config).build()
