"""Verify the shared coordinate system for stimuli and receptive fields."""

from __future__ import annotations

import numpy as np
import pytest

from cortexprobe.geometry import Grid


def test_a_grid_needs_at_least_two_samples() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        Grid(1)


@pytest.mark.parametrize("resolution", [8.5, np.nan, np.inf, True])
def test_grid_resolution_requires_an_integer(resolution) -> None:
    with pytest.raises(ValueError, match="integer"):
        Grid(resolution)


@pytest.mark.parametrize("resolution", [2, 8, 64])
def test_shape_and_radius_follow_the_resolution(resolution) -> None:
    grid = Grid(resolution)

    assert grid.shape == (resolution, resolution)
    assert grid.x.shape == grid.shape
    assert grid.y.shape == grid.shape
    assert grid.radius == pytest.approx(resolution / 2.0)


def test_the_origin_sits_at_the_centre_of_the_field() -> None:
    grid = Grid(64)

    assert grid.x.mean() == pytest.approx(0.0)
    assert grid.y.mean() == pytest.approx(0.0)
    assert grid.eccentricity.min() == pytest.approx(np.hypot(0.5, 0.5))


def test_x_increases_rightward() -> None:
    grid = Grid(8)

    assert grid.x[0, 0] < 0.0 < grid.x[0, -1]
    assert (np.diff(grid.x, axis=1) > 0).all()


def test_y_increases_upward_not_downward() -> None:
    """Image arrays index downward; a fitted y0 must read the way a person expects."""
    grid = Grid(8)

    assert grid.y[0, 0] > 0.0 > grid.y[-1, 0]
    assert (np.diff(grid.y, axis=0) < 0).all()


def test_the_axis_is_symmetric_about_zero() -> None:
    grid = Grid(8)

    assert np.allclose(grid.x[0], -grid.x[0][::-1])


def test_eccentricity_is_the_distance_from_the_origin() -> None:
    grid = Grid(16)

    assert np.allclose(grid.eccentricity, np.hypot(grid.x, grid.y))


@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [(1.0, 0.0, 0.0), (0.0, 1.0, 90.0), (-1.0, 0.0, 180.0), (0.0, -1.0, 270.0)],
)
def test_polar_angle_runs_counter_clockwise_from_the_positive_x_axis(x, y, expected) -> None:
    assert np.degrees(np.arctan2(y, x)) % 360.0 == pytest.approx(expected)


def test_polar_angle_covers_the_full_circle() -> None:
    angle = Grid(32).polar_angle

    assert angle.min() >= 0.0
    assert angle.max() < 360.0


def test_field_mask_is_the_inscribed_circle() -> None:
    grid = Grid(64)

    assert grid.field_mask[32, 32]
    assert not grid.field_mask[0, 0]
    assert np.array_equal(grid.field_mask, grid.eccentricity <= grid.radius)


def test_field_mask_is_symmetric() -> None:
    mask = Grid(64).field_mask

    assert np.array_equal(mask, np.fliplr(mask))
    assert np.array_equal(mask, np.flipud(mask))


def test_derived_arrays_are_cached_not_rebuilt() -> None:
    """Every fit builds candidate fields on the same grid; rebuilding these would be waste."""
    grid = Grid(32)

    assert grid.x is grid.x
    assert grid.eccentricity is grid.eccentricity
    assert grid.field_mask is grid.field_mask


def test_grids_are_frozen_and_comparable() -> None:
    assert Grid(64) == Grid(64)
    assert Grid(64) != Grid(32)

    with pytest.raises(AttributeError):
        Grid(64).resolution = 32  # type: ignore[misc]
