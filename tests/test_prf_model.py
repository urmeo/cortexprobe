"""Gaussian weights, overlap predictions, and the amplitude convention."""

from __future__ import annotations

import numpy as np
import pytest

from cortexprobe.geometry import Grid
from cortexprobe.prf.fit import _r_squared, _solve_amplitude
from cortexprobe.prf.model import GaussianReceptiveField, design_matrix, predict


def test_a_receptive_field_needs_a_positive_sigma() -> None:
    with pytest.raises(ValueError, match="sigma must be positive"):
        GaussianReceptiveField(0.0, 0.0, 0.0)

    with pytest.raises(ValueError, match="sigma must be positive"):
        GaussianReceptiveField(0.0, 0.0, -1.0)


@pytest.mark.parametrize(
    ("x0", "y0", "eccentricity", "angle"),
    [
        (3.0, 4.0, 5.0, 53.13010235),
        (0.0, 0.0, 0.0, 0.0),
        (-1.0, 0.0, 1.0, 180.0),
        (0.0, -2.0, 2.0, 270.0),
    ],
)
def test_position_is_reported_in_polar_coordinates_too(x0, y0, eccentricity, angle) -> None:
    field = GaussianReceptiveField(x0, y0, 3.0)

    assert field.eccentricity == pytest.approx(eccentricity)
    assert field.polar_angle == pytest.approx(angle)


def test_weights_carry_unit_volume() -> None:
    grid = Grid(64)

    for sigma in (1.0, 3.0, 5.0, 10.0):
        assert GaussianReceptiveField(0.0, 0.0, sigma).weights(grid).sum() == pytest.approx(
            1.0, abs=5e-3
        )


def test_a_wider_field_spreads_the_same_weight_more_thinly() -> None:
    """Unit volume sets the weight scale before fitting a free amplitude."""
    grid = Grid(64)

    narrow = GaussianReceptiveField(0.0, 0.0, 2.0).weights(grid)
    wide = GaussianReceptiveField(0.0, 0.0, 6.0).weights(grid)

    assert wide.max() < narrow.max()
    assert wide.sum() == pytest.approx(narrow.sum(), rel=1e-3)


def test_weights_peak_at_the_declared_centre() -> None:
    grid = Grid(64)

    weights = GaussianReceptiveField(10.0, -6.0, 3.0).weights(grid)
    row, column = np.unravel_index(int(np.argmax(weights)), weights.shape)

    assert grid.x[row, column] == pytest.approx(10.0, abs=1.0)
    assert grid.y[row, column] == pytest.approx(-6.0, abs=1.0)


def test_prediction_is_the_overlap_with_each_aperture() -> None:
    grid = Grid(32)
    weights = GaussianReceptiveField(0.0, 0.0, 3.0).weights(grid)
    apertures = np.stack([np.ones(grid.shape), np.zeros(grid.shape)])

    response = predict(weights, apertures)

    assert response[0] == pytest.approx(weights.sum())
    assert response[1] == pytest.approx(0.0)


def test_prediction_rejects_a_field_that_does_not_match_the_frames() -> None:
    grid = Grid(32)
    weights = GaussianReceptiveField(0.0, 0.0, 3.0).weights(grid)

    with pytest.raises(ValueError, match="does not match aperture frames"):
        predict(weights, np.ones((4, 16, 16)))


def test_prediction_scales_linearly_with_the_aperture() -> None:
    grid = Grid(32)
    weights = GaussianReceptiveField(2.0, 2.0, 4.0).weights(grid)
    apertures = np.random.default_rng(0).random((5, *grid.shape))

    assert np.allclose(predict(weights, 2.0 * apertures), 2.0 * predict(weights, apertures))


def test_design_matrix_stacks_one_column_per_candidate() -> None:
    grid = Grid(32)
    apertures = np.random.default_rng(1).random((7, *grid.shape))
    fields = [
        GaussianReceptiveField(0.0, 0.0, 3.0),
        GaussianReceptiveField(5.0, -5.0, 2.0),
        GaussianReceptiveField(-4.0, 4.0, 4.0),
    ]

    matrix = design_matrix(fields, grid, apertures)

    assert matrix.shape == (7, 3)
    for column, field in enumerate(fields):
        assert np.allclose(matrix[:, column], predict(field.weights(grid), apertures))


def test_a_field_far_outside_the_grid_contributes_nothing() -> None:
    grid = Grid(32)

    weights = GaussianReceptiveField(500.0, 500.0, 2.0).weights(grid)

    assert np.isfinite(weights).all()
    assert weights.sum() == pytest.approx(0.0, abs=1e-12)


def test_prediction_is_linear_in_the_receptive_field() -> None:
    """Weighted fields produce the same weighted sum of predictions."""
    grid = Grid(32)
    apertures = np.random.default_rng(2).random((8, *grid.shape))
    first = GaussianReceptiveField(3.0, 3.0, 2.0).weights(grid)
    second = GaussianReceptiveField(-5.0, 1.0, 4.0).weights(grid)

    combined = predict(2.0 * first + 3.0 * second, apertures)
    separately = 2.0 * predict(first, apertures) + 3.0 * predict(second, apertures)

    assert np.allclose(combined, separately)


def test_translating_field_and_aperture_together_leaves_the_response_unchanged() -> None:
    """Joint translation preserves overlap when edge weights are negligible."""
    grid = Grid(64)
    apertures = np.random.default_rng(0).random((8, *grid.shape))
    here = GaussianReceptiveField(4.0, -2.0, 3.0).weights(grid)
    one_right = GaussianReceptiveField(5.0, -2.0, 3.0).weights(grid)

    assert np.allclose(
        predict(here, apertures),
        predict(one_right, np.roll(apertures, 1, axis=2)),
    )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, True])
@pytest.mark.parametrize("parameter", ["x0", "y0", "sigma"])
def test_gaussian_parameters_must_be_finite(parameter, value) -> None:
    parameters = {"x0": 0.0, "y0": 0.0, "sigma": 3.0}
    parameters[parameter] = value
    with pytest.raises(ValueError, match="finite"):
        GaussianReceptiveField(**parameters)


@pytest.mark.parametrize("sigma", [1e-300, 1e300])
def test_unrepresentable_gaussian_variance_is_rejected(sigma) -> None:
    with pytest.raises(ValueError, match="represented"):
        GaussianReceptiveField(0, 0, sigma)


@pytest.mark.parametrize("sigma", [2.0, 5.0, 8.0])
def test_free_amplitude_absorbs_unit_peak_normalization(grid, apertures, sigma) -> None:
    response = 3 * predict(GaussianReceptiveField(12, 8, 5).weights(grid), apertures) + 0.5
    volume = predict(GaussianReceptiveField(12, 8, sigma).weights(grid), apertures)
    scale = 2 * np.pi * sigma**2
    volume_coefficients, volume_fit = _solve_amplitude(volume, response)
    peak_coefficients, peak_fit = _solve_amplitude(volume * scale, response)
    assert peak_fit == pytest.approx(volume_fit, abs=1e-12)
    assert peak_coefficients[0] * scale == pytest.approx(volume_coefficients[0])
    assert _r_squared(response, peak_fit) == pytest.approx(_r_squared(response, volume_fit))


def test_centered_volume_guard_does_not_guarantee_edge_mass(grid) -> None:
    centered = GaussianReceptiveField(0, 0, 5).weights(grid)[grid.field_mask].sum()
    edge = GaussianReceptiveField(grid.radius, 0, 5).weights(grid)[grid.field_mask].sum()
    assert centered > 0.99
    assert edge < 0.6


def test_prediction_requires_a_frame_stack_and_candidates(grid) -> None:
    weights = GaussianReceptiveField(0, 0, 3).weights(grid)
    with pytest.raises(ValueError, match="shape"):
        predict(weights, np.ones(grid.shape))
    with pytest.raises(ValueError, match="at least one aperture"):
        predict(weights, np.empty((0, *grid.shape)))
    with pytest.raises(ValueError, match="at least one candidate"):
        design_matrix([], grid, np.ones((2, *grid.shape)))
