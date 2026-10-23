import re
from dataclasses import replace

import numpy as np
import pytest
from scipy.special import stdtr

import cortexprobe.prf.fit as fit_module
from conftest import RESOLUTION, autocorrelated_noise, synthesise
from cortexprobe.config import ConfigError, FitConfig, StimulusConfig
from cortexprobe.geometry import Grid
from cortexprobe.prf.fit import (
    BOUND_TOLERANCE,
    MIN_ON_GRID_VOLUME,
    RESOLUTION_PER_SIGMA,
    PRFFitter,
    UnitFit,
    _count_distinct_frames,
    _r_squared,
    _solve_amplitude,
    _standard_errors,
)
from cortexprobe.prf.model import GaussianReceptiveField, design_matrix, predict
from cortexprobe.prf.validation import (
    CrossValidatedFit,
    CrossValidator,
    LeaveOneGroupOut,
)
from cortexprobe.stimuli import build_apertures


def geometry(fit: UnitFit) -> tuple[float, float, float]:
    return fit.x0, fit.y0, fit.sigma


def errors(fit: UnitFit) -> tuple[float, float, float]:
    return fit.se_x0, fit.se_y0, fit.se_sigma


FIT_TRUTH = (10.0, -8.0, 5.0)
TRUTH = (12.0, 8.0, 5.0)


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
    grid = Grid(32)
    apertures = np.random.default_rng(2).random((8, *grid.shape))
    first = GaussianReceptiveField(3.0, 3.0, 2.0).weights(grid)
    second = GaussianReceptiveField(-5.0, 1.0, 4.0).weights(grid)

    combined = predict(2.0 * first + 3.0 * second, apertures)
    separately = 2.0 * predict(first, apertures) + 3.0 * predict(second, apertures)

    assert np.allclose(combined, separately)


def test_translating_field_and_aperture_together_leaves_the_response_unchanged() -> None:
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


@pytest.mark.parametrize("sigma", [1e-300, 1e-160, 1e300, 10**155])
def test_unrepresentable_gaussian_normalization_is_rejected(sigma) -> None:
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


def test_constant_prediction_fits_only_the_baseline() -> None:
    response = np.arange(12.0)

    coefficients, fitted = _solve_amplitude(np.full(12, 1e-200), response)

    assert coefficients == pytest.approx((0.0, response.mean()))
    assert fitted == pytest.approx(np.full(12, response.mean()))


def test_amplitude_solve_preserves_variation_around_a_large_baseline() -> None:
    prediction = np.arange(32.0) / 31.0
    response = 1e16 + 2.0 * np.arange(32.0)

    coefficients, fitted = _solve_amplitude(prediction, response)

    assert coefficients[0] == pytest.approx(62.0, abs=1e-10)
    assert coefficients[1] == 1e16
    assert np.array_equal(fitted, response)


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


@pytest.mark.parametrize("value", [0.0, 1e-200, 2.0, 3.0, 1e200])
@pytest.mark.parametrize(("metric", "n_frames"), [("r2", 1), ("r2", 10), ("second_field", 80)])
@pytest.mark.parametrize("perfect", [True, False])
def test_constant_responses_have_undefined_scores(fitter, value, metric, n_frames, perfect):
    response = np.full(n_frames, value)
    fitted = (
        response
        if perfect
        else response + max(abs(value), 4.0 if metric == "second_field" else 1.0)
    )
    score = _r_squared if metric == "r2" else fitter._second_field_r2
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        assert np.isnan(score(response, fitted))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("invalid_target", [True, False])
@pytest.mark.parametrize("metric", ["r2", "second_field"])
def test_nonfinite_inputs_have_undefined_scores(fitter, value, invalid_target, metric):
    response = np.array([1.0, 2.0, 3.0]) if metric == "r2" else np.arange(float(fitter.n_frames))
    fitted = response.copy()
    (response if invalid_target else fitted)[1] = value
    score = _r_squared if metric == "r2" else fitter._second_field_r2
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        assert np.isnan(score(response, fitted))


@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200])
@pytest.mark.parametrize(("prediction", "score"), [([1.0, 2.0, 3.0], 1.0), ([3.0, 2.0, 1.0], -3.0)])
def test_r_squared_preserves_perfect_and_negative_scores_across_response_units(
    scale, prediction, score
) -> None:
    response = scale * np.array([1.0, 2.0, 3.0])
    fitted = scale * np.array(prediction)

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        assert _r_squared(response, fitted) == pytest.approx(score)


@pytest.mark.parametrize("scale", [1.0, 1e-200])
def test_excessively_large_predictions_have_negative_infinite_r_squared(scale) -> None:
    response = scale * np.array([1.0, 2.0, 3.0])
    fitted = 1e200 * np.array([1.0, 2.0, 3.0])

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        assert _r_squared(response, fitted) == -np.inf


NOISE_TOLERANCE = {0.0: 0.05, 0.1: 0.5, 0.2: 0.8, 0.4: 1.5, 0.8: 3.0}


GROUND_TRUTH = [
    (0.0, 0.0, 5.0),
    (12.0, 8.0, 4.0),
    (-10.0, 14.0, 7.0),
    (16.0, -12.0, 6.0),
    (-15.0, -9.0, 9.0),
]


@pytest.mark.parametrize("truth", GROUND_TRUTH)
@pytest.mark.parametrize(("noise", "score", "position_error"), [(0.0, 0.99, 0.5), (0.2, 0.8, 1.5)])
def test_recovery(fitter, grid, apertures, truth, noise, score, position_error) -> None:
    fit = fitter.fit_unit(synthesise(grid, apertures, truth, noise=noise, seed=7))
    assert fit.accepted
    assert fit.r2 > score
    assert np.hypot(fit.x0 - truth[0], fit.y0 - truth[1]) < position_error
    lower, upper = fitter.bounds
    assert all(low <= value <= high for value, low, high in zip(geometry(fit), lower, upper))
    if noise == 0.0:
        assert abs(fit.sigma - truth[2]) / truth[2] < 0.05


def test_rejects_pure_noise(fitter) -> None:
    rng = np.random.default_rng(11)
    responses = rng.normal(size=(len(fitter.apertures), 40))

    fits = fitter.fit_all(responses)
    rejected = sum(not fit.accepted for fit in fits)

    assert rejected / len(fits) >= 0.95


def test_amplitude_and_baseline_recovered(fitter, grid, apertures) -> None:
    fit = fitter.fit_unit(synthesise(grid, apertures, (8.0, -6.0, 5.0), beta=4.0, baseline=1.5))

    assert fit.beta == pytest.approx(4.0, rel=0.05)
    assert fit.baseline == pytest.approx(1.5, abs=0.05)


@pytest.mark.parametrize("value", [0.0, 2.0, np.nan, np.inf, -np.inf])
def test_unfittable_response_has_undefined_parameters(fitter, value) -> None:
    response = np.zeros(fitter.n_frames)
    response[:] = value if np.isfinite(value) else 0.0
    response[3] = value
    fit = fitter.fit_unit(response)
    assert not fit.accepted
    assert not fit.converged
    assert np.isnan([fit.r2, fit.x0]).all()


@pytest.mark.parametrize(
    ("shape", "method", "message"),
    [
        ((81,), "fit_unit", "one value per stimulus frame"),
        ((82, 3), "fit_all", "one row per stimulus frame"),
        ((80,), "fit_all", "frames, units"),
        ((81, 2), "validate_all", "one row per stimulus frame"),
        ((80,), "validate_all", "frames, units"),
    ],
)
def test_fitting_and_validation_reject_invalid_response_shapes(
    fitter, validator, shape, method, message
):
    target = validator if method.startswith("validate") else fitter
    with pytest.raises(ValueError, match=message):
        getattr(target, method)(np.zeros(shape))


def test_fit_is_deterministic(fitter, grid, apertures) -> None:
    response = synthesise(grid, apertures, (10.0, 10.0, 6.0), noise=0.1, seed=3)

    first, second = fitter.fit_unit(response), fitter.fit_unit(response)

    assert geometry(first) == geometry(second)


@pytest.mark.parametrize(("noise", "tolerance"), sorted(NOISE_TOLERANCE.items()))
def test_position_error_stays_within_tolerance_for_fixed_noise_draw(
    fitter, grid, apertures, noise, tolerance
) -> None:
    fit = fitter.fit_unit(synthesise(grid, apertures, FIT_TRUTH, noise=noise, seed=13))

    assert np.hypot(fit.x0 - FIT_TRUTH[0], fit.y0 - FIT_TRUTH[1]) < tolerance


def test_r2_decreases_for_fixed_noise_draw(fitter, grid, apertures) -> None:
    scores = [
        fitter.fit_unit(synthesise(grid, apertures, FIT_TRUTH, noise=noise, seed=13)).r2
        for noise in (0.0, 0.2, 0.4, 0.8)
    ]

    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize("kind", ["bar", "wedge", "ring"])
def test_fitter_runs_on_every_stimulus_design(kind) -> None:
    config = StimulusConfig(resolution=RESOLUTION, n_steps=24, directions=(0, 45, 90, 135))
    grid = Grid(RESOLUTION)
    apertures = build_apertures(config, kind).as_float()
    fitter = PRFFitter(grid, apertures, FitConfig(grid_size=8, sigma_bounds=(1.0, 10.0)))

    fit = fitter.fit_unit(synthesise(grid, apertures, FIT_TRUTH))

    assert np.isfinite([fit.x0, fit.y0, fit.sigma]).all()
    assert fit.r2 > 0.9


def test_bar_sweeps_localise_better_than_rings() -> None:
    config = StimulusConfig(resolution=RESOLUTION, n_steps=24, directions=(0, 45, 90, 135))
    grid = Grid(RESOLUTION)
    fit_config = FitConfig(grid_size=8, sigma_bounds=(1.0, 10.0))

    errors = {}
    for kind in ("bar", "ring"):
        apertures = build_apertures(config, kind).as_float()
        fitter = PRFFitter(grid, apertures, fit_config)
        fit = fitter.fit_unit(synthesise(grid, apertures, FIT_TRUTH, noise=0.2, seed=17))
        errors[kind] = np.hypot(fit.x0 - FIT_TRUTH[0], fit.y0 - FIT_TRUTH[1])

    assert errors["bar"] < errors["ring"]


@pytest.mark.parametrize("amplitude", [1e-5, 10.0])
def test_amplitude_scaling_does_not_change_geometry(fitter, grid, apertures, amplitude) -> None:
    reference = fitter.fit_unit(synthesise(grid, apertures, FIT_TRUTH, beta=1.0))
    scaled = fitter.fit_unit(synthesise(grid, apertures, FIT_TRUTH, beta=amplitude))

    assert scaled.accepted
    assert geometry(scaled) == pytest.approx(FIT_TRUTH, abs=1e-6)
    assert geometry(scaled) == pytest.approx(geometry(reference), abs=1e-6)
    assert scaled.beta == pytest.approx(amplitude, rel=1e-6)
    assert scaled.baseline == pytest.approx(0.5, abs=1e-10)


@pytest.mark.parametrize(
    ("response_scale", "aperture_scale", "noise"),
    [
        *[(scale, 1.0, noise) for scale in (1e-200, 1e200) for noise in (0.0, 0.2)],
        *[(1.0, scale, 0.2) for scale in (1e-13, 1e-16, 1e-200, 1e200)],
    ],
)
def test_extreme_units_preserve_fit_and_diagnostics(
    fitter, grid, apertures, fit_config, response_scale, aperture_scale, noise
) -> None:
    response = synthesise(grid, apertures, FIT_TRUTH, noise=noise, seed=5)
    reference = fitter.fit_unit(response)
    scaled_fitter = PRFFitter(grid, aperture_scale * apertures, fit_config)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        fit = scaled_fitter.fit_unit(response_scale * response)
    assert fit.accepted
    assert geometry(fit) == pytest.approx(geometry(reference), abs=1e-6)
    assert fit.beta * aperture_scale / response_scale == pytest.approx(reference.beta, rel=1e-6)
    baseline_tolerance = {"rel": 1e-6} if response_scale != 1 else {"abs": 1e-9}
    assert fit.baseline / response_scale == pytest.approx(reference.baseline, **baseline_tolerance)
    assert fit.r2 == pytest.approx(reference.r2, abs=1e-10)
    assert fit.second_field_r2 == pytest.approx(reference.second_field_r2, abs=1e-10)
    assert errors(fit) == pytest.approx(errors(reference), rel=1e-5)


def test_large_finite_response_ranges_support_fitting_and_validation(
    grid, apertures, sequence, fit_config
) -> None:
    scaled_apertures = 10.0 * apertures
    signal = predict(GaussianReceptiveField(*FIT_TRUTH).weights(grid), scaled_apertures)
    response = 1e308 * (2.0 * (signal - signal.min()) / np.ptp(signal) - 1.0)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        fit = PRFFitter(grid, scaled_apertures, fit_config).fit_unit(response)
    assert fit.accepted
    assert geometry(fit) == pytest.approx(FIT_TRUTH, abs=1e-6)
    assert np.isfinite([fit.beta, fit.baseline, fit.r2, fit.second_field_r2]).all()
    assert fit.r2 == pytest.approx(1.0)
    signal = predict(GaussianReceptiveField(*TRUTH).weights(grid), scaled_apertures)
    response = 1e308 * (2.0 * (signal - signal.min()) / np.ptp(signal) - 1.0)
    validator = CrossValidator(grid, scaled_apertures, sequence.group, fit_config)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = validator.validate_unit(response)
    assert result.fit.accepted
    assert result.validation_complete
    assert result.cv_r2 == pytest.approx(1.0)
    assert result.n_valid_folds == result.n_total_folds == 4


def test_unrepresentable_amplitude_is_not_reported_as_a_fit(grid, apertures, fit_config) -> None:
    response = synthesise(grid, apertures, FIT_TRUTH)
    fitter = PRFFitter(grid, 1e-308 * apertures, fit_config)

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        fit = fitter.fit_unit(response)

    assert not fit.accepted
    assert not fit.converged
    assert np.isnan([fit.beta, fit.baseline, fit.r2]).all()


@pytest.mark.parametrize("noise", [0.0, 0.2])
@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_aperture_units_preserve_second_field_diagnostic(
    grid, apertures, fit_config, noise, scale
) -> None:
    weights = GaussianReceptiveField(14.0, 0.0, 3.0).weights(grid)
    weights += GaussianReceptiveField(-14.0, 0.0, 3.0).weights(grid)
    response = 3.0 * predict(weights, apertures) + 0.5
    response += np.random.default_rng(7).normal(0.0, noise * response.std(), len(response))
    reference = PRFFitter(grid, apertures, fit_config).fit_unit(response)
    fit = PRFFitter(grid, scale * apertures, fit_config).fit_unit(response)

    assert fit.second_field_r2 > 0.2
    assert fit.second_field_r2 == pytest.approx(reference.second_field_r2, abs=1e-8)


@pytest.mark.parametrize(
    "truth",
    [
        (28.0, 0.0, 4.0),
        (-28.0, 0.0, 4.0),
        (0.0, 28.0, 4.0),
        (0.0, -28.0, 4.0),
        (20.0, 20.0, 4.0),
    ],
)
def test_recovers_prf_near_field_boundary(fitter, grid, apertures, truth) -> None:
    fit = fitter.fit_unit(synthesise(grid, apertures, truth))
    x0, y0, _ = truth

    assert fit.accepted
    assert np.hypot(fit.x0 - x0, fit.y0 - y0) < 1.0


def test_centre_never_escapes_the_field(fitter, grid, apertures) -> None:
    lower, upper = fitter.bounds

    for truth in [(31.0, 0.0, 3.0), (0.0, -31.0, 3.0), (30.0, 30.0, 3.0)]:
        fit = fitter.fit_unit(synthesise(grid, apertures, truth))
        assert lower[0] <= fit.x0 <= upper[0]
        assert lower[1] <= fit.y0 <= upper[1]


@pytest.mark.parametrize(
    ("truth", "tolerance"),
    [
        *(((6.0, 6.0, sigma), 0.1 * sigma) for sigma in (2.0, 4.0, 6.0, 9.0)),
        ((0.0, 0.0, 1.5), 0.5),
        ((0.0, 0.0, 8.0), 1.2),
    ],
)
def test_recovery_across_field_sizes(fitter, grid, apertures, truth, tolerance) -> None:
    fit = fitter.fit_unit(synthesise(grid, apertures, truth))
    assert fit.accepted
    assert abs(fit.sigma - truth[2]) < tolerance


def test_sigma_above_the_search_ceiling_is_flagged_and_rejected(grid, apertures) -> None:
    fitter = PRFFitter(grid, apertures, FitConfig(grid_size=8, sigma_bounds=(1.0, 6.0)))
    fit = fitter.fit_unit(synthesise(grid, apertures, (0.0, 0.0, 20.0)))
    assert fit.converged
    assert fit.r2 > fitter.config.r2_threshold
    assert fit.at_bound
    assert fit.sigma_at_bound
    assert abs(fit.sigma - 6.0) <= BOUND_TOLERANCE
    assert not fit.accepted


def test_interior_fit_is_not_flagged_at_bound(fitter, grid, apertures) -> None:
    fit = fitter.fit_unit(synthesise(grid, apertures, (8.0, -6.0, 5.0)))

    assert not fit.at_bound


@pytest.mark.parametrize(
    ("shape", "message"),
    [((4, 64, 64), "frames"), ((40, 32, 32), "grid"), ((64, 64), "frames, height, width")],
)
def test_fitter_rejects_invalid_frame_stacks(grid, shape, message) -> None:
    with pytest.raises(ValueError, match=message):
        PRFFitter(grid, np.ones(shape), FitConfig())


@pytest.mark.parametrize("validate", [False, True])
def test_caller_mutation_keeps_training_and_scoring_consistent(
    grid, apertures, sequence, fit_config, validate
):
    original = apertures.copy()
    response = synthesise(grid, original, TRUTH)
    target = (
        CrossValidator(grid, original, sequence.group, fit_config)
        if validate
        else PRFFitter(grid, original, fit_config)
    )
    original[:] = 0.0
    result = target.validate_unit(response) if validate else target.fit_unit(response)
    fit = result.fit if validate else result
    assert fit.accepted
    assert fit.r2 > 0.99
    assert geometry(fit) == pytest.approx(TRUTH, abs=0.01)
    if validate:
        assert min(result.fold_r2) > 0.99


@pytest.mark.parametrize("owner", ["fitter", "validator"])
def test_public_apertures_cannot_invalidate_cached_predictions(fitter, validator, apertures, owner):
    target = fitter if owner == "fitter" else validator
    with pytest.raises(ValueError, match="read-only"):
        target.apertures[0] = 0.0
    with pytest.raises(ValueError, match="read-only"):
        target.apertures *= 2.0
    with pytest.raises(ValueError, match="WRITEABLE"):
        target.apertures.setflags(write=True)
    with pytest.raises(AttributeError):
        target.apertures = apertures.copy()
    exposed = target.apertures
    exposed.resize((len(apertures), apertures[0].size))
    np.testing.assert_array_equal(target.apertures, apertures)


def test_public_candidates_cannot_invalidate_cached_predictions(fitter, grid, apertures) -> None:
    candidates = fitter.candidates
    with pytest.raises(AttributeError):
        candidates.clear()
    with pytest.raises(TypeError):
        candidates[0] = GaussianReceptiveField(-30.0, -30.0, 1.0)
    with pytest.raises(AttributeError):
        fitter.candidates = candidates[::-1]

    fit = fitter.fit_unit(synthesise(grid, apertures, (12.0, 8.0, 5.0)))

    assert fitter.candidates == candidates
    assert fit.accepted
    assert geometry(fit) == pytest.approx((12.0, 8.0, 5.0), abs=0.01)


def test_a_clean_fit_to_noise_converges_but_is_not_accepted(fitter) -> None:
    response = np.random.default_rng(11).normal(size=fitter.n_frames)

    fit = fitter.fit_unit(response)

    assert fit.converged
    assert fit.r2 < fitter.config.r2_threshold
    assert not fit.accepted


def test_a_centre_at_the_field_edge_is_still_accepted(grid, apertures) -> None:
    fitter = PRFFitter(grid, apertures, FitConfig(grid_size=10, sigma_bounds=(1.0, 10.0)))

    fit = fitter.fit_unit(synthesise(grid, apertures, (32.0, 0.0, 4.0)))

    assert fit.x0_at_bound
    assert not fit.sigma_at_bound
    assert fit.accepted


def test_at_bound_is_the_disjunction_of_the_three_flags() -> None:
    base = {
        "x0": 0.0,
        "y0": 0.0,
        "sigma": 1.0,
        "beta": 1.0,
        "baseline": 0.0,
        "r2": 1.0,
        "converged": True,
        "n_fev": 1,
    }

    assert not UnitFit(**base).at_bound
    assert UnitFit(**base, x0_at_bound=True).at_bound
    assert UnitFit(**base, y0_at_bound=True).at_bound
    assert UnitFit(**base, sigma_at_bound=True).at_bound


def test_a_fit_without_a_threshold_is_never_accepted() -> None:
    fit = UnitFit(0.0, 0.0, 1.0, 1.0, 0.0, r2=1.0, converged=True, n_fev=1)

    assert fit.converged
    assert not fit.accepted


def test_a_suppressed_unit_is_not_accepted_as_a_prf(fitter, grid, apertures) -> None:
    driven = synthesise(grid, apertures, (12.0, 8.0, 5.0), beta=3.0, baseline=0.5)
    suppressed = synthesise(grid, apertures, (12.0, 8.0, 5.0), beta=-3.0, baseline=10.0)

    driven_fit = fitter.fit_unit(driven)
    suppressed_fit = fitter.fit_unit(suppressed)

    assert suppressed_fit.beta < 0.0
    assert suppressed_fit.r2 > 0.99
    assert suppressed_fit.converged
    assert not suppressed_fit.accepted

    assert driven_fit.beta > 0.0
    assert driven_fit.accepted
    assert np.hypot(driven_fit.x0 - 12.0, driven_fit.y0 - 8.0) < 0.5


def test_sigma_ceiling_the_grid_cannot_represent_is_rejected(grid, apertures) -> None:
    with pytest.raises(ConfigError, match="keeps only"):
        PRFFitter(grid, apertures, FitConfig(grid_size=8, sigma_bounds=(1.0, 20.0)))


def test_a_sigma_ceiling_the_grid_can_represent_is_accepted(grid, apertures) -> None:
    PRFFitter(grid, apertures, FitConfig(grid_size=8, sigma_bounds=(1.0, 10.0)))


@pytest.mark.parametrize("sigma", [1.0, 5.0, 10.0])
def test_unit_volume_survives_across_the_admissible_range(grid, sigma) -> None:
    weights = GaussianReceptiveField(0.0, 0.0, sigma).weights(grid)

    assert weights[grid.field_mask].sum() >= MIN_ON_GRID_VOLUME


def test_unit_volume_is_lost_above_the_admissible_range(grid) -> None:
    weights = GaussianReceptiveField(0.0, 0.0, 20.0).weights(grid)

    assert weights[grid.field_mask].sum() < MIN_ON_GRID_VOLUME


@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200])
def test_a_two_lobed_unit_is_flagged_as_misspecified(fitter, grid, apertures, scale) -> None:
    lobes = GaussianReceptiveField(14.0, 0.0, 3.0).weights(grid) + GaussianReceptiveField(
        -14.0, 0.0, 3.0
    ).weights(grid)
    response = 3.0 * predict(lobes, apertures) + 0.5

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        fit = fitter.fit_unit(scale * response)

    assert fit.accepted
    assert fit.r2 > fitter.config.r2_threshold
    assert fit.second_field_r2 > 0.2


def test_a_single_lobed_unit_is_not_flagged(fitter, grid, apertures) -> None:
    for noise, seed in ((0.0, 0), (0.2, 7), (0.5, 7), (0.8, 7)):
        fit = fitter.fit_unit(synthesise(grid, apertures, (12.0, 8.0, 5.0), noise=noise, seed=seed))

        assert fit.second_field_r2 < 0.1


@pytest.mark.slow
def test_misspecification_separates_single_from_two_lobed_units(fitter, grid, apertures) -> None:
    geometries = [
        GaussianReceptiveField(14.0, 0.0, 3.0).weights(grid)
        + GaussianReceptiveField(-14.0, 0.0, 3.0).weights(grid),
        GaussianReceptiveField(16.0, 10.0, 3.0).weights(grid)
        + GaussianReceptiveField(-16.0, -10.0, 3.0).weights(grid),
    ]

    single, two_lobed = [], []
    for noise in (0.0, 0.2, 0.5, 0.8):
        single.append(
            fitter.fit_unit(
                synthesise(grid, apertures, (12.0, 8.0, 5.0), noise=noise, seed=7)
            ).second_field_r2
        )
        for weights in geometries:
            clean = 3.0 * predict(weights, apertures) + 0.5
            response = clean
            if noise > 0:
                rng = np.random.default_rng(7)
                response = clean + rng.normal(0.0, noise * float(clean.std()), size=clean.shape)
            two_lobed.append(fitter.fit_unit(response).second_field_r2)

    noise_only = fitter.fit_unit(
        np.random.default_rng(11).normal(size=fitter.n_frames)
    ).second_field_r2

    assert max(single) < 0.05
    assert noise_only < 0.1
    assert min(two_lobed) > 0.2
    assert max(two_lobed) < 0.5
    assert min(two_lobed) > max(single) * 5


def test_a_perfect_fit_leaves_no_residual_structure(fitter, grid, apertures) -> None:
    fit = fitter.fit_unit(synthesise(grid, apertures, (12.0, 8.0, 5.0)))

    assert fit.r2 > 0.999
    assert fit.second_field_r2 < 0.01


def test_failed_fit_reports_undefined_scores_and_intervals():
    fit = UnitFit.failed()
    assert np.isnan([fit.r2, fit.second_field_r2, *fit.confidence_interval("x0")]).all()


def test_fit_reports_eccentricity_of_the_fitted_centre(fitter, grid, apertures) -> None:
    fit = fitter.fit_unit(synthesise(grid, apertures, (3.0, 4.0, 5.0)))

    assert fit.eccentricity == pytest.approx(5.0, abs=0.5)


def test_second_field_r2_is_undefined_for_a_constant_target_and_varying_prediction(fitter) -> None:
    response = np.full(fitter.n_frames, 3.0)
    fitted = np.linspace(-7.0, 7.0, fitter.n_frames)

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        assert np.isnan(fitter._second_field_r2(response, fitted))


def test_a_stimulus_with_no_distinguishable_frames_reports_no_second_field() -> None:
    grid = Grid(32)
    identical = np.ones((10, *grid.shape))
    fitter = PRFFitter(grid, identical, FitConfig(grid_size=5, sigma_bounds=(1.0, 5.0)))

    assert fitter._second_field_r2(np.arange(10.0), np.zeros(10)) == 0.0


def test_an_optimiser_failure_is_reported_as_an_unfitted_unit(
    monkeypatch, fitter, grid, apertures
) -> None:
    def explode(*args, **kwargs):
        raise np.linalg.LinAlgError("SVD did not converge")

    monkeypatch.setattr(fit_module, "least_squares", explode)

    fit = fitter.fit_unit(synthesise(grid, apertures, (12.0, 8.0, 5.0)))

    assert not fit.accepted
    assert np.isnan(fit.x0)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_apertures_are_refused(grid, value) -> None:
    apertures = np.zeros((20, *grid.shape))
    apertures[0, 0, 0] = value
    with pytest.raises(ValueError, match="apertures must be finite"):
        PRFFitter(grid, apertures, FitConfig(grid_size=5, sigma_bounds=(1.0, 10.0)))


@pytest.mark.parametrize(
    ("resolution", "sigma"),
    [(32, 8.0), (64, 16.0), (96, 24.0), (128, 32.0), (160, 40.0), (200, 50.0), (64, 20.0)],
)
def test_sigma_ceiling_advice_works(resolution, sigma) -> None:
    grid = Grid(resolution)
    apertures = np.zeros((20, *grid.shape))
    with pytest.raises(ConfigError) as raised:
        PRFFitter(grid, apertures, FitConfig(grid_size=5, sigma_bounds=(1.0, sigma)))
    message = str(raised.value)
    advised_sigma = float(re.search(r"bound to about (\d+) px", message).group(1))
    advised_resolution = int(re.search(r"resolution to about (\d+) px", message).group(1))
    PRFFitter(grid, apertures, FitConfig(grid_size=5, sigma_bounds=(1.0, advised_sigma)))
    assert advised_resolution % 2 == 0
    wider = Grid(advised_resolution)
    PRFFitter(
        wider, np.zeros((20, *wider.shape)), FitConfig(grid_size=5, sigma_bounds=(1.0, sigma))
    )


def test_a_sigma_of_exactly_resolution_over_six_is_rejected(grid) -> None:
    assert RESOLUTION_PER_SIGMA > 6.0

    with pytest.raises(ConfigError, match="keeps only"):
        PRFFitter(
            grid,
            np.zeros((20, *grid.shape)),
            FitConfig(grid_size=5, sigma_bounds=(1.0, grid.resolution / 6.0)),
        )


@pytest.mark.parametrize("mode", ["zero", "missing", "duplicate", "nan", "inf"])
def test_unidentifiable_or_nonfinite_jacobians_have_undefined_errors(mode) -> None:
    jacobian = np.random.default_rng(0).normal(size=(80, 3))
    if mode == "zero":
        jacobian[:] = 0
    elif mode == "missing":
        jacobian[:, 2] = 0
    elif mode == "duplicate":
        jacobian[:, 2] = jacobian[:, 1]
    else:
        jacobian[0, 0] = np.nan if mode == "nan" else np.inf
    assert np.isnan(_standard_errors(jacobian, 1.0, 80)).all()


def test_identified_jacobian_retains_the_covariance_formula() -> None:
    jacobian = np.vstack([np.eye(3)] * 4)
    assert _standard_errors(jacobian, 3.5, 12) == pytest.approx((0.5, 0.5, 0.5))


def test_pseudoinverse_cutoff_cannot_report_false_zero_precision() -> None:
    jacobian = np.vstack([np.eye(3)] * 4)
    jacobian[:, 2] *= 1e-10
    assert np.isnan(_standard_errors(jacobian, 3.5, 12)).all()


@pytest.mark.parametrize(
    ("jacobian", "cost", "count"),
    [
        *[(np.vstack([np.eye(3)] * 4), cost, 12) for cost in (-1, np.nan, np.inf)],
        (np.ones((20, 2)), 1.0, 20),
        (np.vstack([np.eye(3)] * 4), 1.0, 13),
        (np.vstack([np.eye(3), np.eye(3)[:2]]), 1.0, 5),
    ],
)
def test_invalid_covariance_inputs_have_undefined_errors(jacobian, cost, count) -> None:
    assert np.isnan(_standard_errors(jacobian, cost, count)).all()


def test_uncertainty_grows_for_fixed_noise_draw(fitter, grid, apertures) -> None:
    widths = []
    for noise in (0.1, 0.3, 0.6):
        fit = fitter.fit_unit(synthesise(grid, apertures, TRUTH, noise=noise, seed=5))
        low, high = fit.confidence_interval("x0")
        widths.append(high - low)

    assert widths == sorted(widths)


def test_response_units_do_not_change_geometry_uncertainty(fitter, grid, apertures) -> None:
    response = synthesise(grid, apertures, TRUTH, noise=0.2, seed=5)
    reference = fitter.fit_unit(response)
    scaled = fitter.fit_unit(1e-5 * response + 0.5)

    assert errors(scaled) == pytest.approx(errors(reference), rel=1e-5)
    assert scaled.r2 == pytest.approx(reference.r2)
    assert scaled.beta == pytest.approx(1e-5 * reference.beta, rel=1e-6)
    assert scaled.baseline == pytest.approx(0.5 + 1e-5 * reference.baseline, abs=1e-10)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_interval_covers_the_generating_parameter(fitter, grid, apertures, seed) -> None:
    fit = fitter.fit_unit(synthesise(grid, apertures, TRUTH, noise=0.2, seed=seed))

    low, high = fit.confidence_interval("x0")

    assert low <= TRUTH[0] <= high


def test_noiseless_fit_uses_distinct_frame_degrees_of_freedom(fitter, grid, apertures):
    fit = fitter.fit_unit(synthesise(grid, apertures, TRUTH))
    assert fit.dof == fitter.n_frames - 5


def test_wider_confidence_level_gives_wider_interval(fitter, grid, apertures) -> None:
    fit = fitter.fit_unit(synthesise(grid, apertures, TRUTH, noise=0.3, seed=8))

    narrow = fit.confidence_interval("x0", level=0.80)
    wide = fit.confidence_interval("x0", level=0.99)

    assert wide[0] < narrow[0]
    assert narrow[1] < wide[1]


@pytest.mark.parametrize(
    "level",
    [0.95, 0.99, np.nextafter(1.0, 0.0), np.nextafter(0.0, 1.0)],
)
def test_confidence_interval_preserves_finite_probability_boundaries(
    fitter, grid, apertures, level
) -> None:
    fit = fitter.fit_unit(synthesise(grid, apertures, TRUTH, noise=0.2, seed=5))
    assert fit.dof == 75
    assert fit.se_x0 > 0.0

    low, high = fit.confidence_interval("x0", level=level)

    assert np.isfinite([low, high]).all()
    assert low <= fit.x0 <= high
    tail = (1.0 - level) / 2.0
    if tail == 0.5:
        assert low == fit.x0 == high
        centered = replace(fit, x0=0.0).confidence_interval("x0", level=level)
        assert np.isfinite(centered).all()
        assert centered[0] <= 0.0 <= centered[1]
    else:
        assert stdtr(fit.dof, (low - fit.x0) / fit.se_x0) == pytest.approx(tail, rel=1e-8, abs=0.0)
    if level in (0.95, 0.99):
        width = float(fit_module.student_t.ppf(0.5 + level / 2.0, fit.dof)) * fit.se_x0
        assert (low, high) == pytest.approx((fit.x0 - width, fit.x0 + width))


@pytest.mark.parametrize(
    ("parameter", "level", "message"),
    [("beta", 0.95, "parameter must be one of"), ("x0", 1.5, "level")],
)
def test_confidence_interval_rejects_invalid_arguments(parameter, level, message):
    with pytest.raises(ValueError, match=message):
        UnitFit.failed().confidence_interval(parameter, level=level)


@pytest.mark.parametrize("parameter", ["x0", "y0", "sigma"])
def test_confidence_interval_covers_each_fitted_parameter(fitter, grid, apertures, parameter):
    fit = fitter.fit_unit(synthesise(grid, apertures, TRUTH, noise=0.2, seed=5))
    assert np.isfinite(errors(fit)).all()
    assert fit.se_x0 > 0.0
    low, high = fit.confidence_interval(parameter)
    assert low < getattr(fit, parameter) < high


@pytest.mark.parametrize("signed_zeros", [False, True])
def test_duplicate_frames_do_not_shrink_the_error_bars(
    grid, apertures, fit_config, signed_zeros
) -> None:
    response = synthesise(grid, apertures, TRUTH, noise=0.2, seed=5)
    duplicates = apertures.copy()
    if signed_zeros:
        duplicates[duplicates == 0.0] = -0.0

    once = PRFFitter(grid, apertures, fit_config)
    twice = PRFFitter(grid, np.concatenate([apertures, duplicates]), fit_config)

    single = once.fit_unit(response)
    repeated = twice.fit_unit(np.concatenate([response, response]))

    assert twice.n_frames == 2 * once.n_frames
    assert twice.n_independent_frames == once.n_independent_frames
    assert repeated.dof == single.dof
    assert errors(repeated) == pytest.approx(errors(single), rel=1e-6)


def test_signed_zero_copies_count_as_six_distinct_masks() -> None:
    masks = np.zeros((6, 2, 3))
    masks[:, 0, 0] = np.arange(1, 7) / 7.0
    duplicates = masks.copy()
    duplicates[duplicates == 0.0] = -0.0
    frames = np.concatenate([masks, duplicates])
    original_bytes = frames.tobytes()

    assert _count_distinct_frames(frames) == 6
    assert frames.tobytes() == original_bytes


def test_distinct_nonzero_float_values_remain_distinct_masks() -> None:
    frames = np.array(
        [
            [[0.0, 0.25]],
            [[-0.0, 0.25]],
            [[np.nextafter(0.0, 1.0), 0.25]],
            [[0.0, np.nextafter(0.25, 1.0)]],
        ]
    )

    assert _count_distinct_frames(frames) == 3


def test_degrees_of_freedom_count_distinct_frames(grid, fit_config) -> None:
    eight = build_apertures(StimulusConfig(resolution=RESOLUTION, n_steps=20), "bar").as_float()
    fitter = PRFFitter(grid, eight, fit_config)

    assert fitter.n_frames == 160
    assert fitter.n_independent_frames == 80


@pytest.mark.slow
@pytest.mark.parametrize("truth", [(12.0, 8.0, 5.0), (28.0, 0.0, 4.0)])
@pytest.mark.parametrize("noise", [0.2, 0.5])
def test_interval_coverage_over_repeated_noise_draws(fitter, grid, apertures, truth, noise) -> None:
    realisations = 200
    covered = 0
    for seed in range(realisations):
        response = synthesise(grid, apertures, truth, noise=noise, seed=1000 + seed)
        low, high = fitter.fit_unit(response).confidence_interval("x0")
        covered += bool(low <= truth[0] <= high)

    assert 0.90 <= covered / realisations <= 0.99


SEEDS = range(21, 41)


LEAK_THRESHOLD = 0.2


@pytest.fixture(scope="module")
def validator(grid, apertures, sequence, fit_config) -> CrossValidator:
    return CrossValidator(grid, apertures, sequence.group, fit_config)


@pytest.mark.parametrize(
    "groups",
    [
        pytest.param((0, 0, 1, 1, 2, 2, 3, 3), id="balanced"),
        pytest.param((20, -3, 20, 7, -3, 20, 7), id="uneven-noncontiguous"),
    ],
)
def test_splitter_partitions_every_frame_by_held_out_group(groups) -> None:
    groups = np.asarray(groups)
    splitter = LeaveOneGroupOut(groups)
    folds = list(splitter.splits())
    unique_groups = np.unique(groups)

    assert len(splitter) == len(folds) == len(unique_groups)
    for held_out, (train, test) in zip(unique_groups, folds):
        np.testing.assert_array_equal(test, np.flatnonzero(groups == held_out))
        np.testing.assert_array_equal(train, np.flatnonzero(groups != held_out))


@pytest.mark.parametrize(
    ("groups", "message"),
    [
        (np.zeros(10, dtype=int), "two sweep groups"),
        (["a", "b"], "finite numeric"),
        (np.zeros((4, 2), dtype=int), "one-dimensional"),
        *[([0.0, 1.0, missing], "finite") for missing in (np.nan, np.inf, -np.inf)],
    ],
)
def test_splitter_rejects_invalid_groups(groups, message) -> None:
    with pytest.raises(ValueError, match=message):
        LeaveOneGroupOut(groups)


@pytest.mark.parametrize("offset", [-1, 1])
def test_cross_validation_requires_a_group_for_every_frame(
    grid, apertures, fit_config, offset
) -> None:
    groups = np.arange(len(apertures) + offset) % 4
    with pytest.raises(ValueError, match="one group per stimulus frame"):
        CrossValidator(grid, apertures, groups, fit_config)


def test_splitter_retains_groups_after_caller_mutates_labels() -> None:
    groups = np.array([0, 0, 1, 1])
    splitter = LeaveOneGroupOut(groups)

    groups[:] = 7
    for labels in (splitter.groups, splitter.unique):
        with pytest.raises(ValueError, match="read-only"):
            labels[0] = 7
        with pytest.raises(ValueError, match="WRITEABLE"):
            labels.setflags(write=True)
    folds = list(splitter.splits())

    assert len(folds) == 2
    assert np.array_equal(folds[0][1], [0, 1])
    assert np.array_equal(folds[1][1], [2, 3])


def test_synthetic_signal_generalises(validator, grid, apertures) -> None:
    result = validator.validate_unit(synthesise(grid, apertures, TRUTH, noise=0.3, seed=9))

    assert result.cv_r2 > 0.8
    assert result.n_folds == 4
    assert result.n_valid_folds == result.n_total_folds == 4
    assert result.n_skipped_folds == 0
    assert result.validation_complete
    assert np.isfinite(result.cv_r2_sd)
    assert result.cv_r2_sd >= 0.0


@pytest.mark.parametrize(
    ("scores", "mean", "sd", "valid"),
    [
        ((0.9, np.nan, np.nan, np.nan), 0.9, np.nan, 1),
        ((1.0, 0.0, -0.5, -2.0), -0.375, 1.25, 4),
        ((), np.nan, np.nan, 0),
        ((np.nan, np.inf, -np.inf), np.nan, np.nan, 0),
        *[((0.6, missing, 0.8, 1.0), 0.8, 0.2, 3) for missing in (np.nan, np.inf, -np.inf)],
        *[
            ((scalar(score), scalar(score)), score, 0.0, 2)
            for scalar in (float, np.float64)
            for score in (0.0, 1e308, -1e308)
        ],
        *(((-scale, 0.0), -scale / 2.0, scale / np.sqrt(2.0), 2) for scale in (1e-200, 1e308)),
        *[
            ((scalar(-scale), scalar(scale)), 0.0, sd, 2)
            for scalar in (float, np.float64)
            for scale, sd in ((1.7e308, np.inf), (1e308, 1e308 * np.sqrt(2.0)))
        ],
    ],
)
def test_fold_summary_reports_finite_scores_and_complete_coverage(scores, mean, sd, valid) -> None:
    result = CrossValidatedFit(UnitFit.failed(), scores)
    with np.errstate(all="raise"):
        assert result.cv_r2 == pytest.approx(mean, rel=1e-12, abs=0.0, nan_ok=True)
        assert result.cv_r2_sd == pytest.approx(sd, rel=1e-12, abs=0.0, nan_ok=True)
    assert result.n_valid_folds == valid
    assert result.n_folds == result.n_total_folds == len(scores)
    assert result.validation_complete == (valid == len(scores) and valid > 0)
    assert not result.fit.accepted


def test_constant_held_out_target_leaves_validation_incomplete(
    validator, grid, apertures, sequence
) -> None:
    response = synthesise(grid, apertures, TRUTH)
    response[sequence.group == sequence.group[0]] = 0.5

    result = validator.validate_unit(response)

    assert np.isnan(result.fold_r2[0])
    assert result.n_valid_folds == 3
    assert result.n_folds == result.n_total_folds == 4
    assert not result.validation_complete


def test_noise_fails_to_generalise(validator) -> None:
    response = np.random.default_rng(4).normal(size=len(validator.apertures))

    result = validator.validate_unit(response)

    assert result.cv_r2 < result.fit.r2
    assert result.cv_r2 < 0.1


def test_optimism_is_larger_for_noise_than_for_signal(validator, grid, apertures) -> None:
    signal = validator.validate_unit(synthesise(grid, apertures, TRUTH, noise=0.3, seed=9))
    noise = validator.validate_unit(np.random.default_rng(4).normal(size=len(validator.apertures)))

    assert noise.optimism > signal.optimism


@pytest.fixture(scope="module")
def random_split_validator(grid, apertures, fit_config) -> CrossValidator:
    shuffled = np.random.default_rng(0).permutation(len(apertures)) % 4
    return CrossValidator(grid, apertures, shuffled, fit_config)


def _score_inflation(validator, random_split, response) -> float:
    return random_split.validate_unit(response).cv_r2 - validator.validate_unit(response).cv_r2


@pytest.mark.slow
def test_random_frame_split_leaks_on_autocorrelated_noise(
    validator, random_split_validator, apertures
) -> None:
    leaks = [
        _score_inflation(
            validator,
            random_split_validator,
            autocorrelated_noise(len(apertures), width=3, seed=seed),
        )
        for seed in SEEDS
    ]

    assert float(np.mean(leaks)) > LEAK_THRESHOLD
    assert sum(leak > 0.0 for leak in leaks) >= 15


@pytest.mark.slow
def test_white_noise_split_difference_is_small_relative_to_seed_variation(
    validator, random_split_validator, apertures
) -> None:
    leaks = [
        _score_inflation(
            validator,
            random_split_validator,
            np.random.default_rng(seed).normal(size=len(apertures)),
        )
        for seed in SEEDS
    ]

    assert abs(float(np.mean(leaks))) < float(np.std(leaks, ddof=1))
    assert 0 < sum(leak > 0.0 for leak in leaks) < len(leaks)


def test_the_autocorrelated_fixture_has_the_correlation_it_claims(apertures) -> None:
    lags = []
    for seed in SEEDS:
        series = autocorrelated_noise(len(apertures), width=3, seed=seed)
        lags.append(float(np.corrcoef(series[:-1], series[1:])[0, 1]))

    assert float(np.mean(lags)) == pytest.approx(1.0 - 1.0 / 3.0, abs=0.05)


def test_held_out_score_uses_train_amplitude(validator, grid, apertures) -> None:
    response = synthesise(grid, apertures, TRUTH, noise=0.2, seed=3)
    train, test, fitter = validator._folds[0]

    train_fit = fitter.fit_unit(response[train])
    field = GaussianReceptiveField(train_fit.x0, train_fit.y0, train_fit.sigma)
    expected = _r_squared(
        response[test],
        train_fit.beta * predict(field.weights(grid), apertures[test]) + train_fit.baseline,
    )

    assert validator.validate_unit(response).fold_r2[0] == pytest.approx(expected)


@pytest.fixture(scope="module")
def small_stack():
    config = StimulusConfig(resolution=32, n_steps=4, directions=(0, 90))
    grid = Grid(32)
    apertures = build_apertures(config, "bar").as_float()
    return grid, apertures, FitConfig(grid_size=5, sigma_bounds=(1.0, 5.0))


def test_every_frame_can_be_its_own_group(small_stack) -> None:
    grid, apertures, config = small_stack
    groups = np.arange(len(apertures))

    validator = CrossValidator(grid, apertures, groups, config)

    assert len(validator._folds) == len(apertures)


def test_a_fold_with_too_few_training_frames_is_skipped(small_stack) -> None:
    grid, apertures, config = small_stack
    groups = np.array([0] * (len(apertures) - 1) + [1])

    validator = CrossValidator(grid, apertures, groups, config)

    assert len(validator._folds) == 1
    _, test, _ = validator._folds[0]
    assert list(test) == [len(apertures) - 1]


def test_skipped_training_fold_keeps_finite_validation_incomplete(small_stack) -> None:
    grid, apertures, config = small_stack
    groups = np.array([0, 0] + [1] * (len(apertures) - 2))
    validator = CrossValidator(grid, apertures, groups, config)

    result = validator.validate_unit(synthesise(grid, apertures, (0.0, 0.0, 3.0)))

    assert np.isfinite(result.cv_r2)
    assert result.n_valid_folds == result.n_folds == 1
    assert result.n_skipped_folds == 1
    assert result.n_total_folds == 2
    assert not result.validation_complete


def test_a_split_where_no_fold_is_large_enough_is_rejected(small_stack) -> None:
    grid, apertures, config = small_stack
    half = len(apertures) // 2
    groups = np.array([0] * half + [1] * (len(apertures) - half))

    with pytest.raises(ValueError, match="no fold retains enough frames"):
        CrossValidator(grid, apertures, groups, config)


def test_an_unfittable_unit_scores_undefined_rather_than_zero(validator) -> None:
    result = validator.validate_unit(np.full(len(validator.apertures), 3.0))

    assert np.isnan(result.cv_r2)
    assert np.isnan(result.cv_r2_sd)
    assert result.n_valid_folds == 0
    assert result.n_folds == result.n_total_folds == 4
    assert not result.validation_complete


def test_validate_all_preserves_unit_identity_and_column_order(validator, grid, apertures) -> None:
    truths = (TRUTH, (-10.0, -7.0, 4.0))
    activations = np.column_stack(
        [synthesise(grid, apertures, truth, noise=0.2, seed=i) for i, truth in enumerate(truths)]
    )
    expected = [validator.validate_unit(response) for response in activations.T]

    results = validator.validate_all(activations)

    assert len(results) == len(truths)
    for result, reference in zip(results, expected):
        assert geometry(result.fit) == pytest.approx(geometry(reference.fit))
        assert result.fold_r2 == pytest.approx(reference.fold_r2)
        assert result.cv_r2 > 0.8
    assert results[0].fit.x0 > 0.0 > results[1].fit.x0
