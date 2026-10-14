"""Check recovery of known synthetic pRF parameters."""

from __future__ import annotations

import numpy as np
import pytest

from .conftest import synthesise

GROUND_TRUTH = [
    (0.0, 0.0, 5.0),
    (12.0, 8.0, 4.0),
    (-10.0, 14.0, 7.0),
    (16.0, -12.0, 6.0),
    (-15.0, -9.0, 9.0),
]


@pytest.mark.parametrize("truth", GROUND_TRUTH)
def test_recovers_noiseless_prf(fitter, grid, apertures, truth) -> None:
    fit = fitter.fit_unit(synthesise(grid, apertures, truth))
    x0, y0, sigma = truth

    assert fit.accepted
    assert fit.r2 > 0.99
    assert np.hypot(fit.x0 - x0, fit.y0 - y0) < 0.5
    assert abs(fit.sigma - sigma) / sigma < 0.05


@pytest.mark.parametrize("truth", GROUND_TRUTH)
def test_recovers_prf_under_moderate_noise(fitter, grid, apertures, truth) -> None:
    fit = fitter.fit_unit(synthesise(grid, apertures, truth, noise=0.2, seed=7))
    x0, y0, _ = truth

    assert fit.accepted
    assert fit.r2 > 0.8
    assert np.hypot(fit.x0 - x0, fit.y0 - y0) < 1.5


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


def test_constant_response_is_not_fitted(fitter) -> None:
    fit = fitter.fit_unit(np.full(len(fitter.apertures), 2.0))

    assert not fit.accepted
    assert not fit.converged
    assert np.isnan(fit.r2)
    assert np.isnan(fit.x0)


def test_non_finite_response_is_not_fitted(fitter) -> None:
    response = np.zeros(len(fitter.apertures))
    response[3] = np.nan

    assert not fitter.fit_unit(response).accepted


def test_wrong_length_response_raises(fitter) -> None:
    with pytest.raises(ValueError, match="one value per stimulus frame"):
        fitter.fit_unit(np.zeros(len(fitter.apertures) + 1))


def test_fitted_parameters_respect_bounds(fitter, grid, apertures) -> None:
    lower, upper = fitter.bounds

    for truth in GROUND_TRUTH:
        fit = fitter.fit_unit(synthesise(grid, apertures, truth))
        assert lower[0] <= fit.x0 <= upper[0]
        assert lower[1] <= fit.y0 <= upper[1]
        assert lower[2] <= fit.sigma <= upper[2]


def test_fit_is_deterministic(fitter, grid, apertures) -> None:
    response = synthesise(grid, apertures, (10.0, 10.0, 6.0), noise=0.1, seed=3)

    first, second = fitter.fit_unit(response), fitter.fit_unit(response)

    assert (first.x0, first.y0, first.sigma) == (second.x0, second.y0, second.sigma)


def test_fit_all_rejects_wrong_frame_count(fitter) -> None:
    with pytest.raises(ValueError, match="one row per stimulus frame"):
        fitter.fit_all(np.zeros((len(fitter.apertures) + 2, 3)))
