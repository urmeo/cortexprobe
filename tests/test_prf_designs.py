"""Check parameter recovery and uncertainty across stimulus designs and noise levels."""

from __future__ import annotations

import numpy as np
import pytest

from cortexprobe.config import FitConfig, StimulusConfig
from cortexprobe.geometry import Grid
from cortexprobe.prf.fit import PRFFitter
from cortexprobe.stimuli import build_apertures

from .conftest import RESOLUTION, synthesise

TRUTH = (10.0, -8.0, 5.0)
NOISE_TOLERANCE = {0.0: 0.05, 0.1: 0.5, 0.2: 0.8, 0.4: 1.5, 0.8: 3.0}


@pytest.mark.parametrize(("noise", "tolerance"), sorted(NOISE_TOLERANCE.items()))
def test_position_error_stays_within_tolerance_as_noise_grows(
    fitter, grid, apertures, noise, tolerance
) -> None:
    fit = fitter.fit_unit(synthesise(grid, apertures, TRUTH, noise=noise, seed=13))

    assert np.hypot(fit.x0 - TRUTH[0], fit.y0 - TRUTH[1]) < tolerance


def test_r2_decreases_monotonically_with_noise(fitter, grid, apertures) -> None:
    scores = [
        fitter.fit_unit(synthesise(grid, apertures, TRUTH, noise=noise, seed=13)).r2
        for noise in (0.0, 0.2, 0.4, 0.8)
    ]

    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize("kind", ["bar", "wedge", "ring"])
def test_fitter_runs_on_every_stimulus_design(kind) -> None:
    config = StimulusConfig(resolution=RESOLUTION, n_steps=24, directions=(0, 45, 90, 135))
    grid = Grid(RESOLUTION)
    apertures = build_apertures(config, kind).as_float()
    fitter = PRFFitter(grid, apertures, FitConfig(grid_size=8, sigma_bounds=(1.0, 10.0)))

    fit = fitter.fit_unit(synthesise(grid, apertures, TRUTH))

    assert np.isfinite([fit.x0, fit.y0, fit.sigma]).all()
    assert fit.r2 > 0.9


def test_bar_sweeps_localise_better_than_rings() -> None:
    """An expanding ring constrains eccentricity but not polar angle, so position is weaker."""
    config = StimulusConfig(resolution=RESOLUTION, n_steps=24, directions=(0, 45, 90, 135))
    grid = Grid(RESOLUTION)
    fit_config = FitConfig(grid_size=8, sigma_bounds=(1.0, 10.0))

    errors = {}
    for kind in ("bar", "ring"):
        apertures = build_apertures(config, kind).as_float()
        fitter = PRFFitter(grid, apertures, fit_config)
        fit = fitter.fit_unit(synthesise(grid, apertures, TRUTH, noise=0.2, seed=17))
        errors[kind] = np.hypot(fit.x0 - TRUTH[0], fit.y0 - TRUTH[1])

    assert errors["bar"] < errors["ring"]


@pytest.mark.parametrize("sigma", [2.0, 4.0, 6.0, 9.0])
def test_recovery_across_receptive_field_sizes(fitter, grid, apertures, sigma) -> None:
    fit = fitter.fit_unit(synthesise(grid, apertures, (6.0, 6.0, sigma)))

    assert fit.accepted
    assert abs(fit.sigma - sigma) / sigma < 0.1


@pytest.mark.parametrize("amplitude", [1e-5, 10.0])
def test_amplitude_scaling_does_not_change_geometry(fitter, grid, apertures, amplitude) -> None:
    reference = fitter.fit_unit(synthesise(grid, apertures, TRUTH, beta=1.0))
    scaled = fitter.fit_unit(synthesise(grid, apertures, TRUTH, beta=amplitude))

    assert scaled.accepted
    assert (scaled.x0, scaled.y0, scaled.sigma) == pytest.approx(TRUTH, abs=1e-6)
    assert (scaled.x0, scaled.y0, scaled.sigma) == pytest.approx(
        (reference.x0, reference.y0, reference.sigma), abs=1e-6
    )
    assert scaled.beta == pytest.approx(amplitude, rel=1e-6)
    assert scaled.baseline == pytest.approx(0.5, abs=1e-10)


@pytest.mark.parametrize("scale", [1e-200, 1e200])
@pytest.mark.parametrize("noise", [0.0, 0.2])
def test_extreme_response_units_preserve_fit_and_diagnostics(
    fitter, grid, apertures, scale, noise
) -> None:
    response = synthesise(grid, apertures, TRUTH, noise=noise, seed=5)
    reference = fitter.fit_unit(response)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        scaled = fitter.fit_unit(scale * response)

    assert scaled.accepted
    assert (scaled.x0, scaled.y0, scaled.sigma) == pytest.approx(
        (reference.x0, reference.y0, reference.sigma), abs=1e-6
    )
    assert scaled.beta / scale == pytest.approx(reference.beta, rel=1e-6)
    assert scaled.baseline / scale == pytest.approx(reference.baseline, rel=1e-6)
    assert scaled.r2 == pytest.approx(reference.r2, abs=1e-10)
    assert scaled.second_field_r2 == pytest.approx(reference.second_field_r2, abs=1e-10)
