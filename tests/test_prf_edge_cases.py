"""Check fit acceptance at field edges, parameter bounds, and model mismatches."""

from __future__ import annotations

import numpy as np
import pytest

from cortexprobe.config import FitConfig
from cortexprobe.geometry import Grid
from cortexprobe.prf.fit import BOUND_TOLERANCE, PRFFitter, UnitFit
from cortexprobe.prf.model import GaussianReceptiveField

from .conftest import synthesise


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


def test_recovers_smallest_admissible_sigma(grid, apertures) -> None:
    config = FitConfig(grid_size=10, sigma_bounds=(1.0, 10.0))
    fitter = PRFFitter(grid, apertures, config)

    fit = fitter.fit_unit(synthesise(grid, apertures, (0.0, 0.0, 1.5)))

    assert fit.accepted
    assert abs(fit.sigma - 1.5) < 0.5


def test_recovers_largest_admissible_sigma(grid, apertures) -> None:
    config = FitConfig(grid_size=10, sigma_bounds=(1.0, 10.0))
    fitter = PRFFitter(grid, apertures, config)

    fit = fitter.fit_unit(synthesise(grid, apertures, (0.0, 0.0, 8.0)))

    assert fit.accepted
    assert abs(fit.sigma - 8.0) / 8.0 < 0.15


def test_sigma_outside_search_range_is_flagged_at_bound(grid, apertures) -> None:
    """A pRF wider than the search allows must be reported as pinned, not as a measurement."""
    config = FitConfig(grid_size=8, sigma_bounds=(1.0, 6.0))
    fitter = PRFFitter(grid, apertures, config)

    fit = fitter.fit_unit(synthesise(grid, apertures, (0.0, 0.0, 20.0)))

    assert fit.at_bound
    assert abs(fit.sigma - 6.0) <= BOUND_TOLERANCE


def test_interior_fit_is_not_flagged_at_bound(fitter, grid, apertures) -> None:
    fit = fitter.fit_unit(synthesise(grid, apertures, (8.0, -6.0, 5.0)))

    assert not fit.at_bound


def test_fitter_rejects_too_few_frames(grid) -> None:
    with pytest.raises(ValueError, match="frames"):
        PRFFitter(grid, np.ones((4, 64, 64)), FitConfig())


def test_fitter_rejects_mismatched_grid(grid) -> None:
    with pytest.raises(ValueError, match="grid"):
        PRFFitter(grid, np.ones((40, 32, 32)), FitConfig())


def test_fitter_rejects_non_frame_stack(grid) -> None:
    with pytest.raises(ValueError, match="frames, height, width"):
        PRFFitter(grid, np.ones((64, 64)), FitConfig())


def test_zero_response_is_not_fitted(fitter) -> None:
    assert not fitter.fit_unit(np.zeros(fitter.n_frames)).accepted


def test_infinite_response_is_not_fitted(fitter) -> None:
    response = np.zeros(fitter.n_frames)
    response[0] = np.inf

    assert not fitter.fit_unit(response).accepted


def test_receptive_field_far_outside_field_is_finite(grid) -> None:
    weights = GaussianReceptiveField(500.0, 500.0, 3.0).weights(grid)

    assert np.isfinite(weights).all()
    assert weights.sum() == pytest.approx(0.0, abs=1e-12)


def test_a_clean_fit_to_noise_converges_but_is_not_accepted(fitter) -> None:
    """The distinction that matters: the numerics worked, the answer is uninteresting."""
    response = np.random.default_rng(11).normal(size=fitter.n_frames)

    fit = fitter.fit_unit(response)

    assert fit.converged
    assert fit.r2 < fitter.config.r2_threshold
    assert not fit.accepted


def test_sigma_pinned_at_the_ceiling_is_not_accepted(grid, apertures) -> None:
    """A sigma at its bound records where the search stopped, not what the data support."""
    fitter = PRFFitter(grid, apertures, FitConfig(grid_size=8, sigma_bounds=(1.0, 6.0)))

    fit = fitter.fit_unit(synthesise(grid, apertures, (0.0, 0.0, 20.0)))

    assert fit.converged
    assert fit.r2 > fitter.config.r2_threshold
    assert fit.sigma_at_bound
    assert not fit.accepted


def test_a_centre_at_the_field_edge_is_still_accepted(grid, apertures) -> None:
    """Receptive fields really do sit near the field boundary; that is a measurement."""
    fitter = PRFFitter(grid, apertures, FitConfig(grid_size=10, sigma_bounds=(1.0, 10.0)))

    fit = fitter.fit_unit(synthesise(grid, apertures, (32.0, 0.0, 4.0)))

    assert fit.x0_at_bound
    assert not fit.sigma_at_bound
    assert fit.accepted


def test_at_bound_is_the_disjunction_of_the_three_flags() -> None:
    from cortexprobe.prf.fit import UnitFit

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
    """Fail closed: an assembled UnitFit must not pass for a measured pRF by default."""
    from cortexprobe.prf.fit import UnitFit

    fit = UnitFit(0.0, 0.0, 1.0, 1.0, 0.0, r2=1.0, converged=True, n_fev=1)

    assert fit.converged
    assert not fit.accepted


def test_a_suppressed_unit_is_not_accepted_as_a_prf(fitter, grid, apertures) -> None:
    """A response that falls where the aperture covers is suppression, not a receptive field.

    The amplitude is solved by unconstrained least squares, so this fits perfectly with a
    negative beta at exactly the right location. Only the sign distinguishes it.
    """
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


def test_sigma_floor_below_the_pixel_pitch_is_rejected() -> None:
    """The lower guard: an under-sampled Gaussian silently loses its unit volume."""
    from cortexprobe.config import ConfigError

    with pytest.raises(ConfigError, match="at least 1 pixel"):
        FitConfig(sigma_bounds=(0.5, 10.0))


def test_sigma_ceiling_the_grid_cannot_represent_is_rejected(grid, apertures) -> None:
    """The upper guard: a Gaussian wider than about resolution / 6 is truncated by the field."""
    from cortexprobe.config import ConfigError

    with pytest.raises(ConfigError, match="keeps only"):
        PRFFitter(grid, apertures, FitConfig(grid_size=8, sigma_bounds=(1.0, 20.0)))


def test_a_sigma_ceiling_the_grid_can_represent_is_accepted(grid, apertures) -> None:
    PRFFitter(grid, apertures, FitConfig(grid_size=8, sigma_bounds=(1.0, 10.0)))


@pytest.mark.parametrize("sigma", [1.0, 5.0, 10.0])
def test_unit_volume_survives_across_the_admissible_range(grid, sigma) -> None:
    from cortexprobe.prf.fit import MIN_ON_GRID_VOLUME

    weights = GaussianReceptiveField(0.0, 0.0, sigma).weights(grid)

    assert weights[grid.field_mask].sum() >= MIN_ON_GRID_VOLUME


def test_unit_volume_is_lost_above_the_admissible_range(grid) -> None:
    """Pins the reason the ceiling exists, so removing the guard fails a test."""
    from cortexprobe.prf.fit import MIN_ON_GRID_VOLUME

    weights = GaussianReceptiveField(0.0, 0.0, 20.0).weights(grid)

    assert weights[grid.field_mask].sum() < MIN_ON_GRID_VOLUME


def test_a_two_lobed_unit_is_flagged_as_misspecified(fitter, grid, apertures) -> None:
    """One lobe wins the fit, the sigma belongs to neither, and R2 alone looks respectable."""
    lobes = GaussianReceptiveField(14.0, 0.0, 3.0).weights(grid) + GaussianReceptiveField(
        -14.0, 0.0, 3.0
    ).weights(grid)
    from cortexprobe.prf.model import predict

    response = 3.0 * predict(lobes, apertures) + 0.5

    fit = fitter.fit_unit(response)

    assert fit.accepted
    assert fit.r2 > fitter.config.r2_threshold
    assert fit.second_field_r2 > 0.2


def test_a_single_lobed_unit_is_not_flagged(fitter, grid, apertures) -> None:
    for noise, seed in ((0.0, 0), (0.2, 7), (0.5, 7), (0.8, 7)):
        fit = fitter.fit_unit(synthesise(grid, apertures, (12.0, 8.0, 5.0), noise=noise, seed=seed))

        assert fit.second_field_r2 < 0.1


@pytest.mark.slow
def test_misspecification_separates_single_from_two_lobed_units(fitter, grid, apertures) -> None:
    """Verify the recorded single- and two-lobe diagnostic ranges across noise levels."""
    from cortexprobe.prf.model import predict

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


def test_misspecification_does_not_change_acceptance(fitter, grid, apertures) -> None:
    """The diagnostic is reported, not enforced; downstream analysis states its own cut."""
    from cortexprobe.prf.model import predict

    lobes = GaussianReceptiveField(14.0, 0.0, 3.0).weights(grid) + GaussianReceptiveField(
        -14.0, 0.0, 3.0
    ).weights(grid)

    fit = fitter.fit_unit(3.0 * predict(lobes, apertures) + 0.5)

    assert fit.second_field_r2 > 0.2
    assert fit.accepted


def test_a_perfect_fit_leaves_no_residual_structure(fitter, grid, apertures) -> None:
    """A residual of size 1e-16 must not read as a second lobe."""
    fit = fitter.fit_unit(synthesise(grid, apertures, (12.0, 8.0, 5.0)))

    assert fit.r2 > 0.999
    assert fit.second_field_r2 < 0.01


def test_unfittable_response_reports_no_second_field(fitter) -> None:
    assert np.isnan(UnitFit.failed().second_field_r2)


def test_r_squared_of_a_constant_response_is_zero() -> None:
    from cortexprobe.prf.fit import _r_squared

    constant = np.full(10, 3.0)

    assert _r_squared(constant, constant) == 0.0


def test_standard_errors_are_undefined_without_degrees_of_freedom() -> None:
    from cortexprobe.prf.fit import _standard_errors

    errors = _standard_errors(np.ones((3, 3)), 1.0, n_independent=3)

    assert all(np.isnan(error) for error in errors)


def test_standard_errors_are_undefined_when_the_cost_is_not_finite() -> None:
    """Undefined errors are reported as undefined, never as a number."""
    from cortexprobe.prf.fit import _standard_errors

    with np.errstate(invalid="ignore"):
        errors = _standard_errors(np.zeros((20, 3)), float("inf"), n_independent=20)

    assert all(np.isnan(error) for error in errors)


def test_fit_reports_eccentricity_of_the_fitted_centre(fitter, grid, apertures) -> None:
    fit = fitter.fit_unit(synthesise(grid, apertures, (3.0, 4.0, 5.0)))

    assert fit.eccentricity == pytest.approx(5.0, abs=0.5)


def test_fit_all_rejects_a_non_matrix(fitter) -> None:
    with pytest.raises(ValueError, match="frames, units"):
        fitter.fit_all(np.zeros(fitter.n_frames))


def test_second_field_r2_of_a_constant_response_is_zero(fitter) -> None:
    constant = np.full(fitter.n_frames, 2.0)

    assert fitter._second_field_r2(constant, constant) == 0.0


def test_a_stimulus_with_no_distinguishable_frames_reports_no_second_field() -> None:
    """Every candidate prediction is constant, so none can explain any residual."""
    grid = Grid(32)
    identical = np.ones((10, *grid.shape))
    fitter = PRFFitter(grid, identical, FitConfig(grid_size=5, sigma_bounds=(1.0, 5.0)))

    assert fitter._second_field_r2(np.arange(10.0), np.zeros(10)) == 0.0


def test_an_optimiser_failure_is_reported_as_an_unfitted_unit(
    monkeypatch, fitter, grid, apertures
) -> None:
    """A pathological unit must not propagate a numerical exception to the caller."""
    import cortexprobe.prf.fit as fit_module

    def explode(*args, **kwargs):
        raise np.linalg.LinAlgError("SVD did not converge")

    monkeypatch.setattr(fit_module, "least_squares", explode)

    fit = fitter.fit_unit(synthesise(grid, apertures, (12.0, 8.0, 5.0)))

    assert not fit.accepted
    assert np.isnan(fit.x0)


def test_non_finite_apertures_are_refused(grid) -> None:
    """A NaN aperture poisons every candidate prediction; refuse where the cause is visible."""
    apertures = np.zeros((20, *grid.shape))
    apertures[0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="apertures must be finite"):
        PRFFitter(grid, apertures, FitConfig(grid_size=5, sigma_bounds=(1.0, 10.0)))

    apertures[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="apertures must be finite"):
        PRFFitter(grid, apertures, FitConfig(grid_size=5, sigma_bounds=(1.0, 10.0)))


def test_confidence_interval_rejects_an_unknown_parameter(fitter, grid, apertures) -> None:
    """Naming a parameter that was never fitted must say so, not raise AttributeError."""
    fit = fitter.fit_unit(synthesise(grid, apertures, (12.0, 8.0, 5.0)))

    with pytest.raises(ValueError, match="parameter must be one of"):
        fit.confidence_interval("beta")


@pytest.mark.parametrize("parameter", ["x0", "y0", "sigma"])
def test_confidence_interval_accepts_every_fitted_parameter(
    fitter, grid, apertures, parameter
) -> None:
    fit = fitter.fit_unit(synthesise(grid, apertures, (12.0, 8.0, 5.0), noise=0.2, seed=5))
    low, high = fit.confidence_interval(parameter)

    assert low < getattr(fit, parameter) < high


@pytest.mark.parametrize("resolution", [32, 64, 96, 128, 160, 200])
def test_sigma_ceiling_advice_works_at_every_resolution(resolution) -> None:
    """Check suggested sigma bounds against the volume tolerance and require even resolutions."""
    import re

    from cortexprobe.config import ConfigError

    wide = Grid(resolution)
    apertures = np.zeros((20, *wide.shape))
    over = resolution / 4.0
    with pytest.raises(ConfigError) as raised:
        PRFFitter(wide, apertures, FitConfig(grid_size=5, sigma_bounds=(1.0, over)))
    message = str(raised.value)

    advised_sigma = float(re.search(r"bound to about (\d+) px", message).group(1))
    advised_resolution = int(re.search(r"resolution to about (\d+) px", message).group(1))

    PRFFitter(wide, apertures, FitConfig(grid_size=5, sigma_bounds=(1.0, advised_sigma)))

    assert advised_resolution % 2 == 0, "an odd resolution is rejected by StimulusConfig"
    bigger = Grid(advised_resolution)
    PRFFitter(
        bigger, np.zeros((20, *bigger.shape)), FitConfig(grid_size=5, sigma_bounds=(1.0, over))
    )


def test_the_sigma_ceiling_error_gives_advice_that_works(grid) -> None:
    """An error that explains the fix must not suggest one that reproduces the error.

    An earlier message advised raising the resolution to `sigma * 6`, which was itself rejected:
    the 2-D radial mass within 3 sigma is 0.9889, under the 0.99 tolerance, so the true ratio is
    about 6.1 rather than 6.
    """
    import re

    from cortexprobe.config import ConfigError

    apertures = np.zeros((20, *grid.shape))
    with pytest.raises(ConfigError) as raised:
        PRFFitter(grid, apertures, FitConfig(grid_size=5, sigma_bounds=(1.0, 20.0)))
    message = str(raised.value)

    advised_sigma = float(re.search(r"bound to about (\d+) px", message).group(1))
    advised_resolution = int(re.search(r"resolution to about (\d+) px", message).group(1))

    PRFFitter(grid, apertures, FitConfig(grid_size=5, sigma_bounds=(1.0, advised_sigma)))
    wider = Grid(advised_resolution)
    PRFFitter(
        wider,
        np.zeros((20, *wider.shape)),
        FitConfig(grid_size=5, sigma_bounds=(1.0, 20.0)),
    )


def test_a_sigma_of_exactly_resolution_over_six_is_rejected(grid) -> None:
    """The resolution/6 rule of thumb is slightly too generous; the guard is the authority."""
    from cortexprobe.config import ConfigError
    from cortexprobe.prf.fit import RESOLUTION_PER_SIGMA

    assert RESOLUTION_PER_SIGMA > 6.0

    with pytest.raises(ConfigError, match="keeps only"):
        PRFFitter(
            grid,
            np.zeros((20, *grid.shape)),
            FitConfig(grid_size=5, sigma_bounds=(1.0, grid.resolution / 6.0)),
        )
