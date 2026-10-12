"""Check grouped validation and random-split leakage on white and autocorrelated noise."""

from __future__ import annotations

import numpy as np
import pytest

from cortexprobe.config import FitConfig, StimulusConfig
from cortexprobe.geometry import Grid
from cortexprobe.prf.fit import _r_squared
from cortexprobe.prf.model import GaussianReceptiveField, predict
from cortexprobe.prf.validation import CrossValidator, LeaveOneGroupOut
from cortexprobe.stimuli import build_apertures

from .conftest import autocorrelated_noise, synthesise

TRUTH = (12.0, 8.0, 5.0)
SEEDS = range(21, 41)
LEAK_THRESHOLD = 0.2


@pytest.fixture(scope="module")
def validator(grid, apertures, sequence, fit_config) -> CrossValidator:
    return CrossValidator(grid, apertures, sequence.group, fit_config)


def test_folds_hold_out_whole_groups(sequence) -> None:
    splitter = LeaveOneGroupOut(sequence.group)

    for train, test in splitter.splits():
        assert set(sequence.group[train]).isdisjoint(sequence.group[test])
        assert len(train) + len(test) == len(sequence.group)


def test_splitter_requires_two_groups() -> None:
    with pytest.raises(ValueError, match="two sweep groups"):
        LeaveOneGroupOut(np.zeros(10, dtype=int))


@pytest.mark.parametrize("offset", [-1, 1])
def test_cross_validation_requires_a_group_for_every_frame(
    grid, apertures, fit_config, offset
) -> None:
    groups = np.arange(len(apertures) + offset) % 4
    with pytest.raises(ValueError, match="one group per stimulus frame"):
        CrossValidator(grid, apertures, groups, fit_config)


@pytest.mark.parametrize("missing", [np.nan, np.inf, -np.inf])
def test_splitter_rejects_nonfinite_group_labels(missing) -> None:
    with pytest.raises(ValueError, match="finite"):
        LeaveOneGroupOut(np.array([0.0, 1.0, missing]))


def test_splitter_rejects_nonnumeric_groups_with_a_clear_error() -> None:
    with pytest.raises(ValueError, match="finite numeric"):
        LeaveOneGroupOut(["a", "b"])


def test_folds_test_every_frame_exactly_once(sequence) -> None:
    tests = [test for _, test in LeaveOneGroupOut(sequence.group).splits()]
    assert np.array_equal(np.sort(np.concatenate(tests)), np.arange(sequence.n_frames))


def test_splitter_retains_groups_after_caller_mutates_labels() -> None:
    groups = np.array([0, 0, 1, 1])
    splitter = LeaveOneGroupOut(groups)

    groups[:] = 7
    folds = list(splitter.splits())

    assert len(folds) == 2
    assert np.array_equal(folds[0][1], [0, 1])
    assert np.array_equal(folds[1][1], [2, 3])


def test_validation_uses_the_same_aperture_snapshot_for_training_and_scoring(
    grid, apertures, sequence, fit_config
) -> None:
    original = apertures.copy()
    response = synthesise(grid, original, TRUTH)
    validator = CrossValidator(grid, original, sequence.group, fit_config)

    original[:] = 0.0
    result = validator.validate_unit(response)

    assert result.fit.accepted
    assert result.fit.r2 > 0.99
    assert min(result.fold_r2) > 0.99


def test_real_signal_generalises(validator, grid, apertures) -> None:
    result = validator.validate_unit(synthesise(grid, apertures, TRUTH, noise=0.3, seed=9))

    assert result.cv_r2 > 0.8
    assert result.n_folds == 4


def test_noise_fails_to_generalise(validator) -> None:
    """Seeded white noise has lower held-out R² than in-sample R²."""
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
    """Shuffled frame groups for comparison with whole-sweep validation."""
    shuffled = np.random.default_rng(0).permutation(len(apertures)) % 4
    return CrossValidator(grid, apertures, shuffled, fit_config)


def _score_inflation(validator, random_split, response) -> float:
    return random_split.validate_unit(response).cv_r2 - validator.validate_unit(response).cv_r2


@pytest.mark.slow
def test_random_frame_split_leaks_on_autocorrelated_noise(
    validator, random_split_validator, apertures
) -> None:
    """Check random-split score inflation over 20 seeds of temporally correlated noise."""
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
def test_white_noise_leak_is_indistinguishable_from_zero(
    validator, random_split_validator, apertures
) -> None:
    """Check that the random-split leakage estimate varies around zero for white noise."""
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
    """A width-3 boxcar has lag-1 autocorrelation 1 - 1/3; the fixture must actually deliver it."""
    lags = []
    for seed in SEEDS:
        series = autocorrelated_noise(len(apertures), width=3, seed=seed)
        lags.append(float(np.corrcoef(series[:-1], series[1:])[0, 1]))

    assert float(np.mean(lags)) == pytest.approx(1.0 - 1.0 / 3.0, abs=0.05)


def test_held_out_score_uses_train_amplitude(validator, grid, apertures) -> None:
    """Re-solving amplitude on the test fold would leak; confirm it is not done."""
    response = synthesise(grid, apertures, TRUTH, noise=0.2, seed=3)
    train, test, fitter = validator._folds[0]

    train_fit = fitter.fit_unit(response[train])
    field = GaussianReceptiveField(train_fit.x0, train_fit.y0, train_fit.sigma)
    expected = _r_squared(
        response[test],
        train_fit.beta * predict(field.weights(grid), apertures[test]) + train_fit.baseline,
    )

    assert validator.validate_unit(response).fold_r2[0] == pytest.approx(expected)


def test_validate_all_rejects_wrong_frame_count(validator) -> None:
    with pytest.raises(ValueError, match="one row per stimulus frame"):
        validator.validate_all(np.zeros((len(validator.apertures) + 1, 2)))


def test_cross_validated_fit_reports_fold_dispersion(validator, grid, apertures) -> None:
    result = validator.validate_unit(synthesise(grid, apertures, TRUTH, noise=0.3, seed=9))

    assert np.isfinite(result.cv_r2_sd)
    assert result.cv_r2_sd >= 0.0


@pytest.fixture(scope="module")
def small_stack():
    """Eight frames keep fold-size tests small."""
    config = StimulusConfig(resolution=32, n_steps=4, directions=(0, 90))
    grid = Grid(32)
    apertures = build_apertures(config, "bar").as_float()
    return grid, apertures, FitConfig(grid_size=5, sigma_bounds=(1.0, 5.0))


def test_every_frame_can_be_its_own_group(small_stack) -> None:
    """Single-frame test groups leave seven training frames in this sequence."""
    grid, apertures, config = small_stack
    groups = np.arange(len(apertures))

    validator = CrossValidator(grid, apertures, groups, config)

    assert len(validator._folds) == len(apertures)


def test_a_fold_with_too_few_training_frames_is_skipped(small_stack) -> None:
    """Holding out almost everything leaves too few frames to fit five parameters."""
    grid, apertures, config = small_stack
    groups = np.array([0] * (len(apertures) - 1) + [1])

    validator = CrossValidator(grid, apertures, groups, config)

    assert len(validator._folds) == 1
    _, test, _ = validator._folds[0]
    assert list(test) == [len(apertures) - 1]


def test_a_split_where_no_fold_is_large_enough_is_rejected(small_stack) -> None:
    grid, apertures, config = small_stack
    half = len(apertures) // 2
    groups = np.array([0] * half + [1] * (len(apertures) - half))

    with pytest.raises(ValueError, match="no fold retains enough frames"):
        CrossValidator(grid, apertures, groups, config)


def test_splitter_rejects_multidimensional_groups() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        LeaveOneGroupOut(np.zeros((4, 2), dtype=int))


def test_splitter_length_is_the_group_count(sequence) -> None:
    assert len(LeaveOneGroupOut(sequence.group)) == len(np.unique(sequence.group))


def test_an_unfittable_unit_scores_undefined_rather_than_zero(validator) -> None:
    """A fold that could not be fitted must not contribute a score of zero to the mean."""
    result = validator.validate_unit(np.full(len(validator.apertures), 3.0))

    assert np.isnan(result.cv_r2)
    assert np.isnan(result.cv_r2_sd)


def test_validate_all_rejects_a_non_matrix(validator) -> None:
    with pytest.raises(ValueError, match="frames, units"):
        validator.validate_all(np.zeros(len(validator.apertures)))


def test_validate_all_fits_every_column(validator, grid, apertures) -> None:
    activations = np.column_stack(
        [synthesise(grid, apertures, TRUTH, noise=0.2, seed=s) for s in (1, 2)]
    )

    results = validator.validate_all(activations)

    assert len(results) == 2
    assert all(result.cv_r2 > 0.8 for result in results)
