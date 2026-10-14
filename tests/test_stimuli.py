"""Check stimulus construction and cross-fold duplicate and similarity constraints."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cortexprobe.arrays import FloatArray
from cortexprobe.config import ConfigError, StimulusConfig
from cortexprobe.geometry import Grid
from cortexprobe.stimuli import (
    GENERATORS,
    ApertureSequence,
    BarSweep,
    ExpandingRing,
    RotatingWedge,
    build_apertures,
    frame_similarity,
)

from .conftest import RESOLUTION

KINDS = ("bar", "wedge", "ring")


def _cross_group_similarity(sequence: ApertureSequence) -> FloatArray:
    """Largest similarity between each held-out frame and any training frame, over all folds."""
    similarity = frame_similarity(sequence.apertures)
    worst: list[float] = []
    for group in np.unique(sequence.group):
        test = np.flatnonzero(sequence.group == group)
        train = np.flatnonzero(sequence.group != group)
        worst.extend(similarity[np.ix_(test, train)].max(axis=1).tolist())
    return np.asarray(worst)


def _assert_no_cross_group_duplicates(sequence: ApertureSequence) -> None:
    flat = sequence.apertures.reshape(sequence.n_frames, -1)
    for group in np.unique(sequence.group):
        held_out = {flat[i].tobytes() for i in np.flatnonzero(sequence.group == group)}
        training = {flat[i].tobytes() for i in np.flatnonzero(sequence.group != group)}
        assert held_out.isdisjoint(training)


@pytest.mark.parametrize("kind", KINDS)
def test_no_duplicate_frame_crosses_a_fold_boundary(stimulus_config, kind) -> None:
    _assert_no_cross_group_duplicates(build_apertures(stimulus_config, kind))


@pytest.mark.parametrize("kind", KINDS)
def test_held_out_frames_are_dissimilar_from_training_frames(stimulus_config, kind) -> None:
    sequence = build_apertures(stimulus_config, kind)

    assert _cross_group_similarity(sequence).max() <= stimulus_config.max_fold_similarity


@pytest.mark.parametrize("kind", KINDS)
def test_default_config_holds_the_same_invariant(default_stimulus_config, kind) -> None:
    sequence = build_apertures(default_stimulus_config, kind)

    _assert_no_cross_group_duplicates(sequence)
    assert _cross_group_similarity(sequence).max() <= default_stimulus_config.max_fold_similarity


@pytest.mark.parametrize("kind", KINDS)
def test_every_design_yields_at_least_two_groups(default_stimulus_config, kind) -> None:
    sequence = build_apertures(default_stimulus_config, kind)

    assert len(np.unique(sequence.group)) >= 2


def test_opposite_sweeps_share_a_group() -> None:
    config = StimulusConfig(resolution=RESOLUTION, n_steps=20, directions=(30, 210, 120))
    sequence = build_apertures(config, "bar")

    assert sorted(np.unique(sequence.group)) == [30, 120]


def test_opposite_sweeps_produce_identical_frames() -> None:
    config = StimulusConfig(resolution=RESOLUTION, n_steps=20, directions=(30, 210, 120))
    sequence = build_apertures(config, "bar")
    flat = sequence.apertures.reshape(sequence.n_frames, -1)
    n_steps = config.n_steps

    for step in range(n_steps):
        assert np.array_equal(flat[step], flat[2 * n_steps - 1 - step])


def test_a_four_direction_set_can_still_contain_opposite_sweeps() -> None:
    config = StimulusConfig(resolution=RESOLUTION, n_steps=20, directions=(0, 90, 180, 270))
    sequence = build_apertures(config, "bar")

    assert sorted(np.unique(sequence.group)) == [0, 90]
    _assert_no_cross_group_duplicates(sequence)
    assert _cross_group_similarity(sequence).max() <= config.max_fold_similarity


def test_a_single_sweep_axis_still_builds() -> None:
    """Aperture generation permits a single group; cross-validation rejects it separately."""
    config = StimulusConfig(resolution=RESOLUTION, n_steps=20, directions=(30, 210))

    assert build_apertures(config, "bar").n_frames == 40


@pytest.mark.parametrize("directions", [(0, 180), (0, 180, 90), [0, 180, 90]])
def test_opposite_directions_keep_unique_frame_identifiers(directions) -> None:
    config = StimulusConfig(resolution=RESOLUTION, n_steps=8, directions=directions)
    sequence = build_apertures(config, "bar")

    assert sequence.n_frames == len(directions) * config.n_steps
    assert len(np.unique(sequence.frame_index)) == sequence.n_frames
    assert (sequence.group[: 2 * config.n_steps] == 0).all()


def test_distinct_axes_get_distinct_groups(stimulus_config) -> None:
    sequence = build_apertures(stimulus_config, "bar")

    assert sorted(np.unique(sequence.group)) == [0, 45, 90, 135]


def test_ring_drops_frames_that_straddle_a_block_boundary(stimulus_config) -> None:
    """Ring blocks come from index arithmetic; adjacent annuli overlap across the boundary."""
    sequence = build_apertures(stimulus_config, "ring")

    assert sequence.n_frames < ExpandingRing(stimulus_config).n_frames


def test_bar_and_wedge_keep_every_frame(stimulus_config) -> None:
    assert build_apertures(stimulus_config, "bar").n_frames == BarSweep(stimulus_config).n_frames
    assert (
        build_apertures(stimulus_config, "wedge").n_frames
        == RotatingWedge(stimulus_config).n_frames
    )


def test_a_stricter_threshold_drops_more_frames(stimulus_config) -> None:
    strict = replace(stimulus_config, max_fold_similarity=0.3)

    assert (
        build_apertures(strict, "ring").n_frames < build_apertures(stimulus_config, "ring").n_frames
    )


def test_impossible_threshold_is_rejected_rather_than_silently_collapsing() -> None:
    config = StimulusConfig(
        resolution=RESOLUTION, n_steps=8, ring_thickness_frac=0.9, max_fold_similarity=0.05
    )

    with pytest.raises(ConfigError, match="only 1 survive pruning"):
        build_apertures(config, "ring")


def test_frame_similarity_is_a_cosine(stimulus_config) -> None:
    sequence = build_apertures(stimulus_config, "bar")
    similarity = frame_similarity(sequence.apertures)

    assert similarity.shape == (sequence.n_frames, sequence.n_frames)
    assert np.allclose(np.diag(similarity), 1.0)
    assert np.allclose(similarity, similarity.T)
    assert similarity.min() >= 0.0


@pytest.mark.parametrize("kind", KINDS)
def test_frames_are_confined_to_the_field_mask(stimulus_config, kind) -> None:
    sequence = build_apertures(stimulus_config, kind)
    outside = ~sequence.grid.field_mask

    assert not sequence.apertures[:, outside].any()


@pytest.mark.parametrize("kind", KINDS)
def test_coverage_is_a_fraction_of_the_field(stimulus_config, kind) -> None:
    sequence = build_apertures(stimulus_config, kind)
    coverage = sequence.coverage

    assert coverage.shape == (sequence.n_frames,)
    assert ((coverage > 0.0) & (coverage < 1.0)).all()


@pytest.mark.parametrize("kind", KINDS)
def test_labels_and_groups_label_every_frame(stimulus_config, kind) -> None:
    sequence = build_apertures(stimulus_config, kind)

    assert len(sequence.frame_index) == sequence.n_frames
    assert len(sequence.group) == sequence.n_frames
    assert len(set(sequence.frame_index.tolist())) == sequence.n_frames


def test_long_bar_sweeps_keep_direction_and_step_labels_distinct() -> None:
    config = StimulusConfig(resolution=8, n_steps=1001, directions=(0, 1))
    frames, labels, _ = BarSweep(config)._frames()

    assert len(set(labels)) == len(frames)
    decoded = [divmod(label, config.n_steps) for label in labels]
    assert decoded == [
        (direction, step) for direction in config.directions for step in range(config.n_steps)
    ]


def test_declared_frame_count_matches_the_unpruned_design(stimulus_config) -> None:
    assert BarSweep(stimulus_config).n_frames == stimulus_config.n_bar_frames
    assert RotatingWedge(stimulus_config).n_frames == stimulus_config.n_steps
    assert ExpandingRing(stimulus_config).n_frames == stimulus_config.n_steps


def test_as_float_matches_the_boolean_stack(stimulus_config) -> None:
    sequence = build_apertures(stimulus_config, "bar")

    assert np.array_equal(sequence.as_float(), sequence.apertures.astype(np.float64))


def test_sequence_rejects_non_boolean_apertures(grid) -> None:
    with pytest.raises(TypeError, match="boolean"):
        ApertureSequence(np.ones((3, *grid.shape)), grid, "bar", np.arange(3), np.zeros(3))


def test_sequence_rejects_frames_that_do_not_match_the_grid(grid) -> None:
    with pytest.raises(ValueError, match="match the grid"):
        ApertureSequence(np.ones((3, 8, 8), dtype=bool), grid, "bar", np.arange(3), np.zeros(3))


def test_sequence_rejects_unlabelled_frames(grid) -> None:
    stack = np.ones((3, *grid.shape), dtype=bool)

    with pytest.raises(ValueError, match="frame_index"):
        ApertureSequence(stack, grid, "bar", np.arange(2), np.zeros(3))
    with pytest.raises(ValueError, match="group"):
        ApertureSequence(stack, grid, "bar", np.arange(3), np.zeros(2))


def test_unknown_stimulus_kind_names_the_alternatives(stimulus_config) -> None:
    with pytest.raises(ValueError, match="unknown stimulus kind"):
        build_apertures(stimulus_config, "spiral")


def test_generator_registry_covers_every_kind() -> None:
    assert set(GENERATORS) == set(KINDS)


@pytest.mark.parametrize("kind", KINDS)
def test_generation_is_deterministic(stimulus_config, kind) -> None:
    first = build_apertures(stimulus_config, kind)
    second = build_apertures(stimulus_config, kind)

    assert np.array_equal(first.apertures, second.apertures)
    assert np.array_equal(first.frame_index, second.frame_index)
    assert np.array_equal(first.group, second.group)


def test_pruning_error_does_not_advise_raising_n_steps() -> None:
    """Increasing n_steps shortens the step without narrowing the aperture."""
    config = StimulusConfig(
        resolution=RESOLUTION, n_steps=8, ring_thickness_frac=0.9, max_fold_similarity=0.05
    )

    with pytest.raises(ConfigError) as raised:
        build_apertures(config, "ring")
    message = str(raised.value)

    assert "narrow the aperture" in message
    assert "Raising n_steps does not help" in message


def test_more_steps_means_more_cross_group_overlap() -> None:
    worst = []
    for n_steps in (4, 8, 16, 32):
        config = StimulusConfig(
            resolution=32, n_steps=n_steps, ring_thickness_frac=0.5, max_fold_similarity=0.999
        )
        worst.append(_cross_group_similarity(build_apertures(config, "ring")).max())

    assert worst == sorted(worst)


@pytest.mark.parametrize(
    ("kind", "overrides"),
    [
        ("bar", {"bar_width_frac": 0.001}),
        ("ring", {"ring_thickness_frac": 0.001}),
        ("wedge", {"wedge_span_deg": 0.001}),
    ],
)
def test_empty_generated_apertures_are_rejected(kind, overrides) -> None:
    config = StimulusConfig(resolution=8, n_steps=8, directions=(0, 90), **overrides)
    with pytest.raises(ConfigError, match="empty apertures at resolution=8"):
        build_apertures(config, kind)


def test_coverage_counts_only_the_circular_field() -> None:
    grid = Grid(8)
    whole = grid.field_mask
    half = whole & (grid.x >= 0)
    stack = np.stack([whole, half, np.ones(grid.shape, dtype=bool)])
    sequence = ApertureSequence(stack, grid, "test", np.arange(3), np.zeros(3))
    assert sequence.coverage == pytest.approx([1, half.sum() / whole.sum(), 1])


@pytest.mark.parametrize("shape", [(8, 8), (1, 8, 8, 1), (0, 8, 8)])
def test_sequence_requires_a_nonempty_three_dimensional_stack(shape) -> None:
    with pytest.raises(ValueError, match=r"nonempty .* stack"):
        ApertureSequence(np.ones(shape, dtype=bool), Grid(8), "test", np.arange(8), np.zeros(8))


def test_sequence_rejects_an_empty_field_frame_and_multidimensional_metadata() -> None:
    grid = Grid(8)
    stack = np.stack([grid.field_mask, grid.field_mask])
    with pytest.raises(ValueError, match="frame_index"):
        ApertureSequence(stack, grid, "test", np.arange(2)[:, None], np.zeros(2))
    with pytest.raises(ValueError, match="group"):
        ApertureSequence(stack, grid, "test", np.arange(2), np.zeros((2, 1)))
    stack[0] = False
    with pytest.raises(ConfigError, match="at least one field pixel"):
        ApertureSequence(stack, grid, "test", np.arange(2), np.zeros(2))


def test_empty_frame_similarity_is_undefined() -> None:
    with pytest.raises(ValueError, match="nonempty aperture frames"):
        frame_similarity(np.zeros((2, 8, 8), dtype=bool))


def test_sequence_preserves_list_metadata_compatibility() -> None:
    grid = Grid(8)
    sequence = ApertureSequence(np.stack([grid.field_mask]), grid, "test", [0], [0])
    assert sequence.coverage.tolist() == [1]
