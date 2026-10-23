import json
import math
import os
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from conftest import RESOLUTION
from cortexprobe.arrays import FloatArray
from cortexprobe.config import (
    ConfigError,
    FitConfig,
    ModelConfig,
    RunConfig,
    StimulusConfig,
)
from cortexprobe.geometry import Grid
from cortexprobe.prf.fit import MIN_ON_GRID_VOLUME
from cortexprobe.prf.model import GaussianReceptiveField
from cortexprobe.stimuli import (
    GENERATORS,
    ApertureSequence,
    BarSweep,
    build_apertures,
    frame_similarity,
)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        (StimulusConfig, "resolution", 4, "at least 8 pixels"),
        (StimulusConfig, "resolution", 65, "must be even"),
        (StimulusConfig, "n_steps", 1, "n_steps must be at least 2"),
        (StimulusConfig, "bar_width_frac", 0.0, "bar_width_frac"),
        (StimulusConfig, "bar_width_frac", 1.0, "bar_width_frac"),
        (StimulusConfig, "directions", (), "at least one sweep direction"),
        (StimulusConfig, "directions", (360,), "degrees in .0, 360."),
        (StimulusConfig, "directions", (-1,), "degrees in .0, 360."),
        (StimulusConfig, "directions", (0, 0, 90), "directions must be unique"),
        (StimulusConfig, "directions", (0, 90, 0), "directions must be unique"),
        (StimulusConfig, "directions", [0, 0, 90], "directions must be unique"),
        (StimulusConfig, "directions", [0, 90, 0], "directions must be unique"),
        (StimulusConfig, "wedge_span_deg", 0.0, "wedge_span_deg"),
        (StimulusConfig, "wedge_span_deg", 361.0, "wedge_span_deg"),
        (StimulusConfig, "ring_thickness_frac", 0.0, "ring_thickness_frac"),
        (StimulusConfig, "ring_thickness_frac", 1.0, "ring_thickness_frac"),
        (StimulusConfig, "max_fold_similarity", 0.0, "max_fold_similarity"),
        (StimulusConfig, "max_fold_similarity", 1.0, "max_fold_similarity"),
        (ModelConfig, "name", "", "model name"),
        (ModelConfig, "layers", (), "at least one tap layer"),
        (ModelConfig, "layers", ("a", "a"), "unique"),
        (ModelConfig, "pool_to", 1, "pool_to"),
        (FitConfig, "grid_size", 2, "grid_size"),
        *[
            (FitConfig, "sigma_bounds", (low, 10.0), "at least 1 pixel")
            for low in (0.999, 0.2, 0.5)
        ],
        *[
            (FitConfig, "sigma_bounds", bounds, "exactly two")
            for bounds in ((), (1.0,), (1.0, 10.0, 20.0), None)
        ],
        *[
            (FitConfig, "sigma_bounds", bounds, "sigma bound must be finite")
            for bad in (math.nan, math.inf, -math.inf)
            for bounds in ((bad, 10.0), (1.0, bad))
        ],
        (FitConfig, "sigma_bounds", (1.0, 10**400), "finite"),
        (RunConfig, "stimulus", None, "StimulusConfig"),
        (ModelConfig, "layers", ("",), "tap layers"),
        (StimulusConfig, "directions", (0, math.nan), "integer"),
        (FitConfig, "sigma_bounds", (5.0, 5.0), "increasing"),
        (FitConfig, "sigma_bounds", (10.0, 5.0), "increasing"),
        (FitConfig, "max_nfev", 0, "max_nfev"),
        (FitConfig, "r2_threshold", -0.1, "r2_threshold"),
        (FitConfig, "r2_threshold", 1.1, "r2_threshold"),
    ],
)
def test_configuration_rejects_invalid_values(section, field, value, message) -> None:
    with pytest.raises(ConfigError, match=message):
        section(**{field: value})


def test_stimulus_dimensions() -> None:
    config = StimulusConfig(resolution=64, n_steps=10, directions=(0, 90))
    assert config.n_bar_frames == 20
    assert config.bar_width_px == config.ring_thickness_px == pytest.approx(8.0)


def test_json_configuration_rejects_repeated_directions(tmp_path) -> None:
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"stimulus": {"directions": [0, 90, 0]}}), encoding="utf-8")

    with pytest.raises(ConfigError, match="directions must be unique"):
        RunConfig.load(path)


@pytest.mark.parametrize("layers", [["conv1", "conv2"], ("conv1", "conv2")])
def test_serialized_layer_sequences_keep_complete_names(layers) -> None:
    restored = ModelConfig.from_dict({"layers": layers})
    assert restored.layers == ("conv1", "conv2")


def test_default_sigma_ceiling_suits_the_default_stimulus() -> None:
    config = RunConfig()
    grid = Grid(config.stimulus.resolution)
    _, sigma_high = config.fit.sigma_bounds

    weights = GaussianReceptiveField(0.0, 0.0, sigma_high).weights(grid)

    assert weights[grid.field_mask].sum() >= MIN_ON_GRID_VOLUME


@pytest.mark.parametrize(
    "config",
    [
        StimulusConfig(directions=(0, 45, 90)),
        ModelConfig(weights_seed=None),
        RunConfig(
            stimulus=StimulusConfig(resolution=64, directions=(0, 90)),
            model=ModelConfig(layers=("features.1",), weights_seed=7),
            fit=FitConfig(grid_size=6),
            seed=3,
        ),
    ],
)
def test_json_round_trip_preserves_configuration_and_canonical_order(config) -> None:
    payload = json.loads(json.dumps(config.to_dict()))
    restored = type(config).from_dict(dict(reversed(list(payload.items()))))
    assert restored == config
    assert restored.to_dict() == config.to_dict()
    if isinstance(restored, StimulusConfig):
        assert isinstance(restored.directions, tuple)
    if isinstance(restored, RunConfig):
        assert isinstance(restored.stimulus, StimulusConfig)
        assert isinstance(restored.fit, FitConfig)
        assert restored.to_json() == config.to_json()


@pytest.mark.parametrize(
    ("section", "value", "message"),
    [
        *[
            (section, value, message)
            for section in (StimulusConfig, ModelConfig, FitConfig, RunConfig)
            for value, message in (({"nonsense": 1}, "unknown keys"), (None, "object"))
        ],
        (StimulusConfig, {"directions": None}, "invalid container"),
        (ModelConfig, {"layers": "conv"}, "invalid container"),
        (ModelConfig, {"layers": {"conv1": 1, "conv2": 2}}, "invalid container"),
    ],
)
def test_configuration_loading_rejects_invalid_sections(section, value, message) -> None:
    with pytest.raises(ConfigError, match=message):
        section.from_dict(value)


@pytest.mark.parametrize(
    "changed",
    [
        RunConfig(seed=1),
        RunConfig(fit=FitConfig(grid_size=13)),
        RunConfig(stimulus=StimulusConfig(max_fold_similarity=0.5)),
    ],
)
def test_configuration_digest_tracks_settings(changed) -> None:
    assert RunConfig().digest() == RunConfig().digest()
    assert len(RunConfig().digest()) == 64
    assert RunConfig().digest() != changed.digest()


@pytest.mark.parametrize(
    "config", [RunConfig(), RunConfig(seed=11, stimulus=StimulusConfig(resolution=64))]
)
def test_saved_configuration_preserves_settings_and_digest(tmp_path, config) -> None:
    path = tmp_path / "run.json"
    config.save(path)
    restored = RunConfig.load(path)
    assert restored == config
    assert restored.digest() == config.digest()
    assert set(json.loads(path.read_text())) == {"stimulus", "model", "fit", "seed"}
    assert path.read_text().endswith("\n")


@pytest.mark.skipif(sys.platform == "win32", reason="The C locale selects ASCII on POSIX.")
def test_utf8_configuration_loads_and_resaves_in_ascii_locale(tmp_path) -> None:
    config = RunConfig(model=ModelConfig(name="modèle"))
    source = tmp_path / "source.json"
    saved = tmp_path / "saved.json"
    source.write_text(json.dumps(config.to_dict(), ensure_ascii=False), encoding="utf-8")
    command = """
import locale
import sys
from pathlib import Path
from cortexprobe.config import RunConfig

assert locale.getpreferredencoding(False).lower() in ("ascii", "us-ascii", "ansi_x3.4-1968")
source, saved = map(Path, sys.argv[1:3])
config = RunConfig.load(source)
assert config.digest() == sys.argv[3]
config.save(saved)
assert RunConfig.load(saved).digest() == config.digest()
"""
    environment = dict(os.environ, LC_ALL="C", PYTHONCOERCECLOCALE="0", PYTHONUTF8="0")
    result = subprocess.run(
        [sys.executable, "-c", command, str(source), str(saved), config.digest()],
        env=environment,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    assert RunConfig.load(saved) == config


@pytest.mark.parametrize("bad", [3.5, math.nan, math.inf, True])
@pytest.mark.parametrize(
    ("section", "field"),
    [
        (StimulusConfig, "resolution"),
        (StimulusConfig, "n_steps"),
        (FitConfig, "grid_size"),
        (FitConfig, "max_nfev"),
        (ModelConfig, "pool_to"),
        (ModelConfig, "weights_seed"),
        (RunConfig, "seed"),
    ],
)
def test_counts_and_seeds_require_integers(section, field, bad) -> None:
    with pytest.raises(ConfigError, match="integer"):
        section(**{field: bad})


@pytest.mark.parametrize(("section", "field"), [(RunConfig, "seed"), (ModelConfig, "weights_seed")])
def test_seeds_must_be_nonnegative(section, field) -> None:
    with pytest.raises(ConfigError, match="nonnegative"):
        section(**{field: -1})


@pytest.mark.parametrize(
    ("section", "config_type", "field", "values", "replacement"),
    [
        ("stimulus", StimulusConfig, "directions", [0, 90], [0, 90, 0]),
        ("model", ModelConfig, "layers", ["features.2"], ["features.2", "features.2"]),
        ("fit", FitConfig, "sigma_bounds", [1.0, 10.0], [20.0, 30.0]),
    ],
)
def test_configuration_copies_mutable_settings(section, config_type, field, values, replacement):
    values = values.copy()
    expected = tuple(values)
    settings = config_type(**{field: values})
    config = RunConfig(**{section: settings})
    digest = config.digest()
    values[:] = replacement
    assert getattr(settings, field) == expected
    assert settings == config_type(**{field: expected})
    assert config.digest() == digest
    assert RunConfig.from_dict(json.loads(config.to_json())) == config
    if section == "stimulus":
        assert settings.n_bar_frames == 64 == 2 * settings.n_steps


def test_a_grid_needs_at_least_two_samples() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        Grid(1)


@pytest.mark.parametrize("resolution", [8.5, np.nan, np.inf, True])
def test_grid_resolution_requires_an_integer(resolution) -> None:
    with pytest.raises(ValueError, match="integer"):
        Grid(resolution)


@pytest.mark.parametrize("resolution", [2, 8, 16, 32, 64])
def test_grid_geometry(resolution) -> None:
    grid = Grid(resolution)
    assert grid.shape == grid.x.shape == grid.y.shape == (resolution, resolution)
    assert grid.radius == pytest.approx(resolution / 2.0)
    assert grid.x.mean() == grid.y.mean() == pytest.approx(0.0)
    assert grid.eccentricity.min() == pytest.approx(np.hypot(0.5, 0.5))
    assert grid.x[0, 0] < 0.0 < grid.x[0, -1]
    assert grid.y[0, 0] > 0.0 > grid.y[-1, 0]
    assert (np.diff(grid.x, axis=1) > 0).all()
    assert (np.diff(grid.y, axis=0) < 0).all()
    np.testing.assert_allclose(grid.x[0], -grid.x[0][::-1])
    np.testing.assert_allclose(grid.eccentricity, np.hypot(grid.x, grid.y))
    assert 0.0 <= grid.polar_angle.min() <= grid.polar_angle.max() < 360.0
    np.testing.assert_array_equal(grid.field_mask, grid.eccentricity <= grid.radius)
    np.testing.assert_array_equal(grid.field_mask, np.fliplr(grid.field_mask))
    np.testing.assert_array_equal(grid.field_mask, np.flipud(grid.field_mask))
    assert grid.field_mask[resolution // 2, resolution // 2]
    if resolution >= 8:
        assert not grid.field_mask[0, 0]


@pytest.mark.parametrize(
    ("row", "column", "expected"),
    [(1, 2, 0.0), (0, 1, 90.0), (1, 0, 180.0), (2, 1, 270.0)],
)
def test_polar_angle_runs_counter_clockwise_from_the_positive_x_axis(row, column, expected) -> None:
    assert Grid(3).polar_angle[row, column] == pytest.approx(expected)


@pytest.mark.parametrize("name", ["x", "y", "eccentricity", "polar_angle", "field_mask"])
def test_grid_arrays_share_protected_geometry(name) -> None:
    grid = Grid(32)
    exposed = getattr(grid, name)
    original = exposed.copy()
    assert np.shares_memory(exposed, getattr(grid, name))
    with pytest.raises(ValueError, match="read-only"):
        exposed[0, 0] = 0
    with pytest.raises(ValueError, match="WRITEABLE"):
        exposed.setflags(write=True)
    exposed.resize((original.size,))
    assert getattr(grid, name).shape == grid.shape
    np.testing.assert_array_equal(getattr(grid, name), original)
    np.testing.assert_array_equal(grid.eccentricity, np.hypot(grid.x, grid.y))


def test_grids_are_frozen_and_comparable() -> None:
    assert Grid(64) == Grid(64)
    assert Grid(64) != Grid(32)

    with pytest.raises(AttributeError):
        Grid(64).resolution = 32  # type: ignore[misc]


KINDS = ("bar", "wedge", "ring")


def _cross_group_similarity(sequence: ApertureSequence) -> FloatArray:
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
@pytest.mark.parametrize(
    "config",
    [StimulusConfig(resolution=64, n_steps=20, directions=(0, 45, 90, 135)), StimulusConfig()],
    ids=["four-axes", "default"],
)
def test_generated_designs_preserve_geometry_groups_and_identity(config, kind) -> None:
    sequence = build_apertures(config, kind)
    repeated = build_apertures(config, kind)
    _assert_no_cross_group_duplicates(sequence)
    assert _cross_group_similarity(sequence).max() <= config.max_fold_similarity
    assert len(np.unique(sequence.group)) >= 2
    declared = GENERATORS[kind](config).n_frames
    assert declared == (config.n_bar_frames if kind == "bar" else config.n_steps)
    assert sequence.n_frames < declared if kind == "ring" else sequence.n_frames == declared
    if kind == "bar":
        assert sorted(np.unique(sequence.group)) == sorted({d % 180 for d in config.directions})
    similarity = frame_similarity(sequence.apertures)
    assert similarity.shape == (sequence.n_frames, sequence.n_frames)
    np.testing.assert_allclose(np.diag(similarity), 1.0)
    np.testing.assert_allclose(similarity, similarity.T)
    assert similarity.min() >= 0.0
    assert not sequence.apertures[:, ~sequence.grid.field_mask].any()
    assert sequence.coverage.shape == (sequence.n_frames,)
    assert ((sequence.coverage > 0.0) & (sequence.coverage < 1.0)).all()
    assert len(sequence.group) == len(sequence.frame_index) == sequence.n_frames
    assert len(np.unique(sequence.frame_index)) == sequence.n_frames
    np.testing.assert_array_equal(sequence.as_float(), sequence.apertures.astype(np.float64))
    for attribute in ("apertures", "frame_index", "group"):
        np.testing.assert_array_equal(getattr(sequence, attribute), getattr(repeated, attribute))


@pytest.mark.parametrize(
    ("directions", "n_steps"),
    [
        ((30, 210, 120), 20),
        ((0, 90, 180, 270), 20),
        ((30, 210), 20),
        ((0, 180), 8),
        ((0, 180, 90), 8),
        ([0, 180, 90], 8),
    ],
)
def test_opposite_sweeps_share_groups_and_keep_unique_frame_identifiers(directions, n_steps):
    config = StimulusConfig(resolution=RESOLUTION, n_steps=n_steps, directions=directions)
    sequence = build_apertures(config, "bar")
    groups = sorted({direction % 180 for direction in directions})
    assert sorted(np.unique(sequence.group)) == groups
    assert sequence.n_frames == len(directions) * n_steps
    assert len(np.unique(sequence.frame_index)) == sequence.n_frames
    for index, direction in enumerate(directions):
        assert (sequence.group[index * n_steps : (index + 1) * n_steps] == direction % 180).all()
        if direction >= 180 and direction - 180 in directions:
            other = directions.index(direction - 180)
            np.testing.assert_array_equal(
                sequence.apertures[index * n_steps : (index + 1) * n_steps],
                sequence.apertures[other * n_steps : (other + 1) * n_steps][::-1],
            )
    if len(groups) > 1:
        _assert_no_cross_group_duplicates(sequence)
        assert _cross_group_similarity(sequence).max() <= config.max_fold_similarity


@pytest.mark.parametrize("directions", [(0, 180), (30, 210), (45, 225), (90, 270), (135, 315)])
@pytest.mark.parametrize("n_steps", [3, 8, 20])
def test_opposite_sweeps_remain_exact_copies_at_pixel_boundaries(directions, n_steps) -> None:
    config = StimulusConfig(resolution=8, n_steps=n_steps, directions=directions)
    sequence = build_apertures(config, "bar")

    assert np.array_equal(sequence.apertures[:n_steps], sequence.apertures[n_steps:][::-1])
    assert len(np.unique(sequence.frame_index)) == sequence.n_frames
    assert np.unique(sequence.group).tolist() == [directions[0]]


def test_a_stricter_threshold_drops_more_frames(stimulus_config) -> None:
    strict = replace(stimulus_config, max_fold_similarity=0.3)

    assert (
        build_apertures(strict, "ring").n_frames < build_apertures(stimulus_config, "ring").n_frames
    )


def test_impossible_pruning_threshold_gives_actionable_advice() -> None:
    config = StimulusConfig(
        resolution=RESOLUTION, n_steps=8, ring_thickness_frac=0.9, max_fold_similarity=0.05
    )
    with pytest.raises(ConfigError, match="only 1 survive pruning") as raised:
        build_apertures(config, "ring")
    assert "narrow the aperture" in str(raised.value)
    assert "Raising n_steps does not help" in str(raised.value)


def test_long_bar_sweeps_keep_direction_and_step_labels_distinct() -> None:
    config = StimulusConfig(resolution=8, n_steps=1001, directions=(0, 1))
    frames, labels, _ = BarSweep(config)._frames()

    assert len(set(labels)) == len(frames)
    decoded = [divmod(label, config.n_steps) for label in labels]
    assert decoded == [
        (direction, step) for direction in config.directions for step in range(config.n_steps)
    ]


@pytest.mark.parametrize(
    ("resolution", "shape", "dtype", "indices", "groups", "error", "message"),
    [
        (64, (3, 64, 64), float, np.arange(3), np.zeros(3), TypeError, "boolean"),
        (64, (3, 8, 8), bool, np.arange(3), np.zeros(3), ValueError, "match the grid"),
        (64, (3, 64, 64), bool, np.arange(2), np.zeros(3), ValueError, "frame_index"),
        (64, (3, 64, 64), bool, np.arange(3), np.zeros(2), ValueError, "group"),
        *[
            (8, shape, bool, np.arange(8), np.zeros(8), ValueError, r"nonempty .* stack")
            for shape in ((8, 8), (1, 8, 8, 1), (0, 8, 8))
        ],
        (8, (2, 8, 8), bool, np.arange(2)[:, None], np.zeros(2), ValueError, "frame_index"),
        (8, (2, 8, 8), bool, np.arange(2), np.zeros((2, 1)), ValueError, "group"),
    ],
)
def test_sequence_rejects_invalid_frames_and_metadata(
    resolution, shape, dtype, indices, groups, error, message
):
    with pytest.raises(error, match=message):
        ApertureSequence(np.ones(shape, dtype=dtype), Grid(resolution), "bar", indices, groups)


def test_unknown_stimulus_kind_names_the_alternatives(stimulus_config) -> None:
    with pytest.raises(ValueError, match="unknown stimulus kind"):
        build_apertures(stimulus_config, "spiral")


def test_generator_registry_covers_every_kind() -> None:
    assert set(GENERATORS) == set(KINDS)


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


def test_sequence_rejects_an_empty_field_frame() -> None:
    grid = Grid(8)
    stack = np.stack([grid.field_mask, grid.field_mask])
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
