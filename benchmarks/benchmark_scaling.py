"""Print measured pRF fitting and grouped cross-validation times by unit count."""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from cortexprobe.config import FitConfig, StimulusConfig
from cortexprobe.geometry import Grid
from cortexprobe.prf.fit import RESOLUTION_PER_SIGMA, PRFFitter
from cortexprobe.prf.model import GaussianReceptiveField, predict
from cortexprobe.prf.validation import CrossValidator
from cortexprobe.stimuli import build_apertures

RESOLUTION = 64
UNIT_COUNTS = (1, 5, 10, 25, 50, 100)
SIGMA_HIGH = 10.0
MIN_RESOLUTION = 2 * math.ceil(SIGMA_HIGH * RESOLUTION_PER_SIGMA / 2)


def make_units(grid: Grid, apertures: np.ndarray, n_units: int, seed: int = 0) -> np.ndarray:
    """Activations from randomly placed pRFs, one column per unit."""
    if isinstance(n_units, bool) or not isinstance(n_units, (int, np.integer)) or n_units <= 0:
        raise ValueError("n_units must be a positive integer")
    rng = np.random.default_rng(seed)
    radius = grid.radius * 0.7
    columns = []
    for _ in range(n_units):
        x0, y0 = rng.uniform(-radius, radius, size=2)
        sigma = rng.uniform(3.0, 8.0)
        clean = predict(GaussianReceptiveField(x0, y0, sigma).weights(grid), apertures)
        columns.append(clean + rng.normal(0.0, 0.2 * clean.std(), size=clean.shape))
    return np.column_stack(columns)


def time_call(function, *args) -> tuple[float, object]:
    start = time.perf_counter()
    result = function(*args)
    return time.perf_counter() - start, result


def positive_int(value: str) -> int:
    """An argparse count that cannot produce an empty benchmark."""
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolution",
        type=positive_int,
        default=RESOLUTION,
        help=f"even field width, at least {MIN_RESOLUTION}px for fixed sigma bounds 1-10",
    )
    parser.add_argument("--max-units", type=positive_int, default=max(UNIT_COUNTS))
    arguments = parser.parse_args(argv)
    if arguments.resolution < MIN_RESOLUTION or arguments.resolution % 2:
        parser.error(
            f"--resolution must be even and at least {MIN_RESOLUTION}px "
            "for the benchmark's fixed sigma bounds (1, 10)"
        )

    config = StimulusConfig(
        resolution=arguments.resolution, n_steps=20, directions=(0, 45, 90, 135)
    )
    sequence = build_apertures(config, "bar")
    grid = sequence.grid
    apertures = sequence.as_float()
    fit_config = FitConfig(grid_size=10, sigma_bounds=(1.0, SIGMA_HIGH))

    setup_seconds, fitter = time_call(PRFFitter, grid, apertures, fit_config)
    print(
        f"grid {arguments.resolution}px | {fitter.n_frames} frames | "
        f"{len(fitter.candidates)} candidates | setup {setup_seconds * 1000:.0f} ms\n"
    )

    counts = [count for count in UNIT_COUNTS if count <= arguments.max_units]
    print(
        f"{'units':>6} {'fit (s)':>9} {'per unit (ms)':>14} "
        f"{'cv (s)':>9} {'cv per unit (ms)':>17} {'cv factor':>10}"
    )
    print("-" * 70)

    validator = CrossValidator(grid, apertures, sequence.group, fit_config)
    for n_units in counts:
        activations = make_units(grid, apertures, n_units)
        fit_seconds, _ = time_call(fitter.fit_all, activations)
        cv_seconds, _ = time_call(validator.validate_all, activations)
        print(
            f"{n_units:>6} {fit_seconds:>9.3f} {fit_seconds / n_units * 1000:>14.1f} "
            f"{cv_seconds:>9.3f} {cv_seconds / n_units * 1000:>17.1f} "
            f"{cv_seconds / fit_seconds:>10.1f}x"
        )

    projected_minutes = fit_seconds / n_units * 10_000 / 60
    print(
        f"\nLinear projection from {n_units} units: 10,000 units ~ {projected_minutes:.1f} min "
        "(not measured)"
    )


if __name__ == "__main__":
    main()
