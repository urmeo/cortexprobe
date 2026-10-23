import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from cortexprobe.arrays import FloatArray
from cortexprobe.config import FitConfig, StimulusConfig
from cortexprobe.geometry import Grid
from cortexprobe.prf.fit import PRFFitter
from cortexprobe.prf.model import GaussianReceptiveField, predict
from cortexprobe.stimuli import ApertureSequence, build_apertures

RESOLUTION = 64


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: runs hundreds of fits; deselect with -m 'not slow'")


def load_script(relative_path: str) -> ModuleType:
    path = Path(__file__).resolve().parent.parent / relative_path
    spec = importlib.util.spec_from_file_location(f"_test_{path.stem}", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with pytest.MonkeyPatch.context() as paths:
        paths.syspath_prepend(str(path.parent))
        spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def grid() -> Grid:
    return Grid(RESOLUTION)


@pytest.fixture(scope="session")
def stimulus_config() -> StimulusConfig:
    return StimulusConfig(resolution=RESOLUTION, n_steps=20, directions=(0, 45, 90, 135))


@pytest.fixture(scope="session")
def sequence(stimulus_config: StimulusConfig) -> ApertureSequence:
    return build_apertures(stimulus_config, "bar")


@pytest.fixture(scope="session")
def apertures(sequence: ApertureSequence) -> FloatArray:
    return sequence.as_float()


@pytest.fixture(scope="session")
def fit_config() -> FitConfig:
    return FitConfig(grid_size=10, sigma_bounds=(1.0, 10.0))


@pytest.fixture(scope="session")
def fitter(grid: Grid, apertures: FloatArray, fit_config: FitConfig) -> PRFFitter:
    return PRFFitter(grid, apertures, fit_config)


def synthesise(
    grid: Grid,
    apertures: FloatArray,
    truth: tuple[float, float, float],
    beta: float = 3.0,
    baseline: float = 0.5,
    noise: float = 0.0,
    seed: int = 0,
) -> FloatArray:
    x0, y0, sigma = truth
    weights = GaussianReceptiveField(x0, y0, sigma).weights(grid)
    clean = beta * predict(weights, apertures) + baseline
    if noise <= 0.0:
        return clean
    rng = np.random.default_rng(seed)
    return clean + rng.normal(0.0, noise * float(np.std(clean)), size=clean.shape)


def autocorrelated_noise(n_frames: int, width: int = 3, seed: int = 21) -> FloatArray:
    rng = np.random.default_rng(seed)
    kernel = np.ones(width) / width
    return np.convolve(rng.normal(size=n_frames), kernel, mode="same")
