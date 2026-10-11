"""Fit a known Gaussian field and score separate models on held-out sweeps.

Run from an installed checkout with ``python examples/fit_synthetic.py``.
The response is noiseless and synthetic. All field parameters are in pixels.
"""

from cortexprobe import FitConfig, StimulusConfig
from cortexprobe.geometry import Grid
from cortexprobe.prf.fit import PRFFitter
from cortexprobe.prf.model import GaussianReceptiveField, predict
from cortexprobe.prf.validation import CrossValidator
from cortexprobe.stimuli import build_apertures

grid = Grid(64)
sequence = build_apertures(StimulusConfig(resolution=64, n_steps=20, directions=(0, 45, 90, 135)))
apertures = sequence.as_float()
config = FitConfig(grid_size=10, sigma_bounds=(1.0, 10.0))

known_field = GaussianReceptiveField(x0=12.0, y0=8.0, sigma=5.0)
response = 3.0 * predict(known_field.weights(grid), apertures) + 0.5
fit = PRFFitter(grid, apertures, config).fit_unit(response)
print(f"center=({fit.x0:.1f}, {fit.y0:.1f}), sigma={fit.sigma:.1f}, accepted={fit.accepted}")

validated = CrossValidator(grid, apertures, sequence.group, config).validate_unit(response)
print(f"held-out R²={validated.cv_r2:.3f}")
