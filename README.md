# CortexProbe

CortexProbe fits 2D Gaussian population receptive fields to responses from moving
apertures. It estimates each field's center and width, fits response amplitude
and baseline, and provides grouped cross-validation and local parameter
uncertainty. It generates bar, wedge, and ring aperture sequences. It uses
Python 3.9+, NumPy, and SciPy. Matplotlib is optional for figures.

The tests and recorded results use synthetic responses with known fields.
No neural network or human data has been tested. The package has no network
loader or activation runner.

## Install

```sh
git clone https://github.com/urmeo/cortexprobe.git
cd cortexprobe
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Fit a synthetic response

Run this Python code from the installed environment:

```python
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
```

Expected output:

```text
center=(12.0, 8.0), sigma=5.0, accepted=True
held-out R²=1.000
```

Coordinates and sigma are in pixels, with the origin at the grid center, x
rightward, and y upward. Predictions sum Gaussian weights over each aperture;
there is no haemodynamic convolution.

`fit_unit` takes one response per frame. `fit_all` takes an array shaped
`(frames, units)`. Apertures have shape `(frames, height, width)` and must match
the grid. Use the retained `sequence.n_frames` when preparing responses:
aperture generation can prune similar frames across groups. `CrossValidator`
requires one finite numeric group label per frame and at least two groups.
It fits separate models on training folds and keeps their geometry, amplitude,
and baseline fixed for held-out scoring.

## Validation and limits

The recorded experiments measure recovery of known fields, noise rejection,
interval coverage, grouped versus random-split scores, and runtime.
[Validation details](docs/validation.md) include the results, seeds, conditions,
and instructions to reproduce them.

- Network depth, lesions, and variation across network seeds remain untested.
  `ModelConfig` stores reserved metadata; it does not load a model.
- A fit's `accepted` flag uses convergence, in-sample R², positive amplitude,
  and sigma bounds. It does not require a held-out score.
- Sigma must be at least one pixel. The upper bound must retain at least 99%
  of a centered Gaussian's sampled mass in the circular field, roughly requiring
  `sigma <= resolution / 6.1`. Near-edge fields can still lose mass.
- Confidence intervals use a local approximation. Singular geometry produces
  undefined errors. A single Gaussian can miss multiple lobes; the second-field
  diagnostic has no universal cutoff.

For tests, formatting, type checks, and report updates, see
[Contributing](CONTRIBUTING.md).

[MIT license](LICENSE) · [Citation](CITATION.cff)
