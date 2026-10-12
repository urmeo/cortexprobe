# Fit a synthetic response

Install from the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python examples/fit_synthetic.py
```

Expected output:

```text
center=(12.0, 8.0), sigma=5.0, accepted=True
held-out R²=1.000
```

The example creates 80 bar-aperture frames, generates a noiseless response from
a known Gaussian field, and fits its center, width, amplitude, and baseline.
The held-out score comes from separate models trained on the remaining sweeps.
`CrossValidatedFit` stores the full-data `fit` and held-out `fold_r2` scores
separately. Its `cv_r2` is the mean of finite fold scores.

The same steps in Python:

```python
from cortexprobe import FitConfig, StimulusConfig
from cortexprobe.prf.fit import PRFFitter
from cortexprobe.prf.model import GaussianReceptiveField, predict
from cortexprobe.prf.validation import CrossValidator
from cortexprobe.stimuli import build_apertures

sequence = build_apertures(StimulusConfig(resolution=64, n_steps=20, directions=(0, 45, 90, 135)))
grid = sequence.grid
apertures = sequence.as_float()
config = FitConfig(grid_size=10, sigma_bounds=(1.0, 10.0))

known_field = GaussianReceptiveField(x0=12.0, y0=8.0, sigma=5.0)
response = 3.0 * predict(known_field.weights(grid), apertures) + 0.5
fit = PRFFitter(grid, apertures, config).fit_unit(response)
print(f"center=({fit.x0:.1f}, {fit.y0:.1f}), sigma={fit.sigma:.1f}, accepted={fit.accepted}")

validated = CrossValidator(grid, apertures, sequence.group, config).validate_unit(response)
print(f"held-out R²={validated.cv_r2:.3f}")
```

## Inputs and coordinates

| Input or output | Shape or unit |
|---|---|
| Apertures | `(frames, height, width)`; spatial dimensions match the grid |
| One response | `(frames,)` |
| Multiple unit responses | `(frames, units)` |
| Group labels | One finite numeric label per frame; at least two groups |
| Center and sigma | Pixels; origin at the grid center, x rightward, y upward |

`PRFFitter.fit_unit` takes one response per frame. `fit_all` fits one field per
response column. `CrossValidator.validate_unit` and `validate_all` use the same
response shapes. The returned fit uses `x0`, `y0`, and `sigma` for geometry,
`beta` for amplitude, and `baseline` for the offset.

Aperture generation prunes similar frames across groups. Use the retained
`sequence.n_frames` when preparing responses. Externally supplied apertures do
not receive this pruning automatically. Fitting needs more than five frames;
cross-validation skips folds with at most five training frames.

At each trial center and sigma, the fitter solves amplitude and baseline by
linear least squares. Predictions sum Gaussian weights multiplied by each
aperture. There is no haemodynamic convolution or conversion to visual degrees.

The example explicitly sets sigma bounds of 1–10 pixels for a 64-pixel grid.
The default sigma ceiling of 20 pixels fits the default 128-pixel stimulus
resolution. Sigma must be at least one pixel. The upper bound must retain at
least 99% of a centered Gaussian's sampled mass in the circular field, roughly
requiring `sigma <= resolution / 6.1`. Edge fields can still lose mass.

## Draw the figures

Matplotlib is optional for fitting and required for the 11 scientific plots:

```sh
python -m pip install -e '.[viz]'
python scripts/generate_validation_report.py --check
python scripts/generate_figures.py
```

The generator reads `configs/validation.json` and the committed numeric record
in `results/validation.json`. It writes the scientific plots to `docs/figures/`.
The pipeline diagram is a committed Figma export; see the [figure guide](figures/)
for its PNG, SVG, and editable source. The response and autocorrelation curves
are reproducible illustrations with stated seeds; they are separate from the
recorded tables.

For intentional measurement changes, see
[the validation report](validation.md#reproduce-the-report). For the full test,
formatting, and type checks, see [Contributing](../CONTRIBUTING.md).

The [video guide](media/) contains animated demonstrations and instructions for
regenerating them with FFmpeg.

## Limits

The tests and recorded results use synthetic responses. No neural network or
human data has been tested. `ModelConfig` stores reserved metadata; it does not
load a network or extract activations.

A fit's `accepted` flag uses convergence, in-sample R², positive amplitude, and
sigma bounds. It does not require a held-out score. Confidence intervals use a
local approximation; singular geometry produces undefined errors. A single
Gaussian can miss multiple lobes, and the second-field diagnostic has no
universal cutoff. See [Validation](validation.md#method-and-constraints) for
the complete evaluation conditions and assumptions.
