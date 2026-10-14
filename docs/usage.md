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
held-out R²=1.000; valid folds=4/4; validation=complete
```

The example fits center, width, amplitude, and baseline from 80 bar-aperture
frames and a known Gaussian's noiseless response. Separate models trained on
the remaining sweeps produce the held-out scores. `CrossValidatedFit` stores
the full-data `fit` and held-out `fold_r2`. Its `cv_r2` is the mean of finite
fold scores.

`n_valid_folds` counts finite scores. `n_folds` counts retained folds, including
undefined scores; `n_skipped_folds` counts groups with at most five training
frames. `n_total_folds` includes both retained and skipped folds.
`validation_complete` is true only when every requested fold has a finite score.
It describes score coverage, not fit quality or optimizer convergence.

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
validation_status = "complete" if validated.validation_complete else "incomplete"
print(
    f"held-out R²={validated.cv_r2:.3f}; "
    f"valid folds={validated.n_valid_folds}/{validated.n_total_folds}; "
    f"validation={validation_status}"
)
```

## Inputs and coordinates

| Input or output | Shape or unit |
|---|---|
| Apertures | `(frames, height, width)`; spatial dimensions match the grid |
| One response | `(frames,)` |
| Multiple unit responses | `(frames, units)` |
| Group labels | One finite numeric label per frame; at least two groups |
| Center and sigma | Pixels; origin at the grid center, x rightward, y upward |

`PRFFitter.fit_unit` fits one response column; `fit_all` fits each column.
`CrossValidator.validate_unit` and `validate_all` use the same response shapes.
Fit parameters are `x0`, `y0`, and `sigma` for geometry, `beta` for amplitude,
and `baseline` for the offset.

Fitters and validators copy their aperture inputs and expose read-only snapshots.
Create a new fitter or validator to use different masks.

Bar directions must be unique integer degrees in `[0, 360)`. Opposite directions
remain separate sweeps and share a validation group.

Aperture generation prunes similar frames across groups. Prepare responses for
the retained `sequence.n_frames`. Externally supplied apertures are not pruned
automatically. Fitting needs more than five frames; cross-validation skips folds
with at most five training frames.

R² is undefined (`NaN`) for constant targets, including one-frame groups,
and for nonfinite inputs. This applies to both perfect and wrong predictions
of a constant target. Failed fits also report `NaN` for in-sample R².
Undefined fold scores do not contribute to `cv_r2` or `cv_r2_sd`.

At each trial center and sigma, the fitter solves amplitude and baseline by
linear least squares. Predictions sum Gaussian weights multiplied by each
aperture. There is no haemodynamic convolution or conversion to visual degrees.

The example uses sigma bounds of 1–10 pixels for a 64-pixel grid. The default
ceiling is 20 pixels for the default 128-pixel grid. Sigma must be at least one
pixel. Its upper bound must retain at least 99% of a centered Gaussian's sampled
mass in the circular field, roughly requiring `sigma <= resolution / 6.1`.
Edge fields can still lose mass.

## Draw the figures

Matplotlib is optional for fitting and required for the 11 scientific plots:

```sh
python -m pip install -e '.[viz]'
python scripts/generate_validation_report.py --check
python scripts/generate_figures.py
```

The generator reads `configs/validation.json` and `results/validation.json`, then
writes plots to `docs/figures/`. The [figure guide](figures/) includes the
pipeline's editable SVG and PNG preview. Response and autocorrelation curves
are seeded illustrations, separate from the recorded tables.

See [the validation report](validation.md#reproduce-the-report) to change
measurements and the [video guide](media/) to regenerate animations with FFmpeg.

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
