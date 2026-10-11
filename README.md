# CortexProbe

Fit 2D Gaussian population receptive fields to synthetic aperture responses
with Python, NumPy, and SciPy. Matplotlib draws the figures.
**No neural network or human data has been tested.**

![Pipeline from moving apertures to synthetic responses, fitting, validation, and results](docs/figures/pipeline.png)

## From aperture to response

Bar, wedge, and ring apertures sample the field. A Gaussian predicts each
response by summing its weights over the aperture. The fitter estimates the
center, width, amplitude, and baseline.

![Bar, wedge, and ring aperture sequences](docs/figures/stimuli.png)

![Observed response, full-data fit, and separately trained held-out predictions](docs/figures/fitted-response.png)

## Recovery and uncertainty

The recovery experiment uses five known fields and one shared noise seed per
condition. Noise is measured relative to the clean response's standard deviation.
The largest observed error is specific to these five fits.

![Position and size recovery across noise levels](docs/figures/recovery.png)

![Local parameter uncertainty and measured x0 interval coverage](docs/figures/uncertainty.png)

## Autocorrelation and validation

Correlated noise can inflate random-frame validation scores. Compare the noise
curves with held-out scores from grouped and random splits. Each held-out
prediction uses a model fitted on the training groups.

![Noise timecourses and autocorrelation curves](docs/figures/noise-autocorrelation.png)

![Grouped and random-frame validation on noise-only responses](docs/figures/leakage.png)

[Validation tables and conditions](docs/validation.md) include the numeric
results, seeds, uncertainty assumptions, historical runtimes, and limits.
[All figures](docs/figures/) include model, mass-limit, and model-mismatch plots.

## Run

```sh
git clone https://github.com/urmeo/cortexprobe.git
cd cortexprobe
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[viz]'
python examples/fit_synthetic.py
python scripts/generate_figures.py
```

The example recovers `(x0, y0, sigma) = (12.0, 8.0, 5.0)` pixels from a noiseless
synthetic response, with held-out R² of `1.000`.
[Usage](docs/usage.md) explains inputs, coordinates, bounds, and plot generation.

## Files

| Folder | Contents |
|---|---|
| `src/cortexprobe/` | Apertures, Gaussian fitting, grouped validation |
| `examples/` | Runnable synthetic fitting example |
| `configs/` | Recorded experiment configuration |
| `results/` | Numeric record and measurement provenance |
| `docs/` | Usage and validation conditions |
| `docs/figures/` | Generated pipeline diagram, curves, and charts |
| `scripts/` | Report and figure generation |
| `benchmarks/` | Measured runtime by unit count |
| `tests/` | Fitting, configuration, stimulus, and reporting checks |
| `.github/workflows/` | Verification on Python 3.9 and 3.12 |

[Checks and updates](CONTRIBUTING.md) · [MIT license](LICENSE) · [Citation](CITATION.cff)
