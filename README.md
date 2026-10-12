# CortexProbe

Fit 2D Gaussian population receptive fields to synthetic aperture responses
with Python, NumPy, and SciPy. Matplotlib draws experiment plots and animations.
**No neural network or human data has been tested.**

[Documentation](https://urmeo.github.io/cortexprobe/) · [Usage](docs/usage.md) · [Validation](docs/validation.md)

## Pipeline

Match aperture masks with responses, then estimate the field and test held-out
groups with separate models.

[![Matched synthetic inputs branch into a full-data Gaussian fit and independent training-fold refits. Candidate search and bounded refinement solve amplitude and baseline at each geometry. Recorded synthetic checks appear below the workflow.](docs/figures/pipeline.png)](docs/figures/pipeline.png)

Both paths search candidate centers and widths, then refine the geometry.
Amplitude and baseline are solved at each trial geometry. Validation fits the
training groups only and keeps all five parameters fixed on the held-out group.

[Pipeline details](docs/pipeline.md) · [Figma source](https://www.figma.com/design/SLnekFxRy403s5HqmehcOz?node-id=2-2) · [SVG](docs/figures/pipeline.svg)

Click any figure to view it at full size.

## Watch the workflow

**Apertures and responses.** Moving bar, wedge, and ring masks sample a known
Gaussian field. The clean response follows their overlap. Each sequence uses its
own retained frames; playback shows sequence progress.

[![Animated bar, wedge, and ring apertures with their clean response curves](docs/media/aperture-response.gif)](https://urmeo.github.io/cortexprobe/docs/media/#apertures-and-responses)

[Watch the video](https://urmeo.github.io/cortexprobe/docs/media/#apertures-and-responses) · [MP4 file](docs/media/aperture-response.mp4)

**Held-out prediction.** Four separate training-fold fits predict the excluded
group. This uses the field `(12, 8, 5)`, 20% Gaussian noise, and seed 7. Parameters
stay fixed during held-out prediction.

[![Animated training groups, excluded observations, and predictions from separately fitted fold models](docs/media/grouped-validation.gif)](https://urmeo.github.io/cortexprobe/docs/media/#held-out-prediction)

[Watch the video](https://urmeo.github.io/cortexprobe/docs/media/#held-out-prediction) · [MP4 file](docs/media/grouped-validation.mp4)

These are illustrations of the existing synthetic example. See
[video conditions and reproduction](docs/media/) for frame counts and details.

## From aperture to response

The fitter searches candidate centers and widths, then refines the Gaussian.
It solves amplitude and baseline by linear least squares at each trial geometry.

**Apertures.** Bar sweeps, rotating wedges, and expanding rings sample different
parts of the field. Frames carry group labels for validation.

[![Bar, wedge, and ring aperture sequences with validation groups](docs/figures/stimuli.png)](docs/figures/stimuli.png)

**Forward model.** More overlap between the Gaussian and the aperture produces
a larger predicted response. Coordinates and width are in pixels.

[![Gaussian field, two aperture positions, and the resulting response curve](docs/figures/model.png)](docs/figures/model.png)

**Response curves.** This example uses the field `(12, 8, 5)`, 20% Gaussian noise,
and seed 7. It separates the full-data fit, held-out predictions, and residuals.

[![Observed response, full-data fit, separately trained held-out predictions, and residuals](docs/figures/fitted-response.png)](docs/figures/fitted-response.png)

## Recovery and uncertainty

The recovery experiment uses five known fields and one shared noise seed per
condition. Noise is measured relative to the clean response's standard deviation.
The largest observed error is specific to these five fits in one noise draw.

The recorded setup uses a 64 × 64 pixel field, 80 bar frames, four groups,
300 candidate fields, and sigma bounds of 1–10 pixels.

| Noise | Mean position error (px) | Largest position error (px) | Mean sigma error |
|---|---|---|---|
| None | 0.000 | 0.000 | 0.0% |
| 20% | 0.197 | 0.287 | 2.3% |
| 50% | 0.511 | 0.772 | 5.6% |

These are the five-field results with shared seed 7. A separate pure-noise check
accepted 0 of 40 response columns at seed 11 with an in-sample R² threshold of 0.2.
The [numeric record](results/validation.json) and
[validation report](docs/validation.md#recovery) retain the full results and provenance.

[![Position and sigma recovery across noise levels for five known fields](docs/figures/recovery.png)](docs/figures/recovery.png)

The nominal 95% `x0` intervals covered the true value in 91.5–93.5% of trials
across the four recorded conditions. Each condition uses 200 trials; seeds are
paired across conditions. The intervals use a local approximation, and the
noiseless standard error is a numerical floor.

[![Local parameter uncertainty and measured x0 interval coverage](docs/figures/uncertainty.png)](docs/figures/uncertainty.png)

## Autocorrelation and validation

**Noise curves.** White noise and width-3 moving-average noise are compared over
seeds 21–40. The autocorrelation plot shows sample curves and the stationary
reference for that noise generator.

[![White and moving-average noise timecourses and autocorrelation curves](docs/figures/noise-autocorrelation.png)](docs/figures/noise-autocorrelation.png)

**Split comparison.** On correlated noise alone, random-frame R² minus grouped
R² was `+0.459 ± 0.402` across 20 seeds (mean ± SD). Both mean scores were
negative. A higher score from a random split can occur without a field signal.

[![Grouped and random-frame held-out scores on noise-only responses](docs/figures/leakage.png)](docs/figures/leakage.png)

**Frame overlap.** This supplemental diagnostic compares cross-group cosine
similarity before and after grouping return sweeps together and pruning similar
frames. Aperture overlap is one source of dependence to check.

[![Cross-group frame similarity before and after grouping and pruning](docs/figures/fold-leakage.png)](docs/figures/fold-leakage.png)

## Model limits

**Finite grid.** Sigma must be at least one pixel. The upper bound must retain
at least 99% of a centered Gaussian's sampled mass inside the circular field.
Fields near an edge can still lose mass. The plot illustrates these grid limits.

[![Sampled Gaussian mass and the effects of widths below one pixel](docs/figures/sigma-guard.png)](docs/figures/sigma-guard.png)

**Multiple lobes.** A single Gaussian can fit only part of a two-lobe response
and still pass the acceptance rules. The supplemental second-field diagnostic
measures remaining structure; it has no universal rejection threshold.

[![Single-Gaussian fit to two lobes and residual second-field gain](docs/figures/misspecification.png)](docs/figures/misspecification.png)

## Recorded runtime

These timings cover 1, 10, and 100 units on the recorded machine. Cross-validation
includes four training-fold fits and a full-data fit. The measurements are
hardware-specific; the report check preserves them without remeasuring time.

[![Recorded fitting time per unit and cross-validation overhead](docs/figures/runtime.png)](docs/figures/runtime.png)

[Validation tables and conditions](docs/validation.md) include the numeric
results, seeds, uncertainty assumptions, historical runtimes, and limits.
[Figure details](docs/figures/) distinguish recorded results from seeded examples
and supplemental diagnostics.

## Run

```sh
git clone https://github.com/urmeo/cortexprobe.git
cd cortexprobe
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[viz]'
python examples/fit_synthetic.py
```

Expected output for the noiseless synthetic example:

```text
center=(12.0, 8.0), sigma=5.0, accepted=True
held-out R²=1.000
```

To check the recorded measurements and redraw the figures:

```sh
python scripts/generate_validation_report.py --check
python scripts/generate_figures.py
```

[Usage](docs/usage.md) explains inputs, coordinates, bounds, and plot generation.
The [video guide](docs/media/#reproduce) explains animation generation with FFmpeg.

## Files

| Folder | Contents |
|---|---|
| `src/cortexprobe/` | Apertures, Gaussian fitting, grouped validation |
| `examples/` | Runnable synthetic fitting example |
| `configs/` | Recorded experiment configuration |
| `results/` | Numeric record and measurement provenance |
| `docs/` | Pipeline, usage, and validation conditions |
| `docs/figures/` | Pipeline diagram, curves, and charts |
| `docs/media/` | Animated previews, videos, and demonstration conditions |
| `scripts/` | Report, figure, and video generation |
| `benchmarks/` | Measured runtime by unit count |
| `tests/` | Fitting, configuration, stimulus, and reporting checks |
| `.github/workflows/` | Python 3.9/3.12 verification and Pages publication |

[Checks and updates](CONTRIBUTING.md) · [MIT license](LICENSE) · [Citation](CITATION.cff)
