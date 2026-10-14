# Synthetic validation

These experiments test the fitting and validation code on generated responses.
They do not test a neural network or human data. The tables come from
[`results/validation.json`](../results/validation.json), using
[`configs/validation.json`](../configs/validation.json). Each table retains its
recorded environment, date, and configuration digest.

The shared setup is a 64 × 64 pixel field, 80 bar frames, four sweep groups,
and 300 candidate fields. The coarse search uses `grid_size=10`, followed by
bounded refinement with sigma between 1 and 10 pixels. Responses are Gaussian
aperture overlaps with amplitude 3 and baseline 0.5, except where specified.
Noise levels give the noise standard deviation as a fraction of the clean
response's standard deviation.

## Recovery

The five known fields have `(x0, y0, sigma)` values `(0, 0, 5)`, `(12, 8, 4)`,
`(-10, 14, 7)`, `(16, -12, 6)`, and `(-15, -9, 9)`, all in pixels. Each condition
uses the same noise seed, 7, across these fields. The largest error is the
largest of these five fits in one noise realization; it is not a worst-case
bound. The pure-noise check uses 40 independent response columns drawn with
seed 11. Its rejection count is specific to that draw and threshold. The R²
values in this table are in-sample scores from fits using all frames.

<!-- BEGIN GENERATED: recovery -->
| Condition | Position error, largest (px) | mean (px) | Sigma error, largest (%) | mean (%) | Minimum R² |
|---|---|---|---|---|---|
| Noiseless | 0.000 | 0.000 | 0.0 | 0.0 | 1.0000 |
| 20% noise | 0.287 | 0.197 | 5.2 | 2.3 | 0.9696 |
| 50% noise | 0.772 | 0.511 | 12.8 | 5.6 | 0.8326 |
| Pure noise | n/a | n/a | n/a | n/a | **40 / 40 rejected** |

<sub>n = 5, shared noise seed 7; pure noise 0/40 accepted (best R² 0.1947, threshold 0.2). Recorded on macOS-26.6.1 arm64, Python 3.12.13, NumPy 2.5.2, SciPy 1.18.0 | config digest `4d6a856a499e` | generated 2026-08-19</sub>
<!-- END GENERATED: recovery -->

![Known and recovered fields across noise levels](figures/recovery.png)

## Parameter uncertainty

The first table fits the field `(12, 8, 5)` at four noise levels with seed 5.
The standard error and confidence interval describe `x0`, in pixels. At zero
noise, residuals reach machine precision and the interval collapses. That row
shows a numerical floor, not evidence of exact uncertainty calibration.

Coverage measures how often the nominal 95% interval contains the true `x0`.
Each cell uses 200 trials, with seeds 1000–1199 reused across noise levels and
positions. The fields are `(12, 8, 5)` for the interior and `(28, 0, 4)` near
the edge. Cells share seeds and should not be pooled as independent trials.
The binomial intervals around coverage use the Wald approximation.

<!-- BEGIN GENERATED: uncertainty -->
| Noise | Fitted x0 | SE | 95 % CI width |
|---|---|---|---|
| 0.0 | 12.000 | 0.0000 | 0.000 |
| 0.1 | 11.962 | 0.0961 | 0.383 |
| 0.3 | 11.909 | 0.2806 | 1.118 |
| 0.6 | 11.881 | 0.5354 | 2.133 |

| Position | Noise | n | Coverage | 95 % Wald CI |
|---|---|---|---|---|
| interior | 0.2 | 200 | 0.935 | [0.901, 0.969] |
| interior | 0.5 | 200 | 0.930 | [0.895, 0.965] |
| near edge | 0.2 | 200 | 0.930 | [0.895, 0.965] |
| near edge | 0.5 | 200 | 0.915 | [0.876, 0.954] |

<sub>Nominal 95%; paired seeds across cells; noiseless SE is a numerical floor. Recorded on macOS-26.6.1 arm64, Python 3.12.13, NumPy 2.5.2, SciPy 1.18.0 | config digest `4d6a856a499e` | generated 2026-08-19</sub>
<!-- END GENERATED: uncertainty -->

![Local uncertainty and empirical interval coverage](figures/uncertainty.png)

## Grouped and random splits

The response contains noise only. Each seed is evaluated with both a whole-sweep
split and a random frame split. White noise is sampled independently per frame.
Autocorrelated noise is white noise convolved with a width-3 boxcar using
`mode="same"`. The random split uses seed 0 and four folds.

Scores are held-out R², averaged across finite fold scores. The column labeled
"Leak" is random-split R² minus grouped R²; "Leak positive" counts seeds for
which this difference exceeds zero. Negative R² means the prediction is worse
than the held-out response's mean. The correlated-noise result shows that a
random frame split can inflate this score without a receptive-field signal.

<!-- BEGIN GENERATED: cross_validation -->
| Response | n seeds | Grouped CV | Random-split CV | Leak | Leak positive |
|---|---|---|---|---|---|
| White | 20 | -0.248 ± 0.135 | -0.339 ± 0.192 | -0.091 ± 0.231 | 8 / 20 |
| Autocorrelated | 20 | -0.624 ± 0.385 | -0.164 ± 0.189 | +0.459 ± 0.402 | 17 / 20 |

<sub>Mean ± SD, seeds 21-40; width-3 boxcar lag-1 0.651 ± 0.086 (theory 0.667). Recorded on macOS-26.6.1 arm64, Python 3.12.13, NumPy 2.5.2, SciPy 1.18.0 | config digest `4d6a856a499e` | generated 2026-08-19</sub>
<!-- END GENERATED: cross_validation -->

![Grouped and random-split scores for noise responses](figures/leakage.png)

## Runtime

The recorded timings cover fitting and cross-validation after fitter setup.
They use synthetic fields with 20% noise, amplitude 1, and baseline 0.
Seed 0 samples each center coordinate uniformly within ±70% of the field
radius and sigma uniformly between 3 and 8 pixels. Cross-validation includes
four fold fits and a full-data fit. These are measurements from the recorded
machine, not guarantees for other systems or unit counts. `--check` preserves
them without remeasuring runtime.

<!-- BEGIN GENERATED: runtime -->
| Units | Fit (s) | Per unit (ms) | CV (s) | CV factor |
|---|---|---|---|---|
| 1 | 0.009 | 8.5 | 0.040 | 4.7× |
| 10 | 0.078 | 7.8 | 0.391 | 5.0× |
| 100 | 0.792 | 7.9 | 3.978 | 5.0× |

<sub>Hardware timings; 4 folds plus the full fit; not remeasured by --check. Recorded on macOS-27.0.1 arm64, Python 3.12.13, NumPy 2.5.3, SciPy 1.18.1 | config digest `4d6a856a499e` | generated 2026-10-06</sub>
<!-- END GENERATED: runtime -->

![Recorded fit and cross-validation times](figures/runtime.png)

## Method and constraints

- A field predicts each frame's response by summing Gaussian weights multiplied
  by that frame's aperture. Coordinates and sigma are in pixels. There is no
  haemodynamic convolution or conversion to visual degrees.
- The fitter searches candidate centers and widths, then refines them with
  SciPy's bounded least-squares solver. It solves amplitude and baseline by
  linear least squares at each trial geometry. Responses are centered and
  scaled for optimization. Reported amplitude and baseline retain the input units.
- A fit is accepted when the optimizer converges, R² meets the configured
  threshold, amplitude is positive, and sigma is off its bounds. Center bounds,
  undefined standard errors, and the second-field diagnostic do not change
  acceptance. Acceptance does not require a held-out score.
- Aperture generation prunes frames until every cross-group cosine similarity
  is at most the configured threshold. Return sweeps share an axis group.
  `CrossValidator` does not apply this pruning to externally supplied apertures.
  The check covers aperture overlap, not every possible source of dependence
  in external data.
- Cross-validation holds out whole groups. Geometry, amplitude, and baseline
  fitted on training frames stay fixed during scoring. Folds with at most five
  training frames are skipped. The mean excludes undefined scores; inspect
  `fold_r2` to see which folds contributed.
- Standard errors use a local Jacobian approximation. The implementation uses
  the number of distinct aperture masks minus five as residual degrees of
  freedom; repeated masks do not increase this count. This is a conservative
  policy when repetitions have fresh independent response noise. Singular fits
  have undefined errors. Intervals use a Student-t critical value and assume
  local linearity and approximately Gaussian residuals.
- Gaussian weights use continuous unit-volume normalization. A finite grid can
  truncate their mass, especially near an edge. Sigma must be at least one
  pixel; the fitter also requires at least 99% of the centered Gaussian's
  sampled mass inside the circular field at the upper sigma bound, roughly
  `resolution / 6.1`. This centered check does not guarantee mass near edges.
  The free fitted amplitude absorbs peak-versus-volume scaling.
- A single Gaussian can miss multiple lobes. `second_field_r2` reports the largest
  additional R² from a candidate field after fitting. It has no universal
  acceptance cutoff. Ring stimuli alone do not identify polar angle.

## Reproduce the report

Install the development dependencies, then check the report:

```sh
python -m pip install -e '.[dev]'
python scripts/generate_validation_report.py --check
```

This recomputes deterministic measurements and checks the tables against the
numeric record. Float comparisons allow relative error `1e-5` or absolute
error `1e-6`. The check writes no files. It does not compare figure pixels or
remeasure hardware timings.

To update measurements after an intentional analysis change:

```sh
python scripts/generate_validation_report.py
python scripts/generate_figures.py
```

When deterministic measurements still reproduce, the report keeps the recorded
values and provenance. Use `python scripts/generate_validation_report.py --runtime`
to refresh hardware timings explicitly. Review the numeric record, tables, and
figures together.

The [figure gallery](figures/) also includes seeded examples and supplemental
diagnostics. These do not add recorded measurements or change their provenance.

## References

- [Dumoulin and Wandell (2008)](https://doi.org/10.1016/j.neuroimage.2007.09.034):
  moving-aperture receptive-field fitting, adapted here without haemodynamic
  convolution and using pixel coordinates.
- [Kriegeskorte et al. (2009)](https://doi.org/10.1038/nn.2303):
  independent selection and validation.
- [SciPy least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html):
  bounded nonlinear refinement.
- [SciPy covariance guidance](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html):
  limitations of covariance estimates based on local linear approximations.
