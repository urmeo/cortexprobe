# Pipeline

CortexProbe fits Gaussian receptive fields and scores held-out groups. The
examples and recorded experiments use synthetic responses.

[![Prepare matched frames, then follow two separate paths: estimate a field from all frames, and test independent training fits using held-out masks for prediction and held-out responses for scoring.](figures/pipeline.png)](figures/pipeline.png)

[Figma source](https://www.figma.com/design/SLnekFxRy403s5HqmehcOz?node-id=9-2) · [Editable SVG](figures/pipeline.svg) · [Figure guide](figures/) · [Videos](media/)

## Prepare the inputs

Build bar, wedge, or ring aperture masks on a circular pixel grid, with one group
label per frame. Return bar sweeps share an axis group. The generator prunes
frames whose cross-group cosine similarity exceeds the configured threshold.
This checks aperture overlap; other dependence may remain.

For synthetic responses, sum a known Gaussian's weights inside each retained
aperture, apply amplitude and baseline, then optionally add seeded noise.
The recorded experiments use noise-only responses as a negative control.

| Input | Meaning |
|---|---|
| Apertures `(frames, height, width)` | The retained masks, aligned with the response |
| Response `(frames,)` | One value per retained frame |
| Groups `(frames,)` | One finite numeric label per frame; at least two groups for validation |
| Fit configuration | Candidate grid, parameter bounds, optimizer limit, and acceptance threshold |

The [recorded bar setup](../configs/validation.json) uses a 64 × 64 pixel grid,
80 frames, four groups, 300 candidate fields, and sigma bounds of 1–10 pixels.
Other aperture designs can retain different numbers of frames after pruning.

## Fit a field

Fit all retained frames for the field estimate and each training subset for
validation:

1. Cache the response predicted by each candidate Gaussian for those apertures.
2. Search candidate centers and widths. At each geometry, solve amplitude and
   baseline by linear least squares and compare in-sample R².
3. Refine the best geometry with SciPy's bounded least-squares solver. Only
   `x0`, `y0`, and `sigma` enter this nonlinear search; amplitude and baseline
   are solved again at every trial geometry.
4. Return the fitted parameters, in-sample R², convergence and boundary flags,
   local geometry standard errors, and the residual second-field diagnostic.

Candidate predictions and fold fitters are reused across response columns.
`fit_all` and `validate_all` apply these steps to arrays shaped `(frames, units)`.

## Keep estimation and validation separate

| Path | Frames used to fit | Output |
|---|---|---|
| Full-data estimate | All retained frames | Field parameters, in-sample R², acceptance, and diagnostics |
| Held-out validation | All groups except the current test group | R² on that held-out group from a separate fitted model |

For each fold, hold out one complete group, fit the remaining groups, then
predict the held-out frames with geometry, amplitude, and baseline fixed.
The full-data estimate supplies no parameters to fold models. Held-out masks
enter prediction; held-out responses enter scoring. Both bypass training.
Repeat for each usable group.

The reported `cv_r2` is the mean of finite fold R² scores; `cv_r2_sd` is their
sample standard deviation. The score is not computed from pooled predictions.
Folds with at most five training frames are skipped. Invalid fitted geometry,
constant held-out targets, and nonfinite inputs yield undefined (`NaN`) scores.
A finite fold fit is scored even if its optimizer did not converge.

`n_valid_folds` counts finite scores; `n_folds` includes all retained folds,
and `n_skipped_folds` counts skipped groups. `n_total_folds` includes both.
`validation_complete` is true only when every requested fold has a finite score.
Check this coverage alongside the mean and inspect `fold_r2` for each retained
fold. Complete validation does not imply a good fit or optimizer convergence.

## Interpret the checks

| Check | Rule and practical limit |
|---|---|
| Acceptance | Optimizer converged, in-sample R² meets the threshold, amplitude is positive, and sigma is away from either bound. Held-out scores do not gate acceptance. |
| Uncertainty | A local Jacobian approximation uses distinct aperture masks minus five as residual degrees of freedom. Confidence intervals use Student-t critical values. Singular geometry gives undefined errors. |
| Sampled mass | Sigma is at least one pixel, and its upper bound must retain at least 99% of a centered Gaussian's sampled mass inside the circular field. Edge fields can still lose mass. |
| Model mismatch | Project cached candidate response curves off the fitted prediction and intercept; report the largest incremental R². This diagnostic has no universal rejection threshold. |

Center bounds, undefined uncertainty, and the second-field diagnostic do not
change acceptance. An accepted fit can still predict poorly on held-out groups
or miss part of a multi-lobe response. Repeated masks do not increase the
uncertainty calculation's frame count; this is conservative when repeated
measurements have fresh independent noise.

## Inspect and reproduce the results

The recorded experiments check known-field recovery, noise rejection, interval
coverage, grouped versus random splits, and runtime. The
[validation report](validation.md) states the seeds, noise levels, trial counts,
approximation limits, and measurement provenance.

```sh
python scripts/generate_validation_report.py --check
python scripts/generate_figures.py
```

The first command recomputes deterministic measurements and checks the recorded
tables without writing files or remeasuring historical timings. The second
draws the 11 scientific plots and preserves the pipeline SVG and PNG preview.
The [figure index](figures/) identifies recorded results, seeded examples, and
supplemental diagnostics.

Implementation:
[stimuli.py](https://github.com/urmeo/cortexprobe/blob/main/src/cortexprobe/stimuli.py),
[model.py](https://github.com/urmeo/cortexprobe/blob/main/src/cortexprobe/prf/model.py),
[fit.py](https://github.com/urmeo/cortexprobe/blob/main/src/cortexprobe/prf/fit.py), and
[validation.py](https://github.com/urmeo/cortexprobe/blob/main/src/cortexprobe/prf/validation.py).
See [Usage](usage.md) for runnable code and operating limits.
