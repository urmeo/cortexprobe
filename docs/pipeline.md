# Pipeline

CortexProbe fits a Gaussian receptive field and checks its predictions on held-out
groups. The examples and recorded experiments use synthetic responses. The
full-data estimate and validation scores come from separate fits to shared inputs.

[![Inputs branch into full-data estimation and grouped training-fold refits before their outputs are compared](figures/pipeline.png)](figures/pipeline.png)

## Prepare the inputs

Build bar, wedge, or ring aperture masks on a circular pixel grid. Assign a group
to each frame. Return bar sweeps share an axis group, and the generator prunes
frames whose cross-group cosine similarity exceeds the configured threshold.
This checks aperture overlap; other forms of dependence may remain.

For a synthetic response, sum a known Gaussian's weights inside each aperture,
apply amplitude and baseline, then optionally add seeded noise. The recorded
experiments also use noise-only responses as a negative control.

| Input | Meaning |
|---|---|
| Apertures `(frames, height, width)` | The retained masks, aligned with the response |
| Response `(frames,)` | One value per retained frame |
| Groups `(frames,)` | One finite numeric label per frame; at least two groups for validation |
| Fit configuration | Candidate grid, parameter bounds, optimizer limit, and acceptance threshold |

The [recorded bar setup](../configs/validation.json) uses a 64 × 64 pixel grid,
80 frames, four groups, 300 candidate fields, and sigma bounds of 1–10 pixels.
These counts describe that configuration; other aperture designs can retain
different numbers of frames after pruning.

## Fit a field

The same fitting procedure runs on all frames for the field estimate, and on each
training subset for held-out validation:

1. Cache the response predicted by each candidate Gaussian for those apertures.
2. Search candidate centers and widths. At each geometry, solve amplitude and
   baseline by linear least squares and compare in-sample R².
3. Refine the best geometry with SciPy's bounded least-squares solver. Only
   `x0`, `y0`, and `sigma` enter this nonlinear search; amplitude and baseline
   are solved again at every trial geometry.
4. Return the fitted parameters, in-sample R², convergence and boundary flags,
   local standard errors, and the residual second-field diagnostic.

Candidate predictions and fold fitters are reused across response columns.
`fit_all` and `validate_all` apply these steps to arrays shaped `(frames, units)`.

## Keep estimation and validation separate

| Path | Frames used to fit | Output |
|---|---|---|
| Full-data estimate | All retained frames | Field parameters, in-sample R², acceptance, and diagnostics |
| Held-out validation | All groups except the current test group | R² on that held-out group from a separate fitted model |

For each validation fold, hold out one complete group, fit the remaining groups,
then predict the held-out frames. Geometry, amplitude, and baseline stay fixed
during this prediction. The full-data estimate supplies no parameters to the
fold models.

The reported `cv_r2` is the mean of finite fold R² scores. It is not a single R²
computed over pooled predictions. Folds with at most five training frames are
skipped; invalid fitted geometry yields an undefined score. Inspect `fold_r2`
to see which scores contributed. A finite fold fit is scored even if its
optimizer did not converge.

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
coverage, grouped versus random splits, and runtime. Compare fitted and held-out
response curves, residuals, fold scores, and the supplemental diagnostics.
The [validation report](validation.md) states the seeds, noise levels, trial
counts, approximation limits, and measurement provenance.

```sh
python scripts/generate_validation_report.py --check
python scripts/generate_figures.py
```

The first command recomputes deterministic measurements and checks the recorded
tables without writing files or remeasuring historical timings. The second
draws the figures. Recorded tables, seeded response curves, and supplemental
diagnostics have different roles; their descriptions are in the
[figure index](figures/).

The implementation lives in
[stimuli.py](https://github.com/urmeo/cortexprobe/blob/main/src/cortexprobe/stimuli.py),
[model.py](https://github.com/urmeo/cortexprobe/blob/main/src/cortexprobe/prf/model.py),
[fit.py](https://github.com/urmeo/cortexprobe/blob/main/src/cortexprobe/prf/fit.py), and
[validation.py](https://github.com/urmeo/cortexprobe/blob/main/src/cortexprobe/prf/validation.py).
See [Usage](usage.md) for runnable code. No network runner, human-data evaluation,
or haemodynamic convolution is implemented in this pipeline.
