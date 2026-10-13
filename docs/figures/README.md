# Figures

The pipeline starts with matched frames, then separates the field estimate
from held-out prediction tests. Read each path from top to bottom.
The editable [SVG](pipeline.svg), [PNG preview](pipeline.png), and responsive
HTML in the documentation show the same workflow. Keep them in sync.

The other 11 figures are scientific plots generated with Matplotlib from the
committed configuration, recorded tables, and stated synthetic demonstrations.
Run from the repository root:

```sh
python scripts/generate_validation_report.py --check
python scripts/generate_figures.py
```

These commands leave the pipeline diagram unchanged.

| Figure | What it shows | Basis |
|---|---|---|
| [Pipeline](pipeline.png) | Matched frames → separate paths for the all-frame estimate and grouped prediction tests | Implemented synthetic workflow; editable SVG with a PNG preview |
| [Apertures](stimuli.png) | Bar, wedge, and ring masks with group labels | Configured stimulus designs |
| [Gaussian overlap](model.png) | A field's overlap with the aperture produces a response | Forward-model illustration |
| [Response curves](fitted-response.png) | Observations, full-data fit, held-out predictions, and residuals | Seeded demonstration |
| [Autocorrelation](noise-autocorrelation.png) | White and moving-average noise across lags | Validation noise seeds |
| [Recovery](recovery.png) | Position and sigma errors at three noise levels | Five known fields, shared seed 7 |
| [Uncertainty](uncertainty.png) | Standard errors, interval widths, and x0 coverage | Recorded coverage trials |
| [Split comparison](leakage.png) | Grouped versus random-frame held-out scores | Noise-only responses, seeds 21–40 |
| [Fold overlap](fold-leakage.png) | Cross-group frame similarity before and after grouping/pruning | Supplemental diagnostic |
| [Sigma bounds](sigma-guard.png) | Sampled Gaussian mass across widths and grid positions | Supplemental diagnostic |
| [Model mismatch](misspecification.png) | Single-Gaussian fits to single- and two-lobe responses | Supplemental diagnostic |
| [Runtime](runtime.png) | Fit and cross-validation timings by unit count | Recorded hardware timings |

See [Validation](../validation.md) for experiment conditions, uncertainty
assumptions, and provenance, or [Pipeline](../pipeline.md) for the fitting steps.
Autocorrelation curves describe the noise inputs, not model fit quality.

[Videos](../media/) show aperture movement and held-out prediction, with animated
previews, playback controls, and the full demonstration conditions.
