# Figures

The figures are generated with Matplotlib from the committed configuration,
recorded tables, and stated synthetic demonstrations. Run from the repository
root:

```sh
python scripts/generate_validation_report.py --check
python scripts/generate_figures.py
```

| Figure | What it shows | Basis |
|---|---|---|
| [Pipeline](pipeline.png) | Apertures → synthetic response → fit → held-out groups → results | Seeded demonstration |
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

See [Validation](../validation.md) for measured conditions, uncertainty
assumptions, and provenance. The pipeline and response curves illustrate a
single synthetic run; they do not add a new network or human-data evaluation.
The autocorrelation curves describe the noise inputs, not model fit quality.
