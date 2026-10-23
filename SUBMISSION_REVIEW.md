# CortexProbe submission review

Reviewed: **24 October 2026**.

**Verdict: ready for a source-code or demonstration submission within its
stated synthetic scope. Overall score: 8.5/10.** No assessment rubric was
provided; this score is an engineering judgment, not an accuracy measurement.

## What the project demonstrates

CortexProbe estimates a two-dimensional Gaussian receptive field from matched
aperture masks and responses. It searches candidate fields, refines geometry,
and solves amplitude and baseline by linear least squares. Grouped validation
refits the training groups and predicts excluded frames with frozen parameters.
Fit acceptance and held-out validation are reported separately.

The implementation uses Python, NumPy, and SciPy. Runtime use supports Python
3.9 or later. Tests, plots, and documentation require Python 3.10 or later.
The supported workflow is a source checkout with `PYTHONPATH=src`.

## Assessment

| Area | Score | Evidence |
|---|---:|---|
| Functionality | 9.0/10 | Fitting, validation, input checks, and the documented example pass. |
| Verification | 9.0/10 | 480 test cases; numerical and validation regressions retained. |
| Code clarity | 8.5/10 | Seven core modules; plotting and documentation tools kept outside the library. |
| Presentation | 8.5/10 | One pipeline, five reproducible plots, explicit conditions, and working media. |
| Reproducibility | 7.5/10 | Current checks reproduce; original experiment and media generators are absent. |

The overall score is the equally weighted mean of these five scores.

## Verification

| Check | Result |
|---|---|
| Python 3.12 suite with patched pytest | 480 passed |
| Core statement coverage | 783/795 lines, 98.49% |
| Core branch coverage | 200/208 branches, 96.15% |
| Combined statement and branch coverage | 98.01%; above the 97% gate |
| Ruff checks and formatting | Passed for all Python files |
| Strict mypy | Passed for all seven core modules |
| Python 3.9 runtime example | Passed with only NumPy and SciPy |
| Independent result audit | All 678 numeric CSV values matched recomputation |
| Documentation | One page, 17 figures, two GIFs, and two videos; local links passed |
| Browser checks | 320, 390, 768, and 1440 pixels, plus 200% text layouts |
| Media integrity | All 23 PNG, GIF, and MP4 assets decoded |

Coverage applies only to `src/cortexprobe`. Tool behavior is checked separately
through builds, numeric reproduction, link checks, and browser inspection.

The noiseless example recovers center `(12.0, 8.0)` and sigma `5.0`, accepts the
fit, and reports held-out R² `1.000` with four valid folds out of four. In the
recorded noisy example, four separately trained folds predict all 80 frames
exactly once. Mean fold R² is `0.9644975687980548`. This is one specified
synthetic noise draw, not an accuracy guarantee.

A confidence-interval boundary defect was corrected: confidence levels very
close to one now use the inverse survival function, avoiding probability
rounding to one. Four regression cases cover ordinary and extreme levels.
Source and plotting-script hashes identify the regenerated measurements.

## Security

No exposed credentials, private keys, or `.env` files were found in the checked
files or reachable history, including both pull-request refs. A custom scan
covered 115 commits and 537 distinct blobs; Gitleaks independently checked
current files and available local and remote history. Environment and key
files are excluded from future commits.

The development requirements now require pytest 9.0.3 or later to address
[CVE-2025-71176](https://github.com/advisories/GHSA-6w46-j5rx-g56g).
The patched environment was checked against OSV and PyPI advisory records.
These checks apply to the exact audited versions and available advisory data;
they do not establish that every allowed dependency version is safe.

## Limits

All evaluations are synthetic. The model is one isotropic Gaussian with linear
amplitude and baseline, without haemodynamic convolution. No human-data or
neural-network evaluation is included. Fit acceptance does not establish model
adequacy: a two-lobe response can pass the single-Gaussian criteria.

Local Student-t intervals are approximate. Historical nominal 95% intervals
covered `x0` in 91.5–93.5% of recorded trials. Those intervals are not calibrated
95% guarantees. Historical figures and tables retain their recorded conditions,
but their original generators are absent. The retained plotting script
reproduces the five current checks.

## Five improvements

1. Repeat the current checks across more noise draws and report distributions
   of error, convergence, and valid-fold coverage.
2. Archive original experiment recipes if historical figures are part of the
   assessed results.
3. Preserve an exact tested dependency snapshot with the submitted release.
4. Automate numeric plot reproduction with tolerances rather than image-pixel
   equality across library versions.
5. Evaluate a documented external dataset before making claims about measured
   neural responses.

These are extensions to the submitted evidence. The documented synthetic
workflow works without them.
