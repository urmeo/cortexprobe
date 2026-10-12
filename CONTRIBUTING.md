# Contributing

Use Python 3.9 or later. From the repository root, create a virtual environment
and install the development and plotting dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,viz,docs]'
```

Run the same checks as CI before submitting a change:

```sh
python -m ruff format --check .
python -m ruff check .
python -m mypy
python -m pytest --cov
python scripts/generate_validation_report.py --check
python scripts/build_docs.py
```

Coverage must stay at or above 85%. The full test suite includes slow recovery
and interval-coverage tests. Use `python -m pytest -m 'not slow'` for a shorter
local run, then run the full suite before submitting.

When an analysis change affects results, explain the change and its evidence.
Update the numeric record, validation tables, and figures together:

```sh
python scripts/generate_validation_report.py
python scripts/generate_figures.py
```

Use `python scripts/generate_validation_report.py --runtime` only when you
intend to refresh hardware timings. When all deterministic measurements still
reproduce, the report keeps its original values and provenance. The report
check does not compare figure pixels or remeasure timings. See
[Validation](docs/validation.md) for experiment conditions and comparison
tolerances.

Keep changes focused. Use short commit messages that name the update, such as
`updated readme`, `updated results`, or `updated visuals`. Describe the behavior
and checks in the pull request. Preserve documented limits and report
unmeasured behavior as unmeasured.

The project contributor is `urmeo`. Keep commits attributed to that account.

The [video guide](docs/media/#reproduce) explains how to regenerate the synthetic
demonstrations. FFmpeg is required for that command, alongside the plotting extra.
Keep media captions tied to the actual inputs, frame counts, and fitting steps.

## Documentation site

The Pages site renders the README, pipeline guide, usage guide, validation tables,
figure gallery, and videos from their Markdown sources. Build and preview it locally:

```sh
python -m pip install -e '.[docs]'
python scripts/build_docs.py
python -m http.server 8000 --directory _site
```

Open `http://localhost:8000`. The build checks local page, image, video, data, and
anchor links. It copies committed media and data without rerunning the experiments.
Generated files stay in the ignored `_site/` directory.

The overview and pipeline pages use `docs/templates/pipeline.html` and
`docs/assets/pipeline.css`. Keep them aligned with the Figma source and the
exported README diagram. The component's SVG assets live in `docs/assets/pipeline/`.

Fitting and plotting do not require the documentation dependency.
