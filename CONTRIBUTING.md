# Contributing

Use Python 3.9 or later. From the repository root, create a virtual environment
and install the development and plotting dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,viz]'
```

Run the same checks as CI before submitting a change:

```sh
python -m ruff format --check .
python -m ruff check .
python -m mypy
python -m pytest --cov
python scripts/generate_validation_report.py --check
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
reproduce, the report keeps its original values and provenance. The report check does not compare
figure pixels or remeasure timings. See [Validation](docs/validation.md) for
experiment conditions and comparison tolerances.

Keep changes focused. Describe the affected behavior in commit messages and
pull requests, and include the checks you ran. Preserve documented limits and
report unmeasured behavior as unmeasured.
