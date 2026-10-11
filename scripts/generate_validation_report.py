"""Measure the synthetic validation tables in docs/validation.md.

Load configs/validation.json, write results/validation.json and update four marked tables.
--check compares deterministic measurements within tolerance and verifies those tables,
without writing files. It does not check supplemental figure pixels or remeasure timings.

Reproduced values retain their original numbers, date and environment. --runtime explicitly
refreshes hardware timings, recording their provenance separately when needed.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from cortexprobe.config import RunConfig
from cortexprobe.geometry import Grid
from cortexprobe.prf.fit import PRFFitter
from cortexprobe.prf.model import GaussianReceptiveField, predict
from cortexprobe.prf.validation import CrossValidator
from cortexprobe.stimuli import build_apertures

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

CONFIG = RunConfig.load(ROOT / "configs" / "validation.json")

GROUND_TRUTH = [
    (0.0, 0.0, 5.0),
    (12.0, 8.0, 4.0),
    (-10.0, 14.0, 7.0),
    (16.0, -12.0, 6.0),
    (-15.0, -9.0, 9.0),
]
RECOVERY_NOISE = [0.0, 0.2, 0.5]
RECOVERY_SEED = 7
N_NOISE_UNITS = 40
NOISE_SEED = 11

UNCERTAINTY_TRUTH = (12.0, 8.0, 5.0)
UNCERTAINTY_NOISE = [0.0, 0.1, 0.3, 0.6]
UNCERTAINTY_SEED = 5
COVERAGE_TRUTHS = [(12.0, 8.0, 5.0), (28.0, 0.0, 4.0)]
COVERAGE_NOISE = [0.2, 0.5]
COVERAGE_REALISATIONS = 200
COVERAGE_SEED_BASE = 1000

CV_SEEDS = list(range(21, 41))
CARRIER_WIDTH = 3
SPLIT_SEED = 0

RUNTIME_UNITS = [1, 10, 100]
RUNTIME_SEED = 0


def synthesise(grid, apertures, truth, beta=3.0, baseline=0.5, noise=0.0, seed=0):
    """Activation timecourse from a known pRF, with noise scaled to the clean signal."""
    clean = beta * predict(GaussianReceptiveField(*truth).weights(grid), apertures) + baseline
    if noise <= 0.0:
        return clean
    rng = np.random.default_rng(seed)
    return clean + rng.normal(0.0, noise * float(np.std(clean)), size=clean.shape)


def autocorrelated_noise(n_frames, width=CARRIER_WIDTH, seed=21):
    rng = np.random.default_rng(seed)
    return np.convolve(rng.normal(size=n_frames), np.ones(width) / width, mode="same")


def measure_recovery(grid, apertures, fitter) -> dict[str, Any]:
    rows = []
    for noise in RECOVERY_NOISE:
        position_errors, sigma_errors, scores = [], [], []
        for truth in GROUND_TRUTH:
            fit = fitter.fit_unit(
                synthesise(grid, apertures, truth, noise=noise, seed=RECOVERY_SEED)
            )
            position_errors.append(float(np.hypot(fit.x0 - truth[0], fit.y0 - truth[1])))
            sigma_errors.append(abs(fit.sigma - truth[2]) / truth[2] * 100.0)
            scores.append(fit.r2)
        rows.append(
            {
                "noise": noise,
                "n": len(GROUND_TRUTH),
                "position_max": max(position_errors),
                "position_mean": float(np.mean(position_errors)),
                "position_sd": float(np.std(position_errors, ddof=1)),
                "sigma_max": max(sigma_errors),
                "sigma_mean": float(np.mean(sigma_errors)),
                "sigma_sd": float(np.std(sigma_errors, ddof=1)),
                "r2_min": min(scores),
            }
        )

    rng = np.random.default_rng(NOISE_SEED)
    fits = fitter.fit_all(rng.normal(size=(fitter.n_frames, N_NOISE_UNITS)))
    return {
        "rows": rows,
        "pure_noise": {
            "n_units": N_NOISE_UNITS,
            "rejected": sum(not fit.accepted for fit in fits),
            "best_r2": max(fit.r2 for fit in fits),
            "threshold": CONFIG.fit.r2_threshold,
        },
    }


def measure_uncertainty(grid, apertures, fitter) -> dict[str, Any]:
    rows = []
    for noise in UNCERTAINTY_NOISE:
        fit = fitter.fit_unit(
            synthesise(grid, apertures, UNCERTAINTY_TRUTH, noise=noise, seed=UNCERTAINTY_SEED)
        )
        low, high = fit.confidence_interval("x0")
        rows.append(
            {
                "noise": noise,
                "x0": fit.x0,
                "se_x0": fit.se_x0,
                "ci_width": high - low,
                "dof": fit.dof,
            }
        )

    coverage = []
    for truth in COVERAGE_TRUTHS:
        for noise in COVERAGE_NOISE:
            covered = 0
            for seed in range(COVERAGE_REALISATIONS):
                response = synthesise(
                    grid, apertures, truth, noise=noise, seed=COVERAGE_SEED_BASE + seed
                )
                low, high = fitter.fit_unit(response).confidence_interval("x0")
                covered += bool(low <= truth[0] <= high)
            rate = covered / COVERAGE_REALISATIONS
            half_width = 1.96 * float(np.sqrt(rate * (1.0 - rate) / COVERAGE_REALISATIONS))
            coverage.append(
                {
                    "position": "interior" if truth == COVERAGE_TRUTHS[0] else "near edge",
                    "noise": noise,
                    "n": COVERAGE_REALISATIONS,
                    "coverage": rate,
                    "ci_low": rate - half_width,
                    "ci_high": rate + half_width,
                }
            )
    return {"rows": rows, "coverage": coverage, "nominal": 0.95}


def measure_cross_validation(grid, apertures, sequence) -> dict[str, Any]:
    grouped = CrossValidator(grid, apertures, sequence.group, CONFIG.fit)
    shuffled = np.random.default_rng(SPLIT_SEED).permutation(len(apertures)) % 4
    random_split = CrossValidator(grid, apertures, shuffled, CONFIG.fit)

    rows, lags = [], []
    for label in ("white", "autocorrelated"):
        grouped_scores, random_scores = [], []
        for seed in CV_SEEDS:
            if label == "white":
                response = np.random.default_rng(seed).normal(size=len(apertures))
            else:
                response = autocorrelated_noise(len(apertures), CARRIER_WIDTH, seed)
                lags.append(float(np.corrcoef(response[:-1], response[1:])[0, 1]))
            grouped_scores.append(grouped.validate_unit(response).cv_r2)
            random_scores.append(random_split.validate_unit(response).cv_r2)

        differences = np.asarray(random_scores) - np.asarray(grouped_scores)
        rows.append(
            {
                "response": label,
                "n": len(CV_SEEDS),
                "grouped_mean": float(np.mean(grouped_scores)),
                "grouped_sd": float(np.std(grouped_scores, ddof=1)),
                "leaky_mean": float(np.mean(random_scores)),
                "leaky_sd": float(np.std(random_scores, ddof=1)),
                "leak_mean": float(np.mean(differences)),
                "leak_sd": float(np.std(differences, ddof=1)),
                "leak_positive": int((differences > 0.0).sum()),
            }
        )
    return {
        "rows": rows,
        "kernel_width": CARRIER_WIDTH,
        "lag1_mean": float(np.mean(lags)),
        "lag1_sd": float(np.std(lags, ddof=1)),
        "lag1_theoretical": 1.0 - 1.0 / CARRIER_WIDTH,
        "seeds": CV_SEEDS,
    }


def measure_runtime(grid, apertures, sequence, fitter) -> dict[str, Any]:
    validator = CrossValidator(grid, apertures, sequence.group, CONFIG.fit)
    rng = np.random.default_rng(RUNTIME_SEED)
    radius = grid.radius * 0.7

    def generate_units(n_units):
        columns = []
        for _ in range(n_units):
            x0, y0 = rng.uniform(-radius, radius, size=2)
            clean = predict(
                GaussianReceptiveField(x0, y0, rng.uniform(3.0, 8.0)).weights(grid), apertures
            )
            columns.append(clean + rng.normal(0.0, 0.2 * clean.std(), size=clean.shape))
        return np.column_stack(columns)

    rows = []
    for n_units in RUNTIME_UNITS:
        activations = generate_units(n_units)
        start = time.perf_counter()
        fitter.fit_all(activations)
        fit_seconds = time.perf_counter() - start

        start = time.perf_counter()
        validator.validate_all(activations)
        cv_seconds = time.perf_counter() - start

        rows.append(
            {
                "units": n_units,
                "fit_seconds": fit_seconds,
                "per_unit_ms": fit_seconds / n_units * 1000.0,
                "cv_seconds": cv_seconds,
                "cv_factor": cv_seconds / fit_seconds,
            }
        )
    return {"rows": rows, "n_folds": len(validator._folds)}


def current_environment() -> dict[str, str]:
    return {
        "platform": platform.platform(terse=True),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def footer(environment: dict[str, str], digest: str, generated: str, note: str = "") -> str:
    return (
        f"<sub>{note} Recorded on {environment['platform']} {environment['machine']}, "
        f"Python {environment['python']}, NumPy {environment['numpy']}, "
        f"SciPy {environment['scipy']} | config digest `{digest[:12]}` | "
        f"generated {generated}</sub>"
    )


def render(report: dict[str, Any]) -> dict[str, str]:
    """One Markdown block per validation section, keyed by its marker name."""
    blocks = {}

    recovery = report["recovery"]

    def condition(noise: float) -> str:
        return "Noiseless" if noise == 0 else f"{noise:.0%} noise"

    rows = "\n".join(
        f"| {condition(r['noise'])} "
        f"| {r['position_max']:.3f} | {r['position_mean']:.3f} | "
        f"{r['sigma_max']:.1f} | {r['sigma_mean']:.1f} | {r['r2_min']:.4f} |"
        for r in recovery["rows"]
    )
    noise = recovery["pure_noise"]
    blocks["recovery"] = (
        f"| Condition | Position error, largest (px) | mean (px) | "
        f"Sigma error, largest (%) | mean (%) | Minimum R² |\n"
        f"|---|---|---|---|---|---|\n{rows}\n"
        f"| Pure noise | n/a | n/a | n/a | n/a | "
        f"**{noise['rejected']} / {noise['n_units']} rejected** |\n\n"
        + footer(
            report["environment"],
            report["config_digest"],
            report["generated"],
            f"n = {recovery['rows'][0]['n']}, shared noise seed {RECOVERY_SEED}; pure noise "
            f"{noise['n_units'] - noise['rejected']}/{noise['n_units']} accepted "
            f"(best R² {noise['best_r2']:.4f}, threshold {noise['threshold']}).",
        )
    )

    uncertainty = report["uncertainty"]
    rows = "\n".join(
        f"| {r['noise']:.1f} | {r['x0']:.3f} | {r['se_x0']:.4f} | {r['ci_width']:.3f} |"
        for r in uncertainty["rows"]
    )
    coverage = "\n".join(
        f"| {c['position']} | {c['noise']:.1f} | {c['n']} | {c['coverage']:.3f} | "
        f"[{c['ci_low']:.3f}, {c['ci_high']:.3f}] |"
        for c in uncertainty["coverage"]
    )
    blocks["uncertainty"] = (
        f"| Noise | Fitted x0 | SE | 95 % CI width |\n|---|---|---|---|\n{rows}\n\n"
        f"| Position | Noise | n | Coverage | 95 % Wald CI |\n|---|---|---|---|---|\n"
        f"{coverage}\n\n"
        + footer(
            report["environment"],
            report["config_digest"],
            report["generated"],
            f"Nominal {uncertainty['nominal']:.0%}; paired seeds across cells; "
            "noiseless SE is a numerical floor.",
        )
    )

    cv = report["cross_validation"]
    rows = "\n".join(
        f"| {r['response'].capitalize()} | {r['n']} | "
        f"{r['grouped_mean']:+.3f} ± {r['grouped_sd']:.3f} | "
        f"{r['leaky_mean']:+.3f} ± {r['leaky_sd']:.3f} | "
        f"{r['leak_mean']:+.3f} ± {r['leak_sd']:.3f} | {r['leak_positive']} / {r['n']} |"
        for r in cv["rows"]
    )
    blocks["cross_validation"] = (
        f"| Response | n seeds | Grouped CV | Random-split CV | Leak | Leak positive |\n"
        f"|---|---|---|---|---|---|\n{rows}\n\n"
        + footer(
            report["environment"],
            report["config_digest"],
            report["generated"],
            f"Mean ± SD, seeds {cv['seeds'][0]}-{cv['seeds'][-1]}; width-"
            f"{cv['kernel_width']} boxcar lag-1 {cv['lag1_mean']:.3f} ± "
            f"{cv['lag1_sd']:.3f} (theory {cv['lag1_theoretical']:.3f}).",
        )
    )

    runtime = report["runtime"]
    rows = "\n".join(
        f"| {r['units']} | {r['fit_seconds']:.3f} | {r['per_unit_ms']:.1f} | "
        f"{r['cv_seconds']:.3f} | {r['cv_factor']:.1f}× |"
        for r in runtime["rows"]
    )
    blocks["runtime"] = (
        f"| Units | Fit (s) | Per unit (ms) | CV (s) | CV factor |\n|---|---|---|---|---|\n"
        f"{rows}\n\n"
        + footer(
            report.get("runtime_environment", report["environment"]),
            report.get("runtime_config_digest", report["config_digest"]),
            report.get("runtime_generated", report["generated"]),
            f"Hardware timings; {runtime['n_folds']} folds plus the full fit; "
            "not remeasured by --check.",
        )
    )
    return blocks


def splice(text: str, blocks: dict[str, str]) -> str:
    """Replace each marked region with its generated block."""
    for name, block in blocks.items():
        pattern = re.compile(
            rf"(<!-- BEGIN GENERATED: {name} -->).*?(<!-- END GENERATED: {name} -->)",
            re.DOTALL,
        )
        if not pattern.search(text):
            raise SystemExit(f"docs/validation.md has no generated region named {name!r}")
        text = pattern.sub(
            lambda match, body=block: f"{match.group(1)}\n{body}\n{match.group(2)}", text
        )
    return text


def rounded(value: Any, places: int = 6) -> Any:
    """Record six decimal places; the validation tables display fewer places."""
    if isinstance(value, float):
        return round(value, places)
    if isinstance(value, dict):
        return {key: rounded(item, places) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(item, places) for item in value]
    return value


def drifted(recorded: Any, fresh: Any, path: str = "") -> list[str]:
    """Find changes exceeding relative 1e-5 or absolute 1e-6 float tolerance."""
    if isinstance(recorded, dict) and isinstance(fresh, dict):
        problems = []
        for key in sorted(recorded.keys() | fresh.keys()):
            if key not in recorded or key not in fresh:
                problems.append(f"{path}{key}: present in only one report")
            else:
                problems.extend(drifted(recorded[key], fresh[key], f"{path}{key}."))
        return problems
    if isinstance(recorded, list) and isinstance(fresh, list):
        if len(recorded) != len(fresh):
            return [f"{path.rstrip('.')}: length {len(recorded)} became {len(fresh)}"]
        return [
            problem
            for index, (recorded_item, fresh_item) in enumerate(zip(recorded, fresh))
            for problem in drifted(recorded_item, fresh_item, f"{path}{index}.")
        ]
    if isinstance(recorded, float) or isinstance(fresh, float):
        try:
            if math.isclose(recorded, fresh, rel_tol=1e-5, abs_tol=1e-6):
                return []
        except (OverflowError, TypeError):
            pass
    elif recorded == fresh:
        return []
    return [f"{path.rstrip('.')}: {recorded!r} became {fresh!r}"]


def measurements(report: dict[str, Any]) -> dict[str, Any]:
    """Deterministic measurements, excluding date and hardware provenance."""
    ignored = {
        "runtime",
        "environment",
        "generated",
        "runtime_environment",
        "runtime_generated",
        "runtime_config_digest",
    }
    return {key: value for key, value in report.items() if key not in ignored}


def run_check(report: dict[str, Any], previous: dict[str, Any], document: str) -> int:
    """Verify the committed report still reproduces. Writes nothing."""
    if not previous:
        print("no committed report to check against; run without --check first")
        return 1

    problems = drifted(measurements(previous), measurements(report))
    if problems:
        print("measurements have drifted from the committed report:")
        for problem in problems:
            print(f"  {problem}")
        print("\nrerun without --check to record the new numbers")
        return 1

    blocks = render(previous)
    if splice(document, blocks) != document:
        print("docs/validation.md tables do not match the committed report; rerun without --check")
        return 1
    print(
        "recorded deterministic measurements reproduce within tolerance; validation tables match "
        "(timings and figure pixels not rechecked); "
        f"recorded on {previous['environment']['platform']}, checked on "
        f"{current_environment()['platform']} Python {platform.python_version()}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate the validation report.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--runtime",
        action="store_true",
        help="re-measure the runtime table instead of carrying forward the recorded one",
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help="verify the committed report still reproduces, writing nothing; exits non-zero "
        "if a measurement has drifted or the committed tables do not match it",
    )
    arguments = parser.parse_args(argv)

    document_path = ROOT / "docs" / "validation.md"
    document = document_path.read_text()

    grid = Grid(CONFIG.stimulus.resolution)
    sequence = build_apertures(CONFIG.stimulus, "bar")
    apertures = sequence.as_float()
    fitter = PRFFitter(grid, apertures, CONFIG.fit)

    previous_path = RESULTS / "validation.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}

    report: dict[str, Any] = {
        "config_digest": CONFIG.digest(),
        "config": CONFIG.to_dict(),
        "stimulus": {
            "frames": sequence.n_frames,
            "distinct_frames": fitter.n_independent_frames,
            "groups": sorted({int(g) for g in sequence.group}),
            "candidates": len(fitter.candidates),
        },
        "recovery": measure_recovery(grid, apertures, fitter),
        "uncertainty": measure_uncertainty(grid, apertures, fitter),
        "cross_validation": measure_cross_validation(grid, apertures, sequence),
    }

    report = rounded(report)

    if arguments.check:
        return run_check(report, previous, document)

    running_in = current_environment()
    recorded_measurements = measurements(previous)
    unchanged = bool(previous) and not drifted(recorded_measurements, measurements(report))
    if unchanged:
        report.update(recorded_measurements)
    report["environment"] = previous["environment"] if unchanged else running_in
    report["generated"] = previous["generated"] if unchanged else date.today().isoformat()

    refresh_runtime = arguments.runtime or "runtime" not in previous
    if refresh_runtime:
        report["runtime"] = rounded(measure_runtime(grid, apertures, sequence, fitter))
        if previous:
            report["runtime_environment"] = running_in
            report["runtime_generated"] = date.today().isoformat()
            report["runtime_config_digest"] = report["config_digest"]
    else:
        report["runtime"] = previous["runtime"]
        if not unchanged or "runtime_environment" in previous:
            report["runtime_environment"] = previous.get(
                "runtime_environment", previous["environment"]
            )
            report["runtime_generated"] = previous.get("runtime_generated", previous["generated"])
            report["runtime_config_digest"] = previous.get(
                "runtime_config_digest", previous["config_digest"]
            )

    blocks = render(report)
    spliced = splice(document, blocks)

    RESULTS.mkdir(exist_ok=True)
    previous_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    document_path.write_text(spliced)

    print(f"wrote {previous_path.relative_to(ROOT)}")
    state = "unchanged" if unchanged else "updated"
    timing = "re-measured" if refresh_runtime else "carried forward"
    print(f"Validation tables updated; measurements {state}, runtime {timing}")
    if unchanged and running_in != report["environment"]:
        print(
            "note: these numbers reproduced here, in a different environment than the one on "
            f"record ({running_in['platform']} {running_in['machine']}, Python "
            f"{running_in['python']}, NumPy {running_in['numpy']}, SciPy {running_in['scipy']})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
