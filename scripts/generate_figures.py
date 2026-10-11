"""Regenerate project and validation figures in docs/figures/.

Coverage and timing panels use results/validation.json. The pipeline describes the workflow;
other panels show synthetic model behavior and diagnostics. A matching configuration digest
is required. Run the report's --check first to verify recorded numbers; it does not compare
figure pixels.
Needs the ``viz`` extra::

    python3 -m pip install -e '.[viz]'
    python3 scripts/generate_figures.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_validation_report import (
    CARRIER_WIDTH,
    CONFIG,
    CV_SEEDS,
    GROUND_TRUTH,
    RECOVERY_NOISE,
    RECOVERY_SEED,
    RESULTS,
    SPLIT_SEED,
    UNCERTAINTY_NOISE,
    UNCERTAINTY_SEED,
    UNCERTAINTY_TRUTH,
    autocorrelated_noise,
    synthesise,
)

from cortexprobe.config import StimulusConfig
from cortexprobe.geometry import Grid
from cortexprobe.prf.fit import MIN_ON_GRID_VOLUME, RESOLUTION_PER_SIGMA, PRFFitter, _r_squared
from cortexprobe.prf.model import GaussianReceptiveField, predict
from cortexprobe.prf.validation import CrossValidator, LeaveOneGroupOut
from cortexprobe.stimuli import build_apertures, frame_similarity

FIGURES = Path(__file__).resolve().parent.parent / "docs" / "figures"
INK = "#1b1f23"
ACCENT = "#2f6f9f"
WARN = "#b4442e"
GOOD = "#3f7d58"
EXAMPLE_NOISE = 0.2

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.edgecolor": "#d0d7de",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def worst_cross_group_similarity(apertures, groups) -> float:
    similarity = frame_similarity(apertures)
    worst_similarity = 0.0
    for group in np.unique(groups):
        test_indices = np.flatnonzero(groups == group)
        train_indices = np.flatnonzero(groups != group)
        if len(train_indices):
            worst_similarity = max(
                worst_similarity, float(similarity[np.ix_(test_indices, train_indices)].max())
            )
    return worst_similarity


def load_recorded_report() -> dict:
    """Reject absent or stale validation records before drawing figures."""
    path = RESULTS / "validation.json"
    try:
        report = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {path}; generate the validation report first") from error
    if not isinstance(report, dict) or report.get("config_digest") != CONFIG.digest():
        raise ValueError("validation config digest is stale; regenerate the report first")
    return report


def pixel_coordinates(grid: Grid, x0: float, y0: float) -> tuple[float, float]:
    """Convert upward-y field coordinates to imshow's column and downward row."""
    origin = (grid.resolution - 1) / 2
    return x0 + origin, origin - y0


def synthetic_fit_example():
    """One reproducible noisy response for the prediction figure."""
    grid = Grid(CONFIG.stimulus.resolution)
    sequence = build_apertures(CONFIG.stimulus, "bar")
    apertures = sequence.as_float()
    clean = synthesise(grid, apertures, UNCERTAINTY_TRUTH)
    response = synthesise(
        grid, apertures, UNCERTAINTY_TRUTH, noise=EXAMPLE_NOISE, seed=RECOVERY_SEED
    )
    fit = PRFFitter(grid, apertures, CONFIG.fit).fit_unit(response)
    return grid, sequence, apertures, clean, response, fit


def figure_pipeline() -> None:
    """Separate full-data estimation from training-only validation refits."""
    fig = plt.figure(figsize=(12, 8))
    muted = "#59636e"
    border = "#c9d4df"

    def box(x, y, width, height, fill="white"):
        fig.add_artist(
            Rectangle(
                (x, y),
                width,
                height,
                transform=fig.transFigure,
                facecolor=fill,
                edgecolor=border,
                linewidth=1.2,
            )
        )

    def arrow(start, end):
        fig.add_artist(
            FancyArrowPatch(
                start,
                end,
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=15,
                color=muted,
                linewidth=1.4,
                shrinkA=0,
                shrinkB=0,
            )
        )

    fig.text(
        0.5,
        0.96,
        "CortexProbe: from synthetic inputs to results",
        ha="center",
        va="center",
        fontsize=23,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.915,
        "Estimate one field; evaluate predictions with separate fold refits.",
        ha="center",
        va="center",
        fontsize=14,
        color=muted,
    )

    box(0.06, 0.76, 0.88, 0.125, fill="#eef4f9")
    fig.text(0.085, 0.859, "1  Prepare matched inputs", fontsize=16, fontweight="bold")
    fig.text(
        0.085,
        0.832,
        "Build bar, wedge or ring masks; group related frames; prune cross-group overlap.",
        fontsize=13,
    )
    fig.text(
        0.085,
        0.805,
        "Known Gaussian overlap × amplitude + baseline + optional noise → response.",
        fontsize=13,
    )
    fig.text(
        0.085,
        0.778,
        "Align retained masks, responses and group labels by frame.",
        fontsize=12.5,
        color=muted,
    )
    fig.add_artist(
        plt.Line2D(
            [0.27, 0.27, 0.73, 0.73],
            [0.72, 0.742, 0.742, 0.72],
            transform=fig.transFigure,
            color=muted,
            linewidth=1.4,
        )
    )
    fig.add_artist(
        plt.Line2D(
            [0.5, 0.5],
            [0.76, 0.742],
            transform=fig.transFigure,
            color=muted,
            linewidth=1.4,
        )
    )
    arrow((0.27, 0.72), (0.27, 0.705))
    arrow((0.73, 0.72), (0.73, 0.705))

    box(0.06, 0.36, 0.42, 0.345)
    box(0.52, 0.36, 0.42, 0.345)
    fig.text(0.085, 0.671, "2a  Fit on all frames", fontsize=16, fontweight="bold", color=ACCENT)
    fig.text(0.085, 0.626, "Fit one model using all retained frames.", fontsize=13)
    fig.text(0.085, 0.589, "Estimate center, width, amplitude and baseline.", fontsize=12.5)
    fig.text(0.085, 0.537, "Accept the fit only when:", fontsize=13, fontweight="bold")
    for y, criterion in zip(
        (0.494, 0.461, 0.428, 0.395),
        (
            "Optimizer converged",
            "In-sample R² ≥ configured threshold",
            "Amplitude > 0",
            "σ is away from both search bounds",
        ),
    ):
        fig.text(0.1, y, criterion, fontsize=13)

    fig.text(
        0.545,
        0.671,
        "2b  Evaluate held-out groups",
        fontsize=16,
        fontweight="bold",
        color=GOOD,
    )
    for y, instruction in zip(
        (0.626, 0.589, 0.552, 0.515, 0.478),
        (
            "Hold out one group.",
            "Refit using the other groups only.",
            "Keep fitted parameters fixed.",
            "Predict and score the held-out group.",
            "Repeat for each usable fold.",
        ),
    ):
        fig.text(0.545, y, instruction, fontsize=13)
    fig.text(0.545, 0.418, "Report R² for each held-out group.", fontsize=12.5)
    fig.text(0.545, 0.385, "Summarize finite scores with mean and SD.", fontsize=12.5)

    box(0.18, 0.238, 0.64, 0.092, fill="#f4f6f8")
    fig.text(
        0.5,
        0.307,
        "Same fitting algorithm in both paths",
        ha="center",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.279,
        "Coarse grid of center and width → bounded nonlinear refinement",
        ha="center",
        fontsize=12.5,
    )
    fig.text(
        0.5,
        0.251,
        "At every geometry, solve amplitude + baseline by least squares.",
        ha="center",
        fontsize=12.5,
    )
    arrow((0.10, 0.36), (0.10, 0.19))
    arrow((0.90, 0.36), (0.90, 0.19))

    box(0.06, 0.07, 0.88, 0.12, fill="#eef4f9")
    fig.text(0.085, 0.158, "3  Inspect results", fontsize=16, fontweight="bold")
    fig.text(
        0.085,
        0.119,
        "Parameters and acceptance · in-sample and held-out scores · tables and plots",
        fontsize=13,
    )
    fig.text(
        0.085,
        0.086,
        "Held-out scores, local errors and residual second-field gain do not gate acceptance.",
        fontsize=12.5,
        color=muted,
    )
    fig.text(
        0.5,
        0.025,
        "Synthetic only: no neural-network responses or human data evaluated.",
        ha="center",
        fontsize=12,
        color=muted,
    )
    fig.savefig(FIGURES / "pipeline.png", dpi=180)
    plt.close(fig)


def figure_fitted_response() -> None:
    """Distinguish a full-data fit from predictions made by separate training folds."""
    grid, sequence, apertures, clean, response, fit = synthetic_fit_example()
    groups = np.unique(sequence.group)
    field = GaussianReceptiveField(fit.x0, fit.y0, fit.sigma)
    full_prediction = fit.beta * predict(field.weights(grid), apertures) + fit.baseline
    held_out_prediction = np.full_like(response, np.nan)
    fold_scores = []
    for train, test in LeaveOneGroupOut(sequence.group).splits():
        fold_fit = PRFFitter(grid, apertures[train], CONFIG.fit).fit_unit(response[train])
        fold_field = GaussianReceptiveField(fold_fit.x0, fold_fit.y0, fold_fit.sigma)
        held_out_prediction[test] = (
            fold_fit.beta * predict(fold_field.weights(grid), apertures[test]) + fold_fit.baseline
        )
        fold_scores.append(_r_squared(response[test], held_out_prediction[test]))
    finite_scores = [score for score in fold_scores if np.isfinite(score)]
    mean_fold_r2 = np.mean(finite_scores) if finite_scores else float("nan")
    frames = np.arange(len(response))
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 5.4), sharex=True, height_ratios=[2.5, 1])
    axes[0].scatter(
        frames,
        response,
        s=14,
        color="#8c959f",
        alpha=0.8,
        label="observed synthetic response",
        zorder=3,
    )
    axes[0].plot(frames, clean, color=INK, ls=":", lw=1.3, label="known clean response")
    axes[0].plot(
        frames,
        full_prediction,
        color=ACCENT,
        lw=2,
        label=f"fit to all frames (in-sample R² = {fit.r2:.3f})",
    )
    axes[0].plot(
        frames,
        held_out_prediction,
        color=WARN,
        ls="--",
        lw=1.5,
        label=f"held-out predictions (mean fold R² = {mean_fold_r2:.3f})",
    )
    axes[0].set_ylabel("response (arbitrary units)")
    axes[0].legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    axes[0].margins(y=0.25)
    axes[1].axhline(0, color=INK, lw=0.9)
    axes[1].plot(frames, response - full_prediction, color=ACCENT, lw=1.2)
    axes[1].plot(frames, response - held_out_prediction, color=WARN, ls="--", lw=1.2)
    axes[1].set_ylabel("residual")
    axes[1].set_xlabel("retained frame")
    for index, group in enumerate(groups):
        positions = frames[sequence.group == group]
        for axis in axes:
            axis.axvspan(
                positions.min() - 0.5,
                positions.max() + 0.5,
                color=ACCENT,
                alpha=0.035 if index % 2 == 0 else 0,
            )
        axes[1].text(
            positions.mean(),
            0.95,
            f"group {group}",
            transform=axes[1].get_xaxis_transform(),
            va="top",
            ha="center",
            fontsize=7,
            color="#59636e",
        )
    for axis in axes:
        axis.grid(axis="y", color="#eef1f4", lw=1)
        axis.set_axisbelow(True)
        axis.set_xlim(-0.5, len(response) - 0.5)
    fig.suptitle(
        "Response curves: fit to all frames and predict held-out frames",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.025,
        f"Known center ({UNCERTAINTY_TRUTH[0]:g}, {UNCERTAINTY_TRUTH[1]:g}) px; "
        f"σ = {UNCERTAINTY_TRUTH[2]:g} px. "
        f"Gaussian noise = {EXAMPLE_NOISE:.0%} of clean signal SD; seed {RECOVERY_SEED}.\n"
        f"Held-out curve: {len(groups)} separately trained models; "
        "each keeps its parameters fixed on its test group.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.94))
    fig.savefig(FIGURES / "fitted-response.png", dpi=170)
    plt.close(fig)


def figure_noise_autocorrelation() -> None:
    """Describe the existing noise generator with timecourses and lag correlations."""
    n_frames = build_apertures(CONFIG.stimulus, "bar").n_frames
    lags = np.arange(13)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for label, colour in (("white noise", ACCENT), (f"{CARRIER_WIDTH}-frame moving average", WARN)):
        correlations = []
        for seed in CV_SEEDS:
            noise = (
                np.random.default_rng(seed).normal(size=n_frames)
                if label == "white noise"
                else autocorrelated_noise(n_frames, CARRIER_WIDTH, seed)
            )
            correlations.append(
                [1.0] + [float(np.corrcoef(noise[:-lag], noise[lag:])[0, 1]) for lag in lags[1:]]
            )
            if seed == CV_SEEDS[0]:
                standardized = (noise - noise.mean()) / noise.std()
                axes[0].plot(standardized, color=colour, lw=1.3, label=label)
        axes[1].plot(
            lags,
            np.mean(correlations, axis=0),
            "o-",
            color=colour,
            ms=4,
            lw=1.7,
            label=f"{label}: sample mean",
        )
    axes[0].set_title(f"One paired realization (seed {CV_SEEDS[0]})")
    axes[0].set_xlabel("frame")
    axes[0].set_ylabel("noise (standard deviation units)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].axhline(0, color=INK, lw=0.9)
    axes[1].plot(
        lags,
        np.maximum(1 - lags / CARRIER_WIDTH, 0),
        color="#8c959f",
        ls=":",
        lw=1.6,
        label="stationary moving-average reference",
    )
    axes[1].set_title(f"Mean sample autocorrelation ({len(CV_SEEDS)} seeds)")
    axes[1].set_xlabel("lag (frames)")
    axes[1].set_ylabel("Pearson correlation at each lag")
    axes[1].set_xticks(lags[::2])
    axes[1].set_ylim(-0.3, 1.06)
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(axis="y", color="#eef1f4", lw=1)
        axis.set_axisbelow(True)
    fig.suptitle(
        "Noise inputs used in the grouped-validation experiment", fontsize=11.5, fontweight="bold"
    )
    fig.text(
        0.5,
        0.025,
        f"{n_frames} frames per seed; seeds {CV_SEEDS[0]}–{CV_SEEDS[-1]}. "
        "Lagged correlations are computed per seed, then averaged.\n"
        f"Moving-average reference: max(1 − lag/{CARRIER_WIDTH}, 0). "
        "Finite samples can differ. These curves describe inputs, not fit quality.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))
    fig.savefig(FIGURES / "noise-autocorrelation.png", dpi=170)
    plt.close(fig)


def figure_stimulus() -> None:
    """One row per stimulus design, with cross-validation groups labelled."""
    fig, axes = plt.subplots(3, 6, figsize=(8.4, 4.4))
    for row, kind in enumerate(("bar", "wedge", "ring")):
        sequence = build_apertures(CONFIG.stimulus, kind)
        frame_indices = np.linspace(0, sequence.n_frames - 1, 6).astype(int)
        for column, frame_index in enumerate(frame_indices):
            axis = axes[row, column]
            axis.imshow(sequence.apertures[frame_index], cmap="gray_r", interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(f"group {sequence.group[frame_index]}", fontsize=7, pad=2)
            for spine in axis.spines.values():
                spine.set_visible(False)
        axes[row, 0].set_ylabel(
            f"{kind}\n{sequence.n_frames} frames",
            fontsize=9,
            rotation=0,
            ha="right",
            va="center",
            labelpad=26,
        )
    fig.suptitle("Mapping stimuli, frames labelled by cross-validation group", fontsize=10, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURES / "stimuli.png", dpi=150)
    plt.close(fig)


def figure_fold_leakage() -> None:
    """Worst held-out/training similarity, grouping by direction versus by axis."""
    stimulus_config = StimulusConfig(resolution=64, n_steps=20)

    labels, before_similarities, after_similarities = [], [], []

    sequence = build_apertures(stimulus_config, "bar")
    direction_groups = sequence.frame_index // max(1000, stimulus_config.n_steps)
    labels.append("bar\n(8 directions)")
    before_similarities.append(worst_cross_group_similarity(sequence.apertures, direction_groups))
    after_similarities.append(worst_cross_group_similarity(sequence.apertures, sequence.group))

    for kind in ("wedge", "ring"):
        unpruned = build_apertures(
            StimulusConfig(resolution=64, n_steps=20, max_fold_similarity=0.999), kind
        )
        pruned = build_apertures(stimulus_config, kind)
        labels.append(kind)
        before_similarities.append(worst_cross_group_similarity(unpruned.apertures, unpruned.group))
        after_similarities.append(worst_cross_group_similarity(pruned.apertures, pruned.group))

    positions = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(6.2, 3.2))
    axis.bar(positions - 0.19, before_similarities, 0.36, label="before", color=WARN)
    axis.bar(positions + 0.19, after_similarities, 0.36, label="after", color=GOOD)
    axis.axhline(
        stimulus_config.max_fold_similarity,
        color=INK,
        ls="--",
        lw=1,
        label=f"threshold {stimulus_config.max_fold_similarity}",
    )
    for index, (before_similarity, after_similarity) in enumerate(
        zip(before_similarities, after_similarities)
    ):
        axis.text(
            index - 0.19,
            before_similarity + 0.02,
            f"{before_similarity:.3f}",
            ha="center",
            fontsize=8,
            color=WARN,
        )
        axis.text(
            index + 0.19,
            after_similarity + 0.02,
            f"{after_similarity:.3f}",
            ha="center",
            fontsize=8,
            color=GOOD,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylim(0, 1.15)
    axis.set_ylabel("worst held-out ↔ training\ncosine similarity")
    axis.set_title("Cross-group frame similarity")
    axis.legend(frameon=False, fontsize=8, ncol=3, loc="upper center")
    fig.tight_layout()
    fig.savefig(FIGURES / "fold-leakage.png", dpi=150)
    plt.close(fig)


def figure_recovery() -> None:
    """Position and size error against noise, one line per ground-truth pRF."""
    grid = Grid(CONFIG.stimulus.resolution)
    apertures = build_apertures(CONFIG.stimulus, "bar").as_float()
    fitter = PRFFitter(grid, apertures, CONFIG.fit)

    position_errors = np.zeros((len(GROUND_TRUTH), len(RECOVERY_NOISE)))
    sigma_errors = np.zeros_like(position_errors)
    for field_index, truth in enumerate(GROUND_TRUTH):
        for noise_index, noise in enumerate(RECOVERY_NOISE):
            fit = fitter.fit_unit(
                synthesise(grid, apertures, truth, noise=noise, seed=RECOVERY_SEED)
            )
            position_errors[field_index, noise_index] = float(
                np.hypot(fit.x0 - truth[0], fit.y0 - truth[1])
            )
            sigma_errors[field_index, noise_index] = abs(fit.sigma - truth[2]) / truth[2] * 100.0

    noise_percentages = [noise * 100 for noise in RECOVERY_NOISE]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    for axis, errors, error_label, decimal_places in (
        (axes[0], position_errors, "position error (px)", 3),
        (axes[1], sigma_errors, "sigma error (%)", 1),
    ):
        largest_errors = errors.max(axis=0)
        for field_index in range(len(GROUND_TRUTH)):
            axis.plot(
                noise_percentages,
                errors[field_index],
                marker="o",
                ms=3.5,
                lw=1,
                color="#c3cad1",
                zorder=1,
                label="individual pRF" if field_index == 0 else None,
            )
        axis.plot(
            noise_percentages,
            largest_errors,
            marker="o",
            ms=6,
            lw=2.2,
            color=WARN,
            zorder=3,
            label=f"largest of {len(GROUND_TRUTH)}",
        )
        axis.plot(
            noise_percentages,
            errors.mean(axis=0),
            marker="o",
            ms=6,
            lw=2.2,
            color=ACCENT,
            zorder=3,
            label=f"mean of {len(GROUND_TRUTH)}",
        )
        for noise_index, error_value in enumerate(largest_errors):
            displayed_error = 0.0 if error_value < 1e-6 else error_value
            axis.annotate(
                f"{displayed_error:.{decimal_places}f}",
                (noise_percentages[noise_index], error_value),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=8.5,
                color=WARN,
                fontweight="bold",
            )
        axis.set_xlabel("noise (% of signal standard deviation)")
        axis.set_ylabel(error_label)
        axis.set_xticks(noise_percentages)
        axis.set_xticklabels([f"{int(noise)}" for noise in noise_percentages])
        axis.margins(y=0.28)
        axis.set_ylim(bottom=-errors.max() * 0.06)
        axis.grid(axis="y", color="#eef1f4", lw=1)
        axis.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(
        f"Recovery error: {len(GROUND_TRUTH)} synthetic fields", fontsize=11, fontweight="bold"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIGURES / "recovery.png", dpi=150)
    plt.close(fig)


def figure_leakage() -> None:
    """Grouped and random-frame held-out scores for the validation seeds."""
    grid = Grid(CONFIG.stimulus.resolution)
    sequence = build_apertures(CONFIG.stimulus, "bar")
    apertures = sequence.as_float()
    grouped = CrossValidator(grid, apertures, sequence.group, CONFIG.fit)
    random_groups = np.random.default_rng(SPLIT_SEED).permutation(len(apertures)) % 4
    random_split = CrossValidator(grid, apertures, random_groups, CONFIG.fit)

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True)
    for axis, label in zip(axes, ("white", "autocorrelated")):
        grouped_scores, random_scores = [], []
        for seed in CV_SEEDS:
            if label == "white":
                response = np.random.default_rng(seed).normal(size=len(apertures))
            else:
                response = autocorrelated_noise(len(apertures), CARRIER_WIDTH, seed)
            grouped_scores.append(grouped.validate_unit(response).cv_r2)
            random_scores.append(random_split.validate_unit(response).cv_r2)

        for grouped_score, random_score in zip(grouped_scores, random_scores):
            axis.plot([0, 1], [grouped_score, random_score], color="#8c959f", lw=0.7, zorder=1)
        axis.scatter(np.zeros(len(CV_SEEDS)), grouped_scores, s=22, color=GOOD, zorder=2)
        axis.scatter(np.ones(len(CV_SEEDS)), random_scores, s=22, color=WARN, zorder=2)
        axis.axhline(0, color=INK, lw=1)
        axis.set_xticks([0, 1])
        axis.set_xticklabels(["grouped\n(axis)", "random\nframe split"])
        axis.set_xlim(-0.35, 1.35)
        differences = np.asarray(random_scores) - np.asarray(grouped_scores)
        axis.set_title(
            f"{label} noise\nmean difference {differences.mean():+.3f}, "
            f"{int((differences > 0).sum())}/{len(CV_SEEDS)} positive"
        )
    axes[0].set_ylabel("held-out R²")
    fig.suptitle(f"Grouped and random-split validation: {len(CV_SEEDS)} noise seeds", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIGURES / "leakage.png", dpi=150)
    plt.close(fig)


def figure_model() -> None:
    """How one predicted timecourse is formed: a pRF, the apertures it sees, the response."""
    grid = Grid(CONFIG.stimulus.resolution)
    sequence = build_apertures(CONFIG.stimulus, "bar")
    apertures = sequence.as_float()
    field = GaussianReceptiveField(12.0, 8.0, 5.0)
    weights = field.weights(grid)
    response = predict(weights, apertures)

    fig = plt.figure(figsize=(9.2, 3.6))
    layout = fig.add_gridspec(
        1,
        4,
        width_ratios=[1, 1, 1, 2.6],
        wspace=0.3,
        left=0.04,
        right=0.98,
        top=0.72,
        bottom=0.17,
    )

    axis = fig.add_subplot(layout[0, 0])
    axis.imshow(weights, cmap="magma", interpolation="nearest")
    axis.set_title("pRF $w(x,y)$", fontsize=9)
    axis.set_xlabel(r"$x_0{=}12,\ y_0{=}8,\ \sigma{=}5$", fontsize=8)
    axis.set_xticks([])
    axis.set_yticks([])

    for column, frame in ((1, 8), (2, 14)):
        axis = fig.add_subplot(layout[0, column])
        axis.imshow(apertures[frame], cmap="gray_r", interpolation="nearest")
        axis.contour(weights, levels=3, colors=WARN, linewidths=0.9)
        axis.set_title(f"aperture $a_{{t}}$, $t{{=}}{frame}$", fontsize=9)
        axis.set_xlabel("little overlap" if frame == 8 else "much overlap", fontsize=8)
        axis.set_xticks([])
        axis.set_yticks([])

    axis = fig.add_subplot(layout[0, 3])
    axis.plot(response, color=ACCENT, lw=1.5)
    for frame in (8, 14):
        axis.axvline(frame, color="#8c959f", ls="--", lw=1)
        axis.plot([frame], [response[frame]], "o", ms=7, color=WARN, zorder=3)
    for frame, vertical_offset in ((8, 10), (14, -4)):
        axis.annotate(
            f"t={frame}",
            (frame, response[frame]),
            textcoords="offset points",
            xytext=(6, vertical_offset),
            fontsize=8,
            color=WARN,
        )
    axis.set_xlabel("frame")
    axis.set_ylabel("predicted response")
    axis.set_title("timecourse, one peak per sweep", fontsize=9)
    axis.grid(axis="y", color="#eef1f4", lw=1)
    axis.set_axisbelow(True)

    fig.suptitle(
        r"The forward model:  $r(t)=\sum_{x,y} w(x,y)\,a_t(x,y)$",
        fontsize=11.5,
        fontweight="bold",
        y=0.93,
    )
    fig.savefig(FIGURES / "model.png", dpi=150)
    plt.close(fig)


def figure_sigma_guard() -> None:
    """Centered-field truncation and subpixel sampling diagnostics."""
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4))

    grid = Grid(CONFIG.stimulus.resolution)
    sigmas = np.linspace(0.1, 30.0, 220)
    retained_mass = [
        float(GaussianReceptiveField(0.0, 0.0, float(sigma)).weights(grid)[grid.field_mask].sum())
        for sigma in sigmas
    ]
    ceiling = grid.resolution / RESOLUTION_PER_SIGMA
    axes[0].axvspan(0, 1.0, color=WARN, alpha=0.10)
    axes[0].axvspan(ceiling, 30, color=WARN, alpha=0.10)
    axes[0].plot(sigmas, retained_mass, color=ACCENT, lw=2.2, zorder=3)
    axes[0].axhline(MIN_ON_GRID_VOLUME, color=INK, ls="--", lw=1, zorder=2)
    axes[0].annotate(
        f"tolerance {MIN_ON_GRID_VOLUME}",
        (29.5, MIN_ON_GRID_VOLUME),
        ha="right",
        va="bottom",
        fontsize=8,
        color=INK,
    )
    axes[0].annotate(
        "floor\n1 px",
        (1.0, 0.30),
        ha="left",
        va="center",
        fontsize=8,
        color=WARN,
        xytext=(3.0, 0.30),
        textcoords="data",
        arrowprops={"arrowstyle": "->", "color": WARN, "lw": 0.9},
    )
    axes[0].annotate(
        f"ceiling {ceiling:.1f} px",
        (ceiling, 0.55),
        ha="center",
        va="bottom",
        fontsize=8,
        color=WARN,
        rotation=90,
    )
    axes[0].set_xlabel(rf"$\sigma$ (px) on a {grid.resolution} px field")
    axes[0].set_ylabel("unit volume retained on grid")
    axes[0].set_ylim(0, 1.12)
    axes[0].set_xlim(0, 30)
    axes[0].set_title("Centered Gaussian on a finite field")

    offsets = np.linspace(0.0, 0.5, 60)
    sampled_mass = [
        float(GaussianReceptiveField(float(offset), 0.0, 0.2).weights(grid).sum())
        for offset in offsets
    ]
    on_pixel = float(GaussianReceptiveField(0.5, -0.5, 0.2).weights(grid).sum())
    axes[1].axhline(1.0, color=GOOD, ls="--", lw=1.2, zorder=2)
    axes[1].annotate("unit volume = 1.0", (0.02, 1.0), va="bottom", fontsize=8, color=GOOD)
    axes[1].plot(offsets, sampled_mass, color=ACCENT, lw=2.2, zorder=3)
    axes[1].scatter([0.5], [on_pixel], s=60, color=WARN, zorder=4)
    axes[1].annotate(
        f"on a pixel centre\n{on_pixel:.2f}",
        (0.5, on_pixel),
        ha="right",
        va="center",
        fontsize=8,
        color=WARN,
        xytext=(0.42, on_pixel),
        textcoords="data",
    )
    axes[1].annotate(
        f"between pixels\n{sampled_mass[0]:.3f}",
        (0.0, sampled_mass[0]),
        ha="left",
        va="bottom",
        fontsize=8,
        color=ACCENT,
        xytext=(0.03, sampled_mass[0] * 1.4),
        textcoords="data",
        arrowprops={"arrowstyle": "->", "color": ACCENT, "lw": 0.9},
    )
    axes[1].set_xlabel("centre offset from the field origin (px)")
    axes[1].set_ylabel(r"sum at $\sigma = 0.2$")
    axes[1].set_yscale("log")
    axes[1].set_ylim(0.02, 12)
    axes[1].set_title("Under-sampled below the floor")

    for axis in axes:
        axis.grid(axis="y", color="#eef1f4", lw=1)
        axis.set_axisbelow(True)
    fig.suptitle("Sampling and centered-field limits", fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(FIGURES / "sigma-guard.png", dpi=150)
    plt.close(fig)


def figure_uncertainty() -> None:
    """Plot x0 standard errors, interval widths, and measured interval coverage."""
    grid = Grid(CONFIG.stimulus.resolution)
    apertures = build_apertures(CONFIG.stimulus, "bar").as_float()
    fitter = PRFFitter(grid, apertures, CONFIG.fit)

    standard_errors, interval_widths = [], []
    for noise in UNCERTAINTY_NOISE:
        fit = fitter.fit_unit(
            synthesise(grid, apertures, UNCERTAINTY_TRUTH, noise=noise, seed=UNCERTAINTY_SEED)
        )
        lower, upper = fit.confidence_interval("x0")
        standard_errors.append(fit.se_x0)
        interval_widths.append(upper - lower)

    report = load_recorded_report()
    coverage = report["uncertainty"]["coverage"]

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4))

    axes[0].plot(
        UNCERTAINTY_NOISE,
        standard_errors,
        marker="o",
        ms=6,
        lw=2.2,
        color=ACCENT,
        label=r"SE of $x_0$",
    )
    axes[0].plot(
        UNCERTAINTY_NOISE,
        interval_widths,
        marker="s",
        ms=6,
        lw=2.2,
        color=WARN,
        label="95 % CI width",
    )
    for noise, interval_width in zip(UNCERTAINTY_NOISE, interval_widths):
        axes[0].annotate(
            f"{interval_width:.3f}",
            (noise, interval_width),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            color=WARN,
        )
    axes[0].set_xlabel("noise (fraction of signal SD)")
    axes[0].set_ylabel("pixels")
    axes[0].set_title("Standard error and interval width")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    axes[0].margins(y=0.24)

    labels = [f"{row['position']}\n{row['noise']:.0%} noise" for row in coverage]
    coverage_rates = np.array([row["coverage"] for row in coverage])
    lower_bounds = np.array([row["ci_low"] for row in coverage])
    upper_bounds = np.array([row["ci_high"] for row in coverage])
    positions = np.arange(len(labels))
    nominal = report["uncertainty"]["nominal"]
    axes[1].axhline(nominal, color=INK, ls="--", lw=1)
    axes[1].annotate(
        f"nominal {nominal:.2f}",
        (len(labels) - 0.5, nominal + 0.001),
        ha="right",
        va="bottom",
        fontsize=8,
        color=INK,
    )
    axes[1].errorbar(
        positions,
        coverage_rates,
        yerr=[coverage_rates - lower_bounds, upper_bounds - coverage_rates],
        fmt="o",
        ms=7,
        lw=1.6,
        capsize=5,
        color=ACCENT,
    )
    for position, coverage_rate in zip(positions, coverage_rates):
        axes[1].annotate(
            f"{coverage_rate:.3f}",
            (position, coverage_rate),
            textcoords="offset points",
            xytext=(9, -3),
            fontsize=8,
            color=ACCENT,
        )
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylim(0.86, 1.0)
    axes[1].set_ylabel("empirical coverage")
    axes[1].set_title(f"Coverage of the 95 % x0 interval (n = {coverage[0]['n']} each)")

    for axis in axes:
        axis.grid(axis="y", color="#eef1f4", lw=1)
        axis.set_axisbelow(True)
    fig.suptitle(
        "Local uncertainty and empirical coverage",
        fontsize=11.5,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIGURES / "uncertainty.png", dpi=150)
    plt.close(fig)


def figure_misspecification() -> None:
    """Compare residual second-field gain for single- and two-lobe responses."""
    grid = Grid(CONFIG.stimulus.resolution)
    apertures = build_apertures(CONFIG.stimulus, "bar").as_float()
    fitter = PRFFitter(grid, apertures, CONFIG.fit)

    def noisy_response(weights, noise, seed):
        clean = 3.0 * predict(weights, apertures) + 0.5
        if noise <= 0:
            return clean
        rng = np.random.default_rng(seed)
        return clean + rng.normal(0.0, noise * float(clean.std()), size=clean.shape)

    near_lobes = GaussianReceptiveField(14.0, 0.0, 3.0).weights(grid) + GaussianReceptiveField(
        -14.0, 0.0, 3.0
    ).weights(grid)
    far_lobes = GaussianReceptiveField(16.0, 10.0, 3.0).weights(grid) + GaussianReceptiveField(
        -16.0, -10.0, 3.0
    ).weights(grid)
    single_field = GaussianReceptiveField(12.0, 8.0, 5.0).weights(grid)

    noise_levels = [0.0, 0.2, 0.5, 0.8]
    weights_by_label = {
        "two lobes, far apart": far_lobes,
        "two lobes, close": near_lobes,
        "single Gaussian": single_field,
    }
    fits_by_label = {
        label: [fitter.fit_unit(noisy_response(weights, noise, 7)) for noise in noise_levels]
        for label, weights in weights_by_label.items()
    }
    pure_noise_gain = fitter.fit_unit(
        np.random.default_rng(11).normal(size=fitter.n_frames)
    ).second_field_r2

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4))

    axes[0].imshow(near_lobes, cmap="magma", interpolation="nearest")
    fit = fits_by_label["two lobes, close"][0]
    circle = plt.Circle(
        pixel_coordinates(grid, fit.x0, fit.y0), fit.sigma, fill=False, color=GOOD, lw=2
    )
    axes[0].add_patch(circle)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    status = "accepted" if fit.accepted else "rejected"
    axes[0].set_title(f"two lobes, one fitted pRF\n$R^2$ = {fit.r2:.4f}, {status}", fontsize=9)
    axes[0].set_xlabel(f"fitted $\\sigma$ = {fit.sigma:.2f} px", fontsize=8)

    noise_percentages = [noise * 100 for noise in noise_levels]
    for (label, fits), colour, marker in zip(
        fits_by_label.items(), (WARN, "#d98a3a", ACCENT), ("o", "s", "^")
    ):
        axes[1].plot(
            noise_percentages,
            [fit.second_field_r2 for fit in fits],
            marker=marker,
            ms=6,
            lw=2.2,
            color=colour,
            label=label,
        )
    axes[1].axhline(pure_noise_gain, color=INK, ls="--", lw=1)
    axes[1].annotate(
        f"pure noise {pure_noise_gain:.3f}",
        (80, pure_noise_gain),
        ha="right",
        va="bottom",
        fontsize=8,
        color=INK,
    )
    axes[1].set_xlabel("noise (% of signal SD)")
    axes[1].set_ylabel("second-field $R^2$ gain")
    axes[1].set_title("Residual second-field gain")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", color="#eef1f4", lw=1)
    axes[1].set_axisbelow(True)
    axes[1].margins(y=0.2)

    fig.suptitle(
        "Single-Gaussian fit to two synthetic lobes",
        fontsize=11.5,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIGURES / "misspecification.png", dpi=150)
    plt.close(fig)


def figure_runtime() -> None:
    """Recorded hardware timings for fit and cross-validation."""
    report = load_recorded_report()
    rows = report["runtime"]["rows"]
    unit_counts = [row["units"] for row in rows]
    per_unit_ms = [row["per_unit_ms"] for row in rows]
    cv_factors = [row["cv_factor"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4))

    axes[0].plot(unit_counts, per_unit_ms, marker="o", ms=7, lw=2.2, color=ACCENT)
    for unit_count, timing_ms in zip(unit_counts, per_unit_ms):
        axes[0].annotate(
            f"{timing_ms:.1f}",
            (unit_count, timing_ms),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=8,
            color=ACCENT,
        )
    axes[0].set_xscale("log")
    axes[0].set_xticks(unit_counts)
    axes[0].set_xticklabels([str(unit_count) for unit_count in unit_counts])
    axes[0].set_ylim(0, max(per_unit_ms) * 1.5)
    axes[0].set_xlabel("units fitted")
    axes[0].set_ylabel("per unit (ms)")
    axes[0].set_title("Measured time per unit")

    axes[1].bar([str(unit_count) for unit_count in unit_counts], cv_factors, color=GOOD, width=0.55)
    for index, cv_factor in enumerate(cv_factors):
        axes[1].annotate(
            f"{cv_factor:.1f}×",
            (index, cv_factor),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=8,
            color=GOOD,
        )
    axes[1].set_xlabel("units fitted")
    axes[1].set_ylabel("CV cost ÷ fit cost")
    axes[1].set_ylim(0, max(cv_factors) * 1.35)
    axes[1].set_title("Cross-validation overhead")

    for axis in axes:
        axis.grid(axis="y", color="#eef1f4", lw=1)
        axis.set_axisbelow(True)
    fig.suptitle("Recorded hardware timings", fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIGURES / "runtime.png", dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate 12 figures in docs/figures/.")
    parser.parse_args(argv)
    try:
        load_recorded_report()
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    FIGURES.mkdir(parents=True, exist_ok=True)
    for draw in (
        figure_pipeline,
        figure_fitted_response,
        figure_noise_autocorrelation,
        figure_stimulus,
        figure_fold_leakage,
        figure_recovery,
        figure_leakage,
        figure_model,
        figure_sigma_guard,
        figure_uncertainty,
        figure_misspecification,
        figure_runtime,
    ):
        draw()
        print(f"drew {draw.__name__.removeprefix('figure_')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
