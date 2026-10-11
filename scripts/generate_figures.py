"""Regenerate the figures in README.md.

Coverage and timing panels use outputs/validation.json; other panels run supplemental
synthetic diagnostics. A matching configuration digest is required. Run the report's
--check first to verify recorded numbers; it does not compare figure pixels.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_validation_report import (
    CONFIG,
    CV_SEEDS,
    GROUND_TRUTH,
    OUTPUTS,
    RECOVERY_NOISE,
    RECOVERY_SEED,
    SPLIT_SEED,
    UNCERTAINTY_NOISE,
    UNCERTAINTY_SEED,
    UNCERTAINTY_TRUTH,
    autocorrelated_noise,
    synthesise,
)

from cortexprobe.geometry import Grid
from cortexprobe.prf.fit import MIN_ON_GRID_VOLUME, RESOLUTION_PER_SIGMA, PRFFitter
from cortexprobe.prf.model import GaussianReceptiveField, predict
from cortexprobe.prf.validation import CrossValidator
from cortexprobe.stimuli import build_apertures, frame_similarity

FIGURES = OUTPUTS
INK = "#1b1f23"
ACCENT = "#2f6f9f"
WARN = "#b4442e"
GOOD = "#3f7d58"

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
    worst = 0.0
    for group in np.unique(groups):
        test = np.flatnonzero(groups == group)
        train = np.flatnonzero(groups != group)
        if len(train):
            worst = max(worst, float(similarity[np.ix_(test, train)].max(axis=1).max()))
    return worst


def load_recorded_report() -> dict:
    """Reject absent or stale validation records before drawing figures."""
    path = OUTPUTS / "validation.json"
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


def figure_stimulus() -> None:
    """One row per design, frames coloured by cross-validation group."""
    fig, axes = plt.subplots(3, 6, figsize=(8.4, 4.4))
    for row, kind in enumerate(("bar", "wedge", "ring")):
        sequence = build_apertures(CONFIG.stimulus, kind)
        picks = np.linspace(0, sequence.n_frames - 1, 6).astype(int)
        for column, index in enumerate(picks):
            axis = axes[row, column]
            axis.imshow(sequence.apertures[index], cmap="gray_r", interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(f"group {sequence.group[index]}", fontsize=7, pad=2)
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
    grid_config = CONFIG.stimulus
    default = grid_config.__class__(resolution=64, n_steps=20)

    labels, before, after = [], [], []

    sequence = build_apertures(default, "bar")
    by_direction = np.array([d for d in default.directions for _ in range(default.n_steps)])
    labels.append("bar\n(8 directions)")
    before.append(worst_cross_group_similarity(sequence.apertures, by_direction))
    after.append(worst_cross_group_similarity(sequence.apertures, sequence.group))

    for kind in ("wedge", "ring"):
        unpruned = build_apertures(
            grid_config.__class__(resolution=64, n_steps=20, max_fold_similarity=0.999), kind
        )
        pruned = build_apertures(default, kind)
        labels.append(kind)
        before.append(worst_cross_group_similarity(unpruned.apertures, unpruned.group))
        after.append(worst_cross_group_similarity(pruned.apertures, pruned.group))

    x = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(6.2, 3.2))
    axis.bar(x - 0.19, before, 0.36, label="before", color=WARN)
    axis.bar(x + 0.19, after, 0.36, label="after", color=GOOD)
    axis.axhline(
        default.max_fold_similarity,
        color=INK,
        ls="--",
        lw=1,
        label=f"threshold {default.max_fold_similarity}",
    )
    for index, (b, a) in enumerate(zip(before, after)):
        axis.text(index - 0.19, b + 0.02, f"{b:.3f}", ha="center", fontsize=8, color=WARN)
        axis.text(index + 0.19, a + 0.02, f"{a:.3f}", ha="center", fontsize=8, color=GOOD)
    axis.set_xticks(x)
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

    position = np.zeros((len(GROUND_TRUTH), len(RECOVERY_NOISE)))
    size = np.zeros_like(position)
    for row, truth in enumerate(GROUND_TRUTH):
        for column, noise in enumerate(RECOVERY_NOISE):
            fit = fitter.fit_unit(
                synthesise(grid, apertures, truth, noise=noise, seed=RECOVERY_SEED)
            )
            position[row, column] = float(np.hypot(fit.x0 - truth[0], fit.y0 - truth[1]))
            size[row, column] = abs(fit.sigma - truth[2]) / truth[2] * 100.0

    noise_pct = [n * 100 for n in RECOVERY_NOISE]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    for axis, data, label, places in (
        (axes[0], position, "position error (px)", 3),
        (axes[1], size, "sigma error (%)", 1),
    ):
        for row in range(len(GROUND_TRUTH)):
            axis.plot(
                noise_pct,
                data[row],
                marker="o",
                ms=3.5,
                lw=1,
                color="#c3cad1",
                zorder=1,
                label="individual pRF" if row == 0 else None,
            )
        axis.plot(
            noise_pct,
            data.max(axis=0),
            marker="o",
            ms=6,
            lw=2.2,
            color=WARN,
            zorder=3,
            label="largest of 5",
        )
        axis.plot(
            noise_pct,
            data.mean(axis=0),
            marker="o",
            ms=6,
            lw=2.2,
            color=ACCENT,
            zorder=3,
            label="mean of 5",
        )
        for column, value in enumerate(data.max(axis=0)):
            shown = 0.0 if value < 1e-6 else value
            axis.annotate(
                f"{shown:.{places}f}",
                (noise_pct[column], value),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=8.5,
                color=WARN,
                fontweight="bold",
            )
        axis.set_xlabel("noise (% of signal standard deviation)")
        axis.set_ylabel(label)
        axis.set_xticks(noise_pct)
        axis.set_xticklabels([f"{int(n)}" for n in noise_pct])
        axis.margins(y=0.28)
        axis.set_ylim(bottom=-data.max() * 0.06)
        axis.grid(axis="y", color="#eef1f4", lw=1)
        axis.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Recovery error: 5 synthetic fields", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIGURES / "recovery.png", dpi=150)
    plt.close(fig)


def figure_leakage() -> None:
    """Grouped and random-frame held-out scores over 20 seeds."""
    grid = Grid(CONFIG.stimulus.resolution)
    sequence = build_apertures(CONFIG.stimulus, "bar")
    apertures = sequence.as_float()
    grouped = CrossValidator(grid, apertures, sequence.group, CONFIG.fit)
    shuffled = np.random.default_rng(SPLIT_SEED).permutation(len(apertures)) % 4
    leaky = CrossValidator(grid, apertures, shuffled, CONFIG.fit)

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True)
    for axis, label in zip(axes, ("white", "autocorrelated")):
        honest, split = [], []
        for seed in CV_SEEDS:
            if label == "white":
                response = np.random.default_rng(seed).normal(size=len(apertures))
            else:
                response = autocorrelated_noise(len(apertures), 3, seed)
            honest.append(grouped.validate_unit(response).cv_r2)
            split.append(leaky.validate_unit(response).cv_r2)

        for h, s in zip(honest, split):
            axis.plot([0, 1], [h, s], color="#8c959f", lw=0.7, zorder=1)
        axis.scatter(np.zeros(len(honest)), honest, s=22, color=GOOD, zorder=2, label="grouped")
        axis.scatter(np.ones(len(split)), split, s=22, color=WARN, zorder=2, label="random split")
        axis.axhline(0, color=INK, lw=1)
        axis.set_xticks([0, 1])
        axis.set_xticklabels(["grouped\n(axis)", "random\nframe split"])
        axis.set_xlim(-0.35, 1.35)
        leak = np.mean(np.asarray(split) - np.asarray(honest))
        axis.set_title(
            f"{label} noise\nmean leak {leak:+.3f}, "
            f"{int((np.asarray(split) > np.asarray(honest)).sum())}/{len(CV_SEEDS)} positive"
        )
    axes[0].set_ylabel("held-out R²")
    fig.suptitle("Grouped and random-split validation: 20 noise seeds", fontsize=10)
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
    spec = fig.add_gridspec(
        1,
        4,
        width_ratios=[1, 1, 1, 2.6],
        wspace=0.3,
        left=0.04,
        right=0.98,
        top=0.72,
        bottom=0.17,
    )

    axis = fig.add_subplot(spec[0, 0])
    axis.imshow(weights, cmap="magma", interpolation="nearest")
    axis.set_title("pRF $w(x,y)$", fontsize=9)
    axis.set_xlabel(r"$x_0{=}12,\ y_0{=}8,\ \sigma{=}5$", fontsize=8)
    axis.set_xticks([])
    axis.set_yticks([])

    for column, frame in ((1, 8), (2, 14)):
        axis = fig.add_subplot(spec[0, column])
        axis.imshow(apertures[frame], cmap="gray_r", interpolation="nearest")
        axis.contour(weights, levels=3, colors=WARN, linewidths=0.9)
        axis.set_title(f"aperture $a_{{t}}$, $t{{=}}{frame}$", fontsize=9)
        axis.set_xlabel("little overlap" if frame == 8 else "much overlap", fontsize=8)
        axis.set_xticks([])
        axis.set_yticks([])

    axis = fig.add_subplot(spec[0, 3])
    axis.plot(response, color=ACCENT, lw=1.5)
    for frame in (8, 14):
        axis.axvline(frame, color="#8c959f", ls="--", lw=1)
        axis.plot([frame], [response[frame]], "o", ms=7, color=WARN, zorder=3)
    axis.annotate(
        "t=8", (8, response[8]), textcoords="offset points", xytext=(6, 10), fontsize=8, color=WARN
    )
    axis.annotate(
        "t=14",
        (14, response[14]),
        textcoords="offset points",
        xytext=(6, -4),
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
    volume = [
        float(GaussianReceptiveField(0.0, 0.0, float(s)).weights(grid)[grid.field_mask].sum())
        for s in sigmas
    ]
    ceiling = grid.resolution / RESOLUTION_PER_SIGMA
    axes[0].axvspan(0, 1.0, color=WARN, alpha=0.10)
    axes[0].axvspan(ceiling, 30, color=WARN, alpha=0.10)
    axes[0].plot(sigmas, volume, color=ACCENT, lw=2.2, zorder=3)
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
    sums = [float(GaussianReceptiveField(float(o), 0.0, 0.2).weights(grid).sum()) for o in offsets]
    on_pixel = float(GaussianReceptiveField(0.5, -0.5, 0.2).weights(grid).sum())
    axes[1].axhline(1.0, color=GOOD, ls="--", lw=1.2, zorder=2)
    axes[1].annotate("unit volume = 1.0", (0.02, 1.0), va="bottom", fontsize=8, color=GOOD)
    axes[1].plot(offsets, sums, color=ACCENT, lw=2.2, zorder=3)
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
        f"between pixels\n{sums[0]:.3f}",
        (0.0, sums[0]),
        ha="left",
        va="bottom",
        fontsize=8,
        color=ACCENT,
        xytext=(0.03, sums[0] * 1.4),
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
    """Standard errors grow with noise, and the intervals are checked for coverage."""
    grid = Grid(CONFIG.stimulus.resolution)
    apertures = build_apertures(CONFIG.stimulus, "bar").as_float()
    fitter = PRFFitter(grid, apertures, CONFIG.fit)

    noises, errors, widths = [], [], []
    for noise in UNCERTAINTY_NOISE:
        fit = fitter.fit_unit(
            synthesise(grid, apertures, UNCERTAINTY_TRUTH, noise=noise, seed=UNCERTAINTY_SEED)
        )
        low, high = fit.confidence_interval("x0")
        noises.append(noise)
        errors.append(fit.se_x0)
        widths.append(high - low)

    report = load_recorded_report()
    coverage = report["uncertainty"]["coverage"]

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4))

    axes[0].plot(noises, errors, marker="o", ms=6, lw=2.2, color=ACCENT, label=r"SE of $x_0$")
    axes[0].plot(noises, widths, marker="s", ms=6, lw=2.2, color=WARN, label="95 % CI width")
    for x, y in zip(noises, widths):
        axes[0].annotate(
            f"{y:.3f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            color=WARN,
        )
    axes[0].set_xlabel("noise (fraction of signal SD)")
    axes[0].set_ylabel("pixels")
    axes[0].set_title("Uncertainty tracks the noise")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    axes[0].margins(y=0.24)

    labels = [f"{r['position']}\n{r['noise']:.0%} noise" for r in coverage]
    values = [r["coverage"] for r in coverage]
    lows = [r["ci_low"] for r in coverage]
    highs = [r["ci_high"] for r in coverage]
    x = np.arange(len(labels))
    axes[1].axhline(0.95, color=INK, ls="--", lw=1)
    axes[1].annotate(
        "nominal 0.95", (len(labels) - 0.5, 0.951), ha="right", va="bottom", fontsize=8, color=INK
    )
    axes[1].errorbar(
        x,
        values,
        yerr=[np.array(values) - np.array(lows), np.array(highs) - np.array(values)],
        fmt="o",
        ms=7,
        lw=1.6,
        capsize=5,
        color=ACCENT,
    )
    for xi, v in zip(x, values):
        axes[1].annotate(
            f"{v:.3f}",
            (xi, v),
            textcoords="offset points",
            xytext=(9, -3),
            fontsize=8,
            color=ACCENT,
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylim(0.86, 1.0)
    axes[1].set_ylabel("empirical coverage")
    axes[1].set_title(f"Coverage of the 95 % interval (n = {coverage[0]['n']} each)")

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
    """A single Gaussian cannot represent two lobes, and the diagnostic says so."""
    grid = Grid(CONFIG.stimulus.resolution)
    apertures = build_apertures(CONFIG.stimulus, "bar").as_float()
    fitter = PRFFitter(grid, apertures, CONFIG.fit)

    def response(weights, noise, seed):
        clean = 3.0 * predict(weights, apertures) + 0.5
        if noise <= 0:
            return clean
        rng = np.random.default_rng(seed)
        return clean + rng.normal(0.0, noise * float(clean.std()), size=clean.shape)

    lobes_near = GaussianReceptiveField(14.0, 0.0, 3.0).weights(grid) + GaussianReceptiveField(
        -14.0, 0.0, 3.0
    ).weights(grid)
    lobes_far = GaussianReceptiveField(16.0, 10.0, 3.0).weights(grid) + GaussianReceptiveField(
        -16.0, -10.0, 3.0
    ).weights(grid)
    single = GaussianReceptiveField(12.0, 8.0, 5.0).weights(grid)

    noises = [0.0, 0.2, 0.5, 0.8]
    series = {
        "two lobes, far apart": [
            fitter.fit_unit(response(lobes_far, n, 7)).second_field_r2 for n in noises
        ],
        "two lobes, close": [
            fitter.fit_unit(response(lobes_near, n, 7)).second_field_r2 for n in noises
        ],
        "single Gaussian": [
            fitter.fit_unit(response(single, n, 7)).second_field_r2 for n in noises
        ],
    }
    pure = fitter.fit_unit(np.random.default_rng(11).normal(size=fitter.n_frames)).second_field_r2

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4))

    axes[0].imshow(lobes_near, cmap="magma", interpolation="nearest")
    fit = fitter.fit_unit(response(lobes_near, 0.0, 7))
    circle = plt.Circle(
        pixel_coordinates(grid, fit.x0, fit.y0), fit.sigma, fill=False, color=GOOD, lw=2
    )
    axes[0].add_patch(circle)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_title(f"two lobes, one fitted pRF\n$R^2$ = {fit.r2:.4f}, accepted", fontsize=9)
    axes[0].set_xlabel(f"fitted $\\sigma$ = {fit.sigma:.2f} px", fontsize=8)

    for (label, values), colour, marker in zip(
        series.items(), (WARN, "#d98a3a", ACCENT), ("o", "s", "^")
    ):
        axes[1].plot(
            [n * 100 for n in noises],
            values,
            marker=marker,
            ms=6,
            lw=2.2,
            color=colour,
            label=label,
        )
    axes[1].axhline(pure, color=INK, ls="--", lw=1)
    axes[1].annotate(
        f"pure noise {pure:.3f}", (80, pure), ha="right", va="bottom", fontsize=8, color=INK
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
    units = [r["units"] for r in rows]
    per_unit = [r["per_unit_ms"] for r in rows]
    factor = [r["cv_factor"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4))

    axes[0].plot(units, per_unit, marker="o", ms=7, lw=2.2, color=ACCENT)
    for u, v in zip(units, per_unit):
        axes[0].annotate(
            f"{v:.1f}",
            (u, v),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=8,
            color=ACCENT,
        )
    axes[0].set_xscale("log")
    axes[0].set_xticks(units)
    axes[0].set_xticklabels([str(u) for u in units])
    axes[0].set_ylim(0, max(per_unit) * 1.5)
    axes[0].set_xlabel("units fitted")
    axes[0].set_ylabel("per unit (ms)")
    axes[0].set_title("Measured time per unit")

    axes[1].bar([str(u) for u in units], factor, color=GOOD, width=0.55)
    for i, v in enumerate(factor):
        axes[1].annotate(
            f"{v:.1f}×",
            (i, v),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=8,
            color=GOOD,
        )
    axes[1].set_xlabel("units fitted")
    axes[1].set_ylabel("CV cost ÷ fit cost")
    axes[1].set_ylim(0, max(factor) * 1.35)
    axes[1].set_title("Cross-validation overhead")

    for axis in axes:
        axis.grid(axis="y", color="#eef1f4", lw=1)
        axis.set_axisbelow(True)
    fig.suptitle("Recorded hardware timings", fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIGURES / "runtime.png", dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate nine validation figures in outputs/.")
    parser.parse_args(argv)
    try:
        load_recorded_report()
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    FIGURES.mkdir(parents=True, exist_ok=True)
    for draw in (
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
