"""Synthetic checks."""

from __future__ import annotations

import csv
import hashlib
import platform
from dataclasses import replace
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.patches import Circle

from cortexprobe.config import FitConfig, StimulusConfig
from cortexprobe.prf.fit import PRFFitter
from cortexprobe.prf.model import GaussianReceptiveField, predict
from cortexprobe.prf.validation import CrossValidator
from cortexprobe.stimuli import build_apertures

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "results"
BLUE, ORANGE, INK = "#195f9d", "#bf571b", "#172b43"
COLORS = (BLUE, ORANGE, "#26775c", "#8057a4")


def canvas(title, subtitle, columns=1, size=(12, 5.8)):
    fig, axes = plt.subplots(1, columns, figsize=size, squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.25, top=0.76, wspace=0.35)
    fig.text(0.07, 0.96, title, fontsize=18, weight="bold", color=INK, va="top")
    fig.text(0.07, 0.88, subtitle, fontsize=11, color="#53647a", va="top")
    return fig, axes[0]


def save(fig, name, note):
    fig.text(0.07, 0.035, note, fontsize=10, color="#53647a", va="bottom")
    fig.savefig(OUTPUT / f"{name}.png", dpi=180, facecolor="white")
    plt.close(fig)


def record(rows, check, case, **values):
    rows.extend((check, case, key, value) for key, value in values.items())


def response(sequence, field, noise=0.0, seed=7):
    clean = 3.0 * predict(field.weights(sequence.grid), sequence.as_float()) + 0.5
    return clean + np.random.default_rng(seed).normal(0.0, noise * clean.std(), len(clean))


def spatial_coverage(config, rows):
    fig, axes = canvas(
        "Where each design samples the field",
        "Percentage of retained frames exposing each pixel · 64 × 64 grid",
        columns=3,
    )
    for ax, kind in zip(axes, ("bar", "wedge", "ring")):
        sequence = build_apertures(config, kind)
        coverage = 100.0 * sequence.apertures.mean(axis=0)
        inside = coverage[sequence.grid.field_mask]
        groups = len(np.unique(sequence.group))
        image = ax.imshow(
            np.ma.masked_where(~sequence.grid.field_mask, coverage),
            extent=(-32, 32, -32, 32),
            origin="upper",
            cmap="Blues",
            vmin=0,
            vmax=30,
            interpolation="nearest",
        )
        ax.set(
            title=f"{kind.title()} · {sequence.n_frames} frames · {groups} groups",
            xlabel="x (px)",
        )
        ax.set_xticks((-32, 0, 32))
        ax.set_yticks((-32, 0, 32))
        record(
            rows,
            "coverage",
            kind,
            frames=sequence.n_frames,
            groups=groups,
            min_percent=inside.min(),
            max_percent=inside.max(),
            mean_percent=inside.mean(),
        )
    axes[0].set_ylabel("y (px)")
    colorbar = fig.colorbar(
        image, cax=fig.add_axes((0.34, 0.13, 0.38, 0.025)), orientation="horizontal"
    )
    colorbar.set_label("Retained frames exposing pixel (%)")
    save(
        fig,
        "aperture-coverage",
        "Generated masks after pruning. Exposure frequency does not measure fit accuracy.",
    )


def localization(sequence, fitter, rows):
    fig, axes = canvas(
        "Recovering 16 known field centers",
        "Full-data fits · σ = 4 px · one noise draw per unit, seeds 100–115",
        columns=3,
    )
    centers = np.asarray(list(product((-18, -6, 6, 18), repeat=2)))
    errors = []
    for ax, noise, label in zip(axes[:2], (0.0, 0.2), ("Noiseless", "20% noise")):
        fits = [
            fitter.fit_unit(
                response(
                    sequence,
                    GaussianReceptiveField(float(x), float(y), 4.0),
                    noise,
                    100 + unit,
                )
            )
            for unit, (x, y) in enumerate(centers)
        ]
        estimates = np.asarray([(fit.x0, fit.y0) for fit in fits])
        error = np.linalg.norm(estimates - centers, axis=1)
        assert np.isfinite(error).all()
        assert all(fit.accepted for fit in fits)
        if noise == 0:
            assert error.max() < 1e-4
        errors.append(error)
        ax.add_patch(Circle((0, 0), 32, fill=False, color="#c4ceda", linewidth=1))
        ax.scatter(*centers.T, marker="+", s=75, color=INK, label="Known")
        ax.scatter(*estimates.T, facecolors="none", edgecolors=BLUE, s=55, label="Fitted")
        for known, fitted in zip(centers, estimates):
            ax.plot((known[0], fitted[0]), (known[1], fitted[1]), color=ORANGE)
        if noise:
            for unit, center in enumerate(centers, start=1):
                ax.annotate(
                    str(unit),
                    center,
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color=INK,
                )
        ax.set(title=label, xlabel="x (px)", xlim=(-34, 34), ylim=(-34, 34), aspect="equal")
        ax.set_xticks((-24, 0, 24))
        ax.set_yticks((-24, 0, 24))
        for unit, ((x, y), fit, distance) in enumerate(zip(centers, fits, error)):
            record(
                rows,
                "localization",
                f"noise{noise:g}-unit{unit + 1:02d}",
                seed=100 + unit,
                true_x_px=x,
                true_y_px=y,
                true_sigma_px=4,
                fit_x_px=fit.x0,
                fit_y_px=fit.y0,
                fit_sigma_px=fit.sigma,
                center_error_px=distance,
                r2=fit.r2,
                accepted=int(fit.accepted),
            )
    axes[0].set_ylabel("y (px)")
    axes[0].legend(loc="lower left", fontsize=9)
    for error, color, label in zip(errors, (BLUE, ORANGE), ("Noiseless", "20% noise")):
        axes[2].plot(np.arange(1, 17), error, "o-", color=color, label=label, markersize=4)
    axes[2].set(
        title="Center error",
        xlabel="Unit",
        ylabel="Distance from known center (px)",
        ylim=(0, max(0.1, float(np.max(errors)) * 1.15)),
    )
    axes[2].set_xticks((1, 4, 8, 12, 16))
    axes[2].legend(fontsize=9)
    save(
        fig,
        "field-localization",
        f"20% noise: mean error {errors[1].mean():.3f} px; largest {errors[1].max():.3f} px. "
        "One synthetic batch, not a population bound.",
    )


def held_out(sequence, config, observed, rows):
    validator = CrossValidator(sequence.grid, sequence.as_float(), sequence.group, config)
    validated = validator.validate_unit(observed)
    record(
        rows,
        "full_fit",
        "seed7",
        fit_x_px=validated.fit.x0,
        fit_y_px=validated.fit.y0,
        fit_sigma_px=validated.fit.sigma,
        amplitude=validated.fit.beta,
        baseline=validated.fit.baseline,
        r2=validated.fit.r2,
    )
    predictions = np.full(len(observed), np.nan)
    visits = np.zeros(len(observed), dtype=int)
    fits, scores = [], []
    for group, (train, test) in zip(validator.splitter.unique, validator.splitter.splits()):
        assert not np.intersect1d(train, test).size
        fit = PRFFitter(sequence.grid, sequence.as_float()[train], config).fit_unit(observed[train])
        predicted = (
            fit.beta
            * predict(
                GaussianReceptiveField(fit.x0, fit.y0, fit.sigma).weights(sequence.grid),
                sequence.as_float()[test],
            )
            + fit.baseline
        )
        score = 1.0 - np.sum((observed[test] - predicted) ** 2) / np.sum(
            (observed[test] - observed[test].mean()) ** 2
        )
        assert fit.converged
        assert np.isfinite(score)
        predictions[test] = predicted
        visits[test] += 1
        fits.append(fit)
        scores.append(score)
        record(
            rows,
            "fold",
            str(group),
            train_frames=len(train),
            held_out_frames=len(test),
            train_r2=fit.r2,
            held_out_r2=score,
            fit_x_px=fit.x0,
            fit_y_px=fit.y0,
            fit_sigma_px=fit.sigma,
            amplitude=fit.beta,
            baseline=fit.baseline,
        )
    assert np.all(visits == 1)
    assert validated.validation_complete
    np.testing.assert_allclose(scores, validated.fold_r2, rtol=1e-12, atol=1e-12)
    for frame, (observed_value, predicted_value, group) in enumerate(
        zip(observed, predictions, sequence.group)
    ):
        record(
            rows,
            "prediction",
            str(frame),
            group=group,
            observed=observed_value,
            held_out=predicted_value,
        )

    fig, (ax,) = canvas(
        "Predicting frames excluded from fitting",
        "Four separate training fits · 60 training / 20 held-out frames per fold",
        size=(10, 7),
    )
    fig.subplots_adjust(right=0.68)
    limits = (
        min(observed.min(), predictions.min()) - 0.1,
        max(observed.max(), predictions.max()) + 0.1,
    )
    ax.plot(limits, limits, "--", color="#9ca9b8", linewidth=1, label="Identity")
    for group, score, color, marker in zip(
        validator.splitter.unique, scores, COLORS, ("o", "s", "^", "D")
    ):
        selected = sequence.group == group
        ax.scatter(
            observed[selected],
            predictions[selected],
            s=35,
            color=color,
            marker=marker,
            label=f"Group {group} · R² {score:.3f}",
            alpha=0.85,
        )
    ax.set(
        xlabel="Observed response (a.u.)",
        ylabel="Held-out prediction (a.u.)",
        xlim=limits,
        ylim=limits,
        aspect="equal",
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.04, 0.6), frameon=False, fontsize=10)
    fig.text(
        0.72,
        0.34,
        f"Mean fold R² {validated.cv_r2:.3f}\n"
        f"Valid folds {validated.n_valid_folds}/{validated.n_total_folds}",
        fontsize=11,
        color=INK,
    )
    save(
        fig,
        "held-out-predictions",
        "Synthetic bars · field (12, 8, 5) px · 20% noise · seed 7. "
        "The summary averages fold scores.",
    )

    fig, axes = canvas(
        "How estimates change with the held-out group",
        "Each point uses 60 training frames; its 20 held-out frames are excluded",
        columns=3,
    )
    for ax, parameter, known in zip(axes, ("x0", "y0", "sigma"), (12, 8, 5)):
        ax.axhline(known, color=INK, linewidth=1, label="Known value")
        ax.axhline(
            getattr(validated.fit, parameter),
            color=ORANGE,
            linestyle="--",
            label="Full-data fit (80 frames)",
        )
        ax.scatter(
            np.arange(4),
            [getattr(fit, parameter) for fit in fits],
            color=BLUE,
            s=45,
            label="Training-fold fit (60 frames)",
        )
        ax.set(
            title={
                "x0": "Horizontal center",
                "y0": "Vertical center",
                "sigma": "Field width",
            }[parameter],
            ylabel=f"{parameter} (px)",
            xlabel="Held-out group",
            xlim=(-0.4, 3.4),
        )
        ax.set_xticks(np.arange(4), [str(group) for group in validator.splitter.unique])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.11),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    save(
        fig,
        "fold-stability",
        "Field (12, 8, 5) px · 20% noise · seed 7. Training sets overlap; "
        "these are not independent uncertainty estimates.",
    )


def solver_budget(sequence, config, observed, rows):
    budgets = (2, 5, 10, 25, 50, 100, 200)
    fits = [
        PRFFitter(sequence.grid, sequence.as_float(), replace(config, max_nfev=budget)).fit_unit(
            observed
        )
        for budget in budgets
    ]
    assert all(np.isfinite([fit.x0, fit.y0, fit.sigma, fit.r2]).all() for fit in fits)
    assert any(not fit.converged for fit in fits)
    assert any(fit.converged for fit in fits)
    values = (
        [fit.r2 for fit in fits],
        [np.hypot(fit.x0 - 12, fit.y0 - 8) for fit in fits],
        [fit.n_fev for fit in fits],
    )
    fig, axes = canvas(
        "Solver budget and convergence",
        "One fixed noisy response · field (12, 8, 5) px · 20% noise · seed 7",
        columns=3,
    )
    for ax, values_for_axis, label in zip(
        axes, values, ("In-sample R²", "Center error (px)", "Solver-reported n_fev")
    ):
        ax.plot(budgets, values_for_axis, color="#b5c0cc", linewidth=1)
        for budget, value, fit in zip(budgets, values_for_axis, fits):
            ax.scatter(
                budget,
                value,
                color=BLUE if fit.converged else ORANGE,
                marker="o" if fit.converged else "x",
                s=50,
                zorder=3,
            )
        ax.set(xscale="log", xlabel="max_nfev", ylabel=label)
        ax.set_xticks(budgets, [str(budget) for budget in budgets])
        ax.minorticks_off()
    axes[0].scatter([], [], color=BLUE, label="Converged", s=40)
    axes[0].scatter([], [], color=ORANGE, marker="x", label="Not converged", s=40)
    axes[0].legend(fontsize=9)
    for budget, fit, error in zip(budgets, fits, values[1]):
        assert fit.n_fev <= budget
        record(
            rows,
            "solver",
            str(budget),
            max_nfev=budget,
            n_fev=fit.n_fev,
            r2=fit.r2,
            center_error_px=error,
            converged=int(fit.converged),
            accepted=int(fit.accepted),
        )
    save(
        fig,
        "solver-budget",
        "Convergence is required for acceptance. n_fev is the solver's reported count, "
        "not runtime or total computation.",
    )


def main():
    OUTPUT.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": INK,
            "text.color": INK,
            "axes.titlepad": 12,
        }
    )
    stimulus = StimulusConfig(resolution=64, n_steps=20, directions=(0, 45, 90, 135))
    config = FitConfig(grid_size=10, sigma_bounds=(1.0, 10.0))
    sequence = build_apertures(stimulus)
    fitter = PRFFitter(sequence.grid, sequence.as_float(), config)
    assert sequence.n_frames == 80
    assert len(fitter.candidates) == 300
    rows = []
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src/cortexprobe").rglob("*.py")):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    record(
        rows,
        "metadata",
        "source",
        source_sha256=digest.hexdigest(),
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        date_utc=datetime.now(timezone.utc).date().isoformat(),
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        matplotlib=matplotlib.__version__,
    )
    record(
        rows,
        "metadata",
        "setup",
        bar_frames=sequence.n_frames,
        groups=len(np.unique(sequence.group)),
        candidates=len(fitter.candidates),
        amplitude=3,
        baseline=0.5,
        noise_fraction=0.2,
        noise_scaling="clean-response population SD (ddof=0)",
        noise_distribution="independent Gaussian",
        prediction_seed=7,
        prediction_x_px=12,
        prediction_y_px=8,
        prediction_sigma_px=5,
    )
    record(rows, "metadata", "stimulus", **stimulus.to_dict())
    record(rows, "metadata", "fit", **config.to_dict())
    spatial_coverage(stimulus, rows)
    localization(sequence, fitter, rows)
    observed = response(sequence, GaussianReceptiveField(12.0, 8.0, 5.0), 0.2, 7)
    held_out(sequence, config, observed, rows)
    solver_budget(sequence, config, observed, rows)
    with (OUTPUT / "measurements.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("check", "case", "metric", "value"))
        writer.writerows(rows)
    print("Saved 5 figures and measurements.csv in results/.")


if __name__ == "__main__":
    main()
