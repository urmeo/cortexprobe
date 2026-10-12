"""Generate the two synthetic experiment animations in docs/media/.

Uses the project's NumPy/SciPy model and configs/validation.json. Install the ``viz``
extra for Matplotlib, and install FFmpeg with its libx264 encoder separately::

    python3 -m pip install -e '.[viz]'
    python3 scripts/generate_media.py

Each ten-second animation has a 1280x720 H.264 MP4, a looping 960-pixel GIF and a PNG
poster. Frames stream to FFmpeg; no frame directory is written. Playback shows retained
sequence frames and held-out predictions, not acquisition time or optimizer iterations.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from generate_validation_report import CONFIG, RECOVERY_SEED, UNCERTAINTY_TRUTH, synthesise
from matplotlib.patches import Circle

from cortexprobe.geometry import Grid
from cortexprobe.prf.fit import N_PARAMETERS, PRFFitter, _r_squared
from cortexprobe.prf.model import GaussianReceptiveField, predict
from cortexprobe.prf.validation import LeaveOneGroupOut
from cortexprobe.stimuli import build_apertures

OUTPUT = Path(__file__).resolve().parent.parent / "docs" / "media"
WIDTH, HEIGHT = 1280, 720
DPI = 100
FPS = 16
GIF_WIDTH = 960
SECONDS = 10
FRAME_COUNT = FPS * SECONDS
NOISE = 0.2
INK = "#172b43"
BLUE = "#246b9e"
ORANGE = "#b35a1c"
GRAY = "#7c8997"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.edgecolor": "#c9d4df",
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def encode_media(name, figure, draw_frame, poster_frame, output_dir, ffmpeg):
    """Publish the MP4, looping GIF and poster only after both encoders succeed."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=f".{name}-", dir=output_dir) as temporary:
            staging = Path(temporary)
            video = staging / f"{name}.mp4"
            command = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{WIDTH}x{HEIGHT}",
                "-framerate",
                str(FPS),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "22",
                "-threads",
                "2",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(video),
            ]
            process = subprocess.Popen(command, stdin=subprocess.PIPE)
            try:
                for frame in range(FRAME_COUNT):
                    draw_frame(frame)
                    figure.canvas.draw()
                    if frame == poster_frame:
                        figure.savefig(staging / f"{name}.png", dpi=DPI)
                    process.stdin.write(np.asarray(figure.canvas.buffer_rgba())[:, :, :3].tobytes())
                process.stdin.close()
                if process.wait():
                    raise RuntimeError(f"FFmpeg could not encode {output_dir / video.name}")
            finally:
                if process.poll() is None:
                    process.kill()
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass
                finally:
                    process.wait()
            subprocess.run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(video),
                    "-filter_complex_threads",
                    "1",
                    "-filter_complex",
                    f"fps=8,scale={GIF_WIDTH}:-1:flags=lanczos,split[a][b];"
                    "[a]palettegen=max_colors=96:stats_mode=diff[p];"
                    "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
                    "-loop",
                    "0",
                    str(staging / f"{name}.gif"),
                ],
                check=True,
            )
            for extension in ("mp4", "gif", "png"):
                path = staging / f"{name}.{extension}"
                path.replace(output_dir / path.name)
    finally:
        plt.close(figure)
    print(f"wrote {name}.mp4, .gif and .png", flush=True)


def aperture_response(output_dir, ffmpeg):
    """Compare actual retained aperture sequences with their clean model responses."""
    grid = Grid(CONFIG.stimulus.resolution)
    sequences = [build_apertures(CONFIG.stimulus, kind) for kind in ("bar", "wedge", "ring")]
    clean_responses = [
        synthesise(grid, sequence.as_float(), UNCERTAINTY_TRUTH) for sequence in sequences
    ]
    figure, axes = plt.subplots(
        2, 3, figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI, gridspec_kw={"height_ratios": [1.5, 1]}
    )
    figure.subplots_adjust(left=0.065, right=0.98, top=0.80, bottom=0.13, wspace=0.34, hspace=0.43)
    figure.text(
        0.5,
        0.955,
        "Moving apertures and Gaussian responses",
        ha="center",
        fontsize=22,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.912,
        f"Known center ({UNCERTAINTY_TRUTH[0]:g}, {UNCERTAINTY_TRUTH[1]:g}) px · "
        f"σ = {UNCERTAINTY_TRUTH[2]:g} px · amplitude = 3 · baseline = 0.5 · no noise",
        ha="center",
        fontsize=12.5,
    )
    figure.text(
        0.5,
        0.868,
        "Solid fill: exposed pixels · dashed circle and +: known Gaussian center and 1σ width",
        ha="center",
        fontsize=12,
        color=GRAY,
    )
    status = figure.text(0.5, 0.035, "", ha="center", fontsize=12)
    images, cursors, markers = [], [], []
    for column, (sequence, clean) in enumerate(zip(sequences, clean_responses)):
        mask_axis, curve_axis = axes[:, column]
        image = mask_axis.imshow(
            np.ma.array(sequence.apertures[0], mask=~grid.field_mask),
            cmap="Blues",
            vmin=0,
            vmax=1,
            interpolation="nearest",
            extent=(-grid.radius, grid.radius, -grid.radius, grid.radius),
        )
        mask_axis.add_patch(
            Circle(
                UNCERTAINTY_TRUTH[:2],
                UNCERTAINTY_TRUTH[2],
                fill=False,
                edgecolor=ORANGE,
                linestyle="--",
                linewidth=2.2,
            )
        )
        mask_axis.plot(
            *UNCERTAINTY_TRUTH[:2], marker="+", color=ORANGE, markersize=9, markeredgewidth=2
        )
        mask_axis.set(
            xlabel="x (px)",
            ylabel="y (px)",
            xticks=[-grid.radius, 0, grid.radius],
            yticks=[-grid.radius, 0, grid.radius],
        )
        frames = np.arange(1, sequence.n_frames + 1)
        curve_axis.plot(frames, clean, color=BLUE, linewidth=2, label="clean response")
        cursor = curve_axis.axvline(1, color=INK, linestyle=":", linewidth=1.2)
        (marker,) = curve_axis.plot(
            [], [], marker="o", color=ORANGE, markersize=6, linestyle="none", label="current frame"
        )
        curve_axis.set(
            xlabel="retained frame",
            ylabel="response (a.u.)",
            xlim=(1, sequence.n_frames),
            ylim=(0.35, 3.75),
        )
        curve_axis.legend(frameon=False, fontsize=10, loc="upper right")
        curve_axis.grid(axis="y", color="#edf1f5")
        images.append(image)
        cursors.append(cursor)
        markers.append(marker)

    def draw_frame(frame):
        phase = frame / (FRAME_COUNT - 1)
        for column, (sequence, clean) in enumerate(zip(sequences, clean_responses)):
            index = min(int(phase * sequence.n_frames), sequence.n_frames - 1)
            images[column].set_data(np.ma.array(sequence.apertures[index], mask=~grid.field_mask))
            axes[0, column].set_title(
                f"{sequence.kind.title()} · frame {index + 1}/{sequence.n_frames} · "
                f"group {sequence.group[index]}",
                fontsize=12.5,
                pad=12,
            )
            cursors[column].set_xdata([index + 1, index + 1])
            markers[column].set_data([index + 1], [clean[index]])
        status.set_text(
            f"Sequence progress: {phase:.0%} · each design uses its own retained frames, "
            "not a shared time axis"
        )

    encode_media("aperture-response", figure, draw_frame, 27, output_dir, ffmpeg)


def grouped_validation(output_dir, ffmpeg):
    """Show fresh training-only fits and their unchanged held-out predictions."""
    grid = Grid(CONFIG.stimulus.resolution)
    sequence = build_apertures(CONFIG.stimulus, "bar")
    apertures = sequence.as_float()
    response = synthesise(grid, apertures, UNCERTAINTY_TRUTH, noise=NOISE, seed=RECOVERY_SEED)
    frames = np.arange(1, sequence.n_frames + 1)
    folds = []
    for train, test in LeaveOneGroupOut(sequence.group).splits():
        if len(train) <= N_PARAMETERS:
            continue
        fit = PRFFitter(grid, apertures[train], CONFIG.fit).fit_unit(response[train])
        field = GaussianReceptiveField(fit.x0, fit.y0, fit.sigma)
        fitted = fit.beta * predict(field.weights(grid), apertures) + fit.baseline
        folds.append((train, test, fit, fitted, _r_squared(response[test], fitted[test])))
    scores = np.array([fold[-1] for fold in folds])
    finite_scores = scores[np.isfinite(scores)]
    mean_score = np.mean(finite_scores) if len(finite_scores) else float("nan")
    score_sd = np.std(finite_scores, ddof=1) if len(finite_scores) > 1 else float("nan")
    group_labels = [str(sequence.group[test[0]]) for _, test, *_ in folds]

    figure = plt.figure(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI)
    figure.text(
        0.5,
        0.955,
        "Held-out groups: refit on training frames",
        ha="center",
        fontsize=22,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.912,
        f"Synthetic bar response · {sequence.n_frames} retained frames · {len(folds)} folds · "
        f"noise {NOISE:.0%} of clean signal SD · seed {RECOVERY_SEED}",
        ha="center",
        fontsize=12.5,
    )
    status = figure.text(0.5, 0.855, "", ha="center", fontsize=13.5, fontweight="bold")
    mask_axis = figure.add_axes((0.065, 0.365, 0.26, 0.41))
    image = mask_axis.imshow(
        np.ma.array(sequence.apertures[0], mask=~grid.field_mask),
        cmap="Blues",
        vmin=0,
        vmax=1,
        interpolation="nearest",
        extent=(-grid.radius, grid.radius, -grid.radius, grid.radius),
    )
    mask_axis.add_patch(
        Circle(
            UNCERTAINTY_TRUTH[:2],
            UNCERTAINTY_TRUTH[2],
            fill=False,
            edgecolor=ORANGE,
            linewidth=2,
            label="known: 1σ",
        )
    )
    fitted_circle = Circle(
        (0, 0), 1, fill=False, edgecolor=INK, linestyle="--", linewidth=2, label="training fit: 1σ"
    )
    mask_axis.add_patch(fitted_circle)
    mask_axis.set(
        title="Current held-out aperture",
        xlabel="x (px)",
        ylabel="y (px)",
        xticks=[-grid.radius, 0, grid.radius],
        yticks=[-grid.radius, 0, grid.radius],
    )
    mask_axis.legend(
        frameon=False,
        fontsize=10,
        ncol=2,
        columnspacing=1,
        handlelength=1.6,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
    )
    parameters = figure.text(0.065, 0.14, "", fontsize=11, va="bottom", linespacing=1.5)

    curve_axis = figure.add_axes((0.43, 0.40, 0.54, 0.375))
    (train_points,) = curve_axis.plot(
        [], [], "o", color=GRAY, markersize=3.5, label="training observations"
    )
    (test_points,) = curve_axis.plot(
        [],
        [],
        "o",
        color=ORANGE,
        markerfacecolor="white",
        markersize=5,
        label="held-out observations",
    )
    (train_curve,) = curve_axis.plot([], [], color=BLUE, linewidth=2, label="training fit")
    (test_curve,) = curve_axis.plot(
        [], [], color=ORANGE, linestyle="--", linewidth=2.2, label="held-out prediction"
    )
    cursor = curve_axis.axvline(1, color=INK, linestyle=":", linewidth=1)
    (current,) = curve_axis.plot([], [], "D", color=INK, markersize=5, label="current test frame")
    response_range = np.ptp(response)
    curve_axis.set(
        title="Observed response and predictions from this training fit",
        xlabel="retained frame",
        ylabel="response (a.u.)",
        xlim=(0.5, sequence.n_frames + 0.5),
        ylim=(response.min() - 0.1 * response_range, response.max() + 0.5 * response_range),
    )
    curve_axis.legend(frameon=False, fontsize=9.5, ncol=2, loc="upper right")
    curve_axis.grid(axis="y", color="#edf1f5")

    score_axis = figure.add_axes((0.43, 0.13, 0.54, 0.16))
    bars = score_axis.bar(group_labels, scores, color="#a9c3d7", width=0.6)
    for bar, score in zip(bars, scores):
        score_axis.text(
            bar.get_x() + bar.get_width() / 2,
            score + 0.025,
            f"{score:.3f}",
            ha="center",
            fontsize=11,
        )
    score_axis.set(
        title=f"Held-out R² by group · finite-fold mean {mean_score:.3f} · "
        f"sample SD {score_sd:.3f}",
        xlabel="held-out group (one separately trained model each)",
        ylabel="R²",
        ylim=(min(0, finite_scores.min() * 1.1), max(1.15, finite_scores.max() * 1.15)),
    )
    score_axis.grid(axis="y", color="#edf1f5")
    score_axis.set_axisbelow(True)
    figure.text(
        0.5,
        0.025,
        "Each scene uses a fresh train-only fit; "
        "all five parameters stay fixed while test frames advance.",
        ha="center",
        fontsize=11.5,
    )

    def draw_frame(frame):
        fold_index = min(frame * len(folds) // FRAME_COUNT, len(folds) - 1)
        train, test, fit, fitted, score = folds[fold_index]
        fold_start = fold_index * FRAME_COUNT // len(folds)
        fold_end = (fold_index + 1) * FRAME_COUNT // len(folds)
        test_index = min((frame - fold_start) * len(test) // (fold_end - fold_start), len(test) - 1)
        index = test[test_index]
        image.set_data(np.ma.array(sequence.apertures[index], mask=~grid.field_mask))
        fitted_circle.center = (fit.x0, fit.y0)
        fitted_circle.radius = fit.sigma
        train_points.set_data(frames[train], response[train])
        test_points.set_data(frames[test], response[test])
        train_prediction = np.full_like(response, np.nan)
        test_prediction = np.full_like(response, np.nan)
        train_prediction[train] = fitted[train]
        test_prediction[test] = fitted[test]
        train_curve.set_data(frames, train_prediction)
        test_curve.set_data(frames, test_prediction)
        cursor.set_xdata([index + 1, index + 1])
        current.set_data([index + 1], [response[index]])
        for bar_index, bar in enumerate(bars):
            bar.set_edgecolor(INK if bar_index == fold_index else "none")
            bar.set_linewidth(2.5 if bar_index == fold_index else 0)
        status.set_text(
            f"Fold {fold_index + 1}/{len(folds)} · hold out group {sequence.group[index]} · "
            f"train {len(train)} frames · test frame {test_index + 1}/{len(test)} · "
            f"held-out R² = {score:.3f}"
        )
        parameters.set_text(
            f"Training center: ({fit.x0:.2f}, {fit.y0:.2f}) px\n"
            f"σ = {fit.sigma:.2f} px · amplitude = {fit.beta:.2f}\n"
            f"baseline = {fit.baseline:.2f} · fixed on test frames"
        )

    encode_media("grouped-validation", figure, draw_frame, 105, output_dir, ffmpeg)
    print(f"fold R²: {scores.tolist()}; finite mean={mean_score:.12f}; sample SD={score_sd:.12f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate two reproducible synthetic animations.")
    parser.add_argument(
        "--only",
        choices=("aperture-response", "grouped-validation"),
        help="generate one animation instead of both",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--ffmpeg", default="ffmpeg", help="FFmpeg executable or path")
    args = parser.parse_args(argv)
    ffmpeg = shutil.which(args.ffmpeg)
    if ffmpeg is None:
        parser.error("FFmpeg executable not found; install it or pass --ffmpeg PATH")
    for name, generate in (
        ("aperture-response", aperture_response),
        ("grouped-validation", grouped_validation),
    ):
        if args.only is None or args.only == name:
            generate(args.output_dir, ffmpeg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
