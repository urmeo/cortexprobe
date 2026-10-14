# Videos

Two 10-second, silent videos use the response-curve example's configuration and
known field. They add no independent evaluation result. README GIF previews open
the videos with playback controls.

## Apertures and responses

<video controls playsinline preload="metadata" width="1280" height="720" poster="aperture-response.png" aria-label="Bar, wedge, and ring apertures with their Gaussian overlap responses">
  <source src="aperture-response.mp4" type="video/mp4">
  <a href="aperture-response.mp4">Download the aperture and response video.</a>
</video>

[Download MP4](aperture-response.mp4) · [Animated preview](aperture-response.gif) · [Still image](aperture-response.png)

Upper panels show masks with the known Gaussian's center and one-sigma outline;
lower panels show the clean response curve and value at each retained frame.
All designs use a 64 × 64 pixel grid, center `(12, 8)` pixels, and sigma `5` pixels. The response is
`3 × aperture–field overlap + 0.5`, with no added noise.

| Design | Retained frames | Validation groups |
|---|---:|---:|
| Bars | 80 | 4 |
| Wedges | 20 | 4 |
| Rings | 17 | 4 |

The ring sequence retains 17 of its 20 generated frames after overlap pruning.
Each panel reports its own frame index and group. Playback aligns sequence
progress; it does not imply shared acquisition times or equal frame counts.

## Held-out prediction

<video controls playsinline preload="metadata" width="1280" height="720" poster="grouped-validation.png" aria-label="Separate training-fold fits predict observations in each held-out group">
  <source src="grouped-validation.mp4" type="video/mp4">
  <a href="grouped-validation.mp4">Download the held-out prediction video.</a>
</video>

[Download MP4](grouped-validation.mp4) · [Animated preview](grouped-validation.gif) · [Still image](grouped-validation.png)

The 80 bar frames use the same field, amplitude, and baseline, with Gaussian
noise at 20% of the clean response's standard deviation, using seed 7.

Each of four folds fits the other three groups. Geometry, amplitude, and baseline
stay fixed during predictions on the excluded group. Frames show predictions,
not optimizer iterations or new fits. Held-out observations never enter the fit.

Scores cover this one synthetic response. The mean uses finite fold R² values,
rather than one R² over pooled predictions.
See [Pipeline](../pipeline.md) for the fitting procedure and
[Validation](../validation.md) for the recorded experiments and their limits.

## Reproduce

From the repository root, install the plotting extra and add FFmpeg with its
H.264 encoder, `libx264`, to your command path.

```sh
python -m pip install -e '.[viz]'
python scripts/generate_media.py
```

The script reads `configs/validation.json`, computes the masks and predictions
with the project code, and writes MP4, GIF, and PNG files to `docs/media/`.
FFmpeg is only needed to regenerate the media. Fitting and viewing the videos
do not require it. Encoded file bytes can differ across FFmpeg and Matplotlib
versions even when the underlying masks and predictions match.

Both clips have visible labels and manual playback controls, without autoplay.
