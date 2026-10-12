# Videos

These two 10-second, silent videos show the synthetic workflow. The README uses
animated GIF previews that open videos with playback controls. Both clips use the
same configuration and known field as the response-curve example; they add no
independent evaluation result.

## Apertures and responses

<video controls playsinline preload="metadata" width="1280" height="720" poster="aperture-response.png" aria-label="Bar, wedge, and ring apertures with their Gaussian overlap responses">
  <source src="aperture-response.mp4" type="video/mp4">
  <a href="aperture-response.mp4">Download the aperture and response video.</a>
</video>

[Download MP4](aperture-response.mp4) · [Animated preview](aperture-response.gif) · [Still image](aperture-response.png)

The upper panels show aperture masks with the known Gaussian's center and
one-sigma outline. The lower panels show the clean response and its value at the
current retained frame.
All three designs use a 64 × 64 pixel grid and the same field: center `(12, 8)`
pixels, sigma `5` pixels. The response is `3 × aperture–field overlap + 0.5`,
with no added noise.

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

This demonstration uses the 80 bar frames and the same known field, amplitude,
and baseline as the first video. Gaussian noise has standard deviation equal to
20% of the clean response's standard deviation, using seed 7.

Each of four folds fits only the other three groups. Geometry, amplitude, and
baseline stay fixed while the video steps through predictions on the excluded group.
The animation does not show optimizer iterations or repeated fits to each new
test frame. Displaying held-out observations does not feed them to the fit.

The fold scores summarize this one synthetic response. Their mean uses finite
fold R² values, rather than one R² over pooled predictions.
See [Pipeline](../pipeline.md) for the fitting procedure and
[Validation](../validation.md) for the recorded experiments and their limits.

## Reproduce

From the repository root, install the plotting extra and make FFmpeg available
on your command path. FFmpeg needs its H.264 encoder, `libx264`.

```sh
python -m pip install -e '.[viz]'
python scripts/generate_media.py
```

The script reads `configs/validation.json`, computes the masks and predictions
with the project code, and writes MP4, GIF, and PNG files to `docs/media/`.
FFmpeg is only needed to regenerate the media. Fitting and viewing the videos
do not require it. Encoded file bytes can differ across FFmpeg and Matplotlib
versions even when the underlying masks and predictions match.

Both clips contain visible labels and have no audio. The videos use manual
playback controls and do not autoplay. GIF previews animate in the README.
