"""Media generation preserves existing assets when rendering or encoding fails."""

from __future__ import annotations

import subprocess
import sys

import pytest

from .conftest import load_script


@pytest.fixture
def media_script(monkeypatch):
    script = load_script("generate_media.py")
    for name, value in (("WIDTH", 4), ("HEIGHT", 4), ("DPI", 10), ("FRAME_COUNT", 2)):
        monkeypatch.setattr(script, name, value)
    return script


@pytest.fixture
def figure(media_script):
    figure = media_script.plt.figure(figsize=(0.4, 0.4), dpi=10)
    yield figure
    media_script.plt.close(figure)


@pytest.fixture
def media_output(tmp_path):
    output_dir = tmp_path / "media"
    output_dir.mkdir()
    for extension in ("mp4", "gif", "png"):
        (output_dir / f"example.{extension}").write_bytes(f"existing {extension}".encode())
    return output_dir


@pytest.fixture
def encoder(tmp_path):
    """Consume the streamed input and write tiny outputs without requiring FFmpeg."""
    executable = tmp_path / "encoder"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "destination = Path(sys.argv[-1])\n"
        "if destination.suffix == '.mp4':\n"
        "    sys.stdin.buffer.read()\n"
        "failed = os.environ.get('ENCODER_FAIL') == destination.suffix[1:]\n"
        "destination.write_bytes(b'partial' if failed else b'new ' + destination.suffix.encode())\n"
        "sys.exit(1 if failed else 0)\n"
    )
    executable.chmod(0o700)
    return str(executable)


@pytest.fixture
def encoder_processes(media_script, monkeypatch):
    processes = []
    popen = media_script.subprocess.Popen

    def record_process(*args, **kwargs):
        process = popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(media_script.subprocess, "Popen", record_process)
    return processes


def saved_files(output_dir):
    return {path.name: path.read_bytes() for path in output_dir.iterdir()}


def assert_resources_closed(media_script, figure, processes):
    assert not media_script.plt.fignum_exists(figure.number)
    assert all(process.poll() is not None for process in processes)
    assert all(process.stdin is None or process.stdin.closed for process in processes)


@pytest.mark.parametrize("stage", ["mp4", "gif"])
def test_failed_encoder_preserves_existing_assets_and_removes_staging(
    media_script, figure, media_output, encoder, encoder_processes, monkeypatch, stage
):
    previous = saved_files(media_output)
    monkeypatch.setenv("ENCODER_FAIL", stage)
    error = RuntimeError if stage == "mp4" else subprocess.CalledProcessError

    with pytest.raises(error):
        media_script.encode_media("example", figure, lambda frame: None, 0, media_output, encoder)

    assert saved_files(media_output) == previous
    assert_resources_closed(media_script, figure, encoder_processes)


def test_failed_encoder_start_preserves_assets_and_closes_figure(
    media_script, figure, media_output, tmp_path, encoder_processes
):
    previous = saved_files(media_output)

    with pytest.raises(FileNotFoundError):
        media_script.encode_media(
            "example", figure, lambda frame: None, 0, media_output, str(tmp_path / "missing")
        )

    assert saved_files(media_output) == previous
    assert_resources_closed(media_script, figure, encoder_processes)


def test_rendering_failure_preserves_assets_and_terminates_encoder(
    media_script, figure, media_output, encoder, encoder_processes
):
    previous = saved_files(media_output)

    def draw_frame(frame):
        if frame == 1:
            raise ValueError("frame rendering failed")

    with pytest.raises(ValueError, match="frame rendering failed"):
        media_script.encode_media("example", figure, draw_frame, 0, media_output, encoder)

    assert saved_files(media_output) == previous
    assert len(encoder_processes) == 1
    assert_resources_closed(media_script, figure, encoder_processes)


def test_successful_encoding_replaces_all_assets_and_removes_staging(
    media_script, figure, media_output, encoder, encoder_processes
):
    media_script.encode_media("example", figure, lambda frame: None, 0, media_output, encoder)

    published = saved_files(media_output)
    assert set(published) == {"example.mp4", "example.gif", "example.png"}
    assert published["example.mp4"] == b"new .mp4"
    assert published["example.gif"] == b"new .gif"
    assert published["example.png"].startswith(b"\x89PNG\r\n\x1a\n")
    assert len(encoder_processes) == 2
    assert_resources_closed(media_script, figure, encoder_processes)
