"""Check published documentation paths, anchors, data, plots, and media."""

from __future__ import annotations

from pathlib import Path

import pytest

from .conftest import load_script


@pytest.fixture(scope="module")
def docs_script():
    return load_script("build_docs.py")


@pytest.fixture
def built_site(docs_script, monkeypatch, tmp_path):
    output = tmp_path / "_site"
    output.mkdir()
    (output / "stale.html").write_text("Previous generated page")
    monkeypatch.setattr(docs_script, "OUTPUT", output)
    docs_script.build_site()
    return output


def test_site_builds_every_page_and_copies_assets_unchanged(docs_script, built_site):
    assert not (built_site / "stale.html").exists()
    assert (built_site / ".nojekyll").is_file()
    for destination, _ in docs_script.PAGES.values():
        assert (built_site / destination).is_file()
    for directory, suffix in (
        ("docs/figures", ".png"),
        ("docs/figures", ".svg"),
        ("docs/media", ".gif"),
        ("docs/media", ".mp4"),
        ("docs/media", ".png"),
        ("results", ".json"),
        ("configs", ".json"),
    ):
        originals = sorted((docs_script.ROOT / directory).glob(f"*{suffix}"))
        copies = sorted((built_site / directory).glob(f"*{suffix}"))
        assert originals
        assert [path.name for path in copies] == [path.name for path in originals]
        for original, copied in zip(originals, copies):
            assert copied.read_bytes() == original.read_bytes()
    for original in (docs_script.ROOT / "docs/assets").rglob("*"):
        if original.is_file():
            copied = built_site / "assets" / original.relative_to(docs_script.ROOT / "docs/assets")
            assert copied.read_bytes() == original.read_bytes()
    docs_script.validate_links(built_site)


@pytest.mark.parametrize("page", ["index.html", "docs/pipeline.html"])
def test_pipeline_is_readable_html_with_local_figma_assets(docs_script, built_site, page):
    html = (built_site / page).read_text()
    assert html.count('class="pipeline-view"') == 1
    assert 'aria-labelledby="pipeline-title"' in html
    assert "Full-data estimate" in html
    assert "Training-fold refits" in html
    assert "Freeze all five parameters." in html
    assert "The all-frame fit does not initialize the fold models." in html
    assert "Synthetic only: no human-data or neural-network evaluation." in html
    assert "200 trials/condition" in html
    assert "Noise is relative to clean-response SD." in html
    assert 'src="docs/figures/pipeline.png"' not in html
    assert 'src="figures/pipeline.png"' not in html
    assert "figma.com/api/mcp/asset/" not in html
    parser = docs_script.PageLinks()
    parser.feed(html)
    for asset in (built_site / "assets/pipeline").glob("*.svg"):
        url = docs_script.relative_url(asset.relative_to(built_site), Path(page))
        assert url in parser.links
    stylesheet = docs_script.relative_url(Path("assets/pipeline.css"), Path(page))
    assert stylesheet in parser.links


def test_validation_preserves_tables_provenance_and_links(built_site):
    html = (built_site / "docs/validation.html").read_text()
    assert '<a href="../results/validation.json">' in html
    assert '<a href="../configs/validation.json">' in html
    assert 'href="../CONTRIBUTING.html"' in html
    assert 'href="../assets/site.css"' in html
    assert "<table>" in html
    assert "0.935" in html
    assert "4d6a856a499e" in html
    assert "generated 2026-08-19" in html
    assert 'href="#reproduce-the-report"' in html


def test_gallery_displays_all_plots_with_accessible_descriptions(built_site):
    html = (built_site / "docs/figures/index.html").read_text()
    assert html.count('<figure class="figure">') == 12
    assert 'src="noise-autocorrelation.png"' in html
    assert 'href="../validation.html"' in html
    assert 'href="../../index.html"' in html
    assert html.count('loading="eager"') == 1
    assert html.count('loading="lazy"') == 11
    assert html.count('alt="') == 12
    assert html.count('width="') == 12
    assert html.count('height="') == 12


def test_media_page_keeps_local_video_sources_posters_and_compact_navigation(built_site):
    html = (built_site / "docs/media/index.html").read_text()
    assert html.count("<video ") == 2
    for name in ("aperture-response", "grouped-validation"):
        assert f'poster="{name}.png"' in html
        assert f'src="{name}.mp4"' in html
        assert (built_site / f"docs/media/{name}.gif").is_file()
    navigation = html.split('<nav class="primary-nav"', 1)[1].split("</nav>", 1)[0]
    assert navigation.count("<a ") == 5
    assert ">Videos</a>" not in navigation


@pytest.mark.parametrize(
    "header",
    [
        b"GIF87a\x40\x01\xf0\x00",
        b"GIF89a\x40\x01\xf0\x00",
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x01\x40\x00\x00\x00\xf0",
    ],
)
def test_image_dimensions_read_png_and_both_gif_versions(docs_script, tmp_path, header):
    image = tmp_path / "preview"
    image.write_bytes(header)
    assert docs_script.image_dimensions(image) == (320, 240)


@pytest.mark.parametrize(
    ("header", "error"),
    [
        (b"", "invalid PNG/GIF header"),
        (b"GIF89a", "invalid PNG/GIF header"),
        (b"GIF90a\x40\x01\xf0\x00", "invalid PNG/GIF header"),
        (b"not a PNG or GIF file!!!!", "invalid PNG/GIF header"),
        (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR", "invalid PNG/GIF header"),
        (b"GIF89a\x00\x00\xf0\x00", "dimensions must be positive"),
        (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x01\x40\x00\x00\x00\x00",
            "dimensions must be positive",
        ),
    ],
)
def test_invalid_image_headers_fail_clearly(docs_script, tmp_path, header, error):
    image = tmp_path / "preview"
    image.write_bytes(header)
    with pytest.raises(ValueError, match=error):
        docs_script.image_dimensions(image)


def test_markdown_gif_gets_dimensions_and_keeps_query_string(docs_script, monkeypatch, tmp_path):
    (tmp_path / "README.md").write_text("# Preview\n\n![Moving aperture](preview.gif?version=1)\n")
    (tmp_path / "preview.gif").write_bytes(b"GIF89a\x40\x01\xf0\x00")
    monkeypatch.setattr(docs_script, "ROOT", tmp_path)
    html = docs_script.render_page(Path("README.md"))
    assert 'src="preview.gif?version=1"' in html
    assert 'width="320"' in html
    assert 'height="240"' in html
    assert 'loading="eager"' in html
    assert 'alt="Moving aperture"' in html


def test_remote_images_do_not_require_a_local_file(docs_script, monkeypatch, tmp_path):
    (tmp_path / "README.md").write_text("![Preview](https://example.com/preview.gif)\n")
    monkeypatch.setattr(docs_script, "ROOT", tmp_path)
    html = docs_script.render_page(Path("README.md"))
    assert 'src="https://example.com/preview.gif"' in html
    assert 'width="' not in html


@pytest.mark.parametrize(
    ("source", "target", "expected"),
    [
        ("README.md", "docs/figures/", "docs/figures/index.html"),
        ("README.md", "docs/media/", "docs/media/index.html"),
        (
            "docs/usage.md",
            "validation.md#method-and-constraints",
            "validation.html#method-and-constraints",
        ),
        ("docs/validation.md", "../results/validation.json", "../results/validation.json"),
        ("README.md", "https://example.com/page.md", "https://example.com/page.md"),
        ("README.md", "#run", "#run"),
    ],
)
def test_repository_links_survive_html_conversion(docs_script, source, target, expected):
    assert docs_script.SiteContent(Path(source)).rewrite_url(target) == expected


@pytest.mark.parametrize("target", ["missing.png", "#missing", "../outside.html"])
def test_broken_assets_anchors_and_paths_fail_the_build(docs_script, tmp_path, target):
    (tmp_path / "index.html").write_text(f'<a href="{target}">Broken</a>')
    with pytest.raises(ValueError, match="missing"):
        docs_script.validate_links(tmp_path)


def test_duplicate_page_ids_fail_the_build(docs_script, tmp_path):
    (tmp_path / "index.html").write_text('<h1 id="intro">Title</h1><h2 id="intro">Section</h2>')
    with pytest.raises(ValueError, match=r"index\.html: duplicate HTML ids.*intro"):
        docs_script.validate_links(tmp_path)


@pytest.mark.parametrize("missing", [None, "preview.png", "clip.mp4"])
def test_raw_html_video_sources_and_posters_are_validated(docs_script, tmp_path, missing):
    media = tmp_path / "docs/media"
    media.mkdir(parents=True)
    (media / "index.html").write_text(
        '<video controls preload="none" poster="preview.png">'
        '<source src="clip.mp4" type="video/mp4"></video>'
    )
    for name in ("preview.png", "clip.mp4"):
        if name != missing:
            (media / name).touch()
    if missing is None:
        docs_script.validate_links(tmp_path)
    else:
        with pytest.raises(ValueError, match=missing):
            docs_script.validate_links(tmp_path)
