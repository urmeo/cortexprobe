"""Check published documentation paths, anchors, data, and plot preservation."""

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
    for directory, suffix in (("docs/figures", ".png"), ("results", ".json"), ("configs", ".json")):
        originals = sorted((docs_script.ROOT / directory).glob(f"*{suffix}"))
        copies = sorted((built_site / directory).glob(f"*{suffix}"))
        assert [path.name for path in copies] == [path.name for path in originals]
        for original, copied in zip(originals, copies):
            assert copied.read_bytes() == original.read_bytes()
    docs_script.validate_links(built_site)


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


@pytest.mark.parametrize(
    ("source", "target", "expected"),
    [
        ("README.md", "docs/figures/", "docs/figures/index.html"),
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
