import pytest

from conftest import load_script


@pytest.fixture(scope="module")
def docs_script():
    return load_script("scripts/build_docs.py")


@pytest.fixture
def built_site(docs_script, monkeypatch, tmp_path):
    output = tmp_path / "_site"
    output.mkdir()
    (output / "stale.html").write_text("Previous generated page")
    monkeypatch.setattr(docs_script, "OUTPUT", output)
    docs_script.build_site()
    return output


def test_documentation_build_preserves_assets_links_pipeline_and_results(docs_script, built_site):
    html = (built_site / "index.html").read_text()
    assert not (built_site / "stale.html").exists()
    assert not (built_site / "docs/pipeline.html").exists()
    assert (built_site / ".nojekyll").is_file()
    assert not list(built_site.rglob("*.json"))
    assert list(built_site.rglob("*.html")) == [built_site / "index.html"]
    for directory, suffix in (
        ("docs/figures", ".png"),
        ("docs/figures", ".svg"),
        ("docs/media", ".gif"),
        ("docs/media", ".mp4"),
        ("docs/media", ".png"),
    ):
        originals = sorted((docs_script.ROOT / directory).glob(f"*{suffix}"))
        copies = sorted((built_site / directory).glob(f"*{suffix}"))
        assert originals
        assert [path.name for path in copies] == [path.name for path in originals]
        assert all(
            original.read_bytes() == copied.read_bytes()
            for original, copied in zip(originals, copies)
        )
    for name in (
        "aperture-coverage",
        "field-localization",
        "held-out-predictions",
        "fold-stability",
        "solver-budget",
    ):
        relative = f"results/{name}.png"
        assert f'src="{relative}"' in html
        assert (built_site / relative).read_bytes() == (docs_script.ROOT / relative).read_bytes()
    assert len(list((built_site / "results").glob("*.png"))) == 5
    assert 'href="results/measurements.csv"' in html
    assert (built_site / "results/measurements.csv").read_bytes() == (
        docs_script.ROOT / "results/measurements.csv"
    ).read_bytes()
    for original in (docs_script.ROOT / "docs/assets").rglob("*"):
        if original.is_file():
            copied = built_site / "assets" / original.relative_to(docs_script.ROOT / "docs/assets")
            assert copied.read_bytes() == original.read_bytes()
    image = "docs/figures/pipeline.png"
    width, height = docs_script.image_dimensions(docs_script.ROOT / image)
    assert html.count(f'src="{image}"') == 1
    assert f'<a href="{image}"><img ' in html
    pipeline = html.split(f'<a href="{image}">', 1)[1].split("</a>", 1)[0]
    for attribute in (
        f'width="{width}"',
        f'height="{height}"',
        'loading="eager"',
        'alt="Matched masks, responses, and groups feed full-data fitting '
        'and separate training fits for held-out validation."',
    ):
        assert attribute in pipeline
    for target in (
        "docs/figures/pipeline.svg",
        "https://www.figma.com/design/SLnekFxRy403s5HqmehcOz?node-id=9-2",
        "https://github.com/urmeo/cortexprobe/blob/main/scripts/fit_synthetic.py",
        "assets/site.css",
    ):
        assert f'href="{target}"' in html
    assert "figma.com/api/mcp/asset/" not in html
    assert "Original experiment and media generators are not included" in html
    for value in ("<table>", "0.935", "4d6a856a499e", "generated 2026-08-19"):
        assert value in html
    assert html.count("<video ") == 2
    for name in ("aperture-response", "grouped-validation"):
        assert f'poster="docs/media/{name}.png"' in html
        assert f'src="docs/media/{name}.mp4"' in html
        assert (built_site / f"docs/media/{name}.gif").is_file()
    navigation = html.split('<nav class="primary-nav"', 1)[1].split("</nav>", 1)[0]
    assert navigation.count("<a ") == 3
    assert all(f'href="#{section}"' in navigation for section in ("run", "results", "references"))
    docs_script.validate_links(built_site)


@pytest.mark.parametrize(
    ("header", "error"),
    [
        (b"GIF87a\x40\x01\xf0\x00", None),
        (b"GIF89a\x40\x01\xf0\x00", None),
        (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x01\x40\x00\x00\x00\xf0", None),
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
def test_image_header_dimensions(docs_script, tmp_path, header, error):
    image = tmp_path / "preview"
    image.write_bytes(header)
    if error:
        with pytest.raises(ValueError, match=error):
            docs_script.image_dimensions(image)
    else:
        assert docs_script.image_dimensions(image) == (320, 240)


@pytest.mark.parametrize("remote", [False, True])
def test_markdown_images_keep_urls_and_accessible_metadata(
    docs_script, monkeypatch, tmp_path, remote
):
    url = "https://example.com/preview.gif" if remote else "preview.gif?version=1"
    (tmp_path / "README.md").write_text(f"![Moving aperture]({url})\n")
    if not remote:
        (tmp_path / "preview.gif").write_bytes(b"GIF89a\x40\x01\xf0\x00")
    monkeypatch.setattr(docs_script, "ROOT", tmp_path)
    html = docs_script.render_page()
    for attribute in (f'src="{url}"', 'loading="eager"', 'alt="Moving aperture"'):
        assert attribute in html
    if remote:
        assert 'width="' not in html
    else:
        assert 'width="320"' in html
        assert 'height="240"' in html


def test_collapsed_details_render_markdown_tables_and_code(docs_script, monkeypatch, tmp_path):
    (tmp_path / "README.md").write_text(
        '<details markdown="1"><summary>Recorded results</summary>\n\n'
        "| Noise | R² |\n|---|---|\n| 20% | 0.9 |\n\n"
        "```python\nprint('result')\n```\n\n"
        "[Run](README.md#run)\n\n</details>\n"
    )
    monkeypatch.setattr(docs_script, "ROOT", tmp_path)
    html = docs_script.render_page()
    assert "<details>" in html
    assert "<summary>Recorded results</summary>" in html
    assert '<div aria-label="Scrollable table"' in html
    assert "<td>20%</td>" in html
    assert '<pre tabindex="0"><code class="language-python">' in html
    assert 'href="index.html#run"' in html
    assert 'markdown="1"' not in html


@pytest.mark.parametrize("missing", [None, "preview.png", "preview.mp4"])
def test_linked_animation_uses_video_only_with_source_and_poster(
    docs_script, monkeypatch, tmp_path, missing
):
    (tmp_path / "README.md").write_text(
        "[![Moving aperture](preview.gif)](preview.mp4?version=2)\n"
    )
    (tmp_path / "preview.gif").write_bytes(b"GIF89a\x40\x01\xf0\x00")
    if missing != "preview.png":
        (tmp_path / "preview.png").write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x05\x00\x00\x00\x02\xd0"
        )
    if missing != "preview.mp4":
        (tmp_path / "preview.mp4").touch()
    monkeypatch.setattr(docs_script, "ROOT", tmp_path)
    html = docs_script.render_page()
    if missing is None:
        assert '<video aria-label="Moving aperture"' in html
        assert 'poster="preview.png"' in html
        assert 'src="preview.mp4?version=2"' in html
        assert 'width="1280"' in html
        assert 'height="720"' in html
        assert "<img " not in html
    else:
        assert "<video " not in html
        assert 'src="preview.gif"' in html


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("README.md#run", "index.html#run"),
        ("LICENSE", "LICENSE"),
        ("docs/media/clip.mp4?version=2", "docs/media/clip.mp4?version=2"),
        ("https://example.com/page.md", "https://example.com/page.md"),
        ("#run", "#run"),
    ],
)
def test_repository_links_survive_html_conversion(docs_script, target, expected):
    content = docs_script.SiteContent()
    assert content.rewrite_url(target) == expected
    assert (
        content.rewrite_url("docs/figures/pipeline.png?v=2#preview")
        == "docs/figures/pipeline.png?v=2#preview"
    )
    figma = "https://www.figma.com/design/SLnekFxRy403s5HqmehcOz?node-id=9-2"
    assert content.rewrite_url(figma) == figma
    with pytest.raises(ValueError, match="not in the subpath"):
        content.rewrite_url("../outside.png")


@pytest.mark.parametrize("target", ["missing.png", "#missing", "../outside.html", None])
def test_broken_links_and_duplicate_ids_fail_the_build(docs_script, tmp_path, target):
    markup = (
        f'<a href="{target}">Broken</a>'
        if target
        else '<h1 id="intro">Title</h1><h2 id="intro">Section</h2>'
    )
    (tmp_path / "index.html").write_text(markup)
    with pytest.raises(
        ValueError, match="missing" if target else r"index\.html: duplicate HTML ids.*intro"
    ):
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


@pytest.fixture
def results_site(docs_script, monkeypatch, tmp_path):
    for directory in ("docs/assets", "docs/figures", "docs/media", "results"):
        (tmp_path / directory).mkdir(parents=True)
    (tmp_path / "LICENSE").write_text("MIT")
    (tmp_path / "docs/assets/site.css").touch()
    (tmp_path / "README.md").write_text(
        "# Run\n\n# Results\n\n# References\n\n"
        "![Check](results/linked.png?version=1)\n\n"
        "[Data](results/measurements.csv)\n"
    )
    image = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x01\x40\x00\x00\x00\xf0"
    for name in ("linked.png", "unlinked.png"):
        (tmp_path / "results" / name).write_bytes(image)
    for name in ("measurements.csv", "private.csv", "script.py"):
        (tmp_path / "results" / name).write_text("value\n1\n")
    monkeypatch.setattr(docs_script, "ROOT", tmp_path)
    monkeypatch.setattr(docs_script, "OUTPUT", tmp_path / "_site")
    return tmp_path


def test_results_publication_copies_only_linked_images_and_data(docs_script, results_site, capsys):
    docs_script.build_site()
    output = results_site / "_site/results"
    assert {path.name for path in output.iterdir()} == {"linked.png", "measurements.csv"}
    for path in output.iterdir():
        assert path.read_bytes() == (results_site / "results" / path.name).read_bytes()
    assert "1 figures" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("reference", "error"),
    [
        ("results/leak.csv", "not in the subpath"),
        ("results/script.py", "missing local target"),
        ("results/alias/../../measurements.csv", "not in the subpath"),
    ],
)
def test_results_publication_rejects_private_or_unsupported_files(
    docs_script, results_site, reference, error
):
    (results_site / "private.csv").write_text("private")
    (results_site / "results/leak.csv").symlink_to(results_site / "private.csv")
    nested = results_site / "results/deep/nested"
    nested.mkdir(parents=True)
    (results_site / "results/alias").symlink_to(nested, target_is_directory=True)
    document = results_site / "README.md"
    document.write_text(document.read_text() + f'\n<a href="{reference}">Data</a>\n')
    with pytest.raises(ValueError, match=error):
        docs_script.build_site()
    assert not (results_site / "_site" / reference).exists()
