"""Render the committed Markdown, figures, and videos as a static documentation site."""

from __future__ import annotations

import os
import shutil
import struct
import xml.etree.ElementTree as ET
from html import escape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit

from markdown import Markdown
from markdown.treeprocessors import Treeprocessor

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "_site"
REPOSITORY = "https://github.com/urmeo/cortexprobe"
PAGES = {
    Path("README.md"): (Path("index.html"), "Overview"),
    Path("docs/pipeline.md"): (Path("docs/pipeline.html"), "Pipeline"),
    Path("docs/usage.md"): (Path("docs/usage.html"), "Usage"),
    Path("docs/validation.md"): (Path("docs/validation.html"), "Validation"),
    Path("docs/figures/README.md"): (Path("docs/figures/index.html"), "Figures"),
    Path("docs/media/README.md"): (Path("docs/media/index.html"), "Videos"),
}


def relative_url(target: Path, page: Path) -> str:
    return Path(os.path.relpath(target, page.parent)).as_posix()


def image_dimensions(path: Path) -> tuple[int, int]:
    """Read PNG or GIF dimensions without loading the full image or animation."""
    with path.open("rb") as image:
        header = image.read(24)
    if (
        len(header) == 24
        and header[:8] == b"\x89PNG\r\n\x1a\n"
        and header[8:16] == b"\x00\x00\x00\rIHDR"
    ):
        width, height = struct.unpack(">II", header[16:24])
    elif len(header) >= 10 and header[:6] in (b"GIF87a", b"GIF89a"):
        width, height = struct.unpack("<HH", header[6:10])
    else:
        raise ValueError(f"Unsupported or invalid PNG/GIF header: {path}")
    if width == 0 or height == 0:
        raise ValueError(f"Image dimensions must be positive: {path}")
    return width, height


class SiteContent(Treeprocessor):
    """Keep repository links valid after Markdown pages become HTML."""

    def __init__(self, source: Path) -> None:
        super().__init__()
        self.source = source
        self.page = PAGES[source][0]

    def rewrite_url(self, value: str, base: Path | None = None) -> str:
        url = urlsplit(value)
        if url.scheme or url.netloc or not url.path:
            return value
        base = ROOT / self.source.parent if base is None else base
        target = (base / unquote(url.path)).resolve().relative_to(ROOT)
        if (target / "README.md") in PAGES:
            target /= "README.md"
        destination = PAGES[target][0] if target in PAGES else target
        return urlunsplit(("", "", relative_url(destination, self.page), url.query, url.fragment))

    def add_gallery(self, root: ET.Element) -> None:
        gallery = ET.SubElement(root, "section", {"class": "figure-gallery"})
        ET.SubElement(gallery, "h2", {"id": "plots"}).text = "Plots"
        for row in root.findall(".//table/tbody/tr"):
            cells = row.findall("td")
            link = cells[0].find("a") if cells else None
            if link is None or len(cells) != 3:
                continue
            label = "".join(link.itertext())
            description = "".join(cells[1].itertext())
            basis = "".join(cells[2].itertext())
            figure = ET.SubElement(gallery, "figure", {"class": "figure"})
            image_link = ET.SubElement(figure, "a", {"href": link.attrib["href"]})
            ET.SubElement(image_link, "img", {"src": link.attrib["href"], "alt": description})
            caption = ET.SubElement(figure, "figcaption")
            ET.SubElement(caption, "strong").text = label
            caption[-1].tail = f". {description}. {basis}."

    def add_pipeline(self, root: ET.Element) -> None:
        """Keep the shared diagram on desktop and readable flow steps on mobile."""
        if self.source not in (Path("README.md"), Path("docs/pipeline.md")):
            return
        for index, paragraph in enumerate(root):
            image = paragraph.find(".//img")
            if image is None:
                continue
            if image.get("src") != relative_url(Path("docs/figures/pipeline.png"), self.page):
                continue
            pipeline = ET.parse(ROOT / "docs/templates/pipeline.html").getroot()
            for asset in pipeline.iter():
                for attribute in ("href", "src", "poster"):
                    if attribute in asset.attrib:
                        value = asset.attrib[attribute]
                        asset.set(attribute, self.rewrite_url(value, ROOT))
                        if asset.tag == "img" and attribute == "src":
                            url = urlsplit(value)
                            if not url.scheme and not url.netloc:
                                width, height = image_dimensions(ROOT / unquote(url.path))
                                asset.set("width", str(width))
                                asset.set("height", str(height))
                            asset.set("loading", "eager")
                            asset.set("decoding", "async")
            root.remove(paragraph)
            root.insert(index, pipeline)
            return

    def run(self, root: ET.Element) -> None:
        if self.source == Path("docs/figures/README.md"):
            self.add_gallery(root)
        first_image = True
        for element in root.iter():
            if element.tag == "img":
                url = urlsplit(element.attrib["src"])
                if not url.scheme and not url.netloc and url.path:
                    image = (ROOT / self.source.parent / unquote(url.path)).resolve()
                    image.relative_to(ROOT)
                    width, height = image_dimensions(image)
                    element.set("width", str(width))
                    element.set("height", str(height))
                element.set("loading", "eager" if first_image else "lazy")
                element.set("decoding", "async")
                first_image = False
            for attribute in ("href", "src", "poster"):
                if attribute in element.attrib:
                    element.set(attribute, self.rewrite_url(element.attrib[attribute]))
            if element.tag == "pre":
                element.set("tabindex", "0")
        for parent in list(root.iter()):
            for index, child in enumerate(list(parent)):
                if child.tag == "table":
                    wrapper = ET.Element(
                        "div",
                        {
                            "class": "table-scroll",
                            "tabindex": "0",
                            "role": "region",
                            "aria-label": "Scrollable table",
                        },
                    )
                    parent.remove(child)
                    parent.insert(index, wrapper)
                    wrapper.append(child)
        self.add_pipeline(root)


def render_page(source: Path) -> str:
    page, label = PAGES[source]
    renderer = Markdown(
        extensions=["fenced_code", "tables", "toc"],
        extension_configs={"toc": {"toc_depth": "2-3"}},
    )
    renderer.treeprocessors.register(SiteContent(source), "site_content", 5)
    content = renderer.convert((ROOT / source).read_text(encoding="utf-8"))
    content = content.replace("<pre>", '<pre tabindex="0">')
    pipeline_styles = (
        f'<link rel="stylesheet" href="{relative_url(Path("assets/pipeline.css"), page)}">'
        if 'class="pipeline-view"' in content
        else ""
    )
    navigation = []
    for destination, text in list(PAGES.values())[:5]:
        current = ' aria-current="page"' if destination == page else ""
        navigation.append(f'<a href="{relative_url(destination, page)}"{current}>{text}</a>')
    if source in (Path("docs/pipeline.md"), Path("docs/usage.md"), Path("docs/validation.md")):
        contents = (
            '<nav class="page-toc" aria-label="On this page">'
            f'<p class="toc-label">On this page</p>{renderer.toc}</nav>'
        )
        content = content.replace("</h1>", f"</h1>{contents}", 1)
    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="Fit synthetic receptive fields with Python, NumPy, and SciPy.">
  <title>{escape(label)} · CortexProbe</title>
  <link rel="stylesheet" href="{relative_url(Path("assets/site.css"), page)}">
  {pipeline_styles}
</head>
<body>
  <a class="skip-link" href="#main">Skip to content</a>
  <header class="site-header">
    <div class="header-inner">
      <a class="brand" href="{relative_url(Path("index.html"), page)}">CortexProbe</a>
      <nav class="primary-nav" aria-label="Main navigation">{" ".join(navigation)}</nav>
    </div>
  </header>
  <main id="main" class="container">
    <article class="content">{content}</article>
  </main>
  <footer class="site-footer">
    <div class="footer-inner">
      <a href="{REPOSITORY}">Repository</a>
      <a href="{REPOSITORY}/blob/main/{source.as_posix()}">Page source</a>
      <a href="{relative_url(Path("LICENSE"), page)}">MIT license</a>
      <span>By urme</span>
    </div>
  </footer>
</body>
</html>
'''


class PageLinks(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []
        self.ids: set[str] = set()
        self.duplicate_ids: set[str] = set()

    def handle_starttag(self, _tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if value is not None:
                if name in ("href", "src", "poster"):
                    self.links.append(value)
                elif name == "id":
                    if value in self.ids:
                        self.duplicate_ids.add(value)
                    self.ids.add(value)


def validate_links(output: Path) -> None:
    """Fail the build if a local page, media file, stylesheet, or anchor is missing."""
    output = output.resolve()
    pages = {}
    for page in output.rglob("*.html"):
        parser = PageLinks()
        parser.feed(page.read_text(encoding="utf-8"))
        if parser.duplicate_ids:
            raise ValueError(
                f"{page.relative_to(output)}: duplicate HTML ids {sorted(parser.duplicate_ids)}"
            )
        pages[page] = parser
    for page, parser in pages.items():
        for value in parser.links:
            url = urlsplit(value)
            if url.scheme or url.netloc:
                continue
            target = (page.parent / unquote(url.path)).resolve() if url.path else page
            if not target.is_relative_to(output) or not target.is_file():
                raise ValueError(f"{page.relative_to(output)}: missing local target {value!r}")
            if url.fragment and target in pages and unquote(url.fragment) not in pages[target].ids:
                raise ValueError(f"{page.relative_to(output)}: missing anchor {value!r}")


def build_site() -> None:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    for source, (destination, _) in PAGES.items():
        target = OUTPUT / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render_page(source), encoding="utf-8")
    shutil.copytree(ROOT / "docs/assets", OUTPUT / "assets")
    for directory, pattern in (
        ("docs/figures", "*.png"),
        ("docs/figures", "*.svg"),
        ("docs/media", "*.gif"),
        ("docs/media", "*.mp4"),
        ("docs/media", "*.png"),
        ("results", "*.json"),
        ("configs", "*.json"),
    ):
        target = OUTPUT / directory
        target.mkdir(parents=True, exist_ok=True)
        for source in (ROOT / directory).glob(pattern):
            shutil.copy2(source, target / source.name)
    shutil.copy2(ROOT / "LICENSE", OUTPUT / "LICENSE")
    (OUTPUT / ".nojekyll").touch()
    validate_links(OUTPUT)
    figures = len(list((OUTPUT / "docs/figures").glob("*.png")))
    animations = len(list((OUTPUT / "docs/media").glob("*.gif")))
    videos = len(list((OUTPUT / "docs/media").glob("*.mp4")))
    print(
        f"built {OUTPUT.name}: {len(PAGES)} pages, {figures} figures, "
        f"{animations} GIFs, {videos} videos"
    )


if __name__ == "__main__":
    build_site()
