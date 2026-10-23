"""README site."""

from __future__ import annotations

import shutil
import struct
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit

from markdown import Markdown
from markdown.treeprocessors import Treeprocessor

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "_site"
REPOSITORY = "https://github.com/urmeo/cortexprobe"


def image_dimensions(path: Path) -> tuple[int, int]:
    """Image dimensions."""
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
    """README assets."""

    def rewrite_url(self, value: str) -> str:
        url = urlsplit(value)
        if url.scheme or url.netloc or not url.path:
            return value
        target = (ROOT / unquote(url.path)).resolve().relative_to(ROOT)
        destination = "index.html" if target == Path("README.md") else target.as_posix()
        return urlunsplit(("", "", destination, url.query, url.fragment))

    def rewrite_element(self, element: ET.Element, *, eager: bool = False) -> None:
        if element.tag == "img":
            url = urlsplit(element.attrib["src"])
            if not url.scheme and not url.netloc and url.path:
                image = (ROOT / unquote(url.path)).resolve()
                image.relative_to(ROOT)
                width, height = image_dimensions(image)
                element.set("width", str(width))
                element.set("height", str(height))
            element.set("loading", "eager" if eager else "lazy")
            element.set("decoding", "async")
        for attribute in ("href", "src", "poster"):
            if attribute in element.attrib:
                element.set(attribute, self.rewrite_url(element.attrib[attribute]))

    def add_videos(self, root: ET.Element) -> None:
        """Video playback."""
        for parent in list(root.iter()):
            for index, link in enumerate(list(parent)):
                image = link.find("img") if link.tag == "a" else None
                if image is None:
                    continue
                url = urlsplit(link.get("href", ""))
                if url.scheme or url.netloc or not url.path.endswith(".mp4"):
                    continue
                video_path = (ROOT / unquote(url.path)).resolve()
                video_path.relative_to(ROOT)
                poster = video_path.with_suffix(".png")
                if not video_path.is_file() or not poster.is_file():
                    continue
                width, height = image_dimensions(poster)
                video = ET.Element(
                    "video",
                    {
                        "controls": "",
                        "playsinline": "",
                        "preload": "metadata",
                        "width": str(width),
                        "height": str(height),
                        "poster": poster.relative_to(ROOT).as_posix(),
                        "aria-label": image.get("alt", "Video preview"),
                    },
                )
                source = link.attrib["href"]
                ET.SubElement(video, "source", {"src": source, "type": "video/mp4"})
                ET.SubElement(video, "a", {"href": source}).text = "Download MP4"
                video.tail = link.tail
                parent.remove(link)
                parent.insert(index, video)

    def run(self, root: ET.Element) -> None:
        self.add_videos(root)
        first_image = True
        for element in list(root.iter()):
            self.rewrite_element(element, eager=first_image)
            if element.tag == "img":
                first_image = False
            if element.tag == "pre":
                element.set("tabindex", "0")
            for index, child in enumerate(list(element)):
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
                    element.remove(child)
                    element.insert(index, wrapper)
                    wrapper.append(child)


def render_page() -> str:
    renderer = Markdown(
        extensions=["fenced_code", "tables", "toc", "md_in_html"],
        extension_configs={"toc": {"toc_depth": "2-3"}},
    )
    renderer.treeprocessors.register(SiteContent(renderer), "site_content", 5)
    content = renderer.convert((ROOT / "README.md").read_text(encoding="utf-8"))
    content = content.replace("<pre>", '<pre tabindex="0">')
    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="Fit synthetic receptive fields with Python, NumPy, and SciPy.">
  <title>CortexProbe</title>
  <link rel="stylesheet" href="assets/site.css">
</head>
<body>
  <a class="skip-link" href="#main">Skip to content</a>
  <header class="site-header">
    <div class="header-inner">
      <a class="brand" href="#main">CortexProbe</a>
      <nav class="primary-nav" aria-label="Main navigation">
        <a href="#run">Run</a>
        <a href="#results">Results</a>
        <a href="#references">References</a>
      </nav>
    </div>
  </header>
  <main id="main" class="container">
    <article class="content">{content}</article>
  </main>
  <footer class="site-footer">
    <div class="footer-inner">
      <a href="{REPOSITORY}">Repository</a>
      <a href="{REPOSITORY}/blob/main/README.md">Page source</a>
      <a href="LICENSE">MIT license</a>
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
            if value is None:
                continue
            if name in ("href", "src", "poster"):
                self.links.append(value)
            elif name == "id":
                if value in self.ids:
                    self.duplicate_ids.add(value)
                self.ids.add(value)


def validate_links(output: Path) -> None:
    """Validate links."""
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
    page = render_page()
    (OUTPUT / "index.html").write_text(page, encoding="utf-8")
    shutil.copytree(ROOT / "docs/assets", OUTPUT / "assets")
    for directory, suffixes in {
        "docs/figures": {".png", ".svg"},
        "docs/media": {".gif", ".mp4", ".png"},
    }.items():
        target = OUTPUT / directory
        target.mkdir(parents=True, exist_ok=True)
        for source in (ROOT / directory).iterdir():
            if source.is_file() and source.suffix in suffixes:
                shutil.copy2(source, target / source.name)
    links = PageLinks()
    links.feed(page)
    for value in set(links.links):
        url = urlsplit(value)
        relative = Path(unquote(url.path))
        if url.scheme or url.netloc or relative.parts[:1] != ("results",):
            continue
        if relative.suffix not in {".png", ".csv"}:
            continue
        source = ROOT / relative
        source.resolve().relative_to(ROOT / "results")
        target = (OUTPUT / relative).resolve()
        target.relative_to(OUTPUT / "results")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    shutil.copy2(ROOT / "LICENSE", OUTPUT / "LICENSE")
    (OUTPUT / ".nojekyll").touch()
    validate_links(OUTPUT)
    figures = sum(
        len(list((OUTPUT / directory).rglob("*.png"))) for directory in ("docs/figures", "results")
    )
    animations = len(list((OUTPUT / "docs/media").glob("*.gif")))
    videos = len(list((OUTPUT / "docs/media").glob("*.mp4")))
    print(f"built {OUTPUT.name}: 1 page, {figures} figures, {animations} GIFs, {videos} videos")


if __name__ == "__main__":
    build_site()
