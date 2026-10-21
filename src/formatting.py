"""Safe Markdown rendering and rich-message-aware chunking."""
from __future__ import annotations

import html
import re
from collections.abc import Iterable

from markdown_it import MarkdownIt

RICH_MESSAGE_MAX_CHARS = 30_000
_FENCE_OPEN_RE = re.compile(r"^( {0,3})(`{3,}|~{3,})([^\r\n]*)(?:\r?\n|$)")
# ``agy`` sometimes describes a generated artifact using a ``file://`` URL.
# That URL names the worker's private filesystem and is neither meaningful nor
# usable in Telegram; the bridge delivers the artifact as a Telegram upload.
_LOCAL_FILE_MARKDOWN_LINK_RE = re.compile(
    r"!?\[([^\]\r\n]*)\]\(\s*<?file://[^\s)>]+>?[^)]*\)", re.IGNORECASE,
)
_LOCAL_FILE_AUTOLINK_RE = re.compile(r"<file://[^>\r\n]+>", re.IGNORECASE)
_LOCAL_FILE_URL_RE = re.compile(r"(?<![\w/])file://[^\s<>\[\](){}]+", re.IGNORECASE)


def _render_image(tokens: list[object], idx: int, *_: object) -> str:
    """Render image syntax as inert alt text; never ask Telegram to fetch media."""
    content = getattr(tokens[idx], "content", "")
    return html.escape(str(content), quote=False)


_MARKDOWN = MarkdownIt(
    "commonmark",
    {"html": False, "linkify": False, "typographer": False},
)
_MARKDOWN.renderer.rules["image"] = _render_image


def markdown_to_safe_html(markdown: str) -> str:
    """Render supported Markdown to HTML without raw HTML or remote media."""
    return _MARKDOWN.render(markdown)


def strip_local_file_references(markdown: str) -> str:
    """Remove worker-local file URLs while retaining a human-readable label.

    A generated file is uploaded separately after a successful turn. Keeping a
    local URL in the text would expose an internal path and invite the user to
    click a link Telegram cannot open. This is intentionally applied to both
    streaming drafts and the final reply.
    """
    without_links = _LOCAL_FILE_MARKDOWN_LINK_RE.sub(r"\1", markdown)
    without_autolinks = _LOCAL_FILE_AUTOLINK_RE.sub("", without_links)
    return _LOCAL_FILE_URL_RE.sub(_strip_bare_local_file_url, without_autolinks)


def _strip_bare_local_file_url(match: re.Match[str]) -> str:
    """Drop a bare URL but retain ordinary sentence punctuation after it."""
    value = match.group(0)
    return value[len(value.rstrip(".,;:!?")):]


def chunk_markdown(
    markdown: str, max_chars: int = RICH_MESSAGE_MAX_CHARS
) -> list[str]:
    """Split Markdown without leaving a fenced code block open in any chunk."""
    if not markdown:
        return []
    if max_chars < 16:
        raise ValueError("max_chars is too small for safe Markdown chunking")

    units = list(_markdown_units(markdown, max_chars))
    chunks: list[str] = []
    current = ""
    for unit in units:
        if current and len(current) + len(unit) > max_chars:
            chunks.append(current)
            current = ""
        current += unit
    if current:
        chunks.append(current)
    return chunks


def _markdown_units(markdown: str, max_chars: int) -> Iterable[str]:
    lines = markdown.splitlines(keepends=True)
    plain: list[str] = []
    index = 0
    while index < len(lines):
        match = _FENCE_OPEN_RE.match(lines[index])
        if match is None:
            plain.append(lines[index])
            index += 1
            continue

        if plain:
            yield from _split_plain("".join(plain), max_chars)
            plain = []

        opener = lines[index]
        marker = match.group(2)
        index += 1
        body: list[str] = []
        closer = ""
        close_re = re.compile(
            rf"^ {{0,3}}{re.escape(marker[0])}{{{len(marker)},}}[ \t]*(?:\r?\n|$)"
        )
        while index < len(lines):
            if close_re.match(lines[index]):
                closer = lines[index]
                index += 1
                break
            body.append(lines[index])
            index += 1
        yield from _split_fence(opener, "".join(body), closer, marker, max_chars)

    if plain:
        yield from _split_plain("".join(plain), max_chars)


def _split_fence(
    opener: str,
    body: str,
    closer: str,
    marker: str,
    max_chars: int,
) -> Iterable[str]:
    closing = closer or marker
    whole = opener + body + closing
    if len(whole) <= max_chars:
        yield whole
        return

    normalized_opener = opener if opener.endswith(("\n", "\r")) else opener + "\n"
    normalized_closer = closing.lstrip(" \t")
    if not normalized_closer.endswith(("\n", "\r")):
        normalized_closer += "\n"
    capacity = max_chars - len(normalized_opener) - len(normalized_closer) - 1
    if capacity < 1:
        raise ValueError("max_chars cannot contain the fenced-code wrapper")
    for part in _split_plain(body, capacity):
        separator = "" if not part or part.endswith(("\n", "\r")) else "\n"
        yield normalized_opener + part + separator + normalized_closer


def _split_plain(text: str, max_chars: int) -> Iterable[str]:
    remaining = text
    while len(remaining) > max_chars:
        cut = remaining.rfind("\n", 0, max_chars + 1)
        if cut <= 0:
            cut = max_chars
        else:
            cut += 1
        yield remaining[:cut]
        remaining = remaining[cut:]
    if remaining:
        yield remaining
