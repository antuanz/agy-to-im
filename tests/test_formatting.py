"""Tests for safe Telegram rich-message rendering."""
from __future__ import annotations

from src.formatting import chunk_markdown, markdown_to_safe_html


def test_renders_supported_markdown_features() -> None:
    rendered = markdown_to_safe_html(
        "# Heading\n\n**bold** and *italic* with [link](https://example.com) "
        "and `code`.\n\n- one\n- two\n\n```python\nprint('<ok>')\n```\n"
    )
    assert "<h1>Heading</h1>" in rendered
    assert "<strong>bold</strong>" in rendered
    assert "<em>italic</em>" in rendered
    assert '<a href="https://example.com">link</a>' in rendered
    assert "<ul>" in rendered and "<li>one</li>" in rendered
    assert "<code>code</code>" in rendered
    assert '<pre><code class="language-python">' in rendered


def test_raw_html_and_remote_media_are_disabled() -> None:
    rendered = markdown_to_safe_html(
        '<script>alert("x")</script>\n\n'
        "![tracking pixel](https://attacker.example/pixel.png)\n\n"
        "[bad](javascript:alert(1))"
    )
    assert "<script" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "<img" not in rendered
    assert "pixel.png" not in rendered
    assert 'href="javascript:' not in rendered
    assert "tracking pixel" in rendered


def test_long_fenced_code_is_split_into_balanced_fences() -> None:
    chunks = chunk_markdown("```python\n" + ("print('x')\n" * 40) + "```\n", max_chars=100)
    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)
    assert all(chunk.startswith("```python\n") for chunk in chunks)
    assert all(chunk.rstrip().endswith("```") for chunk in chunks)
    assert all(chunk.count("```") == 2 for chunk in chunks)


def test_unclosed_streaming_fence_is_closed_for_rendering() -> None:
    chunks = chunk_markdown("before\n```js\nconst x = 1;", max_chars=100)
    assert len(chunks) == 1
    assert chunks[0].rstrip().endswith("```")
