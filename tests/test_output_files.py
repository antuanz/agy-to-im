from __future__ import annotations

import os
from pathlib import Path

import pytest
from src.output_files import (
    MAX_OUTPUT_FILE_BYTES,
    discover_new_outputs,
    snapshot_workspace_files,
)


def test_discovers_every_new_regular_file_as_a_document_or_photo(tmp_path: Path) -> None:
    (tmp_path / "old.md").write_text("old", encoding="utf-8")
    before = snapshot_workspace_files(tmp_path)
    (tmp_path / "report.md").write_text("new", encoding="utf-8")
    (tmp_path / "chart.png").write_bytes(b"image")
    (tmp_path / "archive.tar.gz").write_bytes(b"binary")

    found = discover_new_outputs(tmp_path, before)

    assert [(item.filename, item.kind) for item in found.files] == [
        ("archive.tar.gz", "document"),
        ("chart.png", "photo"), ("report.md", "document"),
    ]
    assert found.skipped is False


def test_does_not_send_existing_file_renamed_to_an_allowed_extension(tmp_path: Path) -> None:
    original = tmp_path / "secret.bin"
    original.write_bytes(b"old")
    before = snapshot_workspace_files(tmp_path)
    original.rename(tmp_path / "secret.md")

    assert discover_new_outputs(tmp_path, before).files == ()


def test_never_follows_or_sends_a_symlink(tmp_path: Path) -> None:
    if os.name == "nt":
        pytest.skip("Windows symlink creation requires a privilege not present in CI")
    before = snapshot_workspace_files(tmp_path)
    target = tmp_path / "target.md"
    target.write_text("target", encoding="utf-8")
    os.symlink(target, tmp_path / "link.md")

    found = discover_new_outputs(tmp_path, before)

    assert [item.filename for item in found.files] == ["target.md"]


def test_never_sends_a_hardlink(tmp_path: Path) -> None:
    original = tmp_path / "existing.md"
    original.write_text("old", encoding="utf-8")
    before = snapshot_workspace_files(tmp_path)
    os.link(original, tmp_path / "copy.md")

    found = discover_new_outputs(tmp_path, before)

    assert found.files == ()


def test_rejects_too_large_file(tmp_path: Path) -> None:
    before = snapshot_workspace_files(tmp_path)
    large = tmp_path / "large.pdf"
    with large.open("wb") as handle:
        handle.truncate(MAX_OUTPUT_FILE_BYTES + 1)

    found = discover_new_outputs(tmp_path, before)

    assert found.files == ()
    assert found.skipped is True


def test_never_returns_inbound_bridge_files(tmp_path: Path) -> None:
    inbox = tmp_path / ".bridge-inbox"
    inbox.mkdir()
    before = snapshot_workspace_files(tmp_path)
    (inbox / "upload.md").write_text("input", encoding="utf-8")

    found = discover_new_outputs(tmp_path, before)

    assert found.files == ()
    assert found.skipped is True
