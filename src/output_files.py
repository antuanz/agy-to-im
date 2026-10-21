"""Safely discover files newly produced during one successful chat turn.

The worker can write only inside its per-chat workspace.  The controller is the
only component with a Telegram credential, so it compares the workspace before
and after a turn and sends back a small, conservative allowlist of new files.
This module deliberately treats scan uncertainty as "send nothing".
"""
from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


MAX_OUTPUT_FILES = 5
MAX_OUTPUT_FILE_BYTES = 20 * 1024 * 1024
MAX_OUTPUT_TOTAL_BYTES = 25 * 1024 * 1024
MAX_SCAN_DEPTH = 8
MAX_SCANNED_ENTRIES = 4096

_PHOTO_EXTENSIONS = frozenset({".jpeg", ".jpg", ".png", ".webp"})
_DOCUMENT_EXTENSIONS = frozenset({
    ".csv", ".docx", ".json", ".md", ".pdf", ".pptx", ".py", ".txt", ".xlsx",
})
_EXCLUDED_PARTS = frozenset({".bridge-inbox", ".git", "__pycache__", "okf_memory"})

# Linux exposes stable device/inode pairs. Windows can report both as zero, so
# use a conservative metadata fallback there; a false non-delivery is safer
# than returning an existing file.
FileIdentity = tuple[int, ...]
OutputKind = Literal["photo", "document"]


@dataclass(frozen=True)
class WorkspaceSnapshot:
    """File identities observed before a CLI turn.

    ``complete`` is false if an unreadable or unexpectedly large tree makes a
    comparison unreliable.  In that case no output is delivered.
    """

    identities: frozenset[FileIdentity]
    complete: bool


@dataclass(frozen=True)
class OutputFile:
    path: Path
    filename: str
    kind: OutputKind


@dataclass(frozen=True)
class OutputDiscovery:
    files: tuple[OutputFile, ...]
    skipped: bool = False


def _safe_root(chat_dir: str | Path) -> Path | None:
    root = Path(chat_dir)
    try:
        if not root.is_dir() or root.is_symlink():
            return None
        return root.resolve(strict=True)
    except OSError:
        return None


def _scan_regular_files(root: Path) -> tuple[list[tuple[Path, os.stat_result]], bool]:
    """Return regular files below root without ever following symlinks."""
    files: list[tuple[Path, os.stat_result]] = []
    pending: list[tuple[Path, int]] = [(root, 0)]
    seen_entries = 0
    try:
        while pending:
            directory, depth = pending.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    seen_entries += 1
                    if seen_entries > MAX_SCANNED_ENTRIES:
                        return files, False
                    path = Path(entry.path)
                    try:
                        # os.lstat is used instead of DirEntry.stat because a
                        # Windows network/share implementation can return zero
                        # inode values from the latter even for local files.
                        entry_stat = os.lstat(path)
                    except OSError:
                        return files, False
                    if stat.S_ISREG(entry_stat.st_mode):
                        files.append((path, entry_stat))
                    elif stat.S_ISDIR(entry_stat.st_mode):
                        if depth >= MAX_SCAN_DEPTH:
                            return files, False
                        pending.append((path, depth + 1))
        return files, True
    except OSError:
        return files, False


def _file_identity(item: os.stat_result) -> FileIdentity:
    if item.st_dev or item.st_ino:
        return (item.st_dev, item.st_ino)
    return (item.st_ctime_ns, item.st_mtime_ns, item.st_size)


def snapshot_workspace_files(chat_dir: str | Path) -> WorkspaceSnapshot:
    """Capture every existing regular-file identity before a CLI turn."""
    root = _safe_root(chat_dir)
    if root is None:
        return WorkspaceSnapshot(frozenset(), False)
    files, complete = _scan_regular_files(root)
    identities = frozenset(_file_identity(item) for _, item in files)
    return WorkspaceSnapshot(identities, complete)


def _is_deliverable_relative_path(relative: Path) -> bool:
    return not any(part.startswith(".") or part in _EXCLUDED_PARTS for part in relative.parts)


def _kind_for_path(path: Path) -> OutputKind | None:
    suffix = path.suffix.lower()
    if suffix in _PHOTO_EXTENSIONS:
        return "photo"
    if suffix in _DOCUMENT_EXTENSIONS:
        return "document"
    return None


def _safe_filename(path: Path) -> str | None:
    name = path.name
    if not name or len(name) > 128 or any(ord(character) < 32 for character in name):
        return None
    return name


def discover_new_outputs(
    chat_dir: str | Path, before: WorkspaceSnapshot,
) -> OutputDiscovery:
    """Find a bounded allowlist of files with identities absent before the turn.

    Existing files renamed into an allowed extension retain their inode and are
    not sent. Symlinks, hard links, hidden/internal paths, unsupported files,
    and anything beyond the configured size/count limits are ignored.
    """
    if not before.complete:
        return OutputDiscovery((), skipped=True)
    root = _safe_root(chat_dir)
    if root is None:
        return OutputDiscovery((), skipped=True)
    files, complete = _scan_regular_files(root)
    if not complete:
        return OutputDiscovery((), skipped=True)

    accepted: list[OutputFile] = []
    total_bytes = 0
    skipped = False
    for path, item in sorted(files, key=lambda pair: str(pair[0])):
        identity = _file_identity(item)
        if identity in before.identities:
            continue
        try:
            relative = path.relative_to(root)
        except ValueError:
            skipped = True
            continue
        kind = _kind_for_path(path)
        filename = _safe_filename(path)
        if (
            not _is_deliverable_relative_path(relative)
            or kind is None
            or filename is None
            or item.st_nlink != 1
            or item.st_size <= 0
            or item.st_size > MAX_OUTPUT_FILE_BYTES
            or len(accepted) >= MAX_OUTPUT_FILES
            or total_bytes + item.st_size > MAX_OUTPUT_TOTAL_BYTES
        ):
            skipped = True
            continue
        # Re-check immediately before exposing a file to the controller. This
        # makes a replacement with a symlink, directory, or oversized file fail
        # closed even if the workspace changes after the initial directory scan.
        try:
            current = path.stat(follow_symlinks=False)
        except OSError:
            skipped = True
            continue
        if (
            not stat.S_ISREG(current.st_mode)
            or current.st_nlink != 1
            or current.st_dev != item.st_dev
            or current.st_ino != item.st_ino
            or current.st_size != item.st_size
        ):
            skipped = True
            continue
        accepted.append(OutputFile(path=path, filename=filename, kind=kind))
        total_bytes += item.st_size
    return OutputDiscovery(tuple(accepted), skipped)
