"""Safely discover and stage files newly produced during one chat turn.

The controller is the only component with a Telegram credential.  It compares
the shared per-chat workspace before and after a turn and uploads a small,
conservative allowlist of newly created or changed files. Some agy features save their artifacts in
the worker's isolated ``brain`` directory instead of the current workspace;
the worker stages those verified new files into the shared workspace first.
This module deliberately treats scan or copy uncertainty as "send nothing".
"""
from __future__ import annotations

import os
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


MAX_OUTPUT_FILES = 5
MAX_OUTPUT_FILE_BYTES = 20 * 1024 * 1024
MAX_OUTPUT_TOTAL_BYTES = 25 * 1024 * 1024
MAX_SCAN_DEPTH = 8
MAX_SCANNED_ENTRIES = 4096

_PHOTO_EXTENSIONS = frozenset({".jpeg", ".jpg", ".png", ".webp"})
_EXCLUDED_PARTS = frozenset({".bridge-inbox", ".git", "__pycache__", "okf_memory"})

# A version includes metadata that changes when an agent overwrites a file in
# place.  Device/inode alone misses that common editor behaviour.  Excluding
# ctime avoids treating a harmless rename as a new output.
FileVersion = tuple[int, ...]
OutputKind = Literal["photo", "document"]


@dataclass(frozen=True)
class WorkspaceSnapshot:
    """File versions observed before a CLI turn.

    ``complete`` is false if an unreadable or unexpectedly large tree makes a
    comparison unreliable.  In that case no output is delivered.
    """

    versions: frozenset[FileVersion]
    complete: bool


@dataclass(frozen=True)
class OutputFile:
    path: Path
    filename: str
    kind: OutputKind
    device: int
    inode: int
    size: int
    mtime_ns: int


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


def _file_version(item: os.stat_result) -> FileVersion:
    if item.st_dev or item.st_ino:
        return (item.st_dev, item.st_ino, item.st_mtime_ns, item.st_size)
    return (item.st_mtime_ns, item.st_size)


def snapshot_workspace_files(chat_dir: str | Path) -> WorkspaceSnapshot:
    """Capture every existing regular-file version before a CLI turn."""
    root = _safe_root(chat_dir)
    if root is None:
        return WorkspaceSnapshot(frozenset(), False)
    files, complete = _scan_regular_files(root)
    versions = frozenset(_file_version(item) for _, item in files)
    return WorkspaceSnapshot(versions, complete)


def _is_deliverable_relative_path(relative: Path) -> bool:
    return not any(part.startswith(".") or part in _EXCLUDED_PARTS for part in relative.parts)


def _kind_for_path(path: Path) -> OutputKind:
    if path.suffix.lower() in _PHOTO_EXTENSIONS:
        return "photo"
    # Telegram documents may have any file extension. Returning every eligible
    # new file is useful for agent-produced archives, data, and source formats
    # that cannot be anticipated here; only common image types use sendPhoto.
    return "document"


def _safe_filename(path: Path) -> str | None:
    name = path.name
    if not name or len(name) > 128 or any(ord(character) < 32 for character in name):
        return None
    return name


def discover_new_outputs(
    chat_dir: str | Path, before: WorkspaceSnapshot,
) -> OutputDiscovery:
    """Find a bounded allowlist of files created or changed during the turn.

    Renaming an existing file does not alter its version and is not sent.
    An in-place rewrite is delivered, which lets an agent repair an output
    left by an earlier failed turn. Symlinks, hard links, hidden/internal paths,
    and anything beyond the configured size/count limits are ignored. Non-image
    files are sent as Telegram documents regardless of their extension.
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
        version = _file_version(item)
        if version in before.versions:
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
            or current.st_mtime_ns != item.st_mtime_ns
        ):
            skipped = True
            continue
        accepted.append(OutputFile(
            path=path, filename=filename, kind=kind,
            device=item.st_dev, inode=item.st_ino, size=item.st_size,
            mtime_ns=item.st_mtime_ns,
        ))
        total_bytes += item.st_size
    return OutputDiscovery(tuple(accepted), skipped)


def stage_new_outputs(
    source_dir: str | Path,
    before: WorkspaceSnapshot,
    chat_dir: str | Path,
) -> OutputDiscovery:
    """Copy newly discovered isolated-worker artifacts into a chat workspace.

    The source must already be a tightly scoped worker-only artifact directory,
    not a caller-provided path.  Destination files are copied into a fresh
    ``generated/<random>`` directory so they appear as new files to the
    controller's normal delivery scan.  The copy never follows symlinks and
    verifies the source identity captured during discovery.
    """
    found = discover_new_outputs(source_dir, before)
    if not found.files:
        return found
    root = _safe_root(chat_dir)
    if root is None:
        return OutputDiscovery((), skipped=True)
    generated = root / "generated"
    try:
        if generated.exists():
            if generated.is_symlink() or not generated.is_dir():
                return OutputDiscovery((), skipped=True)
        else:
            generated.mkdir(mode=0o750)
        turn_dir = generated / uuid.uuid4().hex
        turn_dir.mkdir(mode=0o750)
    except OSError:
        return OutputDiscovery((), skipped=True)

    copied: list[OutputFile] = []
    skipped = found.skipped
    for output in found.files:
        destination = turn_dir / output.filename
        try:
            source_stat = os.lstat(output.path)
            if (
                not stat.S_ISREG(source_stat.st_mode)
                or source_stat.st_nlink != 1
                or source_stat.st_dev != output.device
                or source_stat.st_ino != output.inode
                or source_stat.st_size != output.size
                or source_stat.st_mtime_ns != output.mtime_ns
            ):
                raise OSError("artifact changed after discovery")
            source_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            source_fd = os.open(output.path, source_flags)
            try:
                opened_stat = os.fstat(source_fd)
                if (
                    not stat.S_ISREG(opened_stat.st_mode)
                    or opened_stat.st_nlink != 1
                    or opened_stat.st_dev != output.device
                    or opened_stat.st_ino != output.inode
                    or opened_stat.st_size != output.size
                    or opened_stat.st_mtime_ns != output.mtime_ns
                ):
                    raise OSError("artifact changed while opening")
                destination_fd = os.open(
                    destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640,
                )
                try:
                    remaining = output.size
                    while remaining:
                        chunk = os.read(source_fd, min(64 * 1024, remaining))
                        if not chunk:
                            raise OSError("artifact was truncated while copying")
                        pending = memoryview(chunk)
                        while pending:
                            written = os.write(destination_fd, pending)
                            if written <= 0:
                                raise OSError("artifact write failed")
                            pending = pending[written:]
                        remaining -= len(chunk)
                finally:
                    os.close(destination_fd)
            finally:
                os.close(source_fd)
            copied.append(OutputFile(
                path=destination, filename=output.filename, kind=output.kind,
                device=output.device, inode=output.inode, size=output.size,
                mtime_ns=output.mtime_ns,
            ))
        except OSError:
            skipped = True
            try:
                destination.unlink(missing_ok=True)
            except OSError:
                pass
    return OutputDiscovery(tuple(copied), skipped)
