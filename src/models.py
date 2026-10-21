"""Discover model slugs exposed by the installed ``agy`` CLI."""
from __future__ import annotations

import asyncio
import re
import shutil
import time
from dataclasses import dataclass
from collections.abc import Awaitable, Callable

from src.state import is_valid_model
from src.processes import process_group_options, terminate_process_group

MODEL_CACHE_TTL_S = 300.0
MODEL_DISCOVERY_TIMEOUT_S = 10.0
MODEL_STDOUT_CAP_BYTES = 262_144
MODEL_STDERR_CAP_BYTES = 65_536
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
# agy uses aligned spaces on a terminal but one tab between columns when stdout is captured.
_MODEL_LINE_RE = re.compile(
    r"^\s*([A-Za-z0-9._][A-Za-z0-9._-]*)(?:(?: {2,}|\t+)\S.*)?\s*$"
)


@dataclass(frozen=True)
class ModelSnapshot:
    models: tuple[str, ...]
    error: str = ""
    stale: bool = False


def parse_model_slugs(output: str) -> tuple[str, ...]:
    """Parse the exact slug column from human-readable ``agy models`` output."""
    found: list[str] = []
    seen: set[str] = set()
    for raw_line in output.splitlines():
        line = _ANSI_RE.sub("", raw_line)
        match = _MODEL_LINE_RE.match(line)
        if match is None:
            continue
        slug = match.group(1)
        if is_valid_model(slug) and slug not in seen:
            found.append(slug)
            seen.add(slug)
    return tuple(found)


class ModelCatalog:
    """Five-minute, success-only cache around ``agy models``."""

    def __init__(
        self,
        agy_path: str,
        *,
        ttl: float = MODEL_CACHE_TTL_S,
        clock: Callable[[], float] = time.monotonic,
        discover: Callable[[], Awaitable[ModelSnapshot]] | None = None,
    ) -> None:
        self._agy_path = shutil.which(agy_path) or agy_path
        self._ttl = ttl
        self._clock = clock
        self._models: tuple[str, ...] | None = None
        self._cached_at = 0.0
        self._lock = asyncio.Lock()
        self._discover_override = discover

    async def get(self) -> ModelSnapshot:
        now = self._clock()
        if self._models is not None and now - self._cached_at < self._ttl:
            return ModelSnapshot(self._models)

        async with self._lock:
            now = self._clock()
            if self._models is not None and now - self._cached_at < self._ttl:
                return ModelSnapshot(self._models)
            if self._discover_override is None:
                snapshot = await self._discover()
            else:
                snapshot = await self._discover_override()
            if snapshot.models:
                self._models = snapshot.models
                self._cached_at = self._clock()
                return snapshot
            if self._models is not None:
                # An old, successfully validated catalogue is safer than
                # needlessly blocking a selected model during a transient CLI
                # failure. Keep attempting refreshes on future calls.
                return ModelSnapshot(self._models, snapshot.error, stale=True)
            return snapshot

    async def _discover(self) -> ModelSnapshot:
        try:
            proc = await asyncio.create_subprocess_exec(
                self._agy_path,
                "models",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=MODEL_STDOUT_CAP_BYTES + 1,
                **process_group_options(),
            )
        except OSError as exc:
            return ModelSnapshot((), f"could not run agy models: {exc}")

        try:
            stdout_result, stderr_result, _ = await asyncio.wait_for(
                asyncio.gather(
                    _read_capped(proc.stdout, MODEL_STDOUT_CAP_BYTES),
                    _read_capped(proc.stderr, MODEL_STDERR_CAP_BYTES),
                    proc.wait(),
                ), timeout=MODEL_DISCOVERY_TIMEOUT_S
            )
        except asyncio.TimeoutError:
            await terminate_process_group(proc)
            return ModelSnapshot((), "agy models timed out")
        except asyncio.CancelledError:
            await terminate_process_group(proc)
            raise

        stdout, stdout_exceeded = stdout_result
        _, stderr_exceeded = stderr_result
        if stdout_exceeded or stderr_exceeded:
            return ModelSnapshot((), "agy models output exceeded size limit")

        if proc.returncode != 0:
            # Never surface CLI diagnostics through Telegram; they may contain
            # local paths or authentication details.
            return ModelSnapshot((), f"agy models exited {proc.returncode}")
        models = parse_model_slugs(stdout.decode("utf-8", errors="replace"))
        if not models:
            return ModelSnapshot((), "agy models returned no model slugs")
        return ModelSnapshot(models)


async def _read_capped(
    stream: asyncio.StreamReader | None, cap: int
) -> tuple[bytes, bool]:
    if stream is None:
        return b"", False
    captured = bytearray()
    exceeded = False
    while True:
        block = await stream.read(8192)
        if not block:
            break
        room = cap - len(captured)
        if room > 0:
            captured.extend(block[:room])
        if len(block) > room:
            exceeded = True
    return bytes(captured), exceeded
