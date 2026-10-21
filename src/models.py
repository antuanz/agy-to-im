"""Discover model slugs exposed by the installed ``agy`` CLI."""
from __future__ import annotations

import asyncio
import re
import shutil
import time
from dataclasses import dataclass
from typing import Callable

from src.state import is_valid_model
from src.processes import process_group_options, terminate_process_group

MODEL_CACHE_TTL_S = 300.0
MODEL_DISCOVERY_TIMEOUT_S = 10.0
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_MODEL_LINE_RE = re.compile(
    r"^\s*([A-Za-z0-9._][A-Za-z0-9._-]*)(?:(?:[ \t]{2,})\S.*)?\s*$"
)


@dataclass(frozen=True)
class ModelSnapshot:
    models: tuple[str, ...]
    error: str = ""


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
    ) -> None:
        self._agy_path = shutil.which(agy_path) or agy_path
        self._ttl = ttl
        self._clock = clock
        self._models: tuple[str, ...] | None = None
        self._cached_at = 0.0
        self._lock = asyncio.Lock()

    async def get(self) -> ModelSnapshot:
        now = self._clock()
        if self._models is not None and now - self._cached_at < self._ttl:
            return ModelSnapshot(self._models)

        async with self._lock:
            now = self._clock()
            if self._models is not None and now - self._cached_at < self._ttl:
                return ModelSnapshot(self._models)
            snapshot = await self._discover()
            if snapshot.models:
                self._models = snapshot.models
                self._cached_at = self._clock()
            return snapshot

    async def _discover(self) -> ModelSnapshot:
        try:
            proc = await asyncio.create_subprocess_exec(
                self._agy_path,
                "models",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **process_group_options(),
            )
        except OSError as exc:
            return ModelSnapshot((), f"could not run agy models: {exc}")

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=MODEL_DISCOVERY_TIMEOUT_S
            )
        except asyncio.TimeoutError:
            await terminate_process_group(proc)
            return ModelSnapshot((), "agy models timed out")
        except asyncio.CancelledError:
            await terminate_process_group(proc)
            raise

        if proc.returncode != 0:
            # Never surface CLI diagnostics through Telegram; they may contain
            # local paths or authentication details.
            del stderr
            return ModelSnapshot((), f"agy models exited {proc.returncode}")
        models = parse_model_slugs(stdout.decode("utf-8", errors="replace"))
        if not models:
            return ModelSnapshot((), "agy models returned no model slugs")
        return ModelSnapshot(models)
