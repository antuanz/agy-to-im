"""Tests for dynamic agy model discovery and caching."""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path
from unittest.mock import AsyncMock

from src.models import ModelCatalog, ModelSnapshot, parse_model_slugs


def test_parse_model_slugs_preserves_exact_slug_order() -> None:
    output = """
Available models:
gemini-4-argon-high       Gemini 4 Argon (High)
gemini-3.8-flash-medium   Gemini 3.8 Flash (Medium)
gemini-4-argon-high       duplicate
not a model row
"""
    assert parse_model_slugs(output) == (
        "gemini-4-argon-high",
        "gemini-3.8-flash-medium",
    )


async def test_success_is_cached_for_five_minutes() -> None:
    now = [100.0]
    catalog = ModelCatalog("agy", clock=lambda: now[0])
    catalog._discover = AsyncMock(  # type: ignore[method-assign]
        return_value=ModelSnapshot(("model-a", "model-b"))
    )
    assert (await catalog.get()).models == ("model-a", "model-b")
    now[0] += 299.9
    assert (await catalog.get()).models == ("model-a", "model-b")
    assert catalog._discover.await_count == 1  # type: ignore[attr-defined]


async def test_expired_cache_failure_preserves_last_successful_catalogue() -> None:
    now = [0.0]
    catalog = ModelCatalog("agy", clock=lambda: now[0])
    catalog._discover = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            ModelSnapshot(("model-a",)),
            ModelSnapshot((), "offline"),
            ModelSnapshot(("model-b",)),
        ]
    )
    assert (await catalog.get()).models == ("model-a",)
    now[0] = 301.0
    failed = await catalog.get()
    assert failed.models == ("model-a",)
    assert failed.error == "offline"
    assert failed.stale is True
    assert (await catalog.get()).models == ("model-b",)


async def test_initial_failure_returns_actionable_error() -> None:
    catalog = ModelCatalog("definitely-not-an-agy-binary")
    result = await catalog.get()
    assert result.models == ()
    assert "could not run agy models" in result.error.lower()


def _fake_models(tmp_path: Path) -> Path:
    script = tmp_path / "fake_models.py"
    script.write_text(
        "import sys\n"
        "sys.stdout.write('x' * 300000)\n"
        "sys.stdout.flush()\n"
    )
    if os.name == "nt":
        wrapper = tmp_path / "agy.cmd"
        wrapper.write_text(f'@"{sys.executable}" "{script}" %*\n')
    else:
        wrapper = tmp_path / "agy"
        wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    return wrapper


async def test_model_discovery_bounds_oversized_output(tmp_path: Path) -> None:
    catalog = ModelCatalog(str(_fake_models(tmp_path)))
    result = await catalog.get()
    assert result.models == ()
    assert "size limit" in result.error
