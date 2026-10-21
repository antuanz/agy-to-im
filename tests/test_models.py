"""Tests for dynamic agy model discovery and caching."""
from __future__ import annotations

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


async def test_expired_cache_failure_is_not_used_or_cached() -> None:
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
    assert failed.models == ()
    assert failed.error == "offline"
    assert (await catalog.get()).models == ("model-b",)


async def test_initial_failure_returns_actionable_error() -> None:
    catalog = ModelCatalog("definitely-not-an-agy-binary")
    result = await catalog.get()
    assert result.models == ()
    assert "could not run agy models" in result.error.lower()
