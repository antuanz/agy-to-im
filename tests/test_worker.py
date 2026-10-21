"""Validation tests for the root-supervised worker boundary."""
from __future__ import annotations

from pathlib import Path

import src.worker as worker
from src.models import MODEL_CACHE_TTL_S
from src.agy_runner import AgyResult
from src.diagnostics import safe_failure_reason
from src.worker import _valid_chat_dir


def test_worker_translates_only_shared_controller_chat_paths(
    monkeypatch, tmp_path: Path,
) -> None:
    controller_root = tmp_path / "controller-chats"
    worker_root = tmp_path / "worker-chats"
    (controller_root / "42").mkdir(parents=True)
    (worker_root / "42").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setenv("AGY_BRIDGE_CONTROLLER_CHATS_ROOT", str(controller_root))
    monkeypatch.setenv("AGY_BRIDGE_CHATS_ROOT", str(worker_root))

    assert _valid_chat_dir(str(controller_root / "42")) == str(worker_root / "42")
    assert _valid_chat_dir(str(outside)) is None


async def test_worker_reuses_verified_model_catalogue_within_ttl(monkeypatch) -> None:
    monkeypatch.setattr(worker, "_MODEL_CACHE", ("gemini-3.8-flash-medium",))
    monkeypatch.setattr(worker, "_MODEL_CACHE_AT", 100.0)
    monkeypatch.setattr(worker.time, "monotonic", lambda: 100.0 + MODEL_CACHE_TTL_S - 1)

    async def should_not_run(*_args, **_kwargs):
        raise AssertionError("a fresh agy models process should not be started")

    monkeypatch.setattr(worker.asyncio, "create_subprocess_exec", should_not_run)
    models, error = await worker._discover_models()

    assert models == ("gemini-3.8-flash-medium",)
    assert error == ""


async def test_worker_uses_last_verified_models_after_refresh_failure(monkeypatch) -> None:
    monkeypatch.setattr(worker, "_MODEL_CACHE", ("gemini-3.8-flash-medium",))
    monkeypatch.setattr(worker, "_MODEL_CACHE_AT", 100.0)
    monkeypatch.setattr(worker.time, "monotonic", lambda: 100.0 + MODEL_CACHE_TTL_S + 1)

    async def unavailable(*_args, **_kwargs):
        raise OSError("not available")

    monkeypatch.setattr(worker.asyncio, "create_subprocess_exec", unavailable)
    models, error = await worker._discover_models()

    assert models == ("gemini-3.8-flash-medium",)
    assert error == "worker could not run agy models"


def test_worker_failure_reason_never_returns_raw_cli_output() -> None:
    result = AgyResult(
        text="", exit_code=1,
        stderr="EACCES: permission denied /home/agy/private-token",
        status="ERROR",
    )
    assert safe_failure_reason(result) == "isolated runtime storage unavailable"


def test_worker_failure_reason_classifies_cli_setup_without_exposing_detail() -> None:
    result = AgyResult(
        text="", exit_code=1,
        stderr="invalid argument: --input-format cannot be used with this mode",
        status="ERROR",
    )
    assert safe_failure_reason(result) == "agy launch configuration rejected"
    assert "argument" not in safe_failure_reason(result)


def test_worker_failure_reason_classifies_missing_runtime_settings() -> None:
    result = AgyResult(
        text="", exit_code=1,
        stderr="unable to load workspace configuration",
        status="ERROR",
    )
    assert safe_failure_reason(result) == "worker runtime configuration unavailable"
