"""Validation tests for the root-supervised worker boundary."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

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


async def test_worker_stages_new_brain_artifacts_only_after_a_successful_turn(
    monkeypatch, tmp_path: Path,
) -> None:
    controller_root = tmp_path / "controller-chats"
    worker_root = tmp_path / "worker-chats"
    agent_home = tmp_path / "agent-home"
    (controller_root / "42").mkdir(parents=True)
    (worker_root / "42").mkdir(parents=True)
    monkeypatch.setenv("AGY_BRIDGE_CONTROLLER_CHATS_ROOT", str(controller_root))
    monkeypatch.setenv("AGY_BRIDGE_CHATS_ROOT", str(worker_root))
    monkeypatch.setenv("HOME", str(agent_home))

    class Policy:
        execution_mode = "plan"

        def for_execution_mode(self, requested: str):
            assert requested == "plan"
            return self

    class Writer:
        def __init__(self) -> None:
            self.events: list[dict[str, object]] = []

        def write(self, raw: bytes) -> None:
            self.events.append(json.loads(raw))

        async def drain(self) -> None:
            pass

    async def fake_run_agy(*_args, **_kwargs) -> AgyResult:
        artifact = agent_home / ".gemini/antigravity-cli/brain/turn/image.png"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"png")
        return AgyResult("done", 0, "", "SUCCESS", "conversation")

    staged: list[tuple[Path, Path]] = []

    def fake_stage(source: Path, _before: object, destination: str) -> None:
        staged.append((source, Path(destination)))

    monkeypatch.setattr(worker, "load_access_policy", lambda: Policy())
    monkeypatch.setattr(worker, "run_agy", fake_run_agy)
    monkeypatch.setattr(worker, "stage_new_outputs", fake_stage)
    writer = Writer()

    await worker._run_request({
        "prompt": "make image", "chat_dir": str(controller_root / "42"),
        "model": "", "conversation_id": "", "execution_mode": "plan", "request_id": "a" * 24,
    }, writer)

    assert staged == [(
        agent_home / ".gemini/antigravity-cli/brain", worker_root / "42",
    )]
    assert writer.events[-1]["type"] == "result"


async def test_worker_cancel_only_stops_the_matching_active_request(monkeypatch, tmp_path: Path) -> None:
    worker._ACTIVE_REQUESTS.clear()
    started = asyncio.Event()
    controller_root = tmp_path / "controller-chats"
    worker_root = tmp_path / "worker-chats"
    (controller_root / "42").mkdir(parents=True)
    (worker_root / "42").mkdir(parents=True)
    monkeypatch.setenv("AGY_BRIDGE_CONTROLLER_CHATS_ROOT", str(controller_root))
    monkeypatch.setenv("AGY_BRIDGE_CHATS_ROOT", str(worker_root))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    class Policy:
        execution_mode = "plan"

        def for_execution_mode(self, requested: str):
            assert requested == "plan"
            return self

    class Writer:
        def __init__(self) -> None:
            self.events: list[dict[str, object]] = []

        def write(self, raw: bytes) -> None:
            self.events.append(json.loads(raw))

        async def drain(self) -> None:
            pass

    async def slow_run_agy(*_args, **_kwargs) -> AgyResult:
        started.set()
        await asyncio.Event().wait()
        return AgyResult("unreachable", 0, "", "SUCCESS")

    monkeypatch.setattr(worker, "load_access_policy", lambda: Policy())
    monkeypatch.setattr(worker, "run_agy", slow_run_agy)
    run_writer = Writer()
    cancel_writer = Writer()
    request_id = "b" * 24
    task = asyncio.create_task(worker._run_request({
        "prompt": "wait", "chat_dir": str(controller_root / "42"), "model": "",
        "conversation_id": "", "execution_mode": "plan", "request_id": request_id,
    }, run_writer))
    await asyncio.wait_for(started.wait(), timeout=0.2)
    await worker._cancel_request({"request_id": request_id}, cancel_writer)
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancel_writer.events == [{"type": "cancel", "cancelled": True}]
    assert request_id not in worker._ACTIVE_REQUESTS


async def test_worker_rejects_mode_not_allowed_by_root_policy(monkeypatch, tmp_path: Path) -> None:
    controller_root = tmp_path / "controller-chats"
    worker_root = tmp_path / "worker-chats"
    (controller_root / "42").mkdir(parents=True)
    (worker_root / "42").mkdir(parents=True)
    monkeypatch.setenv("AGY_BRIDGE_CONTROLLER_CHATS_ROOT", str(controller_root))
    monkeypatch.setenv("AGY_BRIDGE_CHATS_ROOT", str(worker_root))

    class Policy:
        def for_execution_mode(self, requested: str):
            raise worker.AccessPolicyError(f"blocked {requested}")

    class Writer:
        def __init__(self) -> None:
            self.events: list[dict[str, object]] = []

        def write(self, raw: bytes) -> None:
            self.events.append(json.loads(raw))

        async def drain(self) -> None:
            pass

    monkeypatch.setattr(worker, "load_access_policy", lambda: Policy())
    writer = Writer()
    await worker._run_request({
        "prompt": "try", "chat_dir": str(controller_root / "42"), "model": "",
        "conversation_id": "", "execution_mode": "code", "request_id": "c" * 24,
    }, writer)
    assert writer.events == [{"type": "error"}]


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
