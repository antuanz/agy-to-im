"""Validation tests for the root-supervised worker boundary."""
from __future__ import annotations

from pathlib import Path

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
