"""Tests for src.turn — agy turn execution."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from src.agy_runner import AgyResult
from src.config import AgyConfig, Config, TelegramConfig
from src.state import ChatState
from src.turn import _DraftPublisher, execute_agy


@dataclass(frozen=True)
class _FakeMsg:
    chat_id: int = 42
    text: str = "hello"
    chat_type: str = ""


class _FakeTG:
    def __init__(self) -> None:
        self.actions: list[tuple[int, str]] = []
        self.drafts: list[tuple[int, int, str]] = []
        self.fail_drafts = False

    async def send_chat_action(self, chat_id: int, action: str = "typing") -> None:
        self.actions.append((chat_id, action))

    async def send_rich_message_draft(
        self, chat_id: int, draft_id: int, markdown: str
    ) -> None:
        self.drafts.append((chat_id, draft_id, markdown))
        if self.fail_drafts:
            raise RuntimeError("draft rejected")


class _SlowDraftTG(_FakeTG):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def send_rich_message_draft(
        self, chat_id: int, draft_id: int, markdown: str
    ) -> None:
        self.drafts.append((chat_id, draft_id, markdown))
        self.started.set()
        await self.release.wait()


async def test_execute_agy_returns_reply_and_records_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_run_agy(prompt: str, **kwargs: Any) -> AgyResult:
        calls.append({"prompt": prompt, **kwargs})
        await asyncio.sleep(0.01)
        return AgyResult(
            text="reply text", exit_code=0, stderr="",
            conversation_id="123e4567-e89b-12d3-a456-426614174000",
        )

    monkeypatch.setattr("src.turn.run_agy", fake_run_agy)

    tg = _FakeTG()
    cs = ChatState(chat_dir="/tmp/chat")
    cfg = Config(telegram=TelegramConfig(bot_token="t", allowed_user_ids=[42]), agy=AgyConfig())
    text, code = await execute_agy(tg, 42, _FakeMsg(text="hello"), cs, cfg, "/usr/bin/agy")

    assert code == 0
    assert text == "reply text"
    assert tg.actions
    assert calls[0]["prompt"] == "hello"
    assert calls[0]["chat_dir"] == "/tmp/chat"
    assert cs.conversation_id == "123e4567-e89b-12d3-a456-426614174000"
    assert cs.has_session is True


async def test_execute_agy_returns_timeout_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run_agy(**_: Any) -> AgyResult:
        return AgyResult(text="", exit_code=124, stderr="timeout")

    monkeypatch.setattr("src.turn.run_agy", fake_run_agy)

    tg = _FakeTG()
    cs = ChatState(chat_dir="/tmp/chat")
    cfg = Config(telegram=TelegramConfig(bot_token="t", allowed_user_ids=[42]), agy=AgyConfig())
    text, code = await execute_agy(tg, 42, _FakeMsg(), cs, cfg, "/usr/bin/agy")
    assert code == 124
    assert text == ""


async def test_private_chat_streams_drafts_without_typing_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_run_agy(**kwargs: Any) -> AgyResult:
        await kwargs["on_output"]("partial")
        await asyncio.sleep(0)
        return AgyResult("complete", 0, "", "SUCCESS")

    monkeypatch.setattr("src.turn.run_agy", fake_run_agy)
    tg = _FakeTG()
    cs = ChatState(chat_dir="/tmp/chat", model="")
    cfg = Config(
        telegram=TelegramConfig(bot_token="t", allowed_user_ids=[42]),
        agy=AgyConfig(model="configured-model"),
    )
    text, code = await execute_agy(
        tg, 42, _FakeMsg(chat_type="private"), cs, cfg, "/usr/bin/agy"
    )
    assert (text, code) == ("complete", 0)
    assert tg.actions == []
    assert tg.drafts[0][2] == "partial"


async def test_draft_publisher_throttles_and_keeps_one_id() -> None:
    tg = _FakeTG()
    publisher = _DraftPublisher(tg, 42, 777, interval=0.03)
    await publisher.publish("a")
    await asyncio.sleep(0.01)
    await publisher.publish("ab")
    await publisher.publish("abc")
    await asyncio.sleep(0.04)
    await publisher.close()
    assert tg.drafts == [(42, 777, "a"), (42, 777, "abc")]


async def test_draft_failure_does_not_escape() -> None:
    tg = _FakeTG()
    tg.fail_drafts = True
    publisher = _DraftPublisher(tg, 42, 888, interval=0)
    await publisher.publish("still running")
    await asyncio.sleep(0)
    await publisher.close()
    assert tg.drafts == [(42, 888, "still running")]


async def test_slow_draft_request_never_backpressures_publisher() -> None:
    tg = _SlowDraftTG()
    publisher = _DraftPublisher(tg, 42, 999, interval=0)
    await publisher.publish("first")
    await asyncio.wait_for(tg.started.wait(), timeout=0.2)
    start = asyncio.get_running_loop().time()
    await publisher.publish("second")
    assert asyncio.get_running_loop().time() - start < 0.05
    tg.release.set()
    await publisher.close()
