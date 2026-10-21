"""Tests for the async TelegramClient HTTP wrapper."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from pytest_httpx import HTTPXMock

from src.telegram import TelegramClient


pytestmark = pytest.mark.asyncio


async def test_get_updates_long_polls_with_offset_and_timeout(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(
        url="https://api.telegram.org/botTOKEN/getUpdates?offset=42&timeout=30&allowed_updates=%5B%22message%22%2C%22callback_query%22%5D",
        method="GET",
        json={"ok": True, "result": [{"update_id": 50}]},
    )
    async with TelegramClient("TOKEN") as tg:
        updates = await tg.get_updates(offset=42, timeout=30)
    assert updates == [{"update_id": 50}]


async def test_get_updates_raises_on_telegram_error(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="https://api.telegram.org/botTOKEN/getUpdates?offset=1&timeout=30&allowed_updates=%5B%22message%22%2C%22callback_query%22%5D",
        method="GET",
        json={"ok": False, "description": "Unauthorized"},
        status_code=401,
    )
    async with TelegramClient("TOKEN") as tg:
        with pytest.raises(RuntimeError, match="Unauthorized"):
            await tg.get_updates(offset=1)


async def test_send_message_splits_long_text(httpx_mock: HTTPXMock) -> None:
    body = "x" * 9000
    for _ in range(3):
        httpx_mock.add_response(
            url="https://api.telegram.org/botTOKEN/sendMessage",
            method="POST",
            json={"ok": True, "result": {}},
        )
    async with TelegramClient("TOKEN") as tg:
        await tg.send_message(chat_id=10, text=body)
    requests = httpx_mock.get_requests()
    assert len(requests) == 3
    for r in requests:
        assert r.method == "POST"
        assert r.url.path == "/botTOKEN/sendMessage"


async def test_send_message_skips_empty_text(httpx_mock: HTTPXMock) -> None:
    async with TelegramClient("TOKEN") as tg:
        await tg.send_message(chat_id=10, text="")
    assert httpx_mock.get_requests() == []


async def test_set_my_commands_posts_scoped_safe_menu(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="https://api.telegram.org/botTOKEN/setMyCommands",
        method="POST", json={"ok": True, "result": True},
    )
    commands = [{"command": "help", "description": "Show help"}]
    scope = {"type": "chat_member", "chat_id": 10, "user_id": 42}
    async with TelegramClient("TOKEN") as tg:
        await tg.set_my_commands(commands, scope=scope)
    payload = json.loads(httpx_mock.get_requests()[0].content)
    assert payload == {"commands": commands, "scope": scope}


async def test_send_chat_action_posts_typing(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="https://api.telegram.org/botTOKEN/sendChatAction",
        method="POST",
        json={"ok": True, "result": True},
    )
    async with TelegramClient("TOKEN") as tg:
        await tg.send_chat_action(chat_id=10, action="typing")
    req = httpx_mock.get_requests()[0]
    assert req.method == "POST"
    body = req.read().decode()
    assert "typing" in body
    assert "10" in body


async def test_send_rich_message_posts_safe_html(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="https://api.telegram.org/botTOKEN/sendRichMessage",
        method="POST",
        json={"ok": True, "result": {"message_id": 123}},
    )
    async with TelegramClient("TOKEN") as tg:
        message_id = await tg.send_rich_message(
            10, "# Heading\n\n![pixel](https://example.com/pixel.png) <b>raw</b>"
        )
    assert message_id == 123
    payload = json.loads(httpx_mock.get_requests()[0].content)
    rich = payload["rich_message"]
    assert "<h1>Heading</h1>" in rich["html"]
    assert "<img" not in rich["html"]
    assert "pixel.png" not in rich["html"]
    assert "&lt;b&gt;raw&lt;/b&gt;" in rich["html"]
    assert "media" not in rich


async def test_send_rich_message_falls_back_to_plain_text(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(
        url="https://api.telegram.org/botTOKEN/sendRichMessage",
        method="POST",
        json={"ok": False, "description": "can't parse rich message"},
        status_code=400,
    )
    httpx_mock.add_response(
        url="https://api.telegram.org/botTOKEN/sendMessage",
        method="POST",
        json={"ok": True, "result": {"message_id": 456}},
    )
    async with TelegramClient("TOKEN") as tg:
        message_id = await tg.send_rich_message(10, "**still readable**")
    assert message_id == 456
    requests = httpx_mock.get_requests()
    assert [request.url.path for request in requests] == [
        "/botTOKEN/sendRichMessage", "/botTOKEN/sendMessage"
    ]
    fallback = json.loads(requests[1].content)
    assert fallback["text"] == "**still readable**"


async def test_send_rich_draft_uses_stable_id_without_stop_button(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(
        url="https://api.telegram.org/botTOKEN/sendRichMessageDraft",
        method="POST",
        json={"ok": True, "result": True},
    )
    async with TelegramClient("TOKEN") as tg:
        await tg.send_rich_message_draft(10, 987654, "**partial**")
    payload = json.loads(httpx_mock.get_requests()[0].content)
    assert payload["draft_id"] == 987654
    assert "<strong>partial</strong>" in payload["rich_message"]["html"]
    assert "can_stop" not in payload


async def test_send_document_uploads_the_controller_approved_file(
    tmp_path: Path, httpx_mock: HTTPXMock,
) -> None:
    source = tmp_path / "report.md"
    source.write_text("# Report", encoding="utf-8")
    httpx_mock.add_response(
        url="https://api.telegram.org/botTOKEN/sendDocument",
        method="POST", json={"ok": True, "result": {}},
    )
    async with TelegramClient("TOKEN") as tg:
        await tg.send_document(10, source, "report.md")
    request = httpx_mock.get_requests()[0]
    assert request.url.path == "/botTOKEN/sendDocument"
    body = request.read().decode("utf-8", "replace")
    assert "report.md" in body
    assert "# Report" in body


async def test_send_photo_uploads_the_controller_approved_file(
    tmp_path: Path, httpx_mock: HTTPXMock,
) -> None:
    source = tmp_path / "chart.png"
    source.write_bytes(b"not a real png")
    httpx_mock.add_response(
        url="https://api.telegram.org/botTOKEN/sendPhoto",
        method="POST", json={"ok": True, "result": {}},
    )
    async with TelegramClient("TOKEN") as tg:
        await tg.send_photo(10, source, "chart.png")
    request = httpx_mock.get_requests()[0]
    assert request.url.path == "/botTOKEN/sendPhoto"
    assert "chart.png" in request.read().decode("utf-8", "replace")


async def test_rich_draft_uses_newest_chunk_when_answer_exceeds_limit(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(
        url="https://api.telegram.org/botTOKEN/sendRichMessageDraft",
        method="POST",
        json={"ok": True, "result": True},
    )
    text = "first chunk\n" + ("x" * 30_000) + "\nnewest tail"
    async with TelegramClient("TOKEN") as tg:
        await tg.send_rich_message_draft(10, 1, text)
    payload = json.loads(httpx_mock.get_requests()[0].content)
    html = payload["rich_message"]["html"]
    assert "newest tail" in html
    assert "first chunk" not in html
