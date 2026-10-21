"""Protocol tests for the token-free isolated worker client."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from src.worker_client import discover_worker_models, run_worker_job


pytestmark = pytest.mark.skipif(os.name == "nt", reason="Unix-domain sockets require POSIX")


async def test_worker_job_streams_drafts_and_final_result(tmp_path: Path) -> None:
    socket = tmp_path / "worker.sock"

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        request = json.loads(await reader.readline())
        assert request["op"] == "run"
        assert request["conversation_id"] == ""
        assert request["execution_mode"] == "plan"
        for event in (
            {"type": "delta", "text": "hello"},
            {"type": "result", "text": "hello world", "exit_code": 0, "status": "SUCCESS"},
        ):
            writer.write(json.dumps(event).encode() + b"\n")
            await writer.drain()
        writer.close()

    server = await asyncio.start_unix_server(handler, path=str(socket))
    updates: list[str] = []

    async def on_output(text: str) -> None:
        updates.append(text)

    try:
        result = await run_worker_job(
            str(socket), prompt="hi", chat_dir=str(tmp_path), conversation_id="",
            model="", execution_mode="plan", timeout=1, on_output=on_output,
        )
    finally:
        server.close()
        await server.wait_closed()
    assert updates == ["hello"]
    assert result.text == "hello world"
    assert result.exit_code == 0


async def test_worker_model_error_is_not_silently_treated_as_empty(tmp_path: Path) -> None:
    socket = tmp_path / "worker.sock"

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        assert json.loads(await reader.readline()) == {"op": "models"}
        writer.write(b'{"type":"models","models":[],"error":"worker unavailable"}\n')
        await writer.drain()
        writer.close()

    server = await asyncio.start_unix_server(handler, path=str(socket))
    try:
        models, error = await discover_worker_models(str(socket), timeout=1)
    finally:
        server.close()
        await server.wait_closed()
    assert models == ()
    assert error == "worker unavailable"


async def test_worker_model_warning_is_preserved_with_stale_models(tmp_path: Path) -> None:
    socket = tmp_path / "worker.sock"

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        assert json.loads(await reader.readline()) == {"op": "models"}
        writer.write(
            b'{"type":"models","models":["model-a"],"error":"worker model discovery timed out"}\n'
        )
        await writer.drain()
        writer.close()

    server = await asyncio.start_unix_server(handler, path=str(socket))
    try:
        models, error = await discover_worker_models(str(socket), timeout=1)
    finally:
        server.close()
        await server.wait_closed()
    assert models == ("model-a",)
    assert error == "worker model discovery timed out"


async def test_worker_job_accepts_only_fixed_failure_reasons(tmp_path: Path) -> None:
    socket = tmp_path / "worker.sock"

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.readline()
        writer.write(
            b'{"type":"result","text":"","exit_code":1,"status":"ERROR",'
            b'"reason":"authentication unavailable"}\n'
        )
        await writer.drain()
        writer.close()

    server = await asyncio.start_unix_server(handler, path=str(socket))
    try:
        result = await run_worker_job(
            str(socket), prompt="hello", chat_dir="/tmp/chat", conversation_id="",
            model="", execution_mode="plan", timeout=1,
        )
    finally:
        server.close()
        await server.wait_closed()
    assert result.stderr == "authentication unavailable"


async def test_cancelling_a_worker_job_sends_its_opaque_cancel_request(tmp_path: Path) -> None:
    socket = tmp_path / "worker.sock"
    run_started = asyncio.Event()
    cancelled = asyncio.Event()
    seen_request_id = ""

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        nonlocal seen_request_id
        request = json.loads(await reader.readline())
        if request["op"] == "run":
            seen_request_id = request["request_id"]
            run_started.set()
            await reader.read()
        else:
            assert request == {"op": "cancel", "request_id": seen_request_id}
            cancelled.set()
            writer.write(b'{"type":"cancel","cancelled":true}\n')
            await writer.drain()
        writer.close()

    server = await asyncio.start_unix_server(handler, path=str(socket))
    try:
        task = asyncio.create_task(run_worker_job(
            str(socket), prompt="hi", chat_dir=str(tmp_path), conversation_id="",
            model="", execution_mode="plan", timeout=30,
        ))
        await asyncio.wait_for(run_started.wait(), timeout=0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(cancelled.wait(), timeout=0.2)
    finally:
        server.close()
        await server.wait_closed()
