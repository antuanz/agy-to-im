"""Controller-side client for the isolated agy worker service."""
from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable

from src.agy_runner import AgyResult


_LINE_LIMIT = 1_048_576
OutputCallback = Callable[[str], Awaitable[None]]


class WorkerError(RuntimeError):
    """The server-managed worker could not accept or complete a job."""


async def run_worker_job(
    socket_path: str,
    *,
    prompt: str,
    chat_dir: str,
    has_session: bool,
    model: str,
    timeout: float,
    on_output: OutputCallback | None = None,
) -> AgyResult:
    """Run one job without giving the Telegram controller agy access."""
    try:
        reader, writer = await asyncio.open_unix_connection(socket_path, limit=_LINE_LIMIT)
    except OSError as exc:
        return AgyResult("", 1, "worker unavailable", "ERROR")
    try:
        request = {
            "op": "run", "prompt": prompt, "chat_dir": chat_dir,
            "has_session": has_session, "model": model,
        }
        writer.write(json.dumps(request, ensure_ascii=False).encode("utf-8") + b"\n")
        await writer.drain()

        async def receive() -> AgyResult:
            while True:
                raw = await reader.readline()
                if not raw:
                    raise WorkerError("worker connection closed before a result")
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise WorkerError("worker sent invalid data") from exc
                if not isinstance(event, dict):
                    raise WorkerError("worker sent invalid data")
                kind = event.get("type")
                if kind == "delta":
                    text = event.get("text")
                    if isinstance(text, str) and on_output is not None:
                        try:
                            await on_output(text)
                        except Exception:
                            pass
                    continue
                if kind == "result":
                    text = event.get("text")
                    exit_code = event.get("exit_code")
                    status = event.get("status")
                    if not isinstance(text, str) or not isinstance(exit_code, int) or not isinstance(status, str):
                        raise WorkerError("worker sent invalid result")
                    return AgyResult(text, exit_code, "", status)
                if kind == "error":
                    raise WorkerError("worker rejected the request")
                raise WorkerError("worker sent an unknown event")

        try:
            return await asyncio.wait_for(receive(), timeout=timeout)
        except asyncio.TimeoutError:
            return AgyResult("", 124, "worker timed out", "TIMEOUT")
        except WorkerError:
            return AgyResult("", 1, "worker protocol error", "ERROR")
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except OSError:
            pass


async def discover_worker_models(socket_path: str, *, timeout: float = 15.0) -> tuple[tuple[str, ...], str]:
    """Ask the worker to list models; it never exposes agy diagnostics."""
    try:
        reader, writer = await asyncio.open_unix_connection(socket_path, limit=_LINE_LIMIT)
    except OSError:
        return (), "worker unavailable"
    try:
        writer.write(b'{"op":"models"}\n')
        await writer.drain()
        try:
            raw = await asyncio.wait_for(reader.readline(), timeout=timeout)
        except asyncio.TimeoutError:
            return (), "worker model discovery timed out"
        try:
            event = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return (), "worker returned invalid model data"
        models = event.get("models") if isinstance(event, dict) else None
        error = event.get("error") if isinstance(event, dict) else None
        if not isinstance(models, list) or not all(isinstance(item, str) for item in models):
            return (), "worker returned invalid model data"
        if not models:
            return (), error if isinstance(error, str) and error else "worker returned no models"
        return tuple(models), ""
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except OSError:
            pass
