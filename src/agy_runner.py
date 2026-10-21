"""Asynchronous stream-JSON runner for the Antigravity CLI (``agy``)."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from src.processes import (
    process_group_options as _process_group_options,
    terminate_process_group as _terminate_process_group,
)

_STDOUT_CAP_BYTES = 524_288
_STDERR_CAP_BYTES = 65_536
_SUCCESS = "SUCCESS"

OutputCallback = Callable[[str], Awaitable[None]]


@dataclass(frozen=True)
class AgyResult:
    text: str
    exit_code: int
    stderr: str
    status: str = ""


def _build_args(
    *,
    agy_path: str,
    prompt: str,
    has_session: bool,
    model: str,
    mode: str,
    print_timeout: str,
    chat_dir: str = "",
) -> list[str]:
    """Build argv; the prompt is intentionally sent through stdin, not argv."""
    del prompt, chat_dir
    args = [
        shutil.which(agy_path) or agy_path,
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
    ]
    args.append("--continue" if has_session else "--new-project")
    if model:
        args.extend(["--model", model])
    args.append("--dangerously-skip-permissions")
    if mode == "plan":
        args.extend(["--mode", "plan", "--sandbox"])
    elif mode == "code":
        args.extend(["--mode", "accept-edits"])
    args.extend(["--print-timeout", print_timeout])
    return args


async def _read_stderr(stream: asyncio.StreamReader) -> str:
    captured = bytearray()
    while True:
        block = await stream.read(8192)
        if not block:
            break
        if len(captured) < _STDERR_CAP_BYTES:
            captured.extend(block[: _STDERR_CAP_BYTES - len(captured)])
    return bytes(captured).decode("utf-8", errors="replace")


async def _read_events(
    stream: asyncio.StreamReader,
    on_output: OutputCallback | None,
) -> tuple[str, dict[str, object] | None]:
    pieces: list[str] = []
    captured_bytes = 0
    terminal: dict[str, object] | None = None
    truncated = False
    while True:
        raw = await stream.readline()
        if not raw:
            break
        try:
            event = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(event, dict):
            continue
        if event.get("event") == "step_update":
            update = event.get("step_update")
            if not isinstance(update, dict) or update.get("step_type") != "agent_response":
                continue
            delta = update.get("text_delta")
            if not isinstance(delta, str) or not delta:
                continue
            encoded = delta.encode("utf-8")
            room = _STDOUT_CAP_BYTES - captured_bytes
            if room <= 0:
                truncated = True
                continue
            accepted = encoded[:room].decode("utf-8", errors="ignore")
            if len(encoded) > room:
                truncated = True
            if accepted:
                pieces.append(accepted)
                captured_bytes += len(accepted.encode("utf-8"))
                if on_output is not None:
                    try:
                        await on_output("".join(pieces))
                    except Exception:
                        # Draft publication is best-effort and must not affect agy.
                        pass
        elif event.get("event") == "result":
            result = event.get("result")
            if isinstance(result, dict):
                terminal = result
    text = "".join(pieces)
    if truncated:
        text += "\n\n[output truncated at 512 KiB]"
    return text, terminal


async def run_agy(
    prompt: str,
    *,
    chat_dir: str,
    has_session: bool,
    model: str,
    mode: str,
    agy_path: str,
    timeout: float | None = None,
    print_timeout: str = "15m",
    on_output: OutputCallback | None = None,
) -> AgyResult:
    """Run one agy turn and return only streamed agent-response text."""
    args = _build_args(
        agy_path=agy_path,
        prompt=prompt,
        has_session=has_session,
        model=model,
        mode=mode,
        print_timeout=print_timeout,
        chat_dir=chat_dir,
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=chat_dir,
            env=os.environ.copy(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=_STDOUT_CAP_BYTES + 65_536,
            **_process_group_options(),
        )
    except OSError as exc:
        return AgyResult("", 1, str(exc), "ERROR")

    assert proc.stdin is not None
    assert proc.stdout is not None
    assert proc.stderr is not None
    payload = json.dumps(
        {"event": "user", "message": {"content": prompt}},
        ensure_ascii=False,
    ).encode("utf-8") + b"\n"
    try:
        try:
            proc.stdin.write(payload)
            await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            proc.stdin.close()
            try:
                await proc.stdin.wait_closed()
            except (BrokenPipeError, ConnectionResetError):
                pass
    except asyncio.CancelledError:
        await _terminate_process_group(proc)
        raise

    event_task = asyncio.create_task(_read_events(proc.stdout, on_output))
    stderr_task = asyncio.create_task(_read_stderr(proc.stderr))
    wait_task = asyncio.create_task(proc.wait())

    async def _finish() -> tuple[tuple[str, dict[str, object] | None], str, int]:
        events, stderr, exit_code = await asyncio.gather(
            event_task, stderr_task, wait_task
        )
        return events, stderr, exit_code

    finish_task = asyncio.create_task(_finish())
    try:
        events, stderr, exit_code = await asyncio.wait_for(
            asyncio.shield(finish_task), timeout=timeout
        )
    except asyncio.TimeoutError:
        await _terminate_process_group(proc)
        await asyncio.gather(finish_task, return_exceptions=True)
        return AgyResult("", 124, f"agy timed out after {timeout}s", "TIMEOUT")
    except asyncio.CancelledError:
        await _terminate_process_group(proc)
        await asyncio.gather(finish_task, return_exceptions=True)
        raise

    text, terminal = events
    if terminal is None:
        detail = "agy stream ended without a terminal result"
        return AgyResult(text, exit_code or 1, _join_errors(stderr, detail), "INVALID")
    status_value = terminal.get("status")
    status = status_value if isinstance(status_value, str) else "INVALID"
    if status != _SUCCESS or exit_code != 0:
        error_value = terminal.get("error")
        error = error_value if isinstance(error_value, str) else ""
        return AgyResult(text, exit_code or 1, _join_errors(stderr, error), status)
    return AgyResult(text, 0, stderr, status)


def _join_errors(*values: str) -> str:
    return "\n".join(value.strip() for value in values if value.strip())
