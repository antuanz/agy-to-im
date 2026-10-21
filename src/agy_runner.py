"""Asynchronous stream-JSON runner for the Antigravity CLI (``agy``)."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from src.access import AccessPolicy, load_access_policy
from src.processes import (
    process_group_options as _process_group_options,
    terminate_process_group as _terminate_process_group,
)
from src.state import normalize_conversation_id

_STDOUT_CAP_BYTES = 524_288
_STDERR_CAP_BYTES = 65_536
_STREAM_STDOUT_CAP_BYTES = 1_048_576
_STREAM_LINE_CAP_BYTES = 524_288
_FINAL_RESPONSE_CAP_BYTES = 524_288
_SUCCESS = "SUCCESS"

OutputCallback = Callable[[str], Awaitable[None]]


@dataclass(frozen=True)
class AgyResult:
    text: str
    exit_code: int
    stderr: str
    status: str = ""
    conversation_id: str = ""


def _build_args(
    *,
    agy_path: str,
    prompt: str,
    conversation_id: str,
    model: str,
    mode: str,
    print_timeout: str,
    chat_dir: str = "",
    access: AccessPolicy | None = None,
) -> list[str]:
    """Build argv; the prompt is intentionally sent through stdin, not argv."""
    del prompt, chat_dir, mode
    policy = access or load_access_policy()
    args = [
        shutil.which(agy_path) or agy_path,
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
    ]
    # Stream-json owns its stdin session. Do not combine it with --new-project
    # or --continue; a later process resumes a chat only through the documented
    # opaque --conversation ID returned by the prior terminal result.
    if conversation_id:
        args.extend(["--conversation", conversation_id])
    if model:
        args.extend(["--model", model])
    if policy.permits_unattended_tools:
        # Host access is a server-side decision. It still runs as the unprivileged
        # agy account; it is never a root, sudo, or Docker capability.
        args.append("--dangerously-skip-permissions")
        args.extend(["--mode", "accept-edits"])
    else:
        # The production worker already has a root-created systemd filesystem
        # boundary. This VM blocks unprivileged user namespaces, so agy's
        # --sandbox flag exits before a turn can start. Plan mode remains
        # server-enforced and Telegram cannot add unattended approval.
        args.extend(["--mode", "plan"])
    args.extend(["--print-timeout", print_timeout])
    return args


def _child_environment() -> dict[str, str]:
    """Give agy only the runtime it needs; never inherit bot credentials."""
    allowed = (
        "HOME", "LANG", "LC_ALL", "PATH", "XDG_RUNTIME_DIR",
        "DBUS_SESSION_BUS_ADDRESS",
    )
    child = {key: os.environ[key] for key in allowed if os.environ.get(key)}
    child["PATH"] = child.get("PATH", "/usr/local/bin:/usr/bin:/bin")
    child["LANG"] = child.get("LANG", "C.UTF-8")
    child["AGY_CLI_DISABLE_AUTO_UPDATE"] = "true"
    return child


async def _read_stderr(stream: asyncio.StreamReader) -> tuple[str, bool]:
    captured = bytearray()
    exceeded = False
    while True:
        block = await stream.read(8192)
        if not block:
            break
        room = _STDERR_CAP_BYTES - len(captured)
        if room > 0:
            captured.extend(block[:room])
        if len(block) > room:
            exceeded = True
    return bytes(captured).decode("utf-8", errors="replace"), exceeded


@dataclass(frozen=True)
class _StreamEvents:
    deltas: str
    terminal: dict[str, object] | None
    error: str = ""


async def _read_events(
    stream: asyncio.StreamReader,
    on_output: OutputCallback | None,
) -> _StreamEvents:
    pieces: list[str] = []
    captured_bytes = 0
    terminal: dict[str, object] | None = None
    total_bytes = 0
    error = ""
    while True:
        try:
            raw = await stream.readline()
        except ValueError:
            # StreamReader discards an over-limit line before raising. Continue
            # draining the process but never treat its output as trustworthy.
            error = error or "agy stream-json line exceeded size limit"
            continue
        if not raw:
            break
        total_bytes += len(raw)
        if total_bytes > _STREAM_STDOUT_CAP_BYTES:
            error = error or "agy stream-json stdout exceeded size limit"
            continue
        if len(raw) > _STREAM_LINE_CAP_BYTES:
            error = error or "agy stream-json line exceeded size limit"
            continue
        try:
            event = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            error = error or "agy emitted malformed stream-json"
            continue
        if not isinstance(event, dict):
            error = error or "agy emitted an invalid stream-json event"
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
                error = error or "agy response deltas exceeded size limit"
                continue
            accepted = encoded[:room].decode("utf-8", errors="ignore")
            if len(encoded) > room:
                error = error or "agy response deltas exceeded size limit"
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
                if terminal is not None:
                    error = error or "agy emitted multiple terminal results"
                terminal = result
                # A stream-json process remains alive to accept another user
                # event.  The caller must receive this turn's terminal result
                # before it closes stdin; waiting for EOF here would deadlock
                # a one-turn bridge invocation.
                return _StreamEvents("".join(pieces), terminal, error)
            else:
                error = error or "agy emitted an invalid terminal result"
    return _StreamEvents("".join(pieces), terminal, error)


async def run_agy(
    prompt: str,
    *,
    chat_dir: str,
    conversation_id: str,
    model: str,
    mode: str,
    agy_path: str,
    timeout: float | None = None,
    print_timeout: str = "15m",
    on_output: OutputCallback | None = None,
    access: AccessPolicy | None = None,
) -> AgyResult:
    """Run one agy turn and return only streamed agent-response text."""
    args = _build_args(
        agy_path=agy_path,
        prompt=prompt,
        conversation_id=conversation_id,
        model=model,
        mode=mode,
        print_timeout=print_timeout,
        chat_dir=chat_dir,
        access=access,
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=chat_dir,
            env=_child_environment(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=_STREAM_LINE_CAP_BYTES + 1,
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
    event_task = asyncio.create_task(_read_events(proc.stdout, on_output))
    stderr_task = asyncio.create_task(_read_stderr(proc.stderr))
    wait_task = asyncio.create_task(proc.wait())
    try:
        # Keep stdin open until the terminal result for this turn.  In the
        # pinned CLI, stream-json denotes a continuous session: closing stdin
        # immediately after writing makes the process wait indefinitely on
        # some supported releases instead of completing the submitted turn.
        proc.stdin.write(payload)
        await proc.stdin.drain()
        events = await asyncio.wait_for(asyncio.shield(event_task), timeout=timeout)
    except asyncio.TimeoutError:
        await _terminate_process_group(proc)
        await asyncio.gather(event_task, stderr_task, wait_task, return_exceptions=True)
        return AgyResult("", 124, f"agy timed out after {timeout}s", "TIMEOUT")
    except asyncio.CancelledError:
        await _terminate_process_group(proc)
        await asyncio.gather(event_task, stderr_task, wait_task, return_exceptions=True)
        raise

    # A result ends this bridge turn, not the CLI's stream-json session.  Close
    # its input only now, then give it a short bounded grace period to exit.
    # If it keeps the session open, terminate its process group after the
    # authoritative terminal result has been received.
    try:
        proc.stdin.close()
        await proc.stdin.wait_closed()
    except (BrokenPipeError, ConnectionResetError):
        pass
    forced_shutdown = False
    try:
        await asyncio.wait_for(asyncio.shield(wait_task), timeout=3.0)
    except asyncio.TimeoutError:
        forced_shutdown = True
        await _terminate_process_group(proc)
        await asyncio.gather(wait_task, return_exceptions=True)
    stderr_result = await stderr_task
    exit_code = 0 if forced_shutdown and events.terminal is not None else proc.returncode

    stderr, stderr_exceeded = stderr_result
    terminal = events.terminal
    if stderr_exceeded:
        return AgyResult(events.deltas, exit_code or 1, _join_errors(stderr, "agy stderr exceeded size limit"), "INVALID")
    if events.error:
        return AgyResult(events.deltas, exit_code or 1, _join_errors(stderr, events.error), "INVALID")
    if terminal is None:
        detail = "agy stream ended without a terminal result"
        return AgyResult(events.deltas, exit_code or 1, _join_errors(stderr, detail), "INVALID")
    status_value = terminal.get("status")
    status = status_value if isinstance(status_value, str) else "INVALID"
    if status != _SUCCESS or exit_code != 0:
        error_value = terminal.get("error")
        error = error_value if isinstance(error_value, str) else ""
        return AgyResult(events.deltas, exit_code or 1, _join_errors(stderr, error), status)
    denied_actions = terminal.get("denied_actions", [])
    if not isinstance(denied_actions, list) or denied_actions:
        return AgyResult(events.deltas, 1, _join_errors(stderr, "agy result contains denied actions"), "DENIED")
    response = terminal.get("response")
    if not isinstance(response, str):
        return AgyResult(events.deltas, 1, _join_errors(stderr, "agy result has no valid response"), "INVALID")
    if len(response.encode("utf-8")) > _FINAL_RESPONSE_CAP_BYTES:
        return AgyResult(events.deltas, 1, _join_errors(stderr, "agy result response exceeded size limit"), "INVALID")
    return AgyResult(
        response, 0, stderr, status,
        normalize_conversation_id(terminal.get("conversation_id")),
    )


def _join_errors(*values: str) -> str:
    return "\n".join(value.strip() for value in values if value.strip())
