"""Root-supervised, sandboxed agy worker with a narrow Unix-socket protocol."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import stat
from pathlib import Path
from typing import Any

from src.access import AccessPolicyError, load_access_policy
from src.agy_runner import AgyResult, _child_environment, run_agy
from src.models import MODEL_DISCOVERY_TIMEOUT_S, MODEL_STDERR_CAP_BYTES, MODEL_STDOUT_CAP_BYTES, _read_capped, parse_model_slugs
from src.processes import process_group_options, terminate_process_group
from src.state import is_valid_model


LOG = logging.getLogger("antigravity_telegram_worker")
_LINE_LIMIT = 1_048_576
_MAX_PROMPT_BYTES = 262_144
_WORKER_LOCK = asyncio.Lock()


def _socket_path() -> Path:
    raw = os.environ.get("AGY_BRIDGE_WORKER_SOCKET", "/run/agy-telegram-worker/worker.sock")
    path = Path(raw)
    if not path.is_absolute() or path.name != "worker.sock":
        raise RuntimeError("AGY_BRIDGE_WORKER_SOCKET must be an absolute worker.sock path")
    return path


def _chats_root() -> Path:
    return Path(os.environ.get("AGY_BRIDGE_CHATS_ROOT", "/home/agy/telegram-chats")).resolve()


def _controller_chats_root() -> Path:
    """Return the controller-side spelling of the shared chat mount.

    The controller runs in the normal host namespace and never mounts the
    agent's home. The worker translates only paths below this root to its own
    mountpoint, preventing a controller request from choosing an arbitrary cwd.
    """
    return Path(
        os.environ.get("AGY_BRIDGE_CONTROLLER_CHATS_ROOT", "/home/agy/telegram-chats")
    ).resolve()


def _valid_chat_dir(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        supplied = Path(value).resolve(strict=True)
        relative = supplied.relative_to(_controller_chats_root())
        path = (_chats_root() / relative).resolve(strict=True)
        path.relative_to(_chats_root())
    except (OSError, ValueError):
        return None
    return str(path)


def _valid_prompt(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    if len(value.encode("utf-8")) > _MAX_PROMPT_BYTES:
        return None
    return value


def _valid_model(value: object) -> str | None:
    if value == "":
        return ""
    return value if isinstance(value, str) and is_valid_model(value) else None


async def _send(writer: asyncio.StreamWriter, event: dict[str, object]) -> None:
    writer.write(json.dumps(event, ensure_ascii=False).encode("utf-8") + b"\n")
    await writer.drain()


async def _discover_models() -> tuple[tuple[str, ...], str]:
    agy_path = os.environ.get("AGY_BIN", "agy")
    try:
        proc = await asyncio.create_subprocess_exec(
            agy_path, "models", env=_child_environment(),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            limit=MODEL_STDOUT_CAP_BYTES + 1, **process_group_options(),
        )
    except OSError:
        return (), "worker could not run agy models"
    try:
        stdout_result, stderr_result, _ = await asyncio.wait_for(
            asyncio.gather(
                _read_capped(proc.stdout, MODEL_STDOUT_CAP_BYTES),
                _read_capped(proc.stderr, MODEL_STDERR_CAP_BYTES), proc.wait(),
            ), timeout=MODEL_DISCOVERY_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        await terminate_process_group(proc)
        return (), "worker model discovery timed out"
    stdout, stdout_exceeded = stdout_result
    _, stderr_exceeded = stderr_result
    if stdout_exceeded or stderr_exceeded or proc.returncode != 0:
        return (), "worker model discovery failed"
    models = parse_model_slugs(stdout.decode("utf-8", errors="replace"))
    return (models, "") if models else ((), "worker returned no models")


async def _run_request(request: dict[str, Any], writer: asyncio.StreamWriter) -> None:
    prompt = _valid_prompt(request.get("prompt"))
    chat_dir = _valid_chat_dir(request.get("chat_dir"))
    model = _valid_model(request.get("model"))
    if prompt is None or chat_dir is None or model is None or not isinstance(request.get("has_session"), bool):
        await _send(writer, {"type": "error"})
        return
    try:
        policy = load_access_policy()
    except AccessPolicyError:
        await _send(writer, {"type": "error"})
        return

    async def on_output(text: str) -> None:
        await _send(writer, {"type": "delta", "text": text})

    async with _WORKER_LOCK:
        result = await run_agy(
            prompt, chat_dir=chat_dir, has_session=request["has_session"], model=model,
            mode=policy.execution_mode, agy_path=os.environ.get("AGY_BIN", "agy"),
            timeout=900.0, on_output=on_output, access=policy,
        )
    # stderr may contain credentials, paths, or prompt fragments; never transfer it.
    await _send(writer, {
        "type": "result", "text": result.text, "exit_code": result.exit_code,
        "status": result.status,
    })


async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        raw = await reader.readline()
        if not raw or len(raw) > _LINE_LIMIT:
            return
        try:
            request = json.loads(raw)
        except json.JSONDecodeError:
            await _send(writer, {"type": "error"})
            return
        if not isinstance(request, dict):
            await _send(writer, {"type": "error"})
            return
        op = request.get("op")
        if op == "run":
            await _run_request(request, writer)
        elif op == "models":
            async with _WORKER_LOCK:
                models, error = await _discover_models()
            await _send(writer, {"type": "models", "models": list(models), "error": error})
        else:
            await _send(writer, {"type": "error"})
    except (ConnectionError, asyncio.IncompleteReadError):
        return
    except Exception:
        LOG.exception("worker request failed")
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except OSError:
            pass


async def _serve() -> None:
    path = _socket_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if not stat.S_ISSOCK(path.lstat().st_mode):
            raise RuntimeError("worker socket path exists and is not a socket")
        path.unlink()
    server = await asyncio.start_unix_server(_handle, path=str(path), limit=_LINE_LIMIT)
    # The unprivileged controller is in the dedicated agybridge group. It can
    # reach this socket, but it cannot read the agy home or execute the CLI.
    os.chmod(path, 0o660)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    async with server:
        await stop.wait()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(_serve())


if __name__ == "__main__":
    main()
