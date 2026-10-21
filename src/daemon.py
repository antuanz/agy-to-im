"""Long-running daemon: poll Telegram, route through agy, reply."""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

from src.agy_runner import AgyResult
from src.commands import (
    BridgeReply,
    effective_model_value,
    handle_callback,
    handle_text_command,
    is_cancel_command,
)
from src.config import Config, load_config
from src.health import memory_healthy, record_error, record_turn, start_server
from src.media import build_media_prompt, clean_inbox
from src.models import ModelCatalog, ModelSnapshot
from src.worker_client import discover_worker_models
from src.okf_memory import attach_memory
from src.output_files import discover_new_outputs, snapshot_workspace_files
from src.queue import TurnQueue
from src.state import ChatState, State, load_state, save_state
from src.telegram import (
    SAFE_COMMAND_MENU, CallbackQuery, InboundMessage, TelegramClient, is_authorized,
)
from src.turn import CLOSED_CONTEXT_BLOCKED_EXIT_CODE, execute_agy
from src.webhook import drain_webhook_updates, start_webhook_server

LOG = logging.getLogger("antigravity_telegram_bridge")
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.json"
DEFAULT_STATE_PATH = Path.home() / ".antigravity" / "bridge" / "state.json"
DEFAULT_CHATS_ROOT = Path.home() / ".antigravity" / "bridge" / "chats"
DEFAULT_AGY_BIN = "agy"
LONG_POLL_TIMEOUT = 30
AGY_TIMEOUT_S = 900.0
_QUEUE = TurnQueue()
_TURN_RUNNER: asyncio.Task[None] | None = None
_ACTIVE_TURN: "_ActiveTurn | None" = None
PROGRESS_UPDATE_SECONDS = 10.0


class _TelegramLike(Protocol):
    async def __aenter__(self) -> "_TelegramLike": ...
    async def __aexit__(self, *exc: object) -> None: ...
    async def get_me(self) -> dict[str, Any]: ...
    async def get_updates(self, offset: int, timeout: int = 30) -> list[dict[str, Any]]: ...
    async def set_my_commands(
        self, commands: Any, *, scope: dict[str, int | str],
    ) -> None: ...
    async def send_message(
        self, chat_id: int, text: str, *, keyboard: Any | None = None
    ) -> int | None: ...
    async def send_rich_message(self, chat_id: int, markdown: str) -> int | None: ...
    async def send_rich_message_draft(
        self, chat_id: int, draft_id: int, markdown: str
    ) -> None: ...
    async def edit_message_text(
        self, chat_id: int, message_id: int, text: str, *, keyboard: Any | None = None
    ) -> None: ...
    async def delete_message(self, chat_id: int, message_id: int) -> None: ...
    async def answer_callback_query(self, callback_query_id: str, *, text: str = "") -> None: ...
    async def send_chat_action(self, chat_id: int, action: str = "typing") -> None: ...
    async def send_document(self, chat_id: int, path: Path, filename: str) -> None: ...
    async def send_photo(self, chat_id: int, path: Path, filename: str) -> None: ...


RunAgyFunc = Callable[..., Awaitable[AgyResult]]


@dataclass(frozen=True)
class DaemonInfo:
    bot_username: str
    started_at: float
    agy_version: str


@dataclass
class _ActiveTurn:
    chat_id: int
    task: asyncio.Task[None]


class _TurnProgress:
    """A bounded visible status while a turn has no response text yet."""

    def __init__(self, tg: _TelegramLike, chat_id: int) -> None:
        self._tg = tg
        self._chat_id = chat_id
        self._started_at = time.monotonic()
        self._message_id: int | None = None
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        try:
            self._message_id = await self._tg.send_message(
                self._chat_id,
                "⏳ Thinking… (0s)\nSend /cancel to stop this request.",
            )
        except Exception:
            return
        if self._message_id is not None:
            self._task = asyncio.create_task(self._refresh())

    async def close(self, *, cancelled: bool = False) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        if self._message_id is None:
            return
        try:
            if cancelled:
                await self._tg.edit_message_text(
                    self._chat_id, self._message_id, "🛑 Request cancelled."
                )
            else:
                await self._tg.delete_message(self._chat_id, self._message_id)
        except Exception:
            # Progress is a convenience; it must not delay or suppress a reply.
            pass

    def _text(self) -> str:
        elapsed = int(time.monotonic() - self._started_at)
        return f"⏳ Thinking… ({elapsed}s)\nSend /cancel to stop this request."

    async def _refresh(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    await asyncio.wait_for(self._stop.wait(), PROGRESS_UPDATE_SECONDS)
                    return
                except asyncio.TimeoutError:
                    pass
                if self._message_id is not None:
                    try:
                        await self._tg.edit_message_text(
                            self._chat_id, self._message_id, self._text()
                        )
                    except Exception:
                        pass
        except asyncio.CancelledError:
            return


def _resolve_agy_path() -> str:
    return os.environ.get("AGY_BIN") or DEFAULT_AGY_BIN


def _resolve_chats_root(cfg: Config) -> Path:
    if cfg.agy.chats_root:
        return Path(cfg.agy.chats_root).expanduser()
    return DEFAULT_CHATS_ROOT


def _detect_agy_version(agy_path: str) -> str:
    if os.environ.get("AGY_BRIDGE_WORKER_SOCKET"):
        # Only the isolated worker is allowed to execute the CLI in production.
        return "managed worker"
    try:
        out = subprocess.run(
            [agy_path, "--version"],
            capture_output=True, text=True, check=False, timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception as err:
        LOG.warning("agy --version failed: %s", err)
    return "unknown"


def _ensure_chat_state(state: State, chat_id: int, chats_root: Path) -> ChatState:
    cs = state.chats.get(chat_id)
    if cs is not None:
        Path(cs.chat_dir).mkdir(parents=True, exist_ok=True)
        return cs
    chat_dir = chats_root / str(chat_id)
    chat_dir.mkdir(parents=True, exist_ok=True)
    cs = ChatState(chat_dir=str(chat_dir))
    state.chats[chat_id] = cs
    return cs


def _format_timeout_reply() -> str:
    minutes = AGY_TIMEOUT_S / 60
    return (
        f"⏱️ agy turn cut off at {minutes:.0f} min.\n\n"
        "• Send a shorter follow-up\n"
        "• Split into smaller steps\n"
        "• /reset to start fresh"
    )


async def _do_turn(
    msg: InboundMessage, tg: _TelegramLike, cs: ChatState,
    cfg: Config, agy_path: str, catalog: ModelCatalog,
) -> None:
    prompt = await build_media_prompt(msg, tg, cs, cfg)
    if prompt is None:
        return
    selected_model = effective_model_value(cs, cfg)
    if selected_model:
        snapshot = await catalog.get()
        if snapshot.error and not snapshot.models:
            await tg.send_message(
                msg.chat_id,
                f"⚠️ Cannot validate saved model {selected_model!r}: "
                f"{snapshot.error}. No agy turn was started.",
            )
            return
        if selected_model not in snapshot.models:
            await tg.send_message(
                msg.chat_id,
                f"⚠️ Saved model {selected_model!r} is no longer available. "
                "Use /model to choose another model or Use agy default.",
            )
            return
    try:
        await tg.send_chat_action(msg.chat_id, "typing")
    except Exception:
        pass
    progress = _TurnProgress(tg, msg.chat_id)
    await progress.start()
    # Capture the state after inbound media has been put in .bridge-inbox, so
    # an uploaded input cannot be renamed and returned as a new agent output.
    output_before = snapshot_workspace_files(cs.chat_dir)
    execution_msg = replace(msg, text=prompt)
    try:
        text, code = await execute_agy(
            tg, msg.chat_id, execution_msg, cs, cfg, agy_path,
        )
    except asyncio.CancelledError:
        await progress.close(cancelled=True)
        return
    await progress.close()
    if code == CLOSED_CONTEXT_BLOCKED_EXIT_CODE:
        # A denied Sandbox/Workspace tool action is an expected policy outcome,
        # not an operational error to expose as a raw agy exit code.
        reply = text
    elif code == 124:
        record_error()
        reply = _format_timeout_reply()
    elif code != 0:
        record_error()
        reply = f"⚠️ agy error (exit {code})"
    else:
        record_turn()
        cs.turn_count += 1
        reply = text or "(empty reply)"
    outputs = discover_new_outputs(cs.chat_dir, output_before) if code == 0 else None
    # Post-turn maintenance: update OKF memory layer, clean inbox
    try:
        attach_memory(cs.chat_dir, None, bridge_version="0.2.0")
        clean_inbox(cs.chat_dir)
    except Exception:
        pass
    try:
        if code == 0:
            await tg.send_rich_message(msg.chat_id, reply)
        else:
            await tg.send_message(msg.chat_id, reply)
    except Exception as err:
        LOG.error("sendMessage failed chat=%s: %s", msg.chat_id, err)
    if outputs is not None:
        LOG.info(
            "generated output scan chat=%s files=%s skipped=%s",
            msg.chat_id, len(outputs.files), outputs.skipped,
        )
        delivery_failed = False
        for output in outputs.files:
            try:
                if output.kind == "photo":
                    await tg.send_photo(msg.chat_id, output.path, output.filename)
                else:
                    await tg.send_document(msg.chat_id, output.path, output.filename)
            except Exception:
                delivery_failed = True
                LOG.warning("generated output delivery failed chat=%s kind=%s", msg.chat_id, output.kind)
        if outputs.skipped or delivery_failed:
            try:
                await tg.send_message(
                    msg.chat_id,
                    "⚠️ Some generated files were not sent because they exceed delivery limits or failed a safety check.",
                )
            except Exception:
                LOG.warning("generated output warning failed chat=%s", msg.chat_id)


async def _run_turns(
    first: InboundMessage, tg: _TelegramLike, state: State,
    state_path: Path, chats_root: Path, cfg: Config,
    agy_path: str, catalog: ModelCatalog,
) -> None:
    """Run the active turn and then drain the FIFO without blocking polling."""
    global _ACTIVE_TURN, _TURN_RUNNER
    current = first
    current_future: asyncio.Future[str | None] | None = None
    try:
        while True:
            cs = _ensure_chat_state(state, current.chat_id, chats_root)
            task = asyncio.create_task(_do_turn(current, tg, cs, cfg, agy_path, catalog))
            _ACTIVE_TURN = _ActiveTurn(current.chat_id, task)
            try:
                await task
            except asyncio.CancelledError:
                # A user cancellation is an expected terminal outcome. The
                # active task either updates its progress message or the
                # /cancel acknowledgement tells the user what happened.
                pass
            except Exception as exc:
                LOG.exception("turn crashed chat=%s", current.chat_id)
                if current_future is not None and not current_future.done():
                    current_future.set_exception(exc)
            else:
                if current_future is not None and not current_future.done():
                    current_future.set_result(None)
            finally:
                _ACTIVE_TURN = None
                _QUEUE.complete()
            next_item = _QUEUE.next()
            if next_item is None:
                break
            current, current_future = next_item
    finally:
        _ACTIVE_TURN = None
        _TURN_RUNNER = None
        save_state(state_path, state)


async def _cancel_active_turn(msg: InboundMessage, tg: _TelegramLike) -> None:
    active = _ACTIVE_TURN
    if active is None or active.chat_id != msg.chat_id or active.task.done():
        await tg.send_message(msg.chat_id, "There is no active request in this chat.")
        return
    active.task.cancel()
    await tg.send_message(msg.chat_id, "🛑 Cancelling the active request…")


async def _process_text(
    msg: InboundMessage, tg: _TelegramLike, state: State,
    state_path: Path, chats_root: Path, cfg: Config,
    agy_path: str, info: DaemonInfo, catalog: ModelCatalog,
) -> None:
    if not is_authorized(msg, cfg.telegram):
        LOG.info("drop unauth user=%s", msg.user_id)
        return

    if is_cancel_command(msg.text):
        await _cancel_active_turn(msg, tg)
        return

    cs = _ensure_chat_state(state, msg.chat_id, chats_root)
    save_state(state_path, state)

    snapshot = ModelSnapshot(())
    command = msg.text.strip().split(maxsplit=1)
    if command and command[0].lower() == "/model":
        snapshot = await catalog.get()
    reply = await handle_text_command(
        msg, cs, cfg, snapshot.models, snapshot.error
    )
    if reply is not None:
        save_state(state_path, state)
        try:
            await tg.send_message(msg.chat_id, reply.text, keyboard=reply.keyboard)
        except Exception as err:
            LOG.error("sendMessage failed chat=%s: %s", msg.chat_id, err)
        return

    status = await _QUEUE.submit(msg)
    if status is not None:
        await tg.send_message(msg.chat_id, status)
        return
    global _TURN_RUNNER
    _TURN_RUNNER = asyncio.create_task(
        _run_turns(msg, tg, state, state_path, chats_root, cfg, agy_path, catalog)
    )


async def _process_callback(
    cq: CallbackQuery, tg: _TelegramLike, state: State,
    state_path: Path, chats_root: Path, cfg: Config, info: DaemonInfo,
    catalog: ModelCatalog,
) -> None:
    if not is_authorized(
        InboundMessage(update_id=cq.update_id, chat_id=cq.chat_id,
                       user_id=cq.user_id, text=""),
        cfg.telegram,
    ):
        LOG.info("drop unauth callback user=%s", cq.user_id)
        await tg.answer_callback_query(cq.callback_query_id, text="Not authorized")
        return

    cs = _ensure_chat_state(state, cq.chat_id, chats_root)
    snapshot = ModelSnapshot(())
    if cq.data == "nav:model" or cq.data.startswith("m:"):
        snapshot = await catalog.get()
    reply = handle_callback(cq, cs, cfg, snapshot.models, snapshot.error)
    save_state(state_path, state)

    try:
        await tg.answer_callback_query(cq.callback_query_id, text=reply.toast)
    except Exception:
        pass
    try:
        await tg.edit_message_text(
            cq.chat_id, cq.message_id, reply.text, keyboard=reply.keyboard
        )
    except Exception as err:
        LOG.warning("editMessageText failed (sending fresh): %s", err)
        try:
            await tg.send_message(cq.chat_id, reply.text, keyboard=reply.keyboard)
        except Exception as err2:
            LOG.error("fallback sendMessage failed: %s", err2)


async def _fetch_updates(tg: _TelegramLike, offset: int) -> list[dict[str, Any]]:
    try:
        return await asyncio.wait_for(
            tg.get_updates(offset=offset, timeout=LONG_POLL_TIMEOUT),
            timeout=LONG_POLL_TIMEOUT + 5,
        )
    except asyncio.TimeoutError:
        return []
    except Exception as err:
        LOG.warning("getUpdates failed: %s", err)
        await asyncio.sleep(2)
        return []


async def _register_command_menu(tg: _TelegramLike, cfg: Config) -> None:
    """Expose only safe commands to each explicitly allowlisted chat.

    Telegram does not permit the ``chat_member`` command scope in private
    chats.  ``chat`` is the narrowest valid scope there.  The bridge still
    authorizes each incoming message separately, so the menu grants nothing.
    """
    chat_ids = cfg.telegram.allowed_chat_ids
    if not chat_ids:
        # A global menu would not grant access, but exposing an interface on an
        # otherwise default-deny bot is unnecessary. The VM setup requires the
        # owner to configure chat allowlists before enabling the controller.
        LOG.info("command menu not registered: no allowed_chat_ids configured")
        return
    for chat_id in chat_ids:
        scope = {"type": "chat", "chat_id": chat_id}
        try:
            await tg.set_my_commands(SAFE_COMMAND_MENU, scope=scope)
        except Exception as err:
            # A command menu is convenience only; a temporary Bot API failure
            # must never stop the owner from using the bridge.
            LOG.warning("setMyCommands failed for configured chat: %s", err)


async def run(
    *, cfg: Config, state_path: Path, chats_root: Path,
    tg: _TelegramLike, agy_path: str, info: DaemonInfo,
    stop_event: asyncio.Event,
) -> None:
    _QUEUE.active = False
    _QUEUE.pending.clear()
    _QUEUE._limits.clear()
    _QUEUE.owner_chat_id = cfg.telegram.allowed_user_ids[0] if cfg.telegram.allowed_user_ids else 0
    _QUEUE.max_per_user = cfg.safety.queue.max_per_user
    _QUEUE.cooldown_seconds = cfg.safety.queue.cooldown_seconds
    chats_root.mkdir(parents=True, exist_ok=True)
    state = load_state(state_path, chats_root)
    worker_socket = os.environ.get("AGY_BRIDGE_WORKER_SOCKET", "")
    if worker_socket:
        async def discover() -> ModelSnapshot:
            models, error = await discover_worker_models(worker_socket)
            return ModelSnapshot(models, error)
        catalog = ModelCatalog(agy_path, discover=discover)
    else:
        catalog = ModelCatalog(agy_path)
    tick = 0
    async with tg:
        await _register_command_menu(tg, cfg)
        while not stop_event.is_set():
            tick += 1
            # Health gate: check memory every ~30 loop iterations (~30s at idle,
            # faster when processing bursts). Self-terminate if over limit so
            # systemd Restart=on-failure brings up a clean replacement.
            if tick % 30 == 0 and not memory_healthy():
                LOG.critical("memory limit exceeded — triggering self-restart")
                stop_event.set()
                break

            updates = await _fetch_updates(tg, state.last_update_id + 1)
            wh_updates = drain_webhook_updates()
            updates.extend(wh_updates)
            for upd in updates:
                state.last_update_id = max(state.last_update_id, int(upd.get("update_id", 0)))

                msg = parse_update_for_daemon(upd)
                if msg is not None:
                    allowed, wait = _QUEUE.check_ratelimit(msg.user_id)
                    if not allowed:
                        await tg.send_message(msg.chat_id, f"⏳ Rate limit. Wait {wait}s.")
                        continue
                    await _process_text(
                        msg, tg, state, state_path, chats_root,
                        cfg, agy_path, info, catalog,
                    )
                    continue

                cq = parse_callback_for_daemon(upd)
                if cq is not None:
                    await _process_callback(
                        cq, tg, state, state_path, chats_root, cfg, info, catalog,
                    )
                    continue

            if not updates:
                await asyncio.sleep(0)
            save_state(state_path, state)


# Aliases to avoid name collisions in the loop above
from src.telegram import parse_callback_query as parse_callback_for_daemon
from src.telegram import parse_update as parse_update_for_daemon


async def _detect_bot_username(cfg: Config) -> str:
    try:
        async with TelegramClient(cfg.telegram.bot_token) as tg:
            me = await tg.get_me()
            uname = me.get("username")
            return f"@{uname}" if isinstance(uname, str) and uname else "?"
    except Exception as err:
        LOG.warning("getMe failed: %s", err)
    return "?"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    cfg_path = Path(os.environ.get("AGY_BRIDGE_CONFIG") or DEFAULT_CONFIG_PATH)
    state_path = Path(os.environ.get("AGY_BRIDGE_STATE") or DEFAULT_STATE_PATH)
    cfg = load_config(cfg_path)
    chats_root = _resolve_chats_root(cfg)
    agy_path = _resolve_agy_path()
    stop = asyncio.Event()

    async def _go() -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, stop.set)

        health_srv = await start_server()

        wh_task: asyncio.Task | None = None
        wh_secret = ""
        wh_url = os.environ.get("CTI_WEBHOOK_URL", "")
        wh_port = int(os.environ.get("CTI_WEBHOOK_PORT") or "8080")

        tg = TelegramClient(cfg.telegram.bot_token)
        try:
            bot_username = await _detect_bot_username(cfg)
            info = DaemonInfo(
                bot_username=bot_username,
                started_at=time.time(),
                agy_version=_detect_agy_version(agy_path),
            )
            LOG.info(
                "bridge starting bot=%s agy=%s chats_root=%s mode=%s model=%s",
                info.bot_username, info.agy_version, chats_root,
                cfg.agy.mode, cfg.agy.model or "(default)",
            )

            if wh_url:
                wh_secret = cfg.telegram.bot_token[:20]
                wh_res = await tg.set_webhook(wh_url, wh_secret)
                LOG.info("webhook setup: %s", wh_res.get("description") if wh_res.get("ok") else wh_res)
                wh_task = asyncio.create_task(start_webhook_server(wh_port, tg, wh_secret))

            await run(
                cfg=cfg, state_path=state_path, chats_root=chats_root,
                tg=tg, agy_path=agy_path, info=info, stop_event=stop,
            )
        finally:
            if wh_task is not None:
                wh_task.cancel()
                try:
                    await wh_task
                except Exception:
                    pass
            try:
                await tg.delete_webhook()
            except Exception:
                pass
            health_srv.close()
            await health_srv.wait_closed()

    asyncio.run(_go())


if __name__ == "__main__":
    main()
