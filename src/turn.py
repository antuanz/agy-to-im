"""Turn execution — agy print-mode invocation."""
from __future__ import annotations

import asyncio
import logging
import os
import secrets
import time
from typing import TYPE_CHECKING, Callable

from src.agy_runner import run_agy
from src.diagnostics import safe_failure_reason
from src.worker_client import run_worker_job
from src.commands import effective_model_value

if TYPE_CHECKING:
    from src.config import Config
    from src.daemon import _TelegramLike
    from src.state import ChatState
    from src.telegram import InboundMessage

LOG = logging.getLogger("antigravity_telegram_bridge")
AGY_TIMEOUT_S = 900.0


async def _compile_memory_layer(chat_dir: str) -> None:
    """Run graphify extract and export in the background to build the OKF memory vault."""
    try:
        import shutil
        graphify_bin = shutil.which("graphify") or "/home/i/.local/bin/graphify"
        # Run graphify extract . (AST + semantic codebase index)
        proc1 = await asyncio.create_subprocess_exec(
            graphify_bin, "extract", ".",
            cwd=chat_dir,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc1.wait()
        
        # Run graphify export okf --dir concepts (Google OKF v0.1 bundle)
        proc2 = await asyncio.create_subprocess_exec(
            graphify_bin, "export", "okf", "--dir", "concepts",
            cwd=chat_dir,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc2.wait()
        LOG.info("OKF-Graphify memory layer successfully updated for %s", chat_dir)
    except Exception as exc:
        LOG.warning("Failed to update OKF-Graphify memory layer: %s", exc)

async def execute_agy(
    tg: "_TelegramLike", chat_id: int, msg: "InboundMessage",
    cs: "ChatState", cfg: "Config", agy_path: str,
) -> tuple[str, int]:
    """Run one agy turn with private-chat drafts or a typing heartbeat."""
    is_private = getattr(msg, "chat_type", "") == "private"
    hb_stop: asyncio.Event | None = None
    hb_task: asyncio.Task[None] | None = None
    publisher: _DraftPublisher | None = None
    on_output = None
    if is_private:
        publisher = _DraftPublisher(tg, chat_id, secrets.randbits(63) or 1)
        on_output = publisher.publish
    else:
        hb_stop = asyncio.Event()
        hb_task = asyncio.create_task(_heartbeat(tg, chat_id, hb_stop))
    turn_start = time.perf_counter()
    try:
        worker_socket = os.environ.get("AGY_BRIDGE_WORKER_SOCKET", "")
        if worker_socket:
            # The Telegram controller cannot invoke agy directly in production.
            result = await run_worker_job(
                worker_socket, prompt=msg.text, chat_dir=cs.chat_dir,
                conversation_id=cs.conversation_id, model=effective_model_value(cs, cfg),
                timeout=AGY_TIMEOUT_S, on_output=on_output,
            )
        else:
            # Local execution exists only for development and unit tests.
            result = await run_agy(
                prompt=msg.text,
                chat_dir=cs.chat_dir,
                conversation_id=cs.conversation_id,
                model=effective_model_value(cs, cfg),
                # Execution policy is selected by the root-owned service.
                mode=cfg.access.execution_mode,
                agy_path=agy_path,
                timeout=AGY_TIMEOUT_S,
                on_output=on_output,
                access=cfg.access,
            )
        # graphify is a local process and must never run in the token-holding
        # controller service. The production worker protocol does not expose it.
        if result.exit_code == 0 and not worker_socket:
            asyncio.create_task(_compile_memory_layer(cs.chat_dir))
        if result.exit_code == 0 and result.conversation_id:
            cs.conversation_id = result.conversation_id
            cs.has_session = True
    finally:
        if publisher is not None:
            await publisher.close()
        if hb_stop is not None and hb_task is not None:
            hb_stop.set()
            hb_task.cancel()
            try:
                await hb_task
            except (asyncio.CancelledError, Exception):
                pass
    elapsed = int((time.perf_counter() - turn_start) * 1000)
    if result.exit_code != 0:
        LOG.error(
            "turn failed chat=%d exit=%d ms=%d reason=%s",
            chat_id, result.exit_code, elapsed, safe_failure_reason(result),
        )
    else:
        LOG.info(
            "turn chat=%d exit=%d ms=%d reply_len=%d",
            chat_id, result.exit_code, elapsed, len(result.text or ""),
        )
    return result.text or "", result.exit_code


class _DraftPublisher:
    """Capacity-one, best-effort draft publisher.

    ``publish`` never waits for Telegram. The worker retains only the newest
    cumulative value and applies the update interval before publishing it.
    """

    def __init__(
        self,
        tg: "_TelegramLike",
        chat_id: int,
        draft_id: int,
        *,
        interval: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._tg = tg
        self._chat_id = chat_id
        self._draft_id = draft_id
        self._interval = interval
        self._clock = clock
        self._last_sent = float("-inf")
        self._latest: str | None = None
        self._wake = asyncio.Event()
        self._closed = False
        self._task = asyncio.create_task(self._run())

    async def publish(self, cumulative_text: str) -> None:
        if not self._closed:
            self._latest = cumulative_text
            self._wake.set()

    async def close(self, *, timeout: float = 2.0) -> None:
        """Stop the worker without letting a stuck draft request delay a turn."""
        if self._closed:
            return
        self._closed = True
        self._wake.set()
        try:
            await asyncio.wait_for(asyncio.shield(self._task), timeout=timeout)
        except asyncio.TimeoutError:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        except Exception as exc:
            LOG.debug("draft publisher shutdown failed chat=%s: %s", self._chat_id, exc)

    async def _run(self) -> None:
        try:
            while True:
                await self._wake.wait()
                self._wake.clear()
                if self._closed and self._latest is None:
                    return
                delay = self._last_sent + self._interval - self._clock()
                if delay > 0:
                    await asyncio.sleep(delay)
                text = self._latest
                self._latest = None
                if text is None:
                    if self._closed:
                        return
                    continue
                try:
                    await self._tg.send_rich_message_draft(
                        self._chat_id, self._draft_id, text
                    )
                    self._last_sent = self._clock()
                except Exception as exc:
                    LOG.debug("draft update failed chat=%s: %s", self._chat_id, exc)
                if self._closed:
                    return
        except asyncio.CancelledError:
            return


async def _heartbeat(
    tg: "_TelegramLike", chat_id: int, stop_event: asyncio.Event
) -> None:
    """Typing indicator refresh every 4 s."""
    try:
        while not stop_event.is_set():
            try:
                await tg.send_chat_action(chat_id, "typing")
            except Exception:
                pass
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=4.0)
            except asyncio.TimeoutError:
                pass
    except asyncio.CancelledError:
        return
