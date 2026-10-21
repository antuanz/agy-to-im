"""Bridge command handlers — text slash commands and inline-keyboard callbacks."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from src.media import clean_inbox, list_inbox
from src.telegram import InlineKeyboard

if TYPE_CHECKING:
    from src.config import Config
    from src.daemon import _TelegramLike
    from src.state import ChatState
    from src.telegram import CallbackQuery, InboundMessage

_DEFAULT_TOKEN = "_DEFAULT"
_CONFIG_TOKEN = "_CONFIG"
AGY_DEFAULT_LABEL = "Use agy default"

WELCOME_TEXT = (
    "Welcome! I'm your Antigravity bridge bot.\n\n"
    "Send me any message and I'll forward it to the agy CLI. "
    "Each chat has its own session that persists across messages.\n\n"
    "Commands: /start · /help · /status · /settings · /model · /reset · /cancel"
)

HELP_TEXT = (
    "Antigravity Bridge Help\n\n"
    "Send any text to chat with agy.\n\n"
    "Commands:\n"
    "/status   — read-only access + chat summary\n"
    "/settings — control panel with buttons\n"
    "/model    — pick a model (per-chat override)\n"
    "/reset    — fresh agy session for this chat\n"
    "/cancel   — stop this chat's active request\n"
    "/info     — same as /status\n"
    "/image on|off — toggle photo processing\n"
    "/files    — list recent uploads\n"
    "/start, /help — these messages\n\n"
    "Model selection is per-chat. VM access is controlled only by the server owner."
)


@dataclass(frozen=True)
class BridgeReply:
    """Daemon's structured reply: text, optional inline keyboard, optional toast."""

    text: str
    keyboard: InlineKeyboard | None = None
    toast: str = ""


def effective_model_value(cs: "ChatState", cfg: "Config") -> str:
    """Return the exact model slug to pass to agy, or empty to omit --model."""
    if cs.model is not None:
        return cs.model
    return cfg.agy.model


def is_cancel_command(text: str) -> bool:
    """Recognize Telegram's optional @BotUsername command suffix."""
    command = text.strip().split(maxsplit=1)[0].lower() if text.strip() else ""
    return command.partition("@")[0] == "/cancel"


def _effective_model(cs: "ChatState", cfg: "Config") -> tuple[str, str]:
    model = effective_model_value(cs, cfg)
    if not model:
        return AGY_DEFAULT_LABEL, "agy"
    return model, "chat" if cs.model is not None else "config"


def render_status(cs: "ChatState", cfg: "Config") -> str:
    model, model_src = _effective_model(cs, cfg)
    session = "active (resumes next turn)" if cs.has_session else "fresh (next turn starts new)"
    home = os.path.expanduser("~")
    workdir = cs.chat_dir.replace(home, "~", 1)
    return (
        "🟢 Antigravity Bridge — Status\n"
        f"Model:      {model}  [{model_src}]\n"
        f"Access:     {cfg.access.label}  [{cfg.access.profile}]\n"
        f"Execution:  {cfg.access.execution_mode}  [server-managed]\n"
        "\n"
        "This chat:\n"
        f"  Session:  {session}\n"
        f"  Turns:    {cs.turn_count}\n"
        f"  Workdir:  {workdir}"
    )


def _settings_keyboard() -> InlineKeyboard:
    return [
        [{"text": "Model", "callback_data": "nav:model"}],
        [{"text": "Reset session", "callback_data": "R"}],
        [{"text": "Refresh", "callback_data": "nav:settings"}],
    ]


def _model_keyboard(
    current_per_chat: str | None,
    models: Sequence[str],
    *,
    has_config_default: bool,
) -> InlineKeyboard:
    rows: InlineKeyboard = []
    for index, m in enumerate(models):
        marker = "● " if m == current_per_chat else "○ "
        digest = hashlib.sha256(m.encode("utf-8")).hexdigest()[:12]
        rows.append([{
            "text": marker + m,
            "callback_data": f"m:{index}:{digest}",
        }])
    uses_agy_default = current_per_chat == "" or (
        current_per_chat is None and not has_config_default
    )
    default_marker = "● " if uses_agy_default else "○ "
    rows.append([
        {"text": default_marker + AGY_DEFAULT_LABEL, "callback_data": f"m:{_DEFAULT_TOKEN}"}
    ])
    if has_config_default:
        config_marker = "● " if current_per_chat is None else "○ "
        rows.append([{
            "text": config_marker + "Use configured default",
            "callback_data": f"m:{_CONFIG_TOKEN}",
        }])
    rows.append([{"text": "← Back to settings", "callback_data": "nav:settings"}])
    return rows


def _render_settings(cs: "ChatState", cfg: "Config") -> BridgeReply:
    return BridgeReply(text=render_status(cs, cfg), keyboard=_settings_keyboard())


def _render_model_picker(
    cs: "ChatState",
    cfg: "Config",
    models: Sequence[str],
    discovery_error: str = "",
) -> BridgeReply:
    cur, src = _effective_model(cs, cfg)
    selected = effective_model_value(cs, cfg)
    notices: list[str] = []
    if discovery_error:
        notices.append(f"⚠️ Model list may be stale: {discovery_error}")
    elif selected and selected not in models:
        notices.append(
            f"⚠️ Saved selection {selected!r} is no longer available. "
            "Choose a model or Use agy default."
        )
    notice = "\n\n" + "\n".join(notices) if notices else ""
    return BridgeReply(
        text=f"🤖 Choose a model for this chat\n\nCurrent: {cur}  [{src}]{notice}",
        keyboard=_model_keyboard(
            cs.model, models, has_config_default=bool(cfg.agy.model)
        ),
    )


async def handle_text_command(
    msg: "InboundMessage",
    cs: "ChatState",
    cfg: "Config",
    models: Sequence[str] = (),
    discovery_error: str = "",
) -> BridgeReply | None:
    """Return a reply for a slash command, else None (forward to agy)."""
    stripped = msg.text.strip()
    if not stripped:
        return None
    parts = stripped.split(maxsplit=1)
    # Telegram may append ``@BotUsername`` to a command, particularly when a
    # command is invoked from its command menu. Commands remain per-bot, so
    # treat that suffix exactly like the plain private-chat spelling.
    cmd = parts[0].lower().partition("@")[0]
    args = parts[1].strip() if len(parts) > 1 else ""

    if cmd == "/start":
        return BridgeReply(WELCOME_TEXT)
    if cmd == "/help":
        return BridgeReply(HELP_TEXT)
    if cmd in ("/status", "/info"):
        return BridgeReply(render_status(cs, cfg))
    if cmd == "/settings":
        return _render_settings(cs, cfg)
    if cmd == "/model":
        if args:
            if args in {"default", "agy-default"}:
                cs.model = ""
                return BridgeReply(f"🤖 Model set to {AGY_DEFAULT_LABEL}")
            if args in models:
                cs.model = args
                return BridgeReply(f"🤖 Model set to {args}")
            if discovery_error:
                return BridgeReply(
                    f"⚠️ Cannot validate model right now: {discovery_error}"
                )
            return BridgeReply(f"⚠️ Unsupported model: {args}")
        return _render_model_picker(cs, cfg, models, discovery_error)
    if cmd == "/mode":
        return BridgeReply(
            "Execution and VM access are server-managed. The owner changes them through SSH."
        )
    if cmd == "/reset":
        cs.has_session = False
        cs.conversation_id = ""
        return BridgeReply("🧹 Session reset. The next message starts fresh.")
    if cmd == "/thinking":
        mode = args.strip().lower()
        if mode in ("on", "true", "1"):
            return BridgeReply("💭 Thinking visibility is not available in agy print mode.")
        if mode in ("off", "false", "0"):
            return BridgeReply("💭 agy print mode already suppresses thinking streams.")
        return BridgeReply("💭 agy print mode does not expose thinking streams.")
    if cmd == "/compact":
        return BridgeReply("🗜️ Context compaction is not supported by agy print mode.")
    if cmd == "/image":
        mode = args.strip().lower()
        if mode in ("on", "true", "1"):
            cs.photo_enabled = True  # type: ignore[attr-defined]
            return BridgeReply("📸 Photo processing: ON")
        if mode in ("off", "false", "0"):
            cs.photo_enabled = False  # type: ignore[attr-defined]
            return BridgeReply("📸 Photo processing: OFF")
        return BridgeReply("📸 Photo processing toggle not available in this build.")
    if cmd == "/files":
        if args == "clean":
            removed = clean_inbox(cs.chat_dir, max_age_hours=0)
            return BridgeReply(f"🧹 Inbox cleaned: {removed} files removed")
        files = list_inbox(cs.chat_dir, limit=20)
        if not files:
            return BridgeReply("📭 Inbox is empty")
        return BridgeReply("📎 Inbox:\n" + "\n".join(f"• {f}" for f in files))
    if cmd == "/queue":
        return BridgeReply("📋 Queue status is available via daemon internals.")
    return None


def handle_callback(
    cq: "CallbackQuery",
    cs: "ChatState",
    cfg: "Config",
    models: Sequence[str] = (),
    discovery_error: str = "",
) -> BridgeReply:
    """Handle inline-keyboard button taps. Always returns a reply to render."""
    data = cq.data

    if data == "nav:status":
        return BridgeReply(render_status(cs, cfg))
    if data == "nav:settings":
        return _render_settings(cs, cfg)
    if data == "nav:model":
        return _render_model_picker(cs, cfg, models, discovery_error)
    if data == "nav:mode":
        return BridgeReply(
            "Execution and VM access are server-managed. The owner changes them through SSH.",
            toast="Server-managed",
        )
    if data == "R":
        cs.has_session = False
        cs.conversation_id = ""
        rep = _render_settings(cs, cfg)
        return BridgeReply(text=rep.text, keyboard=rep.keyboard, toast="Session reset")

    if data.startswith("m:"):
        choice = _resolve_model_callback(data[2:], models)
        if choice == _DEFAULT_TOKEN:
            cs.model = ""
            toast = AGY_DEFAULT_LABEL
        elif choice == _CONFIG_TOKEN and cfg.agy.model:
            cs.model = None
            toast = "Using configured default"
        elif choice in models:
            cs.model = choice
            toast = f"Model: {choice}"
        elif discovery_error:
            toast = "Model list unavailable"
        else:
            toast = "Unknown choice"
        rep = _render_settings(cs, cfg)
        return BridgeReply(text=rep.text, keyboard=rep.keyboard, toast=toast)

    if data.startswith("M:"):
        rep = _render_settings(cs, cfg)
        return BridgeReply(text=rep.text, keyboard=rep.keyboard, toast="Server-managed")

    return _render_settings(cs, cfg)


def _resolve_model_callback(payload: str, models: Sequence[str]) -> str | None:
    if payload in {_DEFAULT_TOKEN, _CONFIG_TOKEN}:
        return payload
    # Keep accepting old slug-bearing buttons, but validate them against the
    # current discovery result before changing state.
    if payload in models:
        return payload
    try:
        index_text, supplied_digest = payload.split(":", 1)
        model = models[int(index_text)]
    except (ValueError, IndexError):
        return None
    expected = hashlib.sha256(model.encode("utf-8")).hexdigest()[:12]
    return model if supplied_digest == expected else None
