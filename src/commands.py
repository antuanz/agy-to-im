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
    "Commands: /start · /help · /status · /settings · /model · /mode · /reset · /cancel"
)

HELP_TEXT = (
    "Antigravity Bridge Help\n\n"
    "Send any text to chat with agy.\n\n"
    "Commands:\n"
    "/status   — read-only access + chat summary\n"
    "/settings — control panel with buttons\n"
    "/model    — pick a model (per-chat override)\n"
    "/mode     — choose plan or execution in this Sandbox chat\n"
    "/reset    — fresh agy session for this chat\n"
    "/cancel   — stop this chat's active request\n"
    "/info     — same as /status\n"
    "/image on|off — toggle photo processing\n"
    "/files    — list recent uploads\n"
    "/start, /help — these messages\n\n"
    "Model and permitted execution-mode choices are per-chat. VM access is controlled only by the server owner."
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


def effective_execution_mode(cs: "ChatState", cfg: "Config") -> tuple[str, str]:
    """Return the root-approved server default or a saved chat choice."""
    # Only Sandbox may offer per-chat modes.  Old saved state is deliberately
    # ignored when an administrator activates Workspace or Host access.
    if (
        cfg.access.profile == "sandbox"
        and cs.mode in cfg.access.allowed_execution_modes
    ):
        return cs.mode, "chat"
    return cfg.access.execution_mode, "server default"


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
    execution_mode, execution_src = effective_execution_mode(cs, cfg)
    session = "active (resumes next turn)" if cs.has_session else "fresh (next turn starts new)"
    home = os.path.expanduser("~")
    workdir = cs.chat_dir.replace(home, "~", 1)
    return (
        "🟢 Antigravity Bridge — Status\n"
        f"Model:      {model}  [{model_src}]\n"
        f"Access:     {cfg.access.label}  [{cfg.access.profile}]\n"
        f"Execution:  {execution_mode}  [{execution_src}]\n"
        "\n"
        "This chat:\n"
        f"  Session:  {session}\n"
        f"  Turns:    {cs.turn_count}\n"
        f"  Workdir:  {workdir}"
    )


def _settings_keyboard(cs: "ChatState", cfg: "Config") -> InlineKeyboard:
    execution_mode, _ = effective_execution_mode(cs, cfg)
    return [
        [{"text": "Model", "callback_data": "nav:model"}],
        [{"text": f"Execution: {execution_mode.title()}", "callback_data": "nav:mode"}],
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
    return BridgeReply(text=render_status(cs, cfg), keyboard=_settings_keyboard(cs, cfg))


def _mode_keyboard(cs: "ChatState", cfg: "Config") -> InlineKeyboard:
    """Offer only modes pre-authorised by the root-owned Sandbox policy."""
    rows: InlineKeyboard = []
    current, _ = effective_execution_mode(cs, cfg)
    labels = {"plan": "Plan", "code": "Execution"}
    for mode in ("plan", "code"):
        if mode in cfg.access.allowed_execution_modes:
            marker = "● " if current == mode else "○ "
            rows.append([{"text": marker + labels[mode], "callback_data": f"e:{mode}"}])
    rows.append([{"text": "← Back to settings", "callback_data": "nav:settings"}])
    return rows


def _render_mode_picker(cs: "ChatState", cfg: "Config") -> BridgeReply:
    current, source = effective_execution_mode(cs, cfg)
    if cfg.access.profile != "sandbox":
        return BridgeReply(
            text=(
                f"Execution is server-managed for {cfg.access.label}: {current}.\n\n"
                "This control cannot change the active VM access profile."
            ),
            keyboard=[[{"text": "← Back to settings", "callback_data": "nav:settings"}]],
        )
    return BridgeReply(
        text=(
            "⚙️ Choose execution mode for this chat\n\n"
            f"Current: {current.title()}  [{source}]\n\n"
            "Execution auto-approves tools only inside the existing Sandbox. "
            "It never changes VM access."
        ),
        keyboard=_mode_keyboard(cs, cfg),
    )


def _select_execution_mode(cs: "ChatState", cfg: "Config", requested: str) -> str:
    """Change only an allowed Sandbox chat preference and return a safe toast."""
    if cfg.access.profile != "sandbox":
        return "Server-managed"
    if requested not in {"plan", "code"} or requested not in cfg.access.allowed_execution_modes:
        return "Mode not allowed"
    cs.mode = requested
    return f"Execution: {requested.title()}"


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
        requested = args.lower()
        if cfg.access.profile != "sandbox":
            return _render_mode_picker(cs, cfg)
        if not requested:
            return _render_mode_picker(cs, cfg)
        if requested not in {"plan", "code"}:
            return BridgeReply("Use /mode, /mode plan, or /mode code.")
        if requested not in cfg.access.allowed_execution_modes:
            return BridgeReply(f"This server profile does not allow {requested} execution.")
        toast = _select_execution_mode(cs, cfg, requested)
        return BridgeReply(f"{toast}. VM access remains {cfg.access.label}.")
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
        return _render_mode_picker(cs, cfg)
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

    if data.startswith("e:"):
        toast = _select_execution_mode(cs, cfg, data[2:])
        rep = _render_settings(cs, cfg)
        return BridgeReply(text=rep.text, keyboard=rep.keyboard, toast=toast)

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
