"""Tests for the asynchronous agy stream-JSON runner."""
from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
import textwrap
from pathlib import Path

import pytest

from src.agy_runner import (
    AgyResult,
    _build_args,
    _process_group_options,
    _terminate_process_group,
    run_agy,
)


def _fake_agy(tmp_path: Path) -> Path:
    implementation = tmp_path / "fake_agy.py"
    implementation.write_text(
        textwrap.dedent(
            """\
            import json, os, pathlib, subprocess, sys, time

            pathlib.Path(os.environ["ARGV_FILE"]).write_text(json.dumps(sys.argv[1:]))
            line = sys.stdin.readline()
            pathlib.Path(os.environ["INPUT_FILE"]).write_text(line)
            case = os.environ.get("FAKE_AGY_CASE", "success")
            if case == "timeout":
                marker = os.environ["CHILD_MARKER"]
                subprocess.Popen([sys.executable, "-c", "import pathlib,time,sys; time.sleep(1); pathlib.Path(sys.argv[1]).write_text('alive')", marker])
                time.sleep(30)
            elif case == "non_success":
                print(json.dumps({"event":"step_update","step_update":{"step_type":"agent_response","text_delta":"partial"}}), flush=True)
                print(json.dumps({"event":"result","result":{"status":"ERROR","error":"failed"}}), flush=True)
            elif case == "no_result":
                print("not json", flush=True)
            elif case == "malformed":
                print("not json", flush=True)
                print(json.dumps({"event":"result","result":{"status":"SUCCESS","response":"ignored"}}), flush=True)
            elif case == "denied":
                print(json.dumps({"event":"result","result":{"status":"SUCCESS","response":"ignored","denied_actions":["write"]}}), flush=True)
            elif case == "oversized":
                print("x" * 600000, flush=True)
                print(json.dumps({"event":"result","result":{"status":"SUCCESS","response":"ignored"}}), flush=True)
            elif case == "stderr_oversized":
                sys.stderr.write("x" * 70000); sys.stderr.flush()
                print(json.dumps({"event":"result","result":{"status":"SUCCESS","response":"ignored"}}), flush=True)
            else:
                sys.stderr.write("private diagnostic\\n")
                event = json.dumps({"event":"step_update","step_update":{"step_type":"tool","tool_info":{"output":"SECRET TOOL"}}}) + "\\n"
                sys.stdout.write(event)
                delta = json.dumps({"event":"step_update","step_update":{"step_type":"agent_response","text_delta":"hello "}}) + "\\n"
                middle = len(delta) // 2
                sys.stdout.write(delta[:middle]); sys.stdout.flush(); time.sleep(0.02)
                sys.stdout.write(delta[middle:]); sys.stdout.flush()
                print(json.dumps({"event":"step_update","step_update":{"step_type":"agent_response","text_delta":"world"}}), flush=True)
                print(json.dumps({"event":"result","result":{"status":"SUCCESS","response":"terminal response"}}), flush=True)
            """
        )
    )
    if os.name == "nt":
        wrapper = tmp_path / "agy.cmd"
        wrapper.write_text(f'@"{sys.executable}" "{implementation}" %*\n')
    else:
        wrapper = tmp_path / "agy"
        wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{implementation}" "$@"\n')
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    return wrapper


def _env_for_fake(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str) -> None:
    # Production deliberately gives agy an allowlisted environment. Tests inject
    # only the four values needed by the fake executable through that same seam.
    monkeypatch.setattr(
        "src.agy_runner._child_environment",
        lambda: {
            "ARGV_FILE": str(tmp_path / "argv.json"),
            "INPUT_FILE": str(tmp_path / "input.jsonl"),
            "FAKE_AGY_CASE": case,
            "CHILD_MARKER": str(tmp_path / "child-alive"),
        },
    )


async def test_streams_only_agent_text_and_sends_user_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agy = _fake_agy(tmp_path)
    _env_for_fake(monkeypatch, tmp_path, "success")
    updates: list[str] = []

    async def on_output(text: str) -> None:
        updates.append(text)

    result = await run_agy(
        "hello input",
        chat_dir=str(tmp_path),
        has_session=False,
        model="",
        mode="code",
        agy_path=str(agy),
        on_output=on_output,
    )

    assert result.text == "terminal response"
    assert result.exit_code == 0
    assert result.stderr.strip() == "private diagnostic"
    assert result.status == "SUCCESS"
    assert updates == ["hello ", "hello world"]
    sent = json.loads((tmp_path / "input.jsonl").read_text())
    assert sent == {"event": "user", "message": {"content": "hello input"}}
    assert "SECRET TOOL" not in result.text
    assert "private diagnostic" not in result.text


async def test_builds_stream_argv_and_omits_empty_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agy = _fake_agy(tmp_path)
    _env_for_fake(monkeypatch, tmp_path, "success")
    await run_agy(
        "prompt is not argv",
        chat_dir=str(tmp_path),
        has_session=True,
        model="",
        mode="plan",
        agy_path=str(agy),
    )
    args = json.loads((tmp_path / "argv.json").read_text())
    assert args[:4] == [
        "--input-format", "stream-json", "--output-format", "stream-json"
    ]
    assert "--continue" in args
    assert "--sandbox" not in args
    assert "--model" not in args
    assert "prompt is not argv" not in args


def test_build_args_includes_exact_selected_model() -> None:
    args = _build_args(
        agy_path="/usr/bin/agy",
        prompt="not-on-argv",
        has_session=False,
        model="gemini-4-argon-high",
        mode="code",
        print_timeout="15m",
    )
    assert args[0] == "/usr/bin/agy"
    assert args[args.index("--model") + 1] == "gemini-4-argon-high"
    assert "--new-project" in args
    assert "-p" not in args


def test_sandbox_policy_omits_dangerous_permission_bypass() -> None:
    args = _build_args(
        agy_path="/usr/bin/agy", prompt="x", has_session=False,
        model="", mode="code", print_timeout="15m",
    )
    assert "--dangerously-skip-permissions" not in args
    assert args[args.index("--mode") + 1] == "plan"
    assert "--sandbox" not in args


@pytest.mark.parametrize(
    "case,status", [
        ("non_success", "ERROR"), ("no_result", "INVALID"),
        ("malformed", "INVALID"), ("denied", "INVALID"),
        ("oversized", "INVALID"), ("stderr_oversized", "INVALID"),
    ],
)
async def test_rejects_non_success_or_missing_terminal_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    status: str,
) -> None:
    agy = _fake_agy(tmp_path)
    _env_for_fake(monkeypatch, tmp_path, case)
    result = await run_agy(
        "x", chat_dir=str(tmp_path), has_session=False, model="", mode="code",
        agy_path=str(agy),
    )
    assert result.exit_code != 0
    assert result.status == status


async def test_terminal_result_is_authoritative_over_deltas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agy = _fake_agy(tmp_path)
    _env_for_fake(monkeypatch, tmp_path, "success")
    result = await run_agy(
        "x", chat_dir=str(tmp_path), has_session=False, model="", mode="code",
        agy_path=str(agy),
    )
    assert result.text == "terminal response"
    assert result.text != "hello world"


async def test_timeout_terminates_process_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agy = _fake_agy(tmp_path)
    _env_for_fake(monkeypatch, tmp_path, "timeout")
    result = await run_agy(
        "x", chat_dir=str(tmp_path), has_session=False, model="", mode="code",
        agy_path=str(agy), timeout=0.15,
    )
    assert result.exit_code == 124
    assert result.status == "TIMEOUT"
    assert "timed out" in result.stderr


async def test_cancellation_terminates_process_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agy = _fake_agy(tmp_path)
    _env_for_fake(monkeypatch, tmp_path, "timeout")
    task = asyncio.create_task(run_agy(
        "x", chat_dir=str(tmp_path), has_session=False, model="", mode="code",
        agy_path=str(agy),
    ))
    await asyncio.sleep(0.15)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()


async def test_process_group_cleanup_kills_descendants(tmp_path: Path) -> None:
    marker = tmp_path / "orphan-marker"
    code = (
        "import subprocess,sys,time; "
        "subprocess.Popen([sys.executable,'-c',"
        "\"import pathlib,sys,time; time.sleep(1); pathlib.Path(sys.argv[1]).write_text('alive')\","
        "sys.argv[1]]); time.sleep(30)"
    )
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", code, str(marker), **_process_group_options()
    )
    await asyncio.sleep(0.15)
    await _terminate_process_group(proc)
    assert proc.returncode is not None
    await asyncio.sleep(1.1)
    assert not marker.exists()
