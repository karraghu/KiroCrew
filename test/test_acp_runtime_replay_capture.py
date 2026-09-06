"""AcpRuntime transcript-replay capture (``dashboard.replay_from_acp``).

During ``session/load`` kiro-cli replays the whole prior transcript as
``session/update`` frames tagged with the resumed sid. That sid has no
registered queue yet, so the reader's default is the counted-drop path. With
``capture_replay=True`` the runtime retains those frames for the sid that
``load_session()`` armed and hands them to the handle; with it off (the
default) nothing changes.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from kiro_crew.acp import runtime as runtime_mod
from kiro_crew.acp.runtime import AcpRuntime
from kiro_crew.acp.types import JsonRpcMessage

SID = "resume-sid-1"


def _make_runtime(capture: bool) -> tuple[AcpRuntime, asyncio.StreamReader]:
    rt = AcpRuntime(work_dir="/tmp", capture_replay=capture)
    reader = asyncio.StreamReader()
    proc = MagicMock()
    proc.stdout = reader
    proc.stdin = MagicMock()
    proc.stdin.write = MagicMock()
    proc.stdin.drain = AsyncMock()
    proc.returncode = None
    proc.pid = 4242
    rt._process = proc
    rt._pid = 4242
    rt._initialized = True
    rt._can_load_session = True
    return rt, reader


def _update(sid: str, text: str) -> dict:
    return {
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {
            "sessionId": sid,
            "update": {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": text},
            },
        },
    }


def _msg(sid: str, text: str) -> JsonRpcMessage:
    return JsonRpcMessage.from_dict(_update(sid, text))


# ── helper-level behaviour ─────────────────────────────────────────────────


def test_capture_off_never_retains() -> None:
    rt, _ = _make_runtime(capture=False)
    rt._arm_replay_capture(SID)  # no-op when capture is off
    assert rt._capture_replay_frame(SID, _msg(SID, "a")) is False
    frames, overflow = rt._take_replay_capture(SID)
    assert frames == [] and overflow == 0


def test_capture_on_retains_only_armed_sid() -> None:
    rt, _ = _make_runtime(capture=True)
    rt._arm_replay_capture(SID)
    assert rt._capture_replay_frame(SID, _msg(SID, "a")) is True
    assert rt._capture_replay_frame("other-sid", _msg("other-sid", "b")) is False
    frames, overflow = rt._take_replay_capture(SID)
    assert [f["update"]["content"]["text"] for f in frames] == ["a"]
    assert overflow == 0
    # popped: a second take is empty and the sid is disarmed
    assert rt._take_replay_capture(SID) == ([], 0)
    assert rt._capture_replay_frame(SID, _msg(SID, "late")) is False


def test_capture_ignores_non_dict_params() -> None:
    rt, _ = _make_runtime(capture=True)
    rt._arm_replay_capture(SID)
    bad = JsonRpcMessage.from_dict(
        {"jsonrpc": "2.0", "method": "session/update", "params": ["not", "a", "dict"]}
    )
    assert rt._capture_replay_frame(SID, bad) is False
    assert rt._capture_replay_frame(None, _msg(SID, "x")) is False


def test_capture_cap_discards_whole_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_mod, "_REPLAY_CAPTURE_MAX_FRAMES", 3)
    rt, _ = _make_runtime(capture=True)
    rt._arm_replay_capture(SID)
    kept = [rt._capture_replay_frame(SID, _msg(SID, str(i))) for i in range(5)]
    assert kept == [True, True, True, False, False]
    frames, overflow = rt._take_replay_capture(SID)
    # Same discipline as the byte ceiling: a partial replay is not served.
    assert frames == []
    assert overflow == 2


def test_rearm_clears_previous_overflow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_mod, "_REPLAY_CAPTURE_MAX_FRAMES", 1)
    rt, _ = _make_runtime(capture=True)
    rt._arm_replay_capture(SID)
    rt._capture_replay_frame(SID, _msg(SID, "a"))
    rt._capture_replay_frame(SID, _msg(SID, "b"))
    rt._arm_replay_capture(SID)
    assert rt._capture_replay_frame(SID, _msg(SID, "c")) is True
    frames, overflow = rt._take_replay_capture(SID)
    assert len(frames) == 1 and overflow == 0


def test_byte_ceiling_discards_whole_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    # Two small frames fit; a large tool-result frame blows the cumulative cap.
    # The capture is then discarded OUTRIGHT (not truncated) and stays off for
    # the rest of the load, so the consumer renders its own transcript.
    monkeypatch.setattr(runtime_mod, "_REPLAY_CAPTURE_MAX_BYTES", 600)
    rt, _ = _make_runtime(capture=True)
    rt._arm_replay_capture(SID)
    assert rt._capture_replay_frame(SID, _msg(SID, "small one")) is True
    assert rt._capture_replay_frame(SID, _msg(SID, "small two")) is True
    assert rt._capture_replay_frame(SID, _msg(SID, "x" * 2000)) is False
    assert rt._capture_replay_frame(SID, _msg(SID, "after")) is False
    frames, overflow = rt._take_replay_capture(SID)
    assert frames == [] and overflow == 0
    assert SID not in rt._replay_capture_discarded
    assert SID not in rt._replay_capture_bytes
    # Re-arming the same sid starts clean.
    rt._arm_replay_capture(SID)
    assert rt._capture_replay_frame(SID, _msg(SID, "fresh")) is True
    frames, _ = rt._take_replay_capture(SID)
    assert len(frames) == 1


# ── reader-loop routing ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reader_loop_retains_replay_for_armed_sid_and_drops_otherwise() -> None:
    rt, reader = _make_runtime(capture=True)
    rt._arm_replay_capture(SID)
    dropped: list[tuple[object, object]] = []
    rt._note_dropped_frame = lambda sid, method: dropped.append((sid, method))  # type: ignore[method-assign, assignment]
    task = asyncio.ensure_future(rt._reader_loop())
    await asyncio.sleep(0)
    reader.feed_data((json.dumps(_update(SID, "kept")) + "\n").encode())
    reader.feed_data((json.dumps(_update("unknown", "dropped")) + "\n").encode())
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass
    frames, _ = rt._take_replay_capture(SID)
    assert [f["update"]["content"]["text"] for f in frames] == ["kept"]
    assert ("unknown", "session/update") in dropped
    assert (SID, "session/update") not in dropped


@pytest.mark.asyncio
async def test_reader_loop_capture_off_keeps_counted_drop() -> None:
    rt, reader = _make_runtime(capture=False)
    dropped: list[tuple[object, object]] = []
    rt._note_dropped_frame = lambda sid, method: dropped.append((sid, method))  # type: ignore[method-assign, assignment]
    task = asyncio.ensure_future(rt._reader_loop())
    await asyncio.sleep(0)
    reader.feed_data((json.dumps(_update(SID, "x")) + "\n").encode())
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass
    assert (SID, "session/update") in dropped
    assert rt._replay_capture == {}


# ── load_session hands the frames to the handle ────────────────────────────


@pytest.mark.asyncio
async def test_load_session_attaches_replay_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    rt, _ = _make_runtime(capture=True)

    async def fake_send(method: str, params: dict, timeout: float = 0.0) -> dict:
        assert method == "session/load"
        # Frames arrive WHILE the load request is in flight.
        assert rt._capture_replay_frame(SID, _msg(SID, "hello")) is True
        return {"modes": {"availableModes": [], "currentModeId": "kirocrew"}, "models": {}}

    monkeypatch.setattr(rt, "_send_and_await", fake_send)
    monkeypatch.setattr(rt, "_session_start_budget", AsyncMock(return_value=5.0))
    monkeypatch.setattr(runtime_mod, "pooled_session_servers", lambda *_a, **_k: [])
    monkeypatch.setattr(runtime_mod, "_load_watchdog_settings", lambda *_a, **_k: None)

    handle = await rt.load_session("/tmp/x.json", SID, cwd="/tmp")
    assert [f["update"]["content"]["text"] for f in handle.replay_updates] == ["hello"]
    # the bucket is gone: a later frame for the same id takes the drop path
    assert rt._capture_replay_frame(SID, _msg(SID, "late")) is False


@pytest.mark.asyncio
async def test_load_session_failure_disarms_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    rt, _ = _make_runtime(capture=True)

    async def failing_send(method: str, params: dict, timeout: float = 0.0) -> dict:
        rt._capture_replay_frame(SID, _msg(SID, "partial"))
        raise runtime_mod.AcpRuntimeError("Session is active in another process (PID 1)")

    monkeypatch.setattr(rt, "_send_and_await", failing_send)
    monkeypatch.setattr(rt, "_session_start_budget", AsyncMock(return_value=5.0))
    monkeypatch.setattr(runtime_mod, "pooled_session_servers", lambda *_a, **_k: [])

    with pytest.raises(runtime_mod.AcpRuntimeError):
        await rt.load_session("/tmp/x.json", SID, cwd="/tmp")
    assert rt._replay_capture == {}
    assert rt._replay_capture_overflow == {}


@pytest.mark.asyncio
async def test_load_session_capture_off_leaves_handle_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rt, _ = _make_runtime(capture=False)

    async def fake_send(method: str, params: dict, timeout: float = 0.0) -> dict:
        return {"modes": {"availableModes": [], "currentModeId": "kirocrew"}, "models": {}}

    monkeypatch.setattr(rt, "_send_and_await", fake_send)
    monkeypatch.setattr(rt, "_session_start_budget", AsyncMock(return_value=5.0))
    monkeypatch.setattr(runtime_mod, "pooled_session_servers", lambda *_a, **_k: [])
    monkeypatch.setattr(runtime_mod, "_load_watchdog_settings", lambda *_a, **_k: None)

    handle = await rt.load_session("/tmp/x.json", SID, cwd="/tmp")
    assert handle.replay_updates == []
