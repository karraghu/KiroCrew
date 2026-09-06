"""``chat_replay.merge_replay_transcript`` — transcript from ACP replay + JSONL sidecar.

Frame shapes below mirror what kiro-cli 2.21.0 replays during ``session/load``
(measured in research/acp-replay): one merged ``agent_message_chunk`` per model
message, an (often empty) ``agent_thought_chunk`` before it, the FULL assembled
prompt as ``user_message_chunk``, and ``tool_call`` + ``tool_call_update`` pairs
whose ``title`` is the raw tool name.
"""

from __future__ import annotations

from typing import Any

from kiro_crew.dashboard.chat_replay import (
    SOURCE_JSONL,
    SOURCE_REPLAY,
    clean_user_text,
    merge_replay_transcript,
)

SID = "sid-1"
MARKER = "[CURRENT USER REQUEST -- respond to this]"


def _u(session_update: str, **update: Any) -> dict[str, Any]:
    return {"sessionId": SID, "update": {"sessionUpdate": session_update, **update}}


def _text(kind: str, text: str) -> dict[str, Any]:
    return _u(kind, content={"type": "text", "text": text})


def _prompt(user_text: str) -> dict[str, Any]:
    return _text(
        "user_message_chunk", f"[AGENT SYSTEM PROMPT]\nlots of context\n{MARKER}\n{user_text}"
    )


def _tool_call(
    tcid: str, name: str, raw_input: dict[str, Any], kind: str = "read"
) -> dict[str, Any]:
    return _u("tool_call", toolCallId=tcid, title=name, kind=kind, rawInput=raw_input)


def _tool_done(tcid: str, output: Any) -> dict[str, Any]:
    return _u("tool_call_update", toolCallId=tcid, status="completed", rawOutput=output)


def _row(role: str, content: str, **meta: Any) -> dict[str, Any]:
    return {"role": role, "content": content, "ts": "2026-09-06T00:00:00+00:00", "meta": meta}


def _sources(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return [(r["role"], r["meta"]["source"]) for r in rows]


# ── clean_user_text ────────────────────────────────────────────────────────


def test_clean_user_text_strips_assembled_prompt_after_last_marker() -> None:
    prompt = f"[RUNTIME] dashboard\n{MARKER}\nold\n{MARKER}\nread the file"
    assert clean_user_text(prompt) == "read the file"


def test_clean_user_text_keeps_short_raw_prompt_and_bounds_long_one() -> None:
    assert clean_user_text("hello there") == "hello there"
    long = "x" * 10_000
    assert len(clean_user_text(long)) <= 4_000
    assert clean_user_text("") == ""


# ── merge ──────────────────────────────────────────────────────────────────


def test_empty_replay_returns_jsonl_untouched_but_tagged() -> None:
    rows = [_row("user", "hi", mid="m1"), _row("assistant", "hello", mid="m2")]
    out, report = merge_replay_transcript([], rows)
    assert [r["content"] for r in out] == ["hi", "hello"]
    assert all(r["meta"]["source"] == SOURCE_JSONL for r in out)
    assert report["turns"] == 0 and report["jsonl"] == 2


def test_replay_builds_turn_and_jsonl_user_row_wins_for_prompt() -> None:
    updates = [
        _prompt("read notes"),
        _text("agent_thought_chunk", ""),
        _text("agent_message_chunk", ""),
        _tool_call(
            "t1", "read", {"__tool_use_purpose": "Read notes", "operations": [{"path": "/n"}]}
        ),
        _tool_done("t1", {"items": [{"Text": "FIRST"}]}),
        _text("agent_thought_chunk", "the file says FIRST"),
        _text("agent_message_chunk", "The first line is FIRST."),
    ]
    jsonl = [
        _row("user", "read notes", mid="u1", sendId="s1"),
        _row(
            "tool",
            "🔧 Reading notes:1",
            tool_call_id="t1",
            purpose="Read notes",
            input="{}",
            kind="read",
            done=True,
            output="FIRST",
        ),
        _row("assistant", "The first line is FIRST.", mid="a1", turn_stats={"ms": 5}),
    ]
    out, report = merge_replay_transcript(updates, jsonl)
    roles = [r["role"] for r in out]
    assert roles == ["user", "tool", "thinking", "assistant"]
    user, tool, thinking, assistant = out
    # prompt bubble: JSONL row (clean text, mid, sendId) tagged jsonl but marked covered
    assert user["content"] == "read notes"
    assert user["meta"]["mid"] == "u1"
    assert user["meta"]["source"] == SOURCE_JSONL
    # tool row: replay-built, enriched with the humanized JSONL title
    assert tool["meta"]["source"] == SOURCE_REPLAY
    assert tool["content"] == "🔧 Reading notes:1"
    assert tool["meta"]["replay_title"] == "🔧 read"
    assert tool["meta"]["tool_call_id"] == "t1"
    assert tool["meta"]["done"] is True
    assert "FIRST" in tool["meta"]["output"]
    assert tool["ts"] == "2026-09-06T00:00:00+00:00"
    # thinking survives (the live path never persisted it)
    assert thinking["meta"]["source"] == SOURCE_REPLAY
    assert thinking["content"] == "the file says FIRST"
    # assistant text from replay, identity + turn stats from JSONL
    assert assistant["meta"]["source"] == SOURCE_REPLAY
    assert assistant["meta"]["mid"] == "a1"
    assert assistant["meta"]["turn_stats"] == {"ms": 5}
    assert report["turns"] == 1 and report["replay"] == 3 and report["jsonl"] == 1


def test_empty_thought_and_empty_text_chunks_produce_no_rows() -> None:
    updates = [
        _prompt("hi"),
        _text("agent_thought_chunk", ""),
        _text("agent_message_chunk", ""),
        _text("agent_message_chunk", "yo"),
    ]
    out, _ = merge_replay_transcript(updates, [_row("user", "hi")])
    assert [r["role"] for r in out] == ["user", "assistant"]
    assert out[1]["content"] == "yo"


def test_sidecar_rows_anchor_after_their_tool() -> None:
    updates = [
        _prompt("run it"),
        _tool_call("t1", "shell", {"command": "echo 1"}, kind="execute"),
        _tool_done("t1", {"items": [{"Json": {"stdout": "1\n"}}]}),
        _tool_call("t2", "shell", {"command": "echo 2"}, kind="execute"),
        _tool_done("t2", {"items": [{"Json": {"stdout": "2\n"}}]}),
        _text("agent_message_chunk", "done"),
    ]
    jsonl = [
        _row("user", "run it"),
        _row("permission", "approve echo 1?", approval_id="p1", resolved="approved"),
        _row("tool", "🔧 Running: echo 1", tool_call_id="t1", kind="execute"),
        _row("inject", "[Tool blocked — reason sent to the agent] policy", injectKind="policy"),
        _row("tool", "🔧 Running: echo 2", tool_call_id="t2", kind="execute"),
        _row("assistant", "done"),
        _row("error", "⟳ Backend hiccup — retrying…", kind="transient_retry"),
    ]
    out, report = merge_replay_transcript(updates, jsonl)
    assert _sources(out) == [
        ("user", SOURCE_JSONL),
        ("permission", SOURCE_JSONL),  # preceded every tool -> right after the prompt
        ("tool", SOURCE_REPLAY),
        ("inject", SOURCE_JSONL),  # followed t1 -> after t1
        ("tool", SOURCE_REPLAY),
        ("error", SOURCE_JSONL),  # followed t2 -> after t2 (assistant text matched replay)
        ("assistant", SOURCE_REPLAY),
    ]
    assert report["jsonl"] == 4 and report["replay"] == 3


def test_nudge_rows_match_dash_folded_prompt() -> None:
    # Kiro Crew writes `—` to the JSONL but sends `--` on the wire.
    nudge = "[auto-nudge cycle 1]\n[work ledger — durable state]"
    updates = [
        _text("user_message_chunk", f"[RUNTIME] dashboard\n{MARKER}\n{nudge.replace('—', '--')}"),
        _text("agent_message_chunk", "ok"),
    ]
    out, report = merge_replay_transcript(
        updates, [_row("nudge", nudge, nudge=True), _row("assistant", "ok")]
    )
    assert out[0]["role"] == "nudge" and out[0]["meta"]["source"] == SOURCE_JSONL


def test_replay_turn_without_jsonl_prompt_uses_stripped_wire_text() -> None:
    updates = [_prompt("only in replay"), _text("agent_message_chunk", "answer")]
    out, _ = merge_replay_transcript(updates, [])
    assert out[0] == {
        "role": "user",
        "content": "only in replay",
        "cls": "msg msg-u",
        "meta": {"source": SOURCE_REPLAY},
    }


def test_jsonl_prompts_missing_from_replay_are_kept_in_order() -> None:
    updates = [_prompt("second"), _text("agent_message_chunk", "b")]
    jsonl = [
        _row("user", "first", mid="u0"),
        _row("assistant", "a", mid="a0"),
        _row("user", "second", mid="u1"),
        _row("assistant", "b", mid="a1"),
    ]
    out, report = merge_replay_transcript(updates, jsonl)
    assert [r["content"] for r in out] == ["first", "a", "second", "b"]
    assert out[0]["meta"]["source"] == SOURCE_JSONL and out[1]["meta"]["source"] == SOURCE_JSONL
    assert out[3]["meta"]["source"] == SOURCE_REPLAY


def test_short_prompt_matches_only_by_equality() -> None:
    # JSONL: "a" then "a longer question about a"; replay turn 1's clean slice is
    # "a longer question about a" (the first turn was dropped from the replay).
    # Containment would pair "a" with it and shift every later turn.
    updates = [_prompt("a longer question about a"), _text("agent_message_chunk", "long answer")]
    jsonl = [
        _row("user", "a", mid="u0"),
        _row("assistant", "short answer", mid="a0"),
        _row("user", "a longer question about a", mid="u1"),
        _row("assistant", "long answer", mid="a1"),
    ]
    out, _ = merge_replay_transcript(updates, jsonl)
    assert [r["content"] for r in out] == [
        "a",
        "short answer",
        "a longer question about a",
        "long answer",
    ]
    assert out[2]["meta"]["mid"] == "u1"
    assert out[3]["meta"]["source"] == SOURCE_REPLAY


def test_short_prompt_followed_by_guidance_lines_matches_exactly() -> None:
    # The writer appends per-turn guidance after the request on its own lines;
    # the short JSONL row must still claim its turn (newline-bounded prefix).
    updates = [
        _prompt(
            "say ok\n\n(If presenting choices, end with [OPTIONS: a | b] as the very last line.)"
        ),
        _text("agent_message_chunk", "ok"),
    ]
    out, report = merge_replay_transcript(
        updates, [_row("user", "say ok", mid="u1"), _row("assistant", "ok")]
    )
    assert [r["role"] for r in out] == ["user", "assistant"]
    assert out[0]["meta"]["mid"] == "u1" and report["jsonl"] == 1


def test_quick_prompt_row_matches_its_expanded_macro() -> None:
    # The JSONL keeps "/plain"; the agent received the expanded macro, which opens
    # with the header both derive from. Without this the row is unmatched and
    # the turn renders twice (replay prompt + JSONL prompt).
    from kiro_crew.quick_prompts import expand_quick_prompt

    expanded = expand_quick_prompt("/plain")
    assert expanded is not None
    updates = [_prompt(expanded), _text("agent_message_chunk", "plain answer")]
    out, report = merge_replay_transcript(
        updates, [_row("user", "/plain", mid="u1"), _row("assistant", "plain answer")]
    )
    assert [r["role"] for r in out] == ["user", "assistant"]
    assert out[0]["content"] == "/plain" and out[0]["meta"]["mid"] == "u1"
    assert report["turns"] == 1 and report["jsonl"] == 1


def test_rewritten_turn_falls_back_to_jsonl() -> None:
    # The user regenerated after the resume: the JSONL now holds "new answer",
    # the replay still carries "old answer". The old answer must not come back.
    updates = [
        _prompt("q"),
        _text("agent_thought_chunk", "old reasoning"),
        _text("agent_message_chunk", "old answer"),
    ]
    jsonl = [_row("user", "q", mid="u1"), _row("assistant", "new answer", mid="a2")]
    out, report = merge_replay_transcript(updates, jsonl)
    assert [(r["role"], r["content"]) for r in out] == [("user", "q"), ("assistant", "new answer")]
    assert all(r["meta"]["source"] == SOURCE_JSONL for r in out)
    assert report["replay"] == 0


def test_cached_merge_misses_on_in_place_mutation() -> None:
    from kiro_crew.dashboard.chat_replay import _MERGE_CACHE, merge_replay_transcript_cached

    _MERGE_CACHE.clear()
    updates = [
        _prompt("run"),
        _tool_call("t1", "shell", {"command": "echo 1"}, kind="execute"),
        _tool_done("t1", {"items": [{"Json": {"stdout": "1\n"}}]}),
        _text("agent_message_chunk", "done"),
    ]
    perm = _row("permission", "approve?", approval_id="p1")
    jsonl = [
        _row("user", "run"),
        perm,
        _row("tool", "🔧 Running: echo 1", tool_call_id="t1"),
        _row("assistant", "done"),
    ]
    first, _ = merge_replay_transcript_cached("slot-a", updates, jsonl)
    again, _ = merge_replay_transcript_cached("slot-a", updates, jsonl)
    assert again == first and again is not first  # served from cache, as a copy
    # Resolve the permission IN PLACE (no new row): the witness must change.
    perm["meta"]["resolved"] = "approved"
    third, _ = merge_replay_transcript_cached("slot-a", updates, jsonl)
    assert next(r for r in third if r["role"] == "permission")["meta"]["resolved"] == "approved"


def test_discard_replay_for_slot_clears_cache_and_provider() -> None:
    from kiro_crew.dashboard.chat_replay import _MERGE_CACHE, discard_replay_for_slot

    class _Slot:
        key = "slot-z"
        session_key = "slot-z"

    class _Provider:
        discarded = False

        def discard_replay(self) -> None:
            self.discarded = True

    class _Sessions:
        def __init__(self) -> None:
            self.provider = _Provider()

        def get_provider(self, key: str) -> _Provider:
            return self.provider

    _MERGE_CACHE["slot-z"] = ((0, 0), [], {"replay": 0, "jsonl": 0, "turns": 0})
    sessions = _Sessions()
    discard_replay_for_slot(sessions, _Slot())
    assert "slot-z" not in _MERGE_CACHE
    assert sessions.provider.discarded is True


def test_steer_appended_prompt_still_matches_long_row_by_containment() -> None:
    base = "Please summarise the design document for the replay prototype in detail"
    updates = [_prompt(base + "\n\n[steer] also mention risks"), _text("agent_message_chunk", "ok")]
    out, _ = merge_replay_transcript(
        updates, [_row("user", base, mid="u1"), _row("assistant", "ok")]
    )
    assert out[0]["meta"]["mid"] == "u1" and out[0]["meta"]["source"] == SOURCE_JSONL


def test_steer_user_row_is_sidecar_not_a_turn() -> None:
    updates = [_prompt("go"), _text("agent_message_chunk", "going")]
    jsonl = [_row("user", "go"), _row("user", "also do X", steer=True), _row("assistant", "going")]
    out, _ = merge_replay_transcript(updates, jsonl)
    assert _sources(out) == [
        ("user", SOURCE_JSONL),
        ("user", SOURCE_JSONL),
        ("assistant", SOURCE_REPLAY),
    ]
    assert out[1]["meta"]["steer"] is True


def test_compaction_notice_stays_as_sidecar_after_its_turn() -> None:
    updates = [_prompt("q"), _text("agent_message_chunk", "a")]
    jsonl = [
        _row("user", "q"),
        _row("assistant", "a"),
        _row("assistant", "🔄 Auto-compacted at 91%.", kind="compaction"),
    ]
    out, _ = merge_replay_transcript(updates, jsonl)
    assert [r["content"] for r in out] == ["q", "a", "🔄 Auto-compacted at 91%."]
    assert out[2]["meta"]["source"] == SOURCE_JSONL


def test_tool_result_without_call_is_kept_visible() -> None:
    updates = [
        _prompt("x"),
        _tool_done("ghost", {"items": [{"Text": "orphan output"}]}),
        _text("agent_message_chunk", "ok"),
    ]
    out, _ = merge_replay_transcript(updates, [_row("user", "x")])
    tool = next(r for r in out if r["role"] == "tool")
    assert tool["meta"]["tool_call_id"] == "ghost" and "orphan output" in tool["meta"]["output"]


def test_pre_turn_engine_frames_are_ignored() -> None:
    # v3 replays an engine-internal tool call before the first user prompt.
    updates = [
        _tool_call("cfg", "fetch_cloud_config", {}, kind="other"),
        _tool_done("cfg", "{}"),
        _prompt("hi"),
        _text("agent_message_chunk", "hello"),
    ]
    out, _ = merge_replay_transcript(updates, [_row("user", "hi")])
    assert [r["role"] for r in out] == ["user", "assistant"]


def test_replay_text_is_redacted() -> None:
    updates = [_prompt("show"), _text("agent_message_chunk", "token: AKIAIOSFODNN7EXAMPLE end")]
    out, _ = merge_replay_transcript(updates, [_row("user", "show")])
    assert "AKIAIOSFODNN7EXAMPLE" not in out[1]["content"]
