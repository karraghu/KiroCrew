"""Conductor work ledger store — Phase 1 exit criteria, one test per criterion.

Pins what the conductor-work-ledger RFC (pull request #8842) §Migration plan
Phase 1 lists: every enum and cap fails a test if its value changes; two concurrent
writers against one item leave a parseable record and an uninterleaved event log; a
torn, truncated or oversized file reads as absent; a refused cap leaves the prior
bytes untouched; ``depth`` at the cap refuses ``create``; consecutive ``progress``
reports coalesce; a duplicated event line collapses on read; and nothing in
``src/kiro_crew`` imports the module, so the phase reverts by deleting two files.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from kiro_crew import work_ledger as wl

CONDUCTOR = "chat-1-conductor"
WORKER = "chat-2-worker"


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Every test writes into its own data home, never the live one."""
    monkeypatch.setenv("KIROCREW_HOME", str(tmp_path / "home"))
    yield


def _new_item(*, title: str = "port the gate", acceptance: dict | None = None) -> str:
    wl.ensure_conductor(CONDUCTOR, goal="drive the fleet")
    result = wl.apply_conductor_action(
        CONDUCTOR,
        "create",
        title=title,
        acceptance=acceptance if acceptance is not None else {"kind": "human_approval"},
    )
    return result["item"].item_id


def _bytes_on_disk(item_id: str) -> tuple[bytes, bytes]:
    item = wl.item_path(CONDUCTOR, item_id).read_bytes()
    events = wl.item_events_path(CONDUCTOR, item_id).read_bytes()
    return item, events


# ── vocabularies and caps, pinned ─────────────────────────────────────────


def test_item_states_are_exactly_the_four_dispositions():
    assert wl.ITEM_STATES == {"open", "accepted", "rejected", "abandoned"}
    assert wl.TERMINAL_ITEM_STATES == {"accepted", "rejected", "abandoned"}
    assert "open" not in wl.TERMINAL_ITEM_STATES


def test_verdicts_are_accept_evals_five_values():
    assert wl.VERDICTS == {"pass", "fail", "pending", "refused", "error"}


def test_worker_statuses_are_exactly_four_and_separate_blocked_from_question():
    assert wl.WORKER_STATUSES == {"progress", "done", "blocked", "question"}


def test_event_kinds_are_exactly_six():
    assert wl.EVENT_KINDS == {
        "create",
        "bind",
        "report",
        "decision",
        "verdict",
        "close",
    }


def test_conductor_actions_are_exactly_six():
    assert wl.CONDUCTOR_ACTIONS == {
        "create",
        "bind",
        "decide",
        "verdict",
        "close",
        "goal",
    }


def test_caps_hold_their_rfc_values():
    assert wl.MAX_ITEMS_PER_CONDUCTOR == 32
    assert wl.MAX_EVENTS_PER_ITEM == 200
    assert wl.MAX_DEPTH == 2
    assert wl.MAX_GOAL_CHARS == 2000
    assert wl.MAX_TITLE_CHARS == 200
    assert wl.MAX_DECISION_CHARS == 2000
    assert wl.MAX_SUMMARY_CHARS == 500
    assert wl.MAX_EVENT_TEXT_CHARS == 500
    assert wl.MAX_ARTIFACT_KEYS == 16
    assert wl.MAX_ARTIFACT_KEY_CHARS == 64
    assert wl.MAX_ARTIFACT_VALUE_CHARS == 512
    assert (wl.MIN_PR, wl.MAX_PR) == (1, 1_000_000_000)
    assert wl.SCHEMA_VERSION == 1


# ── paths and ids ─────────────────────────────────────────────────────────


def test_item_id_is_server_minted_and_shape_checked():
    minted = wl.mint_item_id()
    assert wl._ITEM_ID_RE.match(minted)
    assert wl.mint_item_id() != minted


@pytest.mark.parametrize(
    "bad",
    ["", "it_", "it_XYZ", "it_1a2b3c4", "../etc/passwd", "/abs/it_1a2b3c4d", "it_1a2b3c4d.json"],
)
def test_a_model_supplied_string_cannot_reach_a_path_component(bad):
    with pytest.raises(wl.WorkLedgerError) as caught:
        wl.item_path(CONDUCTOR, bad)
    assert caught.value.code == wl.CODE_INVALID_VALUE


def test_directory_name_is_readable_plus_digest_and_distinguishes_case():
    name = wl._store_name("chat-1-abc")
    assert name.startswith("chat-1-abc-")
    assert len(name.rsplit("-", 1)[1]) == 8
    assert wl._store_name("Chat") != wl._store_name("chat")
    assert "/" not in wl._store_name("slack:C1/x")


@pytest.mark.parametrize("bad", ["", "a/b", "a\\b", "a\0b"])
def test_conductor_dir_refuses_a_key_that_could_escape_the_root(bad):
    with pytest.raises(wl.WorkLedgerError):
        wl.conductor_dir(bad)


def test_binding_path_shares_the_conductor_naming_scheme():
    assert wl.binding_path(WORKER).name == f"{wl._store_name(WORKER)}.json"
    assert wl.binding_path(WORKER).parent == wl.bindings_dir()
    with pytest.raises(wl.WorkLedgerError):
        wl.binding_path("has\0null")


# ── conductor record ──────────────────────────────────────────────────────


def test_ensure_conductor_is_idempotent_and_never_resets_progress():
    first = wl.ensure_conductor(CONDUCTOR, goal="ship it")
    wl.apply_conductor_action(CONDUCTOR, "goal", goal="ship it", round_number=4)
    again = wl.ensure_conductor(CONDUCTOR, goal="something else")
    assert again.round == 4
    assert again.goal == "ship it"
    assert again.created_at == first.created_at
    assert (wl.conductor_dir(CONDUCTOR) / "slot_key").read_text().strip() == CONDUCTOR


def test_read_conductor_is_none_before_anything_is_written():
    assert wl.read_conductor(CONDUCTOR) is None
    with pytest.raises(wl.WorkLedgerError) as caught:
        wl.apply_conductor_action(CONDUCTOR, "create", title="x", acceptance={})
    assert caught.value.code == wl.CODE_NO_LEDGER


def test_conductor_record_has_no_item_roster_field():
    wl.ensure_conductor(CONDUCTOR)
    stored = json.loads((wl.conductor_dir(CONDUCTOR) / "conductor.json").read_text())
    assert "items" not in stored
    assert "item_roster" not in stored


def test_goal_action_leaves_the_goal_alone_when_only_the_round_moves():
    wl.ensure_conductor(CONDUCTOR, goal="keep me")
    record = wl.apply_conductor_action(CONDUCTOR, "goal", round_number=2)["conductor"]
    assert (record.goal, record.round) == ("keep me", 2)


def test_ensure_conductor_refuses_a_depth_past_the_cap():
    with pytest.raises(wl.WorkLedgerError) as caught:
        wl.ensure_conductor("deep", depth=3)
    assert caught.value.code == wl.CODE_DEPTH_EXCEEDED


def test_ensure_conductor_records_a_parent_item_and_checks_its_shape():
    parent = wl.mint_item_id()
    record = wl.ensure_conductor("child", depth=1, parent_item=parent)
    assert (record.depth, record.parent_item) == (1, parent)
    with pytest.raises(wl.WorkLedgerError):
        wl.ensure_conductor("child2", depth=1, parent_item="not-an-id")


# ── depth ─────────────────────────────────────────────────────────────────


def test_child_depth_advances_one_level_and_refuses_at_the_cap():
    assert wl.child_depth(0) == 1
    assert wl.child_depth(1) == 2
    with pytest.raises(wl.WorkLedgerError) as caught:
        wl.child_depth(2)
    assert caught.value.code == wl.CODE_DEPTH_EXCEEDED


def test_create_is_refused_when_the_conductor_is_at_the_depth_cap():
    wl.ensure_conductor("capped", goal="g", depth=wl.MAX_DEPTH)
    with pytest.raises(wl.WorkLedgerError) as caught:
        wl.apply_conductor_action("capped", "create", title="t", acceptance={})
    assert caught.value.code == wl.CODE_DEPTH_EXCEEDED
    assert wl.list_work_items("capped") == []


def test_a_conductor_one_below_the_cap_may_still_dispatch():
    wl.ensure_conductor("mid", goal="g", depth=1)
    result = wl.apply_conductor_action("mid", "create", title="t", acceptance={})
    assert result["item"].item_id.startswith("it_")


# ── item lifecycle ────────────────────────────────────────────────────────


def test_create_mints_an_item_and_appends_exactly_one_create_event():
    item_id = _new_item()
    item = wl.read_work_item(CONDUCTOR, item_id)
    assert item is not None
    assert (item.state, item.status, item.verdict) == ("open", None, None)
    events = wl.read_events(CONDUCTOR, item_id)
    assert [e.kind for e in events] == ["create"]


def test_items_are_derived_from_the_directory_and_sorted_oldest_first():
    first = _new_item(title="one")
    second = _new_item(title="two")
    listed = [item.item_id for item in wl.list_work_items(CONDUCTOR)]
    assert set(listed) == {first, second}
    assert wl.list_work_items("never-existed") == []


def test_bind_writes_the_binding_and_refuses_a_second_one():
    item_id = _new_item()
    wl.apply_conductor_action(CONDUCTOR, "bind", item_id=item_id, worker_session_key=WORKER)
    assert wl.read_binding(WORKER) == (CONDUCTOR, item_id)
    with pytest.raises(wl.WorkLedgerError) as caught:
        wl.apply_conductor_action(CONDUCTOR, "bind", item_id=item_id, worker_session_key="chat-3")
    assert caught.value.code == wl.CODE_ALREADY_BOUND


def test_read_binding_is_none_when_absent_or_malformed():
    assert wl.read_binding(WORKER) is None
    path = wl.binding_path(WORKER)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"conductor_slot_key": CONDUCTOR, "item_id": "../x"}))
    assert wl.read_binding(WORKER) is None
    path.write_text("not json")
    assert wl.read_binding(WORKER) is None


def test_decide_verdict_and_close_each_move_one_field_and_log_one_event():
    item_id = _new_item()
    wl.apply_conductor_action(CONDUCTOR, "decide", item_id=item_id, decision="retry once")
    wl.apply_conductor_action(CONDUCTOR, "verdict", item_id=item_id, verdict="fail", fails=1)
    mid = wl.read_work_item(CONDUCTOR, item_id)
    assert mid is not None
    assert (mid.decision, mid.verdict, mid.fails, mid.state) == ("retry once", "fail", 1, "open")
    wl.apply_conductor_action(
        CONDUCTOR, "close", item_id=item_id, state="accepted", decision="landed"
    )
    closed = wl.read_work_item(CONDUCTOR, item_id)
    assert closed is not None
    assert (closed.state, closed.decision) == ("accepted", "landed")
    assert closed.closed_at
    assert [e.kind for e in wl.read_events(CONDUCTOR, item_id)] == [
        "create",
        "decision",
        "verdict",
        "close",
    ]


def test_a_terminal_item_refuses_every_further_write():
    item_id = _new_item()
    wl.apply_conductor_action(CONDUCTOR, "close", item_id=item_id, state="abandoned")
    for call in (
        lambda: wl.apply_conductor_action(CONDUCTOR, "decide", item_id=item_id, decision="again"),
        lambda: wl.apply_worker_report(CONDUCTOR, item_id, status="progress", summary="still here"),
    ):
        with pytest.raises(wl.WorkLedgerError) as caught:
            call()
        assert caught.value.code == wl.CODE_ITEM_CLOSED


def test_close_refuses_open_as_a_terminal_state():
    item_id = _new_item()
    with pytest.raises(wl.WorkLedgerError) as caught:
        wl.apply_conductor_action(CONDUCTOR, "close", item_id=item_id, state="open")
    assert caught.value.field == "state"


def test_an_unknown_item_and_an_unknown_action_are_named_distinctly():
    wl.ensure_conductor(CONDUCTOR)
    with pytest.raises(wl.WorkLedgerError) as unknown:
        wl.apply_conductor_action(CONDUCTOR, "decide", item_id=wl.mint_item_id(), decision="x")
    assert unknown.value.code == wl.CODE_UNKNOWN_ITEM
    with pytest.raises(wl.WorkLedgerError) as action:
        wl.apply_conductor_action(CONDUCTOR, "teleport")
    assert action.value.code == wl.CODE_INVALID_ACTION


def test_round_rides_along_with_a_conductor_action():
    item_id = _new_item()
    wl.apply_conductor_action(CONDUCTOR, "decide", item_id=item_id, decision="d", round_number=7)
    item = wl.read_work_item(CONDUCTOR, item_id)
    assert item is not None and item.round == 7


# ── writer ownership ──────────────────────────────────────────────────────


def test_the_worker_path_writes_only_worker_fields():
    item_id = _new_item()
    wl.apply_conductor_action(CONDUCTOR, "decide", item_id=item_id, decision="conductor said")
    wl.apply_worker_report(
        CONDUCTOR,
        item_id,
        status="done",
        summary="tests pass",
        artifacts={"pr": "8842"},
        pr=8842,
    )
    item = wl.read_work_item(CONDUCTOR, item_id)
    assert item is not None
    assert (item.status, item.summary, item.pr) == ("done", "tests pass", 8842)
    assert item.artifacts == {"pr": "8842"}
    assert item.last_report_at
    # Untouched by the worker: the conductor still owns the bar and the disposition.
    assert (item.decision, item.state, item.verdict) == ("conductor said", "open", None)


def test_apply_worker_report_has_no_conductor_field_parameter():
    import inspect

    names = set(inspect.signature(wl.apply_worker_report).parameters)
    assert names == {"slot_key", "item_id", "status", "summary", "artifacts", "pr"}
    assert not names & {"verdict", "state", "acceptance", "decision", "fails", "round_number"}


def test_accept_batch_is_built_from_acceptance_and_never_from_a_claimed_pr():
    item_id = _new_item(acceptance={"kind": "pr_checks", "repo": "owner/name"})
    wl.apply_worker_report(CONDUCTOR, item_id, status="done", summary="s", pr=999)
    batch = wl.accept_batch(wl.list_work_items(CONDUCTOR))
    assert batch == {
        "items": [{"item_id": item_id, "acceptance": {"kind": "pr_checks", "repo": "owner/name"}}]
    }
    assert "999" not in json.dumps(batch)


def test_accept_batch_drops_terminal_items_and_items_with_no_bar():
    kept = _new_item(acceptance={"kind": "human_approval"})
    bare = _new_item(acceptance={})
    closed = _new_item(acceptance={"kind": "human_approval"})
    wl.apply_conductor_action(CONDUCTOR, "close", item_id=closed, state="accepted")
    ids = [entry["item_id"] for entry in wl.accept_batch(wl.list_work_items(CONDUCTOR))["items"]]
    assert ids == [kept]
    assert bare not in ids


def test_work_brief_shows_the_item_and_not_the_conductors_goal():
    item_id = _new_item(title="one job")
    brief = wl.read_work_brief(CONDUCTOR, item_id)
    assert brief is not None
    assert brief["title"] == "one job"
    assert set(brief) == {
        "item_id",
        "title",
        "acceptance",
        "round",
        "decision",
        "status",
        "summary",
    }
    assert "goal" not in brief
    assert "worker_session_key" not in brief
    assert wl.read_work_brief(CONDUCTOR, wl.mint_item_id()) is None


# ── caps refuse, and a refusal changes no bytes ───────────────────────────


def test_the_item_cap_refuses_the_thirty_third_item():
    wl.ensure_conductor(CONDUCTOR, goal="g")
    for index in range(wl.MAX_ITEMS_PER_CONDUCTOR):
        wl.apply_conductor_action(CONDUCTOR, "create", title=f"t{index}", acceptance={})
    before = sorted(p.name for p in wl.items_dir(CONDUCTOR).iterdir())
    with pytest.raises(wl.WorkLedgerError) as caught:
        wl.apply_conductor_action(CONDUCTOR, "create", title="one too many", acceptance={})
    assert caught.value.code == wl.CODE_ITEM_CAP_EXCEEDED
    assert sorted(p.name for p in wl.items_dir(CONDUCTOR).iterdir()) == before


@pytest.mark.parametrize(
    "kwargs,expected_field",
    [
        ({"status": "progress", "summary": "x" * (wl.MAX_SUMMARY_CHARS + 1)}, "summary"),
        (
            {
                "status": "progress",
                "summary": "ok",
                "artifacts": {f"k{i}": "v" for i in range(wl.MAX_ARTIFACT_KEYS + 1)},
            },
            "artifacts",
        ),
        (
            {
                "status": "progress",
                "summary": "ok",
                "artifacts": {"k" * (wl.MAX_ARTIFACT_KEY_CHARS + 1): "v"},
            },
            "artifacts",
        ),
        (
            {
                "status": "progress",
                "summary": "ok",
                "artifacts": {"k": "v" * (wl.MAX_ARTIFACT_VALUE_CHARS + 1)},
            },
            "artifacts",
        ),
    ],
)
def test_a_refused_worker_cap_leaves_both_files_byte_identical(kwargs, expected_field):
    item_id = _new_item()
    wl.apply_worker_report(CONDUCTOR, item_id, status="progress", summary="baseline")
    before = _bytes_on_disk(item_id)
    with pytest.raises(wl.WorkLedgerError) as caught:
        wl.apply_worker_report(CONDUCTOR, item_id, **kwargs)
    assert caught.value.code == wl.CODE_FIELD_TOO_LONG
    assert caught.value.field == expected_field
    assert _bytes_on_disk(item_id) == before


def test_a_refused_conductor_cap_leaves_both_files_byte_identical():
    item_id = _new_item()
    before = _bytes_on_disk(item_id)
    with pytest.raises(wl.WorkLedgerError) as caught:
        wl.apply_conductor_action(
            CONDUCTOR,
            "decide",
            item_id=item_id,
            decision="d" * (wl.MAX_DECISION_CHARS + 1),
        )
    assert (caught.value.code, caught.value.field) == (wl.CODE_FIELD_TOO_LONG, "decision")
    assert _bytes_on_disk(item_id) == before


def test_an_oversized_title_and_goal_are_refused_not_truncated():
    wl.ensure_conductor(CONDUCTOR, goal="g")
    for action, kwargs, name in (
        ("create", {"title": "t" * (wl.MAX_TITLE_CHARS + 1), "acceptance": {}}, "title"),
        ("goal", {"goal": "g" * (wl.MAX_GOAL_CHARS + 1)}, "goal"),
    ):
        with pytest.raises(wl.WorkLedgerError) as caught:
            wl.apply_conductor_action(CONDUCTOR, action, **kwargs)
        assert (caught.value.code, caught.value.field) == (wl.CODE_FIELD_TOO_LONG, name)
    record = wl.read_conductor(CONDUCTOR)
    assert record is not None and record.goal == "g"


def test_a_boundary_length_value_is_accepted():
    item_id = _new_item(title="t" * wl.MAX_TITLE_CHARS)
    wl.apply_worker_report(
        CONDUCTOR,
        item_id,
        status="progress",
        summary="s" * wl.MAX_SUMMARY_CHARS,
        artifacts={"k" * wl.MAX_ARTIFACT_KEY_CHARS: "v" * wl.MAX_ARTIFACT_VALUE_CHARS},
    )
    item = wl.read_work_item(CONDUCTOR, item_id)
    assert item is not None and len(item.summary) == wl.MAX_SUMMARY_CHARS


@pytest.mark.parametrize("bad_pr", [0, -1, wl.MAX_PR + 1, True, 1.5, float("nan"), "12", object()])
def test_pr_is_bounded_and_a_bool_is_not_an_integer(bad_pr):
    item_id = _new_item()
    before = _bytes_on_disk(item_id)
    with pytest.raises(wl.WorkLedgerError) as caught:
        wl.apply_worker_report(CONDUCTOR, item_id, status="done", summary="s", pr=bad_pr)
    assert caught.value.field == "pr"
    assert _bytes_on_disk(item_id) == before


def test_an_unknown_status_is_refused_with_its_own_code():
    item_id = _new_item()
    with pytest.raises(wl.WorkLedgerError) as caught:
        wl.apply_worker_report(CONDUCTOR, item_id, status="finished", summary="s")
    assert caught.value.code == wl.CODE_INVALID_STATUS


def test_wrong_typed_fields_are_refused_rather_than_coerced():
    item_id = _new_item()
    with pytest.raises(wl.WorkLedgerError):
        wl.apply_worker_report(CONDUCTOR, item_id, status="done", summary=17)
    with pytest.raises(wl.WorkLedgerError):
        wl.apply_worker_report(CONDUCTOR, item_id, status="done", summary="s", artifacts=["x"])
    with pytest.raises(wl.WorkLedgerError):
        wl.apply_worker_report(CONDUCTOR, item_id, status="done", summary="s", artifacts={"k": 1})
    with pytest.raises(wl.WorkLedgerError):
        wl.apply_conductor_action(CONDUCTOR, "create", title="t", acceptance=["nope"])
    with pytest.raises(wl.WorkLedgerError):
        wl.apply_conductor_action(CONDUCTOR, "verdict", item_id=item_id, verdict="fail", fails=-1)
    with pytest.raises(wl.WorkLedgerError):
        wl.apply_conductor_action(CONDUCTOR, "verdict", item_id=item_id, verdict="maybe")
    with pytest.raises(wl.WorkLedgerError):
        wl.apply_conductor_action(CONDUCTOR, "bind", item_id=item_id, worker_session_key="a\0b")


def test_acceptance_must_be_json_serialisable():
    wl.ensure_conductor(CONDUCTOR)
    with pytest.raises(wl.WorkLedgerError) as caught:
        wl.apply_conductor_action(CONDUCTOR, "create", title="t", acceptance={"fn": lambda: None})
    assert caught.value.field == "acceptance"


def test_an_oversized_acceptance_is_refused(monkeypatch):
    wl.ensure_conductor(CONDUCTOR)
    monkeypatch.setattr(wl, "MAX_RECORD_BYTES", 200)
    with pytest.raises(wl.WorkLedgerError) as caught:
        wl.apply_conductor_action(CONDUCTOR, "create", title="t", acceptance={"blob": "x" * 400})
    assert caught.value.code == wl.CODE_FIELD_TOO_LONG


def test_acceptance_is_stored_verbatim_and_never_interpreted():
    payload = {"kind": "pr_checks", "pr": 123, "repo": "owner/name", "extra": [1, {"a": None}]}
    item_id = _new_item(acceptance=payload)
    item = wl.read_work_item(CONDUCTOR, item_id)
    assert item is not None and item.acceptance == payload


# ── events ────────────────────────────────────────────────────────────────


def test_event_id_is_content_addressed_and_sixteen_hex():
    first = wl.event_id("2026-01-01T00:00:00+00:00", "it_1a2b3c4d", "report", "hello")
    assert len(first) == 16
    assert first == wl.event_id("2026-01-01T00:00:00+00:00", "it_1a2b3c4d", "report", "hello")
    assert first != wl.event_id("2026-01-01T00:00:00+00:00", "it_1a2b3c4d", "report", "hi")


def test_event_id_separators_keep_two_different_tuples_apart():
    # Bare concatenation would make these two identical.
    assert wl.event_id("a", "b", "create", "cd") != wl.event_id("a", "bc", "create", "d")


def test_a_duplicated_event_line_collapses_on_read():
    item_id = _new_item()
    path = wl.item_events_path(CONDUCTOR, item_id)
    line = path.read_text(encoding="utf-8").strip()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.write(line + "\n")
    assert len(wl.read_events(CONDUCTOR, item_id)) == 1


def test_a_torn_event_line_is_skipped_and_the_history_before_it_survives():
    item_id = _new_item()
    wl.apply_conductor_action(CONDUCTOR, "decide", item_id=item_id, decision="keep me")
    path = wl.item_events_path(CONDUCTOR, item_id)
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"id": "abc", "kind": "rep')
    kinds = [event.kind for event in wl.read_events(CONDUCTOR, item_id)]
    assert kinds == ["create", "decision"]


def test_a_line_with_an_unknown_kind_or_a_non_object_is_skipped():
    item_id = _new_item()
    path = wl.item_events_path(CONDUCTOR, item_id)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"id": "x", "kind": "teleport", "text": ""}) + "\n")
        handle.write("[1, 2, 3]\n")
        handle.write("\n")
    assert [event.kind for event in wl.read_events(CONDUCTOR, item_id)] == ["create"]


def test_the_event_cap_drops_the_oldest_line():
    item_id = _new_item()
    for index in range(wl.MAX_EVENTS_PER_ITEM + 5):
        wl.apply_conductor_action(CONDUCTOR, "decide", item_id=item_id, decision=f"round {index}")
    events = wl.read_events(CONDUCTOR, item_id)
    assert len(events) == wl.MAX_EVENTS_PER_ITEM
    assert events[0].kind != "create"
    assert events[-1].text == f"round {wl.MAX_EVENTS_PER_ITEM + 4}"


def test_read_events_limit_keeps_the_newest():
    item_id = _new_item()
    for index in range(4):
        wl.apply_conductor_action(CONDUCTOR, "decide", item_id=item_id, decision=str(index))
    assert [e.text for e in wl.read_events(CONDUCTOR, item_id, limit=2)] == ["2", "3"]
    assert wl.read_events(CONDUCTOR, item_id, limit=0) == []


def test_event_text_is_bounded_on_the_line_without_refusing_the_write():
    item_id = _new_item()
    wl.apply_conductor_action(
        CONDUCTOR, "decide", item_id=item_id, decision="d" * wl.MAX_DECISION_CHARS
    )
    item = wl.read_work_item(CONDUCTOR, item_id)
    events = wl.read_events(CONDUCTOR, item_id)
    assert item is not None and len(item.decision) == wl.MAX_DECISION_CHARS
    assert len(events[-1].text) == wl.MAX_EVENT_TEXT_CHARS


def test_appending_an_unknown_event_kind_is_refused():
    item_id = _new_item()
    with wl.item_lock(CONDUCTOR, item_id):
        with pytest.raises(wl.WorkLedgerError):
            wl._append_event_locked(CONDUCTOR, item_id, "teleport", "x")


def test_read_events_is_empty_for_an_absent_or_oversized_log(monkeypatch):
    item_id = _new_item()
    assert wl.read_events(CONDUCTOR, wl.mint_item_id()) == []
    monkeypatch.setattr(wl, "MAX_RECORD_BYTES", 1)
    assert wl.read_events(CONDUCTOR, item_id) == []


# ── the progress coalescing rule (RFC Q6) ─────────────────────────────────


def test_consecutive_progress_reports_collapse_to_the_newest():
    item_id = _new_item()
    for step in ("first", "second", "third"):
        wl.apply_worker_report(CONDUCTOR, item_id, status="progress", summary=step)
    events = wl.read_events(CONDUCTOR, item_id)
    assert [(e.kind, e.text) for e in events] == [
        ("create", "port the gate"),
        ("report", "third"),
    ]
    item = wl.read_work_item(CONDUCTOR, item_id)
    assert item is not None and item.summary == "third"


def test_a_status_change_between_two_progress_reports_stops_the_merge():
    item_id = _new_item()
    wl.apply_worker_report(CONDUCTOR, item_id, status="progress", summary="a")
    wl.apply_worker_report(CONDUCTOR, item_id, status="blocked", summary="b")
    wl.apply_worker_report(CONDUCTOR, item_id, status="progress", summary="c")
    texts = [e.text for e in wl.read_events(CONDUCTOR, item_id) if e.kind == "report"]
    assert texts == ["a", "b", "c"]


def test_a_conductor_event_between_two_progress_reports_stops_the_merge():
    item_id = _new_item()
    wl.apply_worker_report(CONDUCTOR, item_id, status="progress", summary="a")
    wl.apply_conductor_action(CONDUCTOR, "decide", item_id=item_id, decision="carry on")
    wl.apply_worker_report(CONDUCTOR, item_id, status="progress", summary="b")
    assert [e.kind for e in wl.read_events(CONDUCTOR, item_id)] == [
        "create",
        "report",
        "decision",
        "report",
    ]


def test_only_progress_reports_merge_and_never_done_reports():
    item_id = _new_item()
    wl.apply_worker_report(CONDUCTOR, item_id, status="done", summary="a")
    wl.apply_worker_report(CONDUCTOR, item_id, status="done", summary="b")
    texts = [e.text for e in wl.read_events(CONDUCTOR, item_id) if e.kind == "report"]
    assert texts == ["a", "b"]


# ── corruption reads as absent ────────────────────────────────────────────


def test_a_truncated_item_file_reads_as_absent():
    item_id = _new_item()
    path = wl.item_path(CONDUCTOR, item_id)
    raw = path.read_text(encoding="utf-8")
    path.write_text(raw[: len(raw) // 2], encoding="utf-8")
    assert wl.read_work_item(CONDUCTOR, item_id) is None
    assert wl.list_work_items(CONDUCTOR) == []


def test_a_non_utf8_item_file_reads_as_absent():
    item_id = _new_item()
    wl.item_path(CONDUCTOR, item_id).write_bytes(b"\xff\xfe not utf-8")
    assert wl.read_work_item(CONDUCTOR, item_id) is None


def test_an_oversized_item_file_reads_as_absent(monkeypatch, caplog):
    item_id = _new_item()
    monkeypatch.setattr(wl, "MAX_RECORD_BYTES", 4)
    with caplog.at_level("WARNING"):
        assert wl.read_work_item(CONDUCTOR, item_id) is None
    assert "size ceiling" in caplog.text


def test_a_corrupt_conductor_file_reads_as_absent():
    wl.ensure_conductor(CONDUCTOR, goal="g")
    (wl.conductor_dir(CONDUCTOR) / "conductor.json").write_text("{ torn", encoding="utf-8")
    assert wl.read_conductor(CONDUCTOR) is None


def test_one_torn_item_does_not_hide_its_siblings():
    good = _new_item(title="good")
    bad = _new_item(title="bad")
    wl.item_path(CONDUCTOR, bad).write_text("{", encoding="utf-8")
    assert [item.item_id for item in wl.list_work_items(CONDUCTOR)] == [good]


def test_a_wrong_typed_stored_field_resets_to_its_default_without_raising():
    item_id = _new_item()
    path = wl.item_path(CONDUCTOR, item_id)
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored.update(
        {
            "state": "teleported",
            "status": 17,
            "verdict": "maybe",
            "round": "many",
            "artifacts": {"keep": "me", "drop": 5},
            "title": None,
            "pr": True,
            "acceptance": "not an object",
        }
    )
    path.write_text(json.dumps(stored), encoding="utf-8")
    item = wl.read_work_item(CONDUCTOR, item_id)
    assert item is not None
    assert (item.state, item.status, item.verdict, item.round) == ("open", None, None, 0)
    assert item.artifacts == {"keep": "me"}
    assert (item.title, item.pr, item.acceptance) == ("", None, {})


def test_records_built_from_a_non_object_are_empty_defaults():
    assert wl.ConductorRecord.from_dict("nope").goal == ""
    assert wl.WorkItem.from_dict(None).state == "open"
    assert wl.WorkEvent.from_dict([1]) is None


def test_read_helpers_backfill_an_id_the_stored_record_lost():
    item_id = _new_item()
    path = wl.item_path(CONDUCTOR, item_id)
    stored = json.loads(path.read_text(encoding="utf-8"))
    del stored["item_id"]
    path.write_text(json.dumps(stored), encoding="utf-8")
    item = wl.read_work_item(CONDUCTOR, item_id)
    assert item is not None and item.item_id == item_id

    conductor_file = wl.conductor_dir(CONDUCTOR) / "conductor.json"
    header = json.loads(conductor_file.read_text(encoding="utf-8"))
    del header["slot_key"]
    conductor_file.write_text(json.dumps(header), encoding="utf-8")
    record = wl.read_conductor(CONDUCTOR)
    assert record is not None and record.slot_key == CONDUCTOR


# ── derived flags ─────────────────────────────────────────────────────────


def test_orphaned_is_derived_and_is_never_a_stored_field():
    item_id = _new_item()
    item = wl.read_work_item(CONDUCTOR, item_id)
    assert item is not None
    assert "orphaned" not in item.to_dict()
    assert "stale" not in item.to_dict()
    assert wl.is_orphaned(item, conductor_slot_exists=False) is True
    assert wl.is_orphaned(item, conductor_slot_exists=True) is False


def test_a_terminal_item_is_neither_orphaned_nor_stale():
    item_id = _new_item()
    wl.apply_conductor_action(CONDUCTOR, "close", item_id=item_id, state="accepted")
    item = wl.read_work_item(CONDUCTOR, item_id)
    assert item is not None
    assert wl.is_orphaned(item, conductor_slot_exists=False) is False
    assert wl.is_stale(item, worker_running=False, window_secs=0) is False


def test_stale_needs_both_a_quiet_item_and_a_stopped_worker():
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    quiet = wl.WorkItem(
        item_id=wl.mint_item_id(),
        created_at=(now - timedelta(hours=2)).isoformat(),
        last_report_at=(now - timedelta(hours=1)).isoformat(),
    )
    assert wl.is_stale(quiet, worker_running=False, now=now, window_secs=60) is True
    # Running is the whole point of the conjunction: a long build is never stale.
    assert wl.is_stale(quiet, worker_running=True, now=now, window_secs=60) is False
    assert wl.is_stale(quiet, worker_running=False, now=now, window_secs=7200) is False


def test_an_item_with_no_report_yet_is_measured_from_creation():
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    fresh = wl.WorkItem(item_id=wl.mint_item_id(), created_at=now.isoformat())
    assert wl.is_stale(fresh, worker_running=False, now=now, window_secs=60) is False
    old = wl.WorkItem(item_id=wl.mint_item_id(), created_at=(now - timedelta(hours=1)).isoformat())
    assert wl.is_stale(old, worker_running=False, now=now, window_secs=60) is True


def test_an_unparseable_or_missing_timestamp_reads_as_stale():
    assert wl.is_stale(wl.WorkItem(), worker_running=False) is True
    assert wl.is_stale(wl.WorkItem(last_report_at="not a date"), worker_running=False) is True


def test_naive_timestamps_are_compared_without_raising():
    naive_now = datetime(2026, 1, 1, 12, 0)
    item = wl.WorkItem(last_report_at="2026-01-01T10:00:00")
    assert wl.is_stale(item, worker_running=False, now=naive_now, window_secs=60) is True


def test_stale_uses_a_default_window_when_none_is_given():
    assert wl.DEFAULT_STALE_WINDOW_SECS > 0
    recent = wl.WorkItem(last_report_at=datetime.now().astimezone().isoformat())
    assert wl.is_stale(recent, worker_running=False) is False


# ── concurrency ───────────────────────────────────────────────────────────


def test_two_concurrent_writers_leave_a_parseable_item_and_a_clean_event_log():
    """One report loop and one conductor loop against the SAME item.

    Threads rather than processes so the test runs unchanged on Windows: the locks
    are taken on separate descriptors, which serialise across threads exactly as
    they do across processes, and nothing here uses a POSIX-only API.
    """
    item_id = _new_item()
    rounds = 25
    errors: list[BaseException] = []

    def report() -> None:
        try:
            for index in range(rounds):
                wl.apply_worker_report(
                    CONDUCTOR,
                    item_id,
                    status="blocked" if index % 2 else "done",
                    summary=f"worker {index}",
                    artifacts={"step": str(index)},
                )
        except BaseException as exc:  # pragma: no cover - surfaced by the assert
            errors.append(exc)

    def decide() -> None:
        try:
            for index in range(rounds):
                wl.apply_conductor_action(
                    CONDUCTOR, "decide", item_id=item_id, decision=f"conductor {index}"
                )
        except BaseException as exc:  # pragma: no cover - surfaced by the assert
            errors.append(exc)

    threads = [threading.Thread(target=report), threading.Thread(target=decide)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    assert not errors, errors
    assert not any(thread.is_alive() for thread in threads)

    # The item still parses, and holds one writer's field beside the other's.
    stored = json.loads(wl.item_path(CONDUCTOR, item_id).read_text(encoding="utf-8"))
    assert stored["item_id"] == item_id
    assert stored["decision"].startswith("conductor ")
    assert stored["summary"].startswith("worker ")

    # Every line is a whole JSON object — no interleaving, no partial line.
    raw = wl.item_events_path(CONDUCTOR, item_id).read_text(encoding="utf-8")
    lines = [line for line in raw.splitlines() if line.strip()]
    assert lines
    for line in lines:
        parsed = json.loads(line)
        assert parsed["item_id"] == item_id
        assert parsed["kind"] in wl.EVENT_KINDS
    assert len(wl.read_events(CONDUCTOR, item_id)) == len(lines)


def test_the_item_cap_holds_under_concurrent_creates():
    wl.ensure_conductor(CONDUCTOR, goal="g")
    refused: list[str] = []

    def create() -> None:
        for _ in range(12):
            try:
                wl.apply_conductor_action(CONDUCTOR, "create", title="t", acceptance={})
            except wl.WorkLedgerError as exc:
                refused.append(exc.code)

    threads = [threading.Thread(target=create) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    assert len(wl.list_work_items(CONDUCTOR)) == wl.MAX_ITEMS_PER_CONDUCTOR
    assert refused and set(refused) == {wl.CODE_ITEM_CAP_EXCEEDED}


def test_the_lock_order_is_conductor_then_item_and_is_documented():
    doc = wl.__doc__ or ""
    assert "conductor -> item" in doc
    assert "conductor" in (wl.conductor_lock.__doc__ or "").lower()
    assert "FIRST" in (wl.conductor_lock.__doc__ or "")
    assert "SECOND" in (wl.item_lock.__doc__ or "")


# ── revertability ─────────────────────────────────────────────────────────


def test_nothing_in_the_package_imports_the_module_yet():
    """Phase 1 reverts by deleting one module and one test.

    Asserted on IMPORT statements rather than any mention of the name, and the
    candidate set is asserted non-empty so a moved source tree fails this test
    instead of hollowing it out.
    """
    package = Path(__file__).resolve().parents[1] / "src" / "kiro_crew"
    sources = [path for path in package.rglob("*.py") if path.name != "work_ledger.py"]
    assert len(sources) > 100, f"expected the package tree, found {len(sources)} files"
    offenders = []
    for path in sources:
        text = path.read_text(encoding="utf-8", errors="replace")
        if "import work_ledger" in text or "from kiro_crew.work_ledger" in text:
            offenders.append(str(path.relative_to(package)))
    assert offenders == []
