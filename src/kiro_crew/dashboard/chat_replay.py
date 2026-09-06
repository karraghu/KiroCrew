"""Render a resumed session's transcript from kiro-cli's ``session/load`` replay.

Prototype behind ``dashboard.replay_from_acp`` (default off).

Today a resumed dashboard session shows the JSONL Kiro Crew wrote turn by turn,
while the frames kiro-cli replays during ``session/load`` -- the agent's OWN
record of the conversation -- are counted and dropped by the runtime's reader.
With the flag on, :class:`~kiro_crew.acp.runtime.AcpRuntime` keeps those frames
(``AcpSessionHandle.replay_updates``) and this module turns them into transcript
rows through the SAME parser the live path uses -- reached through
:func:`kiro_crew.agent_sdk.fold_replay_updates`, the SDK's driver seam, so this
dashboard module never imports the ACP layer -- then overlays from the JSONL only
what the agent never saw.

Measured split (research/acp-replay, kiro-cli 2.21.0, engines v1/v2/v3):

* replay is authoritative for assistant text, thinking text, and every tool
  call's id / rawInput / rawOutput / final status;
* replay carries each user turn as the FULL assembled prompt (system prompt,
  context blocks, the user's words after ``[CURRENT USER REQUEST -- respond to
  this]``), so the clean user text comes from the JSONL row when one matches;
* on the v1/v2 engines a replayed tool call's ``title`` collapses to the raw
  tool name and ``locations`` are gone, so the humanized title / purpose / kind
  are borrowed from the JSONL tool row with the same ``tool_call_id``;
* notices, errors, approvals, recovery/steer/inject provenance, compaction
  banners and cron/subagent labels exist only in the JSONL -- they are the
  sidecar and are re-inserted at their turn, anchored on the tool call they
  followed when there was one.

Every emitted row is tagged ``meta.source`` (``acp_replay`` | ``jsonl``) so the
dashboard can show where it came from. The JSONL keeps being written -- this
changes what a resumed session READS, never what it persists.
"""

from __future__ import annotations

import logging
from typing import Any

from kiro_crew.agent_sdk import fold_replay_updates
from kiro_crew.context import USER_REQUEST_HEADER
from kiro_crew.dashboard.chat_utils import (
    _MAX_TOOL_PURPOSE,
    _redact_tool_field,
    effective_session_key,
)
from kiro_crew.quick_prompts import quick_prompt_header
from kiro_crew.security import redact_credentials, redact_exfiltration_urls

logger = logging.getLogger(__name__)

SOURCE_REPLAY = "acp_replay"
SOURCE_JSONL = "jsonl"

#: The assembled prompt marks the human's words with :data:`USER_REQUEST_HEADER`
#: (owned by ``context.py``, imported rather than respelled); the replayed
#: ``user_message_chunk`` is the whole prompt, so the clean text is whatever
#: follows the LAST occurrence. Compared dash-folded because the outbound
#: sanitizer rewrites the header's em dash to ASCII on the wire. Measured present
#: in 113/115 prompts of a real session; the rest were short raw prompts.
#: Bound on the raw prompt kept when no marker is present and no JSONL row
#: matches: a multi-KB context blob is not a chat bubble.
_RAW_PROMPT_KEEP = 4_000
#: JSONL roles that mean "this row started a prompt the agent received".
_PROMPT_ROLES = frozenset({"user", "nudge"})


def _fold_dashes(text: str) -> str:
    """ASCII-fold em/en dashes the way the outbound prompt sanitizer does."""
    return (text or "").replace("\u2014", "--").replace("\u2013", "-")


def _norm(text: str) -> str:
    """Whitespace-fold and ASCII-fold dashes so JSONL text matches the wire text.

    Kiro Crew rewrites em/en dashes to ASCII before a prompt reaches kiro-cli,
    so without folding both sides every nudge row looks unmatched.
    """
    return " ".join(_fold_dashes(text).split())


def _redact_text(text: str) -> str:
    safe, _ = redact_exfiltration_urls(text or "")
    safe, _ = redact_credentials(safe)
    return safe


def clean_user_text(prompt: str) -> str:
    """The human's words out of an assembled prompt (see :data:`USER_REQUEST_HEADER`)."""
    if not prompt:
        return ""
    folded = _fold_dashes(prompt)
    marker = _fold_dashes(USER_REQUEST_HEADER)
    idx = folded.rfind(marker)
    if idx >= 0:
        # Dash folding never changes the character count, so the index is valid
        # in the unfolded prompt too — the user's own dashes are kept as typed.
        return prompt[idx + len(marker) :].strip()
    if len(prompt) > _RAW_PROMPT_KEEP:
        return prompt[-_RAW_PROMPT_KEEP:].strip()
    return prompt.strip()


def _meta_of(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("meta")
    return meta if isinstance(meta, dict) else {}


def _row_text(row: dict[str, Any]) -> str:
    content = row.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return ""


def _is_prompt_row(row: dict[str, Any]) -> bool:
    if row.get("role") not in _PROMPT_ROLES:
        return False
    meta = _meta_of(row)
    # A steer is delivered INSIDE the running turn, not as its own prompt.
    return not meta.get("steer")


def _jsonl_turns(
    rows: list[dict[str, Any]],
) -> list[tuple[dict[str, Any] | None, list[dict[str, Any]]]]:
    """Group JSONL rows as (prompt_row | None, rows_after_it_until_next_prompt)."""
    groups: list[tuple[dict[str, Any] | None, list[dict[str, Any]]]] = []
    pre: list[dict[str, Any]] = []
    cur: list[dict[str, Any]] | None = None
    for row in rows:
        if _is_prompt_row(row):
            cur = []
            groups.append((row, cur))
            continue
        if cur is None:
            pre.append(row)
        else:
            cur.append(row)
    if pre:
        groups.insert(0, (None, pre))
    return groups


#: Below this many normalized characters a JSONL prompt is matched by EQUALITY
#: only: a short prompt ("a", "ok", "/plain") is a substring of almost any
#: later assembled prompt, so containment would pair the first answer with a
#: later question and reorder the transcript.
_CONTAINMENT_MIN_CHARS = 40


def _match_prompt(
    groups: list[tuple[dict[str, Any] | None, list[dict[str, Any]]]], start: int, prompt_text: str
) -> int:
    """Index of the first JSONL prompt group at/after ``start`` that is this turn's prompt.

    Two passes, both on dash-folded / whitespace-folded text. First an EXACT
    match of the JSONL row against the human's slice of the replayed prompt
    (:func:`clean_user_text`), which is what the writer sent and is exact for
    plain prompts and nudges. Only when nothing is equal -- a steer appended to
    the prompt, a quick-prompt macro expanded in place -- is containment inside
    the whole replayed prompt tried, and only for rows long enough that a
    substring hit cannot be a coincidence.
    """
    hay = _norm(prompt_text)
    if not hay:
        return -1
    clean_raw = _fold_dashes(clean_user_text(prompt_text)).strip()
    clean = _norm(clean_raw)
    for j in range(start, len(groups)):
        prow = groups[j][0]
        if prow is None:
            continue
        needle_raw = _fold_dashes(_row_text(prow)).strip()
        if not needle_raw:
            continue
        # Equal, or the human's words followed by a LINE BREAK: the writer
        # appends per-turn guidance after the request on its own lines, so a
        # newline boundary is exact where a bare prefix would let "a" claim
        # "a longer question".
        if _norm(needle_raw) == clean or clean_raw.startswith(needle_raw + "\n"):
            return j
        # A persisted quick prompt ("/plain") reached the agent as its expanded
        # macro, which opens with the [QUICK PROMPT <token>] header both derive
        # from; the token itself is never in the wire text.
        qp_header = quick_prompt_header(needle_raw)
        if qp_header and clean_raw.startswith(qp_header):
            return j
    for j in range(start, len(groups)):
        prow = groups[j][0]
        if prow is None:
            continue
        needle = _norm(_row_text(prow))
        if len(needle) >= _CONTAINMENT_MIN_CHARS and needle[:120] in hay:
            return j
    return -1


def _tag(row: dict[str, Any], source: str) -> dict[str, Any]:
    out = dict(row)
    meta = dict(out.get("meta") or {}) if isinstance(out.get("meta"), dict) else {}
    meta["source"] = source
    out["meta"] = meta
    return out


def _overlay_turn(
    turn: dict[str, Any], prompt_row: dict[str, Any] | None, jsonl_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """One turn's final rows: replay rows enriched + sidecar rows anchored.

    ``turn`` is one element of :func:`fold_replay_updates`'s result:
    ``{"prompt_text", "rows", "tool_index"}``.
    """
    # 1. the prompt bubble: JSONL row (clean text, ts, mid, steer/nudge flags) wins
    if prompt_row is not None:
        head = _tag(prompt_row, SOURCE_JSONL)
    else:
        head = {
            "role": "user",
            "content": _redact_text(clean_user_text(turn["prompt_text"])),
            "cls": "msg msg-u",
            "meta": {"source": SOURCE_REPLAY},
        }
    out: list[dict[str, Any]] = [head]

    rows = [dict(r, meta=dict(r["meta"])) for r in turn["rows"]]
    tool_pos = {tcid: i for tcid, i in turn["tool_index"].items()}

    # 2. enrich replayed tool rows from their JSONL twin; collect sidecar rows
    #    with an anchor (the last tool row that preceded them in the JSONL).
    sidecar: list[tuple[str | None, dict[str, Any]]] = []
    last_tool: str | None = None
    matched_assistant: set[int] = set()
    for row in jsonl_rows:
        role = row.get("role")
        meta = _meta_of(row)
        if role == "tool":
            tcid = str(meta.get("tool_call_id") or "")
            idx = tool_pos.get(tcid)
            if idx is not None:
                last_tool = tcid
                rmeta = rows[idx]["meta"]
                # v1/v2 replay: title is the raw tool name, purpose/kind may be
                # thinner than what the live path persisted. Borrow, do not
                # overwrite what the replay did carry.
                jsonl_content = _row_text(row)
                if jsonl_content and "replay_title" not in rmeta:
                    # The JSONL persists two rows per approved tool (🔧 then ✅);
                    # the first carries the humanized title the live UI groups
                    # on, so borrow that one and keep the replay's own for audit.
                    rmeta["replay_title"] = rows[idx]["content"]
                    rows[idx]["content"] = jsonl_content
                for key in ("purpose", "kind"):
                    if not rmeta.get(key) and meta.get(key):
                        rmeta[key] = meta[key]
                if "output" not in rmeta and meta.get("output"):
                    rmeta["output"] = meta["output"]
                if meta.get("done"):
                    rmeta["done"] = True
                if row.get("ts"):
                    rows[idx]["ts"] = row["ts"]
                if meta.get("mid"):
                    rmeta["mid"] = meta["mid"]
                continue
            # A tool row the replay lacks: keep it as sidecar so nothing vanishes.
            sidecar.append((last_tool, _tag(row, SOURCE_JSONL)))
            continue
        if role in ("assistant", "chunk", "streaming", "thinking"):
            text = _norm(_row_text(row))
            if role == "assistant" and text:
                for i, r in enumerate(rows):
                    if i in matched_assistant or r["role"] != "assistant":
                        continue
                    rt = _norm(r["content"])
                    if rt == text or (len(text) > 40 and (text in rt or rt in text)):
                        matched_assistant.add(i)
                        if row.get("ts"):
                            r["ts"] = row["ts"]
                        if meta.get("mid"):
                            r["meta"]["mid"] = meta["mid"]
                        # Kiro Crew-side facts riding an assistant row.
                        for key in ("turn_stats", "file_changes"):
                            if key in meta:
                                r["meta"][key] = meta[key]
                        break
                else:
                    # Notice-shaped assistant rows (compaction banner) and text
                    # the replay does not have: sidecar.
                    sidecar.append((last_tool, _tag(row, SOURCE_JSONL)))
            # chunk/streaming/thinking rows are transient wire rows; the replay
            # carries the finished text.
            continue
        # user (steer), inject, notice, error, permission, system, done, ...
        sidecar.append((last_tool, _tag(row, SOURCE_JSONL)))

    # 3. A REWRITTEN turn: the JSONL has assistant text for this turn and none
    #    of it is what the replay carries (regenerate / edit-and-resend / variant
    #    switch happened after the resume). The replay is then a record of a
    #    response that no longer exists, so the JSONL wins the WHOLE turn --
    #    rendering both would resurrect the old answer beside the new one.
    #    Self-correcting even if the rewrite endpoint's invalidation did not run.
    jsonl_assistant = sum(
        1 for r in jsonl_rows if r.get("role") == "assistant" and _row_text(r).strip()
    )
    replay_assistant = sum(1 for r in rows if r["role"] == "assistant")
    if jsonl_assistant and replay_assistant and not matched_assistant:
        return [head] + [_tag(r, SOURCE_JSONL) for r in jsonl_rows]

    # 4. splice sidecar rows after their anchor tool (or right after the prompt)
    by_anchor: dict[str | None, list[dict[str, Any]]] = {}
    for anchor, row in sidecar:
        by_anchor.setdefault(anchor, []).append(row)
    out.extend(by_anchor.pop(None, []))
    for i, r in enumerate(rows):
        out.append(r)
        if r["role"] == "tool":
            out.extend(by_anchor.pop(r["meta"].get("tool_call_id"), []))
    for leftovers in by_anchor.values():
        out.extend(leftovers)
    return out


def merge_replay_transcript(
    updates: list[dict[str, Any]], jsonl_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build the transcript from replay frames, overlaying JSONL sidecar rows.

    Returns ``(rows, report)``; ``report`` counts rows by source so the caller
    can surface how much of the transcript the replay carried. When the replay
    holds no turn at all the JSONL rows are returned untouched (tagged), so a
    caller never loses history to an empty replay.
    """
    turns = fold_replay_updates(
        updates,
        redact_text=_redact_text,
        redact_field=_redact_tool_field,
        purpose_limit=_MAX_TOOL_PURPOSE,
    )
    if not turns:
        rows = [_tag(r, SOURCE_JSONL) for r in jsonl_rows]
        return rows, {"replay": 0, "jsonl": len(rows), "turns": 0}

    groups = _jsonl_turns(jsonl_rows)
    out: list[dict[str, Any]] = []
    cursor = 0
    # Rows before the first JSONL prompt (welcome notices etc.) lead the transcript.
    if groups and groups[0][0] is None:
        out.extend(_tag(r, SOURCE_JSONL) for r in groups[0][1])
        cursor = 1
    for turn in turns:
        j = _match_prompt(groups, cursor, turn["prompt_text"])
        if j >= 0:
            # JSONL prompts the replay skipped (steer-only turns, dropped
            # prompts) are still shown, in order, as sidecar.
            for k in range(cursor, j):
                prow, rest = groups[k]
                if prow is not None:
                    out.append(_tag(prow, SOURCE_JSONL))
                out.extend(_tag(r, SOURCE_JSONL) for r in rest)
            prow, rest = groups[j]
            cursor = j + 1
        else:
            prow, rest = None, []
        out.extend(_overlay_turn(turn, prow, rest))
    for k in range(cursor, len(groups)):
        prow, rest = groups[k]
        if prow is not None:
            out.append(_tag(prow, SOURCE_JSONL))
        out.extend(_tag(r, SOURCE_JSONL) for r in rest)

    report = {
        "replay": sum(1 for r in out if r.get("meta", {}).get("source") == SOURCE_REPLAY),
        "jsonl": sum(1 for r in out if r.get("meta", {}).get("source") == SOURCE_JSONL),
        "turns": len(turns),
    }
    return out, report


# ── Merge cache + rewrite invalidation ─────────────────────────────────────────
#
# The frames never change for a resumed session, so one merge is reused while
# the JSONL corpus is unchanged -- the detail endpoint is polled on every
# reconnect and turn end, and re-walking a multi-thousand-frame replay each time
# is a real cost. Two things break the reuse and are handled here:
#
# * rows mutate IN PLACE after they are appended (a permission row gets
#   ``resolved``, a tool row gets ``done``/``output``, an assistant row gets
#   ``turn_stats``), so the cache witness folds those fields in per row;
# * the transcript is REWRITTEN (rewind, regenerate, edit-and-resend, variant
#   switch): the replay then describes a conversation that no longer exists, so
#   the rewrite endpoints call :func:`discard_replay_for_slot`, which drops both
#   the cached rows and the provider's frames -- the next render is the JSONL.

#: slot key -> ((frame count, corpus witness), rows, report). Bounded so an
#: operator with many slots does not accumulate one merged transcript per slot.
_MERGE_CACHE: dict[str, tuple[tuple[Any, ...], list[dict[str, Any]], dict[str, int]]] = {}
_MERGE_CACHE_MAX = 64


def corpus_witness(corpus: list[dict[str, Any]]) -> int:
    """A cheap fingerprint of a JSONL corpus INCLUDING the row fields that mutate in place.

    One linear pass with no regex or serialisation -- an order of magnitude
    cheaper than the merge it guards.
    """
    acc = len(corpus)
    for row in corpus:
        meta = _meta_of(row)
        content = row.get("content")
        output = meta.get("output")
        acc = hash(
            (
                acc,
                row.get("role"),
                row.get("ts"),
                len(content) if isinstance(content, str) else None,
                meta.get("mid"),
                meta.get("resolved"),
                meta.get("done"),
                len(output) if isinstance(output, str) else None,
                bool(meta.get("turn_stats")),
                bool(meta.get("file_changes")),
            )
        )
    return acc


def merge_replay_transcript_cached(
    slot_key: str, updates: list[dict[str, Any]], jsonl_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """:func:`merge_replay_transcript`, reusing the last result for an unchanged corpus.

    Returns fresh copies so a caller mutating its response cannot poison the
    cache. Synchronous and CPU-bound like the merge -- call it off-loop.
    """
    key = (len(updates), corpus_witness(jsonl_rows))
    cached = _MERGE_CACHE.get(slot_key)
    if cached is not None and cached[0] == key:
        return list(cached[1]), dict(cached[2])
    merged, report = merge_replay_transcript(updates, jsonl_rows)
    if len(_MERGE_CACHE) >= _MERGE_CACHE_MAX:
        _MERGE_CACHE.pop(next(iter(_MERGE_CACHE)))
    _MERGE_CACHE[slot_key] = (key, merged, report)
    return list(merged), dict(report)


def discard_replay_for_slot(sessions: Any, slot: Any) -> None:
    """Forget the resume replay for ``slot`` because its transcript was rewritten.

    Drops the cached merge AND asks the live provider to drop its frames
    (``LLMProvider.discard_replay``, a no-op on backends without a replay), so the
    next detail fetch renders the JSONL instead of resurrecting the pre-rewrite
    response. ``sessions`` is the session manager (``state.sessions``); resolution
    failures are swallowed because a rewrite must never fail on a cleanup step.
    """
    _MERGE_CACHE.pop(str(getattr(slot, "key", "") or ""), None)
    try:
        provider = sessions.get_provider(effective_session_key(slot))
    except Exception:
        return
    if provider is not None:
        try:
            provider.discard_replay()
        except Exception:
            logger.debug("discard_replay failed", exc_info=True)
