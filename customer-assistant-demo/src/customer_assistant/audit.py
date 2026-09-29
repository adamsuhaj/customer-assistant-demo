"""Step 6 - store the decisions and evidence behind each assistant answer.

- ``record_event`` appends routing, MCP tool, denial, and model events to the
  local SQLite ``audit_events`` table. A shared trace ID connects one question's
  user, assistant, source IDs, authorized evidence, provider, and prompt version.
- Model response events can retain accepted or rejected candidate text for
  review. Denials retain no private evidence or answer text, and recursive
  checks reject credential fields or known runtime secret values.
- ``risk_for_action`` derives each event's tier from its tool and stored rows;
  callers cannot supply a lower tier. ``read_trace`` reconstructs raw events
  in insertion order, while ``clear_events`` resets only rehearsal history.
- ``list_recent_traces`` returns metadata-only request summaries and can scope
  them to one signed customer. ``purge_expired_traces`` removes a whole trace
  when its newest event is older than 72 hours; Streamlit calls it on reruns.
- Step 13 handoff events can retain a small, exact copy of the continuity
  packet (summary, verified displayed turns, and source lineage). The payload
  lives in the same trace table and expires with its trace after 72 hours.
- Writes require an existing loaded database, so a wrong path cannot silently
  create a second empty source of truth.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .database import DEFAULT_DB_PATH, connect_readonly
from .risk import assess_composition, risk_for_action


# A signature or credential has no value in a third-party evidence review. A
# nested object can accidentally carry one, so inspect all keys recursively.
_FORBIDDEN_KEY_FRAGMENTS = (
    "signature", "signingsecret", "identitycontext", "apikey",
    "accesstoken", "authorization", "privatekey",
)
TRACE_RETENTION = timedelta(days=3)

# Match the Step 12 handoff packet's field bounds while allowing a union of up
# to six turns' source IDs. Exact keys prevent a caller from adding raw MCP
# rows or unrelated session data to a provider-switch audit event.
_HANDOFF_KEYS = frozenset({
    "handoff_id", "from_provider", "to_provider", "summary_origin",
    "summary", "source_trace_ids", "source_ids", "recent_turns",
})
_HANDOFF_TURN_KEYS = frozenset({
    "question", "answer", "trace_id", "source_ids", "provider",
})
_MAX_HANDOFF_TURNS = 6
_MAX_HANDOFF_JSON_BYTES = 100_000
_CREDENTIAL_HINT = re.compile(
    r"\b(?:OPENAI_API_KEY|ANTHROPIC_API_KEY|DEMO_SIGNING_SECRET)\s*[:=]|"
    r"\bsk-(?:ant-)?[A-Za-z0-9_-]{12,}", re.IGNORECASE,
)


def _retention_cutoff(now: datetime | None) -> str:
    """Return the 72-hour cutoff in the same sortable UTC format as events."""

    current = datetime.now(timezone.utc) if now is None else now
    # A naive datetime has no unambiguous UTC meaning. Reject it instead of
    # silently using the host's local timezone for trace deletion.
    if not isinstance(current, datetime) or current.utcoffset() is None:
        raise ValueError("now must be a timezone-aware datetime")
    return (current.astimezone(timezone.utc) - TRACE_RETENTION).isoformat(
        timespec="milliseconds"
    ).replace("+00:00", "Z")


def _connect_existing(db_path: Path) -> sqlite3.Connection:
    """Open the loaded demo DB for audit writes without creating a new file."""

    path = Path(db_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Demo database not found: {path}")
    # sqlite3.connect(path) would silently create an empty database after a
    # mistaken reset path. URI mode=rw makes that failure explicit.
    connection = sqlite3.connect(path.as_uri() + "?mode=rw", uri=True)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _ensure_schema(connection: sqlite3.Connection) -> None:
    """Add a trace table without changing the imported synthetic source tables."""

    # One question produces several events. Indexing trace_id plus event_id
    # makes a chronological reconstruction cheap without assuming timestamps
    # from separate actions are unique.
    connection.execute(
        "CREATE TABLE IF NOT EXISTS audit_events ("
        "event_id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "timestamp_utc TEXT NOT NULL, trace_id TEXT NOT NULL, "
        "user_id TEXT NOT NULL, agent_id TEXT NOT NULL, "
        "action TEXT NOT NULL, outcome TEXT NOT NULL, "
        "evidence_ids_json TEXT NOT NULL, "
        "authorized_evidence_snapshot_json TEXT NOT NULL, "
        "provider TEXT, model TEXT, response_text TEXT, "
        "prompt_version TEXT NOT NULL, "
        "risk_skill TEXT, risk_tier TEXT, handoff_payload_json TEXT"
        ")"
    )
    # Existing demo databases already have audit_events. Add only the new
    # nullable field so their older traces, source tables, and event IDs stay
    # intact. This runs in record_event's write transaction.
    columns = {row[1] for row in connection.execute("PRAGMA table_info(audit_events)")}
    if "handoff_payload_json" not in columns:
        connection.execute("ALTER TABLE audit_events ADD COLUMN handoff_payload_json TEXT")
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_audit_events_trace "
        "ON audit_events (trace_id, event_id)"
    )


def _reject_signing_material(value: Any) -> None:
    """Search nested evidence and answer text for credential fields or values."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError("Audit evidence keys must be strings")
            # Normalize spelling so ``api_key`` and ``API-Key`` receive the
            # same treatment, including in an unexpected nested object.
            normalized = "".join(char for char in key.casefold() if char.isalnum())
            if any(fragment in normalized for fragment in _FORBIDDEN_KEY_FRAGMENTS):
                raise ValueError(f"Audit evidence cannot contain credential field {key!r}")
            _reject_signing_material(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_signing_material(child)
    elif isinstance(value, str):
        # Exact runtime value checks cover a credential placed under an
        # innocent key such as "note" without revealing it in the error.
        for name in ("DEMO_SIGNING_SECRET", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            secret = os.getenv(name)
            if secret and secret in value:
                raise ValueError("Audit evidence cannot contain credential material")


def _json_snapshot(rows: Sequence[Mapping[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Freeze the authorized tool projection as a validated JSON snapshot."""

    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError("authorized_evidence_snapshot must be a sequence of mappings")
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("Each audit evidence row must be a mapping")
    # The orchestrator should already have projected customer-safe fields.
    # This final write boundary still rejects leaked credentials if a future
    # caller passes a broader source row by mistake.
    _reject_signing_material(rows)
    # A JSON round trip prevents mutation of a caller-owned dictionary from
    # changing the risk input or what the event appears to have seen.
    encoded = json.dumps(rows, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return encoded, json.loads(encoded)


def _handoff_text(value: Any, name: str, maximum: int) -> str:
    """Keep one required text field nonempty and within the packet bound."""

    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"handoff {name} must be a nonempty string of at most {maximum} characters")
    return value


def _handoff_ids(value: Any, name: str, maximum: int) -> list[str]:
    """Normalize short, unique provenance IDs without accepting raw rows."""

    if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= maximum:
        raise ValueError(f"handoff {name} must contain 1 to {maximum} identifiers")
    ids = [_handoff_text(item, name, 128) for item in value]
    if len(set(ids)) != len(ids):
        raise ValueError(f"handoff {name} cannot contain repeated identifiers")
    return ids


def _handoff_json(payload: Mapping[str, Any]) -> str:
    """Validate and freeze exactly the context sent to the receiving model.

    The handoff is already assembled from accepted displayed answers. This
    final persistence boundary still rejects added fields, credentials, and
    oversized text before it becomes part of the local audit history.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("handoff_payload must be a mapping")
    _reject_signing_material(payload)
    if set(payload) != _HANDOFF_KEYS:
        raise ValueError("handoff_payload has missing or unsupported fields")

    handoff_id = _handoff_text(payload["handoff_id"], "handoff_id", 128)
    from_provider = _handoff_text(payload["from_provider"], "from_provider", 40)
    to_provider = _handoff_text(payload["to_provider"], "to_provider", 40)
    if from_provider == to_provider:
        raise ValueError("handoff providers must differ")
    summary_origin = payload["summary_origin"]
    if not isinstance(summary_origin, str) or summary_origin not in {"model", "fallback"}:
        raise ValueError("handoff summary_origin must be model or fallback")
    summary = _handoff_text(payload["summary"], "summary", 1600)
    source_trace_ids = _handoff_ids(payload["source_trace_ids"], "source_trace_ids", 6)
    source_ids = _handoff_ids(payload["source_ids"], "source_ids", 180)

    raw_turns = payload["recent_turns"]
    if not isinstance(raw_turns, (list, tuple)) or not 1 <= len(raw_turns) <= _MAX_HANDOFF_TURNS:
        raise ValueError("handoff recent_turns must contain 1 to 6 exchanges")
    turns: list[dict[str, Any]] = []
    for raw_turn in raw_turns:
        if not isinstance(raw_turn, Mapping) or set(raw_turn) != _HANDOFF_TURN_KEYS:
            raise ValueError("handoff recent_turns contain missing or unsupported fields")
        turns.append({
            "question": _handoff_text(raw_turn["question"], "question", 500),
            "answer": _handoff_text(raw_turn["answer"], "answer", 1200),
            "trace_id": _handoff_text(raw_turn["trace_id"], "trace_id", 128),
            "source_ids": _handoff_ids(raw_turn["source_ids"], "turn source_ids", 30),
            "provider": _handoff_text(raw_turn["provider"], "provider", 40),
        })

    # The top-level IDs are the packet's compact lineage index, not a second
    # source of truth. Reject a mismatch so a reviewer can trace every item
    # back to an actual included exchange.
    if list(dict.fromkeys(turn["trace_id"] for turn in turns)) != source_trace_ids:
        raise ValueError("handoff source_trace_ids must match recent_turns")
    if list(dict.fromkeys(
        source_id for turn in turns for source_id in turn["source_ids"]
    )) != source_ids:
        raise ValueError("handoff source_ids must match recent_turns")

    normalized = {
        "handoff_id": handoff_id, "from_provider": from_provider,
        "to_provider": to_provider, "summary_origin": summary_origin,
        "summary": summary, "source_trace_ids": source_trace_ids,
        "source_ids": source_ids, "recent_turns": turns,
    }
    encoded = json.dumps(normalized, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    # The recursive guard catches named fields and current runtime secrets.
    # Also reject recognizable token syntax when its value is not installed in
    # this process (for example a pasted key in a customer's old question).
    if _CREDENTIAL_HINT.search(encoded):
        raise ValueError("handoff_payload cannot contain credential material")
    if len(encoded.encode("utf-8")) > _MAX_HANDOFF_JSON_BYTES:
        raise ValueError("handoff_payload is too large for an audit event")
    return encoded


def record_event(
    *,
    db_path: Path = DEFAULT_DB_PATH,
    trace_id: str,
    user_id: str,
    agent_id: str,
    action: str,
    outcome: str,
    evidence_ids: Sequence[str] = (),
    authorized_evidence_snapshot: Sequence[Mapping[str, Any]] = (),
    provider: str | None = None,
    model: str | None = None,
    response_text: str | None = None,
    prompt_version: str = "step5-evidence-v1",
    risk_skill: str | None = None,
    risk_skills: Sequence[str] | None = None,
    handoff_payload: Mapping[str, Any] | None = None,
) -> int:
    """Append an action event and return its monotonically increasing ID.

    ``risk_skill`` names the tool actually used, allowing this boundary to
    compute risk from its registered floor and the snapshot. A denial is
    intentionally recorded with no evidence or response text. Only a provider
    handoff action may carry the bounded continuity payload; it shares the
    trace's 72-hour retention rather than creating a separate log.

    For a composite answer, ``risk_skills`` lists every executed tool. Their
    registered floors and the stored rows determine the maximum; numeric
    tiers are never accepted from the model or this caller.
    """

    # Empty identifiers would make it impossible to join an event to a
    # person, assistant, or trace during a later incident review.
    for name, value in (
        ("trace_id", trace_id), ("user_id", user_id), ("agent_id", agent_id),
        ("action", action), ("outcome", outcome),
        ("prompt_version", prompt_version),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty string")
    if risk_skill is not None and (not isinstance(risk_skill, str) or not risk_skill.strip()):
        raise ValueError("risk_skill must be a nonempty string or None")
    if risk_skills is not None:
        if (risk_skill != "composite" or isinstance(risk_skills, (str, bytes))
                or not isinstance(risk_skills, Sequence) or not risk_skills
                or len(risk_skills) > 4
                or any(not isinstance(skill, str) or not skill.strip() for skill in risk_skills)
                or len(set(risk_skills)) != len(risk_skills)):
            raise ValueError("risk_skills requires a bounded composite tool list")
    if response_text is not None and not isinstance(response_text, str):
        raise TypeError("response_text must be a string or None")
    if handoff_payload is not None and not action.startswith("provider_handoff_"):
        raise ValueError("handoff_payload is allowed only on provider handoff events")

    denied = outcome.casefold() in {"deny", "denied"}
    # A denied handoff should preserve only its metadata, just like a denied
    # tool read. In all other cases freeze the exact packet before writing.
    handoff_json = None if denied or handoff_payload is None else _handoff_json(handoff_payload)
    # Keep text only for model-response events, including a rejected candidate
    # that a reviewer may need to inspect. Tool and denial events drop any
    # incidental text; a denied read must never persist another user's row.
    clean_response_text = response_text if action == "model_response" and not denied else None
    if clean_response_text is not None:
        _reject_signing_material(clean_response_text)
    if denied:
        # A denial is the most likely place to leak someone else's row into a
        # trace. Empty both collections before validation or serialization.
        clean_ids: list[str] = []
        snapshot_json, clean_snapshot = "[]", []
    else:
        if isinstance(evidence_ids, (str, bytes)) or not isinstance(evidence_ids, Sequence):
            raise TypeError("evidence_ids must be a sequence of strings")
        clean_ids = list(evidence_ids)
        if any(not isinstance(source_id, str) or not source_id for source_id in clean_ids):
            raise ValueError("evidence_ids must contain nonempty strings")
        snapshot_json, clean_snapshot = _json_snapshot(authorized_evidence_snapshot)

    # The event inherits risk from trusted tool policy and the exact snapshot
    # being stored. A caller-provided numeric tier could understate exposure.
    # A combined answer inherits every executed tool floor and accessed row.
    # These names come from validated application calls, never a model tier.
    risk_tier = (
        str(assess_composition([(skill, clean_snapshot) for skill in risk_skills]).tier)
        if risk_skills is not None else
        str(risk_for_action(risk_skill, clean_snapshot).tier)
        if risk_skill is not None else None
    )
    # Outcome timestamp only; start/end durations are production telemetry work.
    timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )

    connection = _connect_existing(db_path)
    try:
        # Schema setup and insertion share one transaction. Neither a failed
        # insert nor an interrupted first call leaves a half-built audit table.
        with connection:
            _ensure_schema(connection)
            cursor = connection.execute(
                "INSERT INTO audit_events ("
                "timestamp_utc, trace_id, user_id, agent_id, action, outcome, "
                "evidence_ids_json, authorized_evidence_snapshot_json, provider, "
                "model, response_text, prompt_version, risk_skill, risk_tier, "
                "handoff_payload_json"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    timestamp, trace_id, user_id, agent_id, action, outcome,
                    json.dumps(clean_ids, ensure_ascii=False, separators=(",", ":")),
                    snapshot_json, provider, model, clean_response_text,
                    prompt_version, risk_skill,
                    risk_tier, handoff_json,
                ),
            )
            return int(cursor.lastrowid)
    finally:
        connection.close()


def read_trace(
    trace_id: str, *, db_path: Path = DEFAULT_DB_PATH
) -> list[dict[str, Any]]:
    """Return one question's events in write order with decoded evidence."""

    if not isinstance(trace_id, str) or not trace_id.strip():
        raise ValueError("trace_id must be a nonempty string")
    # This is the raw local inspection API: it can return private evidence and
    # rejected model text. A customer UI must first scope trace IDs with
    # list_recent_traces and render a safe subset. Reading must not mutate
    # evidence or create a table merely because somebody opened a trace.
    connection = connect_readonly(db_path)
    connection.row_factory = sqlite3.Row
    try:
        table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            ("audit_events",),
        ).fetchone()
        if table is None:
            return []
        # event_id is the insertion order; wall-clock timestamps can tie when
        # events happen within the same millisecond.
        rows = connection.execute(
            "SELECT * FROM audit_events WHERE trace_id = ? ORDER BY event_id",
            (trace_id,),
        ).fetchall()
        events: list[dict[str, Any]] = []
        for row in rows:
            event = dict(row)
            event["evidence_ids"] = json.loads(event.pop("evidence_ids_json"))
            event["authorized_evidence_snapshot"] = json.loads(
                event.pop("authorized_evidence_snapshot_json")
            )
            # read_trace also supports a pre-Step 13 database that has never
            # been written since the new column was introduced.
            payload_json = event.pop("handoff_payload_json", None)
            event["handoff_payload"] = json.loads(payload_json) if payload_json is not None else None
            events.append(event)
        return events
    finally:
        connection.close()


def list_recent_traces(
    *,
    user_id: str | None = None,
    db_path: Path = DEFAULT_DB_PATH,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """List distinct, unexpired traces, newest first, with safe metadata only.

    ``user_id`` scopes the list to one signed customer. A trace containing
    events for two users is excluded from a scoped list so its other events
    cannot be exposed by opening that trace in the UI.
    """

    if user_id is not None and (not isinstance(user_id, str) or not user_id.strip()):
        raise ValueError("user_id must be a nonempty string or None")
    cutoff = _retention_cutoff(now)
    connection = connect_readonly(db_path)
    connection.row_factory = sqlite3.Row
    try:
        table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            ("audit_events",),
        ).fetchone()
        if table is None:
            return []

        # One question can write tool and model events. Rank within each
        # trace so only its latest event becomes a summary, while the count
        # still includes every event in the retained request. The projection
        # deliberately never reads response text or evidence JSON.
        query = (
            "WITH ranked AS ("
            "SELECT event_id, trace_id, timestamp_utc, user_id, agent_id, "
            "action, outcome, provider, model, "
            "COUNT(*) OVER (PARTITION BY trace_id) AS event_count, "
            "MIN(user_id) OVER (PARTITION BY trace_id) AS first_user, "
            "MAX(user_id) OVER (PARTITION BY trace_id) AS last_user, "
            "ROW_NUMBER() OVER (PARTITION BY trace_id "
            "ORDER BY timestamp_utc DESC, event_id DESC) AS trace_position "
            "FROM audit_events"
            ") SELECT trace_id, timestamp_utc, user_id, agent_id, action, "
            "outcome, provider, model, event_count FROM ranked "
            "WHERE trace_position = 1 AND timestamp_utc >= ?"
        )
        parameters: list[str] = [cutoff]
        if user_id is not None:
            query += " AND first_user = ? AND last_user = ?"
            parameters.extend((user_id, user_id))
        query += " ORDER BY timestamp_utc DESC, event_id DESC"
        return [dict(row) for row in connection.execute(query, parameters)]
    finally:
        connection.close()


def purge_expired_traces(
    *, db_path: Path = DEFAULT_DB_PATH, now: datetime | None = None
) -> int:
    """Delete complete traces inactive for over 72 hours; return rows removed."""

    cutoff = _retention_cutoff(now)
    connection = _connect_existing(db_path)
    try:
        with connection:
            table = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                ("audit_events",),
            ).fetchone()
            if table is None:
                return 0
            # Streamlit calls this on page load/rerun; no background timer is
            # involved. Delete by trace, not by individual event: a question straddling
            # the cutoff must still reconstruct as one complete audit trail.
            # SQLite serializes this selection and deletion in one write
            # transaction, so no caller sees a half-purged request.
            cursor = connection.execute(
                "DELETE FROM audit_events WHERE trace_id IN ("
                "SELECT trace_id FROM audit_events GROUP BY trace_id "
                "HAVING MAX(timestamp_utc) < ?"
                ")",
                (cutoff,),
            )
            return cursor.rowcount
    finally:
        connection.close()


def clear_events(*, db_path: Path = DEFAULT_DB_PATH) -> None:
    """Clear only local audit events so a demo can start with an empty trace."""

    connection = _connect_existing(db_path)
    try:
        with connection:
            table = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                ("audit_events",),
            ).fetchone()
            # Source tables and synthetic customer records remain available;
            # this reset is limited to the review history.
            if table is not None:
                connection.execute("DELETE FROM audit_events")
                connection.execute(
                    "DELETE FROM sqlite_sequence WHERE name = ?", ("audit_events",)
                )
    finally:
        connection.close()
