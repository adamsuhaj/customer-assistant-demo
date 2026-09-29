"""Step 12 - retain customer-scoped chat turns for cross-provider handovers.

- Save each completed user/assistant pair in the existing local demo SQLite
  database, so a Streamlit rerun or a provider switch can reload the same chat.
- Require the trusted caller's resolved customer ID to exist in ``customers``;
  never use a provider name as a conversation partition or accept a model's
  claim about which customer owns a turn.
- Store only the displayed messages and a validated ``AssistantResult`` view.
  Raw MCP rows, signed identity contexts, credentials, and arbitrary result
  fields do not belong in this conversational continuity store.
- Return turns in chronological order for the current customer and remove
  whole pairs after 72 hours. Audit events retain their separate lineage and
  retention behavior in ``audit.py``.
"""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from .database import DEFAULT_DB_PATH, connect_readonly


CHAT_RETENTION = timedelta(days=3)
_MESSAGE_KEYS = frozenset({"role", "content", "created_at_utc", "result", "notice"})
_RESULT_KEYS = frozenset({
    "outcome", "skill", "answer_text", "source_ids", "trace_id",
    "provider", "model", "token_usage",
})
_USAGE_KEYS = frozenset({"prompt_tokens", "completion_tokens", "total_tokens"})
_SAFE_PROVIDER = re.compile(r"[a-z][a-z0-9_-]{0,39}\Z")
_SAFE_SOURCE_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}\Z")
_SAFE_TRACE_ID = re.compile(r"[0-9a-f]{32}\Z")
# This is a guard against accidentally persisting common credential forms in
# visible text. Structured inputs are constrained separately by exact keys.
_CREDENTIAL_TEXT = re.compile(
    r"(?:\b(?:sk-(?:ant-)?[A-Za-z0-9_-]{12,}|Bearer\s+\S{12,})\b|"
    r"-----BEGIN\s+(?:[A-Z ]+\s+)?PRIVATE KEY-----|"
    r"\b(?:OPENAI_API_KEY|ANTHROPIC_API_KEY|DEMO_SIGNING_SECRET)\s*[:=]\s*\S+)",
    re.IGNORECASE,
)


def _utc_timestamp(value: datetime | str) -> str:
    """Normalize an aware instant to lexically sortable UTC milliseconds."""

    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("created_at_utc must be a UTC timestamp") from exc
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat(
        timespec="milliseconds"
    ).replace("+00:00", "Z")


def _check_text(value: Any, field: str, *, allow_empty: bool = False) -> str:
    """Keep saved display text textual and reject recognizable credentials."""

    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f"{field} must be non-empty text")
    if _CREDENTIAL_TEXT.search(value):
        raise ValueError(f"{field} contains credential material")
    return value


def _checked_result(value: Any, provider: str, answer: str) -> dict[str, Any]:
    """Copy the public AssistantResult fields, rejecting all unexpected data."""

    if not isinstance(value, Mapping) or set(value) != _RESULT_KEYS:
        raise ValueError("assistant result has unexpected or missing fields")
    outcome = _check_text(value["outcome"], "result.outcome")
    skill = value["skill"]
    if skill is not None:
        skill = _check_text(skill, "result.skill")
    answer_text = _check_text(value["answer_text"], "result.answer_text")
    if answer_text != answer:
        raise ValueError("result answer does not match displayed assistant text")
    sources = value["source_ids"]
    if not isinstance(sources, (list, tuple)) or any(
        not isinstance(source, str) or not _SAFE_SOURCE_ID.fullmatch(source)
        for source in sources
    ):
        raise ValueError("result.source_ids must contain source IDs only")
    trace_id = value["trace_id"]
    if not isinstance(trace_id, str) or not _SAFE_TRACE_ID.fullmatch(trace_id):
        raise ValueError("result.trace_id must be a generated trace ID")
    if value["provider"] != provider:
        raise ValueError("result provider differs from selected provider")
    model = _check_text(value["model"], "result.model")
    usage = value["token_usage"]
    if usage is not None:
        if (
            not isinstance(usage, Mapping)
            or not set(usage).issubset(_USAGE_KEYS)
            or any(type(amount) is not int or amount < 0 for amount in usage.values())
        ):
            raise ValueError("result.token_usage must contain token counts only")
        usage = dict(usage)
    return {
        "outcome": outcome,
        "skill": skill,
        "answer_text": answer_text,
        "source_ids": list(sources),
        "trace_id": trace_id,
        "provider": provider,
        "model": model,
        "token_usage": usage,
    }


def _checked_message(
    message: Mapping[str, Any], role: str, provider: str,
) -> dict[str, Any]:
    """Serialize only fields consumed by the Streamlit chat renderer."""

    if not isinstance(message, Mapping) or not set(message).issubset(_MESSAGE_KEYS):
        raise ValueError("chat message has unexpected fields")
    if message.get("role") != role:
        raise ValueError(f"chat message role must be {role}")
    content = _check_text(message.get("content"), "message.content")
    created = _utc_timestamp(message.get("created_at_utc"))
    copy: dict[str, Any] = {
        "role": role,
        "content": content,
        "created_at_utc": created,
    }
    if role == "user":
        if "result" in message or "notice" in message:
            raise ValueError("user messages cannot contain result or notice data")
    else:
        if "result" in message:
            copy["result"] = _checked_result(message["result"], provider, content)
        if "notice" in message:
            copy["notice"] = _check_text(message["notice"], "message.notice")
    return copy


def _connect_existing(db_path: Path) -> sqlite3.Connection:
    """Open a loaded database for writes without creating a second copy."""

    path = Path(db_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Demo database not found: {path}")
    connection = sqlite3.connect(path.as_uri() + "?mode=rw", uri=True)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _customer_exists(connection: sqlite3.Connection, customer_id: str) -> bool:
    """Check the source-of-truth customer table, not the model's prose."""

    if not isinstance(customer_id, str) or not customer_id:
        return False
    return connection.execute(
        "SELECT 1 FROM customers WHERE customer_id = ?", (customer_id,)
    ).fetchone() is not None


def _ensure_schema(connection: sqlite3.Connection) -> None:
    """Keep conversational continuity separate from source and audit tables."""

    # No foreign key points to ``customers``: the CSV loader drops and
    # replaces that table on refresh, while retained conversation rows should
    # survive for the same customer ID. Every read/write still checks that ID
    # against the current customer table before exposing a row.
    connection.execute(
        "CREATE TABLE IF NOT EXISTS chat_turns ("
        "turn_id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "customer_id TEXT NOT NULL, created_at_utc TEXT NOT NULL, "
        "provider TEXT NOT NULL, user_message_json TEXT NOT NULL, "
        "assistant_message_json TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_chat_turns_customer_time "
        "ON chat_turns (customer_id, created_at_utc, turn_id)"
    )


def save_chat_turn(
    customer_id: str,
    user_message: Mapping[str, Any],
    assistant_message: Mapping[str, Any],
    provider: str,
    *,
    db_path: Path = DEFAULT_DB_PATH,
) -> int:
    """Commit one customer-owned chat pair and return its local turn ID.

    The caller must resolve ``customer_id`` from its trusted demo selector or
    authenticated identity; this storage function checks the ID still exists
    but cannot authenticate a browser session by a bare string alone.
    """

    if not isinstance(provider, str) or not _SAFE_PROVIDER.fullmatch(provider):
        raise ValueError("provider must be a provider identifier")
    user = _checked_message(user_message, "user", provider)
    assistant = _checked_message(assistant_message, "assistant", provider)
    if user["created_at_utc"] != assistant["created_at_utc"]:
        raise ValueError("a chat pair must share one creation timestamp")
    with closing(_connect_existing(db_path)) as connection:
        # Check ownership and insert as one transaction. BEGIN IMMEDIATE also
        # prevents a source reload from changing customers between the check
        # and write on another SQLite connection.
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            if not _customer_exists(connection, customer_id):
                raise ValueError("unknown customer ID")
            _ensure_schema(connection)
            cursor = connection.execute(
                "INSERT INTO chat_turns "
                "(customer_id, created_at_utc, provider, user_message_json, assistant_message_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    customer_id,
                    user["created_at_utc"],
                    provider,
                    json.dumps(user, separators=(",", ":"), ensure_ascii=False),
                    json.dumps(assistant, separators=(",", ":"), ensure_ascii=False),
                ),
            )
        return int(cursor.lastrowid)


def load_chat_history(
    customer_id: str,
    *,
    db_path: Path = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """Return this customer's retained user/assistant messages oldest first."""

    cutoff = _utc_timestamp(datetime.now(timezone.utc) - CHAT_RETENTION)
    with closing(connect_readonly(db_path)) as connection:
        if not _customer_exists(connection, customer_id):
            raise ValueError("unknown customer ID")
        # A first visit has no chat table yet; reads must not mutate SQLite.
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='chat_turns'"
        ).fetchone()
        if exists is None:
            return []
        rows = connection.execute(
            "SELECT created_at_utc, provider, user_message_json, assistant_message_json "
            "FROM chat_turns WHERE customer_id = ? AND created_at_utc >= ? "
            "ORDER BY created_at_utc ASC, turn_id ASC",
            (customer_id, cutoff),
        ).fetchall()
    # Revalidate local rows on read as well as write. A manually edited DB
    # must not turn this history API into a route for raw tool evidence when a
    # future UI builds handoff context from these messages.
    history: list[dict[str, Any]] = []
    for created_at, provider, raw_user, raw_assistant in rows:
        user = _checked_message(json.loads(raw_user), "user", provider)
        assistant = _checked_message(json.loads(raw_assistant), "assistant", provider)
        if user["created_at_utc"] != created_at or assistant["created_at_utc"] != created_at:
            raise ValueError("stored chat timestamps do not match the turn")
        history.extend((user, assistant))
    return history


def clear_chat_history(
    customer_id: str,
    *,
    db_path: Path = DEFAULT_DB_PATH,
) -> int:
    """Delete this customer's saved chat pairs and return the count removed.

    As with save/load, the caller must resolve the customer from its trusted
    session identity. Clear every provider's turns so a reload or handoff
    cannot restore the cleared conversation. Source records and audit events
    remain available under their existing retention rules.
    """

    with closing(_connect_existing(db_path)) as connection:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            if not _customer_exists(connection, customer_id):
                raise ValueError("unknown customer ID")
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='chat_turns'"
            ).fetchone()
            if exists is None:
                return 0
            cursor = connection.execute(
                "DELETE FROM chat_turns WHERE customer_id = ?", (customer_id,)
            )
            return int(cursor.rowcount)


def purge_expired_chat(
    *, db_path: Path = DEFAULT_DB_PATH, now: datetime | None = None,
) -> int:
    """Delete whole chat pairs older than 72 hours; return pairs deleted."""

    current = datetime.now(timezone.utc) if now is None else now
    if not isinstance(current, datetime) or current.utcoffset() is None:
        raise ValueError("now must be a timezone-aware datetime")
    cutoff = _utc_timestamp(current - CHAT_RETENTION)
    with closing(_connect_existing(db_path)) as connection:
        with connection:
            # The app can call purge on every rerun, including first launch.
            # Avoid creating a table until a completed turn is saved.
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='chat_turns'"
            ).fetchone()
            if exists is None:
                return 0
            cursor = connection.execute(
                "DELETE FROM chat_turns WHERE created_at_utc < ?", (cutoff,)
            )
            return int(cursor.rowcount)
