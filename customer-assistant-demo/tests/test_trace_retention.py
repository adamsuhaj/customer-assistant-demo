"""Step 10 - Verify local audit history shown in the Streamlit Audit Log.

- Check that history lists each request once, newest first, without exposing
  stored answer text or authorized evidence snapshots.
- Scope visible traces to one customer and hide any malformed mixed-user
  trace rather than risk opening another customer's audit events.
- Expire whole traces after 72 hours of inactivity, preserving a multi-event
  request that straddles the cutoff and retaining a trace on the boundary.
- Leave source tables and an audit-free database untouched by history reads.
- The rows are local assistant/MCP events; a later A2A adapter would need to
  create its own message/task rather than forward this audit format directly.
"""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

from customer_assistant.audit import (
    list_recent_traces, purge_expired_traces, read_trace, record_event,
)
from customer_assistant.database import DEFAULT_SOURCE_DIR, load_database


NOW = datetime(2026, 9, 27, 12, 0, tzinfo=timezone.utc)
ALICE = "CUST-1001"
BOB = "CUST-1002"


class TraceRetentionTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.db_path = Path(temporary.name) / "demo.sqlite3"
        load_database(DEFAULT_SOURCE_DIR, self.db_path)

    def _event(
        self, trace_id: str, at: datetime, *, user_id: str = ALICE,
        action: str = "route", outcome: str = "allow",
    ) -> int:
        # The public writer stamps real time. Move only this disposable row
        # to a fixed timestamp so the retention boundary is deterministic.
        event_id = record_event(
            db_path=self.db_path,
            trace_id=trace_id,
            user_id=user_id,
            agent_id="customer_assistant_v1",
            action=action,
            outcome=outcome,
            provider="openai",
            model="openai/test-model",
            response_text="A reviewed answer" if action == "model_response" else None,
        )
        timestamp = at.astimezone(timezone.utc).isoformat(
            timespec="milliseconds"
        ).replace("+00:00", "Z")
        with closing(sqlite3.connect(self.db_path)) as connection:
            with connection:
                connection.execute(
                    "UPDATE audit_events SET timestamp_utc = ? WHERE event_id = ?",
                    (timestamp, event_id),
                )
        return event_id

    def test_history_is_scoped_safe_and_newest_first(self) -> None:
        self._event("alice-earlier", NOW - timedelta(hours=2))
        self._event("alice-latest", NOW - timedelta(hours=1), action="get_order_status")
        self._event(
            "alice-latest", NOW - timedelta(minutes=59), action="model_response",
        )
        self._event("bob", NOW - timedelta(minutes=30), user_id=BOB)
        # A malformed trace with two user IDs must not appear in either
        # customer's scoped list, since opening it would show both users.
        self._event("mixed", NOW - timedelta(minutes=20), user_id=ALICE)
        self._event("mixed", NOW - timedelta(minutes=19), user_id=BOB)
        self._event("expired", NOW - timedelta(days=4))

        alice = list_recent_traces(user_id=ALICE, db_path=self.db_path, now=NOW)
        self.assertEqual(
            [item["trace_id"] for item in alice], ["alice-latest", "alice-earlier"]
        )
        self.assertEqual(alice[0]["event_count"], 2)
        self.assertEqual(alice[0]["action"], "model_response")
        self.assertEqual(alice[0]["provider"], "openai")
        self.assertEqual(
            set(alice[0]), {
                "trace_id", "timestamp_utc", "user_id", "agent_id", "action",
                "outcome", "provider", "model", "event_count",
            },
        )
        self.assertEqual(
            [item["trace_id"] for item in list_recent_traces(
                user_id=BOB, db_path=self.db_path, now=NOW
            )], ["bob"],
        )
        self.assertNotIn(
            "expired", [item["trace_id"] for item in list_recent_traces(
                db_path=self.db_path, now=NOW
            )],
        )

    def test_purge_removes_old_requests_but_preserves_complete_recent_trace(self) -> None:
        self._event("old", NOW - timedelta(days=4), action="route")
        self._event("old", NOW - timedelta(days=4) + timedelta(seconds=1), action="model_response")
        self._event("boundary", NOW - timedelta(days=3))
        self._event("straddling", NOW - timedelta(days=4), action="route")
        self._event("straddling", NOW - timedelta(days=2), action="model_response")

        self.assertEqual(purge_expired_traces(db_path=self.db_path, now=NOW), 2)
        self.assertEqual(read_trace("old", db_path=self.db_path), [])
        self.assertEqual(len(read_trace("straddling", db_path=self.db_path)), 2)
        self.assertEqual(len(read_trace("boundary", db_path=self.db_path)), 1)
        self.assertEqual(purge_expired_traces(db_path=self.db_path, now=NOW), 0)
        with closing(sqlite3.connect(self.db_path)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0] > 0, True)

    def test_empty_audit_history_and_naive_clock(self) -> None:
        self.assertEqual(list_recent_traces(db_path=self.db_path, now=NOW), [])
        self.assertEqual(purge_expired_traces(db_path=self.db_path, now=NOW), 0)
        with closing(sqlite3.connect(self.db_path)) as connection:
            self.assertIsNone(connection.execute(
                "SELECT name FROM sqlite_master WHERE name = 'audit_events'"
            ).fetchone())
        with self.assertRaises(ValueError):
            list_recent_traces(db_path=self.db_path, now=NOW.replace(tzinfo=None))
        with self.assertRaises(ValueError):
            purge_expired_traces(db_path=self.db_path, now=NOW.replace(tzinfo=None))


if __name__ == "__main__":
    unittest.main()
