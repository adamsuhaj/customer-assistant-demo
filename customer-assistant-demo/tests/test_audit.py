"""Step 6 - Prove the SQLite audit explains what the assistant did.

- Exercise record_event() and read_trace() with linked tool and model events.
- Require enough user, agent, source, model, and snapshot data to reconstruct an allowed answer.
- Prove a denial writes no private evidence even if a caller supplies some by mistake.
- Prove risk is computed from trusted tool/data tiers and credentials are rejected.
- Reset traces in a disposable database so repeated rehearsals start cleanly.
- Step 13 verifies that only a bounded provider handoff can carry the exact
  continuity memo, including migration of old audit tables and 72-hour expiry.
"""

import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from customer_assistant.audit import (
    clear_events, purge_expired_traces, read_trace, record_event,
)
from customer_assistant.database import DEFAULT_SOURCE_DIR, load_database


TEST_SECRET = "test-only-audit-signing-secret"
TEST_OPENAI_KEY = "test-only-openai-api-key"
TEST_ANTHROPIC_KEY = "test-only-anthropic-api-key"


class AuditTests(unittest.TestCase):
    def setUp(self) -> None:
        # A disposable copy tests writes without changing the shipped demo DB
        # or the six source CSVs used during the interview walkthrough.
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.db_path = Path(temporary.name) / "demo.sqlite3"
        load_database(DEFAULT_SOURCE_DIR, self.db_path)

    def _event(self, **overrides):
        # Start each scenario with one realistic authorized order read; each
        # test changes only the field whose audit behavior it needs to prove.
        fields = {
            "db_path": self.db_path,
            "trace_id": "trace-alice-1",
            "user_id": "CUST-1001",
            "agent_id": "customer-assistant-demo",
            "action": "tool:get_order_status",
            "outcome": "allow",
            "evidence_ids": ["DEMO-ORD-1007"],
            "authorized_evidence_snapshot": [{
                "order_id": "DEMO-ORD-1007", "order_status": "in_transit",
            }],
            "provider": "openai",
            "model": "openai/gpt-6-sol",
            "risk_skill": "get_order_status",
        }
        fields.update(overrides)
        return record_event(**fields)

    def _handoff_payload(self):
        # This mirrors only the fields given to a newly selected provider.
        # The actual order row and signed identity are deliberately absent.
        return {
            "handoff_id": "handoff-1",
            "from_provider": "openai",
            "to_provider": "anthropic",
            "summary_origin": "model",
            "summary": "The customer asked about order DEMO-ORD-1007.",
            "source_trace_ids": ["trace-alice-1"],
            "source_ids": ["DEMO-ORD-1007"],
            "recent_turns": [{
                "question": "Where is my order DEMO-ORD-1007?",
                "answer": "It is in transit. [DEMO-ORD-1007]",
                "trace_id": "trace-alice-1",
                "source_ids": ["DEMO-ORD-1007"],
                "provider": "openai",
            }],
        }

    def test_composite_inherits_tool_floors_and_cannot_dilute_row_tier(self) -> None:
        self._event(
            action="model_response", risk_skill="composite",
            risk_skills=("get_order_status", "get_service_history", "search_troubleshooting"),
        )
        self.assertEqual(read_trace("trace-alice-1", db_path=self.db_path)[-1]["risk_tier"], "2")
        self._event(
            action="model_response", risk_skill="composite",
            risk_skills=("get_order_status", "search_troubleshooting"),
            authorized_evidence_snapshot=[{"order_id": "DEMO-ORD-1007", "access_tier": "3"}],
        )
        self.assertEqual(read_trace("trace-alice-1", db_path=self.db_path)[-1]["risk_tier"], "3")
        self._event(action="model_response", risk_skill="composite", risk_skills=("unknown_tool",))
        self.assertEqual(read_trace("trace-alice-1", db_path=self.db_path)[-1]["risk_tier"], "3")

    def test_composite_rejects_empty_numeric_or_mislabelled_tool_inputs(self) -> None:
        for label, names in (("composite", ()), ("composite", (0,)),
                             ("get_order_status", ("search_troubleshooting",)),
                             ("composite", "get_order_status")):
            with self.subTest(label=label, names=names), self.assertRaises(ValueError):
                self._event(risk_skill=label, risk_skills=names)

    def test_allowed_tool_and_model_events_share_trace_and_reconstruct_answer(self) -> None:
        # The tool snapshot says what evidence the answer was allowed to use;
        # the later model event must preserve that lineage under one trace.
        tool_id = self._event()
        model_id = self._event(
            action="model_response", outcome="allow",
            response_text="Order DEMO-ORD-1007 is in transit. [DEMO-ORD-1007]",
        )
        events = read_trace("trace-alice-1", db_path=self.db_path)

        self.assertEqual([event["event_id"] for event in events], [tool_id, model_id])
        self.assertEqual(events[0]["timestamp_utc"][-1], "Z")
        self.assertEqual(events[0]["user_id"], "CUST-1001")
        self.assertEqual(events[0]["agent_id"], "customer-assistant-demo")
        self.assertEqual(events[0]["evidence_ids"], ["DEMO-ORD-1007"])
        self.assertEqual(events[0]["authorized_evidence_snapshot"], [{
            "order_id": "DEMO-ORD-1007", "order_status": "in_transit",
        }])
        self.assertEqual(events[0]["risk_tier"], "1")
        self.assertEqual(events[0]["prompt_version"], "step5-evidence-v1")
        self.assertIsNone(events[0]["response_text"])
        self.assertEqual(events[1]["response_text"],
                         "Order DEMO-ORD-1007 is in transit. [DEMO-ORD-1007]")
        self.assertEqual(events[1]["provider"], "openai")
        self.assertEqual(events[1]["model"], "openai/gpt-6-sol")
        self.assertIsNone(events[0]["handoff_payload"])
        self.assertIsNone(events[1]["handoff_payload"])

    def test_handoff_payload_is_reconstructable_only_on_handoff_events(self) -> None:
        payload = self._handoff_payload()
        event_id = self._event(
            trace_id="handoff-1", action="provider_handoff_to_anthropic",
            outcome="model", risk_skill="provider_handoff",
            handoff_payload=payload,
            authorized_evidence_snapshot=[],
        )
        # Mutating the caller's dictionary cannot alter the stored handoff.
        payload["summary"] = "a later change"
        event = read_trace("handoff-1", db_path=self.db_path)[0]
        self.assertEqual(event["event_id"], event_id)
        self.assertEqual(event["handoff_payload"]["summary"],
                         "The customer asked about order DEMO-ORD-1007.")
        self.assertEqual(event["handoff_payload"]["recent_turns"][0]["answer"],
                         "It is in transit. [DEMO-ORD-1007]")
        self.assertEqual(event["risk_tier"], "3")
        self.assertEqual(event["authorized_evidence_snapshot"], [])
        with closing(sqlite3.connect(self.db_path)) as connection:
            stored = connection.execute(
                "SELECT handoff_payload_json FROM audit_events WHERE event_id = ?",
                (event_id,),
            ).fetchone()[0]
        self.assertNotIn("order_status", stored)
        self.assertNotIn("a later change", stored)

    def test_handoff_payload_rejects_extras_credentials_and_oversized_fields(self) -> None:
        valid = self._handoff_payload()
        with self.assertRaises(ValueError):
            self._event(handoff_payload=valid)
        bad_cases = []
        unknown = self._handoff_payload()
        unknown["raw_mcp_rows"] = [{"order_status": "in_transit"}]
        bad_cases.append(unknown)
        nested_secret = self._handoff_payload()
        nested_secret["recent_turns"][0]["signature"] = "secret"
        bad_cases.append(nested_secret)
        large_question = self._handoff_payload()
        large_question["recent_turns"][0]["question"] = "Q" * 501
        bad_cases.append(large_question)
        large_answer = self._handoff_payload()
        large_answer["recent_turns"][0]["answer"] = "A" * 1201
        bad_cases.append(large_answer)
        wrong_lineage = self._handoff_payload()
        wrong_lineage["source_ids"] = ["DEMO-ORD-9999"]
        bad_cases.append(wrong_lineage)
        pasted_key = self._handoff_payload()
        pasted_key["summary"] = "OPENAI_API_KEY=example"
        bad_cases.append(pasted_key)
        with patch.dict(os.environ, {"OPENAI_API_KEY": TEST_OPENAI_KEY}):
            runtime_secret = self._handoff_payload()
            runtime_secret["summary"] = f"value={TEST_OPENAI_KEY}"
            bad_cases.append(runtime_secret)
            for payload in bad_cases:
                with self.subTest(payload_keys=tuple(payload)), self.assertRaises(ValueError):
                    self._event(
                        trace_id="handoff-1", action="provider_handoff_to_anthropic",
                        outcome="model", risk_skill="provider_handoff",
                        handoff_payload=payload,
                    )
        self.assertEqual(read_trace("handoff-1", db_path=self.db_path), [])

    def test_old_audit_table_migrates_without_losing_existing_trace(self) -> None:
        # The shipped local SQLite file may predate the handoff column. Read
        # it without a write, then migrate only when the next event is added.
        with closing(sqlite3.connect(self.db_path)) as connection:
            with connection:
                connection.execute(
                    "CREATE TABLE audit_events ("
                    "event_id INTEGER PRIMARY KEY AUTOINCREMENT, "
                    "timestamp_utc TEXT NOT NULL, trace_id TEXT NOT NULL, "
                    "user_id TEXT NOT NULL, agent_id TEXT NOT NULL, "
                    "action TEXT NOT NULL, outcome TEXT NOT NULL, "
                    "evidence_ids_json TEXT NOT NULL, "
                    "authorized_evidence_snapshot_json TEXT NOT NULL, "
                    "provider TEXT, model TEXT, response_text TEXT, "
                    "prompt_version TEXT NOT NULL, risk_skill TEXT, risk_tier TEXT)"
                )
                connection.execute(
                    "INSERT INTO audit_events (timestamp_utc, trace_id, user_id, "
                    "agent_id, action, outcome, evidence_ids_json, "
                    "authorized_evidence_snapshot_json, prompt_version) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    ("2026-09-27T12:00:00.000Z", "legacy", "CUST-1001",
                     "customer-assistant-demo", "route_question", "allow",
                     "[]", "[]", "step5-evidence-v1"),
                )
        self.assertIsNone(read_trace("legacy", db_path=self.db_path)[0]["handoff_payload"])
        self._event(
            trace_id="handoff-1", action="provider_handoff_to_anthropic",
            outcome="model", risk_skill="provider_handoff",
            handoff_payload=self._handoff_payload(),
        )
        with closing(sqlite3.connect(self.db_path)) as connection:
            columns = {row[1] for row in connection.execute("PRAGMA table_info(audit_events)")}
        self.assertIn("handoff_payload_json", columns)
        self.assertIsNone(read_trace("legacy", db_path=self.db_path)[0]["handoff_payload"])
        self.assertEqual(read_trace("handoff-1", db_path=self.db_path)[0]["handoff_payload"]
                         ["handoff_id"], "handoff-1")

    def test_handoff_payload_expires_with_its_trace_after_72_hours(self) -> None:
        event_id = self._event(
            trace_id="handoff-1", action="provider_handoff_to_anthropic",
            outcome="model", risk_skill="provider_handoff",
            handoff_payload=self._handoff_payload(),
        )
        now = datetime(2026, 9, 27, 12, 0, tzinfo=timezone.utc)
        old_time = (now - timedelta(days=3, milliseconds=1)).isoformat(
            timespec="milliseconds"
        ).replace("+00:00", "Z")
        with closing(sqlite3.connect(self.db_path)) as connection:
            with connection:
                connection.execute(
                    "UPDATE audit_events SET timestamp_utc = ? WHERE event_id = ?",
                    (old_time, event_id),
                )
        self.assertEqual(purge_expired_traces(db_path=self.db_path, now=now), 1)
        self.assertEqual(read_trace("handoff-1", db_path=self.db_path), [])

    def test_denial_discards_supplied_private_evidence_and_response(self) -> None:
        self._event(
            trace_id="trace-bob-1", user_id="CUST-1002", outcome="deny",
            evidence_ids=["DEMO-ORD-1007"],
            authorized_evidence_snapshot=[{
                "order_id": "DEMO-ORD-1007", "signature": "must-never-persist",
            }],
            response_text="must-never-persist",
        )
        event = read_trace("trace-bob-1", db_path=self.db_path)[0]
        self.assertEqual(event["evidence_ids"], [])
        self.assertEqual(event["authorized_evidence_snapshot"], [])
        self.assertIsNone(event["response_text"])
        self.assertEqual(event["risk_tier"], "1")
        # Read the physical DB too: an API projection alone would not prove
        # the denied order/signature was absent from stored audit material.
        with closing(sqlite3.connect(self.db_path)) as connection:
            stored = connection.execute(
                "SELECT evidence_ids_json, authorized_evidence_snapshot_json, "
                "response_text FROM audit_events WHERE trace_id = ?",
                ("trace-bob-1",),
            ).fetchone()
        self.assertEqual(stored, ("[]", "[]", None))

    def test_nested_credentials_and_runtime_secret_are_rejected(self) -> None:
        with patch.dict(os.environ, {
            "DEMO_SIGNING_SECRET": TEST_SECRET,
            "OPENAI_API_KEY": TEST_OPENAI_KEY,
            "ANTHROPIC_API_KEY": TEST_ANTHROPIC_KEY,
        }):
            bad_rows = (
                [{"order_id": "DEMO-ORD-1007", "meta": {"signature": "abc"}}],
                [{"order_id": "DEMO-ORD-1007", "note": f"value={TEST_SECRET}"}],
                [{"order_id": "DEMO-ORD-1007", "note": TEST_OPENAI_KEY}],
                [{"order_id": "DEMO-ORD-1007", "note": TEST_ANTHROPIC_KEY}],
            )
            for rows in bad_rows:
                with self.subTest(rows=rows), self.assertRaises(ValueError):
                    self._event(authorized_evidence_snapshot=rows)
            with self.assertRaises(ValueError):
                self._event(action="model_response", response_text=TEST_OPENAI_KEY)
        self.assertEqual(read_trace("trace-alice-1", db_path=self.db_path), [])

    def test_tier_uses_accessed_row_and_trace_reset_restarts_ids(self) -> None:
        self._event(authorized_evidence_snapshot=[{
            "order_id": "DEMO-ORD-1007", "access_tier": "2",
        }])
        self._event(trace_id="other-trace", risk_skill="unknown_tool")
        self.assertEqual(read_trace("trace-alice-1", db_path=self.db_path)[0]["risk_tier"], "2")
        self.assertEqual(read_trace("other-trace", db_path=self.db_path)[0]["risk_tier"], "3")
        clear_events(db_path=self.db_path)
        self.assertEqual(read_trace("trace-alice-1", db_path=self.db_path), [])
        self.assertEqual(read_trace("other-trace", db_path=self.db_path), [])
        self.assertEqual(self._event(), 1)

    def test_missing_database_is_not_created(self) -> None:
        absent = self.db_path.parent / "absent.sqlite3"
        with self.assertRaises(FileNotFoundError):
            self._event(db_path=absent)
        with self.assertRaises(FileNotFoundError):
            read_trace("trace-alice-1", db_path=absent)
        with self.assertRaises(FileNotFoundError):
            clear_events(db_path=absent)
        self.assertFalse(absent.exists())


if __name__ == "__main__":
    unittest.main()
