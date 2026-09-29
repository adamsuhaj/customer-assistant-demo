"""Step 12 - verify durable customer-scoped chat for provider handovers.

- Save OpenAI and Claude turns for one customer and reload them in the same
  order after the original process state has gone away.
- Keep Bob's stored messages separate from Alice's, and reject unknown IDs or
  missing database paths instead of creating a new empty SQLite file.
- Reject unexpected result fields and recognizable credentials before any
  write, so chat continuity does not become an alternate raw-evidence store.
- Remove whole expired pairs at 72 hours while retaining current messages and
  the separate validated source tables.
"""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

from customer_assistant.audit import read_trace, record_event
from customer_assistant.conversation import (
    clear_chat_history, load_chat_history, purge_expired_chat, save_chat_turn,
)
from customer_assistant.database import HEADERS, load_database


ALICE = "CUST-1001"
BOB = "CUST-1002"


def messages(
    question: str, answer: str, provider: str, created: datetime, trace_id: str,
) -> tuple[dict, dict]:
    stamp = created.astimezone(timezone.utc).isoformat(
        timespec="milliseconds"
    ).replace("+00:00", "Z")
    user = {"role": "user", "content": question, "created_at_utc": stamp}
    assistant = {
        "role": "assistant", "content": answer, "created_at_utc": stamp,
        "result": {
            "outcome": "answered", "skill": "list_customer_orders",
            "answer_text": answer, "source_ids": ["DEMO-ORD-1008"],
            "trace_id": trace_id, "provider": provider,
            "model": f"{provider}/test-model", "token_usage": {"total_tokens": 12},
        },
    }
    return user, assistant


class ConversationStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.db_path = Path(temporary.name) / "demo.sqlite3"
        load_database(db_path=self.db_path)
        self.now = datetime.now(timezone.utc).replace(microsecond=0)

    def test_provider_switch_reloads_one_customer_without_mixing_bob(self) -> None:
        self.assertEqual(load_chat_history(ALICE, db_path=self.db_path), [])
        with closing(sqlite3.connect(self.db_path)) as connection:
            # A read-only first visit should leave schema creation to save.
            self.assertIsNone(connection.execute(
                "SELECT name FROM sqlite_master WHERE name = 'chat_turns'"
            ).fetchone())

        first = messages(
            "List my orders", "Your order is processing.", "openai",
            self.now - timedelta(minutes=2), "a" * 32,
        )
        second = messages(
            "And what is its status?", "It is still processing.", "anthropic",
            self.now - timedelta(minutes=1), "b" * 32,
        )
        bob = messages(
            "List Bob's orders", "One order is delivered.", "openai",
            self.now, "c" * 32,
        )
        save_chat_turn(ALICE, *first, "openai", db_path=self.db_path)
        save_chat_turn(ALICE, *second, "anthropic", db_path=self.db_path)
        save_chat_turn(BOB, *bob, "openai", db_path=self.db_path)

        # A fresh function call reads SQLite, with no browser session cache.
        alice = load_chat_history(ALICE, db_path=self.db_path)
        self.assertEqual([entry["content"] for entry in alice], [
            "List my orders", "Your order is processing.",
            "And what is its status?", "It is still processing.",
        ])
        self.assertEqual(
            [entry["result"]["provider"] for entry in alice if "result" in entry],
            ["openai", "anthropic"],
        )
        self.assertEqual(len(load_chat_history(BOB, db_path=self.db_path)), 2)
        self.assertNotIn("Bob", str(alice))

        # Re-importing the six CSVs should preserve the separate chat table.
        load_database(db_path=self.db_path)
        self.assertEqual(load_chat_history(ALICE, db_path=self.db_path), alice)

    def test_unknown_customer_and_missing_database_do_not_create_history(self) -> None:
        user, assistant = messages(
            "Question", "Answer", "openai", self.now, "d" * 32,
        )
        with self.assertRaises(ValueError):
            save_chat_turn("CUST-UNKNOWN", user, assistant, "openai", db_path=self.db_path)
        with self.assertRaises(ValueError):
            load_chat_history("CUST-UNKNOWN", db_path=self.db_path)
        missing = self.db_path.parent / "missing.sqlite3"
        with self.assertRaises(FileNotFoundError):
            save_chat_turn(ALICE, user, assistant, "openai", db_path=missing)
        with self.assertRaises(FileNotFoundError):
            load_chat_history(ALICE, db_path=missing)
        self.assertFalse(missing.exists())

    def test_rejects_raw_rows_and_credentials_before_writing(self) -> None:
        user, assistant = messages(
            "Question", "Answer", "openai", self.now, "e" * 32,
        )
        assistant["result"]["authorized_evidence_snapshot"] = [{"customer_id": ALICE}]
        with self.assertRaises(ValueError):
            save_chat_turn(ALICE, user, assistant, "openai", db_path=self.db_path)
        del assistant["result"]["authorized_evidence_snapshot"]
        user["content"] = "OPENAI_API_KEY=sk-example-secret-value"
        with self.assertRaises(ValueError):
            save_chat_turn(ALICE, user, assistant, "openai", db_path=self.db_path)
        user["content"] = "Question"
        assistant["result"]["provider"] = "anthropic"
        with self.assertRaises(ValueError):
            save_chat_turn(ALICE, user, assistant, "openai", db_path=self.db_path)
        with closing(sqlite3.connect(self.db_path)) as connection:
            self.assertIsNone(connection.execute(
                "SELECT name FROM sqlite_master WHERE name = 'chat_turns'"
            ).fetchone())

    def test_clear_removes_all_customer_providers_and_preserves_other_data(self) -> None:
        for provider, trace_id in (("openai", "2" * 32), ("anthropic", "3" * 32)):
            pair = messages("Alice question", "Alice answer", provider, self.now, trace_id)
            save_chat_turn(ALICE, *pair, provider, db_path=self.db_path)
        bob = messages("Bob question", "Bob answer", "openai", self.now, "4" * 32)
        save_chat_turn(BOB, *bob, "openai", db_path=self.db_path)
        bob_history = load_chat_history(BOB, db_path=self.db_path)
        record_event(
            db_path=self.db_path, trace_id="2" * 32, user_id=ALICE,
            agent_id="customer-assistant", action="model_response", outcome="answered",
            evidence_ids=["DEMO-ORD-1008"], response_text="Alice answer",
        )
        audit_before = read_trace("2" * 32, db_path=self.db_path)
        source_tables = [name.removesuffix(".csv") for name in HEADERS]
        with closing(sqlite3.connect(self.db_path)) as connection:
            sources_before = {
                table: connection.execute(f'SELECT * FROM "{table}"').fetchall()
                for table in source_tables
            }

        self.assertEqual(clear_chat_history(ALICE, db_path=self.db_path), 2)
        self.assertEqual(load_chat_history(ALICE, db_path=self.db_path), [])
        self.assertEqual(clear_chat_history(ALICE, db_path=self.db_path), 0)
        self.assertEqual(load_chat_history(BOB, db_path=self.db_path), bob_history)
        self.assertEqual(read_trace("2" * 32, db_path=self.db_path), audit_before)
        with closing(sqlite3.connect(self.db_path)) as connection:
            sources_after = {
                table: connection.execute(f'SELECT * FROM "{table}"').fetchall()
                for table in source_tables
            }
        self.assertEqual(sources_after, sources_before)

    def test_clear_validates_customer_and_does_not_create_missing_storage(self) -> None:
        self.assertEqual(clear_chat_history(ALICE, db_path=self.db_path), 0)
        with closing(sqlite3.connect(self.db_path)) as connection:
            self.assertIsNone(connection.execute(
                "SELECT name FROM sqlite_master WHERE name = 'chat_turns'"
            ).fetchone())
        pair = messages("Question", "Answer", "openai", self.now, "5" * 32)
        save_chat_turn(ALICE, *pair, "openai", db_path=self.db_path)
        history_before = load_chat_history(ALICE, db_path=self.db_path)
        for customer_id in ("CUST-UNKNOWN", "", None, "CUST-1001' OR 1=1 --"):
            with self.subTest(customer_id=customer_id):
                with self.assertRaises(ValueError):
                    clear_chat_history(customer_id, db_path=self.db_path)
        self.assertEqual(load_chat_history(ALICE, db_path=self.db_path), history_before)
        missing = self.db_path.parent / "missing.sqlite3"
        with self.assertRaises(FileNotFoundError):
            clear_chat_history(ALICE, db_path=missing)
        self.assertFalse(missing.exists())

    def test_purge_removes_expired_pairs_and_keeps_recent_sources(self) -> None:
        old = messages(
            "Old question", "Old answer", "openai",
            self.now - timedelta(days=4), "f" * 32,
        )
        recent = messages(
            "Recent question", "Recent answer", "anthropic",
            self.now - timedelta(days=2), "1" * 32,
        )
        save_chat_turn(ALICE, *old, "openai", db_path=self.db_path)
        save_chat_turn(ALICE, *recent, "anthropic", db_path=self.db_path)
        self.assertEqual(purge_expired_chat(db_path=self.db_path, now=self.now), 1)
        self.assertEqual(purge_expired_chat(db_path=self.db_path, now=self.now), 0)
        self.assertEqual(
            [entry["content"] for entry in load_chat_history(ALICE, db_path=self.db_path)],
            ["Recent question", "Recent answer"],
        )
        with closing(sqlite3.connect(self.db_path)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0], 11)
        with self.assertRaises(ValueError):
            purge_expired_chat(db_path=self.db_path, now=self.now.replace(tzinfo=None))


if __name__ == "__main__":
    unittest.main()
