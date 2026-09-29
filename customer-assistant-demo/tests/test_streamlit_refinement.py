"""Step 13 - Verify durable customer chat and three-day audit retention.

- Start the UI against an isolated synthetic database without reading .env or
  contacting a provider, then submit two ordinary chat turns.
- Confirm a fresh browser session reloads both the chat and audit trace list
  from SQLite, with no model call on an ordinary rerun.
- Age one complete turn and its trace past three days; the next UI render
  removes that turn and the entire trace while preserving the newer pair.
- The Audit Log uses the same customer scope and three-day retention for
  ordinary question traces and provider-handoff details.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from customer_assistant import audit, config, database


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class StreamlitRefinementTests(unittest.TestCase):
    def setUp(self) -> None:
        # A disposable DB proves the trace list is persisted by the app rather
        # than accidentally inherited from a previous test or browser session.
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.db_path = Path(temporary.name) / "demo.sqlite3"
        db_patch = patch.object(database, "DEFAULT_DB_PATH", self.db_path)
        db_patch.start()
        self.addCleanup(db_patch.stop)

        # A missing provider key stops planning safely. Mock configuration so
        # this persistence test cannot open the real .env or make an API call.
        dotenv_patch = patch.object(config, "dotenv_values", return_value={})
        dotenv_patch.start()
        self.addCleanup(dotenv_patch.stop)
        environment = {
            name: value for name, value in os.environ.items()
            if name not in {
                "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEMO_OPENAI_MODEL",
                "DEMO_ANTHROPIC_MODEL", "DEMO_MODEL_PROVIDER",
                "DEMO_MODEL_TIMEOUT_SECONDS", "DEMO_SIGNING_SECRET",
            }
        }
        environment["DEMO_SIGNING_SECRET"] = "synthetic-refinement-test-signing-key"
        env_patch = patch.dict(os.environ, environment, clear=True)
        env_patch.start()
        self.addCleanup(env_patch.stop)

    @staticmethod
    def _rendered_text(app: AppTest) -> str:
        # Streamlit may choose markdown or captions for trace summaries. Read
        # rendered text only, not private audit snapshots or session secrets.
        return "\n".join(
            str(part.value)
            for kind in ("title", "header", "markdown", "caption", "text", "info")
            for part in app.get(kind)
        )

    def test_continuous_chat_persists_and_old_traces_expire(self) -> None:
        app = AppTest.from_file(str(APP_PATH), default_timeout=45).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual([part.value for part in app.title], ["Your friendly AI assistant"])
        self.assertEqual([part.value for part in app.get("header")], ["Settings"])
        self.assertEqual([part.label for part in app.expander], ["Audit Log"])
        self.assertNotIn("Evaluation", self._rendered_text(app))

        # Missing-key planning failures still create separate auditable traces
        # and complete chat pairs without contacting a model provider.
        app.chat_input[0].set_value("Where is my order?").run()
        old_trace = app.session_state["chat_history"][-1]["result"]["trace_id"]
        app.chat_input[0].set_value("Can you find my order?").run()
        new_trace = app.session_state["chat_history"][-1]["result"]["trace_id"]
        self.assertNotEqual(old_trace, new_trace)
        self.assertEqual(len(app.session_state["chat_history"]), 4)
        self.assertEqual(len(app.chat_message), 4)
        self.assertIn(old_trace, self._rendered_text(app))
        self.assertIn(new_trace, self._rendered_text(app))

        # A script rerun and a brand-new AppTest both reload the same saved
        # conversation and trace list from the local SQLite database.
        app.run()
        self.assertEqual(len(app.session_state["chat_history"]), 4)
        self.assertEqual(len(app.chat_message), 4)
        fresh_session = AppTest.from_file(str(APP_PATH), default_timeout=45).run()
        self.assertEqual(len(fresh_session.exception), 0)
        self.assertEqual(len(fresh_session.session_state["chat_history"]), 4)
        self.assertEqual(len(fresh_session.chat_message), 4)
        rendered = self._rendered_text(fresh_session)
        self.assertIn(old_trace, rendered)
        self.assertIn(new_trace, rendered)
        self.assertLess(rendered.index(old_trace), rendered.index(new_trace))

        # Set one stored chat pair and every event of its matching trace
        # beyond the 72-hour boundary. Retention deletes whole records rather
        # than leaving half of a question/answer or partial audit lineage.
        expired_at = (
            (datetime.now(timezone.utc) - timedelta(days=4))
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
        # sqlite3's own context manager commits but does not close the file
        # handle, which prevents temporary DB cleanup on Windows.
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute(
                "UPDATE audit_events SET timestamp_utc = ? WHERE trace_id = ?",
                (expired_at, old_trace),
            )
            changed = connection.execute(
                "UPDATE chat_turns SET created_at_utc = ? "
                "WHERE assistant_message_json LIKE ?",
                (expired_at, f'%{old_trace}%'),
            ).rowcount
            self.assertEqual(changed, 1)
            connection.commit()
        after_expiry = AppTest.from_file(str(APP_PATH), default_timeout=45).run()
        self.assertEqual(len(after_expiry.exception), 0)
        self.assertNotIn(old_trace, self._rendered_text(after_expiry))
        self.assertIn(new_trace, self._rendered_text(after_expiry))
        self.assertEqual(len(after_expiry.session_state["chat_history"]), 2)
        self.assertEqual(len(after_expiry.chat_message), 2)
        self.assertEqual(
            after_expiry.session_state["chat_history"][-1]["result"]["trace_id"],
            new_trace,
        )
        self.assertEqual(audit.read_trace(old_trace, db_path=self.db_path), [])
        self.assertTrue(audit.read_trace(new_trace, db_path=self.db_path))
        with closing(sqlite3.connect(self.db_path)) as connection:
            remaining = connection.execute("SELECT COUNT(*) FROM chat_turns").fetchone()[0]
        self.assertEqual(remaining, 1)


if __name__ == "__main__":
    unittest.main()
