"""Step 13 - verify durable chat and inspectable provider handoffs in Streamlit.

- Supply synthetic .env settings through a mock, so tests never open the real file.
- Submit Alice's order question, switch from OpenAI to Claude, and confirm the
  outgoing model sends one bounded recap before Claude receives the audited
  history and handoff packet on its next answer.
- Confirm ordinary reruns do not duplicate the handoff, and a fresh browser
  session reloads the same customer's transcript from SQLite.
- Inspect the Audit Log handoff entry for the exact bounded memo and verified
  question/answer lineage sent between model routes, with three-day retention.
- Check Bob's authorized answer and private-record denial without carrying
  any of Alice's conversation context into his model request or trace list.
- Check that a missing selected key stops a model call while an ID clarification
  still comes from the existing orchestrator router.
- Keep Step 11 MCP authorization and routing in the orchestrator; these UI
  tests exercise the same entry point rather than mocking away that boundary.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing, contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, patch

# Streamlit's AppTest scans installed components on its own thread. Preload
# this optional FastMCP dependency before that scan, avoiding a transient
# circular import when several AppTest cases run in the same Python process.
try:
    import beartype.claw._clawstate  # noqa: F401
except ImportError:
    pass

from streamlit.testing.v1 import AppTest

from customer_assistant import audit, config, database, mcp_client


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
TEST_SIGNING_SECRET = "test-only-streamlit-signing-secret-for-synthetic-data"
TEST_OPENAI_KEY = "synthetic-openai-key-never-send"
TEST_ANTHROPIC_KEY = "synthetic-anthropic-key-never-send"
TEST_OPENAI_MODEL = "openai/env-test-model"
TEST_ANTHROPIC_MODEL = "anthropic/env-test-model"


class StreamlitAppTests(unittest.TestCase):
    def setUp(self) -> None:
        # Give each UI test its own generated database. Its MCP subprocess
        # still uses the real loader and authorization path.
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.db_path = Path(temporary.name) / "demo.sqlite3"
        # Leave the generated DB absent so the first app render also proves
        # a fresh project copy can initialize it through the Step 1 loader.
        for target, name, value in (
            (database, "DEFAULT_DB_PATH", self.db_path),
        ):
            active = patch.object(target, name, value)
            active.start()
            self.addCleanup(active.stop)
        # Remove inherited provider settings only for this test process. The
        # synthetic dotenv mock below must be the source of the selected key.
        removed = {
            "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEMO_OPENAI_MODEL",
            "DEMO_ANTHROPIC_MODEL", "DEMO_MODEL_PROVIDER",
            "DEMO_MODEL_TIMEOUT_SECONDS", "DEMO_SIGNING_SECRET",
        }
        test_environment = {
            name: value for name, value in os.environ.items() if name not in removed
        }
        test_environment["DEMO_SIGNING_SECRET"] = TEST_SIGNING_SECRET
        environment = patch.dict(os.environ, test_environment, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def _event_count(self) -> int:
        with closing(sqlite3.connect(self.db_path)) as connection:
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='audit_events'"
            ).fetchone()
            if exists is None:
                return 0
            return connection.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]

    def _synthetic_dotenv(self, *, with_anthropic_key: bool = True):
        settings = {
            "OPENAI_API_KEY": TEST_OPENAI_KEY,
            "DEMO_OPENAI_MODEL": TEST_OPENAI_MODEL,
            "DEMO_ANTHROPIC_MODEL": TEST_ANTHROPIC_MODEL,
            "DEMO_MODEL_TIMEOUT_SECONDS": "17",
        }
        if with_anthropic_key:
            settings["ANTHROPIC_API_KEY"] = TEST_ANTHROPIC_KEY

        def read_synthetic_settings(path: Path) -> dict[str, str]:
            # Confirm the UI asks the existing config helper for the project
            # .env path. This mock returns fixed data without opening it.
            self.assertEqual(Path(path), config.PROJECT_ROOT / ".env")
            return settings

        return read_synthetic_settings

    @contextmanager
    def _mock_litellm(self, completion: AsyncMock):
        # Importing the real LiteLLM package can fetch its optional model
        # price map. Give the gateway's lazy import only an async test double.
        fake_module = ModuleType("litellm")
        fake_module.acompletion = completion
        # Restore only this mock: restoring all of sys.modules unloads NumPy
        # first imported by st.image, whose native extension cannot reload.
        original = sys.modules.get("litellm")
        sys.modules["litellm"] = fake_module
        try:
            yield
        finally:
            if original is None:
                sys.modules.pop("litellm", None)
            else:
                sys.modules["litellm"] = original

    def _selected_audit_events(self, app: AppTest) -> list[dict]:
        # Inspect the same JSON projection that a presenter sees in the
        # expanded Audit Log, rather than reading a private raw audit snapshot.
        self.assertEqual(len(app.get("json")), 1)
        return json.loads(app.get("json")[0].value)

    @staticmethod
    def _answer_calls(completion: AsyncMock):
        """Answer and recap calls stay distinct from native tool proposals."""

        return [call for call in completion.await_args_list if "tools" not in call.kwargs]

    @staticmethod
    def _completion_with_planner(answer_responses):
        responses = iter(answer_responses)

        async def complete(**kwargs):
            if "tools" not in kwargs:
                return next(responses)
            payload = json.loads(kwargs["messages"][-1]["content"])
            if payload.get("current_tool_results"):
                return {"choices": [{"message": {"content": "Evidence is ready."}}]}
            question = payload["question"]
            order_id = next((
                order for order in ("DEMO-ORD-1007", "DEMO-ORD-2001")
                if order in question
            ), None)
            if order_id is None:
                return {"choices": [{"message": {
                    "content": "Please provide the demo order ID.",
                }}]}
            return {"choices": [{"message": {
                "content": None,
                "tool_calls": [{
                    "id": "test_order_status", "type": "function",
                    "function": {
                        "name": "get_order_status",
                        "arguments": json.dumps({"order_id": order_id}),
                    },
                }],
            }}]}

        return AsyncMock(side_effect=complete)

    def test_provider_switch_reruns_and_customer_separated_trace_history(self) -> None:
        # Each queued result is tied to one answer or recap call. Tool planning
        # uses separate native responses. The outgoing OpenAI recap must run
        # on provider selection,
        # before Claude handles another customer question.
        def model_response(text: str) -> dict:
            return {
                "choices": [{"message": {"content": text}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 9},
            }

        completion = self._completion_with_planner([
            model_response("Order DEMO-ORD-1007 is in transit. [DEMO-ORD-1007]"),
            model_response("The customer is tracking DEMO-ORD-1007."),
            model_response("Order DEMO-ORD-1007 is in transit. [DEMO-ORD-1007]"),
            model_response("Order DEMO-ORD-2001 is processing. [DEMO-ORD-2001]"),
            model_response("The customer is still tracking DEMO-ORD-1007."),
        ])
        mcp_errors: list[str] = []
        original_order_status = mcp_client.order_status

        async def order_status_with_diagnostics(*args, **kwargs):
            # The orchestrator intentionally hides transport exceptions from
            # customers. Capture them only in this synthetic test so a failed
            # MCP subprocess can be distinguished from a model failure.
            try:
                return await original_order_status(*args, **kwargs)
            except Exception as exc:
                mcp_errors.append(f"{type(exc).__name__}: {exc}")
                raise

        with (
            patch.object(
                config, "dotenv_values", side_effect=self._synthetic_dotenv()
            ),
            self._mock_litellm(completion),
            patch.object(mcp_client, "order_status", side_effect=order_status_with_diagnostics),
        ):
            app = AppTest.from_file(str(APP_PATH), default_timeout=45).run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(self.db_path.is_file())
            self.assertEqual(self._event_count(), 0)
            self.assertEqual(len(app.chat_input), 1)
            self.assertEqual(len(app.checkbox), 0)
            self.assertEqual([part.label for part in app.expander], ["Audit Log"])
            self.assertEqual([part.value for part in app.title], ["Your friendly AI assistant"])
            self.assertEqual([part.value for part in app.get("header")], ["Settings"])

            # An explicit order ID reaches the existing MCP order tool, then
            # the selected live adapter. Assert the .env key stays out of UI.
            app.chat_input[0].set_value("Where is order DEMO-ORD-1007?").run()
            self.assertEqual(len(app.exception), 0)
            alice_openai = app.session_state["chat_history"][-1]["result"]
            self.assertEqual(alice_openai["outcome"], "allow", (alice_openai, mcp_errors))
            self.assertEqual(alice_openai["source_ids"], ["DEMO-ORD-1007"])
            self.assertEqual(
                (alice_openai["provider"], alice_openai["model"]),
                ("openai", TEST_OPENAI_MODEL),
            )
            self.assertEqual(self._event_count(), 4)
            self.assertEqual(len(self._answer_calls(completion)), 1)
            planning_calls = [
                call for call in completion.await_args_list if "tools" in call.kwargs
            ]
            self.assertEqual(len(planning_calls), 2)
            self.assertEqual({
                tool["function"]["name"]
                for tool in planning_calls[0].kwargs["tools"]
            }, {"get_order_status", "list_customer_orders", "get_service_history", "search_troubleshooting"})
            self.assertNotIn("identity_context", json.dumps(planning_calls[0].kwargs["tools"]))
            refreshed = json.loads(planning_calls[1].kwargs["messages"][-1]["content"])
            self.assertEqual(
                refreshed["current_tool_results"][0]["source_ids"], ["DEMO-ORD-1007"],
            )
            self.assertNotIn("signature", json.dumps(refreshed))
            self.assertEqual(self._answer_calls(completion)[0].kwargs["model"], TEST_OPENAI_MODEL)
            self.assertEqual(self._answer_calls(completion)[0].kwargs["api_key"], TEST_OPENAI_KEY)
            self.assertEqual(self._answer_calls(completion)[0].kwargs["timeout"], 17.0)
            self.assertNotIn(
                TEST_OPENAI_KEY,
                json.dumps(self._answer_calls(completion)[0].kwargs["messages"]),
            )
            # The refined chat uses one compact metadata line per answer.
            captions = "\n".join(part.value for part in app.get("caption"))
            self.assertIn(f"Trace ID: {alice_openai['trace_id']}", captions)
            self.assertIn("Sources: DEMO-ORD-1007", captions)

            # A plain rerun does not call a model. A provider switch starts
            # exactly one recap with the outgoing OpenAI route, but leaves
            # the visible conversation intact and adds no customer question.
            app.run()
            self.assertEqual(len(self._answer_calls(completion)), 1)
            app.selectbox[1].set_value("anthropic").run()
            self.assertEqual(len(self._answer_calls(completion)), 2)
            self.assertEqual(self._event_count(), 5)
            self.assertEqual(len(app.session_state["chat_history"]), 2)
            handoff_kwargs = self._answer_calls(completion)[1].kwargs
            self.assertEqual(handoff_kwargs["model"], TEST_OPENAI_MODEL)
            self.assertEqual(handoff_kwargs["api_key"], TEST_OPENAI_KEY)
            handoff_prompt = json.loads(handoff_kwargs["messages"][1]["content"])
            self.assertEqual(len(handoff_prompt), 1)
            self.assertEqual(handoff_prompt[0]["trace_id"], alice_openai["trace_id"])
            self.assertEqual(handoff_prompt[0]["source_ids"], ["DEMO-ORD-1007"])
            self.assertNotIn(TEST_OPENAI_KEY, json.dumps(handoff_kwargs["messages"]))
            packet = next(iter(app.session_state["provider_handoffs"].values()))
            handoff_events = audit.read_trace(packet.handoff_id, db_path=self.db_path)
            self.assertEqual(len(handoff_events), 1)
            self.assertEqual(handoff_events[0]["action"], "provider_handoff_to_anthropic")
            self.assertEqual(handoff_events[0]["risk_tier"], "3")
            self.assertEqual(handoff_events[0]["evidence_ids"], ["DEMO-ORD-1007"])
            self.assertEqual(handoff_events[0]["authorized_evidence_snapshot"], [])
            # Opening the handoff's own Audit Log entry must show precisely
            # what crossed the provider boundary. It is persisted lineage,
            # not just the transient Streamlit handoff packet.
            self.assertEqual(app.selectbox[2].label, "Inspect saved trace")
            app.selectbox[2].set_value(packet.handoff_id).run()
            self.assertEqual(len(self._answer_calls(completion)), 2)
            visible_handoff = next(
                event["handoff"] for event in self._selected_audit_events(app)
                if event["action"] == "provider_handoff_to_anthropic"
            )
            self.assertEqual(visible_handoff["summary"], packet.summary)
            self.assertEqual(visible_handoff["summary_origin"], "model")
            self.assertEqual(visible_handoff["from_provider"], "openai")
            self.assertEqual(visible_handoff["to_provider"], "anthropic")
            self.assertEqual(visible_handoff["source_trace_ids"], [alice_openai["trace_id"]])
            self.assertEqual(visible_handoff["source_ids"], ["DEMO-ORD-1007"])
            self.assertEqual(len(visible_handoff["recent_turns"]), 1)
            prior_turn = visible_handoff["recent_turns"][0]
            self.assertEqual(prior_turn["question"], "Where is order DEMO-ORD-1007?")
            self.assertEqual(
                prior_turn["answer"],
                "Order DEMO-ORD-1007 is in transit. [DEMO-ORD-1007]",
            )
            self.assertEqual(prior_turn["trace_id"], alice_openai["trace_id"])
            self.assertEqual(prior_turn["source_ids"], ["DEMO-ORD-1007"])
            self.assertEqual(prior_turn["provider"], "openai")
            app.run()
            self.assertEqual(len(self._answer_calls(completion)), 2)
            self.assertEqual(self._event_count(), 5)
            self.assertEqual(len(app.session_state["chat_history"]), 2)

            app.chat_input[0].set_value("Where is order DEMO-ORD-1007?").run()
            alice_anthropic = app.session_state["chat_history"][-1]["result"]
            self.assertEqual(alice_anthropic["outcome"], "allow")
            self.assertEqual(
                (alice_anthropic["provider"], alice_anthropic["model"]),
                ("anthropic", TEST_ANTHROPIC_MODEL),
            )
            self.assertEqual(self._event_count(), 10)
            self.assertEqual(len(self._answer_calls(completion)), 3)
            self.assertEqual(
                self._answer_calls(completion)[2].kwargs["model"], TEST_ANTHROPIC_MODEL
            )
            self.assertEqual(
                self._answer_calls(completion)[2].kwargs["api_key"], TEST_ANTHROPIC_KEY
            )
            self.assertNotIn(
                TEST_ANTHROPIC_KEY,
                json.dumps(self._answer_calls(completion)[2].kwargs["messages"]),
            )
            claude_payload = json.loads(
                self._answer_calls(completion)[2].kwargs["messages"][1]["content"]
            )
            context = claude_payload["conversation_context"]
            self.assertEqual(context["recent_turns"][0]["trace_id"], alice_openai["trace_id"])
            self.assertEqual(context["provider_handoff"]["handoff_id"], packet.handoff_id)
            self.assertEqual(context["provider_handoff"]["from_provider"], "openai")
            self.assertEqual(context["provider_handoff"]["to_provider"], "anthropic")
            receiving_events = audit.read_trace(alice_anthropic["trace_id"], db_path=self.db_path)
            self.assertEqual(
                [event["action"] for event in receiving_events].count(
                    f"provider_handoff_received:{packet.handoff_id}"
                ), 1,
            )
            received = next(
                event for event in receiving_events
                if event["action"] == f"provider_handoff_received:{packet.handoff_id}"
            )
            self.assertEqual(received["risk_tier"], "3")
            self.assertEqual(received["handoff_payload"], visible_handoff)

            # Bob's own order uses his saved/default OpenAI route with no Alice transcript or
            # handoff. A model-proposed read of Alice's order is then denied
            # before any private evidence reaches the answer model.
            app.selectbox[0].set_value("bob").run()
            self.assertEqual(app.session_state["chat_history"], [])
            self.assertEqual(len(self._answer_calls(completion)), 3)
            app.chat_input[0].set_value("Where is order DEMO-ORD-2001?").run()
            bob_own = app.session_state["chat_history"][-1]["result"]
            self.assertEqual(bob_own["outcome"], "allow")
            self.assertEqual(bob_own["source_ids"], ["DEMO-ORD-2001"])
            self.assertEqual(bob_own["provider"], "openai")
            self.assertEqual(len(self._answer_calls(completion)), 4)
            bob_payload = json.loads(
                self._answer_calls(completion)[3].kwargs["messages"][1]["content"]
            )
            self.assertNotIn("conversation_context", bob_payload)
            self.assertNotIn(alice_openai["trace_id"], json.dumps(bob_payload))
            self.assertNotIn(packet.handoff_id, json.dumps(bob_payload))
            app.chat_input[0].set_value("Where is order DEMO-ORD-1007?").run()
            bob = app.session_state["chat_history"][-1]["result"]
            self.assertEqual(bob["outcome"], "deny")
            self.assertEqual(bob["source_ids"], [])
            self.assertEqual(bob["provider"], "openai")
            self.assertEqual(len(self._answer_calls(completion)), 4)
            self.assertEqual(self._event_count(), 16)
            denial_events = audit.read_trace(bob["trace_id"], db_path=self.db_path)
            self.assertEqual([event["action"] for event in denial_events], [
                "tool_selection", "get_order_status",
            ])
            self.assertEqual(denial_events[-1]["outcome"], "deny")
            self.assertTrue(all(
                event["authorized_evidence_snapshot"] == [] for event in denial_events
            ))
            self.assertEqual(len(app.session_state["chat_history"]), 4)
            self.assertEqual(
                len(app.session_state["chat_history_by_customer"]["alice"]), 4
            )
            rendered = "\n".join(
                str(part.value)
                for part in [*app.get("markdown"), *app.get("caption")]
            )
            self.assertIn(bob["trace_id"], rendered)
            self.assertNotIn(alice_openai["trace_id"], rendered)
            self.assertNotIn(alice_anthropic["trace_id"], rendered)
            # Bob can inspect his own audit rows, but cannot see Alice's
            # handoff memo or its verified prior exchange there.
            bob_audit = json.dumps(self._selected_audit_events(app))
            self.assertNotIn(packet.summary, bob_audit)
            self.assertNotIn(packet.handoff_id, bob_audit)
            self.assertNotIn(alice_openai["trace_id"], bob_audit)

            # Switching back restores Alice's four persisted messages and her
            # handoff lineage without exposing Bob's messages or traces.
            app.selectbox[0].set_value("alice").run()
            self.assertEqual(len(app.session_state["chat_history"]), 4)
            alice_rendered = "\n".join(
                str(part.value)
                for part in [*app.get("markdown"), *app.get("caption")]
            )
            self.assertIn(alice_openai["trace_id"], alice_rendered)
            self.assertIn(alice_anthropic["trace_id"], alice_rendered)
            self.assertIn(packet.handoff_id, alice_rendered)
            self.assertNotIn(bob["trace_id"], alice_rendered)
            app.selectbox[0].set_value("bob").run()
            self.assertEqual(len(app.session_state["chat_history"]), 4)

            # Ordinary reruns leave the stored conversation and events alone.
            app.run()
            self.assertEqual(self._event_count(), 16)
            self.assertEqual(len(self._answer_calls(completion)), 4)
            self.assertEqual(len(app.session_state["chat_history"]), 4)
            self.assertEqual([part.label for part in app.expander], ["Audit Log"])
            self.assertEqual([button.label for button in app.button], ["New chat"])
            # A fresh Streamlit session still sees Alice's saved conversation.
            # It restores Alice's last provider, so reconnecting alone cannot
            # trigger a paid reverse handoff.
            reconnect = AppTest.from_file(str(APP_PATH), default_timeout=45).run()
            self.assertEqual(len(reconnect.exception), 0)
            self.assertEqual(len(reconnect.session_state["chat_history"]), 4)
            self.assertEqual(reconnect.selectbox[1].value, "anthropic")
            self.assertEqual(len(self._answer_calls(completion)), 4)
            self.assertEqual(self._event_count(), 16)
            reconnect.selectbox[2].set_value(packet.handoff_id).run()
            recovered_handoff = next(
                event["handoff"] for event in self._selected_audit_events(reconnect)
                if event["action"] == "provider_handoff_to_anthropic"
            )
            self.assertEqual(recovered_handoff, visible_handoff)
            reconnect.run()
            self.assertEqual(len(self._answer_calls(completion)), 4)
            self.assertEqual(self._event_count(), 16)
            # Neither provider's key appears in the displayed answer or
            # trace metadata, even though the live gateway received it.
            customer_chats = json.dumps(app.session_state["chat_history_by_customer"])
            self.assertNotIn(TEST_OPENAI_KEY, customer_chats)
            self.assertNotIn(TEST_ANTHROPIC_KEY, customer_chats)
            self.assertEqual(mcp_errors, [])

            # Age both durable copies of this handoff: its dedicated send
            # trace and the receiving question trace. The next load applies
            # the same 72-hour whole-trace retention to memo details as to
            # ordinary MCP audit events and the matching chat turn.
            expired_at = (
                (datetime.now(timezone.utc) - timedelta(days=4))
                .isoformat(timespec="milliseconds")
                .replace("+00:00", "Z")
            )
            with closing(sqlite3.connect(self.db_path)) as connection:
                connection.execute(
                    "UPDATE audit_events SET timestamp_utc = ? WHERE trace_id IN (?, ?)",
                    (expired_at, packet.handoff_id, alice_anthropic["trace_id"]),
                )
                changed = connection.execute(
                    "UPDATE chat_turns SET created_at_utc = ? "
                    "WHERE assistant_message_json LIKE ?",
                    (expired_at, f'%{alice_anthropic["trace_id"]}%'),
                ).rowcount
                self.assertEqual(changed, 1)
                connection.commit()
            after_expiry = AppTest.from_file(str(APP_PATH), default_timeout=45).run()
            self.assertEqual(len(after_expiry.exception), 0)
            self.assertEqual(audit.read_trace(packet.handoff_id, db_path=self.db_path), [])
            self.assertEqual(
                audit.read_trace(alice_anthropic["trace_id"], db_path=self.db_path),
                [],
            )
            retained = audit.list_recent_traces(
                user_id=packet.customer_id, db_path=self.db_path,
            )
            self.assertNotIn(packet.handoff_id, [row["trace_id"] for row in retained])
            self.assertNotIn(packet.summary, json.dumps(self._selected_audit_events(after_expiry)))
            self.assertEqual(len(self._answer_calls(completion)), 4)

    def test_new_chat_clears_saved_turns_and_handoff_context(self) -> None:
        answer = "Order DEMO-ORD-1007 is in transit. [DEMO-ORD-1007]"
        completion = self._completion_with_planner([
            {"choices": [{"message": {"content": answer}}]},
            {"choices": [{"message": {"content": "The customer is tracking DEMO-ORD-1007."}}]},
            {"choices": [{"message": {"content": answer}}]},
        ])
        with (
            patch.object(config, "dotenv_values", side_effect=self._synthetic_dotenv()),
            self._mock_litellm(completion),
        ):
            app = AppTest.from_file(str(APP_PATH), default_timeout=45).run()
            self.assertFalse(app.button[0].disabled)
            app.chat_input[0].set_value("Where is order DEMO-ORD-1007?").run()
            self.assertEqual(len(app.exception), 0)
            self.assertFalse(app.button[0].disabled)
            alice_trace = app.session_state["chat_history"][-1]["result"]["trace_id"]
            app.selectbox[1].set_value("anthropic").run()
            packet = next(iter(app.session_state["provider_handoffs"].values()))
            self.assertEqual(len(self._answer_calls(completion)), 2)

            # Give Bob a distinct saved conversation, then return to Alice.
            app.selectbox[0].set_value("bob").run()
            app.chat_input[0].set_value("Where is my order?").run()
            app.run()  # Compare Bob's normalized persisted timestamps on both reads.
            bob_history = list(app.session_state["chat_history"])
            app.selectbox[0].set_value("alice").run()
            app.selectbox[1].set_value("anthropic").run()
            self.assertEqual(app.selectbox[1].value, "anthropic")
            bob_cache_key = ("bob", "openai", "anthropic", "b" * 32)
            app.session_state["provider_handoffs"][bob_cache_key] = None
            before_clear_events = self._event_count()
            previous_input_key = app.chat_input[0].key

            app.button[0].click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.chat_message), 0)
            self.assertEqual(app.session_state["chat_history"], [])
            self.assertEqual(app.session_state["chat_history_by_customer"]["alice"], [])
            self.assertEqual(app.session_state["chat_history_by_customer"]["bob"], bob_history)
            self.assertEqual(app.session_state["provider_handoffs"], {bob_cache_key: None})
            self.assertEqual(app.selectbox[1].value, "anthropic")
            self.assertNotEqual(app.chat_input[0].key, previous_input_key)
            self.assertIsNone(app.chat_input[0].value)
            self.assertFalse(app.button[0].disabled)
            self.assertEqual(self._event_count(), before_clear_events)
            self.assertTrue(audit.read_trace(alice_trace, db_path=self.db_path))
            self.assertTrue(audit.read_trace(packet.handoff_id, db_path=self.db_path))
            self.assertEqual(len(self._answer_calls(completion)), 2)

            # A browser reconnect cannot restore the deleted chat or recap it.
            reconnect = AppTest.from_file(str(APP_PATH), default_timeout=45).run()
            self.assertEqual(len(reconnect.exception), 0)
            self.assertEqual(len(reconnect.chat_message), 0)
            self.assertEqual(len(self._answer_calls(completion)), 2)

            # The next answer gets fresh evidence with no previous context.
            app.chat_input[0].set_value("Where is order DEMO-ORD-1007?").run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.chat_message), 2)
            self.assertEqual(len(self._answer_calls(completion)), 3)
            payload = json.loads(self._answer_calls(completion)[2].kwargs["messages"][1]["content"])
            self.assertNotIn("conversation_context", payload)
            app.selectbox[0].set_value("bob").run()
            self.assertEqual(app.session_state["chat_history"], bob_history)

    def test_missing_selected_key_stops_tool_planning_before_sdk_call(self) -> None:
        completion = AsyncMock()
        with (
            patch.object(
                config, "dotenv_values",
                side_effect=self._synthetic_dotenv(with_anthropic_key=False),
            ),
            self._mock_litellm(completion),
        ):
            app = AppTest.from_file(str(APP_PATH), default_timeout=45).run()
            app.selectbox[1].set_value("anthropic").run()

            # Native tool selection needs the selected provider key even for
            # an ambiguous question. The app returns a safe failure before
            # SDK traffic rather than using an implicit keyword fallback.
            app.chat_input[0].set_value("where is my order?").run()
            self.assertEqual(len(app.exception), 0)
            clarification = app.session_state["chat_history"][-1]["result"]
            self.assertEqual(clarification["outcome"], "no_match")
            self.assertIn("unavailable", clarification["answer_text"])
            self.assertEqual(clarification["source_ids"], [])
            self.assertIn(
                "ANTHROPIC_API_KEY",
                app.session_state["chat_history"][-1]["notice"],
            )
            self.assertEqual(self._event_count(), 1)
            completion.assert_not_awaited()

            # An explicit record ID also cannot bypass the key requirement.
            app.chat_input[0].set_value("Where is order DEMO-ORD-1007?").run()
            self.assertEqual(len(app.exception), 0)
            missing_key = app.session_state["chat_history"][-1]["result"]
            self.assertEqual(missing_key["outcome"], "no_match")
            self.assertEqual(missing_key["provider"], "anthropic")
            self.assertEqual(missing_key["model"], TEST_ANTHROPIC_MODEL)
            self.assertIn("unavailable", missing_key["answer_text"])
            self.assertIn(
                "ANTHROPIC_API_KEY",
                app.session_state["chat_history"][-1]["notice"],
            )
            self.assertEqual(self._event_count(), 2)
            completion.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
