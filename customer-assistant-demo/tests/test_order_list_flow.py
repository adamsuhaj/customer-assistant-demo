"""Step 9 - Verify customer order lists across MCP, orchestration, and UI.

- Call the real ``list_customer_orders`` MCP tool with signed Alice and Bob
  identities to prove each receives only their own synthetic orders.
- Check that an ID-free list question reaches the tool and returns all six
  Alice orders and five Bob orders with status citations and an audit trace.
- Confirm the allowed rows carry only the signed customer's joined profile,
  so a later self-account question has evidence without exposing Bob to Alice.
- Reject an incomplete model list so a single citation cannot masquerade as
  an answer to a request for all orders.
- Submit the same question through Streamlit with a mocked live completion;
  provider credentials and ``.env`` contents are never opened by this test.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, patch

from streamlit.testing.v1 import AppTest

from customer_assistant import config, database
from customer_assistant.audit import read_trace
from customer_assistant.config import GatewayConfig
from customer_assistant.database import DEFAULT_SOURCE_DIR, load_database
from customer_assistant.gateway import FakeModelGateway
from customer_assistant.identity import mint_demo_identity
from customer_assistant.mcp_client import create_client, customer_orders
from customer_assistant.orchestrator import answer_question, make_offline_demo_gateway


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
TEST_SIGNING_SECRET = "test-only-list-orders-signing-secret-for-synthetic-data"
TEST_OPENAI_KEY = "synthetic-order-list-key-never-send"
TEST_MODEL = "openai/synthetic-order-list-model"
ALICE_ORDERS = (
    ("DEMO-ORD-1008", "processing"),
    ("DEMO-ORD-1007", "in_transit"),
    ("DEMO-ORD-1009", "in_transit"),
    ("DEMO-ORD-1006", "in_transit"),
    ("DEMO-ORD-1004", "delivered"),
    ("DEMO-ORD-1002", "delivered"),
)
BOB_ORDERS = (
    ("DEMO-ORD-2005", "processing"),
    ("DEMO-ORD-2001", "processing"),
    ("DEMO-ORD-2004", "in_transit"),
    ("DEMO-ORD-2003", "delivered"),
    ("DEMO-ORD-2002", "delivered"),
)
ALICE_IDS = tuple(order_id for order_id, _ in ALICE_ORDERS)
BOB_IDS = tuple(order_id for order_id, _ in BOB_ORDERS)
ALICE_ACTIVE_IDS = tuple(
    order_id for order_id, status in ALICE_ORDERS if status != "delivered"
)
BOB_ACTIVE_IDS = tuple(
    order_id for order_id, status in BOB_ORDERS if status != "delivered"
)
BOB_ID = "DEMO-ORD-2001"


class RecordingGateway:
    """Use the real offline answer builder and retain only its model prompts."""

    def __init__(self) -> None:
        self.config = GatewayConfig("openai", TEST_MODEL, None, 30.0)
        self._offline = make_offline_demo_gateway(self.config)
        self.messages: list[list[dict[str, str]]] = []

    async def select_tools(self, messages, tools):
        return await self._offline.select_tools(messages, tools)

    async def complete(self, messages):
        self.messages.append([dict(message) for message in messages])
        return await self._offline.complete(messages)


class OrderListFlowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        # A disposable validated database lets the test cross the real MCP
        # subprocess boundary without touching the presenter's local state.
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.db_path = Path(temporary.name) / "demo.sqlite3"
        load_database(DEFAULT_SOURCE_DIR, self.db_path)
        signing = patch.dict(os.environ, {"DEMO_SIGNING_SECRET": TEST_SIGNING_SECRET})
        signing.start()
        self.addCleanup(signing.stop)

    async def test_mcp_order_list_is_signed_and_customer_scoped(self) -> None:
        alice = mint_demo_identity("alice", db_path=self.db_path)
        bob = mint_demo_identity("bob", db_path=self.db_path)
        async with create_client(self.db_path) as client:
            names = {tool.name for tool in await client.list_tools()}
            self.assertIn("list_customer_orders", names)
            alice_result = (await client.call_tool("list_customer_orders", {
                "identity_context": alice.as_dict(),
            })).data
            bob_result = (await client.call_tool("list_customer_orders", {
                "identity_context": bob.as_dict(),
            })).data

        self.assertEqual(alice_result["outcome"], "allow")
        self.assertEqual(alice_result["trace_id"], alice.trace_id)
        self.assertEqual(alice_result["source_ids"], list(ALICE_IDS))
        self.assertEqual(
            [(row["order_id"], row["order_status"]) for row in alice_result["rows"]],
            list(ALICE_ORDERS),
        )
        for order_id in BOB_IDS:
            self.assertNotIn(order_id, json.dumps(alice_result))
        self.assertEqual(bob_result["outcome"], "allow")
        self.assertEqual(bob_result["trace_id"], bob.trace_id)
        self.assertEqual(bob_result["source_ids"], list(BOB_IDS))
        self.assertEqual(
            [(row["order_id"], row["order_status"]) for row in bob_result["rows"]],
            list(BOB_ORDERS),
        )
        for alice_id in ALICE_IDS:
            self.assertNotIn(alice_id, json.dumps(bob_result))

        # The convenience wrapper must use the same registered MCP call.
        wrapped = await customer_orders(alice, db_path=self.db_path)
        self.assertEqual(wrapped["source_ids"], list(ALICE_IDS))
        active = await customer_orders(alice, active_only=True, db_path=self.db_path)
        self.assertEqual(active["source_ids"], list(ALICE_ACTIVE_IDS))
        self.assertEqual(
            [row["order_status"] for row in active["rows"]],
            [status for _, status in ALICE_ORDERS if status != "delivered"],
        )

        # A changed identity signature must not reveal any rows or IDs.
        forged = alice.as_dict()
        forged["user_id"] = bob.user_id
        async with create_client(self.db_path) as client:
            denied = (await client.call_tool("list_customer_orders", {
                "identity_context": forged,
            })).data
        self.assertEqual(denied["outcome"], "deny")
        self.assertEqual(denied["rows"], [])
        self.assertEqual(denied["source_ids"], [])

    async def test_id_free_list_has_all_statuses_citations_and_trace(self) -> None:
        alice_gateway = RecordingGateway()
        alice = await answer_question(
            "alice", "List my orders, with all statuses.", alice_gateway,
            db_path=self.db_path,
        )
        self.assertEqual(alice.outcome, "allow")
        self.assertEqual(alice.skill, "list_customer_orders")
        self.assertEqual(alice.source_ids, ALICE_IDS)
        self.assertEqual(len(alice_gateway.messages), 1)
        for order_id in ALICE_IDS:
            self.assertIn(order_id, alice.answer_text)
            self.assertIn(f"[{order_id}]", alice.answer_text)
        self.assertIn("in transit", alice.answer_text)
        self.assertIn("delivered", alice.answer_text)
        self.assertIn("processing", alice.answer_text)
        prompt = json.dumps(alice_gateway.messages)
        # The joined self profile is now approved evidence for questions
        # about the owner of these orders; Bob's profile stays excluded.
        self.assertIn("CUST-1001", prompt)
        self.assertNotIn("CUST-1002", prompt)
        self.assertNotIn("identity_context", prompt)
        self.assertNotIn("signature", prompt)
        self.assertNotIn(TEST_SIGNING_SECRET, prompt)
        for order_id in BOB_IDS:
            self.assertNotIn(order_id, prompt)

        trace = read_trace(alice.trace_id, db_path=self.db_path)
        self.assertEqual([event["action"] for event in trace], [
            "tool_selection", "list_customer_orders", "tool_selection", "model_response",
        ])
        self.assertTrue(all(event["outcome"] == "allow" for event in trace))
        tool_event = next(event for event in trace if event["action"] == "list_customer_orders")
        answer_event = next(event for event in trace if event["action"] == "model_response")
        self.assertEqual(tool_event["evidence_ids"], list(ALICE_IDS))
        self.assertEqual(answer_event["evidence_ids"], list(ALICE_IDS))
        self.assertEqual(
            [row["order_status"] for row in tool_event["authorized_evidence_snapshot"]],
            [status for _, status in ALICE_ORDERS],
        )
        self.assertTrue(all(
            row["customer_id"] == "CUST-1001"
            for row in tool_event["authorized_evidence_snapshot"]
        ))

        active_gateway = RecordingGateway()
        active_alice = await answer_question(
            "alice", "What are my active orders?", active_gateway,
            db_path=self.db_path,
        )
        self.assertEqual(active_alice.outcome, "allow")
        self.assertEqual(active_alice.skill, "list_customer_orders")
        self.assertEqual(active_alice.source_ids, ALICE_ACTIVE_IDS)
        self.assertIn("in transit", active_alice.answer_text)
        for order_id, status in ALICE_ORDERS:
            self.assertEqual(order_id in active_alice.answer_text, status != "delivered")

        bob_gateway = RecordingGateway()
        bob = await answer_question(
            "bob", "What are my active orders?", bob_gateway,
            db_path=self.db_path,
        )
        self.assertEqual(bob.outcome, "allow")
        self.assertEqual(bob.skill, "list_customer_orders")
        self.assertEqual(bob.source_ids, BOB_ACTIVE_IDS)
        self.assertIn("processing", bob.answer_text)
        for order_id in BOB_ACTIVE_IDS:
            self.assertIn(f"[{order_id}]", bob.answer_text)
        for alice_id in ALICE_IDS:
            self.assertNotIn(alice_id, bob.answer_text)
        self.assertNotEqual(alice.trace_id, bob.trace_id)

    async def test_partial_model_list_cannot_pass_as_all_orders(self) -> None:
        # A model that mentions only one row has not fulfilled the request for
        # all statuses. The result may refuse that answer or reconstruct the
        # full list from reviewed evidence, but cannot label it complete as is.
        gateway = FakeModelGateway(
            GatewayConfig("openai", TEST_MODEL, None, 30.0),
            answer_text="Order DEMO-ORD-1007 is in transit. [DEMO-ORD-1007]",
        )
        result = await answer_question(
            "alice", "List my orders, with all statuses.", gateway,
            db_path=self.db_path,
        )
        self.assertTrue(
            result.outcome != "allow" or all(
                order_id in result.answer_text for order_id in ALICE_IDS
            )
        )
        self.assertTrue(
            result.outcome != "allow" or "delivered" in result.answer_text
        )
        self.assertTrue(
            result.outcome != "allow" or result.source_ids == ALICE_IDS
        )

    async def test_mixed_customer_tool_payload_never_reaches_model(self) -> None:
        # A compromised or regressed MCP tool could claim an allowed list
        # containing a foreign record. The orchestrator must recheck every
        # row, not accept the whole envelope because Alice's first row is valid.
        identity = mint_demo_identity("alice", db_path=self.db_path)
        forged = {
            "outcome": "allow", "trace_id": identity.trace_id,
            "source_ids": ["DEMO-ORD-1007", BOB_ID],
            "rows": [
                {
                    "order_id": "DEMO-ORD-1007", "customer_id": "CUST-1001",
                    "order_status": "in_transit",
                },
                {
                    "order_id": BOB_ID, "customer_id": "CUST-1002",
                    "order_status": "processing",
                },
            ],
        }
        gateway = RecordingGateway()
        with (
            patch("customer_assistant.orchestrator.mint_demo_identity", return_value=identity),
            patch(
                "customer_assistant.orchestrator.mcp_client.customer_orders",
                new=AsyncMock(return_value=forged),
            ),
        ):
            result = await answer_question(
                "alice", "List my orders, with all statuses.", gateway,
                db_path=self.db_path,
            )
        self.assertEqual(result.outcome, "no_match")
        self.assertEqual(result.source_ids, ())
        self.assertEqual(gateway.messages, [])
        self.assertNotIn(BOB_ID, json.dumps(result.as_dict()))
        trace = read_trace(result.trace_id, db_path=self.db_path)
        self.assertEqual([event["action"] for event in trace], [
            "tool_selection", "list_customer_orders",
        ])
        self.assertEqual(trace[-1]["outcome"], "invalid_evidence")
        self.assertEqual(trace[-1]["authorized_evidence_snapshot"], [])


class OrderListStreamlitTests(unittest.TestCase):
    @staticmethod
    @contextmanager
    def _mock_litellm(module: ModuleType):
        # Restore only LiteLLM; AppTest can import NumPy's native extension,
        # which must stay loaded for later UI tests in this same process.
        original = sys.modules.get("litellm")
        sys.modules["litellm"] = module
        try:
            yield
        finally:
            if original is None:
                sys.modules.pop("litellm", None)
            else:
                sys.modules["litellm"] = original

    def test_alice_list_uses_live_provider_and_survives_rerun(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        db_path = Path(temporary.name) / "demo.sqlite3"
        db_patch = patch.object(database, "DEFAULT_DB_PATH", db_path)
        db_patch.start()
        self.addCleanup(db_patch.stop)

        # Replace dotenv parsing before the app starts. The mock provides a
        # fixed synthetic key, and the test never reads the project's .env.
        def synthetic_dotenv(path: Path) -> dict[str, str]:
            self.assertEqual(Path(path), config.PROJECT_ROOT / ".env")
            return {"OPENAI_API_KEY": TEST_OPENAI_KEY, "DEMO_OPENAI_MODEL": TEST_MODEL}

        # A live model's proposed list must cover every owned order. Build the
        # fixture from current expected rows so adding orders cannot leave an
        # outdated short response passing as a complete list.
        listed_orders = "\n".join(
            f"- {order_id} is {status.replace('_', ' ')} [{order_id}]."
            for order_id, status in ALICE_ORDERS
        )
        async def model_completion(**kwargs):
            if "tools" in kwargs:
                payload = json.loads(kwargs["messages"][-1]["content"])
                if payload.get("current_tool_results"):
                    return {"choices": [{"message": {"content": "Evidence is ready."}}]}
                return {"choices": [{"message": {
                    "content": None,
                    "tool_calls": [{
                        "id": "test_order_list", "type": "function",
                        "function": {"name": "list_customer_orders", "arguments": "{}"},
                    }],
                }}]}
            return {
                "choices": [{"message": {"content": "Your orders:\n" + listed_orders}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 13},
            }

        completion = AsyncMock(side_effect=model_completion)
        litellm_stub = ModuleType("litellm")
        litellm_stub.acompletion = completion
        clean_environment = {
            name: value for name, value in os.environ.items()
            if name not in {
                "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEMO_OPENAI_MODEL",
                "DEMO_ANTHROPIC_MODEL", "DEMO_MODEL_PROVIDER",
                "DEMO_MODEL_TIMEOUT_SECONDS", "DEMO_SIGNING_SECRET",
            }
        }
        clean_environment["DEMO_SIGNING_SECRET"] = TEST_SIGNING_SECRET
        with (
            patch.dict(os.environ, clean_environment, clear=True),
            patch.object(config, "dotenv_values", side_effect=synthetic_dotenv),
            self._mock_litellm(litellm_stub),
        ):
            app = AppTest.from_file(str(APP_PATH), default_timeout=45).run()
            app.chat_input[0].set_value("List my orders, with all statuses.").run()
            self.assertEqual(len(app.exception), 0)
            result = app.session_state["chat_history"][-1]["result"]
            self.assertEqual(result["outcome"], "allow")
            self.assertEqual(result["skill"], "list_customer_orders")
            self.assertEqual(result["source_ids"], list(ALICE_IDS))
            self.assertEqual(result["provider"], "openai")
            self.assertEqual(result["model"], TEST_MODEL)
            self.assertEqual(completion.await_count, 3)
            self.assertEqual(
                ["tools" in call.kwargs for call in completion.await_args_list],
                [True, True, False],
            )
            self.assertEqual(completion.await_args.kwargs["api_key"], TEST_OPENAI_KEY)
            self.assertEqual(completion.await_args.kwargs["model"], TEST_MODEL)
            self.assertNotIn(
                TEST_OPENAI_KEY,
                json.dumps(completion.await_args.kwargs["messages"]),
            )
            self.assertNotIn(TEST_OPENAI_KEY, json.dumps(app.session_state["chat_history"]))
            # The refined chat presents these fields on one compact line,
            # while retaining each source ID and the full trace identifier.
            captions = "\n".join(part.value for part in app.get("caption"))
            self.assertIn("Sources: " + ", ".join(ALICE_IDS), captions)
            self.assertIn(f"Trace ID: {result['trace_id']}", captions)
            app.run()
            self.assertEqual(completion.await_count, 3)
            self.assertEqual(len(app.session_state["chat_history"]), 2)


if __name__ == "__main__":
    unittest.main()
