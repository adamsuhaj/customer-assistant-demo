"""Step 11 - Keep customer relationships and broad questions grounded.

- Replay Alice's order-list conversation and prove a request for Bob's orders
  cannot reuse Alice's evidence or send another private read to the model.
- Confirm self-account follow-ups expose only the signed customer's joined ID,
  organization name, and demo email, each backed by an owned order citation.
- Check that ordering by date uses the recorded order creation date, and reject
  a model answer that substitutes delivery estimates or claims dates are absent.
- Exercise ID-free service history and broad troubleshooting through the normal
  orchestrator and MCP path, with complete owned or approved-public citations.
- Keep every test on a disposable synthetic database and an offline gateway;
  provider credentials and the project's .env file are never read.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import AsyncMock, patch

from customer_assistant.audit import read_trace
from customer_assistant.config import GatewayConfig
from customer_assistant.database import DEFAULT_SOURCE_DIR, load_database
from customer_assistant.gateway import FakeModelGateway
from customer_assistant.orchestrator import answer_question, make_offline_demo_gateway


TEST_SIGNING_SECRET = "test-only-customer-relationship-signing-secret"
TEST_CONFIG = GatewayConfig("openai", "openai/offline-relationship-test", None, 30.0)
ALICE_ORDER_DATES = (
    ("DEMO-ORD-1008", "2026-09-26", "processing"),
    ("DEMO-ORD-1007", "2026-09-22", "in_transit"),
    ("DEMO-ORD-1009", "2026-09-21", "in_transit"),
    ("DEMO-ORD-1006", "2026-09-19", "in_transit"),
    ("DEMO-ORD-1004", "2026-08-24", "delivered"),
    ("DEMO-ORD-1002", "2026-08-08", "delivered"),
)


class RecordingOfflineGateway:
    """Capture the actual approved prompt while using the local answer path."""

    def __init__(self) -> None:
        self.config = TEST_CONFIG
        self._offline = make_offline_demo_gateway(self.config)
        self.messages: list[list[dict[str, str]]] = []

    async def select_tools(self, messages, tools):
        return await self._offline.select_tools(messages, tools)

    async def complete(self, messages):
        self.messages.append([dict(message) for message in messages])
        return await self._offline.complete(messages)


class CustomerRelationshipFlowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.db_path = Path(temporary.name) / "demo.sqlite3"
        load_database(DEFAULT_SOURCE_DIR, self.db_path)
        signing = patch.dict(os.environ, {"DEMO_SIGNING_SECRET": TEST_SIGNING_SECRET})
        signing.start()
        self.addCleanup(signing.stop)

    async def test_alice_cannot_ask_for_bobs_orders_after_her_own_list(self) -> None:
        gateway = RecordingOfflineGateway()
        own = await answer_question(
            "alice", "List all my orders by status.", gateway,
            db_path=self.db_path,
        )
        self.assertEqual(own.outcome, "allow")
        self.assertEqual(len(gateway.messages), 1)

        # The second turn uses the same gateway, as a UI chat would. A named
        # foreign customer must stop at routing, before any private MCP read or
        # new model prompt can relabel the prior Alice order list as Bob's.
        with patch(
            "customer_assistant.orchestrator.mcp_client.customer_orders",
            new_callable=AsyncMock,
        ) as customer_orders:
            foreign = await answer_question(
                "alice", "Can you tell me Bob's orders?", gateway,
                db_path=self.db_path,
            )
        customer_orders.assert_not_awaited()
        self.assertEqual(foreign.outcome, "deny")
        self.assertEqual(foreign.source_ids, ())
        self.assertEqual(len(gateway.messages), 1)
        for order_id in (*own.source_ids, "DEMO-ORD-2001", "DEMO-ORD-2005"):
            self.assertNotIn(order_id, foreign.answer_text)
        trace = read_trace(foreign.trace_id, db_path=self.db_path)
        self.assertEqual(len(trace), 1)
        self.assertEqual(trace[0]["action"], "route_question")
        self.assertEqual(trace[0]["outcome"], "deny")
        self.assertEqual(trace[0]["authorized_evidence_snapshot"], [])

    async def test_model_cannot_relabel_alices_owned_orders_as_bobs(self) -> None:
        # Defense in depth: a prompt naming Bob is stopped before retrieval,
        # and an unrelated model hallucination must also fail after retrieval.
        mislabeled = "Bob's orders:\n" + "\n".join(
            f"- {order_id}: {status.replace('_', ' ')} [{order_id}]"
            for order_id, _date, status in ALICE_ORDER_DATES
        )
        gateway = FakeModelGateway(TEST_CONFIG, answer_text=mislabeled)
        result = await answer_question(
            "alice", "List all my orders by status.", gateway,
            db_path=self.db_path,
        )
        self.assertNotEqual(result.outcome, "allow")
        self.assertEqual(result.source_ids, ())
        self.assertNotIn("Bob's orders", result.answer_text)
        trace = read_trace(result.trace_id, db_path=self.db_path)
        self.assertEqual(trace[-1]["outcome"], "rejected_customer_fields")

    async def test_self_customer_fields_are_joined_and_answered_truthfully(self) -> None:
        for question in (
            "Do you see the customer IDs and a name or email address?",
            "Do you see any customer ID, name or email related to these orders?",
        ):
            with self.subTest(question=question):
                gateway = RecordingOfflineGateway()
                result = await answer_question(
                    "alice", question, gateway, db_path=self.db_path,
                )
                self.assertEqual(result.outcome, "allow", result.answer_text)
                self.assertEqual(result.skill, "list_customer_orders")
                self.assertEqual(len(gateway.messages), 1)
                payload = json.loads(gateway.messages[0][-1]["content"])
                self.assertEqual(len(payload["evidence"]), 6)
                self.assertTrue(all(
                    row["customer_id"] == "CUST-1001"
                    and row["customer_name"] == "Northstar Bioanalytics Demo Ltd"
                    and row["contact_email"] == "alice@example.com"
                    for row in payload["evidence"]
                ))
                self.assertIn("CUST-1001", result.answer_text)
                self.assertIn("Northstar Bioanalytics Demo Ltd", result.answer_text)
                self.assertIn("alice@example.com", result.answer_text)
                self.assertIn("[DEMO-ORD-", result.answer_text)
                self.assertNotIn("CUST-1002", result.answer_text)
                self.assertNotIn("bob@example.net", result.answer_text)

    async def test_order_by_date_uses_created_on_and_rejects_false_absence(self) -> None:
        question = "Hi! Please list all my orders, by status. Order by date."
        gateway = RecordingOfflineGateway()
        result = await answer_question("alice", question, gateway, db_path=self.db_path)
        self.assertEqual(result.outcome, "allow")
        payload = json.loads(gateway.messages[0][-1]["content"])
        self.assertEqual(
            [(row["order_id"], row["created_on"], row["order_status"])
             for row in payload["evidence"]],
            list(ALICE_ORDER_DATES),
        )
        answer_positions = []
        for order_id, created_on, _status in ALICE_ORDER_DATES:
            self.assertIn(created_on, result.answer_text)
            answer_positions.append(result.answer_text.index(f"[{order_id}]"))
        self.assertEqual(answer_positions, sorted(answer_positions))
        self.assertNotIn("dates weren", result.answer_text.casefold())

        # This resembles the reported live response: every order/status has a
        # citation, but the model says order dates were unavailable and shows
        # estimated delivery dates. It must not be displayed as an allowed
        # grounded answer to an order-date request.
        misleading = (
            "Order dates weren't provided; these are delivery estimates.\n"
            "- DEMO-ORD-1008 processing, 2026-10-05 [DEMO-ORD-1008]\n"
            "- DEMO-ORD-1007 in transit, 2026-09-30 [DEMO-ORD-1007]\n"
            "- DEMO-ORD-1009 in transit, 2026-10-02 [DEMO-ORD-1009]\n"
            "- DEMO-ORD-1006 in transit, 2026-09-29 [DEMO-ORD-1006]\n"
            "- DEMO-ORD-1004 delivered, 2026-08-30 [DEMO-ORD-1004]\n"
            "- DEMO-ORD-1002 delivered, 2026-08-12 [DEMO-ORD-1002]"
        )
        misleading_gateway = FakeModelGateway(TEST_CONFIG, answer_text=misleading)
        stopped = await answer_question(
            "alice", question, misleading_gateway, db_path=self.db_path,
        )
        self.assertNotEqual(stopped.outcome, "allow")
        self.assertEqual(stopped.source_ids, ())
        self.assertNotIn("Order dates weren't provided", stopped.answer_text)
        trace = read_trace(stopped.trace_id, db_path=self.db_path)
        self.assertEqual(trace[-1]["outcome"], "rejected_coverage")

    async def test_model_cannot_claim_linked_customer_fields_are_absent(self) -> None:
        # Reproduce the misleading live-model response in the reported chat.
        # Even a cited answer must be checked against the joined customer row
        # when the question explicitly asks for those fields.
        misleading_gateway = FakeModelGateway(
            TEST_CONFIG,
            answer_text=(
                "No. The order records do not show a customer ID, name, or "
                "email. [DEMO-ORD-1008]"
            ),
        )
        result = await answer_question(
            "alice", "Do you see any customer ID, name or email related to these orders?",
            misleading_gateway, db_path=self.db_path,
        )
        self.assertNotEqual(result.outcome, "allow")
        self.assertEqual(result.source_ids, ())
        self.assertNotIn("do not show a customer ID", result.answer_text)
        trace = read_trace(result.trace_id, db_path=self.db_path)
        self.assertEqual(trace[-1]["outcome"], "rejected_customer_fields")

    async def test_id_free_service_history_stays_with_signed_customer(self) -> None:
        for login, owned, foreign in (
            ("alice", ("DEMO-SVC-1009", "DEMO-SVC-1001"), ("DEMO-SVC-2001",)),
            ("bob", ("DEMO-SVC-2001",), ("DEMO-SVC-1009", "DEMO-SVC-1001")),
        ):
            with self.subTest(login=login):
                gateway = RecordingOfflineGateway()
                result = await answer_question(
                    login, "Show my service history.", gateway,
                    db_path=self.db_path,
                )
                self.assertEqual(result.outcome, "allow")
                self.assertEqual(result.skill, "get_service_history")
                self.assertEqual(result.source_ids, owned)
                for source_id in owned:
                    self.assertIn(f"[{source_id}]", result.answer_text)
                for source_id in foreign:
                    self.assertNotIn(source_id, result.answer_text)
                payload = json.loads(gateway.messages[0][-1]["content"])
                self.assertEqual(payload["source_ids"], list(owned))
                self.assertEqual(len(payload["evidence"]), len(owned))
                for source_id in foreign:
                    self.assertNotIn(source_id, json.dumps(payload))

    async def test_id_free_service_history_requires_every_owned_event(self) -> None:
        # The source dataset has two Alice service events. Add another
        # valid Alice event in this disposable DB so a one-record answer cannot
        # accidentally satisfy a request for the full history.
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute(
                "INSERT INTO service_history "
                "(service_event_id, customer_id, instrument_id, event_date, "
                "event_type, reported_symptom, technician_summary, "
                "resolution_status, access_tier, record_source) "
                "SELECT 'DEMO-SVC-1002', customer_id, instrument_id, "
                "'2026-09-10', event_type, reported_symptom, "
                "'Synthetic additional service event.', resolution_status, "
                "access_tier, record_source FROM service_history "
                "WHERE service_event_id = 'DEMO-SVC-1001'"
            )
            connection.commit()
        gateway = RecordingOfflineGateway()
        complete = await answer_question(
            "alice", "Show my service history.", gateway, db_path=self.db_path,
        )
        self.assertEqual(complete.outcome, "allow")
        self.assertEqual(complete.source_ids, ("DEMO-SVC-1009", "DEMO-SVC-1002", "DEMO-SVC-1001"))
        for source_id in complete.source_ids:
            self.assertIn(f"[{source_id}]", complete.answer_text)

        incomplete = FakeModelGateway(
            TEST_CONFIG,
            answer_text="One visit is closed. [DEMO-SVC-1001]",
        )
        stopped = await answer_question(
            "alice", "Show my service history.", incomplete,
            db_path=self.db_path,
        )
        self.assertNotEqual(stopped.outcome, "allow")
        self.assertEqual(stopped.source_ids, ())
        self.assertEqual(
            read_trace(stopped.trace_id, db_path=self.db_path)[-1]["outcome"],
            "rejected_coverage",
        )

    async def test_broad_troubleshooting_lists_only_approved_public_articles(self) -> None:
        gateway = RecordingOfflineGateway()
        result = await answer_question(
            "alice", "What troubleshooting guidance is available?", gateway,
            db_path=self.db_path,
        )
        self.assertEqual(result.outcome, "allow")
        self.assertEqual(result.skill, "search_troubleshooting")
        self.assertEqual(result.source_ids, ("DEMO-KB-001", "DEMO-KB-002"))
        for source_id in result.source_ids:
            self.assertIn(f"[{source_id}]", result.answer_text)
        payload = json.loads(gateway.messages[0][-1]["content"])
        self.assertEqual(payload["source_ids"], list(result.source_ids))
        self.assertTrue(all(
            row["access_tier"] == "0" for row in payload["evidence"]
        ))

    async def test_dataset_question_describes_relationship_without_private_rows(self) -> None:
        gateway = RecordingOfflineGateway()
        result = await answer_question(
            "alice", "What datasets do you see in your database?", gateway,
            db_path=self.db_path,
        )
        self.assertEqual(result.outcome, "allow")
        self.assertEqual(result.source_ids, ())
        self.assertEqual(gateway.messages, [])
        for label in ("customers", "instruments", "orders", "service history"):
            self.assertIn(label, result.answer_text)
        self.assertNotIn("alice@example.com", result.answer_text)


if __name__ == "__main__":
    unittest.main()
