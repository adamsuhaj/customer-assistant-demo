"""Step 12 - Verify provider handoffs preserve only audit-backed chat context.

- Build disposable audit traces and adjacent Streamlit-shaped chat messages.
- Reject foreign, denied, altered, or malformed assistant results before they
  can be included in another provider's continuity summary.
- Exercise one outgoing model summary call and its deterministic fallback with
  offline gateways; no provider API or populated .env file is accessed.
- Confirm raw authorized evidence rows stay out of the summary prompt, while
  the packet retains trace and source IDs for review.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from customer_assistant.audit import record_event
from customer_assistant.config import GatewayConfig
from customer_assistant.database import DEFAULT_SOURCE_DIR, load_database
from customer_assistant.gateway import GatewayResult
from customer_assistant.handoff import (
    CONTINUITY_LABEL, create_handoff, verified_turns,
)


class CountingGateway:
    """Return an offline summary and count calls without provider I/O."""

    def __init__(self, *, answer: str = "The user asked about order 1007.", fail: bool = False):
        self.config = GatewayConfig(
            provider="openai", model="openai/offline-test", api_key=None,
        )
        self.answer = answer
        self.fail = fail
        self.messages = []

    async def complete(self, messages):
        self.messages.append(messages)
        if self.fail:
            raise RuntimeError("synthetic provider failure")
        return GatewayResult(
            provider="openai", model=self.config.model,
            answer_text=self.answer, token_usage=None,
        )


class HandoffTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.db_path = Path(temporary.name) / "demo.sqlite3"
        load_database(DEFAULT_SOURCE_DIR, self.db_path)

    def _history_pair(
        self,
        *,
        trace_id: str = "allowed-alice-trace",
        customer_id: str = "CUST-1001",
        outcome: str = "allow",
        answer: str = "Order DEMO-ORD-1007 is in transit. [DEMO-ORD-1007]",
    ):
        # The extra private row field proves handoff uses the displayed answer
        # and IDs, not the raw evidence snapshot retained for local audit.
        details = {
            "order_id": "DEMO-ORD-1007", "order_status": "in_transit",
            "contact_email": "private-row-field@example.invalid",
        }
        record_event(
            db_path=self.db_path, trace_id=trace_id, user_id=customer_id,
            agent_id="customer_assistant_v1", action="get_order_status",
            outcome=outcome, evidence_ids=["DEMO-ORD-1007"],
            authorized_evidence_snapshot=[details], provider="openai",
            model="openai/offline-test", risk_skill="get_order_status",
        )
        if outcome == "allow":
            record_event(
                db_path=self.db_path, trace_id=trace_id,
                user_id=customer_id, agent_id="customer_assistant_v1",
                action="model_response", outcome="allow",
                evidence_ids=["DEMO-ORD-1007"],
                authorized_evidence_snapshot=[details], provider="openai",
                model="openai/offline-test", risk_skill="get_order_status",
                response_text=answer,
            )
        return [
            {"role": "user", "content": "Where is my order 1007?"},
            {
                "role": "assistant", "content": answer,
                "result": {
                    "outcome": outcome, "skill": "get_order_status",
                    "answer_text": answer, "source_ids": ["DEMO-ORD-1007"],
                    "trace_id": trace_id, "provider": "openai",
                    "model": "openai/offline-test",
                },
            },
        ]

    def test_only_exact_same_customer_allowed_answer_is_verified(self) -> None:
        good = self._history_pair()
        self.assertEqual(len(verified_turns(good, customer_id="CUST-1001", db_path=self.db_path)), 1)
        self.assertEqual(verified_turns(good, customer_id="CUST-1002", db_path=self.db_path), ())

        # A copied source ID, provider, model, answer, or outcome from another
        # UI message cannot turn an unaudited exchange into a valid handoff.
        for field, value in (
            ("source_ids", ["DEMO-ORD-1006"]),
            ("provider", "anthropic"),
            ("model", "anthropic/offline-test"),
            ("answer_text", "Changed answer"),
            ("outcome", "deny"),
        ):
            with self.subTest(field=field):
                altered = [dict(good[0]), {**good[1], "result": {**good[1]["result"], field: value}}]
                self.assertEqual(
                    verified_turns(altered, customer_id="CUST-1001", db_path=self.db_path),
                    (),
                )
        changed_display = [good[0], {**good[1], "content": "Different displayed answer"}]
        self.assertEqual(
            verified_turns(changed_display, customer_id="CUST-1001", db_path=self.db_path),
            (),
        )

    def test_denials_unpaired_messages_and_foreign_traces_are_skipped(self) -> None:
        good = self._history_pair()
        denied = self._history_pair(trace_id="denied-alice-trace", outcome="deny")
        foreign = self._history_pair(
            trace_id="allowed-bob-trace", customer_id="CUST-1002",
        )
        history = [
            {"role": "assistant", "content": "No prior user"},
            *denied,
            {"role": "user", "content": "An unanswered follow-up"},
            *foreign,
            *good,
        ]
        turns = verified_turns(history, customer_id="CUST-1001", db_path=self.db_path)
        self.assertEqual([turn.trace_id for turn in turns], ["allowed-alice-trace"])

    def test_model_handoff_is_bounded_and_never_passes_raw_rows(self) -> None:
        history = self._history_pair()
        gateway = CountingGateway()
        packet = asyncio.run(create_handoff(
            history, customer_id="CUST-1001", from_provider="openai",
            to_provider="anthropic", gateway=gateway, db_path=self.db_path,
        ))
        self.assertIsNotNone(packet)
        self.assertEqual(len(gateway.messages), 1)
        self.assertEqual(packet.summary_origin, "model")
        self.assertTrue(packet.summary.startswith(CONTINUITY_LABEL))
        self.assertEqual(packet.source_ids, ("DEMO-ORD-1007",))
        self.assertEqual(packet.source_trace_ids, ("allowed-alice-trace",))
        self.assertEqual(packet.customer_id, "CUST-1001")
        self.assertEqual((packet.from_provider, packet.to_provider), ("openai", "anthropic"))
        serialized_prompt = json.dumps(gateway.messages)
        self.assertNotIn("private-row-field@example.invalid", serialized_prompt)
        self.assertNotIn("contact_email", serialized_prompt)

    def test_failed_model_uses_local_recap_and_no_valid_turn_skips_call(self) -> None:
        history = self._history_pair()
        gateway = CountingGateway(fail=True)
        packet = asyncio.run(create_handoff(
            history, customer_id="CUST-1001", from_provider="openai",
            to_provider="anthropic", gateway=gateway, db_path=self.db_path,
        ))
        self.assertIsNotNone(packet)
        self.assertEqual(packet.summary_origin, "fallback")
        self.assertIn("DEMO-ORD-1007", packet.summary)
        self.assertEqual(len(gateway.messages), 1)

        foreign_gateway = CountingGateway()
        no_packet = asyncio.run(create_handoff(
            history, customer_id="CUST-1002", from_provider="openai",
            to_provider="anthropic", gateway=foreign_gateway, db_path=self.db_path,
        ))
        self.assertIsNone(no_packet)
        self.assertEqual(foreign_gateway.messages, [])

        # A mismatched gateway must not send the chat to the wrong provider;
        # a local recap still lets the UI complete the selection change.
        wrong_route = CountingGateway()
        wrong_route.config = GatewayConfig(
            provider="anthropic", model="anthropic/offline-test", api_key=None,
        )
        safe_packet = asyncio.run(create_handoff(
            history, customer_id="CUST-1001", from_provider="openai",
            to_provider="anthropic", gateway=wrong_route, db_path=self.db_path,
        ))
        self.assertEqual(safe_packet.summary_origin, "fallback")
        self.assertEqual(wrong_route.messages, [])


if __name__ == "__main__":
    unittest.main()
