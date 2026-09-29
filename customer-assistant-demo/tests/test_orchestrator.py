"""Step 5 - Exercise answer_question() through MCP and an offline model.

- Route single-order, instrument-service, and public warning questions through
  the real MCP subprocess, rather than calling policy functions in isolation.
- Check allowed answers use authorized evidence and retain source citations;
  Bob's denied private read and unverified rows stop before an answer request.
- Switch the provider label while keeping the same tool and entitlement path.
- Inspect model messages so the Step 11 joined self-customer fields can answer
  an account question while signatures, secrets, and foreign profiles stay out.
- Step 12 verifies provider handoff turns against audit lineage. Follow-up
  references trigger a fresh authorized MCP read, and a model cannot reuse an
  old citation or another customer's conversation as current evidence.
- Complement the Step 9 order-list and Step 11 broad service/catalog tests in
  their dedicated flow modules, rather than duplicating those scenarios here.
- Step 14 exercises the fake's native selection contract separately from
  answer completion, while private authorization stays at the tool boundary.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from customer_assistant.config import GatewayConfig
from customer_assistant.database import DEFAULT_SOURCE_DIR, load_database
from customer_assistant.gateway import FakeModelGateway
from customer_assistant.handoff import create_handoff
from customer_assistant.identity import mint_demo_identity
from customer_assistant.orchestrator import answer_question, make_offline_demo_gateway
from customer_assistant.audit import read_trace


TEST_SIGNING_SECRET = "test-only-orchestrator-secret-for-synthetic-data"


class RecordingOfflineGateway:
    """Wrap the grounded fake so tests can inspect what reaches the model."""

    def __init__(self, provider: str = "openai") -> None:
        model = (
            "openai/gpt-6-sol" if provider == "openai"
            else "anthropic/claude-sonnet-5"
        )
        self.config = GatewayConfig(provider, model, None, 30.0)
        self.fake = make_offline_demo_gateway(self.config)
        self.messages: list[list[dict]] = []
        self.selection_messages: list[list[dict]] = []

    async def select_tools(self, messages, tools):
        self.selection_messages.append([dict(message) for message in messages])
        return await self.fake.select_tools(messages, tools)

    async def complete(self, messages):
        self.messages.append([dict(message) for message in messages])
        return await self.fake.complete(messages)


class OrchestratorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        # Every workflow test uses the supplied synthetic CSVs in a disposable
        # database. The signing key exists only in this test process and MCP
        # subprocesses it starts.
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db_path = Path(self.temp.name) / "demo.sqlite3"
        load_database(DEFAULT_SOURCE_DIR, self.db_path)
        env_patch = patch.dict(os.environ, {"DEMO_SIGNING_SECRET": TEST_SIGNING_SECRET})
        env_patch.start()
        self.addCleanup(env_patch.stop)

    async def _alice_order_handoff(self):
        """Build a real allowed turn and an outgoing-provider continuity packet."""

        question = "Where is order DEMO-ORD-1007?"
        outgoing = RecordingOfflineGateway("openai")
        first = await answer_question(
            "alice", question, outgoing, db_path=self.db_path,
        )
        self.assertEqual(first.outcome, "allow")
        # The summarizer stands in for the outgoing OpenAI model. It receives
        # only an already displayed, audit-backed exchange, never an MCP row.
        summary_gateway = FakeModelGateway(
            outgoing.config,
            answer_text="Alice asked about DEMO-ORD-1007 and may ask a follow-up.",
        )
        packet = await create_handoff(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": first.answer_text,
                 "result": first.as_dict()},
            ],
            customer_id="CUST-1001",
            from_provider="openai",
            to_provider="anthropic",
            gateway=summary_gateway,
            db_path=self.db_path,
        )
        self.assertIsNotNone(packet)
        self.assertEqual(packet.summary_origin, "model")
        return first, packet

    async def test_alice_order_and_service_use_authorized_evidence(self) -> None:
        # Use the real MCP subprocess and a recording fake model together so
        # this checks the whole route without making a provider API call.
        gateway = RecordingOfflineGateway()
        order = await answer_question(
            "alice", "Where is order DEMO-ORD-1007?", gateway,
            db_path=self.db_path,
        )
        service = await answer_question(
            "alice", "What happened at the last service visit for DEMO-INS-1001?",
            gateway, db_path=self.db_path,
        )

        self.assertEqual(order.outcome, "allow")
        self.assertEqual(order.skill, "get_order_status")
        self.assertEqual(order.source_ids, ("DEMO-ORD-1007",))
        self.assertIn("in transit", order.answer_text)
        self.assertIn("DEMO-ORD-1007", order.answer_text)
        self.assertEqual(service.outcome, "allow")
        self.assertEqual(service.skill, "get_service_history")
        self.assertEqual(service.source_ids, ("DEMO-SVC-1009", "DEMO-SVC-1001"))
        self.assertIn("closed", service.answer_text)
        self.assertIn("DEMO-SVC-1001", service.answer_text)
        self.assertIn("DEMO-SVC-1009", service.answer_text)
        self.assertNotEqual(order.trace_id, service.trace_id)

        # Step 11 includes the verified *current* customer ID and joined
        # profile fields so self-account questions are answerable. It still
        # excludes the signed context, secret, and other customer's profile.
        sent = json.dumps(gateway.messages)
        self.assertNotIn("identity_context", sent)
        self.assertNotIn("signature", sent)
        self.assertNotIn(TEST_SIGNING_SECRET, sent)
        self.assertIn("CUST-1001", sent)
        self.assertNotIn("CUST-1002", sent)
        self.assertEqual(len(gateway.messages), 2)

    async def test_bob_denial_has_no_answer_call_or_alice_evidence(self) -> None:
        gateway = RecordingOfflineGateway()
        denied = await answer_question(
            "bob", "Show me order DEMO-ORD-1007.", gateway,
            db_path=self.db_path,
        )
        self.assertEqual(denied.outcome, "deny")
        self.assertEqual(denied.skill, "get_order_status")
        self.assertEqual(denied.source_ids, ())
        self.assertEqual(denied.provider, "openai")
        self.assertEqual(denied.model, "openai/gpt-6-sol")
        self.assertTrue(denied.trace_id)
        self.assertNotIn("in_transit", json.dumps(denied.as_dict()))
        self.assertNotIn("DEMO-TRACK-1007", json.dumps(denied.as_dict()))
        self.assertEqual(gateway.messages, [])

    async def test_public_pressure_guidance_cites_approved_article(self) -> None:
        gateway = RecordingOfflineGateway()
        result = await answer_question(
            "alice", "What does Pressure Below Lower Limit mean?", gateway,
            db_path=self.db_path,
        )
        self.assertEqual(result.outcome, "allow")
        self.assertEqual(result.skill, "search_troubleshooting")
        self.assertEqual(result.source_ids, ("DEMO-KB-001",))
        self.assertIn("configured lower limit", result.answer_text)
        self.assertIn("DEMO-KB-001", result.answer_text)
        prompt = json.dumps(gateway.messages)
        self.assertIn("G7111AUser.pdf", prompt)
        self.assertNotIn("identity_context", prompt)

        # Mentioning an instrument while asking what a pressure alert means
        # should still choose public guidance, not private service history.
        with_instrument = await answer_question(
            "alice", "What does the pressure warning mean for DEMO-INS-1001?",
            gateway, db_path=self.db_path,
        )
        self.assertEqual(with_instrument.skill, "search_troubleshooting")
        self.assertEqual(with_instrument.source_ids, ("DEMO-KB-001",))

    async def test_provider_switch_changes_only_model_route(self) -> None:
        results = []
        for provider in ("openai", "anthropic"):
            gateway = RecordingOfflineGateway(provider)
            result = await answer_question(
                "alice", "Where is order DEMO-ORD-1007?", gateway,
                db_path=self.db_path,
            )
            results.append(result)
            self.assertEqual(len(gateway.messages), 1)
        self.assertEqual([result.provider for result in results], ["openai", "anthropic"])
        self.assertEqual(
            [result.model for result in results],
            ["openai/gpt-6-sol", "anthropic/claude-sonnet-5"],
        )
        self.assertEqual([result.source_ids for result in results], [
            ("DEMO-ORD-1007",), ("DEMO-ORD-1007",),
        ])
        self.assertTrue(all("in transit" in result.answer_text for result in results))

    async def test_provider_handoff_resolves_that_order_with_fresh_mcp_read(self) -> None:
        first, packet = await self._alice_order_handoff()
        incoming = RecordingOfflineGateway("anthropic")

        # The new model gets a conversation cue, but the order tool is called
        # again with a newly signed trace; the old answer is not reused.
        from customer_assistant import mcp_client

        with patch(
            "customer_assistant.orchestrator.mcp_client.order_status",
            new=AsyncMock(wraps=mcp_client.order_status),
        ) as fresh_read:
            followup = await answer_question(
                "alice", "What is the status of that order?", incoming,
                db_path=self.db_path, recent_turns=packet.turns, handoff=packet,
            )
        self.assertEqual(followup.outcome, "allow")
        self.assertEqual(followup.skill, "get_order_status")
        self.assertEqual(followup.source_ids, ("DEMO-ORD-1007",))
        self.assertEqual(followup.provider, "anthropic")
        self.assertNotEqual(followup.trace_id, first.trace_id)
        fresh_read.assert_awaited_once()
        self.assertEqual(fresh_read.await_args.args[1], "DEMO-ORD-1007")
        actions = [event["action"] for event in read_trace(
            followup.trace_id, db_path=self.db_path,
        )]
        self.assertIn("get_order_status", actions)
        self.assertTrue(any(action.startswith("provider_handoff_received:") for action in actions))

    async def test_bob_handoff_cannot_enter_alice_conversation(self) -> None:
        bob_question = "Where is order DEMO-ORD-2001?"
        outgoing = RecordingOfflineGateway("openai")
        bob_answer = await answer_question(
            "bob", bob_question, outgoing, db_path=self.db_path,
        )
        self.assertEqual(bob_answer.outcome, "allow")
        packet = await create_handoff(
            [
                {"role": "user", "content": bob_question},
                {"role": "assistant", "content": bob_answer.answer_text,
                 "result": bob_answer.as_dict()},
            ],
            customer_id="CUST-1002", from_provider="openai",
            to_provider="anthropic",
            gateway=FakeModelGateway(
                outgoing.config, answer_text="Bob asked about DEMO-ORD-2001."
            ),
            db_path=self.db_path,
        )
        self.assertIsNotNone(packet)

        incoming = RecordingOfflineGateway("anthropic")
        result = await answer_question(
            "alice", "What is the status of that order?", incoming,
            db_path=self.db_path, recent_turns=packet.turns, handoff=packet,
        )
        # With no Alice-owned verified turn, the reference is ambiguous. The
        # handoff must not supply Bob's ID to planning or trigger a private
        # MCP read or answer completion.
        self.assertEqual(result.outcome, "no_match")
        self.assertEqual(result.source_ids, ())
        self.assertEqual(incoming.messages, [])
        self.assertNotIn("DEMO-ORD-2001", result.answer_text)
        actions = [event["action"] for event in read_trace(
            result.trace_id, db_path=self.db_path,
        )]
        self.assertEqual(actions, ["tool_selection"])
        self.assertEqual(len(incoming.selection_messages), 1)
        planner_payload = json.loads(incoming.selection_messages[0][-1]["content"])
        self.assertNotIn("conversation_context", planner_payload)
        self.assertNotIn("DEMO-ORD-2001", json.dumps(planner_payload))
        self.assertNotIn("CUST-1002", json.dumps(planner_payload))

    async def test_handoff_prompt_has_continuity_without_secrets_or_foreign_rows(self) -> None:
        _, packet = await self._alice_order_handoff()
        incoming = RecordingOfflineGateway("anthropic")
        result = await answer_question(
            "alice", "What is the status of that order?", incoming,
            db_path=self.db_path, recent_turns=packet.turns, handoff=packet,
        )
        self.assertEqual(result.outcome, "allow")
        self.assertEqual(len(incoming.messages), 1)
        prompt = json.loads(incoming.messages[0][-1]["content"])
        context = prompt["conversation_context"]
        self.assertEqual(context["recent_turns"][0]["trace_id"], packet.turns[0].trace_id)
        self.assertEqual(context["provider_handoff"]["handoff_id"], packet.handoff_id)
        self.assertIn("DEMO-ORD-1007", context["provider_handoff"]["summary"])
        context_text = json.dumps(context)
        self.assertNotIn("authorized_evidence_snapshot", context_text)
        self.assertNotIn("identity_context", context_text)
        self.assertNotIn("signature", context_text)
        self.assertNotIn(TEST_SIGNING_SECRET, context_text)
        self.assertNotIn("CUST-1002", context_text)
        self.assertNotIn("DEMO-ORD-2001", context_text)

    async def test_handoff_cannot_make_old_order_id_a_current_citation(self) -> None:
        _, packet = await self._alice_order_handoff()
        # The question requests a different Alice-owned order. The prior ID
        # may occur in continuity context, but the recipient may cite only
        # the newly fetched order's source ID.
        incoming = FakeModelGateway(
            GatewayConfig("anthropic", "anthropic/claude-sonnet-5", None, 30.0),
            answer_text="The earlier order was in transit. [DEMO-ORD-1007]",
        )
        result = await answer_question(
            "alice", "Where is order DEMO-ORD-1008?", incoming,
            db_path=self.db_path, recent_turns=packet.turns, handoff=packet,
        )
        self.assertEqual(result.outcome, "no_match")
        self.assertEqual(result.source_ids, ())
        self.assertNotIn("DEMO-ORD-1007", result.answer_text)
        events = read_trace(result.trace_id, db_path=self.db_path)
        self.assertEqual(events[-1]["outcome"], "rejected_historical_source")
        self.assertEqual(events[-1]["evidence_ids"], ["DEMO-ORD-1008"])

    async def test_uncited_model_text_is_not_relabelled_as_grounded(self) -> None:
        config = GatewayConfig("openai", "openai/gpt-6-sol", None, 30.0)
        unsupported = FakeModelGateway(
            config, answer_text="Your order has been delivered."
        )
        result = await answer_question(
            "alice", "Where is order DEMO-ORD-1007?", unsupported,
            db_path=self.db_path,
        )
        self.assertEqual(result.outcome, "no_match")
        self.assertEqual(result.source_ids, ())
        self.assertNotIn("delivered", result.answer_text)

    async def test_cited_false_single_order_status_is_rejected(self) -> None:
        # A handoff can carry an older answer. Even with a current citation,
        # a recipient must not call Alice's in-transit order delivered.
        config = GatewayConfig("openai", "openai/gpt-6-sol", None, 30.0)
        false_status = FakeModelGateway(
            config,
            answer_text="Order DEMO-ORD-1007 is delivered. [DEMO-ORD-1007]",
        )
        result = await answer_question(
            "alice", "Where is order DEMO-ORD-1007?", false_status,
            db_path=self.db_path,
        )
        self.assertEqual(result.outcome, "no_match")
        self.assertIn("could not verify its status", result.answer_text)
        self.assertEqual(result.source_ids, ())

    async def test_missing_or_unverified_evidence_stops_before_answer(self) -> None:
        gateway = RecordingOfflineGateway()
        missing = await answer_question(
            "alice", "Where is my order?", gateway, db_path=self.db_path,
        )
        wrong_model = await answer_question(
            "alice", "What does the 1290 pump pressure warning mean?", gateway,
            db_path=self.db_path,
        )
        self.assertEqual(missing.outcome, "no_match")
        self.assertEqual(wrong_model.outcome, "no_match")
        self.assertEqual(missing.source_ids, ())
        self.assertEqual(wrong_model.source_ids, ())
        self.assertEqual(gateway.messages, [])

        # Even if an MCP implementation regressed and labeled a foreign row
        # 'allow', the orchestrator must refuse to place it in a model prompt.
        identity = mint_demo_identity("alice", db_path=self.db_path)
        forged = {
            "outcome": "allow", "trace_id": identity.trace_id,
            "source_ids": ["DEMO-ORD-2001"],
            "rows": [{"order_id": "DEMO-ORD-2001", "customer_id": "CUST-1002"}],
        }
        with patch(
            "customer_assistant.orchestrator.mint_demo_identity",
            return_value=identity,
        ), patch(
            "customer_assistant.orchestrator.mcp_client.order_status",
            new=AsyncMock(return_value=forged),
        ):
            rejected = await answer_question(
                "alice", "Where is order DEMO-ORD-2001?", gateway,
                db_path=self.db_path,
            )
        self.assertEqual(rejected.outcome, "no_match")
        self.assertEqual(rejected.source_ids, ())
        self.assertEqual(gateway.messages, [])
        self.assertNotIn("CUST-1002", json.dumps(gateway.selection_messages))

        incomplete = {
            "outcome": "allow", "trace_id": identity.trace_id,
            "source_ids": ["DEMO-ORD-1007"],
            "rows": [{"order_id": "DEMO-ORD-1007", "customer_id": "CUST-1001"}],
        }
        with patch(
            "customer_assistant.orchestrator.mint_demo_identity",
            return_value=identity,
        ), patch(
            "customer_assistant.orchestrator.mcp_client.order_status",
            new=AsyncMock(return_value=incomplete),
        ):
            rejected = await answer_question(
                "alice", "Where is order DEMO-ORD-1007?", gateway,
                db_path=self.db_path,
            )
        self.assertEqual(rejected.outcome, "no_match")
        self.assertEqual(gateway.messages, [])

        wrong_id = {
            "outcome": "allow", "trace_id": identity.trace_id,
            "source_ids": ["DEMO-ORD-1002"],
            "rows": [{
                "order_id": "DEMO-ORD-1002", "customer_id": "CUST-1001",
                "order_status": "delivered",
            }],
        }
        with patch(
            "customer_assistant.orchestrator.mint_demo_identity",
            return_value=identity,
        ), patch(
            "customer_assistant.orchestrator.mcp_client.order_status",
            new=AsyncMock(return_value=wrong_id),
        ):
            rejected = await answer_question(
                "alice", "Where is order DEMO-ORD-1007?", gateway,
                db_path=self.db_path,
            )
        self.assertEqual(rejected.outcome, "no_match")
        self.assertEqual(gateway.messages, [])


if __name__ == "__main__":
    unittest.main()
