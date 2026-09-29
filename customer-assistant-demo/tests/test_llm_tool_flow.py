"""Exercise model-selected MCP planning, composition, and security boundaries.

The catalog is discovered once from the real stdio MCP server. Each scenario
uses a disposable database and explicit offline/injected model responses;
owned multi-tool reads and denial traverse the real MCP process boundary.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
import inspect
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from customer_assistant import mcp_client
from customer_assistant.audit import read_trace
from customer_assistant.config import GatewayConfig
from customer_assistant.database import DEFAULT_SOURCE_DIR, load_database
from customer_assistant.gateway import (
    FakeModelGateway, LiteLLMGateway, SAFE_TOOL_CLARIFICATION, ToolCall, ToolSelectionResult,
)
from customer_assistant.identity import verify_demo_identity
from customer_assistant.orchestrator import (
    MAX_PLANNING_ROUNDS, answer_question, make_offline_demo_gateway,
)


TEST_SIGNING_SECRET = "synthetic-test-llm-tool-flow-signing-secret"
TEST_CONFIG = GatewayConfig("openai", "openai/offline-tool-flow", None, 30.0)


def order_call(call_id="order-1", order_id="DEMO-ORD-1007"):
    return ToolCall(call_id, "get_order_status", {"order_id": order_id})


def service_call(call_id="service-1", instrument_id="DEMO-INS-1001"):
    return ToolCall(call_id, "get_service_history", {"instrument_id": instrument_id})


def owned_order_row(order_id="DEMO-ORD-1007"):
    """Minimal authorized-looking fixture for rejection/projection tests."""

    return {
        "order_id": order_id,
        "customer_id": "CUST-1001",
        "customer_name": "Northstar Bioanalytics Demo Ltd",
        "contact_email": "alice@example.com",
        "created_on": "2026-09-22",
        "order_status": "in_transit",
        "instrument_id": "DEMO-INS-1001",
        "estimated_delivery": "2026-09-30",
    }


class RecordingPlannerGateway:
    """Script model choices while recording every planner/answer input.

    Normal scenarios use FakeModelGateway's public selector seam. A raw
    injected-result option separately tests the orchestrator's own checks,
    without relying on the provider adapter to reject malformed proposals.
    """

    def __init__(self, selector, *, config=TEST_CONFIG, raw_selection=False):
        self.config = config
        self.selector = selector
        self.raw_selection = raw_selection
        self._planner = FakeModelGateway(config, tool_selector=selector)
        self._answer = make_offline_demo_gateway(config)
        self.planning_messages = []
        self.planning_catalogs = []
        self.answer_messages = []

    async def select_tools(self, messages, tools):
        self.planning_messages.append(deepcopy(messages))
        self.planning_catalogs.append(deepcopy(tools))
        if not self.raw_selection:
            return await self._planner.select_tools(messages, tools)
        result = self.selector(messages, tools)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, ToolSelectionResult):
            return result
        return ToolSelectionResult(self.config.provider, self.config.model, tuple(result))

    async def complete(self, messages):
        self.answer_messages.append(deepcopy(messages))
        return await self._answer.complete(messages)

    def planning_payload(self, index):
        return json.loads(self.planning_messages[index][-1]["content"])


class LLMToolFlowTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        # Cache only the advertised definitions, never tool results or identity.
        # Test calls later use their own databases and fresh signed claims.
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "catalog.sqlite3"
            load_database(DEFAULT_SOURCE_DIR, db_path)
            with patch.dict(os.environ, {"DEMO_SIGNING_SECRET": TEST_SIGNING_SECRET}):
                cls.catalog = asyncio.run(mcp_client.discover_model_tools(db_path))

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.db_path = Path(directory.name) / "demo.sqlite3"
        load_database(DEFAULT_SOURCE_DIR, self.db_path)
        environment = patch.dict(os.environ, {"DEMO_SIGNING_SECRET": TEST_SIGNING_SECRET})
        environment.start()
        self.addCleanup(environment.stop)
        discovery = patch(
            "customer_assistant.orchestrator.mcp_client.discover_model_tools",
            new=AsyncMock(return_value=deepcopy(self.catalog)),
        )
        self.discovery = discovery.start()
        self.addCleanup(discovery.stop)

    async def ask(self, gateway, question="Please summarize my records.", *, login="alice"):
        return await answer_question(login, question, gateway, db_path=self.db_path)

    async def test_simultaneous_owned_order_and_service_are_combined_through_real_mcp(self):
        def selector(messages, tools):
            payload = json.loads(messages[-1]["content"])
            return () if payload.get("current_tool_results") else (order_call(), service_call())

        gateway = RecordingPlannerGateway(selector)
        actual_call_tool = mcp_client.call_tool
        with patch(
            "customer_assistant.mcp_client.call_tool", wraps=actual_call_tool,
        ) as tool_hop:
            result = await self.ask(
                gateway,
                "Tell me about DEMO-ORD-1007 and the visits for DEMO-INS-1001 together.",
            )
        self.assertEqual(result.outcome, "allow")
        self.assertEqual(result.skill, "composite")
        self.assertEqual(result.source_ids, (
            "DEMO-ORD-1007", "DEMO-SVC-1009", "DEMO-SVC-1001",
        ))
        self.assertIn("in transit", result.answer_text)
        self.assertIn("[DEMO-SVC-1001]", result.answer_text)
        self.assertEqual(tool_hop.await_count, 2)
        signed_claims = []
        for awaited in tool_hop.await_args_list:
            self.assertIn(awaited.args[0], {"get_order_status", "get_service_history"})
            identity = verify_demo_identity(awaited.args[1]["identity_context"])
            self.assertEqual(identity.user_id, "CUST-1001")
            self.assertEqual(identity.trace_id, result.trace_id)
            signed_claims.append(identity)
        self.assertEqual(signed_claims[0], signed_claims[1])
        self.assertEqual(len(gateway.planning_messages), 2)
        self.assertEqual(len(gateway.answer_messages), 1)
        second = gateway.planning_payload(1)["current_tool_results"]
        self.assertEqual([batch["skill"] for batch in second], [
            "get_order_status", "get_service_history",
        ])
        sent = json.dumps(gateway.planning_messages + gateway.answer_messages)
        self.assertNotIn("identity_context", sent)
        self.assertNotIn("signature", sent)
        self.assertNotIn(TEST_SIGNING_SECRET, sent)
        self.assertNotIn("CUST-1002", sent)
        self.assertEqual(gateway.planning_catalogs[0], self.catalog)
        self.assertEqual([
            row["action"] for row in read_trace(result.trace_id, db_path=self.db_path)
        ], ["tool_selection", "get_order_status", "get_service_history", "tool_selection", "model_response"])

    async def test_dependent_service_id_is_learned_from_authorized_order_result(self):
        def selector(messages, tools):
            payload = json.loads(messages[-1]["content"])
            batches = payload.get("current_tool_results", [])
            if not batches:
                return (order_call(),)
            if len(batches) == 1:
                instrument_id = batches[0]["evidence"][0]["instrument_id"]
                return (service_call(instrument_id=instrument_id),)
            return ()

        gateway = RecordingPlannerGateway(selector)
        result = await self.ask(gateway, "Check DEMO-ORD-1007, then its associated equipment's visits.")
        self.assertEqual(result.outcome, "allow")
        self.assertEqual(result.skill, "composite")
        self.assertEqual(len(gateway.planning_messages), 3)
        self.assertNotIn("DEMO-INS-1001", gateway.planning_messages[0][-1]["content"])
        order_batch = gateway.planning_payload(1)["current_tool_results"][0]
        self.assertEqual(order_batch["evidence"][0]["instrument_id"], "DEMO-INS-1001")
        self.assertEqual(gateway.planning_payload(2)["current_tool_results"][1]["arguments"], {
            "instrument_id": "DEMO-INS-1001",
        })
        self.assertIn("[DEMO-ORD-1007]", result.answer_text)
        self.assertIn("[DEMO-SVC-1009]", result.answer_text)

    async def test_unfamiliar_paraphrase_uses_model_choice_without_keyword_gating(self):
        question = "Can you shed some light on that little box I am waiting for?"
        def selector(messages, tools):
            payload = json.loads(messages[-1]["content"])
            return () if payload.get("current_tool_results") else (order_call(),)

        gateway = RecordingPlannerGateway(selector)
        result = await self.ask(gateway, question)
        self.assertEqual(result.outcome, "allow")
        self.assertEqual(result.skill, "get_order_status")
        self.assertEqual(gateway.planning_payload(0)["question"], question)
        self.assertIn("[DEMO-ORD-1007]", result.answer_text)

    async def test_whole_batch_is_rejected_before_any_read_for_bad_arguments(self):
        bad_calls = (
            ToolCall("bad-1", "get_order_status", {}),
            ToolCall("bad-1", "get_service_history", {"instrument_id": 1001}),
            ToolCall("bad-1", "list_customer_orders", {"active_only": "true"}),
            ToolCall("bad-1", "list_customer_orders", {"customer_id": "CUST-1002"}),
            ToolCall("bad-1", "get_order_status", {
                "order_id": "DEMO-ORD-1007", "identity_context": {"user_id": "CUST-1002"},
            }),
            ToolCall("bad-1", "get_order_status", {"order_id": "DEMO-ORD-1007", "risk_tier": 0}),
            ToolCall("bad-1", "get_order_status", {"order_id": "DEMO-ORD-1007 OR 1=1"}),
            ToolCall("bad-1", "admin_sql", {"query": "SELECT * FROM orders"}),
        )
        for bad_call in bad_calls:
            with self.subTest(call=bad_call):
                gateway = RecordingPlannerGateway(
                    lambda messages, tools: (order_call(), bad_call), raw_selection=True,
                )
                with patch(
                    "customer_assistant.orchestrator._execute_selected_tool", new_callable=AsyncMock,
                ) as execute:
                    result = await self.ask(gateway)
                execute.assert_not_awaited()
                self.assertEqual(result.outcome, "no_match")
                self.assertEqual(result.source_ids, ())
                self.assertEqual(gateway.answer_messages, [])
                self.assertEqual(read_trace(result.trace_id, db_path=self.db_path)[-1]["outcome"], "rejected_arguments")

    async def test_changed_identity_or_open_schema_blocks_every_call(self):
        for change in ("identity", "open_schema"):
            with self.subTest(change=change):
                changed_catalog = deepcopy(self.catalog)
                parameters = changed_catalog[0]["function"]["parameters"]
                if change == "identity":
                    parameters["properties"]["identity_context"] = {"type": "object"}
                else:
                    parameters["additionalProperties"] = True
                self.discovery.return_value = changed_catalog
                gateway = RecordingPlannerGateway(
                    lambda messages, tools: (order_call(), service_call()), raw_selection=True,
                )
                with patch(
                    "customer_assistant.orchestrator._execute_selected_tool", new_callable=AsyncMock,
                ) as execute:
                    result = await self.ask(gateway)
                execute.assert_not_awaited()
                self.assertEqual(result.outcome, "no_match")
                self.assertEqual(gateway.answer_messages, [])

    async def test_unowned_real_mcp_read_denies_without_second_plan_or_answer(self):
        gateway = RecordingPlannerGateway(lambda messages, tools: (order_call(),))
        result = await self.ask(gateway, "What happened to DEMO-ORD-1007?", login="bob")
        self.assertEqual(result.outcome, "deny")
        self.assertEqual(result.source_ids, ())
        self.assertEqual(len(gateway.planning_messages), 1)
        self.assertEqual(gateway.answer_messages, [])
        self.assertNotIn("in_transit", json.dumps(result.as_dict()))
        self.assertNotIn("DEMO-TRACK-1007", json.dumps(gateway.planning_messages))
        self.assertEqual(read_trace(result.trace_id, db_path=self.db_path)[-1]["authorized_evidence_snapshot"], [])

    async def test_wrong_provider_or_model_selection_stops_before_execution(self):
        mismatches = (
            ToolSelectionResult("anthropic", "anthropic/claude-sonnet-5", (order_call(),)),
            ToolSelectionResult(TEST_CONFIG.provider, "openai/other-route", (order_call(),)),
        )
        for selected in mismatches:
            with self.subTest(selected=selected):
                gateway = RecordingPlannerGateway(
                    lambda messages, tools: selected, raw_selection=True,
                )
                with patch(
                    "customer_assistant.orchestrator._execute_selected_tool", new_callable=AsyncMock,
                ) as execute:
                    result = await self.ask(gateway)
                execute.assert_not_awaited()
                self.assertEqual(result.outcome, "no_match")
                self.assertEqual(gateway.answer_messages, [])
                self.assertEqual(read_trace(result.trace_id, db_path=self.db_path)[-1]["outcome"], "rejected_route")

    async def test_duplicate_calls_or_call_ids_reject_whole_batch(self):
        batches = (
            (order_call("first"), order_call("second")),
            (order_call("same"), service_call("same")),
        )
        for calls in batches:
            with self.subTest(calls=calls):
                gateway = RecordingPlannerGateway(lambda messages, tools: calls, raw_selection=True)
                with patch(
                    "customer_assistant.orchestrator._execute_selected_tool", new_callable=AsyncMock,
                ) as execute:
                    result = await self.ask(gateway)
                execute.assert_not_awaited()
                self.assertEqual(result.outcome, "no_match")
                self.assertEqual(gateway.answer_messages, [])

    async def test_completed_call_cannot_repeat_in_next_round(self):
        def selector(messages, tools):
            payload = json.loads(messages[-1]["content"])
            return (order_call("again" if payload.get("current_tool_results") else "first"),)

        gateway = RecordingPlannerGateway(selector)
        calls = 0
        async def evidence(call, arguments, identity, db_path):
            nonlocal calls
            calls += 1
            return {"outcome": "allow", "trace_id": identity.trace_id,
                    "rows": [owned_order_row()], "source_ids": ["DEMO-ORD-1007"]}

        with patch("customer_assistant.orchestrator._execute_selected_tool", side_effect=evidence):
            result = await self.ask(gateway)
        self.assertEqual(calls, 1)
        self.assertEqual(len(gateway.planning_messages), 2)
        self.assertEqual(gateway.answer_messages, [])
        self.assertEqual(result.outcome, "no_match")

    async def test_initial_batch_over_four_calls_runs_none(self):
        calls = tuple(order_call(f"order-{index}", f"DEMO-ORD-{1002 + index}") for index in range(5))
        gateway = RecordingPlannerGateway(lambda messages, tools: calls)
        with patch(
            "customer_assistant.orchestrator._execute_selected_tool", new_callable=AsyncMock,
        ) as execute:
            result = await self.ask(gateway)
        execute.assert_not_awaited()
        self.assertEqual(result.outcome, "no_match")
        self.assertEqual(read_trace(result.trace_id, db_path=self.db_path)[-1]["outcome"], "rejected_limit")

    async def test_total_call_budget_applies_across_rounds(self):
        def selector(messages, tools):
            payload = json.loads(messages[-1]["content"])
            start = 1005 if payload.get("current_tool_results") else 1002
            count = 2 if payload.get("current_tool_results") else 3
            return tuple(order_call(f"order-{index}", f"DEMO-ORD-{index}") for index in range(start, start + count))

        gateway = RecordingPlannerGateway(selector)
        async def evidence(call, arguments, identity, db_path):
            order_id = arguments["order_id"]
            return {"outcome": "allow", "trace_id": identity.trace_id,
                    "rows": [owned_order_row(order_id)], "source_ids": [order_id]}

        with patch(
            "customer_assistant.orchestrator._execute_selected_tool", side_effect=evidence,
        ) as execute:
            result = await self.ask(gateway)
        self.assertEqual(execute.await_count, 3)
        self.assertEqual(len(gateway.planning_messages), 2)
        self.assertEqual(gateway.answer_messages, [])
        self.assertEqual(result.outcome, "no_match")
        self.assertEqual(read_trace(result.trace_id, db_path=self.db_path)[-1]["outcome"], "rejected_limit")

    async def test_planning_rounds_are_bounded(self):
        def selector(messages, tools):
            count = len(json.loads(messages[-1]["content"]).get("current_tool_results", []))
            return (order_call(f"round-{count}", f"DEMO-ORD-{1002 + count}"),)

        gateway = RecordingPlannerGateway(selector)
        async def evidence(call, arguments, identity, db_path):
            order_id = arguments["order_id"]
            return {"outcome": "allow", "trace_id": identity.trace_id,
                    "rows": [owned_order_row(order_id)], "source_ids": [order_id]}

        with patch(
            "customer_assistant.orchestrator._execute_selected_tool", side_effect=evidence,
        ) as execute:
            result = await self.ask(gateway)
        self.assertEqual(len(gateway.planning_messages), MAX_PLANNING_ROUNDS)
        self.assertEqual(execute.await_count, MAX_PLANNING_ROUNDS)
        self.assertLessEqual(len(gateway.answer_messages), 1)
        self.assertNotIn("DEMO-ORD-1005", result.answer_text)

    async def test_denied_foreign_stale_and_malformed_results_never_reach_replanning(self):
        def mutation(case, identity):
            row = owned_order_row()
            result = {"outcome": "allow", "trace_id": identity.trace_id,
                      "rows": [row], "source_ids": ["DEMO-ORD-1007"]}
            if case == "deny":
                result["outcome"] = "deny"
                row["tracking_reference"] = "foreign-data-marker"
            elif case == "foreign":
                row["customer_id"] = "CUST-1002"
                row["customer_name"] = "foreign-data-marker"
            elif case == "stale_trace":
                result["trace_id"] = "another-trace"
            elif case == "malformed_rows":
                result["rows"] = ["foreign-data-marker"]
            elif case == "missing_fact":
                del row["created_on"]
            elif case == "extra_source":
                result["source_ids"].append("foreign-data-marker")
            return result

        for case in ("deny", "foreign", "stale_trace", "malformed_rows", "missing_fact", "extra_source"):
            with self.subTest(case=case):
                gateway = RecordingPlannerGateway(lambda messages, tools: (order_call(),))
                async def evidence(call, arguments, identity, db_path):
                    return mutation(case, identity)

                with patch("customer_assistant.orchestrator._execute_selected_tool", side_effect=evidence):
                    result = await self.ask(gateway)
                self.assertEqual(result.outcome, "deny" if case == "deny" else "no_match")
                self.assertEqual(result.source_ids, ())
                self.assertEqual(len(gateway.planning_messages), 1)
                self.assertEqual(gateway.answer_messages, [])
                self.assertNotIn("foreign-data-marker", json.dumps(gateway.planning_messages))
                self.assertNotIn("foreign-data-marker", json.dumps(result.as_dict()))
                self.assertNotIn("current_tool_results", gateway.planning_payload(0))

    async def test_approved_projection_strips_internal_fields_before_second_plan(self):
        def selector(messages, tools):
            return () if json.loads(messages[-1]["content"]).get("current_tool_results") else (order_call(),)

        gateway = RecordingPlannerGateway(selector)
        async def evidence(call, arguments, identity, db_path):
            row = {**owned_order_row(), "identity_context": identity.as_dict(),
                   "signature": identity.signature, "signing_secret": TEST_SIGNING_SECRET,
                   "internal_database_path": "internal-path-marker"}
            return {"outcome": "allow", "trace_id": identity.trace_id,
                    "rows": [row], "source_ids": ["DEMO-ORD-1007"],
                    "internal_notes": "internal-result-marker"}

        with patch("customer_assistant.orchestrator._execute_selected_tool", side_effect=evidence):
            result = await self.ask(gateway)
        self.assertEqual(result.outcome, "allow")
        self.assertEqual(len(gateway.planning_messages), 2)
        sent = json.dumps(gateway.planning_messages + gateway.answer_messages)
        for marker in ("identity_context", "signature", TEST_SIGNING_SECRET,
                       "internal-path-marker", "internal-result-marker"):
            self.assertNotIn(marker, sent)
        self.assertIn("in_transit", gateway.planning_messages[1][-1]["content"])

    async def test_native_planner_text_cannot_become_evidence_free_facts(self):
        response = {"choices": [{"message": {
            "content": "DEMO-ORD-1007 was delivered yesterday.", "tool_calls": [],
        }}]}
        completion = AsyncMock(return_value=response)
        gateway = LiteLLMGateway(
            GatewayConfig("openai", "openai/offline-injected-response", "synthetic-test-key", 30),
            completion_fn=completion,
        )
        with patch(
            "customer_assistant.orchestrator._execute_selected_tool", new_callable=AsyncMock,
        ) as execute:
            result = await self.ask(gateway, "Where is DEMO-ORD-1007?")
        execute.assert_not_awaited()
        self.assertEqual(completion.await_count, 1)
        self.assertEqual(result.outcome, "no_match")
        self.assertEqual(result.answer_text, SAFE_TOOL_CLARIFICATION)
        self.assertNotIn("delivered", result.answer_text)
        self.assertEqual(result.source_ids, ())

    async def test_explicit_order_target_cannot_be_replaced_with_another_owned_order(self):
        def selector(messages, tools):
            return () if json.loads(messages[-1]["content"]).get("current_tool_results") else (
                order_call(order_id="DEMO-ORD-1008"),
            )

        gateway = RecordingPlannerGateway(selector)
        async def evidence(call, arguments, identity, db_path):
            return {"outcome": "allow", "trace_id": identity.trace_id,
                    "rows": [owned_order_row("DEMO-ORD-1008")], "source_ids": ["DEMO-ORD-1008"]}

        with patch("customer_assistant.orchestrator._execute_selected_tool", side_effect=evidence):
            result = await self.ask(gateway, "Where is DEMO-ORD-1007?")
        self.assertEqual(result.outcome, "no_match")
        self.assertEqual(result.source_ids, ())
        self.assertEqual(len(gateway.answer_messages), 1)
        self.assertNotIn("1008", result.answer_text)
        self.assertEqual(read_trace(result.trace_id, db_path=self.db_path)[-1]["outcome"], "rejected_target")

    async def test_invented_citation_is_rejected_even_beside_valid_order_facts(self):
        def selector(messages, tools):
            return () if json.loads(messages[-1]["content"]).get("current_tool_results") else (order_call(),)

        gateway = RecordingPlannerGateway(selector)
        gateway._answer = FakeModelGateway(TEST_CONFIG, answer_text=(
            "Order DEMO-ORD-1007 is in transit. [DEMO-ORD-1007]\n"
            "It is scheduled for delivery today. [fake-source]"
        ))
        async def evidence(call, arguments, identity, db_path):
            return {"outcome": "allow", "trace_id": identity.trace_id,
                    "rows": [owned_order_row()], "source_ids": ["DEMO-ORD-1007"]}

        with patch("customer_assistant.orchestrator._execute_selected_tool", side_effect=evidence):
            result = await self.ask(gateway)
        self.assertEqual(result.outcome, "no_match")
        self.assertEqual(result.source_ids, ())
        self.assertNotIn("fake-source", result.answer_text)
        self.assertEqual(len(gateway.answer_messages), 1)
        self.assertEqual(read_trace(result.trace_id, db_path=self.db_path)[-1]["outcome"], "rejected_citation")

    async def test_arbitrary_injected_planner_clarification_cannot_assert_facts(self):
        gateway = RecordingPlannerGateway(
            lambda messages, tools: ToolSelectionResult(
                TEST_CONFIG.provider, TEST_CONFIG.model, (),
                clarification_text="DEMO-ORD-1007 was delivered yesterday.",
            ),
            raw_selection=True,
        )
        with patch(
            "customer_assistant.orchestrator._execute_selected_tool", new_callable=AsyncMock,
        ) as execute:
            result = await self.ask(gateway)
        execute.assert_not_awaited()
        self.assertEqual(result.outcome, "no_match")
        self.assertEqual(result.answer_text, SAFE_TOOL_CLARIFICATION)
        self.assertEqual(gateway.answer_messages, [])

    async def test_equivalent_default_arguments_cannot_repeat_a_completed_read(self):
        def selector(messages, tools):
            payload = json.loads(messages[-1]["content"])
            arguments = {"active_only": False} if payload.get("current_tool_results") else {}
            call_id = "repeat" if payload.get("current_tool_results") else "first"
            return (ToolCall(call_id, "list_customer_orders", arguments),)

        gateway = RecordingPlannerGateway(selector)
        async def evidence(call, arguments, identity, db_path):
            return {"outcome": "allow", "trace_id": identity.trace_id,
                    "rows": [owned_order_row()], "source_ids": ["DEMO-ORD-1007"]}

        with patch(
            "customer_assistant.orchestrator._execute_selected_tool", side_effect=evidence,
        ) as execute:
            result = await self.ask(gateway, "List my orders.")
        self.assertEqual(execute.await_count, 1)
        self.assertEqual(len(gateway.planning_messages), 2)
        self.assertEqual(result.outcome, "no_match")
        self.assertEqual(gateway.answer_messages, [])
        self.assertEqual(read_trace(result.trace_id, db_path=self.db_path)[-1]["outcome"], "rejected_arguments")

    async def test_selection_failure_never_falls_back_to_keyword_execution(self):
        gateway = RecordingPlannerGateway(AsyncMock(side_effect=RuntimeError("provider failed")))
        with patch(
            "customer_assistant.orchestrator._execute_selected_tool", new_callable=AsyncMock,
        ) as execute:
            result = await self.ask(gateway, "Where is DEMO-ORD-1007?")
        execute.assert_not_awaited()
        self.assertEqual(result.outcome, "no_match")
        self.assertEqual(gateway.answer_messages, [])
        self.assertEqual(read_trace(result.trace_id, db_path=self.db_path)[-1]["outcome"], "error")


if __name__ == "__main__":
    unittest.main()
