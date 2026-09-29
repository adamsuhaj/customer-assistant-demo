"""Exercise the native planner adapter with injected responses, never live calls."""

import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from customer_assistant.config import GatewayConfig, GatewayConfigurationError
from customer_assistant.gateway import (
    FakeModelGateway,
    GatewayCallError,
    LiteLLMGateway,
    SAFE_TOOL_CLARIFICATION,
    ToolCall,
    ToolSelectionResult,
)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Read status for one authorized order.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_customer_orders",
            "description": "Read orders for the application-authenticated customer.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]
MESSAGES = [{"role": "user", "content": "Where is DEMO-ORD-1007?"}]


def native_call(
    *, call_id="call_1", name="get_order_status", arguments='{"order_id":"DEMO-ORD-1007"}',
):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def native_response(calls=None, content=None, usage=None):
    return {
        "choices": [{"message": {"tool_calls": calls, "content": content}}],
        "usage": usage,
    }


class GatewayToolSelectionTests(unittest.IsolatedAsyncioTestCase):
    def live_gateway(self, response, *, key="synthetic-never-send-key"):
        completion = AsyncMock(return_value=response)
        config = GatewayConfig("openai", "openai/gpt-6-sol", key, 12.0)
        return LiteLLMGateway(config, completion_fn=completion), completion

    async def test_native_tool_calls_and_selected_key_for_both_providers(self):
        for provider, model in (
            ("openai", "openai/gpt-6-sol"),
            ("anthropic", "anthropic/claude-sonnet-5"),
        ):
            with self.subTest(provider=provider):
                completion = AsyncMock(return_value=native_response(
                    [native_call()],
                    usage={"prompt_tokens": 11, "completion_tokens": 9, "total_tokens": 20},
                ))
                config = GatewayConfig(provider, model, "synthetic-never-send-key", 12.0)
                result = await LiteLLMGateway(
                    config, completion_fn=completion,
                ).select_tools(MESSAGES, TOOLS)
                completion.assert_awaited_once_with(
                    model=model, messages=MESSAGES, tools=TOOLS, tool_choice="auto",
                    api_key="synthetic-never-send-key", timeout=12.0,
                )
                self.assertEqual(result.tool_calls, (
                    ToolCall("call_1", "get_order_status", {"order_id": "DEMO-ORD-1007"}),
                ))
                self.assertEqual(result.provider, provider)
                self.assertEqual(result.model, model)
                self.assertIsNone(result.clarification_text)
                self.assertEqual(result.token_usage["total_tokens"], 20)
                self.assertNotIn(config.api_key, json.dumps(result.as_dict()))

    async def test_sdk_attribute_objects_are_normalized(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=None,
                tool_calls=[SimpleNamespace(
                    id="toolu_01", type="function", function=SimpleNamespace(
                        name="get_order_status", arguments='{"order_id":"DEMO-ORD-1007"}',
                    ),
                )],
            ))],
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=None, total_tokens=-1),
        )
        gateway, _ = self.live_gateway(response)
        result = await gateway.select_tools(MESSAGES, TOOLS)
        self.assertEqual(result.tool_calls[0].id, "toolu_01")
        self.assertEqual(result.token_usage, {"prompt_tokens": 7})

    async def test_multiple_calls_and_result_copy_do_not_mutate_arguments(self):
        gateway, _ = self.live_gateway(native_response([
            native_call(), native_call(call_id="call_2", name="list_customer_orders", arguments="{}"),
        ]))
        result = await gateway.select_tools(MESSAGES, TOOLS)
        self.assertEqual(len(result.tool_calls), 2)
        copied = result.as_dict()
        copied["tool_calls"][0]["arguments"]["order_id"] = "changed"
        self.assertEqual(result.tool_calls[0].arguments["order_id"], "DEMO-ORD-1007")

    async def test_no_calls_discards_unsupported_provider_facts(self):
        gateway, _ = self.live_gateway(native_response(
            [], content="DEMO-ORD-1007 was delivered yesterday.",
        ))
        result = await gateway.select_tools(MESSAGES, TOOLS)
        self.assertEqual(result.tool_calls, ())
        self.assertEqual(result.clarification_text, SAFE_TOOL_CLARIFICATION)
        self.assertNotIn("delivered", json.dumps(result.as_dict()))

    async def test_empty_choice_or_malformed_call_list_fails_closed(self):
        for response in (
            {"choices": []},
            {"choices": [{"message": None}]},
            native_response({"name": "get_order_status"}),
        ):
            with self.subTest(response=response):
                gateway, _ = self.live_gateway(response)
                with self.assertRaises(GatewayCallError):
                    await gateway.select_tools(MESSAGES, TOOLS)

    async def test_malformed_json_nonobjects_duplicate_keys_and_nan_rejected(self):
        for arguments in (
            "{", "[]", "null", '"text"',
            '{"order_id":"DEMO-ORD-1007","order_id":"DEMO-ORD-1008"}',
            '{"order_id":NaN}',
            {"order_id": "DEMO-ORD-1007"},
        ):
            with self.subTest(arguments=arguments):
                gateway, _ = self.live_gateway(native_response([
                    native_call(arguments=arguments),
                ]))
                with self.assertRaises(GatewayCallError):
                    await gateway.select_tools(MESSAGES, TOOLS)

    async def test_invalid_fields_and_unknown_tool_rejected(self):
        calls = [
            native_call(call_id=""), native_call(call_id="call id"),
            native_call(call_id=4), native_call(name="unreviewed_admin_tool"),
            native_call(name="../get_order_status"),
            {**native_call(), "type": "shell"},
        ]
        for call in calls:
            with self.subTest(call=call):
                gateway, _ = self.live_gateway(native_response([call]))
                with self.assertRaises(GatewayCallError):
                    await gateway.select_tools(MESSAGES, TOOLS)

    async def test_reused_tool_call_id_rejected(self):
        gateway, _ = self.live_gateway(native_response([native_call(), native_call()]))
        with self.assertRaises(GatewayCallError):
            await gateway.select_tools(MESSAGES, TOOLS)

    async def test_model_cannot_supply_signed_identity_arguments(self):
        gateway, _ = self.live_gateway(native_response([native_call(arguments=json.dumps({
            "order_id": "DEMO-ORD-1007", "identity_context": {"user_id": "CUST-ALICE"},
        }))]))
        with self.assertRaises(GatewayCallError):
            await gateway.select_tools(MESSAGES, TOOLS)

    async def test_key_echoes_are_blocked_in_text_id_and_arguments(self):
        key = "synthetic-never-send-key"
        responses = [
            native_response([], content=f"echoed {key}"),
            native_response([native_call(call_id=key)]),
            native_response([native_call(arguments=json.dumps({"order_id": key}))]),
        ]
        for response in responses:
            with self.subTest(response=response):
                gateway, _ = self.live_gateway(response, key=key)
                with self.assertRaises(GatewayCallError) as caught:
                    await gateway.select_tools(MESSAGES, TOOLS)
                self.assertNotIn(key, str(caught.exception))

    async def test_json_escaped_key_echo_is_also_blocked(self):
        key = 'synthetic-"escaped-key'
        gateway, _ = self.live_gateway(native_response([], content=key), key=key)
        with self.assertRaises(GatewayCallError) as caught:
            await gateway.select_tools(MESSAGES, TOOLS)
        self.assertNotIn(key, str(caught.exception))

    async def test_credential_in_request_is_not_sent(self):
        gateway, completion = self.live_gateway(native_response([native_call()]))
        with self.assertRaises(GatewayCallError):
            await gateway.select_tools(
                [{"role": "user", "content": gateway.config.api_key}], TOOLS,
            )
        completion.assert_not_awaited()

    async def test_catalog_cannot_expose_identity_and_is_not_sent(self):
        bad_tools = json.loads(json.dumps(TOOLS))
        bad_tools[0]["function"]["parameters"]["properties"]["identity_context"] = {"type": "object"}
        gateway, completion = self.live_gateway(native_response([native_call()]))
        with self.assertRaises(GatewayCallError):
            await gateway.select_tools(MESSAGES, bad_tools)
        completion.assert_not_awaited()

    async def test_provider_exception_hides_credentials_and_request_details(self):
        gateway, completion = self.live_gateway(None)
        completion.side_effect = RuntimeError(f"prompt secret {gateway.config.api_key}")
        with self.assertRaises(GatewayCallError) as caught:
            await gateway.select_tools(MESSAGES, TOOLS)
        self.assertNotIn(gateway.config.api_key, str(caught.exception))
        self.assertNotIn("prompt secret", str(caught.exception))
        self.assertIn("RuntimeError", str(caught.exception))

    async def test_missing_key_stops_before_live_call(self):
        completion = AsyncMock()
        gateway = LiteLLMGateway(
            GatewayConfig("anthropic", "anthropic/claude-sonnet-5", None, 30),
            completion_fn=completion,
        )
        with self.assertRaises(GatewayConfigurationError):
            await gateway.select_tools(MESSAGES, TOOLS)
        completion.assert_not_awaited()

    async def test_native_adapter_copies_nested_input(self):
        async def mutate(**kwargs):
            kwargs["messages"][0]["content"] = "changed"
            kwargs["tools"][0]["function"]["parameters"]["properties"].clear()
            return native_response([native_call()])

        config = GatewayConfig("openai", "openai/gpt-6-sol", "synthetic-key", 30)
        messages = json.loads(json.dumps(MESSAGES))
        tools = json.loads(json.dumps(TOOLS))
        await LiteLLMGateway(config, completion_fn=mutate).select_tools(messages, tools)
        self.assertEqual(messages, MESSAGES)
        self.assertEqual(tools, TOOLS)

    async def test_offline_selector_supports_sync_async_and_result_contract(self):
        config = GatewayConfig("openai", "openai/gpt-6-sol", None, 30)
        call = ToolCall("offline_1", "get_order_status", {"order_id": "DEMO-ORD-1007"})
        selectors = (
            lambda messages, tools: [call],
            AsyncMock(return_value=(call,)),
            lambda messages, tools: ToolSelectionResult("openai", config.model, (call,)),
        )
        for selector in selectors:
            with self.subTest(selector=selector):
                result = await FakeModelGateway(config, tool_selector=selector).select_tools(
                    MESSAGES, TOOLS,
                )
                self.assertEqual(result.tool_calls, (call,))
                self.assertIsNone(result.token_usage)

    async def test_offline_default_is_lazy_and_does_not_import_litellm(self):
        config = GatewayConfig("openai", "openai/gpt-6-sol", None, 30)
        call = ToolCall("offline_1", "list_customer_orders", {})
        module = SimpleNamespace(select_tools=lambda messages, tools: [call])
        with patch.dict("sys.modules", {"customer_assistant.offline_tool_selector": module}):
            result = await FakeModelGateway(config).select_tools(MESSAGES, TOOLS)
        self.assertEqual(result.tool_calls, (call,))

    async def test_offline_trusted_clarification_is_preserved_for_both_routes(self):
        for provider, model in (
            ("openai", "openai/gpt-6-sol"),
            ("anthropic", "anthropic/claude-sonnet-5"),
        ):
            with self.subTest(provider=provider):
                config = GatewayConfig(provider, model, None, 30)
                clarification = "Please include the DEMO-ORD order ID."
                result = await FakeModelGateway(
                    config,
                    tool_selector=lambda messages, tools: ToolSelectionResult(
                        "ignored-provider", "ignored-model", (), clarification_text=clarification,
                    ),
                ).select_tools(MESSAGES, TOOLS)
                self.assertEqual(result.clarification_text, clarification)
                self.assertEqual(result.provider, provider)
                self.assertEqual(result.model, model)
                self.assertEqual(result.tool_calls, ())

    async def test_offline_malformed_or_credential_output_is_suppressed(self):
        key = "synthetic-key"
        config = GatewayConfig("openai", "openai/gpt-6-sol", key, 30)
        selectors = (
            lambda messages, tools: {"raw": key},
            lambda messages, tools: [ToolCall("offline_1", "get_order_status", {"order_id": key})],
            lambda messages, tools: ToolSelectionResult(
                "openai", config.model, (), clarification_text=key,
            ),
        )
        for selector in selectors:
            with self.subTest(selector=selector):
                with self.assertRaises(GatewayCallError) as caught:
                    await FakeModelGateway(config, tool_selector=selector).select_tools(MESSAGES, TOOLS)
                self.assertNotIn(key, str(caught.exception))


if __name__ == "__main__":
    unittest.main()
