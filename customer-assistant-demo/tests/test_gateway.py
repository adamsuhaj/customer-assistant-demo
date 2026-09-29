"""Step 4 - Check both model gateway implementations without network calls.

- Use FakeModelGateway to rehearse OpenAI and Anthropic routes offline.
- Inject a completion test double into LiteLLMGateway to inspect call arguments.
- Verify both paths return the same normalized answer and token-usage fields.
- Verify missing keys, provider errors, and repr output do not expose secrets.
"""

import json
import unittest
from unittest.mock import AsyncMock

from customer_assistant.config import GatewayConfig, GatewayConfigurationError
from customer_assistant.gateway import (
    FakeModelGateway,
    GatewayCallError,
    LiteLLMGateway,
)


class GatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_offline_fake_uses_one_result_shape_for_both_providers(self) -> None:
        messages = [{"role": "user", "content": "Synthetic test question"}]
        for provider, model in (
            ("openai", "openai/gpt-6-sol"),
            ("anthropic", "anthropic/claude-sonnet-5"),
        ):
            with self.subTest(provider=provider):
                # No key is present: the fake must support a complete offline
                # provider-switch rehearsal with no LiteLLM import or HTTP.
                config = GatewayConfig(provider, model, None, 30.0)
                result = await FakeModelGateway(
                    config, answer_text="Synthetic answer"
                ).complete(messages)
                self.assertEqual(result.as_dict(), {
                    "provider": provider,
                    "model": model,
                    "answer_text": "Synthetic answer",
                    "token_usage": None,
                })

    async def test_live_adapter_uses_selected_key_and_normalizes_usage(self) -> None:
        test_key = "synthetic-test-key-never-send"
        config = GatewayConfig("openai", "openai/gpt-6-sol", test_key, 12.0)
        fake_completion = AsyncMock(return_value={
            "choices": [{"message": {"content": "Order is in transit."}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11},
        })
        messages = [{"role": "user", "content": "Synthetic order question"}]

        # Inject a completion response into the real adapter. This checks its
        # vendor call arguments without making a chargeable external request.
        result = await LiteLLMGateway(
            config, completion_fn=fake_completion
        ).complete(messages)
        fake_completion.assert_awaited_once_with(
            model="openai/gpt-6-sol",
            messages=messages,
            api_key=test_key,
            timeout=12.0,
        )
        self.assertEqual(result.provider, "openai")
        self.assertEqual(result.answer_text, "Order is in transit.")
        self.assertEqual(result.token_usage, {
            "prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11,
        })
        self.assertNotIn(test_key, json.dumps(result.as_dict()))
        self.assertNotIn(test_key, repr(config))

    async def test_missing_key_stops_before_call_and_null_usage_is_omitted(self) -> None:
        config = GatewayConfig("anthropic", "anthropic/claude-sonnet-5", None, 30.0)
        fake_completion = AsyncMock()
        with self.assertRaises(GatewayConfigurationError):
            await LiteLLMGateway(
                config, completion_fn=fake_completion
            ).complete([{"role": "user", "content": "Synthetic question"}])
        fake_completion.assert_not_awaited()

        # LiteLLM's own mock mode can return null token values. The gateway
        # reports usage only when the provider actually supplies counts.
        keyed = GatewayConfig(config.provider, config.model, "test-key", 30.0)
        fake_completion.return_value = {
            "choices": [{"message": {"content": "Synthetic reply"}}],
            "usage": {"prompt_tokens": None, "completion_tokens": None},
        }
        result = await LiteLLMGateway(
            keyed, completion_fn=fake_completion
        ).complete([{"role": "user", "content": "Synthetic question"}])
        self.assertIsNone(result.token_usage)

    async def test_provider_error_does_not_surface_secret_or_raw_message(self) -> None:
        secret = "synthetic-secret-for-error-test"
        config = GatewayConfig("openai", "openai/gpt-6-sol", secret, 30.0)
        fake_completion = AsyncMock(side_effect=RuntimeError(f"provider echoed {secret}"))
        with self.assertRaises(GatewayCallError) as caught:
            await LiteLLMGateway(
                config, completion_fn=fake_completion
            ).complete([{"role": "user", "content": "Synthetic question"}])
        self.assertNotIn(secret, str(caught.exception))
        self.assertNotIn("provider echoed", str(caught.exception))

        # A credential echoed in otherwise valid answer text must also be
        # stopped before it can reach the normalized/audit-ready result.
        echoed_answer = AsyncMock(return_value={
            "choices": [{"message": {"content": f"Echoed {secret}"}}],
        })
        with self.assertRaises(GatewayCallError) as echoed:
            await LiteLLMGateway(
                config, completion_fn=echoed_answer
            ).complete([{"role": "user", "content": "Synthetic question"}])
        self.assertNotIn(secret, str(echoed.exception))


if __name__ == "__main__":
    unittest.main()
