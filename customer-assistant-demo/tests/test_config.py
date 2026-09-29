"""Step 4 - Check how load_gateway_config() chooses a model route.

- Verify the default OpenAI route and an explicit Anthropic selection.
- Verify process settings override .env values so an operator can change one run.
- Reject mismatched model prefixes and invalid timeouts before any provider call.
- Keep API keys out of config repr output while still passing them to the gateway.
"""

import tempfile
import unittest
from pathlib import Path

from customer_assistant.config import (
    GatewayConfigurationError,
    load_gateway_config,
)


class GatewayConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        # Supplying an explicit environment mapping prevents a developer's
        # real provider keys or settings from affecting these offline tests.
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.env_file = Path(self.temp.name) / ".env"

    def test_default_and_explicit_provider_selection(self) -> None:
        # Defaults let a clean offline checkout run, while an explicit choice
        # proves the notebook can change providers without editing code.
        default = load_gateway_config(env_file=self.env_file, environ={})
        self.assertEqual(default.provider, "openai")
        self.assertEqual(default.model, "openai/gpt-6-sol")
        self.assertEqual(default.timeout_seconds, 30.0)
        self.assertIsNone(default.api_key)

        anthropic = load_gateway_config(
            provider="anthropic", env_file=self.env_file, environ={}
        )
        self.assertEqual(anthropic.provider, "anthropic")
        self.assertEqual(anthropic.model, "anthropic/claude-sonnet-5")
        self.assertIsNone(anthropic.api_key)

    def test_environment_overrides_dotenv_and_explicit_provider_wins(self) -> None:
        # Each source contributes settings, but the live process has priority.
        self.env_file.write_text(
            "DEMO_MODEL_PROVIDER=anthropic\n"
            "DEMO_ANTHROPIC_MODEL=anthropic/old-model\n"
            "ANTHROPIC_API_KEY=dotenv-secret\n"
            "DEMO_MODEL_TIMEOUT_SECONDS=15\n",
            encoding="utf-8",
        )
        config = load_gateway_config(
            env_file=self.env_file,
            environ={
                "DEMO_ANTHROPIC_MODEL": "anthropic/new-model",
                "ANTHROPIC_API_KEY": "process-secret",
                "DEMO_MODEL_TIMEOUT_SECONDS": "42.5",
            },
        )
        self.assertEqual(config.provider, "anthropic")
        self.assertEqual(config.model, "anthropic/new-model")
        self.assertEqual(config.api_key, "process-secret")
        self.assertEqual(config.timeout_seconds, 42.5)
        self.assertNotIn("process-secret", repr(config))

        selected = load_gateway_config(
            provider="openai", env_file=self.env_file,
            environ={"OPENAI_API_KEY": "openai-secret"},
        )
        self.assertEqual(selected.provider, "openai")
        self.assertEqual(selected.model, "openai/gpt-6-sol")
        self.assertEqual(selected.api_key, "openai-secret")
        self.assertNotIn("openai-secret", repr(selected))

    def test_invalid_provider_model_and_timeout_are_rejected(self) -> None:
        # Reject bad routes at configuration time so the gateway never sends
        # an API key to the wrong provider or waits without a finite timeout.
        for setting in ("unknown", "", "openai/secret-key"):
            with self.subTest(provider=setting):
                with self.assertRaises(GatewayConfigurationError):
                    load_gateway_config(
                        provider=setting, env_file=self.env_file, environ={}
                    )

        for model in ("anthropic/claude-sonnet-5", "openai/", ""):
            with self.subTest(model=model):
                with self.assertRaises(GatewayConfigurationError):
                    load_gateway_config(
                        env_file=self.env_file,
                        environ={"DEMO_OPENAI_MODEL": model},
                    )

        for timeout in ("0", "-1", "nan", "inf", "not-a-number"):
            with self.subTest(timeout=timeout):
                with self.assertRaises(GatewayConfigurationError) as raised:
                    load_gateway_config(
                        env_file=self.env_file,
                        environ={"DEMO_MODEL_TIMEOUT_SECONDS": timeout},
                    )
                self.assertNotIn(timeout, str(raised.exception))


if __name__ == "__main__":
    unittest.main()
