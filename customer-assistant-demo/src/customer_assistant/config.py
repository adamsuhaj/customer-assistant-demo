"""Step 4 - resolve the selected model provider and its live-call settings.

- ``load_gateway_config`` reads the optional project ``.env`` and overlays
  process environment values. The caller's provider choice, including the
  Streamlit selector, determines which provider-specific model and key it uses.
- ``GatewayConfig`` carries that provider, qualified model name, timeout, and
  optional API key. The dataclass hides the key from its printed representation.
- Validation rejects an unknown provider, a model routed to another provider,
  or a nonpositive/nonfinite timeout before any live request is made.
- A missing key remains valid for the offline notebook and evaluation gateway;
  the live adapter checks for the selected key immediately before a call.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values


# Resolve the default .env relative to the project, not the process directory.
# The notebook, IDE, and pytest may each start from a different directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# These are provider-qualified LiteLLM names. Separate model and key variables
# let the Streamlit provider selector switch both settings together without
# editing code or relying on the other provider's credential.
DEFAULT_MODELS = {
    "openai": "openai/gpt-6-sol",
    "anthropic": "anthropic/claude-sonnet-5",
}
API_KEY_VARIABLES = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}
MODEL_VARIABLES = {
    "openai": "DEMO_OPENAI_MODEL",
    "anthropic": "DEMO_ANTHROPIC_MODEL",
}
DEFAULT_TIMEOUT_SECONDS = 30.0


class GatewayConfigurationError(ValueError):
    """A non-secret setting is invalid for the selected model provider."""


@dataclass(frozen=True)
class GatewayConfig:
    """Validated provider settings shared by live and offline gateway paths."""

    provider: str
    model: str
    # The live adapter needs this value to call its provider. Hiding it from
    # repr prevents accidental disclosure when the config is printed or logged.
    api_key: str | None = field(repr=False)
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS


def load_gateway_config(
    *,
    provider: str | None = None,
    env_file: Path = PROJECT_ROOT / ".env",
    environ: Mapping[str, str] | None = None,
) -> GatewayConfig:
    """Resolve one provider configuration without making a network request."""

    # Reading .env without mutating os.environ prevents a notebook run from
    # changing unrelated processes or later tests. An explicit merge also
    # documents the precedence: process settings win over .env values.
    settings = {key: value for key, value in dotenv_values(env_file).items()
                if value is not None}
    settings.update(os.environ if environ is None else environ)

    # A caller may choose a provider for one run (the swap demonstration);
    # otherwise DEMO_MODEL_PROVIDER controls the normal notebook default.
    raw_provider = provider if provider is not None else settings.get(
        "DEMO_MODEL_PROVIDER", "openai"
    )
    selected_provider = raw_provider.strip().lower()
    if selected_provider not in DEFAULT_MODELS:
        raise GatewayConfigurationError(
            "DEMO_MODEL_PROVIDER must be 'openai' or 'anthropic'."
        )

    model_variable = MODEL_VARIABLES[selected_provider]
    model = settings.get(model_variable, DEFAULT_MODELS[selected_provider]).strip()
    # LiteLLM expects a provider-qualified name. Enforcing its prefix keeps
    # the chosen credential and model route together; a typo must fail here
    # instead of silently sending a request through another provider.
    if not model.startswith(f"{selected_provider}/") or not model.split("/", 1)[1]:
        raise GatewayConfigurationError(
            f"{model_variable} must name a {selected_provider}/... model."
        )

    # Reject zero, negative, infinity, and NaN values now so a later live
    # call cannot hang or behave differently across provider SDKs.
    raw_timeout = settings.get(
        "DEMO_MODEL_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)
    )
    try:
        timeout_seconds = float(raw_timeout)
    except (TypeError, ValueError):
        raise GatewayConfigurationError(
            "DEMO_MODEL_TIMEOUT_SECONDS must be a positive finite number."
        ) from None
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise GatewayConfigurationError(
            "DEMO_MODEL_TIMEOUT_SECONDS must be a positive finite number."
        )

    # Empty .env template values are not credentials. Represent them as None:
    # the offline fake still works, and the live adapter gives a clear error
    # at the point where a real provider call actually needs a key.
    api_key = settings.get(API_KEY_VARIABLES[selected_provider], "").strip() or None
    return GatewayConfig(
        provider=selected_provider,
        model=model,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    )
