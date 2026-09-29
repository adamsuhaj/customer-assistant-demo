"""Step 4 - provider-neutral planning and grounded answer interfaces.

- ``ModelGateway.select_tools`` receives the reviewed MCP catalog and proposes
  calls. The orchestrator validates arguments and supplies signed identity;
  the provider adapter never executes a tool or grants permission.
- ``ModelGateway.complete`` still generates an answer only after authorized,
  projected MCP evidence is available. Existing answer callers use this seam.
- ``LiteLLMGateway`` calls only the selected OpenAI or Anthropic model with its
  configured key. ``FakeModelGateway`` gives notebook and evaluation runs a
  deterministic answer without a provider request.
- ``GatewayResult`` and ``_normalize_response`` expose answer text and valid
  token counts in one provider-neutral shape; raw SDK objects are not returned.
- Empty responses and a response echoing the selected API key are rejected.
  Provider exception details are suppressed because they can contain prompt
  content or credentials; the caller audits a safe outcome instead.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import inspect
import json
import re
from typing import Any, Protocol

from .config import GatewayConfig, GatewayConfigurationError


class GatewayCallError(RuntimeError):
    """A model call failed; its message omits raw provider request details."""


@dataclass(frozen=True)
class GatewayResult:
    """Provider-neutral answer and usage fields safe for the caller to inspect."""

    provider: str
    model: str
    answer_text: str
    token_usage: dict[str, int] | None

    def as_dict(self) -> dict[str, Any]:
        """Copy normalized fields for display or audit without exposing a key."""

        return {
            "provider": self.provider,
            "model": self.model,
            "answer_text": self.answer_text,
            "token_usage": dict(self.token_usage) if self.token_usage is not None else None,
        }


@dataclass(frozen=True)
class ToolCall:
    """A model-proposed operation, carrying no application identity or authority."""

    id: str
    name: str
    arguments: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": deepcopy(self.arguments)}


@dataclass(frozen=True)
class ToolSelectionResult:
    """Normalized planning output; free-form provider facts are never returned."""

    provider: str
    model: str
    tool_calls: tuple[ToolCall, ...]
    clarification_text: str | None = None
    token_usage: dict[str, int] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "tool_calls": [call.as_dict() for call in self.tool_calls],
            "clarification_text": self.clarification_text,
            "token_usage": dict(self.token_usage) if self.token_usage is not None else None,
        }


SAFE_TOOL_CLARIFICATION = (
    "Please clarify whether you need order status, your order list, service history, "
    "or troubleshooting. Include an order or instrument ID when applicable."
)
_FUNCTION_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,63}\Z")
_CALL_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_APPLICATION_ARGUMENTS = frozenset({
    "identity_context", "user_id", "agent_id", "trace_id", "signature", "expires_at",
})


class ModelGateway(Protocol):
    """Contract used for controlled tool selection and grounded answer generation."""

    config: GatewayConfig

    async def complete(self, messages: Sequence[Mapping[str, Any]]) -> GatewayResult:
        ...

    async def select_tools(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> ToolSelectionResult:
        ...


def _field(value: Any, name: str) -> Any:
    # LiteLLM response objects often expose attributes; injected test doubles
    # and serialized responses may use dictionaries. Reading both forms here
    # keeps response validation identical in live and offline adapter tests.
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _normalized_usage(response: Any) -> dict[str, int] | None:
    raw_usage = _field(response, "usage")
    usage = {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = _field(raw_usage, field)
        if type(value) is int and value >= 0:
            usage[field] = value
    return usage or None


def _protected_text(config: GatewayConfig, value: Any) -> None:
    """Check JSON-safe model-facing/output data without echoing it in errors."""

    try:
        serialized = json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, OverflowError):
        raise GatewayCallError("Tool selection contained invalid data") from None
    if config.api_key:
        encoded_key = json.dumps(config.api_key, ensure_ascii=False)[1:-1]
        if config.api_key in serialized or encoded_key in serialized:
            raise GatewayCallError("Tool selection contained protected credential")


def _validated_tools(
    config: GatewayConfig, tools: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], frozenset[str]]:
    """Accept only function catalogs with public, model-supplied parameters."""

    if not isinstance(tools, (list, tuple)) or not tools:
        raise GatewayCallError("No approved tools were supplied")
    _protected_text(config, tools)
    copied = deepcopy(list(tools))
    names: set[str] = set()
    for tool in copied:
        function = _field(tool, "function")
        name = _field(function, "name")
        parameters = _field(function, "parameters")
        if (
            not isinstance(tool, Mapping)
            or tool.get("type") != "function"
            or not isinstance(function, Mapping)
            or not isinstance(name, str)
            or not _FUNCTION_NAME.fullmatch(name)
            or name in names
            or not isinstance(parameters, Mapping)
            or parameters.get("type") != "object"
        ):
            raise GatewayCallError("Approved tool definition is invalid")
        properties = parameters.get("properties", {})
        required = parameters.get("required", [])
        if (
            not isinstance(properties, Mapping)
            or not isinstance(required, (list, tuple))
            or any(not isinstance(item, str) for item in required)
            or _APPLICATION_ARGUMENTS.intersection(properties)
            or _APPLICATION_ARGUMENTS.intersection(required)
        ):
            # Signed context is attached after validation in trusted code.
            # Advertising it would invite the model to invent its authority.
            raise GatewayCallError("Tool catalog exposes application identity")
        names.add(name)
    return copied, frozenset(names)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate key")
        result[key] = value
    return result


def _invalid_json_constant(value: str) -> None:
    raise ValueError("Nonfinite JSON value")


def _normalize_tool_selection(
    config: GatewayConfig, response: Any, allowed_names: frozenset[str],
) -> ToolSelectionResult:
    choices = _field(response, "choices")
    if not isinstance(choices, (list, tuple)) or not choices:
        raise GatewayCallError("Model returned no tool selection choice")
    message = _field(choices[0], "message")
    if message is None:
        raise GatewayCallError("Model returned no tool selection message")
    content = _field(message, "content")
    if content is not None:
        _protected_text(config, content)
    raw_calls = _field(message, "tool_calls")
    if raw_calls is not None and not isinstance(raw_calls, (list, tuple)):
        raise GatewayCallError("Model returned malformed tool calls")
    calls = []
    seen_ids = set()
    for raw_call in raw_calls or ():
        call_id = _field(raw_call, "id")
        function = _field(raw_call, "function")
        name = _field(function, "name")
        arguments_json = _field(function, "arguments")
        if (
            _field(raw_call, "type") != "function"
            or not isinstance(call_id, str)
            or not _CALL_ID.fullmatch(call_id)
            or call_id in seen_ids
            or not isinstance(name, str)
            or not _FUNCTION_NAME.fullmatch(name)
            or name not in allowed_names
            or not isinstance(arguments_json, str)
        ):
            raise GatewayCallError("Model returned invalid tool call fields")
        _protected_text(config, [call_id, name, arguments_json])
        try:
            arguments = json.loads(
                arguments_json,
                object_pairs_hook=_unique_json_object,
                parse_constant=_invalid_json_constant,
            )
        except (ValueError, TypeError, RecursionError):
            raise GatewayCallError("Model returned invalid tool arguments JSON") from None
        if not isinstance(arguments, dict):
            raise GatewayCallError("Model tool arguments must be an object")
        if _APPLICATION_ARGUMENTS.intersection(arguments):
            raise GatewayCallError("Model attempted to supply application identity")
        _protected_text(config, arguments)
        calls.append(ToolCall(call_id, name, arguments))
        seen_ids.add(call_id)
    # Planner text may invent order facts before any authorized read. Return
    # an application-owned clarification instead; the separate answer stage
    # is responsible for checked evidence. Existing evidence lets the caller
    # treat a no-call result as the end of planning and synthesize an answer.
    return ToolSelectionResult(
        provider=config.provider,
        model=config.model,
        tool_calls=tuple(calls),
        clarification_text=None if calls else SAFE_TOOL_CLARIFICATION,
        token_usage=_normalized_usage(response),
    )


def _normalize_response(config: GatewayConfig, response: Any) -> GatewayResult:
    # A missing text choice cannot support the demo's evidence-based answer.
    # Reject it instead of letting the caller treat a partial SDK result as
    # a successful model reply.
    choices = _field(response, "choices")
    if not isinstance(choices, (list, tuple)) or not choices:
        raise GatewayCallError("Model returned no answer choice")
    answer = _field(_field(choices[0], "message"), "content")
    if not isinstance(answer, str) or not answer.strip():
        # The orchestrator executes MCP tools before this call, then requests
        # a text answer. A tool-call-only or empty model response is unusable.
        raise GatewayCallError("Model returned no answer text")
    if config.api_key and config.api_key in answer:
        # A model can echo text from its input. Prevent a mistakenly included
        # API key from reaching notebook output, AssistantResult, or audit rows.
        raise GatewayCallError("Model returned protected credential")

    return GatewayResult(
        provider=config.provider,
        model=config.model,
        answer_text=answer,
        token_usage=_normalized_usage(response),
    )


class LiteLLMGateway:
    """Send one live completion through LiteLLM and normalize its response."""

    def __init__(
        self,
        config: GatewayConfig,
        *,
        completion_fn: Callable[..., Awaitable[Any]] | None = None,
    ) -> None:
        self.config = config
        # Tests can inject a completion function to exercise this adapter's
        # key handling and response parsing without an external model call.
        self._completion_fn = completion_fn

    async def complete(self, messages: Sequence[Mapping[str, Any]]) -> GatewayResult:
        # Config loading permits a missing key for offline work. A live call
        # must stop here before importing LiteLLM or attempting network I/O.
        if not self.config.api_key:
            key_name = f"{self.config.provider.upper()}_API_KEY"
            raise GatewayConfigurationError(f"Set {key_name} for live model calls")

        try:
            completion_fn = self._completion_fn
            if completion_fn is None:
                # Import only on the live path. Config inspection, tests using
                # an injected completion function, and the fake need no SDK.
                from litellm import acompletion

                completion_fn = acompletion
            # Pass only the selected provider's key explicitly. A Streamlit
            # provider change builds a new config, so this request uses that
            # provider's model/key pair. Copy message mappings so downstream
            # adapters cannot mutate caller-owned data.
            response = await completion_fn(
                model=self.config.model,
                messages=[dict(message) for message in messages],
                api_key=self.config.api_key,
                timeout=self.config.timeout_seconds,
            )
        except Exception as exc:
            # Provider errors may embed prompts, request IDs, or credentials.
            # Retain only the exception type in the user/audit-facing message.
            raise GatewayCallError(
                f"{self.config.provider} model call failed ({type(exc).__name__})"
            ) from None
        return _normalize_response(self.config, response)

    async def select_tools(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> ToolSelectionResult:
        """Request native tool calls through either supported provider adapter."""

        if not self.config.api_key:
            key_name = f"{self.config.provider.upper()}_API_KEY"
            raise GatewayConfigurationError(f"Set {key_name} for live model calls")
        copied_tools, names = _validated_tools(self.config, tools)
        _protected_text(self.config, messages)
        try:
            completion_fn = self._completion_fn
            if completion_fn is None:
                from litellm import acompletion

                completion_fn = acompletion
            response = await completion_fn(
                model=self.config.model,
                messages=deepcopy(list(messages)),
                tools=copied_tools,
                tool_choice="auto",
                api_key=self.config.api_key,
                timeout=self.config.timeout_seconds,
            )
        except Exception as exc:
            raise GatewayCallError(
                f"{self.config.provider} tool selection failed ({type(exc).__name__})"
            ) from None
        return _normalize_tool_selection(self.config, response, names)


class FakeModelGateway:
    """Answer through the gateway contract without SDK import or network I/O."""

    def __init__(
        self,
        config: GatewayConfig,
        *,
        answer_text: str = "Synthetic test answer.",
        answer_builder: Callable[[Sequence[Mapping[str, Any]]], str] | None = None,
        tool_selector: Callable[
            [Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]], Any
        ] | None = None,
    ):
        self.config = config
        if not answer_text.strip():
            raise ValueError("Fake answer must contain text")
        self.answer_text = answer_text
        # The orchestrator supplies an evidence-aware builder for the offline
        # notebook. A fixed default makes lower-level tests deterministic.
        self._answer_builder = answer_builder
        self._tool_selector = tool_selector

    async def complete(self, messages: Sequence[Mapping[str, Any]]) -> GatewayResult:
        # This path uses only local input and the optional builder. It can
        # demonstrate provider switching without claiming a live model call.
        answer_text = (
            self._answer_builder(messages)
            if self._answer_builder is not None else self.answer_text
        )
        if not isinstance(answer_text, str) or not answer_text.strip():
            raise GatewayCallError("Offline fake returned no answer text")
        if self.config.api_key and self.config.api_key in answer_text:
            raise GatewayCallError("Offline fake returned protected credential")
        return GatewayResult(
            provider=self.config.provider,
            model=self.config.model,
            answer_text=answer_text,
            token_usage=None,
        )

    async def select_tools(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> ToolSelectionResult:
        """Use an explicitly offline selector through the same validation seam."""

        copied_tools, names = _validated_tools(self.config, tools)
        _protected_text(self.config, messages)
        try:
            selector = self._tool_selector
            if selector is None:
                from .offline_tool_selector import select_tools

                selector = select_tools
            selected = selector(deepcopy(list(messages)), copied_tools)
            if inspect.isawaitable(selected):
                selected = await selected
            if isinstance(selected, ToolSelectionResult) and selected.clarification_text is not None:
                if (
                    not isinstance(selected.clarification_text, str)
                    or not selected.clarification_text.strip()
                ):
                    raise ValueError("Offline selector returned invalid clarification")
            calls = selected.tool_calls if isinstance(selected, ToolSelectionResult) else selected
            if not isinstance(calls, (list, tuple)) or any(
                not isinstance(call, ToolCall) for call in calls
            ):
                raise ValueError("Offline selector returned invalid tool calls")
            response = {
                "choices": [{"message": {
                    "content": (
                        selected.clarification_text
                        if isinstance(selected, ToolSelectionResult) else None
                    ),
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments, allow_nan=False),
                            },
                        } for call in calls
                    ],
                }}],
                "usage": selected.token_usage if isinstance(selected, ToolSelectionResult) else None,
            }
        except Exception as exc:
            raise GatewayCallError(
                f"Offline tool selection failed ({type(exc).__name__})"
            ) from None
        result = _normalize_tool_selection(self.config, response, names)
        if (
            not result.tool_calls
            and isinstance(selected, ToolSelectionResult)
            and selected.clarification_text is not None
        ):
            # The deterministic local selector supplies application-authored
            # clarification, preserving existing demo messages. Live model
            # text is still discarded by _normalize_tool_selection above.
            return ToolSelectionResult(
                result.provider, result.model, result.tool_calls,
                selected.clarification_text, result.token_usage,
            )
        return result
