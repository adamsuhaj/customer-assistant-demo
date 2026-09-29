"""Step 3 - Call the separate MCP server from trusted application code.

- Start mcp_server.py as a Python stdio subprocess for tool discovery and
  calls. The orchestrator uses this client instead of importing tool
  functions directly, so the real MCP boundary is exercised in the demo.
- Pass the chosen SQLite path and signing secret through the trusted process
  environment. Neither is taken from a model-written tool argument, and the
  secret is not placed in the evidence returned to the model.
- Send an application-minted signed context to private tools: one order,
  Step 9 all-orders, and service history. Step 11 made an omitted instrument
  ID mean all history for the signed customer, with no customer-ID parameter.
- Call the public troubleshooting tool without an identity. Step 11 made
  omitted search terms mean the approved tier 0 article catalog; targeted
  calls still supply model and symptom.
- Accept only the reviewed tool names and a structured dictionary response.
  The orchestrator needs explicit outcome, rows, and source IDs for routing,
  audit, and grounded answers.
- Discover model-facing schemas over MCP, removing application-owned identity
  inputs. Validate the model's choice before trusted code adds signed context;
  a changed catalog needs review before it can expand the tool boundary.
"""

from __future__ import annotations

import copy
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping

from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from .database import DEFAULT_DB_PATH, PROJECT_ROOT
from .identity import DemoIdentity


TOOL_NAMES = frozenset(
    {
        "get_order_status",
        "list_customer_orders",
        "get_service_history",
        "search_troubleshooting",
    }
)

# Discovery supplies descriptions and schemas, while this reviewed envelope
# prevents a newly advertised parameter from becoming model-controlled authority.
_MODEL_ARGUMENT_TYPES = {
    "get_order_status": {"order_id": frozenset({"string"})},
    "list_customer_orders": {"active_only": frozenset({"boolean"})},
    "get_service_history": {"instrument_id": frozenset({"string", "null"})},
    "search_troubleshooting": {
        "model": frozenset({"string", "null"}),
        "symptom": frozenset({"string", "null"}),
    },
}
_MODEL_REQUIRED = {"get_order_status": frozenset({"order_id"})}
_PRIVATE_TOOL_NAMES = TOOL_NAMES - {"search_troubleshooting"}
_RECORD_ID_PATTERNS = {
    "order_id": re.compile(r"DEMO-ORD-[0-9]{4}"),
    "instrument_id": re.compile(r"DEMO-INS-[0-9]{4}"),
}


def _schema_types(schema: Any) -> frozenset[str]:
    """Accept the simple primitive schemas used by these reviewed read tools."""

    if not isinstance(schema, dict):
        raise ValueError("Malformed reviewed MCP argument schema")
    # Reject references and metadata rather than forwarding opaque definitions,
    # paths, or new authority fields from a changed server into a model prompt.
    if not set(schema).issubset({"type", "anyOf", "default"}):
        raise ValueError("Unsupported reviewed MCP argument schema")
    if "anyOf" in schema:
        alternatives = schema["anyOf"]
        if "type" in schema or not isinstance(alternatives, list) or not alternatives:
            raise ValueError("Malformed reviewed MCP argument schema")
        if any(not isinstance(item, dict) or set(item) != {"type"} for item in alternatives):
            raise ValueError("Unsupported reviewed MCP union schema")
        types = frozenset().union(*(_schema_types(item) for item in alternatives))
    else:
        primitive = schema.get("type")
        if not isinstance(primitive, str) or primitive not in {"string", "boolean", "null"}:
            raise ValueError("Unsupported reviewed MCP argument type")
        types = frozenset({primitive})
    return types


def _validated_model_schema(name: str, schema: Any) -> dict[str, Any]:
    """Check the closed model-facing schema against the reviewed parameter set."""

    if (
        not isinstance(schema, dict)
        or schema.get("type") != "object"
        or schema.get("additionalProperties") is not False
        or not set(schema).issubset({"type", "properties", "required", "additionalProperties"})
    ):
        raise ValueError("Malformed model-facing MCP schema")
    properties = schema.get("properties")
    if not isinstance(properties, dict) or set(properties) != set(_MODEL_ARGUMENT_TYPES[name]):
        raise ValueError("MCP parameters differ from the reviewed tool contract")
    required = schema.get("required", [])
    if (
        not isinstance(required, list)
        or any(not isinstance(field, str) for field in required)
        or len(required) != len(set(required))
        or set(required) != _MODEL_REQUIRED.get(name, frozenset())
    ):
        raise ValueError("MCP required parameters differ from the reviewed tool contract")
    for field, expected_types in _MODEL_ARGUMENT_TYPES[name].items():
        if _schema_types(properties[field]) != expected_types:
            raise ValueError("MCP argument types differ from the reviewed tool contract")
        if "default" in properties[field]:
            default = properties[field]["default"]
            if not (
                (field == "active_only" and default is False)
                or ("null" in expected_types and default is None)
            ):
                raise ValueError("MCP defaults differ from the reviewed tool contract")
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise ValueError("Malformed model-facing MCP schema") from exc
    return copy.deepcopy(schema)


def create_client(db_path: Path = DEFAULT_DB_PATH) -> Client:
    """Build the stdio transport that launches the server for each MCP session."""

    # FastMCP's subprocess does not inherit every environment variable. Give it
    # the same signing secret that minted the context or private verification
    # would fail. This is a trusted process setting, never a model/tool argument.
    env = {
        "PYTHONPATH": str(PROJECT_ROOT / "src"),
        "PYTHONIOENCODING": "utf-8",
        "DEMO_DB_PATH": str(Path(db_path).resolve()),
    }
    secret = os.environ.get("DEMO_SIGNING_SECRET")
    if secret:
        env["DEMO_SIGNING_SECRET"] = secret

    transport = StdioTransport(
        command=sys.executable,
        args=["-m", "customer_assistant.mcp_server"],
        cwd=str(PROJECT_ROOT),
        env=env,
        # Step 5 notebook kernels replace stderr with an object without a real
        # OS file descriptor. Use a real null handle so subprocess startup
        # works there; tool failures still surface through client results.
        log_file=Path(os.devnull),
    )
    return Client(transport)


async def list_tool_names(db_path: Path = DEFAULT_DB_PATH) -> list[str]:
    """Ask the subprocess for its advertised MCP tools."""

    async with create_client(db_path) as client:
        return sorted(tool.name for tool in await client.list_tools())


async def discover_model_tools(db_path: Path = DEFAULT_DB_PATH) -> list[dict[str, Any]]:
    """Discover reviewed MCP tools without exposing application identity inputs.

    A catalog change is a review boundary: unknown tools are omitted, and a
    missing or changed reviewed contract fails closed before any model call.
    """

    async with create_client(db_path) as client:
        advertised = await client.list_tools()
    if not isinstance(advertised, list):
        raise ValueError("Malformed advertised MCP catalog")
    definitions: dict[str, dict[str, Any]] = {}
    for tool in advertised:
        name = getattr(tool, "name", None)
        if not isinstance(name, str):
            raise ValueError("Malformed advertised MCP tool name")
        if name not in TOOL_NAMES:
            continue
        if name in definitions:
            raise ValueError("Duplicate reviewed MCP tool in discovered catalog")
        raw_schema = getattr(tool, "input_schema", None)
        if not isinstance(raw_schema, dict):
            raise ValueError("Malformed advertised MCP schema")
        raw_properties = raw_schema.get("properties")
        raw_required = raw_schema.get("required", [])
        trusted_fields = {"identity_context"} if name in _PRIVATE_TOOL_NAMES else set()
        expected_fields = set(_MODEL_ARGUMENT_TYPES[name]) | trusted_fields
        expected_required = set(_MODEL_REQUIRED.get(name, frozenset())) | trusted_fields
        if (
            raw_schema.get("type") != "object"
            or raw_schema.get("additionalProperties") is not False
            or not set(raw_schema).issubset({"type", "properties", "required", "additionalProperties"})
            or not isinstance(raw_properties, dict)
            or set(raw_properties) != expected_fields
            or not isinstance(raw_required, list)
            or any(not isinstance(field, str) for field in raw_required)
            or len(raw_required) != len(set(raw_required))
            or set(raw_required) != expected_required
        ):
            raise ValueError("Advertised MCP schema differs from the reviewed tool contract")
        parameters = _validated_model_schema(name, {
            "type": "object",
            "properties": {
                field: copy.deepcopy(raw_properties[field])
                for field in _MODEL_ARGUMENT_TYPES[name]
            },
            "required": [field for field in raw_required if field not in trusted_fields],
            "additionalProperties": False,
        })
        description = getattr(tool, "description", None)
        if not isinstance(description, str) or not description.strip():
            raise ValueError("Reviewed MCP tool has no description")
        # The opening paragraph is the capability summary; later paragraphs
        # explain signing and SQLite internals to developers, not to the model.
        description = description.strip().split("\n\n", 1)[0]
        definitions[name] = {
            "type": "function",
            "function": {"name": name, "description": description, "parameters": parameters},
        }
    if set(definitions) != TOOL_NAMES:
        raise ValueError("Discovered MCP catalog is missing a reviewed tool")
    return [definitions[name] for name in sorted(definitions)]


def validate_model_arguments(
    name: str,
    arguments: dict[str, Any],
    catalog: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return a fresh argument map validated against the discovered safe schema.

    This accepts no identity, customer, agent, trace, process, or database
    parameter. The orchestrator adds its own signed identity only afterward.
    """

    if not isinstance(name, str) or name not in TOOL_NAMES:
        raise ValueError("Unknown model-selected MCP tool")
    if not isinstance(arguments, dict) or any(not isinstance(field, str) for field in arguments):
        raise ValueError("MCP tool arguments must be a dictionary with string keys")
    if not isinstance(catalog, list):
        raise ValueError("Malformed model-facing MCP catalog")
    schemas: dict[str, dict[str, Any]] = {}
    for entry in catalog:
        if not isinstance(entry, dict) or set(entry) != {"type", "function"} or entry["type"] != "function":
            raise ValueError("Malformed model-facing MCP tool definition")
        function = entry["function"]
        if not isinstance(function, dict) or set(function) != {"name", "description", "parameters"}:
            raise ValueError("Malformed model-facing MCP function definition")
        tool_name = function["name"]
        if not isinstance(tool_name, str) or tool_name not in TOOL_NAMES or tool_name in schemas:
            raise ValueError("Unreviewed or duplicate model-facing MCP tool")
        if not isinstance(function["description"], str) or not function["description"].strip():
            raise ValueError("Model-facing MCP tool has no description")
        schemas[tool_name] = _validated_model_schema(tool_name, function["parameters"])
    if name not in schemas:
        raise ValueError("Selected MCP tool is missing from the discovered catalog")
    if set(arguments) - set(_MODEL_ARGUMENT_TYPES[name]):
        raise ValueError("Model supplied an unreviewed or application-owned MCP argument")
    if not Draft202012Validator(schemas[name]).is_valid(arguments):
        raise ValueError("MCP arguments do not match the reviewed schema")
    for field, pattern in _RECORD_ID_PATTERNS.items():
        value = arguments.get(field)
        if value is not None and pattern.fullmatch(value) is None:
            raise ValueError("MCP record ID must use the canonical demo format")
    return copy.deepcopy(arguments)


async def call_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    db_path: Path = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Call a registered tool and unpack structured MCP data."""

    if name not in TOOL_NAMES:
        # Only the reviewed server tools are callable through this
        # helper; a model-generated name cannot select a newly exposed tool.
        raise ValueError(f"Unknown demo MCP tool: {name}")
    async with create_client(db_path) as client:
        result = await client.call_tool(name, arguments)
    if not isinstance(result.data, dict):
        # Later policy and audit logic expect named fields. A text-only or
        # malformed MCP response must fail here rather than look like a deny.
        raise TypeError(f"MCP tool {name} returned no structured dictionary")
    return result.data


def _identity_context(identity: DemoIdentity | Mapping[str, Any]) -> dict[str, Any]:
    # Serialization happens immediately before the tool call. The trusted
    # application holds the signed context; the model receives filtered tool
    # evidence afterward, not this context or its signing secret.
    return identity.as_dict() if isinstance(identity, DemoIdentity) else dict(identity)


async def order_status(
    identity: DemoIdentity | Mapping[str, Any],
    order_id: str,
    *,
    db_path: Path = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Call get_order_status through MCP with application-supplied identity."""

    return await call_tool(
        "get_order_status",
        {"order_id": order_id, "identity_context": _identity_context(identity)},
        db_path=db_path,
    )


async def customer_orders(
    identity: DemoIdentity | Mapping[str, Any],
    *,
    active_only: bool = False,
    db_path: Path = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Call the all-orders MCP tool, optionally filtering to active orders."""

    # No customer ID is accepted here. The server derives it from the signed
    # context so a prompt cannot turn this into a cross-customer search.
    return await call_tool(
        "list_customer_orders",
        {
            "identity_context": _identity_context(identity),
            "active_only": active_only,
        },
        db_path=db_path,
    )


async def service_history(
    identity: DemoIdentity | Mapping[str, Any],
    instrument_id: str | None = None,
    *,
    db_path: Path = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Call signed service history, for one instrument or all owned events."""

    # Omitting the key, rather than inventing a customer ID or placeholder,
    # selects the server's all-owned-instruments route. The signed identity
    # continues to define which customer the query may read.
    arguments = {"identity_context": _identity_context(identity)}
    if instrument_id is not None:
        arguments["instrument_id"] = instrument_id
    return await call_tool(
        "get_service_history",
        arguments,
        db_path=db_path,
    )


async def troubleshooting(
    model: str | None = None,
    symptom: str | None = None,
    *,
    db_path: Path = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Call a public article search, or list the approved catalog."""

    # An empty argument map asks for the approved public catalog. If a caller
    # supplies only one term, the server returns no_match instead of broadening
    # that targeted question into unrelated advice.
    arguments = {}
    if model is not None:
        arguments["model"] = model
    if symptom is not None:
        arguments["symptom"] = symptom
    return await call_tool(
        "search_troubleshooting",
        arguments,
        db_path=db_path,
    )
