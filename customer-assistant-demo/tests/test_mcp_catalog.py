"""Exercise actual MCP discovery and the boundary for model-selected arguments."""

import asyncio
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from customer_assistant.mcp_client import (
    TOOL_NAMES,
    call_tool,
    create_client,
    discover_model_tools,
    validate_model_arguments,
)


class MCPCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.db_path = Path(cls.temp.name) / "unused-discovery.sqlite3"

        async def load_catalog():
            # Discovery requires no database read or signing secret. These are
            # actual server Tool objects, not a hand-maintained model fixture.
            async with create_client(cls.db_path) as client:
                advertised = await client.list_tools()
            catalog = await discover_model_tools(cls.db_path)
            return advertised, catalog

        cls.advertised, cls.catalog = asyncio.run(load_catalog())

    def _discover_advertised(self, tools):
        client = MagicMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        client.list_tools = AsyncMock(return_value=tools)
        with patch("customer_assistant.mcp_client.create_client", return_value=client) as build:
            result = asyncio.run(discover_model_tools(self.db_path))
        build.assert_called_once_with(self.db_path)
        client.list_tools.assert_awaited_once_with()
        return result

    def test_actual_server_catalog_is_complete_and_identity_is_hidden(self) -> None:
        self.assertEqual([entry["function"]["name"] for entry in self.catalog], sorted(TOOL_NAMES))
        self.assertEqual({tool.name for tool in self.advertised}, TOOL_NAMES)
        for entry in self.catalog:
            with self.subTest(tool=entry["function"]["name"]):
                self.assertEqual(entry["type"], "function")
                parameters = entry["function"]["parameters"]
                self.assertIs(parameters["additionalProperties"], False)
                self.assertNotIn("identity_context", json.dumps(entry))
                self.assertNotIn("SQLite", entry["function"]["description"])
        serialized = json.dumps(self.catalog)
        for hidden in (str(self.db_path), "DEMO_SIGNING_SECRET", "PYTHONPATH", "signature"):
            self.assertNotIn(hidden, serialized)

    def test_discovery_omits_unreviewed_advertised_tool(self) -> None:
        unknown = SimpleNamespace(
            name="delete_orders", description="Delete customer records",
            input_schema={"type": "object", "properties": {}},
        )
        self.assertEqual(self._discover_advertised([*self.advertised, unknown]), self.catalog)

    def test_discovery_does_not_mutate_server_schema(self) -> None:
        before = [tool.model_dump() for tool in self.advertised]
        self._discover_advertised(self.advertised)
        self.assertEqual([tool.model_dump() for tool in self.advertised], before)
        private = next(tool for tool in self.advertised if tool.name == "get_order_status")
        self.assertIn("identity_context", private.input_schema["properties"])
        self.assertIn("identity_context", private.input_schema["required"])

    def test_discovery_fails_closed_on_missing_or_duplicate_reviewed_tool(self) -> None:
        for advertised in (self.advertised[:-1], [*self.advertised, self.advertised[0]]):
            with self.subTest(names=[tool.name for tool in advertised]), self.assertRaises(ValueError):
                self._discover_advertised(advertised)

    def test_discovery_rejects_new_authority_parameter(self) -> None:
        advertised = copy.deepcopy(self.advertised)
        selected = next(tool for tool in advertised if tool.name == "list_customer_orders")
        selected.input_schema["properties"]["customer_id"] = {"type": "string"}
        with self.assertRaises(ValueError):
            self._discover_advertised(advertised)

    def test_discovery_rejects_changed_required_field_or_primitive_type(self) -> None:
        for change in ("required", "type"):
            advertised = copy.deepcopy(self.advertised)
            selected = next(tool for tool in advertised if tool.name == "get_order_status")
            if change == "required":
                selected.input_schema["required"].remove("order_id")
            else:
                selected.input_schema["properties"]["order_id"]["type"] = "object"
            with self.subTest(change=change), self.assertRaises(ValueError):
                self._discover_advertised(advertised)

    def test_discovery_rejects_schema_refs_and_defaults_that_could_leak(self) -> None:
        for schema in (
            {"$ref": "https://example.test/schema"},
            {"type": "string", "default": "private-path-or-secret"},
        ):
            advertised = copy.deepcopy(self.advertised)
            selected = next(tool for tool in advertised if tool.name == "get_order_status")
            selected.input_schema["properties"]["order_id"] = schema
            with self.subTest(schema=schema), self.assertRaises(ValueError):
                self._discover_advertised(advertised)

    def test_required_order_argument_and_canonical_id(self) -> None:
        supplied = {"order_id": "DEMO-ORD-1007"}
        validated = validate_model_arguments("get_order_status", supplied, self.catalog)
        self.assertEqual(validated, supplied)
        self.assertIsNot(validated, supplied)
        validated["order_id"] = "DEMO-ORD-1009"
        self.assertEqual(supplied["order_id"], "DEMO-ORD-1007")
        for arguments in ({}, {"order_id": None}, {"order_id": True}, {"order_id": 1007}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                validate_model_arguments("get_order_status", arguments, self.catalog)

    def test_bool_cannot_be_coerced_from_number_or_string(self) -> None:
        for value in (False, True):
            self.assertEqual(
                validate_model_arguments("list_customer_orders", {"active_only": value}, self.catalog),
                {"active_only": value},
            )
        for value in (0, 1, "true", "false", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_model_arguments("list_customer_orders", {"active_only": value}, self.catalog)

    def test_optional_service_argument_keeps_owned_collection_route(self) -> None:
        for arguments in ({}, {"instrument_id": None}, {"instrument_id": "DEMO-INS-1001"}):
            with self.subTest(arguments=arguments):
                self.assertEqual(validate_model_arguments("get_service_history", arguments, self.catalog), arguments)
        with self.assertRaises(ValueError):
            validate_model_arguments("get_service_history", {"instrument_id": {"user_id": "CUST-1002"}}, self.catalog)

    def test_optional_public_arguments_keep_catalog_and_targeted_routes(self) -> None:
        cases = (
            {},
            {"model": None, "symptom": None},
            {"model": "1260", "symptom": "pressure warning"},
            {"model": "1260"},
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                self.assertEqual(validate_model_arguments("search_troubleshooting", arguments, self.catalog), arguments)
        for value in (True, 1260, [], {"identity_context": {}}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_model_arguments("search_troubleshooting", {"model": value}, self.catalog)

    def test_identity_customer_and_process_injection_is_rejected(self) -> None:
        hidden_fields = (
            "identity_context", "user_id", "customer_id", "agent_id", "trace_id",
            "signature", "issued_at", "expires_at", "db_path", "DEMO_DB_PATH",
            "signing_secret", "identityContext", "unexpected",
        )
        for field in hidden_fields:
            arguments = {"order_id": "DEMO-ORD-1007", field: "model-supplied"}
            before = copy.deepcopy(arguments)
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_model_arguments("get_order_status", arguments, self.catalog)
            self.assertEqual(arguments, before)

    def test_record_ids_reject_sql_path_or_embedded_authority_text(self) -> None:
        for name, field in (("get_order_status", "order_id"), ("get_service_history", "instrument_id")):
            for value in ("1007", "demo-ord-1007", "DEMO-ORD-1007 OR 1=1", "../demo.sqlite3", "CUST-1002", "DEMO-ORD-1007\n"):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    validate_model_arguments(name, {field: value}, self.catalog)

    def test_unreviewed_tool_and_missing_catalog_entry_fail_closed(self) -> None:
        for name, catalog in (("delete_orders", self.catalog), ("get_order_status", [])):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_model_arguments(name, {}, catalog)

    def test_forged_catalog_cannot_broaden_arguments(self) -> None:
        catalog = copy.deepcopy(self.catalog)
        selected = next(entry for entry in catalog if entry["function"]["name"] == "get_order_status")
        parameters = selected["function"]["parameters"]
        parameters["properties"]["identity_context"] = {"type": "string"}
        with self.assertRaises(ValueError):
            validate_model_arguments("get_order_status", {"order_id": "DEMO-ORD-1007", "identity_context": "forged"}, catalog)
        parameters["properties"].pop("identity_context")
        parameters["additionalProperties"] = True
        with self.assertRaises(ValueError):
            validate_model_arguments("get_order_status", {"order_id": "DEMO-ORD-1007"}, catalog)

    def test_duplicate_catalog_or_unknown_entry_is_rejected(self) -> None:
        unknown = {
            "type": "function",
            "function": {"name": "delete_orders", "description": "Delete orders", "parameters": {}},
        }
        for catalog in ([*self.catalog, self.catalog[0]], [*self.catalog, unknown]):
            with self.assertRaises(ValueError):
                validate_model_arguments("get_order_status", {"order_id": "DEMO-ORD-1007"}, catalog)

    def test_argument_container_must_be_dict(self) -> None:
        for arguments in (None, "{}", [], {1: "DEMO-ORD-1007"}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                validate_model_arguments("get_order_status", arguments, self.catalog)

    def test_unreviewed_name_cannot_start_mcp_process(self) -> None:
        with patch("customer_assistant.mcp_client.create_client") as build:
            with self.assertRaises(ValueError):
                asyncio.run(call_tool("delete_orders", {}))
        build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
