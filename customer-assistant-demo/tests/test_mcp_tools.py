"""Step 3 - Call the MCP server tools through a real stdio client.

- Confirm the server advertises single-order, all-orders, service, and public
  troubleshooting tools.
- Pass signed identity to all three private tools across the subprocess boundary.
- Show Alice receives six and Bob five owned orders with all statuses, while
  neither receives the other's rows. A forged identity receives no evidence.
- Show single-order and service reads still deny cross-customer requests.
- Show an omitted instrument returns only the signed customer's service events.
- Show public troubleshooting returns either matching guidance or the tier-0
  approved catalog without a login; neither route exposes private records.
"""

import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from customer_assistant.database import DEFAULT_SOURCE_DIR, load_database
from customer_assistant.identity import mint_demo_identity
from customer_assistant.mcp_client import (
    create_client, customer_orders, order_status, service_history,
    troubleshooting,
)


TEST_SIGNING_SECRET = "test-only-mcp-secret-for-synthetic-records-1d4cb37e"
ALICE_ORDERS = (
    ("DEMO-ORD-1008", "processing"),
    ("DEMO-ORD-1007", "in_transit"),
    ("DEMO-ORD-1009", "in_transit"),
    ("DEMO-ORD-1006", "in_transit"),
    ("DEMO-ORD-1004", "delivered"),
    ("DEMO-ORD-1002", "delivered"),
)
BOB_ORDERS = (
    ("DEMO-ORD-2005", "processing"),
    ("DEMO-ORD-2001", "processing"),
    ("DEMO-ORD-2004", "in_transit"),
    ("DEMO-ORD-2003", "delivered"),
    ("DEMO-ORD-2002", "delivered"),
)


class MCPToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        # The real stdio server receives a disposable SQLite copy. The test
        # never points an MCP subprocess at a production or shared database.
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db_path = Path(self.temp.name) / "demo.sqlite3"
        load_database(DEFAULT_SOURCE_DIR, self.db_path)
        env_patch = patch.dict(os.environ, {"DEMO_SIGNING_SECRET": TEST_SIGNING_SECRET})
        env_patch.start()
        self.addCleanup(env_patch.stop)

    async def test_order_tool_is_listed_and_denies_bob_without_order_data(self) -> None:
        # Listing proves the separate server exposes only the intended tools;
        # calling them here proves policy still runs across the stdio hop.
        alice = mint_demo_identity("alice", db_path=self.db_path)
        bob = mint_demo_identity("bob", db_path=self.db_path)
        async with create_client(self.db_path) as client:
            names = {tool.name for tool in await client.list_tools()}
            self.assertEqual(names, {
                "get_order_status", "list_customer_orders",
                "get_service_history", "search_troubleshooting",
            })
            approved = (await client.call_tool("get_order_status", {
                "order_id": "DEMO-ORD-1007",
                "identity_context": alice.as_dict(),
            })).data
            denied = (await client.call_tool("get_order_status", {
                "order_id": "DEMO-ORD-1007",
                "identity_context": bob.as_dict(),
            })).data

        self.assertEqual(approved["outcome"], "allow")
        self.assertEqual(approved["source_ids"], ["DEMO-ORD-1007"])
        self.assertEqual(approved["rows"][0]["order_status"], "in_transit")
        self.assertEqual(denied["outcome"], "deny")
        self.assertEqual(denied["rows"], [])
        self.assertEqual(denied["source_ids"], [])
        self.assertNotIn("DEMO-ORD-1007", json.dumps(denied))
        self.assertNotIn("DEMO-TRACK-1007", json.dumps(denied))
        # The convenience client used by later orchestration must take the
        # same protocol path and decode the structured result correctly.
        wrapped = await order_status(alice, "DEMO-ORD-1007", db_path=self.db_path)
        self.assertEqual(wrapped["source_ids"], ["DEMO-ORD-1007"])

    async def test_all_orders_tool_filters_by_signed_customer(self) -> None:
        alice = mint_demo_identity("alice", db_path=self.db_path)
        bob = mint_demo_identity("bob", db_path=self.db_path)
        forged = alice.as_dict()
        forged["user_id"] = bob.user_id

        # The same MCP subprocess boundary used by the app must enforce the
        # signed customer claim and return a citation for each visible order.
        async with create_client(self.db_path) as client:
            alice_result = (await client.call_tool("list_customer_orders", {
                "identity_context": alice.as_dict(),
            })).data
            bob_result = (await client.call_tool("list_customer_orders", {
                "identity_context": bob.as_dict(),
            })).data
            forged_result = (await client.call_tool("list_customer_orders", {
                "identity_context": forged,
            })).data

        self.assertEqual(alice_result["outcome"], "allow")
        self.assertEqual(
            alice_result["source_ids"], [order_id for order_id, _ in ALICE_ORDERS]
        )
        self.assertEqual(
            [(row["order_id"], row["order_status"]) for row in alice_result["rows"]],
            list(ALICE_ORDERS),
        )
        self.assertTrue(all(row["customer_id"] == alice.user_id
                            for row in alice_result["rows"]))
        for order_id, _ in BOB_ORDERS:
            self.assertNotIn(order_id, json.dumps(alice_result))
        self.assertEqual(
            bob_result["source_ids"], [order_id for order_id, _ in BOB_ORDERS]
        )
        self.assertEqual(
            [(row["order_id"], row["order_status"]) for row in bob_result["rows"]],
            list(BOB_ORDERS),
        )
        for order_id, _ in ALICE_ORDERS:
            self.assertNotIn(order_id, json.dumps(bob_result))

        self.assertEqual(forged_result["outcome"], "deny")
        self.assertEqual(forged_result["reason"], "invalid_identity")
        self.assertEqual(forged_result["rows"], [])
        self.assertEqual(forged_result["source_ids"], [])
        self.assertIsNone(forged_result["trace_id"])

        wrapped = await customer_orders(alice, db_path=self.db_path)
        self.assertEqual(wrapped["source_ids"], alice_result["source_ids"])

        active = await customer_orders(alice, active_only=True, db_path=self.db_path)
        self.assertEqual(active["outcome"], "allow")
        self.assertEqual(
            active["source_ids"],
            [order_id for order_id, status in ALICE_ORDERS if status != "delivered"],
        )
        self.assertTrue(all(row["order_status"] != "delivered" for row in active["rows"]))

    async def test_no_matching_orders_is_allowed_empty_list(self) -> None:
        alice = mint_demo_identity("alice", db_path=self.db_path)
        # Change only the disposable test copy: an authenticated customer who
        # has no matching orders is different from a forged or foreign user.
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute("DELETE FROM orders WHERE customer_id = ?", (alice.user_id,))
            connection.commit()
        result = await customer_orders(alice, active_only=True, db_path=self.db_path)
        self.assertEqual(result["outcome"], "allow")
        self.assertEqual(result["rows"], [])
        self.assertEqual(result["source_ids"], [])
        self.assertIsNone(result["reason"])
        self.assertEqual(result["trace_id"], alice.trace_id)

    async def test_service_tool_checks_instrument_ownership(self) -> None:
        alice = mint_demo_identity("alice", db_path=self.db_path)
        bob = mint_demo_identity("bob", db_path=self.db_path)
        async with create_client(self.db_path) as client:
            approved = (await client.call_tool("get_service_history", {
                "instrument_id": "DEMO-INS-1001",
                "identity_context": alice.as_dict(),
            })).data
            denied = (await client.call_tool("get_service_history", {
                "instrument_id": "DEMO-INS-1001",
                "identity_context": bob.as_dict(),
            })).data

        self.assertEqual(approved["outcome"], "allow")
        self.assertEqual(approved["source_ids"], ["DEMO-SVC-1009", "DEMO-SVC-1001"])
        self.assertEqual(approved["rows"][0]["resolution_status"], "closed")
        self.assertEqual(denied["outcome"], "deny")
        self.assertEqual(denied["rows"], [])
        self.assertEqual(denied["source_ids"], [])
        self.assertNotIn("DEMO-SVC-1001", json.dumps(denied))
        self.assertNotIn("DEMO-SVC-1009", json.dumps(denied))

    async def test_service_tool_without_instrument_scopes_to_signed_customer(self) -> None:
        alice = mint_demo_identity("alice", db_path=self.db_path)
        bob = mint_demo_identity("bob", db_path=self.db_path)
        forged = alice.as_dict()
        forged["user_id"] = bob.user_id

        # Omitting the instrument asks policy for all of the signed user's
        # events. It must not turn a natural-language reference to another
        # customer into a cross-customer query.
        async with create_client(self.db_path) as client:
            alice_result = (await client.call_tool("get_service_history", {
                "identity_context": alice.as_dict(),
            })).data
            bob_result = (await client.call_tool("get_service_history", {
                "identity_context": bob.as_dict(),
            })).data
            forged_result = (await client.call_tool("get_service_history", {
                "identity_context": forged,
            })).data

        self.assertEqual(alice_result["outcome"], "allow")
        self.assertEqual(alice_result["source_ids"], ["DEMO-SVC-1009", "DEMO-SVC-1001"])
        self.assertEqual(alice_result["trace_id"], alice.trace_id)
        self.assertNotIn("DEMO-SVC-2001", json.dumps(alice_result))
        self.assertEqual(bob_result["outcome"], "allow")
        self.assertEqual(bob_result["source_ids"], ["DEMO-SVC-2001"])
        self.assertNotIn("DEMO-SVC-1001", json.dumps(bob_result))
        self.assertNotIn("DEMO-SVC-1009", json.dumps(bob_result))
        self.assertEqual(forged_result["outcome"], "deny")
        self.assertEqual(forged_result["rows"], [])
        self.assertEqual(forged_result["source_ids"], [])
        self.assertIsNone(forged_result["trace_id"])

        # The wrapper used by orchestration must keep the same no-ID shape.
        wrapped = await service_history(alice, db_path=self.db_path)
        self.assertEqual(wrapped["source_ids"], alice_result["source_ids"])

    async def test_public_troubleshooting_tool_returns_cited_article_without_login(self) -> None:
        # A public tier 0 lookup must not require the private signing secret.
        with patch.dict(os.environ, {"DEMO_SIGNING_SECRET": ""}):
            async with create_client(self.db_path) as client:
                result = (await client.call_tool("search_troubleshooting", {
                    "model": "1260 Infinity II Quaternary Pump",
                    "symptom": "pressure warning",
                })).data
                wrong_model = (await client.call_tool("search_troubleshooting", {
                    "model": "9999 Infinity II Pump",
                    "symptom": "pressure warning",
                })).data
                ambiguous_model = (await client.call_tool("search_troubleshooting", {
                    "model": "Infinity II Pump",
                    "symptom": "pressure warning",
                })).data

        self.assertEqual(result["outcome"], "allow")
        self.assertEqual(result["source_ids"], ["DEMO-KB-001"])
        self.assertEqual(result["rows"][0]["access_tier"], "0")
        self.assertIn("G7111AUser.pdf", result["rows"][0]["source_url"])
        self.assertIn("Pressure Below Lower Limit", result["rows"][0]["source_locator"])
        self.assertEqual(wrong_model["outcome"], "no_match")
        self.assertEqual(wrong_model["source_ids"], [])
        self.assertEqual(ambiguous_model["outcome"], "no_match")
        self.assertEqual(ambiguous_model["source_ids"], [])

    async def test_public_catalog_lists_only_approved_tier_zero_articles(self) -> None:
        # Add two tempting rows to the disposable copy. Neither an internal
        # article nor an unpublished summary may enter the public catalog.
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute(
                "INSERT INTO troubleshooting_articles "
                "SELECT 'DEMO-KB-PRIVATE', model_family, symptom, article_summary, "
                "approved_assistant_response, escalation_rule, source_title, "
                "source_locator, source_url, '2', content_status "
                "FROM troubleshooting_articles WHERE article_id = 'DEMO-KB-001'"
            )
            connection.execute(
                "INSERT INTO troubleshooting_articles "
                "SELECT 'DEMO-KB-DRAFT', model_family, symptom, article_summary, "
                "approved_assistant_response, escalation_rule, source_title, "
                "source_locator, source_url, '0', 'draft' "
                "FROM troubleshooting_articles WHERE article_id = 'DEMO-KB-001'"
            )
            connection.commit()

        with patch.dict(os.environ, {"DEMO_SIGNING_SECRET": ""}):
            async with create_client(self.db_path) as client:
                result = (await client.call_tool("search_troubleshooting", {})).data
            wrapped = await troubleshooting(db_path=self.db_path)

        for catalog in (result, wrapped):
            self.assertEqual(catalog["outcome"], "allow")
            self.assertEqual(catalog["source_ids"], ["DEMO-KB-001", "DEMO-KB-002"])
            self.assertTrue(all(row["access_tier"] == "0" for row in catalog["rows"]))
            self.assertTrue(all(
                row["content_status"] == "synthetic_summary_with_public_source"
                for row in catalog["rows"]
            ))
            rendered = json.dumps(catalog)
            for forbidden in (
                "DEMO-KB-PRIVATE", "DEMO-KB-DRAFT", "DEMO-SVC-1001",
                "DEMO-ORD-1007", "CUST-1001", "CUST-1002",
            ):
                self.assertNotIn(forbidden, rendered)


if __name__ == "__main__":
    unittest.main()
