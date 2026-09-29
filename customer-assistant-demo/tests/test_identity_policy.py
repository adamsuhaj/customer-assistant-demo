"""Step 2 - Verify signed demo identity and customer-owned data reads.

- Mint Alice/Bob contexts and confirm identity fields are signed and time limited.
- Check get_order_status() and get_service_history() return only owned rows.
- Step 11 verifies the signed customer's order rows include a matching
  customer name and demo contact email, with no foreign profile in denials.
- Step 11 checks the ID-free service list against each signed customer,
  invalid signatures, and inconsistent service/instrument owner links.
- Confirm Bob and altered claims cannot expose Alice's order or service evidence.
- Keep user, fixed agent, request trace, and process signing secret distinct.
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
from customer_assistant.identity import (
    AGENT_ID,
    IdentityConfigurationError,
    InvalidIdentity,
    mint_demo_identity,
    verify_demo_identity,
)
from customer_assistant.policy import (
    get_order_status, get_service_history, list_customer_orders,
    list_customer_service_history,
)


TEST_SIGNING_SECRET = "test-only-secret-for-synthetic-data-7f031a99"


class IdentityPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        # A fresh SQLite copy and process-local test secret isolate every
        # entitlement check from real environment values and other tests.
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db_path = Path(self.temp.name) / "demo.sqlite3"
        load_database(DEFAULT_SOURCE_DIR, self.db_path)
        env_patch = patch.dict(os.environ, {"DEMO_SIGNING_SECRET": TEST_SIGNING_SECRET})
        env_patch.start()
        self.addCleanup(env_patch.stop)

    def identity(self, login):
        return mint_demo_identity(login, db_path=self.db_path)

    def order(self, identity, order_id):
        return get_order_status(
            identity, order_id, db_path=self.db_path,
        )

    def service(self, identity, instrument_id):
        return get_service_history(
            identity, instrument_id, db_path=self.db_path,
        )

    def test_alice_can_read_her_order_and_service_history(self) -> None:
        # Passing the context both as a mapping and a dataclass checks the
        # form used across MCP transport and the direct trusted-code form.
        alice = self.identity("alice")
        order = self.order(alice.as_dict(), "DEMO-ORD-1007")
        service = self.service(alice, "DEMO-INS-1001")

        self.assertEqual(order.outcome, "allow")
        self.assertEqual(order.source_ids, ("DEMO-ORD-1007",))
        self.assertEqual(order.rows[0]["customer_id"], "CUST-1001")
        self.assertEqual(order.rows[0]["order_status"], "in_transit")
        self.assertEqual(order.trace_id, alice.trace_id)
        self.assertEqual(service.outcome, "allow")
        self.assertEqual(service.source_ids, ("DEMO-SVC-1009", "DEMO-SVC-1001"))
        self.assertEqual(service.rows[0]["resolution_status"], "closed")

    def test_bob_cannot_read_alice_order_or_service_row(self) -> None:
        # Assert on the whole serialized denial, because a future MCP tool
        # will pass that form onward to orchestration and audit code.
        bob = self.identity("bob")
        for result in (
            self.order(bob, "DEMO-ORD-1007"),
            self.service(bob.as_dict(), "DEMO-INS-1001"),
        ):
            with self.subTest(result=result.reason):
                self.assertEqual(result.outcome, "deny")
                self.assertEqual(result.rows, ())
                self.assertEqual(result.source_ids, ())
                payload = json.dumps(result.as_dict())
                self.assertNotIn("in_transit", payload)
                self.assertNotIn("DEMO-TRACK-1007", payload)
                self.assertNotIn("DEMO-SVC-1001", payload)

        self.assertEqual(
            self.order(bob, "DEMO-ORD-2001").rows[0]["customer_id"],
            "CUST-1002",
        )

    def test_authorized_order_rows_join_only_the_signed_customer_profile(self) -> None:
        # The CSV customer/order relationship must survive the policy query:
        # the answer layer needs these facts to state who owns each order.
        for login, customer_id, name, email, order_count in (
            ("alice", "CUST-1001", "Northstar Bioanalytics Demo Ltd", "alice@example.com", 6),
            ("bob", "CUST-1002", "Meridian Research Demo Lab Ltd", "bob@example.net", 5),
        ):
            with self.subTest(login=login):
                identity = self.identity(login)
                listed = list_customer_orders(identity, db_path=self.db_path)
                self.assertEqual(listed.outcome, "allow")
                self.assertEqual(len(listed.rows), order_count)
                for row in listed.rows:
                    self.assertEqual(row["customer_id"], customer_id)
                    self.assertEqual(row["customer_name"], name)
                    self.assertEqual(row["contact_email"], email)
                    # created_on is the actual order date; the downstream
                    # answer can sort by it without substituting delivery.
                    self.assertTrue(row["created_on"])
                single = self.order(identity, listed.rows[0]["order_id"])
                self.assertEqual(single.rows[0]["customer_name"], name)
                self.assertEqual(single.rows[0]["contact_email"], email)

        # A foreign ID still yields no order *or* joined customer fields.
        alice = self.identity("alice")
        denial = self.order(alice, "DEMO-ORD-2001")
        self.assertEqual(denial.outcome, "deny")
        self.assertEqual(denial.rows, ())
        self.assertNotIn("bob@example.net", json.dumps(denial.as_dict()))

    def test_service_list_stays_with_the_signed_customer_and_joins_profile(self) -> None:
        # An ID-free history request should cover the selected customer only.
        for login, event_ids, customer_id, name, email in (
            ("alice", ("DEMO-SVC-1009", "DEMO-SVC-1001"), "CUST-1001", "Northstar Bioanalytics Demo Ltd", "alice@example.com"),
            ("bob", ("DEMO-SVC-2001",), "CUST-1002", "Meridian Research Demo Lab Ltd", "bob@example.net"),
        ):
            with self.subTest(login=login):
                identity = self.identity(login)
                result = list_customer_service_history(identity, db_path=self.db_path)
                self.assertEqual(result.outcome, "allow")
                self.assertEqual(result.trace_id, identity.trace_id)
                self.assertEqual(result.source_ids, event_ids)
                for row in result.rows:
                    self.assertEqual(row["customer_id"], customer_id)
                    self.assertEqual(row["customer_name"], name)
                    self.assertEqual(row["contact_email"], email)

    def test_service_list_denies_invalid_identity_before_read(self) -> None:
        # The list has no instrument ID to check, so signature verification
        # must still gate the entire query before any row enters the result.
        forged = self.identity("alice").as_dict()
        forged["user_id"] = "CUST-1002"
        result = list_customer_service_history(forged, db_path=self.db_path)
        self.assertEqual(result.outcome, "deny")
        self.assertEqual(result.reason, "invalid_identity")
        self.assertIsNone(result.trace_id)
        self.assertEqual(result.rows, ())
        self.assertEqual(result.source_ids, ())

    def test_service_list_excludes_an_inconsistent_instrument_owner(self) -> None:
        # Defense in depth: if a later DB mutation mislabels the customer on
        # Alice's events, neither Alice nor Bob may see them through the list.
        with closing(sqlite3.connect(self.db_path)) as connection:
            with connection:
                connection.execute(
                    "UPDATE service_history SET customer_id = 'CUST-1002' "
                    "WHERE service_event_id IN ('DEMO-SVC-1001', 'DEMO-SVC-1009')"
                )
        alice = list_customer_service_history(self.identity("alice"), db_path=self.db_path)
        bob = list_customer_service_history(self.identity("bob"), db_path=self.db_path)
        self.assertEqual(alice.outcome, "allow")
        self.assertEqual(alice.rows, ())
        self.assertEqual(alice.source_ids, ())
        self.assertEqual(bob.source_ids, ("DEMO-SVC-2001",))
        self.assertNotIn("DEMO-SVC-1001", json.dumps(bob.as_dict()))
        self.assertNotIn("DEMO-SVC-1009", json.dumps(bob.as_dict()))

    def test_tampered_signature_or_user_id_is_denied_before_read(self) -> None:
        # Changing a signed claim must invalidate the context before either
        # private query runs, even when the new customer ID exists.
        alice = self.identity("alice")
        bad_signature = alice.as_dict()
        bad_signature["signature"] = "0" * 64
        malformed_signature = alice.as_dict()
        malformed_signature["signature"] = "invalid-ü-signature"
        forged_user = alice.as_dict()
        forged_user["user_id"] = "CUST-1002"

        for context in (bad_signature, malformed_signature, forged_user):
            with self.subTest(context=context["user_id"]):
                with self.assertRaises(InvalidIdentity):
                    verify_demo_identity(context)
                for result in (
                    self.order(context, "DEMO-ORD-1007"),
                    self.service(context, "DEMO-INS-1001"),
                ):
                    self.assertEqual(result.outcome, "deny")
                    self.assertEqual(result.reason, "invalid_identity")
                    self.assertEqual(result.rows, ())
                    self.assertEqual(result.source_ids, ())

    def test_user_agent_and_trace_are_distinct_and_secret_is_not_serialized(self) -> None:
        alice = self.identity("alice")
        bob = self.identity("bob")
        self.assertEqual(alice.user_id, "CUST-1001")
        self.assertEqual(bob.user_id, "CUST-1002")
        self.assertEqual(alice.agent_id, AGENT_ID)
        self.assertNotEqual(alice.user_id, alice.agent_id)
        self.assertNotIn(alice.trace_id, {alice.user_id, alice.agent_id})
        self.assertNotEqual(alice.trace_id, bob.trace_id)
        self.assertEqual(
            verify_demo_identity(alice.as_dict()),
            alice,
        )
        self.assertNotIn(TEST_SIGNING_SECRET, json.dumps(alice.as_dict()))
        self.assertNotIn("DEMO_SIGNING_SECRET", alice.as_dict())

    def test_private_reads_require_a_configured_signing_secret(self) -> None:
        alice = self.identity("alice")
        with patch.dict(os.environ, {"DEMO_SIGNING_SECRET": ""}):
            with self.assertRaises(IdentityConfigurationError):
                mint_demo_identity("alice", db_path=self.db_path)
            with self.assertRaises(IdentityConfigurationError):
                self.order(alice, "DEMO-ORD-1007")

    def test_missing_database_read_does_not_create_an_empty_file(self) -> None:
        alice = self.identity("alice")
        missing = Path(self.temp.name) / "missing.sqlite3"
        with self.assertRaises(FileNotFoundError):
            mint_demo_identity("alice", db_path=missing)
        with self.assertRaises(FileNotFoundError):
            get_order_status(alice, "DEMO-ORD-1007", db_path=missing)
        self.assertFalse(missing.exists())


if __name__ == "__main__":
    unittest.main()
