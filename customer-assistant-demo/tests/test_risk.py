"""Step 6 - Verify risk_for_action() and assess_composition() inheritance.

- Check public, single and listed order, and service tools start at tiers
  0, 1, 1, and 2; an order list is still private customer data.
- Prove row classifications can raise a tier but cannot lower a tool floor.
- Prove a composition inherits the highest tier of every action and row touched.
- Treat unknown tools or malformed tiers as the most restrictive tier 3.
- Step 12 assigns a registered tier-3 floor to cross-provider handoffs.
"""

from __future__ import annotations

import unittest

from customer_assistant.risk import TOOL_TIER_FLOORS, assess_composition, risk_for_action


class RiskAssessmentTests(unittest.TestCase):
    def test_tool_floors_apply_even_without_rows_or_on_denial(self) -> None:
        self.assertEqual(risk_for_action("search_troubleshooting").tier, 0)
        self.assertEqual(risk_for_action("get_order_status").tier, 1)
        self.assertEqual(risk_for_action("list_customer_orders").tier, 1)
        self.assertEqual(risk_for_action("get_service_history").tier, 2)
        self.assertEqual(risk_for_action("provider_handoff").tier, 3)
        # A caller cannot redefine the floor and downgrade private evidence.
        with self.assertRaises(TypeError):
            TOOL_TIER_FLOORS["get_order_status"] = 0

    def test_row_tier_can_raise_but_not_lower_tool_floor(self) -> None:
        # Orders have no tier column in the CSV; its private tool floor applies.
        self.assertEqual(
            risk_for_action("get_order_status", [{"order_id": "DEMO-ORD-1007"}]).tier,
            1,
        )
        # Even a purported caller tier or a lower stored row tier cannot
        # downgrade the registered private order tool.
        lowered = risk_for_action(
            "get_order_status",
            [{"order_id": "DEMO-ORD-1007", "access_tier": "0", "tier": 0}],
        )
        self.assertEqual(lowered.tier, 1)
        listed = risk_for_action(
            "list_customer_orders",
            [
                {"order_id": "DEMO-ORD-1007", "access_tier": "0"},
                {"order_id": "DEMO-ORD-1002", "access_tier": "2"},
            ],
        )
        self.assertEqual(listed.tier, 2)
        self.assertEqual(
            [component["name"] for component in listed.as_dict()["components"]],
            ["list_customer_orders", "DEMO-ORD-1007", "DEMO-ORD-1002"],
        )
        self.assertEqual(risk_for_action("get_service_history", [{"access_tier": "0"}]).tier, 2)
        raised = risk_for_action(
            "search_troubleshooting",
            [{"article_id": "restricted-KB", "access_tier": "3"}],
        )
        self.assertEqual(raised.tier, 3)
        self.assertEqual(raised.as_dict()["components"][-1]["name"], "restricted-KB")

    def test_composition_inherits_highest_touched_tier(self) -> None:
        # A public article and private order cannot hide a service-history
        # read: the full composition must inherit that highest classification.
        assessed = assess_composition(
            [
                ("search_troubleshooting", [{"article_id": "DEMO-KB-001", "access_tier": "0"}]),
                ("get_order_status", []),
                ("get_service_history", [{"service_event_id": "DEMO-SVC-1001", "access_tier": "2"}]),
            ]
        )
        self.assertEqual(assessed.tier, 2)
        self.assertEqual(assess_composition([]).tier, 0)
        self.assertEqual(len(assessed.as_dict()["components"]), 5)

    def test_unknown_and_malformed_classification_fail_closed(self) -> None:
        self.assertEqual(risk_for_action("unregistered_tool").tier, 3)
        for malformed in (None, True, "two", "-1", "4", 99):
            with self.subTest(value=malformed):
                self.assertEqual(
                    risk_for_action("search_troubleshooting", [{"access_tier": malformed}]).tier,
                    3,
                )
        self.assertEqual(risk_for_action("search_troubleshooting", [None]).tier, 3)


if __name__ == "__main__":
    unittest.main()
