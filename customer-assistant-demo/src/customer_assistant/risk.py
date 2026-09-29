"""Step 6 - calculate the risk inherited from each tool and accessed row.

- ``risk_for_action`` starts with a registered MCP tool floor: approved public
  troubleshooting is tier 0, one or all customer orders are tier 1, and one or
  all customer service events are tier 2. Step 11 joined customer fields do
  not lower the tier of the private order or service read that produced them.
- An explicit row ``access_tier`` can raise its tool floor. Missing row tiers
  leave the floor in place; an unknown tool or malformed tier becomes tier 3.
- ``assess_composition`` takes the maximum across all accessed tools and rows
  and retains each component so a reviewer can see what set the result.
- These classifications live in code, outside model prompts and caller
  arguments; a combined workflow cannot declare itself safer than its reads.
- Step 12 provider handoff is tier 3 because previously answered customer
  information crosses to another model provider. It is a registered audit
  action, not an additional MCP data tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence


# These tool/action floors are harness policy, not arguments accepted from a
# notebook or model. Orders in the supplied CSV have no access_tier column,
# so an order read still receives tier 1 from its tool. Returning five orders or a joined
# owner profile does not erase that private floor; a future classified row can
# raise it but never lower it.
TOOL_TIER_FLOORS: Mapping[str, int] = MappingProxyType({
    "get_order_status": 1,
    "list_customer_orders": 1,
    "get_service_history": 2,
    "search_troubleshooting": 0,
    "provider_handoff": 3,
})
MAX_TIER = 3
_RECORD_ID_KEYS = ("order_id", "service_event_id", "article_id", "instrument_id")


@dataclass(frozen=True)
class RiskComponent:
    """One inspectable tool or row classification behind a final tier."""

    kind: str
    name: str
    tier: int
    reason: str

    def as_dict(self) -> dict[str, str | int]:
        return {
            "kind": self.kind,
            "name": self.name,
            "tier": self.tier,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RiskAssessment:
    """The assessed tier together with the contributors a reviewer can audit."""

    tier: int
    components: tuple[RiskComponent, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "components": [component.as_dict() for component in self.components],
        }


def _record_tier(value: object) -> tuple[int, bool]:
    """Parse a row's explicit tier and assign tier 3 if that tier is malformed."""

    # bool is an int subclass in Python, but an access_tier of True/False is
    # malformed data rather than a deliberate classification.
    if isinstance(value, bool):
        return MAX_TIER, False
    if isinstance(value, int) and 0 <= value <= MAX_TIER:
        return value, True
    # Accept the exact CSV strings only. Trimming or guessing from values like
    # ``2-ish`` would hide data-quality failures in a security classification.
    if isinstance(value, str) and value in {"0", "1", "2", "3"}:
        return int(value), True
    return MAX_TIER, False


def risk_for_action(
    skill: str,
    rows: Sequence[Mapping[str, Any]] = (),
) -> RiskAssessment:
    """Return the highest tier contributed by a tool and its accessed rows.

    A denied private-tool attempt still has that tool's floor. A row without
    ``access_tier`` adds no new classification, while an invalid explicit
    value raises the assessment to tier 3.
    """

    # Unknown names must not inherit the public tier by accident. Their
    # tier-3 component also makes the reason visible in the report.
    floor = TOOL_TIER_FLOORS.get(skill, MAX_TIER)
    components = [
        RiskComponent(
            kind="tool",
            name=skill,
            tier=floor,
            reason="registered action floor" if skill in TOOL_TIER_FLOORS else "unknown tool",
        )
    ]
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            # A malformed evidence item cannot safely be treated as public.
            components.append(RiskComponent("data", f"row_{index}", MAX_TIER, "malformed row"))
            continue

        # No field means there is no separate data classification to add;
        # the registered tool floor still participates in the final maximum.
        if "access_tier" not in row:
            continue
        tier, valid = _record_tier(row["access_tier"])
        # Preserve a useful source label in the explanation without trusting
        # that source label to determine the classification itself.
        record_id = next(
            (str(row[key]) for key in _RECORD_ID_KEYS if row.get(key)),
            f"row_{index}",
        )
        components.append(
            RiskComponent(
                kind="data",
                name=record_id,
                tier=tier,
                reason="record access_tier" if valid else "invalid record access_tier",
            )
        )

    # Max, rather than an average, prevents one public tool call from diluting
    # the effect of a sensitive row touched in the same action.
    return RiskAssessment(
        tier=max(component.tier for component in components),
        components=tuple(components),
    )


def assess_composition(
    actions: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
) -> RiskAssessment:
    """Combine action assessments so the whole workflow inherits the maximum."""

    # Flatten the explanations from each action, not just their numeric tiers,
    # so a reviewer can see the exact tool or row responsible for promotion.
    components = tuple(
        component
        for skill, rows in actions
        for component in risk_for_action(skill, rows).components
    )
    # A composition with no actions has touched no classified tool or data.
    return RiskAssessment(
        tier=max((component.tier for component in components), default=0),
        components=components,
    )
