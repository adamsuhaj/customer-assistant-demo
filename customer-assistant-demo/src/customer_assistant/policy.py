"""Step 2 - Enforce ownership before private rows leave SQLite.

- Verify the application's signed identity on every private read. The
  customer ID comes from that identity, never from a user question, tool
  supplied name, or model claim.
- ``get_order_status`` returns one order only when both its customer_id and
  linked instrument belong to the signed customer. Step 9 added
  ``list_customer_orders`` with the same checks for all or active orders.
- ``get_service_history`` requires ownership of the requested instrument
  before returning its events. Step 11 added
  ``list_customer_service_history`` to return all events across the signed
  customer's instruments when no instrument ID was supplied.
- Step 11 joins authorized orders and service events to the matching customer
  row. This lets grounded answers state the customer ID, organization, and
  demo contact email from verified data, without exposing Bob to Alice.
- Apply ownership predicates in SQL so foreign rows cannot become Python
  results, model evidence, citations, or audit evidence. Missing and foreign
  IDs have the same denial shape; a valid signed trace may be retained for
  audit, while a forged context supplies no trusted trace ID.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .database import DEFAULT_DB_PATH, connect_readonly
from .identity import DemoIdentity, InvalidIdentity, verify_demo_identity


@dataclass(frozen=True)
class AccessResult:
    """One MCP response shape for allowed rows or evidence-free denials."""

    outcome: str
    trace_id: str | None
    rows: tuple[dict[str, str], ...] = ()
    source_ids: tuple[str, ...] = ()
    reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "outcome": self.outcome,
            "trace_id": self.trace_id,
            "rows": list(self.rows),
            "source_ids": list(self.source_ids),
            "reason": self.reason,
        }


def _connection(db_path: Path) -> sqlite3.Connection:
    # Policy should only inspect the validated demo copy. Read-only mode also
    # makes a missing DB an explicit setup error rather than creating a blank
    # one that would look like a legitimate authorization denial.
    connection = connect_readonly(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def _verified_or_denied(
    context: DemoIdentity | Mapping[str, Any],
) -> DemoIdentity | AccessResult:
    try:
        return verify_demo_identity(context)
    except InvalidIdentity:
        # Do not trust even the trace ID of a forged/expired context: copying
        # it into a denial could pollute a real audit trail. No data is fetched.
        return AccessResult("deny", None, reason="invalid_identity")


def get_order_status(
    context: DemoIdentity | Mapping[str, Any],
    order_id: str,
    *,
    db_path: Path = DEFAULT_DB_PATH,
) -> AccessResult:
    """Return an order only when the signed user's customer owns that row."""

    verified = _verified_or_denied(context)
    if isinstance(verified, AccessResult):
        return verified

    # Bind the signed user ID into the query and check that the order's
    # instrument has the same owner. Join that *same* customer before copying
    # profile fields, so the model can describe the relationship only after
    # authorization. Fetching by order ID first, then checking in Python,
    # would move private fields across the authorization boundary.
    with closing(_connection(db_path)) as connection:
        row = connection.execute(
            "SELECT o.*, c.customer_name, c.contact_email FROM orders AS o "
            "JOIN instruments AS i ON i.instrument_id = o.instrument_id "
            "AND i.customer_id = o.customer_id "
            "JOIN customers AS c ON c.customer_id = o.customer_id "
            "WHERE o.order_id = ? AND o.customer_id = ?",
            (order_id, verified.user_id),
        ).fetchone()
    if row is None:
        # A missing ID and a foreign ID produce the same denial. Distinguishing
        # them would let a caller probe which order numbers exist.
        return AccessResult(
            "deny", verified.trace_id, reason="not_found_or_not_authorized"
        )
    return AccessResult(
        "allow", verified.trace_id, (dict(row),), (row["order_id"],)
    )


def list_customer_orders(
    context: DemoIdentity | Mapping[str, Any],
    *,
    active_only: bool = False,
    db_path: Path = DEFAULT_DB_PATH,
) -> AccessResult:
    """Return owned orders, optionally limited to processing or in transit."""

    verified = _verified_or_denied(context)
    if isinstance(verified, AccessResult):
        return verified

    # Customer identity comes only from the verified signature. Keep the
    # customer predicate inside SQL, and require the linked instrument to
    # belong to that same customer, just as the single-order read does. Join
    # the matching customer row to give a verified ownership relationship to
    # the answer layer. The question and model cannot choose another customer.
    # This optional fragment is a fixed SQL literal chosen by trusted code,
    # not text supplied by the question. "Active" has the two validated
    # statuses below; delivered orders remain available in the full list.
    # Apply the filter inside the authorized query so excluded rows do not
    # become intermediate evidence in Python.
    active_clause = (
        "AND o.order_status IN ('processing', 'in_transit') "
        if active_only else ""
    )
    with closing(_connection(db_path)) as connection:
        query = (
            "SELECT o.*, c.customer_name, c.contact_email FROM orders AS o "
            "JOIN instruments AS i ON i.instrument_id = o.instrument_id "
            "AND i.customer_id = o.customer_id "
            "JOIN customers AS c ON c.customer_id = o.customer_id "
            "WHERE o.customer_id = ? "
            + active_clause
            + "ORDER BY o.created_on DESC, o.order_id DESC"
        )
        rows = connection.execute(query, (verified.user_id,)).fetchall()

    # An authenticated customer with no orders is an allowed empty result,
    # not an authorization failure. The stable order keeps answers and source
    # citations reproducible for the demo and its evaluations.
    return AccessResult(
        "allow",
        verified.trace_id,
        tuple(dict(row) for row in rows),
        tuple(row["order_id"] for row in rows),
    )


def get_service_history(
    context: DemoIdentity | Mapping[str, Any],
    instrument_id: str,
    *,
    db_path: Path = DEFAULT_DB_PATH,
) -> AccessResult:
    """Return service events only for an instrument owned by the signed user."""

    verified = _verified_or_denied(context)
    if isinstance(verified, AccessResult):
        return verified

    # Verify the parent instrument first: an owned instrument with no events
    # is a valid empty result, whereas a foreign or missing instrument is a
    # denial and must not reveal whether its service events exist.
    with closing(_connection(db_path)) as connection:
        owned = connection.execute(
            "SELECT 1 FROM instruments WHERE instrument_id = ? AND customer_id = ?",
            (instrument_id, verified.user_id),
        ).fetchone()
        if owned is None:
            return AccessResult(
                "deny", verified.trace_id, reason="not_found_or_not_authorized"
            )
        # Reapply the customer predicate to each service row. If inconsistent
        # data somehow reaches SQLite, an owned instrument alone must not
        # authorize a stray event labeled as another customer's record.
        rows = connection.execute(
            "SELECT s.*, c.customer_name, c.contact_email "
            "FROM service_history AS s "
            "JOIN customers AS c ON c.customer_id = s.customer_id "
            "WHERE s.instrument_id = ? AND s.customer_id = ? "
            "ORDER BY s.event_date DESC, s.service_event_id DESC",
            (instrument_id, verified.user_id),
        ).fetchall()

    return AccessResult(
        "allow",
        verified.trace_id,
        tuple(dict(row) for row in rows),
        tuple(row["service_event_id"] for row in rows),
    )


def list_customer_service_history(
    context: DemoIdentity | Mapping[str, Any],
    *,
    db_path: Path = DEFAULT_DB_PATH,
) -> AccessResult:
    """Return all service events owned by the signed customer's instruments."""

    verified = _verified_or_denied(context)
    if isinstance(verified, AccessResult):
        return verified

    # The no-ID route is still customer-scoped. Both relationships are checked
    # inside SQLite: the event and instrument must name the signed customer,
    # and profile fields come from that same customer row. A question cannot
    # supply a different customer or make a foreign event cross into evidence.
    with closing(_connection(db_path)) as connection:
        rows = connection.execute(
            "SELECT s.*, c.customer_name, c.contact_email "
            "FROM service_history AS s "
            "JOIN instruments AS i ON i.instrument_id = s.instrument_id "
            "AND i.customer_id = s.customer_id "
            "JOIN customers AS c ON c.customer_id = s.customer_id "
            "WHERE s.customer_id = ? "
            "ORDER BY s.event_date DESC, s.service_event_id DESC",
            (verified.user_id,),
        ).fetchall()

    # A signed customer with no service events has an allowed empty result.
    # This differs from probing a foreign/missing instrument by ID, which
    # remains a denial in get_service_history.
    return AccessResult(
        "allow",
        verified.trace_id,
        tuple(dict(row) for row in rows),
        tuple(row["service_event_id"] for row in rows),
    )
