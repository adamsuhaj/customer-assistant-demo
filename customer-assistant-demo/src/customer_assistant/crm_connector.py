"""Step 7 - Mark the production CRM connection as unfinished on purpose.

- ``fetch_order`` and ``fetch_service_history`` mark the single-record live
  CRM reads a production adapter would need for targeted order and service
  questions. Step 11's ID-free, customer-wide service history also needs a
  separately authorized account-scoped read in a real connector.
- Both functions raise ``NotImplementedError`` immediately. This demo has no
  CRM endpoint or credentials, and an empty result could be mistaken for a
  genuine "no records" response in a customer conversation.
- The synthetic SQLite adapter remains the only working data source here.
  A later implementation must validate identity and customer entitlement at
  the source, keep customer-wide reads scoped there, bound its network calls,
  and return versioned source references for the same audit and evaluation
  checks used by this demo.
"""

from __future__ import annotations

from typing import NoReturn


# PRODUCTION STUB (CRM CONNECTOR): no credentials, SDK, endpoint, or network
# request belongs here until a reviewed data-product adapter replaces this
# module. This explicit failure prevents a plausible-looking empty result from
# being mistaken for a successful customer lookup.
def fetch_order(*, order_id: str) -> NoReturn:
    """Reject a live order lookup until an authorized CRM adapter is built."""

    # No stubbed row is returned: order absence and connector absence are
    # different outcomes for both the customer and the audit trail.
    raise NotImplementedError(
        "PRODUCTION STUB (CRM CONNECTOR): live order reads are not implemented"
    )


def fetch_service_history(*, instrument_id: str) -> NoReturn:
    """Reject a live service lookup until an authorized CRM adapter is built."""

    # A placeholder service event could falsely imply a repair was completed.
    # Fail visibly until an owner supplies source authorization and lineage.
    raise NotImplementedError(
        "PRODUCTION STUB (CRM CONNECTOR): live service reads are not implemented"
    )


# A future adapter must accept validated OIDC/JWT-derived identity from trusted
# application code, enforce customer ownership at the source for both targeted
# and full-history reads, bound timeouts, and return source versions for the
# same per-action audit and evaluation gates.
