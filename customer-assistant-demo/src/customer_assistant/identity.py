"""Step 2 - Carry the selected demo customer's identity to private tools.

- Resolve the presenter's Alice/Bob selection against the customers table,
  then sign its customer_id in trusted application code. This establishes
  which synthetic customer the demo may read; selecting an alias in the UI
  is not authentication of a real person.
- Include a fixed assistant agent ID, a fresh per-request trace ID, and a
  one-hour expiry in the HMAC-signed context. The secret comes from the
  process environment and is never sent as a model prompt or tool argument.
- Verify the exact claim set, types, expiry, agent ID, and signature at every
  private policy read after an MCP hop. A forged or stale context cannot be
  used to select Bob's rows while Alice is active.
- Keep the demo login resolver as a production replacement point. Real
  customer data would require validated OIDC/JWT login claims and managed
  signing keys before this ownership policy could be used in production.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import time
import uuid
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .database import DEFAULT_DB_PATH, connect_readonly


# This value names the *application* making requests. Keeping it fixed here
# prevents a prompt or tool argument from impersonating another trusted agent.
AGENT_ID = "customer_assistant_v1"
IDENTITY_TTL_SECONDS = 3600
# Only these claims cross the MCP boundary. An exact field set prevents an
# unexpected caller-supplied claim, or the signing secret, from being treated
# as part of the identity accepted by downstream tools.
IDENTITY_FIELDS = frozenset(
    {"user_id", "agent_id", "trace_id", "issued_at", "expires_at", "signature"}
)


class IdentityConfigurationError(RuntimeError):
    """The application has no signing secret configured."""


class UnknownDemoLogin(ValueError):
    """A simulated login does not match a customer in the demo copy."""


class InvalidIdentity(ValueError):
    """The identity context is malformed, expired, or not signed by this app."""


@dataclass(frozen=True)
class DemoIdentity:
    user_id: str
    agent_id: str
    trace_id: str
    issued_at: int
    expires_at: int
    signature: str

    def as_dict(self) -> dict[str, str | int]:
        """Return only the fields needed for a tool hop; never the secret."""

        return {
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "trace_id": self.trace_id,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "signature": self.signature,
        }


def _signing_secret() -> bytes:
    # Private-data checks depend on a secret controlled by the application.
    # A built-in fallback would let anyone who sees this source forge a user.
    value = os.environ.get("DEMO_SIGNING_SECRET")
    if not value:
        raise IdentityConfigurationError("Set DEMO_SIGNING_SECRET before using private data")
    return value.encode("utf-8")


def _message(fields: Mapping[str, Any]) -> bytes:
    # JSON key order can change across the MCP process boundary. Canonical
    # serialization makes minting and verification sign the same bytes. The
    # signature field itself is excluded to avoid signing its own value.
    payload = {key: fields[key] for key in IDENTITY_FIELDS if key != "signature"}
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def mint_demo_identity(
    demo_login: str,
    *,
    db_path: Path = DEFAULT_DB_PATH,
) -> DemoIdentity:
    """Resolve a presenter-selected alias and mint the app's signed tool context.

    PRODUCTION STUB (OIDC/JWT): this  local alias lookup stands in for
    validated user login claims. Replace it before connecting real customer data.
    The model must never supply ``demo_login`` or receive the signing secret.
    """

    secret = _signing_secret()
    # The  caller selects a demo alias, then this lookup resolves its
    # customer ID through the customers table. The UI selection is a demo
    # fixture choice, not a customer-authentication check. Only the resolved
    # ID is signed; the model sees approved evidence after the policy read.
    with closing(connect_readonly(db_path)) as connection:
        row = connection.execute(
            "SELECT customer_id FROM customers WHERE demo_login = ?", (demo_login,)
        ).fetchone()
    if row is None:
        raise UnknownDemoLogin("Unknown demo login")

    # Each request gets its own trace for audit correlation, even when Alice
    # asks twice. Expiry limits the usefulness of a copied signed context.
    issued_at = int(time.time())
    fields = {
        "user_id": row[0],
        "agent_id": AGENT_ID,
        "trace_id": uuid.uuid4().hex,
        "issued_at": issued_at,
        "expires_at": issued_at + IDENTITY_TTL_SECONDS,
    }
    signature = hmac.new(secret, _message(fields), hashlib.sha256).hexdigest()
    return DemoIdentity(**fields, signature=signature)


def verify_demo_identity(
    context: DemoIdentity | Mapping[str, Any],
) -> DemoIdentity:
    """Verify the context at every private data read, including MCP tool hops."""

    secret = _signing_secret()
    if isinstance(context, DemoIdentity):
        identity = context
    elif isinstance(context, Mapping) and set(context) == IDENTITY_FIELDS:
        # The MCP JSON transport converts the dataclass to a plain mapping.
        # Reconstruct it only when all intended claims, and no extras, arrive;
        # matching a Python type alone would not establish trust.
        try:
            identity = DemoIdentity(**context)
        except TypeError as exc:
            raise InvalidIdentity("Malformed identity context") from exc
    else:
        raise InvalidIdentity("Malformed identity context")

    # Check types and fixed claims before the HMAC. In particular, Python's
    # bool is an int subclass, so exact int checks keep true/false from being
    # accepted as timestamps in a JSON-derived context.
    if (
        not isinstance(identity.user_id, str)
        or not identity.user_id
        or identity.agent_id != AGENT_ID
        or identity.user_id == identity.agent_id
        or not isinstance(identity.trace_id, str)
        or not isinstance(identity.signature, str)
        or re.fullmatch(r"[0-9a-f]{64}", identity.signature) is None
        or type(identity.issued_at) is not int
        or type(identity.expires_at) is not int
    ):
        raise InvalidIdentity("Malformed identity context")
    try:
        uuid.UUID(identity.trace_id)
    except ValueError as exc:
        raise InvalidIdentity("Malformed trace ID") from exc

    # Reject malformed or expired claims before policy can use their user ID.
    # The short lifetime is part of the signed context, not a caller override.
    now = int(time.time())
    if not (
        identity.issued_at <= now < identity.expires_at
        and 0 < identity.expires_at - identity.issued_at <= IDENTITY_TTL_SECONDS
    ):
        raise InvalidIdentity("Expired or invalid identity lifetime")

    expected = hmac.new(secret, _message(identity.as_dict()), hashlib.sha256).hexdigest()
    # Constant-time comparison avoids revealing signature differences through
    # comparison timing when an invalid context reaches this boundary.
    if not hmac.compare_digest(identity.signature, expected):
        raise InvalidIdentity("Invalid identity signature")
    return identity
