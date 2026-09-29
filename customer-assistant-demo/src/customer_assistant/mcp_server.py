"""Step 3 - Expose checked customer data and public guidance as MCP tools.

- Register four read-only FastMCP tools on a separate stdio server process.
  Its protocol output is machine-readable; ordinary prints would break calls.
- ``get_order_status`` returns a single order only after Step 2 verifies the
  signed customer and the order/instrument ownership relationship in SQLite.
- Step 9 ``list_customer_orders`` handles requests without an order ID. It
  returns all of the signed customer's orders, or just active orders, in a
  repeatable order for answers and citations.
- ``get_service_history`` accepts an owned instrument ID. Step 11 added the
  no-ID path, which retrieves history across all of the signed customer's
  instruments while applying the same ownership boundary.
- ``search_troubleshooting`` is public but restricted to approved tier 0
  articles. A model and symptom produce a targeted match; Step 11 added an
  empty-query path that lists the approved catalog.
- The private tools delegate authorization to policy.py. The public tool
  reads only approved article rows, never orders or service records.
"""

from __future__ import annotations

import os
import re
import sqlite3
from contextlib import closing
from pathlib import Path

from fastmcp import FastMCP

from .database import DEFAULT_DB_PATH, connect_readonly
from .policy import get_order_status as read_order
from .policy import get_service_history as read_service
from .policy import list_customer_service_history as read_customer_service_history
from .policy import list_customer_orders as read_customer_orders


# The server is a separate trust boundary from the assistant orchestrator.
# Mask unexpected exceptions so a database path or record cannot leak in an
# error message; keep expected authorization denials as structured results so
# the caller can audit the decision without seeing denied evidence.
# Three retrieval skills share one MCP protocol via four tool operations.
mcp = FastMCP(
    "Customer Assistant Demo",
    mask_error_details=True,
    strict_input_validation=True,
)

MODEL_GENERIC_WORDS = frozenset(
    {"agilent", "infinitylab", "infinity", "lc", "series", "pump", "pumps"}
)
QUESTION_FILLER_WORDS = frozenset(
    {"the", "and", "for", "with", "what", "does", "mean", "warning", "message"}
)


def _db_path() -> Path:
    # The trusted client selects the local database through the subprocess
    # environment. It is not a tool argument, so model output cannot redirect
    # these queries to an arbitrary file.
    return Path(os.environ.get("DEMO_DB_PATH", DEFAULT_DB_PATH))


def _terms(value: str) -> set[str]:
    """Tokenize article/request text for a deterministic, case-free match."""

    return {word for word in re.findall(r"[a-z0-9]+", value.casefold()) if len(word) >= 3}


@mcp.tool(name="get_order_status", annotations={"readOnlyHint": True})
def get_order_status(order_id: str, identity_context: dict[str, str | int]) -> dict:
    """Return one order if the application-signed customer owns it.

    ``identity_context`` comes from trusted application code, not the model.
    Policy verifies its signature and filters by customer inside SQLite.
    """

    return read_order(identity_context, order_id, db_path=_db_path()).as_dict()


@mcp.tool(name="list_customer_orders", annotations={"readOnlyHint": True})
def list_customer_orders(
    identity_context: dict[str, str | int], active_only: bool = False
) -> dict:
    """Return the signed customer's orders, optionally only active ones.

    The tool accepts no customer ID or order ID. Policy verifies the
    application-minted context and applies customer ownership inside SQLite.
    ``active_only`` limits statuses to processing/in transit; false includes
    delivered orders as well.
    """

    return read_customer_orders(
        identity_context, active_only=active_only, db_path=_db_path()
    ).as_dict()


@mcp.tool(name="get_service_history", annotations={"readOnlyHint": True})
def get_service_history(
    identity_context: dict[str, str | int], instrument_id: str | None = None
) -> dict:
    """Return one owned instrument's events, or all signed-customer events.

    The missing instrument is intentional: policy derives the customer from
    the signed context instead of accepting a customer ID from the question.
    """

    # The no-ID path still runs through signed policy. A prompt cannot select
    # another customer's service history by supplying a customer ID here.
    if instrument_id is None:
        return read_customer_service_history(
            identity_context, db_path=_db_path()
        ).as_dict()
    return read_service(identity_context, instrument_id, db_path=_db_path()).as_dict()


@mcp.tool(name="search_troubleshooting", annotations={"readOnlyHint": True})
def search_troubleshooting(
    model: str | None = None, symptom: str | None = None
) -> dict:
    """Find a public article, or list the approved catalog with no terms.

    A targeted search needs both a sufficiently specific model and symptom.
    Public access means no customer signature is needed; article approval is
    still enforced by the SQLite predicates below.
    """

    # The catalog route is deliberately available only when *both* search
    # terms are absent. A partial or overly broad targeted request should not
    # silently receive an unrelated article as troubleshooting advice.
    if not model and not symptom:
        with closing(connect_readonly(_db_path())) as connection:
            connection.row_factory = sqlite3.Row
            articles = connection.execute(
                "SELECT * FROM troubleshooting_articles "
                "WHERE access_tier = '0' "
                "AND content_status = 'synthetic_summary_with_public_source' "
                "ORDER BY article_id"
            ).fetchall()
        if not articles:
            return {
                "outcome": "no_match", "trace_id": None, "rows": [],
                "source_ids": [], "reason": "no_public_article_match",
            }
        rows = [dict(article) for article in articles]
        return {
            "outcome": "allow", "trace_id": None, "rows": rows,
            "source_ids": [row["article_id"] for row in rows], "reason": None,
        }

    # Strip words such as "pump" and "warning" because they describe many
    # articles. If either side becomes empty, refuse to guess a target article.
    model_terms = _terms(model or "") - MODEL_GENERIC_WORDS
    symptom_terms = _terms(symptom or "") - QUESTION_FILLER_WORDS
    if not model_terms or not symptom_terms:
        return {
            "outcome": "no_match", "trace_id": None, "rows": [],
            "source_ids": [], "reason": "no_public_article_match",
        }

    # Public guidance needs no signed identity, but it is still constrained:
    # only published synthetic summaries at tier 0 can become evidence. The
    # query never touches orders, service history, or evaluation candidates.
    with closing(connect_readonly(_db_path())) as connection:
        connection.row_factory = sqlite3.Row
        articles = connection.execute(
            "SELECT * FROM troubleshooting_articles "
            "WHERE access_tier = '0' "
            "AND content_status = 'synthetic_summary_with_public_source'"
        ).fetchall()

    ranked: list[tuple[int, dict[str, str]]] = []
    for article in articles:
        row = dict(article)
        family_terms = _terms(row["model_family"])
        model_codes = {term for term in family_terms if any(char.isdigit() for char in term)}
        # Shared names such as "Infinity II Pump" are too broad to identify a
        # manual. When the article has a numeric model code, require that code
        # in the request before showing guidance for the article.
        if model_codes and not model_codes.intersection(model_terms):
            continue
        # All remaining model terms must match the article family. This keeps
        # a nearby model from inheriting guidance just because it shares a
        # broad manufacturer or product-family name.
        if not model_terms.issubset(family_terms):
            continue
        # Once the model matches, score actual symptom overlap. Otherwise a
        # pressure question could retrieve an unrelated status card for the
        # same instrument family and produce a plausible but wrong answer.
        guidance_terms = _terms(row["symptom"] + " " + row["article_summary"])
        overlap = symptom_terms & guidance_terms
        if overlap:
            ranked.append((len(overlap), row))

    if not ranked:
        return {
            "outcome": "no_match", "trace_id": None, "rows": [],
            "source_ids": [], "reason": "no_public_article_match",
        }

    highest_score = max(score for score, _ in ranked)
    # Return every best-scoring card in a stable order. The caller can then
    # cite each supplied article ID, and repeated demo runs stay reproducible.
    matches = sorted(
        (row for score, row in ranked if score == highest_score),
        key=lambda row: row["article_id"],
    )
    return {
        "outcome": "allow",
        "trace_id": None,
        "rows": matches,
        "source_ids": [row["article_id"] for row in matches],
        "reason": None,
    }


if __name__ == "__main__":
    # FastMCP speaks JSON-RPC on stdout. A banner or debug print would corrupt
    # the protocol stream before the client can parse the first tool result.
    mcp.run(transport="stdio", show_banner=False)
