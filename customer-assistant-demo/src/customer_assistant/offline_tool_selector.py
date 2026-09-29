"""Deterministic tool selection for explicit offline rehearsals and tests.

This module is used only by FakeModelGateway. A live provider failure never
falls back to these keyword rules or fabricates a successful model decision.
"""
from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from .gateway import ToolCall, ToolSelectionResult

ORDER_ID = re.compile(r"\bDEMO-ORD-\d+\b", re.IGNORECASE)
INSTRUMENT_ID = re.compile(r"\bDEMO-INS-\d+\b", re.IGNORECASE)
MODEL_CODE = re.compile(r"\b\d{4}\b")

def _offline_route_question(
    question: str, instrument_model: str,
) -> tuple[str | None, dict[str, str | bool | None], str | None, str | None]:
    """Deterministic rehearsal only; live requests use native model tool calls."""

    lower = question.casefold()
    if re.search(r"\b(?:datasets?|databases?|tables?)\b", lower):
        # This is a reviewed schema description, not a query over private
        # rows. Evaluation fixtures are separate from customer evidence.
        return None, {}, "allow", (
            "This demo links customers, instruments, orders, and service "
            "history. Orders and instruments use customer_id; service events "
            "use customer_id and instrument_id. It also has approved public "
            "troubleshooting articles. Private records are limited to the "
            "selected demo customer."
        )

    # Multiple record IDs make the target ambiguous. Ask for one target so
    # neither a private read nor its eventual citation uses the wrong record.
    order_ids = set(match.upper() for match in ORDER_ID.findall(question))
    instrument_ids = set(match.upper() for match in INSTRUMENT_ID.findall(question))
    if len(order_ids) + len(instrument_ids) > 1:
        return None, {}, "no_match", "Please ask about one order or instrument at a time."
    if order_ids:
        # An explicit order ID has priority; the route supplies the ID only.
        # The private tool gets signed user context from the trusted client.
        return "get_order_status", {"order_id": next(iter(order_ids))}, None, None

    if (re.search(r"\borders\b", lower) or "order history" in lower
            or "order statuses" in lower):
        # A plural/list request is scoped by the signed demo identity in the
        # private tool; the question supplies no customer ID or authority.
        # An instrument-filtered order list is not implemented, so do not
        # silently return every order for a narrower request.
        if instrument_ids:
            return None, {}, "no_match", (
                "Please ask for your full order list or provide an order ID."
            )
        active_only = bool(re.search(r"\bactive\b", lower)) and "all statuses" not in lower
        return "list_customer_orders", {"active_only": active_only}, None, None
    if re.search(
        r"\b(?:customer\s+(?:ids?|names?)|(?:contact\s+)?e-?mail|"
        r"organization|organisation|my\s+name|my\s+account|who\s+am\s+i)\b",
        lower,
    ):
        # The existing signed order-list tool returns joined customer facts.
        # This answers self-account questions without accepting a prompt-
        # supplied customer ID or creating a general customer lookup tool.
        return "list_customer_orders", {"active_only": False}, None, None
    # An order request without its ID must not fall through to a different
    # private skill just because it mentions an instrument.
    if "order" in lower or "tracking" in lower:
        return None, {}, "no_match", "Please provide the demo order ID."
    public_terms = (
        "pressure", "pump", "troubleshoot", "trouble-shoot", "warning",
        "error", "guidance",
    )
    service_terms = ("service", "visit", "repair", "technician", "maintenance")
    if instrument_ids and (
        any(term in lower for term in service_terms)
        or not any(term in lower for term in public_terms)
    ):
        # A supplied instrument ID narrows the signed customer's service read.
        # A bare symptom question can use public articles without reading
        # private customer records.
        return "get_service_history", {"instrument_id": next(iter(instrument_ids))}, None, None

    if any(term in lower for term in service_terms) and (
        "service" in lower or "visit" in lower or "technician" in lower
        or "maintenance history" in lower
    ):
        # No instrument ID means *all* events for this signed customer. The
        # MCP policy still filters both the service and parent instrument.
        return "get_service_history", {"instrument_id": None}, None, None

    if any(term in lower for term in public_terms):
        # Instrument/order IDs contain four digits too. Remove those record
        # identifiers before interpreting a number as a pump model code.
        model_words = ORDER_ID.sub("", INSTRUMENT_ID.sub("", question))
        codes = set(MODEL_CODE.findall(model_words))
        if len(codes) > 1:
            return None, {}, "no_match", "Please specify one instrument model."
        # The notebook's synthetic model supplies context when the question
        # says only "Pressure Below Lower Limit". A different numeric code in
        # the question must not silently retrieve the 1260 manual.
        requested_model = instrument_model
        if codes and next(iter(codes)) not in MODEL_CODE.findall(instrument_model):
            requested_model = f"{next(iter(codes))} Pump"
        # A broad search has no exact warning or model requirement. Let the
        # MCP public tool list approved cards rather than inventing a symptom
        # from filler words such as "show troubleshooting guidance".
        generic_words = {
            "about", "all", "and", "any", "are", "articles", "available",
            "can", "could", "do", "does", "errors", "find", "for", "get",
            "guidance", "have", "help", "information", "is", "list", "manual",
            "me", "my", "need", "on", "please", "provide", "public", "pump",
            "search", "show", "some", "support", "the", "there", "topics",
            "troubleshoot", "troubleshooting", "trouble", "shooting", "want",
            "warnings", "warning", "what", "which", "with", "you",
        }
        specific_words = {
            word for word in re.findall(r"[a-z0-9]+", lower)
            if word not in generic_words and not word.isdigit()
        }
        if not specific_words:
            return "search_troubleshooting", {
                "model": None, "symptom": None,
            }, None, None
        return "search_troubleshooting", {
            "model": requested_model, "symptom": question,
        }, None, None

    if any(term in lower for term in service_terms):
        # Catch looser repair or maintenance wording after the public-guidance
        # branch. With no instrument ID, use only this signed customer's full
        # service history; this keyword route does not interpret every possible
        # service-related sentence.
        return "get_service_history", {"instrument_id": None}, None, None
    return None, {}, "unsupported", (
        "I can help with a demo order ID, instrument service history, or a "
        "pump warning. Please include the relevant ID or message."
    )



def select_tools(
    messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]],
) -> ToolSelectionResult:
    """Rehearse the native selection contract without calling a provider."""

    payload = json.loads(str(messages[-1]["content"]))
    route = payload["selected_route"]
    if payload.get("current_tool_results"):
        return ToolSelectionResult(route["provider"], route["model"], ())
    skill, arguments, outcome, clarification = _offline_route_question(
        payload["question"], payload["instrument_model_context"],
    )
    calls = (ToolCall("offline-call-1", skill, arguments),) if skill else ()
    return ToolSelectionResult(
        route["provider"], route["model"], calls, clarification if outcome else None,
    )
