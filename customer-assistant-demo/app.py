"""Step 8 - Present the existing customer assistant in a local Streamlit UI.

- Select Alice or Bob as a demo login and OpenAI or Anthropic as the live model
  route; the shared config loader reads settings without showing API key values.
- Submit each chat prompt once to answer_question(), which owns tool planning, signed
  identity, MCP calls, policy checks, evidence grounding, and audit writes.
- Display the answer with its source IDs, provider/model, and audit trace ID;
  keep separate in-session conversations for Alice and Bob across reruns.
- Step 9 adds ID-free owned order lists through the existing assistant path.
- Step 10 adds the dark continuous chat and a customer-scoped SQLite trace
  history. Expired traces are removed on app reruns after 72 hours; Step 12
  adds local provider-handoff events, now shown in the Audit Log.
- Step 11 adds owned customer links, ID-free service history, and an approved
  public troubleshooting catalog in the existing MCP/orchestrator layers. The
  UI forwards those questions without duplicating their authorization rules.
- Step 12 saves complete chat turns in local SQLite, so the selected customer's
  transcript survives provider switches, browser reconnects, and app restarts.
- On an OpenAI/Claude switch, the outgoing model summarizes audit-verified
  turns for a bounded provider handoff. The next provider receives that memo
  as continuity context, then the normal MCP and policy path rechecks facts.
- The handoff is A2A-style local lineage between model routes of this one
  assistant, not a claim of interoperable A2A agent servers or task transport.
- Step 13 renames the sidebar trace browser Audit Log and shows the bounded
  handoff summary, verified prior turns, provider direction, and source lineage
  on prepared and delivered events under the existing 72-hour retention.
- New chat clears the selected customer's saved conversation and cached
  provider context, while keeping the separate audit history available.
"""

from __future__ import annotations

import asyncio
import csv
import os
import secrets
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st


# Streamlit runs this file directly instead of installing the package. Add the
# same source folder that the notebook uses, without copying application code.
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from customer_assistant.audit import (
    list_recent_traces, purge_expired_traces, read_trace, record_event,
)
from customer_assistant.config import (
    API_KEY_VARIABLES, GatewayConfigurationError, load_gateway_config,
)
from customer_assistant.database import (
    DEFAULT_DB_PATH, DEFAULT_SOURCE_DIR, HEADERS, connect_readonly, load_database,
)
from customer_assistant.conversation import (
    clear_chat_history, load_chat_history, purge_expired_chat, save_chat_turn,
)
from customer_assistant.gateway import LiteLLMGateway
from customer_assistant.handoff import (
    create_handoff, handoff_audit_payload, verified_turns,
)
from customer_assistant.identity import mint_demo_identity
from customer_assistant.orchestrator import answer_question


def _orders_match_source() -> bool:
    """Detect an older local order copy without inspecting private settings."""

    columns = HEADERS["orders.csv"]
    with (DEFAULT_SOURCE_DIR / "orders.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        source_rows = sorted(
            tuple(row[column] for column in columns)
            for row in csv.DictReader(stream)
        )
    with closing(connect_readonly(DEFAULT_DB_PATH)) as connection:
        stored_rows = sorted(
            tuple(row)
            for row in connection.execute(
                "SELECT " + ", ".join(columns) + " FROM orders"
            )
        )
    return source_rows == stored_rows


def _prepare_local_demo() -> None:
    """Refresh changed synthetic orders and keep a process demo signing key."""

    try:
        needs_load = not DEFAULT_DB_PATH.is_file() or not _orders_match_source()
    except (OSError, sqlite3.Error, KeyError):
        needs_load = True
    if needs_load:
        # A presenter may have a SQLite copy from before the ten-order CSV was
        # added. Reload the validated source tables when orders differ; the
        # separate audit and chat tables survive the source refresh.
        load_database(db_path=DEFAULT_DB_PATH)
    if not os.environ.get("DEMO_SIGNING_SECRET"):
        # The signed context must survive the separate MCP subprocess. Keep
        # this key stable for the Streamlit process and never show or log it.
        # The provider config loader reads .env separately for model settings.
        os.environ["DEMO_SIGNING_SECRET"] = secrets.token_hex(32)


def _gateway_for(provider: str) -> LiteLLMGateway:
    """Load the selected .env-backed provider route for a live model answer."""

    # The shared loader merges project .env with process overrides and picks
    # only this provider's key/model. Keep that selection in the existing
    # gateway so the orchestrator and its evidence checks stay unchanged.
    config = load_gateway_config(provider=provider)
    # A missing key is checked only if approved evidence reaches a model call.
    # Local clarifications and denials therefore still work with no API key;
    # ID-free order/service/catalog requests follow their normal MCP routes.
    return LiteLLMGateway(config)


def _apply_chat_style() -> None:
    """Add restrained heading, message, and spacing refinements to the theme."""

    # The theme TOML supplies stable widget colors. This CSS only adjusts
    # layout details Streamlit does not expose there; it contains no content
    # from a customer question or provider response.
    st.markdown(
        """<style>
        .block-container { max-width: 920px; padding-top: 2.5rem; padding-bottom: 7rem; }
        h1, h2, h3 { color: #39A9DF !important; letter-spacing: -0.025em; }
        h1 { font-weight: 720 !important; }
        [data-testid="stSidebar"] { border-right: 1px solid #35404B; }
        [data-testid="stChatMessage"] {
            background: #242A33; border: 1px solid #343D49;
            border-radius: 16px; padding: 0.55rem 0.9rem;
            margin-bottom: 0.7rem;
        }
        [data-testid="stChatInput"] { border-radius: 18px; }
        [data-testid="stExpander"] { border: 1px solid #35404B; border-radius: 12px; }
        </style>""",
        unsafe_allow_html=True,
    )


def _current_timestamp() -> str:
    """Timestamp one chat turn in UTC for three-day SQLite retention."""

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _show_message(message: dict[str, Any]) -> None:
    """Render a saved chat entry without serializing raw tool rows or keys."""

    avatar = "👤" if message["role"] == "user" else "💠"
    with st.chat_message(message["role"], avatar=avatar):
        st.write(message["content"])
        result = message.get("result")
        if result is not None:
            sources = ", ".join(result["source_ids"]) or "(none)"
            st.caption(
                f"Sources: {sources} · Provider/model: "
                f"{result['provider']} / {result['model']} · "
                f"Trace ID: {result['trace_id']}"
            )
        if message.get("notice"):
            # The notice contains only a variable name, never its value.
            st.warning(message["notice"])


def _safe_trace(trace_id: str) -> list[dict[str, Any]]:
    """Show safe lineage and the bounded content of provider handoffs."""

    visible = (
        "timestamp_utc", "user_id", "agent_id", "action", "outcome",
        "evidence_ids", "provider", "model", "prompt_version", "risk_tier",
    )
    projected = []
    for event in read_trace(trace_id, db_path=DEFAULT_DB_PATH):
        shown = {
            **{key: event.get(key) for key in visible},
            "authorized_evidence_rows": len(event["authorized_evidence_snapshot"]),
            "model_answer_recorded": event["response_text"] is not None,
        }
        if event.get("handoff_payload") is not None:
            # Only audit.py's checked packet fields reach this view. Show the
            # direction and state explicitly: a provider switch prepares a
            # packet, while the later model request actually delivers it.
            shown["handoff_stage"] = (
                "delivered" if event["action"].startswith("provider_handoff_received:")
                else "prepared"
            )
            shown["handoff"] = event["handoff_payload"]
        projected.append(shown)
    return projected


def _show_trace_history(customer: str) -> None:
    """List retained traces for the selected signed demo customer only."""

    user_id = mint_demo_identity(customer, db_path=DEFAULT_DB_PATH).user_id
    traces = list_recent_traces(user_id=user_id, db_path=DEFAULT_DB_PATH)
    with st.expander("Audit Log"):
        # The same customer-scoped browser covers ordinary MCP calls and the
        # bounded provider packet. A prepared packet can exist before any new
        # question; its delivered event appears after the receiving model call.
        st.caption("Assistant, MCP, and provider handoff events. Retained for 72 hours.")
        if not traces:
            st.info("No retained traces for this customer yet.")
            return
        # A compact recent list keeps a busy rehearsal from filling the whole
        # sidebar; the selector still offers every retained trace for review.
        st.caption(f"Recent traces · {len(traces)} saved")
        for trace in traces[:8]:
            shown_at = datetime.fromisoformat(
                trace["timestamp_utc"].replace("Z", "+00:00")
            ).strftime("%d %b %H:%M UTC")
            st.caption(
                f"{shown_at} · {trace['trace_id']} · "
                f"{trace['action']} / {trace['outcome']}"
            )
        by_id = {trace["trace_id"]: trace for trace in traces}
        selected = st.selectbox(
            "Inspect saved trace",
            tuple(by_id),
            format_func=lambda trace_id: (
                f"{by_id[trace_id]['timestamp_utc']} · {trace_id[:10]}"
            ),
            key=f"selected_trace_{customer}",
        )
        if selected in by_id:
            # Ordinary audit rows show counts and lineage. Handoff rows also
            # show the checked memo and verified turns sent as continuity.
            st.json(_safe_trace(selected))


def _handoff_for(
    customer: str, customer_id: str, agent_id: str, provider: str,
    history: list[dict[str, Any]],
):
    """Summarize one provider transition and reuse it across widget reruns."""

    latest = next(
        (message["result"] for message in reversed(history)
         if message.get("role") == "assistant" and isinstance(message.get("result"), dict)),
        None,
    )
    if latest is None or latest.get("provider") == provider:
        return None
    previous_provider = latest["provider"]
    # Key the packet to the latest saved turn. Opening a trace, changing a
    # widget, or rerunning Streamlit must not pay for another summary or append
    # duplicate audit events. A new answer changes this key naturally.
    cache_key = (customer, previous_provider, provider, latest["trace_id"])
    packets = st.session_state.setdefault("provider_handoffs", {})
    if cache_key in packets:
        return packets[cache_key]

    try:
        outgoing_gateway = _gateway_for(previous_provider)
    except GatewayConfigurationError:
        # A broken outgoing route cannot erase the conversation. The handoff
        # module can still make a local recap from verified displayed turns.
        outgoing_gateway = None
    with st.spinner("Handing over the conversation..."):
        packet = asyncio.run(create_handoff(
            history, customer_id=customer_id,
            from_provider=previous_provider, to_provider=provider,
            gateway=outgoing_gateway, db_path=DEFAULT_DB_PATH,
        ))
    if packet is not None:
        # The handoff has its own trace and is marked prepared at the switch.
        # Cache it only after the durable audit write succeeds, so a failed
        # write cannot leave a packet available for an unlogged delivery.
        record_event(
            db_path=DEFAULT_DB_PATH, trace_id=packet.handoff_id,
            user_id=customer_id, agent_id=agent_id,
            action=f"provider_handoff_to_{provider}",
            outcome=packet.summary_origin,
            evidence_ids=packet.source_ids,
            provider=previous_provider,
            model=(outgoing_gateway.config.model if outgoing_gateway is not None
                   else latest.get("model")),
            prompt_version="step12-provider-handoff-v1",
            risk_skill="provider_handoff",
            handoff_payload=handoff_audit_payload(packet),
        )
    packets[cache_key] = packet
    return packet


def _start_new_chat(customer: str, customer_id: str) -> None:
    """Clear saved turns and current-session context for one trusted customer."""

    clear_chat_history(customer_id, db_path=DEFAULT_DB_PATH)
    st.session_state.setdefault("chat_history_by_customer", {}).pop(customer, None)
    st.session_state["chat_history"] = []
    packets = st.session_state.setdefault("provider_handoffs", {})
    for cache_key in list(packets):
        if cache_key[0] == customer:
            del packets[cache_key]
    generations = st.session_state.setdefault("chat_generation_by_customer", {})
    previous = generations.get(customer, 0)
    st.session_state.pop(f"chat_input_{customer}_{previous}", None)
    # A fresh widget key also discards an unsent draft in the browser.
    generations[customer] = previous + 1


def main() -> None:
    st.set_page_config(
        page_title="Your friendly AI assistant", page_icon="💠",
        layout="centered", initial_sidebar_state="expanded",
    )
    _apply_chat_style()

    try:
        _prepare_local_demo()
        # Cleanup runs on every load/rerun, not on a background timer. A fresh
        # session therefore sees only traces retained within the last 72 hours.
        purge_expired_traces(db_path=DEFAULT_DB_PATH)
        purge_expired_chat(db_path=DEFAULT_DB_PATH)
    except Exception:
        st.error("The local demo data or trace history could not be prepared. Run the Step 1 loader.")
        st.stop()

    with st.sidebar:
        st.header("Settings")
        # Friendly labels keep the controls readable while the selected
        # values remain the exact IDs expected by identity and model config.
        customer = st.selectbox(
            "Demo customer", ("alice", "bob"), format_func=str.title,
        )
        # Restore the last used provider for this customer on a new browser
        # connection. Merely reconnecting must not cause a paid reverse
        # handoff because the selector fell back to an unrelated default.
        identity = mint_demo_identity(customer, db_path=DEFAULT_DB_PATH)
        history = load_chat_history(identity.user_id, db_path=DEFAULT_DB_PATH)
        latest_result = next(
            (message["result"] for message in reversed(history)
             if isinstance(message.get("result"), dict)),
            None,
        )
        last_provider = latest_result.get("provider") if latest_result else None
        provider_key = f"provider_{customer}"
        if provider_key not in st.session_state:
            st.session_state[provider_key] = (
                "anthropic" if last_provider == "anthropic" else "openai"
            )
        provider = st.selectbox(
            "Provider", ("openai", "anthropic"),
            format_func={"openai": "OpenAI", "anthropic": "Claude"}.get,
            key=provider_key,
        )
        if st.button(
            "New chat", key=f"new_chat_{customer}", width="stretch",
            help=f"Clear {customer.title()}'s saved chat and start fresh. Audit Log stays available.",
        ):
            try:
                _start_new_chat(customer, identity.user_id)
            except Exception:
                st.error("The chat could not be cleared. Please try again.")
            else:
                st.rerun()

    # SQLite is the source of chat continuity, including a browser reconnect.
    # The current selector is resolved to a trusted customer ID before any
    # history is read; provider changes never select a different partition.
    histories = st.session_state.setdefault("chat_history_by_customer", {})
    histories[customer] = history
    st.session_state["chat_history"] = history

    # The live promotion-gate demo runs separately in the notebook or CLI.
    heading, mascot = st.columns([5, 1], vertical_alignment="center", wrap=False)
    with heading:
        st.title("Your friendly AI assistant")
        st.caption("Ask about your orders, service history, or pump warnings. Answers use verified demo sources.")
    with mascot:
        robot_image = PROJECT_ROOT / "assets" / "friendly-robot.png"
        if robot_image.is_file():
            st.image(str(robot_image), width="stretch")
    handoff = _handoff_for(
        customer, identity.user_id, identity.agent_id, provider, history,
    )
    if handoff is not None:
        provider_labels = {"openai": "OpenAI", "anthropic": "Claude"}
        st.caption(
            f"{provider_labels[handoff.from_provider]} handed over the conversation to "
            f"{provider_labels[handoff.to_provider]} ({handoff.summary_origin} recap). "
            "Current answers still check fresh sources."
        )

    for message in history:
        _show_message(message)

    generation = st.session_state.get("chat_generation_by_customer", {}).get(customer, 0)
    prompt = st.chat_input(
        "Message your assistant", key=f"chat_input_{customer}_{generation}",
    )
    if prompt:
        recent_turns = verified_turns(
            history, customer_id=identity.user_id, db_path=DEFAULT_DB_PATH,
        )
        # st.chat_input returns text only on its submission rerun. Calling the
        # orchestrator here prevents control changes and ordinary reruns from
        # starting another MCP stdio subprocess for the same question.
        created_at = _current_timestamp()
        user_message = {
            "role": "user", "content": prompt, "created_at_utc": created_at,
        }
        history.append(user_message)
        _show_message(user_message)
        try:
            gateway = _gateway_for(provider)
            with st.spinner("Checking authorized evidence..."):
                result = asyncio.run(
                    answer_question(
                        customer, prompt, gateway, db_path=DEFAULT_DB_PATH,
                        recent_turns=recent_turns, handoff=handoff,
                    )
                )
            assistant_message = {
                "role": "assistant", "content": result.answer_text,
                "result": result.as_dict(), "created_at_utc": created_at,
            }
            if not gateway.config.api_key:
                # Identity policy still runs locally. Tool selection and
                # grounded answers both need the selected provider's key.
                key_name = API_KEY_VARIABLES[provider]
                assistant_message["notice"] = (
                    f"Set {key_name} in the project .env or process environment "
                    "for live tool selection and model answers."
                )
        except GatewayConfigurationError:
            assistant_message = {
                "role": "assistant",
                "content": "Model route settings are invalid. Check the selected provider's .env or process settings.",
                "created_at_utc": created_at,
            }
        except Exception:
            # Transport/provider exceptions may carry request or credential
            # details. Keep the UI response generic and do not log them.
            assistant_message = {
                "role": "assistant",
                "content": "The request could not be completed. Check the local demo setup.",
                "created_at_utc": created_at,
            }
        history.append(assistant_message)
        # Commit the visible pair after the answer is known. Reloading the app
        # or selecting the other provider will read the same customer history.
        # Storage validates public result fields and refuses recognizable keys.
        save_chat_turn(
            identity.user_id, user_message, assistant_message, provider,
            db_path=DEFAULT_DB_PATH,
        )
        _show_message(assistant_message)

    with st.sidebar:
        st.divider()
        _show_trace_history(customer)


if __name__ == "__main__":
    main()
