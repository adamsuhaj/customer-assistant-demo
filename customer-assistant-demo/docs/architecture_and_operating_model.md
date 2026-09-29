# Step 7 - Target architecture and operating model

- Map the notebook to the intended runtime, orchestrator, MCP tool, governed data-product, and model-gateway boundaries.
- Explain how registered tool and data tiers determine inherited risk and which gate owners review promotion.
- Identify the OIDC/JWT and CRM production stubs, likely failure modes, staffing sequence, and assumptions that could change the plan.
- Distinguish the Step 12 local, audit-linked provider handoff from future interoperable A2A between independently owned agents.
- Step 9 extends the private order boundary with a signed customer-scoped list
  and checks every listed status/citation before a complete answer is shown.
- Step 10 adds ten synthetic orders, a continuous local chat, and a customer-scoped
  trace browser that clears whole traces after 72 hours on the next UI run.
- Step 11 joins customer identity fields only through authorized private reads,
  adds signed full-service and approved public-catalog routes, and rejects
  incomplete status, date, service, or article lists before display.
- Step 12 retains completed per-customer chat turns in SQLite for 72 hours and
  passes a bounded, audit-verified recap from the outgoing to incoming model route.
- Step 13 makes the bounded handoff packet inspectable in the customer-scoped
  Audit Log, with prepared and delivered events under the same 72-hour retention.
- Step 14 lets the selected model propose reviewed MCP tools and arguments,
  while trusted code validates and executes them within a bounded loop.

## Target architecture

```text
Customer / validated OIDC login [production stub]
    -> Streamlit chat -> customer-scoped SQLite chat turns (72 h)
    -> orchestrator (preflight, signed user + distinct agent identity)
    -> MCP list_tools -> reviewed descriptions + business-argument schemas
    -> model gateway -> OpenAI or Anthropic proposes native tool calls
    -> orchestrator validates arguments and attaches signed identity
    -> MCP client -> order status/list | service | public troubleshooting tools
                     -> governed data products / CRM adapter [production stub]
    -> projected authorized evidence -> next model planning round
       (up to 3 rounds / 4 tool calls; stop collecting or fail safely)
    -> model answer -> application grounding and citation checks
    -> cited answer, trace ID, per-action SQLite audit, evaluation harness

On provider switch: audit-verified prior turns -> outgoing model recap (or local
fallback) -> incoming model context -> fresh orchestrator/MCP authorization.
Across teams later: interoperable A2A for independently owned agents.
```

The working laptop slice uses synthetic CSVs copied into SQLite, a simulated Alice/Bob login, HMAC signed tool context, four FastMCP tools over stdio, and one provider-neutral gateway. The Streamlit app is a local presentation layer over the same orchestrator and audit reader; it owns no separate tool or entitlement logic. Step 12 stores completed chat pairs in SQLite by the verified customer ID, so Alice and Bob keep separate conversations across provider switches, browser reconnects, and app restarts. The Step 9 order-list tool reads only rows owned by the signed customer, with an optional active-status filter applied inside SQLite. Step 10 expands the fixture to five orders per customer, three active each. The data-product boundary enforces customer ownership before private rows leave SQLite. Only reviewed evidence fields reach a model. Step 14 replaces keyword routing with native model tool selection. The model sees the MCP catalog's reviewed names, descriptions, and JSON business-argument schemas; signed identity and customer-selection fields are withheld. The orchestrator rejects unregistered tools, identity injection, schema-invalid arguments, repeated requests, and requests beyond its loop bounds before execution. It supplies the signed customer context itself, then returns only projected current evidence for another planning round. A planning response without calls either stops evidence collection or triggers an approved static clarification; arbitrary planner prose is discarded and only the final evidence-backed answer goes through the normal grounding checks. Runtime, orchestration, and MCP schemas remain the same when the configured model provider changes; Step 12 adds a separate audited handoff action. A provider SDK called inside an MCP skill, provider-specific routing in the orchestrator, or model-supplied identity would break those boundaries.

Step 11 makes the customer-to-record relationships explicit in authorized evidence. The order query joins `orders.customer_id` to `customers.customer_id` and requires its linked instrument to have that same owner; the service query also requires the event and instrument to belong to the signed customer. The joined `customer_name` is a synthetic organization name and `contact_email` is a synthetic demo address. A prompt naming Bob while Alice is selected cannot choose Bob's rows: the orchestrator rejects explicit other-customer names or IDs before any private MCP or model call, and the SQL boundary still checks the signed identity. The MCP service tool can return all events across the signed customer's instruments when no instrument ID is supplied. Public troubleshooting can return the full approved tier 0 article catalog when no model or symptom is supplied; this path never reads private records. A dataset question receives a fixed relationship description, not a general database query.

The current fixture has six Alice orders and five Bob orders. The extra `DEMO-ORD-1009` / `DEMO-SVC-1009` records deliberately conflict on receipt, installation, and shipping dates for a demo discussion. Source reconciliation belongs in a live governed data product; the assistant retains the retrieved facts and citations.

The order-list query sorts by `orders.created_on` newest first. That column is the order date; `estimated_delivery` is a separate forecast. The answer checker requires every authorized order to have its own cited line with the exact status, and, for a date request, the actual `created_on` value in source order. A single-order answer must state the current row's status without a conflicting status. Full service and public-article lists must cite every returned source ID. If a provider omits a required citation or fails those order status/date checks, the answer is withheld with a safe reply and the rejected response is recorded in the audit trail.

On an OpenAI/Claude change, `handoff.py` selects up to six recent completed turns whose displayed answers match allowed audit events for that same customer. It asks the outgoing provider to summarize those question/answer pairs and source/trace IDs, or uses a local recap of the latest verified turn if that provider fails. The bounded packet carries the origin and lineage; it excludes raw MCP evidence snapshots and credentials. `orchestrator.py` accepts it only for the signed customer and receiving provider, passes it as conversation context, and performs a fresh signed MCP read for the next factual answer. The handoff does not grant access to another customer's records.

The UI's **Audit Log** shows customer-scoped local assistant, MCP, and provider-handoff audit events. It shows the latest eight summaries and lets the presenter select any retained trace ID. A provider switch records a tier 3 **prepared** handoff event, and the first successful, verified receiving tool-selection response records a tier 3 **delivered** event, before tool retrieval and final answer completion; both share the handoff ID. Each event stores a bounded packet with the provider direction, model or fallback summary origin, summary, up to six audit-verified question/answer turns, and their source and trace IDs. It stores no raw MCP evidence rows or credentials. Delivery records receipt of the recap by the planner, so a later tool denial or answer failure does not undo it. A prepared packet that never reaches a successful, verified receiving planning response has no delivered event. The UI purges whole traces 72 hours after their latest event on its next run or rerun, including any packet attached to those events; `conversation.py` purges complete chat pairs older than 72 hours on that same schedule. The provider switch alone does not erase chat. SQLite audit and chat history survive a source-data refresh because the loader replaces source tables separately; an intentional deletion of the generated SQLite database does erase both. The evaluation harness remains a notebook/CLI gate and is not part of the chat UI. Its intentional EVAL-005 failure still blocks promotion.

The local prototype has no cloud runtime, production identity provider, CRM connection, or durable shared event service. `identity.py` is the marked **PRODUCTION STUB (OIDC/JWT)**; `crm_connector.py` is the marked **PRODUCTION STUB (CRM CONNECTOR)** and raises before network activity. Production would need verified login claims, service authentication, customer entitlements at each data product, governed source versions, retention rules, and an audit store independent of the demo database.

## Risk inheritance and promotion

The harness assigns trusted catalog tiers: **0** public approved troubleshooting material; **1** private customer order data; **2** private service history and its instrument context; **3** a cross-provider recap of previously answered customer information. It computes a composition's tier as `max(tier of each tool and record touched)`. A public search plus an order is tier 1; adding service history makes it tier 2. A provider handoff is a registered tier 3 audit action, not a fifth MCP tool. Tool and data definitions, not a prompt or caller's proposed tier, supply these values. A denied private read is still a private tool action and is audited, but its authorized evidence snapshot is empty. For a question using several tools, the checked answer records the maximum tool/data tier from that composition. Each tool event keeps its own assessed tier. Tiers remain audit classification and offline promotion requirements in this demo; they do not change runtime tool permissions or add approval gates. A tier 3 handoff is its own registered action, not a sticky session value. The next order read is assessed from its own order tool and evidence, even after a handoff.

Builders work in a sandbox with synthetic fixtures. Promotion to certified consumer use requires the gate report: cross-customer access denial, factual grounding against authorized source facts, complete trace and lineage, and the test set required at the computed tier. Security/data-product owners own the entitlement gate; the domain owner owns factual grounding and operator guidance; the agent-plane team owns trace and tier enforcement. A candidate fails closed if any required gate fails. **EVAL-005** deliberately injects a candidate-only “delivered” answer for Alice's in-transit order; the grounding gate must fail and the overall promotion result is **blocked**. That text must not enter the normal customer answer path.

The implemented fact checker covers these exact synthetic statuses and summaries. Broader customer language needs domain-owned evaluation before certification.

A failure sends the affected pod back to its sandbox for a targeted fix and rerun; it does not create an exception to certification. To shorten that week, retain small synthetic fixtures, deterministic offline provider fakes, a one-command evaluation report, clear gate ownership, and a narrow rerun of the failed gate before the full suite. Certification still uses the full gate set and a reviewed source/prompt version.

## Staffing, sequence, and decisions

Start with **two harness engineers**, both reporting to the agent-plane lead: embed one with the customer assistant pod and one with the field-service triage pod. Reserve shared review time for MCP contracts, identity propagation, audit schema, and evaluation fixtures so pod-specific adapters do not fragment the platform. Add a third only when a third funded pod is ready. In year one, certify the customer slice after the intentional gate is resolved, extend the same seams to read-only field-service triage, then standardize the harness and operational ownership before allowing writes or autonomous actions. Measure time from failed gate to corrected candidate, unauthorized-read regressions, trace completeness, and provider-switch effort.

Assumptions that could change this plan: (1) data products expose stable customer and instrument IDs with enforceable entitlements; (2) the chosen providers and manual sources are approved for the intended data and region; (3) a single orchestrator can own the first workflows. Likely failures are stale order status, a citation attached to a false claim, drift in customer-to-instrument ownership, unavailable MCP/CRM service, or a model outage. Source timestamps, fact-level evaluation, data-boundary checks, timeouts, and safe failure replies are the first controls; the demo proves only a narrow synthetic version of them.

**Interoperable A2A remains deferred.** Step 12 supplies an A2A-style local handoff between OpenAI and Claude model routes of one assistant, with verified conversational context and audit lineage. It does not create independently owned agents, agent endpoints, task messages, or identity delegation under an A2A protocol. Introduce that protocol when an independently owned specialist agent must accept a bounded task, return evidence and status, and preserve the original user/agent lineage across that boundary.
