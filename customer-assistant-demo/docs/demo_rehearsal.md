# Step 7 - Clean-start demo rehearsal

- Rebuild the generated SQLite copy from six source CSVs, run tests and evaluation, then restart the notebook kernel.
- Follow a 10-12 minute sequence covering Alice's order/service access, public guidance, Bob's denial, provider selection, evaluation, and traces.
- Explain why the test harness passes while the intentionally false candidate keeps promotion blocked.
- Distinguish the offline gateway comparison from an optional live provider call and name the production identity/CRM work still required.
- Step 9 adds a signed order-list check; Step 10 expands it so Alice sees five
  owned statuses, while an active-only request returns her three non-delivered orders.
- Step 10 provides a continuous Streamlit chat and a three-day local trace browser;
  keep the intentional evaluation failure in the separate CLI/notebook harness.
- Step 11 demonstrates authorized customer joins, order dates, no-ID service
  history, and a no-term public troubleshooting catalog without crossing the
  selected customer's boundary.
- Step 12 demonstrates retained per-customer chat and a local OpenAI-to-Claude
  handoff whose recap never substitutes for a fresh authorized MCP read.
- Step 13 lets the presenter inspect the prepared packet and its later delivery
  in Audit Log, with the same 72-hour retention as the surrounding audit trace.
- Step 14 shows native model selection from the reviewed MCP catalog, followed
  by argument validation, signed execution, fresh evidence, and a checked answer.
  The explicit offline fake keeps rehearsal deterministic.

## Clean start from the project folder (PowerShell)

```powershell
python3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = 'src'
Remove-Item -LiteralPath .\data\local\customer_assistant.sqlite3 -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe -m customer_assistant.database
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
$env:DEMO_SIGNING_SECRET = (& .\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_hex(32))")
.\.venv\Scripts\python.exe -m customer_assistant.evaluation
```

These clean-start commands delete the generated SQLite database, including retained chat and audit traces; use them only when a reset is intended. The evaluation's intentional EVAL-005 grounding failure should block promotion while the Python test suite passes. The command creates a temporary process signing secret without displaying it; the notebook creates its own ephemeral secret when none is configured in its kernel. Do not print or save either secret. Open `notebooks/01_customer_assistant_demo.ipynb` in VS Code using `.venv`, **Restart Kernel**, and **Run All**. Keep the default offline mode for a repeatable first pass. If a clean-start command fails, stop and diagnose that exact seam before the presentation. The Streamlit chat does not run or show evaluation; use the CLI or notebook for that gate.

For the optional local UI presentation, run `.\.venv\Scripts\python.exe -m streamlit run app.py` from the project folder. Its **Settings** sidebar selects Alice or Bob and a live OpenAI or Anthropic provider. The main panel is **Your friendly AI assistant** with a continuous chat for each customer. Completed turns are saved in SQLite by customer ID and reloaded after provider changes, browser reconnects, or app restarts. A selected provider needs its API key in the process environment or `.env` for live tool selection and final answers. When changing providers after a verified answer, the outgoing model summarizes recent verified turns; a local recap is used if it cannot run. The receiving route gets the bounded recap as context, then the normal orchestrator obtains fresh authorized MCP evidence for the next answer. The **Audit Log** sidebar section shows eight recent summaries and lets you select any retained trace. Its tier 3 handoff events show the provider direction, summary origin, bounded summary, up to six verified question/answer turns, and source/trace IDs; no raw MCP rows or credentials are stored in that packet. The provider switch records a **prepared** event. A **delivered** event appears after the first successful, verified receiving tool-selection response, before MCP retrieval and the final answer. It records the planner receiving the recap even if a later tool or answer fails; an unused handoff remains prepared. This is a local A2A-style exchange between model routes of one assistant, not an interoperable A2A protocol or separate agents. Chat pairs and whole audit traces, including handoff packets, older than 72 hours are removed on the next app run or rerun.

In Step 11, the selected demo login becomes a signed customer ID before a private query. An authorized order or service result may include that customer's ID, synthetic organization, and demo contact email from the customer join. A name or customer ID typed into chat cannot grant access to another customer. An explicit request for Bob's private records while Alice is selected is denied before MCP or model traffic. A question about the datasets receives a fixed relationship explanation rather than unrestricted database access.

Step 14 planning uses up to three native selection rounds and four MCP calls for a question. Inspect a normal trace to show selection, authorized tool execution, a stop-selection round, and the grounded answer. Try a combined request such as "Show order DEMO-ORD-1009 and the service history for DEMO-INS-1001" to demonstrate composition. The 1009 chronology conflict is intentional and would be reconciled upstream in a live data product. Risk tiers classify the individual events and composed answer; they do not change runtime actions in this demo. A provider key is required for live selection as well as final answers; missing keys or provider failures stop planning safely. A no-call initial decision produces an approved static clarification, and arbitrary planner prose is discarded. The deterministic selector belongs only to the explicit offline fake.

## 10–12 minute sequence

| Time | Action | Point to show |
| --- | --- | --- |
| 0:00-1:00 | State the synthetic scope and inspect the four MCP skills. | The model receives reviewed names/descriptions and argument schemas. Signed identity stays in code; private tools enforce ownership. |
| 1:00-2:30 | Select Alice and ask "List all my orders, by status. Order by date." Then ask for the customer ID, organization, and demo email linked to those orders. | Six individually cited Alice orders, sorted by `created_on` newest first. Show the verified customer join; `estimated_delivery` is not an order date. |
| 2:30-3:30 | Ask "Show my full service history" without an instrument ID, then ask for `DEMO-INS-1001`. | The no-ID route returns all signed-customer events; the specific route checks instrument ownership. Both use tier 2 private evidence. |
| 3:30-4:30 | Ask "What troubleshooting articles are available?" and then about "Pressure Below Lower Limit". | The broad route lists all approved tier 0 articles with citations; the targeted route uses the matching public card and escalation language. |
| 4:30-5:30 | While Alice remains selected, ask for Bob's orders. Also ask what datasets are in the database. | Bob's private request is denied before MCP/model work. The dataset reply describes relationships without exposing private rows or arbitrary schema access. |
| 5:30-6:30 | Select Bob and list his orders. Try Alice's `DEMO-ORD-1007`. | Bob sees five owned orders; the explicit Alice order ID returns no private evidence. |
| 6:30-7:45 | In Streamlit, ask Alice about one owned order with OpenAI, then select Claude and ask a follow-up about that same order. | The displayed conversation persists. Open **Audit Log** to show the prepared packet, the delivered event on the first receiving planning response, then the new answer's fresh MCP authorization and source citation. |
| 7:45-9:15 | Run the evaluation report. | Cross-customer, trace, and tier gates pass; the EVAL-005 candidate-only wrong "delivered" answer fails grounding and promotion remains blocked. |
| 9:15-10:30 | Inspect an allowed and a denied trace in the notebook or local UI. | Selection round, executed tool, final answer, user/agent, outcome, evidence count (zero for denial), provider/model, timestamp, and assessed tier. |
| 10:30-12:00 | Explain seams and take one failure question. | OIDC/JWT and CRM stubs, no cloud deployment, and why the local provider handoff is not an interoperable A2A agent protocol. |

The two-provider notebook comparison uses an offline fake and proves the configuration and code seam without provider traffic. The Streamlit handoff demonstration uses live provider routes only if valid keys and model routes are available for both; if the outgoing route fails, identify the recap as a local fallback, not a model summary. Do not claim a live call, deployed service, production login, CRM integration, or interoperable A2A protocol without evidence. The false EVAL-005 answer belongs only to evaluation output and must never be shown as a customer-facing result.

## Rehearsal checklist

- Confirm a fresh database load reports the expected six synthetic CSV sources.
- Confirm the current order source contains eleven valid orders: six for Alice (four active), five for Bob (three active), including the intentional 1009 chronology conflict.
- Confirm date-sorted order answers use `created_on`, give each order its exact status and own citation, and do not call `estimated_delivery` the order date.
- Confirm the customer ID, organization, and demo email on private evidence belong to the selected signed customer; another customer's named request is denied before a tool or model call.
- Confirm an ID-free service request returns all service events for the selected customer, while an instrument-specific read checks ownership.
- Confirm a broad troubleshooting request lists every approved public article with its citation, and a dataset question returns only the fixed relationship description.
- Confirm all tests pass, while the evaluation reports the intended grounding failure and **blocked** promotion.
- Confirm Alice receives cited order/service answers and Bob receives no private evidence.
- Confirm the public card cites the manual and avoids internal repair instructions.
- Confirm audit trace inspection matches the same request ID across its actions.
- Confirm selection receives the reviewed MCP schemas without signed identity fields, then receives only projected current evidence after execution; the final answer retains all required citations.
- Confirm a composed order/service answer is tier 2, while each order and service event retains its own tier; a separate tier 3 handoff does not raise every later order read.
- Confirm Alice's displayed chat persists after switching OpenAI/Claude and reloading the app, while Bob's separate history contains no Alice turns.
- Rehearse the provider handoff once; check **Audit Log** for model or fallback origin, the bounded summary and verified turns, and tier 3 prepared/delivered events under one handoff ID. Confirm the follow-up has a fresh authorized MCP result.
- Use a live provider demonstration only if both provider keys/routes were verified in advance.
