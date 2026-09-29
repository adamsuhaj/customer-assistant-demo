# Step 0 - Customer Facing Assistant Demo

- Use this README as the map for rebuilding, running, testing, and presenting the local synthetic assistant.
- Step 1 validates six supplied CSV files and loads a replaceable SQLite copy, leaving the source datasets untouched.
- Step 2 turns the Alice/Bob demo login into signed user, agent, and trace context; private reads check ownership before returning a row.
- Step 3 introduces the first three MCP tools: single-order status, service history, and public troubleshooting; Step 9 adds the fourth, customer-scoped order list.
- Step 4 routes model requests through a common OpenAI/Anthropic gateway and provides an offline fake for a repeatable rehearsal.
- Step 5 chooses one MCP tool per question, forwards only authorized evidence to the model, and returns citations with the trace ID.
- Step 6 writes per-action audit events, inherits the highest tool/data risk tier, and blocks a candidate that claims an in-transit order was delivered.
- Step 7 adds architecture and operating-model notes, production stubs, and a timed demo script.
- Step 8 adds a small Streamlit chat UI over the same orchestrator and trace reader.
- Step 9 adds a signed, customer-scoped order-list tool so Alice can see all or only active orders without supplying each ID.
- Step 10 refines the dark Streamlit chat, keeps recent per-customer trace history for 72 hours, and expands the synthetic data to ten orders.
- Step 11 joins authorized private evidence to the selected customer, handles ID-free service history and public troubleshooting lists, and verifies complete cited answers.
- Step 12 saves customer-scoped chat turns in SQLite and hands verified conversation context between OpenAI and Claude routes without changing MCP authorization.
- Step 13 renames the sidebar trace browser Audit Log and retains a bounded copy of each provider handoff packet with its audit events for 72 hours.
- Step 14 gives the selected model reviewed MCP tool schemas, validates its proposed calls, and gathers fresh authorized evidence in a bounded tool loop before checking the final answer.

## Build sequence

1. **Step 0, scaffold (guide page 5):** create a runnable local project and keep the supplied synthetic data as read-only source files. Other module files are placeholders until their guide pages are executed.
2. **Step 1, data (guide page 6):** validate and copy the CSV data into a local SQLite database. Validation catches broken links before the assistant can use the rows.
3. **Step 2, identity and policy (guide page 7):** sign the simulated login context and check ownership in the data access functions. This keeps private reads tied to the selected demo customer.
4. **Step 3, MCP tools (guide page 8):** expose the initial three protocol calls for single-order status, service history, and public troubleshooting. Step 9 adds the fourth call for a customer-scoped order list. The private tools reuse the signed context and SQL ownership checks.
5. **Step 4, model gateway (guide page 9):** select OpenAI or Anthropic through configuration and return the same answer and usage fields from either provider. An offline fake supports tests without API calls.
6. **Step 5, assistant orchestration (guide page 10):** classify the question, call one MCP skill, and use only its authorized evidence to form a response. The notebook demonstrates the result, sources, trace, and selected provider in offline mode by default.
7. **Step 6, audit and evaluation (guide page 11):** record tool actions, denials, and model responses in SQLite; compute the maximum trusted tier touched; evaluate entitlement, grounding, trace, and tier gates. EVAL-005 is an evaluation-only wrong-answer candidate that must fail grounding and block promotion.
8. **Step 7, document and rehearse (guide page 12):** rehearse a 10–12 minute notebook walkthrough, document the target architecture and operating model, and mark the production identity and CRM connector stubs.
9. **Step 8, local Streamlit UI:** show a customer/provider selector, chat results, and trace metadata without copying the MCP or policy code.
10. **Step 9, order list:** add a fourth private MCP tool that lists only the signed customer's orders; route plural order requests to it and require every returned order's status and citation in a complete answer.
11. **Step 10, chat and data refinement:** show a continuous chat for each selected demo customer, keep a customer-scoped trace browser with 72-hour retention, and expand the fixture to five orders each for Alice and Bob.
12. **Step 11, linked evidence and broader reads:** join customer details only after signed ownership checks, deny a request for the other demo customer before a private read or model call, use `created_on` for order-date questions, and support complete customer service and approved public-article lists without a record ID.
13. **Step 12, provider handoff and durable chat:** store each completed customer chat turn in SQLite for 72 hours; on an OpenAI/Claude switch, recap audit-verified turns with the outgoing model when available and carry a bounded handoff to the incoming route. Recheck authorization and current sources for every new answer.
14. **Step 13, inspectable handoff audit:** record the prepared handoff packet when the provider changes and its delivery after the first successful, verified receiving tool-selection response, before evidence retrieval and the final answer. Show the provider direction, summary origin, bounded summary, verified question/answer turns, and source/trace IDs in the customer-scoped **Audit Log**, under the existing 72-hour trace retention.
15. **Step 14, model-selected tools:** discover the real MCP catalog, expose only reviewed tools and business arguments, and let the model propose calls. Application code validates each proposal, supplies signed identity, executes MCP, and returns projected evidence for the next planning round. The loop allows at most three planning rounds and four tool calls per question; final answers still require current citations and grounding checks.

Each authored file starts with the step in which its current behavior was built and short bullets explaining its purpose. The synthetic source datasets have no code comments; generated JSON and SQLite carry inspectable metadata because those formats cannot begin with comments.

This project is a local, synthetic-data prototype for the customer-facing assistant case study. The notebook drives the walkthrough; reusable application code belongs in `src/customer_assistant`. The CSV loader, signed demo identity, private-row authorization, four local MCP tools, model gateway, evidence-routing orchestrator, per-action audit, risk inheritance, and evaluation harness form the working slice. The intentionally failed evaluation keeps promotion blocked.

## Layout

- `data/source/` contains six synthetic CSVs. The current order fixture has eleven rows; the loader treats all source files as read-only inputs.
- `data/local/` holds the generated SQLite copy and validation/evaluation reports; it is ignored by Git. The validation JSON starts with Step 1 metadata, and the binary database stores its Step 1 summary in `artifact_metadata`.
- `notebooks/01_customer_assistant_demo.ipynb` runs a configurable question and curated offline examples.
- `app.py` presents the same assistant through a continuous Streamlit chat, reloads each customer's retained SQLite history, and coordinates provider handoffs and the sidebar **Audit Log**.
- `.streamlit/config.toml` sets the dark gray theme and blue accent colors.
- `src/customer_assistant/database.py` validates and loads all six CSVs.
- `src/customer_assistant/identity.py` mints and verifies a signed demo identity.
- `src/customer_assistant/policy.py` enforces ownership in order and service reads.
- `src/customer_assistant/mcp_server.py` exposes four FastMCP tools, including the signed order-list tool.
- `src/customer_assistant/mcp_client.py` discovers reviewed tool schemas, validates model arguments, and calls the tools through a separate stdio process.
- `src/customer_assistant/config.py` loads provider, model, timeout, and selected key from `.env` and process settings.
- `src/customer_assistant/gateway.py` normalizes native model tool proposals and final text through LiteLLM, with an offline fake for tests and rehearsal.
- `src/customer_assistant/orchestrator.py` validates and executes model-selected MCP calls, builds an evidence-bounded answer request, and returns a traceable assistant result.
- `src/customer_assistant/offline_tool_selector.py` selects tools deterministically only for the explicit offline fake; a failed live provider call does not activate it.
- `src/customer_assistant/conversation.py` stores completed chat pairs by verified customer ID and purges pairs older than 72 hours.
- `src/customer_assistant/handoff.py` verifies displayed turns against allowed audit events, then builds a bounded outgoing-provider summary or local fallback for the incoming provider.
- `src/customer_assistant/audit.py` stores and reads per-action traces, including bounded provider handoff packets, in the local SQLite copy and purges complete traces after 72 hours when the UI runs.
- `src/customer_assistant/risk.py` derives the maximum tier of tools and records touched.
- `src/customer_assistant/evaluation.py` runs the gate set and reports promotion state.
- `src/customer_assistant/crm_connector.py` is an explicit fail-closed production stub with no network call.
- `docs/architecture_and_operating_model.md` records the target seams, risk, gate ownership, staffing, assumptions, and A2A decision.
- `docs/demo_rehearsal.md` provides the clean-start commands and timed 10–12 minute script.
- `output/pdf/customer_assistant_architecture_and_handoff.pdf` is the Step 13 visual snapshot. It predates Step 14 model-selected tools; this README and the architecture note describe the current request path.
- `tests/` contains focused loader, identity, entitlement, protocol, gateway, orchestration, audit, risk, and evaluation checks.

## Environment

Use Python 3.12. From this folder in PowerShell:

```powershell
python3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

No API keys are needed for the offline tests or the notebook's default walkthrough. Private reads require `DEMO_SIGNING_SECRET` in the application process environment; the tests set a temporary test-only value and the notebook makes an ephemeral one if neither the process nor `.env` supplies it. To use a live model, copy `.env.example` to `.env`, set the provider and its key, and keep `.env` out of version control. Step 4 configuration reads that file without putting credentials in the gateway result.

To check the package import after creating the environment:

```powershell
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'src'); import customer_assistant; print(customer_assistant.__version__)"
```

## Validate and load the synthetic data

From this folder in PowerShell:

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m customer_assistant.database
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The loader prints a per-file JSON report and saves it to `data/local/validation_report.json`. The current fixtures contain 2 customers, 2 instruments, 11 orders (6 for Alice and 5 for Bob), 3 service events, 2 troubleshooting articles, and 6 evaluation cases. A failed validation names the file and failed check, exits with status 1, and leaves any previous database copy intact. Use `--validate-only` to report on source files without changing the database.

All IDs and source fields are stored as strings. The loader checks exact headers, primary ID uniqueness, customer and instrument references, instrument ownership, demo login references, and order statuses. The accepted order statuses for this dataset are `processing`, `in_transit`, and `delivered`. Re-running the loader replaces the local source tables in a transaction rather than duplicating rows, and preserves the separate audit and chat tables. The source CSVs are never written by the loader.

Open `notebooks/01_customer_assistant_demo.ipynb` in VS Code and select the `.venv` interpreter as its kernel. Run all cells from a fresh kernel for the offline walkthrough, or edit its presenter settings first.

## Demo identity and private reads

The notebook's `alice` or `bob` setting is a **simulated login**. Trusted application code calls `mint_demo_identity(demo_login)` to resolve it through the local customers table. The resulting context contains a customer `user_id`, fixed `agent_id=customer_assistant_v1`, generated `trace_id`, expiry, and HMAC signature. Its serialized form has no signing secret. A private data read verifies the context before querying SQLite.

`get_order_status(identity, order_id)` returns an order only when its `customer_id` matches the signed user and its instrument belongs to that customer. `list_customer_orders(identity, active_only=False)` returns all orders owned by that signed customer, or only `processing` and `in_transit` orders when `active_only=True`; it accepts no customer ID. Step 11 joins each authorized order to that same customer's record, exposing the verified customer ID, organization name, and synthetic contact email in reviewed evidence. Alice and Bob are demo login aliases, not the organization names. The model cannot use a name or customer ID supplied in a question to select another customer's rows. An explicit request for the other demo customer is denied before a private tool or model call.

`get_service_history(identity, instrument_id)` checks instrument ownership first, then returns only that customer's events. Its Step 11 no-ID route lists service events across all instruments owned by the signed customer. The service join also includes that customer's verified organization and demo email. Both paths require the event and its parent instrument to belong to the signed customer. A denied result has no rows or source IDs and uses the same reason for an unknown record and another customer's record. An owned customer with no matching orders or service events receives an allowed empty result, not a foreign-record denial.

To run a local smoke check without printing or saving a signing secret:

```powershell
$env:PYTHONPATH = 'src'
$env:DEMO_SIGNING_SECRET = (& .\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_hex(32))")
.\.venv\Scripts\python.exe -c "from customer_assistant.identity import mint_demo_identity; from customer_assistant.policy import get_order_status; alice=mint_demo_identity('alice'); bob=mint_demo_identity('bob'); print('Alice:', get_order_status(alice, 'DEMO-ORD-1007').outcome); print('Bob:', get_order_status(bob, 'DEMO-ORD-1007').outcome)"
Remove-Item Env:DEMO_SIGNING_SECRET
```

**PRODUCTION STUB (OIDC/JWT):** the local alias selector and HMAC context are for this synthetic demo. Real authentication must validate OIDC/JWT claims before minting tool context. The Step 5 orchestrator sends the signed context only to the MCP boundary and sends authorized evidence, never the signature or secret, to the model gateway. The second marked stub, `crm_connector.py`, raises `NotImplementedError` rather than opening a network connection; a production adapter must authenticate to a governed source and recheck customer entitlements there.

## Local MCP tools

The server advertises `get_order_status`, `list_customer_orders`, `get_service_history`, and `search_troubleshooting`. The first three take an application-signed `identity_context` because the user identity must survive the stdio hop. The order-list tool takes only that context and an optional active-only flag, then filters by the signed customer's ID inside SQLite. The service tool accepts an optional instrument ID: with one it checks that instrument; without one it returns the signed customer's full service history. The public search accepts optional `model` and `symptom` terms. When both are absent, it lists the approved tier 0 article catalog with article IDs and manual attribution; a targeted query still needs a supported match. The catalog is public and does not query private order or service rows.

The client launches `python -m customer_assistant.mcp_server` as a separate process using FastMCP's stdio transport. It explicitly passes the local database path and, for private calls, `DEMO_SIGNING_SECRET` into that process. It never puts the secret in tool arguments. The orchestrator uses this client; the notebook does not call server functions directly.

Step 14 replaces the original Step 5 keyword routing. It obtains the tool catalog through MCP `list_tools` and exposes the four reviewed names, descriptions, and JSON schemas to the selected model. The model-visible schemas omit `identity_context`; the model may propose business arguments such as an order ID or `active_only`, but cannot choose a customer, agent, trace, or signature. Application code rejects unknown tools, extra arguments, and schema-invalid values before execution. It then attaches the signed identity to private calls, and SQL ownership checks remain authoritative.

After an allowed call, the next planning round receives only the projected current evidence and source IDs. The model can request another approved tool or stop collecting evidence. The runtime bounds planning to three rounds and four calls, rejects repeated or invalid requests, and sends all accepted evidence to the final answer step. Arbitrary planning text is not displayed. With no evidence, the runtime uses only its approved static clarification messages. An explicit foreign-customer request and the fixed dataset explanation still use trusted application preflight checks.

The implementation follows FastMCP's [server tool](https://gofastmcp.com/servers/tools) and [stdio client transport](https://gofastmcp.com/clients/transports) documentation. Version 4.0.10 is pinned in `requirements.txt` because that is the version used for the protocol tests.

To run the protocol-level checks from the project folder:

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_mcp_tools.py -v
```

The tests list the four tools, call each over stdio, confirm six owned orders for Alice and five for Bob, deny cross-customer requests without returning Alice evidence, and retrieve `DEMO-KB-001` with its public source locator. Step 11 also exercises the signed no-ID service list and no-term approved public catalog. They use a temporary database and a test-only secret.

## Model gateway

`load_gateway_config()` reads the optional `.env` file and then applies process environment overrides. Set `DEMO_MODEL_PROVIDER` to `openai` or `anthropic`; the defaults are `openai/gpt-6-sol` and `anthropic/claude-sonnet-5`, respectively. `DEMO_OPENAI_MODEL` and `DEMO_ANTHROPIC_MODEL` can change those provider-qualified names without editing the gateway or MCP tools. `DEMO_MODEL_TIMEOUT_SECONDS` defaults to 30. Only the selected provider's API key is used, and a missing key stops a live call before it reaches LiteLLM.

`LiteLLMGateway.select_tools(messages, tools)` uses native tool calling through the same provider-neutral LiteLLM route. It returns normalized tool call IDs, names, and argument dictionaries. A no-call decision stops evidence collection or triggers an approved static clarification; arbitrary planner prose is discarded. `LiteLLMGateway.complete(messages)` remains the answer and handoff-summary interface: it returns `provider`, `model`, `answer_text`, and `token_usage` when available. Both interfaces use a finite timeout and withhold raw provider errors and credentials.

`FakeModelGateway` implements the same two interfaces without a key or network traffic. Its selector is deliberately deterministic for offline tests and notebook rehearsal. Live mode requires the selected provider key for tool selection as well as final answers; an unavailable provider produces a safe failure, not a hidden offline routing fallback.

Run the offline configuration and gateway checks with the full suite:

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

When provider keys are configured, this optional synthetic smoke check makes one short call to each available provider and prints only the normalized result:

```powershell
$env:PYTHONPATH = 'src'
@'
import asyncio
from customer_assistant.config import load_gateway_config
from customer_assistant.gateway import LiteLLMGateway

async def main():
    for provider in ("openai", "anthropic"):
        config = load_gateway_config(provider=provider)
        if config.api_key is None:
            print(f"{provider}: no key configured; skipped")
            continue
        result = await LiteLLMGateway(config).complete([
            {"role": "user", "content": "Reply in one short sentence: this is a synthetic test."}
        ])
        print(result.as_dict())

asyncio.run(main())
'@ | .\.venv\Scripts\python.exe -
```

The implementation follows the [LiteLLM SDK](https://docs.litellm.ai/docs/) response and provider-selection interface. `litellm==1.102.1` is pinned because that is the installed version used for the gateway adapter checks.

## Assistant walkthrough

The notebook's presenter cell controls `demo_user`, `provider`, `question`, `instrument_model`, and `demo_mode`. `provider = None` follows `DEMO_MODEL_PROVIDER` or its OpenAI default; set it to `"openai"` or `"anthropic"` to override for that run. `demo_mode = "offline"` uses `make_offline_demo_gateway()` even if an API key is configured. This keeps **Run All** deterministic and free of model charges. The notebook loads the local synthetic SQLite copy if it is missing, reads `DEMO_SIGNING_SECRET` from the process or `.env` when present, and otherwise creates an ephemeral secret for that kernel. Neither secret nor provider key is displayed.

The configured question runs first. The notebook then shows four offline cases: Alice's order, Bob's request for that same order, Alice's service history, and a public "Pressure Below Lower Limit" warning. Each result displays the answer, source IDs, trace ID, provider, model, and token usage. A final offline comparison runs one question through the same routing and evidence path with both provider configurations. The fake gateway uses authorized evidence to produce a deterministic answer; the provider label changes, while no API request is made.

For live model-selected tools and answers, set `demo_mode = "live"` in the presenter cell and configure the selected provider's API key. **Run All** then runs only the configured question through the bounded planning and answer flow; it skips the curated and provider-comparison examples. A question can make several provider calls. Live provider behavior and model availability depend on the external service and are not covered by the offline checks.

For a headless notebook run from the project folder:

```powershell
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute notebooks/01_customer_assistant_demo.ipynb --output 01_customer_assistant_demo.executed.ipynb --output-dir data/local
```

## Local Streamlit UI

Use **New chat** below the provider selector to clear the selected customer's saved messages and start a fresh conversation. It also clears the current draft and cached provider handoff context; the selected provider stays selected. A page reload keeps the conversation empty. Other customers' conversations and the separate **Audit Log** remain available.

For a double-click start on Windows, open **`Launch Customer Assistant.exe`** in the parent `Artifacts` folder. It runs the Streamlit command using this project's existing `.venv`, waits for the server and chat/selector JavaScript files to load, and opens the app in your default browser. VS Code does not need to be open. Closing the small launcher window keeps the app running in the Windows notification area; double-click its tray icon to reopen the browser or right-click it for the launcher controls. **Stop app and close** stops the Python process tree it started. Repeated clicks reuse the same launcher. If port 8501 is occupied, it uses the next available local port through 8520. Startup output is saved in `data/local/launcher.log`.

If an existing browser tab shows **Failed to fetch dynamically imported module** after the server was stopped, relaunch the app and reload that tab with **Ctrl+Shift+R**. The browser cannot retry a failed chat/selector module until the page is reloaded.

If both providers return **the model answer is unavailable** and `launcher.log` shows `WinError 10013` during network requests, the app may have inherited network restrictions from a test process. Use **Stop app and close**, then start the launcher from Windows File Explorer. Repeated double-clicks reuse the existing process, so stopping the restricted instance first matters. Agent-driven live tests must start the launcher with approved network access; a normal sandbox launch is suitable only for startup and local checks.

The launcher is a small Windows program that starts the existing project. It first checks for `customer-assistant-demo` beside its executable, then falls back to the installed project path recorded when it was built. This lets a copy embedded in PowerPoint find the project when PowerPoint opens it from a temporary folder. The project, including its `.venv` and `.env`, must remain installed on the presentation computer. After a launcher update, replace any previously embedded copy in PowerPoint with the updated executable. A hyperlink or action that runs the original executable also uses future launcher updates automatically. To rebuild it after editing its source or moving the installed project, run `powershell -NoProfile -ExecutionPolicy Bypass -File .\launcher\build.ps1` from the project folder. This uses the installed Windows .NET Framework compiler and downloads nothing.

From the project folder, install the updated requirements and start the app:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The Streamlit app uses the same `answer_question()` orchestrator as the notebook. Its dark gray interface has blue major titles, a **Settings** sidebar with the demo customer and provider controls, and a **Your friendly AI assistant** chat. Completed user/assistant pairs are stored in the local SQLite `chat_turns` table by signed customer ID, so switching OpenAI and Claude, reconnecting the browser, or restarting the app reloads the same conversation. Alice and Bob still have separate histories. A new browser session restores each customer's last used provider, avoiding an unintended paid handoff on reconnect. Turns older than 72 hours are deleted on the next app run or rerun. Each answer shows source IDs, provider/model, and trace ID. Each submitted question uses the existing short-lived MCP client; changing a control or opening a sidebar section does not launch another tool server.

When the provider changes and there are prior audit-verified answers, `handoff.py` asks the outgoing model to summarize up to six recent verified turns. If that model is unavailable, it builds a local recap from the latest verified turn. The incoming provider receives this bounded memo and recent verified turns with its next question; neither raw private MCP rows nor credentials go into the handoff. The summary is conversation context, not a source of truth: `orchestrator.py` still signs the selected customer, calls the appropriate MCP tool, rechecks ownership, and grounds the next answer in fresh authorized evidence. Switching the selected customer never transfers the other customer's history. A follow-up about one unambiguous previously cited order can reuse its ID, then fetch that order anew.

The sidebar **Audit Log** section shows the latest eight summaries for the selected customer and offers every retained trace ID in a selector. Select one to inspect its action lineage and authorized evidence row counts without printing private snapshots. A provider switch records a tier 3 **prepared** handoff event; the first successful, verified receiving tool-selection response records a tier 3 **delivered** event under the same handoff ID. The recap has reached the receiving planner at that point, even if a later tool read or answer fails. Those events include the provider direction, summary origin (model or local fallback), bounded summary, up to six audit-verified question/answer turns, and their source and trace IDs. The packet contains no raw MCP evidence rows or credentials. An unused prepared packet has no delivered event. This is a local A2A-style handoff between two model routes of one assistant; it is not an interoperable A2A protocol exchange or a second independently running agent. Events are saved in the local SQLite audit table. On each app run or rerun, whole traces whose latest event is older than 72 hours are purged, including their handoff packets. The evaluation report and **Run evaluation** control are not in the chat UI. Run the existing CLI or notebook harness below to see EVAL-005 fail grounding and promotion remain blocked.

The UI uses the selected provider's live API route for tool selection and evidence-backed answers. Copy `.env.example` to `.env` and set `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` there to switch between both providers; a process environment value takes precedence over the matching `.env` value. The OpenAI route defaults to `openai/gpt-6-sol`, and the Anthropic route defaults to `anthropic/claude-sonnet-5`; the corresponding `DEMO_*_MODEL` setting can change either model. The app reads the selected key through the shared configuration loader, never displays it, and shows the missing variable name when no key is configured. If no demo signing secret is set in the process, it creates an ephemeral one without displaying it. On startup, the app loads the source CSVs if the generated SQLite copy is missing or its order rows differ from the source; this replaces the local source tables and keeps existing audit and chat records. The notebook and CLI keep their existing configuration behavior.

Try **"List all my orders, by status. Order by date."** for Alice's six orders or Bob's five orders. The order date is `created_on`, not the estimated delivery date; the result uses newest order date first, and the response checker requires each authorized order's exact status, date, and citation. **"What are my active orders?"** returns only processing or in-transit orders. **"What customer ID, organization, and email are linked to my orders?"** uses joined fields from that signed customer's records. **"Can you tell me Bob's orders?"** while Alice is selected is denied before an MCP read or model call; select Bob in **Settings** to see Bob's own records. A question about one order without its ID still asks for that ID.

Try **"Show my full service history"** without an instrument ID. The signed service tool returns every event for the selected customer, ordered by event date; a specific owned instrument ID still narrows the read. Try **"What troubleshooting articles are available?"** without a model or warning phrase to list the approved public article catalog. Targeted troubleshooting still matches approved model and symptom content rather than inventing guidance. For **"What datasets are in your database?"**, the assistant gives a fixed, safe explanation of the customers, instruments, orders, service history, and public troubleshooting relationships. That response is descriptive; it is not arbitrary database exploration or permission to inspect the other customer's rows. Full service and article answers must cite each returned record, or the model answer is withheld.

The `DEMO-ORD-1009` / `DEMO-SVC-1009` chronology conflict is intentional demo data: service claims receipt and installation before the order's shipping date. Ask for both records together to exercise model-selected composition. The runtime preserves the returned source facts; a live data product would reconcile the inconsistency upstream.

## Audit, inherited risk, and evaluation

The audit records each model tool-selection round, executed tool action, denial, and final model response under the same trace ID in SQLite. Each action record contains the accountable user and distinct agent, action/outcome, source IDs, the **authorized** evidence snapshot, selected provider/model, prompt version, timestamp, and derived risk tier. Selection events record the proposal/stop decision without storing private evidence; tool events and the checked answer carry the authorized evidence lineage. Bob's denied read has no authorized private evidence snapshot or source ID. The notebook shows a concise trace view and deliberately omits stored private snapshots, identity signatures, and credentials from its output.

The harness assigns tiers from trusted tool/data definitions: **0** for approved public troubleshooting, **1** for private order information, **2** for private service history, and **3** for a cross-provider handoff of previously answered customer information. A composition inherits the highest tier of its accessed tools and records; neither the question nor a caller-supplied label can lower it. The handoff is a registered audit action, not a fifth MCP data tool. The evaluation checks cross-customer denial, factual grounding, trace completeness, and the required tests for the inherited tier. EVAL-005 supplies a wrong **candidate-only** “delivered” claim for Alice's in-transit order. That case must fail grounding and leave promotion **blocked**. The normal assistant path never uses this candidate as an answer. Risk tiers classify audit events and evaluation requirements; they do not change tool permissions, request approval, or trigger live action gates in this demo. The final answer inherits the maximum tier of the tools/evidence used for that question. A tier 3 provider handoff is a separate audited action; it does not permanently raise later order reads to tier 3. A passing Python test suite confirms that the intentional failure is detected; it does not certify this candidate.

From the project folder, run the evaluation separately to inspect the gate and promotion report:

```powershell
$env:PYTHONPATH = 'src'
$env:DEMO_SIGNING_SECRET = (& .\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_hex(32))")
.\.venv\Scripts\python.exe -m customer_assistant.evaluation
```

## Clean start and rehearsal

These commands reset the generated local SQLite copy, including saved chat and audit traces. They preserve all six source CSV files. Run them from the project folder in PowerShell only when you want a clean rehearsal:

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

Then open the notebook with the `.venv` kernel, choose **Restart Kernel**, and **Run All**. Default offline mode runs Alice's approved order/service requests, Bob's denial, public troubleshooting, the two-provider configuration comparison, the intentional failed gate, and trace inspection. The provider comparison uses a fake response to demonstrate the gateway seam without provider traffic. A true live swap requires valid keys and available model routes for **both** providers: set `demo_mode = 'live'`, run the configured question with `provider = 'openai'`, then change only `provider = 'anthropic'` and run the question again. Live mode skips the offline examples and evaluation; current provider availability and behavior need a separate live smoke check.

The [timed rehearsal](docs/demo_rehearsal.md) gives a 10–12 minute sequence and checklist. The [architecture and operating-model note](docs/architecture_and_operating_model.md) covers the runtime, orchestration, MCP/data-product and model-gateway seams; gate ownership, tier inheritance, staffing, assumptions, failure modes, and the A2A decision. Step 12's provider handoff is local conversation continuity between OpenAI and Claude routes of one orchestrator. Interoperable A2A between independently owned agents remains future work.

The working project folder is the current version. This local prototype has no cloud deployment, production login, or live CRM connector; those are explicitly marked production work.
