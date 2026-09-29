# Customer Assistant — reviewer quick start

This is a local Python 3.12 demo using synthetic customer, order, instrument,
and service data. It includes a Streamlit UI, four real MCP tools, and offline
tests/evaluation. No provider credentials or saved conversations are included.

## Install once

Extract the ZIP completely. In Windows PowerShell, enter the project folder:

```powershell
cd customer-assistant-demo
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m customer_assistant.database
```

Python 3.12 and internet access for dependency installation are prerequisites.
The loader validates the six source CSVs and creates the local SQLite database.

## Review without model API keys

From that same project folder and PowerShell session:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m customer_assistant.evaluation
.\.venv\Scripts\python.exe -m jupyter lab notebooks/01_customer_assistant_demo.ipynb
```

For the notebook, select the `.venv` Python kernel and run all cells with the
default `demo_mode = "offline"`. Offline runs use an explicit deterministic
model fake while executing real MCP, authorization, evidence, and audit paths.
They make no model API calls and incur no provider charges.

## Run the live Streamlit UI

Copy `.env.example` to `.env` and add your own `OPENAI_API_KEY` and/or
`ANTHROPIC_API_KEY`. Both keys are needed to demonstrate switching providers.
The configured model names can be changed in `.env` for routes available to
your account. Live questions and provider handoffs use the selected provider
and can incur API charges. Keep the populated `.env` private.

```powershell
Copy-Item .env.example .env
# Add your own keys to .env before continuing.
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit. On Windows, after this setup, you can
also double-click `Launch Customer Assistant.exe` in the parent folder. Keep
it beside `customer-assistant-demo/`. It starts that installed project; it
does not bundle Python, install dependencies, or supply provider credentials.

On other platforms, create/activate a Python 3.12 virtual environment, install
`requirements.txt`, and run `python -m streamlit run app.py` from the project
folder. The `.exe` launcher is Windows only.

## Deliberate demo choices

- `DEMO-ORD-1009` and the matching service record deliberately contradict one
  another. In a live environment the source owners would reconcile them.
- `EVAL-005` supplies an intentionally wrong candidate answer. Expected report:
  `factual_grounding: fail`, `promotion: blocked`, and `harness_status: pass` because
  the evaluation correctly detects it.
- The A2A-style provider handoff is local by design. It carries an audited
  recap between two model routes of the same assistant.
- Risk tiers classify audit events and evaluation requirements. Authorization
  comes from signed identity and ownership checks; tiers add no live approvals.

For the source map and full instructions, see `customer-assistant-demo/README.md`.
The current tool flow is described in Step 14: the LLM proposes MCP calls;
trusted code validates them, attaches identity, executes MCP, and checks the
evidence-backed answer. Planning is bounded to three rounds/four tool calls.

The optional decorative robot image is omitted from the public release. Its
absence does not affect the assistant, tools, notebook, or evaluation.
