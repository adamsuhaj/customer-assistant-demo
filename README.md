# Customer Assistant  Demo

A local customer-assistant prototype built with Streamlit, LLM-selected MCP tools, signed user identity, grounded responses, a local A2A handoff, and an evaluation gate. The included customer, order, instrument, and service records are synthetic demonstration data.

## Run it

Start with the [short reviewer quickstart](RUN_ME.md). Python 3.12 and dependency installation are required. Offline tests and the notebook need no API keys; live Streamlit use requires your own OpenAI or Anthropic key.

On Windows, keep `Launch Customer Assistant.exe` beside the `customer-assistant-demo` folder and run it after setup. The launcher does not bundle Python or install dependencies. On other systems, start Streamlit directly using the quickstart.

## Review the implementation

- [Detailed project README](customer-assistant-demo/README.md)
- [Architecture and operating model](customer-assistant-demo/docs/architecture_and_operating_model.md)
- [Orchestrator and model tool loop](customer-assistant-demo/src/customer_assistant/orchestrator.py)
- [MCP tool descriptions and schemas](customer-assistant-demo/src/customer_assistant/mcp_server.py)
- [Demonstration notebook](customer-assistant-demo/notebooks/01_customer_assistant_demo.ipynb)
- [Tests](customer-assistant-demo/tests)

## Deliberate demo choices

- A2A handoff is local by design; it demonstrates portable conversation state and provider switching.
- Order 1009 contains an intentional source-data conflict that would be reconciled upstream in a live environment.
- Risk tiers classify and audit actions; runtime human approval workflows are outside this prototype. The offline certification harness applies the promotion gate.
- The offline selector supports deterministic tests and the notebook. Live tool selection uses the model-visible MCP catalog.

Credentials, virtual environments, generated databases, saved conversations, and local audit logs are excluded from this repository.

## Public package

This repository contains the code, synthetic data, and a portable Windows
launcher. Presentation slides and the memo are shared separately for the interview.
The optional decorative robot artwork is omitted from this public package;
the app runs normally without it. The launcher uses the adjacent project folder
and contains no embedded personal installation path.
