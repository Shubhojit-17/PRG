# Prompt Regression Guardian Notes

## Purpose
- Build an agent that detects regressions between new prompt versions and a known-good baseline.
- Run evaluations via Phoenix MCP, compare scores, and alert via Slack before production rollout.

## Stack
- Google Cloud Agent Builder + Gemini 2.0 Flash
- Arize Phoenix + Phoenix MCP server
- FastAPI on Cloud Run with Cloud Scheduler triggers
- Slack Incoming Webhooks
- Python 3.11

## Required Files (per spec)
- Root: README.md, LICENSE, .env.example, .gitignore, requirements.txt, Dockerfile, cloudbuild.yaml
- agent/: __init__.py, main.py, guardian.py, mcp_client.py, evaluator.py, alerter.py, models.py
- config/: __init__.py, settings.py
- scripts/: seed_phoenix.py, trigger_local.py
- tests/: __init__.py, conftest.py, test_evaluator.py, test_mcp_client.py

## Key Behaviors
- Phoenix MCP client uses JSON-RPC over stdio via subprocess: `npx -y @arizeai/phoenix-mcp --baseUrl <host>`.
- Evaluate latest vs baseline (most recent "stable" tag). Skip prompts without baseline or dataset.
- Max 3 prompts in parallel; experiments run concurrently per prompt.
- Regression logic: score delta + per-dimension drops vs threshold.
- Slack alerts use Block Kit; include Phoenix experiment URL on failures.

## Code Quality Constraints
- Full Google-style docstrings and type annotations for every function/method.
- All I/O must be async; no blocking calls on event loop.
- structlog for logging (no stdlib logging; no print except seed script).
- Tool names, model names, URLs via settings or module-level constants.
- Specific exception types only; log when catching.

## Tests
- Evaluator: pass/fail/inconclusive logic + summary includes version tags.
- MCP client: initialize handshake, list prompts, retry logic, MCPToolError.

## Post-Generation Checklist
- Confirm Phoenix REST API endpoints and MCP tool schemas match deployment.
- Replace placeholder GCP project IDs and Slack webhook.
- Ensure Cloud Run secrets exist in Secret Manager.
