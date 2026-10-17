# PROJECT CONTEXT

PROJECT: DataPilot, a verified Excel analyst agent (1-week portfolio project).
A user uploads Excel/CSV and asks for a report; every number comes from a tool.
Flow: Router -> agents <-> Excel MCP server (6 tools) -> Observation (retry max 2) -> report.
Tools: list_sheets, load_data, aggregate, compare_groups, run_pandas, make_chart.
LLM: OpenRouter only (OpenAI-compatible API via langchain-openai). The key comes ONLY from OPENROUTER_API_KEY.
PRIVACY TRADE-OFF: column names, the 5-row load_data preview, tool results and capped run_pandas output (20 rows) are sent to
OpenRouter and the model provider; the full files, charts and reports stay local. The app needs the user's consent before each run.

## RULES
- The LLM never does arithmetic.
- Tools never raise; they return {"error": "..."}.
- Data stays on the server (previews max 20 rows).
- Read only in DATA_DIR, write only in OUTPUT_DIR; never modify inputs.
- Never print to stdout in the server (stdio protocol); log to stderr.
- Logs never contain raw rows.
- The API key is never logged, printed, stored, committed or shown in the UI; it is excluded from every repr, trace, error message and eval result.
- run_pandas printed output sent back to the model stays capped; do not raise the cap.

## STACK
Python 3.10+, mcp (FastMCP), pandas, openpyxl, scipy, matplotlib, pytest.

## OUT OF SCOPE
KPI registry, templates, Planner/Verifier agents, MLflow, SQL, PDF.

## WORKING AGREEMENT
Type hints, docstrings, small commits. If ambiguous, pick the simplest
option and list it under "Decisions". Say plainly what you could not verify.

## EXTRA RULES (from the Day 5 handoff)
- Environment: OpenRouter only. Model via --model / OPENROUTER_MODEL (alias MODEL_NAME; default openai/gpt-4o-mini); it must support tool calling.
- Eval: do not tune to the 20 questions; fixes must be general. Do not edit truth functions unless one is provably wrong (list it under Decisions).
- README, CV and LinkedIn text use only numbers computed by eval/summarize.
- data/ and eval/data/ are private: never print rows, never commit them.

## ARCHITECTURE (short)
Router -> analysis/chart agents <-> Excel MCP server (list_sheets, load_data, aggregate, compare_groups, run_pandas, make_chart)
-> Observation (retry max N) -> report (facts in code, writer LLM never types numbers, number check, MD + DOCX).
Streamlit app (app.py) + CLI (run.py). Eval harness: eval/questions.py, scoring.py, run_eval.py, summarize.py.

## DECISIONS
- Docker Compose: one app container; app port bound to 127.0.0.1 (no login); key passed through from .env, never baked in; pytest/psutil not in the image.
- MIT licence (copyright holder Basmala, 2026) so the portfolio code can be reused.
- requirements.lock is `pip freeze` of the environment the tests ran in (Linux, Python 3.13); regenerate on your own machine if you want Windows-exact pins.
- Session limits: 5 sessions, 30 min idle (DATAPILOT_MAX_SESSIONS / DATAPILOT_IDLE_MINUTES); orphan sweep only deletes 12-hex session folders.
- OpenRouter switch (Oct 17) reverses the Day 5 decision, because a 26B local model was too slow on a CPU-only machine.
- check=True makes TWO cheap GETs, not one: GET /models (model id exists, "tools" in supported_parameters) and GET /key. Reason: /models is public and answers
  even for a wrong key, so it cannot confirm the key. Only 401/403 from /key count as a rejected key; any other answer there is ignored. Not verified against the live API.
- Sessions build the model WITHOUT the per-job check (the app has a "Check key and model" button; run.py and eval check once up front), to avoid downloading the model list for every run.
- --concurrency (1 to 4) implemented: each worker owns its MCP server and folders; runs are independent; records are appended under a lock.
- Rate limits: the client retries 429/5xx/timeouts twice; an eval run that still ends in 429 is waited for (30/60/120 s) and re-run; it counts as rate_limit only after that.
- Dry-run works without the datasets (figure counts shown as '?'); its LLM-call estimate is an assumed range (4 to 20 calls/run) unless the run folder has measured llm_calls.
- Curated model list in the app (5 ids) is a convenience; ids may go stale, the Check button reports "model not found".
- This file has no numbered sections 1 and 9: the LLM and privacy statements went into the header and RULES instead.
