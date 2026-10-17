# DataPilot progress (compact state; CONTEXT.md is the rulebook)

LLM: OpenRouter only (OPENROUTER_API_KEY from the environment; model via --model / OPENROUTER_MODEL, default openai/gpt-4o-mini).
Datasets (eval/data, gitignored): shopping_trends.csv (no date column), SeoulBikeData.csv (cp1252, dd/mm/yyyy).

## Done
- Tools (6), MCP server, LangGraph loop, report + DOCX, number check, Streamlit app, CLI (run.py), Unicode uploads, CSV cp1252 + day-first dates
- Eval harness: 20 questions with pandas truth, scoring, run_eval (resume, repeats, --concurrency, llm_calls, rate-limit retry, dry-run estimate), summarize (+ --update-readme)
- Hardening: SessionManager (5 sessions, 30 min idle, orphan-folder sweep)
- OpenRouter switch: llm.py, app sidebar + privacy checkbox, eval, tests, Docker (single app service), docs
- Docs: README, docs/DEPLOY.md, docs/DEMO.md, docs/CV.md (TODO(results) placeholders), LICENSE (MIT), requirements.lock

## STILL TO DO (needs a real OpenRouter key and your machine)
- Check: `OPENROUTER_API_KEY=... python run.py --file <file> "<question>"`, then in the app press "Check key and model"
- Smoke: CLI on the eval datasets (aggregate, group-by + chart, vague question); app with a CSV and an Arabic-named file; screenshots to docs/img/
- Eval: `python -m eval.run_eval --name x --dry-run` (120 runs), slice `--limit 4 --repeats 1` (read time per run and llm_calls), then the full run
  in the background; `python -m eval.summarize eval/results/run1 --update-readme`; review failures by reason; fix general bugs only; rerun as run2 and use `--compare run1`
- Replace every TODO(results) in README / docs/CV.md from summary.md
- `pytest -q` with the key set (live tests run), `git tag v1.0`, build datapilot-final.zip WITHOUT data/ and eval/data/

## Changelog
- Day 5: OpenRouter removed, Ollama only (gemma4:26b), embedding models never selectable
- Oct 14: Docker Compose (app + Ollama), Ollama timeout/keep-alive/warm-up, session manager, summarize --update-readme, docs
- Oct 17: switched back to OpenRouter only (Ollama code, warm-up, GPU and host overlays, pull script removed); privacy checkbox; --concurrency; rate-limit retry; llm_calls
