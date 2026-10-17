# Deploying DataPilot (Docker Compose, self-hosted app)

One container: `app` (Streamlit + the MCP excel server it spawns). The model is **OpenRouter**, an external API:
column names, the 5-row preview, tool results and capped `run_pandas` output are sent to OpenRouter and the model
provider. **Do not upload confidential data.** The app makes users tick a box to confirm this before every run.

## Requirements
- Docker with Compose v2, outbound HTTPS to openrouter.ai, an OpenRouter API key with credit. No GPU and no model download.

## First start
```bash
cp .env.example .env              # put your key after OPENROUTER_API_KEY=
docker compose up -d --build
open http://localhost:8501
```
Stop: `docker compose down` (volumes are kept). `docker compose down -v` deletes uploads and reports.

## Configuration (.env)
| Variable | Default | Meaning |
|---|---|---|
| `OPENROUTER_API_KEY` | required | read only from the environment; never logged or shown |
| `OPENROUTER_MODEL` | openai/gpt-4o-mini | tool-capable OpenRouter model id |
| `OPENROUTER_TIMEOUT_S` | 120 | per request |
| `DATAPILOT_BIND` / `DATAPILOT_PORT` | 127.0.0.1 / 8501 | host interface and port |

## Security notes (read before exposing it)
- The key lives in `.env` (gitignored) and is passed to the container as an environment variable; it is not in the image.
  Anyone who can run `docker inspect` on the host can read it.
- The app has **no login**. It is bound to localhost by default. To share it, put a reverse proxy with authentication in
  front, and only then set `DATAPILOT_BIND=0.0.0.0`. Everyone using it spends your OpenRouter credit.
- Each browser session has private folders under the `datapilot_data` / `datapilot_outputs` volumes; idle sessions close
  after 30 minutes and at most 5 run at once.
- `run_pandas` is a guardrail, **not a sandbox**. Do not expose it to untrusted users.
- The container runs as a non-root user (uid 10001).

## Live eval (costs money)
```bash
export OPENROUTER_API_KEY=...                     # PowerShell: $env:OPENROUTER_API_KEY = "..."
./scripts/run_live_eval.sh slice                  # 4 runs: read time per run and llm_calls per run
./scripts/run_live_eval.sh full run1 3 2          # background, resumable, 2 workers
python -m eval.summarize eval/results/run1 --update-readme
```
Copy `eval/data/shopping_trends.csv` and `SeoulBikeData.csv` into `eval/data/` first (gitignored). The scripts run natively
(no container needed): `python -m eval.run_eval --name run1 --repeats 3` does the same.

## Verification status
Verified in the build sandbox: tests, `docker compose config`, shell syntax. **Not verified:** `docker build`, the container
healthcheck, and any real OpenRouter call (the sandbox has no Docker daemon and cannot reach openrouter.ai).
