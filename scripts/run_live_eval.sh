#!/usr/bin/env bash
# Live eval through OpenRouter (costs money: every run makes several paid model calls).
# Step 1: a 4-question slice to measure time per run and llm_calls per run.
# Step 2 (after you read the numbers): ./scripts/run_live_eval.sh full <name> <repeats> [concurrency]
# Needs OPENROUTER_API_KEY in the environment (or in .env) and eval/data/shopping_trends.csv + SeoulBikeData.csv (gitignored).
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -f .env ]; then
  OPENROUTER_API_KEY="$(grep -E '^OPENROUTER_API_KEY=' .env | cut -d= -f2-)"; export OPENROUTER_API_KEY
fi
[ -n "${OPENROUTER_API_KEY:-}" ] || { echo "set OPENROUTER_API_KEY (environment or .env)" >&2; exit 2; }
case "${1:-slice}" in
  slice) python -m eval.run_eval --name try --limit 4 --repeats 1 ;;
  full)  nohup python -m eval.run_eval --name "${2:?name}" --repeats "${3:-3}" --concurrency "${4:-1}" > "eval/${2}.log" 2>&1 &
         echo "started in background; tail -f eval/${2}.log (resumable: rerun the same command)";;
  *) echo "usage: $0 [slice | full <name> <repeats> [concurrency]]" >&2; exit 2;;
esac
