# DataPilot

Upload a CSV/Excel file, ask a question in plain English, get an answer with a verified report. Every number is computed by tools (pandas/scipy) — the LLM only plans and writes.

> **Privacy:** Column names, 5-row preview, and tool outputs go to OpenRouter. Don't upload confidential data.

## Quickstart

```bash
# Setup (Python 3.10+)
python -m venv .venv && source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Add your key to .env (already gitignored)
echo "OPENROUTER_API_KEY=sk-or-v1-your-key" > .env

# Run the app
streamlit run app.py
```

Get an OpenRouter API key at https://openrouter.ai/keys (needs credit).

## How to use

1. Open http://localhost:8501
2. Pick a model (default: `openai/gpt-4o-mini`)
3. Click **Check key and model** → should show "ready"
4. Upload your `.xlsx` or `.csv` file
5. Ask: "Total sales by region" or "Show top 5 products"
6. Click **Run** and review the result

Report downloads (Markdown/DOCX) unlock after review.

## What it does

```
Question → Router → Analysis agents → Tools (pandas) → Facts → Writer (placeholders) → Number check → Report
```

**6 Tools:** `list_sheets`, `load_data`, `aggregate`, `compare_groups`, `run_pandas`, `make_chart`  
**Safety:** Numbers only from tools, verified before the report.

## Command line

```bash
python run.py --file sales.csv "total by region" --report
```

Report saved to `outputs/reports/`.

## Tests

```bash
pytest -q   # No network needed, uses temp files
```

## Configuration

Optional settings via environment or `.env`:
- `OPENROUTER_MODEL` (default: `openai/gpt-4o-mini`)
- `OPENROUTER_TIMEOUT_S` (default: 120)
- `DATA_DIR` / `OUTPUT_DIR` (default: `./data` / `./outputs`)

## More documentation

- Architecture & rules: `docs/CONTEXT.md`
- Progress tracking: `docs/PROGRESS.md`
- Docker deployment: `docs/DEPLOY.md`

---

<!-- EVAL:START -->
## Verification Status

**Model tested:** `openai/gpt-4o-mini` via OpenRouter  
**Date:** October 17, 2026  
**Status:** Core functionality verified, full evaluation pending

### ✅ Verified Working

1. **Basic Analysis Pipeline**
   - MCP server communication functional
   - Tool calls executing correctly (tested with `aggregate`)
   - Report generation working
   - Number verification system operational

2. **Sample Test Run**
   - Dataset: 7-row sales CSV (product, region, amount, quantity)
   - Query: "Total amount by region"
   - Results: All 4 regional totals computed correctly
   - Report: Generated successfully with proper fact tracking
   - Latency: ~2-3 seconds for simple aggregation

3. **Report Quality**
   - ✅ All figures sourced from tool outputs (no LLM-generated numbers)
   - ✅ Fact registry working (F1-F6 tracked correctly)
   - ✅ Key figures table populated accurately
   - ⚠️ Minor: Event loop closure caused narrative fallback (figures unaffected)

### 📋 Full Evaluation Status

**Pending:** Comprehensive evaluation on standard datasets requires:
- `shopping_trends.csv` - Public shopping dataset (~3,900 rows)
- `SeoulBikeData.csv` - Seoul bike sharing dataset

**To run full evaluation:**
```bash
# Place datasets in eval/data/
python -m eval.run_eval --name production --limit 4 --repeats 1  # test slice
python -m eval.run_eval --name production --repeats 3             # full run
python -m eval.summarize eval/results/production --update-readme
```

**Expected metrics** (once evaluation completes):
- `tool_correct`: Percentage of runs where expected figures appear in tool outputs
- `report_correct`: Percentage where correct figures reach the final report
- `completion`: Percentage of runs that finish without timeout/error
- Mean retries, self-corrections, and latency per question type

This section will be replaced with actual evaluation results after running the full 20-question test suite.
<!-- EVAL:END -->
