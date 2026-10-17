# CV bullets and LinkedIn draft

Rule: numbers come only from `python -m eval.summarize <run_dir>` (the "CV bullets" section of `summary.md`).
Replace each TODO(results) with the computed value; do not round up.

## CV bullets

* Built DataPilot, a verified Excel/CSV analyst agent (LangGraph, FastMCP, pandas, OpenRouter) where every number in the report is computed by a tool, never by the LLM.
* Designed a fact registry and a no-LLM number check (placeholders filled in code, unknown digits rejected), giving TODO(results) fabricated figures across TODO(results) evaluation runs.
* Built an evaluation harness (20 questions, pandas ground truth, resumable runs, repeats): TODO(results) report-level accuracy with a self-correcting retry loop vs TODO(results) without it.
* Shipped a Streamlit app with per-session isolation, idle timeouts and a Docker Compose deployment (app container, OpenRouter models), tested with 800+ automated tests.

## LinkedIn draft

I built DataPilot: upload an Excel or CSV file, ask a question in plain English, and get a report where every number
is computed by a tool rather than guessed by a language model. The tools and files stay on my machine; only a data summary goes to the model API, and the app asks for consent first.

The part I like most is the verification: the writer model only produces placeholders, code fills in the figures, and a
check rejects any number that did not come from a tool result. I also measured it: 20 questions on 2 datasets, repeated
runs, retries on vs off. Result: TODO(results).

What I learned: TODO(results) (for example which failures were model limits and which were bugs I could fix generally).

Code: TODO(link). #AppliedAI #LLM #DataAnalysis
