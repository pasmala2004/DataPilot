"""Prompts and prompt-building helpers for the router, the two agents and the decision node.

Nothing here puts raw data rows in a prompt: the schema is column names, types, null counts
and date ranges only (the preview rows that ``load_data`` returns are dropped before the
state is built).
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

MAX_TASKS = 4
MAX_TOOL_ITERATIONS = 6
DECISION_MAX_WORDS = 150
MAX_OBSERVATION_CHARS = 2000


# --------------------------------------------------------------------------- #
# Schema rendering
# --------------------------------------------------------------------------- #
def render_schema(dataset_id: str, schema: Mapping[str, Any]) -> str:
    """Plain-text description of the loaded dataset (no cell values except date min/max)."""
    lines = [f'dataset_id: "{dataset_id}"', f"rows: {schema.get('rows', '?')}", "columns (name: type):"]
    for name, dtype in (schema.get("columns") or {}).items():
        lines.append(f"  - {name}: {dtype}")
    nulls = schema.get("nulls") or {}
    lines.append("missing values: " + (", ".join(f"{c}={n}" for c, n in nulls.items()) if nulls else "none"))
    ranges = schema.get("date_ranges") or {}
    if ranges:
        lines.append("date ranges: " + "; ".join(f"{c}: {lo} to {hi}" for c, (lo, hi) in ranges.items()))
    for warning in schema.get("warnings") or []:
        lines.append(f"warning: {warning}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
ROUTER_SYSTEM = f"""You are the router of a spreadsheet analysis assistant. A spreadsheet is already loaded. \
You receive its schema and the user's question. Your only job is to PLAN: split the question into \
1 to {MAX_TASKS} tasks and assign each task to an agent. You never answer the question and never compute numbers.

Agents:
- "analysis": computes numbers with tools: totals, averages, medians, counts, minimum/maximum, distinct counts, \
grouped or filtered figures, top-N rankings, trends by month/quarter/year, correlations, t-tests, ANOVA, and \
anything else that needs a calculation.
- "chart": draws ONE chart (bar, line, histogram or scatter) from the loaded data.

Rules:
1. Each task must be self-contained. The agent sees only its own task text and the schema, not the user's \
question and not the other tasks. Name the exact column names from the schema (same spelling), the \
aggregation (sum, mean, count, ...), any filters, the date column and period/frequency, and the grouping.
2. Use only columns that exist in the schema. Never invent a column or a value.
3. Put chart tasks last, and add a chart task only when the user asked for a chart, plot, graph or visual. \
One chart per chart task.
4. Prefer few tasks: merge steps that one tool call can do. Do not exceed {MAX_TASKS} tasks.
5. If the request is too vague to plan (no measurable question, the columns involved are unclear, or it is \
unrelated to the data), set "clarification" to ONE short question for the user and return no tasks. \
Otherwise "clarification" must be null."""


def router_messages(question: str, dataset_id: str, schema: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Messages for the router call."""
    human = f"Schema of the loaded spreadsheet:\n{render_schema(dataset_id, schema)}\n\nUser question:\n{question}"
    return [("system", ROUTER_SYSTEM), ("human", human)]


# --------------------------------------------------------------------------- #
# Agents
# --------------------------------------------------------------------------- #
_COMMON_RULES = f"""Rules:
1. NEVER do arithmetic, estimation, rounding or unit conversion yourself. Every number you report must come \
from a tool result, copied exactly as returned.
2. Use the dataset_id exactly as given below in every tool call. Never call load_data again and never make up an id.
3. Use column names exactly as written in the schema (spelling, case, spaces).
4. If a tool returns an error, read the message (it lists the valid columns or ids) and call the tool again \
with corrected arguments.
5. Use at most {MAX_TOOL_ITERATIONS} rounds of tool calls.
6. When the task is done, reply WITHOUT calling a tool: one factual line stating what was computed, with the \
numbers exactly as the tools returned them."""

ANALYSIS_SYSTEM = """You are the analysis agent of a spreadsheet analysis system. You complete exactly one \
task by calling tools.

{rules}
7. Prefer aggregate (sums, means, counts, groups, filters, periods) and compare_groups (correlation, t-test, \
ANOVA). Use run_pandas only when those cannot express the computation, and print the result or assign it to \
`result`.

Dataset:
{schema}"""

CHART_SYSTEM = """You are the chart agent of a spreadsheet analysis system. You complete exactly one task by \
calling make_chart.

{rules}
7. make_chart takes kind (bar, line, hist, scatter), x, y, agg, freq and title. For bar and line, x is the \
grouping column and y is the column to aggregate with agg. For a date x use freq (month, quarter or year). \
For hist give only x; for scatter give two different numeric columns x and y.
8. When the chart is saved, reply with one line that names the chart kind and the file path the tool returned.

Dataset:
{schema}"""


def agent_system_prompt(agent: str, dataset_id: str, schema: Mapping[str, Any]) -> str:
    """System prompt for the ``analysis`` or ``chart`` agent."""
    template = CHART_SYSTEM if agent == "chart" else ANALYSIS_SYSTEM
    return template.format(rules=_COMMON_RULES, schema=render_schema(dataset_id, schema))


# --------------------------------------------------------------------------- #
# Decision
# --------------------------------------------------------------------------- #
DECISION_SYSTEM = f"""You write the final answer to the user's question about a spreadsheet. You are given the \
user's question and OBSERVATIONS, which are results returned by tools. Use ONLY the observations.

Rules:
1. Every number in your answer must appear in an observation; copy it exactly. Do not calculate, round, \
convert or compare to create new numbers (no sums, differences, percentages or ratios).
2. After each figure, cite the tool_call_id of the observation it came from in square brackets, \
for example [call_abc123].
3. Be brief: under {DECISION_MAX_WORDS} words, plain sentences, no preamble.
4. If a chart was produced, mention its file path.
5. If some tasks failed, end with "Could not complete:" and name each failed task with its reason.
6. If there are no observations, say that no verified results were produced. Never guess."""


def _compact(value: Any, limit: int) -> str:
    text = json.dumps(value, default=str, separators=(",", ":"), ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def render_observations(observations: Sequence[Mapping[str, Any]]) -> str:
    """One line per clean tool result: ``[tool_call_id] tool(args) -> output``."""
    if not observations:
        return "(none)"
    return "\n".join(
        f"[{o['tool_call_id']}] {o['tool']}({_compact(o['args'], 400)}) -> "
        f"{_compact(o['output'], MAX_OBSERVATION_CHARS)}"
        for o in observations
    )


def decision_messages(
    question: str,
    task_status: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
) -> list[tuple[str, str]]:
    """Messages for the decision call.

    ``task_status`` entries look like ``{"idx", "agent", "task", "status", "reasons"}``;
    ``observations`` are the clean results-log entries.
    """
    task_lines = []
    for status in task_status:
        line = f"{status['idx'] + 1}. [{status['agent']}] {status['task']} -> {status['status'].upper()}"
        if status["status"] == "failed" and status.get("reasons"):
            line += f" (reason: {'; '.join(status['reasons'])})"
        task_lines.append(line)
    human = (
        f"User question:\n{question}\n\n"
        f"Tasks:\n{chr(10).join(task_lines) or '(none)'}\n\n"
        f"Observations:\n{render_observations(observations)}"
    )
    return [("system", DECISION_SYSTEM), ("human", human)]


def no_results_answer(task_status: Sequence[Mapping[str, Any]]) -> str:
    """Deterministic answer used when no task produced a clean observation (no LLM call)."""
    text = "No verified results were produced, so I cannot answer this question."
    failed = [s for s in task_status if s["status"] == "failed"]
    if failed:
        text += " Could not complete: " + "; ".join(
            f"task {s['idx'] + 1} ({s['task']})"
            + (f", reason: {'; '.join(s['reasons'])}" if s.get("reasons") else "")
            for s in failed
        ) + "."
    return text


# --------------------------------------------------------------------------- #
# Report writer (Day 3)
# --------------------------------------------------------------------------- #
WRITER_SYSTEM = """You write the narrative of a data-analysis report. You are given the user's question and, \
for each finished task, the task and its FACTS. A fact is a figure already computed by a tool and verified; you \
see only its id, the tool and a label, never its value.

Rules:
1. NEVER type a number. Refer to every figure as a placeholder such as {{F3}}; code replaces it with the verified \
value. No digits, no percentages, no counts in words or digits: use a placeholder for every figure. The only \
digits you may type are four-digit years, quarter names such as Q3, ISO month/date labels such as 2024-03, and \
numbered-list markers.
2. You cannot see the values, so do not say which figure is higher, lower, larger, growing or shrinking, and do \
not compute totals, differences, ratios or percentages. Present what was measured and cite the figures.
3. Use only the listed fact ids. Never invent an id. Refer to a fact only in the section of its own task.
4. Write one section per task, using the task_idx shown. Each section is a few plain sentences. A task without \
facts gets a short sentence saying what it covers.
5. Write the summary LAST: 2 to 3 sentences answering the question, with figures as placeholders only.
6. Describe association, not cause: do not use "because", "caused by", "due to", "leads to" or "results in".
7. Do not mention these rules."""

MAX_WRITER_FACTS_PER_TASK = 80


def writer_messages(
    question: str,
    tasks: Sequence[Mapping[str, Any]],
    facts: Sequence[Mapping[str, Any]],
) -> list[tuple[str, str]]:
    """Messages for the writer.

    ``tasks`` are the FINISHED tasks (``{"idx", "agent", "task"}``); ``facts`` come from
    ``agent.facts.build_facts``. The writer sees fact id, tool and label only: no values, no display
    text, and not the agents' free-text notes (they may contain numbers).
    """
    blocks = []
    for task in tasks:
        own = [f for f in facts if f["task_idx"] == task["idx"]]
        lines = [f"- {f['fact_id']} | {f['tool']} | {f['label']}" for f in own[:MAX_WRITER_FACTS_PER_TASK]]
        if len(own) > MAX_WRITER_FACTS_PER_TASK:
            lines.append(f"- ({len(own) - MAX_WRITER_FACTS_PER_TASK} more facts are listed in the report's Key figures table)")
        blocks.append(
            f"task_idx {task['idx']} [{task['agent']}]: {task['task']}\n"
            + ("facts (id | tool | label):\n" + "\n".join(lines) if lines else "facts: (none)")
        )
    human = f"User question:\n{question}\n\nFinished tasks:\n\n" + "\n\n".join(blocks)
    return [("system", WRITER_SYSTEM), ("human", human)]


def writer_feedback(issues: Sequence[str], previous: str) -> str:
    """Message sent back to the writer when its text failed the number check."""
    listed = "\n".join(f"- {issue}" for issue in issues)
    return (
        "Your report text was rejected by an automatic number check:\n"
        f"{listed}\n\n"
        f"Your previous text:\n{previous}\n\n"
        "Return the complete corrected report (every section and the summary). Remember: never type a number; "
        "use only the listed {{F...}} placeholders."
    )
