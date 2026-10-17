"""Scoring of one evaluation run. Code only: no LLM, no network.

Inputs are the ``run_agent`` result (``results`` log, ``task_status``, ``clarification``, ``error``) and the
``build_report`` dict (``facts``, ``doc``, ``fallbacks``). Output is one flat record with no data rows.

Definitions
-----------
* ``tool_correct``: every expected figure appears among the CLEAN tool outputs (no error, no NaN, ...; the
  same check the agent uses) with relative tolerance 1e-3 and absolute tolerance 1e-6. A group figure must sit
  under its own group label. ``run_pandas`` figures count when the number is printed in its output.
  Unanswerable question: no clean ``aggregate`` / ``compare_groups`` output at all.
* ``report_correct``: for every expected figure, the matching fact's display text is in the report's Key figures
  table (a ``run_pandas`` figure: printed in the report's run_pandas output section, because it has no fact).
  Unanswerable question: the report has zero facts.
* ``chart_ok``: only for questions that ask for a chart: a clean ``make_chart`` PNG exists and is not empty.
* ``fabricated``: any number-check fallback in the report, or any fact on an unanswerable question.
* A run that did not complete (error, timeout, crash) is never correct.
"""

from __future__ import annotations

import math
import os
import re
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from agent.observe import entry_problems
from eval.questions import Expect, Question

from agent.llm import is_rate_limit

REL_TOL = 1e-3
ABS_TOL = 1e-6
FACT_TOOLS = ("aggregate", "compare_groups")
REASONS = ("wrong_value", "no_tool_call", "tool_error", "timeout", "crash", "rate_limit")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

Candidate = tuple[str, Optional[str], float]  # (field, group label, value)


# --------------------------------------------------------------------------- #
# Numbers
# --------------------------------------------------------------------------- #
def close(expected: float, actual: float) -> bool:
    """``math.isclose`` with relative tolerance 1e-3 and absolute tolerance 1e-6 (NaN never matches)."""
    if math.isnan(expected) or math.isnan(actual):
        return False
    return math.isclose(expected, actual, rel_tol=REL_TOL, abs_tol=ABS_TOL)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def label_text(value: Any) -> str:
    """Group label as text: ``18.0`` -> ``18``, ``"Winter"`` -> ``Winter``."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def numbers_in_text(text: str) -> list[float]:
    """Every number printed in a text (``run_pandas`` output)."""
    return [float(m) for m in _NUMBER.findall(text or "")]


# --------------------------------------------------------------------------- #
# Tool outputs
# --------------------------------------------------------------------------- #
def clean_tool_entries(
    results: Iterable[Mapping[str, Any]], path_exists: Callable[[str], bool] = os.path.isfile
) -> list[Mapping[str, Any]]:
    """Results-log entries with no problem (errors, NaN, empty groups, ...), whatever the task's final status."""
    return [e for e in results if not entry_problems(e, path_exists=path_exists)]


def candidates(entry: Mapping[str, Any]) -> list[Candidate]:
    """Every figure one clean entry reports, as ``(field, group label, value)``."""
    tool, out = str(entry.get("tool")), entry.get("output")
    if not isinstance(out, Mapping):
        return []
    found: list[Candidate] = []
    if tool == "aggregate":
        if _is_number(out.get("value")):
            found.append(("value", None, float(out["value"])))
        for group in out.get("groups") or []:
            if isinstance(group, Mapping) and _is_number(group.get("value")):
                label = " / ".join(label_text(v) for k, v in group.items() if k != "value")
                found.append(("value", label, float(group["value"])))
    elif tool == "compare_groups":
        for key in ("r", "p_value", "statistic"):
            if _is_number(out.get(key)):
                found.append((key, None, float(out[key])))
        for label, value in (out.get("group_means") or {}).items():
            if _is_number(value):
                found.append(("group_means", label_text(label), float(value)))
    elif tool == "run_pandas":
        found.extend(("text", None, n) for n in numbers_in_text(str(out.get("output", ""))))
    return found


def _expect_in(expect: Expect, found: Sequence[Candidate]) -> bool:
    return any(
        field == expect.field and (expect.group is None or label == expect.group) and close(expect.value, value)
        for field, label, value in found
        if expect.group is not None or label is None
    )


def tool_check(expects: Sequence[Expect], entries: Sequence[Mapping[str, Any]]) -> tuple[bool, list[str]]:
    """``(all found, names of the missing figures)`` for answerable questions."""
    found = [c for e in entries for c in candidates(e)]
    missing = [x.name for x in expects if not _expect_in(x, found)]
    return not missing, missing


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _fact_matches(fact: Mapping[str, Any], expect: Expect) -> bool:
    value, label = fact.get("value"), str(fact.get("label", ""))
    if not _is_number(value) or not close(expect.value, float(value)):
        return False
    if label.startswith(("rows used", "rows dropped")):
        return False
    if expect.group is not None:
        return label.endswith(f"={expect.group}") or f"={expect.group}," in label
    if expect.field == "r":
        return label.endswith(": r")
    if expect.field == "p_value":
        return label.endswith(": p-value")
    if expect.field == "statistic":
        return label.endswith(": statistic")
    return " for " not in label and ": " not in label  # a plain scalar value


def _table(doc: Any) -> list[tuple[set[str], str]]:
    return [(set(re.split(r"\s*,\s*", str(r.ids))), str(r.value)) for r in (getattr(doc, "key_figures", None) or [])]


def report_check(expects: Sequence[Expect], report: Optional[Mapping[str, Any]]) -> tuple[bool, list[str]]:
    """``(all shown, names of the figures missing from the Key figures table)`` for answerable questions."""
    if not report:
        return False, [x.name for x in expects]
    facts, doc = report.get("facts") or [], report.get("doc")
    table = _table(doc)
    pandas_numbers = [n for p in getattr(doc, "pandas_outputs", None) or [] for n in numbers_in_text(p.text)]
    missing: list[str] = []
    for expect in expects:
        if expect.field == "text":
            ok = any(close(expect.value, n) for n in pandas_numbers)
        else:
            ok = any(
                _fact_matches(f, expect)
                and any(f["fact_id"] in ids and str(f["text"]) in re.split(r"\s*/\s*", value) for ids, value in table)
                for f in facts
            )
        if not ok:
            missing.append(expect.name)
    return not missing, missing


def chart_check(entries: Sequence[Mapping[str, Any]]) -> bool:
    """A clean ``make_chart`` output whose PNG exists and is not empty."""
    for entry in entries:
        path = (entry.get("output") or {}).get("path") if entry.get("tool") == "make_chart" else None
        if isinstance(path, str) and os.path.isfile(path) and os.path.getsize(path) > 0:
            return True
    return False


# --------------------------------------------------------------------------- #
# One run
# --------------------------------------------------------------------------- #
def error_type(message: Optional[str]) -> Optional[str]:
    """``"ConnectError: refused"`` -> ``ConnectError``; a free-text message -> ``Error``."""
    if not message:
        return None
    head = str(message).split(":", 1)[0].strip()
    return head if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", head) else "Error"


def score_run(
    question: Question,
    expects: Sequence[Expect],
    result: Optional[Mapping[str, Any]],
    report: Optional[Mapping[str, Any]],
    *,
    job_error: Optional[str] = None,
    latency_s: float = 0.0,
    timed_out: bool = False,
    path_exists: Callable[[str], bool] = os.path.isfile,
) -> dict[str, Any]:
    """Score one run into a flat record (no data rows).

    Args:
        question: the question asked.
        expects: its ground truth (``question.truth(df)``).
        result: the ``run_agent`` result (``None`` when the job itself failed).
        report: the ``build_report`` dict (``None`` when there was no report).
        job_error: a failure of the job itself (crash text); ``timed_out`` marks a timeout.
        latency_s: wall-clock seconds.
    """
    result = result or {}
    results = list(result.get("results") or [])
    statuses = list(result.get("task_status") or [])
    run_error = job_error or result.get("error")
    completed = not timed_out and not run_error
    entries = clean_tool_entries(results, path_exists)
    facts = list((report or {}).get("facts") or [])
    fallbacks = list((report or {}).get("fallbacks") or [])

    missing: list[str] = []
    if not completed:
        tool_ok = report_ok = False
    elif question.unanswerable:
        tool_ok = not any(e.get("tool") in FACT_TOOLS for e in entries)
        report_ok = not facts
    else:
        tool_ok, missing = tool_check(expects, entries)
        report_ok, _ = report_check(expects, report)
    chart_ok = (chart_check(entries) if completed else False) if question.expects_chart else None
    fabricated = bool(fallbacks) or (question.unanswerable and bool(facts))

    reason: Optional[str] = None
    if timed_out:
        reason = "timeout"
    elif run_error:
        reason = "rate_limit" if is_rate_limit(str(run_error)) else "crash"
    elif not (tool_ok and report_ok):
        if question.unanswerable:
            reason = "wrong_value"  # it produced figures it could not have
        elif not results:
            reason = "no_tool_call"
        elif not any(e.get("tool") in (*FACT_TOOLS, "run_pandas") for e in entries):
            reason = "tool_error"
        else:
            reason = "wrong_value"

    return {
        "completed": completed,
        "failed_tasks": sum(1 for s in statuses if s.get("status") == "failed"),
        "retries": sum(int(s.get("retries") or 0) for s in statuses),
        "self_corrections": sum(int(s.get("self_corrections") or 0) for s in statuses),
        "tool_calls": len(results),
        "number_check_fallbacks": len(fallbacks),
        "latency_s": round(float(latency_s), 2),
        "error_type": "Timeout" if timed_out else error_type(str(run_error) if run_error else None),
        "failure_reason": reason,
        "tool_correct": tool_ok,
        "report_correct": report_ok,
        "chart_ok": chart_ok,
        "fabricated": fabricated,
        "facts": len(facts),
        "clarification": bool(result.get("clarification")),
        "missing": missing,
    }
