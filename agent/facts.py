"""Facts: every figure the report may show, built in CODE from the results log.

The writer LLM never types a number. It refers to a figure as ``{{F3}}`` and code swaps in the
fact's display ``text``. A fact is::

    {"fact_id": "F3", "tool_call_id": "call_1", "tool": "aggregate", "task_idx": 0,
     "label": "sum of amount for region=North", "value": 470.25, "text": "470.25"}

Which entries produce facts
---------------------------
* only CLEAN entries (``observe.entry_problems`` finds nothing wrong);
* only from tasks that did not fail (a failed task is reported in the Caveats instead);
* only from ``aggregate`` and ``compare_groups``. ``run_pandas`` returns free text, so it
  produces no facts (it appears only in the Sources appendix); ``make_chart`` has no figures;
* a repeated identical call in the same task (same tool and arguments) yields facts once.

What is extracted
-----------------
``value`` (scalar), each group value (the label names the group), ``n`` / ``rows_used``,
``rows_dropped_null``, ``r``, ``p_value``, ``statistic``, every ``group_means`` and
``group_sizes`` entry.

Display text (code-formatted, no currency or units are ever guessed)
--------------------------------------------------------------------
ints: thousands separators; floats: 2 decimals and thousands separators; ``r``: 3 decimals;
``p_value``: 4 significant digits, or ``<0.001``; text values (such as a minimum date) as they are,
with a midnight time dropped.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from agent.observe import entry_problems

FACT_TOOLS = ("aggregate", "compare_groups")
P_VALUE_FLOOR = 0.001
Fact = dict[str, Any]


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def format_int(value: int) -> str:
    """``1234567`` -> ``1,234,567``."""
    return f"{int(value):,}"


def format_float(value: float) -> str:
    """``1234.5`` -> ``1,234.50`` (2 decimals; never ``-0.00``)."""
    text = f"{value:,.2f}"
    return text[1:] if text == "-0.00" else text


def format_r(value: float) -> str:
    """Correlation with 3 decimals: ``0.9345`` -> ``0.935``."""
    text = f"{value:.3f}"
    return text[1:] if text == "-0.000" else text


def format_p(value: float) -> str:
    """p-value to 4 significant digits (``0.04123`` -> ``0.04123``, ``0.5`` -> ``0.5000``), or ``<0.001``."""
    return "<0.001" if value < P_VALUE_FLOOR else f"{value:#.4g}"


def format_text(value: str) -> str:
    """A text value as it is; an ISO timestamp at midnight is shown as a plain date."""
    if len(value) == 19 and value[10] == "T" and value.endswith("T00:00:00"):
        return value[:10]
    return value.replace("T", " ", 1) if len(value) == 19 and value[10] == "T" else value


def format_value(value: Any) -> str:
    """Display text for a scalar or group value (ints and floats by the rules above)."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return format_int(value)
    if isinstance(value, float):
        return format_float(value)
    return format_text(str(value))


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
def _group_columns(args: Mapping[str, Any]) -> list[str]:
    group_by = args.get("group_by")
    if isinstance(group_by, str):
        return [group_by]
    return [str(c) for c in group_by] if isinstance(group_by, list) else []


def _aggregate_base(args: Mapping[str, Any]) -> str:
    """``sum of amount [region == North; 2024-01-01..2024-03-31]``: what was computed, with its filters."""
    base = f"{args.get('agg')} of {args.get('column')}"
    parts = [
        f"{f.get('column')} {f.get('op')} {f.get('value')}"
        for f in (args.get("filters") or [])
        if isinstance(f, Mapping)
    ]
    period = args.get("period")
    if isinstance(period, Mapping) and period:
        parts.append(f"{period.get('start', '')}..{period.get('end', '')}")
    return f"{base} [{'; '.join(parts)}]" if parts else base


def _group_suffix(group: Mapping[str, Any]) -> str:
    keys = [f"{k}={format_value(v)}" for k, v in group.items() if k != "value"]
    return " for " + ", ".join(keys) if keys else ""


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def _aggregate_facts(args: Mapping[str, Any], output: Mapping[str, Any]) -> list[tuple[str, Any, str]]:
    """(label, value, text) triples for one clean aggregate output."""
    base = _aggregate_base(args)
    found: list[tuple[str, Any, str]] = []
    if "value" in output and output["value"] is not None:
        found.append((base, output["value"], format_value(output["value"])))
    for group in output.get("groups") or []:
        if isinstance(group, Mapping) and group.get("value") is not None:
            found.append((base + _group_suffix(group), group["value"], format_value(group["value"])))
    if _is_number(output.get("rows_used")):
        found.append((f"rows used for {base}", output["rows_used"], format_int(output["rows_used"])))
    if _is_number(output.get("rows_dropped_null")):
        found.append(
            (f"rows dropped for missing values in {base}", output["rows_dropped_null"], format_int(output["rows_dropped_null"]))
        )
    return found


def _compare_facts(args: Mapping[str, Any], output: Mapping[str, Any]) -> list[tuple[str, Any, str]]:
    """(label, value, text) triples for one clean compare_groups output."""
    test, x, y = args.get("test"), args.get("x"), args.get("y")
    if test == "correlation":
        base = f"correlation of {x} and {y}"
    else:
        base = f"{'t-test' if test == 'ttest' else 'ANOVA'} of {y} by {x}"
    found: list[tuple[str, Any, str]] = []
    if _is_number(output.get("n")):
        found.append((f"{base}: n", output["n"], format_int(output["n"])))
    if _is_number(output.get("r")):
        found.append((f"{base}: r", output["r"], format_r(output["r"])))
    if _is_number(output.get("statistic")):
        found.append((f"{base}: statistic", output["statistic"], format_float(output["statistic"])))
    if _is_number(output.get("p_value")):
        found.append((f"{base}: p-value", output["p_value"], format_p(output["p_value"])))
    for name, key in (("mean", "group_means"), ("size", "group_sizes")):
        for group, value in (output.get(key) or {}).items():
            if _is_number(value):
                found.append((f"{base}: {name} of {y} in {x}={group}", value, format_value(value)))
    if _is_number(output.get("rows_dropped_null")):
        found.append(
            (f"rows dropped for missing values in {base}", output["rows_dropped_null"], format_int(output["rows_dropped_null"]))
        )
    return found


def failed_task_indices(task_status: Optional[Sequence[Mapping[str, Any]]]) -> set[int]:
    """Indices of tasks marked ``failed`` in ``task_status``."""
    return {int(s["idx"]) for s in task_status or [] if s.get("status") == "failed"}


def build_facts(
    results: Sequence[Mapping[str, Any]],
    task_status: Optional[Sequence[Mapping[str, Any]]] = None,
    *,
    path_exists: Callable[[str], bool] = os.path.isfile,
) -> list[Fact]:
    """Facts from the results log (see the module docstring for what counts).

    Args:
        results: the full results log ``{task_idx, tool_call_id, tool, args, output}``.
        task_status: per-task status; entries of tasks marked ``failed`` are left out.
        path_exists: how to test that a chart file exists (passed to the cleanliness check).

    Returns:
        Facts with ids ``F1``, ``F2``, ... in log order. Empty when nothing is clean.
    """
    skipped = failed_task_indices(task_status)
    facts: list[Fact] = []
    seen_calls: set[tuple[Any, ...]] = set()
    for entry in results:
        tool, idx = str(entry.get("tool")), entry.get("task_idx")
        if tool not in FACT_TOOLS or idx in skipped or entry_problems(entry, path_exists=path_exists):
            continue
        args, output = entry.get("args") or {}, entry["output"]
        key = (idx, tool, json.dumps(args, sort_keys=True, default=str))
        if key in seen_calls:
            continue
        seen_calls.add(key)
        extract = _aggregate_facts if tool == "aggregate" else _compare_facts
        for label, value, text in extract(args, output):
            facts.append({
                "fact_id": f"F{len(facts) + 1}",
                "tool_call_id": entry.get("tool_call_id"),
                "tool": tool,
                "task_idx": idx,
                "label": label,
                "value": value,
                "text": text,
            })
    return facts


def facts_by_id(facts: Iterable[Mapping[str, Any]]) -> dict[str, Fact]:
    """``{"F1": fact, ...}``."""
    return {f["fact_id"]: dict(f) for f in facts}
