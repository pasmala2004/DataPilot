"""Observation: deterministic checks on what the tools returned. Pure code, NO LLM.

After an agent finishes an attempt at a task, ``observe`` looks at the tool results of
that attempt and answers one question: is this usable evidence, or should the agent try
again? It returns one of three outcomes:

* ``finish``: nothing wrong, the task is done.
* ``retry``:  something is wrong and retries are left; ``feedback`` tells the agent why.
* ``failed``: something is wrong and no retries are left (or ``max_retries`` is 0).

Checks (any one of them triggers a retry):

1. a tool returned ``"error"``
2. no tool was called
3. NaN / inf anywhere in an output
4. empty ``groups`` or ``rows_used == 0``
5. ``p_value`` outside [0, 1]
6. ``|r| > 1``
7. a ``make_chart`` result whose ``path`` is missing

A problem is only held against the attempt if the agent did not fix it: when the same
tool is called again later in the same attempt and that call is clean, the earlier
problem is considered resolved. Without this, an agent that obeys "on a tool error read
the message and fix the arguments" would still lose a retry for an error it already fixed.

The retry limit is the ``max_retries`` parameter (0 turns retrying off). ``resolved_problems``
counts the problems that were fixed within an attempt, which is the self-correction that did
not cost a retry (the graph keeps it per task as ``self_corrections``).
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Sequence

DEFAULT_MAX_RETRIES = 2
R_TOLERANCE = 1e-9  # |r| may exceed 1 by float rounding only
MAX_ERROR_CHARS = 300

Action = Literal["finish", "retry", "failed"]

_LONE_NON_FINITE = re.compile(r"^[+-]?(nan|inf|infinity)$", re.IGNORECASE)


@dataclass(frozen=True)
class Observation:
    """Outcome of observing one attempt.

    ``reasons`` is empty for ``finish``. ``feedback`` is the text to show the agent and
    is only set for ``retry``.
    """

    action: Action
    reasons: tuple[str, ...] = ()
    feedback: str = ""


# --------------------------------------------------------------------------- #
# Per-output checks
# --------------------------------------------------------------------------- #
def _has_non_finite_number(value: Any) -> bool:
    """True if any float anywhere inside ``value`` is NaN or +/-inf."""
    if isinstance(value, bool):
        return False
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, Mapping):
        return any(_has_non_finite_number(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_non_finite_number(v) for v in value)
    return False


def _prints_lone_non_finite(text: Any) -> bool:
    """True if a line of free text is just nan/inf (what ``print(x)`` shows for such a value).

    Only whole lines count: NaN cells inside a printed table are normal missing values.
    """
    if not isinstance(text, str):
        return False
    return any(_LONE_NON_FINITE.match(line.strip()) for line in text.splitlines())


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def entry_problems(
    entry: Mapping[str, Any], *, path_exists: Callable[[str], bool] = os.path.isfile
) -> list[str]:
    """Problems found in ONE results-log entry ({"tool", "output", ...}). Empty list = clean."""
    tool = str(entry.get("tool", "tool"))
    output = entry.get("output")
    if not isinstance(output, Mapping):
        return [f"{tool} returned an unexpected result (not a JSON object)"]

    error = output.get("error")
    if error:
        return [f"{tool} returned an error: {str(error)[:MAX_ERROR_CHARS]}"]

    problems: list[str] = []
    if _has_non_finite_number(output) or (tool == "run_pandas" and _prints_lone_non_finite(output.get("output"))):
        problems.append(f"{tool} produced a NaN or infinite value")
    if output.get("groups") == []:
        problems.append(f"{tool} returned no groups (empty result); check the filters, period and column names")
    if output.get("rows_used") == 0:
        problems.append(f"{tool} used 0 rows (rows_used == 0); check the filters, period and column names")
    p_value = output.get("p_value")
    if _is_number(p_value) and not (0 <= p_value <= 1):
        problems.append(f"{tool} returned p_value {p_value}, which is outside [0, 1]")
    r = output.get("r")
    if _is_number(r) and abs(r) > 1 + R_TOLERANCE:
        problems.append(f"{tool} returned correlation r {r}, but |r| cannot exceed 1")
    if tool == "make_chart":
        path = output.get("path")
        if not path or not isinstance(path, str):
            problems.append("make_chart returned no chart path")
        elif not path_exists(path):
            problems.append("make_chart returned a chart path, but the file does not exist")
    return problems


# --------------------------------------------------------------------------- #
# Per-attempt check
# --------------------------------------------------------------------------- #
def _classify(
    entries: Sequence[Mapping[str, Any]], path_exists: Callable[[str], bool]
) -> tuple[list[list[str]], list[bool]]:
    """Per entry: its problems, and whether a later clean call of the same tool resolved them."""
    per_entry = [entry_problems(e, path_exists=path_exists) for e in entries]
    last_clean: dict[str, int] = {}
    for i, (entry, problems) in enumerate(zip(entries, per_entry)):
        if not problems:
            last_clean[str(entry.get("tool"))] = i
    resolved = [
        bool(problems) and last_clean.get(str(entry.get("tool")), -1) > i
        for i, (entry, problems) in enumerate(zip(entries, per_entry))
    ]
    return per_entry, resolved


def attempt_problems(
    entries: Sequence[Mapping[str, Any]], *, path_exists: Callable[[str], bool] = os.path.isfile
) -> list[str]:
    """Unresolved problems of one attempt (its results-log entries, in call order).

    A problem is resolved when the same tool is called again LATER in the attempt and
    that call is clean. Duplicate reasons are reported once.
    """
    if not entries:
        return ["no tool was called: every number must come from a tool call"]

    per_entry, resolved = _classify(entries, path_exists)
    reasons: list[str] = []
    for problems, fixed in zip(per_entry, resolved):
        if problems and not fixed:
            for reason in problems:
                if reason not in reasons:
                    reasons.append(reason)
    return reasons


def resolved_problems(
    entries: Sequence[Mapping[str, Any]], *, path_exists: Callable[[str], bool] = os.path.isfile
) -> int:
    """Number of problems the agent fixed by itself later in the same attempt.

    It uses the same rule as ``attempt_problems``: a problem counts as fixed when the same
    tool is called again later in the attempt and that call is clean. Every problem of every
    fixed call is counted (an error that took two more tries to fix counts as two), so this is
    the amount of self-correction that happened WITHOUT a retry. Problems that were never
    fixed are not counted (they cost a retry or fail the task).
    """
    per_entry, resolved = _classify(entries, path_exists)
    return sum(len(problems) for problems, fixed in zip(per_entry, resolved) if fixed)


def build_feedback(reasons: Sequence[str], retries_left: int) -> str:
    """The message the agent sees when an attempt is rejected."""
    lines = "\n".join(f"- {reason}" for reason in reasons)
    return (
        "Your previous attempt was rejected by an automatic check:\n"
        f"{lines}\n"
        f"You have {retries_left} retr{'y' if retries_left == 1 else 'ies'} left. "
        "Fix the problem and call the tools again with corrected arguments. "
        "Do not do any arithmetic yourself; every number must come from a tool result. "
        "When done, finish with one factual line."
    )


def observe(
    entries: Sequence[Mapping[str, Any]],
    *,
    retries_used: int,
    max_retries: int = DEFAULT_MAX_RETRIES,
    path_exists: Callable[[str], bool] = os.path.isfile,
) -> Observation:
    """Judge one attempt.

    Args:
        entries: results-log entries produced during this attempt only.
        retries_used: retries already spent on this task (0 on the first attempt).
        max_retries: retries allowed per task; 0 turns retrying off.
        path_exists: how to test that a chart file exists (injectable for tests).
    """
    reasons = attempt_problems(entries, path_exists=path_exists)
    if not reasons:
        return Observation("finish")
    allowed = max(0, int(max_retries))
    if retries_used >= allowed:
        return Observation("failed", tuple(reasons))
    return Observation("retry", tuple(reasons), build_feedback(reasons, allowed - retries_used))
