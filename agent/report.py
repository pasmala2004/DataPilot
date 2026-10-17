"""Report: a Markdown report whose every number comes from code.

    build_report(run_result, llm) -> {"markdown", "facts", "issues", "path", ...}

Code builds everything except the narrative. The narrative is written by the Writer LLM, which
never sees a value and never types a number: it refers to figures as ``{{F3}}`` and code swaps in
the fact's display text (``agent/facts.py``).

The number check (code, no LLM) runs on the Writer's text BEFORE rendering:

1. every ``{{Fn}}`` exists;
2. no digits outside placeholders (allowed: 4-digit years 1990-2100, quarter tokens Q1-Q4, ISO
   month/date tokens such as 2024-03, numbered list markers, and names from the data that contain
   both a letter and a digit, such as a column ``q1_sales``);
3. every finished task has a section.

If it fails, the issues go back to the Writer for one more round (at most 2 rounds in all). A part
(a section or the summary) that still fails is replaced by a sentence built in code from that
task's facts, and the Caveats say so. Text that failed the check is never shipped.

Report sections, in order: Title, Summary, Key figures, Analysis, Charts, Caveats, Sources.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from pydantic import BaseModel, Field

from agent.facts import Fact, build_facts, facts_by_id, failed_task_indices
from agent.llm import ainvoke_structured
from agent.observe import entry_problems
from agent.prompts import writer_feedback, writer_messages
from agent.report_doc import (
    AnalysisSection, ChartRef, KeyFigureRow, PandasOutput, ReportDoc, SourceRow, to_markdown,
)
from core import Config, safe_path

MAX_ROUNDS = 2
MAX_PANDAS_OUTPUT_CHARS = 2000
MAX_PANDAS_OUTPUT_LINES = 5
TRUNCATED_MARK = "(truncated)"
MAX_FALLBACK_FACTS = 12
DROPPED_PREFIX = "rows dropped for missing values in "

PLACEHOLDER = re.compile(r"\{\{\s*(F\d+)\s*\}\}")
CAUSAL = re.compile(r"\b(because|caused by|due to|leads? to|led to|results? in|resulted in)\b", re.IGNORECASE)
_ISO_DATE = re.compile(r"\b(?:199\d|20\d\d|2100)-(?:0[1-9]|1[0-2])(?:-(?:0[1-9]|[12]\d|3[01]))?\b")
_QUARTER = re.compile(r"\bQ[1-4]\b")
_YEAR = re.compile(r"\b(?:199\d|20\d\d|2100)\b")
_LIST_MARKER = re.compile(r"(?m)^[ \t]*\d{1,2}[.)](?=\s)")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")

SUMMARY = "summary"
Part = Any  # "summary" or a task index


# --------------------------------------------------------------------------- #
# Writer output schema (sections first, so the summary is generated last)
# --------------------------------------------------------------------------- #
class WriterSection(BaseModel):
    """The narrative of one task."""

    task_idx: int = Field(description="task_idx of the task, as shown in the input")
    text: str = Field(description="A few plain sentences; every figure is a {{F..}} placeholder, never a typed number")


class WriterOutput(BaseModel):
    """The writer's reply: one section per finished task, then the summary."""

    sections: list[WriterSection] = Field(default_factory=list, description="One section per finished task")
    summary: str = Field(default="", description="2 to 3 sentences, written last, figures as placeholders only")


# --------------------------------------------------------------------------- #
# Number check (pure code)
# --------------------------------------------------------------------------- #
def has_letter_and_digit(name: str) -> bool:
    return any(c.isalpha() for c in name) and any(c.isdigit() for c in name)


def allowed_literals(run_result: Mapping[str, Any]) -> set[str]:
    """Names from the data that may appear in the narrative even though they contain digits.

    Column names, the file and sheet names, and group labels, but only those with BOTH a letter and a
    digit (``q1_sales``, ``Zone 3``). A purely numeric label such as ``3`` is never allowed, because
    allowing it would let that digit through everywhere.
    """
    names: set[str] = {str(run_result.get("file") or ""), str(run_result.get("sheet") or "")}
    names.update(str(c) for c in ((run_result.get("schema") or {}).get("columns") or {}))
    for entry in run_result.get("results") or []:
        output = entry.get("output")
        if not isinstance(output, Mapping) or "error" in output:
            continue
        for group in output.get("groups") or []:
            if isinstance(group, Mapping):
                names.update(str(v) for k, v in group.items() if k != "value" and isinstance(v, str))
        for key in ("group_means", "group_sizes"):
            if isinstance(output.get(key), Mapping):
                names.update(str(k) for k in output[key])
    return {n for n in names if has_letter_and_digit(n)}


def check_text(text: str, valid_ids: set[str], literals: Sequence[str] = ()) -> list[str]:
    """Number-check one piece of Writer text. Returns issues (empty list = passes)."""
    issues: list[str] = []
    for fact_id in PLACEHOLDER.findall(text):
        message = f"{{{{{fact_id}}}}} is not a known fact id"
        if fact_id not in valid_ids and message not in issues:
            issues.append(message)
    masked = PLACEHOLDER.sub(" ", text)
    for literal in sorted(literals, key=len, reverse=True):
        masked = masked.replace(literal, " ")
    for pattern in (_ISO_DATE, _QUARTER, _YEAR, _LIST_MARKER):
        masked = pattern.sub(" ", masked)
    if "{{" in masked or "}}" in masked:
        issues.append("malformed placeholder: write figures exactly as {{F1}}, {{F2}}, ...")
    for token in _NUMBER.findall(masked):
        message = f"stray number '{token}' outside placeholders: use a {{{{F..}}}} placeholder"
        if message not in issues:
            issues.append(message)
    return issues


def check_output(
    output: WriterOutput, valid_ids: set[str], required_tasks: Sequence[int], literals: Sequence[str] = ()
) -> dict[Part, list[str]]:
    """Number-check a whole Writer reply. Returns ``{part: issues}`` for the parts that fail only.

    Parts are ``"summary"`` and the task indices that must have a section. Sections for other
    tasks are ignored (they are never shipped).
    """
    problems: dict[Part, list[str]] = {}
    summary = output.summary.strip()
    found = check_text(summary, valid_ids, literals) if summary else ["the summary is empty"]
    if found:
        problems[SUMMARY] = found
    seen: dict[int, str] = {}
    duplicates: set[int] = set()
    for section in output.sections:
        if section.task_idx in seen:
            duplicates.add(section.task_idx)
        else:
            seen[section.task_idx] = section.text.strip()
    for idx in required_tasks:
        text = seen.get(idx, "")
        found = [f"there is no section for task_idx {idx}"] if not text else check_text(text, valid_ids, literals)
        if idx in duplicates:
            found.append(f"task_idx {idx} has more than one section: write exactly one")
        if found:
            problems[idx] = found
    return problems


def render(text: str, facts: Mapping[str, Mapping[str, Any]]) -> str:
    """Replace every ``{{Fn}}`` with the fact's display text (ids were validated before)."""
    return PLACEHOLDER.sub(lambda m: str(facts[m.group(1)]["text"]), text)


def part_name(part: Part) -> str:
    return "the summary" if part == SUMMARY else f"section for task {int(part) + 1}"


# --------------------------------------------------------------------------- #
# Writer rounds and code fallbacks
# --------------------------------------------------------------------------- #
def fallback_section(task_facts: Sequence[Mapping[str, Any]]) -> str:
    """Section built in code from a task's facts (placeholders, rendered later like any other text)."""
    if not task_facts:
        return "This task produced no figures that can be quoted here. See the Sources appendix."
    shown = task_facts[:MAX_FALLBACK_FACTS]
    lines = [f"- {f['label']}: {{{{{f['fact_id']}}}}}" for f in shown]
    more = len(task_facts) - len(shown)
    tail = f"\n- ... and {more} more figures (see Key figures)." if more > 0 else ""
    return "The figures computed for this task were:\n" + "\n".join(lines) + tail


FALLBACK_SUMMARY = (
    "This text was generated without the language model. The figures computed for each task are listed in "
    "Key figures and in the sections below."
)


class WriterResult(BaseModel):
    """What the writer rounds produced (all text still contains placeholders)."""

    summary: str
    sections: dict[int, str]
    llm_parts: list[Any] = Field(default_factory=list, description='parts whose text came from the LLM ("summary" or a task index)')
    fallbacks: list[str] = Field(default_factory=list, description="Caveat lines for parts built without the LLM")
    issues: list[str] = Field(default_factory=list, description="every number-check issue found, with round and part")
    rounds: int = 0


def _previous_text(output: Optional[WriterOutput]) -> str:
    if output is None:
        return "(the previous reply could not be parsed)"
    parts = [f"task_idx {s.task_idx}: {s.text}" for s in output.sections] + [f"summary: {output.summary}"]
    return "\n".join(parts)[:3000]


async def write_narrative(
    llm: Any,
    question: str,
    tasks: Sequence[Mapping[str, Any]],
    facts: Sequence[Fact],
    *,
    literals: Sequence[str] = (),
    max_rounds: int = MAX_ROUNDS,
) -> WriterResult:
    """Run the Writer with the number check and at most ``max_rounds`` rounds; fall back per part.

    A part that passed the check in any round is kept (the latest passing version wins), so a later
    round cannot make a good part worse. Parts that never passed are built in code.
    """
    valid_ids = {f["fact_id"] for f in facts}
    required = [int(t["idx"]) for t in tasks]
    accepted: dict[Part, str] = {}
    result = WriterResult(summary="", sections={})
    messages = writer_messages(question, tasks, facts)
    pending: dict[Part, list[str]] = {SUMMARY: ["not written yet"], **{i: ["not written yet"] for i in required}}
    previous: Optional[WriterOutput] = None
    llm_error = ""

    for round_no in range(1, max(1, max_rounds) + 1):
        if round_no > 1:
            flat = [f"{part_name(p)}: {m}" for p, found in pending.items() for m in found]
            messages = messages[:2] + [("human", writer_feedback(flat, _previous_text(previous)))]
        result.rounds = round_no
        try:
            output = await ainvoke_structured(llm, WriterOutput, messages)
        except Exception as exc:  # noqa: BLE001 - a writer failure must not crash the report
            llm_error = f"{type(exc).__name__}: {' '.join(str(exc).split())[:200]}"
            result.issues.append(f"round {round_no}: the writer failed ({llm_error})")
            previous = None
            if "StructuredOutputError" not in type(exc).__name__:
                break  # a network/API failure: asking again would not help
            continue
        previous = output
        pending = check_output(output, valid_ids, required, literals)
        for part, found in pending.items():
            result.issues.extend(f"round {round_no}, {part_name(part)}: {m}" for m in found)
        by_idx = {}
        for section in output.sections:
            by_idx.setdefault(section.task_idx, section.text.strip())
        if SUMMARY not in pending:
            accepted[SUMMARY] = output.summary.strip()
        for idx in required:
            if idx not in pending:
                accepted[idx] = by_idx[idx]
        if not pending:
            break

    for part in [SUMMARY, *required]:
        if part in accepted:
            result.llm_parts.append(part)
            continue
        why = f"writer error: {llm_error}" if llm_error else "number check failed"
        label = "summary" if part == SUMMARY else f"section {int(part) + 1}"
        result.fallbacks.append(f"{label}: generated without the LLM ({why})")
        accepted[part] = (
            FALLBACK_SUMMARY if part == SUMMARY else fallback_section([f for f in facts if f["task_idx"] == part])
        )
    result.summary = accepted[SUMMARY]
    result.sections = {idx: accepted[idx] for idx in required}
    return result


# --------------------------------------------------------------------------- #
# Markdown pieces
# --------------------------------------------------------------------------- #
ROWS_USED_PREFIX = "rows used for "


def key_figure_rows(facts: Sequence[Fact]) -> list[KeyFigureRow]:
    """Rows of the Key figures table as ``(ids, figure, value)``.

    Every fact keeps its own row, except the "rows used" and "rows dropped" facts of one tool call, which are shown
    ONCE as a pair (``9 / 1``) at the position of the first of them. All facts stay in the facts list and Sources.
    """
    def kind(fact: Fact) -> Optional[str]:
        label = str(fact["label"])
        if label.startswith(ROWS_USED_PREFIX) and fact["tool"] == "aggregate":
            return "used"
        if label.endswith(": n") and fact["tool"] == "compare_groups":
            return "used"
        if label.startswith(DROPPED_PREFIX):
            return "dropped"
        return None

    pairs: dict[str, dict[str, Fact]] = {}
    for fact in facts:
        which = kind(fact)
        if which:
            pairs.setdefault(str(fact["tool_call_id"]), {}).setdefault(which, fact)
    rows: list[KeyFigureRow] = []
    done: set[str] = set()
    for fact in facts:
        which, call_id = kind(fact), str(fact["tool_call_id"])
        if not which:
            rows.append(KeyFigureRow(fact["fact_id"], str(fact["label"]), str(fact["text"])))
        elif call_id not in done:
            done.add(call_id)
            used, dropped = pairs[call_id].get("used"), pairs[call_id].get("dropped")
            if used and dropped:
                base = str(dropped["label"])[len(DROPPED_PREFIX):]
                rows.append(KeyFigureRow(f"{used['fact_id']}, {dropped['fact_id']}",
                                         f"rows used / rows dropped for missing values: {base}",
                                         f"{used['text']} / {dropped['text']}"))
            else:
                only = used or dropped
                rows.append(KeyFigureRow(only["fact_id"], str(only["label"]), str(only["text"])))
    return rows


def clean_entries(run_result: Mapping[str, Any], path_exists: Callable[[str], bool]) -> list[Mapping[str, Any]]:
    """Clean results-log entries of tasks that did not fail."""
    skipped = failed_task_indices(run_result.get("task_status"))
    return [
        e for e in run_result.get("results") or []
        if e.get("task_idx") not in skipped and not entry_problems(e, path_exists=path_exists)
    ]


def chart_refs(entries: Sequence[Mapping[str, Any]], report_dir: Path) -> list[ChartRef]:
    """Every saved chart once, with a link relative to the Markdown report's folder."""
    refs: list[ChartRef] = []
    links: set[str] = set()
    for entry in entries:
        if entry.get("tool") != "make_chart":
            continue
        args, path = entry.get("args") or {}, str(entry["output"]["path"])
        link = Path(os.path.relpath(path, report_dir)).as_posix().replace(" ", "%20")
        alt = f"{args.get('kind', 'chart')} chart: {args.get('x', '')}" + (f" by {args.get('y')}" if args.get("y") else "")
        if link not in links:
            links.add(link)
            refs.append(ChartRef(path=path, link=link, alt=alt))
    return refs


def cap_lines(text: str, max_lines: int = MAX_PANDAS_OUTPUT_LINES) -> str:
    """At most ``max_lines`` lines; a longer text keeps its first lines and ends with ``(truncated)``."""
    lines = text.split("\n")
    if lines and lines[-1] == "":  # a trailing newline is not a line
        lines.pop()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[:max_lines]) + "\n" + TRUNCATED_MARK


def source_rows(facts: Sequence[Fact], entries: Sequence[Mapping[str, Any]]) -> tuple[list[SourceRow], list[PandasOutput]]:
    """Every tool call that was used: fact ids, tool, arguments, tool_call_id (+ run_pandas printed output)."""
    by_call: dict[str, list[str]] = {}
    for fact in facts:
        by_call.setdefault(str(fact["tool_call_id"]), []).append(fact["fact_id"])
    rows: list[SourceRow] = []
    outputs: list[PandasOutput] = []
    seen: set[str] = set()
    for entry in entries:
        call_id = str(entry.get("tool_call_id"))
        if entry.get("tool") not in ("aggregate", "compare_groups", "run_pandas", "make_chart") or call_id in seen:
            continue
        seen.add(call_id)
        ids = by_call.get(call_id, [])
        shown = f"{ids[0]}-{ids[-1]}" if len(ids) > 2 else ", ".join(ids)
        args = json.dumps(entry.get("args") or {}, separators=(",", ":"), default=str, ensure_ascii=False)
        rows.append(SourceRow(facts=shown or "-", tool=str(entry["tool"]), arguments=args, tool_call_id=call_id))
        if entry["tool"] == "run_pandas":
            outputs.append(PandasOutput(call_id, cap_lines(str(entry["output"].get("output", ""))[:MAX_PANDAS_OUTPUT_CHARS])))
    return rows, outputs


def build_caveats(
    run_result: Mapping[str, Any],
    facts: Sequence[Fact],
    writer: Optional[WriterResult],
    narrative_text: Sequence[tuple[str, str]],
) -> list[str]:
    """Caveats, all generated in code: failed tasks, retries, dropped rows, fallbacks, causal wording."""
    notes: list[str] = []
    status = list(run_result.get("task_status") or [])
    if run_result.get("error"):
        notes.append(f"The run stopped with an error: {run_result['error']}")
    for item in status:
        n, task = int(item["idx"]) + 1, item.get("task", "")
        if item.get("status") == "failed":
            reasons = "; ".join(item.get("reasons") or []) or "no reason recorded"
            notes.append(f"Task {n} failed and is not part of the figures: {task} (reason: {reasons})")
        if item.get("retries"):
            notes.append(f"Task {n} needed {item['retries']} retr{'y' if item['retries'] == 1 else 'ies'} before it was accepted.")
        if item.get("self_corrections"):
            notes.append(f"Task {n}: the agent corrected {item['self_corrections']} tool problem(s) by itself within an attempt.")
    dropped, seen_labels = [], set()
    for fact in facts:
        if fact["label"].startswith(DROPPED_PREFIX) and fact["value"] and fact["label"] not in seen_labels:
            seen_labels.add(fact["label"])
            dropped.append(f"{fact['text']} row(s) were left out for missing values in {fact['label'][len(DROPPED_PREFIX):]}.")
    notes.extend(dropped)
    if writer is not None:
        notes.extend(writer.fallbacks)
        for name, text in narrative_text:
            for match in sorted({m.group(0).lower() for m in CAUSAL.finditer(text)}):
                notes.append(
                    f"Causal wording ('{match}') appears in {name}. These figures show association only; "
                    "they do not establish a cause."
                )
    tools = sorted({f["tool"] for f in facts}) or ["the analysis tools"]
    notes.append(
        f"All figures in this report were computed by tools ({', '.join(tools)}) from the loaded data. "
        "The language model wrote only the narrative text and never typed a number."
    )
    return notes


def build_doc(
    run_result: Mapping[str, Any],
    facts: Sequence[Fact],
    entries: Sequence[Mapping[str, Any]],
    summary: str,
    sections: Mapping[int, str],
    caveats: Sequence[str],
    report_dir: Path,
) -> ReportDoc:
    """Put the report together as a structured ``ReportDoc`` (rendered by ``to_markdown`` / ``to_docx``)."""
    sheet = f" ({run_result['sheet']})" if run_result.get("sheet") else ""
    finished = [s for s in run_result.get("task_status") or [] if s.get("status") == "finished"]
    rows, pandas_outputs = source_rows(facts, entries)
    return ReportDoc(
        title=f"Analysis report: {run_result.get('file', 'data')}{sheet}",
        question=str(run_result.get("question", "")),
        summary=summary,
        key_figures=key_figure_rows(facts),
        analysis=[
            AnalysisSection(int(s["idx"]), str(s.get("task", "")), sections[int(s["idx"])])
            for s in finished if int(s["idx"]) in sections
        ],
        charts=chart_refs(entries, report_dir),
        caveats=list(caveats),
        sources=rows,
        pandas_outputs=pandas_outputs,
    )


def assemble(*args: Any, **kwargs: Any) -> str:
    """Markdown of ``build_doc(...)`` (same arguments)."""
    return to_markdown(build_doc(*args, **kwargs))


NO_RESULTS_SUMMARY = (
    "Nothing could be verified: no task produced a figure that passed the automatic checks, "
    "so this report states no results."
)


def _report_path(report_dir: Path, now: datetime) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    stem = f"report_{now:%Y%m%d_%H%M%S}"
    for counter in range(1, 1000):
        name = f"{stem}.md" if counter == 1 else f"{stem}_{counter}.md"
        candidate = safe_path(name, report_dir)
        if isinstance(candidate, dict):
            raise ValueError(candidate["error"])
        if not candidate.exists():
            return candidate
    raise FileExistsError("could not find a free report file name")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
async def build_report(
    run_result: Mapping[str, Any],
    llm: Any,
    *,
    output_dir: Optional[str | os.PathLike[str]] = None,
    now: Optional[datetime] = None,
    max_rounds: int = MAX_ROUNDS,
    path_exists: Callable[[str], bool] = os.path.isfile,
) -> dict[str, Any]:
    """Build and save the Markdown report for a finished agent run.

    Args:
        run_result: the dict returned by ``agent.graph.run_agent``.
        llm: the chat model used by the Writer (not called when there are no verified figures).
        output_dir: where ``reports/report_<timestamp>.md`` goes (default: OUTPUT_DIR).
        now: timestamp for the file name (default: the current time).
        max_rounds: Writer rounds (1 = no repair round).

    Returns:
        ``markdown`` (the text), ``facts``, ``issues`` (every number-check issue found, with its
        round), ``path`` (absolute path of the saved file), ``doc`` (the structured ``ReportDoc`` for other
        renderers such as ``to_docx``), plus ``rounds``, ``fallbacks`` and ``caveats``.
    """
    base = Path(output_dir) if output_dir is not None else Config().output_dir
    report_dir = (base / "reports").resolve()
    facts = build_facts(run_result.get("results") or [], run_result.get("task_status"), path_exists=path_exists)
    entries = clean_entries(run_result, path_exists)
    path = _report_path(report_dir, now or datetime.now())

    writer: Optional[WriterResult] = None
    narrative: list[tuple[str, str]] = []
    if facts:
        finished = [s for s in run_result.get("task_status") or [] if s.get("status") == "finished"]
        literals = sorted(allowed_literals(run_result))
        writer = await write_narrative(
            llm, str(run_result.get("question", "")), finished, facts, literals=literals, max_rounds=max_rounds
        )
        lookup = facts_by_id(facts)
        summary = render(writer.summary, lookup)
        sections = {idx: render(text, lookup) for idx, text in writer.sections.items()}
        narrative = [
            (part_name(part), summary if part == SUMMARY else sections[part]) for part in writer.llm_parts
        ]  # only text the LLM wrote is checked for causal wording
        issues, rounds, fallbacks = writer.issues, writer.rounds, writer.fallbacks
    else:
        summary, sections, issues, rounds, fallbacks = NO_RESULTS_SUMMARY, {}, [], 0, []

    caveats = build_caveats(run_result, facts, writer, narrative)
    doc = build_doc(run_result, facts, entries, summary, sections, caveats, path.parent)
    markdown = to_markdown(doc)
    path.write_text(markdown, encoding="utf-8")
    return {
        "markdown": markdown, "doc": doc, "facts": facts, "issues": issues, "path": str(path),
        "rounds": rounds, "fallbacks": fallbacks, "caveats": caveats,
    }
