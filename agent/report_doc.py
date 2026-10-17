"""The structured report (``ReportDoc``) and its Markdown renderer.

``agent/report.py`` builds a ``ReportDoc`` (title, summary, key-figure rows, analysis sections, chart refs,
caveats, sources rows) from the run; renderers turn it into a file: ``to_markdown`` here and ``to_docx`` in
``agent/docx_export.py``. The doc holds plain text only. Every number in it is already final text
(placeholders replaced, figures formatted by ``agent/facts.py``); renderers never format or compute a number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MAX_CELL_CHARS = 160
MAX_LABEL_CHARS = 240


@dataclass(frozen=True)
class KeyFigureRow:
    """One row of the Key figures table. ``ids`` is ``F1`` or ``F3, F4`` for a used/dropped pair."""

    ids: str
    figure: str
    value: str


@dataclass(frozen=True)
class AnalysisSection:
    """The narrative of one finished task."""

    task_idx: int
    task: str
    text: str


@dataclass(frozen=True)
class ChartRef:
    """A saved chart: ``path`` (absolute file), ``link`` (relative to the Markdown report) and ``alt`` text."""

    path: str
    link: str
    alt: str


@dataclass(frozen=True)
class SourceRow:
    """One tool call that was used: fact ids (``F1-F4``, or ``-``), tool, arguments as compact JSON, call id."""

    facts: str
    tool: str
    arguments: str
    tool_call_id: str


@dataclass(frozen=True)
class PandasOutput:
    """Printed output of a ``run_pandas`` call (free text; shown only in Sources, never as a figure)."""

    tool_call_id: str
    text: str


@dataclass
class ReportDoc:
    """Everything a renderer needs. Empty lists mean "none"; the renderers print the matching note."""

    title: str
    question: str
    summary: str
    key_figures: list[KeyFigureRow] = field(default_factory=list)
    analysis: list[AnalysisSection] = field(default_factory=list)
    charts: list[ChartRef] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    sources: list[SourceRow] = field(default_factory=list)
    pandas_outputs: list[PandasOutput] = field(default_factory=list)


# Notes shown by every renderer when a section is empty
NO_FIGURES_NOTE = "No verified figures were produced."
NO_ANALYSIS_NOTE = "No analysis could be verified."
NO_CHARTS_NOTE = "No charts were produced."
NO_SOURCES_NOTE = "No tool results were used."
PANDAS_NOTE = "output (free text, not used as a figure)"


def flat(text: Any) -> str:
    """Collapse all whitespace (including newlines) to single spaces."""
    return " ".join(str(text).split())


def clip(text: Any, limit: int = MAX_CELL_CHARS) -> str:
    """Flatten and shorten text for a table cell (``...`` marks a cut)."""
    one_line = flat(text)
    return one_line if len(one_line) <= limit else one_line[: limit - 3] + "..."


def md_cell(text: Any, limit: int = MAX_CELL_CHARS) -> str:
    """Make text safe inside a Markdown table cell."""
    return clip(text, limit).replace("|", "\\|")


def to_markdown(doc: ReportDoc) -> str:
    """Render the report as Markdown (sections in the fixed order: Title, Summary, Key figures, Analysis,
    Charts, Caveats, Sources)."""
    if doc.key_figures:
        rows = ["| ID | Figure | Value |", "|---|---|---|"]
        rows += [f"| {r.ids} | {md_cell(r.figure, MAX_LABEL_CHARS)} | {md_cell(r.value)} |" for r in doc.key_figures]
        figures = "\n".join(rows)
    else:
        figures = NO_FIGURES_NOTE
    analysis = [f"### Task {s.task_idx + 1}\n\n*{flat(s.task)}*\n\n{s.text}" for s in doc.analysis]
    charts = [f"![{md_cell(c.alt)}]({c.link})" for c in doc.charts]

    if doc.sources:
        src = ["| Facts | Tool | Arguments | tool_call_id |", "|---|---|---|---|"]
        src += [
            f"| {r.facts} | {r.tool} | `{md_cell(r.arguments).replace('`', chr(39))}` | {md_cell(r.tool_call_id)} |"
            for r in doc.sources
        ]
        sources = "\n".join(src)
    else:
        sources = NO_SOURCES_NOTE
    blocks = [
        f"`{md_cell(p.tool_call_id)}` {PANDAS_NOTE}:\n\n```\n{p.text.replace('```', chr(39) * 3)}\n```" for p in doc.pandas_outputs
    ]
    parts = [
        f"# {doc.title}",
        f"*Question: {flat(doc.question)}*",
        "## Summary\n\n" + doc.summary,
        "## Key figures\n\n" + figures,
        "## Analysis\n\n" + ("\n\n".join(analysis) if analysis else NO_ANALYSIS_NOTE),
        "## Charts\n\n" + ("\n\n".join(charts) if charts else NO_CHARTS_NOTE),
        "## Caveats\n\n" + "\n".join(f"- {c}" for c in doc.caveats),
        "## Sources\n\n" + sources + ("\n\n" + "\n\n".join(blocks) if blocks else ""),
    ]
    return "\n\n".join(parts) + "\n"
