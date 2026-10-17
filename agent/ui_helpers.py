"""Small pure helpers for the Streamlit app (no Streamlit import, so they are easy to test).

Nothing here touches data rows: the schema view is built from column names, types, null counts and date
ranges only.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional, Sequence

from agent.llm import hint_for_error
from agent.report_doc import ChartRef

PRIVACY_NOTICE = (
    "External model (OpenRouter): column names, the 5-row preview from load_data, tool results and run_pandas output "
    "(up to 20 rows) are sent to OpenRouter and the model provider. Do not upload confidential data."
)
PRIVACY_CHECKBOX = "I understand my data summary is sent to an external API"
CUSTOM_MODEL_LABEL = "Custom OpenRouter model id (optional, overrides the list)"
# A convenience list only: the "Check key and model" button confirms that an id exists and supports tools.
CURATED_MODELS = (
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "google/gemini-2.0-flash-001",
    "anthropic/claude-3.5-haiku",
    "meta-llama/llama-3.3-70b-instruct",
)
_IMAGE_LINE = re.compile(r"^!\[[^\]]*\]\(([^)]+)\)\s*$")


def effective_model(selected: Optional[str], custom: Optional[str]) -> str:
    """The model id to use: the free-text id when it is not empty, else the selected one."""
    return (custom or "").strip() or (selected or "").strip()


def run_block_reason(*, key_found: bool, model_error: Optional[str] = None, privacy_ack: bool, has_file: bool,
                     has_question: bool, running: bool = False) -> Optional[str]:
    """Why Run must be disabled (a short hint for the user), or ``None`` when it may run."""
    if not key_found:
        return "Set OPENROUTER_API_KEY in the environment that starts the app, then restart it."
    if model_error:
        return f"The model check failed: {model_error}"
    if not privacy_ack:
        return "Tick the box to confirm that your data summary is sent to an external API."
    if running:
        return "A run is in progress."
    if not has_file:
        return "Upload a file first."
    if not has_question:
        return "Type a question."
    return None


def schema_rows(schema: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One row per column for the schema table: name, type, nulls, date range. Never any cell values."""
    nulls = schema.get("nulls") or {}
    ranges = schema.get("date_ranges") or {}
    rows = []
    for column, dtype in (schema.get("columns") or {}).items():
        span = ranges.get(column)
        rows.append({
            "column": str(column),
            "type": str(dtype),
            "nulls": int(nulls.get(column, 0)),
            "date range": f"{str(span[0])[:10]} to {str(span[1])[:10]}" if span and len(span) == 2 else "",
        })
    return rows


def split_for_preview(markdown: str, charts: Sequence[ChartRef]) -> list[tuple[str, str]]:
    """Cut the report Markdown into ``("markdown", text)`` and ``("image", local_path)`` pieces.

    Streamlit cannot load Markdown images from local files, so each ``![alt](link)`` line of a known chart
    becomes an image piece with the chart's real path (shown with ``st.image``); other text stays Markdown.
    """
    by_link = {chart.link: chart.path for chart in charts}
    pieces: list[tuple[str, str]] = []
    buffer: list[str] = []

    def flush() -> None:
        text = "\n".join(buffer).strip("\n")
        if text.strip():
            pieces.append(("markdown", text))
        buffer.clear()

    for line in markdown.splitlines():
        match = _IMAGE_LINE.match(line)
        if match and match.group(1) in by_link:
            flush()
            pieces.append(("image", by_link[match.group(1)]))
        else:
            buffer.append(line)
    flush()
    return pieces


def format_elapsed(seconds: float) -> str:
    """``73.4`` -> ``1m 13s``; ``9`` -> ``9s``."""
    total = int(max(seconds, 0))
    return f"{total // 60}m {total % 60:02d}s" if total >= 60 else f"{total}s"


def problem_messages(job_error: Optional[str], job_hint: Optional[str], result: Optional[Mapping[str, Any]],
                     model: Optional[str] = None) -> list[tuple[str, str]]:
    """Messages to show for a finished job as ``(level, text)``; ``level`` is ``error`` or ``warning``.

    Covers a failed job (key rejected, no credits, rate limit, timeout, crash), an error inside the run, number-check
    fallbacks and failed tasks. Never a stack trace: only the short messages the agent produced, with the
    fix hint when there is one.
    """
    messages: list[tuple[str, str]] = []
    if job_error:
        messages.append(("error", job_error + (f"\n\nFix: {job_hint}" if job_hint else "")))
    if not result:
        return messages
    if result.get("error"):
        hint = hint_for_error(str(result["error"]))
        messages.append(("error", str(result["error"]) + (f"\n\nFix: {hint}" if hint else "")))
    for item in result.get("task_status") or []:
        if item.get("status") == "failed":
            why = "; ".join(str(r) for r in item.get("reasons") or [])
            messages.append(("warning", f"Task {int(item['idx']) + 1} failed and is not in the figures: {item.get('task', '')}"
                                        + (f" ({why})" if why else "")))
    for note in (result.get("report") or {}).get("fallbacks") or []:
        messages.append(("warning", f"Number check: {note}"))
    return messages
