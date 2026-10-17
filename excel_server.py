"""DataPilot Excel MCP server (stdio).

Thin wrappers over ``core.py``: six tools, one shared in-memory dataset store.
Tools never raise; failures come back as ``{"error": "..."}``.

IMPORTANT: stdout is the MCP protocol channel. Nothing in this process may print
to it. Logging goes to stderr (see ``core.traced``).

Run:  python excel_server.py          (reads DATA_DIR / OUTPUT_DIR from the environment)
"""

from __future__ import annotations

import os
from typing import Any

from mcp.server.fastmcp import FastMCP

import core

# The MCP library logs every request at INFO to stderr; keep that quiet by default. The tool-call logs
# from core.py (tool name, argument names, result size) use their own logger and are not affected.
mcp = FastMCP("datapilot-excel", log_level=os.environ.get("FASTMCP_LOG_LEVEL", "WARNING").upper())  # type: ignore[arg-type]

# Server-side state. DataFrames never leave this process; tools return summaries.
STORE: core.Store = {}
CONFIG = core.Config()  # DATA_DIR / OUTPUT_DIR from the environment, read once at startup


@mcp.tool()
def list_sheets(path: str) -> dict[str, Any]:
    """List the sheets of an Excel (.xlsx) or CSV file in the data folder.

    Args:
        path: File name inside the data folder, e.g. "sales.xlsx".

    Returns {"sheets": [{"name", "rows", "columns"}]}; a CSV has one sheet named "csv".
    On failure returns {"error": "..."}.
    """
    return core.list_sheets(path, config=CONFIG)


@mcp.tool()
def load_data(path: str, sheet: str | None = None) -> dict[str, Any]:
    """Load one sheet into server memory and return its profile.

    Call this once per sheet, then pass the returned dataset_id to the other tools.
    Columns whose name contains "date" are parsed as datetimes. Max 500,000 rows.

    Args:
        path: File name inside the data folder.
        sheet: Sheet name (default: the first sheet; use "csv" for CSV files).

    Returns {"dataset_id", "rows", "columns", "nulls", "date_ranges", "preview"}
    (preview = first 5 rows). On failure returns {"error": "..."}.
    """
    return core.load_data(path, sheet, store=STORE, config=CONFIG)


@mcp.tool()
def aggregate(
    dataset_id: str,
    column: str,
    agg: str,
    group_by: list[str] | str | None = None,
    filters: list[dict[str, Any]] | None = None,
    date_column: str | None = None,
    period: dict[str, str] | None = None,
    freq: str | None = None,
    top_n: int | None = None,
) -> dict[str, Any]:
    """Compute a number (or grouped numbers) from a loaded dataset. All arithmetic happens here.

    Args:
        dataset_id: Id returned by load_data, e.g. "ds1".
        column: Column to aggregate.
        agg: One of sum, count, mean, median, min, max, nunique (sum/mean/median need a numeric column).
        group_by: Column name(s) to group by.
        filters: List of {"column", "op", "value"}; op is one of ==, !=, >, >=, <, <=, in
            ("in" takes a list). Filters are ANDed; null cells never match.
        date_column: Date column used by period and freq.
        period: {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}, inclusive, applied to date_column.
        freq: month, quarter or year; groups date_column by that period.
        top_n: Keep the n largest groups, sorted descending (needs group_by or freq).

    Returns {"value": x} or {"groups": [{<group columns>, "value": x}]} (max 50 groups,
    "truncated": true if cut), always with rows_used and rows_dropped_null.
    On failure returns {"error": "..."} (bad column errors list the available columns).
    """
    return core.aggregate(
        dataset_id, column, agg, group_by, filters, date_column, period, freq, top_n, store=STORE
    )


@mcp.tool()
def compare_groups(dataset_id: str, test: str, x: str, y: str) -> dict[str, Any]:
    """Run a statistical test on a loaded dataset. All arithmetic happens here.

    Args:
        dataset_id: Id returned by load_data.
        test: "correlation" (Pearson; x and y numeric), "ttest" (Welch; x is a column with
            exactly 2 groups, y numeric) or "anova" (x has 2+ groups, y numeric).
        x: Numeric column (correlation) or group column (ttest, anova).
        y: Numeric column.

    correlation returns n, r, p_value; ttest/anova return statistic, p_value,
    group_sizes, group_means. On failure (wrong group count, non-numeric column,
    NaN result) returns {"error": "..."}.
    """
    return core.compare_groups(dataset_id, test, x, y, store=STORE)


@mcp.tool()
def run_pandas(dataset_id: str, code: str) -> dict[str, Any]:
    """Run a short pandas/numpy snippet on a COPY of a loaded dataset (escape hatch).

    The code sees only `df`, `pd` and `np`. Show results with print() or by assigning to a
    variable named `result` (a DataFrame/Series is shown as its first 20 rows). Output is
    capped at 2000 characters. Prefer aggregate and compare_groups when they can do the job.

    Not allowed: import, open, exec, eval, compile, os, sys, subprocess, getattr/setattr,
    double-underscore names, read_* and file-writing to_* methods (to_datetime, to_numeric,
    to_dict, to_list, to_numpy, to_frame, to_period, to_timestamp, to_timedelta and to_series
    are fine). A banned word anywhere in the code is rejected, even inside a string or a
    column name. Runs in a separate process with a 10 second timeout. This is a guardrail,
    not a sandbox.

    Returns {"output": "..."} (plus "truncated": true if cut) or {"error": "..."}.
    """
    return core.run_pandas(dataset_id, code, store=STORE)


@mcp.tool()
def make_chart(
    dataset_id: str,
    kind: str,
    x: str,
    y: str | None = None,
    agg: str = "sum",
    freq: str | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Draw a chart of a loaded dataset and save it as a PNG file.

    Args:
        dataset_id: Id returned by load_data.
        kind: "bar", "line", "hist" or "scatter".
        x: bar/line: grouping column. hist: numeric column. scatter: numeric x column.
        y: bar/line: column to aggregate (may be omitted when agg is "count"). scatter: numeric
            y column. Ignored for hist.
        agg: bar/line aggregation: sum, count, mean, median, min, max, nunique.
        freq: bar/line only: month, quarter or year; groups the date column x by period.
        title: Optional chart title.

    bar/line are limited to 50 bars/points. Returns {"path", "kind", "rows_used",
    "rows_dropped_null"}; path is the absolute path of the PNG. On failure returns {"error": "..."}.
    """
    return core.make_chart(dataset_id, kind, x, y, agg, freq, title, store=STORE, config=CONFIG)


if __name__ == "__main__":
    mcp.run(transport="stdio")
