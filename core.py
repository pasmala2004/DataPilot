"""DataPilot core: configuration, path safety, tracing, and the analysis tools.

Everything here is a plain function. Nothing is global state: the dataset
``store`` and the ``Config`` are passed in as parameters, so tests (and, later,
the MCP server) decide where data lives.

Rules enforced in this module:
* Tools never raise; they return ``{"error": "..."}``.
* Files are only read from ``DATA_DIR`` and never modified.
* Nothing is printed to stdout (reserved for the MCP stdio protocol);
  logs go to stderr and never contain argument values or cell values.
"""

from __future__ import annotations

import functools
import inspect
import json
import logging
import math
import operator
import os
import pickle
import re
import subprocess
import sys
import uuid
import warnings
import warnings as _warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
import pandas as pd
from scipy import stats

MAX_ROWS = 500_000
PREVIEW_ROWS = 5
ALLOWED_EXTENSIONS = (".xlsx", ".csv")
CSV_SHEET_NAME = "csv"

AGGS = ("sum", "count", "mean", "median", "min", "max", "nunique")
NUMERIC_AGGS = ("sum", "mean", "median")  # need a numeric column
FILTER_OPS = ("==", "!=", ">", ">=", "<", "<=", "in")
FREQS = ("month", "quarter", "year")
MAX_GROUPS = 50
TESTS = ("correlation", "ttest", "anova")

RUN_PANDAS_TIMEOUT = 10.0  # seconds
MAX_OUTPUT_CHARS = 2000
MAX_CODE_CHARS = 10_000
RESULT_PREVIEW_ROWS = 20
CHILD_MEMORY_MB = 4096  # address-space limit for the worker (POSIX only)
SAFE_BUILTINS = (
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float", "int",
    "isinstance", "iter", "len", "list", "map", "max", "min", "next", "pow", "print",
    "range", "repr", "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple",
    "zip", "Exception", "ValueError", "KeyError", "TypeError", "IndexError",
    "ZeroDivisionError", "ArithmeticError",
)

CHART_KINDS = ("bar", "line", "hist", "scatter")
MAX_BARS = 50
_OPERATORS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}

Store = dict[str, pd.DataFrame]
F = TypeVar("F", bound=Callable[..., Any])


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Config:
    """Directories the tools may touch.

    Defaults come from the ``DATA_DIR`` / ``OUTPUT_DIR`` environment variables,
    falling back to ``./data`` and ``./outputs``. Defaults are read when a
    ``Config`` is created, not at import time.
    """

    data_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("DATA_DIR", "./data"))
    )
    output_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("OUTPUT_DIR", "./outputs"))
    )


# --------------------------------------------------------------------------- #
# Logging (stderr only)
# --------------------------------------------------------------------------- #
class _StderrHandler(logging.StreamHandler):
    """Stream handler that always writes to the *current* ``sys.stderr``."""

    def __init__(self) -> None:
        super().__init__(sys.stderr)

    @property  # type: ignore[override]
    def stream(self):  # noqa: D102
        return sys.stderr

    @stream.setter
    def stream(self, value) -> None:  # noqa: D102
        pass  # always follow sys.stderr


logger = logging.getLogger("datapilot")
if not logger.handlers:
    _handler = _StderrHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def traced(func: F) -> F:
    """Wrap a tool: log its name, argument NAMES and result size to stderr.

    Argument values are never logged. Any exception is converted into
    ``{"error": "<Type>: <message>"}`` so tools never raise. For exceptions only
    the exception type is logged, because its message may quote cell values.
    """
    signature = inspect.signature(func)
    name = func.__name__

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            arg_names = list(signature.bind_partial(*args, **kwargs).arguments)
        except TypeError:
            arg_names = list(kwargs)
        try:
            result = func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - tools must never raise
            logger.info("tool=%s args=%s error_type=%s", name, arg_names, type(exc).__name__)
            return {"error": f"{type(exc).__name__}: {exc}"}
        size = len(json.dumps(result, default=str).encode("utf-8"))
        status = "error" if isinstance(result, dict) and "error" in result else "ok"
        logger.info("tool=%s args=%s status=%s result_bytes=%d", name, arg_names, status, size)
        return result

    return wrapper  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Path safety
# --------------------------------------------------------------------------- #
def safe_path(path: str | os.PathLike[str], base: str | os.PathLike[str]) -> Path | dict[str, str]:
    """Resolve ``path`` and make sure it stays inside ``base``.

    Relative paths are interpreted relative to ``base``. Symlinks and ``..``
    are resolved before the check. Returns the resolved ``Path`` on success, or
    ``{"error": "path outside allowed directory"}`` otherwise.
    """
    base_resolved = Path(base).resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = base_resolved / candidate
    try:
        resolved = candidate.resolve()
    except (OSError, RuntimeError, ValueError):  # symlink loop, NUL byte, ...
        return {"error": "invalid path"}
    if not resolved.is_relative_to(base_resolved):
        return {"error": "path outside allowed directory"}
    return resolved


def _resolve_input(path: str, config: Config) -> Path | dict[str, str]:
    """Validate an input file: inside DATA_DIR, .xlsx/.csv, and existing."""
    resolved = safe_path(path, config.data_dir)
    if isinstance(resolved, dict):
        return resolved
    suffix = resolved.suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return {"error": f"unsupported file type '{suffix}': only .xlsx and .csv are supported"}
    if not resolved.is_file():
        return {"error": "file not found"}
    return resolved


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
@traced
def list_sheets(path: str, config: Config | None = None) -> dict[str, Any]:
    """List the sheets of an Excel/CSV file in DATA_DIR.

    Returns ``{"sheets": [{"name", "rows", "columns"}]}``. ``rows`` excludes the
    header row. A ``.csv`` file has a single sheet named ``"csv"``.
    """
    config = config or Config()
    resolved = _resolve_input(path, config)
    if isinstance(resolved, dict):
        return resolved

    if resolved.suffix.lower() == ".csv":
        frames = {CSV_SHEET_NAME: _read_csv(resolved)}
    else:
        frames = pd.read_excel(resolved, sheet_name=None, engine="openpyxl")

    return {
        "sheets": [
            {"name": str(name), "rows": int(len(df)), "columns": int(df.shape[1])}
            for name, df in frames.items()
        ]
    }


CSV_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    """Read a CSV, trying UTF-8 (with BOM), then Windows-1252, then Latin-1 (never fails to decode)."""
    for encoding in CSV_ENCODINGS[:-1]:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding=CSV_ENCODINGS[-1], **kwargs)


def _parse_date_columns(df: pd.DataFrame) -> list[str]:
    """Convert columns whose name contains "date" to datetimes, in place.

    Numeric/boolean columns are skipped (e.g. ``update_count``). If a column
    cannot be parsed it is left untouched and a warning is returned, rather
    than silently turning bad values into nulls.
    """
    warnings: list[str] = []
    for i, col in enumerate(df.columns):
        if "date" not in str(col).lower():
            continue
        series = df.iloc[:, i]
        if (
            pd.api.types.is_datetime64_any_dtype(series)
            or pd.api.types.is_numeric_dtype(series)
            or pd.api.types.is_bool_dtype(series)
        ):
            continue
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", UserWarning)
            try:
                df.isetitem(i, pd.to_datetime(series))
                continue
            except (ValueError, TypeError):
                pass
            try:
                df.isetitem(i, pd.to_datetime(series, dayfirst=True))
                warnings.append(f"column '{col}' parsed as day/month/year (month-first did not fit)")
            except (ValueError, TypeError):
                warnings.append(
                    f"column '{col}' looks like a date but could not be parsed; left as {series.dtype}"
                )
    return warnings


@traced
def load_data(
    path: str,
    sheet: str | None = None,
    *,
    store: Store,
    config: Config | None = None,
) -> dict[str, Any]:
    """Load one sheet into ``store`` and return a profile of it.

    The DataFrame stays in ``store`` (server side); only a summary is returned:
    ``dataset_id`` ("ds1", "ds2", ...), ``rows``, ``columns`` {name: dtype},
    ``nulls`` {col: n} (only n > 0), ``date_ranges`` {col: [min, max]} and a
    ``preview`` of the first 5 rows. ``sheet=None`` loads the first sheet.
    At most ``MAX_ROWS`` rows are accepted. A ``warnings`` list is added only
    when something needs attention (e.g. an unparseable date column).
    """
    config = config or Config()
    resolved = _resolve_input(path, config)
    if isinstance(resolved, dict):
        return resolved

    is_csv = resolved.suffix.lower() == ".csv"
    # Read one row more than the limit so we can detect "too many" cheaply.
    if is_csv:
        names = [CSV_SHEET_NAME]
        target = sheet if sheet is not None else names[0]
        if target not in names:
            return {"error": f"unknown sheet '{target}'; available sheets: {names}"}
        df = _read_csv(resolved, nrows=MAX_ROWS + 1)
    else:
        with pd.ExcelFile(resolved, engine="openpyxl") as workbook:
            names = [str(n) for n in workbook.sheet_names]
            target = sheet if sheet is not None else names[0]
            if target not in names:
                return {"error": f"unknown sheet '{target}'; available sheets: {names}"}
            df = workbook.parse(target, nrows=MAX_ROWS + 1)

    if len(df) > MAX_ROWS:
        return {"error": f"too many rows: more than {MAX_ROWS} (limit is {MAX_ROWS})"}

    df.columns = [str(c) for c in df.columns]
    warnings = _parse_date_columns(df)

    null_counts = df.isna().sum()
    date_ranges: dict[str, list[str]] = {}
    for i, col in enumerate(df.columns):
        series = df.iloc[:, i]
        if pd.api.types.is_datetime64_any_dtype(series) and series.notna().any():
            date_ranges[col] = [series.min().isoformat(), series.max().isoformat()]

    n = len(store) + 1
    while f"ds{n}" in store:
        n += 1
    dataset_id = f"ds{n}"
    store[dataset_id] = df

    result: dict[str, Any] = {
        "dataset_id": dataset_id,
        "rows": int(len(df)),
        "columns": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "nulls": {col: int(c) for col, c in null_counts.items() if c > 0},
        "date_ranges": date_ranges,
        "preview": json.loads(
            df.head(PREVIEW_ROWS).to_json(orient="records", date_format="iso", date_unit="s")
        ),
    }
    if warnings:
        result["warnings"] = warnings
    return result


# --------------------------------------------------------------------------- #
# Shared helpers for the analysis tools
# --------------------------------------------------------------------------- #
def _unknown_dataset(dataset_id: Any, store: Store) -> dict[str, Any]:
    """Error for an unknown dataset id, listing the ids that are loaded."""
    return {
        "error": f"unknown dataset_id '{dataset_id}'",
        "loaded": sorted(store, key=lambda k: (len(k), k)),
    }


def _get_dataset(dataset_id: Any, store: Store) -> pd.DataFrame | None:
    """Return the stored DataFrame, or ``None`` if the id is not loaded."""
    if isinstance(dataset_id, str) and dataset_id in store:
        return store[dataset_id]
    return None


def _has_column(df: pd.DataFrame, name: Any) -> bool:
    return isinstance(name, str) and name in df.columns


def _unknown_column(df: pd.DataFrame, name: Any) -> dict[str, str]:
    """Error for a bad column name, listing the available columns."""
    return {"error": f"unknown column '{name}'; available columns: {list(df.columns)}"}


def _is_numeric(series: pd.Series) -> bool:
    """True for int/float columns (booleans are deliberately not numeric)."""
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _py(value: Any) -> Any:
    """Convert a pandas/numpy scalar to a JSON-safe Python value."""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite result")
    return value


# --------------------------------------------------------------------------- #
# aggregate
# --------------------------------------------------------------------------- #
def _apply_filters(df: pd.DataFrame, filters: Any) -> pd.DataFrame | dict[str, str]:
    """AND together structured filters. Null cells never match any filter."""
    if filters is None:
        return df
    if not isinstance(filters, list):
        return {"error": "filters must be a list of {column, op, value}"}
    mask = pd.Series(True, index=df.index)
    for i, flt in enumerate(filters, start=1):
        if not isinstance(flt, dict) or set(flt) != {"column", "op", "value"}:
            return {"error": f"filter #{i} must be a dict with exactly the keys column, op, value"}
        column, op, value = flt["column"], flt["op"], flt["value"]
        if not _has_column(df, column):
            return _unknown_column(df, column)
        if not isinstance(op, str) or op not in FILTER_OPS:
            return {"error": f"filter #{i}: unsupported op; use one of {list(FILTER_OPS)}"}
        series = df[column]
        is_dt = pd.api.types.is_datetime64_any_dtype(series)
        try:
            if op == "in":
                if not isinstance(value, (list, tuple)):
                    return {"error": f"filter #{i}: op 'in' needs a list value"}
                items = [pd.Timestamp(v) if is_dt else v for v in value]
                part = series.isin(items)
            else:
                if value is None or isinstance(value, (list, tuple, dict, set)):
                    return {"error": f"filter #{i}: op '{op}' needs a single non-null value"}
                part = _OPERATORS[op](series, pd.Timestamp(value) if is_dt else value)
        except (ValueError, TypeError) as exc:
            return {"error": f"filter #{i}: cannot compare column '{column}' with the given value ({type(exc).__name__})"}
        mask &= part & series.notna()
    return df[mask]


def _apply_period(
    df: pd.DataFrame, date_column: str, period: Any
) -> pd.DataFrame | dict[str, str]:
    """Keep rows whose date is within ``period`` {start, end}, both inclusive.

    A date-only ``end`` (e.g. "2024-03-31") includes that whole day. Rows with
    a null date never match.
    """
    if not isinstance(period, dict) or not period or set(period) - {"start", "end"}:
        return {"error": "period must be a dict with 'start' and/or 'end' (ISO dates)"}
    try:
        lo = pd.Timestamp(period["start"]) if period.get("start") is not None else None
        hi = pd.Timestamp(period["end"]) if period.get("end") is not None else None
    except (ValueError, TypeError):
        return {"error": "period start/end must be valid ISO dates, e.g. '2024-01-31'"}
    if lo is None and hi is None:
        return {"error": "period needs a start and/or an end"}
    if lo is not None and hi is not None and lo > hi:
        return {"error": "period start is after end"}
    series = df[date_column]
    mask = series.notna()
    try:
        if lo is not None:
            mask &= series >= lo
        if hi is not None:
            raw_end = period["end"]
            if isinstance(raw_end, str) and len(raw_end.strip()) <= 10:  # date only
                mask &= series < hi + pd.Timedelta(days=1)
            else:
                mask &= series <= hi
    except TypeError:
        return {"error": f"cannot compare column '{date_column}' with period dates (timezone mismatch?)"}
    return df[mask]


def _period_labels(series: pd.Series, freq: str) -> pd.Series:
    """Label each date with its month ("2024-01"), quarter ("2024-Q1") or year ("2024")."""
    if freq == "month":
        labels = series.dt.strftime("%Y-%m")
    elif freq == "quarter":
        labels = (
            series.dt.year.astype("Int64").astype("string")
            + "-Q"
            + series.dt.quarter.astype("Int64").astype("string")
        )
    else:
        labels = series.dt.year.astype("Int64").astype("string")
    return labels.astype(object).where(labels.notna(), None)


@traced
def aggregate(
    dataset_id: str,
    column: str,
    agg: str,
    group_by: list[str] | None = None,
    filters: list[dict[str, Any]] | None = None,
    date_column: str | None = None,
    period: dict[str, str] | None = None,
    freq: str | None = None,
    top_n: int | None = None,
    *,
    store: Store,
) -> dict[str, Any]:
    """Aggregate one column of a loaded dataset. All arithmetic happens here.

    Order of operations: ``filters`` (AND) -> ``period`` on ``date_column`` ->
    drop rows with a null in ``column`` or in any group key -> aggregate.

    * ``agg``: sum | count | mean | median | min | max | nunique
      (sum/mean/median need a numeric column).
    * ``filters``: ``[{"column", "op", "value"}]``, op in == != > >= < <= in.
      Null cells never match a filter.
    * ``period``: ``{"start", "end"}`` inclusive ISO dates on ``date_column``.
    * ``freq``: month | quarter | year; adds a group column named
      ``<date_column>_<freq>`` (labels "2024-01", "2024-Q1", "2024").
    * ``group_by``: list of columns. ``top_n``: keep the n largest groups,
      sorted descending (needs grouping).

    Returns ``{"value": x}`` or ``{"groups": [{<group cols>, "value": x}]}``
    (at most 50 groups; ``"truncated": true`` and ``"total_groups"`` if cut),
    always with ``rows_used`` and ``rows_dropped_null``. Rows removed by
    filters/period are not counted as dropped. Only structured arguments are
    accepted; there are no free-text formulas.
    """
    df = _get_dataset(dataset_id, store)
    if df is None:
        return _unknown_dataset(dataset_id, store)
    if not isinstance(agg, str) or agg not in AGGS:
        return {"error": f"unknown agg; use one of {list(AGGS)}"}
    if not _has_column(df, column):
        return _unknown_column(df, column)

    if group_by is None:
        group_cols: list[str] = []
    elif isinstance(group_by, str):
        group_cols = [group_by]
    elif isinstance(group_by, list):
        group_cols = list(group_by)
    else:
        return {"error": "group_by must be a list of column names"}
    for name in group_cols:
        if not _has_column(df, name):
            return _unknown_column(df, name)
    if len(set(group_cols)) != len(group_cols):
        return {"error": "group_by contains duplicate columns"}

    if freq is not None and (not isinstance(freq, str) or freq not in FREQS):
        return {"error": f"unknown freq; use one of {list(FREQS)}"}
    if date_column is not None:
        if not _has_column(df, date_column):
            return _unknown_column(df, date_column)
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
            return {"error": f"column '{date_column}' is not a date column; date columns: {datetime_cols}"}
    if (period is not None or freq is not None) and date_column is None:
        return {"error": "date_column is required when using period or freq"}

    grouped = bool(group_cols) or freq is not None
    if top_n is not None:
        if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
            return {"error": "top_n must be an integer >= 1"}
        if not grouped:
            return {"error": "top_n needs group_by or freq"}

    series_ok = df[column]
    if agg in NUMERIC_AGGS and not _is_numeric(series_ok):
        return {"error": f"agg '{agg}' needs a numeric column; '{column}' is {series_ok.dtype}"}

    key_names = list(group_cols)
    if freq is not None:
        key_names.insert(0, f"{date_column}_{freq}")
    if "value" in key_names:
        return {"error": "a group column named 'value' would clash with the result field; cannot group by it"}

    # Row selection: filters, then period.
    frame = _apply_filters(df, filters)
    if isinstance(frame, dict):
        return frame
    if period is not None:
        frame = _apply_period(frame, date_column, period)  # type: ignore[arg-type]
        if isinstance(frame, dict):
            return frame

    # Null handling: a row is unusable if the value or any group key is null.
    keys: list[pd.Series] = []
    if freq is not None:
        keys.append(_period_labels(frame[date_column], freq).rename(key_names[0]))  # type: ignore[index]
    keys.extend(frame[name] for name in group_cols)
    values = frame[column]
    valid = values.notna()
    for key in keys:
        valid &= key.notna()
    rows_used = int(valid.sum())
    meta = {"rows_used": rows_used, "rows_dropped_null": int(len(frame) - rows_used)}
    values = values[valid]

    if not grouped:
        if rows_used == 0:
            value: Any = 0 if agg in ("count", "nunique") else None
        else:
            value = _py(values.agg(agg))
        return {"value": value, **meta}

    keys = [k[valid] for k in keys]
    rows: list[tuple[tuple[Any, ...], Any]] = []
    if rows_used:
        result = values.groupby(keys, sort=True, observed=True).agg(agg)
        for index, val in result.items():
            rows.append((index if isinstance(index, tuple) else (index,), val))
    if top_n is not None:
        rows = sorted(rows, key=lambda r: r[1], reverse=True)[:top_n]  # stable: ties keep key order
    total = len(rows)
    out: dict[str, Any] = {
        "groups": [
            {**{name: _py(k) for name, k in zip(key_names, key_tuple)}, "value": _py(val)}
            for key_tuple, val in rows[:MAX_GROUPS]
        ],
        **meta,
    }
    if total > MAX_GROUPS:
        out["truncated"] = True
        out["total_groups"] = total
    return out


# --------------------------------------------------------------------------- #
# compare_groups
# --------------------------------------------------------------------------- #
def _group_labels_hint(labels: list[str]) -> str:
    """Return the group labels if there are few of them, else nothing."""
    return f": {labels}" if len(labels) <= 10 else ""


def _correlation(df: pd.DataFrame, x: str, y: str) -> dict[str, Any]:
    """Pearson correlation between two numeric columns (pairwise null drop)."""
    for name in (x, y):
        if not _is_numeric(df[name]):
            return {"error": f"column '{name}' must be numeric for correlation; it is {df[name].dtype}"}
    ok = df[x].notna() & df[y].notna()
    n = int(ok.sum())
    if n < 3:
        return {"error": f"correlation needs at least 3 rows with both values present; found {n}"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = stats.pearsonr(df[x][ok].astype(float), df[y][ok].astype(float))
    r, p = float(res.statistic), float(res.pvalue)
    if not (math.isfinite(r) and math.isfinite(p)):
        return {"error": "correlation is NaN (a column is probably constant)"}
    return {"n": n, "r": r, "p_value": p, "rows_dropped_null": int(len(df) - n)}


def _group_test(df: pd.DataFrame, test: str, x: str, y: str) -> dict[str, Any]:
    """Welch t-test (2 groups) or one-way ANOVA (2+ groups) of numeric ``y`` by ``x``."""
    if not _is_numeric(df[y]):
        return {"error": f"column '{y}' must be numeric for {test}; it is {df[y].dtype}"}
    ok = df[x].notna() & df[y].notna()
    groups = {
        str(label): sub.astype(float).to_numpy()
        for label, sub in df[y][ok].groupby(df[x][ok], sort=True)
    }
    labels = list(groups)
    k = len(labels)
    if test == "ttest" and k != 2:
        return {"error": f"ttest needs exactly 2 groups in '{x}'; found {k}{_group_labels_hint(labels)}"}
    if test == "anova":
        if k < 2:
            return {"error": f"anova needs at least 2 groups in '{x}'; found {k}{_group_labels_hint(labels)}"}
        if k > MAX_GROUPS:
            return {"error": f"anova supports at most {MAX_GROUPS} groups; '{x}' has {k}"}
    smallest = min(len(a) for a in groups.values())
    if smallest < 2:
        return {"error": f"every group needs at least 2 non-null values of '{y}'; the smallest has {smallest}"}

    arrays = list(groups.values())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if test == "ttest":
            res = stats.ttest_ind(arrays[0], arrays[1], equal_var=False)  # Welch
        else:
            res = stats.f_oneway(*arrays)
    statistic, p = float(res.statistic), float(res.pvalue)
    if not (math.isfinite(statistic) and math.isfinite(p)):
        return {"error": f"{test} produced a NaN/non-finite result (e.g. identical values within groups)"}
    return {
        "statistic": statistic,
        "p_value": p,
        "group_sizes": {label: int(len(a)) for label, a in groups.items()},
        "group_means": {label: float(a.mean()) for label, a in groups.items()},
        "rows_dropped_null": int(len(df) - ok.sum()),
    }


@traced
def compare_groups(dataset_id: str, test: str, x: str, y: str, *, store: Store) -> dict[str, Any]:
    """Run a statistical test on a loaded dataset. All arithmetic happens here.

    * ``"correlation"``: Pearson between numeric ``x`` and ``y`` ->
      ``n, r, p_value`` (rows with a null in either column are dropped).
    * ``"ttest"``: Welch t-test of numeric ``y`` between the exactly 2 groups
      of column ``x``. The statistic is (first group - second group) with groups
      in sorted order, as listed in ``group_sizes``.
    * ``"anova"``: one-way ANOVA of numeric ``y`` across the 2-50 groups of ``x``.

    t-test/ANOVA return ``statistic, p_value, group_sizes, group_means``; every
    group needs at least 2 non-null values. A NaN/non-finite result is an error.
    """
    df = _get_dataset(dataset_id, store)
    if df is None:
        return _unknown_dataset(dataset_id, store)
    if not isinstance(test, str) or test not in TESTS:
        return {"error": f"unknown test; use one of {list(TESTS)}"}
    for name in (x, y):
        if not _has_column(df, name):
            return _unknown_column(df, name)
    if test == "correlation":
        return _correlation(df, x, y)
    return _group_test(df, test, x, y)


# --------------------------------------------------------------------------- #
# run_pandas
# --------------------------------------------------------------------------- #
# Pure in-memory conversions that may be called as ``to_*``; every other ``to_*``
# (to_csv, to_excel, to_pickle, to_sql, to_string(buf), ...) writes somewhere and is blocked.
_SAFE_TO = "datetime|numeric|timedelta|period|timestamp|frame|list|numpy|dict|series"
_BLOCKED_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (label, re.compile(pattern))
    for label, pattern in (
        ("dunder names ('__')", r"__"),
        ("import", r"\bimport\b"),
        ("open", r"\bopen\b"),
        ("exec", r"\bexec\b"),
        ("eval", r"\beval\b"),
        ("compile", r"\bcompile\b"),
        ("os", r"\bos\b"),
        ("sys", r"\bsys\b"),
        ("subprocess", r"\bsubprocess\b"),
        ("getattr/setattr/delattr", r"\b(?:getattr|setattr|delattr)\b"),
        ("read_* functions", r"\bread_\w*"),
        (f"to_* output methods (only to_{{{_SAFE_TO}}} are allowed)", rf"\bto_(?!(?:{_SAFE_TO})\b)\w*"),
        (
            "file I/O helpers (np.load/save/memmap..., ExcelWriter, .io)",
            r"\.(?:load|loads|save|savez|savez_compressed|savetxt|loadtxt|genfromtxt|fromfile"
            r"|tofile|memmap|fromregex)\b|\b(?:ExcelWriter|ExcelFile|HDFStore)\b|\.io\b",
        ),
    )
)

# Runs in a fresh interpreter. Reads a pickle from stdin, prints ONE JSON line to the
# real stdout. User code only ever sees a capped in-memory stdout.
_RUNNER = r'''
import sys
sys.path[:] = [p for p in sys.path if p not in ("", ".")]
import builtins, json, pickle, warnings
import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
real_stdout = sys.stdout
payload = pickle.loads(sys.stdin.buffer.read())
cap, rows = payload["cap"], payload["preview_rows"]

try:
    import resource
    limit = int(payload["mem_mb"]) * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
except Exception:
    pass

pd.set_option("display.max_rows", rows)
pd.set_option("display.min_rows", rows)
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 60)


class Capped:
    """Collects at most cap+1 characters of what the code prints."""
    def __init__(self, cap):
        self.cap, self.parts, self.kept, self.seen = cap, [], 0, 0
    def write(self, text):
        text = str(text)
        self.seen += len(text)
        room = self.cap + 1 - self.kept
        if room > 0:
            self.parts.append(text[:room])
            self.kept += min(room, len(text))
        return len(text)
    def flush(self):
        pass
    def getvalue(self):
        return "".join(self.parts)


def fmt(obj):
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        n = len(obj)
        text = obj.head(rows).to_string()
        if n > rows:
            text += "\n... (%d rows total, first %d shown)" % (n, rows)
        return text
    return str(obj)


safe = {name: getattr(builtins, name) for name in payload["builtins"]}
ns = {"__builtins__": safe, "df": payload["df"], "pd": pd, "np": np}
out = Capped(cap)
try:
    tree = compile(payload["code"], "<run_pandas>", "exec")
    sys.stdout = out
    try:
        exec(tree, ns)
    finally:
        sys.stdout = real_stdout
    text = out.getvalue()
    result = ns.get("result")
    seen = out.seen
    if result is not None:
        rtext = fmt(result)
        text = text[:cap] + ("\n" if text and not text.endswith("\n") else "") + rtext
        seen = max(seen, len(text))
    if not text.strip():
        response = {"error": "no output: use print() or assign a value to `result`"}
    else:
        response = {"output": text[:cap], "truncated": seen > cap or len(text) > cap}
except Exception as exc:
    line = None
    tb = exc.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code.co_filename == "<run_pandas>":
            line = tb.tb_lineno
        tb = tb.tb_next
    if isinstance(exc, SyntaxError):
        line = None  # the message already says where
    msg = "%s: %s" % (type(exc).__name__, exc)
    if len(msg) > 500:
        msg = msg[:500] + "..."
    response = {"error": msg + (" (line %d)" % line if line else "")}
real_stdout.write(json.dumps(response))
real_stdout.flush()
'''


def _blocked_by_guardrail(code: str) -> str | None:
    """Return the label of the first guardrail pattern found in ``code``, else ``None``."""
    for label, pattern in _BLOCKED_PATTERNS:
        if pattern.search(code):
            return label
    return None


def _child_env() -> dict[str, str]:
    """Minimal environment for the worker: no API keys or other secrets are passed on."""
    keep = (
        "PATH", "SYSTEMROOT", "SYSTEMDRIVE", "LD_LIBRARY_PATH", "LANG", "LC_ALL", "TMPDIR",
        "TEMP", "TMP", "HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA",
    )
    env = {key: os.environ[key] for key in keep if key in os.environ}
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = "1"
    return env


@traced
def run_pandas(
    dataset_id: str, code: str, *, store: Store, timeout: float | None = None
) -> dict[str, Any]:
    """Run a short pandas/numpy snippet against a COPY of a loaded dataset.

    The snippet sees only ``df`` (a copy), ``pd`` and ``np``. Results come back
    through ``print()`` and/or a variable named ``result`` (a DataFrame/Series is
    shown as its first 20 rows), capped at 2000 characters:
    ``{"output": str}`` plus ``"truncated": true`` if it was cut.

    SECURITY: this is a GUARDRAIL, NOT ISOLATION. The code is screened with a
    regex deny-list (dunder names, import, open, exec, eval, compile, os, sys,
    subprocess, getattr/setattr, read_*, file-writing to_* methods and a few numpy/
    pandas file helpers) and runs with a whitelist of builtins in a separate Python
    process with a timeout (default 10 s, then the process is killed), a memory cap
    (POSIX) and a minimal environment. That stops accidents and sloppy generated
    code. It does NOT stop a determined attacker: the worker is an ordinary child
    process with the server's file permissions. Do not run untrusted code with it
    outside a container/VM.

    The deny-list is deliberately blunt: a banned word anywhere in the code,
    even inside a string or a column name such as "open", is rejected. Use
    ``df.iloc[:, i]`` or rename columns if you hit that. Each call starts a fresh
    interpreter (about half a second of overhead). ``timeout`` is for tests.
    """
    df = _get_dataset(dataset_id, store)
    if df is None:
        return _unknown_dataset(dataset_id, store)
    if not isinstance(code, str) or not code.strip():
        return {"error": "code must be a non-empty string"}
    if len(code) > MAX_CODE_CHARS:
        return {"error": f"code is too long ({len(code)} chars; max {MAX_CODE_CHARS})"}
    blocked = _blocked_by_guardrail(code)
    if blocked:
        return {
            "error": f"blocked by guardrail: {blocked}. Only in-memory pandas/numpy work on `df` is allowed."
        }

    limit = RUN_PANDAS_TIMEOUT if timeout is None else timeout
    payload = pickle.dumps(
        {
            "df": df,
            "code": code,
            "cap": MAX_OUTPUT_CHARS,
            "preview_rows": RESULT_PREVIEW_ROWS,
            "mem_mb": CHILD_MEMORY_MB,
            "builtins": list(SAFE_BUILTINS),
        }
    )
    try:
        # subprocess.run kills the child when the timeout expires.
        proc = subprocess.run(
            [sys.executable, "-c", _RUNNER],
            input=payload,
            capture_output=True,
            timeout=limit,
            env=_child_env(),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"timeout: code ran longer than {limit:g} s and was terminated"}

    lines = proc.stdout.decode("utf-8", "replace").strip().splitlines()
    try:
        data = json.loads(lines[-1]) if lines else None
    except ValueError:
        data = None
    if not isinstance(data, dict):
        return {"error": f"worker process failed (exit code {proc.returncode}); possibly out of memory"}
    if "error" in data:
        return {"error": str(data["error"])[:600]}
    output = str(data.get("output", ""))[:MAX_OUTPUT_CHARS]
    result: dict[str, Any] = {"output": output}
    if data.get("truncated"):
        result["truncated"] = True
    return result


# --------------------------------------------------------------------------- #
# make_chart
# --------------------------------------------------------------------------- #
_CHART_BLUE = "#2a78d6"  # single series: first slot of the reference palette
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_GRID = "#e6e5e1"
_AXIS = "#c9c8c3"


def _slug(text: str, limit: int = 40) -> str:
    """Filesystem-safe fragment: letters, digits and '-' only."""
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")[:limit].strip("-") or "chart"


def _compact(value: float, _pos: Any = None) -> str:
    """Axis tick label: 1500 -> 1.5K, 2000000 -> 2M."""
    magnitude = abs(value)
    for limit, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if magnitude >= limit:
            return f"{value / limit:g}{suffix}"
    return f"{value:g}"


def _chart_group_data(
    df_id: str, kind: str, x: str, y: str | None, agg: str, freq: str | None, store: Store
) -> dict[str, Any]:
    """Aggregate for bar/line with ``aggregate``; returns xs, ys, counts or an error dict."""
    if y is None and agg != "count":
        return {"error": f"kind '{kind}' needs y (the column to aggregate), unless agg is 'count'"}
    column = x if y is None else y
    if freq is not None:
        res = aggregate(df_id, column, agg, date_column=x, freq=freq, store=store)
        key = f"{x}_{freq}"
    else:
        res = aggregate(df_id, column, agg, group_by=[x], store=store)
        key = x
    if "error" in res:
        return res
    if res.get("truncated"):
        what = "bars" if kind == "bar" else "points"
        return {
            "error": f"too many {what}: {res['total_groups']} groups (max {MAX_BARS}). "
            "Group by a coarser column or use freq='quarter'/'year'."
        }
    groups = res["groups"]
    if not groups:
        return {"error": "no data to plot (no rows left after dropping nulls)"}
    xs = [g[key] for g in groups]
    ys = [g["value"] for g in groups]
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in ys):
        return {"error": f"agg '{agg}' of '{column}' is not numeric, so it cannot be charted"}
    return {"xs": xs, "ys": ys, "rows_used": res["rows_used"], "rows_dropped_null": res["rows_dropped_null"]}


@traced
def make_chart(
    dataset_id: str,
    kind: str,
    x: str,
    y: str | None = None,
    agg: str = "sum",
    freq: str | None = None,
    title: str | None = None,
    *,
    store: Store,
    config: Config | None = None,
) -> dict[str, Any]:
    """Draw a chart of a loaded dataset and save it as a PNG in OUTPUT_DIR/charts/.

    * ``bar`` / ``line``: group by ``x`` and aggregate ``y`` with ``agg`` using the
      same code as ``aggregate`` (``y`` may be omitted when ``agg="count"`` to count
      rows per ``x``). ``freq`` (month|quarter|year) groups a date column ``x`` by
      period. Lines are sorted by x. At most 50 bars/points; more is an error.
    * ``hist``: distribution of numeric ``x`` (30 bins). ``y``/``agg``/``freq`` ignored.
    * ``scatter``: numeric ``x`` against numeric ``y`` (pairwise null drop).

    Returns ``{"path", "kind", "rows_used", "rows_dropped_null"}``; ``path`` is
    absolute. The file name is generated (never taken from user text) and unique,
    so earlier charts are never overwritten. Uses the Agg backend and always closes
    the figure.
    """
    config = config or Config()
    df = _get_dataset(dataset_id, store)
    if df is None:
        return _unknown_dataset(dataset_id, store)
    if not isinstance(kind, str) or kind not in CHART_KINDS:
        return {"error": f"unknown kind; use one of {list(CHART_KINDS)}"}
    if not _has_column(df, x):
        return _unknown_column(df, x)
    if y is not None and not _has_column(df, y):
        return _unknown_column(df, y)
    if title is not None and not isinstance(title, str):
        return {"error": "title must be a string"}

    xs: list[Any]
    ys: list[Any]
    if kind in ("bar", "line"):
        data = _chart_group_data(dataset_id, kind, x, y, agg, freq, store)
        if "error" in data:
            return data
        xs, ys = data["xs"], data["ys"]
        used, dropped = data["rows_used"], data["rows_dropped_null"]
        xlabel = f"{x} ({freq})" if freq else x
        ylabel = "count" if y is None else f"{agg} of {y}"
        default_title = f"{ylabel} by {xlabel}"
    elif kind == "hist":
        if not _is_numeric(df[x]):
            return {"error": f"column '{x}' must be numeric for hist; it is {df[x].dtype}"}
        values = df[x].dropna().astype(float)
        if values.empty:
            return {"error": "no data to plot (column has no non-null values)"}
        xs, ys = values.tolist(), []
        used, dropped = len(values), int(len(df) - len(values))
        xlabel, ylabel, default_title = x, "rows", f"Distribution of {x}"
    else:  # scatter
        if y is None:
            return {"error": "kind 'scatter' needs y"}
        if x == y:
            return {"error": "scatter needs two different columns"}
        for name in (x, y):
            if not _is_numeric(df[name]):
                return {"error": f"column '{name}' must be numeric for scatter; it is {df[name].dtype}"}
        both = df[[x, y]].dropna()
        if both.empty:
            return {"error": "no data to plot (no rows with both values present)"}
        xs, ys = both[x].astype(float).tolist(), both[y].astype(float).tolist()
        used, dropped = len(both), int(len(df) - len(both))
        xlabel, ylabel, default_title = x, y, f"{y} vs {x}"

    charts_dir = config.output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    name = f"{kind}_{_slug(x if y is None else f'{x}-{y}')}_{uuid.uuid4().hex[:8]}.png"
    path = safe_path(name, charts_dir)
    if isinstance(path, dict):
        return path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100, facecolor=_SURFACE)
    try:
        ax.set_facecolor(_SURFACE)
        if kind == "bar":
            labels = [str(v) for v in xs]
            slot_px = 640 / len(labels)
            ax.bar(labels, ys, width=min(0.6, 24 / slot_px), color=_CHART_BLUE)  # bars <= 24px
        elif kind == "line":
            numeric_x = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in xs)
            ax.plot(
                xs if numeric_x else [str(v) for v in xs], ys, color=_CHART_BLUE, linewidth=1.5,
                solid_capstyle="round", solid_joinstyle="round",
                marker="o" if len(xs) <= 24 else None, markersize=6,
                markeredgecolor=_SURFACE, markeredgewidth=1.5,
            )
        elif kind == "hist":
            ax.hist(xs, bins=30, color=_CHART_BLUE, edgecolor=_SURFACE, linewidth=1.0)
        else:
            ax.scatter(xs, ys, s=18, color=_CHART_BLUE, alpha=0.6, linewidths=0)

        ax.set_title(title or default_title, loc="left", color=_INK, fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel(xlabel, color=_INK_2, fontsize=9)
        ax.set_ylabel(ylabel, color=_INK_2, fontsize=9)
        ax.tick_params(colors=_INK_2, labelsize=8, length=0)
        ax.yaxis.set_major_formatter(FuncFormatter(_compact))
        if kind in ("hist", "scatter") or (kind == "line" and numeric_x):
            ax.xaxis.set_major_formatter(FuncFormatter(_compact))
        if kind in ("bar", "line") and not (kind == "line" and numeric_x) and len(xs) > 8:
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(_AXIS)
        ax.grid(axis="y", color=_GRID, linewidth=0.75)
        ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(path, format="png", facecolor=fig.get_facecolor())
    finally:
        plt.close(fig)

    if not path.is_file() or path.stat().st_size == 0:
        return {"error": "chart file was not written"}
    return {"path": str(path), "kind": kind, "rows_used": used, "rows_dropped_null": dropped}
