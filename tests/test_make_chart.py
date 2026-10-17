"""Tests for make_chart().

A PNG cannot be compared number by number, so the tests (1) check the file really is a
non-empty PNG in the right place and (2) spy on the matplotlib calls to check that the
plotted data equals what pandas computes inside the test.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.axes  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from core import MAX_BARS, Config, make_chart  # noqa: E402

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
SAFE_NAME = re.compile(r"^[A-Za-z0-9_-]+\.png$")


def chart(store: dict, cfg: Config, kind: str, x: str, y: str | None = None, **kwargs: Any) -> dict[str, Any]:
    return make_chart("ds1", kind, x, y, store=store, config=cfg, **kwargs)


def charts_dir(cfg: Config) -> Path:
    return cfg.output_dir / "charts"


def assert_valid_png(result: dict, cfg: Config) -> Path:
    assert "error" not in result, result
    path = Path(result["path"])
    assert path.is_absolute() and path.is_file()
    assert path.parent == charts_dir(cfg).resolve()
    assert SAFE_NAME.match(path.name)
    data = path.read_bytes()
    assert len(data) > 1000 and data.startswith(PNG_SIGNATURE)
    return path


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch):
    """Record the positional data arguments of Axes.bar/plot/hist/scatter calls."""
    calls: dict[str, list[tuple]] = {}

    def install(name: str) -> list[tuple]:
        original = getattr(matplotlib.axes.Axes, name)
        recorded: list[tuple] = []

        def wrapper(self, *args, **kwargs):
            recorded.append(args)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, name, wrapper)
        calls[name] = recorded
        return recorded

    return install


# -------------------------------------------------------------- PNG output --
@pytest.mark.parametrize("kind,x,y,extra", [
    ("bar", "region", "amount", {}),
    ("bar", "region", None, {"agg": "count"}),
    ("line", "order_date", "amount", {"freq": "month"}),
    ("line", "qty", "amount", {"agg": "mean"}),
    ("hist", "amount", None, {}),
    ("scatter", "qty", "amount", {}),
])
def test_each_kind_writes_a_nonempty_png_and_closes_the_figure(analysis_store, cfg, kind, x, y, extra):
    result = chart(analysis_store, cfg, kind, x, y, **extra)
    assert_valid_png(result, cfg)
    assert result["kind"] == kind
    assert plt.get_fignums() == []  # figure closed


def test_title_is_accepted_and_results_are_json_safe(analysis_store, cfg):
    import json

    result = chart(analysis_store, cfg, "bar", "region", "amount", title="Revenue by region")
    assert_valid_png(result, cfg)
    json.dumps(result, allow_nan=False)


def test_each_call_gets_its_own_file_and_nothing_is_overwritten(analysis_store, cfg):
    first = chart(analysis_store, cfg, "bar", "region", "amount")
    second = chart(analysis_store, cfg, "bar", "region", "amount")
    assert first["path"] != second["path"]
    assert Path(first["path"]).is_file() and Path(second["path"]).is_file()


def test_only_the_output_folder_is_written_and_stdout_stays_empty(analysis_store, cfg, tmp_path, capfd):
    before_data = sorted(p.name for p in cfg.data_dir.iterdir())
    chart(analysis_store, cfg, "bar", "region", "amount")
    chart(analysis_store, cfg, "hist", "amount")
    written = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert written and all(charts_dir(cfg).resolve() in p.resolve().parents for p in written)
    assert sorted(p.name for p in cfg.data_dir.iterdir()) == before_data
    assert capfd.readouterr().out == ""


def test_filenames_are_safe_even_for_hostile_column_names_and_titles(cfg, tmp_path):
    df = pd.DataFrame({"../../evil name!": ["a", "b", "a"], "v/al*ue": [1.0, 2.0, 3.0]})
    result = make_chart("ds1", "bar", "../../evil name!", "v/al*ue", title="../../etc/passwd",
                        store={"ds1": df}, config=cfg)
    path = assert_valid_png(result, cfg)
    assert path.parent == charts_dir(cfg).resolve()
    assert [p for p in tmp_path.rglob("*") if p.is_file() and charts_dir(cfg).resolve() not in p.parents] == []


def test_the_store_is_not_modified(analysis_df, analysis_store, cfg):
    before = analysis_df.copy(deep=True)
    for kind, x, y in [("bar", "region", "amount"), ("hist", "amount", None), ("scatter", "qty", "amount")]:
        chart(analysis_store, cfg, kind, x, y)
    pd.testing.assert_frame_equal(analysis_store["ds1"], before)


# ------------------------------------------------ plotted data == pandas ----
def test_bar_plots_the_pandas_group_sums(analysis_df, analysis_store, cfg, spy):
    recorded = spy("bar")
    result = chart(analysis_store, cfg, "bar", "region", "amount", agg="sum")
    assert_valid_png(result, cfg)
    sub = analysis_df.dropna(subset=["region", "amount"])
    expected = sub.groupby("region")["amount"].sum()
    labels, heights = recorded[0]
    assert list(labels) == list(expected.index)
    assert list(heights) == pytest.approx(list(expected.values))
    assert result["rows_used"] == len(sub)
    assert result["rows_dropped_null"] == len(analysis_df) - len(sub)


@pytest.mark.parametrize("agg", ["mean", "median", "min", "max", "count", "nunique"])
def test_bar_supports_every_aggregation(analysis_df, analysis_store, cfg, spy, agg):
    recorded = spy("bar")
    assert_valid_png(chart(analysis_store, cfg, "bar", "product", "amount", agg=agg), cfg)
    sub = analysis_df.dropna(subset=["product", "amount"])
    expected = sub.groupby("product")["amount"].agg(agg)
    assert list(recorded[0][1]) == pytest.approx(list(expected.values))


def test_bar_without_y_counts_rows_per_group(analysis_df, analysis_store, cfg, spy):
    recorded = spy("bar")
    result = chart(analysis_store, cfg, "bar", "region", agg="count")
    assert_valid_png(result, cfg)
    expected = analysis_df["region"].value_counts().sort_index()
    assert list(recorded[0][0]) == list(expected.index) and list(recorded[0][1]) == list(expected.values)


def test_line_is_sorted_by_x_and_matches_pandas(cfg, spy):
    recorded = spy("plot")
    df = pd.DataFrame({"x": [3, 1, 2, 1, 3, 2], "y": [30.0, 10.0, 20.0, 11.0, 31.0, 21.0]})
    assert_valid_png(make_chart("ds1", "line", "x", "y", store={"ds1": df}, config=cfg), cfg)
    xs, ys = recorded[0]
    assert list(xs) == [1, 2, 3]  # input rows were unsorted
    assert list(ys) == pytest.approx(list(df.groupby("x")["y"].sum().values))


@pytest.mark.parametrize("freq,code", [("month", "M"), ("quarter", "Q"), ("year", "Y")])
def test_line_with_freq_groups_dates_chronologically(analysis_df, analysis_store, cfg, spy, freq, code):
    recorded = spy("plot")
    assert_valid_png(chart(analysis_store, cfg, "line", "order_date", "amount", freq=freq), cfg)
    sub = analysis_df.dropna(subset=["order_date", "amount"])
    expected = sub.groupby(sub["order_date"].dt.to_period(code))["amount"].sum()

    def label(p: pd.Period) -> str:
        return {"month": f"{p.year}-{p.month:02d}", "quarter": f"{p.year}-Q{p.quarter}", "year": str(p.year)}[freq]

    xs, ys = recorded[0]
    assert list(xs) == [label(p) for p in expected.index]
    assert list(ys) == pytest.approx(list(expected.values))


def test_hist_plots_the_non_null_values_and_ignores_y(analysis_df, analysis_store, cfg, spy):
    recorded = spy("hist")
    result = chart(analysis_store, cfg, "hist", "amount", "qty", agg="bogus", freq="bogus")
    assert_valid_png(result, cfg)
    expected = analysis_df["amount"].dropna().tolist()
    assert list(recorded[0][0]) == pytest.approx(expected)
    assert (result["rows_used"], result["rows_dropped_null"]) == (57, 3)


def test_scatter_drops_nulls_pairwise(analysis_df, analysis_store, cfg, spy):
    recorded = spy("scatter")
    result = chart(analysis_store, cfg, "scatter", "qty", "amount")
    assert_valid_png(result, cfg)
    both = analysis_df[["qty", "amount"]].dropna()
    assert list(recorded[0][0]) == pytest.approx(both["qty"].astype(float).tolist())
    assert list(recorded[0][1]) == pytest.approx(both["amount"].tolist())
    assert (result["rows_used"], result["rows_dropped_null"]) == (len(both), len(analysis_df) - len(both))


# --------------------------------------------------------------- 50 bars ----
def test_exactly_50_bars_is_allowed_and_51_is_an_error(cfg):
    ok = pd.DataFrame({"g": [f"g{i:02d}" for i in range(MAX_BARS)], "v": range(MAX_BARS)})
    assert_valid_png(make_chart("ds1", "bar", "g", "v", store={"ds1": ok}, config=cfg), cfg)

    too_many = pd.DataFrame({"g": [f"g{i:02d}" for i in range(MAX_BARS + 1)], "v": range(MAX_BARS + 1)})
    before = sorted(charts_dir(cfg).glob("*.png"))
    for kind in ("bar", "line"):
        error = make_chart("ds1", kind, "g", "v", store={"ds1": too_many}, config=cfg)["error"]
        assert "too many" in error and "51" in error and "max 50" in error
    assert sorted(charts_dir(cfg).glob("*.png")) == before  # nothing written for the failed calls
    assert plt.get_fignums() == []


# ------------------------------------------------------------ bad input ----
def test_unknown_dataset_lists_loaded_ids(cfg):
    store = {"ds1": pd.DataFrame({"a": [1]}), "ds2": pd.DataFrame({"a": [1]})}
    result = make_chart("ds5", "bar", "a", "a", store=store, config=cfg)
    assert "ds5" in result["error"] and result["loaded"] == ["ds1", "ds2"]


@pytest.mark.parametrize("kind,x,y,extra,fragment", [
    ("pie", "region", "amount", {}, "unknown kind"),
    ("", "region", "amount", {}, "unknown kind"),
    ("bar", "nope", "amount", {}, "unknown column 'nope'"),
    ("bar", "region", "nope", {}, "unknown column 'nope'"),
    ("scatter", "nope", "amount", {}, "unknown column 'nope'"),
    ("hist", "nope", None, {}, "unknown column 'nope'"),
    ("bar", "region", None, {}, "needs y"),                       # sum without y
    ("line", "qty", None, {"agg": "mean"}, "needs y"),
    ("bar", "region", "amount", {"agg": "median(amount)"}, "unknown agg"),
    ("bar", "region", "region", {"agg": "sum"}, "numeric"),
    ("bar", "region", "region", {"agg": "min"}, "not numeric"),   # strings cannot be plotted
    ("bar", "qty", "amount", {"freq": "month"}, "not a date column"),
    ("line", "order_date", "amount", {"freq": "week"}, "unknown freq"),
    ("hist", "region", None, {}, "must be numeric"),
    ("scatter", "region", "amount", {}, "must be numeric"),
    ("scatter", "qty", "region", {}, "must be numeric"),
    ("scatter", "qty", None, {}, "needs y"),
    ("bar", "region", "amount", {"title": 5}, "title must be a string"),
])
def test_clear_errors_and_no_file_written(analysis_df, analysis_store, cfg, kind, x, y, extra, fragment):
    result = chart(analysis_store, cfg, kind, x, y, **extra)
    assert fragment in result["error"]
    if charts_dir(cfg).exists():
        assert list(charts_dir(cfg).glob("*")) == []  # a failed call writes nothing
    if "unknown column" in result["error"]:
        assert all(col in result["error"] for col in analysis_df.columns)


@pytest.mark.parametrize("column", ["amount", "qty"])
def test_scatter_with_the_same_column_twice_is_a_clear_error(analysis_store, cfg, column):
    # regression: df[[x, x]] has duplicate columns, which used to surface as an AttributeError
    result = chart(analysis_store, cfg, "scatter", column, column)
    assert result == {"error": "scatter needs two different columns"}
    assert not charts_dir(cfg).exists() or list(charts_dir(cfg).glob("*")) == []
    assert plt.get_fignums() == []


def test_scatter_same_text_column_gets_the_same_error_not_a_type_error(analysis_store, cfg):
    assert chart(analysis_store, cfg, "scatter", "region", "region") == {
        "error": "scatter needs two different columns"}


def test_nothing_left_to_plot_is_an_error(cfg):
    df = pd.DataFrame({"g": ["a", "b"], "v": [float("nan")] * 2, "w": [float("nan")] * 2})
    for kind, x, y in [("bar", "g", "v"), ("hist", "v", None), ("scatter", "v", "w")]:
        assert "no data to plot" in make_chart("ds1", kind, x, y, store={"ds1": df}, config=cfg)["error"]


@pytest.mark.parametrize("kwargs", [
    {"dataset_id": None}, {"dataset_id": 1}, {"dataset_id": ["ds1"]},
    {"kind": None}, {"kind": 5}, {"kind": ["bar"]}, {"kind": {"a": 1}},
    {"x": None}, {"x": 5}, {"x": ["region"]}, {"x": {"a": 1}},
    {"y": 5}, {"y": ["amount"]}, {"y": {"a": 1}},
    {"agg": None}, {"agg": 5}, {"agg": ["sum"]},
    {"freq": 3}, {"freq": ["month"]},
    {"title": 5}, {"title": ["t"]},
    {"store": None}, {"store": []}, {"store": "ds1"}, {"store": {"ds1": None}},
])
def test_no_bad_input_raises(analysis_store, cfg, kwargs):
    call = {"dataset_id": "ds1", "kind": "bar", "x": "region", "y": "amount", "store": analysis_store,
            "config": cfg, **kwargs}
    result = make_chart(call.pop("dataset_id"), call.pop("kind"), call.pop("x"), call.pop("y"), **call)
    assert isinstance(result, dict) and "error" in result


def test_missing_arguments_do_not_raise(cfg):
    assert "error" in make_chart()
    assert "error" in make_chart("ds1", "bar", "region")  # store missing


def test_make_chart_is_traced_and_logs_no_values(analysis_store, cfg, capfd):
    assert hasattr(make_chart, "__wrapped__")
    chart(analysis_store, cfg, "bar", "region", "amount", title="Zanzibar-Q9")
    err = capfd.readouterr().err
    assert "tool=make_chart" in err and "result_bytes=" in err
    assert "Zanzibar-Q9" not in err and "North" not in err
