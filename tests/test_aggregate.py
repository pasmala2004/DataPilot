"""Tests for aggregate(): every expected value is computed with pandas inside the test."""

from __future__ import annotations

import json
import operator
from typing import Any

import numpy as np
import pandas as pd
import pytest

from core import AGGS, aggregate

QUARTER_CODE = {"month": "M", "quarter": "Q", "year": "Y"}


def run(store: dict, column: str = "amount", agg: str = "sum", **kwargs: Any) -> dict[str, Any]:
    return aggregate("ds1", column, agg, store=store, **kwargs)


def as_dict(result: dict, keys: list[str]) -> dict[tuple, Any]:
    """{(group values...): value} from a grouped result."""
    return {tuple(g[k] for k in keys): g["value"] for g in result["groups"]}


def label(period: pd.Period, freq: str) -> str:
    return {"month": f"{period.year}-{period.month:02d}", "quarter": f"{period.year}-Q{period.quarter}",
            "year": str(period.year)}[freq]


# ------------------------------------------------------------ every agg ---
@pytest.mark.parametrize("agg", AGGS)
def test_ungrouped_every_agg_matches_pandas(analysis_df, analysis_store, agg):
    result = run(analysis_store, "amount", agg)
    expected = getattr(analysis_df["amount"].dropna(), agg)()
    assert result["value"] == pytest.approx(expected)
    assert result["rows_used"] == int(analysis_df["amount"].notna().sum()) == 57
    assert result["rows_dropped_null"] == 3


@pytest.mark.parametrize("agg", ["count", "nunique", "min", "max"])
def test_non_numeric_column_aggs(analysis_df, analysis_store, agg):
    expected = getattr(analysis_df["region"].dropna(), agg)()
    assert run(analysis_store, "region", agg)["value"] == expected


@pytest.mark.parametrize("agg", ["sum", "mean", "median"])
def test_numeric_only_aggs_reject_text_columns(analysis_store, agg):
    assert "numeric" in run(analysis_store, "region", agg)["error"]


def test_min_max_on_dates_are_iso_strings(analysis_df, analysis_store):
    assert run(analysis_store, "order_date", "min")["value"] == analysis_df["order_date"].min().isoformat()
    assert run(analysis_store, "order_date", "max")["value"] == analysis_df["order_date"].max().isoformat()


# ------------------------------------------------------------- group_by ---
@pytest.mark.parametrize("agg", AGGS)
@pytest.mark.parametrize("keys", [["region"], ["region", "product"]])
def test_group_by_every_agg_matches_pandas(analysis_df, analysis_store, agg, keys):
    result = run(analysis_store, "amount", agg, group_by=keys)
    sub = analysis_df.dropna(subset=["amount", *keys])
    expected = sub.groupby(keys)["amount"].agg(agg)
    expected = {(k if isinstance(k, tuple) else (k,)): v for k, v in expected.items()}
    got = as_dict(result, keys)
    assert got.keys() == expected.keys()
    for key, value in expected.items():
        assert got[key] == pytest.approx(value)
    assert list(got) == sorted(got)  # groups come back in key order
    assert result["rows_used"] == len(sub)
    assert result["rows_dropped_null"] == len(analysis_df) - len(sub)
    assert "value" not in result and "truncated" not in result


def test_group_by_accepts_a_single_column_name(analysis_store):
    assert run(analysis_store, group_by="region") == run(analysis_store, group_by=["region"])


def test_group_by_the_aggregated_column_itself(analysis_df, analysis_store):
    result = run(analysis_store, "region", "count", group_by=["region"])
    expected = analysis_df["region"].value_counts()
    assert as_dict(result, ["region"]) == {(k,): v for k, v in expected.items()}


# --------------------------------------------------------------- filters ---
def expected_mask(df: pd.DataFrame, column: str, op: str, value: Any) -> pd.Series:
    series = df[column]
    if column == "order_date":
        value = [pd.Timestamp(v) for v in value] if op == "in" else pd.Timestamp(value)
    if op == "in":
        part = series.isin(value)
    else:
        part = {"==": operator.eq, "!=": operator.ne, ">": operator.gt, ">=": operator.ge,
                "<": operator.lt, "<=": operator.le}[op](series, value)
    return part & series.notna()  # null cells never match


FILTER_CASES = [
    ("qty", "==", 5), ("qty", "!=", 5), ("qty", ">", 5), ("qty", ">=", 5), ("qty", "<", 5),
    ("qty", "<=", 5), ("qty", "in", [2, 4, 6]),
    ("region", "==", "North"), ("region", "!=", "North"), ("region", "in", ["North", "East"]),
    ("order_date", ">=", "2024-03-01"), ("order_date", "<", "2024-03-01"),
    ("order_date", "in", ["2024-03-31 15:30:00"]),
]


@pytest.mark.parametrize("column,op,value", FILTER_CASES)
def test_each_filter_op_matches_pandas(analysis_df, analysis_store, column, op, value):
    result = run(analysis_store, "amount", "sum",
                 filters=[{"column": column, "op": op, "value": value}])
    selected = analysis_df[expected_mask(analysis_df, column, op, value)]
    used = selected["amount"].dropna()
    assert result["value"] == pytest.approx(used.sum())
    assert result["rows_used"] == len(used)
    assert result["rows_dropped_null"] == len(selected) - len(used)  # filtered-out rows don't count


def test_not_equal_does_not_match_null_cells(analysis_df, analysis_store):
    result = run(analysis_store, "qty", "count",
                 filters=[{"column": "region", "op": "!=", "value": "North"}])
    with_nulls = int((analysis_df["region"] != "North").sum())
    without_nulls = int(((analysis_df["region"] != "North") & analysis_df["region"].notna()).sum())
    assert result["rows_used"] == without_nulls < with_nulls


def test_multiple_filters_are_anded(analysis_df, analysis_store):
    result = run(analysis_store, "amount", "mean", filters=[
        {"column": "region", "op": "in", "value": ["North", "South"]},
        {"column": "qty", "op": ">=", "value": 4},
        {"column": "product", "op": "==", "value": "A"},
    ])
    d = analysis_df
    sub = d[d.region.isin(["North", "South"]) & (d.qty >= 4) & (d["product"] == "A")]
    assert result["value"] == pytest.approx(sub["amount"].mean())


def test_empty_filter_list_changes_nothing(analysis_store):
    assert run(analysis_store, filters=[]) == run(analysis_store)


# ---------------------------------------------------------------- period ---
PERIOD_CASES = [
    ({"start": "2024-01-01", "end": "2024-03-31"}, "2024-01-01", "2024-03-31 23:59:59.999999"),
    ({"start": "2024-01-01", "end": "2024-03-30"}, "2024-01-01", "2024-03-30 23:59:59.999999"),
    ({"start": "2024-03-31", "end": "2024-03-31"}, "2024-03-31", "2024-03-31 23:59:59.999999"),
    ({"start": "2024-03-31 16:00:00"}, "2024-03-31 16:00:00", None),
    ({"end": "2024-02-29"}, None, "2024-02-29 23:59:59.999999"),
    ({"end": "2024-03-31 15:30:00"}, None, "2024-03-31 15:30:00"),
]


@pytest.mark.parametrize("period,lo,hi", PERIOD_CASES)
def test_period_is_inclusive_and_matches_pandas(analysis_df, analysis_store, period, lo, hi):
    d = analysis_df["order_date"]
    mask = d.notna()
    if lo:
        mask &= d >= pd.Timestamp(lo)
    if hi:
        mask &= d <= pd.Timestamp(hi)
    selected = analysis_df[mask]
    result = run(analysis_store, "amount", "sum", date_column="order_date", period=period)
    assert result["value"] == pytest.approx(selected["amount"].sum())
    assert result["rows_used"] == int(selected["amount"].notna().sum())


def test_period_end_date_includes_the_whole_day(analysis_df, analysis_store):
    # row 0 is 2024-03-31 15:30: in with end="2024-03-31", out with end="2024-03-30"
    inside = run(analysis_store, "qty", "count", date_column="order_date",
                 period={"start": "2024-03-31", "end": "2024-03-31"})
    outside = run(analysis_store, "qty", "count", date_column="order_date",
                  period={"start": "2024-03-30", "end": "2024-03-30"})
    assert inside["rows_used"] >= 1
    assert inside["rows_used"] == int((analysis_df["order_date"].dt.normalize() == "2024-03-31").sum())
    assert outside["rows_used"] == int((analysis_df["order_date"].dt.normalize() == "2024-03-30").sum())


# ------------------------------------------------------------------ freq ---
@pytest.mark.parametrize("freq", ["month", "quarter", "year"])
def test_freq_matches_pandas_periods(analysis_df, analysis_store, freq):
    result = run(analysis_store, "amount", "sum", date_column="order_date", freq=freq)
    sub = analysis_df.dropna(subset=["amount", "order_date"])
    expected = sub.groupby(sub["order_date"].dt.to_period(QUARTER_CODE[freq]))["amount"].sum()
    got = as_dict(result, [f"order_date_{freq}"])
    assert got.keys() == {(label(p, freq),) for p in expected.index}
    for p, value in expected.items():
        assert got[(label(p, freq),)] == pytest.approx(value)
    assert list(got) == sorted(got)  # chronological
    assert result["rows_used"] == len(sub)
    assert result["rows_dropped_null"] == len(analysis_df) - len(sub)


def test_freq_quarter_labels_and_group_key_name(analysis_store):
    groups = run(analysis_store, "amount", "sum", date_column="order_date", freq="quarter")["groups"]
    assert [g["order_date_quarter"] for g in groups] == ["2023-Q4", "2024-Q1", "2024-Q2", "2024-Q3"]


def test_freq_combined_with_group_by_and_filters(analysis_df, analysis_store):
    result = run(analysis_store, "amount", "mean", group_by=["region"], date_column="order_date",
                 freq="quarter", filters=[{"column": "qty", "op": ">", "value": 2}],
                 period={"start": "2024-01-01", "end": "2024-12-31"})
    d = analysis_df
    sub = d[(d.qty > 2) & (d.order_date >= "2024-01-01") & (d.order_date < "2025-01-01")]
    sub = sub.dropna(subset=["amount", "region", "order_date"])
    quarters = sub["order_date"].dt.to_period("Q")
    expected = sub.groupby([quarters, sub["region"]])["amount"].mean()
    got = as_dict(result, ["order_date_quarter", "region"])
    assert got.keys() == {(label(p, "quarter"), r) for p, r in expected.index}
    for (p, r), value in expected.items():
        assert got[(label(p, "quarter"), r)] == pytest.approx(value)


# ----------------------------------------------------------------- top_n ---
def test_top_n_keeps_the_largest_groups_sorted_desc(analysis_df, analysis_store):
    result = run(analysis_store, "amount", "sum", group_by=["region"], top_n=2)
    expected = analysis_df.dropna(subset=["amount", "region"]).groupby("region")["amount"].sum()
    expected = expected.sort_values(ascending=False).head(2)
    assert [g["region"] for g in result["groups"]] == list(expected.index)
    assert [g["value"] for g in result["groups"]] == pytest.approx(list(expected.values))
    assert "truncated" not in result


def test_top_n_larger_than_group_count_returns_all_sorted_desc(analysis_df, analysis_store):
    result = run(analysis_store, "amount", "sum", group_by=["region"], top_n=99)
    values = [g["value"] for g in result["groups"]]
    assert len(values) == analysis_df["region"].nunique() and values == sorted(values, reverse=True)


def test_top_n_with_freq_and_ties_is_deterministic():
    df = pd.DataFrame({"g": ["b", "a", "c", "a", "b", "c"], "v": [1, 1, 5, 1, 1, 5]})
    result = aggregate("ds1", "v", "sum", group_by=["g"], top_n=2, store={"ds1": df})
    assert [(g["g"], g["value"]) for g in result["groups"]] == [("c", 10), ("a", 2)]  # tie a/b -> key order


# -------------------------------------------------------------- truncation ---
@pytest.fixture
def many_groups() -> dict:
    return {"ds1": pd.DataFrame({"g": [f"g{i:03d}" for i in range(60)], "v": list(range(60))})}


def test_more_than_50_groups_is_truncated(many_groups):
    result = aggregate("ds1", "v", "sum", group_by=["g"], store=many_groups)
    assert len(result["groups"]) == 50 and result["truncated"] is True and result["total_groups"] == 60
    assert [g["g"] for g in result["groups"]] == [f"g{i:03d}" for i in range(50)]


def test_top_n_within_the_cap_is_not_truncation_and_top_n_above_it_is(many_groups):
    small = aggregate("ds1", "v", "sum", group_by=["g"], top_n=5, store=many_groups)
    assert len(small["groups"]) == 5 and "truncated" not in small
    assert small["groups"][0] == {"g": "g059", "value": 59}
    big = aggregate("ds1", "v", "sum", group_by=["g"], top_n=60, store=many_groups)
    assert len(big["groups"]) == 50 and big["truncated"] is True
    assert big["groups"][0]["g"] == "g059"


def test_exactly_50_groups_is_not_truncated():
    df = pd.DataFrame({"g": [f"g{i:02d}" for i in range(50)], "v": 1})
    assert "truncated" not in aggregate("ds1", "v", "sum", group_by=["g"], store={"ds1": df})


# ----------------------------------------------------------- null counting ---
@pytest.fixture
def nulls_store() -> dict:
    return {"ds1": pd.DataFrame({
        "v": [1.0, 2.0, None, 4.0, 5.0, 6.0],
        "g": ["a", "a", "a", None, "b", "b"],
    })}


def test_null_counting_ungrouped(nulls_store):
    result = aggregate("ds1", "v", "sum", store=nulls_store)
    assert result == {"value": 18.0, "rows_used": 5, "rows_dropped_null": 1}


def test_null_counting_grouped_drops_null_values_and_null_keys(nulls_store):
    result = aggregate("ds1", "v", "sum", group_by=["g"], store=nulls_store)
    assert result["groups"] == [{"g": "a", "value": 3.0}, {"g": "b", "value": 11.0}]
    assert (result["rows_used"], result["rows_dropped_null"]) == (4, 2)
    counts = aggregate("ds1", "v", "count", group_by=["g"], store=nulls_store)
    assert counts["groups"] == [{"g": "a", "value": 2}, {"g": "b", "value": 2}]


def test_filtered_out_rows_are_not_counted_as_null_drops(nulls_store):
    result = aggregate("ds1", "v", "sum", store=nulls_store,
                       filters=[{"column": "g", "op": "==", "value": "b"}])
    assert result == {"value": 11.0, "rows_used": 2, "rows_dropped_null": 0}


def test_used_plus_dropped_equals_rows_after_filters(analysis_df, analysis_store):
    result = run(analysis_store, "amount", "sum", group_by=["region"],
                 filters=[{"column": "qty", "op": ">", "value": 3}])
    assert result["rows_used"] + result["rows_dropped_null"] == int((analysis_df["qty"] > 3).sum())


# ----------------------------------------------------------- empty results ---
def test_no_rows_left_gives_zero_counts_and_none_otherwise(analysis_store):
    none_match = [{"column": "qty", "op": ">", "value": 1000}]
    assert run(analysis_store, "amount", "count", filters=none_match) == {
        "value": 0, "rows_used": 0, "rows_dropped_null": 0}
    assert run(analysis_store, "amount", "nunique", filters=none_match)["value"] == 0
    for agg in ("sum", "mean", "median", "min", "max"):
        assert run(analysis_store, "amount", agg, filters=none_match)["value"] is None
    assert run(analysis_store, group_by=["region"], filters=none_match) == {
        "groups": [], "rows_used": 0, "rows_dropped_null": 0}


# ------------------------------------------------------------- bad input ---
@pytest.mark.parametrize("kwargs", [
    {"column": "nope"},
    {"group_by": ["nope"]},
    {"group_by": ["region", "nope"]},
    {"filters": [{"column": "nope", "op": "==", "value": 1}]},
    {"date_column": "nope", "freq": "month"},
])
def test_bad_column_lists_available_columns(analysis_df, analysis_store, kwargs):
    kwargs = {"column": "amount", **kwargs}
    result = aggregate("ds1", kwargs.pop("column"), "sum", store=analysis_store, **kwargs)
    assert "unknown column 'nope'" in result["error"]
    assert "available columns" in result["error"]
    assert all(col in result["error"] for col in analysis_df.columns)


def test_unknown_dataset_lists_loaded_ids():
    store = {"ds2": pd.DataFrame({"a": [1]}), "ds10": pd.DataFrame({"a": [1]}), "ds1": pd.DataFrame({"a": [1]})}
    result = aggregate("ds99", "a", "sum", store=store)
    assert "ds99" in result["error"] and result["loaded"] == ["ds1", "ds2", "ds10"]
    assert aggregate("ds1", "a", "sum", store={})["loaded"] == []


@pytest.mark.parametrize("kwargs,fragment", [
    ({"column": "amount * 2"}, "unknown column"),
    ({"column": "amount; DROP TABLE x"}, "unknown column"),
    ({"agg": "sum(amount)"}, "unknown agg"),
    ({"agg": "__import__('os').system('id')"}, "unknown agg"),
    ({"agg": "lambda x: x"}, "unknown agg"),
    ({"filters": [{"column": "qty", "op": "== 1 or 1 ==", "value": 1}]}, "unsupported op"),
    ({"filters": [{"column": "qty", "op": "eval", "value": "1+1"}]}, "unsupported op"),
    ({"group_by": ["region + product"]}, "unknown column"),
])
def test_free_text_expressions_are_rejected(analysis_store, kwargs, fragment):
    args = {"column": "amount", "agg": "sum", **kwargs}
    result = aggregate("ds1", args.pop("column"), args.pop("agg"), store=analysis_store, **args)
    assert fragment in result["error"]


def test_structured_error_cases(analysis_store):
    df = pd.DataFrame({"value": [1, 2], "x": [3, 4]})
    assert "value" in aggregate("ds1", "x", "sum", group_by=["value"], store={"ds1": df})["error"]
    assert "duplicate" in run(analysis_store, group_by=["region", "region"])["error"]
    assert "not a date column" in run(analysis_store, date_column="region", freq="month")["error"]
    assert "order_date" in run(analysis_store, date_column="region", freq="month")["error"]
    assert "date_column is required" in run(analysis_store, freq="month")["error"]
    assert "date_column is required" in run(analysis_store, period={"start": "2024-01-01"})["error"]
    assert "unknown freq" in run(analysis_store, date_column="order_date", freq="week")["error"]
    assert "top_n needs" in run(analysis_store, top_n=3)["error"]


@pytest.mark.parametrize("period", [
    {}, "2024", [], {"foo": "2024-01-01"}, {"start": "not a date"}, {"start": None, "end": None},
    {"start": "2024-05-01", "end": "2024-01-01"}, {"start": ["2024-01-01"]},
])
def test_bad_period(analysis_store, period):
    result = run(analysis_store, date_column="order_date", period=period)
    assert "error" in result and "period" in result["error"]


@pytest.mark.parametrize("top_n", [0, -1, "3", 2.5, True, [1]])
def test_bad_top_n(analysis_store, top_n):
    assert "top_n" in run(analysis_store, group_by=["region"], top_n=top_n)["error"]


@pytest.mark.parametrize("filters", [
    "qty > 3", {"column": "qty"}, [1], ["qty"], [{"column": "qty", "op": "==", "value": 1, "extra": 1}],
    [{"column": "qty", "op": "==", "value": 1}, {"column": "qty"}],
    [{"column": "qty", "op": "like", "value": "1"}], [{"column": "qty", "op": "in", "value": 5}],
    [{"column": "qty", "op": "==", "value": [1, 2]}], [{"column": "qty", "op": "==", "value": None}],
    [{"column": "qty", "op": ">", "value": "abc"}], [{"column": "order_date", "op": ">", "value": "garbage"}],
    [{"column": "order_date", "op": "in", "value": ["garbage"]}],
])
def test_bad_filters(analysis_store, filters):
    assert "error" in run(analysis_store, filters=filters)


@pytest.mark.parametrize("kwargs", [
    {"dataset_id": None}, {"dataset_id": 1}, {"dataset_id": ["ds1"]}, {"dataset_id": {"a": 1}},
    {"column": None}, {"column": 7}, {"column": ["amount"]}, {"column": {"x": 1}},
    {"agg": None}, {"agg": 5}, {"agg": ["sum"]},
    {"group_by": 5}, {"group_by": {"a": 1}}, {"group_by": [1]}, {"group_by": [["region"]]}, {"group_by": [None]},
    {"freq": 3}, {"freq": ["month"]}, {"date_column": 5}, {"date_column": ["order_date"]},
    {"filters": 5}, {"period": 5}, {"top_n": None, "group_by": 5},
    {"store": None}, {"store": []}, {"store": "ds1"}, {"store": {"ds1": None}},
])
def test_no_bad_input_raises(analysis_store, kwargs):
    call = {"dataset_id": "ds1", "column": "amount", "agg": "sum", "store": analysis_store, **kwargs}
    result = aggregate(call.pop("dataset_id"), call.pop("column"), call.pop("agg"), **call)
    assert isinstance(result, dict) and "error" in result


def test_missing_arguments_do_not_raise():
    assert "error" in aggregate()
    assert "error" in aggregate("ds1")
    assert "error" in aggregate("ds1", "amount", "sum")  # store missing


# ------------------------------------------------------------ housekeeping ---
def test_results_are_json_safe_and_store_is_not_modified(analysis_df, analysis_store):
    before = analysis_df.copy(deep=True)
    results = [
        run(analysis_store), run(analysis_store, group_by=["region", "product"]),
        run(analysis_store, "order_date", "max", group_by=["region"]),
        run(analysis_store, date_column="order_date", freq="month", group_by=["product"]),
        run(analysis_store, filters=[{"column": "order_date", "op": ">", "value": "2024-01-01"}],
            date_column="order_date", period={"start": "2024-02-01"}),
    ]
    for result in results:
        assert "error" not in result
        json.dumps(result, allow_nan=False)
    pd.testing.assert_frame_equal(analysis_store["ds1"], before)


def test_aggregate_is_traced_and_logs_no_values(analysis_store, capfd):
    assert hasattr(aggregate, "__wrapped__")
    run(analysis_store, "amount", "sum", group_by=["region"],
        filters=[{"column": "region", "op": "==", "value": "Zanzibar-Q9"}])
    err = capfd.readouterr().err
    assert "tool=aggregate" in err and "'filters'" in err and "result_bytes=" in err
    assert "Zanzibar-Q9" not in err and "North" not in err
    assert capfd.readouterr().out == ""
