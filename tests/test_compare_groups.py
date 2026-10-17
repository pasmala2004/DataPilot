"""Tests for compare_groups(): expected values come from pandas / scipy inside the test."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from core import compare_groups


@pytest.fixture
def cmp_df() -> pd.DataFrame:
    """40 rows: correlated x/y, 2-group arm with unequal size/variance, 3-group grp, nulls."""
    rng = np.random.default_rng(7)
    n = 40
    x = rng.normal(size=n)
    arm = np.array(["ctl"] * 25 + ["trt"] * 15, dtype=object)
    metric = np.where(arm == "ctl", rng.normal(10, 1, n), rng.normal(11, 3, n))
    df = pd.DataFrame({
        "x": x,
        "y": 2 * x + rng.normal(size=n),
        "arm": arm,
        "metric": metric,
        "grp": rng.choice(["a", "b", "c"], n),
        "flag": rng.integers(0, 2, n),
        "is_ok": rng.choice([True, False], n),
        "const": 5.0,
    })
    df.loc[3, "y"] = np.nan          # pairwise drop with x
    df.loc[7, "x"] = np.nan
    df.loc[2, "metric"] = np.nan
    df.loc[10, "arm"] = None
    return df


@pytest.fixture
def cmp_store(cmp_df: pd.DataFrame) -> dict:
    return {"ds1": cmp_df}


def run(store: dict, test: str, x: str, y: str) -> dict[str, Any]:
    return compare_groups("ds1", test, x, y, store=store)


# ------------------------------------------------------------ correlation ---
def test_correlation_matches_pandas_and_scipy(cmp_df, cmp_store):
    result = run(cmp_store, "correlation", "x", "y")
    sub = cmp_df[["x", "y"]].dropna()
    assert result["n"] == len(sub) == 38
    assert result["r"] == pytest.approx(sub.corr().loc["x", "y"])
    assert result["p_value"] == pytest.approx(stats.pearsonr(sub["x"], sub["y"]).pvalue)
    assert result["rows_dropped_null"] == 2
    assert result["n"] < len(cmp_df)  # pairwise drop actually happened


def test_correlation_is_symmetric_and_works_on_int_columns(cmp_df, cmp_store):
    assert run(cmp_store, "correlation", "y", "x")["r"] == pytest.approx(run(cmp_store, "correlation", "x", "y")["r"])
    result = run(cmp_store, "correlation", "flag", "metric")
    sub = cmp_df[["flag", "metric"]].dropna()
    assert result["r"] == pytest.approx(sub.corr().loc["flag", "metric"])


# ------------------------------------------------------------------ ttest ---
def welch_expected(df: pd.DataFrame, group: str, value: str) -> tuple[Any, dict, dict]:
    sub = df.dropna(subset=[group, value])
    parts = {str(k): g[value].to_numpy() for k, g in sub.groupby(group)}
    a, b = list(parts.values())
    return stats.ttest_ind(a, b, equal_var=False), {k: len(v) for k, v in parts.items()}, \
        {k: v.mean() for k, v in parts.items()}


def test_ttest_matches_scipy_welch(cmp_df, cmp_store):
    result = run(cmp_store, "ttest", "arm", "metric")
    expected, sizes, means = welch_expected(cmp_df, "arm", "metric")
    assert result["statistic"] == pytest.approx(expected.statistic)
    assert result["p_value"] == pytest.approx(expected.pvalue)
    assert result["group_sizes"] == sizes == {"ctl": 23, "trt": 15}
    assert list(result["group_sizes"]) == ["ctl", "trt"]  # sorted order; statistic = ctl - trt
    for label, mean in means.items():
        assert result["group_means"][label] == pytest.approx(mean)
    assert result["statistic"] < 0  # trt mean is higher
    assert result["rows_dropped_null"] == 2


def test_ttest_is_welch_not_student(cmp_df, cmp_store):
    sub = cmp_df.dropna(subset=["arm", "metric"])
    a, b = (g["metric"] for _, g in sub.groupby("arm"))
    student = stats.ttest_ind(a, b, equal_var=True)
    result = run(cmp_store, "ttest", "arm", "metric")
    assert abs(result["p_value"] - student.pvalue) > 1e-4  # fixture really distinguishes them


def test_ttest_with_numeric_and_boolean_group_columns(cmp_df, cmp_store):
    for column in ("flag", "is_ok"):
        result = run(cmp_store, "ttest", column, "metric")
        expected, sizes, _ = welch_expected(cmp_df, column, "metric")
        assert result["statistic"] == pytest.approx(expected.statistic)
        assert result["group_sizes"] == sizes


# ------------------------------------------------------------------ anova ---
def test_anova_matches_scipy_with_three_groups(cmp_df, cmp_store):
    result = run(cmp_store, "anova", "grp", "metric")
    sub = cmp_df.dropna(subset=["grp", "metric"])
    parts = {k: g["metric"].to_numpy() for k, g in sub.groupby("grp")}
    expected = stats.f_oneway(*parts.values())
    assert list(parts) == ["a", "b", "c"]
    assert result["statistic"] == pytest.approx(expected.statistic)
    assert result["p_value"] == pytest.approx(expected.pvalue)
    assert result["group_sizes"] == {k: len(v) for k, v in parts.items()}
    for label, values in parts.items():
        assert result["group_means"][label] == pytest.approx(values.mean())
    assert sum(result["group_sizes"].values()) + result["rows_dropped_null"] == len(cmp_df)


def test_anova_with_two_groups_matches_scipy(cmp_df, cmp_store):
    result = run(cmp_store, "anova", "arm", "metric")
    sub = cmp_df.dropna(subset=["arm", "metric"])
    expected = stats.f_oneway(*(g["metric"] for _, g in sub.groupby("arm")))
    assert result["statistic"] == pytest.approx(expected.statistic)
    assert result["p_value"] == pytest.approx(expected.pvalue)


# -------------------------------------------------------- wrong group counts ---
def test_ttest_needs_exactly_two_groups(cmp_store):
    three = run(cmp_store, "ttest", "grp", "metric")
    assert "exactly 2 groups" in three["error"] and "found 3" in three["error"]
    one = run(cmp_store, "ttest", "const", "metric")  # a single group (all rows 5.0)
    assert "exactly 2 groups" in one["error"] and "found 1" in one["error"]


def test_anova_needs_at_least_two_groups(cmp_store):
    result = run(cmp_store, "anova", "const", "metric")
    assert "at least 2 groups" in result["error"] and "found 1" in result["error"]


def test_anova_rejects_more_than_50_groups():
    df = pd.DataFrame({"id": [f"id{i}" for i in range(120)], "v": np.arange(120.0)})
    result = compare_groups("ds1", "anova", "id", "v", store={"ds1": df})
    assert "at most 50" in result["error"] and "120" in result["error"]


def test_every_group_needs_two_values():
    df = pd.DataFrame({"g": ["a", "a", "a", "b"], "v": [1.0, 2.0, 3.0, 4.0]})
    for test in ("ttest", "anova"):
        error = compare_groups("ds1", test, "g", "v", store={"ds1": df})["error"]
        assert "at least 2" in error and "smallest has 1" in error


def test_group_counts_are_after_null_removal():
    df = pd.DataFrame({"g": ["a", "a", "b", "b", "c"], "v": [1.0, 2.0, 3.0, 4.0, None]})
    result = compare_groups("ds1", "ttest", "g", "v", store={"ds1": df})  # 'c' has no valid value
    assert result["group_sizes"] == {"a": 2, "b": 2}


# ------------------------------------------------------------ non-numeric ---
@pytest.mark.parametrize("test,x,y,bad", [
    ("correlation", "arm", "y", "arm"),
    ("correlation", "x", "arm", "arm"),
    ("correlation", "x", "is_ok", "is_ok"),   # booleans are not numeric
    ("ttest", "arm", "grp", "grp"),
    ("anova", "grp", "arm", "arm"),
    ("ttest", "arm", "is_ok", "is_ok"),
])
def test_non_numeric_columns_give_clear_errors(cmp_store, test, x, y, bad):
    error = run(cmp_store, test, x, y)["error"]
    assert f"'{bad}' must be numeric" in error


# ------------------------------------------------------------ NaN results ---
def test_constant_column_correlation_is_an_error(cmp_store):
    assert "NaN" in run(cmp_store, "correlation", "const", "y")["error"]
    assert "error" in run(cmp_store, "correlation", "const", "const")


def test_identical_values_within_groups_is_an_error():
    both_constant = pd.DataFrame({"g": ["a"] * 3 + ["b"] * 3, "v": [1.0] * 3 + [1.0] * 3})
    different_constants = pd.DataFrame({"g": ["a"] * 3 + ["b"] * 3, "v": [1.0] * 3 + [2.0] * 3})
    for df in (both_constant, different_constants):
        for test in ("ttest", "anova"):
            result = compare_groups("ds1", test, "g", "v", store={"ds1": df})
            assert "NaN" in result["error"] and "non-finite" in result["error"]


def test_correlation_needs_three_complete_rows():
    df = pd.DataFrame({"a": [1.0, 2.0, None, 4.0], "b": [2.0, None, 3.0, None]})
    assert "at least 3" in compare_groups("ds1", "correlation", "a", "b", store={"ds1": df})["error"]


# ----------------------------------------------------------- bad inputs ---
def test_unknown_dataset_lists_loaded_ids():
    store = {"ds1": pd.DataFrame({"a": [1]}), "ds2": pd.DataFrame({"a": [1]})}
    result = compare_groups("ds7", "ttest", "a", "a", store=store)
    assert "ds7" in result["error"] and result["loaded"] == ["ds1", "ds2"]
    assert compare_groups("ds1", "ttest", "a", "a", store={})["loaded"] == []


@pytest.mark.parametrize("x,y", [("nope", "y"), ("x", "nope")])
def test_unknown_columns_list_available_columns(cmp_df, cmp_store, x, y):
    error = run(cmp_store, "correlation", x, y)["error"]
    assert "unknown column 'nope'" in error and all(c in error for c in cmp_df.columns)


@pytest.mark.parametrize("test", ["pearson", "t-test", "ttest; drop", "", "Correlation"])
def test_unknown_test_name(cmp_store, test):
    assert "unknown test" in run(cmp_store, test, "x", "y")["error"]


@pytest.mark.parametrize("kwargs", [
    {"dataset_id": None}, {"dataset_id": 3}, {"dataset_id": ["ds1"]},
    {"test": None}, {"test": 5}, {"test": ["ttest"]}, {"test": {"a": 1}},
    {"x": None}, {"x": 3}, {"x": ["arm"]}, {"x": {"a": 1}},
    {"y": None}, {"y": 3}, {"y": ["metric"]}, {"y": {"a": 1}},
    {"store": None}, {"store": []}, {"store": "ds1"}, {"store": {"ds1": None}},
])
def test_no_bad_input_raises(cmp_store, kwargs):
    call = {"dataset_id": "ds1", "test": "ttest", "x": "arm", "y": "metric", "store": cmp_store, **kwargs}
    result = compare_groups(call.pop("dataset_id"), call.pop("test"), call.pop("x"), call.pop("y"), **call)
    assert isinstance(result, dict) and "error" in result


def test_missing_arguments_do_not_raise():
    assert "error" in compare_groups()
    assert "error" in compare_groups("ds1", "ttest", "a", "b")  # store missing


# ------------------------------------------------------------ housekeeping ---
def test_results_are_json_safe_and_store_is_not_modified(cmp_df, cmp_store):
    before = cmp_df.copy(deep=True)
    for test, x, y in [("correlation", "x", "y"), ("ttest", "arm", "metric"), ("anova", "grp", "metric"),
                       ("ttest", "flag", "metric")]:
        result = run(cmp_store, test, x, y)
        assert "error" not in result
        json.dumps(result, allow_nan=False)
    pd.testing.assert_frame_equal(cmp_store["ds1"], before)


def test_compare_groups_is_traced_and_logs_no_values(cmp_store, capfd):
    assert hasattr(compare_groups, "__wrapped__")
    run(cmp_store, "ttest", "arm", "metric")
    err = capfd.readouterr()
    assert "tool=compare_groups" in err.err and "args=['dataset_id', 'test', 'x', 'y', 'store']" in err.err
    assert "ctl" not in err.err and "trt" not in err.err
    assert err.out == ""
