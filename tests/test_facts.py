"""Tests for agent/facts.py: built from a hand-made results log; no server, no LLM, no network."""

from __future__ import annotations

import math
from typing import Any

import pytest

from agent.facts import (
    build_facts,
    facts_by_id,
    format_float,
    format_int,
    format_p,
    format_r,
    format_value,
)


def entry(tool: str, args: dict, output: dict, call_id: str, task_idx: int = 0) -> dict[str, Any]:
    return {"task_idx": task_idx, "tool_call_id": call_id, "tool": tool, "args": args, "output": output}


AGG = {"dataset_id": "ds1", "column": "amount", "agg": "sum"}
SCALAR = entry("aggregate", AGG, {"value": 1234567.891, "rows_used": 9, "rows_dropped_null": 1}, "c1")
GROUPED = entry(
    "aggregate", {**AGG, "group_by": ["region"]},
    {"groups": [{"region": "North", "value": 470.25}, {"region": "South", "value": 1200}], "rows_used": 9, "rows_dropped_null": 1},
    "c2",
)
CORR = entry("compare_groups", {"dataset_id": "ds1", "test": "correlation", "x": "qty", "y": "amount"},
             {"n": 9, "r": -0.93456, "p_value": 0.0004, "rows_dropped_null": 1}, "c3")
TTEST = entry("compare_groups", {"dataset_id": "ds1", "test": "ttest", "x": "arm", "y": "metric"},
              {"statistic": -2.3456, "p_value": 0.04123, "group_sizes": {"ctl": 23, "trt": 15},
               "group_means": {"ctl": 10.1, "trt": 12345.678}, "rows_dropped_null": 2}, "c4")


# ------------------------------------------------------------- formatting ----
@pytest.mark.parametrize("value,text", [(0, "0"), (7, "7"), (1000, "1,000"), (1234567, "1,234,567"), (-1500, "-1,500")])
def test_ints_use_thousands_separators(value, text):
    assert format_int(value) == text and format_value(value) == text


@pytest.mark.parametrize("value,text", [
    (955.0, "955.00"), (1234.5, "1,234.50"), (0.005, "0.01"), (0.004, "0.00"), (-0.001, "0.00"),
    (-1234.567, "-1,234.57"), (1234567.891, "1,234,567.89"), (1e9, "1,000,000,000.00"),
])
def test_floats_have_two_decimals_and_thousands_separators(value, text):
    assert format_float(value) == text and format_value(value) == text


@pytest.mark.parametrize("value,text", [(0.93456, "0.935"), (-0.93456, "-0.935"), (1.0, "1.000"), (0.0, "0.000"),
                                        (-0.0004, "0.000"), (0.5, "0.500")])
def test_r_has_three_decimals(value, text):
    assert format_r(value) == text


@pytest.mark.parametrize("value,text", [
    (0.04123, "0.04123"), (0.5, "0.5000"), (1.0, "1.000"), (0.001, "0.001000"), (0.0012345, "0.001234"),
    (0.0009999, "<0.001"), (1e-12, "<0.001"), (0.0, "<0.001"), (0.123456, "0.1235"),
])
def test_p_value_has_four_significant_digits_or_is_floored(value, text):
    assert format_p(value) == text


def test_text_values_are_kept_and_midnight_dates_become_plain_dates():
    assert format_value("2024-04-01T00:00:00") == "2024-04-01"
    assert format_value("2024-03-31T15:30:00") == "2024-03-31 15:30:00"
    assert format_value("North") == "North"
    assert format_value(True) == "True"


# ----------------------------------------------------------- extraction -----
def by_label(facts: list[dict]) -> dict[str, dict]:
    return {f["label"]: f for f in facts}


def test_scalar_value_rows_used_and_rows_dropped_are_extracted():
    facts = build_facts([SCALAR])
    assert [(f["fact_id"], f["value"], f["text"]) for f in facts] == [
        ("F1", 1234567.891, "1,234,567.89"), ("F2", 9, "9"), ("F3", 1, "1")]
    assert facts[0]["label"] == "sum of amount"
    assert facts[1]["label"] == "rows used for sum of amount"
    assert facts[2]["label"] == "rows dropped for missing values in sum of amount"
    for f in facts:
        assert set(f) == {"fact_id", "tool_call_id", "tool", "task_idx", "label", "value", "text"}
        assert (f["tool_call_id"], f["tool"], f["task_idx"]) == ("c1", "aggregate", 0)


def test_each_group_value_is_a_fact_and_the_label_names_the_group():
    facts = build_facts([GROUPED])
    groups = [f for f in facts if "for region=" in f["label"]]
    assert [(f["label"], f["text"]) for f in groups] == [
        ("sum of amount for region=North", "470.25"), ("sum of amount for region=South", "1,200")]
    assert [f["value"] for f in groups] == [470.25, 1200]


def test_multi_key_groups_and_date_groups_in_the_label():
    out = {"groups": [{"order_date_quarter": "2024-Q1", "region": "North", "value": 3.5}], "rows_used": 1, "rows_dropped_null": 0}
    (group, *_) = build_facts([entry("aggregate", {**AGG, "agg": "mean", "freq": "quarter", "group_by": ["region"]}, out, "c9")])
    assert group["label"] == "mean of amount for order_date_quarter=2024-Q1, region=North"


def test_filters_and_period_appear_in_the_label():
    args = {**AGG, "filters": [{"column": "region", "op": "==", "value": "North"}],
            "period": {"start": "2024-01-01", "end": "2024-03-31"}, "date_column": "order_date"}
    (fact, *_) = build_facts([entry("aggregate", args, {"value": 5.0, "rows_used": 1, "rows_dropped_null": 0}, "c1")])
    assert fact["label"] == "sum of amount [region == North; 2024-01-01..2024-03-31]"


def test_correlation_facts_n_r_p_and_dropped_rows():
    facts = by_label(build_facts([CORR]))
    assert facts["correlation of qty and amount: n"]["text"] == "9"
    assert facts["correlation of qty and amount: r"]["text"] == "-0.935"
    assert facts["correlation of qty and amount: p-value"]["text"] == "<0.001"
    assert facts["rows dropped for missing values in correlation of qty and amount"]["text"] == "1"


def test_ttest_facts_statistic_p_means_and_sizes():
    facts = by_label(build_facts([TTEST]))
    assert facts["t-test of metric by arm: statistic"]["text"] == "-2.35"
    assert facts["t-test of metric by arm: p-value"]["text"] == "0.04123"
    assert facts["t-test of metric by arm: mean of metric in arm=trt"]["text"] == "12,345.68"
    assert facts["t-test of metric by arm: mean of metric in arm=ctl"]["text"] == "10.10"
    assert facts["t-test of metric by arm: size of metric in arm=ctl"]["text"] == "23"
    assert facts["t-test of metric by arm: size of metric in arm=trt"]["value"] == 15


def test_anova_label_and_text_min_value():
    anova = entry("compare_groups", {"dataset_id": "ds1", "test": "anova", "x": "grp", "y": "metric"},
                  {"statistic": 3.0, "p_value": 0.2, "group_sizes": {"a": 3, "b": 4}, "group_means": {"a": 1.0, "b": 2.0}}, "c5")
    assert any(f["label"] == "ANOVA of metric by grp: statistic" for f in build_facts([anova]))
    first_date = entry("aggregate", {**AGG, "column": "order_date", "agg": "min"},
                       {"value": "2024-01-05T00:00:00", "rows_used": 9, "rows_dropped_null": 0}, "c6")
    assert build_facts([first_date])[0]["text"] == "2024-01-05"


def test_ids_are_sequential_across_entries_in_log_order():
    facts = build_facts([SCALAR, GROUPED, CORR])
    assert [f["fact_id"] for f in facts] == [f"F{i}" for i in range(1, len(facts) + 1)]
    assert [f["tool_call_id"] for f in facts][:3] == ["c1", "c1", "c1"] and facts[-1]["tool_call_id"] == "c3"
    assert list(facts_by_id(facts)) == [f["fact_id"] for f in facts]
    assert facts_by_id(facts)["F1"]["text"] == "1,234,567.89"


# ------------------------------------------------------ what is left out -----
def test_run_pandas_output_produces_no_facts():
    pandas_entry = entry("run_pandas", {"dataset_id": "ds1", "code": "print(1)"}, {"output": "5915.0\n"}, "c7")
    assert build_facts([pandas_entry]) == []
    assert [f["tool_call_id"] for f in build_facts([pandas_entry, SCALAR])] == ["c1", "c1", "c1"]


def test_other_tools_produce_no_facts():
    for tool, output in [("make_chart", {"path": "/x/a.png", "kind": "bar", "rows_used": 9, "rows_dropped_null": 0}),
                         ("list_sheets", {"sheets": [{"name": "s", "rows": 3, "columns": 2}]}),
                         ("load_data", {"dataset_id": "ds1", "rows": 3, "nulls": {}, "date_ranges": {}})]:
        assert build_facts([entry(tool, {}, output, "c8")], path_exists=lambda p: True) == []


@pytest.mark.parametrize("output", [
    {"error": "unknown column 'x'"},
    {"value": math.nan, "rows_used": 3, "rows_dropped_null": 0},
    {"groups": [], "rows_used": 0, "rows_dropped_null": 0},
    {"value": None, "rows_used": 0, "rows_dropped_null": 2},
    {"n": 5, "r": 1.5, "p_value": 0.1},
    {"n": 5, "r": 0.5, "p_value": 3.0},
])
def test_unclean_entries_produce_no_facts(output):
    assert build_facts([entry("aggregate", AGG, output, "bad")]) == []
    assert build_facts([entry("compare_groups", {"test": "correlation", "x": "a", "y": "b"}, output, "bad")]) == []


def test_clean_entries_still_count_next_to_unclean_ones():
    bad = entry("aggregate", AGG, {"error": "boom"}, "bad")
    assert [f["tool_call_id"] for f in build_facts([bad, SCALAR])] == ["c1", "c1", "c1"]


def test_entries_of_failed_tasks_are_excluded():
    failed_entry = entry("aggregate", AGG, {"value": 99.0, "rows_used": 3, "rows_dropped_null": 0}, "f1", task_idx=1)
    ok_entry = {**SCALAR, "task_idx": 0}
    status = [{"idx": 0, "status": "finished"}, {"idx": 1, "status": "failed"}]
    facts = build_facts([ok_entry, failed_entry], status)
    assert {f["task_idx"] for f in facts} == {0} and "f1" not in {f["tool_call_id"] for f in facts}
    assert {f["task_idx"] for f in build_facts([ok_entry, failed_entry])} == {0, 1}   # without a status, nothing is excluded
    assert build_facts([failed_entry], [{"idx": 1, "status": "failed"}]) == []


def test_a_repeated_identical_call_in_one_task_gives_facts_once():
    again = {**SCALAR, "tool_call_id": "c1b"}
    facts = build_facts([SCALAR, again])
    assert len(facts) == 3 and {f["tool_call_id"] for f in facts} == {"c1"}
    other_task = {**SCALAR, "tool_call_id": "c1c", "task_idx": 1}
    assert len(build_facts([SCALAR, other_task])) == 6                  # a different task keeps its own copy
    other_args = entry("aggregate", {**AGG, "agg": "mean"}, {"value": 2.0, "rows_used": 9, "rows_dropped_null": 1}, "c1d")
    assert len(build_facts([SCALAR, other_args])) == 6


def test_no_results_and_odd_entries_never_raise():
    assert build_facts([]) == []
    assert build_facts([{"tool": "aggregate"}, {"tool": "aggregate", "output": "not a dict", "task_idx": 0}]) == []
    assert build_facts([entry("aggregate", {}, {"value": 1.0, "rows_used": 1, "rows_dropped_null": 0}, "c1")])[0]["label"] == "None of None"


def test_build_facts_does_not_modify_its_input():
    log = [SCALAR, GROUPED, CORR, TTEST]
    snapshot = [{**e, "output": dict(e["output"])} for e in log]
    build_facts(log)
    assert log == snapshot
