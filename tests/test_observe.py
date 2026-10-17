"""Tests for agent/observe.py: pure code, no LLM, no server, no network."""

from __future__ import annotations

import math
from typing import Any

import pytest

from agent.observe import (
    Observation,
    attempt_problems,
    build_feedback,
    entry_problems,
    observe,
    resolved_problems,
)


def entry(tool: str, output: Any, call_id: str = "c1") -> dict[str, Any]:
    return {"task_idx": 0, "tool_call_id": call_id, "tool": tool, "args": {}, "output": output}


def clean_sum() -> dict[str, Any]:
    return entry("aggregate", {"value": 18.0, "rows_used": 5, "rows_dropped_null": 1})


def judge(entries: list[dict], retries_used: int = 0, max_retries: int = 2, **kw: Any) -> Observation:
    return observe(entries, retries_used=retries_used, max_retries=max_retries, **kw)


EXISTS = lambda path: True  # noqa: E731
MISSING = lambda path: False  # noqa: E731


# ------------------------------------------------------------ clean result --
def test_clean_result_finishes():
    result = judge([clean_sum()])
    assert result == Observation("finish") and result.reasons == () and result.feedback == ""


@pytest.mark.parametrize("output", [
    {"value": 0.0, "rows_used": 3, "rows_dropped_null": 0},                     # zero is a fine value
    {"groups": [{"region": "N", "value": 1.0}], "rows_used": 4, "rows_dropped_null": 0},
    {"n": 38, "r": 0.93, "p_value": 1.2e-9, "rows_dropped_null": 2},
    {"statistic": -2.1, "p_value": 0.04, "group_sizes": {"a": 3, "b": 4}, "group_means": {"a": 1.0, "b": 2.0}},
    {"output": "5915.0\n"},
    {"sheets": [{"name": "Sales", "rows": 8, "columns": 4}]},
    {"dataset_id": "ds1", "rows": 3, "nulls": {}, "date_ranges": {}},
])
def test_ordinary_outputs_are_clean(output):
    assert entry_problems(entry("aggregate", output)) == []
    assert judge([entry("aggregate", output)]).action == "finish"


# ------------------------------------------------------ rule 1: tool error --
def test_tool_error_triggers_retry_and_quotes_the_message():
    result = judge([entry("aggregate", {"error": "unknown column 'nope'; available columns: ['a']"})])
    assert result.action == "retry"
    assert "aggregate returned an error" in result.reasons[0] and "unknown column 'nope'" in result.reasons[0]
    assert "unknown column 'nope'" in result.feedback


def test_very_long_error_is_cut_down():
    result = judge([entry("run_pandas", {"error": "x" * 5000})])
    assert len(result.reasons[0]) < 400


# --------------------------------------------------- rule 2: no tool called --
def test_no_tool_called_triggers_retry():
    result = judge([])
    assert result.action == "retry" and "no tool was called" in result.reasons[0]


# -------------------------------------------------------- rule 3: NaN / inf --
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_number_triggers_retry(bad):
    result = judge([entry("aggregate", {"value": bad, "rows_used": 3})])
    assert result.action == "retry" and "NaN or infinite" in result.reasons[0]


def test_non_finite_number_nested_anywhere_triggers_retry():
    nested = {"groups": [{"region": "N", "value": 1.0}, {"region": "S", "value": math.nan}], "rows_used": 4}
    assert judge([entry("aggregate", nested)]).action == "retry"
    deep = {"group_means": {"a": 1.0, "b": [2.0, {"x": math.inf}]}}
    assert judge([entry("compare_groups", deep)]).action == "retry"


def test_run_pandas_printing_a_lone_nan_or_inf_triggers_retry():
    for text in ("nan\n", "  NaN  ", "inf", "-inf\n", "Infinity", "total\nnan\n"):
        result = judge([entry("run_pandas", {"output": text})])
        assert result.action == "retry" and "NaN or infinite" in result.reasons[0], text


def test_nan_cells_inside_a_printed_table_are_not_a_problem():
    table = "   region  amount\n0   North     NaN\n1   South    80.0\n"
    assert judge([entry("run_pandas", {"output": table})]).action == "finish"
    assert judge([entry("run_pandas", {"output": "mean of nan_count column is 3"})]).action == "finish"


def test_a_group_labelled_inf_is_not_mistaken_for_a_number():
    labelled = {"groups": [{"tier": "inf", "value": 3.0}], "rows_used": 3, "rows_dropped_null": 0}
    assert judge([entry("aggregate", labelled)]).action == "finish"


# ---------------------------------------------- rule 4: empty / zero rows ----
def test_empty_groups_trigger_retry():
    result = judge([entry("aggregate", {"groups": [], "rows_used": 0, "rows_dropped_null": 0})])
    assert result.action == "retry"
    assert any("no groups" in r for r in result.reasons)


def test_empty_groups_alone_trigger_retry():
    assert judge([entry("aggregate", {"groups": []})]).action == "retry"


def test_rows_used_zero_triggers_retry_even_without_groups():
    result = judge([entry("aggregate", {"value": 0, "rows_used": 0, "rows_dropped_null": 0})])
    assert result.action == "retry" and "rows_used == 0" in result.reasons[0]


def test_one_row_is_enough():
    assert judge([entry("aggregate", {"value": 7.0, "rows_used": 1, "rows_dropped_null": 0})]).action == "finish"


# ------------------------------------------------ rule 5: p_value in [0, 1] --
@pytest.mark.parametrize("p", [1.5, -0.1, 2, 1.0000001])
def test_p_value_outside_unit_interval_triggers_retry(p):
    result = judge([entry("compare_groups", {"statistic": 1.0, "p_value": p})])
    assert result.action == "retry" and "outside [0, 1]" in result.reasons[0]


@pytest.mark.parametrize("p", [0, 0.0, 1, 1.0, 0.5])
def test_p_value_on_the_boundary_is_fine(p):
    assert judge([entry("compare_groups", {"statistic": 1.0, "p_value": p})]).action == "finish"


# ------------------------------------------------------ rule 6: |r| > 1 ------
@pytest.mark.parametrize("r", [1.2, -1.2, 5, -1.01])
def test_correlation_beyond_one_triggers_retry(r):
    result = judge([entry("compare_groups", {"n": 10, "r": r, "p_value": 0.1})])
    assert result.action == "retry" and "|r| cannot exceed 1" in result.reasons[0]


@pytest.mark.parametrize("r", [1.0, -1.0, 0, 0.999, 1.0 + 1e-12])
def test_correlation_within_one_is_fine(r):
    assert judge([entry("compare_groups", {"n": 10, "r": r, "p_value": 0.1})]).action == "finish"


# ------------------------------------------------ rule 7: chart path missing --
def test_chart_without_a_path_triggers_retry():
    for output in ({"kind": "bar"}, {"kind": "bar", "path": ""}, {"kind": "bar", "path": None}):
        result = judge([entry("make_chart", output)], path_exists=EXISTS)
        assert result.action == "retry" and "no chart path" in result.reasons[0]


def test_chart_path_that_does_not_exist_triggers_retry():
    result = judge([entry("make_chart", {"path": "/x/y.png", "kind": "bar"})], path_exists=MISSING)
    assert result.action == "retry" and "does not exist" in result.reasons[0]


def test_chart_with_an_existing_file_finishes(tmp_path):
    png = tmp_path / "bar.png"
    png.write_bytes(b"\x89PNG")
    assert judge([entry("make_chart", {"path": str(png), "kind": "bar"})]).action == "finish"  # real isfile
    assert judge([entry("make_chart", {"path": "/x/y.png", "kind": "bar"})], path_exists=EXISTS).action == "finish"


def test_the_path_rule_only_applies_to_make_chart():
    assert judge([entry("aggregate", {"value": 1.0, "rows_used": 2})], path_exists=MISSING).action == "finish"


# ----------------------------------------- a fixed problem is not held against --
def test_error_followed_by_a_clean_call_of_the_same_tool_is_resolved():
    entries = [entry("aggregate", {"error": "unknown column 'nope'"}, "c1"), clean_sum()]
    assert attempt_problems(entries) == []
    assert judge(entries).action == "finish"


def test_clean_call_followed_by_an_error_is_not_resolved():
    entries = [clean_sum(), entry("aggregate", {"error": "boom"}, "c2")]
    assert judge(entries).action == "retry"


def test_a_clean_call_of_a_different_tool_does_not_resolve_it():
    entries = [entry("aggregate", {"error": "boom"}), entry("list_sheets", {"sheets": []}, "c2")]
    assert judge(entries).action == "retry"


def test_every_kind_of_problem_can_be_resolved_by_a_later_clean_call():
    for bad in ({"groups": [], "rows_used": 0}, {"value": math.nan}, {"p_value": 3.0}, {"r": 2.0}):
        assert judge([entry("compare_groups", bad), entry("compare_groups", {"p_value": 0.2})]).action == "finish"


def test_unresolved_problems_are_all_reported_once():
    entries = [
        entry("aggregate", {"error": "bad column"}, "c1"),
        entry("aggregate", {"error": "bad column"}, "c2"),       # duplicate reason
        entry("compare_groups", {"p_value": 4.0}, "c3"),
    ]
    reasons = attempt_problems(entries)
    assert len(reasons) == 2 and "aggregate returned an error" in reasons[0] and "p_value" in reasons[1]


@pytest.mark.parametrize("output", [None, "text", 5, ["a"], {}])
def test_unexpected_outputs_never_raise(output):
    result = judge([entry("aggregate", output)])
    assert isinstance(result, Observation)
    if not isinstance(output, dict):
        assert result.action == "retry"
    else:
        assert result.action == "finish"  # an empty object has nothing wrong with it


def test_entries_missing_keys_do_not_raise():
    assert judge([{}]).action == "retry"
    assert judge([{"output": {"value": 1.0, "rows_used": 1}}]).action == "finish"


# ------------------------------------------------------ retry budget logic ---
def test_retry_while_budget_remains_then_fail():
    bad = [entry("aggregate", {"error": "boom"})]
    assert judge(bad, retries_used=0, max_retries=2).action == "retry"
    assert judge(bad, retries_used=1, max_retries=2).action == "retry"
    last = judge(bad, retries_used=2, max_retries=2)
    assert last.action == "failed" and "boom" in last.reasons[0] and last.feedback == ""


def test_max_retries_zero_fails_immediately():
    result = judge([entry("aggregate", {"error": "boom"})], retries_used=0, max_retries=0)
    assert result.action == "failed" and "boom" in result.reasons[0]
    assert judge([], retries_used=0, max_retries=0).action == "failed"  # no tool called counts too


def test_max_retries_zero_still_finishes_clean_results():
    assert judge([clean_sum()], retries_used=0, max_retries=0).action == "finish"


def test_a_clean_result_after_retries_finishes():
    assert judge([clean_sum()], retries_used=2, max_retries=2).action == "finish"


def test_negative_max_retries_behaves_like_zero():
    assert judge([entry("aggregate", {"error": "boom"})], max_retries=-3).action == "failed"


@pytest.mark.parametrize("max_retries", [1, 3, 5])
def test_the_limit_is_a_parameter(max_retries):
    bad = [entry("aggregate", {"error": "boom"})]
    assert judge(bad, retries_used=max_retries - 1, max_retries=max_retries).action == "retry"
    assert judge(bad, retries_used=max_retries, max_retries=max_retries).action == "failed"


# ----------------------------------------------------------------- feedback --
def test_feedback_lists_every_reason_and_the_retries_left():
    feedback = build_feedback(["reason one", "reason two"], 2)
    assert "- reason one" in feedback and "- reason two" in feedback and "2 retries left" in feedback
    assert "1 retry left" in build_feedback(["x"], 1)
    assert "arithmetic" in feedback


def test_feedback_from_observe_counts_down():
    bad = [entry("aggregate", {"error": "boom"})]
    assert "2 retries left" in judge(bad, retries_used=0, max_retries=2).feedback
    assert "1 retry left" in judge(bad, retries_used=1, max_retries=2).feedback


def test_observe_is_pure_it_does_not_modify_its_input():
    entries = [entry("aggregate", {"error": "boom"})]
    snapshot = [dict(e, output=dict(e["output"])) for e in entries]
    judge(entries)
    assert entries == snapshot


# ------------------------------------------------ resolved_problems (self-corrections) --
def test_resolved_problems_is_zero_for_a_clean_attempt():
    assert resolved_problems([clean_sum()]) == 0
    assert resolved_problems([clean_sum(), clean_sum()]) == 0


def test_resolved_problems_counts_an_error_fixed_by_a_later_call_of_the_same_tool():
    entries = [entry("aggregate", {"error": "unknown column 'nope'"}, "c1"), clean_sum()]
    assert resolved_problems(entries) == 1
    assert judge(entries).action == "finish"  # it is the same rule attempt_problems applies


def test_resolved_problems_is_zero_when_the_problem_was_never_fixed():
    assert resolved_problems([entry("aggregate", {"error": "boom"})]) == 0
    assert resolved_problems([clean_sum(), entry("aggregate", {"error": "boom"}, "c2")]) == 0  # error came last
    assert resolved_problems([entry("aggregate", {"error": "boom"}), entry("list_sheets", {"sheets": []}, "c2")]) == 0
    assert resolved_problems([]) == 0  # no tool called: nothing was fixed


def test_resolved_problems_counts_each_fixed_call_and_each_problem_in_it():
    two_errors = [entry("aggregate", {"error": "a"}, "c1"), entry("aggregate", {"error": "b"}, "c2"), clean_sum()]
    assert resolved_problems(two_errors) == 2
    two_problems_in_one_call = [entry("aggregate", {"value": math.nan, "groups": []}, "c1"), clean_sum()]
    assert resolved_problems(two_problems_in_one_call) == 2


def test_resolved_problems_ignores_unfixed_problems_next_to_fixed_ones():
    entries = [
        entry("aggregate", {"error": "fixed later"}, "c1"),
        clean_sum(),
        entry("compare_groups", {"p_value": 4.0}, "c3"),         # never fixed
    ]
    assert resolved_problems(entries) == 1
    assert judge(entries).action == "retry"  # the unfixed one still costs a retry


def test_resolved_problems_matches_attempt_problems_on_every_kind_of_problem():
    for bad in ({"groups": [], "rows_used": 0}, {"value": math.nan}, {"p_value": 3.0}, {"r": 2.0}):
        entries = [entry("compare_groups", bad), entry("compare_groups", {"p_value": 0.2})]
        assert attempt_problems(entries) == [] and resolved_problems(entries) >= 1


def test_resolved_problems_respects_the_chart_path_check_and_does_not_modify_input():
    entries = [entry("make_chart", {"path": "/x/a.png"}, "c1"), entry("make_chart", {"path": "/x/b.png"}, "c2")]
    snapshot = [dict(e) for e in entries]
    assert resolved_problems(entries, path_exists=lambda p: p.endswith("b.png")) == 1
    assert entries == snapshot
