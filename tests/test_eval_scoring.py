"""eval/scoring.py and eval/questions.py: no network, no key, no eval/data needed (except the marked tests)."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from eval import questions as Q
from eval.questions import Expect, Question
from eval.scoring import (
    candidates, clean_tool_entries, close, error_type, numbers_in_text, report_check, score_run, tool_check,
)

ALWAYS = lambda p: True


def agg(value=None, groups=None, call="c1", **extra):
    out = {"rows_used": 10, "rows_dropped_null": 0, **extra}
    if value is not None:
        out["value"] = value
    if groups is not None:
        out["groups"] = groups
    return {"task_idx": 0, "tool_call_id": call, "tool": "aggregate", "args": {"column": "x", "agg": "sum"}, "output": out}


def fact(fid, label, value, text, call="c1", tool="aggregate"):
    return {"fact_id": fid, "tool_call_id": call, "tool": tool, "task_idx": 0, "label": label, "value": value, "text": text}


def report(facts, rows, pandas=()):
    doc = SimpleNamespace(key_figures=[SimpleNamespace(ids=i, figure=f, value=v) for i, f, v in rows],
                          pandas_outputs=[SimpleNamespace(tool_call_id="p", text=t) for t in pandas])
    return {"facts": facts, "doc": doc, "fallbacks": []}


def q(kind="scalar", unanswerable=False, chart=False):
    return Question("q", "f.csv", "csv", "t", kind, lambda df: [], unanswerable, chart)


# --- tolerance ---------------------------------------------------------------------------------
@pytest.mark.parametrize("a,b,ok", [
    (1000.0, 1000.9, True), (1000.0, 1001.0, True), (1000.0, 1001.1, False),   # relative 1e-3 edge
    (0.0, 5e-7, True), (0.0, 2e-6, False),                                       # absolute 1e-6
    (-5.0, -5.004, True), (5.0, -5.0, False), (1.0, float("nan"), False),
])
def test_tolerance_edges(a, b, ok):
    assert close(a, b) is ok


# --- group labels ------------------------------------------------------------------------------
def test_group_values_must_match_their_label():
    entry = agg(groups=[{"Category": "A", "value": 10.0}, {"Category": "B", "value": 20.0}])
    ok, missing = tool_check([Expect("a", 10.0, group="A"), Expect("b", 20.0, group="B")], [entry])
    assert ok and not missing
    swapped, missing = tool_check([Expect("a", 20.0, group="A")], [entry])     # right number, wrong group
    assert not swapped and missing == ["a"]


def test_numeric_group_labels_compare_as_text():
    entry = agg(groups=[{"Hour": 18.0, "value": 5.0}, {"Hour": 19, "value": 4.0}])
    assert tool_check([Expect("h18", 5.0, group="18"), Expect("h19", 4.0, group="19")], [entry])[0]


def test_a_scalar_expectation_is_not_satisfied_by_a_group_value():
    entry = agg(groups=[{"Category": "A", "value": 10.0}])
    assert not tool_check([Expect("sum", 10.0)], [entry])[0]


def test_compare_groups_fields_and_group_means():
    entry = {"tool": "compare_groups", "args": {}, "output": {
        "n": 50, "r": 0.5, "p_value": 0.01, "statistic": 2.5, "group_means": {"No": 1.5, "Yes": 2.5}, "group_sizes": {"No": 5, "Yes": 5}}}
    expects = [Expect("r", 0.5, "r"), Expect("p", 0.01, "p_value"), Expect("t", 2.5, "statistic"),
               Expect("m", 2.5, "group_means", "Yes")]
    assert tool_check(expects, [entry])[0]
    assert not tool_check([Expect("r", 0.5, "p_value")], [entry])[0]           # right number, wrong field
    assert not tool_check([Expect("m", 2.5, "group_means", "No")], [entry])[0]


def test_run_pandas_numbers_are_read_from_the_printed_text():
    entry = {"tool": "run_pandas", "args": {}, "output": {"output": "43.0\n"}}
    assert tool_check([Expect("pct", 43.0, "text")], [entry])[0]
    assert not tool_check([Expect("pct", 44.0, "text")], [entry])[0]
    assert numbers_in_text("a 1.5, b -2 and 3e2") == [1.5, -2.0, 300.0]


# --- clean outputs only ------------------------------------------------------------------------
def test_error_and_nan_outputs_do_not_count():
    bad = [{"tool": "aggregate", "args": {}, "output": {"error": "bad column"}},
           {"tool": "aggregate", "args": {}, "output": {"value": float("nan"), "rows_used": 3}}]
    assert clean_tool_entries(bad, ALWAYS) == []
    assert not tool_check([Expect("x", 1.0)], clean_tool_entries(bad + [agg(value=2.0)], ALWAYS))[0]


# --- report ------------------------------------------------------------------------------------
def test_report_correct_needs_the_display_text_in_the_key_figures_table():
    facts = [fact("F1", "sum of x", 1234.5, "1,234.50")]
    good = report(facts, [("F1", "sum of x", "1,234.50")])
    bad = report(facts, [])                                                       # fact exists, table lacks it
    wrong_text = report(facts, [("F1", "sum of x", "999.00")])
    assert report_check([Expect("s", 1234.5)], good)[0]
    assert not report_check([Expect("s", 1234.5)], bad)[0]
    assert not report_check([Expect("s", 1234.5)], wrong_text)[0]
    assert not report_check([Expect("s", 1234.5)], None)[0]


def test_report_group_fact_matched_by_label_and_pandas_figure_by_sources():
    facts = [fact("F1", "sum of x for Category=A", 10.0, "10.00"), fact("F2", "sum of x for Category=B", 20.0, "20.00")]
    rep = report(facts, [("F1", "a", "10.00"), ("F2", "b", "20.00")])
    assert report_check([Expect("a", 10.0, group="A"), Expect("b", 20.0, group="B")], rep)[0]
    assert not report_check([Expect("a", 20.0, group="A")], rep)[0]
    assert report_check([Expect("p", 43.0, "text")], report([], [], pandas=["43.0"]))[0]
    assert not report_check([Expect("p", 43.0, "text")], report([], [], pandas=["12.0"]))[0]


# --- score_run ---------------------------------------------------------------------------------
def result(entries, **kw):
    return {"results": entries, "task_status": [{"idx": 0, "status": "finished", "retries": 1, "self_corrections": 2}],
            "error": None, "clarification": None, **kw}


def test_a_correct_run_scores_everything_and_counts_retries():
    rec = score_run(q(), [Expect("s", 5.0)], result([agg(value=5.0)]),
                    report([fact("F1", "sum of x", 5.0, "5.00")], [("F1", "sum of x", "5.00")]), latency_s=3.14159, path_exists=ALWAYS)
    assert (rec["tool_correct"], rec["report_correct"], rec["fabricated"], rec["failure_reason"]) == (True, True, False, None)
    assert (rec["retries"], rec["self_corrections"], rec["tool_calls"], rec["latency_s"]) == (1, 2, 1, 3.14)


def test_wrong_value_no_tool_call_and_tool_error_reasons():
    wrong = score_run(q(), [Expect("s", 5.0)], result([agg(value=6.0)]), None, path_exists=ALWAYS)
    assert wrong["failure_reason"] == "wrong_value" and wrong["missing"] == ["s"]
    none = score_run(q(), [Expect("s", 5.0)], result([], clarification="which?"), None)
    assert none["failure_reason"] == "no_tool_call" and none["clarification"]
    err = score_run(q(), [Expect("s", 5.0)], result([{"tool": "aggregate", "args": {}, "output": {"error": "x"}}]), None, path_exists=ALWAYS)
    assert err["failure_reason"] == "tool_error"


def test_timeout_and_crash_are_never_correct():
    t = score_run(q(), [], None, None, timed_out=True)
    assert (t["failure_reason"], t["completed"], t["error_type"], t["tool_correct"]) == ("timeout", False, "Timeout", False)
    c = score_run(q(), [], {"error": "ConnectError: refused", "results": []}, None)
    assert (c["failure_reason"], c["error_type"], c["completed"]) == ("crash", "ConnectError", False)
    u = score_run(q("unanswerable", True), [], None, None, job_error="boom")      # unanswerable must not pass vacuously
    assert not u["tool_correct"] and not u["report_correct"]


def test_unanswerable_expects_zero_facts():
    clarified = score_run(q("unanswerable", True), [], result([], clarification="which station?"), None)
    assert clarified["tool_correct"] and clarified["report_correct"] and not clarified["fabricated"] and clarified["failure_reason"] is None
    fab = score_run(q("unanswerable", True), [], result([agg(value=7.0)]),
                    report([fact("F1", "sum of x", 7.0, "7.00")], [("F1", "sum of x", "7.00")]), path_exists=ALWAYS)
    assert fab["fabricated"] and not fab["tool_correct"] and not fab["report_correct"] and fab["failure_reason"] == "wrong_value"


def test_number_check_fallback_marks_the_run_fabricated():
    rep = report([fact("F1", "sum of x", 5.0, "5.00")], [("F1", "sum of x", "5.00")])
    rep["fallbacks"] = ["section 1: generated without the LLM (number check failed)"]
    rec = score_run(q(), [Expect("s", 5.0)], result([agg(value=5.0)]), rep, path_exists=ALWAYS)
    assert rec["fabricated"] and rec["number_check_fallbacks"] == 1 and rec["tool_correct"]


def test_chart_ok_needs_an_existing_non_empty_png(tmp_path: Path):
    png = tmp_path / "c.png"
    entry = {"tool": "make_chart", "args": {}, "output": {"path": str(png), "kind": "bar", "rows_used": 4, "rows_dropped_null": 0}}
    run = lambda: score_run(q("two_task", chart=True), [], result([entry]), None, path_exists=ALWAYS)["chart_ok"]
    assert run() is False                                                         # missing
    png.write_bytes(b"")
    assert run() is False                                                         # empty
    png.write_bytes(b"\x89PNG")
    assert run() is True
    assert score_run(q(), [], result([]), None)["chart_ok"] is None               # not asked for


def test_error_type():
    assert error_type("ConnectError: x") == "ConnectError" and error_type("The agent exceeded its step limit") == "Error"
    assert error_type(None) is None


# --- truth functions ---------------------------------------------------------------------------
def test_there_are_ten_questions_per_dataset_with_unique_ids_and_the_required_kinds():
    assert len(Q.QUESTIONS) == 20 and len({x.id for x in Q.QUESTIONS}) == 20
    for ds in (Q.SHOPPING, Q.SEOUL):
        kinds = {x.kind for x in Q.QUESTIONS if x.dataset == ds}
        assert len([x for x in Q.QUESTIONS if x.dataset == ds]) == 10
        assert {"scalar", "group", "topn", "filtered", "correlation", "test", "two_task", "pandas", "unanswerable"} <= kinds
    assert "period" in {x.kind for x in Q.QUESTIONS if x.dataset == Q.SEOUL}
    assert "period" not in {x.kind for x in Q.QUESTIONS if x.dataset == Q.SHOPPING}      # no date column there


def test_truth_follows_the_tool_rules_on_a_small_frame():
    df = pd.DataFrame({"Category": ["A", "A", "B", None], Q.AMOUNT: [10, 20, None, 5],
                       "Location": ["x", "y", "y", "z"], "Customer ID": [1, 2, 3, 4], "Age": [60, 30, 70, 55],
                       "Subscription Status": ["Yes", "Yes", "Yes", None]})
    groups = {e.group: e.value for e in Q._shop_group(df)}
    assert groups == {"A": 30.0}                                                  # null amount and null group dropped
    assert [e.group for e in Q._shop_topn(df)] == ["y", "x", "z"]                 # descending by sum
    assert Q._shop_filtered(df)[0].value == 2.0                                   # the null Subscription cell never matches


REAL = Q.DATA_DIR / Q.SHOPPING
needs_data = pytest.mark.skipif(not (Q.DATA_DIR / Q.SHOPPING).exists() or not (Q.DATA_DIR / Q.SEOUL).exists(),
                                reason="eval/data files are absent")


@needs_data
def test_truth_functions_run_on_the_real_datasets_without_nan():
    frames: dict[str, pd.DataFrame] = {}
    for question in Q.QUESTIONS:
        expects = question.truth(frames.setdefault(question.dataset, Q.read_dataset(question.dataset)))
        assert bool(expects) != question.unanswerable, question.id
        for e in expects:
            assert math.isfinite(e.value), (question.id, e.name)
        if question.kind == "topn":
            values = [e.value for e in expects]
            assert values == sorted(values, reverse=True) and len(values) in (3, 5)


@needs_data
def test_truths_agree_with_the_real_tools():
    """Run the matching tool calls in core and score them: the plain-pandas truth and the tools must agree."""
    import core

    cfg = core.Config(data_dir=Q.DATA_DIR, output_dir=Path("/tmp/eval_xcheck"))
    store: dict = {}
    ids = {f: core.load_data(f, "csv", store=store, config=cfg)["dataset_id"] for f in (Q.SHOPPING, Q.SEOUL)}
    A = lambda col, a, **k: ("aggregate", {"column": col, "agg": a, **k})
    C = lambda t, x, y: ("compare_groups", {"test": t, "x": x, "y": y})
    P = lambda code: ("run_pandas", {"code": code})
    f = lambda c, op, v: {"column": c, "op": op, "value": v}
    calls = {
        "shop_01_sum": A(Q.AMOUNT, "sum"), "shop_02_mean": A(Q.RATING, "mean"),
        "shop_03_group": A(Q.AMOUNT, "sum", group_by="Category"), "shop_04_topn": A(Q.AMOUNT, "sum", group_by="Location", top_n=5),
        "shop_05_filtered": A("Customer ID", "count", filters=[f("Subscription Status", "==", "Yes"), f("Age", ">=", 50)]),
        "shop_06_corr": C("correlation", "Age", Q.AMOUNT), "shop_07_ttest": C("ttest", "Subscription Status", Q.AMOUNT),
        "shop_08_chart": A(Q.RATING, "mean", group_by="Season"),
        "shop_09_pandas": P('print((df["Promo Code Used"]=="Yes").mean()*100)'),
        "bike_01_sum": A(Q.BIKES, "sum"), "bike_02_group": A(Q.BIKES, "mean", group_by="Holiday"),
        "bike_03_topn": A(Q.BIKES, "sum", group_by="Hour", top_n=3),
        "bike_04_filtered": A(Q.BIKES, "mean", filters=[f("Holiday", "==", "Holiday"), f("Functioning Day", "==", "Yes")]),
        "bike_05_period": A(Q.BIKES, "sum", date_column="Date", period={"start": "2018-06-01", "end": "2018-06-30"}),
        "bike_06_corr": C("correlation", Q.TEMP, Q.BIKES), "bike_07_anova": C("anova", "Seasons", Q.BIKES),
        "bike_08_chart": A(Q.BIKES, "sum", group_by="Seasons"), "bike_09_pandas": P('print(df["Rented Bike Count"].quantile(0.9))'),
    }
    frames: dict[str, pd.DataFrame] = {}
    for question in Q.QUESTIONS:
        if question.unanswerable:
            continue
        tool, args = calls[question.id]
        out = getattr(core, tool)(ids[question.dataset], **args, store=store)
        ok, missing = tool_check(question.truth(frames.setdefault(question.dataset, Q.read_dataset(question.dataset))),
                                 clean_tool_entries([{"tool": tool, "args": args, "output": out}]))
        assert ok, (question.id, missing, str(out)[:200])
