"""Tests for agent/report.py: number check, writer rounds, code fallbacks, Markdown assembly.

A hand-built run result and a scripted writer: no server, no network, no API key.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent.report import (
    WriterOutput,
    WriterSection,
    allowed_literals,
    build_report,
    check_output,
    check_text,
    fallback_section,
    render,
)
from agent.facts import build_facts, facts_by_id
from scripted import ScriptedChatModel

NOW = datetime(2026, 10, 11, 14, 30, 5)


# --------------------------------------------------------------------------- #
# Fixtures: a small finished run
# --------------------------------------------------------------------------- #
def entry(tool: str, args: dict, output: dict, call_id: str, task_idx: int) -> dict[str, Any]:
    return {"task_idx": task_idx, "tool_call_id": call_id, "tool": tool, "args": args, "output": output}


def status(idx: int, state: str = "finished", task: str = "", **extra: Any) -> dict[str, Any]:
    base = {"idx": idx, "agent": "analysis", "task": task or f"task text {idx}", "status": state, "retries": 0,
            "self_corrections": 0, "reasons": [], "note": "agent note with number 987654 inside"}
    return {**base, **extra}


@pytest.fixture
def run_result(tmp_path: Path) -> dict[str, Any]:
    chart = tmp_path / "outputs" / "charts" / "bar_region.png"
    chart.parent.mkdir(parents=True)
    chart.write_bytes(b"\x89PNG fake")
    results = [
        entry("aggregate", {"dataset_id": "ds1", "column": "amount", "agg": "sum", "group_by": ["region"]},
              {"groups": [{"region": "North", "value": 1470.25}, {"region": "South", "value": 80.0}],
               "rows_used": 9, "rows_dropped_null": 1}, "call_a", 0),
        entry("compare_groups", {"dataset_id": "ds1", "test": "correlation", "x": "qty", "y": "amount"},
              {"n": 9, "r": 0.83456, "p_value": 0.0051, "rows_dropped_null": 1}, "call_b", 1),
        entry("run_pandas", {"dataset_id": "ds1", "code": "print(df.qty.max())"}, {"output": "5\n"}, "call_p", 1),
        entry("make_chart", {"dataset_id": "ds1", "kind": "bar", "x": "region", "y": "amount"},
              {"path": str(chart), "kind": "bar", "rows_used": 9, "rows_dropped_null": 1}, "call_c", 2),
        entry("aggregate", {"dataset_id": "ds1", "column": "nope", "agg": "sum"}, {"error": "unknown column 'nope'"}, "call_x", 3),
    ]
    return {
        "question": "How are sales?", "file": "sales.xlsx", "sheet": "orders", "error": None,
        "schema": {"columns": {"order_date": "datetime64[ns]", "region": "str", "amount": "float64", "q1_sales": "float64"}},
        "tasks": [], "results": results,
        "task_status": [
            status(0, task="Sum amount by region."), status(1, task="Correlate qty with amount."),
            status(2, task="Bar chart of amount by region.", agent="chart"),
            status(3, "failed", task="Sum a column that is missing.", retries=2,
                   reasons=["aggregate returned an error: unknown column 'nope'"]),
        ],
    }


def writer_reply(sections: dict[int, str], summary: str) -> AIMessage:
    args = {"sections": [{"task_idx": i, "text": t} for i, t in sections.items()], "summary": summary}
    return AIMessage(content="", tool_calls=[{"name": "WriterOutput", "args": args, "id": "w1"}])


GOOD = {0: "Sales by region: {{F1}} for North and {{F2}} for South.", 1: "Correlation r is {{F6}} (p {{F7}}).",
        2: "The chart below shows amount by region."}
GOOD_SUMMARY = "Sales were measured by region; North is {{F1}}. The correlation r is {{F6}}."


def run(coro: Any) -> Any:
    return asyncio.run(asyncio.wait_for(coro, 30))


def report(run_result: dict, llm: Any, tmp_path: Path, **kw: Any) -> dict[str, Any]:
    return run(build_report(run_result, llm, output_dir=tmp_path / "outputs", now=NOW, **kw))


def section_text(markdown: str, heading: str) -> str:
    match = re.search(rf"^## {heading}\n\n(.*?)(?=^## |\Z)", markdown, re.S | re.M)
    assert match, heading
    return match.group(1).strip()


# =========================================================================== #
# Number check
# =========================================================================== #
IDS = {"F1", "F2", "F3"}


def test_unknown_placeholder_is_caught():
    issues = check_text("Total is {{F99}}.", IDS)
    assert issues == ["{{F99}} is not a known fact id"]
    assert check_text("Total is {{F1}} and {{ F2 }}.", IDS) == []          # spaces inside braces are fine


@pytest.mark.parametrize("text,token", [
    ("Sales were 45,650 in total.", "45,650"), ("The mean is 12.5 here.", "12.5"), ("About 3 regions.", "3"),
    ("Growth of 12%.", "12"), ("Value {F1} is odd.", "1"),                  # single braces are not placeholders
    ("It is {{F1}} or maybe 7.", "7"), ("Total ١٢٣ here.", "١٢٣"),           # Arabic-Indic digits count as digits
    ("Born in 1985 and 45,650.", "45,650"), ("The year 2024.5 was odd.", "5"),
])
def test_stray_numbers_are_caught(text, token):
    issues = check_text(text, IDS)
    assert any(f"stray number '{token}'" in i for i in issues), issues


@pytest.mark.parametrize("text", [
    "In 2024 sales rose.", "In 1990 and 2100 only.", "Q3 was the busiest, then Q1.", "2024-Q3 and Q4-2025 both appear.",
    "March (2024-03) and 2024-03-15 are labels.", "1. First point\n2. Second point\n10) Tenth point",
    "Plain words only.", "North has {{F1}} and South {{F2}}.", "See Q2 2025: {{F3}}.",
])
def test_years_quarters_dates_and_list_markers_are_allowed(text):
    assert check_text(text, IDS) == []


@pytest.mark.parametrize("text", ["In 1989 something.", "In 2101 something.", "Month 2024-13 is not a date.", "Quarter Q5.", "The 3. item"])
def test_lookalikes_outside_the_allowed_forms_are_caught(text):
    assert check_text(text, IDS) != []


def test_names_with_letters_and_digits_may_appear_but_bare_numbers_may_not():
    assert check_text("The q1_sales column and Zone 3.", IDS, ["q1_sales", "Zone 3"]) == []
    assert check_text("The q1_sales column has 3 rows.", IDS, ["q1_sales"]) != []   # the 3 is still stray
    assert check_text("Group 3 here.", IDS, []) != []                                 # not known, so flagged


def test_malformed_placeholders_are_caught():
    assert any("malformed" in i for i in check_text("Value {{F1 here.", IDS))
    assert any("malformed" in i for i in check_text("Value {{total}} here.", IDS))


def test_missing_sections_summary_and_duplicates_are_caught():
    ok = WriterOutput(sections=[WriterSection(task_idx=0, text="Fine {{F1}}."), WriterSection(task_idx=1, text="Fine {{F2}}.")],
                      summary="Short {{F3}}.")
    assert check_output(ok, IDS, [0, 1]) == {}
    missing = WriterOutput(sections=[WriterSection(task_idx=0, text="Fine {{F1}}.")], summary="Short {{F3}}.")
    assert check_output(missing, IDS, [0, 1]) == {1: ["there is no section for task_idx 1"]}
    empty_text = WriterOutput(sections=[WriterSection(task_idx=0, text="  ")], summary="Short.")
    assert 0 in check_output(empty_text, IDS, [0])
    no_summary = WriterOutput(sections=[WriterSection(task_idx=0, text="Fine.")], summary=" ")
    assert check_output(no_summary, IDS, [0]) == {"summary": ["the summary is empty"]}
    dup = WriterOutput(sections=[WriterSection(task_idx=0, text="A."), WriterSection(task_idx=0, text="B.")], summary="S.")
    assert any("more than one" in i for i in check_output(dup, IDS, [0])[0])
    extra = WriterOutput(sections=[WriterSection(task_idx=0, text="A."), WriterSection(task_idx=7, text="stray 45,650")], summary="S.")
    assert check_output(extra, IDS, [0]) == {}                               # sections for other tasks are ignored


def test_the_summary_and_every_section_are_checked():
    bad = WriterOutput(sections=[WriterSection(task_idx=0, text="Fine."), WriterSection(task_idx=1, text="Bad 45,650.")],
                       summary="Also bad 12.")
    problems = check_output(bad, IDS, [0, 1])
    assert set(problems) == {"summary", 1}


def test_render_replaces_every_placeholder_with_the_fact_text():
    facts = facts_by_id(build_facts([entry("aggregate", {"column": "a", "agg": "sum"},
                                           {"value": 1234.5, "rows_used": 3, "rows_dropped_null": 0}, "c", 0)]))
    assert render("Total {{F1}}; used {{ F2 }}.", facts) == "Total 1,234.50; used 3."


def test_allowed_literals_only_keeps_names_with_a_letter_and_a_digit(run_result):
    run_result["results"].append(entry("aggregate", {}, {"groups": [{"zone": "Zone 3", "value": 1.0}, {"zone": "7", "value": 2.0}],
                                                        "rows_used": 2, "rows_dropped_null": 0}, "c9", 0))
    literals = allowed_literals(run_result)
    assert {"q1_sales", "Zone 3"} <= literals and "7" not in literals and "North" not in literals


def test_fallback_section_lists_the_tasks_facts_as_placeholders():
    facts = build_facts([entry("aggregate", {"column": "amount", "agg": "sum"},
                               {"value": 5.0, "rows_used": 1, "rows_dropped_null": 0}, "c", 0)])
    text = fallback_section(facts)
    assert "sum of amount: {{F1}}" in text and check_text(text, {"F1", "F2", "F3"}) == []
    assert "no figures" in fallback_section([])
    many = build_facts([entry("aggregate", {"column": "a", "agg": "sum", "group_by": ["g"]},
                              {"groups": [{"g": f"x{i}", "value": float(i)} for i in range(30)], "rows_used": 30,
                               "rows_dropped_null": 0}, "c", 0)])
    assert "more figures" in fallback_section(many)


# =========================================================================== #
# Writer rounds
# =========================================================================== #
def test_round_one_clean_means_one_writer_call_and_no_fallback(run_result, tmp_path):
    llm = ScriptedChatModel(script=[writer_reply(GOOD, GOOD_SUMMARY)])
    result = report(run_result, llm, tmp_path)
    assert result["rounds"] == 1 and result["issues"] == [] and result["fallbacks"] == [] and llm.script == []
    md = result["markdown"]
    assert "Sales by region: 1,470.25 for North and 80.00 for South." in md
    assert "Correlation r is 0.835 (p 0.005100)." in md
    assert "{{" not in md and "}}" not in md                                   # nothing left unreplaced


def test_round_one_stray_number_then_round_two_clean(run_result, tmp_path):
    bad = {**GOOD, 0: "North sold 45,650 units and {{F2}} for South."}
    llm = ScriptedChatModel(script=[writer_reply(bad, GOOD_SUMMARY), writer_reply(GOOD, GOOD_SUMMARY)])
    result = report(run_result, llm, tmp_path)
    assert result["rounds"] == 2 and llm.script == []
    assert any("round 1" in i and "stray number '45,650'" in i for i in result["issues"])
    assert result["fallbacks"] == []                                           # the repair round fixed it
    assert "45,650" not in result["markdown"]                                  # the rejected text never shipped
    assert "Sales by region: 1,470.25 for North" in result["markdown"]
    feedback = llm.seen[1][-1]                                                 # what the writer got in round 2
    assert isinstance(feedback, HumanMessage) and "45,650" in feedback.content and "section for task 1" in feedback.content


def test_always_bad_writer_gets_code_fallback_and_a_caveat(run_result, tmp_path):
    bad = {**GOOD, 1: "The correlation is 0.83 which is strong."}
    llm = ScriptedChatModel(script=[writer_reply(bad, GOOD_SUMMARY), writer_reply(bad, GOOD_SUMMARY)])
    result = report(run_result, llm, tmp_path)
    assert result["rounds"] == 2 and llm.script == []                           # exactly two rounds, never a third
    md = result["markdown"]
    assert "0.83 which" not in md
    assert "The figures computed for this task were:" in md and "correlation of qty and amount: r: 0.835" in md
    assert "section 2: generated without the LLM (number check failed)" in section_text(md, "Caveats")
    assert result["fallbacks"] == ["section 2: generated without the LLM (number check failed)"]
    assert "Sales by region: 1,470.25 for North" in md                          # the good sections are kept


def test_a_part_that_passed_in_an_earlier_round_is_kept(run_result, tmp_path):
    round1 = writer_reply({**GOOD, 1: "bad 12 here"}, GOOD_SUMMARY)                 # section 2 bad, rest fine
    round2 = writer_reply(GOOD, "The summary now has 99 in it.")                          # section 2 fixed, summary broken
    llm = ScriptedChatModel(script=[round1, round2])
    result = report(run_result, llm, tmp_path)
    md = result["markdown"]
    assert "Sales were measured by region; North is 1,470.25." in md                     # summary from round 1
    assert "Correlation r is 0.835" in md and "99" not in md
    assert result["fallbacks"] == []


def test_a_writer_that_never_writes_a_section_gets_the_fallback(run_result, tmp_path):
    no_section_two = {k: v for k, v in GOOD.items() if k != 2}
    llm = ScriptedChatModel(script=[writer_reply(no_section_two, GOOD_SUMMARY)] * 2)
    result = report(run_result, llm, tmp_path)
    assert any("no section for task_idx 2" in i for i in result["issues"])
    assert "section 3: generated without the LLM (number check failed)" in result["caveats"]


def test_max_rounds_one_means_no_repair_round(run_result, tmp_path):
    bad = {**GOOD, 0: "stray 45,650."}
    llm = ScriptedChatModel(script=[writer_reply(bad, GOOD_SUMMARY), writer_reply(GOOD, GOOD_SUMMARY)])
    result = report(run_result, llm, tmp_path, max_rounds=1)
    assert result["rounds"] == 1 and len(llm.script) == 1
    assert "section 1: generated without the LLM (number check failed)" in result["caveats"]


def test_a_writer_that_cannot_produce_structured_output_falls_back_completely(run_result, tmp_path):
    plain = AIMessage(content="Here is a report in prose with 12 numbers.")
    llm = ScriptedChatModel(script=[plain] * 4)                                 # 2 rounds x (try + structured repair)
    result = report(run_result, llm, tmp_path)
    assert llm.script == [] and result["rounds"] == 2
    assert len(result["fallbacks"]) == 4 and all("generated without the LLM" in f for f in result["fallbacks"])
    assert "prose" not in result["markdown"]
    assert re.findall(r"\{\{|\}\}", result["markdown"]) == []


def test_a_writer_that_raises_does_not_crash_the_report(run_result, tmp_path):
    from test_llm import RaisingModel

    result = report(run_result, RaisingModel(), tmp_path)
    assert result["rounds"] == 1                                                # an API failure is not retried
    assert any("writer error" in f for f in result["fallbacks"]) and any("the writer failed" in i for i in result["issues"])
    assert "Key figures" in result["markdown"] and Path(result["path"]).is_file()


def test_the_writer_sees_ids_tools_labels_and_task_text_but_no_values_or_notes(run_result, tmp_path):
    llm = ScriptedChatModel(script=[writer_reply(GOOD, GOOD_SUMMARY)])
    report(run_result, llm, tmp_path)
    prompt = "\n".join(str(m.content) for m in llm.seen[0])
    assert "F1 | aggregate | sum of amount for region=North" in prompt and "Sum amount by region." in prompt
    assert "How are sales?" in prompt
    for secret in ("1,470.25", "1470.25", "0.835", "0.83456", "987654", "agent note", "0.0051", "80.00"):
        assert secret not in prompt, secret
    assert "task_idx 3" not in prompt                                            # the failed task is not sent
    assert "Sum a column that is missing" not in prompt


# =========================================================================== #
# Report content
# =========================================================================== #
def test_sections_come_in_the_required_order(run_result, tmp_path):
    md = report(run_result, ScriptedChatModel(script=[writer_reply(GOOD, GOOD_SUMMARY)]), tmp_path)["markdown"]
    headings = re.findall(r"^#{1,2} (.+)$", md, re.M)
    assert headings == ["Analysis report: sales.xlsx (orders)", "Summary", "Key figures", "Analysis", "Charts", "Caveats", "Sources"]


def test_key_figures_table_is_generated_from_the_facts(run_result, tmp_path):
    result = report(run_result, ScriptedChatModel(script=[writer_reply(GOOD, GOOD_SUMMARY)]), tmp_path)
    table = section_text(result["markdown"], "Key figures").splitlines()
    assert table[0] == "| ID | Figure | Value |"
    rows = {cells[1]: cells[3] for cells in (re.split(r"\s*\|\s*", ln.strip("| ")) for ln in table[2:]) for cells in [["", *cells, ""]]}
    assert len(result["facts"]) == 8 and len(table) - 2 == 6          # 8 facts; two used/dropped pairs share a row
    assert rows["F1"] == "1,470.25" and rows["F2"] == "80.00" and rows["F6"] == "0.835" and rows["F7"] == "0.005100"


def test_rows_used_and_dropped_are_one_paired_row_per_tool_call_but_every_fact_stays(run_result, tmp_path):
    result = report(run_result, ScriptedChatModel(script=[writer_reply(GOOD, GOOD_SUMMARY)]), tmp_path)
    md = result["markdown"]
    table = section_text(md, "Key figures")
    assert "| F3, F4 | rows used / rows dropped for missing values: sum of amount [group_by region] | 9 / 1 |" in table or \
        re.search(r"\| F3, F4 \| rows used / rows dropped for missing values: sum of amount.* \| 9 / 1 \|", table)
    assert "| F5, F8 | rows used / rows dropped for missing values: correlation of qty and amount | 9 / 1 |" in table
    assert not re.search(r"^\| F\d+ \| rows (used|dropped)", table, re.M)           # no separate rows any more
    assert [f["fact_id"] for f in result["facts"]] == [f"F{i}" for i in range(1, 9)]   # the facts list is untouched
    sources = section_text(md, "Sources")
    assert "F1-F4" in sources and "F5-F8" in sources and "call_a" in sources


def test_analysis_has_one_section_per_finished_task_and_skips_failed_ones(run_result, tmp_path):
    md = report(run_result, ScriptedChatModel(script=[writer_reply(GOOD, GOOD_SUMMARY)]), tmp_path)["markdown"]
    analysis = section_text(md, "Analysis")
    assert [h for h in re.findall(r"^### (Task \d+)$", analysis, re.M)] == ["Task 1", "Task 2", "Task 3"]
    assert "Task 4" not in analysis and "*Sum amount by region.*" in analysis


def test_caveats_list_failures_retries_dropped_rows_and_the_tool_statement(run_result, tmp_path):
    run_result["task_status"][1]["retries"] = 1
    run_result["task_status"][0]["self_corrections"] = 2
    caveats = report(run_result, ScriptedChatModel(script=[writer_reply(GOOD, GOOD_SUMMARY)]), tmp_path)["caveats"]
    text = "\n".join(caveats)
    assert "Task 4 failed" in text and "unknown column 'nope'" in text and "Sum a column that is missing." in text
    assert "Task 2 needed 1 retry" in text and "Task 1: the agent corrected 2 tool problem(s)" in text
    assert "1 row(s) were left out for missing values in sum of amount" in text
    assert caveats[-1].startswith("All figures in this report were computed by tools (aggregate, compare_groups)")
    assert "never typed a number" in caveats[-1]


def test_causal_wording_is_a_warning_not_a_failure(run_result, tmp_path):
    causal = {**GOOD, 0: "North is {{F1}} because of its size, which leads to more orders; South is {{F2}}."}
    llm = ScriptedChatModel(script=[writer_reply(causal, GOOD_SUMMARY)])
    result = report(run_result, llm, tmp_path)
    assert result["rounds"] == 1 and result["fallbacks"] == [] and "because of its size" in result["markdown"]
    caveats = section_text(result["markdown"], "Caveats")
    assert "Causal wording ('because') appears in section for task 1" in caveats
    assert "Causal wording ('leads to') appears in section for task 1" in caveats
    assert "association only" in caveats


def test_causal_wording_in_code_built_text_is_not_flagged(run_result, tmp_path):
    bad = {**GOOD, 0: "stray 45,650 because."}
    llm = ScriptedChatModel(script=[writer_reply(bad, GOOD_SUMMARY)] * 2)
    caveats = "\n".join(report(run_result, llm, tmp_path)["caveats"])
    assert "Causal wording" not in caveats                                       # the rejected text was replaced


def test_charts_are_relative_links_to_files_that_exist(run_result, tmp_path):
    result = report(run_result, ScriptedChatModel(script=[writer_reply(GOOD, GOOD_SUMMARY)]), tmp_path)
    charts = section_text(result["markdown"], "Charts")
    (link,) = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", charts)
    assert link == "../charts/bar_region.png" and not Path(link).is_absolute()
    assert (Path(result["path"]).parent / link).resolve().is_file()


def test_sources_lists_every_used_call_with_fact_ids_and_the_run_pandas_output(run_result, tmp_path):
    result = report(run_result, ScriptedChatModel(script=[writer_reply(GOOD, GOOD_SUMMARY)]), tmp_path)
    sources = section_text(result["markdown"], "Sources")
    for call_id in ("call_a", "call_b", "call_p", "call_c"):
        assert call_id in sources
    assert "call_x" not in sources                                               # the failed call is not a source
    row_a = next(ln for ln in sources.splitlines() if "call_a" in ln and ln.startswith("|"))
    assert "F1-F4" in row_a and "aggregate" in row_a and '"group_by":["region"]' in row_a
    assert "```\n5\n\n```" in sources and "free text, not used as a figure" in sources
    used_calls = {f["tool_call_id"] for f in result["facts"]}
    assert all(c in sources for c in used_calls)


def test_run_pandas_numbers_never_become_facts(run_result, tmp_path):
    result = report(run_result, ScriptedChatModel(script=[writer_reply(GOOD, GOOD_SUMMARY)]), tmp_path)
    assert all(f["tool"] != "run_pandas" for f in result["facts"])
    assert not any("call_p" == f["tool_call_id"] for f in result["facts"])


def test_file_is_saved_under_reports_with_a_timestamp_and_never_overwritten(run_result, tmp_path):
    first = report(run_result, ScriptedChatModel(script=[writer_reply(GOOD, GOOD_SUMMARY)]), tmp_path)
    second = report(run_result, ScriptedChatModel(script=[writer_reply(GOOD, GOOD_SUMMARY)]), tmp_path)
    p1, p2 = Path(first["path"]), Path(second["path"])
    assert p1.name == "report_20261011_143005.md" and p2.name == "report_20261011_143005_2.md"
    assert p1.parent == (tmp_path / "outputs" / "reports").resolve() and p1.is_absolute()
    assert p1.read_text(encoding="utf-8") == first["markdown"] and p2.is_file()


def test_pipes_in_labels_do_not_break_the_tables(tmp_path):
    result = {"question": "q", "file": "f.xlsx", "sheet": None, "schema": {"columns": {}},
              "task_status": [status(0, task="t")],
              "results": [entry("aggregate", {"column": "v", "agg": "sum", "group_by": ["g"]},
                                {"groups": [{"g": "a|b", "value": 1.0}], "rows_used": 1, "rows_dropped_null": 0}, "c1", 0)]}
    llm = ScriptedChatModel(script=[writer_reply({0: "See {{F1}}."}, "Done {{F1}}.")])
    md = run(build_report(result, llm, output_dir=tmp_path, now=NOW))["markdown"]
    row = next(ln for ln in md.splitlines() if ln.startswith("| F1 "))
    assert "a\\|b" in row and row.count("|") - row.count("\\|") == 4


# =========================================================================== #
# No clean results: the report is built from code only
# =========================================================================== #
def test_without_clean_results_the_report_is_code_only_and_the_llm_is_never_called(tmp_path):
    result_in = {"question": "How are sales?", "file": "sales.xlsx", "sheet": "orders", "schema": {"columns": {}},
                 "results": [entry("aggregate", {"column": "nope", "agg": "sum"}, {"error": "unknown column 'nope'"}, "c1", 0)],
                 "task_status": [status(0, "failed", task="Sum a missing column.", retries=2,
                                        reasons=["aggregate returned an error: unknown column 'nope'"])]}
    llm = ScriptedChatModel(script=[])                                           # any LLM call raises AssertionError
    result = run(build_report(result_in, llm, output_dir=tmp_path, now=NOW))
    md = result["markdown"]
    assert result["facts"] == [] and result["rounds"] == 0 and llm.seen == []
    assert "Nothing could be verified" in section_text(md, "Summary")
    assert "No verified figures were produced." in section_text(md, "Key figures")
    caveats = section_text(md, "Caveats")
    assert "Task 1 failed" in caveats and "Sum a missing column." in caveats and "unknown column 'nope'" in caveats
    assert "No charts were produced." in section_text(md, "Charts") and "No tool results were used." in section_text(md, "Sources")
    assert Path(result["path"]).is_file()


def test_an_empty_or_minimal_run_result_still_gives_a_report(tmp_path):
    result = run(build_report({}, ScriptedChatModel(script=[]), output_dir=tmp_path, now=NOW))
    assert "Nothing could be verified" in result["markdown"] and Path(result["path"]).is_file()
    stopped = run(build_report({"error": "Cannot open 'x.xlsx': file not found", "task_status": []}, ScriptedChatModel(script=[]),
                               output_dir=tmp_path, now=NOW))
    assert "The run stopped with an error: Cannot open 'x.xlsx'" in stopped["markdown"]


def test_failed_task_entries_that_were_clean_are_not_reported_as_figures(run_result, tmp_path):
    run_result["results"].append(entry("aggregate", {"column": "amount", "agg": "max"},
                                       {"value": 999.0, "rows_used": 9, "rows_dropped_null": 1}, "call_failed_task", 3))
    result = report(run_result, ScriptedChatModel(script=[writer_reply(GOOD, GOOD_SUMMARY)]), tmp_path)
    assert "999.00" not in result["markdown"] and "call_failed_task" not in result["markdown"]


def test_build_report_does_not_modify_the_run_result(run_result, tmp_path):
    import copy

    snapshot = copy.deepcopy(run_result)
    report(run_result, ScriptedChatModel(script=[writer_reply(GOOD, GOOD_SUMMARY)]), tmp_path)
    assert run_result == snapshot


# =========================================================================== #
# Golden Markdown: the structured ReportDoc refactor must not change a single character
# =========================================================================== #
GOLDEN = Path(__file__).parent / "golden" / "report.md"


def golden_markdown(run_result: dict, tmp_path: Path) -> str:
    """A report that exercises every section: retries, a self-correction, a causal word, a failed task."""
    run_result["task_status"][1]["retries"] = 1
    run_result["task_status"][0]["self_corrections"] = 2
    run_result["results"][2]["output"] = {"output": "".join(f"line {i}\n" for i in range(1, 8))}   # 7 printed lines
    sections = {**GOOD, 0: "North is {{F1}} because of its size; South is {{F2}} (| pipe)."}
    return report(run_result, ScriptedChatModel(script=[writer_reply(sections, GOOD_SUMMARY)]), tmp_path)["markdown"]


def test_markdown_output_is_identical_to_the_golden_file(run_result, tmp_path):
    assert golden_markdown(run_result, tmp_path) == GOLDEN.read_text(encoding="utf-8")


# =========================================================================== #
# run_pandas output in Sources: first 5 lines + "(truncated)"
# =========================================================================== #
def test_cap_lines_edges():
    from agent.report import cap_lines
    five = "".join(f"l{i}\n" for i in range(1, 6))
    assert cap_lines("") == "" and cap_lines("x") == "x" and cap_lines("5\n") == "5\n"
    assert cap_lines(five) == five                                    # exactly 5 lines (with a trailing newline): untouched
    assert cap_lines(five.rstrip("\n")) == five.rstrip("\n")
    assert cap_lines(five + "l6\n") == "l1\nl2\nl3\nl4\nl5\n(truncated)"
    assert cap_lines("a\n\nb\n\nc\n\nd") == "a\n\nb\n\nc\n(truncated)"  # blank lines count as lines
    assert cap_lines("l1\nl2\nl3", max_lines=2) == "l1\nl2\n(truncated)"


def test_long_run_pandas_output_is_cut_in_the_markdown_sources(run_result, tmp_path):
    run_result["results"][2]["output"] = {"output": "".join(f"row {i}\n" for i in range(1, 21))}
    md = report(run_result, ScriptedChatModel(script=[writer_reply(GOOD, GOOD_SUMMARY)]), tmp_path)["markdown"]
    sources = section_text(md, "Sources")
    assert "row 5\n(truncated)\n```" in sources and "row 6" not in sources and "row 20" not in sources
    short = "".join(f"row {i}\n" for i in range(1, 6))
    run_result["results"][2]["output"] = {"output": short}
    md = report(run_result, ScriptedChatModel(script=[writer_reply(GOOD, GOOD_SUMMARY)]), tmp_path)["markdown"]
    assert "row 5\n\n```" in md and "(truncated)" not in md             # five lines are shown in full, no marker
