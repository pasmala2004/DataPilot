"""End to end: real excel_server over stdio + scripted LLMs -> agent run -> Markdown report.

No network, no API key. Every number in the finished report (Summary, Key figures, Analysis) must equal a
value computed here with pandas/scipy and formatted by the documented rules, never taken from the report code.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from scipy import stats

import run as cli
from agent.llm import key_present
from agent.report import build_report
from scripted import ScriptedChatModel, agg, call, plan, run_graph, say

NOW = datetime(2026, 10, 11, 14, 30, 5)
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?|<0\.001")


def section(markdown: str, title: str) -> str:
    """Text of the '## <title>' section (up to the next '## ')."""
    match = re.search(rf"^## {re.escape(title)}\s*$(.*?)(?=^## |\Z)", markdown, re.S | re.M)
    assert match, f"no section {title!r} in:\n{markdown}"
    return match.group(1)


def pandas_expected(orders: pd.DataFrame) -> dict[str, str]:
    """Every figure the scenario can show, computed with pandas/scipy and formatted by the documented rules."""
    clean = orders.dropna(subset=["amount"])
    sums = clean.groupby("region")["amount"].sum()
    both = orders[["qty", "amount"]].dropna()
    r, p = stats.pearsonr(both["qty"], both["amount"])
    return {
        "north": f"{sums['North']:,.2f}", "south": f"{sums['South']:,.2f}", "east": f"{sums['East']:,.2f}",
        "rows_used": f"{len(clean):,}", "dropped": f"{int(orders['amount'].isna().sum()):,}",
        "n": f"{len(both):,}", "r": f"{r:.3f}", "p": "<0.001" if p < 0.001 else f"{p:#.4g}",
    }


def fact_id(facts: list[dict], tool: str, *needles: str) -> str:
    (found,) = [f for f in facts if f["tool"] == tool and all(n in f["label"] for n in needles)]
    return "{{" + found["fact_id"] + "}}"


@pytest.fixture
def scenario(workspace):
    """Run the agent (4 tasks: groups / a failing one / correlation + run_pandas / chart), return the run result."""
    llm, result = run_graph(workspace, [
        plan(("analysis", "Sum amount by region."), ("analysis", "Sum a column that does not exist."),
             ("analysis", "Correlate qty with amount."), ("chart", "Bar chart of amount by region.")),
        call("aggregate", agg(group_by=["region"]), "call_groups"), say("Summed."),
        call("aggregate", agg("nope"), "call_bad"), say("That failed."),
        call("compare_groups", {"dataset_id": "ds1", "test": "correlation", "x": "qty", "y": "amount"}, "call_corr"),
        call("run_pandas", {"dataset_id": "ds1", "code": "print(df['qty'].max())"}, "call_pandas"),
        say("Correlated."),
        call("make_chart", {"dataset_id": "ds1", "kind": "bar", "x": "region", "y": "amount"}, "call_chart"),
        say("Chart saved."),
        say("Sales were summed by region [call_groups]."),
    ], max_retries=0)
    assert result["error"] is None
    assert [s["status"] for s in result["task_status"]] == ["finished", "failed", "finished", "finished"]
    return workspace, result


def scripted_writer(result: dict, *, stray_in_round_one: bool) -> ScriptedChatModel:
    from agent.facts import build_facts
    from langchain_core.messages import AIMessage

    facts = build_facts(result["results"], result["task_status"])
    north, south = fact_id(facts, "aggregate", "region=North"), fact_id(facts, "aggregate", "region=South")
    east = fact_id(facts, "aggregate", "region=East")
    r, p = fact_id(facts, "compare_groups", ": r"), fact_id(facts, "compare_groups", "p-value")
    n = fact_id(facts, "compare_groups", ": n")
    good = {
        0: f"North brings in {north}, South {south} and East {east}.",
        2: f"Across {n} orders the correlation r is {r} with p {p}. The largest quantity came from run_pandas.",
        3: "The chart shows the amount by region.",
    }
    summary = f"North is {north} and the correlation is {r}."

    def reply(sections: dict[int, str], text: str, call_id: str) -> AIMessage:
        args = {"sections": [{"task_idx": i, "text": t} for i, t in sections.items()], "summary": text}
        return AIMessage(content="", tool_calls=[{"name": "WriterOutput", "args": args, "id": call_id}])

    script = []
    if stray_in_round_one:
        script.append(reply({**good, 0: "North brings in 45,650 and South " + south + "."}, summary, "w1"))
    script.append(reply(good, summary, "w2"))
    return ScriptedChatModel(script=script)


def test_every_number_in_the_report_comes_from_the_results_and_matches_pandas(scenario):
    workspace, result = scenario
    writer = scripted_writer(result, stray_in_round_one=True)
    report = asyncio.run(build_report(result, writer, output_dir=workspace["out_dir"], now=NOW))
    md = report["markdown"]
    expected = pandas_expected(workspace["orders"])

    assert writer.script == [] and report["rounds"] == 2 and report["fallbacks"] == []   # round 1 caught, round 2 clean
    assert "45,650" not in md                                                              # the stray number never ships
    assert "{{" not in md

    # 1. every number in Summary, Key figures and Analysis is a pandas value
    shown = set(expected.values())
    for title in ("Summary", "Key figures", "Analysis"):
        found = NUMBER.findall(re.sub(r"\bT?F\d+\b|Task \d+", "", section(md, title)))
        stray = [x for x in found if x.rstrip(".,") not in shown]
        assert not stray, f"numbers not computed by pandas in {title}: {stray}"

    # 2. the key figures are the actual values (not just any allowed ones)
    key = section(md, "Key figures")
    for name in ("north", "south", "east", "r", "p"):
        assert expected[name] in key, name
    assert all(expected[k] in section(md, "Analysis") for k in ("north", "south", "east", "r", "p", "n"))

    # 3. the failed task is in Caveats, with its reason
    caveats = section(md, "Caveats")
    assert "Task 2 failed" in caveats and "unknown column 'nope'" in caveats and "Sum a column that does not exist" in caveats
    assert "computed by tools" in caveats

    # 4. the chart link points to an existing file (relative to the report)
    links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", section(md, "Charts"))
    assert len(links) == 1
    assert (Path(report["path"]).parent / links[0]).resolve().is_file()
    assert Path(report["path"]).parent == workspace["out_dir"] / "reports"
    assert Path(report["path"]).name == "report_20261011_143005.md"
    assert Path(report["path"]).read_text(encoding="utf-8") == md

    # 5. Sources lists every tool_call_id that produced something shown (and not the failed call)
    sources = section(md, "Sources")
    for call_id in ("call_groups", "call_corr", "call_pandas", "call_chart"):
        assert call_id in sources, call_id
    assert "call_bad" not in sources
    used = {f["tool_call_id"] for f in report["facts"]}
    assert used == {"call_groups", "call_corr"} and all(u in sources for u in used)


def test_a_writer_that_always_puts_a_stray_number_is_replaced_by_code_text_and_flagged(scenario):
    workspace, result = scenario
    from agent.facts import build_facts
    from langchain_core.messages import AIMessage

    facts = build_facts(result["results"], result["task_status"])
    bad_args = {"sections": [{"task_idx": i, "text": "We sold 45,650 units."} for i in (0, 2, 3)],
                "summary": "Revenue was 45,650."}
    bad = AIMessage(content="", tool_calls=[{"name": "WriterOutput", "args": bad_args, "id": "w"}])
    writer = ScriptedChatModel(script=[bad, bad])
    report = asyncio.run(build_report(result, writer, output_dir=workspace["out_dir"], now=NOW))
    assert "45,650" not in report["markdown"]
    expected = pandas_expected(workspace["orders"])
    assert expected["north"] in section(report["markdown"], "Analysis")                 # code-built from the facts
    caveats = section(report["markdown"], "Caveats")
    assert "section 1: generated without the LLM (number check failed)" in caveats
    assert facts  # sanity


def test_no_clean_results_gives_a_code_only_report_without_calling_the_llm(workspace):
    _, result = run_graph(workspace, [
        plan(("analysis", "Sum a column that does not exist.")),
        call("aggregate", agg("nope"), "call_bad"), say("failed"),
    ], max_retries=0)
    llm = ScriptedChatModel(script=[])
    report = asyncio.run(build_report(result, llm, output_dir=workspace["out_dir"], now=NOW))
    md = report["markdown"]
    assert llm.seen == [] and report["facts"] == []
    assert "nothing could be verified" in md.lower() or "no verified" in md.lower()
    assert "Sum a column that does not exist" in section(md, "Caveats")
    assert Path(report["path"]).is_file()


def test_cli_report_flag_prints_the_path_after_the_answer(workspace, monkeypatch, capsys):
    monkeypatch.setenv("DATA_DIR", str(workspace["data_dir"]))
    monkeypatch.setenv("OUTPUT_DIR", str(workspace["out_dir"]))
    from agent.facts import build_facts  # noqa: F401
    from langchain_core.messages import AIMessage

    run_script = [
        plan(("analysis", "Sum amount.")),
        call("aggregate", agg(), "call_1"), say("Summed."),
        say("The total amount was computed [call_1]."),
    ]
    sections = [{"task_idx": 0, "text": "The total amount is {{F1}}."}]
    writer_msg = AIMessage(content="", tool_calls=[{"name": "WriterOutput", "id": "w", "args": {
        "sections": sections, "summary": "Total amount is {{F1}}."}}])
    llm = ScriptedChatModel(script=run_script + [writer_msg])
    code = cli.main(["--file", "sales.xlsx", "--sheet", "orders", "total amount", "--report"], llm_factory=lambda **kw: llm)
    out = capsys.readouterr().out
    assert code == 0 and out.index("ANSWER") < out.index("REPORT")
    path = Path(out.split("REPORT")[1].strip().splitlines()[0].strip())
    assert path.is_file() and f"{workspace['orders']['amount'].sum():,.2f}" in path.read_text(encoding="utf-8")


def test_cli_without_the_report_flag_writes_no_report(workspace, monkeypatch, capsys):
    monkeypatch.setenv("DATA_DIR", str(workspace["data_dir"]))
    monkeypatch.setenv("OUTPUT_DIR", str(workspace["out_dir"]))
    llm = ScriptedChatModel(script=[
        plan(("analysis", "Sum amount.")), call("aggregate", agg(), "call_1"), say("Summed."), say("Computed [call_1]."),
    ])
    assert cli.main(["--file", "sales.xlsx", "--sheet", "orders", "total amount"], llm_factory=lambda **kw: llm) == 0
    assert "REPORT" not in capsys.readouterr().out
    assert not (workspace["out_dir"] / "reports").exists()


@pytest.mark.skipif(not key_present(), reason="live test: set OPENROUTER_API_KEY to run it")
def test_live_report_has_no_numbers_outside_the_results(workspace, monkeypatch):
    from agent.llm import get_llm
    from agent.mcp_client import excel_session
    from agent.graph import run_agent

    llm = get_llm()

    async def go():
        async with excel_session(workspace["data_dir"], workspace["out_dir"]) as tools:
            result = await run_agent("Total amount by region", "sales.xlsx", llm=llm, tools=tools, sheet="orders")
        return result, await build_report(result, llm, output_dir=workspace["out_dir"])

    result, report = asyncio.run(asyncio.wait_for(go(), 600))
    assert result["error"] is None
    assert Path(report["path"]).is_file()
    assert "{{" not in report["markdown"]
