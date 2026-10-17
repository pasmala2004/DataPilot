"""Tests for the agent loop (agent/graph.py, run.py).

No network and no API key: the language model is a ScriptedChatModel that replays queued
AIMessages. The six main scenarios run the REAL graph against the REAL excel_server.py over
stdio, on a tiny fixture workbook built here in a temp DATA_DIR (nothing from ./data is used).
A few edge cases use fake in-process tools so they do not need a server.

The one live test at the bottom is skipped unless OPENROUTER_API_KEY is set and works.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import Field

import run as cli
from agent.graph import (
    RouterPlan,
    RouterTask,
    clean_ai_message,
    normalise_plan,
    parse_tool_result,
    recursion_limit_for,
    run_agent,
)
from agent.llm import MissingKeyError, key_present
from agent.mcp_client import ALL_TOOL_NAMES, ANALYSIS_TOOL_NAMES, ExcelTools, excel_session
from scripted import (CANARIES, TIMEOUT, ScriptedChatModel, agg, assert_agents_only_got_their_own_tools, call, plan,
                      prompt_text, run_graph, say, trace_text)



# =========================================================================== #
# The six scenarios against the real server
# =========================================================================== #
def test_1_happy_path_value_matches_pandas_and_task_finishes(workspace, capfd):
    orders: pd.DataFrame = workspace["orders"]
    llm, result = run_graph(workspace, [
        plan(("analysis", "Sum the amount column.")),
        call("aggregate", agg(), "call_1"),
        say("The total amount was computed."),
        say("The total amount is 955.0 [call_1]."),
    ])
    assert result["error"] is None and result["clarification"] is None
    assert result["answer"] == "The total amount is 955.0 [call_1]."
    assert result["dataset_id"] == "ds1"

    (entry,) = result["results"]
    assert (entry["task_idx"], entry["tool_call_id"], entry["tool"]) == (0, "call_1", "aggregate")
    assert entry["args"] == agg()
    assert entry["output"]["value"] == pytest.approx(orders["amount"].sum())
    assert (entry["output"]["rows_used"], entry["output"]["rows_dropped_null"]) == (9, 1)

    (status,) = result["task_status"]
    assert (status["status"], status["retries"], status["reasons"]) == ("finished", 0, [])
    assert status["note"] == "The total amount was computed."
    assert llm.script == []  # every scripted reply was consumed, none missing

    text = trace_text(result)
    for expected in ("LOAD sales.xlsx [orders] -> ds1: 10 rows, 5 columns", "ROUTER -> 1 task(s)", "TOOL aggregate",
                     "-> ok", "OBSERVE -> FINISH", "DECISION -> answer written from 1 observation(s)"):
        assert expected in text, expected

    # router sees RouterPlan; the analysis agent is bound to exactly its five tools
    assert llm.bound[0] == ["RouterPlan"]
    assert_agents_only_got_their_own_tools(llm)

    # no raw rows anywhere: prompts, trace, stdout, stderr
    captured = capfd.readouterr()
    for canary in CANARIES:
        assert canary not in prompt_text(llm) and canary not in text and canary not in captured.err
    assert captured.out == ""  # no trace callback given, so nothing was printed


def test_2_wrong_column_error_then_retry_then_finish(workspace):
    orders: pd.DataFrame = workspace["orders"]
    llm, result = run_graph(workspace, [
        plan(("analysis", "Sum the amount column.")),
        call("aggregate", agg("amnt"), "call_bad"),
        say("I could not compute it."),                       # attempt 1 ends with the error unresolved
        call("aggregate", agg("amount"), "call_good"),
        say("The sum was computed."),
        say("The total is 955.0 [call_good]."),
    ])
    assert result["error"] is None
    bad, good = result["results"]
    assert "unknown column 'amnt'" in bad["output"]["error"] and bad["tool_call_id"] == "call_bad"
    assert good["output"]["value"] == pytest.approx(orders["amount"].sum())

    (status,) = result["task_status"]
    assert (status["status"], status["retries"]) == ("finished", 1)  # exactly one retry used
    assert status["self_corrections"] == 0  # the error was fixed by a retry, not by the agent itself
    assert trace_text(result).count("OBSERVE -> RETRY(") == 1
    assert "unknown column 'amnt'" in next(ln for ln in result["trace"] if "RETRY(" in ln)

    # the second attempt received the feedback as a message
    third_call = llm.seen[3]  # calls so far: router, agent, agent (closing line), then the retried agent call
    feedback = [m for m in third_call if isinstance(m, HumanMessage) and "rejected by an automatic check" in str(m.content)]
    assert len(feedback) == 1 and "unknown column 'amnt'" in feedback[0].content
    assert llm.script == []


def test_2b_an_error_the_agent_fixes_by_itself_does_not_cost_a_retry(workspace):
    llm, result = run_graph(workspace, [
        plan(("analysis", "Sum the amount column.")),
        call("aggregate", agg("amnt"), "call_bad"),
        call("aggregate", agg("amount"), "call_good"),        # fixed within the same attempt
        say("Computed."),
        say("Total [call_good]."),
    ])
    assert [r["output"].get("error") is not None for r in result["results"]] == [True, False]
    (status,) = result["task_status"]
    assert (status["status"], status["retries"]) == ("finished", 0)
    assert status["self_corrections"] == 1  # one problem fixed inside the attempt, no retry spent
    assert "RETRY" not in trace_text(result) and llm.script == []
    # the decision only saw the clean observation, not the superseded error
    decision_prompt = str(llm.seen[-1][-1].content)
    assert "call_good" in decision_prompt and "call_bad" not in decision_prompt


def test_a_router_reply_that_does_not_parse_is_repaired_once_and_the_run_continues(workspace):
    bad = AIMessage(content="", tool_calls=[{"name": "RouterPlan", "id": "r0",
                                             "args": {"tasks": [{"agent": "banana", "task": "x"}], "clarification": None}}])
    llm, result = run_graph(workspace, [
        bad,
        plan(("analysis", "Sum the amount column.")),               # the repaired plan
        call("aggregate", agg(), "call_1"), say("Summed."), say("Total [call_1]."),
    ])
    assert result["error"] is None and result["tasks"][0]["task"] == "Sum the amount column."
    assert result["task_status"][0]["status"] == "finished" and llm.script == []
    repair = llm.seen[1][-1]                                         # the second router call
    assert isinstance(repair, HumanMessage) and "could not be parsed" in repair.content


def test_a_router_that_keeps_failing_ends_the_run_cleanly_with_no_tool_calls(workspace):
    bad = AIMessage(content="", tool_calls=[{"name": "RouterPlan", "id": "r0",
                                             "args": {"tasks": [{"agent": "banana", "task": "x"}], "clarification": None}}])
    llm, result = run_graph(workspace, [bad, bad])
    assert result["answer"] is None and result["results"] == []
    assert result["error"].startswith("The router failed: StructuredOutputError")
    assert "after one repair attempt" in result["error"] and len(llm.seen) == 2   # exactly one repair, no third call


def test_self_corrections_accumulate_across_attempts_of_one_task_and_reset_for_the_next(workspace):
    corr = {"dataset_id": "ds1", "test": "correlation", "x": "qty", "y": "amount"}
    llm, result = run_graph(workspace, [
        plan(("analysis", "Sum amount and correlate qty with amount."), ("analysis", "Count the rows.")),
        call("aggregate", agg("amnt"), "a1_bad"),
        call("aggregate", agg("amount"), "a1_fixed"),                  # self-corrected inside attempt 1
        call("compare_groups", {**corr, "x": "nope"}, "a1_corr_bad"),
        say("Only part of it worked."),                                # attempt 1: the correlation error is unresolved
        call("compare_groups", corr, "a2_corr"),                       # attempt 2 (after the retry)
        say("Done."),
        call("aggregate", agg("qty", "count"), "t2_1"),
        say("Counted."),
        say("Answer."),
    ])
    first, second = result["task_status"]
    assert (first["status"], first["retries"], first["self_corrections"]) == ("finished", 1, 1)
    assert (second["status"], second["retries"], second["self_corrections"]) == ("finished", 0, 0)
    assert llm.script == []


def test_3_always_failing_call_fails_after_two_retries_and_the_next_task_still_runs(workspace):
    orders: pd.DataFrame = workspace["orders"]
    llm, result = run_graph(workspace, [
        plan(("analysis", "Sum a column that does not exist."), ("analysis", "Count the rows.")),
        call("aggregate", agg("nope"), "t1_a1"), say("failed 1"),
        call("aggregate", agg("nope"), "t1_a2"), say("failed 2"),
        call("aggregate", agg("nope"), "t1_a3"), say("failed 3"),
        call("aggregate", agg("qty", "count"), "t2_a1"), say("counted"),
        say("There are 10 orders [t2_a1]. Could not complete: task 1."),
    ], max_retries=2)
    assert result["error"] is None
    first, second = result["task_status"]
    assert (first["status"], first["retries"]) == ("failed", 2)
    assert first["self_corrections"] == 0 and second["self_corrections"] == 0  # per task, never fixed
    assert "unknown column 'nope'" in first["reasons"][0]
    assert (second["status"], second["retries"]) == ("finished", 0)

    assert [(r["task_idx"], r["tool_call_id"]) for r in result["results"]] == [
        (0, "t1_a1"), (0, "t1_a2"), (0, "t1_a3"), (1, "t2_a1")]
    assert result["results"][3]["output"]["value"] == len(orders)
    text = trace_text(result)
    assert text.count("OBSERVE -> RETRY(") == 2
    assert "OBSERVE -> FAILED(" in text and "-> NEXT TASK" in text and "OBSERVE -> FINISH" in text
    assert "TASK 2/2 [analysis]" in text

    decision_prompt = str(llm.seen[-1][-1].content)  # the decision saw the failed task and only the clean result
    assert "FAILED" in decision_prompt and "unknown column 'nope'" in decision_prompt
    assert "t2_a1" in decision_prompt and "t1_a1" not in decision_prompt
    assert result["answer"].startswith("There are 10 orders") and llm.script == []


def test_4_max_retries_zero_marks_a_bad_call_failed_immediately(workspace):
    llm, result = run_graph(workspace, [
        plan(("analysis", "Sum a column that does not exist.")),
        call("aggregate", agg("nope"), "call_1"),
        say("It failed."),
    ], max_retries=0)
    (status,) = result["task_status"]
    assert (status["status"], status["retries"]) == ("failed", 0)
    assert len(result["results"]) == 1  # no second attempt
    text = trace_text(result)
    assert "OBSERVE -> FAILED(" in text and "RETRY" not in text
    assert not any("rejected by an automatic check" in str(m.content) for msgs in llm.seen for m in msgs)
    # nothing verified -> the answer is composed in code, with no decision LLM call
    assert result["answer"].startswith("No verified results were produced")
    assert "task 1" in result["answer"] and "unknown column 'nope'" in result["answer"]
    assert "answer composed without the LLM" in text
    assert len(llm.seen) == 3 and llm.script == []  # router + agent (call) + agent (closing line)


def test_5_two_tool_calls_in_one_session_reuse_the_same_dataset_id(workspace):
    orders: pd.DataFrame = workspace["orders"]
    llm, result = run_graph(workspace, [
        plan(("analysis", "Sum amount by region, then count the orders."),
             ("chart", "Bar chart of the sum of amount by region.")),
        call("aggregate", agg(group_by=["region"]), "call_1"),
        call("aggregate", agg("qty", "count"), "call_2"),          # second call, same session, same id
        say("Computed both."),
        call("make_chart", {"dataset_id": "ds1", "kind": "bar", "x": "region", "y": "amount"}, "call_3"),
        say("Chart saved."),
        say("Done [call_1] [call_2] [call_3]."),
    ])
    assert result["error"] is None and result["dataset_id"] == "ds1"
    assert [r["args"]["dataset_id"] for r in result["results"]] == ["ds1", "ds1", "ds1"]
    assert all("error" not in r["output"] for r in result["results"])

    expected = orders.dropna(subset=["amount"]).groupby("region")["amount"].sum()
    got = {g["region"]: g["value"] for g in result["results"][0]["output"]["groups"]}
    assert got == pytest.approx(expected.to_dict())
    assert result["results"][1]["output"]["value"] == len(orders)

    png = Path(result["results"][2]["output"]["path"])
    assert png.is_file() and png.read_bytes().startswith(b"\x89PNG")
    assert png.parent == (workspace["out_dir"] / "charts").resolve()

    assert [s["status"] for s in result["task_status"]] == ["finished", "finished"]
    assert_agents_only_got_their_own_tools(llm)
    assert ["make_chart"] in llm.bound  # the chart agent was bound, and only to make_chart
    assert llm.script == []


def test_6_vague_request_returns_a_clarification_and_calls_no_tools(workspace):
    llm, result = run_graph(workspace, [plan(clarification="Which metric do you want to look at?")])
    assert result["clarification"] == "Which metric do you want to look at?"
    assert result["answer"] is None and result["error"] is None
    assert result["tasks"] == [] and result["task_status"] == [] and result["results"] == []
    text = trace_text(result)
    assert "clarification needed" in text and "TOOL" not in text and "TASK" not in text
    assert llm.bound == [["RouterPlan"]] and len(llm.seen) == 1  # only the router was ever called
    assert llm.script == []


# =========================================================================== #
# More behaviour against the real server
# =========================================================================== #
def test_tool_iteration_cap_ends_the_attempt(workspace):
    llm, result = run_graph(workspace, [
        plan(("analysis", "Keep computing.")),
        call("aggregate", agg(), "c1"), call("aggregate", agg(), "c2"), call("aggregate", agg(), "c3"),
        call("aggregate", agg(), "c4"),        # a 4th round is requested but the cap is 3: never executed
        say("Decision."),
    ], max_tool_iterations=3)
    assert [r["tool_call_id"] for r in result["results"]] == ["c1", "c2", "c3"]
    assert "tool iteration limit (3) reached" in trace_text(result)
    assert result["task_status"][0]["status"] == "finished"  # what was collected is clean evidence
    assert llm.script == []


def test_load_problems_end_the_run_with_a_clear_message_and_no_llm_call(workspace):
    async def go() -> list[tuple[ScriptedChatModel, dict]]:
        out = []
        async with excel_session(workspace["data_dir"], workspace["out_dir"]) as tools:
            for file, sheet in [("missing.xlsx", None), ("sales.xlsx", "nope"), ("../sales.xlsx", None), ("notes.txt", None)]:
                (workspace["data_dir"] / "notes.txt").write_text("hello")
                llm = ScriptedChatModel()  # empty script: any LLM call would raise
                out.append((llm, await run_agent("q", file, llm=llm, tools=tools, sheet=sheet)))
        return out

    runs = asyncio.run(asyncio.wait_for(go(), TIMEOUT))
    (_, missing), (_, bad_sheet), (_, traversal), (_, wrong_type) = runs
    assert "Cannot open 'missing.xlsx'" in missing["error"] and "file not found" in missing["error"]
    assert "Cannot load 'sales.xlsx'" in bad_sheet["error"] and "orders" in bad_sheet["error"] and "targets" in bad_sheet["error"]
    assert "path outside allowed directory" in traversal["error"]
    assert "unsupported file type" in wrong_type["error"]
    for llm, result in runs:
        assert result["answer"] is None and result["results"] == [] and llm.seen == []
        assert result["trace"][-1].startswith("ERROR ")


def test_without_a_sheet_the_first_sheet_is_used_and_the_trace_says_so(workspace):
    llm, result = run_graph(workspace, [plan(clarification="Which metric?")], sheet=None)
    assert result["sheet"] == "orders"
    assert "using the first sheet 'orders' (others: ['targets'])" in trace_text(result)


def test_the_schema_in_state_has_no_preview_rows(workspace):
    llm, result = run_graph(workspace, [plan(clarification="Which metric?")])
    assert set(result["schema"]) == {"rows", "columns", "nulls", "date_ranges"}
    assert result["schema"]["rows"] == 10 and result["schema"]["nulls"] == {"amount": 1}
    assert result["schema"]["date_ranges"]["order_date"][0].startswith("2024-01-05")
    assert "preview" not in json.dumps(result["schema"])
    router_prompt = prompt_text(llm)
    assert "order_date: datetime64" in router_prompt and "amount=1" in router_prompt
    assert not any(c in router_prompt for c in CANARIES)



# =========================================================================== #
# CLI end to end (scripted model, real server)
# =========================================================================== #
def test_cli_runs_end_to_end_and_prints_trace_then_answer(workspace, monkeypatch, capsys):
    monkeypatch.setenv("DATA_DIR", str(workspace["data_dir"]))
    monkeypatch.setenv("OUTPUT_DIR", str(workspace["out_dir"]))
    llm = ScriptedChatModel(script=[
        plan(("analysis", "Sum amount."), ("chart", "Bar chart of qty by region.")),
        call("aggregate", agg(), "call_1"), say("Summed."),
        call("make_chart", {"dataset_id": "ds1", "kind": "bar", "x": "region", "y": "qty"}, "call_2"), say("Chart saved."),
        say("Total amount is 955.0 [call_1]."),
    ])
    code = cli.main(["--file", "sales.xlsx", "--sheet", "orders", "total amount and a chart", "--max-retries", "1"],
                    llm_factory=lambda **kw: llm)
    out = capsys.readouterr().out
    assert code == 0
    assert out.index("LOAD") < out.index("ROUTER") < out.index("TOOL aggregate") < out.index("OBSERVE") < out.index("ANSWER")
    assert "Total amount is 955.0 [call_1]." in out
    assert "1. [analysis] finished" in out and "2. [chart] finished" in out and "chart: " in out
    assert not any(c in out for c in CANARIES)
    assert list((workspace["out_dir"] / "charts").glob("*.png"))


def test_cli_without_an_api_key_is_a_clear_error_not_a_crash(capsys):
    def no_key(**kw) -> BaseChatModel:
        raise MissingKeyError("OPENROUTER_API_KEY is not set.")

    assert cli.main(["--file", "x.xlsx", "question"], llm_factory=no_key) == 2
    assert "OPENROUTER_API_KEY is not set" in capsys.readouterr().err


def test_cli_returns_1_and_prints_the_error_when_the_run_fails(workspace, monkeypatch, capsys):
    monkeypatch.setenv("DATA_DIR", str(workspace["data_dir"]))
    monkeypatch.setenv("OUTPUT_DIR", str(workspace["out_dir"]))
    assert cli.main(["--file", "missing.xlsx", "q"], llm_factory=lambda **kw: ScriptedChatModel()) == 1
    assert "ERROR: Cannot open 'missing.xlsx'" in capsys.readouterr().out


def test_cli_argument_parsing():
    parser = cli.build_parser()
    args = parser.parse_args(["--file", "sales.xlsx", "--sheet", "orders", "my question", "--max-retries", "0"])
    assert (args.file, args.sheet, args.question, args.max_retries) == ("sales.xlsx", "orders", "my question", 0)
    assert parser.parse_args(["--file", "a.csv", "q"]).max_retries == 2 and parser.parse_args(["--file", "a.csv", "q"]).sheet is None
    for bad in (["q"], ["--file", "a.xlsx"], ["--file", "a.xlsx", "q", "--max-retries", "-1"],
                ["--file", "a.xlsx", "q", "--max-retries", "x"]):
        with pytest.raises(SystemExit):
            parser.parse_args(bad)


# =========================================================================== #
# Fast edge cases with fake in-process tools (no server)
# =========================================================================== #
def fake_tools(handlers: dict[str, Any] | None = None) -> tuple[ExcelTools, list[tuple[str, dict]]]:
    """All six tools as in-process fakes; ``handlers[name](args)`` returns the tool's dict."""
    calls: list[tuple[str, dict]] = []
    handlers = handlers or {}

    def make(name: str) -> StructuredTool:
        async def coroutine(**kwargs: Any) -> str:
            calls.append((name, kwargs))
            default = {"sheets": [{"name": "S", "rows": 3, "columns": 2}]} if name == "list_sheets" else {
                "dataset_id": "ds1", "rows": 3, "columns": {"a": "int64"}, "nulls": {}, "date_ranges": {},
                "preview": [{"a": 1}]} if name == "load_data" else {"value": 1.0, "rows_used": 3, "rows_dropped_null": 0}
            return json.dumps(handlers.get(name, lambda a: default)(kwargs))

        return StructuredTool(name=name, description=name, coroutine=coroutine,
                              args_schema={"type": "object", "properties": {}, "additionalProperties": True})

    return ExcelTools(by_name={name: make(name) for name in ALL_TOOL_NAMES}), calls


def run_fake(script: list[AIMessage], tools: ExcelTools, **kwargs: Any) -> tuple[ScriptedChatModel, dict]:
    llm = ScriptedChatModel(script=list(script))
    return llm, asyncio.run(run_agent("q", "f.xlsx", llm=llm, tools=tools, **kwargs))


def test_a_chart_path_that_is_not_on_disk_is_rejected_by_observe():
    tools, _ = fake_tools({"make_chart": lambda a: {"path": "/nonexistent/chart.png", "kind": "bar"}})
    llm, result = run_fake([
        plan(("chart", "Draw a chart.")),
        call("make_chart", {"dataset_id": "ds1", "kind": "bar", "x": "a"}, "c1"), say("saved"),
        call("make_chart", {"dataset_id": "ds1", "kind": "bar", "x": "a"}, "c2"), say("saved"),
    ], tools, max_retries=1)
    (status,) = result["task_status"]
    assert (status["status"], status["retries"]) == ("failed", 1)
    assert "the file does not exist" in status["reasons"][0]
    assert llm.script == []


def test_a_tool_outside_the_agents_toolset_is_an_error_result_not_a_crash():
    tools, calls = fake_tools()
    llm, result = run_fake([
        plan(("chart", "Draw a chart.")),
        call("aggregate", agg(), "c1"), say("x"),            # the chart agent may NOT use aggregate
        call("make_chart", {"dataset_id": "ds1", "kind": "bar", "x": "a"}, "c2"), say("y"),
    ], tools, max_retries=1)
    assert "unknown tool 'aggregate'" in result["results"][0]["output"]["error"]
    assert [name for name, _ in calls if name == "aggregate"] == []  # it was never executed
    assert result["task_status"][0]["retries"] == 1


def test_load_runs_list_sheets_then_load_data_once_and_the_agent_never_reloads():
    tools, calls = fake_tools()
    run_fake([plan(clarification="?")], tools)
    assert [name for name, _ in calls] == ["list_sheets", "load_data"]
    assert calls[1][1] == {"path": "f.xlsx", "sheet": "S"}


def test_tool_call_ids_are_made_unique_and_valid():
    tools, _ = fake_tools()
    llm, result = run_fake([
        plan(("analysis", "Two calls with the same id, then an empty id.")),
        AIMessage(content="", tool_calls=[{"name": "aggregate", "args": agg(), "id": "dup"},
                                          {"name": "aggregate", "args": agg("qty"), "id": "dup"}]),
        AIMessage(content="", tool_calls=[{"name": "aggregate", "args": agg(), "id": ""}]),
        say("done"), say("answer"),
    ], tools)
    ids = [r["tool_call_id"] for r in result["results"]]
    assert len(ids) == 3 and len(set(ids)) == 3 and ids[0] == "dup" and all(ids)
    assert all(len(i) == 9 and i.isalnum() for i in ids[1:])
    # every tool message answers a tool call id that exists in the AI messages before it
    messages = llm.seen[-2]  # the agent's last call; llm.seen[-1] is the decision call with its own prompt
    call_ids = {c["id"] for m in messages if isinstance(m, AIMessage) for c in m.tool_calls}
    assert {m.tool_call_id for m in messages if isinstance(m, ToolMessage)} == call_ids == set(ids)


def test_a_failing_tool_transport_becomes_an_error_result():
    async def explode(args: dict) -> dict:
        raise ConnectionError("server went away")

    tools, _ = fake_tools({"aggregate": lambda a: (_ for _ in ()).throw(ConnectionError("server went away"))})
    llm, result = run_fake([plan(("analysis", "x")), call("aggregate", agg(), "c1"), say("x")], tools, max_retries=0)
    assert "ConnectionError" in result["results"][0]["output"]["error"] and "server went away" in result["results"][0]["output"]["error"]
    assert result["task_status"][0]["status"] == "failed"


def test_router_failure_ends_the_run_with_a_message():
    tools, _ = fake_tools()
    llm = ScriptedChatModel(script=[say("this is not a RouterPlan tool call")])
    result = asyncio.run(run_agent("q", "f.xlsx", llm=llm, tools=tools))
    assert result["error"] and "router" in result["error"].lower() and result["answer"] is None


def test_an_llm_api_failure_is_reported_in_the_result_not_raised():
    tools, _ = fake_tools()
    result = asyncio.run(run_agent("q", "f.xlsx", llm=ScriptedChatModel(), tools=tools))  # empty script raises inside
    assert result["error"] and "AssertionError" in result["error"] or "router" in result["error"].lower()
    assert result["trace"][-1].startswith("ERROR ")


def test_negative_max_retries_is_rejected():
    tools, _ = fake_tools()
    with pytest.raises(ValueError):
        asyncio.run(run_agent("q", "f.xlsx", llm=ScriptedChatModel(), tools=tools, max_retries=-1))


def test_the_trace_callback_receives_every_line_and_never_tool_output():
    tools, _ = fake_tools({"aggregate": lambda a: {"value": 424242.0, "rows_used": 3, "rows_dropped_null": 0}})
    seen: list[str] = []
    llm = ScriptedChatModel(script=[plan(("analysis", "Sum.")), call("aggregate", agg(), "c1"), say("ok"), say("fin")])
    result = asyncio.run(run_agent("q", "f.xlsx", llm=llm, tools=tools, trace=seen.append))
    assert seen == result["trace"] and seen
    assert not any("424242" in line for line in seen)  # outputs are never traced, only name, args and ok/error


def test_long_arguments_are_shortened_in_the_trace():
    tools, _ = fake_tools()
    long_code = "df['x'] = 1  # " + "y" * 1000
    llm = ScriptedChatModel(script=[plan(("analysis", "x")), call("run_pandas", {"dataset_id": "ds1", "code": long_code}, "c1"),
                                    say("ok"), say("fin")])
    result = asyncio.run(run_agent("q", "f.xlsx", llm=llm, tools=tools))
    line = next(ln for ln in result["trace"] if ln.startswith("TOOL run_pandas"))
    assert len(line) < 400 and "..." in line


# =========================================================================== #
# Pure helpers
# =========================================================================== #
def test_normalise_plan_puts_chart_tasks_last_keeps_order_and_caps_at_four():
    p = RouterPlan(tasks=[RouterTask(agent="chart", task="c1"), RouterTask(agent="analysis", task="a1"),
                          RouterTask(agent="chart", task="c2"), RouterTask(agent="analysis", task="a2"),
                          RouterTask(agent="analysis", task="a3")])
    tasks, clarification = normalise_plan(p)
    assert clarification is None
    assert [t["task"] for t in tasks] == ["a1", "a2", "a3", "c1"]  # chart last, then truncated to 4


def test_normalise_plan_clarification_wins_and_empty_plans_ask_for_clarification():
    tasks, clarification = normalise_plan(RouterPlan(tasks=[RouterTask(agent="analysis", task="a")], clarification="Which?"))
    assert tasks == [] and clarification == "Which?"
    tasks, clarification = normalise_plan(RouterPlan(tasks=[], clarification="   "))
    assert tasks == [] and "rephrase" in clarification
    tasks, clarification = normalise_plan(RouterPlan(tasks=[RouterTask(agent="analysis", task="  ")]))
    assert tasks == [] and clarification


def test_parse_tool_result_handles_mcp_content_blocks_plain_errors_and_junk():
    blocks = ToolMessage(content=[{"type": "text", "text": '{"value": 5.0}'}], tool_call_id="x")
    assert parse_tool_result(blocks) == {"value": 5.0}
    assert parse_tool_result(ToolMessage(content='{"error": "boom"}', tool_call_id="x")) == {"error": "boom"}
    plain = ToolMessage(content="Error executing tool aggregate: 1 validation error", tool_call_id="x", status="error")
    assert parse_tool_result(plain)["error"].startswith("Error executing tool")
    assert "error" in parse_tool_result(ToolMessage(content="not json", tool_call_id="x"))
    assert "error" in parse_tool_result(ToolMessage(content="[1, 2]", tool_call_id="x"))
    assert "error" in parse_tool_result(ToolMessage(content="", tool_call_id="x"))
    status_error = ToolMessage(content='{"ok": false}', tool_call_id="x", status="error")
    assert "error" in parse_tool_result(status_error)


def test_clean_ai_message_drops_malformed_calls_and_provider_extras():
    raw = AIMessage(
        content=[{"type": "text", "text": "thinking"}],
        tool_calls=[{"name": "aggregate", "args": {"a": 1}, "id": "ok1", "type": "tool_call"}],
        invalid_tool_calls=[{"name": "aggregate", "args": "{bad json", "id": "bad1", "error": "x", "type": "invalid_tool_call"}],
        additional_kwargs={"tool_calls": [{"id": "bad1"}], "reasoning": "secret"},
    )
    cleaned = clean_ai_message(raw, set())
    assert cleaned.content == "thinking" and [c["id"] for c in cleaned.tool_calls] == ["ok1"]
    assert cleaned.invalid_tool_calls == [] and cleaned.additional_kwargs == {}
    # an id already used by an earlier call is replaced
    assert clean_ai_message(raw, {"ok1"}).tool_calls[0]["id"] != "ok1"


def test_recursion_limit_grows_with_retries_and_covers_the_worst_case():
    assert recursion_limit_for(0, 6) < recursion_limit_for(2, 6) < recursion_limit_for(5, 6)
    assert recursion_limit_for(2, 6) > 25  # LangGraph's default would stop long runs early
    worst = 2 + 4 * (1 + 3 * (2 * 6 + 2)) + 1  # load, router, 4 tasks x 3 attempts, decision
    assert recursion_limit_for(2, 6) > worst


# =========================================================================== #
# Live test (needs a key; skipped otherwise)
# =========================================================================== #
@pytest.mark.skipif(not key_present(), reason="live test: set OPENROUTER_API_KEY to run it")
def test_live_two_task_question_matches_pandas(workspace):
    from agent.llm import get_llm

    orders: pd.DataFrame = workspace["orders"]
    question = ("What is the sum of the amount column for each region? "
                "Also draw a bar chart of the sum of amount by region.")

    async def go() -> dict:
        async with excel_session(workspace["data_dir"], workspace["out_dir"]) as tools:
            return await run_agent(question, "sales.xlsx", llm=get_llm(), tools=tools, sheet="orders")

    result = asyncio.run(asyncio.wait_for(go(), 300))
    assert result["error"] is None, result["error"]
    assert result["answer"]

    expected = orders.dropna(subset=["amount"]).groupby("region")["amount"].sum().to_dict()
    grouped = [
        {g["region"]: g["value"] for g in r["output"]["groups"]}
        for r in result["results"]
        if r["tool"] == "aggregate" and "groups" in r["output"] and "error" not in r["output"]
    ]
    assert any(g == pytest.approx(expected) for g in grouped), grouped  # the number in the log equals pandas
    charts = [r["output"]["path"] for r in result["results"] if r["tool"] == "make_chart" and r["output"].get("path")]
    assert charts and all(Path(p).is_file() for p in charts)
