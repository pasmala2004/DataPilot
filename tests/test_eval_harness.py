"""eval/run_eval.py with a tiny fixture dataset, a scripted LLM and the REAL MCP server. No network, no key, no eval/data."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult

from eval import run_eval
from eval.questions import Expect, Question
from eval.run_eval import RunKey, plan_runs, read_runs
from scripted import ScriptedChatModel, call, plan, say

DATA = pd.DataFrame({"region": ["a", "a", "b"], "amount": [10.0, 20.0, 30.0]})


class EvalLLM(ScriptedChatModel):
    """Scripted model that swaps the placeholder dataset id "DS" for the id in the agent's system prompt."""

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        result = super()._generate(messages, stop, run_manager, **kwargs)
        text = " ".join(str(m.content) for m in messages)
        found = re.search(r'dataset_id: "(ds\d+)"', text)
        for gen in result.generations:
            for tc in getattr(gen.message, "tool_calls", []) or []:
                if tc["args"].get("dataset_id") == "DS" and found:
                    tc["args"]["dataset_id"] = found.group(1)
        return result


def good(tool_args: dict[str, Any], call_id: str, sentence: str = "The total is {{F1}}.") -> list[AIMessage]:
    writer = AIMessage(content="", tool_calls=[{"name": "WriterOutput", "id": "w", "args": {
        "sections": [{"task_idx": 0, "text": sentence}], "summary": sentence}}])
    return [plan(("analysis", "Compute it.")), call("aggregate", {"dataset_id": "DS", **tool_args}, call_id),
            say("Done."), say(f"Result [{call_id}]."), writer]


def bad_first(tool_args: dict[str, Any], call_id: str) -> list[AIMessage]:
    """A wrong column first, and the attempt ENDS there: only a retry can fix it."""
    return [plan(("analysis", "Compute it.")), call("aggregate", {"dataset_id": "DS", "column": "nope", "agg": "sum"}, "bad"),
            say("It failed.")]


SUM = {"column": "amount", "agg": "sum"}
MEAN = {"column": "amount", "agg": "mean"}
Q1 = Question("t1_sum", "tiny.csv", "csv", "Total amount?", "scalar", lambda df: [Expect("sum", float(df["amount"].sum()))])
Q2 = Question("t2_mean", "tiny.csv", "csv", "Average amount?", "scalar", lambda df: [Expect("mean", float(df["amount"].mean()))])
QS = (Q1, Q2)


class Harness:
    """Factory + bookkeeping: builds the script for the run that is about to start (resume-aware)."""

    def __init__(self, runs_path: Path, repeats: int = 2, settings=(0, 2)) -> None:
        self.path, self.calls = runs_path, 0
        self.plan = plan_runs(QS, list(settings), repeats)

    def __call__(self, model: str | None = None) -> EvalLLM:
        self.calls += 1
        key = self.plan[len(read_runs(self.path))]
        if key.question_id == "t1_sum":
            # first repeat, every setting: bad call first; the second repeat is clean
            script = (good(SUM, "c1") if key.repeat == 2 else
                      bad_first(SUM, "c1") if key.setting == 0 else bad_first(SUM, "c1") + good(SUM, "c2")[1:])
        else:
            script = good(MEAN, "c1")
        return EvalLLM(script=script)


@pytest.fixture
def env(tmp_path: Path) -> dict[str, Any]:
    data = tmp_path / "data"
    data.mkdir()
    DATA.to_csv(data / "tiny.csv", index=False)
    results = tmp_path / "results"
    runs = results / "r1" / "runs.jsonl"
    return {"argv": ["--name", "r1", "--model", "m", "--repeats", "2", "--settings", "0", "2",
                     "--data-dir", str(data), "--results-dir", str(results), "--timeout-s", "60"],
            "runs": runs, "results": results, "factory": Harness(runs)}


def go(env: dict[str, Any], *extra: str) -> int:
    return run_eval.main(env["argv"] + list(extra), llm_factory=env["factory"], questions=QS)


def test_plan_interleaves_settings_and_counts_runs():
    keys = plan_runs(QS, [0, 2], 2)
    assert len(keys) == 8
    assert [k.as_tuple() for k in keys[:4]] == [("t1_sum", 0, 1), ("t1_sum", 2, 1), ("t1_sum", 0, 2), ("t1_sum", 2, 2)]


def test_full_run_retries_off_fails_and_retries_on_passes(env, capsys):
    assert go(env) == 0
    records = read_runs(env["runs"])
    assert len(records) == 8 and len({(r["question_id"], r["setting"], r["repeat"]) for r in records}) == 8
    by = {(r["question_id"], r["setting"], r["repeat"]): r for r in records}
    off, on = by[("t1_sum", 0, 1)], by[("t1_sum", 2, 1)]
    assert not off["tool_correct"] and not off["completed"] is False and off["failure_reason"] == "tool_error"
    assert off["failed_tasks"] == 1 and off["retries"] == 0
    assert on["tool_correct"] and on["report_correct"] and on["failure_reason"] is None and on["retries"] == 1
    assert all(r["tool_correct"] and r["report_correct"] for r in records if r["question_id"] == "t2_mean")
    assert all(r["tool_correct"] for r in records if r["question_id"] == "t1_sum" and r["repeat"] == 2)
    assert all(not r["fabricated"] and r["latency_s"] > 0 for r in records)
    out = capsys.readouterr().out
    assert "Time per run:" in out and "ETA" in out and "[8/8]" in out


def test_records_hold_no_data_rows(env):
    go(env)
    text = env["runs"].read_text(encoding="utf-8")
    assert "region" not in text and '"a"' not in text


def test_meta_json(env):
    go(env, "--limit", "1")
    meta = json.loads((env["results"] / "r1" / "meta.json").read_text())
    assert meta["provider"] == "openrouter" and meta["model"] == "m" and meta["settings"] == [0, 2] and meta["repeats"] == 2
    assert {"git_commit", "date", "concurrency"} <= set(meta) and "OLLAMA_NUM_CTX" not in meta


def test_meta_and_records_never_contain_the_key(env, monkeypatch):
    key = "sk-or-v1-EVALKEY0123456789"
    monkeypatch.setenv("OPENROUTER_API_KEY", key)
    go(env, "--limit", "2")
    for path in (env["results"] / "r1").rglob("*"):
        if path.is_file() and path.suffix in (".json", ".jsonl"):
            assert key not in path.read_text(encoding="utf-8")


def test_resume_after_a_kill_has_no_duplicates_and_skips_finished_runs(env):
    assert go(env, "--limit", "3") == 0
    assert len(read_runs(env["runs"])) == 3
    with env["runs"].open("a", encoding="utf-8") as fh:             # a process killed in the middle of a write
        fh.write('{"question_id": "t1_sum", "setti')
    before = env["factory"].calls
    assert go(env) == 0
    records = read_runs(env["runs"])
    assert len(records) == 8 and len({(r["question_id"], r["setting"], r["repeat"]) for r in records}) == 8
    assert env["factory"].calls - before == 5                       # only the 5 missing runs started
    before = env["factory"].calls
    assert go(env) == 0 and env["factory"].calls == before          # nothing left to do


def test_only_and_limit(env):
    assert go(env, "--only", "t2_mean", "--limit", "2") == 0
    records = read_runs(env["runs"])
    assert len(records) == 2 and {r["question_id"] for r in records} == {"t2_mean"}


def test_unknown_question_id_is_an_error(env, capsys):
    assert go(env, "--only", "nope") == 2 and "unknown question id" in capsys.readouterr().err


def test_dry_run_prints_the_plan_and_starts_nothing(env, capsys):
    assert go(env, "--dry-run") == 0
    out = capsys.readouterr().out
    assert "2 questions x 2 settings x 2 repeats = 8 runs" in out and env["factory"].calls == 0
    assert not env["runs"].exists()


def test_a_missing_dataset_file_is_a_clear_error(env, capsys):
    (Path(env["argv"][env["argv"].index("--data-dir") + 1]) / "tiny.csv").unlink()
    assert go(env, "--limit", "1") == 2 and "tiny.csv" in capsys.readouterr().err
    assert go(env, "--dry-run") == 0 and "tiny.csv" in capsys.readouterr().err         # a dry run only needs the plan


def test_resuming_with_another_model_is_refused(env, capsys):
    go(env, "--limit", "1")
    argv = [("other" if a == "m" else a) for a in env["argv"]]
    assert run_eval.main(argv, llm_factory=env["factory"], questions=QS) == 2
    assert "Use another --name" in capsys.readouterr().err


def test_without_a_key_it_exits_2(env, monkeypatch, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    argv = [a for i, a in enumerate(env["argv"]) if a != "--model" and env["argv"][i - 1] != "--model"]
    assert run_eval.main(argv, questions=QS) == 2                       # the real path: no injected model
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err


def test_a_model_that_cannot_use_tools_stops_the_run(env, monkeypatch, capsys):
    from agent.llm import ModelNoToolsError

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-TESTKEY0123456789")

    def refuse(model=None, **kw):
        raise ModelNoToolsError("The model 'm' cannot call tools, which DataPilot needs.")
    monkeypatch.setattr(run_eval, "get_llm", refuse)
    assert run_eval.main(env["argv"], questions=QS) == 2
    assert "cannot call tools" in capsys.readouterr().err and not env["runs"].exists()


def test_a_timeout_is_recorded_and_the_run_goes_on(env):
    class Slow(ScriptedChatModel):
        async def _agenerate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kw: Any) -> ChatResult:
            import asyncio
            await asyncio.sleep(30)
            raise AssertionError("unreachable")

    argv = [a for a in env["argv"]]
    argv[argv.index("--timeout-s") + 1] = "1"
    assert run_eval.main(argv + ["--only", "t2_mean", "--limit", "1"], llm_factory=lambda model=None: Slow(), questions=QS) == 0
    rec = read_runs(env["runs"])[0]
    assert rec["failure_reason"] == "timeout" and not rec["completed"] and rec["error_type"] == "Timeout"
    assert not rec["tool_correct"]


def test_format_duration():
    assert [run_eval.format_duration(x) for x in (8, 95, 3725)] == ["8s", "1m 35s", "1h 02m"]


# --------------------------------------------------------------------------- OpenRouter-specific behaviour
def stateless(model: str | None = None) -> EvalLLM:
    return EvalLLM(script=good(MEAN, "c1"))


def test_llm_calls_are_recorded_per_run(env):
    assert go(env, "--only", "t2_mean") == 0
    records = read_runs(env["runs"])
    assert records and all(r["llm_calls"] >= 3 for r in records)             # router + agent steps + report writer


def test_dry_run_prints_the_plan_and_an_estimate_even_without_the_data_files(tmp_path, capsys):
    code = run_eval.main(["--name", "x", "--dry-run", "--data-dir", str(tmp_path / "nothing"), "--results-dir", str(tmp_path / "r")])
    out = capsys.readouterr().out
    assert code == 0 and "= 120 runs" in out and "ESTIMATE of LLM calls: 480 to 2400" in out and "NOT measured" in out


def test_dry_run_uses_the_measured_mean_when_the_run_folder_has_one(env, capsys):
    go(env, "--only", "t2_mean", "--limit", "2")
    capsys.readouterr()
    assert go(env, "--dry-run") == 0
    assert "measured mean" in capsys.readouterr().out


def test_concurrency_is_limited_to_1_to_4(env, capsys):
    for bad in ("0", "5", "x"):
        with pytest.raises(SystemExit):
            run_eval.build_parser().parse_args(["--name", "n", "--concurrency", bad])
    assert run_eval.build_parser().parse_args(["--name", "n"]).concurrency == 1


def test_concurrent_workers_finish_every_run_once_with_separate_folders(env):
    argv = env["argv"] + ["--only", "t2_mean", "--concurrency", "2"]
    assert run_eval.main(argv, llm_factory=stateless, questions=QS) == 0
    records = read_runs(env["runs"])
    assert len(records) == 4 and len({(r["question_id"], r["setting"], r["repeat"]) for r in records}) == 4
    assert all(r["tool_correct"] and r["report_correct"] for r in records)
    work = env["results"] / "r1" / "_work"
    assert (work / "w1").exists() and (work / "w2").exists()


class RateLimited(ScriptedChatModel):
    def _generate(self, *a: Any, **kw: Any):
        raise RuntimeError("Error code: 429 - slow down")


def one_run(env, factory, waits, sleeps):
    holder = run_eval.DatasetSession(run_eval.Config(data_dir=env["results"] / "d", output_dir=env["results"] / "o"),
                                     Path(env["argv"][env["argv"].index("--data-dir") + 1]), factory)
    try:
        return run_eval.run_one(holder, Q2, Q2.truth(DATA), RunKey("t2_mean", 0, 1), "m", 60, sleep=sleeps.append, rate_limit_waits=waits)
    finally:
        holder.close()


def test_a_rate_limited_run_waits_and_retries_then_succeeds(env):
    sleeps: list[float] = []
    models = iter([RateLimited(), EvalLLM(script=good(MEAN, "c1"))])
    record = one_run(env, lambda model=None: next(models), (5.0, 10.0), sleeps)
    assert sleeps == [5.0] and record["completed"] and record["tool_correct"] and record["failure_reason"] is None
    assert record["llm_calls"] >= 4


def test_a_run_is_counted_as_rate_limit_only_after_the_retries_are_exhausted(env):
    sleeps: list[float] = []
    record = one_run(env, lambda model=None: RateLimited(), (1.0, 2.0), sleeps)
    assert sleeps == [1.0, 2.0] and not record["completed"] and record["failure_reason"] == "rate_limit"


def test_a_plain_crash_is_not_retried(env):
    class Crash(ScriptedChatModel):
        def _generate(self, *a: Any, **kw: Any):
            raise RuntimeError("something else broke")

    sleeps: list[float] = []
    record = one_run(env, lambda model=None: Crash(), (1.0, 2.0), sleeps)
    assert sleeps == [] and record["failure_reason"] == "crash"
