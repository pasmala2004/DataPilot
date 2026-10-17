"""Tests for run_pandas(): worker process, guardrails, output caps, timeout.

Expected numbers are computed with pandas inside the test. Most cases start a real
worker process (about 0.4 s each); the blocked-code cases are rejected before that.
"""

from __future__ import annotations

import ast
import time
from typing import Any

import pandas as pd
import pytest

import core
from core import MAX_OUTPUT_CHARS, _blocked_by_guardrail, _child_env, run_pandas


def run(store: dict, code: str, **kwargs: Any) -> dict[str, Any]:
    return run_pandas("ds1", code, store=store, **kwargs)


# ------------------------------------------------------------------ ok ------
def test_print_output_matches_pandas(analysis_df, analysis_store):
    result = run(analysis_store, 'print(df["amount"].sum())')
    assert float(result["output"]) == pytest.approx(analysis_df["amount"].sum())
    assert "truncated" not in result


def test_result_variable_matches_pandas(analysis_df, analysis_store):
    result = run(analysis_store, 'result = df.groupby("region")["amount"].sum().round(2).to_dict()')
    got = ast.literal_eval(result["output"])
    expected = analysis_df.groupby("region")["amount"].sum().round(2).to_dict()
    assert got.keys() == expected.keys()
    for key, value in expected.items():
        assert got[key] == pytest.approx(value)


def test_print_and_result_are_combined(analysis_store):
    assert run(analysis_store, 'print("a")\nresult = 5')["output"] == "a\n5"


def test_pd_and_np_are_available_and_other_names_are_not(analysis_df, analysis_store):
    code = 'print(np.sqrt(16.0), pd.Series([1, 2, 3]).sum(), len(df), df.shape[1])'
    assert run(analysis_store, code)["output"].split() == ["4.0", "6", str(len(analysis_df)), "5"]
    assert "NameError" in run(analysis_store, "print(store)")["error"]
    assert "NameError" in run(analysis_store, "print(core)")["error"]


def test_df_is_a_copy_and_the_store_is_never_modified(analysis_df, analysis_store):
    before = analysis_df.copy(deep=True)
    changed = run(analysis_store, 'df.loc[:, "amount"] = 0\ndf.drop(df.index, inplace=True)\nprint(len(df))')
    assert changed["output"].strip() == "0"
    pd.testing.assert_frame_equal(analysis_store["ds1"], before)
    # a second call sees the original data again
    assert float(run(analysis_store, 'print(df["amount"].sum())')["output"]) == pytest.approx(before["amount"].sum())


def test_allowed_to_conversions_work_for_real(analysis_df, analysis_store):
    code = 'result = pd.to_datetime(df["order_date"]).dt.year.max()'
    assert float(run(analysis_store, code)["output"]) == analysis_df["order_date"].dt.year.max() == 2024  # NaT makes it a float
    code = 'print(sum(pd.to_numeric(df["qty"]).to_list()))'
    assert int(run(analysis_store, code)["output"]) == int(analysis_df["qty"].sum())


# ------------------------------------------------------------- caps ---------
def test_output_is_capped_at_2000_chars(analysis_store):
    result = run(analysis_store, 'print("x" * 5000)')
    assert len(result["output"]) == MAX_OUTPUT_CHARS == 2000
    assert result["truncated"] is True


def test_output_exactly_at_the_cap_is_not_truncated(analysis_store):
    result = run(analysis_store, 'print("x" * 1999)')  # print adds "\n" -> exactly 2000 characters
    assert len(result["output"]) == 2000 and "truncated" not in result


def test_dataframe_result_shows_at_most_20_rows(analysis_df, analysis_store):
    assert len(analysis_df) == 60
    result = run(analysis_store, "result = df")
    lines = result["output"].splitlines()
    data_rows = [ln for ln in lines if ln.split()[0].isdigit()]
    assert len(data_rows) == 20
    assert "60 rows total, first 20 shown" in result["output"]


def test_printed_dataframe_is_also_limited_to_20_rows(analysis_store):
    out = run(analysis_store, "print(df)")["output"]
    assert "..." in out and "[60 rows x 5 columns]" in out
    assert sum(1 for ln in out.splitlines() if ln.split() and ln.split()[0].isdigit()) == 20


def test_huge_result_object_is_still_capped(analysis_store):
    result = run(analysis_store, "result = list(range(100000))")
    assert len(result["output"]) == 2000 and result["truncated"] is True


# ---------------------------------------------------------- timeout ---------
@pytest.mark.parametrize("code", ["while True:\n    pass", 'while True:\n    print("spam")'])
def test_timeout_terminates_runaway_code(analysis_store, code):
    start = time.monotonic()
    result = run(analysis_store, code, timeout=1)
    elapsed = time.monotonic() - start
    assert "timeout" in result["error"] and "terminated" in result["error"]
    assert elapsed < 6  # it really was stopped, we did not wait for the loop


def test_default_timeout_is_10_seconds():
    assert core.RUN_PANDAS_TIMEOUT == 10.0


def test_the_server_keeps_working_after_a_timeout(analysis_store):
    assert "timeout" in run(analysis_store, "while True: pass", timeout=1)["error"]
    assert run(analysis_store, "print(1 + 1)")["output"].strip() == "2"


# ------------------------------------------------------- guardrail: blocked --
BLOCKED = [
    "import os", "import numpy", "from os import path", "__import__('os')", "open('x.txt')",
    "exec('1')", "eval('1')", "compile('1', '', 'exec')", "os.system('id')", "sys.exit()",
    "subprocess.run(['id'])", "getattr(df, 'shape')", "setattr(df, 'x', 1)", "delattr(df, 'x')",
    "df.__class__", "df.__dict__", "().__class__.__mro__", "pd.read_csv('x.csv')",
    "pd.read_excel('x.xlsx')", "pd.read_pickle('x')", "df.to_csv('o.csv')", "df.to_excel('o.xlsx')",
    "df.to_pickle('o.pkl')", "df.to_sql('t', None)", "df.to_json('o.json')", "df.to_parquet('o')",
    "df.to_string('o.txt')", "df.to_markdown('o.md')", "df.to_clipboard()", "np.load('x.npy')",
    "np.save('x', df.values)", "np.savetxt('x', df.values)", "df.values.tofile('x')",
    "np.memmap('x')", "pd.ExcelWriter('x.xlsx')", "pd.io.common.os", "df.query('@os.system(1)')",
    "df.eval('1')", "pd.eval('1')",
]


@pytest.mark.parametrize("code", BLOCKED)
def test_blocked_code_is_rejected_before_it_runs(analysis_store, tmp_path, monkeypatch, code):
    monkeypatch.chdir(tmp_path)
    result = run(analysis_store, code)
    assert result["error"].startswith("blocked by guardrail")
    assert list(tmp_path.iterdir()) == []  # nothing was written anywhere


@pytest.mark.parametrize("code", [
    "pd.to_datetime(df['order_date'])", "pd.to_numeric(df['amount'])", "df['amount'].to_numpy()",
    "df.to_dict('records')", "df['region'].to_list()", "df['qty'].to_frame()", "df['order_date'].dt.to_period('M')",
    "df.groupby('region').amount.mean()", "df['region'].str.upper()", "df.describe()",
    "np.percentile(df['amount'].dropna(), 90)", "df.sort_values('amount').head(5)",
    "df['amount'].apply(lambda v: v * 2)", "df.pivot_table(index='region', values='amount', aggfunc='sum')",
    "df.sample(5, random_state=0)", "df['region'].isin(['North'])", "result = df.corr(numeric_only=True)",
    "x = [i ** 2 for i in range(5)]", "autotos = 1; reload_count = 2",
])
def test_ordinary_analysis_code_is_not_blocked(code):
    assert _blocked_by_guardrail(code) is None


def test_guardrail_is_blunt_about_banned_words_in_strings():
    # documented behaviour: a column called "open" is rejected too
    assert _blocked_by_guardrail('df["open"].mean()') is not None


@pytest.mark.parametrize("name", ["type", "globals", "locals", "vars", "dir", "input", "breakpoint",
                                  "help", "exit", "quit", "memoryview", "bytes", "object", "callable"])
def test_non_whitelisted_builtins_do_not_exist(analysis_store, name):
    result = run(analysis_store, f"print({name})")
    assert "NameError" in result["error"] and name in result["error"]


def test_class_definitions_are_not_available(analysis_store):
    assert "error" in run(analysis_store, "class A:\n    pass\nprint(1)")


# ----------------------------------------------------------- error paths ----
def test_runtime_error_reports_type_message_and_line(analysis_store):
    result = run(analysis_store, 'x = 1\nprint(df["nope"])')
    assert "KeyError" in result["error"] and "nope" in result["error"] and "line 2" in result["error"]


def test_syntax_error_is_an_error_not_a_crash(analysis_store):
    result = run(analysis_store, "1 +")
    assert "SyntaxError" in result["error"]


def test_code_that_produces_no_output_is_an_error(analysis_store):
    assert "no output" in run(analysis_store, "x = df.shape")["error"]
    assert "no output" in run(analysis_store, "result = None")["error"]
    assert "no output" in run(analysis_store, 'print("")')["error"]


def test_huge_allocation_returns_an_error(analysis_store):
    result = run(analysis_store, "x = np.ones(10**10)\nprint(x.size)")
    assert "MemoryError" in result["error"]


def test_a_crashing_worker_is_reported_not_raised(analysis_store, monkeypatch):
    monkeypatch.setattr(core, "_RUNNER", "import os\nos._exit(3)")
    assert "worker process failed (exit code 3)" in run(analysis_store, "print(1)")["error"]
    monkeypatch.setattr(core, "_RUNNER", "print('this is not json')")
    assert "worker process failed" in run(analysis_store, "print(1)")["error"]


def test_unknown_dataset_lists_loaded_ids():
    store = {"ds2": pd.DataFrame({"a": [1]}), "ds1": pd.DataFrame({"a": [1]})}
    result = run_pandas("ds9", "print(1)", store=store)
    assert "ds9" in result["error"] and result["loaded"] == ["ds1", "ds2"]


@pytest.mark.parametrize("kwargs", [
    {"dataset_id": None}, {"dataset_id": 1}, {"dataset_id": ["ds1"]},
    {"code": None}, {"code": 5}, {"code": ["print(1)"]}, {"code": ""}, {"code": "  \n "},
    {"code": "x = 1\n" * 3000},  # longer than 10,000 characters
    {"store": None}, {"store": []}, {"store": "ds1"}, {"store": {"ds1": None}},
])
def test_no_bad_input_raises(analysis_store, kwargs):
    call = {"dataset_id": "ds1", "code": "print(1)", "store": analysis_store, **kwargs}
    result = run_pandas(call.pop("dataset_id"), call.pop("code"), **call)
    assert isinstance(result, dict) and "error" in result


def test_missing_arguments_do_not_raise():
    assert "error" in run_pandas()
    assert "error" in run_pandas("ds1", "print(1)")  # store missing


# --------------------------------------------------------- safety plumbing --
def test_worker_environment_has_no_secrets(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or_secret")
    env = _child_env()
    assert "GROQ_API_KEY" not in env and "OPENROUTER_API_KEY" not in env
    assert "PATH" in env


def test_worker_output_never_reaches_our_stdout_and_logs_contain_no_code_or_values(analysis_store, capfd):
    result = run(analysis_store, 'print("Zanzibar-Q9", df["amount"].sum())')
    assert "Zanzibar-Q9" in result["output"]
    captured = capfd.readouterr()
    assert captured.out == ""
    assert "tool=run_pandas" in captured.err and "args=['dataset_id', 'code', 'store']" in captured.err
    assert "Zanzibar" not in captured.err and "amount" not in captured.err


def test_results_are_json_safe(analysis_store):
    import json

    for code in ("result = df", "print(df.describe())", "result = df['amount'].mean()"):
        json.dumps(run(analysis_store, code), allow_nan=False)


def test_run_pandas_is_traced():
    assert hasattr(run_pandas, "__wrapped__")
