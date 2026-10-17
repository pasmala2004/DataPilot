"""Tests for agent/session.py: AgentSession, Job, uploads. Real excel_server over stdio + scripted LLMs.

No network, no API key. Process checks use psutil (skipped when it is not installed).
"""

from __future__ import annotations

import asyncio
import io
import json
import threading
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agent.llm import KeyRejectedError, MissingKeyError
from agent.session import (
    MAX_UPLOAD_BYTES,
    AgentSession,
    SessionError,
    UploadError,
    safe_upload_name,
)
from core import Config
from scripted import CANARIES, ScriptedChatModel, agg, call, plan, say

WAIT = 90  # seconds a job may take in a test before it counts as hung


def xlsx_bytes(orders: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        orders.to_excel(writer, sheet_name="orders", index=False)
    return buffer.getvalue()


def server_pids() -> set[int]:
    psutil = pytest.importorskip("psutil")
    found = set()
    for child in psutil.Process().children(recursive=True):
        try:
            if "excel_server" in " ".join(child.cmdline()):
                found.add(child.pid)
        except psutil.Error:
            pass
    return found


class SlowModel(ScriptedChatModel):
    """Never answers within the test's timeout (async sleep, so a cancellation ends it at once)."""

    async def _agenerate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        await asyncio.sleep(60)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="late"))])


@pytest.fixture
def base(tmp_path: Path) -> Config:
    return Config(data_dir=tmp_path / "data", output_dir=tmp_path / "outputs")


@pytest.fixture
def make_session(base: Config):
    """Factory for started sessions that are always closed at the end (no server left behind)."""
    made: list[AgentSession] = []

    def factory(llm: Any = None, **kwargs: Any) -> AgentSession:
        session = AgentSession(base, llm_factory=(lambda **kw: llm) if llm is not None else None, **kwargs)
        made.append(session)
        return session.start()

    yield factory
    for session in made:
        session.close()


# =========================================================================== #
# Upload handling (no server needed)
# =========================================================================== #
@pytest.mark.parametrize("name, expected", [
    ("sales.xlsx", "sales.xlsx"),
    ("My Sales (2024).XLSX", "My Sales (2024).xlsx"),
    ("data.csv", "data.csv"),
    ("../../evil.xlsx", "evil.xlsx"),                      # folders are stripped, the file stays inside
    ("..\\..\\windows\\evil.csv", "evil.csv"),
    ("C:\\Users\\me\\sales.xlsx", "sales.xlsx"),
    ("/etc/passwd.csv", "passwd.csv"),
    ("we!rd*na?me.csv", "we_rdname.csv"),                  # reserved characters are removed, other symbols become _
    ("نتائج.xlsx", "نتائج.xlsx"),                            # Unicode letters are kept
    ("مبيعات 2024 (نهائي).CSV", "مبيعات 2024 (نهائي).csv"),
    ("..\\..\\نتائج.csv", "نتائج.csv"),
    ("café_ventes-2024.xlsx", "café_ventes-2024.xlsx"),
    ("销售数据.csv", "销售数据.csv"),
    ('a<b>c:d"e|f?g*.csv', "abcdefg.csv"),
    ("evil\u202Efdp.csv", "evilfdp.csv"),                  # bidi override characters are stripped
    ("tab\there\x07.csv", "tabhere.csv"),                   # control characters are stripped
    (".xlsx", "upload.xlsx"),                                # an empty stem becomes "upload"
    ("...csv", "upload.csv"),
    (" .csv", "upload.csv"),
    ("name.CSV ", "name.csv"),
    ("a.b.xlsx", "a.b.xlsx"),
    ("CON.xlsx", "CON_.xlsx"),                               # Windows device names are renamed
    ("nul.csv", "nul_.csv"),
    ("Com1.csv", "Com1_.csv"),
    ("LPT9.xlsx", "LPT9_.xlsx"),
    ("aux .csv", "aux_.csv"),
    ("console.csv", "console.csv"),                          # only the exact device names
    ("x" * 300 + ".csv", "x" * 100 + ".csv"),
])
def test_safe_upload_name_strips_folders_and_odd_characters(name, expected):
    assert safe_upload_name(name) == expected


@pytest.mark.parametrize("name", [
    "", "..", ".", "/", "a/b/..", "evil.exe", "report.xlsx.exe", "noextension", "x.xlsm", "a.xls",
    "bad\x00name.xlsx", "sales.xlsx/", "dir\\",
])
def test_unsafe_or_unsupported_names_are_refused(name):
    with pytest.raises(UploadError):
        safe_upload_name(name)


def test_save_upload_writes_only_inside_the_session_folder(base, tmp_path):
    session = AgentSession(base, session_id="s1")
    saved = session.save_upload("../../../outside.xlsx", b"PK-data")
    assert saved == "outside.xlsx"
    assert (session.data_dir / "outside.xlsx").read_bytes() == b"PK-data"
    assert session.data_dir == (tmp_path / "data" / "s1").resolve()
    written = {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()}
    assert written == {"data/s1/outside.xlsx"}               # nothing anywhere else


def test_save_upload_rejects_wrong_extension_empty_and_oversize(base):
    session = AgentSession(base, session_id="s2")
    with pytest.raises(UploadError, match="xlsx and .csv"):
        session.save_upload("notes.txt", b"hello")
    with pytest.raises(UploadError, match="empty"):
        session.save_upload("a.csv", b"")
    with pytest.raises(UploadError, match="25 MB"):
        session.save_upload("big.csv", b"x" * (MAX_UPLOAD_BYTES + 1))
    assert session.save_upload("ok.csv", b"x" * MAX_UPLOAD_BYTES) == "ok.csv"      # exactly at the limit is fine
    assert session.files() == ["ok.csv"]                      # the refused files were not written


def test_session_ids_cannot_escape_the_base_folder(base):
    for bad in ("../x", "a/b", "x" * 65, "a b", "..\\x"):
        with pytest.raises(ValueError):
            AgentSession(base, session_id=bad)


# =========================================================================== #
# Several submits in one session
# =========================================================================== #
def test_several_submits_reuse_one_server_and_one_loop(make_session, orders):
    before = server_pids()
    llm = ScriptedChatModel(script=[plan(clarification="Which column do you mean?") for _ in range(6)])
    session = make_session(llm)
    name = session.save_upload("sales.xlsx", xlsx_bytes(orders))
    pids = server_pids() - before
    assert len(pids) == 1                                    # ONE server process for the session
    seen_threads = set()
    for i in range(6):
        job = session.submit("something vague", name, "orders", model="m", make_report=False)
        assert job.wait(WAIT), f"job {i} hung"
        assert job.error is None
        assert job.result()["clarification"] == "Which column do you mean?"
        seen_threads.add(threading.current_thread().name)
        assert server_pids() - before == pids                # still the same process after every submit
    assert llm.script == []                                  # every job really ran the (scripted) router


def test_dataset_ids_keep_counting_up_inside_one_server(make_session, orders):
    """Each run loads the sheet again; ids ds1, ds2, ds3 prove the SAME server (its store) lived on."""
    script: list[AIMessage] = []
    for i in range(3):
        script += [plan(("analysis", "Sum amount.")), call("aggregate", agg(), f"c{i}"), say("Summed."), say(f"Done [c{i}].")]
    session = make_session(ScriptedChatModel(script=script))
    name = session.save_upload("sales.xlsx", xlsx_bytes(orders))
    ids = []
    for _ in range(3):
        job = session.submit("total amount", name, "orders", make_report=False)
        assert job.wait(WAIT) and job.error is None
        ids.append(job.result()["dataset_id"])
    assert ids == ["ds1", "ds2", "ds3"]


def test_the_trace_queue_delivers_lines_in_order_and_only_once(make_session, orders):
    llm = ScriptedChatModel(script=[
        plan(("analysis", "Sum amount.")), call("aggregate", agg(), "call_1"), say("Summed."), say("Total computed [call_1]."),
    ])
    session = make_session(llm)
    name = session.save_upload("sales.xlsx", xlsx_bytes(orders))
    job = session.submit("total amount", name, "orders", make_report=False)
    lines: list[str] = []
    deadline = time.monotonic() + WAIT
    while not job.done() and time.monotonic() < deadline:
        lines += job.drain()
        time.sleep(0.02)
    lines += job.drain()
    assert job.done() and job.error is None
    text = "\n".join(lines)
    assert text.index("LOAD") < text.index("ROUTER") < text.index("TOOL aggregate") < text.index("OBSERVE")
    assert job.drain() == []                                  # lines are delivered once
    assert job.elapsed() > 0
    frozen = job.elapsed()
    time.sleep(0.05)
    assert job.elapsed() == frozen                            # the clock stops when the job is done
    assert lines == job.result()["trace"]                     # every trace line arrived, in order, none twice


def test_a_job_with_a_report_returns_the_report_dict_inside_the_session_folder(make_session, orders):
    writer = AIMessage(content="", tool_calls=[{"name": "WriterOutput", "id": "w", "args": {
        "sections": [{"task_idx": 0, "text": "The total is {{F1}}."}], "summary": "Total is {{F1}}."}}])
    llm = ScriptedChatModel(script=[
        plan(("analysis", "Sum amount.")), call("aggregate", agg(), "call_1"), say("Summed."), say("Total [call_1]."), writer,
    ])
    session = make_session(llm)
    name = session.save_upload("sales.xlsx", xlsx_bytes(orders))
    job = session.submit("total amount", name, "orders")
    assert job.wait(WAIT) and job.error is None
    report = job.result()["report"]
    path = Path(report["path"])
    assert path.is_file() and path.parent == session.output_dir / "reports"
    assert f"{orders['amount'].sum():,.2f}" in report["markdown"]


# =========================================================================== #
# Timeout, errors, and survival
# =========================================================================== #
def test_the_overall_timeout_stops_the_run_and_the_session_still_works(make_session, orders):
    slow = SlowModel()
    session = make_session(slow)
    name = session.save_upload("sales.xlsx", xlsx_bytes(orders))
    started = time.monotonic()
    job = session.submit("total amount", name, "orders", make_report=False, timeout_s=1)
    assert job.wait(WAIT)
    assert time.monotonic() - started < 30
    assert job.result() is None and "took longer than 1 seconds" in job.error

    ok = ScriptedChatModel(script=[plan(clarification="Which column?")])
    session._llm_factory = lambda **kw: ok                    # same session, same server, a working model now
    again = session.submit("vague", name, "orders", make_report=False)
    assert again.wait(WAIT) and again.error is None and again.result()["clarification"] == "Which column?"


def test_a_missing_key_is_a_clear_job_error_not_a_crash(make_session, orders):
    def no_model(**kw: Any):
        raise MissingKeyError("OPENROUTER_API_KEY is not set. Set it in your environment.")

    session = make_session()
    session._llm_factory = no_model
    name = session.save_upload("sales.xlsx", xlsx_bytes(orders))
    job = session.submit("total amount", name, "orders")
    assert job.wait(WAIT)
    assert job.result() is None and "OPENROUTER_API_KEY is not set" in job.error and "Traceback" not in job.error


def test_a_bad_file_ends_the_job_with_the_tool_message(make_session, orders):
    session = make_session(ScriptedChatModel())
    job = session.submit("total amount", "missing.xlsx", "orders", make_report=False)
    assert job.wait(WAIT) and job.error is None
    assert "Cannot open 'missing.xlsx'" in job.result()["error"]


def test_list_sheets_and_describe_return_names_and_schema_only(make_session, orders):
    session = make_session()
    name = session.save_upload("sales.xlsx", xlsx_bytes(orders))
    assert session.list_sheets(name) == ["orders"]
    schema = session.describe(name, "orders")
    assert set(schema) <= {"rows", "columns", "nulls", "date_ranges", "warnings"} and "preview" not in schema
    assert schema["rows"] == len(orders) and set(schema["columns"]) == set(orders.columns)
    assert schema["nulls"]["amount"] == 1
    assert not any(c in json.dumps(schema, default=str) for c in CANARIES)
    with pytest.raises(SessionError, match="Cannot open 'nope.xlsx'"):
        session.list_sheets("nope.xlsx")
    session.save_upload("data.csv", b"a,b\n1,2\n")
    assert session.list_sheets("data.csv") == ["csv"]


def test_nothing_the_session_exposes_contains_raw_rows(make_session, orders):
    llm = ScriptedChatModel(script=[
        plan(("analysis", "Sum amount by region.")), call("aggregate", agg(group_by=["region"]), "call_1"),
        say("Summed."), say("Done [call_1]."),
        AIMessage(content="", tool_calls=[{"name": "WriterOutput", "id": "w", "args": {
            "sections": [{"task_idx": 0, "text": "North is {{F1}}."}], "summary": "North is {{F1}}."}}]),
    ])
    session = make_session(llm)
    name = session.save_upload("sales.xlsx", xlsx_bytes(orders))
    job = session.submit("total by region", name, "orders")
    lines: list[str] = []
    while not job.done():
        lines += job.drain()
        time.sleep(0.02)
    lines += job.drain()
    exposed = "\n".join(lines) + json.dumps(job.result(), default=str) + json.dumps(session.describe(name, "orders"), default=str)
    assert job.error is None
    assert not any(c in exposed for c in CANARIES)           # the customer column never appears anywhere


# =========================================================================== #
# Isolation and shutdown
# =========================================================================== #
def test_two_sessions_do_not_see_each_others_files_or_servers(make_session, orders):
    before = server_pids()
    a, b = make_session(), make_session(ScriptedChatModel())
    assert a.data_dir != b.data_dir and a.output_dir != b.output_dir and a.session_id != b.session_id
    assert len(server_pids() - before) == 2                  # one process each
    name = a.save_upload("secret.xlsx", xlsx_bytes(orders))
    assert a.files() == [name] and b.files() == []
    assert a.list_sheets(name) == ["orders"]
    with pytest.raises(SessionError, match="Cannot open"):
        b.list_sheets(name)                                  # B's server cannot reach A's folder
    assert not (b.data_dir / name).exists()
    job = b.submit("total", name, "orders", make_report=False)   # nor can a run in B
    assert job.wait(WAIT) and "Cannot open" in job.result()["error"]


def test_close_stops_the_server_process_and_the_thread(base, orders):
    before = server_pids()
    session = AgentSession(base).start()
    started = server_pids() - before
    assert len(started) == 1
    thread = session._thread
    session.close()
    assert not thread.is_alive()
    assert server_pids() - before == set()                   # no server process left
    session.close()                                          # closing twice is fine
    with pytest.raises(SessionError):
        session.submit("q", "f.xlsx")
    with pytest.raises(SessionError):
        session.start()


def test_close_while_a_job_is_running_cancels_it_quickly(base, orders):
    before = server_pids()
    session = AgentSession(base, llm_factory=lambda **kw: SlowModel()).start()
    name = session.save_upload("sales.xlsx", xlsx_bytes(orders))
    job = session.submit("total", name, "orders", make_report=False, timeout_s=300)
    time.sleep(1.5)                                           # let the run reach the (slow) router call
    queued = session.submit("another", name, "orders", make_report=False)
    started = time.monotonic()
    session.close()
    assert time.monotonic() - started < 25
    assert job.wait(5) and "cancelled" in job.error
    assert queued.wait(5) and "closed" in queued.error
    assert server_pids() - before == set()


def test_close_can_delete_the_session_folders(base):
    session = AgentSession(base).start()
    session.save_upload("a.csv", b"a\n1\n")
    session.close(cleanup=True)
    assert not session.data_dir.exists() and not session.output_dir.exists()


def test_a_server_that_cannot_start_is_a_clear_error(base, tmp_path):
    broken = tmp_path / "no_such_server.py"
    with pytest.raises(SessionError):
        AgentSession(base, server_script=broken).start()


# =========================================================================== #
# Unicode (Arabic) file and column names through the real server
# =========================================================================== #
def test_an_arabic_file_name_and_arabic_columns_work_end_to_end(make_session):
    frame = pd.DataFrame({
        "المنطقة": ["شمال", "جنوب", "شمال", "شرق", "جنوب", "شمال"],
        "المبلغ": [120.5, 80.0, 200.0, 95.25, 60.0, None],
    })
    writer = AIMessage(content="", tool_calls=[{"name": "WriterOutput", "id": "w", "args": {
        "sections": [{"task_idx": 0, "text": "الشمال: {{F1}}."}], "summary": "الشمال: {{F1}}."}}])
    llm = ScriptedChatModel(script=[
        plan(("analysis", "اجمع المبلغ حسب المنطقة.")),
        call("aggregate", {"dataset_id": "ds1", "column": "المبلغ", "agg": "sum", "group_by": ["المنطقة"]}, "call_1"),
        say("تم."), say("المجموع [call_1]."), writer,
    ])
    session = make_session(llm)
    name = session.save_upload("نتائج.csv", frame.to_csv(index=False).encode("utf-8"))
    assert name == "نتائج.csv" and (session.data_dir / "نتائج.csv").is_file()
    assert session.list_sheets(name) == ["csv"]
    schema = session.describe(name, "csv")
    assert list(schema["columns"]) == ["المنطقة", "المبلغ"] and schema["rows"] == 6

    job = session.submit("المجموع حسب المنطقة", name, "csv")
    assert job.wait(WAIT) and job.error is None
    result = job.result()
    assert result["error"] is None and [t["status"] for t in result["task_status"]] == ["finished"]
    groups = {g["المنطقة"]: g["value"] for g in result["results"][0]["output"]["groups"]}
    expected = frame.dropna(subset=["المبلغ"]).groupby("المنطقة")["المبلغ"].sum()
    assert groups == expected.to_dict()
    north = f"{expected['شمال']:,.2f}"
    assert north in result["report"]["markdown"] and "المنطقة=شمال" in result["report"]["markdown"]


def test_an_api_failure_inside_the_run_is_a_clean_message_without_the_key(make_session, orders, monkeypatch):
    """A 401 raised by the client mid-run reaches the user as the fixed message; the key never appears."""
    key = "sk-or-v1-TESTKEY0123456789"
    monkeypatch.setenv("OPENROUTER_API_KEY", key)

    class Boom(ScriptedChatModel):
        def _generate(self, *a: Any, **kw: Any):
            raise RuntimeError(f"Error code: 401 - invalid key {key}")

    session = make_session()
    session._llm_factory = lambda **kw: Boom()
    name = session.save_upload("sales.xlsx", xlsx_bytes(orders))
    job = session.submit("total amount", name, "orders", make_report=False)
    assert job.wait(WAIT)
    shown = f"{job.error} {(job.result() or {}).get('error')}"
    assert "key was rejected" in shown and key not in shown and "Traceback" not in shown


def test_each_job_counts_its_model_calls(make_session, orders):
    session = make_session()
    llm = ScriptedChatModel(script=[plan(clarification="Which column?")])
    session._llm_factory = lambda **kw: llm
    name = session.save_upload("sales.xlsx", xlsx_bytes(orders))
    job = session.submit("vague", name, "orders", make_report=False)
    assert job.wait(WAIT) and job.llm_calls == 1
