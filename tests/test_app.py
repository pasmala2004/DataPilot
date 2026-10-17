"""Smoke and flow tests for app.py with streamlit.testing.v1.AppTest. No network, no real API key (a fake one is set).

The key/model check is faked (``check_openrouter`` patched) and the LLM is a scripted model injected through the
session's default factory. The upload test uses a REAL AgentSession (real excel_server over stdio).
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import streamlit as st
from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

import agent.llm as llm_module
import agent.session as session_module
from agent.llm import KeyRejectedError
from agent.session import Job
from agent.ui_helpers import CURATED_MODELS
from scripted import CANARIES, ScriptedChatModel, agg, call, plan, say

APP = str(Path(__file__).resolve().parents[1] / "app.py")
FAKE_KEY = "sk-or-v1-FAKEKEY0123456789"


def xlsx_bytes(orders: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        orders.to_excel(writer, sheet_name="orders", index=False)
        pd.DataFrame({"region": ["North"], "target": [500]}).to_excel(writer, sheet_name="targets", index=False)
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def isolate(monkeypatch, tmp_path):
    """Private DATA_DIR/OUTPUT_DIR, a fake key, no old settings, no network check."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)
    for name in ("OPENROUTER_MODEL", "MODEL_NAME", "OLLAMA_MODEL", "OLLAMA_HOST"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(llm_module, "check_openrouter", lambda settings, **kw: None)
    st.cache_data.clear()
    st.cache_resource.clear()
    yield
    st.cache_data.clear()
    st.cache_resource.clear()


def start() -> AppTest:
    return AppTest.from_file(APP, default_timeout=60).run()


def texts(elements) -> list[str]:
    return [str(e.value) for e in elements]


def everything(at: AppTest) -> str:
    """All text the page shows (to prove the key never appears)."""
    parts = [str(getattr(e, "value", "")) for group in (at.markdown, at.text, at.caption, at.code, at.info, at.warning, at.error, at.success)
             for e in group]
    return "\n".join(parts)


def run_button(at: AppTest):
    return [b for b in at.button if b.label == "Run"][0]


def tick_privacy(at: AppTest) -> AppTest:
    return at.checkbox(key="privacy_ack").check().run()


# =========================================================================== #
# Sidebar: key, model, check button, privacy gate
# =========================================================================== #
def test_the_app_starts_and_shows_the_key_status_and_the_model_list():
    at = start()
    assert not at.exception
    assert any("API key: found" in t for t in texts(at.sidebar.success))
    assert list(at.sidebar.selectbox[0].options) == list(CURATED_MODELS) and at.sidebar.selectbox[0].value == "openai/gpt-4o-mini"
    assert any("not checked yet" in str(c.value) for c in at.sidebar.caption)
    assert FAKE_KEY not in everything(at)
    assert not any("Ollama" in str(e.value) or "ollama" in str(e.value) for e in at.sidebar.markdown)


def test_without_a_key_run_is_disabled_and_the_hint_shows_how_to_set_it(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    at = start()
    assert not at.exception
    assert any("API key: not found" in t for t in texts(at.sidebar.error))
    assert any("OPENROUTER_API_KEY" in str(c.value) and "$env:" in str(c.value) for c in at.sidebar.code)
    assert [b for b in at.sidebar.button if b.label == "Check key and model"][0].disabled
    assert run_button(at).disabled


def test_run_needs_the_privacy_checkbox_even_with_a_file_and_a_question():
    fake = FakeSession(result={"answer": None, "clarification": "?", "error": None, "results": [], "task_status": []})
    at = AppTest.from_file(APP, default_timeout=60)
    prepare(at, fake)
    at.run()
    at.text_area[0].set_value("total amount").run()
    assert any("sent to OpenRouter" in t for t in texts(at.warning))               # the privacy notice is on the page
    assert run_button(at).disabled and any("Tick the box" in str(c.value) for c in at.caption)
    tick_privacy(at)
    assert not run_button(at).disabled


def test_run_is_disabled_until_a_file_is_loaded_and_a_question_typed():
    at = start()
    tick_privacy(at)
    assert run_button(at).disabled and any("Upload a file first" in str(c.value) for c in at.caption)


def test_the_check_button_shows_ready_or_a_clean_error_and_never_the_key(monkeypatch):
    at = start()
    [b for b in at.sidebar.button if b.label == "Check key and model"][0].click().run()
    assert any("Model: ready" in t for t in texts(at.sidebar.success))
    monkeypatch.setattr(llm_module, "check_openrouter", lambda settings, **kw: (_ for _ in ()).throw(KeyRejectedError(llm_module.MESSAGES["rejected"])))
    [b for b in at.sidebar.button if b.label == "Check key and model"][0].click().run()
    assert any("Model: error" in t and "key was rejected" in t for t in texts(at.sidebar.error))
    assert FAKE_KEY not in everything(at) and not at.exception
    tick_privacy(at)
    assert run_button(at).disabled                                                   # a failed check blocks Run


def test_a_custom_model_id_overrides_the_list_and_old_ollama_settings_are_ignored(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "gemma4:26b")
    at = start()
    assert at.sidebar.selectbox[0].value == "openai/gpt-4o-mini"                    # OLLAMA_MODEL changes nothing
    fake = FakeSession(result={"answer": None, "clarification": "?", "error": None, "results": [], "task_status": []})
    at = AppTest.from_file(APP, default_timeout=60)
    prepare(at, fake)
    at.run()
    at.sidebar.text_input[0].set_value("vendor/some-model").run()
    at.text_area[0].set_value("total amount").run()
    tick_privacy(at)
    run_button(at).click().run()
    assert fake.submitted[0][-1] == "vendor/some-model"


def test_a_model_id_from_the_environment_is_selectable(monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODEL", "vendor/from-env")
    at = start()
    assert at.sidebar.selectbox[0].value == "vendor/from-env" and "vendor/from-env" in at.sidebar.selectbox[0].options


# =========================================================================== #
# Full flow with a real session: upload -> schema -> run -> report -> reviewed -> downloads
# =========================================================================== #
def scripted_llm(orders: pd.DataFrame) -> ScriptedChatModel:
    writer = AIMessage(content="", tool_calls=[{"name": "WriterOutput", "id": "w", "args": {
        "sections": [{"task_idx": 0, "text": "The total amount is {{F1}}."}], "summary": "The total amount is {{F1}}."}}])
    return ScriptedChatModel(script=[
        plan(("analysis", "Sum amount.")), call("aggregate", agg(), "call_1"), say("Summed."), say("Total computed [call_1]."), writer,
    ])


def wait_for(at: AppTest, done, seconds: float = 90) -> AppTest:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        at.run()
        assert not at.exception
        if done(at):
            return at
        time.sleep(0.3)
    raise AssertionError("the app did not reach the expected state in time")


def test_upload_schema_run_review_and_download_without_rerunning_the_agent(monkeypatch, orders):
    llm = scripted_llm(orders)
    monkeypatch.setattr(session_module, "_default_llm_factory", lambda **kw: llm)
    at = start()
    at.file_uploader[0].upload("sales.xlsx", xlsx_bytes(orders), "application/vnd.ms-excel").run()
    assert not at.exception
    try:
        assert next(b for b in at.selectbox if b.label == "Sheet").options == ["orders", "targets"]                # sheets of the uploaded workbook
        schema = at.dataframe[0].value                                          # the schema table (no data rows)
        assert list(schema["column"]) == list(orders.columns) and "customer" in set(schema["column"])
        assert not any(c in schema.to_string() for c in CANARIES)
        assert int(schema.loc[schema["column"] == "amount", "nulls"].iloc[0]) == 1

        at.text_area[0].set_value("total amount").run()
        tick_privacy(at)
        run_button(at).click().run()
        at = wait_for(at, lambda a: any(sub.label == "I reviewed this report" for sub in a.checkbox))
        assert llm.script == []                                                 # the whole script was used once

        total = f"{orders['amount'].sum():,.2f}"
        assert any(total in str(m.value) for m in at.markdown)                  # report preview shows the pandas value
        downloads = at.get("download_button")
        assert len(downloads) == 2 and all(d.proto.disabled for d in downloads)  # locked until the review

        at.checkbox(key="reviewed").check().run()
        downloads = at.get("download_button")
        assert not any(d.proto.disabled for d in downloads)
        assert {d.proto.label for d in downloads} == {"Download DOCX", "Download Markdown"}
        at.run()                                                               # more reruns: the agent is NOT run again
        assert llm.script == [] and len(llm.seen) == 5
        assert not any(c in str(at.markdown) + str(at.code) for c in CANARIES)  # no raw rows in the UI
    finally:
        at.session_state["session"].close(cleanup=True)


# =========================================================================== #
# Result states with a fake session (no server): clarification, errors, fallbacks, locked downloads
# =========================================================================== #
class FakeSession:
    """Just enough of AgentSession for the app: a job that is already done."""

    running = True

    def __init__(self, result: Any = None, error: str | None = None, hint: str | None = None):
        self.job = Job()
        self.job._finish(result=result, error=error, hint=hint)
        self.submitted: list[tuple] = []

    def submit(self, *args: Any, **kwargs: Any) -> Job:
        self.submitted.append(args)
        return self.job

    def close(self, **kw: Any) -> None:
        pass


def prepare(at: AppTest, fake: FakeSession) -> None:
    at.session_state["session"] = fake
    at.session_state["file"] = "sales.xlsx"
    at.session_state["sheets"] = ["orders"]
    at.session_state["schema"] = {"rows": 3, "columns": {"amount": "float64"}, "nulls": {}}
    at.session_state["schema_key"] = ("sales.xlsx", "orders")


def run_with(fake: FakeSession) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    prepare(at, fake)
    at.session_state["privacy_ack"] = True
    at.run()
    at.text_area[0].set_value("total amount").run()
    run_button(at).click().run()
    return at


def test_a_vague_request_shows_the_clarification_inline():
    fake = FakeSession(result={"answer": None, "clarification": "Which column do you mean?", "error": None, "results": [], "task_status": []})
    at = run_with(fake)
    assert not at.exception and len(fake.submitted) == 1
    assert any("Which column do you mean?" in t for t in texts(at.info))
    assert at.get("download_button") == []


def test_run_passes_the_sidebar_settings_to_the_session():
    fake = FakeSession(result={"answer": None, "clarification": "?", "error": None, "results": [], "task_status": []})
    run_with(fake)
    question, file, sheet, retries, model = fake.submitted[0]
    assert (question, file, sheet, retries, model) == ("total amount", "sales.xlsx", "orders", 2, "openai/gpt-4o-mini")


def test_a_failed_job_shows_a_clear_message_with_the_hint_and_no_traceback():
    at = run_with(FakeSession(error="OpenRouter: no credits (402). Add credits to your OpenRouter account.",
                              hint="OpenRouter: no credits (402). Add credits to your OpenRouter account."))
    assert not at.exception
    message = "\n".join(texts(at.error))
    assert "no credits" in message and "Fix:" in message and "Traceback" not in message and FAKE_KEY not in message


def test_a_timeout_is_a_clear_message():
    at = run_with(FakeSession(error="The run took longer than 600 seconds and was stopped."))
    assert any("took longer than 600 seconds" in t for t in texts(at.error))


def test_downloads_stay_locked_until_the_review_box_is_ticked_and_fallbacks_are_shown(tmp_path):
    from agent.report_doc import KeyFigureRow, ReportDoc, to_markdown
    doc = ReportDoc(title="Analysis report: sales.xlsx", question="q", summary="Total is 1,006.00.",
                    key_figures=[KeyFigureRow("F1", "sum of amount", "1,006.00")], caveats=["c"])
    report = {"markdown": to_markdown(doc), "doc": doc, "path": str(tmp_path / "report_20261012_100000.md"),
              "fallbacks": ["section 1: generated without the LLM (number check failed)"]}
    result = {"answer": "Total is 1,006.00.", "clarification": None, "error": None, "report": report, "results": [],
              "task_status": [{"idx": 0, "agent": "analysis", "task": "Sum amount.", "status": "failed", "reasons": ["boom"]}]}
    at = run_with(FakeSession(result=result))
    assert not at.exception
    assert any("1,006.00" in str(m.value) for m in at.markdown)
    assert any("Number check: section 1: generated without the LLM" in t for t in texts(at.warning))
    assert any("Task 1 failed" in t and "boom" in t for t in texts(at.warning))
    downloads = at.get("download_button")
    assert len(downloads) == 2 and all(d.proto.disabled for d in downloads)
    at.checkbox(key="reviewed").check().run()
    assert not any(d.proto.disabled for d in at.get("download_button"))
    names = {d.proto.label for d in at.get("download_button")}
    assert names == {"Download DOCX", "Download Markdown"}
