"""DataPilot: a verified Excel analyst. Run with ``streamlit run app.py``.

Thin UI. The logic lives in agent/session.py (background loop + one MCP session), agent/docx_export.py,
agent/llm.py (OpenRouter) and agent/ui_helpers.py. Streamlit reruns this script on every click, so everything
that must survive (the session, the file, the running job and the result) is kept in ``st.session_state`` and
a click on a download button never re-runs the agent.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

# Load .env file at startup (before any other imports that might use env vars)
load_dotenv(override=False)

import pandas as pd
import streamlit as st

from agent.docx_export import DOCX_MIME, to_docx
from agent.llm import LLMConfigError, check_openrouter, describe_exception, key_present, resolve_settings
from agent.session import AgentSession, SessionError, UploadError
from agent.session_manager import SessionManager
from agent.ui_helpers import (
    CURATED_MODELS, CUSTOM_MODEL_LABEL, PRIVACY_CHECKBOX, PRIVACY_NOTICE, effective_model, format_elapsed,
    problem_messages, run_block_reason, schema_rows, split_for_preview,
)

POLL_SECONDS = 1.0
STATE_DEFAULTS: dict[str, Any] = {
    "session": None, "upload_id": None, "file": None, "sheets": [], "schema": None, "schema_key": None,
    "job": None, "trace": [], "result": None, "job_error": None, "job_hint": None, "run_info": None,
    "docx": None, "reviewed": False, "model_check": None, "privacy_ack": False,
}


def init_state() -> None:
    for key, value in STATE_DEFAULTS.items():
        st.session_state.setdefault(key, list(value) if isinstance(value, list) else value)


@st.cache_resource
def session_manager() -> SessionManager:
    """One manager per server process: session limit, idle timeout, cleanup of expired folders."""
    manager = SessionManager()
    manager.start_reaper()
    return manager


def get_session() -> AgentSession:
    """The user's session (created and started on first use, restarted if it died or expired)."""
    session = st.session_state.session
    if session is None or not session.running:
        session = session_manager().create()
        st.session_state.session = session
        st.session_state.update(upload_id=None, file=None, sheets=[], schema=None, schema_key=None)
    return session


def expiry_notice() -> None:
    """Tell the user when their session was closed for being idle (their files were deleted)."""
    old = st.session_state.get("session")
    if old is not None and not old.running and getattr(old, "expired", False):
        minutes = int(session_manager().idle_seconds // 60)
        st.info(f"Your session was closed after {minutes} minutes without activity and your uploaded files were deleted. "
                "Upload the file again to continue.")


def reset_session() -> None:
    old = st.session_state.get("session")
    if old is not None:
        session_manager().release(old)
    for key, value in STATE_DEFAULTS.items():
        st.session_state[key] = list(value) if isinstance(value, list) else value


# --------------------------------------------------------------------------- sidebar
def run_model_check(model: str) -> None:
    """Button handler: confirm the key and the model id with OpenRouter and remember the outcome (never the key)."""
    try:
        warning = check_openrouter(resolve_settings(model))
        outcome = {"model": model, "ok": True, "text": "ready", "warning": warning}
    except LLMConfigError as exc:
        outcome = {"model": model, "ok": False, "text": str(exc), "warning": None}
    except Exception as exc:  # noqa: BLE001 - never show a traceback
        outcome = {"model": model, "ok": False, "text": describe_exception(exc), "warning": None}
    st.session_state.model_check = outcome


def sidebar() -> dict[str, Any]:
    """Draw the settings and return ``{model, max_retries, key_found, model_error}``."""
    with st.sidebar:
        st.header("Model (OpenRouter)")
        key_found = key_present()
        (st.success if key_found else st.error)("API key: found" if key_found else "API key: not found")
        if not key_found:
            st.caption("Set it before starting the app (the key is never shown here):")
            st.code('export OPENROUTER_API_KEY=...          # bash\n$env:OPENROUTER_API_KEY = "..."      # PowerShell', language=None)
        default = os.environ.get("OPENROUTER_MODEL") or os.environ.get("MODEL_NAME") or CURATED_MODELS[0]
        options = list(CURATED_MODELS) + ([default] if default not in CURATED_MODELS else [])  # env id first-class
        selected = st.selectbox("Model", options, index=options.index(default))
        custom = st.text_input(CUSTOM_MODEL_LABEL, placeholder="provider/model-name")
        model = effective_model(selected, custom)
        check = st.session_state.model_check
        if st.button("Check key and model", disabled=not key_found):
            run_model_check(model)
            check = st.session_state.model_check
        model_error = None
        if check and check["model"] == model:
            if check["ok"]:
                st.success(f"Model: ready ({model})")
                if check.get("warning"):
                    st.warning(check["warning"])
            else:
                model_error = check["text"]
                st.error(f"Model: error. {model_error}")
        else:
            st.caption("Model: not checked yet")
        max_retries = st.slider("Max retries per task", 0, 3, 2)
        st.divider()
        if st.button("Start a new session", help="Closes the analysis server and deletes the uploaded files"):
            reset_session()
            st.rerun()
    return {"model": model, "max_retries": max_retries, "key_found": key_found, "model_error": model_error}


# --------------------------------------------------------------------------- 1) upload and schema
def upload_section() -> None:
    st.subheader("1. Your file")
    uploaded = st.file_uploader("Excel or CSV (max 25 MB)", type=["xlsx", "csv"])
    if uploaded is not None and uploaded.file_id != st.session_state.upload_id:
        try:
            session = get_session()
            name = session.save_upload(uploaded.name, uploaded.getvalue())
            st.session_state.update(file=name, sheets=session.list_sheets(name), schema=None, schema_key=None,
                                    upload_id=uploaded.file_id, result=None, docx=None, reviewed=False)
        except (UploadError, SessionError) as exc:
            st.session_state.update(file=None, sheets=[], schema=None, upload_id=None)
            st.error(str(exc))
            return
    if not st.session_state.file:
        st.caption("Upload a file to start. It stays in this session's private folder on this machine; only a summary goes to OpenRouter.")
        return
    sheet = st.selectbox("Sheet", st.session_state.sheets)
    key = (st.session_state.file, sheet)
    if st.session_state.schema_key != key:
        try:
            st.session_state.schema = get_session().describe(st.session_state.file, sheet)
            st.session_state.schema_key = key
        except SessionError as exc:
            st.session_state.schema = None
            st.error(str(exc))
            return
    schema = st.session_state.schema
    st.caption(f"{st.session_state.file}: {schema.get('rows', 0):,} rows. Schema only, no data rows are shown.")
    st.dataframe(pd.DataFrame(schema_rows(schema)), hide_index=True, width="stretch")
    for warning in schema.get("warnings") or []:
        st.warning(str(warning))


# --------------------------------------------------------------------------- 2) request + run
def request_section(settings: dict[str, Any]) -> None:
    st.subheader("2. Your request")
    running = st.session_state.job is not None and not st.session_state.job.done()
    st.warning(PRIVACY_NOTICE)
    st.checkbox(PRIVACY_CHECKBOX, key="privacy_ack", disabled=running)
    question = st.text_area("What do you want to know?", placeholder="e.g. total sales by region, and a bar chart", disabled=running)
    reason = run_block_reason(
        key_found=settings["key_found"], model_error=settings["model_error"], privacy_ack=bool(st.session_state.privacy_ack),
        has_file=bool(st.session_state.file), has_question=bool(question.strip()), running=running)
    if st.button("Run", type="primary", disabled=reason is not None):
        sheet = st.session_state.schema_key[1] if st.session_state.schema_key else None
        st.session_state.update(trace=[], result=None, job_error=None, job_hint=None, docx=None, reviewed=False,
                                run_info={"model": settings["model"]})
        try:
            st.session_state.job = get_session().submit(
                question.strip(), st.session_state.file, sheet, settings["max_retries"], settings["model"])
        except SessionError as exc:
            st.session_state.job_error = str(exc)
        st.rerun()
    if reason and not running:
        st.caption(reason)


# --------------------------------------------------------------------------- 3) live trace
@st.fragment(run_every=POLL_SECONDS)
def trace_panel() -> None:
    """Polls the job's trace queue; only this fragment reruns while the job works, other widgets stay usable."""
    job = st.session_state.job
    if job is None:
        return
    st.session_state.trace += job.drain()
    state = "Working" if not job.done() else "Finished"
    st.caption(f"{state} - elapsed {format_elapsed(job.elapsed())}")
    if st.session_state.trace:
        st.code("\n".join(st.session_state.trace[-200:]), language=None)
    if job.done():
        finish_job()
        st.rerun()


def finish_job() -> None:
    """Move a finished job's outcome into session state (once) and build the DOCX bytes."""
    job = st.session_state.job
    st.session_state.trace += job.drain()
    result = job.result()
    st.session_state.update(result=result, job_error=job.error, job_hint=job.hint, job=None)
    report = (result or {}).get("report")
    if report:
        st.session_state.docx = to_docx(report["doc"])


def trace_section() -> None:
    if st.session_state.job is not None:
        st.subheader("3. Progress")
        trace_panel()
    elif st.session_state.trace:
        with st.expander("Trace of the last run"):
            st.code("\n".join(st.session_state.trace[-200:]), language=None)


# --------------------------------------------------------------------------- 4) result
def result_section() -> None:
    result, info = st.session_state.result, st.session_state.run_info or {}
    for level, text in problem_messages(st.session_state.job_error, st.session_state.job_hint, result,
                                        info.get("model")):
        getattr(st, level)(text)
    if not result:
        return
    if result.get("clarification"):
        st.info(f"I need a bit more detail: {result['clarification']}")
        return
    if result.get("error"):
        return
    st.subheader("4. Result")
    st.markdown(result.get("answer") or "(no answer)")
    report = result.get("report")
    if not report:
        return
    with st.expander("Report preview", expanded=True):
        for kind, content in split_for_preview(report["markdown"], report["doc"].charts):
            if kind == "image":
                st.image(content)
            else:
                st.markdown(content)
    reviewed = st.checkbox("I reviewed this report", key="reviewed")
    stem = Path(report["path"]).stem
    left, right = st.columns(2)
    left.download_button("Download DOCX", st.session_state.docx or b"", file_name=f"{stem}.docx", mime=DOCX_MIME, disabled=not reviewed)
    right.download_button("Download Markdown", report["markdown"].encode("utf-8"), file_name=f"{stem}.md",
                          mime="text/markdown", disabled=not reviewed)
    if not reviewed:
        st.caption("Tick the box to enable the downloads.")


def main() -> None:
    st.set_page_config(page_title="DataPilot", page_icon=None, layout="wide")
    init_state()
    st.title("DataPilot")
    st.caption("Ask a question about an Excel or CSV file. Every number in the answer is computed by a tool, not by the model.")
    settings = sidebar()
    expiry_notice()
    upload_section()
    request_section(settings)
    trace_section()
    result_section()


main()
