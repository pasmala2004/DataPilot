"""A reusable agent session for UIs that are not async (Streamlit reruns its script on every click).

``AgentSession`` owns

* ONE background thread running its own asyncio event loop,
* ONE open ``excel_session`` (one MCP server process) for the whole life of the session, so the
  ``dataset_id``s the server hands out stay valid across many questions, and
* its own folders ``DATA_DIR/<session_id>/`` and ``OUTPUT_DIR/<session_id>/``: two sessions never see
  each other's files (each has its own server process confined to its own folders).

Typical use (from any thread, no ``asyncio.run``)::

    session = AgentSession()
    session.start()
    name = session.save_upload("sales.xlsx", data)       # safe file name inside the session folder
    sheets = session.list_sheets(name)
    job = session.submit("total by region", name, "orders", model="llama3.1")
    while not job.done():
        show(job.drain()); time.sleep(0.5)
    result = job.result()                                # run_agent result (+ "report")
    session.close()

No signals and no fork: only threads and an asyncio loop, so it works on Windows and Linux. The MCP session
is opened and closed by ONE long-lived task (anyio cancel scopes must be entered and left in the same
task); jobs run one after another inside that task. Nothing here prints to stdout.

What the objects expose: trace lines (no data rows; the loop never traces cell values), the run result and
report (figures, labels, group names, and ``run_pandas`` printed output, which the report's Sources appendix
shows), and the schema from ``describe`` (column names, types, null counts, date ranges; no data rows).
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import queue
import re
import shutil
import threading
import time
import unicodedata
import uuid
import weakref
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel

from agent.graph import parse_tool_result, run_agent
from agent.llm import LLMConfigError, describe_exception, get_llm, hint_for_error
from agent.mcp_client import ExcelTools, excel_session
from agent.observe import DEFAULT_MAX_RETRIES
from agent.report import build_report
from core import Config

ALLOWED_SUFFIXES = (".xlsx", ".csv")
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 600
START_TIMEOUT_SECONDS = 60
CALL_TIMEOUT_SECONDS = 120
CLOSE_TIMEOUT_SECONDS = 20
SCHEMA_KEYS = ("rows", "columns", "nulls", "date_ranges", "warnings")

LLMFactory = Callable[..., BaseChatModel]


class SessionError(RuntimeError):
    """The session could not start, or is not running. The message is safe to show to the user."""


class UploadError(ValueError):
    """An uploaded file was refused (name, extension or size). The message is safe to show to the user."""


def _default_llm_factory(model: Optional[str] = None) -> BaseChatModel:
    return get_llm(model)  # no per-job check: the app has a check button, the CLI and eval check once up front


WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
RESERVED_CHARS = set('<>:"|?*')
MAX_STEM_CHARS = 100


def safe_upload_name(name: str) -> str:
    """Reduce an uploaded file name to a safe plain name (no folders), or raise ``UploadError``.

    Unicode letters, marks and digits are kept (so ``نتائج.xlsx`` works), plus spaces and ``_ - . ( )``.
    Folders are stripped (both ``/`` and ``\\``); control and format characters (including bidi overrides) and
    the Windows-reserved characters ``< > : " | ? *`` are removed; any other symbol becomes ``_``. An empty
    stem becomes ``upload``; a Windows device name (``CON``, ``NUL``, ``COM1``...) gets a trailing ``_``.
    Only ``.xlsx`` / ``.csv`` are accepted: ``..``, ``/``, ``evil.exe`` and names with NUL are refused.
    """
    if not isinstance(name, str) or "\x00" in name:
        raise UploadError("The file name is not valid.")
    base = unicodedata.normalize("NFC", re.split(r"[\\/]", name.strip())[-1]).strip()
    kept: list[str] = []
    for char in base:
        category = unicodedata.category(char)
        if category[0] in "CZ" and char != " ":  # control, format (bidi marks), line/paragraph separators, odd spaces
            continue
        if char in RESERVED_CHARS:
            continue
        kept.append(char if category[0] in "LMN" or char in " _-.()" else "_")
    cleaned = "".join(kept).strip().rstrip(" .")
    suffix = cleaned[cleaned.rfind("."):].lower() if "." in cleaned else ""
    if suffix not in ALLOWED_SUFFIXES:
        raise UploadError("Only .xlsx and .csv files are supported.")
    stem = cleaned[: -len(suffix)].strip(" .")[:MAX_STEM_CHARS].strip(" .") or "upload"
    if stem.split(".")[0].strip().upper() in WINDOWS_RESERVED:
        stem += "_"
    return stem + suffix


class CallCounter(BaseCallbackHandler):
    """Counts chat-model calls (one per request the agent makes, not the client's transport retries)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.count = 0

    def on_chat_model_start(self, *args: Any, **kwargs: Any) -> None:
        with self._lock:
            self.count += 1

    def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
        pass  # chat models fire on_chat_model_start; counting both would double count


class Job:
    """One question running in a session. Thread-safe: poll it from the UI thread.

    ``drain()`` returns the trace lines that arrived since the last call; ``done()`` says whether the run ended;
    ``result()`` is the ``run_agent`` result (plus ``"report"``) once done, else ``None``; ``error`` is a
    user-readable message when the job itself failed (timeout, model not reachable, crash), else ``None``;
    ``hint`` is a fix-it line when there is one.
    """

    def __init__(self) -> None:
        self._trace: "queue.Queue[str]" = queue.Queue()
        self._done = threading.Event()
        self._lock = threading.Lock()
        self._result: Optional[dict[str, Any]] = None
        self._error: Optional[str] = None
        self._hint: Optional[str] = None
        self._started = time.monotonic()
        self._finished: Optional[float] = None
        self._on_finish: Optional[Callable[[], None]] = None
        self._counter: Optional[CallCounter] = None

    # --- called by the session (loop thread) -------------------------------------------------
    def _emit(self, line: str) -> None:
        self._trace.put(str(line))

    def _finish(self, result: Optional[dict[str, Any]] = None, error: Optional[str] = None, hint: Optional[str] = None) -> None:
        with self._lock:
            self._result, self._error, self._hint = result, error, hint or None
            already = self._done.is_set()
            self._finished = time.monotonic()
        self._done.set()
        if self._on_finish is not None and not already:
            self._on_finish()

    # --- called by the UI thread --------------------------------------------------------------
    def drain(self) -> list[str]:
        """New trace lines since the last call (possibly empty)."""
        lines: list[str] = []
        while True:
            try:
                lines.append(self._trace.get_nowait())
            except queue.Empty:
                return lines

    def done(self) -> bool:
        return self._done.is_set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until done (or the timeout); returns ``done()``."""
        return self._done.wait(timeout)

    def result(self) -> Optional[dict[str, Any]]:
        with self._lock:
            return self._result

    @property
    def error(self) -> Optional[str]:
        with self._lock:
            return self._error

    @property
    def hint(self) -> Optional[str]:
        with self._lock:
            return self._hint

    @property
    def llm_calls(self) -> int:
        """Chat-model calls made by this job so far (0 before the model was built)."""
        return self._counter.count if self._counter is not None else 0

    def elapsed(self) -> float:
        """Seconds since the job was submitted (frozen once it is done)."""
        with self._lock:
            end = self._finished if self._finished is not None else time.monotonic()
        return end - self._started


_LIVE: "weakref.WeakSet[AgentSession]" = weakref.WeakSet()


@atexit.register
def _close_all_sessions() -> None:  # pragma: no cover - runs at interpreter exit
    for session in list(_LIVE):
        try:
            session.close()
        except Exception:  # noqa: BLE001
            pass


class AgentSession:
    """See the module docstring. Create, ``start()``, use from any thread, ``close()``."""

    def __init__(
        self,
        config: Optional[Config] = None,
        *,
        session_id: Optional[str] = None,
        llm_factory: Optional[LLMFactory] = None,
        server_script: Optional[str | Path] = None,
    ) -> None:
        base = config or Config()
        self.session_id = session_id or uuid.uuid4().hex[:12]
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", self.session_id):
            raise ValueError("session_id may only contain letters, digits, '_' and '-'")
        self.data_dir = Path(base.data_dir).resolve() / self.session_id
        self.output_dir = Path(base.output_dir).resolve() / self.session_id
        self._llm_factory: LLMFactory = llm_factory or _default_llm_factory
        self._server_script = server_script
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._jobs: Optional["asyncio.Queue[Any]"] = None
        self._current: Optional["asyncio.Future[None]"] = None
        self._closing = False
        self._tools: Optional[ExcelTools] = None
        self._ready = threading.Event()
        self._start_error: Optional[BaseException] = None
        self._closed = False
        self._lock = threading.Lock()
        self.last_active: float = time.time()  # wall-clock seconds of the last user action (see touch())
        self.expired = False                    # set by SessionManager when it closed this session for being idle
        self._jobs_open = 0                     # jobs submitted and not finished

    def touch(self, now: Optional[float] = None) -> None:
        """Record user activity (uploads, schema reads, questions)."""
        self.last_active = time.time() if now is None else now

    @property
    def busy(self) -> bool:
        """True while a question is queued or running: such a session is never treated as idle."""
        return self._jobs_open > 0

    # ------------------------------------------------------------------ life cycle
    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive() and self._tools is not None and not self._closed)

    def start(self) -> "AgentSession":
        """Create the folders, start the loop thread and open the MCP session. Safe to call twice."""
        with self._lock:
            if self._closed:
                raise SessionError("This session was closed; create a new one.")
            if self._thread is not None:
                return self
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._thread = threading.Thread(target=self._thread_main, name=f"agent-session-{self.session_id}", daemon=True)
            self._thread.start()
        if not self._ready.wait(START_TIMEOUT_SECONDS):
            raise SessionError("The analysis server did not start in time.")
        if self._start_error is not None:
            raise SessionError(f"The analysis server could not start: {type(self._start_error).__name__}: {str(self._start_error)[:300]}")
        _LIVE.add(self)
        return self

    def close(self, *, cleanup: bool = False) -> None:
        """Stop the server process and the loop thread. Safe to call twice.

        Args:
            cleanup: also delete this session's data and output folders.
        """
        with self._lock:
            already = self._closed
            self._closed = True
        thread, loop, jobs = self._thread, self._loop, self._jobs
        if not already and thread is not None and thread.is_alive() and loop is not None and jobs is not None:
            try:
                loop.call_soon_threadsafe(self._stop)
            except RuntimeError:  # the loop is already closed
                pass
            thread.join(CLOSE_TIMEOUT_SECONDS)
        _LIVE.discard(self)
        if cleanup:
            for folder in (self.data_dir, self.output_dir):
                shutil.rmtree(folder, ignore_errors=True)

    def _job_finished(self) -> None:
        with self._lock:
            self._jobs_open = max(0, self._jobs_open - 1)
        self.touch()

    def _stop(self) -> None:
        """(loop thread) Cancel the running job, if any, and tell the main task to finish."""
        self._closing = True
        if self._current is not None and not self._current.done():
            self._current.cancel()
        assert self._jobs is not None
        self._jobs.put_nowait(None)

    def __enter__(self) -> "AgentSession":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        try:
            loop.run_until_complete(self._main())
        except BaseException as exc:  # noqa: BLE001 - reported through start()
            if not self._ready.is_set():
                self._start_error = exc
                self._ready.set()
        finally:
            self._ready.set()
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:  # noqa: BLE001
                pass
            loop.close()

    async def _main(self) -> None:
        """The ONE long-lived task: opens the MCP session, runs queued jobs in order, closes the session."""
        self._jobs = asyncio.Queue()
        async with excel_session(self.data_dir, self.output_dir, server_script=self._server_script) as tools:
            self._tools = tools
            self._ready.set()
            while True:
                item = await self._jobs.get()
                if item is None or self._closing:
                    if item is not None:
                        item[0]._finish(error="The session was closed before this run started.")
                    break
                self._current = asyncio.ensure_future(self._run_job(*item))
                try:
                    await self._current
                except asyncio.CancelledError:  # close() cancelled the running job; the loop ends next
                    pass
                finally:
                    self._current = None
        self._tools = None
        while not self._jobs.empty():  # jobs still waiting when the session closed
            item = self._jobs.get_nowait()
            if item is not None:
                item[0]._finish(error="The session was closed before this run started.")

    # ------------------------------------------------------------------ files
    def save_upload(self, name: str, data: bytes) -> str:
        """Save an uploaded file in the session's data folder and return its safe file name.

        Raises:
            UploadError: bad name, extension other than .xlsx/.csv, empty file, or more than 25 MB.
        """
        self.touch()
        safe = safe_upload_name(name)
        if not isinstance(data, (bytes, bytearray)) or len(data) == 0:
            raise UploadError("The file is empty.")
        if len(data) > MAX_UPLOAD_BYTES:
            raise UploadError(f"The file is larger than the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        target = (self.data_dir / safe).resolve()
        if target.parent != self.data_dir:  # belt and braces: never write outside the session folder
            raise UploadError("The file name is not valid.")
        target.write_bytes(bytes(data))
        return safe

    def files(self) -> list[str]:
        """Names of the uploaded files in this session."""
        if not self.data_dir.is_dir():
            return []
        return sorted(p.name for p in self.data_dir.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_SUFFIXES)

    def _call(self, coroutine_factory: Callable[[ExcelTools], Any], timeout: float = CALL_TIMEOUT_SECONDS) -> Any:
        if not self.running or self._loop is None or self._tools is None:
            raise SessionError("The session is not running.")
        self.touch()
        future = asyncio.run_coroutine_threadsafe(coroutine_factory(self._tools), self._loop)
        try:
            return future.result(timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise SessionError("The analysis server did not answer in time.") from None

    def list_sheets(self, file: str) -> list[str]:
        """Sheet names of an uploaded file (``["csv"]`` for a CSV file).

        Raises:
            SessionError: the file cannot be opened; the message is the tool's own error.
        """
        async def go(tools: ExcelTools) -> dict[str, Any]:
            return parse_tool_result(await tools.by_name["list_sheets"].ainvoke(
                {"name": "list_sheets", "args": {"path": file}, "id": "ui-sheets", "type": "tool_call"}))

        reply = self._call(go)
        if "error" in reply:
            raise SessionError(f"Cannot open '{file}': {reply['error']}")
        return [str(s["name"]) for s in reply.get("sheets", [])]

    def describe(self, file: str, sheet: Optional[str] = None) -> dict[str, Any]:
        """Schema of one sheet: rows, columns {name: dtype}, nulls, date_ranges, warnings. NEVER data rows."""
        async def go(tools: ExcelTools) -> dict[str, Any]:
            args = {"path": file, "sheet": sheet}
            return parse_tool_result(await tools.by_name["load_data"].ainvoke(
                {"name": "load_data", "args": args, "id": "ui-describe", "type": "tool_call"}))

        reply = self._call(go)
        if "error" in reply:
            raise SessionError(f"Cannot load '{file}': {reply['error']}")
        return {k: reply[k] for k in SCHEMA_KEYS if k in reply}

    # ------------------------------------------------------------------ jobs
    def submit(
        self,
        question: str,
        file: str,
        sheet: Optional[str] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        model: Optional[str] = None,
        make_report: bool = True,
        timeout_s: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> Job:
        """Queue one question; returns at once with a ``Job`` to poll. Jobs of a session run one after another."""
        if not self.running or self._loop is None or self._jobs is None:
            raise SessionError("The session is not running.")
        self.touch()
        job = Job()
        with self._lock:
            self._jobs_open += 1
        job._on_finish = self._job_finished
        params = (job, question, file, sheet, max_retries, model, make_report, timeout_s)
        self._loop.call_soon_threadsafe(self._jobs.put_nowait, params)
        return job

    async def _run_job(
        self, job: Job, question: str, file: str, sheet: Optional[str], max_retries: int,
        model: Optional[str], make_report: bool, timeout_s: float,
    ) -> None:
        """Run one job inside the long-lived task. Never raises: every failure ends up in ``job.error``."""
        try:
            assert self._tools is not None
            loop = asyncio.get_running_loop()
            llm = await loop.run_in_executor(None, partial(self._llm_factory, model=model))
            counter = CallCounter()
            try:
                llm.callbacks = [*(llm.callbacks or []), counter]  # counts the model calls of THIS job
            except (AttributeError, TypeError, ValueError):
                pass
            job._counter = counter

            async def work() -> dict[str, Any]:
                result = await run_agent(question, file, llm=llm, tools=self._tools, sheet=sheet,
                                         max_retries=max_retries, trace=job._emit)
                if make_report and not result.get("error") and not result.get("clarification"):
                    job._emit("REPORT building the report")
                    result = {**result, "report": await build_report(result, llm, output_dir=self.output_dir)}
                return result

            result = await asyncio.wait_for(work(), timeout_s)
            error = result.get("error")
            job._finish(result=result, error=None, hint=hint_for_error(str(error)) if error else None)
        except asyncio.TimeoutError:
            job._finish(error=f"The run took longer than {int(timeout_s)} seconds and was stopped. "
                              "A slow local model may need more time (the first answer loads the model).")
        except LLMConfigError as exc:
            job._finish(error=str(exc))
        except asyncio.CancelledError:
            job._finish(error="The run was cancelled because the session was closed.")
            raise
        except Exception as exc:  # noqa: BLE001
            message = describe_exception(exc)
            job._finish(error=message, hint=hint_for_error(message))
