"""Limits for many browser sessions: a maximum number, an idle timeout, and cleanup of expired folders.

Streamlit gives no reliable "the tab was closed" signal, so sessions would otherwise live until the process ends.
``SessionManager`` keeps the live ``AgentSession`` objects and

* refuses a new session when ``max_sessions`` are live (after reaping expired ones first),
* closes (and deletes the folders of) sessions idle longer than ``idle_seconds``; a session with a job
  running is never idle,
* sweeps orphan folders ``DATA_DIR/<id>`` / ``OUTPUT_DIR/<id>`` that no live session owns and that have not been
  touched for ``idle_seconds`` (left behind by a crash). Only folders named like a session id (12 hex chars) are
  ever deleted, so user files and ``.gitkeep`` are safe.

Environment: ``DATAPILOT_IDLE_MINUTES`` (default 30), ``DATAPILOT_MAX_SESSIONS`` (default 5).
"""

from __future__ import annotations

import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from agent.session import AgentSession, SessionError
from core import Config

DEFAULT_IDLE_MINUTES = 30.0
DEFAULT_MAX_SESSIONS = 5
SESSION_DIR = re.compile(r"[0-9a-f]{12}")


def settings_from_env(env: Optional[Mapping[str, str]] = None) -> tuple[float, int]:
    """``(idle_seconds, max_sessions)`` from the environment; bad values fall back to the defaults."""
    env = os.environ if env is None else env
    try:
        idle = float(env.get("DATAPILOT_IDLE_MINUTES") or DEFAULT_IDLE_MINUTES)
    except ValueError:
        idle = DEFAULT_IDLE_MINUTES
    try:
        limit = int(env.get("DATAPILOT_MAX_SESSIONS") or DEFAULT_MAX_SESSIONS)
    except ValueError:
        limit = DEFAULT_MAX_SESSIONS
    return (idle if idle > 0 else DEFAULT_IDLE_MINUTES) * 60.0, (limit if limit > 0 else DEFAULT_MAX_SESSIONS)


def _newest_mtime(folder: Path) -> float:
    newest = folder.stat().st_mtime
    for path in folder.rglob("*"):
        try:
            newest = max(newest, path.stat().st_mtime)
        except OSError:
            pass
    return newest


class SessionManager:
    """Thread-safe registry of live sessions. ``clock`` is injectable for tests (wall-clock seconds)."""

    def __init__(
        self,
        config: Optional[Config] = None,
        *,
        idle_seconds: Optional[float] = None,
        max_sessions: Optional[int] = None,
        factory: Optional[Callable[[], AgentSession]] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        env_idle, env_max = settings_from_env()
        self.config = config or Config()
        self.idle_seconds = env_idle if idle_seconds is None else idle_seconds
        self.max_sessions = env_max if max_sessions is None else max_sessions
        self._factory = factory or (lambda: AgentSession(self.config))
        self._clock = clock
        self._lock = threading.Lock()
        self._live: dict[str, AgentSession] = {}
        self._reaper: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # ------------------------------------------------------------------ public
    def create(self) -> AgentSession:
        """Start a new session, or raise ``SessionError`` ("busy") when the limit is reached."""
        self.reap()
        with self._lock:
            if len(self._live) >= self.max_sessions:
                raise SessionError(
                    f"The server is busy: {self.max_sessions} sessions are already open. "
                    "Try again in a few minutes (idle sessions close after "
                    f"{int(self.idle_seconds // 60)} minutes)."
                )
            session = self._factory()
            self._live[session.session_id] = session
        try:
            session.start()
        except Exception:
            with self._lock:
                self._live.pop(session.session_id, None)
            session.close(cleanup=True)
            raise
        session.touch(self._clock())
        return session

    def release(self, session: AgentSession) -> None:
        """Close a session and delete its folders (the user pressed "Start a new session")."""
        with self._lock:
            self._live.pop(session.session_id, None)
        session.close(cleanup=True)

    def count(self) -> int:
        with self._lock:
            return len(self._live)

    def reap(self) -> list[str]:
        """Close sessions idle too long (or already dead), sweep orphan folders; returns the closed session ids."""
        now = self._clock()
        expired: list[AgentSession] = []
        with self._lock:
            for sid, session in list(self._live.items()):
                dead = not session.running
                idle = (not session.busy) and now - session.last_active > self.idle_seconds
                if dead or idle:
                    session.expired = idle and not dead
                    expired.append(self._live.pop(sid))
        for session in expired:
            session.close(cleanup=True)
        self.sweep_folders()
        return [s.session_id for s in expired]

    def sweep_folders(self) -> list[Path]:
        """Delete session-id folders nobody owns that were untouched for ``idle_seconds``. Returns what was deleted."""
        with self._lock:
            owned = set(self._live)
        removed: list[Path] = []
        now = self._clock()
        for root in (Path(self.config.data_dir), Path(self.config.output_dir)):
            if not root.is_dir():
                continue
            for child in root.iterdir():
                if not (child.is_dir() and not child.is_symlink() and SESSION_DIR.fullmatch(child.name)) or child.name in owned:
                    continue
                try:
                    if now - _newest_mtime(child) > self.idle_seconds:
                        shutil.rmtree(child, ignore_errors=True)
                        removed.append(child)
                except OSError:
                    pass
        return removed

    def start_reaper(self, interval_seconds: float = 60.0) -> None:
        """Run ``reap`` every ``interval_seconds`` in a daemon thread (once per manager)."""
        with self._lock:
            if self._reaper is not None:
                return
            self._reaper = threading.Thread(target=self._reap_loop, args=(interval_seconds,), name="session-reaper", daemon=True)
            self._reaper.start()

    def stop_reaper(self) -> None:
        self._stop.set()

    def _reap_loop(self, interval: float) -> None:
        while not self._stop.wait(interval):
            try:
                self.reap()
            except Exception:  # noqa: BLE001 - the reaper must never die
                pass
