"""SessionManager: max sessions, idle timeout, orphan-folder cleanup. Real excel_server over stdio, fake clock."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from agent.session import SessionError
from agent.session_manager import SessionManager, settings_from_env
from core import Config


class Clock:
    def __init__(self) -> None:
        self.offset = 0.0

    def __call__(self) -> float:
        return time.time() + self.offset


@pytest.fixture()
def env(tmp_path: Path):
    config = Config(data_dir=tmp_path / "data", output_dir=tmp_path / "out")
    clock = Clock()
    manager = SessionManager(config, idle_seconds=600, max_sessions=2, clock=clock)
    yield manager, config, clock
    for sid in list(manager._live.values()):
        sid.close(cleanup=True)


def test_env_defaults_and_overrides() -> None:
    assert settings_from_env({}) == (30 * 60.0, 5)
    assert settings_from_env({"DATAPILOT_IDLE_MINUTES": "2", "DATAPILOT_MAX_SESSIONS": "9"}) == (120.0, 9)
    assert settings_from_env({"DATAPILOT_IDLE_MINUTES": "x", "DATAPILOT_MAX_SESSIONS": "-1"}) == (1800.0, 5)


def test_max_sessions_then_slot_frees_on_release(env) -> None:
    manager, _, _ = env
    a, b = manager.create(), manager.create()
    with pytest.raises(SessionError, match="busy"):
        manager.create()
    manager.release(a)
    c = manager.create()
    assert manager.count() == 2 and c.running and b.running


def test_idle_session_is_closed_and_folders_deleted(env) -> None:
    manager, _, clock = env
    s = manager.create()
    s.save_upload("a.csv", b"x,y\n1,2\n")
    assert s.data_dir.exists()
    clock.offset = 601
    s.last_active = time.time() - 601  # last user action 601 s ago on the fake clock
    assert manager.reap() == [s.session_id]
    assert s.expired and not s.running and not s.data_dir.exists() and not s.output_dir.exists()
    assert manager.count() == 0


def test_active_session_is_kept(env) -> None:
    manager, _, clock = env
    s = manager.create()
    clock.offset = 300
    assert manager.reap() == [] and s.running


def test_busy_session_is_never_idle(env) -> None:
    manager, _, clock = env
    s = manager.create()
    s._jobs_open = 1  # a question is running
    clock.offset = 99999
    s.last_active = time.time() - 99999
    assert manager.reap() == [] and s.running
    s._jobs_open = 0


def test_expired_slot_is_reused_by_create(env) -> None:
    manager, _, clock = env
    a, b = manager.create(), manager.create()
    clock.offset = 700
    a.last_active = b.last_active = time.time() - 700
    c = manager.create()  # reaps a and b first, so no "busy"
    assert c.running and manager.count() == 1


def test_sweep_removes_only_old_orphan_session_folders(env) -> None:
    manager, config, clock = env
    live = manager.create()
    old = "0123456789ab"
    for root in (config.data_dir, config.output_dir):
        d = Path(root) / old
        d.mkdir(parents=True, exist_ok=True)
        (d / "f.csv").write_text("x")
        past = time.time() - 5000
        os.utime(d / "f.csv", (past, past)); os.utime(d, (past, past))
    fresh = Path(config.data_dir) / "ffffffffffff"
    fresh.mkdir()
    keep_file = Path(config.data_dir) / "mine.csv"
    keep_file.write_text("a")
    other = Path(config.data_dir) / "not-a-session"
    other.mkdir()
    past = time.time() - 5000
    os.utime(other, (past, past))
    removed = {p.name for p in manager.sweep_folders()}
    assert removed == {old}
    assert fresh.exists() and keep_file.exists() and other.exists() and live.data_dir.exists()


def test_dead_session_is_dropped_from_registry(env) -> None:
    manager, _, _ = env
    s = manager.create()
    s.close()
    assert manager.reap() == [s.session_id] and manager.count() == 0
