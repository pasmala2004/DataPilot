"""Run the evaluation: every question x max_retries setting x repeat through the real agent (OpenRouter).

    python -m eval.run_eval --name first --limit 4 --repeats 1      # measure the time per run first
    python -m eval.run_eval --name full --settings 0 2 --repeats 3  # then the real run (hours; each run makes several LLM calls)
    python -m eval.run_eval --name full ...                          # same command again resumes
    python -m eval.run_eval --name x --dry-run                       # print the plan and the run count

Each run is ``AgentSession.submit`` (``run_agent`` + ``build_report``) on the dataset's session. One session
(one MCP server) is reused for all runs of a dataset. Settings are interleaved (retries 0, retries 2, 0, 2 ...)
so slow drift of the local model hits both settings alike. Every finished run is appended to
``<results>/<name>/runs.jsonl`` at once, so a crash loses at most the run in flight; a second call skips every
(question, setting, repeat) already in the file. Stored records contain scores and counters, never data rows.

Exit codes: 0 done (or stopped by --limit), 2 bad configuration / model not usable, 130 interrupted.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import pandas as pd

from agent.llm import LLMConfigError, get_llm, is_rate_limit, resolve_model_name, resolve_settings
from agent.session import AgentSession
from core import Config
from eval.questions import DATA_DIR, QUESTIONS, Expect, Question, read_dataset
from eval.scoring import score_run

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_SETTINGS = (0, 2)
DEFAULT_REPEATS = 3
DEFAULT_TIMEOUT_S = 300.0
MAX_CONCURRENCY = 4
RATE_LIMIT_WAITS_S = (30.0, 60.0, 120.0)   # waits before retrying a run that OpenRouter rate limited
# ASSUMPTION, not a measurement: planning range of chat-model calls per run (router + agent steps + report,
# more with retries). The slice run measures the real mean (llm_calls in runs.jsonl); --dry-run prefers that.
CALLS_PER_RUN_RANGE = (4, 20)
_APPEND_LOCK = threading.Lock()
REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class RunKey:
    """One planned run."""

    question_id: str
    setting: int
    repeat: int

    def as_tuple(self) -> tuple[str, int, int]:
        return (self.question_id, self.setting, self.repeat)


# --------------------------------------------------------------------------- #
# Plan and resume
# --------------------------------------------------------------------------- #
def plan_runs(questions: Sequence[Question], settings: Sequence[int], repeats: int) -> list[RunKey]:
    """Questions in order; for each, repeat by repeat, and inside a repeat every setting (interleaved)."""
    return [RunKey(q.id, s, r) for q in questions for r in range(1, repeats + 1) for s in settings]


def read_runs(path: Path) -> list[dict[str, Any]]:
    """Records of a runs.jsonl; a truncated last line (a killed process) is ignored."""
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def append_record(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON line and flush it to disk (crash-safe)."""
    with _APPEND_LOCK:
        _append_locked(path, record)


def _append_locked(path: Path, record: dict[str, Any]) -> None:
    needs_newline = False
    if path.exists() and path.stat().st_size:
        with path.open("rb") as existing:  # a killed process may have left a half line: start on a fresh line
            existing.seek(-1, os.SEEK_END)
            needs_newline = existing.read(1) != b"\n"
    with path.open("a", encoding="utf-8") as handle:
        if needs_newline:
            handle.write("\n")
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def git_commit() -> str:
    """Short hash of HEAD (``unknown`` outside a git checkout)."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def build_meta(model: str, args: argparse.Namespace) -> dict[str, Any]:
    """Facts about the run that a reader needs to interpret the results."""
    return {
        "name": args.name,
        "provider": "openrouter",
        "model": model,
        "git_commit": git_commit(),
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "settings": list(args.settings),
        "repeats": args.repeats,
        "timeout_s": args.timeout_s,
        "concurrency": getattr(args, "concurrency", 1),
    }


def format_duration(seconds: float) -> str:
    """``3725`` -> ``1h 02m``; ``95`` -> ``1m 35s``; ``8`` -> ``8s``."""
    total = int(max(seconds, 0))
    if total >= 3600:
        return f"{total // 3600}h {total % 3600 // 60:02d}m"
    return f"{total // 60}m {total % 60:02d}s" if total >= 60 else f"{total}s"


# --------------------------------------------------------------------------- #
# One run
# --------------------------------------------------------------------------- #
class DatasetSession:
    """One AgentSession per dataset, started lazily and restarted if it died."""

    def __init__(self, config: Config, data_dir: Path, llm_factory: Optional[Callable[..., Any]]) -> None:
        self.config, self.data_dir, self.llm_factory = config, data_dir, llm_factory
        self.session: Optional[AgentSession] = None
        self.file: Optional[str] = None
        self.name: Optional[str] = None

    def get(self, dataset: str) -> tuple[AgentSession, str]:
        if self.file != dataset or self.session is None or not self.session.running:
            self.close()
            session = AgentSession(self.config, session_id=f"eval_{Path(dataset).stem[:40]}", llm_factory=self.llm_factory)
            session.start()
            self.session, self.file = session, dataset
            self.name = session.save_upload(dataset, (self.data_dir / dataset).read_bytes())
        assert self.session is not None and self.name is not None
        return self.session, self.name

    def close(self) -> None:
        if self.session is not None:
            self.session.close()
        self.session, self.file, self.name = None, None, None


def run_one(
    holder: DatasetSession, question: Question, expects: Sequence[Expect], key: RunKey, model: Optional[str], timeout_s: float,
    *, sleep: Callable[[float], None] = time.sleep, rate_limit_waits: Sequence[float] = RATE_LIMIT_WAITS_S,
) -> dict[str, Any]:
    """Run one question once and return its scored record (never raises for a failed run).

    When OpenRouter rate limits the run (429 after the client's own retries), wait and run it again, up to
    ``len(rate_limit_waits)`` times. Only when those retries are exhausted does the record say ``rate_limit``.
    ``llm_calls`` adds up the chat-model calls of all attempts.
    """
    started = time.monotonic()
    result: Optional[dict[str, Any]] = None
    job_error: Optional[str] = None
    timed_out = False
    llm_calls = 0
    for attempt in range(len(rate_limit_waits) + 1):
        result, job_error, timed_out = None, None, False
        try:
            session, name = holder.get(question.dataset)
            job = session.submit(question.text, name, question.sheet, max_retries=key.setting, model=model, timeout_s=timeout_s)
            if not job.wait(timeout_s + 60):
                timed_out = True
                holder.close()  # the job did not end by itself: do not reuse this session
            else:
                result, job_error = job.result(), job.error
                timed_out = bool(job_error and "took longer than" in job_error)
                if timed_out:
                    job_error = None
            llm_calls += job.llm_calls
        except (OSError, RuntimeError, ValueError) as exc:  # session could not start, file missing ...
            job_error = f"{type(exc).__name__}: {str(exc)[:200]}"
            holder.close()
        if not is_rate_limit(job_error or (result or {}).get("error")) or attempt >= len(rate_limit_waits):
            break
        sleep(rate_limit_waits[attempt])
    latency = time.monotonic() - started
    report = (result or {}).get("report")
    record = score_run(question, expects, result, report, job_error=job_error, latency_s=latency, timed_out=timed_out)
    return {
        "question_id": question.id, "dataset": question.dataset, "kind": question.kind,
        "setting": key.setting, "repeat": key.repeat, "model": model, "llm_calls": llm_calls,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _non_negative(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be 0 or more")
    return value


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be 1 or more")
    return value


def _concurrency(text: str) -> int:
    value = int(text)
    if not 1 <= value <= MAX_CONCURRENCY:
        raise argparse.ArgumentTypeError(f"must be between 1 and {MAX_CONCURRENCY}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m eval.run_eval", description="Run the DataPilot evaluation (OpenRouter).")
    parser.add_argument("--name", required=True, help="run name: results go to eval/results/<name>/")
    parser.add_argument("--settings", type=_non_negative, nargs="+", default=list(DEFAULT_SETTINGS),
                        help="max_retries values to compare (default: 0 2)")
    parser.add_argument("--repeats", type=_positive, default=DEFAULT_REPEATS, help="repeats per question and setting (default 3)")
    parser.add_argument("--only", default=None, help="comma-separated question ids, e.g. shop_01_sum,bike_05_period")
    parser.add_argument("--limit", type=_positive, default=None, help="stop after N runs of this call (try --limit 4 --repeats 1 first)")
    parser.add_argument("--timeout-s", dest="timeout_s", type=float, default=DEFAULT_TIMEOUT_S, help="seconds per run (default 300)")
    parser.add_argument("--model", default=None, help="OpenRouter model id; overrides OPENROUTER_MODEL")
    parser.add_argument("--concurrency", type=_concurrency, default=1,
                        help=f"runs in parallel, 1 to {MAX_CONCURRENCY} (default 1); each worker has its own server and folders")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and the run count, run nothing")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR), help=argparse.SUPPRESS)
    parser.add_argument("--data-dir", default=str(DATA_DIR), help=argparse.SUPPRESS)
    return parser


def select_questions(questions: Sequence[Question], only: Optional[str]) -> list[Question]:
    """All questions, or only the listed ids (unknown ids raise ``ValueError``)."""
    if not only:
        return list(questions)
    wanted = [x.strip() for x in only.split(",") if x.strip()]
    known = {q.id for q in questions}
    unknown = [w for w in wanted if w not in known]
    if unknown:
        raise ValueError(f"unknown question id(s): {', '.join(unknown)}; known: {', '.join(sorted(known))}")
    return [q for q in questions if q.id in wanted]


def compute_truths(questions: Sequence[Question], data_dir: Path) -> dict[str, list[Expect]]:
    """Ground truth per question id from the data files (``FileNotFoundError`` if a file is missing)."""
    frames: dict[str, pd.DataFrame] = {}
    truths: dict[str, list[Expect]] = {}
    for q in questions:
        if q.dataset not in frames:
            frames[q.dataset] = read_dataset(q.dataset, data_dir)
        truths[q.id] = q.truth(frames[q.dataset])
    return truths


def estimate_calls(todo: int, measured_mean: Optional[float]) -> str:
    """One line estimating the chat-model calls still to make: from a measured mean, else the assumed range."""
    if measured_mean:
        return f"ESTIMATE of LLM calls: about {round(todo * measured_mean)} for {todo} runs (measured mean {measured_mean:.1f} calls per run in this run folder)."
    low, high = CALLS_PER_RUN_RANGE
    return (f"ESTIMATE of LLM calls: {todo * low} to {todo * high} for {todo} runs (assumed {low} to {high} calls per run, "
            "NOT measured; run the slice, then llm_calls in runs.jsonl gives the real mean).")


def print_plan(questions: Sequence[Question], truths: dict[str, list[Expect]], runs: Sequence[RunKey], done: set[tuple[str, int, int]],
               measured_mean: Optional[float] = None) -> None:
    print(f"{'question':<22} {'kind':<12} {'expected figures':>16}")
    for q in questions:
        print(f"{q.id:<22} {q.kind:<12} {(len(truths[q.id]) if q.id in truths else '?'):>16}")
    todo = [r for r in runs if r.as_tuple() not in done]
    print(f"\nPLAN: {len(questions)} questions x {len({r.setting for r in runs})} settings x "
          f"{len({r.repeat for r in runs})} repeats = {len(runs)} runs ({len(done & {r.as_tuple() for r in runs})} already done, {len(todo)} to run)")
    print(estimate_calls(len(todo), measured_mean))


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    llm_factory: Optional[Callable[..., Any]] = None,
    questions: Optional[Sequence[Question]] = None,
) -> int:
    """Run the evaluation. ``llm_factory(model=...)`` and ``questions`` are for tests; the real run uses OpenRouter."""
    args = build_parser().parse_args(argv)
    data_dir, run_dir = Path(args.data_dir), Path(args.results_dir) / args.name
    try:
        chosen = select_questions(questions if questions is not None else QUESTIONS, args.only)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    try:
        truths: dict[str, list[Expect]] = compute_truths(chosen, data_dir)
    except FileNotFoundError as exc:
        if not args.dry_run:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        truths = {}  # a dry run only needs the plan; the figure counts need the data files
        print(f"NOTE: {exc} (dry run: expected-figure counts shown as '?')", file=sys.stderr)
    settings = list(dict.fromkeys(args.settings))
    runs = plan_runs(chosen, settings, args.repeats)
    runs_path = run_dir / "runs.jsonl"
    done = {(r["question_id"], r["setting"], r["repeat"]) for r in read_runs(runs_path)}

    if args.dry_run:
        counted = [r["llm_calls"] for r in read_runs(runs_path) if r.get("llm_calls")]
        print_plan(chosen, truths, runs, done, sum(counted) / len(counted) if counted else None)
        return 0

    try:
        if llm_factory is None:
            model = resolve_settings(args.model).model       # needs OPENROUTER_API_KEY
            get_llm(model, check=True)  # fail fast: no key, key rejected, unknown model, no "tools"
        else:
            model = resolve_model_name(args.model)           # tests inject their own model: no key needed
    except LLMConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    run_dir.mkdir(parents=True, exist_ok=True)
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        previous = json.loads(meta_path.read_text(encoding="utf-8"))
        if previous.get("model") != model:
            print(f"ERROR: this run was started with model '{previous.get('model')}', not '{model}'. Use another --name.", file=sys.stderr)
            return 2
    else:
        meta_path.write_text(json.dumps(build_meta(model, args), indent=2), encoding="utf-8")

    todo = [r for r in runs if r.as_tuple() not in done]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"{len(runs) - len([r for r in runs if r.as_tuple() not in done])}/{len(runs)} runs already done; running {len(todo)} now "
          f"(model {model}). Results: {runs_path}")
    by_id = {q.id: q for q in chosen}
    work = Path(args.results_dir) / args.name / "_work"
    latencies: list[float] = []
    workers = max(1, min(args.concurrency, MAX_CONCURRENCY))
    holders = [DatasetSession(Config(data_dir=work / f"w{n}" / "data", output_dir=work / f"w{n}" / "outputs"), data_dir, llm_factory)
               for n in range(1, workers + 1)]

    def finish(i: int, key: RunKey, record: dict[str, Any]) -> None:
        append_record(runs_path, record)
        latencies.append(record["latency_s"])
        mean = sum(latencies) / len(latencies)
        ok = record["tool_correct"] and record["report_correct"] and record["failure_reason"] is None
        status = "ok" if ok else f"FAIL({record['failure_reason'] or 'report'})"
        eta = mean * (len(todo) - i) / workers
        print(f"[{i}/{len(todo)}] {key.question_id} retries={key.setting} repeat={key.repeat} {status} "
              f"{format_duration(record['latency_s'])} | mean {format_duration(mean)}/run | ETA {format_duration(eta)}", flush=True)
        if i == 1:
            print(f"Time per run: {record['latency_s']:.0f}s. A full run of {len(runs)} runs would take about "
                  f"{format_duration(record['latency_s'] * len(runs) / workers)} at this speed"
                  f"{f' with {workers} workers' if workers > 1 else ''}; choose --repeats from this.", flush=True)

    try:
        if workers == 1:
            for i, key in enumerate(todo, 1):
                question = by_id[key.question_id]
                finish(i, key, run_one(holders[0], question, truths[question.id], key, model, args.timeout_s))
        else:
            free: "queue.Queue[DatasetSession]" = queue.Queue()
            for holder in holders:
                free.put(holder)

            def work_item(key: RunKey) -> dict[str, Any]:
                holder = free.get()
                try:
                    question = by_id[key.question_id]
                    return run_one(holder, question, truths[question.id], key, model, args.timeout_s)
                finally:
                    free.put(holder)

            pool = ThreadPoolExecutor(max_workers=workers)
            try:
                futures = {pool.submit(work_item, key): key for key in todo}
                for i, future in enumerate(as_completed(futures), 1):
                    finish(i, futures[future], future.result())
            finally:
                pool.shutdown(wait=True, cancel_futures=True)
    except KeyboardInterrupt:
        print(f"\nInterrupted after {len(latencies)} run(s). Run the same command again to resume.", file=sys.stderr)
        return 130
    finally:
        for holder in holders:
            holder.close()
    remaining = len([r for r in runs if r.as_tuple() not in {(x["question_id"], x["setting"], x["repeat"]) for x in read_runs(runs_path)}])
    print(f"\nDone for now: {remaining} run(s) left. Summary: python -m eval.summarize {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
