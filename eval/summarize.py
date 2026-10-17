"""Summarize an evaluation run: ``python -m eval.summarize <run_dir> [--compare <other_run_dir>] [--update-readme]``.

Reads ``runs.jsonl`` (and ``meta.json``) and writes ``summary.md`` and ``accuracy_by_setting.png`` into the run
folder. Every number in the file is computed from the records; the closing "CV bullets" paragraph uses only
those computed numbers. A run counts as *correct* for a metric when its flag is true; unfinished runs are
simply not in the file, so n is the number of runs that finished.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence

from eval.run_eval import read_runs

README_PATH = Path(__file__).resolve().parent.parent / "README.md"
EVAL_START, EVAL_END = "<!-- EVAL:START -->", "<!-- EVAL:END -->"

METRICS = (("tool_correct", "tool_correct"), ("report_correct", "report_correct"), ("completed", "completion"))
IMAGE_NAME = "accuracy_by_setting.png"


# --------------------------------------------------------------------------- #
# Numbers
# --------------------------------------------------------------------------- #
def frac(k: int, n: int) -> str:
    """``7/9 (77.8%)``; ``0/0 (n/a)`` when there is nothing to count."""
    return f"{k}/{n} ({100 * k / n:.1f}%)" if n else "0/0 (n/a)"


def pct(k: int, n: int) -> Optional[float]:
    return 100 * k / n if n else None


def mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def fmt(value: Optional[float], digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def load_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """``(meta, records)`` of a run folder; a repeated (question, setting, repeat) keeps its last record."""
    runs_path = run_dir / "runs.jsonl"
    if not runs_path.exists():
        raise FileNotFoundError(f"no runs.jsonl in {run_dir}")
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    unique = {(r["question_id"], r["setting"], r["repeat"]): r for r in read_runs(runs_path)}
    return meta, list(unique.values())


def correct(record: dict[str, Any], metric: str) -> bool:
    return bool(record.get(metric))


def setting_stats(records: Sequence[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Per max_retries setting: counts of each metric, and the means of retries, self-corrections, latency."""
    by: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by[int(r["setting"])].append(r)
    stats: dict[int, dict[str, Any]] = {}
    for setting, rows in sorted(by.items()):
        stats[setting] = {
            "n": len(rows),
            **{metric: sum(correct(r, metric) for r in rows) for metric, _ in METRICS},
            "retries": mean([r.get("retries", 0) for r in rows]),
            "self_corrections": mean([r.get("self_corrections", 0) for r in rows]),
            "latency_s": mean([r.get("latency_s", 0.0) for r in rows]),
            "fabricated": sum(bool(r.get("fabricated")) for r in rows),
            "fallbacks": sum(int(r.get("number_check_fallbacks") or 0) for r in rows),
        }
    return stats


def retry_delta(stats: dict[int, dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Highest setting minus the lowest, in percentage points (``None`` with fewer than two settings)."""
    if len(stats) < 2:
        return None
    low, high = min(stats), max(stats)
    out: dict[str, Any] = {"low": low, "high": high}
    for metric, label in METRICS:
        a, b = pct(stats[low][metric], stats[low]["n"]), pct(stats[high][metric], stats[high]["n"])
        out[label] = None if a is None or b is None else b - a
    return out


def by_kind(records: Sequence[dict[str, Any]]) -> dict[str, dict[int, tuple[int, int]]]:
    """``{kind: {setting: (report_correct count, n)}}``."""
    table: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in records:
        cell = table[r["kind"]][int(r["setting"])]
        cell[0] += correct(r, "report_correct")
        cell[1] += 1
    return {k: {s: (c[0], c[1]) for s, c in v.items()} for k, v in table.items()}


def flaky(records: Sequence[dict[str, Any]]) -> list[tuple[str, int, int, int]]:
    """Questions with mixed outcomes across repeats of one setting: ``(question, setting, passes, n)``.

    A pass is ``tool_correct and report_correct``.
    """
    groups: dict[tuple[str, int], list[bool]] = defaultdict(list)
    for r in records:
        groups[(r["question_id"], int(r["setting"]))].append(correct(r, "tool_correct") and correct(r, "report_correct"))
    return sorted((q, s, sum(v), len(v)) for (q, s), v in groups.items() if 0 < sum(v) < len(v))


def failure_reasons(records: Sequence[dict[str, Any]]) -> dict[int, dict[str, int]]:
    out: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in records:
        if r.get("failure_reason"):
            out[int(r["setting"])][r["failure_reason"]] += 1
    return {s: dict(v) for s, v in sorted(out.items())}


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def _setting_table(stats: dict[int, dict[str, Any]]) -> list[str]:
    lines = ["| max_retries | runs | tool_correct | report_correct | completion | mean retries | mean self-corrections | mean latency (s) |",
             "|---|---|---|---|---|---|---|---|"]
    for s, st in stats.items():
        lines.append(f"| {s} | {st['n']} | {frac(st['tool_correct'], st['n'])} | {frac(st['report_correct'], st['n'])} | "
                     f"{frac(st['completed'], st['n'])} | {fmt(st['retries'])} | {fmt(st['self_corrections'])} | {fmt(st['latency_s'], 1)} |")
    return lines


def _delta_line(delta: Optional[dict[str, Any]]) -> str:
    if delta is None:
        return "Retry on vs off: needs at least two settings."
    parts = ", ".join(f"{label} {'n/a' if delta[label] is None else format(delta[label], '+.1f') + ' pp'}" for _, label in METRICS)
    return f"Retry effect (max_retries {delta['high']} minus {delta['low']}): {parts}."


def cv_bullets(records: Sequence[dict[str, Any]], stats: dict[int, dict[str, Any]], meta: dict[str, Any], n_questions: int) -> str:
    """One paragraph built only from computed numbers."""
    n = len(records)
    if not n:
        return "No finished runs yet."
    total = lambda key: sum(st[key] for st in stats.values())  # noqa: E731
    settings = ", ".join(str(s) for s in stats)
    parts = [f"Evaluated the agent on {n} finished runs ({n_questions} questions, max_retries in {{{settings}}}) with the local model "
             f"{meta.get('model', 'unknown')}: tool outputs matched pandas ground truth in {frac(total('tool_correct'), n)}, "
             f"reports showed the correct figure in {frac(total('report_correct'), n)}, and {frac(total('completed'), n)} runs completed."]
    delta = retry_delta(stats)
    if delta is not None and delta["report_correct"] is not None:
        low, high = delta["low"], delta["high"]
        parts.append(f"Report accuracy was {fmt(pct(stats[low]['report_correct'], stats[low]['n']), 1)}% with retries {low} and "
                     f"{fmt(pct(stats[high]['report_correct'], stats[high]['n']), 1)}% with retries {high} ({delta['report_correct']:+.1f} pp).")
    fabricated = sum(bool(r.get("fabricated")) for r in records)
    parts.append(f"Runs flagged as fabricated (number-check fallback or figures on an unanswerable question): {fabricated}/{n}.")
    parts.append(f"Mean latency {fmt(mean([r.get('latency_s', 0.0) for r in records]), 1)} s per run.")
    return " ".join(parts)


def build_summary(run_dir: Path, other: Optional[Path] = None) -> tuple[str, dict[int, dict[str, Any]]]:
    """The summary text and the per-setting stats of the primary run."""
    meta, records = load_run(run_dir)
    stats = setting_stats(records)
    questions = {r["question_id"] for r in records}
    lines = [f"# Evaluation summary: {meta.get('name', run_dir.name)}", ""]
    lines.append(f"Model: {meta.get('model', 'unknown')} (provider {meta.get('provider', 'openrouter')}), "
                 f"commit {meta.get('git_commit', '?')}, started {meta.get('date', '?')}. "
                 f"{len(records)} finished runs, {len(questions)} questions.")
    planned = meta.get("repeats")
    if planned and meta.get("settings"):
        lines.append(f"Planned: {len(questions)} questions x {len(meta['settings'])} settings x {planned} repeats "
                     f"(finished runs above may be fewer if the run is not complete).")
    calls = [r["llm_calls"] for r in records if r.get("llm_calls")]
    if calls:
        lines.append(f"Mean LLM calls per run: {fmt(mean(calls), 1)} (min {min(calls)}, max {max(calls)}).")
    lines += ["", "## Accuracy and cost by setting", ""] + _setting_table(stats)
    lines += ["", "## Retry on vs off", "", _delta_line(retry_delta(stats))]

    kinds = by_kind(records)
    settings = sorted(stats)
    lines += ["", "## report_correct by question kind", "",
              "| kind | " + " | ".join(f"retries {s}" for s in settings) + " |", "|---|" + "---|" * len(settings)]
    for kind in sorted(kinds):
        lines.append(f"| {kind} | " + " | ".join(frac(*kinds[kind].get(s, (0, 0))) for s in settings) + " |")

    charts = [r for r in records if r.get("chart_ok") is not None]
    if charts:
        lines += ["", f"Charts: PNG saved and non-empty in {frac(sum(bool(r['chart_ok']) for r in charts), len(charts))} chart runs."]

    unstable = flaky(records)
    lines += ["", "## Flaky questions (mixed results across repeats)", ""]
    lines += [f"- {q} (retries {s}): passed {k}/{n} repeats" for q, s, k, n in unstable] or ["- none"]

    reasons = failure_reasons(records)
    lines += ["", "## Failure reasons", ""]
    lines += [f"- retries {s}: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())) for s, c in reasons.items()] or ["- no failures"]

    fabricated = [r for r in records if r.get("fabricated")]
    lines += ["", "## Fabricated runs", ""]
    lines += [f"- {r['question_id']} (retries {r['setting']}, repeat {r['repeat']}): "
              f"{r.get('number_check_fallbacks', 0)} fallback(s), {r.get('facts', 0)} fact(s)" for r in fabricated] or ["- none"]

    if other is not None:
        o_meta, o_records = load_run(other)
        o_stats = setting_stats(o_records)
        lines += ["", f"## Comparison: {meta.get('name', run_dir.name)} vs {o_meta.get('name', other.name)}", "",
                  f"Models: {meta.get('model', '?')} vs {o_meta.get('model', '?')}.", "",
                  "| max_retries | metric | " + f"{meta.get('name', run_dir.name)} | {o_meta.get('name', other.name)} |", "|---|---|---|---|"]
        for s in sorted(set(stats) | set(o_stats)):
            for metric, label in METRICS:
                cells = [frac(st[metric], st["n"]) if (st := d.get(s)) else "n/a" for d in (stats, o_stats)]
                lines.append(f"| {s} | {label} | {cells[0]} | {cells[1]} |")
            for key, label, digits in (("retries", "mean retries", 2), ("self_corrections", "mean self-corrections", 2), ("latency_s", "mean latency (s)", 1)):
                cells = [fmt(st[key], digits) if (st := d.get(s)) else "n/a" for d in (stats, o_stats)]
                lines.append(f"| {s} | {label} | {cells[0]} | {cells[1]} |")
        lines += ["", _delta_line(retry_delta(stats)).replace("Retry effect", f"Retry effect ({meta.get('name', run_dir.name)})"),
                  _delta_line(retry_delta(o_stats)).replace("Retry effect", f"Retry effect ({o_meta.get('name', other.name)})")]

    lines += ["", "## CV bullets", "", cv_bullets(records, stats, meta, len(questions)), ""]
    return "\n".join(lines), stats


def write_chart(path: Path, run_dir: Path, other: Optional[Path] = None) -> None:
    """Bar chart: percent correct per setting (tool_correct, report_correct, completion); both runs when comparing."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series: list[tuple[str, dict[int, dict[str, Any]]]] = []
    for folder in [run_dir] + ([other] if other else []):
        meta, records = load_run(folder)
        series.append((str(meta.get("name", folder.name)), setting_stats(records)))
    groups = [(name, s) for name, st in series for s in st]
    fig, ax = plt.subplots(figsize=(max(5.0, 1.8 * len(groups) + 2), 4))
    width, colors = 0.25, ("#3b6fb6", "#d98324", "#5b9a5b")
    for j, (metric, label) in enumerate(METRICS):
        values = []
        for name, s in groups:
            st = dict(series)[name][s]
            values.append(pct(st[metric], st["n"]) or 0.0)
        bars = ax.bar([i + (j - 1) * width for i in range(len(groups))], values, width, label=label, color=colors[j])
        ax.bar_label(bars, fmt="%.0f", fontsize=8)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{name}\nretries={s}" if other else f"retries={s}" for name, s in groups])
    ax.set_ylim(0, 110)
    ax.set_ylabel("% of runs")
    ax.set_title("Accuracy by max_retries setting")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def readme_section(run_dir: Path) -> str:
    """The README block for one run: model, date, runs, per-setting k/n and %, retry effect, latency, fabricated.

    Every number comes from ``runs.jsonl``; nothing is typed by hand.
    """
    meta, records = load_run(run_dir)
    stats = setting_stats(records)
    lines = [f"**Run `{meta.get('name', run_dir.name)}`**: model {meta.get('model', 'unknown')}, date {str(meta.get('date', 'unknown'))[:10]}, "
             f"{len(records)} finished runs ({len({r['question_id'] for r in records})} questions, "
             f"commit {meta.get('git_commit', '?')}).", "",
             "| max_retries | runs | tool_correct | report_correct | completion | mean latency (s) | fabricated |",
             "|---|---|---|---|---|---|---|"]
    for setting, st in stats.items():
        lines.append(f"| {setting} | {st['n']} | {frac(st['tool_correct'], st['n'])} | {frac(st['report_correct'], st['n'])} | "
                     f"{frac(st['completed'], st['n'])} | {fmt(st['latency_s'], 1)} | {st['fabricated']} |")
    lines += ["", _delta_line(retry_delta(stats))]
    return "\n".join(lines)


def update_readme(readme: Path, run_dir: Path, before: Optional[Path] = None) -> None:
    """Replace the text between the EVAL markers in ``readme`` with the block for ``run_dir``.

    With ``before`` (an earlier run, for example run1 when run_dir is run2) both runs are written, earlier first,
    so a before/after comparison stays visible.

    Raises:
        FileNotFoundError: the README or a run is missing.
        ValueError: the markers are not in the README exactly once, in order.
    """
    text = readme.read_text(encoding="utf-8")
    if text.count(EVAL_START) != 1 or text.count(EVAL_END) != 1 or text.index(EVAL_START) > text.index(EVAL_END):
        raise ValueError(f"{readme} must contain {EVAL_START} and {EVAL_END} exactly once, in that order.")
    sections = ([readme_section(before)] if before else []) + [readme_section(run_dir)]
    block = f"{EVAL_START}\n" + "\n\n".join(sections) + f"\n{EVAL_END}"
    head, rest = text.split(EVAL_START, 1)
    tail = rest.split(EVAL_END, 1)[1]
    readme.write_text(head + block + tail, encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.summarize", description="Summarize an evaluation run.")
    parser.add_argument("run_dir", help="eval/results/<name>")
    parser.add_argument("--compare", default=None, help="another run folder to show side by side")
    parser.add_argument("--update-readme", action="store_true",
                        help="write the results block between the EVAL markers in README.md (with --compare: that run first)")
    parser.add_argument("--readme", default=str(README_PATH), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    run_dir, other = Path(args.run_dir), Path(args.compare) if args.compare else None
    try:
        text, _ = build_summary(run_dir, other)
        if not any(True for _ in read_runs(run_dir / "runs.jsonl")):
            print("No finished runs in runs.jsonl yet.", file=sys.stderr)
            return 2
        (run_dir / "summary.md").write_text(text, encoding="utf-8")
        write_chart(run_dir / IMAGE_NAME, run_dir, other)
        if args.update_readme:
            update_readme(Path(args.readme), run_dir, other)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(text)
    print(f"\nWrote {run_dir / 'summary.md'} and {run_dir / IMAGE_NAME}")
    if args.update_readme:
        print(f"Updated the evaluation block in {args.readme}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
