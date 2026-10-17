"""eval/summarize.py on a synthetic runs.jsonl: counts, percentages, retry delta, flaky, reasons, CV bullets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from eval import summarize as S


def rec(q: str, setting: int, repeat: int, *, tool=True, report=True, completed=True, reason=None, kind="scalar",
        retries=0, sc=0, latency=10.0, fabricated=False, fallbacks=0, chart=None) -> dict[str, Any]:
    return {"question_id": q, "dataset": "d.csv", "kind": kind, "setting": setting, "repeat": repeat, "model": "gemma4:26b",
            "completed": completed, "failed_tasks": 0, "retries": retries, "self_corrections": sc, "tool_calls": 1,
            "number_check_fallbacks": fallbacks, "latency_s": latency, "error_type": None, "failure_reason": reason,
            "tool_correct": tool, "report_correct": report, "chart_ok": chart, "fabricated": fabricated, "facts": 1,
            "clarification": False, "missing": []}


def make_run(tmp_path: Path, name: str = "r1", records: list[dict[str, Any]] | None = None) -> Path:
    folder = tmp_path / name
    folder.mkdir()
    (folder / "meta.json").write_text(json.dumps({"name": name, "model": "openai/gpt-4o-mini", "provider": "openrouter",
                                                  "git_commit": "abc123", "date": "2026-10-14T00:00:00+00:00", "settings": [0, 2], "repeats": 5}))
    (folder / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in (records if records is not None else synthetic())) + "\n")
    return folder


def synthetic() -> list[dict[str, Any]]:
    """Setting 0: 10 runs, tool 6, report 5, completed 9. Setting 2: 10 runs, tool 9, report 8, completed 10."""
    rows = []
    for i in range(10):
        rows.append(rec("qa" if i < 5 else "qb", 0, i % 5 + 1, tool=i < 6, report=i < 5, completed=i != 9,
                        reason=None if i < 5 else ("crash" if i == 9 else "wrong_value"), kind="group" if i < 5 else "scalar",
                        retries=0, sc=1 if i == 0 else 0, latency=20.0))
    for i in range(10):
        rows.append(rec("qa" if i < 5 else "qb", 2, i % 5 + 1, tool=i < 9, report=i < 8, completed=True,
                        reason=None if i < 8 else "wrong_value", kind="group" if i < 5 else "scalar",
                        retries=1 if i < 4 else 0, latency=40.0, fabricated=i == 3, fallbacks=1 if i == 3 else 0))
    return rows


def test_counts_percentages_and_means(tmp_path):
    stats = S.setting_stats(S.load_run(make_run(tmp_path))[1])
    assert stats[0]["n"] == 10 and (stats[0]["tool_correct"], stats[0]["report_correct"], stats[0]["completed"]) == (6, 5, 9)
    assert (stats[2]["tool_correct"], stats[2]["report_correct"], stats[2]["completed"]) == (9, 8, 10)
    assert stats[0]["latency_s"] == 20.0 and stats[2]["latency_s"] == 40.0
    assert stats[2]["retries"] == pytest.approx(0.4) and stats[0]["self_corrections"] == pytest.approx(0.1)
    assert stats[2]["fabricated"] == 1 and stats[2]["fallbacks"] == 1
    assert S.frac(7, 9) == "7/9 (77.8%)" and S.frac(0, 0) == "0/0 (n/a)"


def test_retry_delta_is_high_minus_low_in_percentage_points(tmp_path):
    delta = S.retry_delta(S.setting_stats(S.load_run(make_run(tmp_path))[1]))
    assert (delta["low"], delta["high"]) == (0, 2)
    assert delta["tool_correct"] == pytest.approx(30.0) and delta["report_correct"] == pytest.approx(30.0) and delta["completion"] == pytest.approx(10.0)
    assert S.retry_delta({0: S.setting_stats([rec("q", 0, 1)])[0]}) is None


def test_flaky_failure_reasons_and_kinds(tmp_path):
    _, records = S.load_run(make_run(tmp_path))
    flakes = S.flaky(records)
    assert ("qb", 0, 0, 5) not in flakes                                  # always failing is not flaky
    assert ("qa", 0, 5, 5) not in flakes and ("qa", 2, 5, 5) not in flakes
    assert ("qb", 2, 3, 5) in flakes                                      # 3 of 5 repeats passed
    assert S.failure_reasons(records) == {0: {"crash": 1, "wrong_value": 4}, 2: {"wrong_value": 2}}
    kinds = S.by_kind(records)
    assert kinds["group"] == {0: (5, 5), 2: (5, 5)} and kinds["scalar"] == {0: (0, 5), 2: (3, 5)}


def test_summary_text_contains_the_computed_numbers_and_cv_bullets(tmp_path):
    text, _ = S.build_summary(make_run(tmp_path))
    assert "| 0 | 10 | 6/10 (60.0%) | 5/10 (50.0%) | 9/10 (90.0%) |" in text
    assert "| 2 | 10 | 9/10 (90.0%) | 8/10 (80.0%) | 10/10 (100.0%) |" in text
    assert "tool_correct +30.0 pp" in text and "report_correct +30.0 pp" in text
    assert "- qb (retries 2): passed 3/5 repeats" in text
    assert "- retries 0: crash 1, wrong_value 4" in text
    assert "- qa (retries 2, repeat 4): 1 fallback(s), 1 fact(s)" in text
    cv = text.split("## CV bullets")[1]
    assert "20 finished runs" in cv and "openai/gpt-4o-mini" in cv and "15/20 (75.0%)" in cv          # tool 6+9
    assert "13/20 (65.0%)" in cv and "19/20 (95.0%)" in cv                                      # report 5+8, completed 9+10
    assert "50.0% with retries 0 and 80.0% with retries 2 (+30.0 pp)" in cv and "fabricated" in cv and "1/20" in cv
    assert "30.0 s per run" in cv


def test_nothing_in_the_summary_is_invented_for_an_empty_or_single_setting_run(tmp_path):
    only0 = make_run(tmp_path, "r0", [rec("q", 0, 1), rec("q", 0, 2, report=False, reason="wrong_value")])
    text, _ = S.build_summary(only0)
    assert "needs at least two settings" in text and "1/2 (50.0%)" in text and "(+" not in text.split("## CV bullets")[1]
    assert S.cv_bullets([], {}, {}, 0) == "No finished runs yet."


def test_duplicate_records_keep_the_last_and_truncated_line_is_ignored(tmp_path):
    folder = make_run(tmp_path, "r2", [rec("q", 0, 1, report=False, reason="wrong_value"), rec("q", 0, 1)])
    with (folder / "runs.jsonl").open("a") as fh:
        fh.write('{"question_id": "q", "setti')
    _, records = S.load_run(folder)
    assert len(records) == 1 and records[0]["report_correct"] is True


def test_compare_shows_two_runs_side_by_side(tmp_path):
    a = make_run(tmp_path, "a")
    b = make_run(tmp_path, "b", [rec("q", 0, 1), rec("q", 2, 1, report=False, reason="wrong_value")])
    text, _ = S.build_summary(a, b)
    assert "## Comparison: a vs b" in text
    assert "| 0 | report_correct | 5/10 (50.0%) | 1/1 (100.0%) |" in text
    assert "| 2 | report_correct | 8/10 (80.0%) | 0/1 (0.0%) |" in text


def test_main_writes_summary_and_chart(tmp_path, capsys):
    folder = make_run(tmp_path)
    assert S.main([str(folder)]) == 0
    assert (folder / "summary.md").read_text().startswith("# Evaluation summary: r1")
    assert (folder / S.IMAGE_NAME).stat().st_size > 1000
    other = make_run(tmp_path, "o", [rec("q", 0, 1)])
    assert S.main([str(folder), "--compare", str(other)]) == 0
    assert "Comparison" in (folder / "summary.md").read_text()


def test_main_errors(tmp_path, capsys):
    assert S.main([str(tmp_path / "missing")]) == 2 and "no runs.jsonl" in capsys.readouterr().err
    empty = make_run(tmp_path, "e", [])
    assert S.main([str(empty)]) == 2 and not (empty / "summary.md").exists()


# --------------------------------------------------------------------------- README block (--update-readme)
README_TEXT = f"# T\n\nbefore\n\n{S.EVAL_START}\nold text\n{S.EVAL_END}\n\nafter\n"


def test_update_readme_replaces_only_the_block_with_computed_numbers(tmp_path):
    run = make_run(tmp_path, "r1", synthetic())
    readme = tmp_path / "README.md"
    readme.write_text(README_TEXT, encoding="utf-8")
    S.update_readme(readme, run)
    out = readme.read_text(encoding="utf-8")
    assert out.startswith("# T\n\nbefore\n\n") and out.endswith("\n\nafter\n")
    assert "old text" not in out and out.count(S.EVAL_START) == 1 and out.count(S.EVAL_END) == 1
    _, records = S.load_run(run)
    stats = S.setting_stats(records)
    for setting, st in stats.items():
        assert S.frac(st["tool_correct"], st["n"]) in out and S.frac(st["report_correct"], st["n"]) in out
        assert S.frac(st["completed"], st["n"]) in out
    assert "Retry effect" in out and "fabricated" in out and "model" in out


def test_update_readme_is_idempotent_and_keeps_both_runs_for_before_after(tmp_path):
    first, second = make_run(tmp_path, "run1", synthetic()), make_run(tmp_path, "run2", synthetic()[:6])
    readme = tmp_path / "README.md"
    readme.write_text(README_TEXT, encoding="utf-8")
    S.update_readme(readme, second, before=first)
    once = readme.read_text(encoding="utf-8")
    S.update_readme(readme, second, before=first)
    assert readme.read_text(encoding="utf-8") == once
    assert once.index("Run `run1`") < once.index("Run `run2`")


def test_update_readme_needs_the_markers_exactly_once(tmp_path):
    run = make_run(tmp_path, "r1", synthetic())
    readme = tmp_path / "README.md"
    for bad in ("no markers here", README_TEXT + S.EVAL_START, f"{S.EVAL_END}\n{S.EVAL_START}"):
        readme.write_text(bad, encoding="utf-8")
        with pytest.raises(ValueError):
            S.update_readme(readme, run)
        assert readme.read_text(encoding="utf-8") == bad  # untouched


def test_main_update_readme_flag(tmp_path, capsys):
    run = make_run(tmp_path, "r1", synthetic())
    readme = tmp_path / "README.md"
    readme.write_text(README_TEXT, encoding="utf-8")
    assert S.main([str(run), "--update-readme", "--readme", str(readme)]) == 0
    assert "tool_correct" in readme.read_text(encoding="utf-8")
    readme.write_text("no markers", encoding="utf-8")
    assert S.main([str(run), "--update-readme", "--readme", str(readme)]) == 2
    assert "EVAL:START" in capsys.readouterr().err
