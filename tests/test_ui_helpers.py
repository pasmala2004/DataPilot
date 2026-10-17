"""agent/ui_helpers.py: pure helpers of the app (schema table, preview splitting, messages). No Streamlit."""

from __future__ import annotations

from agent.report_doc import ChartRef
from agent.ui_helpers import (
    CURATED_MODELS, PRIVACY_CHECKBOX, PRIVACY_NOTICE, effective_model, format_elapsed, problem_messages, run_block_reason,
    schema_rows, split_for_preview,
)


def test_schema_rows_have_names_types_nulls_and_date_ranges_only():
    schema = {"rows": 10, "columns": {"order_date": "datetime64[ns]", "amount": "float64"}, "nulls": {"amount": 1},
              "date_ranges": {"order_date": ["2024-01-05T00:00:00", "2024-05-02T00:00:00"]}, "preview": [{"amount": 123456}]}
    rows = schema_rows(schema)
    assert rows == [
        {"column": "order_date", "type": "datetime64[ns]", "nulls": 0, "date range": "2024-01-05 to 2024-05-02"},
        {"column": "amount", "type": "float64", "nulls": 1, "date range": ""},
    ]
    assert "123456" not in str(rows)                                       # a preview, if one slipped in, is never shown
    assert schema_rows({}) == []


def test_the_privacy_notice_names_what_is_sent_and_the_checkbox_text_is_fixed():
    assert "OpenRouter" in PRIVACY_NOTICE and "column names" in PRIVACY_NOTICE and "Do not upload confidential data" in PRIVACY_NOTICE
    assert PRIVACY_CHECKBOX == "I understand my data summary is sent to an external API"
    assert "openai/gpt-4o-mini" in CURATED_MODELS and len(CURATED_MODELS) <= 6


def test_effective_model_prefers_the_free_text_id():
    assert effective_model("openai/gpt-4o-mini", "  vendor/x  ") == "vendor/x"
    assert effective_model("openai/gpt-4o-mini", "") == "openai/gpt-4o-mini"
    assert effective_model(None, None) == ""


READY = dict(key_found=True, model_error=None, privacy_ack=True, has_file=True, has_question=True, running=False)


def test_run_is_allowed_only_when_everything_is_in_place():
    assert run_block_reason(**READY) is None
    assert "OPENROUTER_API_KEY" in run_block_reason(**{**READY, "key_found": False})
    assert "model check failed" in run_block_reason(**{**READY, "model_error": "bad"})
    assert "Tick the box" in run_block_reason(**{**READY, "privacy_ack": False})
    assert "Upload a file" in run_block_reason(**{**READY, "has_file": False})
    assert "Type a question" in run_block_reason(**{**READY, "has_question": False})
    assert "in progress" in run_block_reason(**{**READY, "running": True})


def test_split_for_preview_turns_chart_links_into_local_images_and_keeps_the_rest():
    charts = [ChartRef("/out/charts/bar region.png", "../charts/bar%20region.png", "bar chart"),
              ChartRef("/out/charts/line.png", "../charts/line.png", "line chart")]
    markdown = ("## Charts\n\n![bar chart](../charts/bar%20region.png)\n\n![line chart](../charts/line.png)\n\n"
                "## Caveats\n\n- one\n\n![other](http://example.com/x.png)\n")
    assert split_for_preview(markdown, charts) == [
        ("markdown", "## Charts"), ("image", "/out/charts/bar region.png"), ("image", "/out/charts/line.png"),
        ("markdown", "## Caveats\n\n- one\n\n![other](http://example.com/x.png)"),
    ]
    assert split_for_preview("just text\n", []) == [("markdown", "just text")]
    assert split_for_preview("", charts) == []


def test_format_elapsed():
    assert [format_elapsed(x) for x in (-3, 0, 9.9, 59, 60, 73.4, 3725)] == ["0s", "0s", "9s", "59s", "1m 00s", "1m 13s", "62m 05s"]


def test_problem_messages_cover_job_errors_run_errors_failed_tasks_and_fallbacks():
    assert problem_messages(None, None, None) == []
    assert problem_messages("timed out", "try later", None) == [("error", "timed out\n\nFix: try later")]
    result = {
        "error": "The router failed: OpenRouter: no credits (402). Add credits to your OpenRouter account.",
        "task_status": [{"idx": 1, "status": "failed", "task": "Sum x.", "reasons": ["unknown column 'x'"]},
                        {"idx": 0, "status": "finished", "task": "ok"}],
        "report": {"fallbacks": ["section 2: generated without the LLM (number check failed)"]},
    }
    messages = problem_messages(None, None, result, "openai/gpt-4o-mini")
    assert messages[0][0] == "error" and "no credits" in messages[0][1] and "Fix:" in messages[0][1]
    assert ("warning", "Task 2 failed and is not in the figures: Sum x. (unknown column 'x')") in messages
    assert ("warning", "Number check: section 2: generated without the LLM (number check failed)") in messages
    assert len(messages) == 3
    assert problem_messages(None, None, {"error": None, "task_status": [], "report": {"fallbacks": []}}) == []
