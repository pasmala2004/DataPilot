"""Command-line entry point: ask a question about a spreadsheet in DATA_DIR.

    python run.py --file sales.xlsx --sheet orders "total amount by region" [--max-retries 2]

``--file`` is a name inside DATA_DIR (default ./data). The trace (ROUTER / TOOL / OBSERVE ...)
and the answer go to stdout; the MCP server's own one-line-per-call logs go to stderr, so
``2>/dev/null`` hides them. Neither contains raw data rows.

LLM: OpenRouter (set OPENROUTER_API_KEY; model via --model or OPENROUTER_MODEL). With OpenRouter, column names,
the 5-row preview, tool results and capped run_pandas output are sent to OpenRouter and the model provider:
do not use confidential data. Exit code 2 means the key or model is missing, wrong or not reachable.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any, Callable, Sequence

from langchain_core.language_models.chat_models import BaseChatModel

from agent.graph import run_agent
from agent.llm import LLMConfigError, describe_exception, get_llm, hint_for_error
from agent.mcp_client import excel_session
from agent.observe import DEFAULT_MAX_RETRIES
from agent.report import build_report
from core import Config


def _non_negative_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
    if value < 0:
        raise argparse.ArgumentTypeError("must be 0 or more (0 turns retrying off)")
    return value


def build_parser() -> argparse.ArgumentParser:
    """The CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Ask a question about an Excel/CSV file in DATA_DIR. Every number comes from a tool.",
    )
    parser.add_argument("question", help="your question, in quotes")
    parser.add_argument("--file", required=True, help="file name inside DATA_DIR, e.g. sales.xlsx")
    parser.add_argument("--sheet", default=None, help="sheet name (default: the first sheet; use 'csv' for CSV files)")
    parser.add_argument(
        "--max-retries",
        type=_non_negative_int,
        default=DEFAULT_MAX_RETRIES,
        help=f"retries per task after a rejected attempt (default {DEFAULT_MAX_RETRIES}; 0 turns retrying off)",
    )
    parser.add_argument(
        "--model", default=None,
        help="OpenRouter model id (default: $OPENROUTER_MODEL or openai/gpt-4o-mini)",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="also write a Markdown report to OUTPUT_DIR/reports and print its path",
    )
    return parser


def _default_llm_factory(model: str | None = None) -> BaseChatModel:
    """Build the OpenRouter LLM and check the key and the model id."""
    return get_llm(model, check=True)


def _print_trace(line: str) -> None:
    print(line, flush=True)


async def run_question(
    llm: BaseChatModel, file: str, question: str, sheet: str | None, max_retries: int, config: Config
) -> dict[str, Any]:
    """Open ONE MCP session, run the agent loop inside it and return its result dict."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    async with excel_session(config.data_dir, config.output_dir) as tools:
        return await run_agent(
            question, file, llm=llm, tools=tools, sheet=sheet, max_retries=max_retries, trace=_print_trace
        )


def print_summary(result: dict[str, Any], model: str | None = None) -> None:
    """Print the outcome after the trace (``model`` only tailors the fix-it hint)."""
    print()
    if result.get("error"):
        print(f"ERROR: {result['error']}")
        hint = hint_for_error(str(result["error"]))
        if hint:
            print(f"HINT: {hint}")
        return
    if result.get("clarification"):
        print(f"I need a bit more detail: {result['clarification']}")
        return
    print("ANSWER")
    print(result.get("answer") or "(no answer)")
    print("\nTASKS")
    for status in result.get("task_status", []):
        retries = f", {status['retries']} retr{'y' if status['retries'] == 1 else 'ies'}" if status["retries"] else ""
        print(f"  {status['idx'] + 1}. [{status['agent']}] {status['status']}{retries}: {status['task']}")
    charts = [
        r["output"]["path"] for r in result.get("results", []) if r["tool"] == "make_chart" and r["output"].get("path")
    ]
    for path in dict.fromkeys(charts):
        print(f"  chart: {path}")


def main(argv: Sequence[str] | None = None, *, llm_factory: Callable[..., BaseChatModel] = _default_llm_factory) -> int:
    """Run the CLI. Returns the process exit code (0 ok, 1 run failed, 2 LLM not configured/reachable).

    ``llm_factory(model=...)`` builds the model; tests pass a scripted one.
    """
    args = build_parser().parse_args(argv)
    try:
        llm = llm_factory(model=args.model)
    except LLMConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    result = asyncio.run(run_question(llm, args.file, args.question, args.sheet, args.max_retries, Config()))
    print_summary(result, args.model)
    if args.report and not result.get("error") and not result.get("clarification"):
        try:
            report = asyncio.run(build_report(result, llm))
            print(f"\nREPORT\n  {report['path']}")
        except OSError as exc:
            print(f"\nERROR: the report could not be written: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:  # noqa: BLE001 - an API failure while writing the report: clean message, no traceback
            print(f"\nERROR: the report could not be built: {describe_exception(exc)}", file=sys.stderr)
            return 1
    return 1 if result.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
