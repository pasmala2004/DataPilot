"""MCP client: spawns excel_server.py and keeps ONE session open for the whole run.

The dataset store lives inside the server process, so a ``dataset_id`` is only valid for
the session that created it. Opening a new session per tool call would lose every id.
Use ``excel_session`` as an async context manager around the whole agent run:

    async with excel_session(data_dir, output_dir) as tools:
        ...  # tools.analysis_tools, tools.chart_tools, tools.by_name

Tools are async (``ainvoke``), so everything that calls them must be async too.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

SERVER_SCRIPT = Path(__file__).resolve().parents[1] / "excel_server.py"
SERVER_NAME = "excel"

ANALYSIS_TOOL_NAMES = ("list_sheets", "load_data", "aggregate", "compare_groups", "run_pandas")
CHART_TOOL_NAMES = ("make_chart",)
ALL_TOOL_NAMES = ANALYSIS_TOOL_NAMES + CHART_TOOL_NAMES


@dataclass(frozen=True)
class ExcelTools:
    """The server's tools, split by which agent may use them."""

    by_name: dict[str, BaseTool]

    @property
    def analysis_tools(self) -> list[BaseTool]:
        """list_sheets, load_data, aggregate, compare_groups, run_pandas."""
        return [self.by_name[name] for name in ANALYSIS_TOOL_NAMES]

    @property
    def chart_tools(self) -> list[BaseTool]:
        """make_chart."""
        return [self.by_name[name] for name in CHART_TOOL_NAMES]


@asynccontextmanager
async def excel_session(
    data_dir: str | Path, output_dir: str | Path, *, server_script: str | Path | None = None
) -> AsyncIterator[ExcelTools]:
    """Start excel_server.py over stdio and yield its tools; the server stops on exit.

    The server is started with ``sys.executable`` so it uses the same environment as the
    caller. ``DATA_DIR`` / ``OUTPUT_DIR`` are passed as absolute paths. Only these
    variables (plus the MCP library's small safe default set such as PATH) reach the
    server process, so API keys in the caller's environment are not passed on.
    The server's own one-line-per-call logs (tool name, argument names, result size) go to
    stderr, which the server inherits from this process.

    Raises:
        RuntimeError: if the server does not expose all six expected tools.
    """
    script = Path(server_script) if server_script else SERVER_SCRIPT
    client = MultiServerMCPClient(
        {
            SERVER_NAME: {
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(script)],
                "env": {
                    "DATA_DIR": str(Path(data_dir).resolve()),
                    "OUTPUT_DIR": str(Path(output_dir).resolve()),
                },
            }
        }
    )
    async with client.session(SERVER_NAME) as session:
        loaded = await load_mcp_tools(session)
        by_name = {tool.name: tool for tool in loaded}
        missing = [name for name in ALL_TOOL_NAMES if name not in by_name]
        if missing:
            raise RuntimeError(f"excel_server.py does not expose the expected tools: missing {missing}")
        yield ExcelTools(by_name=by_name)
