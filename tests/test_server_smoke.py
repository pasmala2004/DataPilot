"""Smoke test: start excel_server.py over stdio and talk to it with the MCP client.

* The server is pointed at a temp DATA_DIR / OUTPUT_DIR through the environment; the
  tiny fixture workbook is built here with pandas, so nothing in ./data is used.
* Uses asyncio.run (no pytest plugins).
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "excel_server.py"
EXPECTED_TOOLS = ["aggregate", "compare_groups", "list_sheets", "load_data", "make_chart", "run_pandas"]
TIMEOUT = 90  # seconds for a whole scenario; a hung server must fail the test, not freeze it


@pytest.fixture
def workspace(tmp_path: Path) -> dict[str, Any]:
    """Temp DATA_DIR with one workbook (sheet "Sales", 8 rows) and an empty OUTPUT_DIR."""
    data_dir, out_dir = tmp_path / "data", tmp_path / "outputs"
    data_dir.mkdir()
    out_dir.mkdir()
    sales = pd.DataFrame({
        "order_date": pd.to_datetime(["2024-01-05", "2024-01-20", "2024-02-11", "2024-02-18",
                                      "2024-03-02", "2024-03-15", "2024-04-01", "2024-04-09"]),
        "region": ["North", "South", "North", "East", "South", "North", "East", "South"],
        "amount": [120.5, 80.0, 200.0, 95.25, 60.0, 150.75, 70.0, 99.5],
        "qty": [3, 2, 5, 2, 1, 4, 2, 3],
    })
    sales.to_excel(data_dir / "sales.xlsx", sheet_name="Sales", index=False)
    return {"data_dir": data_dir, "out_dir": out_dir, "sales": sales}


def server_env(workspace: dict[str, Any]) -> dict[str, str]:
    return {"DATA_DIR": str(workspace["data_dir"]), "OUTPUT_DIR": str(workspace["out_dir"])}


def payload(result: Any) -> dict[str, Any]:
    """The dict a tool returned (structured content, falling back to the JSON text)."""
    assert not result.isError, result.content
    if result.structuredContent is not None:
        return result.structuredContent
    return json.loads(result.content[0].text)


def test_six_tools_and_one_session_keeps_dataset_ids_valid(workspace):
    sales: pd.DataFrame = workspace["sales"]
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER)], env=server_env(workspace))

    async def scenario() -> None:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # exactly the six tools
                tools = (await session.list_tools()).tools
                assert sorted(t.name for t in tools) == EXPECTED_TOOLS
                assert len(tools) == 6 and all(t.description for t in tools)

                # list_sheets -> load_data -> aggregate, in this one session
                sheets = payload(await session.call_tool("list_sheets", {"path": "sales.xlsx"}))
                assert sheets == {"sheets": [{"name": "Sales", "rows": 8, "columns": 4}]}

                loaded = payload(await session.call_tool("load_data", {"path": "sales.xlsx", "sheet": "Sales"}))
                dataset_id = loaded["dataset_id"]
                assert dataset_id == "ds1" and loaded["rows"] == 8
                assert loaded["columns"]["order_date"].startswith("datetime64")
                assert len(loaded["preview"]) == 5

                call = {"dataset_id": dataset_id, "column": "amount", "agg": "sum", "group_by": ["region"]}
                grouped = payload(await session.call_tool("aggregate", call))
                expected = sales.groupby("region")["amount"].sum()
                assert {g["region"]: g["value"] for g in grouped["groups"]} == pytest.approx(expected.to_dict())
                assert grouped["rows_used"] == 8 and grouped["rows_dropped_null"] == 0

                # the id is still valid after more calls, including a second load_data (ds2)
                second = payload(await session.call_tool("load_data", {"path": "sales.xlsx"}))
                assert second["dataset_id"] == "ds2"
                again = payload(await session.call_tool("aggregate", call))
                assert again == grouped
                total = payload(await session.call_tool(
                    "aggregate", {"dataset_id": dataset_id, "column": "amount", "agg": "sum"}))
                assert total["value"] == pytest.approx(sales["amount"].sum())

                # the other three tools work over the same session and the same id
                corr = payload(await session.call_tool(
                    "compare_groups", {"dataset_id": dataset_id, "test": "correlation", "x": "qty", "y": "amount"}))
                assert corr["r"] == pytest.approx(sales["qty"].corr(sales["amount"]))

                ran = payload(await session.call_tool(
                    "run_pandas", {"dataset_id": dataset_id, "code": 'print(df["amount"].max())'}))
                assert float(ran["output"]) == pytest.approx(sales["amount"].max())

                drawn = payload(await session.call_tool(
                    "make_chart", {"dataset_id": dataset_id, "kind": "bar", "x": "region", "y": "amount"}))
                png = Path(drawn["path"])
                assert png.is_file() and png.stat().st_size > 1000
                assert png.parent == (workspace["out_dir"] / "charts").resolve()

                # failures are ordinary results with an "error" key, not protocol errors
                unknown = payload(await session.call_tool(
                    "aggregate", {"dataset_id": "ds99", "column": "amount", "agg": "sum"}))
                assert "ds99" in unknown["error"] and unknown["loaded"] == ["ds1", "ds2"]
                outside = payload(await session.call_tool("list_sheets", {"path": "../sales.xlsx"}))
                assert outside == {"error": "path outside allowed directory"}
                blocked = payload(await session.call_tool(
                    "run_pandas", {"dataset_id": dataset_id, "code": "import os"}))
                assert blocked["error"].startswith("blocked by guardrail")

    asyncio.run(asyncio.wait_for(scenario(), TIMEOUT))


def test_stdout_carries_only_mcp_messages(workspace):
    """Speak raw JSON-RPC to the server: every stdout line must be a protocol message."""
    def request(msg_id: int, method: str, params: dict) -> dict:
        return {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params}

    messages = [
        request(1, "initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                                  "clientInfo": {"name": "smoke-test", "version": "0"}}),
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        request(2, "tools/list", {}),
        request(3, "tools/call", {"name": "list_sheets", "arguments": {"path": "sales.xlsx"}}),
        request(4, "tools/call", {"name": "load_data", "arguments": {"path": "sales.xlsx"}}),
        request(5, "tools/call", {"name": "run_pandas",
                                  "arguments": {"dataset_id": "ds1", "code": 'print("worker output")'}}),
        request(6, "tools/call", {"name": "make_chart",
                                  "arguments": {"dataset_id": "ds1", "kind": "hist", "x": "amount"}}),
        request(7, "tools/call", {"name": "list_sheets", "arguments": {"path": "../nope.xlsx"}}),
    ]
    proc = subprocess.Popen(
        [sys.executable, str(SERVER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, env={**os.environ, **server_env(workspace)},
    )
    watchdog = threading.Timer(TIMEOUT, proc.kill)  # a hung server must fail the test, not freeze it
    watchdog.start()
    lines: list[str] = []
    try:
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(("\n".join(json.dumps(m) for m in messages) + "\n").encode())
        proc.stdin.flush()
        seen: set[int] = set()
        while seen != {1, 2, 3, 4, 5, 6, 7}:  # keep stdin open until every request is answered
            raw = proc.stdout.readline()
            if not raw:
                break
            line = raw.decode("utf-8").strip()
            if line:
                lines.append(line)
                message = json.loads(line)  # raises if anything non-JSON was printed
                if isinstance(message, dict) and "id" in message:
                    seen.add(message["id"])
        proc.stdin.close()
        rest = proc.stdout.read().decode("utf-8")
        lines += [ln for ln in rest.splitlines() if ln.strip()]
        proc.wait(timeout=30)
    finally:
        watchdog.cancel()
        if proc.poll() is None:
            proc.kill()
        err = proc.stderr.read().decode("utf-8") if proc.stderr else ""
        for stream in (proc.stdout, proc.stderr):
            if stream:
                stream.close()
    assert lines, err
    decoded = []
    for line in lines:
        message = json.loads(line)
        assert message["jsonrpc"] == "2.0", line
        decoded.append(message)
    assert sorted(m["id"] for m in decoded if "id" in m) == [1, 2, 3, 4, 5, 6, 7]
    assert len(next(m for m in decoded if m.get("id") == 2)["result"]["tools"]) == 6

    # logs went to stderr instead: tool names and sizes, no cell values or code
    assert "tool=list_sheets" in err and "tool=load_data" in err and "tool=run_pandas" in err
    assert "worker output" not in err and "North" not in err
