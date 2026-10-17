"""Shared test helpers: a scripted chat model and builders for its replies. No network, no API key."""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from pydantic import Field

from agent.graph import run_agent
from agent.mcp_client import ANALYSIS_TOOL_NAMES, excel_session

TIMEOUT = 120  # seconds per scenario: a hung server must fail the test, not freeze the run


# =========================================================================== #
# Scripted model
# =========================================================================== #
def _tool_name(tool: Any) -> str:
    if isinstance(tool, BaseTool):
        return tool.name
    if isinstance(tool, type):
        return tool.__name__
    if isinstance(tool, dict):
        return tool.get("name") or tool["function"]["name"]
    return str(tool)


class ScriptedChatModel(BaseChatModel):
    """Replays queued AIMessages in order; raises if the graph asks for more than was scripted.

    ``bind_tools`` returns the model itself and records the tool names it was given in
    ``bound``; ``seen`` records the messages of every call. ``with_structured_output`` (used by
    the router) works through the standard LangChain path: the router reply is an AIMessage with
    a ``RouterPlan`` tool call.
    """

    script: list[AIMessage] = Field(default_factory=list)
    seen: list[list[BaseMessage]] = Field(default_factory=list)
    bound: list[list[str]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":  # type: ignore[override]
        self.bound.append([_tool_name(t) for t in tools])
        return self

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        if not self.script:
            raise AssertionError("ScriptedChatModel ran out of scripted responses (the graph asked for one more call)")
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])


def plan(*tasks: tuple[str, str], clarification: str | None = None) -> AIMessage:
    """Scripted router reply."""
    args = {"tasks": [{"agent": a, "task": t} for a, t in tasks], "clarification": clarification}
    return AIMessage(content="", tool_calls=[{"name": "RouterPlan", "args": args, "id": "router-1"}])


def call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    """Scripted agent reply that calls one tool."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def say(text: str) -> AIMessage:
    """Scripted plain-text reply (an agent's closing line or the decision answer)."""
    return AIMessage(content=text)


def agg(column: str = "amount", how: str = "sum", **extra: Any) -> dict[str, Any]:
    return {"dataset_id": "ds1", "column": column, "agg": how, **extra}


# =========================================================================== #
# Fixture-workbook helpers
# =========================================================================== #
CANARIES = [f"Acme-{7000 + i}-Corp" for i in range(10)]


def run_graph(workspace: dict[str, Any], script: list[AIMessage], *, file: str = "sales.xlsx",
              sheet: str | None = "orders", max_retries: int = 2, **kwargs: Any) -> tuple[ScriptedChatModel, dict]:
    """Run the real graph against the real server in ONE MCP session."""
    llm = ScriptedChatModel(script=list(script))

    async def go() -> dict:
        async with excel_session(workspace["data_dir"], workspace["out_dir"]) as tools:
            return await run_agent("the question", file, llm=llm, tools=tools, sheet=sheet,
                                   max_retries=max_retries, **kwargs)

    return llm, asyncio.run(asyncio.wait_for(go(), TIMEOUT))


def assert_agents_only_got_their_own_tools(llm: ScriptedChatModel) -> None:
    """Every bind_tools call is the router's schema, the analysis agent's 5 tools, or make_chart alone."""
    allowed = [["RouterPlan"], ["make_chart"]]
    for bound in llm.bound:
        assert bound in allowed or set(bound) == set(ANALYSIS_TOOL_NAMES), bound
        assert len(bound) == len(set(bound))  # no tool listed twice


def trace_text(result: dict) -> str:
    return "\n".join(result["trace"])


def prompt_text(llm: ScriptedChatModel) -> str:
    return "\n".join(str(m.content) for call_messages in llm.seen for m in call_messages)
