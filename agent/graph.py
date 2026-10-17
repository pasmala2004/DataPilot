"""The agent loop (LangGraph).

    START -> load -> router -> start_task -> agent <-> tools -> observe
                                   ^                              |
                                   +---- next task ---------------+--> retry (back to agent)
                                                                  +--> all tasks done -> decision -> END

* ``load``      (code)  list_sheets + load_data on the requested file; errors end the run.
* ``router``    (LLM)   structured plan: 1-4 tasks for the "analysis" / "chart" agents, or a clarification.
* ``agent``     (LLM)   ReAct loop with ONLY that agent's tools, at most 6 tool rounds per attempt.
* ``tools``     (code)  runs the calls on the ONE open MCP session and appends to the results log.
* ``observe``   (code)  agent/observe.py: retry with feedback, finish the task, or mark it failed.
* ``decision``  (LLM)   short answer from the clean observations only; cites tool_call_ids.

The report writer and the number check are Day 3 and are not built here.
Nothing in the trace contains raw data rows: tool outputs are never printed, only tool names,
arguments and ok/error status.
"""

import json
import uuid
from typing import Any, Callable, Literal, Optional, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agent.llm import ainvoke_structured, describe_exception
from agent.mcp_client import ExcelTools
from agent.observe import DEFAULT_MAX_RETRIES, entry_problems, observe, resolved_problems
from agent.prompts import (
    MAX_TASKS,
    MAX_TOOL_ITERATIONS,
    agent_system_prompt,
    decision_messages,
    no_results_answer,
    router_messages,
)

TraceFn = Callable[[str], None]
MAX_TRACE_VALUE_CHARS = 120
MAX_TRACE_LINE_CHARS = 320
VAGUE_FALLBACK = (
    "I could not work out what to compute from that request. "
    "Could you rephrase it or name the columns you care about?"
)


# --------------------------------------------------------------------------- #
# Router output schema
# --------------------------------------------------------------------------- #
class RouterTask(BaseModel):
    """One unit of work for one agent."""

    agent: Literal["analysis", "chart"] = Field(description='"analysis" computes numbers; "chart" draws one chart')
    task: str = Field(description="Self-contained instruction naming the exact columns, aggregation and filters")


class RouterPlan(BaseModel):
    """The router's plan: tasks (1 to 4), or a clarification question and no tasks."""

    tasks: list[RouterTask] = Field(default_factory=list, description="1 to 4 tasks; empty only when asking for clarification")
    clarification: Optional[str] = Field(default=None, description="One short question when the request is too vague, else null")


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
class AgentState(TypedDict, total=False):
    """Everything the graph carries. Nodes return only the keys they change."""

    # inputs
    question: str
    file: str
    sheet: Optional[str]
    # load
    error: Optional[str]
    dataset_id: str
    schema: dict[str, Any]
    # router
    tasks: list[dict[str, str]]
    clarification: Optional[str]
    # current task
    task_idx: int
    task_attempt: int  # retries already used on the current task
    self_corrections: int  # problems the agent fixed by itself within an attempt, this task so far
    attempt_start: int  # index into `results` where the current attempt began
    tool_iterations: int
    messages: list[BaseMessage]
    route: str
    # outputs
    results: list[dict[str, Any]]
    task_status: list[dict[str, Any]]
    answer: Optional[str]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def message_text(content: Any) -> str:
    """Text of a message whose content may be a string or a list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in (None, "text"):
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return "" if content is None else str(content)


def parse_tool_result(message: Any) -> dict[str, Any]:
    """Turn the ToolMessage of an MCP tool call into the dict the tool returned.

    The server returns every result as JSON text. Protocol-level failures (for example an
    argument of the wrong type) arrive as plain text and become ``{"error": text}``.
    """
    text = message_text(getattr(message, "content", message)).strip()
    try:
        data = json.loads(text)
    except ValueError:
        return {"error": text[:500] or "empty tool result"}
    if not isinstance(data, dict):
        return {"error": "tool returned a result that is not a JSON object"}
    if getattr(message, "status", None) == "error" and "error" not in data:
        return {"error": text[:500]}
    return data


async def call_tool(tool: BaseTool, name: str, args: dict[str, Any], call_id: str) -> dict[str, Any]:
    """Run one MCP tool on the open session. Never raises: failures become ``{"error": ...}``."""
    try:
        message = await tool.ainvoke({"name": name, "args": args, "id": call_id, "type": "tool_call"})
    except Exception as exc:  # noqa: BLE001 - transport/validation failures must not crash the run
        return {"error": f"{type(exc).__name__}: {str(exc)[:400]}"}
    return parse_tool_result(message)


def format_args(args: dict[str, Any]) -> str:
    """Compact one-line argument text for the trace; long strings (e.g. code) are cut."""

    def shorten(value: Any) -> Any:
        if isinstance(value, str) and len(value) > MAX_TRACE_VALUE_CHARS:
            return value[: MAX_TRACE_VALUE_CHARS - 3] + "..."
        if isinstance(value, dict):
            return {k: shorten(v) for k, v in value.items()}
        if isinstance(value, list):
            return [shorten(v) for v in value]
        return value

    text = json.dumps(shorten(args), default=str, separators=(",", ":"), ensure_ascii=False)
    return text if len(text) <= MAX_TRACE_LINE_CHARS else text[: MAX_TRACE_LINE_CHARS - 3] + "..."


def clean_ai_message(ai: AIMessage, used_ids: set[str]) -> AIMessage:
    """A fresh AIMessage with plain-text content, valid tool calls and unique ids.

    * Malformed tool calls (``invalid_tool_calls``) are dropped: chat APIs reject a history
      where a tool call has no matching tool message.
    * A missing or repeated ``tool_call_id`` is replaced (9 alphanumeric characters, which every
      provider accepts), because the decision node cites ids as evidence and they must be unique.
    """
    calls = []
    for call in ai.tool_calls or []:
        call_id = call.get("id") or ""
        while not call_id or call_id in used_ids:
            call_id = uuid.uuid4().hex[:9]
        used_ids.add(call_id)
        calls.append({"name": call["name"], "args": call.get("args") or {}, "id": call_id, "type": "tool_call"})
    return AIMessage(content=message_text(ai.content), tool_calls=calls)


def normalise_plan(plan: RouterPlan) -> tuple[list[dict[str, str]], Optional[str]]:
    """Enforce the router contract in code: chart tasks last, at most 4 tasks, clarification XOR tasks."""
    clarification = (plan.clarification or "").strip() or None
    if clarification:
        return [], clarification
    tasks = [{"agent": t.agent, "task": t.task.strip()} for t in plan.tasks if t.task.strip()]
    if not tasks:
        return [], VAGUE_FALLBACK
    tasks.sort(key=lambda t: t["agent"] == "chart")  # stable: keeps the router's order within each group
    return tasks[:MAX_TASKS], None


def recursion_limit_for(max_retries: int, max_tool_iterations: int) -> int:
    """LangGraph steps needed in the worst case (default limit 25 is far too small for retries)."""
    per_attempt = 2 * max_tool_iterations + 3
    return 10 + MAX_TASKS * (1 + (max(0, max_retries) + 1) * per_attempt)


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #
def build_graph(
    llm: BaseChatModel,
    tools: ExcelTools,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    max_tool_iterations: int = MAX_TOOL_ITERATIONS,
    trace: TraceFn = lambda line: None,
):
    """Compile the agent graph.

    Args:
        llm: chat model used by the router, the agents and the decision node. It must support
            ``bind_tools`` and ``with_structured_output``.
        tools: the MCP tools of the ONE open session (see ``mcp_client.excel_session``).
        max_retries: retries per task after a rejected attempt; 0 turns retrying off.
        max_tool_iterations: tool rounds allowed per attempt.
        trace: called with each human-readable trace line.
    """
    tool_map = {
        "analysis": {t.name: t for t in tools.analysis_tools},
        "chart": {t.name: t for t in tools.chart_tools},
    }

    # ---- load (code, no LLM) ------------------------------------------------
    async def load_node(state: AgentState) -> dict[str, Any]:
        file = state["file"]
        sheets = await call_tool(tools.by_name["list_sheets"], "list_sheets", {"path": file}, "load-1")
        if "error" in sheets:
            return {"error": f"Cannot open '{file}': {sheets['error']}"}
        names = [s["name"] for s in sheets.get("sheets", [])]
        if not names:
            return {"error": f"'{file}' has no sheets"}
        sheet = state.get("sheet") or names[0]
        if not state.get("sheet") and len(names) > 1:
            trace(f"LOAD no --sheet given, using the first sheet '{sheet}' (others: {names[1:]})")
        loaded = await call_tool(tools.by_name["load_data"], "load_data", {"path": file, "sheet": sheet}, "load-2")
        if "error" in loaded:
            return {"error": f"Cannot load '{file}': {loaded['error']}"}
        schema = {k: loaded[k] for k in ("rows", "columns", "nulls", "date_ranges", "warnings") if k in loaded}
        trace(
            f"LOAD {file} [{sheet}] -> {loaded['dataset_id']}: "
            f"{schema.get('rows')} rows, {len(schema.get('columns', {}))} columns"
        )
        # the 5-row preview is deliberately not kept: no cell values go into state or prompts
        return {"dataset_id": loaded["dataset_id"], "schema": schema, "sheet": sheet}

    # ---- router (LLM, structured output) -----------------------------------
    async def router_node(state: AgentState) -> dict[str, Any]:
        try:  # one repair step if the structured reply does not parse (small local models need it)
            plan = await ainvoke_structured(llm, RouterPlan, router_messages(state["question"], state["dataset_id"], state["schema"]))
        except Exception as exc:  # noqa: BLE001
            return {"error": f"The router failed: {describe_exception(exc)}"}
        tasks, clarification = normalise_plan(plan)
        if clarification:
            trace("ROUTER -> clarification needed, no tasks")
            return {"tasks": [], "clarification": clarification}
        trace(f"ROUTER -> {len(tasks)} task(s)")
        for i, task in enumerate(tasks, start=1):
            trace(f"  {i}. [{task['agent']}] {task['task'][:MAX_TRACE_LINE_CHARS]}")
        return {"tasks": tasks, "clarification": None}

    # ---- start_task ----------------------------------------------------------
    async def start_task_node(state: AgentState) -> dict[str, Any]:
        idx = state.get("task_idx", -1) + 1
        task = state["tasks"][idx]
        trace(f"TASK {idx + 1}/{len(state['tasks'])} [{task['agent']}]")
        return {
            "task_idx": idx,
            "task_attempt": 0,
            "self_corrections": 0,
            "attempt_start": len(state["results"]),
            "tool_iterations": 0,
            "messages": [
                SystemMessage(content=agent_system_prompt(task["agent"], state["dataset_id"], state["schema"])),
                HumanMessage(content=task["task"]),
            ],
        }

    # ---- agent (LLM with only this agent's tools) ----------------------------
    async def agent_node(state: AgentState) -> dict[str, Any]:
        agent = state["tasks"][state["task_idx"]]["agent"]
        bound = llm.bind_tools(list(tool_map[agent].values()))
        raw = await bound.ainvoke(state["messages"])
        used_ids = {r["tool_call_id"] for r in state["results"]}
        ai = clean_ai_message(raw, used_ids)
        if ai.tool_calls and state["tool_iterations"] >= max_tool_iterations:
            trace(f"AGENT stopped: tool iteration limit ({max_tool_iterations}) reached")
            ai = AIMessage(content=message_text(ai.content) or "(stopped: tool iteration limit reached)")
        return {"messages": state["messages"] + [ai]}

    def route_after_agent(state: AgentState) -> Literal["tools", "observe"]:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else "observe"

    # ---- tools (code) --------------------------------------------------------
    async def tools_node(state: AgentState) -> dict[str, Any]:
        idx = state["task_idx"]
        available = tool_map[state["tasks"][idx]["agent"]]
        last = state["messages"][-1]
        assert isinstance(last, AIMessage)
        messages = list(state["messages"])
        results = list(state["results"])
        for call in last.tool_calls:  # sequential: the server is synchronous and shares one store
            name, args, call_id = call["name"], call["args"], call["id"]
            tool = available.get(name)
            if tool is None:
                output: dict[str, Any] = {"error": f"unknown tool '{name}'; available tools: {sorted(available)}"}
            else:
                output = await call_tool(tool, name, args, call_id)
            status = f"error: {str(output['error'])[:160]}" if "error" in output else "ok"
            trace(f"TOOL {name} {format_args(args)} -> {status}")
            results.append({"task_idx": idx, "tool_call_id": call_id, "tool": name, "args": args, "output": output})
            messages.append(
                ToolMessage(
                    content=json.dumps(output, default=str, separators=(",", ":")),
                    tool_call_id=call_id,
                    name=name,
                )
            )
        return {"messages": messages, "results": results, "tool_iterations": state["tool_iterations"] + 1}

    # ---- observe (code, no LLM) ----------------------------------------------
    async def observe_node(state: AgentState) -> dict[str, Any]:
        idx = state["task_idx"]
        task = state["tasks"][idx]
        attempt = state["results"][state["attempt_start"]:]
        verdict = observe(attempt, retries_used=state["task_attempt"], max_retries=max_retries)
        self_corrections = state.get("self_corrections", 0) + resolved_problems(attempt)
        if verdict.action == "retry":
            trace(f"OBSERVE -> RETRY({'; '.join(verdict.reasons)})")
            return {
                "messages": state["messages"] + [HumanMessage(content=verdict.feedback)],
                "task_attempt": state["task_attempt"] + 1,
                "self_corrections": self_corrections,
                "attempt_start": len(state["results"]),
                "tool_iterations": 0,
                "route": "agent",
            }
        last = state["messages"][-1]
        status = {
            "idx": idx,
            "agent": task["agent"],
            "task": task["task"],
            "status": "finished" if verdict.action == "finish" else "failed",
            "retries": state["task_attempt"],
            "self_corrections": self_corrections,
            "reasons": list(verdict.reasons),
            "note": message_text(last.content)[:300] if isinstance(last, AIMessage) else "",
        }
        more = idx + 1 < len(state["tasks"])
        outcome = "NEXT TASK" if more else "FINISH"
        if verdict.action == "failed":
            trace(f"OBSERVE -> FAILED({'; '.join(verdict.reasons)}) -> {outcome}")
        else:
            trace(f"OBSERVE -> {outcome}")
        return {"task_status": state["task_status"] + [status], "route": "start_task" if more else "decision"}

    # ---- decision (LLM, or code when there is nothing to report) -------------
    async def decision_node(state: AgentState) -> dict[str, Any]:
        clean = [r for r in state["results"] if not entry_problems(r)]
        if not clean:
            trace("DECISION -> no verified observations, answer composed without the LLM")
            return {"answer": no_results_answer(state["task_status"])}
        reply = await llm.ainvoke(decision_messages(state["question"], state["task_status"], clean))
        answer = message_text(reply.content).strip()
        trace(f"DECISION -> answer written from {len(clean)} observation(s)")
        return {"answer": answer}

    # ---- wiring ----------------------------------------------------------------
    graph = StateGraph(AgentState)
    graph.add_node("load", load_node)
    graph.add_node("router", router_node)
    graph.add_node("start_task", start_task_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("observe", observe_node)
    graph.add_node("decision", decision_node)

    graph.add_edge(START, "load")
    graph.add_conditional_edges("load", lambda s: "end" if s.get("error") else "router", {"end": END, "router": "router"})
    graph.add_conditional_edges(
        "router",
        lambda s: "end" if s.get("error") or s.get("clarification") else "start_task",
        {"end": END, "start_task": "start_task"},
    )
    graph.add_edge("start_task", "agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "observe": "observe"})
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges(
        "observe", lambda s: s["route"], {"agent": "agent", "start_task": "start_task", "decision": "decision"}
    )
    graph.add_edge("decision", END)
    return graph.compile()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
async def run_agent(
    question: str,
    file: str,
    *,
    llm: BaseChatModel,
    tools: ExcelTools,
    sheet: Optional[str] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    max_tool_iterations: int = MAX_TOOL_ITERATIONS,
    trace: Optional[TraceFn] = None,
) -> dict[str, Any]:
    """Run the whole loop for one question and return everything it produced.

    Must be awaited while the MCP session of ``tools`` is open (one session per run).

    Returns a dict with:
        ``answer``         short answer (None when a clarification or an error ended the run)
        ``clarification``  question for the user when the request was too vague, else None
        ``error``          message when the run could not start or crashed, else None
        ``results``        the full results log: {task_idx, tool_call_id, tool, args, output} per tool call
        ``task_status``    per task: idx, agent, task, status (finished/failed), retries, self_corrections, reasons, note
        ``tasks``, ``dataset_id``, ``schema``, ``trace`` (list of trace lines)
    """
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")
    lines: list[str] = []

    def emit(line: str) -> None:
        lines.append(line)
        if trace is not None:
            trace(line)

    graph = build_graph(llm, tools, max_retries=max_retries, max_tool_iterations=max_tool_iterations, trace=emit)
    initial: AgentState = {
        "question": question,
        "file": file,
        "sheet": sheet,
        "error": None,
        "clarification": None,
        "answer": None,
        "tasks": [],
        "results": [],
        "task_status": [],
    }
    try:
        final = await graph.ainvoke(
            initial, config={"recursion_limit": recursion_limit_for(max_retries, max_tool_iterations)}
        )
    except GraphRecursionError:
        final = {**initial, "error": "The agent exceeded its step limit and was stopped."}
    except Exception as exc:  # noqa: BLE001 - e.g. an LLM API failure; report it, do not crash the caller
        final = {**initial, "error": describe_exception(exc)}
    if final.get("error"):
        emit(f"ERROR {final['error']}")
    return {
        "question": question,
        "file": file,
        "sheet": final.get("sheet"),
        "dataset_id": final.get("dataset_id"),
        "schema": final.get("schema"),
        "tasks": final.get("tasks", []),
        "task_status": final.get("task_status", []),
        "results": final.get("results", []),
        "answer": final.get("answer"),
        "clarification": final.get("clarification"),
        "error": final.get("error"),
        "trace": lines,
    }
