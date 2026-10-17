"""Tests for agent/llm.py (OpenRouter settings, key hygiene, error mapping, /models check, structured-output
repair) and the run.py flags. No network and no real key: HTTP goes through httpx.MockTransport or fake clients."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr

import run as cli
from agent import llm as llm_module
from agent.graph import RouterPlan
from agent.llm import (
    DEFAULT_MODEL,
    KeyRejectedError,
    LLMConfigError,
    MissingKeyError,
    ModelNoToolsError,
    ModelNotFoundError,
    NoCreditsError,
    OpenRouterUnreachableError,
    RateLimitedError,
    StructuredOutputError,
    ainvoke_structured,
    build_llm,
    check_openrouter,
    describe_exception,
    error_for_status,
    get_llm,
    hint_for_error,
    is_rate_limit,
    key_present,
    llm_is_reachable,
    resolve_settings,
    scrub,
)
from scripted import ScriptedChatModel, plan, say

KEY = "sk-or-v1-0123456789abcdefSECRET"
ENV = {"OPENROUTER_API_KEY": KEY}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts with no LLM configuration at all (old Ollama variables included)."""
    for name in ("OPENROUTER_API_KEY", "OPENROUTER_MODEL", "MODEL_NAME", "OPENROUTER_TIMEOUT_S",
                 "OLLAMA_MODEL", "OLLAMA_HOST", "OLLAMA_NUM_CTX"):
        monkeypatch.delenv(name, raising=False)


# =========================================================================== #
# Settings resolution
# =========================================================================== #
def test_missing_key_is_a_clear_error_with_both_shells():
    with pytest.raises(MissingKeyError) as info:
        resolve_settings(env={})
    message = str(info.value)
    assert "OPENROUTER_API_KEY" in message and "export" in message and "$env:" in message
    assert isinstance(info.value, LLMConfigError)


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_key_counts_as_missing(blank):
    with pytest.raises(MissingKeyError):
        resolve_settings(env={"OPENROUTER_API_KEY": blank})
    assert not key_present({"OPENROUTER_API_KEY": blank})


def test_defaults_model_and_timeout():
    s = resolve_settings(env=ENV)
    assert s.model == DEFAULT_MODEL == "openai/gpt-4o-mini" and s.timeout_s == 120.0
    assert key_present(ENV)


def test_model_precedence_flag_then_openrouter_model_then_alias_then_default():
    env = {**ENV, "OPENROUTER_MODEL": "a/primary", "MODEL_NAME": "b/alias"}
    assert resolve_settings("c/flag", env=env).model == "c/flag"
    assert resolve_settings(env=env).model == "a/primary"
    assert resolve_settings(env={**ENV, "MODEL_NAME": "b/alias"}).model == "b/alias"


def test_timeout_env_and_bad_values():
    assert resolve_settings(env={**ENV, "OPENROUTER_TIMEOUT_S": "45.5"}).timeout_s == 45.5
    for bad in ("abc", "0", "-3"):
        with pytest.raises(LLMConfigError):
            resolve_settings(env={**ENV, "OPENROUTER_TIMEOUT_S": bad})
    with pytest.raises(LLMConfigError):
        resolve_settings("two words", env=ENV)


def test_old_ollama_variables_are_ignored(monkeypatch):
    env = {**ENV, "OLLAMA_MODEL": "gemma4:26b", "OLLAMA_HOST": "http://h:1", "OLLAMA_NUM_CTX": "999"}
    s = resolve_settings(env=env)
    assert s.model == DEFAULT_MODEL and "gemma" not in repr(s) and "999" not in repr(s)
    with pytest.raises(MissingKeyError):                       # an Ollama setup alone is not a configuration
        resolve_settings(env={"OLLAMA_MODEL": "gemma4:26b"})
    assert not hasattr(llm_module, "check_ollama") and not hasattr(llm_module, "DEFAULT_OLLAMA_HOST")


def test_building_the_client_makes_no_network_call_and_sets_transport_options(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    model = get_llm("x/y")
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "x/y" and model.temperature == 0 and model.max_retries == 2
    assert model.openai_api_base == "https://openrouter.ai/api/v1" and model.request_timeout == 120.0


# =========================================================================== #
# The key never shows up
# =========================================================================== #
def test_key_is_not_in_repr_or_str_of_settings_or_the_client():
    settings = resolve_settings(env=ENV)
    for text in (repr(settings), str(settings), repr(build_llm(settings)), str(build_llm(settings))):
        assert KEY not in text and "SECRET" not in text
    assert isinstance(build_llm(settings).openai_api_key, SecretStr)


def test_scrub_removes_the_configured_key_and_key_shapes():
    text = f"failed with {KEY} and Bearer abcdefghijklmnop and sk-or-v1-zzzzzzzzzzzz"
    out = scrub(text, env=ENV)
    assert KEY not in out and "abcdefghijklmnop" not in out and "zzzzzzzzzzzz" not in out and out.count("[redacted]") == 3


def test_every_fixed_message_and_status_error_is_key_free():
    for message in llm_module.MESSAGES.values():
        assert KEY not in message and "sk-or-" not in message
    for status in (401, 402, 403, 404, 429):
        assert KEY not in str(error_for_status(status))
    assert error_for_status(500) is None and error_for_status(200) is None


def test_describe_exception_scrubs_unknown_errors(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    text = describe_exception(ValueError(f"weird failure with {KEY} inside"))
    assert KEY not in text and text.startswith("ValueError:") and "[redacted]" in text


# =========================================================================== #
# Error mapping through a real OpenAI client on a fake httpx transport
# =========================================================================== #
def client_for(status: int, body: dict[str, Any] | None = None) -> ChatOpenAI:
    """A ChatOpenAI that talks to a fake transport answering ``status`` (no retries: the test must be instant)."""
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == f"Bearer {KEY}"
        return httpx.Response(status, json=body or {"error": {"message": f"upstream said {status} (key {KEY})", "code": status}})

    return ChatOpenAI(model="x/y", base_url="https://openrouter.ai/api/v1", api_key=SecretStr(KEY), max_retries=0,
                      http_client=httpx.Client(transport=httpx.MockTransport(handler)))


@pytest.mark.parametrize("status,error_class,fragment", [
    (401, KeyRejectedError, "key was rejected"),
    (403, KeyRejectedError, "key was rejected"),
    (402, NoCreditsError, "no credits"),
    (404, ModelNotFoundError, "model not found"),
    (429, RateLimitedError, "rate limited"),
])
def test_http_errors_become_clean_messages_without_the_key(status, error_class, fragment, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    with pytest.raises(Exception) as info:
        client_for(status).invoke("hello")
    message = describe_exception(info.value)
    assert fragment in message and KEY not in message and "Traceback" not in message and "upstream" not in message
    assert message == str(error_for_status(status)) and isinstance(error_for_status(status), error_class)
    assert hint_for_error(message) == message                          # the same text doubles as the fix-it hint


def test_a_status_inside_an_error_text_is_mapped_too():
    assert "no credits" in describe_exception(RuntimeError("Error code: 402 - {'error': 'x'}"))
    assert "not a valid model" not in describe_exception(RuntimeError("Error code: 404 - not a valid model ID"))
    assert is_rate_limit(str(error_for_status(429))) and not is_rate_limit("crash") and not is_rate_limit(None)


def test_transport_failures_map_to_unreachable():
    assert "Cannot reach OpenRouter" in describe_exception(httpx.ConnectError("boom"))
    assert "Cannot reach OpenRouter" in describe_exception(httpx.ReadTimeout("slow"))


def test_mid_run_hints():
    assert hint_for_error("something odd") == ""
    assert "rate limited" in hint_for_error("The router failed: OpenRouter says rate limited (429). Wait")


# =========================================================================== #
# check=True: GET /models (+ /key) with a fake client
# =========================================================================== #
def models_payload(*entries: dict[str, Any]) -> dict[str, Any]:
    return {"data": list(entries)}


def mock_client(models_status: int = 200, models_body: Any = None, key_status: int = 200, seen: list[Any] | None = None) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        if request.url.path.endswith("/models"):
            return httpx.Response(models_status, json=models_body if models_body is not None else
                                  models_payload({"id": "openai/gpt-4o-mini", "supported_parameters": ["tools", "temperature"]}))
        return httpx.Response(key_status, json={"data": {"label": "x"}})

    return httpx.Client(transport=httpx.MockTransport(handler))


def settings_for(model: str = "openai/gpt-4o-mini") -> Any:
    return resolve_settings(model, env=ENV)


def test_check_passes_and_sends_the_key_only_as_a_bearer_header():
    seen: list[httpx.Request] = []
    assert check_openrouter(settings_for(), client=mock_client(seen=seen)) is None
    assert [r.url.path for r in seen] == ["/api/v1/models", "/api/v1/key"]
    assert all(r.headers["authorization"] == f"Bearer {KEY}" for r in seen)
    assert all(KEY not in str(r.url) for r in seen)


def test_unknown_model_id_is_model_not_found():
    with pytest.raises(ModelNotFoundError) as info:
        check_openrouter(settings_for("nope/missing"), client=mock_client())
    assert "nope/missing" in str(info.value) and KEY not in str(info.value)


def test_model_without_tools_is_refused():
    body = models_payload({"id": "openai/gpt-4o-mini", "supported_parameters": ["temperature"]})
    with pytest.raises(ModelNoToolsError):
        check_openrouter(settings_for(), client=mock_client(models_body=body))


def test_missing_supported_parameters_is_only_a_warning():
    body = models_payload({"id": "openai/gpt-4o-mini"})
    warning = check_openrouter(settings_for(), client=mock_client(models_body=body))
    assert warning and "Could not confirm" in warning


@pytest.mark.parametrize("status,error_class", [(401, KeyRejectedError), (403, KeyRejectedError), (402, NoCreditsError), (429, RateLimitedError)])
def test_models_status_errors_are_mapped(status, error_class):
    with pytest.raises(error_class):
        check_openrouter(settings_for(), client=mock_client(models_status=status, models_body={"error": "x"}))


def test_a_rejected_key_is_caught_by_the_key_endpoint_because_models_is_public():
    with pytest.raises(KeyRejectedError):
        check_openrouter(settings_for(), client=mock_client(key_status=401))


def test_other_key_endpoint_answers_never_block_a_working_setup():
    assert check_openrouter(settings_for(), client=mock_client(key_status=500)) is None


def test_unreachable_and_unreadable_answers():
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    with pytest.raises(OpenRouterUnreachableError):
        check_openrouter(settings_for(), client=httpx.Client(transport=httpx.MockTransport(down)))
    with pytest.raises(OpenRouterUnreachableError):
        check_openrouter(settings_for(), client=mock_client(models_body={"unexpected": True}))


def test_get_llm_check_is_off_by_default_and_never_runs_at_import(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    calls: list[Any] = []
    monkeypatch.setattr(llm_module, "check_openrouter", lambda settings, **kw: calls.append(settings))
    get_llm("x/y")
    assert calls == []
    get_llm("x/y", check=True)
    assert len(calls) == 1 and calls[0].model == "x/y"


def test_llm_is_reachable_is_false_without_a_key():
    assert llm_is_reachable() is False


# =========================================================================== #
# Structured output: one repair step
# =========================================================================== #
class Verdict(BaseModel):
    answer: str
    score: int


def bad_plan() -> AIMessage:
    """A router reply that fails validation ('banana' is not an agent)."""
    args = {"tasks": [{"agent": "banana", "task": "x"}], "clarification": None}
    return AIMessage(content="", tool_calls=[{"name": "RouterPlan", "args": args, "id": "r-bad"}])


def run(coro: Any) -> Any:
    return asyncio.run(asyncio.wait_for(coro, 30))


MESSAGES = [("system", "plan it"), ("human", "the question")]


def test_a_valid_first_reply_needs_no_repair():
    model = ScriptedChatModel(script=[plan(("analysis", "Sum amount."))])
    result = run(ainvoke_structured(model, RouterPlan, MESSAGES))
    assert result.tasks[0].task == "Sum amount." and len(model.seen) == 1 and model.script == []


def test_a_reply_that_fails_validation_is_repaired_once_with_the_parse_error_appended():
    model = ScriptedChatModel(script=[bad_plan(), plan(("analysis", "Sum amount."))])
    result = run(ainvoke_structured(model, RouterPlan, MESSAGES))
    assert result.tasks[0].agent == "analysis" and model.script == []
    first, second = model.seen
    assert len(second) == len(first) + 1                                   # exactly one message was appended
    assert [m.content for m in second[:-1]] == [m.content for m in first]  # the original messages are untouched
    repair = second[-1]
    assert isinstance(repair, HumanMessage) and "could not be parsed" in repair.content
    assert "banana" in repair.content or "agent" in repair.content          # it carries the actual parse error
    assert "RouterPlan" in repair.content


def test_a_reply_without_any_structured_output_is_repaired_too():
    model = ScriptedChatModel(script=[say("Sure! Here is my plan in prose."), plan(("chart", "Draw it."))])
    result = run(ainvoke_structured(model, RouterPlan, MESSAGES))
    assert result.tasks[0].agent == "chart" and len(model.seen) == 2
    assert "no structured output" in model.seen[1][-1].content


def test_two_bad_replies_fail_cleanly_after_exactly_one_repair():
    model = ScriptedChatModel(script=[bad_plan(), bad_plan(), plan(("analysis", "never reached"))])
    with pytest.raises(StructuredOutputError) as info:
        run(ainvoke_structured(model, RouterPlan, MESSAGES))
    assert "after one repair attempt" in str(info.value) and "RouterPlan" in str(info.value)
    assert len(model.seen) == 2 and len(model.script) == 1                  # no third call was made


def test_the_repair_works_for_any_schema_not_just_the_router():
    good = AIMessage(content="", tool_calls=[{"name": "Verdict", "args": {"answer": "ok", "score": 3}, "id": "v1"}])
    bad = AIMessage(content="", tool_calls=[{"name": "Verdict", "args": {"answer": "ok", "score": "high"}, "id": "v0"}])
    model = ScriptedChatModel(script=[bad, good])
    assert run(ainvoke_structured(model, Verdict, MESSAGES)) == Verdict(answer="ok", score=3)
    assert "score" in model.seen[1][-1].content


class RaisingModel(BaseChatModel):
    """A model whose API call fails (Ollama down, bad key): not a parse problem."""

    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "raising"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "RaisingModel":  # type: ignore[override]
        return self

    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        raise httpx.ConnectError("Connection refused")


def test_transport_errors_are_not_repaired_and_not_swallowed():
    model = RaisingModel()
    with pytest.raises(Exception) as info:
        run(ainvoke_structured(model, RouterPlan, MESSAGES))
    assert not isinstance(info.value, StructuredOutputError) and "Connection refused" in str(info.value)
    assert model.calls == 1                                                # asking again would not help


# =========================================================================== #
# run.py flags and exit codes
# =========================================================================== #
def test_cli_passes_the_model_flag_to_the_factory():
    seen: dict[str, Any] = {}

    def factory(**kw: Any) -> BaseChatModel:
        seen.update(kw)
        raise MissingKeyError("stop here")             # we only care about what the factory received

    assert cli.main(["--file", "x.xlsx", "--model", "some/model", "q"], llm_factory=factory) == 2
    assert seen == {"model": "some/model"}
    seen.clear()
    cli.main(["--file", "x.xlsx", "q"], llm_factory=factory)
    assert seen == {"model": None}


def test_cli_has_no_provider_flag(capsys):
    with pytest.raises(SystemExit) as info:
        cli.main(["--file", "x.xlsx", "--provider", "openrouter", "q"])
    assert info.value.code == 2 and "unrecognized arguments" in capsys.readouterr().err


@pytest.mark.parametrize("error,fragment", [
    (MissingKeyError(llm_module.MESSAGES["missing_key"]), "OPENROUTER_API_KEY"),
    (KeyRejectedError(llm_module.MESSAGES["rejected"]), "key was rejected"),
    (NoCreditsError(llm_module.MESSAGES["no_credits"]), "no credits"),
    (ModelNotFoundError(llm_module.MESSAGES["not_found"]), "model not found"),
])
def test_cli_configuration_errors_exit_with_code_2_and_a_clean_message(error, fragment, capsys):
    def factory(**kw: Any) -> BaseChatModel:
        raise error

    assert cli.main(["--file", "x.xlsx", "q"], llm_factory=factory) == 2
    captured = capsys.readouterr()
    assert fragment in captured.err and captured.out == "" and "Traceback" not in captured.err and KEY not in captured.err


def test_cli_with_the_real_default_factory_and_no_key_exits_2(capsys):
    assert cli.main(["--file", "x.xlsx", "q"]) == 2
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err
