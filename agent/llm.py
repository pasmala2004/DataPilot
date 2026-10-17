"""The language model: always OpenRouter (an OpenAI-compatible API).

Privacy: with OpenRouter the column names, the small load_data preview (5 rows), tool results and the printed
output of run_pandas (capped) are sent to OpenRouter and the model provider. Do not use confidential data.

Configuration (environment, overridable by ``get_llm(model=...)`` and the ``--model`` flag of run.py):

* ``OPENROUTER_API_KEY`` (required). It is read ONLY from the environment and is never logged, printed, stored,
  shown in the UI, or included in a repr, a trace, an error message or an eval result (``scrub`` removes it from
  any text that might carry it).
* ``OPENROUTER_MODEL`` (default ``openai/gpt-4o-mini``; ``MODEL_NAME`` is accepted as an alias), the model must
  support tool calling.
* ``OPENROUTER_TIMEOUT_S`` (per request, default 120). Transport errors (429, 5xx, timeouts) are retried twice
  by the client with its built-in backoff. Temperature is 0.

Nothing is built or contacted at import time. ``get_llm`` only builds the client; pass ``check=True`` to also
verify the key and the model id with two cheap GET requests (see ``check_openrouter``).

Small models are weaker at structured output, so ``ainvoke_structured`` adds ONE repair step: if the reply
cannot be parsed, it asks again with the parse error appended, then fails cleanly with ``StructuredOutputError``.
The router and the report writer use it.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_TIMEOUT_SECONDS = 120.0
CHECK_TIMEOUT_SECONDS = 10.0
TRANSPORT_RETRIES = 2
MAX_PARSE_ERROR_CHARS = 600
KEY_VAR = "OPENROUTER_API_KEY"

S = TypeVar("S", bound=BaseModel)


# --------------------------------------------------------------------------- #
# Errors (run.py turns every LLMConfigError into exit code 2). Messages never contain the key.
# --------------------------------------------------------------------------- #
class LLMConfigError(RuntimeError):
    """The language model is not configured or not usable. The message says how to fix it."""


class MissingKeyError(LLMConfigError):
    """``OPENROUTER_API_KEY`` is not set."""


class KeyRejectedError(LLMConfigError):
    """OpenRouter answered 401/403."""


class NoCreditsError(LLMConfigError):
    """OpenRouter answered 402."""


class RateLimitedError(LLMConfigError):
    """OpenRouter answered 429 (after the client's own retries)."""


class ModelNotFoundError(LLMConfigError):
    """The model id does not exist (or has no available provider)."""


class ModelNoToolsError(LLMConfigError):
    """The model does not list "tools" in ``supported_parameters``."""


class OpenRouterUnreachableError(LLMConfigError):
    """The request never got an answer (no network, DNS, timeout)."""


class StructuredOutputError(RuntimeError):
    """The model's structured reply could not be parsed, even after one repair attempt."""


MESSAGES = {
    "missing_key": (
        "OPENROUTER_API_KEY is not set. Set it in your environment, then start again. "
        "bash: export OPENROUTER_API_KEY=...   PowerShell: $env:OPENROUTER_API_KEY = \"...\""
    ),
    "rejected": "OpenRouter: the key was rejected (401/403). Check OPENROUTER_API_KEY.",
    "no_credits": "OpenRouter: no credits (402). Add credits to your OpenRouter account.",
    "rate_limited": "OpenRouter: rate limited (429), wait and retry.",
    "not_found": "OpenRouter: model not found (404). Check the model id (for example openai/gpt-4o-mini).",
    "unreachable": "Cannot reach OpenRouter (network error or timeout). Check your connection and try again.",
}
STATUS_ERRORS = {
    401: (KeyRejectedError, "rejected"), 403: (KeyRejectedError, "rejected"), 402: (NoCreditsError, "no_credits"),
    404: (ModelNotFoundError, "not_found"), 429: (RateLimitedError, "rate_limited"),
}


# --------------------------------------------------------------------------- #
# Keeping the key out of text
# --------------------------------------------------------------------------- #
_KEY_SHAPES = re.compile(r"sk-or-[A-Za-z0-9_\-]{8,}|(?i:bearer)\s+[A-Za-z0-9._\-]{8,}")


def scrub(text: str, env: Optional[Mapping[str, str]] = None) -> str:
    """Replace the configured key (and anything shaped like an OpenRouter key or a Bearer token) by ``[redacted]``."""
    source = os.environ if env is None else env
    key = (source.get(KEY_VAR) or "").strip()
    if len(key) >= 4:
        text = text.replace(key, "[redacted]")
    return _KEY_SHAPES.sub("[redacted]", text)


def key_present(env: Optional[Mapping[str, str]] = None) -> bool:
    """True when ``OPENROUTER_API_KEY`` is set and not empty (never returns or shows the key)."""
    return bool(((os.environ if env is None else env).get(KEY_VAR) or "").strip())


def error_for_status(status: int) -> Optional[LLMConfigError]:
    """The clean error for an HTTP status (401/402/403/404/429), or ``None`` for any other status."""
    pair = STATUS_ERRORS.get(int(status))
    return pair[0](MESSAGES[pair[1]]) if pair else None


_CODE = re.compile(r"error code:\s*(\d{3})", re.I)


def describe_exception(exc: BaseException) -> str:
    """A short, key-free message for any exception raised while the model was in use.

    API status errors (401/402/403/404/429), timeouts and connection failures become the fixed messages above;
    anything else is ``TypeName: text`` with the key scrubbed and the text capped.
    """
    if isinstance(exc, LLMConfigError):
        return scrub(str(exc))
    status = getattr(exc, "status_code", None)
    if status is None:
        match = _CODE.search(str(exc))
        status = int(match.group(1)) if match else None
    mapped = error_for_status(status) if status is not None else None
    if mapped is not None:
        return str(mapped)
    name = type(exc).__name__
    if name in ("APITimeoutError", "ConnectError", "ConnectTimeout", "ReadTimeout", "APIConnectionError", "TimeoutException"):
        return MESSAGES["unreachable"]
    if status == 400 and "valid model" in str(exc).lower():
        return MESSAGES["not_found"]
    return scrub(f"{name}: {' '.join(str(exc).split())[:400]}")


def hint_for_error(message: str, settings: Any = None) -> str:
    """A fix-it hint for an error text produced mid-run (empty when there is nothing to add)."""
    text = message.lower()
    for needle, key in (("key was rejected", "rejected"), ("no credits", "no_credits"), ("rate limited", "rate_limited"),
                        ("model not found", "not_found"), ("cannot reach openrouter", "unreachable")):
        if needle in text:
            return MESSAGES[key]
    match = _CODE.search(message)
    mapped = error_for_status(int(match.group(1))) if match else None
    return str(mapped) if mapped else ""


def is_rate_limit(text: Optional[str]) -> bool:
    """True when an error text says OpenRouter rate limited the request."""
    return bool(text) and "rate limited" in text.lower()


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LLMSettings:
    """Resolved OpenRouter configuration. The key is excluded from repr/str/compare on purpose."""

    model: str
    timeout_s: float = DEFAULT_TIMEOUT_SECONDS
    base_url: str = BASE_URL
    api_key: str = field(default="", repr=False, compare=False)


def resolve_model_name(model: Optional[str] = None, env: Optional[Mapping[str, str]] = None) -> str:
    """Only the model id (flag, ``OPENROUTER_MODEL``, ``MODEL_NAME`` or the default); needs no key. For labels and meta."""
    env = os.environ if env is None else env
    return (model or env.get("OPENROUTER_MODEL") or env.get("MODEL_NAME") or DEFAULT_MODEL).strip()


def resolve_settings(model: Optional[str] = None, env: Optional[Mapping[str, str]] = None) -> LLMSettings:
    """Work out key, model and timeout from the environment. No network access.

    ``model`` (from the command line) overrides ``OPENROUTER_MODEL`` / ``MODEL_NAME``. Settings of the previous
    local-model setup are ignored.

    Raises:
        MissingKeyError: ``OPENROUTER_API_KEY`` is not set.
        LLMConfigError: a malformed model id or timeout.
    """
    env = os.environ if env is None else env
    key = (env.get(KEY_VAR) or "").strip()
    if not key:
        raise MissingKeyError(MESSAGES["missing_key"])
    name = (model or env.get("OPENROUTER_MODEL") or env.get("MODEL_NAME") or DEFAULT_MODEL).strip()
    if not name or re.search(r"\s", name):
        raise LLMConfigError("The model id must be a non-empty OpenRouter id without spaces, for example openai/gpt-4o-mini.")
    raw = (env.get("OPENROUTER_TIMEOUT_S") or "").strip()
    try:
        timeout_s = float(raw) if raw else DEFAULT_TIMEOUT_SECONDS
    except ValueError:
        raise LLMConfigError(f"OPENROUTER_TIMEOUT_S must be a number of seconds, got {raw!r}.") from None
    if not timeout_s > 0:
        raise LLMConfigError(f"OPENROUTER_TIMEOUT_S must be positive, got {raw!r}.")
    return LLMSettings(model=name, timeout_s=timeout_s, api_key=key)


# --------------------------------------------------------------------------- #
# Reachability / key / model check
# --------------------------------------------------------------------------- #
def _get(url: str, settings: LLMSettings, client: Any) -> Any:
    """GET with the key as a Bearer header. Transport failures become ``OpenRouterUnreachableError``."""
    import httpx

    try:
        return (client or httpx).get(url, headers={"Authorization": f"Bearer {settings.api_key}"}, timeout=CHECK_TIMEOUT_SECONDS)
    except httpx.HTTPError:
        raise OpenRouterUnreachableError(MESSAGES["unreachable"]) from None


def check_openrouter(settings: LLMSettings, *, client: Any = None) -> Optional[str]:
    """Confirm the key works and the model id exists and can call tools.

    Request 1: ``GET /models`` (Bearer header): the model id must be listed, and when the entry has
    ``supported_parameters`` it must contain "tools". Request 2: ``GET /key`` (OpenRouter's key-info endpoint),
    because ``/models`` is public and answers even for a wrong key; only 401/403 from it count as a rejected key,
    any other answer is ignored (it must never block a working setup).

    Returns:
        ``None`` when all is confirmed, or a warning text when tool support could not be checked
        (the field is absent): the run may go on.

    Raises:
        LLMConfigError subclasses: unreachable, key rejected, no credits, rate limited, model not found, no tools.
    """
    response = _get(settings.base_url + "/models", settings, client)
    status = getattr(response, "status_code", 200)
    if status != 200:
        raise error_for_status(status) or OpenRouterUnreachableError(f"OpenRouter answered HTTP {status} when listing models.")
    try:
        payload = response.json()
        models = payload.get("data") if isinstance(payload, dict) else None
    except ValueError:
        models = None
    if not isinstance(models, list):
        raise OpenRouterUnreachableError("OpenRouter's model list could not be read. Try again in a moment.")
    entry = next((m for m in models if isinstance(m, dict) and m.get("id") == settings.model), None)
    if entry is None:
        raise ModelNotFoundError(f"Model not found on OpenRouter: '{settings.model}'. Check the model id.")
    warning: Optional[str] = None
    params = entry.get("supported_parameters")
    if params is None:
        warning = (f"Could not confirm that '{settings.model}' supports tools (OpenRouter did not list its parameters). "
                   "If the run fails with a tools error, pick another model.")
    elif "tools" not in params:
        raise ModelNoToolsError(f"The model '{settings.model}' cannot call tools, which DataPilot needs. Pick another model.")
    key_response = _get(settings.base_url + "/key", settings, client)
    if getattr(key_response, "status_code", 200) in (401, 403):
        raise KeyRejectedError(MESSAGES["rejected"])
    return warning


# --------------------------------------------------------------------------- #
# Building the model
# --------------------------------------------------------------------------- #
def build_llm(settings: LLMSettings) -> BaseChatModel:
    """Build the OpenRouter chat client for resolved settings (no network access)."""
    from langchain_openai import ChatOpenAI
    from pydantic import SecretStr

    return ChatOpenAI(
        model=settings.model,
        base_url=settings.base_url,
        api_key=SecretStr(settings.api_key),
        temperature=0,
        timeout=settings.timeout_s,
        max_retries=TRANSPORT_RETRIES,
    )


def get_llm(model: Optional[str] = None, *, check: bool = False) -> BaseChatModel:
    """Build the chat model (temperature 0).

    Args:
        model: OpenRouter model id; defaults to ``$OPENROUTER_MODEL`` / ``$MODEL_NAME`` / ``openai/gpt-4o-mini``.
        check: also verify the key and the model id (two cheap GET requests).

    Raises:
        LLMConfigError: (or a subclass) when the configuration is missing, wrong or unreachable.
    """
    settings = resolve_settings(model)
    if check:
        check_openrouter(settings)
    return build_llm(settings)


def llm_is_reachable(model: Optional[str] = None) -> bool:
    """True if a key is set, OpenRouter answers and the model exists. For test skips."""
    try:
        get_llm(model, check=True)
    except LLMConfigError:
        return False
    return True


# --------------------------------------------------------------------------- #
# Structured output with one repair step
# --------------------------------------------------------------------------- #
def _error_text(error: Any) -> str:
    text = " ".join(str(error).split()) or type(error).__name__
    return text[:MAX_PARSE_ERROR_CHARS]


async def _structured_once(llm: BaseChatModel, schema: type[S], messages: Sequence[Any], method: str) -> tuple[Optional[S], str]:
    """One structured call. Returns (parsed, "") or (None, parse error text). Transport errors propagate."""
    structured = llm.with_structured_output(schema, method=method, include_raw=True)
    reply = await structured.ainvoke(list(messages))
    parsed = reply.get("parsed") if isinstance(reply, dict) else reply
    error = reply.get("parsing_error") if isinstance(reply, dict) else None
    if isinstance(parsed, schema):
        return parsed, ""
    if error is not None:
        return None, _error_text(error)
    return None, f"the reply did not contain a {schema.__name__} (no structured output was returned)"


async def ainvoke_structured(
    llm: BaseChatModel, schema: type[S], messages: Sequence[Any], *, method: str = "function_calling"
) -> S:
    """Ask for ``schema``; if the reply does not parse, retry ONCE with the parse error appended.

    Only parse/validation failures are repaired. Network and API errors (bad key, no credits, rate limit)
    are raised as they are, because asking again would not help.

    Raises:
        StructuredOutputError: the reply was still unusable after the repair attempt.
    """
    parsed, error = await _structured_once(llm, schema, messages, method)
    if parsed is not None:
        return parsed
    repair = list(messages) + [
        (
            "human",
            f"Your previous reply could not be parsed: {error}\n"
            f"Reply again with a valid {schema.__name__}, following the schema exactly. "
            "Do not add any text outside the required structure.",
        )
    ]
    parsed, second_error = await _structured_once(llm, schema, repair, method)
    if parsed is not None:
        return parsed
    raise StructuredOutputError(
        f"the model did not return a valid {schema.__name__} after one repair attempt: {second_error}"
    )
