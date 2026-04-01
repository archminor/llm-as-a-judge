"""LLM client wrapper with retry logic for multiple vendors (OpenAI / Gemini)."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any

import httpx
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    RateLimitError,
    AzureOpenAI,
    OpenAI,
)
from tenacity import (
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

from llm_judge.config import resolve_vendor_env


RETRYABLE_ERRORS = (APITimeoutError, APIConnectionError, RateLimitError)
_DEFAULT_REQUEST_TIMEOUT_SECONDS = float(
    os.getenv("LLM_JUDGE_REQUEST_TIMEOUT_SECONDS", "120")
)
_RETRY_ATTEMPTS = max(1, int(os.getenv("LLM_JUDGE_RETRY_ATTEMPTS", "2")))

_GEMINI_DEFAULT_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta"


# ── Normalized response wrapper ──────────────────────────


@dataclass
class CompletionResult:
    """Provider-independent wrapper for chat completion responses."""

    text: str
    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    raw_response: Any = field(default=None, repr=False)


# ── Retryable error helpers ──────────────────────────────


class GeminiAPIError(Exception):
    """Error from the Gemini native API."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(f"Gemini API error {status_code}: {message}")


def _is_retryable_api_error(exc: Exception) -> bool:
    """Retry only on server-side API errors (>=500)."""
    if isinstance(exc, APIError):
        status = getattr(exc, "status_code", None)
        return status is None or status >= 500
    if isinstance(exc, GeminiAPIError):
        return exc.status_code >= 500
    return False


RETRYABLE_GEMINI_ERRORS = (httpx.ConnectError, httpx.TimeoutException)


# ── OpenAI-compatible client ─────────────────────────────

_thread_local = threading.local()


def create_client(vendor: str, endpoint: str | None = None) -> OpenAI:
    """Create an OpenAI-compatible client based on vendor name.

    For ``vendor="gemini"`` this function is not used; use
    :func:`judge_chat_completion` directly instead.
    """
    api_key, env_endpoint = resolve_vendor_env(vendor)
    endpoint = endpoint or env_endpoint

    if vendor == "azure-openai":
        return AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version="2024-12-01-preview",
        )
    else:
        # Generic OpenAI-compatible endpoint
        return OpenAI(
            api_key=api_key,
            base_url=endpoint or None,
        )


def get_thread_local_openai_client(
    vendor: str, endpoint: str | None = None
) -> OpenAI:
    """Return a thread-local cached OpenAI-compatible client.

    Each (vendor, endpoint, api_key) combination gets a single client per
    thread, so parallel workers do not share connection state.
    """
    api_key, env_endpoint = resolve_vendor_env(vendor)
    resolved_endpoint = endpoint or env_endpoint

    cache: dict = getattr(_thread_local, "clients", None)
    if cache is None:
        cache = {}
        _thread_local.clients = cache

    cache_key = (vendor, resolved_endpoint, api_key)
    if cache_key not in cache:
        cache[cache_key] = create_client(vendor, endpoint)
    return cache[cache_key]


@retry(
    retry=retry_if_exception(_is_retryable_api_error)
    | retry_if_exception_type(RETRYABLE_ERRORS),
    stop=stop_after_attempt(_RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=2, max=30) + wait_random(0, 2),
    reraise=True,
)
def chat_completion(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    **kwargs: Any,
) -> Any:
    """Call chat completions with automatic retry on transient errors."""
    if "max_tokens" in kwargs and any(x in model for x in ("gpt-5", "o1", "o3")):
        if "max_completion_tokens" not in kwargs:
            kwargs["max_completion_tokens"] = kwargs["max_tokens"]
        kwargs.pop("max_tokens")

    timeout = kwargs.pop("timeout", _DEFAULT_REQUEST_TIMEOUT_SECONDS)

    return client.chat.completions.create(
        model=model,
        messages=messages,
        timeout=timeout,
        **kwargs,
    )


# ── Gemini native transport ──────────────────────────────


def _messages_to_gemini_contents(
    messages: list[dict[str, str]],
) -> tuple[list[dict], dict | None]:
    """Convert OpenAI-style messages to Gemini contents + systemInstruction."""
    system_instruction: dict | None = None
    contents: list[dict] = []

    for msg in messages:
        role = msg["role"]
        text = msg["content"]
        if role == "system":
            system_instruction = {"parts": [{"text": text}]}
        elif role == "assistant":
            contents.append({"role": "model", "parts": [{"text": text}]})
        else:
            contents.append({"role": "user", "parts": [{"text": text}]})

    return contents, system_instruction


def _extract_gemini_text(response_json: dict) -> str:
    candidates = response_json.get("candidates", [])
    if not candidates:
        raise ValueError("Gemini response contains no candidates")
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts)


def _extract_gemini_finish_reason(response_json: dict) -> str | None:
    candidates = response_json.get("candidates", [])
    if not candidates:
        return None
    return candidates[0].get("finishReason")


def _extract_gemini_usage(response_json: dict) -> tuple[int | None, int | None]:
    usage = response_json.get("usageMetadata", {})
    return (
        usage.get("promptTokenCount"),
        usage.get("candidatesTokenCount"),
    )


@retry(
    retry=retry_if_exception(_is_retryable_api_error)
    | retry_if_exception_type(RETRYABLE_GEMINI_ERRORS),
    stop=stop_after_attempt(_RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=2, max=30) + wait_random(0, 2),
    reraise=True,
)
def _gemini_generate_content(
    model: str,
    messages: list[dict[str, str]],
    api_key: str,
    endpoint: str,
    json_mode: bool = False,
    response_schema: dict | None = None,
    generation_params: dict[str, Any] | None = None,
) -> dict:
    """Call the Gemini generateContent API and return raw response JSON."""
    url = f"{endpoint}/models/{model}:generateContent"
    contents, system_instruction = _messages_to_gemini_contents(messages)

    body: dict[str, Any] = {"contents": contents}
    if system_instruction:
        body["systemInstruction"] = system_instruction

    gen_config: dict[str, Any] = {}
    if generation_params:
        gen_config.update(generation_params)
    if json_mode or response_schema:
        gen_config["responseMimeType"] = "application/json"
    if response_schema:
        gen_config["responseSchema"] = response_schema
    if gen_config:
        body["generationConfig"] = gen_config

    timeout = float(os.getenv("LLM_JUDGE_REQUEST_TIMEOUT_SECONDS", "120"))

    resp = httpx.post(
        url,
        params={"key": api_key},
        json=body,
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise GeminiAPIError(resp.status_code, resp.text)
    return resp.json()


# ── Gemini inference helper ───────────────────────────────


def gemini_inference_completion(
    model: str,
    messages: list[dict[str, str]],
    endpoint: str | None = None,
    generation_params: dict[str, Any] | None = None,
    response_schema: dict | None = None,
) -> CompletionResult:
    """Gemini native inference with optional schema-constrained output."""
    api_key, env_endpoint = resolve_vendor_env("gemini")
    endpoint = endpoint or env_endpoint or _GEMINI_DEFAULT_ENDPOINT

    params = {}
    if generation_params:
        params.update(generation_params)
    # Gemini uses maxOutputTokens instead of max_tokens
    if "max_tokens" in params:
        params["maxOutputTokens"] = params.pop("max_tokens")

    raw = _gemini_generate_content(
        model=model,
        messages=messages,
        api_key=api_key,
        endpoint=endpoint,
        response_schema=response_schema,
        generation_params=params,
    )
    text = _extract_gemini_text(raw)
    finish_reason = _extract_gemini_finish_reason(raw)
    input_tokens, output_tokens = _extract_gemini_usage(raw)

    return CompletionResult(
        text=text,
        finish_reason=finish_reason,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        raw_response=raw,
    )


# ── Provider-independent judge helper ────────────────────


def judge_chat_completion(
    vendor: str,
    model: str,
    messages: list[dict[str, str]],
    client: OpenAI,
    endpoint: str | None = None,
    generation_params: dict[str, Any] | None = None,
) -> CompletionResult:
    """Provider-independent chat completion for Judge / Consistency stages."""
    if vendor == "gemini":
        return _gemini_judge_completion(
            model=model,
            messages=messages,
            endpoint=endpoint,
            generation_params=generation_params,
        )

    extra_kwargs: dict[str, Any] = {"temperature": 0}
    if generation_params:
        extra_kwargs.update(generation_params)
    response = chat_completion(
        client=client,
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        **extra_kwargs,
    )
    text = response.choices[0].message.content or ""
    usage = response.usage
    return CompletionResult(
        text=text,
        finish_reason=response.choices[0].finish_reason,
        input_tokens=usage.prompt_tokens if usage else None,
        output_tokens=usage.completion_tokens if usage else None,
        raw_response=response,
    )


def _gemini_judge_completion(
    model: str,
    messages: list[dict[str, str]],
    endpoint: str | None = None,
    generation_params: dict[str, Any] | None = None,
) -> CompletionResult:
    api_key, env_endpoint = resolve_vendor_env("gemini")
    endpoint = endpoint or env_endpoint or _GEMINI_DEFAULT_ENDPOINT

    params = {"temperature": 0}
    if generation_params:
        params.update(generation_params)

    raw = _gemini_generate_content(
        model=model,
        messages=messages,
        api_key=api_key,
        endpoint=endpoint,
        json_mode=True,
        generation_params=params,
    )
    text = _extract_gemini_text(raw)
    finish_reason = _extract_gemini_finish_reason(raw)
    input_tokens, output_tokens = _extract_gemini_usage(raw)

    return CompletionResult(
        text=text,
        finish_reason=finish_reason,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        raw_response=raw,
    )
