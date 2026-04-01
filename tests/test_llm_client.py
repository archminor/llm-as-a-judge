from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

from tenacity import wait_random

from llm_judge.llm_client import (
    chat_completion,
    get_thread_local_openai_client,
    judge_chat_completion,
    _gemini_generate_content,
)


def test_chat_completion_translates_max_tokens_when_completion_missing():
    client = MagicMock()
    expected = object()
    client.chat.completions.create.return_value = expected

    result = chat_completion(
        client=client,
        model="azure-gpt-51",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=1024,
    )

    assert result is expected
    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert "max_tokens" not in call_kwargs
    assert call_kwargs["max_completion_tokens"] == 1024


def test_chat_completion_keeps_explicit_max_completion_tokens():
    client = MagicMock()
    expected = object()
    client.chat.completions.create.return_value = expected

    result = chat_completion(
        client=client,
        model="azure-gpt-51",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=1024,
        max_completion_tokens=2048,
    )

    assert result is expected
    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert "max_tokens" not in call_kwargs
    assert call_kwargs["max_completion_tokens"] == 2048


# ── judge_chat_completion client argument ────────────────


def _make_mock_openai_response():
    response = MagicMock()
    response.choices[0].message.content = '{"score": 1}'
    response.choices[0].finish_reason = "stop"
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 5
    return response


def test_judge_chat_completion_uses_provided_client():
    """When a client is supplied it should be used directly."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_mock_openai_response()

    result = judge_chat_completion(
        vendor="openai",
        model="gpt-4",
        messages=[{"role": "user", "content": "hi"}],
        client=mock_client,
    )

    mock_client.chat.completions.create.assert_called_once()
    assert result.text == '{"score": 1}'


# ── get_thread_local_openai_client ───────────────────────


def _fake_resolve(vendor):
    return ("fake-key", None)


def _fake_create(vendor, endpoint=None):
    from openai import OpenAI
    # Return a plain object so identity comparison works without network
    return object()


def test_thread_local_client_same_instance_in_same_thread(monkeypatch):
    """Same (vendor, endpoint, api_key) in the same thread → same instance."""
    import llm_judge.llm_client as mod

    # Isolate thread-local state so tests don't interfere with each other
    import threading
    monkeypatch.setattr(mod, "_thread_local", threading.local())
    monkeypatch.setattr(mod, "resolve_vendor_env", _fake_resolve)

    created = []

    def fake_create(vendor, endpoint=None):
        obj = object()
        created.append(obj)
        return obj

    monkeypatch.setattr(mod, "create_client", fake_create)

    c1 = get_thread_local_openai_client("openai", None)
    c2 = get_thread_local_openai_client("openai", None)

    assert c1 is c2
    assert len(created) == 1


def test_thread_local_client_different_instances_across_threads(monkeypatch):
    """Different threads should get different client instances."""
    import llm_judge.llm_client as mod

    monkeypatch.setattr(mod, "_thread_local", threading.local())
    monkeypatch.setattr(mod, "resolve_vendor_env", _fake_resolve)

    instances: list[object] = []

    def fake_create(vendor, endpoint=None):
        obj = object()
        instances.append(obj)
        return obj

    monkeypatch.setattr(mod, "create_client", fake_create)

    results: list[object] = []

    def worker():
        c = get_thread_local_openai_client("openai", None)
        results.append(c)

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert len(results) == 2
    assert results[0] is not results[1]


def test_thread_local_client_different_vendor_different_instance(monkeypatch):
    """Different vendor keys → different instances in the same thread."""
    import llm_judge.llm_client as mod

    monkeypatch.setattr(mod, "_thread_local", threading.local())

    def resolve(vendor):
        return ("key-" + vendor, None)

    monkeypatch.setattr(mod, "resolve_vendor_env", resolve)

    created: list[object] = []

    def fake_create(vendor, endpoint=None):
        obj = object()
        created.append(obj)
        return obj

    monkeypatch.setattr(mod, "create_client", fake_create)

    c_a = get_thread_local_openai_client("vendor-a", None)
    c_b = get_thread_local_openai_client("vendor-b", None)

    assert c_a is not c_b
    assert len(created) == 2


# ── retry jitter ─────────────────────────────────────────


def test_chat_completion_retry_has_jitter():
    """chat_completion retry wait should include wait_random for jitter."""
    from tenacity import wait_combine
    wait = chat_completion.retry.wait
    # wait_combine wraps multiple wait strategies; check it contains wait_random
    assert isinstance(wait, wait_combine)
    assert any(isinstance(w, wait_random) for w in wait.wait_funcs)


def test_gemini_generate_content_retry_has_jitter():
    """_gemini_generate_content retry wait should include wait_random for jitter."""
    from tenacity import wait_combine
    wait = _gemini_generate_content.retry.wait
    assert isinstance(wait, wait_combine)
    assert any(isinstance(w, wait_random) for w in wait.wait_funcs)
