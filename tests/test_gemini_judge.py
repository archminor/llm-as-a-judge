"""Tests for Gemini Judge support."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from llm_judge.config import load_run_config
from llm_judge.llm_client import (
    CompletionResult,
    GeminiAPIError,
    _extract_gemini_finish_reason,
    _extract_gemini_text,
    _extract_gemini_usage,
    _messages_to_gemini_contents,
    judge_chat_completion,
)
from llm_judge.models import JudgeRef, RunConfig


# ── Message conversion ───────────────────────────────────


class TestMessagesToGeminiContents:
    def test_user_only(self):
        messages = [{"role": "user", "content": "Hello"}]
        contents, sys_inst = _messages_to_gemini_contents(messages)
        assert len(contents) == 1
        assert contents[0]["role"] == "user"
        assert contents[0]["parts"] == [{"text": "Hello"}]
        assert sys_inst is None

    def test_system_message_extracted(self):
        messages = [
            {"role": "system", "content": "You are a judge."},
            {"role": "user", "content": "Evaluate this."},
        ]
        contents, sys_inst = _messages_to_gemini_contents(messages)
        assert len(contents) == 1
        assert contents[0]["role"] == "user"
        assert sys_inst == {"parts": [{"text": "You are a judge."}]}

    def test_assistant_becomes_model(self):
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "Bye"},
        ]
        contents, sys_inst = _messages_to_gemini_contents(messages)
        assert len(contents) == 3
        assert contents[0]["role"] == "user"
        assert contents[1]["role"] == "model"
        assert contents[2]["role"] == "user"
        assert sys_inst is None


# ── Response extraction ──────────────────────────────────


_SAMPLE_GEMINI_RESPONSE = {
    "candidates": [
        {
            "content": {
                "parts": [
                    {"text": '{"per_metric": {"accuracy": 5}, "overall_score": 4.5}'}
                ]
            },
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 100,
        "candidatesTokenCount": 50,
    },
}


class TestExtractGeminiText:
    def test_basic(self):
        text = _extract_gemini_text(_SAMPLE_GEMINI_RESPONSE)
        parsed = json.loads(text)
        assert parsed["per_metric"]["accuracy"] == 5
        assert parsed["overall_score"] == 4.5

    def test_multi_parts(self):
        resp = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "part1"}, {"text": "part2"}]
                    }
                }
            ]
        }
        assert _extract_gemini_text(resp) == "part1part2"

    def test_no_candidates_raises(self):
        with pytest.raises(ValueError, match="no candidates"):
            _extract_gemini_text({"candidates": []})


class TestExtractGeminiFinishReason:
    def test_stop(self):
        assert _extract_gemini_finish_reason(_SAMPLE_GEMINI_RESPONSE) == "STOP"

    def test_empty(self):
        assert _extract_gemini_finish_reason({"candidates": []}) is None


class TestExtractGeminiUsage:
    def test_basic(self):
        inp, out = _extract_gemini_usage(_SAMPLE_GEMINI_RESPONSE)
        assert inp == 100
        assert out == 50

    def test_missing(self):
        inp, out = _extract_gemini_usage({})
        assert inp is None
        assert out is None


# ── judge_chat_completion (OpenAI path) ──────────────────


class TestJudgeChatCompletionOpenAI:
    @patch("llm_judge.llm_client.chat_completion")
    def test_openai_path(self, mock_chat):
        mock_client = MagicMock()

        mock_choice = MagicMock()
        mock_choice.message.content = '{"overall_score": 5}'
        mock_choice.finish_reason = "stop"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_chat.return_value = mock_response

        result = judge_chat_completion(
            vendor="openai",
            model="gpt-4o",
            messages=[{"role": "user", "content": "test"}],
            client=mock_client,
        )

        assert isinstance(result, CompletionResult)
        assert result.text == '{"overall_score": 5}'
        assert result.input_tokens == 10
        assert result.output_tokens == 20

    @patch("llm_judge.llm_client.chat_completion")
    def test_openai_path_with_endpoint(self, mock_chat):
        mock_client = MagicMock()

        mock_choice = MagicMock()
        mock_choice.message.content = "{}"
        mock_choice.finish_reason = "stop"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        mock_chat.return_value = mock_response

        result = judge_chat_completion(
            vendor="azure-openai",
            model="gpt-4o",
            messages=[{"role": "user", "content": "test"}],
            client=mock_client,
            endpoint="https://custom.endpoint.com",
        )

        assert result.input_tokens is None

    @patch("llm_judge.llm_client.chat_completion")
    def test_openai_path_generation_params_forwarded(self, mock_chat):
        """generation_params must be forwarded to chat_completion on OpenAI path."""
        mock_client = MagicMock()

        mock_choice = MagicMock()
        mock_choice.message.content = "{}"
        mock_choice.finish_reason = "stop"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        mock_chat.return_value = mock_response

        judge_chat_completion(
            vendor="openai",
            model="gpt-4o",
            messages=[{"role": "user", "content": "test"}],
            client=mock_client,
            generation_params={"max_tokens": 2048, "top_p": 0.9, "temperature": 0.3},
        )

        call_kwargs = mock_chat.call_args
        # temperature should be overridden by generation_params (0.3 instead of default 0)
        assert call_kwargs.kwargs["temperature"] == 0.3
        assert call_kwargs.kwargs["max_tokens"] == 2048
        assert call_kwargs.kwargs["top_p"] == 0.9

    @patch("llm_judge.llm_client.chat_completion")
    def test_openai_path_default_temperature_when_no_params(self, mock_chat):
        """Default temperature=0 should be used when generation_params is None."""
        mock_client = MagicMock()

        mock_choice = MagicMock()
        mock_choice.message.content = "{}"
        mock_choice.finish_reason = "stop"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        mock_chat.return_value = mock_response

        judge_chat_completion(
            vendor="openai",
            model="gpt-4o",
            messages=[{"role": "user", "content": "test"}],
            client=mock_client,
        )

        call_kwargs = mock_chat.call_args
        assert call_kwargs.kwargs["temperature"] == 0


# ── judge_chat_completion (Gemini path) ──────────────────


class TestJudgeChatCompletionGemini:
    @patch("llm_judge.llm_client._gemini_generate_content")
    @patch("llm_judge.llm_client.resolve_vendor_env")
    def test_gemini_path(self, mock_env, mock_generate):
        mock_env.return_value = ("test-key", "https://custom-gemini.example.com")
        mock_generate.return_value = _SAMPLE_GEMINI_RESPONSE

        result = judge_chat_completion(
            vendor="gemini",
            model="gemini-2.0-flash",
            messages=[
                {"role": "system", "content": "You are a judge."},
                {"role": "user", "content": "Evaluate this."},
            ],
            client=MagicMock(),
        )

        assert isinstance(result, CompletionResult)
        parsed = json.loads(result.text)
        assert parsed["per_metric"]["accuracy"] == 5
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.finish_reason == "STOP"

        mock_generate.assert_called_once()
        call_kwargs = mock_generate.call_args
        assert call_kwargs.kwargs["json_mode"] is True
        assert call_kwargs.kwargs["generation_params"]["temperature"] == 0

    @patch("llm_judge.llm_client._gemini_generate_content")
    @patch("llm_judge.llm_client.resolve_vendor_env")
    def test_gemini_with_custom_endpoint(self, mock_env, mock_generate):
        mock_env.return_value = ("test-key", "")
        mock_generate.return_value = _SAMPLE_GEMINI_RESPONSE

        judge_chat_completion(
            vendor="gemini",
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": "test"}],
            client=MagicMock(),
            endpoint="https://my-gemini-proxy.example.com/v1",
        )

        call_kwargs = mock_generate.call_args
        assert call_kwargs.kwargs["endpoint"] == "https://my-gemini-proxy.example.com/v1"

    @patch("llm_judge.llm_client._gemini_generate_content")
    @patch("llm_judge.llm_client.resolve_vendor_env")
    def test_gemini_default_endpoint(self, mock_env, mock_generate):
        mock_env.return_value = ("test-key", "")
        mock_generate.return_value = _SAMPLE_GEMINI_RESPONSE

        judge_chat_completion(
            vendor="gemini",
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": "test"}],
            client=MagicMock(),
        )

        call_kwargs = mock_generate.call_args
        assert "generativelanguage.googleapis.com" in call_kwargs.kwargs["endpoint"]

    @patch("llm_judge.llm_client._gemini_generate_content")
    @patch("llm_judge.llm_client.resolve_vendor_env")
    def test_gemini_generation_params_merged(self, mock_env, mock_generate):
        mock_env.return_value = ("test-key", "")
        mock_generate.return_value = _SAMPLE_GEMINI_RESPONSE

        judge_chat_completion(
            vendor="gemini",
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": "test"}],
            client=MagicMock(),
            generation_params={"topK": 40, "temperature": 0.5},
        )

        call_kwargs = mock_generate.call_args
        params = call_kwargs.kwargs["generation_params"]
        # User-specified temperature overrides default
        assert params["temperature"] == 0.5
        assert params["topK"] == 40

    @patch("llm_judge.llm_client._gemini_generate_content")
    @patch("llm_judge.llm_client.resolve_vendor_env")
    def test_gemini_json_mode_not_overridable(self, mock_env, mock_generate):
        """responseMimeType in generation_params must not override JSON mode."""
        mock_env.return_value = ("test-key", "")
        mock_generate.return_value = _SAMPLE_GEMINI_RESPONSE

        judge_chat_completion(
            vendor="gemini",
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": "test"}],
            client=MagicMock(),
            generation_params={"responseMimeType": "text/plain"},
        )

        call_kwargs = mock_generate.call_args
        # json_mode=True must win over user-supplied responseMimeType
        assert call_kwargs.kwargs["json_mode"] is True


# ── _gemini_generate_content body construction ───────────


class TestGeminiGenerateContentBody:
    @patch("llm_judge.llm_client.httpx.post")
    def test_json_mode_survives_generation_params_override(self, mock_post):
        """responseMimeType='application/json' must not be overridden by generation_params."""
        from llm_judge.llm_client import _gemini_generate_content

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _SAMPLE_GEMINI_RESPONSE
        mock_post.return_value = mock_resp

        _gemini_generate_content(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": "hi"}],
            api_key="key",
            endpoint="https://example.com",
            json_mode=True,
            generation_params={"responseMimeType": "text/plain", "topK": 40},
        )

        body = mock_post.call_args.kwargs["json"]
        gen_config = body["generationConfig"]
        assert gen_config["responseMimeType"] == "application/json"
        assert gen_config["topK"] == 40


# ── GeminiAPIError ───────────────────────────────────────


class TestGeminiAPIError:
    def test_error_attributes(self):
        err = GeminiAPIError(429, "rate limit exceeded")
        assert err.status_code == 429
        assert "429" in str(err)
        assert "rate limit" in str(err)


# ── JudgeRef model with endpoint and generation_params ───


class TestJudgeRefExtended:
    def test_defaults(self):
        ref = JudgeRef(
            judge_id="j1",
            vendor="gemini",
            model_id="gemini-2.0-flash",
            rubric_version="v1",
        )
        assert ref.endpoint is None
        assert ref.generation_params == {}

    def test_with_endpoint_and_params(self):
        ref = JudgeRef(
            judge_id="j1",
            vendor="gemini",
            model_id="gemini-2.0-flash",
            rubric_version="v1",
            endpoint="https://custom.example.com",
            generation_params={"topK": 40},
        )
        assert ref.endpoint == "https://custom.example.com"
        assert ref.generation_params["topK"] == 40


# ── run-config validation with vendor=gemini ─────────────


class TestRunConfigGeminiVendor:
    def _base_config(self, judge_vendor: str = "gemini") -> dict:
        return {
            "run_id": "test-gemini",
            "dataset": {"testcases_path": "data/testcases.jsonl"},
            "candidates": [
                {
                    "candidate_id": "c1",
                    "vendor": "openai",
                    "model_id": "gpt-4o",
                },
                {
                    "candidate_id": "c2",
                    "vendor": "openai",
                    "model_id": "gpt-4o",
                },
            ],
            "judges": [
                {
                    "judge_id": "gemini-judge-1",
                    "vendor": judge_vendor,
                    "model_id": "gemini-2.0-flash",
                    "rubric_version": "v1",
                }
            ],
            "protocol": {
                "evaluation_mode": "pairwise",
                "aggregation": {"method": "mean"},
            },
        }

    def test_gemini_vendor_pydantic(self):
        cfg = self._base_config()
        rc = RunConfig.model_validate(cfg)
        assert rc.judges[0].vendor == "gemini"

    def test_gemini_vendor_with_endpoint(self):
        cfg = self._base_config()
        cfg["judges"][0]["endpoint"] = "https://custom-gemini.example.com"
        cfg["judges"][0]["generation_params"] = {"topK": 40}
        rc = RunConfig.model_validate(cfg)
        assert rc.judges[0].endpoint == "https://custom-gemini.example.com"
        assert rc.judges[0].generation_params["topK"] == 40

    def test_gemini_vendor_load_run_config(self, tmp_path):
        cfg = self._base_config()
        path = tmp_path / "run-config.yaml"
        path.write_text(yaml.dump(cfg, allow_unicode=True))
        rc = load_run_config(path)
        assert rc.judges[0].vendor == "gemini"

    def test_gemini_vendor_with_endpoint_load_run_config(self, tmp_path):
        cfg = self._base_config()
        cfg["judges"][0]["endpoint"] = "https://custom.example.com"
        cfg["judges"][0]["generation_params"] = {"topK": 40}
        path = tmp_path / "run-config.yaml"
        path.write_text(yaml.dump(cfg, allow_unicode=True))
        rc = load_run_config(path)
        assert rc.judges[0].endpoint == "https://custom.example.com"
        assert rc.judges[0].generation_params["topK"] == 40
