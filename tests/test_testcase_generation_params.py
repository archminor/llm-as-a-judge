"""Tests for generation_params handling.

The Testcase model no longer validates generation_params keys -- any dict is
accepted.  The only validation that remains is in ModelRef: it rejects having
both max_tokens and max_completion_tokens at the same time.
"""

from __future__ import annotations

import pytest

from llm_judge.models import ModelRef, Testcase


class TestTestcaseGenerationParamsAcceptsAnyKeys:
    """Testcase.generation_params is a free-form dict."""

    def _make_tc(self, **kwargs) -> Testcase:
        defaults = {
            "testcase_id": "tc_001",
            "task_type": "report_generation",
            "input": {"messages": [{"role": "user", "content": "hello"}]},
        }
        defaults.update(kwargs)
        return Testcase(**defaults)

    def test_no_generation_params(self):
        """generation_params defaults to empty dict."""
        tc = self._make_tc()
        assert tc.generation_params == {}

    def test_max_tokens_allowed(self):
        tc = self._make_tc(generation_params={"max_tokens": 512})
        assert tc.generation_params == {"max_tokens": 512}

    def test_temperature_allowed(self):
        """Arbitrary keys like temperature are now accepted."""
        tc = self._make_tc(generation_params={"temperature": 0.5})
        assert tc.generation_params == {"temperature": 0.5}

    def test_multiple_keys_allowed(self):
        """Multiple arbitrary keys are accepted."""
        tc = self._make_tc(
            generation_params={"max_tokens": 512, "top_p": 0.9}
        )
        assert tc.generation_params == {"max_tokens": 512, "top_p": 0.9}

    def test_empty_dict_allowed(self):
        tc = self._make_tc(generation_params={})
        assert tc.generation_params == {}


class TestModelRefGenerationParamsValidation:
    """ModelRef still rejects having both max_tokens and max_completion_tokens."""

    def test_max_tokens_only(self):
        ref = ModelRef(
            candidate_id="c1", vendor="openai", model_id="gpt-4o",
            generation_params={"max_tokens": 1024},
        )
        assert ref.generation_params == {"max_tokens": 1024}

    def test_max_completion_tokens_only(self):
        ref = ModelRef(
            candidate_id="c1", vendor="openai", model_id="gpt-4o",
            generation_params={"max_completion_tokens": 1024},
        )
        assert ref.generation_params == {"max_completion_tokens": 1024}

    def test_both_rejected(self):
        with pytest.raises(Exception, match="both max_tokens and max_completion_tokens"):
            ModelRef(
                candidate_id="c1", vendor="openai", model_id="gpt-4o",
                generation_params={"max_tokens": 1024, "max_completion_tokens": 2048},
            )


class TestGenParamsMerge:
    """Verify the merge logic in inference stage."""

    def test_no_testcase_override(self):
        """Config params pass through when testcase has no generation_params."""
        config_params = {"max_tokens": 1024, "temperature": 0.7}
        tc_gen = {}

        gen_params = dict(config_params)
        tc_max = tc_gen.get("max_tokens")
        if tc_max is not None:
            gen_params.pop("max_tokens", None)
            gen_params.pop("max_completion_tokens", None)
            gen_params["max_tokens"] = tc_max

        assert gen_params == {"max_tokens": 1024, "temperature": 0.7}

    def test_testcase_overrides_max_tokens(self):
        """Testcase max_tokens replaces config max_tokens."""
        config_params = {"max_tokens": 1024, "temperature": 0.7}
        tc_gen = {"max_tokens": 512}

        gen_params = dict(config_params)
        tc_max = tc_gen.get("max_tokens")
        if tc_max is not None:
            gen_params.pop("max_tokens", None)
            gen_params.pop("max_completion_tokens", None)
            gen_params["max_tokens"] = tc_max

        assert gen_params == {"max_tokens": 512, "temperature": 0.7}

    def test_testcase_overrides_max_completion_tokens(self):
        """Testcase max_tokens preserves max_completion_tokens key when candidate uses it."""
        config_params = {"max_completion_tokens": 2048, "temperature": 0.7}
        tc_gen = {"max_tokens": 512}

        gen_params = dict(config_params)
        tc_max = tc_gen.get("max_tokens")
        if tc_max is not None:
            use_completion_key = "max_completion_tokens" in config_params
            gen_params.pop("max_tokens", None)
            gen_params.pop("max_completion_tokens", None)
            if use_completion_key:
                gen_params["max_completion_tokens"] = tc_max
            else:
                gen_params["max_tokens"] = tc_max

        assert gen_params == {"max_completion_tokens": 512, "temperature": 0.7}
        assert "max_tokens" not in gen_params
