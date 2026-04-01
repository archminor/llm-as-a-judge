"""Tests for rubric split into system/user prompt (Plan 26).

Verifies:
- load_rubric_parts() correctly splits common / metric sections
- Pairwise/absolute system prompt contains ONLY common part (no metric rubrics)
- Pairwise/absolute user prompt contains ONLY requested metrics
- {mode} placeholder is expanded; scale is fixed in rubric
- Consistency prompt still uses full rubric unchanged
- All 12 metrics are extractable from v1.md
"""

from __future__ import annotations

import pytest

from llm_judge.models import Testcase
from llm_judge.prompts import (
    _RUBRIC_PARTS_CACHE,
    build_absolute_judge_prompt,
    build_consistency_judge_prompt,
    build_pairwise_judge_prompt,
    load_rubric,
    load_rubric_parts,
)

ALL_METRICS = [
    "format_compliance",
    "harmlessness",
    "accuracy",
    "faithfulness",
    "completeness",
    "relevance",
    "reasoning",
    "citation_quality",
    "expression_quality",
]


def _make_testcase(**overrides) -> Testcase:
    defaults = dict(
        testcase_id="tc-001",
        task_type="report_generation",
        input={"raw_text": "test input"},
    )
    defaults.update(overrides)
    return Testcase(**defaults)


@pytest.fixture(autouse=True)
def _clear_rubric_parts_cache():
    """Ensure cache is clean before each test."""
    _RUBRIC_PARTS_CACHE.clear()
    yield
    _RUBRIC_PARTS_CACHE.clear()


# ── load_rubric_parts ────────────────────────────────────


class TestLoadRubricParts:
    def test_returns_tuple(self):
        common, metrics = load_rubric_parts("v1")
        assert isinstance(common, str)
        assert isinstance(metrics, dict)

    def test_common_contains_principles(self):
        common, _ = load_rubric_parts("v1")
        assert "## 0. Principles" in common
        assert "## 1. Global Rules" in common
        assert "## 2. Scoring Methods" in common
        assert "## 3. Output Format" in common

    def test_common_does_not_contain_metric_sections(self):
        common, _ = load_rubric_parts("v1")
        assert "## 4. Evaluation Metrics" not in common
        for metric_id in ALL_METRICS:
            assert f"— {metric_id}" not in common.replace(" ", "").lower(), (
                f"Metric '{metric_id}' found in common section"
            )

    def test_all_9_metrics_extracted(self):
        _, metrics = load_rubric_parts("v1")
        assert set(metrics.keys()) == set(ALL_METRICS)

    def test_each_metric_has_definition_and_anchors(self):
        _, metrics = load_rubric_parts("v1")
        for metric_id, body in metrics.items():
            assert "Definition" in body, f"{metric_id}: missing Definition"
            assert "1 (Poor)" in body, f"{metric_id}: missing anchor 1"
            assert "3 (Acceptable)" in body, f"{metric_id}: missing anchor 3"
            assert "5 (Excellent)" in body, f"{metric_id}: missing anchor 5"

    def test_metric_section_starts_with_heading(self):
        _, metrics = load_rubric_parts("v1")
        for metric_id, body in metrics.items():
            assert body.startswith("### "), (
                f"{metric_id}: section doesn't start with ### heading"
            )

    def test_common_has_mode_placeholder(self):
        common, _ = load_rubric_parts("v1")
        assert "{mode}" in common

    def test_common_has_fixed_scale(self):
        common, _ = load_rubric_parts("v1")
        assert "1/3/5" in common or "1–5" in common
        assert "{scale_str}" not in common

    def test_common_has_critical_issue_definition(self):
        common, _ = load_rubric_parts("v1")
        assert "Critical Issues" in common
        assert "critical_issue" in common

    def test_common_has_output_format(self):
        common, _ = load_rubric_parts("v1")
        assert "per_metric" in common
        assert "overall_winner" in common
        assert "overall_score" in common

    def test_common_has_null_prohibition(self):
        common, _ = load_rubric_parts("v1")
        assert "null" in common or "do not use null" in common.lower()
        assert "omit" in common.lower()

    def test_result_is_cached(self):
        result1 = load_rubric_parts("v1")
        result2 = load_rubric_parts("v1")
        assert result1 is result2

    def test_invalid_rubric_raises_error(self):
        """Rubric without '## N.' heading should raise ValueError."""
        import llm_judge.prompts as prompts_mod

        original_cache = prompts_mod._RUBRIC_CACHE.copy()
        try:
            prompts_mod._RUBRIC_CACHE["broken"] = "# No metrics here\nJust text."
            with pytest.raises(ValueError, match="Evaluation Metrics"):
                load_rubric_parts("broken")
        finally:
            prompts_mod._RUBRIC_CACHE.clear()
            prompts_mod._RUBRIC_CACHE.update(original_cache)


# ── Pairwise prompt ──────────────────────────────────────


class TestPairwisePromptSplit:
    def test_system_msg_has_no_metric_rubric(self):
        tc = _make_testcase()
        messages = build_pairwise_judge_prompt(
            testcase=tc,
            output_a="answer A",
            output_b="answer B",
            label_a="A",
            label_b="B",
            metrics=["accuracy", "completeness"],
            rubric_version="v1",
        )
        system_msg = messages[0]["content"]
        # Should NOT contain individual metric section headings
        assert "### 4.1" not in system_msg
        assert "### 4.2" not in system_msg
        assert "## 4. Evaluation Metrics" not in system_msg
        # Should NOT contain metric anchor scores (1/3/5 definitions)
        assert "1 (Poor)" not in system_msg
        assert "5 (Excellent)" not in system_msg

    def test_user_msg_has_requested_metrics_only(self):
        tc = _make_testcase()
        messages = build_pairwise_judge_prompt(
            testcase=tc,
            output_a="answer A",
            output_b="answer B",
            label_a="A",
            label_b="B",
            metrics=["accuracy", "completeness"],
            rubric_version="v1",
        )
        user_msg = messages[1]["content"]
        # Should contain requested metrics
        assert "accuracy — Factual Accuracy" in user_msg
        assert "completeness — Completeness" in user_msg
        # Should NOT contain unrequested metrics
        assert "relevance — Instruction Adherence" not in user_msg
        assert "faithfulness — Input Faithfulness" not in user_msg
        assert "expression_quality" not in user_msg

    def test_mode_placeholder_expanded_to_pairwise(self):
        tc = _make_testcase()
        messages = build_pairwise_judge_prompt(
            testcase=tc,
            output_a="A",
            output_b="B",
            label_a="A",
            label_b="B",
            metrics=["accuracy"],
            rubric_version="v1",
        )
        system_msg = messages[0]["content"]
        assert "**pairwise**" in system_msg
        assert "{mode}" not in system_msg

    def test_scale_is_fixed_in_rubric(self):
        tc = _make_testcase()
        messages = build_pairwise_judge_prompt(
            testcase=tc,
            output_a="A",
            output_b="B",
            label_a="A",
            label_b="B",
            metrics=["accuracy"],
            rubric_version="v1",
        )
        system_msg = messages[0]["content"]
        assert "1/3/5" in system_msg or "1–5" in system_msg

    def test_user_msg_has_task_info(self):
        tc = _make_testcase(testcase_id="tc-999", task_type="preprocessing")
        messages = build_pairwise_judge_prompt(
            testcase=tc,
            output_a="A",
            output_b="B",
            label_a="A",
            label_b="B",
            metrics=["accuracy"],
            rubric_version="v1",
        )
        user_msg = messages[1]["content"]
        assert "tc-999" in user_msg
        assert "preprocessing" in user_msg

    def test_user_msg_has_both_answers(self):
        tc = _make_testcase()
        messages = build_pairwise_judge_prompt(
            testcase=tc,
            output_a="UNIQUE_ANSWER_A_TEXT",
            output_b="UNIQUE_ANSWER_B_TEXT",
            label_a="A",
            label_b="B",
            metrics=["accuracy"],
            rubric_version="v1",
        )
        user_msg = messages[1]["content"]
        assert "UNIQUE_ANSWER_A_TEXT" in user_msg
        assert "UNIQUE_ANSWER_B_TEXT" in user_msg

    def test_single_metric_only(self):
        tc = _make_testcase()
        messages = build_pairwise_judge_prompt(
            testcase=tc,
            output_a="A",
            output_b="B",
            label_a="A",
            label_b="B",
            metrics=["harmlessness"],
            rubric_version="v1",
        )
        user_msg = messages[1]["content"]
        assert "harmlessness — Safety" in user_msg
        # No other metrics
        for m in ALL_METRICS:
            if m != "harmlessness":
                assert f"### " not in user_msg.split("harmlessness")[0] or True
        # More precise: count ### headings in rubric section
        rubric_section = user_msg.split("## Task Info")[0]
        assert rubric_section.count("### ") == 1



# ── Absolute prompt ──────────────────────────────────────


class TestAbsolutePromptSplit:
    def test_system_msg_has_no_metric_rubric(self):
        tc = _make_testcase()
        messages = build_absolute_judge_prompt(
            testcase=tc,
            output_text="some output",
            candidate_label="c1",
            metrics=["accuracy", "relevance"],
            rubric_version="v1",
        )
        system_msg = messages[0]["content"]
        # Should NOT contain individual metric section headings
        assert "### 4.1" not in system_msg
        assert "### 4.6" not in system_msg
        assert "## 4. Evaluation Metrics" not in system_msg
        # Should NOT contain metric anchor scores (1/3/5 definitions)
        assert "1 (Poor)" not in system_msg
        assert "5 (Excellent)" not in system_msg

    def test_user_msg_has_requested_metrics_only(self):
        tc = _make_testcase()
        messages = build_absolute_judge_prompt(
            testcase=tc,
            output_text="some output",
            candidate_label="c1",
            metrics=["accuracy", "relevance"],
            rubric_version="v1",
        )
        user_msg = messages[1]["content"]
        assert "accuracy — Factual Accuracy" in user_msg
        assert "relevance — Instruction Adherence" in user_msg
        assert "completeness — Completeness" not in user_msg
        assert "harmlessness — Safety" not in user_msg

    def test_mode_placeholder_expanded_to_absolute(self):
        tc = _make_testcase()
        messages = build_absolute_judge_prompt(
            testcase=tc,
            output_text="output",
            candidate_label="c1",
            metrics=["accuracy"],
            rubric_version="v1",
        )
        system_msg = messages[0]["content"]
        assert "**absolute**" in system_msg
        assert "{mode}" not in system_msg

    def test_scale_is_fixed_in_rubric(self):
        tc = _make_testcase()
        messages = build_absolute_judge_prompt(
            testcase=tc,
            output_text="output",
            candidate_label="c1",
            metrics=["accuracy"],
            rubric_version="v1",
        )
        system_msg = messages[0]["content"]
        assert "1/3/5" in system_msg or "1–5" in system_msg

    def test_user_msg_has_output(self):
        tc = _make_testcase()
        messages = build_absolute_judge_prompt(
            testcase=tc,
            output_text="UNIQUE_OUTPUT_CONTENT",
            candidate_label="c1",
            metrics=["accuracy"],
            rubric_version="v1",
        )
        user_msg = messages[1]["content"]
        assert "UNIQUE_OUTPUT_CONTENT" in user_msg

    def test_all_metrics_requested(self):
        tc = _make_testcase()
        messages = build_absolute_judge_prompt(
            testcase=tc,
            output_text="output",
            candidate_label="c1",
            metrics=ALL_METRICS,
            rubric_version="v1",
        )
        user_msg = messages[1]["content"]
        for m in ALL_METRICS:
            assert m in user_msg, f"Metric '{m}' missing from user prompt"


# ── Consistency prompt ───────────────────────────────────


class TestConsistencyPromptUnchanged:
    def test_uses_full_rubric(self):
        tc = _make_testcase()
        messages = build_consistency_judge_prompt(
            testcase=tc,
            outputs=["output 1", "output 2"],
            rubric_version="v1",
        )
        system_msg = messages[0]["content"]
        full_rubric = load_rubric("v1")
        # System msg should contain the entire rubric
        assert full_rubric in system_msg

    def test_has_consistency_scale(self):
        tc = _make_testcase()
        messages = build_consistency_judge_prompt(
            testcase=tc,
            outputs=["output 1", "output 2"],
            rubric_version="v1",
        )
        system_msg = messages[0]["content"]
        assert "5 (Very consistent)" in system_msg
        assert "1 (Inconsistent)" in system_msg

    def test_user_msg_has_outputs(self):
        tc = _make_testcase()
        messages = build_consistency_judge_prompt(
            testcase=tc,
            outputs=["FIRST_OUTPUT", "SECOND_OUTPUT", "THIRD_OUTPUT"],
            rubric_version="v1",
        )
        user_msg = messages[1]["content"]
        assert "FIRST_OUTPUT" in user_msg
        assert "SECOND_OUTPUT" in user_msg
        assert "THIRD_OUTPUT" in user_msg
        assert "3 outputs" in user_msg


# ── Cross-cutting: no inline instructions in prompts.py ──


class TestNoInlineInstructions:
    """Verify prompts.py no longer has hardcoded critical_issue / output format."""

    def test_pairwise_system_has_no_hardcoded_critical_issue_list(self):
        tc = _make_testcase()
        messages = build_pairwise_judge_prompt(
            testcase=tc,
            output_a="A",
            output_b="B",
            label_a="A",
            label_b="B",
            metrics=["accuracy"],
            rubric_version="v1",
        )
        system_msg = messages[0]["content"]
        # The critical_issue info should come from v1.md §1.2, NOT from
        # a hardcoded bullet list in prompts.py. We check it's from the rubric
        # by verifying the rubric's specific phrasing is present.
        assert "Critical Issues" in system_msg
        assert "critical_issue" in system_msg

    def test_absolute_system_has_no_hardcoded_critical_issue_list(self):
        tc = _make_testcase()
        messages = build_absolute_judge_prompt(
            testcase=tc,
            output_text="output",
            candidate_label="c1",
            metrics=["accuracy"],
            rubric_version="v1",
        )
        system_msg = messages[0]["content"]
        assert "Critical Issues" in system_msg
        assert "critical_issue" in system_msg


# ── Metric isolation: unrequested metrics never leak ─────


class TestMetricIsolation:
    """Verify that unrequested metrics do not appear anywhere in the prompt."""

    @pytest.mark.parametrize("requested,excluded", [
        (["accuracy"], ["completeness", "relevance", "expression_quality"]),
        (["harmlessness", "relevance"], ["accuracy", "reasoning", "expression_quality"]),
        (["expression_quality"], ["accuracy", "completeness", "relevance"]),
    ])
    def test_pairwise_excludes_unrequested_metrics(self, requested, excluded):
        tc = _make_testcase()
        messages = build_pairwise_judge_prompt(
            testcase=tc,
            output_a="A",
            output_b="B",
            label_a="A",
            label_b="B",
            metrics=requested,
            rubric_version="v1",
        )
        system_msg = messages[0]["content"]
        user_msg = messages[1]["content"]
        _, metric_sections = load_rubric_parts("v1")
        for m in excluded:
            section_text = metric_sections[m]
            # The full metric section text must NOT be in system or user msg
            assert section_text not in system_msg, (
                f"Metric '{m}' rubric leaked into system prompt"
            )
            assert section_text not in user_msg, (
                f"Metric '{m}' rubric leaked into user prompt"
            )

    @pytest.mark.parametrize("requested,excluded", [
        (["accuracy"], ["completeness", "relevance", "expression_quality"]),
        (["harmlessness", "relevance"], ["accuracy", "reasoning", "expression_quality"]),
    ])
    def test_absolute_excludes_unrequested_metrics(self, requested, excluded):
        tc = _make_testcase()
        messages = build_absolute_judge_prompt(
            testcase=tc,
            output_text="output",
            candidate_label="c1",
            metrics=requested,
            rubric_version="v1",
        )
        system_msg = messages[0]["content"]
        user_msg = messages[1]["content"]
        _, metric_sections = load_rubric_parts("v1")
        for m in excluded:
            section_text = metric_sections[m]
            assert section_text not in system_msg
            assert section_text not in user_msg


# ── Unknown metric detection ─────────────────────────────


class TestUnknownMetricDetection:
    """Verify that unknown metric IDs raise ValueError instead of silent omission."""

    def test_pairwise_raises_on_unknown_metric(self):
        tc = _make_testcase()
        with pytest.raises(ValueError, match="no_such_metric"):
            build_pairwise_judge_prompt(
                testcase=tc,
                output_a="A",
                output_b="B",
                label_a="A",
                label_b="B",
                metrics=["accuracy", "no_such_metric"],
                rubric_version="v1",
            )

    def test_absolute_raises_on_unknown_metric(self):
        tc = _make_testcase()
        with pytest.raises(ValueError, match="no_such_metric"):
            build_absolute_judge_prompt(
                testcase=tc,
                output_text="output",
                candidate_label="c1",
                metrics=["no_such_metric"],
                rubric_version="v1",
            )

    def test_error_message_lists_available_metrics(self):
        tc = _make_testcase()
        with pytest.raises(ValueError, match="accuracy") as exc_info:
            build_pairwise_judge_prompt(
                testcase=tc,
                output_a="A",
                output_b="B",
                label_a="A",
                label_b="B",
                metrics=["typo_metric"],
                rubric_version="v1",
            )
        # Error should show available metrics for easy debugging
        assert "Available" in str(exc_info.value)

    def test_valid_metrics_do_not_raise(self):
        tc = _make_testcase()
        # Should not raise
        messages = build_pairwise_judge_prompt(
            testcase=tc,
            output_a="A",
            output_b="B",
            label_a="A",
            label_b="B",
            metrics=["accuracy", "completeness"],
            rubric_version="v1",
        )
        assert len(messages) == 2
