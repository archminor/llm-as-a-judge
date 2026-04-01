"""Pydantic models matching all JSON schemas."""

from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Testcase ──────────────────────────────────────────────


class InputMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class TestcaseMetadata(BaseModel):
    difficulty: int | None = Field(None, ge=1, le=5)
    input_length_bucket: str | None = Field(None, pattern=r"^[SML]$")
    tags: list[str] = Field(default_factory=list)


class OutputFormat(BaseModel):
    type: str | None = Field(None, pattern=r"^(free_text|markdown|json)$")
    json_schema_ref: str | None = None


class CitationPolicy(BaseModel):
    required: bool = False
    allowed_sources: list[str] = Field(default_factory=list)


class Constraints(BaseModel):
    required_points: list[str] = Field(default_factory=list)
    forbidden_points: list[str] = Field(default_factory=list)
    output_format: OutputFormat | None = None
    citation_policy: CitationPolicy | None = None


class Testcase(BaseModel):
    testcase_id: str
    task_type: str
    input: dict[str, Any]
    metadata: TestcaseMetadata | None = None
    constraints: Constraints | None = None
    generation_params: dict[str, Any] = Field(default_factory=dict)

    @property
    def has_messages(self) -> bool:
        messages = self.input.get("messages")
        return isinstance(messages, list) and len(messages) > 0

    @property
    def messages(self) -> list[InputMessage] | None:
        if not self.has_messages:
            return None
        return [InputMessage.model_validate(m) for m in self.input["messages"]]

    @model_validator(mode="after")
    def validate_input_mode(self) -> "Testcase":
        has_messages_key = "messages" in self.input
        if not has_messages_key:
            return self
        keys = set(self.input.keys())
        if keys != {"messages"}:
            raise ValueError("input.messages cannot be mixed with other input keys")
        if not isinstance(self.input["messages"], list) or len(self.input["messages"]) == 0:
            raise ValueError("input.messages must contain at least one message")
        for m in self.input["messages"]:
            InputMessage.model_validate(m)
        return self


# ── RunConfig ─────────────────────────────────────────────


class ModelRef(BaseModel):
    candidate_id: str
    vendor: str
    model_id: str
    endpoint: str | None = None
    generation_params: dict[str, Any] = Field(default_factory=dict)
    prompt_version: str | None = None

    @field_validator("generation_params")
    @classmethod
    def validate_generation_params(cls, value: dict[str, Any]) -> dict[str, Any]:
        if "max_tokens" in value and "max_completion_tokens" in value:
            raise ValueError(
                "generation_params cannot specify both max_tokens and max_completion_tokens"
            )
        return value


class JudgeRef(BaseModel):
    judge_id: str
    vendor: str
    model_id: str
    rubric_version: str
    prompt_version: str | None = None
    endpoint: str | None = None
    generation_params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("generation_params")
    @classmethod
    def validate_generation_params(cls, value: dict[str, Any]) -> dict[str, Any]:
        if "max_tokens" in value and "max_completion_tokens" in value:
            raise ValueError(
                "generation_params cannot specify both max_tokens and max_completion_tokens"
            )
        return value


class Blinding(BaseModel):
    enabled: bool = True
    random_seed: int | None = None


class Repeats(BaseModel):
    inference_repeats: int = Field(1, ge=1)
    judge_repeats: int = Field(1, ge=1)


class Aggregation(BaseModel):
    method: str  # mean | majority_vote | worst_case | custom
    weights: dict[str, float] = Field(default_factory=dict)


class Parallelism(BaseModel):
    inference_max_concurrency: int = Field(1, ge=1)
    judge_max_concurrency: int = Field(1, ge=1)
    consistency_max_concurrency: int = Field(1, ge=1)


class Protocol(BaseModel):
    evaluation_mode: str  # pairwise | absolute | hybrid
    blinding: Blinding = Field(default_factory=Blinding)
    repeats: Repeats = Field(default_factory=Repeats)
    metrics: list[str] = Field(default_factory=list)
    aggregation: Aggregation
    parallelism: Parallelism = Field(default_factory=Parallelism)


class DatasetConfig(BaseModel):
    testcases_path: str
    dataset_version: str | None = None
    holdout: bool = False


class RunConfig(BaseModel):
    run_id: str
    created_at: str | None = None
    dataset: DatasetConfig
    candidates: list[ModelRef]
    judges: list[JudgeRef]
    protocol: Protocol

    @model_validator(mode="after")
    def validate_candidate_count_by_mode(self) -> "RunConfig":
        mode = self.protocol.evaluation_mode
        n = len(self.candidates)
        if mode in {"pairwise", "hybrid"} and n < 2:
            raise ValueError(
                f"evaluation_mode={mode} requires at least 2 candidates, got {n}"
            )
        if mode == "absolute" and n < 1:
            raise ValueError(
                "evaluation_mode=absolute requires at least 1 candidate"
            )
        return self


# ── InferenceRecord ───────────────────────────────────────


class ModelInfo(BaseModel):
    vendor: str
    model_id: str
    endpoint: str | None = None


class PromptInfo(BaseModel):
    prompt_version: str | None = None
    prompt_hash: str | None = None


class OutputInfo(BaseModel):
    model_config = {"populate_by_name": True}

    text: str
    reasoning: str | None = None
    format: str | None = None  # free_text | markdown | json
    json_data: Any | None = Field(None, alias="json", serialization_alias="json")


class UsageInfo(BaseModel):
    input_tokens: int = Field(0, ge=0)
    output_tokens: int = Field(0, ge=0)
    reasoning_tokens: int | None = None


class TimingInfo(BaseModel):
    started_at: str | None = None
    ended_at: str | None = None
    latency_ms: float = Field(0, ge=0)


class StatusInfo(BaseModel):
    ok: bool = True
    error_type: str | None = None
    error_message: str | None = None


class InferenceRecord(BaseModel):
    run_id: str
    testcase_id: str
    candidate_id: str
    model: ModelInfo
    output: OutputInfo
    prompt: PromptInfo | None = None
    generation_params: dict[str, Any] | None = None
    input_hash: str | None = None
    usage: UsageInfo | None = None
    timing: TimingInfo | None = None
    status: StatusInfo = Field(default_factory=StatusInfo)


# ── AutoCheckRecord ───────────────────────────────────────


class FormatCompliance(BaseModel):
    passed: bool
    details: str | None = None


class JsonSchemaValidation(BaseModel):
    schema_ref: str | None = None
    passed: bool
    errors: list[str] = Field(default_factory=list)


class Checks(BaseModel):
    format_compliance: FormatCompliance | None = None
    json_schema_validation: JsonSchemaValidation | None = None


class AutoCheckRecord(BaseModel):
    run_id: str
    testcase_id: str
    candidate_id: str
    checks: Checks


# ── Metric score types ───────────────────────────────────


class MetricScore(BaseModel):
    rationale: str = Field(min_length=1)
    score: int = Field(ge=1, le=5)


class PairwiseMetricScore(BaseModel):
    rationale: str = Field(min_length=1)
    score_a: int = Field(ge=1, le=5)
    score_b: int = Field(ge=1, le=5)


# ── JudgementRecord ──────────────────────────────────────


class InferenceRef(BaseModel):
    path: str
    line_index: int = Field(ge=0)


class JudgeTarget(BaseModel):
    candidate_id: str
    inference_ref: InferenceRef


class BlindingInfo(BaseModel):
    enabled: bool
    presented_order: list[str] = Field(default_factory=list)
    random_seed: int | None = None


class JudgeInfo(BaseModel):
    judge_id: str
    vendor: str
    model_id: str
    rubric_version: str
    prompt_version: str | None = None


class Scores(BaseModel):
    per_metric: dict[str, Union[MetricScore, PairwiseMetricScore, int]]
    overall_winner: str | None = None  # pairwise: candidate_id or 'tie'
    overall_score: float | None = None  # absolute


class Evidence(BaseModel):
    source_ref: str | None = None
    span: str | None = None


class JudgementRecord(BaseModel):
    run_id: str
    testcase_id: str
    judge: JudgeInfo
    mode: str  # pairwise | absolute
    targets: list[JudgeTarget]
    scores: Scores
    blinding: BlindingInfo | None = None
    critical_issue: bool = False
    critical_issue_candidates: list[str] = Field(default_factory=list)
    overall_rationale: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)


# ── ConsistencyRecord ─────────────────────────────────────


class ConsistencyScores(BaseModel):
    overall: float | None = None  # 1-5 scale
    rationale: str | None = None


class ConsistencyRecord(BaseModel):
    run_id: str
    testcase_id: str
    candidate_id: str
    judge_id: str
    repeat_count: int
    scores: ConsistencyScores
    status: StatusInfo = Field(default_factory=StatusInfo)


# ── ComparisonReport ─────────────────────────────────────


class NotableFailure(BaseModel):
    testcase_id: str | None = None
    candidate_id: str | None = None
    reason: str | None = None
    refs: list[str] = Field(default_factory=list)


class AggregateBlock(BaseModel):
    win_rate: dict[str, float] = Field(default_factory=dict)
    loss_rate: dict[str, float] = Field(default_factory=dict)
    mean_score: dict[str, dict[str, float]] = Field(default_factory=dict)
    weighted_overall: dict[str, float] = Field(default_factory=dict)
    confidence_intervals: dict[str, Any] = Field(default_factory=dict)
    critical_issue_count: dict[str, int] = Field(default_factory=dict)
    notable_failures: list[NotableFailure] = Field(default_factory=list)
    inference_consistency: dict[str, float] = Field(default_factory=dict)


class JudgeAgreement(BaseModel):
    pairwise_agreement_rate: float | None = None
    notes: str | None = None


class CandidateInfo(BaseModel):
    candidate_id: str
    vendor: str
    model_id: str


class JudgeSummary(BaseModel):
    judge_id: str
    vendor: str
    model_id: str
    rubric_version: str


class ReportSummary(BaseModel):
    total_judgements: int = Field(0, ge=0)
    valid_judgements: int = Field(0, ge=0)
    excluded_judgements: int = Field(0, ge=0)
    repeat_stability: float | None = None


class ReportDataset(BaseModel):
    dataset_version: str | None = None
    testcase_count: int = Field(0, ge=0)


class Results(BaseModel):
    overall: AggregateBlock = Field(default_factory=AggregateBlock)
    by_task: dict[str, AggregateBlock] = Field(default_factory=dict)
    by_bucket: dict[str, AggregateBlock] = Field(default_factory=dict)
    judge_agreement: JudgeAgreement = Field(default_factory=JudgeAgreement)


class ComparisonReport(BaseModel):
    run_id: str
    dataset: ReportDataset
    candidates: list[CandidateInfo]
    judges: list[JudgeSummary]
    protocol: dict[str, Any]
    summary: ReportSummary = Field(default_factory=ReportSummary)
    results: Results
