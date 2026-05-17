"""Tests for the RegressionEvaluator class."""

from __future__ import annotations

from datetime import datetime, timezone

from typing import Literal

from agent.evaluator import RegressionEvaluator
from agent.models import EvalScore, ExperimentResult, PromptVersion, RegressionReport


def build_result(scores: dict[str, float], mean_score: float) -> ExperimentResult:
    """Build an ExperimentResult from a score mapping.

    Args:
        scores: Mapping of dimension to score.
        mean_score: Mean score across dimensions.

    Returns:
        ExperimentResult: Constructed result.
    """

    return ExperimentResult(
        experiment_id="exp-1",
        prompt_version_id="ver-1",
        scores=[
            EvalScore(dimension=dimension, score=value, explanation="ok")
            for dimension, value in scores.items()
        ],
        mean_score=mean_score,
        ran_at=datetime.now(timezone.utc),
    )


def build_report(
    verdict: Literal["pass", "fail", "inconclusive"], score_delta: float
) -> RegressionReport:
    """Build a RegressionReport for summary tests.

    Args:
        verdict: Verdict string.
        score_delta: Score delta value.

    Returns:
        RegressionReport: Constructed report.
    """

    baseline_version = PromptVersion(
        version_id="b1",
        prompt_text="baseline",
        version_tag="v1.2",
        created_at=datetime.now(timezone.utc),
        is_baseline=True,
    )
    candidate_version = PromptVersion(
        version_id="c1",
        prompt_text="candidate",
        version_tag="v1.3",
        created_at=datetime.now(timezone.utc),
        is_baseline=False,
    )
    baseline_result = build_result(
        {"relevance": 0.85, "faithfulness": 0.80, "toxicity": 0.95}, 0.867
    )
    candidate_result = build_result(
        {"relevance": 0.85, "faithfulness": 0.80, "toxicity": 0.95}, 0.867
    )
    return RegressionReport(
        prompt_name="support",
        baseline_version=baseline_version,
        candidate_version=candidate_version,
        baseline_result=baseline_result,
        candidate_result=candidate_result,
        score_delta=score_delta,
        regressed_dimensions=[],
        verdict=verdict,
        summary="",
    )


def test_pass_when_scores_are_equal() -> None:
    """Return pass when mean scores are equal and no regressions exist."""

    evaluator = RegressionEvaluator(threshold=0.05)
    baseline = build_result(
        {"relevance": 0.8, "faithfulness": 0.8, "toxicity": 0.9}, 0.833
    )
    candidate = build_result(
        {"relevance": 0.8, "faithfulness": 0.8, "toxicity": 0.9}, 0.833
    )

    score_delta, regressed, verdict = evaluator.compare(baseline, candidate)

    assert score_delta == 0
    assert regressed == []
    assert verdict == "pass"


def test_fail_when_relevance_drops_significantly() -> None:
    """Return fail when relevance drops beyond the threshold."""

    evaluator = RegressionEvaluator(threshold=0.05)
    baseline = build_result({"relevance": 0.9, "faithfulness": 0.8}, 0.85)
    candidate = build_result({"relevance": 0.8, "faithfulness": 0.8}, 0.8)

    score_delta, regressed, verdict = evaluator.compare(baseline, candidate)

    assert verdict == "fail"
    assert "relevance" in regressed
    assert score_delta == -0.05


def test_inconclusive_when_delta_below_half_threshold() -> None:
    """Return inconclusive when delta is below half the threshold."""

    evaluator = RegressionEvaluator(threshold=0.05)
    baseline = build_result({"relevance": 0.8, "faithfulness": 0.8}, 0.8)
    candidate = build_result({"relevance": 0.82, "faithfulness": 0.8}, 0.82)

    score_delta, regressed, verdict = evaluator.compare(baseline, candidate)

    assert verdict == "inconclusive"
    assert regressed == []
    assert score_delta == 0.02


def test_fail_reports_all_regressed_dimensions() -> None:
    """Return fail and list all regressed dimensions."""

    evaluator = RegressionEvaluator(threshold=0.05)
    baseline = build_result({"relevance": 0.9, "faithfulness": 0.85}, 0.875)
    candidate = build_result({"relevance": 0.8, "faithfulness": 0.75}, 0.775)

    _, regressed, verdict = evaluator.compare(baseline, candidate)

    assert verdict == "fail"
    assert set(regressed) == {"relevance", "faithfulness"}


def test_summary_mentions_version_tags() -> None:
    """Include version tags in the summary output."""

    evaluator = RegressionEvaluator(threshold=0.05)
    report = build_report("pass", 0.01)
    report = report.model_copy(update={"summary": evaluator.build_summary(report)})

    assert "v1.2" in report.summary
    assert "v1.3" in report.summary
