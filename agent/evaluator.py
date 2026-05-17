"""Evaluation logic for prompt regression detection."""

from __future__ import annotations

from typing import Literal

from agent.models import ExperimentResult, RegressionReport


class RegressionEvaluator:
    """Compares experiment results and determines regression verdicts."""

    def __init__(self, threshold: float = 0.05) -> None:
        """Initialize the evaluator.

        Args:
            threshold: Regression threshold for score deltas.
        """

        self._threshold = threshold

    def compare(
        self, baseline: ExperimentResult, candidate: ExperimentResult
    ) -> tuple[float, list[str], Literal["pass", "fail", "inconclusive"]]:
        """Compare baseline and candidate results.

        Args:
            baseline: Baseline experiment result.
            candidate: Candidate experiment result.

        Returns:
            tuple[float, list[str], Literal["pass", "fail", "inconclusive"]]:
                Score delta, regressed dimensions, and verdict.
        """

        baseline_scores = {score.dimension: score.score for score in baseline.scores}
        candidate_scores = {score.dimension: score.score for score in candidate.scores}
        score_delta = round(candidate.mean_score - baseline.mean_score, 10)
        regressed_dimensions = []

        for dimension, candidate_score in candidate_scores.items():
            baseline_score = baseline_scores.get(dimension)
            if baseline_score is None:
                continue
            delta = round(candidate_score - baseline_score, 10)
            if delta < -self._threshold:
                regressed_dimensions.append(dimension)

        if regressed_dimensions:
            verdict: Literal["pass", "fail", "inconclusive"] = "fail"
        elif 0 < abs(score_delta) < self._threshold / 2:
            verdict = "inconclusive"
        else:
            verdict = "pass"

        return score_delta, regressed_dimensions, verdict

    def build_summary(self, report: RegressionReport) -> str:
        """Build a plain-English summary for a regression report.

        Args:
            report: Regression report to summarize.

        Returns:
            str: Summary sentence(s) for Slack messages.
        """

        delta_str = f"{report.score_delta:+.2f}"
        candidate_tag = report.candidate_version.version_tag
        baseline_tag = report.baseline_version.version_tag
        candidate_score = f"{report.candidate_result.mean_score:.2f}"

        if report.verdict == "pass":
            return (
                f"Prompt {candidate_tag} is stable. Mean score held at {candidate_score} "
                f"({delta_str} vs baseline {baseline_tag})."
            )
        if report.verdict == "fail":
            dimensions = ", ".join(report.regressed_dimensions)
            return (
                f"Prompt {candidate_tag} regressed on {dimensions} "
                f"(mean score dropped {delta_str} vs baseline {baseline_tag}). Do not deploy."
            )
        return (
            f"Prompt {candidate_tag} is inconclusive. Mean score is {candidate_score} "
            f"({delta_str} vs baseline {baseline_tag}). Review before deployment."
        )
