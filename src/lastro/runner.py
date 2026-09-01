"""Runs every metric over a paired benchmark and collects the results."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from lastro.dataset import load_responses, load_samples, load_sources, pair
from lastro.metrics import (
    citation_metrics,
    faithfulness_metrics,
    refusal_metrics,
    retrieval_metrics,
)
from lastro.types import MetricResult, Response, Sample, Source


@dataclass
class Evaluation:
    metrics: list[MetricResult]
    sample_count: int
    missing_sample_ids: list[str] = field(default_factory=list)
    suite: str = ""

    def by_name(self) -> dict[str, MetricResult]:
        return {metric.name: metric for metric in self.metrics}

    def to_dict(self) -> dict:
        return {
            "suite": self.suite,
            "samples_scored": self.sample_count,
            "samples_missing": self.missing_sample_ids,
            "metrics": {
                metric.name: {
                    "value": metric.value,
                    "numerator": metric.numerator,
                    "denominator": metric.denominator,
                    "detail": metric.detail,
                    "note": metric.note,
                }
                for metric in self.metrics
            },
        }


def evaluate(
    samples: list[Sample],
    responses: dict[str, Response],
    sources: dict[str, Source],
    suite: str = "",
) -> Evaluation:
    pairs, missing = pair(samples, responses)
    metrics = [
        *retrieval_metrics(pairs),
        *citation_metrics(pairs, sources),
        *faithfulness_metrics(pairs, sources),
        *refusal_metrics(pairs),
    ]
    return Evaluation(
        metrics=metrics, sample_count=len(pairs), missing_sample_ids=missing, suite=suite
    )


def evaluate_suite(suite_dir: Path, responses_path: Path | None = None) -> Evaluation:
    """Score a suite directory holding samples.yaml, sources.yaml and a responses file.

    `responses_path` selects which recorded run to score; it defaults to
    `responses.yaml` inside the suite. Naming the file directly means CI can score
    `runs/grounded.yaml` without copying it into place first.
    """
    responses_path = responses_path or suite_dir / "responses.yaml"
    return evaluate(
        samples=load_samples(suite_dir / "samples.yaml"),
        responses=load_responses(responses_path),
        sources=load_sources(suite_dir / "sources.yaml"),
        suite=responses_path.stem if responses_path.stem != "responses" else suite_dir.name,
    )


__all__ = ["Evaluation", "evaluate", "evaluate_suite"]
