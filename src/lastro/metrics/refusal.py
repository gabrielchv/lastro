"""Refusal metrics, scored in both directions.

A system that refuses everything cannot hallucinate, and would top every other
metric here. Out-of-scope cases catch the system that answers what it cannot
support; in-scope cases catch the system that bought its safety by being useless.
Reporting only the first is what makes an over-cautious model look excellent.
"""

from __future__ import annotations

from lastro.types import MetricResult, Response, Sample, ratio


def refusal_metrics(pairs: list[tuple[Sample, Response]]) -> list[MetricResult]:
    out_of_scope = correct_refusals = 0
    in_scope = over_refusals = 0

    for sample, response in pairs:
        if sample.expects_refusal:
            out_of_scope += 1
            if response.refused:
                correct_refusals += 1
        else:
            in_scope += 1
            if response.refused:
                over_refusals += 1

    return [
        ratio(
            correct_refusals,
            out_of_scope,
            "refusal_correctness",
            note="Out-of-scope questions correctly refused.",
        ),
        ratio(
            over_refusals,
            in_scope,
            "over_refusal_rate",
            note="Answerable questions refused. Lower is better — this is the cost "
            "of the metric above.",
        ),
    ]


__all__ = ["refusal_metrics"]
