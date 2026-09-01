"""Retrieval metrics, so a generation failure is not blamed on the generator.

When an answer is ungrounded there are two possible causes: retrieval never
surfaced the right passage, or it did and the model ignored it. Those need
opposite fixes — chunking and embeddings versus prompting and verification — and a
single end-to-end pass rate cannot tell them apart.
"""

from __future__ import annotations

from lastro import normalize
from lastro.types import MetricResult, Response, Sample, ratio


def retrieval_metrics(pairs: list[tuple[Sample, Response]]) -> list[MetricResult]:
    recall_sum = 0.0
    reciprocal_rank_sum = 0.0
    scored = 0
    complete = 0

    for sample, response in pairs:
        if sample.expects_refusal:
            continue
        expected = {normalize.article_number(s) for s in sample.expected_source_ids}
        if not expected:
            continue
        retrieved = [normalize.article_number(s) for s in response.retrieved_source_ids]
        if not response.retrieved_source_ids:
            # Retrieval was not recorded for this response; scoring it as a miss
            # would understate a system that simply did not report its retrieval.
            continue

        scored += 1
        hits = expected & set(retrieved)
        recall_sum += len(hits) / len(expected)
        if hits == expected:
            complete += 1
        for rank, source_id in enumerate(retrieved, start=1):
            if source_id in expected:
                reciprocal_rank_sum += 1 / rank
                break

    return [
        ratio(
            recall_sum,
            scored,
            "retrieval_recall",
            note="Mean fraction of expected sources present anywhere in the "
            "retrieved set. Caps what generation can possibly ground.",
        ),
        ratio(
            complete,
            scored,
            "retrieval_full_recall_rate",
            note="Cases where every expected source was retrieved.",
        ),
        ratio(
            reciprocal_rank_sum,
            scored,
            "retrieval_mrr",
            note="Mean reciprocal rank of the first expected source. Low values with "
            "high recall mean the right passage is retrieved but ranked below "
            "distractors the model reads first.",
        ),
    ]


__all__ = ["retrieval_metrics"]
