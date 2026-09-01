"""Citation metrics: is a cited quote real, and are the cited sources the right ones.

Two independent failures live here and are reported separately.

*Verification* asks whether the quote exists in the source it is attributed to.
That is the claim `inquilino-ia`'s README makes and the one its own eval never
measured: `run_eval.py` compares cited article numbers against expected ones and
never calls `verify_citations`, so a system could cite the right article with an
invented quote and score a clean pass.

*Precision and recall* ask whether the right sources were cited at all. The
original eval scored `bool(cited & expected)`, so citing four articles when one
was correct counted as success. Precision is what makes shotgun citation cost
something.
"""

from __future__ import annotations

from lastro import normalize
from lastro.match import check_citation
from lastro.types import CitationCheck, MetricResult, Response, Sample, Source, ratio


def check_all(response: Response, sources: dict[str, Source]) -> list[CitationCheck]:
    return [check_citation(citation, sources) for citation in response.citations]


def citation_metrics(
    pairs: list[tuple[Sample, Response]], sources: dict[str, Source]
) -> list[MetricResult]:
    verified = verbatim = total_citations = 0
    missing_sources = 0
    precision_sum = recall_sum = 0.0
    scored_cases = 0
    uncited_answers = 0
    answered_cases = 0

    for sample, response in pairs:
        checks = check_all(response, sources)
        total_citations += len(checks)
        for check in checks:
            if not check.source_found:
                missing_sources += 1
            if check.level.is_supported:
                verified += 1
            if check.level.is_verbatim:
                verbatim += 1

        if response.refused or sample.expects_refusal:
            continue
        answered_cases += 1
        if not response.citations:
            uncited_answers += 1

        cited = {normalize.article_number(c.source_id) for c in response.citations}
        expected = {normalize.article_number(s) for s in sample.expected_source_ids}
        if not expected:
            continue
        hits = len(cited & expected)
        precision_sum += hits / len(cited) if cited else 0.0
        recall_sum += hits / len(expected)
        scored_cases += 1

    precision = ratio(
        precision_sum,
        scored_cases,
        "citation_precision",
        note="Mean fraction of cited sources that were expected. Penalizes citing "
        "everything retrieved, which a set-overlap check rewards.",
    )
    recall = ratio(
        recall_sum,
        scored_cases,
        "citation_recall",
        note="Mean fraction of expected sources actually cited.",
    )

    results = [
        ratio(
            verified,
            total_citations,
            "citation_verification_rate",
            detail={"verbatim": float(verbatim), "citations": float(total_citations)},
            note="Fraction of citations whose quote is found in the source it names. "
            "This is the claim the anti-fabrication design makes.",
        ),
        ratio(
            verbatim,
            total_citations,
            "citation_verbatim_rate",
            note="Subset of the above matching contiguously, before and after "
            "normalization. The gap to verification rate is quotes that were "
            "altered rather than invented.",
        ),
        precision,
        recall,
        _f1(precision, recall),
        ratio(
            missing_sources,
            total_citations,
            "citation_dangling_rate",
            note="Citations naming a source that does not exist. Distinct from a "
            "quote that fails to match a real source.",
        ),
        ratio(
            uncited_answers,
            answered_cases,
            "uncited_answer_rate",
            note="Answers given with no citation at all. A system can score well on "
            "the rates above by rarely citing anything; this is the counterweight.",
        ),
    ]
    return results


def _f1(precision: MetricResult, recall: MetricResult) -> MetricResult:
    if precision.value is None or recall.value is None:
        return MetricResult(name="citation_f1", value=None, note="Requires both P and R.")
    total = precision.value + recall.value
    value = 0.0 if total == 0 else 2 * precision.value * recall.value / total
    return MetricResult(
        name="citation_f1",
        value=value,
        numerator=value,
        denominator=1.0,
        note="Harmonic mean of citation precision and recall.",
    )


__all__ = ["citation_metrics", "check_all"]
