"""Faithfulness: is every claim in the answer supported by what was retrieved.

Citation metrics only see the passages a system chose to cite. An answer can cite
one article perfectly and, in the surrounding prose, assert three things the
statute never says. Faithfulness scores the answer text instead of the citation
list.

Scoring needs the answer split into atomic claims, which is the one step no string
comparison performs well. `lastro.decompose` handles it and caches the result on
disk keyed by the answer text, so the split is paid for once and every rerun is
free and identical. The *support check* stays deterministic: a claim is supported
when its wording is recoverable, in order, from the retrieved context.

A response with no claims recorded is skipped, not scored zero — an unmeasured
answer must not read as an unfaithful one.
"""

from __future__ import annotations

from lastro.match import resolve_source, supports_claim
from lastro.types import MetricResult, Response, Sample, Source, ratio


def _context_for(response: Response, sources: dict[str, Source]) -> str:
    """All retrieved text as one block.

    Claims are checked against everything retrieved rather than against the single
    cited source, because a correct answer routinely combines two articles and
    checking per-source would score the combination as unsupported.

    Ids go through `resolve_source` for the same reason citations do: a system is
    free to report "Art. 4º" where the corpus is keyed "4", and a plain dict
    lookup would return nothing and skip the response rather than score it.
    """
    texts = []
    for source_id in response.retrieved_source_ids:
        source = resolve_source(source_id, sources)
        if source is not None:
            texts.append(source.text)
    return "\n".join(texts)


def faithfulness_metrics(
    pairs: list[tuple[Sample, Response]], sources: dict[str, Source]
) -> list[MetricResult]:
    supported = 0
    total_claims = 0
    scored_responses = 0
    skipped = 0
    fully_supported = 0

    for _sample, response in pairs:
        if response.refused:
            continue
        if not response.claims:
            skipped += 1
            continue
        context = _context_for(response, sources)
        if not context:
            skipped += 1
            continue

        scored_responses += 1
        response_supported = 0
        for claim in response.claims:
            level, _coverage = supports_claim(claim, context)
            total_claims += 1
            if level.is_supported:
                supported += 1
                response_supported += 1
        if response_supported == len(response.claims):
            fully_supported += 1

    return [
        ratio(
            supported,
            total_claims,
            "claim_support_rate",
            detail={"responses_scored": float(scored_responses), "skipped": float(skipped)},
            note="Fraction of atomic claims recoverable from retrieved context. "
            "Responses without cached claims are skipped, not scored zero.",
        ),
        ratio(
            fully_supported,
            scored_responses,
            "fully_grounded_answer_rate",
            note="Answers where every claim is supported. A per-claim average hides "
            "that one unsupported sentence is enough to mislead a reader.",
        ),
    ]


__all__ = ["faithfulness_metrics"]
