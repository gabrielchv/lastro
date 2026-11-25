"""Quote-to-source matching, graded rather than boolean.

`inquilino-ia`'s original check was `normalized(quote) in normalized(article)`,
which answers one question: did the model reproduce a contiguous span. That
collapses three different situations into `False` — the model altered the quote's
punctuation, the model truncated a real passage, the model invented the passage —
and only the last is a hallucination worth failing a build over.

This module reports which of those happened. No model is asked to judge; every
level is decided by string and token comparison against the source text.
"""

from __future__ import annotations

from lastro import normalize
from lastro.types import Citation, CitationCheck, MatchLevel, Source

DEFAULT_COVERAGE_THRESHOLD = 0.75
"""Coverage required to call a *quote* drawn from its source.

Quotes are meant to be verbatim, so a quote that has drifted this far is already
suspect; the threshold is strict on purpose.
"""

CLAIM_COVERAGE_THRESHOLD = 0.55
"""Coverage required to call a *claim* supported by context.

Deliberately lower than the quote threshold, because the two are not the same
task. A quote is a copy and should match nearly exactly; a claim is a paraphrase
by construction -- the statute writes "tenha sido celebrado por escrito" and a
faithful answer writes "deve ter sido celebrado por escrito".

The value is not a guess. It is calibrated against a negative control: claims
produced with no retrieval at all, scored against the context they never saw.
`benchmark/inquilinato/calibrate.py` reports the two distributions and the
separation at this threshold. Reusing the strict quote threshold here put the cut
point on the median of the grounded distribution, where a rounding difference
swings the score.
"""

MIN_QUOTE_TOKENS = 3
"""Below this, coverage is not meaningful: a two-word quote matches most sources
by accident, so a short quote must match contiguously or not at all."""


def _ordered_coverage(quote_tokens: list[str], source_tokens: list[str]) -> float:
    """Fraction of quote tokens appearing in the source *in order*.

    Both sides are expected to be stemmed by the caller.

    A greedy left-to-right scan. Order matters: the same bag of words rearranged
    can reverse a legal meaning ("o locador deve" vs "deve o locatario"), so bag
    overlap would credit a quote that inverts the source. This is a subsequence
    ratio, not a set intersection.
    """
    if not quote_tokens:
        return 0.0
    matched = 0
    cursor = 0
    for token in quote_tokens:
        try:
            cursor = source_tokens.index(token, cursor) + 1
            matched += 1
        except ValueError:
            continue
    return matched / len(quote_tokens)


def match_quote(
    quote: str, source_text: str, coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD
) -> tuple[MatchLevel, float]:
    """Grade one quote against one source text."""
    if not quote.strip():
        return MatchLevel.NONE, 0.0

    basic_quote = normalize.basic(quote)
    basic_source = normalize.basic(source_text)
    if basic_quote and basic_quote in basic_source:
        return MatchLevel.EXACT, 1.0

    loose_quote = normalize.loose(quote)
    loose_source = normalize.loose(source_text)
    if loose_quote and loose_quote in loose_source:
        return MatchLevel.NORMALIZED, 1.0

    quote_tokens = normalize.stems(quote)
    if len(quote_tokens) < MIN_QUOTE_TOKENS:
        return MatchLevel.NONE, 0.0

    # Stems, not surface tokens: Portuguese inflection otherwise splits "poderá"
    # from "pode" and reports a faithful paraphrase as unsupported. The two tiers
    # above never see a stem, so a "verbatim" verdict stays literal.
    coverage = _ordered_coverage(quote_tokens, normalize.stems(source_text))
    if coverage >= coverage_threshold:
        return MatchLevel.PARTIAL, coverage
    return MatchLevel.NONE, coverage


def check_citation(
    citation: Citation,
    sources: dict[str, Source],
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
) -> CitationCheck:
    """Check a citation against the source it names.

    Source ids are compared after `normalize.article_number`, because a model that
    writes "Art. 4º" is referring to the source stored as "4" and a lookup miss
    would be scored as a fabricated citation.
    """
    source = resolve_source(citation.source_id, sources)
    if source is None:
        return CitationCheck(
            citation=citation, level=MatchLevel.NONE, coverage=0.0, source_found=False
        )
    level, coverage = match_quote(citation.quote, source.text, coverage_threshold)
    return CitationCheck(citation=citation, level=level, coverage=coverage, source_found=True)


def resolve_source(source_id: str, sources: dict[str, Source]) -> Source | None:
    """Find the source a citation names, tolerating how the id was spelled.

    Public because every consumer that maps a recorded source id back to corpus
    text must agree on this rule. `faithfulness` builds its context from
    `Response.retrieved_source_ids`, and a bare `sources.get(id)` there silently
    found nothing whenever a system reported "Art. 4º" against a corpus keyed
    "4" — which skipped the answer from faithfulness scoring instead of scoring
    it, so the report showed `n/a` and looked healthy.
    """
    if source_id in sources:
        return sources[source_id]
    canonical = normalize.article_number(source_id)
    if canonical in sources:
        return sources[canonical]
    for key, source in sources.items():
        if normalize.article_number(key) == canonical:
            return source
    return None


def supports_claim(
    claim: str, context: str, coverage_threshold: float = CLAIM_COVERAGE_THRESHOLD
) -> tuple[MatchLevel, float]:
    """Whether retrieved context supports an atomic claim.

    Deliberately the same machinery as quote matching, and deliberately not an
    entailment model. A claim scored `PARTIAL` shares most of its wording with the
    context in order; that is evidence the claim was drawn from the context, not
    proof that the context entails it. `benchmark/` measures how far that proxy
    agrees with human judgement instead of assuming it does — see the README.
    """
    return match_quote(claim, context, coverage_threshold)
