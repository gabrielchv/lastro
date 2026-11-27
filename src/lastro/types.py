"""Core data types.

The vocabulary is deliberately small: a `Sample` is one question with its ground
truth, a `Response` is what a system produced for it, and everything a metric
returns is a `MetricResult` carrying both the number and the counts it came from.
Reporting a bare score loses the ability to tell "0.5 from two cases" apart from
"0.5 from two hundred", which is the difference between noise and a signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class MatchLevel(StrEnum):
    """How closely a quote matched its claimed source.

    Ordered from strongest to weakest. Kept as distinct levels rather than a
    boolean because "the model transcribed the quote with different accents" and
    "the model invented the quote" are both `verified=False` under a strict
    comparison, and only one of them is a hallucination.
    """

    EXACT = "exact"
    """Byte-identical after invisible-character cleanup."""

    NORMALIZED = "normalized"
    """Identical once case, accents and legal-reference spelling are folded."""

    PARTIAL = "partial"
    """Token coverage above threshold, but not a contiguous match. The quote is
    drawn from the source and altered — truncated, reordered, or paraphrased."""

    NONE = "none"
    """Not supported by the cited source."""

    @property
    def is_supported(self) -> bool:
        return self is not MatchLevel.NONE

    @property
    def is_verbatim(self) -> bool:
        return self in (MatchLevel.EXACT, MatchLevel.NORMALIZED)


@dataclass(frozen=True)
class Citation:
    """A claim by the system that `quote` appears in source `source_id`."""

    source_id: str
    quote: str


@dataclass(frozen=True)
class Source:
    """A retrievable unit of ground-truth text — one statute article, one document
    chunk. `id` is what a citation refers to."""

    id: str
    text: str


@dataclass(frozen=True)
class Sample:
    """One benchmark case.

    `expected_source_ids` empty means the correct behaviour is refusal: the case
    exists to catch a system that answers questions its corpus cannot support.
    """

    id: str
    question: str
    expected_source_ids: frozenset[str] = frozenset()
    note: str = ""

    @property
    def expects_refusal(self) -> bool:
        return not self.expected_source_ids


@dataclass(frozen=True)
class Response:
    """What a system under test produced for one `Sample`."""

    sample_id: str
    answer: str
    citations: tuple[Citation, ...] = ()
    retrieved_source_ids: tuple[str, ...] = ()
    refused: bool = False
    claims: tuple[str, ...] = ()
    """Atomic claims the answer makes. Optional: supplied by a decomposer, cached
    on disk so scoring stays free and deterministic. Empty means faithfulness is
    not scored for this response rather than scored as zero."""

    recorded: bool = True
    """False for the placeholder built when a sample has no response at all.

    The distinction matters for retrieval: a *recorded* response with an empty
    `retrieved_source_ids` means the run did not report its retrieval, and
    scoring that as a miss would understate a system that simply stayed quiet
    about it. A sample with no response is different — nobody asked the question
    — and must count against recall rather than vanish from the denominator.
    """


@dataclass(frozen=True)
class CitationCheck:
    """Result of checking one citation against the source it names."""

    citation: Citation
    level: MatchLevel
    coverage: float
    """Fraction of the quote's tokens found in the source, in order. 1.0 for a
    contiguous match."""
    source_found: bool
    """False when the cited source id does not exist at all — a different failure
    from a quote that does not match, and reported separately."""


@dataclass
class MetricResult:
    """A score plus the evidence behind it."""

    name: str
    value: float | None
    numerator: float = 0.0
    denominator: float = 0.0
    detail: dict[str, float] = field(default_factory=dict)
    note: str = ""

    @property
    def is_defined(self) -> bool:
        """False when no case exercised this metric. Distinct from a score of 0."""
        return self.value is not None

    def format_value(self) -> str:
        if self.value is None:
            return "n/a"
        return f"{self.value:.1%}"


def ratio(numerator: float, denominator: float, name: str, **kwargs) -> MetricResult:
    """Build a MetricResult, leaving the value undefined on an empty denominator.

    Scoring an unexercised metric as 0.0 makes a benchmark look worse the fewer
    cases it runs, which is backwards.
    """
    value = numerator / denominator if denominator else None
    return MetricResult(
        name=name, value=value, numerator=numerator, denominator=denominator, **kwargs
    )
