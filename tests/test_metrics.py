"""Metric behaviour, especially the cases the original inquilino-ia eval got wrong."""

from lastro.metrics import citation_metrics, faithfulness_metrics, refusal_metrics
from lastro.metrics.retrieval import retrieval_metrics
from lastro.types import Citation, Response, Sample, Source

ART4 = Source("4", "O locatário poderá devolvê-lo, pagando a multa pactuada.")
ART22 = Source("22", "O locador é obrigado a entregar ao locatário o imóvel em estado de servir.")
SOURCES = {"4": ART4, "22": ART22}


def _sample(expected=("4",), sid="s1"):
    return Sample(id=sid, question="q?", expected_source_ids=frozenset(expected))


def _named(results):
    return {r.name: r for r in results}


def test_shotgun_citation_is_punished_by_precision():
    """The original eval scored `bool(cited & expected)`, so citing everything passed."""
    response = Response(
        sample_id="s1",
        answer="a",
        citations=(
            Citation("4", "pagando a multa pactuada"),
            Citation("22", "em estado de servir"),
        ),
    )
    metrics = _named(citation_metrics([(_sample(("4",)), response)], SOURCES))
    assert metrics["citation_recall"].value == 1.0
    assert metrics["citation_precision"].value == 0.5


def test_verification_catches_a_right_article_with_an_invented_quote():
    """Cites the correct article, fabricates the supporting text.

    This is the failure inquilino-ia's own eval could not see: it compared article
    numbers and never checked the quote.
    """
    response = Response(
        sample_id="s1",
        answer="a",
        citations=(Citation("4", "o locador deve pintar o imóvel anualmente"),),
    )
    metrics = _named(citation_metrics([(_sample(("4",)), response)], SOURCES))
    assert metrics["citation_recall"].value == 1.0
    assert metrics["citation_verification_rate"].value == 0.0


def test_dangling_citation_is_reported_apart_from_a_failed_quote():
    response = Response(sample_id="s1", answer="a", citations=(Citation("999", "qualquer"),))
    metrics = _named(citation_metrics([(_sample(("4",)), response)], SOURCES))
    assert metrics["citation_dangling_rate"].value == 1.0


def test_answering_with_no_citation_at_all_is_counted():
    response = Response(sample_id="s1", answer="uma resposta sem fonte", citations=())
    metrics = _named(citation_metrics([(_sample(("4",)), response)], SOURCES))
    assert metrics["uncited_answer_rate"].value == 1.0


def test_a_system_that_refuses_everything_fails_over_refusal():
    pairs = [
        (_sample(("4",), "in1"), Response(sample_id="in1", answer="", refused=True)),
        (Sample(id="out1", question="q?"), Response(sample_id="out1", answer="", refused=True)),
    ]
    metrics = _named(refusal_metrics(pairs))
    assert metrics["refusal_correctness"].value == 1.0
    assert metrics["over_refusal_rate"].value == 1.0, "refusing answerable questions must show"


def test_faithfulness_skips_responses_without_claims_rather_than_scoring_zero():
    response = Response(
        sample_id="s1", answer="uma resposta", retrieved_source_ids=("4",), claims=()
    )
    metrics = _named(faithfulness_metrics([(_sample(), response)], SOURCES))
    assert metrics["claim_support_rate"].value is None
    assert metrics["claim_support_rate"].detail["skipped"] == 1.0


def test_faithfulness_flags_an_unsupported_sentence_beside_a_good_one():
    response = Response(
        sample_id="s1",
        answer="...",
        retrieved_source_ids=("4",),
        claims=(
            "O locatário pode devolver o imóvel pagando a multa pactuada.",
            "O locatário tem direito a seis meses de carência.",
        ),
    )
    metrics = _named(faithfulness_metrics([(_sample(), response)], SOURCES))
    assert metrics["claim_support_rate"].value == 0.5
    assert metrics["fully_grounded_answer_rate"].value == 0.0


def test_retrieval_recall_separates_a_retrieval_miss_from_a_generation_miss():
    hit = (_sample(("4",), "a"), Response(sample_id="a", answer="x", retrieved_source_ids=("4",)))
    miss = (_sample(("4",), "b"), Response(sample_id="b", answer="x", retrieved_source_ids=("22",)))
    metrics = _named(retrieval_metrics([hit, miss]))
    assert metrics["retrieval_recall"].value == 0.5


def test_mrr_penalizes_ranking_the_right_article_below_distractors():
    first = (
        _sample(("4",), "a"),
        Response(sample_id="a", answer="x", retrieved_source_ids=("4", "22")),
    )
    third = (
        _sample(("4",), "b"),
        Response(sample_id="b", answer="x", retrieved_source_ids=("22", "22", "4")),
    )
    metrics = _named(retrieval_metrics([first, third]))
    assert metrics["retrieval_mrr"].value == (1.0 + 1 / 3) / 2


def test_undefined_metric_is_not_a_perfect_score():
    metrics = _named(refusal_metrics([]))
    assert metrics["refusal_correctness"].value is None
    assert not metrics["refusal_correctness"].is_defined


def test_faithfulness_resolves_retrieved_ids_spelled_like_the_model_wrote_them():
    # A system may report retrieval as "Art. 4º" while the corpus is keyed "4".
    # citation_metrics already tolerated that via match.resolve_source; the
    # faithfulness context builder did a bare dict lookup, found nothing, and
    # skipped the response — so the report showed n/a instead of scoring it.
    sample = _sample(("4", "22"))
    response = Response(
        sample_id="s1",
        answer="a",
        retrieved_source_ids=("Art. 4º", "artigo 22"),
        claims=("o locatario podera devolve-lo",),
    )
    named = _named(faithfulness_metrics([(sample, response)], SOURCES))
    assert named["claim_support_rate"].is_defined
    assert named["claim_support_rate"].denominator == 1
    assert named["claim_support_rate"].value == 1.0


def test_faithfulness_still_skips_a_response_whose_context_is_genuinely_missing():
    sample = _sample(("4",))
    response = Response(
        sample_id="s1", answer="a", retrieved_source_ids=("999",), claims=("x y z",)
    )
    named = _named(faithfulness_metrics([(sample, response)], SOURCES))
    assert not named["claim_support_rate"].is_defined
