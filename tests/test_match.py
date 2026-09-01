from lastro.match import check_citation, match_quote, supports_claim
from lastro.types import Citation, MatchLevel, Source

ARTICLE = (
    "Art. 4º Durante o prazo estipulado para a duração do contrato, não poderá o "
    "locador reaver o imóvel alugado. O locatário, todavia, poderá devolvê-lo, "
    "pagando a multa pactuada, segundo a proporção prevista no art. 413 do Código Civil."
)


def test_verbatim_quote_matches_exactly():
    level, coverage = match_quote("pagando a multa pactuada", ARTICLE)
    assert level is MatchLevel.EXACT
    assert coverage == 1.0


def test_case_and_accent_differences_are_normalized_not_failed():
    level, _ = match_quote("PAGANDO A MULTA PACTUADA", ARTICLE)
    assert level is MatchLevel.NORMALIZED
    assert level.is_verbatim


def test_reworded_quote_drawn_from_source_is_partial_not_none():
    level, coverage = match_quote(
        "o locatario podera devolver o imovel pagando a multa pactuada", ARTICLE
    )
    assert level is MatchLevel.PARTIAL
    assert level.is_supported
    assert not level.is_verbatim
    assert coverage >= 0.75


def test_invented_quote_is_unsupported():
    level, _ = match_quote("o locador deve reformar a cozinha anualmente", ARTICLE)
    assert level is MatchLevel.NONE
    assert not level.is_supported


def test_word_order_matters_so_a_reversed_claim_is_not_credited():
    """The same words in an order that inverts the legal meaning must not pass."""
    forward, _ = match_quote("o locador reaver o imóvel", ARTICLE)
    scrambled, coverage = match_quote("imóvel o reaver locador o", ARTICLE)
    assert forward.is_supported
    assert coverage < 1.0


def test_short_quote_must_match_contiguously():
    """Two common words match most sources by accident."""
    level, _ = match_quote("o locador", ARTICLE)
    assert level is MatchLevel.EXACT
    level, _ = match_quote("xyz abc", ARTICLE)
    assert level is MatchLevel.NONE


def test_empty_quote_is_unsupported():
    assert match_quote("   ", ARTICLE)[0] is MatchLevel.NONE


def test_citation_resolves_article_spelling_to_the_stored_id():
    sources = {"4": Source("4", ARTICLE)}
    check = check_citation(Citation("Art. 4º", "pagando a multa pactuada"), sources)
    assert check.source_found
    assert check.level is MatchLevel.EXACT


def test_citation_to_a_nonexistent_source_is_flagged_separately():
    check = check_citation(Citation("999", "qualquer coisa"), {"4": Source("4", ARTICLE)})
    assert not check.source_found
    assert check.level is MatchLevel.NONE


def test_supports_claim_uses_the_same_grading():
    level, _ = supports_claim("o locatário pode devolver o imóvel pagando multa", ARTICLE)
    assert level.is_supported


# --- Stemming must not buy recall by crediting fabrications ------------------


def test_stemming_credits_inflected_paraphrase():
    """ "poderá devolvê-lo" in the statute vs "pode devolver" in the answer."""
    level, coverage = supports_claim(
        "o locatário pode devolver o imóvel pagando a multa pactuada", ARTICLE
    )
    assert level.is_supported, f"coverage was {coverage}"


def test_stemming_does_not_credit_an_invented_obligation():
    """The dangerous direction: over-stemming would score a fabrication as grounded."""
    for invented in (
        "o locador deve reformar a cozinha todo ano",
        "o locatário tem direito a três meses de aluguel gratuito",
        "o contrato deve ser registrado em cartório sob pena de nulidade",
        "o fiador responde por dívidas trabalhistas do locatário",
    ):
        level, coverage = supports_claim(invented, ARTICLE)
        assert not level.is_supported, f"{invented!r} scored {level} at coverage {coverage}"


def test_stemming_does_not_collapse_distinct_legal_actors():
    """locador and locatário are opposite parties and must not stem together."""
    from lastro import normalize

    assert normalize.stem(normalize.loose("locador")) != normalize.stem(
        normalize.loose("locatário")
    )


# --- Quotes and claims are graded on different scales ------------------------


def test_claim_threshold_is_looser_than_the_quote_threshold():
    """A quote is a copy; a claim is a paraphrase. One threshold cannot serve both."""
    from lastro.match import CLAIM_COVERAGE_THRESHOLD, DEFAULT_COVERAGE_THRESHOLD

    assert CLAIM_COVERAGE_THRESHOLD < DEFAULT_COVERAGE_THRESHOLD


def test_text_between_the_two_thresholds_is_a_claim_but_not_a_quote():
    """The band between the thresholds is the whole reason there are two of them."""
    from lastro.match import CLAIM_COVERAGE_THRESHOLD, DEFAULT_COVERAGE_THRESHOLD

    source = "um dois tres quatro cinco seis sete oito nove dez"
    # Seven of the ten tokens appear in the source, in order: coverage 0.7, which
    # sits between the claim threshold and the quote threshold.
    text = "um dois tres quatro cinco seis sete naoexiste naoexisteb naoexistec"

    _level, coverage = match_quote(text, source)
    assert CLAIM_COVERAGE_THRESHOLD <= coverage < DEFAULT_COVERAGE_THRESHOLD, coverage

    assert supports_claim(text, source)[0].is_supported
    assert not match_quote(text, source)[0].is_supported


def test_the_looser_claim_threshold_still_rejects_fabrication():
    source = "Art. 51. o contrato a renovar tenha sido celebrado por escrito."
    level, coverage = supports_claim(
        "O locatário tem direito a indenização por danos morais em caso de despejo", source
    )
    assert not level.is_supported, f"coverage {coverage}"
