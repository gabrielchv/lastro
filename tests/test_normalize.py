from lastro import normalize


def test_folds_ordinal_spellings_to_one_form():
    assert normalize.loose("Art. 4º") == normalize.loose("art 4o")
    assert normalize.loose("Artigo 4") == normalize.loose("ART. 4°")


def test_strips_accents_so_a_dropped_cedilla_is_not_a_hallucination():
    assert normalize.loose("locação") == normalize.loose("locacao")


def test_rejoins_words_broken_by_a_line_wrap():
    assert normalize.basic("loca-\nção") == "locação"


def test_collapses_nonbreaking_and_exotic_spaces():
    assert normalize.basic("a b c") == "a b c"


def test_folds_typographic_quotes_and_dashes():
    assert normalize.basic("“aspas” — traço") == '"aspas" - traço'


def test_basic_preserves_case_and_accents():
    assert normalize.basic("Locação") == "Locação"


def test_article_number_extracts_digits():
    assert normalize.article_number("Art. 4º") == "4"
    assert normalize.article_number("artigo 23") == "23"


def test_article_number_keeps_a_malformed_reference_visible():
    assert normalize.article_number("sem numero") == "sem numero"


def test_paragraph_sign_and_word_compare_equal():
    assert normalize.loose("§ 1º") == normalize.loose("parágrafo 1")
