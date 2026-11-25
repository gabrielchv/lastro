"""Text normalization for Brazilian Portuguese legal and reference text.

A citation check is a string comparison, and in Portuguese legal text the two
strings disagree for reasons that have nothing to do with whether the claim is
grounded. A model writes "art. 4o" where the statute has "Art. 4º"; it drops the
cedilla from "locação"; a scraped statute carries a soft hyphen from the original
line wrap. Treating those as citation failures reports hallucinations that are not
there, so normalization runs before every comparison.

Each transformation here is lossy in a way that could, in principle, merge two
genuinely different passages. They are ordered from safest to most aggressive and
exposed as separate levels so a caller can decide how much collapsing is
acceptable, rather than getting one opaque "normalized" string.
"""

from __future__ import annotations

import re
import unicodedata

# Characters that scraped legal text carries and that no comparison should see:
# soft hyphen, zero-width space/joiners, and the several non-breaking spaces.
_INVISIBLE = dict.fromkeys(
    [0x00AD, 0x200B, 0x200C, 0x200D, 0xFEFF],
    None,
)
_SPACE_LIKE = dict.fromkeys(
    [0x00A0, 0x2007, 0x202F, 0x2009, 0x200A, 0x2002, 0x2003, 0x3000],
    " ",
)

# Typographic punctuation that differs between a model's output and a scraped
# source purely as a rendering choice.
_PUNCT_FOLD = {
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "‟": '"',
    "′": "'",
    "″": '"',
    "‐": "-",
    "‑": "-",
    "‒": "-",
    "–": "-",
    "—": "-",
    "―": "-",
    "−": "-",
    "…": "...",
}

_TRANSLATION = {**_INVISIBLE, **_SPACE_LIKE, **{ord(k): v for k, v in _PUNCT_FOLD.items()}}

# A line wrap inside a word: "loca-\ncao" is one word in the source document.
_WRAP_HYPHEN = re.compile(r"(\w)-\s*\n\s*(\w)")
_WHITESPACE = re.compile(r"\s+")

# Legal ordinal markers. Portuguese statutes write "Art. 4º"; models produce
# "art. 4o", "artigo 4", "Art 4°" interchangeably. The degree sign (U+00B0) is a
# distinct character from the masculine ordinal (U+00BA) and models mix them.
_ORDINAL_MARKS = re.compile(r"(?<=\d)\s*[ºª°˚]|(?<=\d)\s*[oa]\b")

# Section and article abbreviations, expanded to a single canonical token so
# "§ 1º", "par. 1", "paragrafo 1" compare equal.
#
# Every pattern ends on a lookahead rather than a bare `\s*`. Without a right-hand
# boundary, `\bart(?:...)?\s*` also fires inside ordinary words that merely start
# with "art" — "artefato" became "art efato", "artista" became "art ista". That
# inflates the token count, which is what `match.MIN_QUOTE_TOKENS` uses to reject
# quotes too short to score, and it invents a very common "art" token in text
# that never cited an article. The lookahead requires the abbreviation to stand
# on its own, so a following letter no longer matches.
_ABBREVIATIONS = [
    (re.compile(r"\bart(?:s\.|\.|igos?|s)?(?=\W|$)\s*", re.I), "art "),
    (re.compile(r"[§]+(?=\W|$)\s*", re.I), "par "),
    (re.compile(r"\bparagrafos?(?=\W|$)\s*", re.I), "par "),
    (re.compile(r"\bincisos?(?=\W|$)\s*", re.I), "inc "),
    (re.compile(r"\balineas?(?=\W|$)\s*", re.I), "al "),
]


def strip_invisibles(text: str) -> str:
    """Remove zero-width marks and fold exotic spaces and dashes to ASCII."""
    text = _WRAP_HYPHEN.sub(r"\1\2", text)
    return text.translate(_TRANSLATION)


def collapse_whitespace(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def strip_accents(text: str) -> str:
    """Drop combining marks: "locação" -> "locacao".

    Models transcribing a quote frequently lose diacritics, and no Portuguese word
    pair that matters here is distinguished only by an accent in a way that would
    make a citation check meaningful.
    """
    decomposed = unicodedata.normalize("NFD", text)
    return unicodedata.normalize(
        "NFC", "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    )


def canonical_legal_refs(text: str) -> str:
    """Collapse article/paragraph reference spellings to one form.

    "Art. 4º", "artigo 4o" and "art 4" all become "art 4".
    """
    text = _ORDINAL_MARKS.sub("", text)
    for pattern, replacement in _ABBREVIATIONS:
        text = pattern.sub(replacement, text)
    return text


def basic(text: str) -> str:
    """Safest level: invisible characters, spacing and typographic punctuation only.

    Preserves case and accents, so a match at this level means the quote is
    verbatim in every way a reader would notice.
    """
    return collapse_whitespace(strip_invisibles(text))


def loose(text: str) -> str:
    """Case-folded and accent-stripped, with legal references canonicalized.

    A match here but not at `basic` means the quote is faithful in substance and
    differs in transcription.
    """
    return collapse_whitespace(canonical_legal_refs(strip_accents(basic(text)).lower()))


_WORD = re.compile(r"\w+", re.UNICODE)


def tokens(text: str) -> list[str]:
    """Word tokens of the loose form, for coverage-based comparison."""
    return _WORD.findall(loose(text))


def article_number(raw: str) -> str:
    """Extract the bare digits from however a model wrote an article reference.

    "Art. 4º" -> "4". Returns the input unchanged when it holds no digits, so a
    malformed reference stays visible in a report instead of silently becoming "".
    """
    match = re.search(r"\d+", raw)
    return match.group(0) if match else raw.strip()


# --- Stemming, for coverage comparison only ---------------------------------
#
# Portuguese is morphologically rich, and a grounded-generation checker that
# compares surface tokens pays for it: the statute writes "poderá devolvê-lo" and
# a faithful paraphrase writes "pode devolver". Those are the same lemma, and
# scoring them as different words reports a hallucination where none happened —
# measured on the tenancy corpus, exact-token coverage lost roughly a fifth of the
# tokens of claims that were in fact drawn from the source.
#
# The rules below are deliberately shorter than a full RSLP stemmer. Over-stemming
# is the dangerous direction, because it credits an invented claim as supported,
# so suffixes are only stripped when a stem of at least MIN_STEM characters
# remains, and only the endings that carry inflection rather than derivation.
#
# This is used exclusively by the PARTIAL coverage tier. The EXACT and NORMALIZED
# tiers never see a stem, so "quoted verbatim" keeps its literal meaning.

MIN_STEM = 3

# Ordered longest-first: "-aram" must be tried before "-am".
_SUFFIXES = (
    # plural forms that change the stem
    ("oes", "ao"),
    ("aes", "ao"),
    ("aos", "ao"),
    ("ais", "al"),
    ("eis", "el"),
    ("ois", "ol"),
    ("is", "il"),
    # verb inflection
    ("issem", ""),
    ("assem", ""),
    ("essem", ""),
    ("ariam", ""),
    ("eriam", ""),
    ("iriam", ""),
    ("aria", ""),
    ("eria", ""),
    ("iria", ""),
    ("ando", ""),
    ("endo", ""),
    ("indo", ""),
    ("aram", ""),
    ("eram", ""),
    ("iram", ""),
    ("asse", ""),
    ("esse", ""),
    ("isse", ""),
    ("avam", ""),
    ("iam", ""),
    ("ara", ""),
    ("era", ""),
    ("ira", ""),
    ("ava", ""),
    ("iam", ""),
    ("ar", ""),
    ("er", ""),
    ("ir", ""),
    ("am", ""),
    ("em", ""),
    ("ou", ""),
    # adverb
    ("mente", ""),
    # plain plural
    ("ns", "m"),
    ("s", ""),
)

_FINAL_VOWEL = ("a", "e", "o")


def _strip_final_vowel(token: str) -> str:
    if token and token[-1] in _FINAL_VOWEL and len(token) - 1 >= MIN_STEM:
        return token[:-1]
    return token


def stem(token: str) -> str:
    """Conservative inflectional stem of an already-normalized token.

    The suffix pass and the final-vowel pass both run, in that order. Running only
    one of them desynchronizes singular from plural -- "locatario" would strip its
    final vowel to "locatari" while "locatarios" stopped at "locatario" -- which
    reintroduces exactly the mismatch stemming exists to remove.
    """
    if len(token) <= MIN_STEM:
        return token
    for suffix, replacement in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) + len(replacement) >= MIN_STEM:
            return _strip_final_vowel(token[: -len(suffix)] + replacement)
    return _strip_final_vowel(token)


def stems(text: str) -> list[str]:
    """Stemmed word tokens, for coverage-based comparison."""
    return [stem(token) for token in tokens(text)]
