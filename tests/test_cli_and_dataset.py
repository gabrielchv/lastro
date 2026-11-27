"""End-to-end: a suite directory scores, renders and gates a build."""

import textwrap

import pytest

from lastro.cli import main
from lastro.dataset import load_responses, load_samples, load_sources, pair
from lastro.report import to_json, to_markdown, to_text
from lastro.runner import evaluate_suite

SAMPLES = """
- id: ok
  question: "Posso sair antes do prazo?"
  expected_source_ids: ["4"]
- id: fora
  question: "Qual a capital do Brasil?"
  expected_source_ids: []
"""

SOURCES = """
- id: "4"
  text: "O locatário poderá devolvê-lo, pagando a multa pactuada."
"""

RESPONSES = """
- sample_id: ok
  answer: "Pode, pagando a multa pactuada."
  retrieved_source_ids: ["4"]
  citations:
    - source_id: "Art. 4º"
      quote: "pagando a multa pactuada"
  claims:
    - "O locatário pode devolver o imóvel pagando a multa pactuada."
- sample_id: fora
  answer: ""
  refused: true
"""


@pytest.fixture
def suite(tmp_path):
    (tmp_path / "samples.yaml").write_text(textwrap.dedent(SAMPLES), encoding="utf-8")
    (tmp_path / "sources.yaml").write_text(textwrap.dedent(SOURCES), encoding="utf-8")
    (tmp_path / "responses.yaml").write_text(textwrap.dedent(RESPONSES), encoding="utf-8")
    return tmp_path


def test_a_clean_run_scores_perfectly(suite):
    evaluation = evaluate_suite(suite)
    metrics = evaluation.by_name()
    assert evaluation.sample_count == 2
    assert metrics["citation_verification_rate"].value == 1.0
    assert metrics["refusal_correctness"].value == 1.0
    assert metrics["over_refusal_rate"].value == 0.0
    assert metrics["claim_support_rate"].value == 1.0


def test_missing_response_is_reported_not_silently_dropped(suite):
    (suite / "responses.yaml").write_text(
        textwrap.dedent(RESPONSES).replace("- sample_id: fora", "- sample_id: outro"),
        encoding="utf-8",
    )
    evaluation = evaluate_suite(suite)
    assert evaluation.missing_sample_ids == ["fora"]


def test_duplicate_sample_ids_are_rejected(tmp_path):
    (tmp_path / "samples.yaml").write_text(
        "- id: a\n  question: q\n- id: a\n  question: q2\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate sample id"):
        load_samples(tmp_path / "samples.yaml")


def test_every_renderer_produces_output(suite):
    evaluation = evaluate_suite(suite)
    assert "citation_verification_rate" in to_text(evaluation)
    assert '"citation_verification_rate"' in to_json(evaluation)
    assert "| `citation_verification_rate` |" in to_markdown(evaluation)


def test_cli_exits_zero_when_thresholds_hold(suite, capsys):
    assert main(["run", str(suite), "--fail-under", "citation_verification_rate=0.9"]) == 0


def test_cli_exits_nonzero_when_a_threshold_is_breached(suite, capsys):
    assert main(["run", str(suite), "--fail-under", "citation_precision=1.1"]) == 1
    assert "threshold failures" in capsys.readouterr().err


def test_cli_treats_an_undefined_metric_as_a_breach(suite, capsys):
    """A suite that stopped measuring something must not pass as if it measures it."""
    (suite / "samples.yaml").write_text(
        "- id: ok\n  question: q\n  expected_source_ids: ['4']\n", encoding="utf-8"
    )
    assert main(["run", str(suite), "--fail-under", "refusal_correctness=0.5"]) == 1
    assert "undefined" in capsys.readouterr().err


def test_cli_writes_a_report_file(suite, tmp_path):
    out = tmp_path / "report.md"
    assert main(["run", str(suite), "--format", "markdown", "--out", str(out)]) == 0
    assert "## Citations" in out.read_text(encoding="utf-8")


def test_pair_joins_on_sample_id(suite):
    samples = load_samples(suite / "samples.yaml")
    responses = load_responses(suite / "responses.yaml")
    pairs, missing = pair(samples, responses)
    assert [s.id for s, _ in pairs] == ["ok", "fora"]
    assert missing == []


def test_sources_load_by_id(suite):
    assert set(load_sources(suite / "sources.yaml")) == {"4"}


# --- thresholds respect metric direction -----------------------------------
#
# `--fail-under citation_dangling_rate=0.1` reads as "fail if it exceeds 0.1",
# not "fail if it falls below". Comparing every metric with `<` inverted the
# gate for the lower-is-better metrics, so a run with *more* dangling citations
# than allowed passed and a clean run failed.

DANGLING_RESPONSES = """
- sample_id: ok
  answer: "Pode, pagando a multa pactuada."
  retrieved_source_ids: ["4"]
  citations:
    - source_id: "999"
      quote: "pagando a multa pactuada"
  claims:
    - "O locatário pode devolver o imóvel pagando a multa pactuada."
- sample_id: fora
  answer: ""
  refused: true
"""


@pytest.fixture
def dangling_suite(suite):
    (suite / "responses.yaml").write_text(textwrap.dedent(DANGLING_RESPONSES), encoding="utf-8")
    return suite


def test_threshold_on_a_lower_is_better_metric_fails_when_exceeded(dangling_suite, capsys):
    """One of one citation is dangling (rate 1.0); a ceiling of 0.1 must fail."""
    assert main(["run", str(dangling_suite), "--fail-under", "citation_dangling_rate=0.1"]) == 1
    assert "citation_dangling_rate" in capsys.readouterr().err


def test_threshold_on_a_lower_is_better_metric_passes_when_within(dangling_suite, capsys):
    """The same run against a ceiling it satisfies must exit zero."""
    assert main(["run", str(dangling_suite), "--fail-under", "citation_dangling_rate=1.0"]) == 0


def test_a_clean_run_passes_a_tight_lower_is_better_ceiling(suite, capsys):
    """No dangling citations at all clears any ceiling."""
    assert main(["run", str(suite), "--fail-under", "citation_dangling_rate=0.0"]) == 0
