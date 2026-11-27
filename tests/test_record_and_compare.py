"""Recording round-trips, and the comparison view."""

import json
import textwrap
from pathlib import Path

import pytest

from lastro.cli import main
from lastro.dataset import load_responses, load_samples, load_sources
from lastro.decompose import ClaimCache, decompose
from lastro.record import record, write_responses
from lastro.report import compare_to_markdown, compare_to_text
from lastro.runner import evaluate
from lastro.types import Citation, Response, Sample

SAMPLES = """
- id: a
  question: "Posso sair antes do prazo?"
  expected_source_ids: ["4"]
- id: b
  question: "Capital do Brasil?"
  expected_source_ids: []
"""
SOURCES = """
- id: "4"
  text: "O locatário poderá devolvê-lo, pagando a multa pactuada."
"""


class FakeSystem:
    name = "fake"

    def answer(self, sample: Sample) -> Response:
        if sample.expects_refusal:
            return Response(sample_id=sample.id, answer="", refused=True)
        return Response(
            sample_id=sample.id,
            answer="Pode, pagando a multa pactuada.",
            citations=(Citation("4", "pagando a multa pactuada"),),
            retrieved_source_ids=("4",),
            claims=("O locatário pode devolver pagando a multa pactuada.",),
        )


@pytest.fixture
def suite(tmp_path):
    (tmp_path / "samples.yaml").write_text(textwrap.dedent(SAMPLES), encoding="utf-8")
    (tmp_path / "sources.yaml").write_text(textwrap.dedent(SOURCES), encoding="utf-8")
    return tmp_path


def test_recording_round_trips_through_yaml(suite):
    samples = load_samples(suite / "samples.yaml")
    destination = suite / "runs" / "fake.yaml"
    record(FakeSystem(), samples, destination)

    reloaded = load_responses(destination)
    assert set(reloaded) == {"a", "b"}
    assert reloaded["b"].refused
    assert reloaded["a"].citations[0].quote == "pagando a multa pactuada"
    assert reloaded["a"].retrieved_source_ids == ("4",)
    assert reloaded["a"].claims


def test_a_recorded_run_scores_without_any_model_call(suite):
    samples = load_samples(suite / "samples.yaml")
    sources = load_sources(suite / "sources.yaml")
    destination = suite / "runs" / "fake.yaml"
    record(FakeSystem(), samples, destination)

    evaluation = evaluate(samples, load_responses(destination), sources, suite="fake")
    assert evaluation.by_name()["citation_verification_rate"].value == 1.0


def test_record_reports_progress_per_sample(suite):
    seen = []
    record(
        FakeSystem(),
        load_samples(suite / "samples.yaml"),
        suite / "runs" / "fake.yaml",
        on_progress=lambda i, total, s, r: seen.append((i, total, s.id)),
    )
    assert seen == [(1, 2, "a"), (2, 2, "b")]


def test_compare_renders_one_column_per_run(suite):
    samples = load_samples(suite / "samples.yaml")
    sources = load_sources(suite / "sources.yaml")
    good = evaluate(samples, load_responses_from(FakeSystem(), samples), sources, suite="good")
    text = compare_to_text([good, good])
    assert text.count("good") == 2
    assert "citation_verification_rate" in text
    assert "| `citation_verification_rate` |" in compare_to_markdown([good, good])


def load_responses_from(system, samples):
    return {s.id: system.answer(s) for s in samples}


def test_compare_with_no_runs_says_so():
    assert compare_to_text([]) == "no runs to compare"


def test_cli_compare_reads_named_run_files(suite, capsys):
    samples = load_samples(suite / "samples.yaml")
    record(FakeSystem(), samples, suite / "runs" / "one.yaml")
    record(FakeSystem(), samples, suite / "runs" / "two.yaml")

    assert (
        main(
            [
                "compare",
                str(suite),
                str(suite / "runs" / "one.yaml"),
                str(suite / "runs" / "two.yaml"),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "one" in out and "two" in out


def test_cli_compare_rejects_a_missing_run_file(suite):
    with pytest.raises(SystemExit, match="no such run file"):
        main(["compare", str(suite), str(suite / "runs" / "absent.yaml")])


def test_run_accepts_an_explicit_responses_file(suite):
    samples = load_samples(suite / "samples.yaml")
    record(FakeSystem(), samples, suite / "runs" / "one.yaml")
    assert main(["run", str(suite), "--responses", str(suite / "runs" / "one.yaml")]) == 0


# --- claim cache -------------------------------------------------------------


def test_claim_cache_makes_a_second_decomposition_free(tmp_path):
    calls = []

    def decomposer(text):
        calls.append(text)
        return ["uma afirmação."]

    cache = ClaimCache(tmp_path / "claims.json")
    assert decompose("uma resposta", decomposer, cache) == ["uma afirmação."]
    assert decompose("uma resposta", decomposer, cache) == ["uma afirmação."]
    assert len(calls) == 1, "the second call must be served from cache"


def test_claim_cache_survives_a_reload(tmp_path):
    path = tmp_path / "claims.json"
    cache = ClaimCache(path)
    decompose("resposta", lambda _t: ["afirmação."], cache)
    cache.save()

    assert ClaimCache(path).get("resposta") == ["afirmação."]


def test_empty_answer_decomposes_to_nothing_without_calling_the_model():
    def explode(_text):
        raise AssertionError("must not be called")

    assert decompose("   ", explode) == []


def test_write_responses_records_the_system_name(tmp_path):
    destination = tmp_path / "run.yaml"
    write_responses([Response(sample_id="a", answer="x")], destination, system_name="my-rag")
    assert "my-rag" in destination.read_text(encoding="utf-8")


def test_written_file_is_not_python_specific(tmp_path):
    """The run file is data, so it must not carry Python object tags."""
    destination = tmp_path / "run.yaml"
    write_responses(
        [Response(sample_id="a", answer="x", citations=(Citation("4", "q"),))], destination
    )
    assert "!!python" not in destination.read_text(encoding="utf-8")


def test_paths_are_accepted_as_path_objects(suite):
    assert isinstance(suite, Path)
    assert load_samples(suite / "samples.yaml")


# --- the claim cache is the product of paid model calls ---------------------


def test_a_truncated_cache_file_names_itself_and_the_remedy(tmp_path):
    """A half-written cache used to surface as a bare JSONDecodeError.

    This JSON costs real model calls to rebuild, so silently starting empty
    would re-bill the user without saying so.
    """
    path = tmp_path / "claims.json"
    path.write_text('{"abc": ["x"]', encoding="utf-8")

    with pytest.raises(ValueError) as caught:
        ClaimCache(path)

    message = str(caught.value)
    assert "claims.json" in message
    assert "Delete it" in message


def test_a_cache_that_is_not_an_object_is_refused(tmp_path):
    path = tmp_path / "claims.json"
    path.write_text('["not", "a", "mapping"]', encoding="utf-8")

    with pytest.raises(ValueError, match="not a JSON object"):
        ClaimCache(path)


def test_save_leaves_no_partial_or_temp_file_on_disk(tmp_path):
    """The write is temp-then-rename, so the file is never half-written."""
    path = tmp_path / "claims.json"
    cache = ClaimCache(path)
    cache.put("uma resposta", ["claim um", "claim dois"])
    cache.save()

    assert json.loads(path.read_text(encoding="utf-8"))
    assert not list(tmp_path.glob("*.tmp")), "a temp file was left behind"


def test_a_saved_cache_round_trips(tmp_path):
    path = tmp_path / "claims.json"
    first = ClaimCache(path)
    first.put("uma resposta", ["claim um"])
    first.save()

    assert ClaimCache(path).get("uma resposta") == ["claim um"]
