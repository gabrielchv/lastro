"""Loading benchmark cases, sources and recorded system responses from YAML.

Responses are recorded to disk rather than generated during scoring. That is what
lets the metrics run in CI: no API key, no cost, no variance between runs. A live
run against a system is a separate step that *produces* one of these files, so
"score the benchmark" and "call a model" are never the same command.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from lastro.types import Citation, Response, Sample, Source


def _read_yaml(path: Path):
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or []


def load_samples(path: Path) -> list[Sample]:
    samples = []
    for index, raw in enumerate(_read_yaml(path)):
        expected = raw.get("expected_source_ids") or []
        samples.append(
            Sample(
                id=str(raw.get("id") or f"case-{index + 1}"),
                question=raw["question"],
                expected_source_ids=frozenset(str(s) for s in expected),
                note=raw.get("note", ""),
            )
        )
    _assert_unique(s.id for s in samples)
    return samples


def load_sources(path: Path) -> dict[str, Source]:
    sources = {}
    for raw in _read_yaml(path):
        source = Source(id=str(raw["id"]), text=raw["text"])
        sources[source.id] = source
    return sources


def load_responses(path: Path) -> dict[str, Response]:
    responses = {}
    for raw in _read_yaml(path):
        citations = tuple(
            Citation(source_id=str(c["source_id"]), quote=c["quote"])
            for c in raw.get("citations") or []
        )
        response = Response(
            sample_id=str(raw["sample_id"]),
            answer=raw.get("answer", ""),
            citations=citations,
            retrieved_source_ids=tuple(str(s) for s in raw.get("retrieved_source_ids") or []),
            refused=bool(raw.get("refused", False)),
            claims=tuple(raw.get("claims") or []),
        )
        responses[response.sample_id] = response
    return responses


def pair(
    samples: list[Sample], responses: dict[str, Response]
) -> tuple[list[tuple[Sample, Response]], list[str]]:
    """Join samples to responses, reporting which samples have no response.

    Missing responses are returned rather than silently dropped: a run that
    covered half the benchmark should not present its scores as if it covered all
    of it.
    """
    pairs = []
    missing = []
    for sample in samples:
        response = responses.get(sample.id)
        if response is None:
            missing.append(sample.id)
        else:
            pairs.append((sample, response))
    return pairs, missing


def _assert_unique(ids) -> None:
    seen = set()
    for identifier in ids:
        if identifier in seen:
            raise ValueError(f"duplicate sample id: {identifier!r}")
        seen.add(identifier)


__all__ = ["load_samples", "load_sources", "load_responses", "pair"]
