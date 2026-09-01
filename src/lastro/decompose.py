"""Splitting an answer into atomic claims, cached on disk.

This is the one step in the pipeline that a string comparison cannot do well, and
so the one place a model is used. It is confined and cached deliberately:

- The model *splits* text. It never decides whether a claim is supported — that
  stays in `lastro.match`, where the decision is a comparison against source text
  and cannot be talked into a wrong answer.
- Results are keyed by a hash of the answer text, so the split is bought once.
  Rescoring the same responses is free and returns byte-identical claims, which is
  what lets the benchmark gate CI.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

Decomposer = Callable[[str], list[str]]


def _key(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


class ClaimCache:
    """A JSON file mapping answer-hash to the claims extracted from that answer."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: dict[str, list[str]] = {}
        if path.exists():
            self._data = json.loads(path.read_text(encoding="utf-8"))

    def get(self, answer: str) -> list[str] | None:
        return self._data.get(_key(answer))

    def put(self, answer: str, claims: list[str]) -> None:
        self._data[_key(answer)] = claims

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    def __len__(self) -> int:
        return len(self._data)


def decompose(answer: str, decomposer: Decomposer, cache: ClaimCache | None = None) -> list[str]:
    """Return atomic claims for `answer`, consulting `cache` first."""
    if not answer.strip():
        return []
    if cache is not None:
        cached = cache.get(answer)
        if cached is not None:
            return cached
    claims = [claim.strip() for claim in decomposer(answer) if claim.strip()]
    if cache is not None:
        cache.put(answer, claims)
    return claims


__all__ = ["ClaimCache", "Decomposer", "decompose"]
