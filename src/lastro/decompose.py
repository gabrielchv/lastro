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
        if not path.exists():
            return

        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            # A truncated file is not hypothetical: a crash or a full disk during
            # `save` leaves one, and this JSON is the product of *paid* model
            # calls, so silently starting empty would quietly re-bill them.
            # Failing loudly, naming the file and the remedy, is the useful
            # behaviour.
            raise ValueError(
                f"claim cache at {path} is not readable JSON ({error}). "
                "Delete it to re-decompose the answers, which costs model calls."
            ) from error

        if not isinstance(loaded, dict):
            raise ValueError(f"claim cache at {path} is not a JSON object; delete it to rebuild.")
        self._data = loaded

    def get(self, answer: str) -> list[str] | None:
        return self._data.get(_key(answer))

    def put(self, answer: str, claims: list[str]) -> None:
        self._data[_key(answer)] = claims

    def save(self) -> None:
        """Write the cache atomically.

        `write_text` truncates the file and then writes: a process killed in
        between, or a disk that fills, leaves a half-written file and every later
        run dies on it. Write to a sibling temp file and rename, which on POSIX is
        atomic, so the file on disk is always either the old cache or the new one.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self._data, indent=2, ensure_ascii=False, sort_keys=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(self.path)

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
