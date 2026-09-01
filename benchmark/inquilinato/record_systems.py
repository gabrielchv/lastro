"""Records three RAG configurations over Lei 8.245/1991 into scoreable response files.

The point of recording three is that a metric suite is only credible if it can
tell a good system from a degraded one. Each variant removes exactly one defence:

  grounded       retrieval + citation extraction + an explicit refusal instruction
  no-retrieval   no articles supplied, refusal instruction kept
  always-answer  retrieval intact, refusal instruction removed
  parametric     no articles AND no refusal instruction — the model must answer
                 every question from memory alone

`no-retrieval` was originally intended as the negative control and turned out not
to be one: with the refusal instruction still in place and no articles supplied,
the model refuses all 32 questions rather than inventing anything. That is a real
result about prompt design, and it is why `parametric` exists — removing retrieval
only forces fabrication once the refusal escape hatch is removed too.

`parametric` is therefore the null distribution used by `calibrate.py`: claims
produced with no access to the source, scored against the source anyway.

Requires GEMINI_API_KEY. Retrieval is in-memory over sources.yaml rather than the
pgvector store inquilino-ia deploys, so the benchmark is self-contained.

    python benchmark/inquilinato/record_systems.py --system grounded
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lastro.dataset import load_samples, load_sources  # noqa: E402
from lastro.decompose import ClaimCache, decompose  # noqa: E402
from lastro.record import record  # noqa: E402
from lastro.types import Citation, Response, Sample, Source  # noqa: E402

HERE = Path(__file__).parent
EMBED_CACHE = HERE / ".cache" / "embeddings.json"
CLAIM_CACHE = HERE / ".cache" / "claims.json"
TOP_K = 5

REFUSAL = "Não encontrei base na Lei do Inquilinato para responder a essa pergunta."

ANSWER_PROMPT = """Você responde dúvidas sobre a Lei do Inquilinato (Lei 8.245/1991).

Use exclusivamente os artigos fornecidos abaixo. Responda em português, de forma
direta, em no máximo 4 frases.

Se os artigos fornecidos não sustentarem uma resposta, responda exatamente:
{refusal}

Artigos:
{articles}"""

ANSWER_PROMPT_NO_REFUSAL = """Você responde dúvidas sobre a Lei do Inquilinato (Lei 8.245/1991).

Use os artigos fornecidos abaixo. Responda em português, de forma direta, em no
máximo 4 frases. Sempre forneça uma resposta útil ao usuário.

Artigos:
{articles}"""

CITATION_PROMPT = """Extraia as citações que sustentam a resposta.

Para cada afirmação da resposta, informe o número do artigo e um trecho VERBATIM
copiado exatamente do texto do artigo fornecido. Não parafraseie o trecho."""

CITATION_SCHEMA = {
    "type": "object",
    "properties": {
        "citacoes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"artigo": {"type": "string"}, "trecho": {"type": "string"}},
                "required": ["artigo", "trecho"],
            },
        }
    },
    "required": ["citacoes"],
}

CLAIM_PROMPT = """Divida a resposta em afirmações atômicas, uma por linha.

Cada afirmação deve ser uma frase completa e independente, contendo um único fato
jurídico. Não numere. Não adicione comentários. Se a resposta for uma recusa,
devolva nada."""


def _client():
    from google import genai

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY not set")
    return genai.Client(api_key=key)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class Retriever:
    """In-memory cosine retrieval over the statute, with embeddings cached on disk."""

    def __init__(self, client, sources: dict[str, Source]) -> None:
        self.client = client
        self.sources = sources
        self._cache: dict[str, list[float]] = {}
        if EMBED_CACHE.exists():
            self._cache = json.loads(EMBED_CACHE.read_text(encoding="utf-8"))
        self._index = {sid: self._embed(s.text, "RETRIEVAL_DOCUMENT") for sid, s in sources.items()}
        self._save()

    def _embed(self, text: str, task_type: str) -> list[float]:
        # hashlib, not the builtin hash(): string hashing is salted per process
        # (PYTHONHASHSEED), so a builtin-hash key misses on every new run and
        # re-embeds the whole corpus while the cache file grows without bound.
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
        key = f"{task_type}:{digest}"
        if key in self._cache:
            return self._cache[key]
        from google.genai import types

        result = self.client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=768),
        )
        vector = list(result.embeddings[0].values)
        self._cache[key] = vector
        return vector

    def _save(self) -> None:
        EMBED_CACHE.parent.mkdir(parents=True, exist_ok=True)
        EMBED_CACHE.write_text(json.dumps(self._cache), encoding="utf-8")

    def retrieve(self, question: str, top_k: int = TOP_K) -> list[Source]:
        query = self._embed(question, "RETRIEVAL_QUERY")
        self._save()
        ranked = sorted(
            self._index.items(), key=lambda kv: _cosine(query, kv[1]), reverse=True
        )
        return [self.sources[sid] for sid, _ in ranked[:top_k]]


class InquilinatoSystem:
    def __init__(self, client, retriever: Retriever, variant: str, claim_cache: ClaimCache):
        self.client = client
        self.retriever = retriever
        self.variant = variant
        self.name = variant
        self.claim_cache = claim_cache

    def _generate(self, system_prompt: str, question: str) -> str:
        from google.genai import types

        result = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=question,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt, temperature=0.0
            ),
        )
        return (result.text or "").strip()

    def _generate_json(self, system_prompt: str, content: str) -> dict:
        from google.genai import types

        result = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=content,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=CITATION_SCHEMA,
            ),
        )
        try:
            return json.loads(result.text or "{}")
        except json.JSONDecodeError:
            return {}

    def _claims(self, answer: str) -> list[str]:
        def run(text: str) -> list[str]:
            raw = self._generate(CLAIM_PROMPT, text)
            return [line.strip("-• ").strip() for line in raw.splitlines() if line.strip()]

        return decompose(answer, run, self.claim_cache)

    def answer(self, sample: Sample) -> Response:
        without_retrieval = self.variant in ("no-retrieval", "parametric")
        if without_retrieval:
            retrieved: list[Source] = []
            articles_block = "(nenhum artigo fornecido)"
        else:
            retrieved = self.retriever.retrieve(sample.question)
            articles_block = "\n\n".join(f"Art. {s.id}\n{s.text}" for s in retrieved)

        template = (
            ANSWER_PROMPT_NO_REFUSAL
            if self.variant in ("always-answer", "parametric")
            else ANSWER_PROMPT
        )
        prompt = template.format(articles=articles_block, refusal=REFUSAL)
        answer = self._generate(prompt, sample.question)

        refused = _is_refusal(answer)
        citations: tuple[Citation, ...] = ()
        claims: tuple[str, ...] = ()
        if not refused and answer:
            extraction = self._generate_json(
                CITATION_PROMPT,
                f"Pergunta: {sample.question}\n\nResposta: {answer}\n\nArtigos:\n{articles_block}",
            )
            citations = tuple(
                Citation(source_id=str(c.get("artigo", "")), quote=str(c.get("trecho", "")))
                for c in extraction.get("citacoes", [])
                if c.get("trecho")
            )
            claims = tuple(self._claims(answer))

        return Response(
            sample_id=sample.id,
            answer=answer,
            citations=citations,
            retrieved_source_ids=tuple(s.id for s in retrieved),
            refused=refused,
            claims=claims,
        )


def _is_refusal(answer: str) -> bool:
    """A refusal is detected by intent, not by exact string equality.

    Pinning to the literal sentence would score a model that refuses in its own
    words as having answered, inflating the hallucination rate with what is
    actually correct behaviour.
    """
    if not answer.strip():
        return True
    lowered = answer.lower()
    markers = (
        "não encontrei base",
        "nao encontrei base",
        "não há base",
        "não é possível responder",
        "não posso responder",
        "não trata",
        "fora do escopo",
        "não sustentam",
    )
    return any(marker in lowered for marker in markers)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system",
        choices=["grounded", "no-retrieval", "always-answer", "parametric"],
        default="grounded",
    )
    args = parser.parse_args()

    samples = load_samples(HERE / "samples.yaml")
    sources = load_sources(HERE / "sources.yaml")
    client = _client()
    claim_cache = ClaimCache(CLAIM_CACHE)
    retriever = Retriever(client, sources)
    system = InquilinatoSystem(client, retriever, args.system, claim_cache)

    def progress(index, total, sample, response):
        state = "refused" if response.refused else f"{len(response.citations)} citations"
        print(f"  [{index:>2}/{total}] {sample.id:<32} {state}")

    destination = HERE / "runs" / f"{args.system}.yaml"
    print(f"recording {args.system} -> {destination.relative_to(ROOT)}")
    record(system, samples, destination, on_progress=progress)
    claim_cache.save()
    print(f"done. claim cache holds {len(claim_cache)} answers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
