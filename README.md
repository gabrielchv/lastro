# lastro

Measuring whether generated Portuguese is actually backed by its sources — without
asking a model to be the judge.

*Lastro* is Portuguese for ballast, and for the backing behind a financial
instrument. An **afirmação sem lastro** is a claim with nothing under it. That is
what these metrics look for.

---

## The problem this exists for

A retrieval system that answers legal questions will tell you it is grounded. Mine
did. `inquilino-ia` answers questions about the Brazilian tenancy statute, and its
README says every claim is backed by a verbatim quote checked in code. It has an
eval suite. The suite scored 17 questions and reported a pass rate.

The pass rate was this:

```python
cited = {c["artigo"] for c in extraction["citacoes"]}
passed = bool(cited & expected)
```

Any overlap counts. Cite four articles when one is correct — passes. Cite the
correct article and invent the quote underneath it — passes, because the quote is
never checked: the eval compares article numbers and never calls the project's own
`verify_citations`. The one claim the system is built around is the one thing its
eval could not see.

That gap is not unusual, and it is not fixed by scoring harder. It is fixed by
measuring the right things.

## What it measures

Four families, because "is it grounded" is four different questions with four
different fixes.

| Metric | Question |
| --- | --- |
| `retrieval_recall`, `retrieval_mrr` | Did the right passage even reach the model? |
| `citation_verification_rate` | Does the quoted text exist in the source it names? |
| `citation_precision` / `_recall` | Were the right sources cited, without shotgunning? |
| `claim_support_rate` | Is every sentence of the answer recoverable from context? |
| `refusal_correctness`, `over_refusal_rate` | Does it decline what it cannot support — and only that? |

Separating retrieval from generation is the part that pays for itself. When an
answer is ungrounded, either retrieval never surfaced the passage or the model
ignored it, and those need opposite fixes — chunking and embeddings versus
prompting and verification. One end-to-end pass rate cannot tell you which.

The refusal pair matters for the same reason. A system that refuses everything
cannot hallucinate and would top every other metric here. `over_refusal_rate` is
what it costs.

## The design rule: no model grades another model

Every score above is computed by comparing text against source text. Nothing is
delegated to an LLM judge.

This is not purism. An LLM judge is a second model with the same failure mode as
the first, it costs money on every run, and it returns a different number each
time — so it cannot gate a build. A string comparison is free, identical on every
run, and cannot be talked out of its answer.

One step does need a model: splitting an answer into atomic claims. It is confined
to `lastro.decompose`, cached to disk keyed by a hash of the answer, and it only
ever *splits* text — the decision about whether a claim is supported stays in
`lastro.match`. Buy the split once; every rescoring after that is free and
byte-identical.

### Verification is graded, not boolean

`normalized(quote) in normalized(source)` answers one question — did the model
reproduce a contiguous span — and collapses three situations into `False`:

- the model altered the quote's punctuation or accents,
- the model truncated or reordered a real passage,
- the model invented the passage.

Only the third is a hallucination. `lastro` reports which happened:

| Level | Meaning |
| --- | --- |
| `EXACT` | Verbatim, once invisible characters are cleaned up |
| `NORMALIZED` | Identical after folding case, accents and legal-reference spelling |
| `PARTIAL` | Wording recoverable from the source, in order, above a coverage threshold |
| `NONE` | Not supported |

The gap between `citation_verification_rate` and `citation_verbatim_rate` is
exactly the quotes that were altered rather than fabricated.

## Why this is Portuguese-specific

Because the naive version silently under-reports grounding in Portuguese, and the
reasons are all boring:

- **Legal ordinals.** The statute writes `Art. 4º`. Models write `art. 4o`,
  `artigo 4`, `Art 4°`. The masculine ordinal `º` and the degree sign `°` are
  different characters and models mix them freely.
- **Morphology.** The statute says *poderá devolvê-lo*; a faithful paraphrase says
  *pode devolver*. Same lemmas. Comparing surface tokens scores that as
  unsupported — on this corpus, exact-token matching lost about a fifth of the
  tokens of claims that genuinely came from the source. `lastro` stems before
  computing coverage, with a deliberately conservative Portuguese stemmer:
  over-stemming would credit a fabrication, so suffixes are stripped only when a
  stem of three or more characters survives, and only for the `PARTIAL` tier.
  `EXACT` and `NORMALIZED` never see a stem, so "verbatim" keeps its literal
  meaning.
- **Accents.** Models transcribing a quote lose diacritics. `locação` and
  `locacao` are the same word for any purpose that matters here.
- **Scraped-source artifacts.** Soft hyphens and line-wrap breaks (`loca-\nção`)
  come straight from the source document and are not the model's doing.

Word order is still respected. Coverage is an ordered subsequence ratio, not a bag
of words, because the same words rearranged can invert a legal meaning — *o locador
deve* and *deve o locatário* are opposite obligations.

## Scoring is free and reproducible

Recording a system's answers costs API calls. Scoring reads a file.

```
benchmark/inquilinato/
  samples.yaml     32 questions with the articles that must ground them
  sources.yaml     the 90 articles of Lei 8.245/1991
  runs/*.yaml      recorded system outputs, committed
```

Splitting them is what makes the benchmark usable as a CI gate: no API key, no
cost, no variance between runs. `lastro run` never calls a model.

```bash
lastro run benchmark/inquilinato --format markdown
lastro run benchmark/inquilinato --fail-under citation_verification_rate=0.9
```

`--fail-under` exits non-zero when a metric drops below a threshold. An *undefined*
metric — one no sample exercised — counts as a breach, so a suite that quietly
stops measuring something cannot pass as one that measures it perfectly.

## The benchmark

32 questions over Lei 8.245/1991, the Brazilian tenancy statute. Ground truth is
the article that must back a correct answer, checked against the statute text
rather than assumed. Four questions are out-of-scope traps — one of them asks about
mortgage rates, which is plausibly real-estate adjacent and therefore the useful
kind of trap.

The statute is public domain (Lei 9.610/1998, art. 8, IV). Article text comes from
`inquilino-ia`'s ingestion, which strips revoked amendments so historical wording
is not scored as current law.

## Validating the metrics themselves

A metric suite is only worth trusting if it can tell a good system from a degraded
one. `benchmark/inquilinato/record_systems.py` records three configurations that
each remove exactly one defence:

| Variant | What is removed |
| --- | --- |
| `grounded` | nothing — retrieval, citation extraction, refusal instruction |
| `no-retrieval` | the articles; refusal instruction kept |
| `always-answer` | the refusal instruction; retrieval intact |
| `parametric` | both — the model must answer every question from memory |

`no-retrieval` was meant to be the negative control and is not one. With no
articles supplied but the refusal instruction still in place, the model refuses all
32 questions rather than inventing anything. That is a genuine result about prompt
design — removing the sources does not by itself produce fabrication — and it is
why `parametric` exists. Both the sources and the escape hatch have to go before a
model starts making things up.

Results: [`benchmark/inquilinato/RESULTS.md`](benchmark/inquilinato/RESULTS.md).

## Install

```bash
pip install -e ".[dev]"      # scoring: pyyaml only
pytest                        # 42 tests, no API key
```

Recording additionally needs `google-genai` and `GEMINI_API_KEY`.

## Using it on your own system

Implement one method. The library never imports a model client.

```python
from lastro.types import Citation, Response, Sample

class MySystem:
    name = "my-rag"

    def answer(self, sample: Sample) -> Response:
        chunks = my_retriever(sample.question)
        text = my_model(sample.question, chunks)
        return Response(
            sample_id=sample.id,
            answer=text,
            citations=tuple(Citation(source_id=c.id, quote=c.quote) for c in my_citations(text)),
            retrieved_source_ids=tuple(c.id for c in chunks),
            refused=is_refusal(text),
        )
```

Then `lastro.record.record(MySystem(), samples, Path("runs/mine.yaml"))` and score
the directory.

## Limitations

Worth stating plainly, because a groundedness metric that oversells itself has the
same problem as the systems it measures.

- **`PARTIAL` is evidence, not entailment.** High ordered token coverage means a
  claim was drawn from the context. It does not prove the context entails it — a
  negation flips meaning while barely moving coverage. Treat `claim_support_rate`
  as a floor on unfaithfulness, not a certificate of faithfulness.
- **The stemmer is a heuristic.** It merges some unrelated words. It is tuned to
  fail toward *not* crediting a claim, and the test suite pins that direction, but
  it is not a morphological analyzer.
- **Ground truth is single-corpus.** One statute, one domain, hand-labelled by one
  person. The metric definitions generalize; the benchmark numbers do not.
- **Coverage scales with context size.** A larger retrieved context offers more
  tokens for a claim to match against, so the same claim scores higher when more
  articles are retrieved. Comparing `claim_support_rate` across systems is only
  meaningful at a comparable retrieval depth — all runs here use `top_k = 5`.
- **No human-agreement study yet.** The negative control substitutes for one: it
  establishes that the matcher separates grounded claims from ungrounded ones,
  which is weaker than agreeing with a human on each individual claim. Labelling a
  sample by hand is the honest next step.

## License

MIT.
