
## What the comparison shows

**The refusal pair earns its place.** `no-retrieval` scores 100% on
`refusal_correctness` — a perfect score on the metric most people would report
alone. It got there by refusing all 32 questions, including the 28 it had the
answer to. `over_refusal_rate` at 100% is the only thing in the table that says
so. A suite that reports refusal in one direction ranks a useless system first.

**Removing the refusal instruction degrades grounding, not just refusal.**
`always-answer` was expected to differ from `grounded` only on the refusal
metrics: same retriever, same articles, same citation extraction. Refusal
correctness does collapse as predicted, from 100% to 25%. But `citation_verbatim_rate`
also falls and `claim_support_rate` falls with it — told to always be useful, the
model paraphrases the statute more loosely and quotes it less exactly, on the
in-scope questions too. The instruction that governs when to answer also changes
how closely the answer sticks to the source.

**Removing retrieval does not by itself cause fabrication.** This was the
prediction that failed. `no-retrieval` was supposed to be the negative control —
no articles, so citations with nothing behind them. Instead the model refused
everything, because the refusal instruction was still there and correctly fired
when no articles were supplied. Fabrication needed both the sources and the escape
hatch removed, which is what `parametric` does and why it exists.

## Reading the absolute numbers

`citation_verification_rate` at 100% on `grounded` is a weaker result than it
looks. Citations are produced by a second structured-output call that is given the
article text and asked to copy from it, so the task is transcription, and the
metric mostly confirms transcription works. It would catch a regression that
started paraphrasing quotes — `always-answer` shows exactly that, at 86.9% verbatim
— but it is not evidence that the answer as a whole is grounded. That is what
`claim_support_rate` is for, and it sits far lower.

`fully_grounded_answer_rate` is the strictest number here and the most useful one
to watch: one unsupported sentence in an otherwise correct answer is enough to
mislead a reader, and a per-claim average hides it.
