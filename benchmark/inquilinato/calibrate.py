"""Calibrates the claim-support threshold against a negative control.

Picking a coverage threshold by eye is how an eval framework quietly starts
reporting whatever number its author wanted. This script picks it from data
instead.

The control is the `parametric` run: no articles supplied and no refusal option, so
every answer comes from the model's own memory. Scoring those claims against the
context they never saw gives a null distribution — coverage that arises from two
texts sharing a topic and a legal register, not from one being drawn from the
other. The grounded run gives the positive distribution.

The `no-retrieval` run is deliberately *not* the control. Removing the articles
while leaving the refusal instruction in place makes the model refuse all 32
questions, which produces no claims to measure. Both factors have to go.

A usable threshold separates them. This reports both distributions, the overlap at
each candidate cut, and the value that maximizes separation (Youden's J: the point
where true-positive rate minus false-positive rate is greatest).

    python benchmark/inquilinato/calibrate.py
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lastro.dataset import load_responses, load_sources  # noqa: E402
from lastro.match import match_quote  # noqa: E402

HERE = Path(__file__).parent


def _coverages(run_name: str, sources, context_by_sample) -> list[float]:
    """Coverage of every claim in a run, against the *grounded* run's context.

    A control run records no retrieval of its own, so its claims are scored against
    the context the grounded system retrieved for the same question. That is the
    comparison that matters: would this claim have been credited if it had been
    checked against the real source.
    """
    responses = load_responses(HERE / "runs" / f"{run_name}.yaml")
    values = []
    for sample_id, response in responses.items():
        if response.refused or not response.claims:
            continue
        context = context_by_sample.get(sample_id, "")
        if not context:
            continue
        for claim in response.claims:
            values.append(match_quote(claim, context)[1])
    return values


def _histogram(values: list[float], label: str) -> str:
    lines = [f"{label}  (n={len(values)}, median={statistics.median(values):.3f})"]
    for lower in range(0, 100, 10):
        bucket = [v for v in values if lower / 100 <= v < (lower + 10) / 100]
        if lower == 90:
            bucket = [v for v in values if v >= 0.9]
        bar = "#" * round(len(bucket) / max(1, len(values)) * 60)
        lines.append(f"  {lower:>3}-{lower + 9:<3}% {bar} {len(bucket)}")
    return "\n".join(lines)


def main() -> int:
    sources = load_sources(HERE / "sources.yaml")
    grounded = load_responses(HERE / "runs" / "grounded.yaml")
    context_by_sample = {
        sample_id: "\n".join(
            sources[s].text for s in response.retrieved_source_ids if s in sources
        )
        for sample_id, response in grounded.items()
    }

    positive = _coverages("grounded", sources, context_by_sample)
    negative = _coverages("parametric", sources, context_by_sample)

    if not negative:
        print("no negative control recorded — run record_systems.py --system parametric")
        return 1

    print(_histogram(positive, "grounded (claims drawn from retrieved context)"))
    print()
    print(_histogram(negative, "parametric (no articles, no refusal option) — NEGATIVE CONTROL"))
    print()

    print(f"{'cut':>6}  {'TPR':>7}  {'FPR':>7}  {'J':>7}")
    best = (None, -1.0)
    for step in range(30, 96, 5):
        cut = step / 100
        tpr = sum(1 for v in positive if v >= cut) / len(positive)
        fpr = sum(1 for v in negative if v >= cut) / len(negative)
        youden = tpr - fpr
        if youden > best[1]:
            best = (cut, youden)
        marker = ""
        print(f"{cut:>6.2f}  {tpr:>6.1%}  {fpr:>6.1%}  {youden:>6.3f}{marker}")

    print()
    print(f"best separation at cut={best[0]:.2f} (Youden's J = {best[1]:.3f})")
    print(
        "\nNote: a high FPR at low cuts is expected and is the point of the control -- "
        "\ntwo texts about the same statute share vocabulary even when one was not "
        "\nderived from the other."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
