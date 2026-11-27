"""Rendering an Evaluation as terminal text, JSON or Markdown."""

from __future__ import annotations

import json

from lastro.runner import Evaluation
from lastro.types import MetricResult

# Metrics where a high score is a bad result.
#
# Public because the CLI has to gate them in the opposite direction: a threshold
# on one of these is a ceiling, not a floor. Keeping the list here means the
# renderer and the gate cannot disagree about which metrics those are.
LOWER_IS_BETTER = frozenset({"over_refusal_rate", "citation_dangling_rate", "uncited_answer_rate"})

_GROUPS = [
    ("Retrieval", ["retrieval_recall", "retrieval_full_recall_rate", "retrieval_mrr"]),
    (
        "Citations",
        [
            "citation_verification_rate",
            "citation_verbatim_rate",
            "citation_precision",
            "citation_recall",
            "citation_f1",
            "citation_dangling_rate",
            "uncited_answer_rate",
        ],
    ),
    ("Faithfulness", ["claim_support_rate", "fully_grounded_answer_rate"]),
    ("Refusal", ["refusal_correctness", "over_refusal_rate"]),
]


_DERIVED = {"citation_f1"}


def _counts(metric: MetricResult) -> str:
    if metric.name in _DERIVED or not metric.denominator:
        # A derived metric has no sample count of its own; printing the synthetic
        # denominator of 1 reads as "measured on one case".
        return ""
    if float(metric.numerator).is_integer() and float(metric.denominator).is_integer():
        return f"{int(metric.numerator)}/{int(metric.denominator)}"
    return f"n={int(metric.denominator)}"


def to_text(evaluation: Evaluation) -> str:
    by_name = evaluation.by_name()
    lines = []
    title = f"lastro — {evaluation.suite}" if evaluation.suite else "lastro"
    lines.append(title)
    lines.append("=" * len(title))
    lines.append(f"{evaluation.sample_count} samples scored")
    if evaluation.missing_sample_ids:
        lines.append(f"WARNING: {len(evaluation.missing_sample_ids)} sample(s) had no response")

    for group, names in _GROUPS:
        present = [by_name[name] for name in names if name in by_name]
        if not present:
            continue
        lines.append("")
        lines.append(group)
        for metric in present:
            marker = " (lower is better)" if metric.name in LOWER_IS_BETTER else ""
            counts = _counts(metric)
            counts = f"  [{counts}]" if counts else ""
            lines.append(f"  {metric.name:<32} {metric.format_value():>7}{counts}{marker}")
    return "\n".join(lines)


def to_json(evaluation: Evaluation) -> str:
    return json.dumps(evaluation.to_dict(), indent=2, ensure_ascii=False)


def to_markdown(evaluation: Evaluation) -> str:
    by_name = evaluation.by_name()
    lines = [f"# lastro — {evaluation.suite}".rstrip(" —"), ""]
    lines.append(f"`{evaluation.sample_count}` samples scored.")
    if evaluation.missing_sample_ids:
        lines.append("")
        lines.append(f"> {len(evaluation.missing_sample_ids)} sample(s) had no recorded response.")
    for group, names in _GROUPS:
        present = [by_name[name] for name in names if name in by_name]
        if not present:
            continue
        lines += ["", f"## {group}", "", "| Metric | Value | Counts |", "| --- | ---: | ---: |"]
        for metric in present:
            suffix = " ↓" if metric.name in LOWER_IS_BETTER else ""
            lines.append(
                f"| `{metric.name}`{suffix} | {metric.format_value()} | {_counts(metric) or '—'} |"
            )
    return "\n".join(lines)


__all__ = ["to_text", "to_json", "to_markdown", "compare_to_text", "compare_to_markdown"]


def compare_to_text(evaluations: list[Evaluation]) -> str:
    """Side-by-side table of several runs over the same suite.

    Comparison is the reason to have metrics at all: an absolute score means little
    without a variant that should score worse to check it against.
    """
    if not evaluations:
        return "no runs to compare"
    names = [e.suite or f"run{i}" for i, e in enumerate(evaluations)]
    tables = [e.by_name() for e in evaluations]
    width = max(len(n) for n in names + ["metric"]) + 2
    labels = [
        "  " + m + (" (lower better)" if m in LOWER_IS_BETTER else "")
        for _group, metric_names in _GROUPS
        for m in metric_names
    ]
    label_width = max(len(label) for label in labels) + 2

    lines = ["metric".ljust(label_width) + "".join(n.rjust(width) for n in names)]
    lines.append("-" * (label_width + width * len(names)))
    for group, metric_names in _GROUPS:
        present = [m for m in metric_names if any(m in t for t in tables)]
        if not present:
            continue
        lines.append(group)
        for metric_name in present:
            suffix = " (lower better)" if metric_name in LOWER_IS_BETTER else ""
            row = ("  " + metric_name + suffix).ljust(label_width)
            for table in tables:
                metric = table.get(metric_name)
                row += (metric.format_value() if metric else "-").rjust(width)
            lines.append(row)
    return "\n".join(lines)


def compare_to_markdown(evaluations: list[Evaluation]) -> str:
    if not evaluations:
        return "no runs to compare"
    names = [e.suite or f"run{i}" for i, e in enumerate(evaluations)]
    tables = [e.by_name() for e in evaluations]
    lines = ["| Metric | " + " | ".join(f"`{n}`" for n in names) + " |"]
    lines.append("| --- |" + " ---: |" * len(names))
    for group, metric_names in _GROUPS:
        present = [m for m in metric_names if any(m in t for t in tables)]
        if not present:
            continue
        lines.append(f"| **{group}** |" + " |" * len(names))
        for metric_name in present:
            suffix = " ↓" if metric_name in LOWER_IS_BETTER else ""
            cells = []
            for table in tables:
                metric = table.get(metric_name)
                cells.append(metric.format_value() if metric else "—")
            lines.append(f"| `{metric_name}`{suffix} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
