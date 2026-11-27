"""Command line entry point: `lastro run <suite> [--format ...] [--fail-under ...]`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lastro import report
from lastro.dataset import load_responses, load_samples, load_sources
from lastro.runner import Evaluation, evaluate, evaluate_suite

_FORMATS = {"text": report.to_text, "json": report.to_json, "markdown": report.to_markdown}


def _parse_thresholds(raw: list[str]) -> dict[str, float]:
    thresholds = {}
    for item in raw:
        name, _, value = item.partition("=")
        if not _:
            raise SystemExit(f"--fail-under expects name=value, got {item!r}")
        thresholds[name.strip()] = float(value)
    return thresholds


def _check_thresholds(evaluation: Evaluation, thresholds: dict[str, float]) -> list[str]:
    """Return one message per breached threshold.

    A threshold is a floor for most metrics and a *ceiling* for the ones where a
    high score is bad. `--fail-under citation_dangling_rate=0.1` asks "is the
    dangling rate at most 10%", and comparing it with `<` like every other metric
    inverted the gate: a run with 100% dangling citations cleared it, and a clean
    run with zero failed it. The flag keeps its name for compatibility, but the
    direction now comes from the metric.

    An undefined metric (no case exercised it) is a breach, not a pass: a suite
    that stopped measuring something must not read as a suite that measures it
    perfectly.
    """
    by_name = evaluation.by_name()
    failures = []
    for name, bound in thresholds.items():
        metric = by_name.get(name)
        if metric is None:
            failures.append(f"{name}: not produced by this suite")
        elif metric.value is None:
            failures.append(
                f"{name}: undefined (no sample exercised it), required {_bound_text(name, bound)}"
            )
        elif name in report.LOWER_IS_BETTER:
            if metric.value > bound:
                failures.append(f"{name}: {metric.value:.3f} > allowed {bound}")
        elif metric.value < bound:
            failures.append(f"{name}: {metric.value:.3f} < required {bound}")
    return failures


def _bound_text(name: str, bound: float) -> str:
    """How the bound reads in a message, given the metric's direction."""
    if name in report.LOWER_IS_BETTER:
        return f"<= {bound}"
    return f">= {bound}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lastro", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="score a benchmark suite")
    run.add_argument("suite", type=Path, help="directory with samples/sources/responses YAML")
    run.add_argument(
        "--responses",
        type=Path,
        help="run file to score (default: responses.yaml inside the suite)",
    )
    run.add_argument("--format", choices=sorted(_FORMATS), default="text")
    run.add_argument("--out", type=Path, help="write the report here instead of stdout")
    run.add_argument(
        "--fail-under",
        action="append",
        default=[],
        metavar="METRIC=VALUE",
        help=(
            "exit non-zero when a metric misses VALUE: a floor for most metrics, "
            "a ceiling for the lower-is-better ones (e.g. citation_dangling_rate); "
            "repeatable"
        ),
    )

    compare = subparsers.add_parser(
        "compare", help="score several recorded runs of one suite side by side"
    )
    compare.add_argument("suite", type=Path)
    compare.add_argument(
        "runs", nargs="+", type=Path, help="response files, e.g. runs/grounded.yaml"
    )
    compare.add_argument("--format", choices=["text", "markdown"], default="text")
    compare.add_argument("--out", type=Path)

    args = parser.parse_args(argv)
    if args.command == "compare":
        return _compare(args)

    if not args.suite.is_dir():
        raise SystemExit(f"not a suite directory: {args.suite}")

    evaluation = evaluate_suite(args.suite, args.responses)
    rendered = _FORMATS[args.format](evaluation)
    if args.out:
        args.out.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)

    failures = _check_thresholds(evaluation, _parse_thresholds(args.fail_under))
    if failures:
        print("\nthreshold failures:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    return 0


def _compare(args) -> int:
    samples = load_samples(args.suite / "samples.yaml")
    sources = load_sources(args.suite / "sources.yaml")
    evaluations = []
    for run_path in args.runs:
        if not run_path.exists():
            raise SystemExit(f"no such run file: {run_path}")
        evaluations.append(
            evaluate(samples, load_responses(run_path), sources, suite=run_path.stem)
        )
    renderer = report.compare_to_markdown if args.format == "markdown" else report.compare_to_text
    rendered = renderer(evaluations)
    if args.out:
        args.out.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
