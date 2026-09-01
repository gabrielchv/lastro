"""lastro — evaluate grounded generation in Brazilian Portuguese.

Named for the Portuguese *lastro* (ballast, backing): an *afirmação sem lastro* is
a claim with nothing behind it, which is exactly what these metrics look for.

The design rule is that no model is asked to score another model. Citation
verification, claim support and every rate here are decided by comparing text
against source text. The one step that needs a model — splitting an answer into
atomic claims — is cached to disk, so scoring is deterministic and free to rerun.
"""

from lastro.runner import Evaluation, evaluate, evaluate_suite
from lastro.types import Citation, MatchLevel, MetricResult, Response, Sample, Source

__version__ = "0.1.0"

__all__ = [
    "Citation",
    "Evaluation",
    "MatchLevel",
    "MetricResult",
    "Response",
    "Sample",
    "Source",
    "evaluate",
    "evaluate_suite",
]
