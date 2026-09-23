"""Registry and factory for pipeline-v2 dataset evaluators."""

from __future__ import annotations

from pipeline_v2.evaluators.base import BaseEvaluator
from pipeline_v2.evaluators.citevqa import CiteVQAEvaluator
from pipeline_v2.evaluators.mmlongbench_doc import MMLongBenchDocEvaluator
from pipeline_v2.evaluators.slidevqa import SlideVQAEvaluator

_EVALUATORS: dict[str, type[BaseEvaluator]] = {
    "citevqa": CiteVQAEvaluator,
    "slidevqa": SlideVQAEvaluator,
    "mmlongbench_doc": MMLongBenchDocEvaluator,
    "mmlongbenchdoc": MMLongBenchDocEvaluator,
}


def register_evaluator(name: str, evaluator_cls: type[BaseEvaluator]) -> None:
    """Registers a new evaluator class under a dataset name."""
    _EVALUATORS[name.lower()] = evaluator_cls


def get_evaluator(dataset_name: str) -> BaseEvaluator:
    """Instantiates and returns the evaluator registered for dataset_name."""
    name_key = dataset_name.lower().replace("-", "_").replace(" ", "")
    # Normalize keys
    if name_key in _EVALUATORS:
        return _EVALUATORS[name_key]()
    if "cite" in name_key:
        return CiteVQAEvaluator()
    if "slide" in name_key:
        return SlideVQAEvaluator()
    if "mmlong" in name_key:
        return MMLongBenchDocEvaluator()
    # Default fallback to SlideVQAEvaluator
    return SlideVQAEvaluator()
