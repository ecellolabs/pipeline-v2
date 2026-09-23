"""Evaluators package for dataset benchmarks."""

from __future__ import annotations

from pipeline_v2.evaluators.base import BaseEvaluator
from pipeline_v2.evaluators.citevqa import CiteVQAEvaluator
from pipeline_v2.evaluators.mmlongbench_doc import MMLongBenchDocEvaluator
from pipeline_v2.evaluators.registry import get_evaluator, register_evaluator
from pipeline_v2.evaluators.slidevqa import SlideVQAEvaluator

__all__ = [
    "BaseEvaluator",
    "CiteVQAEvaluator",
    "MMLongBenchDocEvaluator",
    "SlideVQAEvaluator",
    "get_evaluator",
    "register_evaluator",
]
