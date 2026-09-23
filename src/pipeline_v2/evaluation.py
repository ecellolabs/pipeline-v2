"""Evaluation runner for document visual question answering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from atria_core.datasets import Dataset, DatasetConfig
from atria_core.logger import get_logger
from atria_core.types import MultiPageDocumentInstance

from pipeline_v2.evaluators.registry import get_evaluator
from pipeline_v2.models.qwen import QwenVLModel
from pipeline_v2.parsers.docling import DoclingTransform

logger = get_logger(__name__)


def evaluate_dataset(
    dataset: Dataset[MultiPageDocumentInstance, DatasetConfig],
    model: QwenVLModel,
    output_file: Path | None = None,
    docling_transform: DoclingTransform | None = None,
) -> dict[str, Any]:
    """Runs model evaluation across all dataset splits using dataset-specific evaluator."""
    evaluator = get_evaluator(dataset.name)
    return evaluator.evaluate(
        dataset=dataset,
        model=model,
        output_file=output_file,
        docling_transform=docling_transform,
    )
