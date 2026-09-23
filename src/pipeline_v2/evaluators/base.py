"""Abstract base evaluator class for dataset evaluations in pipeline-v2."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from atria_core.datasets import Dataset, DatasetConfig
from atria_core.types import MultiPageDocumentInstance

from pipeline_v2.models.qwen import QwenVLModel
from pipeline_v2.parsers.docling import DoclingTransform


class BaseEvaluator(ABC):
    """Abstract Base Evaluator defining interface for dataset evaluation."""

    @abstractmethod
    def evaluate(
        self,
        dataset: Dataset[MultiPageDocumentInstance, DatasetConfig],
        model: QwenVLModel,
        output_file: Path | None = None,
        docling_transform: DoclingTransform | None = None,
    ) -> dict[str, Any]:
        """Runs model evaluation across dataset splits and returns metric summary."""
        raise NotImplementedError
