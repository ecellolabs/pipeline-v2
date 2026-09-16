from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import overload

from atria_core.datasets import Dataset, datasets
from atria_core.logger import get_logger
from atria_core.types import (
    BoundingBoxMode,
    DatasetMetadata,
    DatasetSplitType,
    DocumentContent,
    ElementArray,
    MultiPageDocumentInstance,
    MultiPageQAPair,
    MultiPageQuestionAnsweringAnnotation,
    SinglePageDocumentInstance,
)

from agentic.datasets.utils import require_manual_path

_HOMEPAGE = "https://github.com/rubenpt91/MP-DocVQA-Framework"
_RRC_PORTAL = "https://rrc.cvc.uab.es/?ch=17&com=downloads"
_SPLIT_NAMES = {
    DatasetSplitType.train: "train",
    DatasetSplitType.validation: "val",
    DatasetSplitType.test: "test",
}

logger = get_logger(__name__)


@dataclass
class _IMDBRecord:
    question_id: int
    question: str
    image_id: str
    image_name: list[str]
    answer_page_idx: int
    answers: list[str]
    ocr_tokens: list[list[str]]
    ocr_normalized_boxes: list[list[tuple[float, float, float, float]]]

    @classmethod
    def from_raw(cls, record: dict[str, object]) -> _IMDBRecord:
        image_name = record["image_name"]
        assert isinstance(image_name, list)

        raw_answers = record["answers"]
        assert isinstance(raw_answers, list)
        deduplicated_answers = list(
            dict.fromkeys(str(answer) for answer in raw_answers)
        )

        ocr_tokens = record["ocr_tokens"]
        assert isinstance(ocr_tokens, list)

        ocr_normalized_boxes = record["ocr_normalized_boxes"]
        assert isinstance(ocr_normalized_boxes, list)

        question_id = record["question_id"]
        assert isinstance(question_id, int)

        answer_page_idx = record["answer_page_idx"]
        assert isinstance(answer_page_idx, int)

        return cls(
            question_id=question_id,
            question=str(record["question"]),
            image_id=str(record["image_id"]),
            image_name=[str(name) for name in image_name],
            answer_page_idx=answer_page_idx,
            answers=deduplicated_answers,
            ocr_tokens=[
                [str(token) for token in page_tokens] for page_tokens in ocr_tokens
            ],
            ocr_normalized_boxes=[
                [
                    (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
                    for box in page_boxes
                ]
                for page_boxes in ocr_normalized_boxes
            ],
        )


class SplitIterator(Sequence[_IMDBRecord]):
    def __init__(self, imdb_dir: Path, split: DatasetSplitType) -> None:
        import numpy as np

        imdb_path = imdb_dir / f"imdb_{_SPLIT_NAMES[split]}.npy"
        logger.info(f"Loading MP-DocVQA {split.value} split from {imdb_path}")
        data = np.load(imdb_path, allow_pickle=True)
        self._records = [_IMDBRecord.from_raw(record) for record in data[1:]]

    @overload
    def __getitem__(self, index: int) -> _IMDBRecord: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[_IMDBRecord]: ...

    def __getitem__(self, index: int | slice) -> _IMDBRecord | Sequence[_IMDBRecord]:
        return self._records[index]

    def __len__(self) -> int:
        return len(self._records)


class InputTransform:
    def __init__(self, images_dir: Path) -> None:
        self._images_dir = images_dir

    def _build_page(
        self,
        image_id: str,
        page_number: int,
        image_name: str,
        tokens: list[str],
        boxes: list[tuple[float, float, float, float]],
    ) -> SinglePageDocumentInstance:
        content = None
        if tokens:
            content = DocumentContent(
                elements=ElementArray.from_words(
                    texts=tokens,
                    bboxes=boxes,
                    bbox_mode=BoundingBoxMode.XYXY,
                    normalized=True,
                )
            )
        return SinglePageDocumentInstance.from_image(
            self._images_dir / f"{image_name}.jpg",
            sample_id=f"{image_id}#{page_number}",
            content=content,
        )

    def __call__(self, record: _IMDBRecord) -> MultiPageDocumentInstance:
        pages = [
            self._build_page(
                record.image_id,
                page_number,
                image_name,
                record.ocr_tokens[page_number],
                record.ocr_normalized_boxes[page_number],
            )
            for page_number, image_name in enumerate(record.image_name)
        ]

        qa_pair = MultiPageQAPair(
            id=record.question_id,
            question_text=record.question,
            answer_text=record.answers[0],
            evidence_pages=[record.answer_page_idx],
            alternative_answers=record.answers[1:],
        )
        return MultiPageDocumentInstance(
            sample_id=record.image_id, pages=pages
        ).add_annotation(
            annotation=MultiPageQuestionAnsweringAnnotation(qa_pairs=[qa_pair])
        )


@datasets.register
class MPDocVQA(Dataset[MultiPageDocumentInstance]):
    """MP-DocVQA: question answering over multi-page scanned documents,
    reusing SP-DocVQA's questions and answers with added document context."""

    __module_name__ = "mpdocvqa"

    def _download(
        self, data_dir: str, access_token: str | None = None
    ) -> dict[str, Path]:
        imdb_dir = require_manual_path(
            data_dir=data_dir,
            expected_path="mpdocvqa/imdb",
            homepage=_HOMEPAGE,
            instructions=(
                f"Register at {_RRC_PORTAL} (Task 4: MP-DocVQA), download the "
                "IMDBs archive, and extract it here so that imdb_train.npy, "
                "imdb_val.npy, and imdb_test.npy are directly inside."
            ),
        )
        images_dir = require_manual_path(
            data_dir=data_dir,
            expected_path="mpdocvqa/images",
            homepage=_HOMEPAGE,
            instructions=(
                f"Register at {_RRC_PORTAL} (Task 4: MP-DocVQA), download the "
                "Images archive, and extract it here so that each page's "
                ".jpg file is directly inside."
            ),
        )
        return {"imdb": imdb_dir, "images": images_dir}

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            description=(
                "MP-DocVQA: question answering over multi-page scanned "
                "documents, extending SP-DocVQA with surrounding document "
                "pages for context."
            ),
            homepage=_HOMEPAGE,
            license=(
                "Data usage requires registration at the RRC portal and is "
                "subject to its terms of use."
            ),
        )

    def _available_splits(self, data_dir: str) -> list[DatasetSplitType]:
        return list(_SPLIT_NAMES)

    def _build_split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> SplitIterator:
        return SplitIterator(imdb_dir=Path(data_dir) / "mpdocvqa" / "imdb", split=split)

    def _build_input_transform(
        self,
    ) -> Callable[[_IMDBRecord], MultiPageDocumentInstance]:
        return InputTransform(images_dir=Path(self.data_dir) / "mpdocvqa" / "images")
