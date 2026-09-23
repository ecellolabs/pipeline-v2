from __future__ import annotations

import csv
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, overload

import numpy as np
from atria_core.datasets import (
    AtriaDownloadManager,
    Dataset,
    DatasetConfig,
    UrlSpec,
    datasets,
)
from atria_core.logger import get_logger
from atria_core.types import (
    AnnotatedObject,
    BoundingBoxMode,
    DatasetMetadata,
    DatasetSplitType,
    MultiPageDocumentInstance,
    MultiPageQAPair,
    MultiPageQuestionAnsweringAnnotation,
    ObjectDetectionAnnotation,
    SinglePageDocumentInstance,
)

_HF_REPO = "opendatalab/CiteVQA"
_PDF_CSV_URL = "https://huggingface.co/datasets/opendatalab/CiteVQA/raw/main/data/download/pdf_source.csv"
_HOMEPAGE = "https://github.com/opendatalab/CiteVQA"
_LICENSE = "MIT"

logger = get_logger(__name__)


@dataclass
class _EvidenceMeta:
    type: str
    content: str
    bbox: list[float]  # [ymin, xmin, ymax, xmax] in 0-1000 scale
    source_pdf_name: str
    source_page_id: int | None  # 1-indexed in CiteVQA, None if unassigned
    source_doc_index: int
    necessity: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _EvidenceMeta:
        raw_bbox = data.get("bbox", [0, 0, 0, 0])
        bbox = [float(v) for v in raw_bbox]
        page_id = data.get("source_page_id")
        return cls(
            type=str(data.get("type", "text")),
            content=str(data.get("content", "")),
            bbox=bbox,
            source_pdf_name=str(data.get("source_pdf_name", "")),
            source_page_id=int(page_id) if page_id is not None else None,
            source_doc_index=int(data.get("source_doc_index", 1)),
            necessity=str(data.get("necessity", "necessary")),
        )


@dataclass
class _RowMeta:
    question_id: str
    question_type: str
    question: str
    standard_answer: str
    evidence_list: list[_EvidenceMeta]
    dataset_type: str
    description: str
    language: str
    pdf_sources: list[str]

    @classmethod
    def from_hf_row(cls, row: dict[str, Any]) -> _RowMeta:
        raw_evidence = row.get("Evidence") or []
        evidence_list = [_EvidenceMeta.from_dict(ev) for ev in raw_evidence]
        raw_sources = row.get("PDF_Source") or []
        pdf_sources = [Path(str(p)).name for p in raw_sources]

        return cls(
            question_id=str(row.get("index", "")),
            question_type=str(row.get("Question_Type", "")),
            question=str(row.get("Question", "")),
            standard_answer=str(row.get("Standard_Answer", "")),
            evidence_list=evidence_list,
            dataset_type=str(row.get("dataset_type", "Single-Doc")),
            description=str(row.get("description", "")),
            language=str(row.get("language", "en")),
            pdf_sources=pdf_sources,
        )


@dataclass
class _Sample:
    doc_id: str
    pdf_path: Path
    qa_metas: list[_RowMeta]


class SplitIterator(Sequence[_Sample]):
    def __init__(
        self,
        data_dir: str,
        split: DatasetSplitType,
        max_samples: int | None = None,
    ) -> None:
        from datasets import load_dataset

        logger.info(f"Loading CiteVQA validation split from Hugging Face ({_HF_REPO})")
        # CiteVQA only provides a validation set on Hugging Face
        hf_split = "validation"
        self._rows = load_dataset(_HF_REPO, split=hf_split, cache_dir=data_dir)

        self._pdf_dir = Path(data_dir) / "pdfs"
        self._pdf_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Indexing CiteVQA rows by document")
        self._row_indices_by_doc = self._index_rows_by_doc()
        self._doc_ids = list(self._row_indices_by_doc)
        if max_samples is not None:
            self._doc_ids = self._doc_ids[:max_samples]

        logger.info(
            f"Downloading / verifying CiteVQA document URLs for {len(self._doc_ids)} docs"
        )
        self._url_by_doc_id = self._load_pdf_sources(Path(data_dir))
        self._download_documents()

    def _index_rows_by_doc(self) -> dict[str, list[int]]:
        row_indices_by_doc: dict[str, list[int]] = {}
        for row_index, row in enumerate(self._rows):
            raw_sources = row.get("PDF_Source") or []
            for p in raw_sources:
                doc_name = Path(str(p)).name
                if doc_name:
                    row_indices_by_doc.setdefault(doc_name, []).append(row_index)
        return row_indices_by_doc

    def _load_pdf_sources(self, data_dir: Path) -> dict[str, str]:
        csv_file = data_dir / "pdf_source.csv"
        manager = AtriaDownloadManager(
            data_dir=self._pdf_dir.parent,
            download_dir=self._pdf_dir.parent / ".download_cache",
        )
        if not csv_file.exists():
            try:
                manager.download_and_extract(
                    data_urls=[
                        UrlSpec(
                            url=_PDF_CSV_URL,
                            url_ext=".csv",
                            rel_output_file_path="pdf_source.csv",
                        )
                    ],
                    extract=False,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Could not download pdf_source.csv: {e}")

        url_map: dict[str, str] = {}
        if csv_file.exists():
            with open(csv_file, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    tid = row.get("track_id", "").strip()
                    url = row.get("url", "").strip()
                    if tid and url:
                        url_map[f"{tid}.pdf"] = url
        return url_map

    def _pdf_path(self, doc_id: str) -> Path:
        return self._pdf_dir / doc_id

    def _download_documents(self) -> None:
        missing_doc_ids = [
            doc_id for doc_id in self._doc_ids if not self._pdf_path(doc_id).exists()
        ]
        if not missing_doc_ids:
            return

        urls_to_download = [
            UrlSpec(
                url=self._url_by_doc_id[doc_id],
                url_ext=".pdf",
                rel_output_file_path=f"pdfs/{doc_id}",
            )
            for doc_id in missing_doc_ids
            if self._url_by_doc_id.get(doc_id)
        ]
        if urls_to_download:
            manager = AtriaDownloadManager(
                data_dir=self._pdf_dir.parent,
                download_dir=self._pdf_dir.parent / ".download_cache",
            )
            for spec in urls_to_download:
                try:
                    manager.download_and_extract([spec], extract=False)
                except Exception as e:  # noqa: BLE001
                    # Fallback to direct request with User-Agent (many external hosts block requests without browser UA)
                    rel_path = spec.rel_output_file_path or ""
                    dest_path = self._pdf_dir / Path(rel_path).name
                    try:
                        import requests

                        resp = requests.get(
                            spec.url,
                            headers={"User-Agent": "Mozilla/5.0"},
                            timeout=60,
                            stream=True,
                        )
                        resp.raise_for_status()
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(dest_path, "wb") as f:
                            for chunk in resp.iter_content(chunk_size=65536):
                                if chunk:
                                    f.write(chunk)
                        logger.info(
                            f"Downloaded {dest_path.name} via direct request fallback"
                        )
                    except Exception as err:  # noqa: BLE001
                        logger.warning(
                            f"Notice: PDF for {spec.rel_output_file_path} could not be downloaded: {err} (Atria error: {e}). "
                            "To obtain the complete PDF archive, use ModelScope: "
                            f"'modelscope download --dataset risemds/CiteVQA_PDF --local_dir {self._pdf_dir}'"
                        )

    @overload
    def __getitem__(self, index: int) -> _Sample: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[_Sample]: ...

    def __getitem__(self, index: int | slice) -> _Sample | Sequence[_Sample]:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]

        doc_id = self._doc_ids[index]
        row_indices = self._row_indices_by_doc[doc_id]
        qa_metas = [
            _RowMeta.from_hf_row(self._rows[row_index]) for row_index in row_indices
        ]
        return _Sample(
            doc_id=doc_id,
            pdf_path=self._pdf_path(doc_id),
            qa_metas=qa_metas,
        )

    def __len__(self) -> int:
        return len(self._doc_ids)


class InputTransform:
    def _bbox_annotation(
        self, bboxes: list[_EvidenceMeta]
    ) -> ObjectDetectionAnnotation:
        converted = [
            AnnotatedObject(
                label_value=idx,
                label_name=ev.type,
                bbox=np.asarray(
                    [ev.bbox[1], ev.bbox[0], ev.bbox[3], ev.bbox[2]],
                    dtype=np.float64,
                ),
            )
            for idx, ev in enumerate(bboxes)
        ]
        return replace(
            ObjectDetectionAnnotation.from_objects(converted),
            bbox_mode=BoundingBoxMode.XYXY,
            normalized=False,
        )

    def __call__(self, sample: _Sample) -> MultiPageDocumentInstance:
        if not sample.pdf_path.exists():
            raise FileNotFoundError(
                f"PDF file for sample '{sample.doc_id}' not found at '{sample.pdf_path}'. "
                "CiteVQA PDFs can be acquired via ModelScope: "
                f"'modelscope download --dataset risemds/CiteVQA_PDF --local_dir {sample.pdf_path.parent}'"
            )

        document = MultiPageDocumentInstance.from_pdf(
            sample.pdf_path,
            sample_id=sample.doc_id,
        )

        bboxes_by_page: dict[int, list[_EvidenceMeta]] = {}
        qa_pairs: list[MultiPageQAPair] = []

        for qa_id, meta in enumerate(sample.qa_metas):
            doc_evidence_pages: list[int] = []
            doc_evidence_sources: list[str] = []

            for ev in meta.evidence_list:
                if (
                    ev.source_pdf_name == sample.doc_id
                    and ev.source_page_id is not None
                ):
                    page_idx = ev.source_page_id - 1
                    if page_idx not in doc_evidence_pages:
                        doc_evidence_pages.append(page_idx)
                    doc_evidence_sources.append(ev.type)
                    bboxes_by_page.setdefault(page_idx, []).append(ev)

            qa_pairs.append(
                MultiPageQAPair(
                    id=qa_id,
                    question_text=meta.question,
                    answer_text=meta.standard_answer,
                    evidence_pages=sorted(doc_evidence_pages),
                    evidence_sources=doc_evidence_sources,
                    answer_format=meta.question_type,
                )
            )

        annotated_pages: list[SinglePageDocumentInstance] = []
        for page_idx, page in enumerate(document.pages):
            page_bboxes = bboxes_by_page.get(page_idx, [])
            if page_bboxes:
                annotated_pages.append(
                    page.add_annotation(annotation=self._bbox_annotation(page_bboxes))
                )
            else:
                annotated_pages.append(page)

        metadata: dict[str, Any] = {
            "doc_id": sample.doc_id,
            "language": sample.qa_metas[0].language if sample.qa_metas else "en",
            "domain": sample.qa_metas[0].description if sample.qa_metas else "",
        }

        return replace(
            document, pages=annotated_pages, metadata=metadata
        ).add_annotation(
            annotation=MultiPageQuestionAnsweringAnnotation(qa_pairs=qa_pairs)
        )


class CiteVQAConfig(DatasetConfig):
    max_samples: int | None = None


@datasets.register
class CiteVQA(Dataset[MultiPageDocumentInstance, CiteVQAConfig]):
    """CiteVQA: Benchmarking Evidence Attribution for Trustworthy Document Intelligence.

    Features 1,897 questions across 711 multi-page PDF documents requiring element-level
    bounding-box evidence citation and evaluated via Strict Attributed Accuracy (SAA).
    """

    __module_name__ = "citevqa"

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            description=(
                "CiteVQA: Benchmarking Evidence Attribution for Trustworthy Document "
                "Intelligence with element-level citation bounding boxes."
            ),
            homepage=_HOMEPAGE,
            license=_LICENSE,
        )

    def _available_splits(self, data_dir: str) -> list[DatasetSplitType]:
        return [DatasetSplitType.validation]

    def _build_split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> SplitIterator:
        return SplitIterator(
            data_dir=data_dir, split=split, max_samples=self.config.max_samples
        )

    def _build_input_transform(self) -> Callable[[_Sample], MultiPageDocumentInstance]:
        return InputTransform()
