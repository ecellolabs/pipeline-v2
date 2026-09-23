from pathlib import Path
from unittest.mock import MagicMock

import httpx
import numpy as np
import pytest
from atria_core.datasets import datasets
from atria_core.types import (
    AnnotationType,
    BoundingBoxMode,
    DatasetSplitType,
    MultiPageDocumentInstance,
    MultiPageQuestionAnsweringAnnotation,
    ObjectDetectionAnnotation,
    SinglePageDocumentInstance,
)
from docling_core.types.doc.document import DoclingDocument

from pipeline_v2.datasets.citevqa import (
    CiteVQA,
    CiteVQAConfig,
    InputTransform,
    _EvidenceMeta,
    _RowMeta,
    _Sample,
)
import importlib.util
from pipeline_v2.parsers.docling import DoclingTransform

import sys

_preprocess_path = Path(__file__).resolve().parent.parent / "usage" / "01_preprocess.py"
_spec = importlib.util.spec_from_file_location("preprocess_mod", _preprocess_path)
assert _spec is not None and _spec.loader is not None
preprocess_mod = importlib.util.module_from_spec(_spec)
sys.modules["preprocess_mod"] = preprocess_mod
_spec.loader.exec_module(preprocess_mod)
Preprocessor = preprocess_mod.Preprocessor


def test_citevqa_registration_and_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    assert "citevqa" in datasets.list()
    dataset_cls = datasets.get("citevqa")
    assert dataset_cls is CiteVQA

    monkeypatch.setattr(
        CiteVQA, "_build_split_iterator", lambda self, split, data_dir: []
    )
    dataset = CiteVQA(config=CiteVQAConfig(max_samples=5))
    metadata = dataset._metadata()
    assert "CiteVQA" in metadata.description
    assert dataset._available_splits("/dummy/dir") == [DatasetSplitType.validation]


def test_evidence_and_row_meta_parsing() -> None:
    raw_evidence = [
        {
            "type": "table",
            "content": "<table>...</table>",
            "bbox": [224, 84, 469, 927],
            "source_pdf_name": "sample_doc.pdf",
            "source_page_id": 10,
            "source_doc_index": 1,
            "necessity": "necessary",
        }
    ]
    ev = _EvidenceMeta.from_dict(raw_evidence[0])
    assert ev.type == "table"
    assert ev.bbox == [224.0, 84.0, 469.0, 927.0]
    assert ev.source_pdf_name == "sample_doc.pdf"
    assert ev.source_page_id == 10

    raw_hf_row = {
        "index": "q_123",
        "Question_Type": "Multimodal Parsing",
        "Question": "What is the OD value?",
        "Standard_Answer": "0.075",
        "Evidence": raw_evidence,
        "dataset_type": "Single-Doc",
        "description": "Drug Package Inserts",
        "language": "en",
        "PDF_Source": ["data/pdf/sample_doc.pdf"],
    }
    row_meta = _RowMeta.from_hf_row(raw_hf_row)
    assert row_meta.question_id == "q_123"
    assert row_meta.question_type == "Multimodal Parsing"
    assert row_meta.standard_answer == "0.075"
    assert row_meta.pdf_sources == ["sample_doc.pdf"]
    assert len(row_meta.evidence_list) == 1


def test_input_transform_missing_pdf() -> None:
    transform = InputTransform()
    sample = _Sample(
        doc_id="missing.pdf",
        pdf_path=Path("/nonexistent/path/missing.pdf"),
        qa_metas=[],
    )
    with pytest.raises(
        FileNotFoundError, match="PDF file for sample 'missing.pdf' not found"
    ):
        transform(sample)


def test_input_transform_with_mocked_pdf(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Create a mock PDF file
    dummy_pdf = tmp_path / "test_doc.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4 dummy content")

    # Mock MultiPageDocumentInstance.from_pdf to return 2 dummy pages
    p0 = SinglePageDocumentInstance(
        sample_id="test_doc.pdf#0", visual=MagicMock(), metadata={}
    )
    p1 = SinglePageDocumentInstance(
        sample_id="test_doc.pdf#1", visual=MagicMock(), metadata={}
    )
    mock_doc = MultiPageDocumentInstance(
        sample_id="test_doc.pdf",
        pages=[p0, p1],
        metadata={},
    )
    monkeypatch.setattr(
        MultiPageDocumentInstance,
        "from_pdf",
        lambda path, sample_id: mock_doc,
    )

    qa_meta = _RowMeta(
        question_id="0",
        question_type="Factoid",
        question="What is the test result?",
        standard_answer="Positive",
        evidence_list=[
            _EvidenceMeta(
                type="text",
                content="Result is Positive",
                bbox=[100.0, 50.0, 200.0, 150.0],
                source_pdf_name="test_doc.pdf",
                source_page_id=2,  # 1-indexed page 2 -> 0-indexed page 1
                source_doc_index=1,
                necessity="necessary",
            )
        ],
        dataset_type="Single-Doc",
        description="Medical Report",
        language="en",
        pdf_sources=["test_doc.pdf"],
    )

    sample = _Sample(
        doc_id="test_doc.pdf",
        pdf_path=dummy_pdf,
        qa_metas=[qa_meta],
    )

    transform = InputTransform()
    doc_instance = transform(sample)

    assert doc_instance.sample_id == "test_doc.pdf"
    assert doc_instance.metadata["doc_id"] == "test_doc.pdf"
    assert doc_instance.metadata["language"] == "en"
    assert doc_instance.metadata["domain"] == "Medical Report"
    assert len(doc_instance.pages) == 2

    # Page 0 has no evidence, Page 1 has object detection annotation
    assert not doc_instance.pages[0].has_annotation_type(
        AnnotationType.object_detection
    )
    assert doc_instance.pages[1].has_annotation_type(AnnotationType.object_detection)

    det_ann = doc_instance.pages[1].get_annotation_by_type(
        AnnotationType.object_detection
    )
    assert isinstance(det_ann, ObjectDetectionAnnotation)
    assert det_ann.bbox_mode == BoundingBoxMode.XYXY
    # Check bbox conversion: [ymin, xmin, ymax, xmax] -> [xmin, ymin, xmax, ymax]
    np.testing.assert_allclose(det_ann.bboxes[0], [50.0, 100.0, 150.0, 200.0])

    # QA annotation
    assert doc_instance.has_annotation_type(
        AnnotationType.multi_page_question_answering
    )
    qa_ann = doc_instance.get_annotation_by_type(
        AnnotationType.multi_page_question_answering
    )
    assert isinstance(qa_ann, MultiPageQuestionAnsweringAnnotation)
    assert len(qa_ann.qa_pairs) == 1
    qa_pair = qa_ann.qa_pairs[0]
    assert qa_pair.id == 0
    assert qa_pair.question_text == "What is the test result?"
    assert qa_pair.answer_text == "Positive"
    assert qa_pair.evidence_pages == [1]  # 0-indexed
    assert qa_pair.evidence_sources == ["text"]


from PIL import Image


def test_citevqa_docling_preprocess_run(tmp_path: Path) -> None:
    # Verify that a CiteVQA MultiPageDocumentInstance can be preprocessed by 01_preprocess.Preprocessor
    raw_doc = DoclingDocument(name="test_page").model_dump()
    mock_payload = {
        "document": {
            "json_content": raw_doc,
            "md_content": "# Parsed content",
            "filename": "test.png",
        },
        "status": "success",
        "processing_time": 0.1,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=mock_payload)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    transform = DoclingTransform(api_url="http://mock-docling:10001", client=client)

    mock_page = MagicMock(spec=SinglePageDocumentInstance)
    mock_page.sample_id = "cite_doc#0"
    mock_page.key = "cite_doc_0"
    img = Image.new("RGB", (50, 50), color="white")
    mock_content = MagicMock()
    mock_content.require_content.return_value = img
    mock_page.load.return_value = mock_content

    sample = MultiPageDocumentInstance(
        sample_id="cite_doc", pages=[mock_page], metadata={}
    )

    out_dir = tmp_path / "docling" / "validation"
    preprocessor = Preprocessor(
        out_dir=out_dir,
        api_url="http://mock-docling:10001",
        num_workers=1,
    )
    preprocessor._process_sample(sample, transform=transform)

    saved_file = out_dir / sample.key / f"{mock_page.key}.json"
    assert saved_file.exists()
    assert "test_page" in saved_file.read_text()
