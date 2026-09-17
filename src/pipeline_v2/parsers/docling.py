"""Docling-based document parsing. Parallelism (multiprocessing, Ray, etc.)
is deliberately not this module's concern -- wrap `DoclingTransform.__call__`
from the outside if you want it."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from atria_core.types import SinglePageDocumentInstance
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, ImageFormatOption

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument


@dataclass
class DoclingTransform:
    """Runs docling's own parsing pipeline (layout analysis + OCR) on page
    images, in memory, and returns docling's own `DoclingDocument` tree --
    no translation into another schema."""

    pipeline_options: PdfPipelineOptions

    _converter: DocumentConverter = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._converter = DocumentConverter(
            format_options={
                InputFormat.IMAGE: ImageFormatOption(
                    pipeline_options=self.pipeline_options
                )
            }
        )

    def __call__(self, page: SinglePageDocumentInstance) -> DoclingDocument:
        image = page.load().require_content()
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        stream = DocumentStream(name=f"{page.key}.png", stream=buffer)
        result = self._converter.convert(stream)
        return result.document
