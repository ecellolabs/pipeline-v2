"""Runs DoclingTransform over a dataset, writing each page's parsed output
next to the dataset's own data."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from atria_core.datasets import Dataset, DatasetBuilder, DatasetConfig
from atria_core.logger import get_logger
from atria_core.types import MultiPageDocumentInstance
from docling.datamodel.pipeline_options import PdfPipelineOptions

from agentic.parsers.docling import DoclingTransform

logger = get_logger(__name__)


@dataclass
class Preprocessor:
    """Parses every page of every sample in a dataset split, writing each
    page's docling output to `data_dir/docling/{sample.key}/{page.key}.json`.
    Skips pages whose output already exists, so a re-run resumes rather than
    reprocessing."""

    out_dir: Path
    pipeline_options: PdfPipelineOptions
    num_workers: int = 1

    def _process_sample(self, sample: MultiPageDocumentInstance) -> None:
        logger.info(f"sample_id={sample.sample_id!r} key={sample.key!r}")
        transform = DoclingTransform(pipeline_options=self.pipeline_options)
        sample_dir = self.out_dir / sample.key
        sample_dir.mkdir(parents=True, exist_ok=True)
        for page in sample.pages:
            out_path = sample_dir / f"{page.key}.json"
            if out_path.exists():
                logger.info(f"skip page_id={page.sample_id!r} -> {out_path}")
                continue
            document = transform(page)
            out_path.write_text(json.dumps(document.export_to_dict()))
            logger.info(f"page_id={page.sample_id!r} key={page.key!r} -> {out_path}")

    def run(self, split_iterator: Iterable[MultiPageDocumentInstance]) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)

        if self.num_workers <= 1:
            for sample in split_iterator:
                self._process_sample(sample)
            return

        with mp.Pool(self.num_workers) as pool:
            for _ in pool.imap_unordered(self._process_sample, split_iterator):
                pass


def main() -> None:
    from agentic.datasets import *  # type: ignore registers all datasets

    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()

    dataset = cast(
        Dataset[MultiPageDocumentInstance, DatasetConfig],
        DatasetBuilder().load(args.name).build(),
    )
    pipeline_options = PdfPipelineOptions(do_ocr=True)
    for split, split_iterator in dataset.split_iterators.items():
        preprocessor = Preprocessor(
            out_dir=dataset.data_dir / "docling" / split.value,
            pipeline_options=pipeline_options,
            num_workers=args.num_workers,
        )
        preprocessor.run(split_iterator)


if __name__ == "__main__":
    main()
