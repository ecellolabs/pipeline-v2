"""Small end-to-end dataset preparation example."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from atria_core.datasets import DatasetBuilder, FileStorageType
from atria_core.logger import get_logger
from atria_core.visualizers import visualize

logger = get_logger(__name__)


def prepare_dataset(
    name: str,
    output_dir: str = "./dataset_visualizations",
    enable_caching: bool = False,
    visualize_samples: bool = True,
    **dataset_kwargs: Any,
) -> None:
    """Load and cache a dataset, then inspect the first sample of each split."""
    dataset = DatasetBuilder().load(name, **dataset_kwargs)
    if enable_caching:
        dataset = dataset.cache(FileStorageType.DELTALAKE, store_images_to_files=True)
    dataset = dataset.build()

    logger.info("Cached dataset:\n%s", dataset)

    for split, split_iterator in dataset.split_iterators.items():
        samples: Any = split_iterator
        sample_count = len(samples)
        if not visualize_samples or sample_count == 0:
            continue

        sample = samples[0].load()
        sample_dir = Path(output_dir) / name / split.value
        sample_dir.mkdir(parents=True, exist_ok=True)
        visualize(sample, output_dir=str(sample_dir))
        logger.info(f"First sample of split `{split}`:\n {sample}")


def main() -> None:
    """Parse a registered dataset name and run its preparation pipeline."""
    import atria_datasets  # type: ignore registers all datasets

    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=sorted(atria_datasets.datasets.list()))
    args = parser.parse_args()
    prepare_dataset(args.name)


if __name__ == "__main__":
    main()
