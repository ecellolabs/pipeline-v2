"""CLI entrypoint to run Vision-Language model baseline evaluation across datasets via API."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, cast

from atria_core.datasets import Dataset, DatasetBuilder, DatasetConfig
from atria_core.logger import get_logger
from atria_core.types import DatasetSplitType, MultiPageDocumentInstance

from pipeline_v2.datasets import *
from pipeline_v2.evaluation import evaluate_dataset
from pipeline_v2.evaluators import get_evaluator
from pipeline_v2.models.qwen import (
    DEFAULT_QWEN_API_URL,
    DEFAULT_QWEN_MODEL_ID,
    QwenVLConfig,
    QwenVLModel,
)
from pipeline_v2.parsers.docling import DoclingTransform

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Qwen Vision-Language baseline evaluation on document datasets via API."
    )
    parser.add_argument(
        "name",
        nargs="?",
        default="citevqa",
        help="Registered dataset name (default: 'citevqa'). Options: citevqa, slidevqa, mmlongbench_doc.",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help=(
            "OpenAI-compatible vLLM / Workers AI server URL (e.g. 'https://model.ecello.net/v1'). "
            "Defaults to $QWEN_API_URL or $VLM_BASE_URL or "
            f"'{DEFAULT_QWEN_API_URL}'."
        ),
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help=(
            "Model repo ID or served name (e.g. '@cf/qwen/qwen3.8-27b' or 'Qwen/Qwen2.5-VL-7B-Instruct'). "
            f"Defaults to $QWEN_MODEL_ID or $VLM_MODEL or '{DEFAULT_QWEN_MODEL_ID}'."
        ),
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API authorization key if required by endpoint (e.g. Bearer token).",
    )
    parser.add_argument(
        "--docling-url",
        type=str,
        default=os.getenv("DOCLING_API_URL"),
        help="Docling API service URL (e.g. 'http://serv-3334:10001').",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Qwen thinking/reasoning tokens (default: False).",
    )
    parser.add_argument(
        "--split",
        choices=[s.value for s in DatasetSplitType],
        default="validation",
        help="Dataset split to evaluate (default: 'validation').",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples/decks to evaluate (ideal for fast pilot runs).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate per question (default: 128).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-request HTTP timeout in seconds (default: 120.0).",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Path to save evaluation summary and predictions JSON.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock inference for dry-run verification without network/server calls.",
    )
    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = Path(f"./evaluation_results/{args.name}_qwen_eval.json")

    # Build model configuration
    config_kwargs: dict[str, Any] = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "timeout": args.timeout,
        "enable_thinking": args.enable_thinking,
        "mock": args.mock,
    }
    if args.api_url:
        config_kwargs["api_url"] = args.api_url
    if args.model_id:
        config_kwargs["model_id"] = args.model_id
    if args.api_key:
        config_kwargs["api_key"] = args.api_key

    config = QwenVLConfig(**config_kwargs)
    model = QwenVLModel(config=config)

    # Pre-flight health check on API endpoint
    if not args.mock:
        logger.info(
            f"Testing connection to Qwen Vision API at {config.api_url} (model: {config.model_id})..."
        )
        ok, conn_msg = model.check_connection()
        if not ok:
            print("\n" + "!" * 80, file=sys.stderr)
            print(f"[PREFLIGHT FAILED] {conn_msg}", file=sys.stderr)
            print("Please check your API key and base URL using:", file=sys.stderr)
            print(
                "  --api-url https://model.ecello.net/v1 --api-key <YOUR_API_KEY>",
                file=sys.stderr,
            )
            print("!" * 80 + "\n", file=sys.stderr)
            sys.exit(1)
        logger.info(f"[PREFLIGHT OK] {conn_msg}")

    # Optional Docling transform initialization
    docling_transform: DoclingTransform | None = None
    if args.docling_url:
        try:
            docling_transform = DoclingTransform(api_url=args.docling_url)
            logger.info(f"Initialized DoclingTransform at {args.docling_url}")
        except Exception as err:
            logger.warning(
                f"Could not initialize DoclingTransform at {args.docling_url}: {err}. Falling back to standard visual QA."
            )

    split = DatasetSplitType(args.split) if args.split is not None else None
    logger.info(
        f"Loading dataset {args.name!r} (split={args.split}, max_samples={args.max_samples})"
    )
    dataset = cast(
        Dataset[MultiPageDocumentInstance, DatasetConfig],
        DatasetBuilder()
        .load(args.name, split=split, max_samples=args.max_samples)
        .build(),
    )

    evaluate_dataset(
        dataset=dataset,
        model=model,
        output_file=args.output_file,
        docling_transform=docling_transform,
    )


if __name__ == "__main__":
    main()
