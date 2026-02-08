#!/usr/bin/env python3
"""YouTube video generation pipeline — CLI entry point.

Usage:
    # Generate 3 videos with auto-generated topics (dry run, no upload):
    python main.py --count 3 --dry-run

    # Generate 1 video with a specific topic:
    python main.py --count 1 --topics "Why Banks Treat You Differently After \$100K"

    # Full run with upload:
    python main.py --count 5

    # Use a custom config file:
    python main.py --config my_config.yaml --count 2
"""

import argparse
import dataclasses
import logging
import sys

from config_loader import load_config
from pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and upload YouTube videos about finance/banking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--topics",
        type=str,
        default=None,
        help="Comma-separated list of topics (overrides config file topics)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of videos to generate (overrides config file video_count)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the full pipeline but skip uploading to YouTube",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("pipeline.log"),
        ],
    )

    # Load configuration
    config = load_config(args.config)

    # Apply CLI overrides
    overrides: dict = {}
    if args.topics:
        overrides["topics"] = [t.strip() for t in args.topics.split(",")]
    if args.count is not None:
        overrides["video_count"] = args.count
    if args.dry_run:
        overrides["dry_run"] = True

    if overrides:
        config = dataclasses.replace(config, **overrides)

    logging.getLogger(__name__).info(
        f"Starting pipeline: {config.video_count} video(s), "
        f"dry_run={config.dry_run}"
    )

    # Run the pipeline
    results = run(config)

    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE RESULTS")
    print("=" * 60)
    for r in results:
        status = "OK" if r.success else "FAILED"
        detail = r.video_url or r.error or "dry-run"
        print(f"  [{status}] {r.topic}")
        print(f"          Title: {r.title}")
        print(f"          {detail}")
    print("=" * 60)
    succeeded = sum(1 for r in results if r.success)
    print(f"Total: {succeeded}/{len(results)} succeeded")


if __name__ == "__main__":
    main()
