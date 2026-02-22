#!/usr/bin/env python3
"""YouTube video generation pipeline — CLI entry point.

Usage:
    # Generate 3 videos for the default channel (dry run, no upload):
    python main.py --count 3 --dry-run

    # Generate videos for a specific channel:
    python main.py --channel heartbreak_chronicles --count 5

    # Generate 1 video with a specific topic:
    python main.py --count 1 --topics "Why Banks Treat You Differently After $100K"

    # Specify manual titles (matched 1:1 with topics):
    python main.py --count 2 \\
        --topics "Topic one,Topic two" \\
        --titles "Custom Title One,Custom Title Two"

    # Set target video length (in seconds):
    python main.py --count 1 --video-length 600

    # Full run with upload:
    python main.py --count 5

    # Use a custom config file (bypasses --channel):
    python main.py --config my_config.yaml --count 2

    # Resume a failed run from its output directory:
    python main.py --resume output/why-banks-treat-you-differently

    # Resume in dry-run mode (skip upload even if it wasn't done yet):
    python main.py --resume output/why-banks-treat-you-differently --dry-run
"""

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

from config_loader import load_config
from pipeline import resume, run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and upload YouTube videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--channel",
        default="mike_explains_money",
        help="Channel name — loads config from channels/<name>/config.yaml "
        "(default: mike_explains_money)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a YAML config file (overrides --channel)",
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
        "--titles",
        type=str,
        default=None,
        help="Comma-separated list of manual titles (matched 1:1 with topics). "
        "Skips title generation for provided titles.",
    )
    parser.add_argument(
        "--video-length",
        type=int,
        default=None,
        help="Target video length in seconds (adjusts script word count). "
        "e.g., 600 for ~10 minutes, 1800 for ~30 minutes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the full pipeline but skip uploading to YouTube",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="OUTPUT_DIR",
        help="Resume a previous run from the given output directory. "
        "Skips stages that already completed successfully.",
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

    # Resolve config path: --config overrides --channel
    if args.config:
        config_path = args.config
    else:
        config_path = str(Path("channels") / args.channel / "config.yaml")

    config = load_config(config_path)

    # Apply CLI overrides
    overrides: dict = {}
    if args.topics:
        overrides["topics"] = [t.strip() for t in args.topics.split(",")]
    if args.titles:
        overrides["titles"] = [t.strip() for t in args.titles.split(",")]
    if args.count is not None:
        overrides["video_count"] = args.count
    if args.video_length is not None:
        overrides["target_video_length"] = args.video_length
        # Adjust word count range based on target length
        # ~2.5 words/sec speaking rate, with ±15% tolerance
        target_words = int(args.video_length * 2.5)
        overrides["script_min_words"] = int(target_words * 0.85)
        overrides["script_max_words"] = int(target_words * 1.15)
    if args.dry_run:
        overrides["dry_run"] = True

    if overrides:
        config = dataclasses.replace(config, **overrides)

    log = logging.getLogger(__name__)
    log.info(f"Channel: {args.channel} | Mode: {config.video_mode}")

    # Resume mode: pick up a single video from its output directory
    if args.resume:
        log.info(f"Resuming pipeline from: {args.resume}")
        result = resume(args.resume, config)
        results = [result]
    else:
        log.info(
            f"Starting pipeline: {config.video_count} video(s), "
            f"dry_run={config.dry_run}"
        )
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

    if succeeded < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
