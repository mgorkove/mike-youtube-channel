"""Core pipeline orchestrator.

Processes each video through all stages sequentially:
topic → title → script → description → voiceover → image prompts →
images → thumbnail → video assembly → upload.

Includes retry logic and quality gates at every step.
"""

import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from google import genai

from assembly import video
from config_loader import Config
from generators import images, speech, text, thumbnail
from quality import checks
from upload import youtube

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Raised when a pipeline stage fails irrecoverably."""


@dataclass
class VideoResult:
    topic: str
    title: str
    video_id: str | None
    video_url: str | None
    success: bool
    error: str | None = None
    quality_results: dict = field(default_factory=dict)


def run(config: Config) -> list[VideoResult]:
    """Execute the full pipeline for all configured topics.

    If topics are not provided in config, generates them via Gemini.
    Processes each video sequentially. Individual failures do not
    stop the batch.
    """
    client = genai.Client()
    results: list[VideoResult] = []

    # Determine topics
    topics = config.topics[:config.video_count] if config.topics else []
    if len(topics) < config.video_count:
        needed = config.video_count - len(topics)
        logger.info(f"Generating {needed} topic(s)...")
        generated = text.generate_topics(needed, config, client)
        topics.extend(generated)
    topics = topics[:config.video_count]

    # Match manual titles to topics (if provided)
    manual_titles = config.titles or []

    logger.info(f"Processing {len(topics)} video(s)")
    for i, topic in enumerate(topics):
        logger.info(f"\n{'='*60}")
        logger.info(f"Video {i + 1}/{len(topics)}: {topic}")
        logger.info(f"{'='*60}")

        manual_title = manual_titles[i] if i < len(manual_titles) else None
        try:
            result = _process_single_video(topic, config, client, manual_title=manual_title)
            results.append(result)
            logger.info(f"Completed: {result.video_url or 'dry-run'}")
        except Exception as e:
            logger.error(f"Failed to process '{topic}': {e}", exc_info=True)
            results.append(
                VideoResult(
                    topic=topic,
                    title="",
                    video_id=None,
                    video_url=None,
                    success=False,
                    error=str(e),
                )
            )

    # Summary
    succeeded = sum(1 for r in results if r.success)
    logger.info(f"\nPipeline complete: {succeeded}/{len(results)} videos succeeded")
    for r in results:
        status = "OK" if r.success else "FAILED"
        logger.info(f"  [{status}] {r.topic}: {r.video_url or r.error}")

    return results


def _process_single_video(
    topic: str,
    config: Config,
    client: genai.Client,
    manual_title: str | None = None,
) -> VideoResult:
    """Process a single video through all pipeline stages."""
    slug = _slugify(topic)
    output_dir = config.output_base_dir / slug
    output_dir.mkdir(parents=True, exist_ok=True)
    quality_results: dict[str, checks.CheckResult] = {}

    # --- Stage 1: Title ---
    if manual_title:
        logger.info(f"Stage 1: Using manual title: {manual_title}")
        title = manual_title
    else:
        logger.info("Stage 1: Generating title...")
        title = _retry_with_check(
            generate_fn=lambda: text.generate_title(topic, config, client),
            check_fn=lambda t: checks.check_title_length(t),
            stage_name="title",
            config=config,
        )
    _save_artifact(output_dir / "title.txt", title)
    logger.info(f"Title: {title}")

    # --- Stage 2: Script ---
    logger.info("Stage 2: Generating script...")
    script_text = _retry_with_check(
        generate_fn=lambda: text.generate_script(topic, config, client),
        check_fn=lambda s: _check_script(s, config),
        stage_name="script",
        config=config,
    )
    _save_artifact(output_dir / "script.txt", script_text)
    word_count = len(script_text.split())
    logger.info(f"Script: {word_count} words")

    # --- Stage 3: Description ---
    logger.info("Stage 3: Generating description...")
    description = _retry_with_check(
        generate_fn=lambda: text.generate_description(
            topic, title, script_text, config, client
        ),
        check_fn=lambda d: _check_description(d, config),
        stage_name="description",
        config=config,
    )
    _save_artifact(output_dir / "description.txt", description)
    logger.info("Description generated")

    # --- Stage 4: Voiceover ---
    logger.info("Stage 4: Generating voiceover...")
    tts_result = _retry_on_error(
        fn=lambda: speech.generate_voiceover(script_text, output_dir, config, client),
        stage_name="tts",
        config=config,
    )
    audio_check = checks.check_audio_file(tts_result.audio_path, word_count)
    quality_results["audio"] = audio_check
    if not audio_check.passed:
        logger.warning(f"Audio check warning: {audio_check.message}")
    logger.info(f"Voiceover: {tts_result.duration_seconds:.1f}s")

    # --- Stage 5: Image prompts ---
    logger.info("Stage 5: Extracting image prompts...")
    num_images = math.ceil(tts_result.duration_seconds / config.seconds_per_image)
    image_prompts = _retry_on_error(
        fn=lambda: text.extract_image_prompts(
            script_text, num_images, config, client
        ),
        stage_name="image_prompts",
        config=config,
    )
    logger.info(f"Image prompts: {len(image_prompts)}")

    # --- Stage 6: Images ---
    logger.info("Stage 6: Generating images...")
    image_paths = _retry_on_error(
        fn=lambda: images.generate_images(
            image_prompts, output_dir, config, client
        ),
        stage_name="image_generation",
        config=config,
    )
    for img_path in image_paths:
        img_check = checks.check_image_exists_and_dimensions(
            img_path, config.image_width, config.image_height
        )
        quality_results[f"image_{img_path.name}"] = img_check
        if not img_check.passed:
            raise PipelineError(f"Image check failed: {img_check.message}")
    logger.info(f"Generated {len(image_paths)} images")

    # --- Stage 7: Thumbnail ---
    logger.info("Stage 7: Generating thumbnail...")
    thumb_path = _retry_on_error(
        fn=lambda: thumbnail.generate_thumbnail(
            title, topic, output_dir, config, client
        ),
        stage_name="thumbnail",
        config=config,
    )
    contrast_check = checks.check_contrast_ratio(
        thumb_path, config.thumbnail_min_contrast
    )
    quality_results["thumbnail_contrast"] = contrast_check
    if not contrast_check.passed:
        logger.warning(f"Thumbnail contrast warning: {contrast_check.message}")
        # Try regenerating once with adjusted prompt
        logger.info("Regenerating thumbnail for better contrast...")
        thumb_path = _retry_on_error(
            fn=lambda: thumbnail.generate_thumbnail(
                title, topic, output_dir, config, client
            ),
            stage_name="thumbnail_retry",
            config=config,
        )
        contrast_check = checks.check_contrast_ratio(
            thumb_path, config.thumbnail_min_contrast
        )
        quality_results["thumbnail_contrast_retry"] = contrast_check
    logger.info(f"Thumbnail: contrast ratio {contrast_check.message}")

    # --- Stage 8: Video assembly ---
    logger.info("Stage 8: Assembling video...")
    video_path = video.assemble_video(
        image_paths, tts_result.audio_path, output_dir, config
    )
    video_check = checks.check_video_file(video_path, tts_result.duration_seconds)
    quality_results["video"] = video_check
    if not video_check.passed:
        raise PipelineError(f"Video check failed: {video_check.message}")
    logger.info("Video assembled successfully")

    # --- Stage 9: Upload ---
    video_id = None
    video_url = None
    if config.dry_run:
        logger.info("Stage 9: Skipping upload (dry run)")
    else:
        logger.info("Stage 9: Uploading to YouTube...")
        upload_result = _retry_on_error(
            fn=lambda: youtube.upload_video(
                video_path, title, description, thumb_path, config
            ),
            stage_name="upload",
            config=config,
        )
        video_id = upload_result.video_id
        video_url = upload_result.video_url
        logger.info(f"Uploaded: {video_url}")

    # --- Stage 10: Save metadata ---
    metadata = {
        "topic": topic,
        "title": title,
        "video_id": video_id,
        "video_url": video_url,
        "audio_duration_seconds": tts_result.duration_seconds,
        "num_images": len(image_paths),
        "word_count": word_count,
        "quality_checks": {
            k: {"passed": v.passed, "message": v.message}
            for k, v in quality_results.items()
        },
    }
    _save_artifact(output_dir / "metadata.json", json.dumps(metadata, indent=2))

    return VideoResult(
        topic=topic,
        title=title,
        video_id=video_id,
        video_url=video_url,
        success=True,
        quality_results={k: v.passed for k, v in quality_results.items()},
    )


# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------


def _retry_with_check(generate_fn, check_fn, stage_name: str, config: Config):
    """Retry generation until quality check passes or max attempts exceeded."""
    for attempt in range(config.retry_max_attempts):
        result = generate_fn()
        check = check_fn(result)
        if check.passed:
            return result
        logger.warning(
            f"[{stage_name}] Attempt {attempt + 1}/{config.retry_max_attempts}: "
            f"check failed — {check.message}"
        )
        if attempt < config.retry_max_attempts - 1:
            delay = min(
                config.retry_base_delay * (2**attempt),
                config.retry_max_delay,
            )
            time.sleep(delay)
    raise PipelineError(
        f"[{stage_name}] Failed quality checks after {config.retry_max_attempts} attempts"
    )


def _retry_on_error(fn, stage_name: str, config: Config):
    """Retry a function on exception with exponential backoff."""
    last_error = None
    for attempt in range(config.retry_max_attempts):
        try:
            return fn()
        except Exception as e:
            last_error = e
            logger.warning(
                f"[{stage_name}] Attempt {attempt + 1}/{config.retry_max_attempts} "
                f"error: {e}"
            )
            if attempt < config.retry_max_attempts - 1:
                delay = min(
                    config.retry_base_delay * (2**attempt),
                    config.retry_max_delay,
                )
                time.sleep(delay)
    raise PipelineError(
        f"[{stage_name}] Failed after {config.retry_max_attempts} attempts: {last_error}"
    ) from last_error


# ---------------------------------------------------------------------------
# Quality check composites
# ---------------------------------------------------------------------------


def _check_script(script_text: str, config: Config) -> checks.CheckResult:
    """Run all script quality checks, return first failure or overall pass."""
    wc = checks.check_word_count(
        script_text, config.script_min_words, config.script_max_words
    )
    if not wc.passed:
        return wc
    bp = checks.check_banned_phrases(script_text, config.banned_phrases)
    if not bp.passed:
        return bp
    return checks.CheckResult(True, "Script passes all checks")


def _check_description(description: str, config: Config) -> checks.CheckResult:
    """Run all description quality checks."""
    kw = checks.check_keywords_present(description, config.required_keywords)
    if not kw.passed:
        return kw
    dc = checks.check_disclaimer_present(description, config.disclaimer)
    if not dc.passed:
        return dc
    return checks.CheckResult(True, "Description passes all checks")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _slugify(text: str) -> str:
    """Convert text to a filesystem-safe slug."""
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    return slug[:80]


def _save_artifact(path: Path, content: str) -> None:
    """Save a text artifact to disk."""
    path.write_text(content, encoding="utf-8")
