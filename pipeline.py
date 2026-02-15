"""Core pipeline orchestrator.

Processes each video through all stages sequentially:
topic → title → script → description → voiceover → image prompts →
images → thumbnail → video assembly → upload.

Includes retry logic, quality gates, and checkpoint/resume support.
After each stage, a checkpoint is saved so the pipeline can be resumed
from the point of failure without re-doing completed work.
"""

import json
import logging
import math
import re
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

from google import genai

from assembly import video
from config_loader import Config
from generators import images, speech, text, thumbnail
from quality import checks
from upload import youtube

logger = logging.getLogger(__name__)

CHECKPOINT_FILE = "checkpoint.json"


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


class _Checkpoint:
    """Tracks completed pipeline stages for a single video.

    Persists to ``checkpoint.json`` inside the video output directory so the
    pipeline can resume from the last successful stage.
    """

    def __init__(self, output_dir: Path) -> None:
        self._path = output_dir / CHECKPOINT_FILE
        self._data: dict = self._load()

    def _load(self) -> dict:
        if self._path.exists():
            return json.loads(self._path.read_text(encoding="utf-8"))
        return {"completed_stages": [], "data": {}}

    def _save(self) -> None:
        self._path.write_text(
            json.dumps(self._data, indent=2, default=str), encoding="utf-8"
        )

    def is_done(self, stage: str) -> bool:
        return stage in self._data["completed_stages"]

    def mark_done(self, stage: str, **kv: object) -> None:
        """Mark *stage* complete and persist any associated key-value data."""
        if stage not in self._data["completed_stages"]:
            self._data["completed_stages"].append(stage)
        self._data["data"].update(kv)
        self._save()

    def get(self, key: str, default: object = None) -> object:
        return self._data["data"].get(key, default)


def resume(output_dir: str | Path, config: Config) -> VideoResult:
    """Resume a single video from a previous output directory.

    Reads the checkpoint to determine which stages are already done,
    then continues from the first incomplete stage.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    ckpt = _Checkpoint(output_dir)
    topic = ckpt.get("topic")
    if not topic:
        raise PipelineError(
            f"No checkpoint found in {output_dir}. "
            "Cannot resume without a previous run."
        )

    client = genai.Client()
    manual_title = ckpt.get("manual_title")
    return _process_single_video(
        topic, config, client, manual_title=manual_title, output_dir_override=output_dir
    )


def run(config: Config) -> list[VideoResult]:
    """Execute the full pipeline for all configured topics.

    If topics are not provided in config, generates them via Gemini.
    Processes each video sequentially. Individual failures do not
    stop the batch.
    """
    client = genai.Client()
    results: list[VideoResult] = []

    # Fetch existing channel titles for deduplication
    existing_titles: list[str] = []
    if not config.dry_run:
        existing_titles = youtube.fetch_existing_titles(config)
    else:
        logger.info("Dry run: skipping YouTube title fetch for deduplication")

    # Determine topics
    topics = config.topics[:config.video_count] if config.topics else []
    if len(topics) < config.video_count:
        needed = config.video_count - len(topics)
        logger.info(f"Generating {needed} topic(s)...")
        generated = text.generate_topics(needed, config, client, existing_titles=existing_titles)
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
            result = _process_single_video(
                topic, config, client, manual_title=manual_title,
                existing_titles=existing_titles,
            )
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
    output_dir_override: Path | None = None,
    existing_titles: list[str] | None = None,
) -> VideoResult:
    """Process a single video through all pipeline stages.

    Uses a checkpoint file to track progress. On resume, completed stages
    are skipped and their artifacts are loaded from disk.
    """
    slug = _slugify(topic)
    output_dir = output_dir_override or (config.output_base_dir / slug)
    output_dir.mkdir(parents=True, exist_ok=True)
    quality_results: dict[str, checks.CheckResult] = {}

    ckpt = _Checkpoint(output_dir)
    # Persist identity info so resume() can reload it later.
    ckpt.mark_done("_init", topic=topic, manual_title=manual_title)

    # --- Stage 1: Title ---
    if ckpt.is_done("title"):
        title = (output_dir / "title.txt").read_text(encoding="utf-8")
        logger.info(f"Stage 1: Loaded cached title: {title}")
    else:
        if manual_title:
            logger.info(f"Stage 1: Using manual title: {manual_title}")
            title = manual_title
        else:
            logger.info("Stage 1: Generating title...")
            title = _retry_with_check(
                generate_fn=lambda: text.generate_title(topic, config, client, existing_titles=existing_titles),
                check_fn=lambda t: checks.check_title_length(t),
                stage_name="title",
                config=config,
            )
        _save_artifact(output_dir / "title.txt", title)
        ckpt.mark_done("title")
        logger.info(f"Title: {title}")

    # --- Stage 2: Script ---
    if ckpt.is_done("script"):
        script_text = (output_dir / "script.txt").read_text(encoding="utf-8")
        word_count = len(script_text.split())
        logger.info(f"Stage 2: Loaded cached script ({word_count} words)")
    else:
        logger.info("Stage 2: Generating script...")
        script_text = _retry_with_check(
            generate_fn=lambda: text.generate_script(topic, config, client),
            check_fn=lambda s: _check_script(s, config),
            stage_name="script",
            config=config,
        )
        _save_artifact(output_dir / "script.txt", script_text)
        word_count = len(script_text.split())
        ckpt.mark_done("script")
        logger.info(f"Script: {word_count} words")

    # --- Stage 3: Description ---
    if ckpt.is_done("description"):
        description = (output_dir / "description.txt").read_text(encoding="utf-8")
        logger.info("Stage 3: Loaded cached description")
    else:
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
        ckpt.mark_done("description")
        logger.info("Description generated")

    # --- Stage 4: Voiceover ---
    audio_path = output_dir / "audio.wav"
    if ckpt.is_done("voiceover") and audio_path.exists():
        audio_duration = _wav_duration(audio_path)
        word_count = len(script_text.split())
        logger.info(
            f"Stage 4: Loaded cached voiceover ({audio_duration:.1f}s)"
        )
    else:
        logger.info("Stage 4: Generating voiceover...")
        tts_result = _retry_on_error(
            fn=lambda: speech.generate_voiceover(
                script_text, output_dir, config, client
            ),
            stage_name="tts",
            config=config,
        )
        audio_path = tts_result.audio_path
        audio_duration = tts_result.duration_seconds
        word_count = len(script_text.split())
        audio_check = checks.check_audio_file(audio_path, word_count)
        quality_results["audio"] = audio_check
        if not audio_check.passed:
            logger.warning(f"Audio check warning: {audio_check.message}")
        ckpt.mark_done("voiceover", audio_duration=audio_duration)
        logger.info(f"Voiceover: {audio_duration:.1f}s")

    # --- Stage 5: Image prompts ---
    prompts_path = output_dir / "image_prompts.json"
    num_images = math.ceil(audio_duration / config.seconds_per_image)
    if ckpt.is_done("image_prompts") and prompts_path.exists():
        image_prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
        logger.info(
            f"Stage 5: Loaded {len(image_prompts)} cached image prompts"
        )
    else:
        logger.info("Stage 5: Extracting image prompts...")
        image_prompts = _retry_on_error(
            fn=lambda: text.extract_image_prompts(
                script_text, num_images, config, client
            ),
            stage_name="image_prompts",
            config=config,
        )
        _save_artifact(
            output_dir / "image_prompts.json",
            json.dumps(image_prompts, indent=2),
        )
        ckpt.mark_done("image_prompts")
        logger.info(f"Image prompts: {len(image_prompts)}")

    # --- Stage 6: Images ---
    if ckpt.is_done("images"):
        image_paths = _load_image_paths(output_dir, len(image_prompts))
        logger.info(f"Stage 6: Loaded {len(image_paths)} cached images")
    else:
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
        ckpt.mark_done("images")
        logger.info(f"Generated {len(image_paths)} images")

    # --- Stage 7: Thumbnail ---
    thumb_path = output_dir / "thumbnail.png"
    if ckpt.is_done("thumbnail") and thumb_path.exists():
        logger.info("Stage 7: Loaded cached thumbnail")
    else:
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
            logger.warning(
                f"Thumbnail contrast warning: {contrast_check.message}"
            )
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
        ckpt.mark_done("thumbnail")

    # --- Stage 8: Video assembly ---
    video_path = output_dir / "video.mp4"
    if ckpt.is_done("video") and video_path.exists():
        logger.info("Stage 8: Loaded cached video")
    else:
        logger.info("Stage 8: Assembling video...")
        video_path = video.assemble_video(
            image_paths, audio_path, output_dir, config
        )
        ckpt.mark_done("video")
        logger.info("Video assembled successfully")

    # --- Stage 9: Upload ---
    video_id = None
    video_url = None
    if ckpt.is_done("upload"):
        video_id = ckpt.get("video_id")
        video_url = ckpt.get("video_url")
        logger.info(f"Stage 9: Already uploaded: {video_url}")
    elif config.dry_run:
        logger.info("Stage 9: Skipping upload (dry run)")
        ckpt.mark_done("upload", video_id=None, video_url=None)
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
        ckpt.mark_done("upload", video_id=video_id, video_url=video_url)
        logger.info(f"Uploaded: {video_url}")

    # --- Save metadata ---
    metadata = {
        "topic": topic,
        "title": title,
        "video_id": video_id,
        "video_url": video_url,
        "audio_duration_seconds": audio_duration,
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
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _wav_duration(path: Path) -> float:
    """Return the duration (seconds) of a WAV file."""
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def _load_image_paths(output_dir: Path, expected_count: int) -> list[Path]:
    """Load previously generated image paths from the images/ subdirectory."""
    images_dir = output_dir / "images"
    paths = sorted(images_dir.glob("*.png"))
    if len(paths) < expected_count:
        raise PipelineError(
            f"Expected {expected_count} images in {images_dir}, found {len(paths)}"
        )
    return paths[:expected_count]


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
