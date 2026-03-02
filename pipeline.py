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
import shutil
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from google import genai

from assembly import shorts, slideshow, static_image, stock_video, video
from config_loader import Config
from generators import images, speech, stock_footage, subtitles, text, thumbnail
from quality import checks
from scheduling import compute_publish_schedule
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


class _UploadBudget:
    """Thread-safe YouTube API quota budget tracker.

    The YouTube Data API v3 has a default daily quota of 10,000 units.
    Each upload cycle (videos.insert + thumbnails.set + videos.update)
    costs ~1,700 units, allowing roughly 5 uploads per day.
    """

    COST_PER_UPLOAD = 1700

    def __init__(self, daily_quota: int = 10_000, reserved: int = 200):
        self._lock = threading.Lock()
        self._remaining = daily_quota - reserved

    def has_budget(self) -> bool:
        """Check if there's enough budget for one more upload."""
        with self._lock:
            return self._remaining >= self.COST_PER_UPLOAD

    def try_use(self) -> bool:
        """Try to consume budget for one upload. Returns True if allowed."""
        with self._lock:
            if self._remaining >= self.COST_PER_UPLOAD:
                self._remaining -= self.COST_PER_UPLOAD
                return True
            return False


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
    publish_at = ckpt.get("publish_at")
    return _process_single_video(
        topic, config, client, manual_title=manual_title,
        output_dir_override=output_dir, publish_at=publish_at,
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
        generated = _retry_on_error(
            fn=lambda: text.generate_topics(needed, config, client, existing_titles=existing_titles),
            stage_name="topic_generation",
            config=config,
        )
        topics.extend(generated)
    topics = topics[:config.video_count]

    # Match manual titles to topics (if provided)
    manual_titles = config.titles or []

    # Compute publish schedule (Mon–Sun, 2/day at configured times)
    if not config.dry_run:
        schedule = compute_publish_schedule(
            video_count=len(topics),
            timezone=config.publish_timezone,
            publish_times=config.publish_times,
        )
        for i, dt in enumerate(schedule):
            logger.info(f"  Video {i + 1} scheduled for: {dt}")
    else:
        schedule = [None] * len(topics)

    max_workers = min(config.max_parallel_videos, len(topics))
    logger.info(f"Processing {len(topics)} video(s) with {max_workers} worker(s)")

    # YouTube API quota: 10,000 units/day, ~1,700 per upload ≈ 5 uploads/day.
    # Videos that exceed the budget are generated but upload is deferred
    # to the next --upload-pending run.
    upload_budget = _UploadBudget() if not config.dry_run else None

    if max_workers <= 1:
        # Sequential (original behavior)
        for i, topic in enumerate(topics):
            logger.info(f"\n{'='*60}")
            logger.info(f"Video {i + 1}/{len(topics)}: {topic}")
            logger.info(f"{'='*60}")

            manual_title = manual_titles[i] if i < len(manual_titles) else None
            try:
                result = _process_single_video(
                    topic, config, client, manual_title=manual_title,
                    existing_titles=existing_titles,
                    publish_at=schedule[i],
                    upload_budget=upload_budget,
                )
                results.append(result)
                logger.info(f"Completed: {result.video_url or 'dry-run'}")
            except Exception as e:
                logger.error(f"Failed to process '{topic}': {e}", exc_info=True)
                results.append(
                    VideoResult(
                        topic=topic, title="", video_id=None, video_url=None,
                        success=False, error=str(e),
                    )
                )
    else:
        # Parallel execution
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {}
            for i, topic in enumerate(topics):
                manual_title = manual_titles[i] if i < len(manual_titles) else None
                # Each thread gets its own genai client
                thread_client = genai.Client()
                future = executor.submit(
                    _process_single_video,
                    topic, config, thread_client,
                    manual_title=manual_title,
                    existing_titles=existing_titles,
                    publish_at=schedule[i],
                    upload_budget=upload_budget,
                )
                future_to_idx[future] = (i, topic)

            # Collect results in submission order
            indexed_results: dict[int, VideoResult] = {}
            for future in as_completed(future_to_idx):
                idx, topic = future_to_idx[future]
                try:
                    result = future.result()
                    indexed_results[idx] = result
                    logger.info(f"Completed video {idx + 1}: {result.video_url or 'dry-run'}")
                except Exception as e:
                    logger.error(f"Failed video {idx + 1} '{topic}': {e}", exc_info=True)
                    indexed_results[idx] = VideoResult(
                        topic=topic, title="", video_id=None, video_url=None,
                        success=False, error=str(e),
                    )

            results = [indexed_results[i] for i in range(len(topics))]

    # Write results.json
    results_data = [
        {
            "topic": r.topic,
            "title": r.title,
            "video_id": r.video_id,
            "video_url": r.video_url,
            "success": r.success,
            "error": r.error,
        }
        for r in results
    ]
    results_path = config.output_base_dir / "results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results_data, indent=2), encoding="utf-8")
    logger.info(f"Results written to {results_path}")

    # Summary
    succeeded = sum(1 for r in results if r.success)
    logger.info(f"\nPipeline complete: {succeeded}/{len(results)} videos succeeded")
    for r in results:
        status = "OK" if r.success else "FAILED"
        logger.info(f"  [{status}] {r.topic}: {r.video_url or r.error}")

    return results


def upload_pending(config: Config) -> list[VideoResult]:
    """Upload videos that were generated but not yet uploaded.

    Scans the output directory for videos with completed generation
    (video stage done) but pending upload, and uploads them within
    the daily YouTube API quota (~5 uploads per day).
    """
    output_base = config.output_base_dir
    if not output_base.exists():
        logger.info("No output directory found")
        return []

    # Find videos with completed generation but pending upload
    pending: list[Path] = []
    for video_dir in sorted(output_base.iterdir()):
        if not video_dir.is_dir():
            continue
        ckpt_path = video_dir / CHECKPOINT_FILE
        if not ckpt_path.exists():
            continue
        ckpt = _Checkpoint(video_dir)
        if ckpt.is_done("video") and not ckpt.is_done("upload"):
            pending.append(video_dir)

    if not pending:
        logger.info("No videos pending upload")
        return []

    logger.info(f"Found {len(pending)} video(s) pending upload")

    budget = _UploadBudget()
    client = genai.Client()
    results: list[VideoResult] = []

    for video_dir in pending:
        if not budget.has_budget():
            remaining = len(pending) - len(results)
            logger.info(f"Daily quota budget reached, {remaining} video(s) still pending")
            break

        ckpt = _Checkpoint(video_dir)
        topic = ckpt.get("topic")
        logger.info(f"Uploading: {topic}")

        try:
            result = _process_single_video(
                topic=topic,
                config=config,
                client=client,
                manual_title=ckpt.get("manual_title"),
                output_dir_override=video_dir,
                publish_at=ckpt.get("publish_at"),
                upload_budget=budget,
            )
            results.append(result)
            logger.info(f"Uploaded: {result.video_url or 'deferred'}")
        except QuotaExceededError:
            remaining = len(pending) - len(results)
            logger.info(f"YouTube API quota exceeded, {remaining} video(s) still pending")
            break
        except Exception as e:
            logger.error(f"Failed to upload {video_dir.name}: {e}", exc_info=True)
            results.append(
                VideoResult(
                    topic=topic or str(video_dir.name), title="",
                    video_id=None, video_url=None,
                    success=False, error=str(e),
                )
            )

    uploaded = sum(1 for r in results if r.video_url)
    logger.info(f"Upload batch complete: {uploaded}/{len(results)} uploaded")
    return results


def _process_single_video(
    topic: str,
    config: Config,
    client: genai.Client,
    manual_title: str | None = None,
    output_dir_override: Path | None = None,
    existing_titles: list[str] | None = None,
    publish_at: str | None = None,
    upload_budget: _UploadBudget | None = None,
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
    ckpt.mark_done("_init", topic=topic, manual_title=manual_title, publish_at=publish_at)

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
        description = _ensure_description_footer(description, config)
        _save_artifact(output_dir / "description.txt", description)
        ckpt.mark_done("description")
        logger.info("Description generated")

    # --- Stage 3b: Tags ---
    tags_path = output_dir / "tags.json"
    if ckpt.is_done("tags") and tags_path.exists():
        video_tags = json.loads(tags_path.read_text(encoding="utf-8"))
        logger.info(f"Stage 3b: Loaded {len(video_tags)} cached tags")
    else:
        logger.info("Stage 3b: Generating per-video tags...")
        try:
            video_tags = _retry_on_error(
                fn=lambda: text.generate_tags(topic, title, config, client),
                stage_name="tags",
                config=config,
            )
            _save_artifact(tags_path, json.dumps(video_tags, indent=2))
            ckpt.mark_done("tags")
            logger.info(f"Tags: {len(video_tags)} generated")
        except Exception as e:
            logger.warning(f"Tag generation failed, using defaults: {e}")
            video_tags = []
            ckpt.mark_done("tags")

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

    # --- Stages 5-8 branch by video_mode ---
    if config.video_mode == "static_image":
        image_paths, num_visuals, video_path, thumb_path = _stages_5_to_8_static_image(
            script_text, audio_duration, audio_path, title, topic,
            output_dir, config, client, ckpt, quality_results,
        )
    elif config.video_mode == "stock_footage":
        image_paths, num_visuals, video_path, thumb_path = _stages_5_to_8_stock_footage(
            script_text, audio_duration, audio_path, title, topic,
            output_dir, config, client, ckpt, quality_results,
        )
    else:
        # ken_burns and slideshow modes both use generated images;
        # assembly step selects zoom vs static based on video_mode.
        image_paths, num_visuals, video_path, thumb_path = _stages_5_to_8_ken_burns(
            script_text, audio_duration, audio_path, title, topic,
            output_dir, config, client, ckpt, quality_results,
        )

    # --- Stage 8b: YouTube Short (stock_footage mode only) ---
    short_path = output_dir / "short.mp4"
    srt_path = output_dir / "subtitles.srt"
    if config.video_mode in ("stock_footage", "static_image") and srt_path.exists():
        if ckpt.is_done("short") and short_path.exists():
            logger.info("Stage 8b: Loaded cached Short")
        else:
            logger.info("Stage 8b: Generating YouTube Short...")
            # For static_image mode, pass background image so the short is
            # built from the image directly (avoids double subtitles since
            # the main video already has burned-in subs).
            # Prefer dedicated shorts background if configured.
            bg_image = None
            if config.video_mode == "static_image":
                bg_image = config.shorts_background_image_path or config.background_image_path
            try:
                short_path = shorts.generate_short(
                    video_path, audio_path, srt_path, output_dir, config,
                    background_image=bg_image,
                )
                ckpt.mark_done("short")
                logger.info("Short generated successfully")
            except Exception as e:
                logger.warning(f"Short generation failed (non-fatal): {e}")
                short_path = None
                ckpt.mark_done("short")
    else:
        short_path = None

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
    elif upload_budget and not upload_budget.try_use():
        logger.info("Stage 9: Upload deferred (daily YouTube API quota budget reached)")
    else:
        logger.info("Stage 9: Uploading to YouTube...")
        upload_result = _retry_on_error(
            fn=lambda: youtube.upload_video(
                video_path, title, description, thumb_path, config,
                publish_at=publish_at,
                video_tags=video_tags,
            ),
            stage_name="upload",
            config=config,
        )
        video_id = upload_result.video_id
        video_url = upload_result.video_url
        ckpt.mark_done("upload", video_id=video_id, video_url=video_url)
        logger.info(f"Uploaded: {video_url}")

    # --- Stage 9b: Upload Short ---
    if short_path and short_path.exists():
        if ckpt.is_done("short_upload"):
            logger.info("Stage 9b: Short already uploaded")
        elif config.dry_run:
            logger.info("Stage 9b: Skipping Short upload (dry run)")
            ckpt.mark_done("short_upload")
        elif upload_budget and not upload_budget.try_use():
            logger.info("Stage 9b: Short upload deferred (daily quota budget reached)")
            ckpt.mark_done("short_upload")
        else:
            logger.info("Stage 9b: Uploading YouTube Short...")
            try:
                short_title = title[:90] + " #Shorts"
                short_desc = f"Full story: https://www.youtube.com/watch?v={video_id}\n\n{description}"
                short_result = youtube.upload_video(
                    short_path, short_title, short_desc, thumb_path, config,
                    video_tags=video_tags,
                )
                ckpt.mark_done("short_upload", short_video_id=short_result.video_id)
                logger.info(f"Short uploaded: {short_result.video_url}")
            except Exception as e:
                logger.warning(f"Short upload failed (non-fatal): {e}")
                ckpt.mark_done("short_upload")

    # --- Stage 10: Cleanup ---
    if not config.dry_run and video_url and config.cleanup_after_upload:
        logger.info("Stage 10: Cleaning up large output files...")
        for item in output_dir.iterdir():
            if item.name in ("metadata.json", "checkpoint.json"):
                continue
            if item.is_dir():
                shutil.rmtree(item)
            elif item.suffix in (".mp4", ".wav"):
                item.unlink()
        logger.info("Cleanup complete")

    # --- Save metadata ---
    metadata = {
        "topic": topic,
        "title": title,
        "video_id": video_id,
        "video_url": video_url,
        "audio_duration_seconds": audio_duration,
        "num_visuals": num_visuals,
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
# Video-mode stage helpers
# ---------------------------------------------------------------------------


def _stages_5_to_8_ken_burns(
    script_text, audio_duration, audio_path, title, topic,
    output_dir, config, client, ckpt, quality_results,
) -> tuple[list[Path], int]:
    """Stages 5-8 for ken_burns/slideshow mode: image prompts → images → thumbnail → video."""
    # Stage 5: Image prompts
    prompts_path = output_dir / "image_prompts.json"
    num_images = math.ceil(audio_duration / config.seconds_per_image)

    # Slideshow uses segment-aware prompts for proportional timing
    use_segments = config.video_mode == "slideshow"
    segments_data: list[dict] | None = None

    if ckpt.is_done("image_prompts") and prompts_path.exists():
        cached = json.loads(prompts_path.read_text(encoding="utf-8"))
        # Detect whether cached data has segment info
        if cached and isinstance(cached[0], dict) and "segment" in cached[0]:
            segments_data = cached
            image_prompts = [s["prompt"] for s in cached]
        else:
            image_prompts = cached
        logger.info(f"Stage 5: Loaded {len(image_prompts)} cached image prompts")
    else:
        logger.info("Stage 5: Extracting image prompts...")
        if use_segments:
            segments_data = _retry_on_error(
                fn=lambda: text.extract_image_prompts_with_segments(
                    script_text, num_images, config, client
                ),
                stage_name="image_prompts",
                config=config,
            )
            image_prompts = [s["prompt"] for s in segments_data]
            _save_artifact(
                prompts_path,
                json.dumps(segments_data, indent=2),
            )
        else:
            image_prompts = _retry_on_error(
                fn=lambda: text.extract_image_prompts(
                    script_text, num_images, config, client
                ),
                stage_name="image_prompts",
                config=config,
            )
            _save_artifact(
                prompts_path,
                json.dumps(image_prompts, indent=2),
            )
        ckpt.mark_done("image_prompts")
        logger.info(f"Image prompts: {len(image_prompts)}")

    # Stage 6: Images
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

    # Stage 7: Thumbnail
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
            logger.warning(f"Thumbnail contrast warning: {contrast_check.message}")
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

    # Stage 8: Video assembly
    video_path = output_dir / "video.mp4"
    if ckpt.is_done("video") and video_path.exists():
        logger.info("Stage 8: Loaded cached video")
    elif config.video_mode == "slideshow":
        logger.info("Stage 8: Assembling video (slideshow)...")
        # Calculate per-image durations from audio word timestamps
        clip_durations = None
        if segments_data:
            logger.info("Transcribing audio for transition timing...")
            whisper_words = subtitles.transcribe_words(audio_path)
            clip_durations = slideshow.calculate_durations(
                segments_data, audio_duration, whisper_words
            )
        video_path = slideshow.assemble_video(
            image_paths, audio_path, output_dir, config,
            clip_durations=clip_durations,
        )
        ckpt.mark_done("video")
        logger.info("Video assembled successfully")
    else:
        logger.info("Stage 8: Assembling video (Ken Burns)...")
        video_path = video.assemble_video(
            image_paths, audio_path, output_dir, config
        )
        ckpt.mark_done("video")
        logger.info("Video assembled successfully")

    return image_paths, len(image_paths), video_path, thumb_path


def _stages_5_to_8_static_image(
    script_text, audio_duration, audio_path, title, topic,
    output_dir, config, client, ckpt, quality_results,
) -> tuple[list[Path], int]:
    """Stages 5-8 for static_image mode: subtitles → thumbnail → video."""
    # Stage 5-6: Skipped (no visuals to generate or download)
    logger.info("Stages 5-6: Skipped (static image mode)")

    # Stage 6b: Generate subtitles
    srt_path = output_dir / "subtitles.srt"
    if ckpt.is_done("subtitles") and srt_path.exists():
        logger.info("Stage 6b: Loaded cached subtitles")
    else:
        logger.info("Stage 6b: Generating subtitles...")
        srt_path = subtitles.generate_srt(script_text, audio_duration, srt_path)
        ckpt.mark_done("subtitles")
        logger.info("Subtitles generated")

    # Stage 7: Thumbnail
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
            logger.warning(f"Thumbnail contrast warning: {contrast_check.message}")
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

    # Stage 8: Static image video assembly
    video_path = output_dir / "video.mp4"
    if ckpt.is_done("video") and video_path.exists():
        logger.info("Stage 8: Loaded cached video")
    else:
        logger.info("Stage 8: Assembling static image video...")
        bg_image = config.background_image_path
        if not bg_image or not bg_image.exists():
            raise PipelineError(
                f"Background image not found: {bg_image}. "
                "Set 'background_image' in config.yaml for static_image mode."
            )
        video_path = static_image.assemble_static_image_video(
            bg_image, audio_path, srt_path, output_dir, config
        )
        ckpt.mark_done("video")
        logger.info("Static image video assembled successfully")

    return [], 0, video_path, thumb_path


def _stages_5_to_8_stock_footage(
    script_text, audio_duration, audio_path, title, topic,
    output_dir, config, client, ckpt, quality_results,
) -> tuple[list[Path], int]:
    """Stages 5-8 for stock_footage mode: search queries → clips → subtitles → video."""
    # Stage 5: Stock footage search queries
    queries_path = output_dir / "stock_queries.json"
    num_clips = math.ceil(audio_duration / config.seconds_per_clip)
    if ckpt.is_done("stock_queries") and queries_path.exists():
        search_queries = json.loads(queries_path.read_text(encoding="utf-8"))
        logger.info(f"Stage 5: Loaded {len(search_queries)} cached search queries")
    else:
        logger.info("Stage 5: Generating stock footage search queries...")
        search_queries = _retry_on_error(
            fn=lambda: text.extract_stock_footage_queries(
                script_text, num_clips, config, client
            ),
            stage_name="stock_queries",
            config=config,
        )
        _save_artifact(queries_path, json.dumps(search_queries, indent=2))
        ckpt.mark_done("stock_queries")
        logger.info(f"Search queries: {len(search_queries)}")

    # Stage 6: Download stock footage clips
    if ckpt.is_done("stock_clips"):
        clip_paths = _load_clip_paths(output_dir, len(search_queries))
        logger.info(f"Stage 6: Loaded {len(clip_paths)} cached stock clips")
    else:
        logger.info("Stage 6: Downloading stock footage clips...")
        clip_paths = _retry_on_error(
            fn=lambda: stock_footage.fetch_stock_clips(
                search_queries, output_dir, config
            ),
            stage_name="stock_clips",
            config=config,
        )
        ckpt.mark_done("stock_clips")
        logger.info(f"Downloaded {len(clip_paths)} stock clips")

    # Stage 6b: Generate subtitles
    srt_path = output_dir / "subtitles.srt"
    if ckpt.is_done("subtitles") and srt_path.exists():
        logger.info("Stage 6b: Loaded cached subtitles")
    else:
        logger.info("Stage 6b: Generating subtitles...")
        srt_path = subtitles.generate_srt(script_text, audio_duration, srt_path)
        ckpt.mark_done("subtitles")
        logger.info("Subtitles generated")

    # Stage 7: Thumbnail
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
            logger.warning(f"Thumbnail contrast warning: {contrast_check.message}")
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

    # Stage 8: Stock footage video assembly
    video_path = output_dir / "video.mp4"
    if ckpt.is_done("video") and video_path.exists():
        logger.info("Stage 8: Loaded cached video")
    else:
        logger.info("Stage 8: Assembling stock footage video...")
        video_path = stock_video.assemble_stock_video(
            clip_paths, audio_path, srt_path, output_dir, config
        )
        ckpt.mark_done("video")
        logger.info("Stock footage video assembled successfully")

    return clip_paths, len(clip_paths), video_path, thumb_path


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


def _load_clip_paths(output_dir: Path, expected_count: int) -> list[Path]:
    """Load previously downloaded clip paths from the clips/ subdirectory."""
    clips_dir = output_dir / "clips"
    paths = sorted(clips_dir.glob("*.mp4"))
    if len(paths) < expected_count:
        raise PipelineError(
            f"Expected {expected_count} clips in {clips_dir}, found {len(paths)}"
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


class QuotaExceededError(PipelineError):
    """Raised when the YouTube API daily quota is exhausted."""


def _retry_on_error(fn, stage_name: str, config: Config):
    """Retry a function on exception with exponential backoff."""
    last_error = None
    for attempt in range(config.retry_max_attempts):
        try:
            return fn()
        except Exception as e:
            last_error = e
            # Quota errors won't resolve until midnight PT — fail immediately
            if "quotaExceeded" in str(e):
                raise QuotaExceededError(
                    f"[{stage_name}] YouTube API quota exceeded"
                ) from e
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
    return checks.CheckResult(True, "Description passes all checks")


def _ensure_description_footer(description: str, config: Config) -> str:
    """Ensure the description always contains the disclaimer.

    The LLM sometimes truncates the keyword line and disclaimer.
    This post-processes to guarantee they're present.
    """
    disclaimer = config.disclaimer
    if disclaimer and disclaimer not in description:
        description = description.rstrip() + "\n\n" + disclaimer
    return description


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
