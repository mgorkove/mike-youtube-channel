"""Video assembly with static images (slideshow mode).

Each image is displayed for a duration derived from its narration segment's
word count, so transitions align with natural breaks in the script.  Falls
back to equal durations when no per-image timing is supplied.

Uses ffmpeg's -loop flag for fast per-image clip rendering, then concatenates
all clips with the audio track.
"""

import logging
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from assembly.video import _get_audio_duration
from config_loader import Config

logger = logging.getLogger(__name__)

DEFAULT_RENDER_WORKERS = 4


def calculate_durations(
    segments: list[dict],
    audio_duration: float,
    whisper_words: list[tuple[str, float, float]],
) -> list[float]:
    """Calculate per-image durations from Whisper word timestamps.

    Maps each segment's word count to the corresponding Whisper-transcribed
    words (sequentially, proportionally) and uses the actual audio timestamps
    to determine exact transition points.
    """
    # Count words per segment from the LLM output
    seg_word_counts = [
        max(len(seg.get("segment", "").split()), 1) for seg in segments
    ]
    total_seg_words = sum(seg_word_counts)
    total_whisper_words = len(whisper_words)

    # Map segment word counts → Whisper word indices proportionally
    # This handles mismatches between LLM segment text and Whisper output
    durations = []
    whisper_pos = 0

    for i, wc in enumerate(seg_word_counts):
        # How many Whisper words this segment covers (proportional)
        share = wc / total_seg_words
        whisper_count = round(share * total_whisper_words)
        whisper_count = max(whisper_count, 1)  # at least 1 word

        # Don't overshoot
        if whisper_pos + whisper_count > total_whisper_words:
            whisper_count = total_whisper_words - whisper_pos
        # Last segment takes all remaining
        if i == len(seg_word_counts) - 1:
            whisper_count = total_whisper_words - whisper_pos

        if whisper_count <= 0:
            # Edge case: ran out of words, give a tiny duration
            durations.append(0.1)
            continue

        seg_start = whisper_words[whisper_pos][1]  # start of first word
        seg_end = whisper_words[whisper_pos + whisper_count - 1][2]  # end of last word
        durations.append(seg_end - seg_start)
        whisper_pos += whisper_count

    # Adjust so durations sum to exactly audio_duration
    # (Whisper timestamps may not perfectly cover the full audio)
    total = sum(durations)
    if total > 0:
        scale = audio_duration / total
        durations = [d * scale for d in durations]

    short = sum(1 for d in durations if d < 2.0)
    long = sum(1 for d in durations if d > 3.0)
    logger.info(
        f"Clip durations: min={min(durations):.2f}s, max={max(durations):.2f}s, "
        f"mean={audio_duration / len(durations):.2f}s "
        f"({short} under 2s, {long} over 3s)"
    )
    return durations


def _render_clip(
    index: int,
    img_path: Path,
    clip_path: Path,
    clip_duration: float,
    config: Config,
    total: int,
) -> Path:
    """Render a single image into a static video clip (no zoom/pan)."""
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", str(img_path),
        "-t", str(clip_duration),
        "-c:v", config.video_codec,
        "-b:v", config.video_bitrate,
        "-r", str(config.video_fps),
        "-pix_fmt", "yuv420p",
        str(clip_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for clip {index + 1}: {result.stderr[-500:]}"
        )

    logger.info(f"Rendered clip {index + 1}/{total} ({clip_duration:.2f}s)")
    return clip_path


def assemble_video(
    image_paths: list[Path],
    audio_path: Path,
    output_dir: Path,
    config: Config,
    clip_durations: list[float] | None = None,
) -> Path:
    """Combine static images + audio into a final MP4 (slideshow style).

    If *clip_durations* is provided, each image is held for its
    corresponding duration.  Otherwise every image gets equal time.
    """
    audio_duration = _get_audio_duration(audio_path)
    num_images = len(image_paths)

    if clip_durations is None:
        clip_durations = [audio_duration / num_images] * num_images

    # Merge clips shorter than 2s into their neighbors so every image
    # stays on screen long enough.  The merged image is dropped and its
    # duration is added to the neighbour, keeping total time unchanged.
    MIN_CLIP = 2.0
    merged_images: list[Path] = []
    merged_durations: list[float] = []
    for i, (img, dur) in enumerate(zip(image_paths, clip_durations)):
        if merged_durations and merged_durations[-1] + dur < MIN_CLIP * 2:
            # Merging into the previous clip is safe — absorb this one
            if dur < MIN_CLIP:
                merged_durations[-1] += dur
                continue
        if dur < MIN_CLIP and merged_durations:
            # Too short — add duration to previous clip
            merged_durations[-1] += dur
            continue
        merged_images.append(img)
        merged_durations.append(dur)

    if len(merged_images) < num_images:
        logger.info(
            f"Merged {num_images - len(merged_images)} short clips "
            f"(< {MIN_CLIP}s) into neighbours: "
            f"{num_images} → {len(merged_images)} clips"
        )
        image_paths = merged_images
        clip_durations = merged_durations
        num_images = len(image_paths)

    render_workers = getattr(config, "render_workers", DEFAULT_RENDER_WORKERS)

    logger.info(
        f"Assembling slideshow: {num_images} images, "
        f"{audio_duration:.1f}s total, "
        f"{render_workers} parallel workers"
    )

    temp_dir = output_dir / "temp_clips"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Render clips in parallel
    clip_paths: dict[int, Path] = {}

    with ThreadPoolExecutor(max_workers=render_workers) as executor:
        futures = {
            executor.submit(
                _render_clip,
                i,
                img_path,
                temp_dir / f"clip_{i:03d}.mp4",
                clip_durations[i],
                config,
                num_images,
            ): i
            for i, img_path in enumerate(image_paths)
        }

        for future in as_completed(futures):
            idx = futures[future]
            clip_paths[idx] = future.result()

    # Create concat file
    concat_file = temp_dir / "concat.txt"
    with open(concat_file, "w") as f:
        for i in range(num_images):
            f.write(f"file '{clip_paths[i].name}'\n")

    # Concatenate clips + add audio
    output_path = output_dir / "video.mp4"
    logger.info("Concatenating clips and adding audio...")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", config.audio_codec,
        "-shortest",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg concat failed: {result.stderr[-500:]}")

    # Clean up temp files
    shutil.rmtree(temp_dir)

    logger.info(f"Slideshow video assembled: {output_path}")
    return output_path
