"""Video assembly with static images (slideshow mode).

Each image is displayed for a fixed duration with no zoom or pan effects.
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

    logger.info(f"Rendered clip {index + 1}/{total}")
    return clip_path


def assemble_video(
    image_paths: list[Path],
    audio_path: Path,
    output_dir: Path,
    config: Config,
) -> Path:
    """Combine static images + audio into a final MP4 (slideshow style).

    Each image is held on screen for an equal duration. No zoom or pan
    effects are applied — images simply cut from one to the next.
    """
    audio_duration = _get_audio_duration(audio_path)
    num_images = len(image_paths)
    clip_duration = audio_duration / num_images

    render_workers = getattr(config, "render_workers", DEFAULT_RENDER_WORKERS)

    logger.info(
        f"Assembling slideshow: {num_images} images, "
        f"{clip_duration:.1f}s each, {audio_duration:.1f}s total, "
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
                clip_duration,
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
