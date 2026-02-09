"""Video assembly with Ken Burns zoom effect via ffmpeg.

Uses ffmpeg's native zoompan filter instead of Python-level frame processing.
Each image is rendered into a clip in parallel, then concatenated with audio.
"""

import logging
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config_loader import Config

logger = logging.getLogger(__name__)

MAX_RENDER_WORKERS = 4


def _get_audio_duration(audio_path: Path) -> float:
    """Get audio duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(audio_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def _render_clip(
    index: int,
    img_path: Path,
    clip_path: Path,
    clip_duration: float,
    config: Config,
    total: int,
) -> Path:
    """Render a single image into a video clip with Ken Burns zoom via ffmpeg."""
    frames = int(clip_duration * config.video_fps)

    # Ken Burns zoom: from 1.0x to (1 + ratio * duration)x over the clip
    # zoompan z expression: on = output frame number, so time = on / fps
    zoom_expr = f"1+{config.ken_burns_ratio}*on/{config.video_fps}"

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", str(img_path),
        "-vf", (
            f"zoompan=z='{zoom_expr}'"
            f":d={frames}"
            f":x='iw/2-(iw/zoom/2)'"
            f":y='ih/2-(ih/zoom/2)'"
            f":s={config.image_width}x{config.image_height}"
            f":fps={config.video_fps},"
            f"format=yuv420p"
        ),
        "-c:v", config.video_codec,
        "-b:v", config.video_bitrate,
        "-t", str(clip_duration),
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
    """Combine images with Ken Burns zoom + audio into a final MP4.

    Each image is rendered into a short video clip using ffmpeg's zoompan
    filter (parallel workers), then all clips are concatenated and the
    audio track is muxed in.
    """
    audio_duration = _get_audio_duration(audio_path)
    num_images = len(image_paths)
    clip_duration = audio_duration / num_images

    logger.info(
        f"Assembling video: {num_images} images, "
        f"{clip_duration:.1f}s each, {audio_duration:.1f}s total, "
        f"{MAX_RENDER_WORKERS} parallel workers"
    )

    temp_dir = output_dir / "temp_clips"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Render clips in parallel
    clip_paths: dict[int, Path] = {}

    with ThreadPoolExecutor(max_workers=MAX_RENDER_WORKERS) as executor:
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

    # Create concat file (clips in order, filenames only since concat.txt is in same dir)
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

    logger.info(f"Video assembled: {output_path}")
    return output_path
