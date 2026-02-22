"""Video assembly from stock footage clips with burned-in subtitles.

Takes pre-downloaded video clips, trims/scales them to uniform segments,
burns SRT subtitles into the video, and muxes with the audio track.
"""

import logging
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config_loader import Config

logger = logging.getLogger(__name__)

# Use project-bundled ffmpeg (has libass/subtitles filter) if available,
# otherwise fall back to system ffmpeg.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_BUNDLED_FFMPEG = _PROJECT_ROOT / "bin" / "ffmpeg"
_BUNDLED_FFPROBE = _PROJECT_ROOT / "bin" / "ffprobe"
FFMPEG = str(_BUNDLED_FFMPEG) if _BUNDLED_FFMPEG.exists() else "ffmpeg"
FFPROBE = str(_BUNDLED_FFPROBE) if _BUNDLED_FFPROBE.exists() else "ffprobe"


def _get_audio_duration(audio_path: Path) -> float:
    """Get audio duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            FFPROBE, "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(audio_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def _get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            FFPROBE, "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def _prepare_clip(
    index: int,
    clip_path: Path,
    output_path: Path,
    target_duration: float,
    config: Config,
    total: int,
) -> Path:
    """Scale and trim/loop a stock footage clip to exact target duration.

    If the source clip is shorter than target_duration, it gets looped.
    If longer, it gets trimmed from the start.
    """
    w = config.image_width
    h = config.image_height
    fps = config.video_fps

    source_duration = _get_video_duration(clip_path)

    if source_duration >= target_duration:
        # Trim to target duration
        cmd = [
            FFMPEG, "-y",
            "-i", str(clip_path),
            "-t", str(target_duration),
            "-vf", f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}",
            "-r", str(fps),
            "-c:v", config.video_codec,
            "-b:v", config.video_bitrate,
            "-pix_fmt", "yuv420p",
            "-an",
            str(output_path),
        ]
    else:
        # Loop the clip to fill target duration
        loops = int(target_duration / source_duration) + 1
        cmd = [
            FFMPEG, "-y",
            "-stream_loop", str(loops),
            "-i", str(clip_path),
            "-t", str(target_duration),
            "-vf", f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}",
            "-r", str(fps),
            "-c:v", config.video_codec,
            "-b:v", config.video_bitrate,
            "-pix_fmt", "yuv420p",
            "-an",
            str(output_path),
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed preparing clip {index + 1}: {result.stderr[-500:]}"
        )

    logger.info(f"Prepared clip {index + 1}/{total}")
    return output_path


def assemble_stock_video(
    clip_paths: list[Path],
    audio_path: Path,
    srt_path: Path,
    output_dir: Path,
    config: Config,
) -> Path:
    """Assemble stock footage clips with subtitles and audio into final MP4.

    1. Prepare each clip (scale, trim/loop to segment duration)
    2. Concatenate all prepared clips
    3. Burn subtitles into the concatenated video
    4. Mux with audio track
    """
    audio_duration = _get_audio_duration(audio_path)
    num_clips = len(clip_paths)
    clip_duration = audio_duration / num_clips

    render_workers = getattr(config, "render_workers", 4)

    logger.info(
        f"Assembling stock video: {num_clips} clips, "
        f"{clip_duration:.1f}s each, {audio_duration:.1f}s total"
    )

    temp_dir = output_dir / "temp_clips"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Prepare clips in parallel
    prepared: dict[int, Path] = {}
    with ThreadPoolExecutor(max_workers=render_workers) as executor:
        futures = {
            executor.submit(
                _prepare_clip,
                i,
                clip_path,
                temp_dir / f"prep_{i:03d}.mp4",
                clip_duration,
                config,
                num_clips,
            ): i
            for i, clip_path in enumerate(clip_paths)
        }

        for future in as_completed(futures):
            idx = futures[future]
            prepared[idx] = future.result()

    # Step 2: Concatenate
    concat_file = temp_dir / "concat.txt"
    with open(concat_file, "w") as f:
        for i in range(num_clips):
            f.write(f"file '{prepared[i].name}'\n")

    concat_path = temp_dir / "concat.mp4"
    cmd_concat = [
        FFMPEG, "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-c:v", "copy",
        str(concat_path),
    ]
    result = subprocess.run(cmd_concat, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg concat failed: {result.stderr[-500:]}")

    logger.info("Clips concatenated")

    # Step 3: Burn subtitles + mux audio
    output_path = output_dir / "video.mp4"
    font_size = config.subtitle_font_size
    margin_v = config.subtitle_margin_v

    # Copy SRT to a simple temp path to avoid ffmpeg filter escaping issues
    # with long/special-character paths.
    simple_srt = temp_dir / "subs.srt"
    shutil.copy2(srt_path, simple_srt)

    # Build subtitle filter with styling.
    # The subtitles filter path must have : and \ escaped.
    srt_escaped = str(simple_srt).replace("\\", "/").replace(":", "\\:")
    force_style = (
        f"FontSize={font_size},"
        f"PrimaryColour=&H00FFFFFF,"
        f"OutlineColour=&H00000000,"
        f"Outline=3,"
        f"Bold=1,"
        f"Alignment=2,"
        f"MarginV={margin_v}"
    )
    subtitle_filter = f"subtitles={srt_escaped}:force_style='{force_style}'"

    cmd_final = [
        FFMPEG, "-y",
        "-i", str(concat_path),
        "-i", str(audio_path),
        "-vf", subtitle_filter,
        "-c:v", config.video_codec,
        "-b:v", config.video_bitrate,
        "-c:a", config.audio_codec,
        "-shortest",
        str(output_path),
    ]

    logger.info(f"Burning subtitles and muxing audio (using {FFMPEG})...")
    result = subprocess.run(cmd_final, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg subtitle/mux failed: {result.stderr[-500:]}")

    # Clean up temp files
    shutil.rmtree(temp_dir)

    logger.info(f"Stock video assembled: {output_path}")
    return output_path
