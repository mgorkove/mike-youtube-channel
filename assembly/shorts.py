"""YouTube Shorts generation from long-form videos.

Extracts the most dramatic segment, crops to 9:16 vertical,
and burns in large subtitles optimized for mobile viewing.
"""

import logging
import shutil
import subprocess
from pathlib import Path

from config_loader import Config

logger = logging.getLogger(__name__)

# Reuse ffmpeg binary detection from stock_video module
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_BUNDLED_FFMPEG = _PROJECT_ROOT / "bin" / "ffmpeg"
_BUNDLED_FFPROBE = _PROJECT_ROOT / "bin" / "ffprobe"
FFMPEG = str(_BUNDLED_FFMPEG) if _BUNDLED_FFMPEG.exists() else "ffmpeg"
FFPROBE = str(_BUNDLED_FFPROBE) if _BUNDLED_FFPROBE.exists() else "ffprobe"

# Shorts specs
SHORTS_WIDTH = 1080
SHORTS_HEIGHT = 1920
SHORTS_MAX_DURATION = 59  # YouTube Shorts must be under 60 seconds


def _get_audio_duration(audio_path: Path) -> float:
    """Get audio duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            FFPROBE, "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(audio_path),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def _find_dramatic_segment(srt_path: Path, audio_duration: float) -> tuple[float, float]:
    """Find the most dramatic segment of the video for a Short.

    Strategy: The confrontation scene is typically at 55-75% of the video
    (based on the script structure). We extract a 45-second window from
    this region to capture the emotional climax.
    """
    clip_duration = min(50, SHORTS_MAX_DURATION)

    # Target the confrontation zone: 55-75% of the video
    target_start = audio_duration * 0.58
    target_end = target_start + clip_duration

    # Clamp to video bounds
    if target_end > audio_duration:
        target_end = audio_duration
        target_start = max(0, target_end - clip_duration)

    return target_start, target_end


def _extract_srt_segment(
    srt_path: Path,
    start_time: float,
    end_time: float,
    output_path: Path,
) -> Path:
    """Extract and re-time SRT subtitles for the Short's time window."""
    lines = srt_path.read_text(encoding="utf-8").strip().split("\n")

    entries = []
    i = 0
    while i < len(lines):
        # Skip blank lines
        if not lines[i].strip():
            i += 1
            continue

        # Subtitle index
        try:
            int(lines[i].strip())
        except ValueError:
            i += 1
            continue
        i += 1

        # Timestamp line
        if i >= len(lines) or "-->" not in lines[i]:
            continue
        ts_line = lines[i].strip()
        i += 1

        # Text lines
        text_lines = []
        while i < len(lines) and lines[i].strip():
            text_lines.append(lines[i].strip())
            i += 1

        # Parse timestamps
        parts = ts_line.split(" --> ")
        sub_start = _srt_time_to_seconds(parts[0].strip())
        sub_end = _srt_time_to_seconds(parts[1].strip())

        # Keep only subtitles within our window
        if sub_end > start_time and sub_start < end_time:
            # Re-time relative to clip start
            new_start = max(0, sub_start - start_time)
            new_end = min(end_time - start_time, sub_end - start_time)
            entries.append((new_start, new_end, "\n".join(text_lines)))

    # Write new SRT
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, (start, end, text) in enumerate(entries, 1):
            f.write(f"{idx}\n")
            f.write(f"{_seconds_to_srt_time(start)} --> {_seconds_to_srt_time(end)}\n")
            f.write(f"{text}\n\n")

    return output_path


def _srt_time_to_seconds(ts: str) -> float:
    """Convert SRT timestamp (HH:MM:SS,mmm) to seconds."""
    ts = ts.replace(",", ".")
    parts = ts.split(":")
    return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])


def _seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def generate_short(
    video_path: Path,
    audio_path: Path,
    srt_path: Path,
    output_dir: Path,
    config: Config,
) -> Path:
    """Generate a YouTube Short from a long-form video.

    1. Find the most dramatic segment (confrontation scene)
    2. Extract audio segment
    3. Extract and re-time subtitles
    4. Render vertical (9:16) video with large subtitles
    """
    audio_duration = _get_audio_duration(audio_path)
    start_time, end_time = _find_dramatic_segment(srt_path, audio_duration)
    clip_duration = end_time - start_time

    logger.info(
        f"Generating Short: {clip_duration:.1f}s clip from "
        f"{start_time:.1f}s to {end_time:.1f}s"
    )

    temp_dir = output_dir / "temp_short"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Extract and re-time subtitles for the short segment
    short_srt = temp_dir / "short_subs.srt"
    _extract_srt_segment(srt_path, start_time, end_time, short_srt)

    # Build subtitle filter with large font for mobile
    srt_escaped = str(short_srt).replace("\\", "/").replace(":", "\\:")
    force_style = (
        "FontSize=42,"
        "PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,"
        "Outline=4,"
        "Bold=1,"
        "Alignment=2,"
        "MarginV=120,"
        "FontName=Arial"
    )
    subtitle_filter = f"subtitles={srt_escaped}:force_style='{force_style}'"

    # Single ffmpeg command: extract segment, crop to 9:16, burn subtitles
    output_path = output_dir / "short.mp4"
    cmd = [
        FFMPEG, "-y",
        "-ss", str(start_time),
        "-t", str(clip_duration),
        "-i", str(video_path),
        "-vf", (
            f"crop=ih*9/16:ih,"
            f"scale={SHORTS_WIDTH}:{SHORTS_HEIGHT},"
            f"{subtitle_filter}"
        ),
        "-c:v", config.video_codec,
        "-b:v", "6000k",
        "-c:a", config.audio_codec,
        "-r", str(config.video_fps),
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg Short generation failed: {result.stderr[-500:]}")

    # Clean up temp files
    shutil.rmtree(temp_dir)

    logger.info(f"Short generated: {output_path} ({clip_duration:.1f}s)")
    return output_path
