"""Video assembly from a static background image with burned-in subtitles.

Takes a single background image, burns SRT subtitles with a semi-transparent
backdrop for contrast, and muxes with the audio track.
"""

import logging
import shutil
import subprocess
from pathlib import Path

from config_loader import Config

logger = logging.getLogger(__name__)

# Use project-bundled ffmpeg if available, otherwise fall back to system.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_BUNDLED_FFMPEG = _PROJECT_ROOT / "bin" / "ffmpeg"
FFMPEG = str(_BUNDLED_FFMPEG) if _BUNDLED_FFMPEG.exists() else "ffmpeg"


def assemble_static_image_video(
    background_image: Path,
    audio_path: Path,
    srt_path: Path,
    output_dir: Path,
    config: Config,
) -> Path:
    """Create a video from a static background image with subtitles and audio.

    Uses ffmpeg -loop 1 to turn the image into a video stream, burns in
    subtitles with a semi-transparent black backdrop, and muxes the audio.
    """
    output_path = output_dir / "video.mp4"
    temp_dir = output_dir / "temp_clips"
    temp_dir.mkdir(parents=True, exist_ok=True)

    w = config.image_width
    h = config.image_height
    fps = config.video_fps
    font_size = config.subtitle_font_size
    margin_v = config.subtitle_margin_v

    # Copy SRT to a simple temp path to avoid ffmpeg filter escaping issues.
    simple_srt = temp_dir / "subs.srt"
    shutil.copy2(srt_path, simple_srt)

    srt_escaped = str(simple_srt).replace("\\", "/").replace(":", "\\:")

    # Subtitle styling: white bold text with semi-transparent black backdrop
    # BorderStyle=4 = opaque/translucent box behind text
    # BackColour &H80000000 = 50% transparent black (ASS format: &HAABBGGRR)
    force_style = (
        f"FontSize={font_size},"
        f"PrimaryColour=&H00FFFFFF,"
        f"OutlineColour=&H00000000,"
        f"BackColour=&H80000000,"
        f"BorderStyle=4,"
        f"Outline=1,"
        f"Bold=1,"
        f"Alignment=2,"
        f"MarginV={margin_v}"
    )
    subtitle_filter = f"subtitles={srt_escaped}:force_style='{force_style}'"

    # Scale the image then burn subtitles
    vf = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},{subtitle_filter}"

    cmd = [
        FFMPEG, "-y",
        "-loop", "1",
        "-i", str(background_image),
        "-i", str(audio_path),
        "-vf", vf,
        "-r", str(fps),
        "-tune", "stillimage",
        "-c:v", config.video_codec,
        "-b:v", config.video_bitrate,
        "-pix_fmt", "yuv420p",
        "-c:a", config.audio_codec,
        "-shortest",
        str(output_path),
    ]

    logger.info(f"Assembling static image video with subtitles (using {FFMPEG})...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg static image assembly failed: {result.stderr[-500:]}")

    # Clean up temp files
    shutil.rmtree(temp_dir)

    logger.info(f"Static image video assembled: {output_path}")
    return output_path
