"""Video assembly for the satisfying_shorts video_mode.

Builds a 60-second YouTube Short:
  [0–2s)  ASMR intro clip (random pick from assets/intro/) with the text
          "perfectly satisfying photos" overlaid. Uses the clip's own audio.
  [2–60s) 29 photos held 2 seconds each, with a random background music
          track from assets/music/ playing under them.

The final video is 1080x1920, H.264, 30fps, with AAC audio.
"""

from __future__ import annotations

import logging
import random
import shutil
import subprocess
from pathlib import Path

from config_loader import Config
from generators import stock_footage

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_BUNDLED_FFMPEG = _PROJECT_ROOT / "bin" / "ffmpeg"
_BUNDLED_FFPROBE = _PROJECT_ROOT / "bin" / "ffprobe"
FFMPEG = str(_BUNDLED_FFMPEG) if _BUNDLED_FFMPEG.exists() else "ffmpeg"
FFPROBE = str(_BUNDLED_FFPROBE) if _BUNDLED_FFPROBE.exists() else "ffprobe"

# Default font for intro text overlay; lives in repo root assets/.
_DEFAULT_FONT = _PROJECT_ROOT / "assets" / "Anton-Regular.ttf"

W = 1080
H = 1920

_VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".m4v"}
_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".aac", ".flac"}


def _resolve_intro_clip(intro_dir: Path, temp_dir: Path, config: Config) -> Path:
    """Pick a local intro clip if any exist, else download one from Pexels.

    Local files in ``assets/intro/`` always win — that lets a user override
    with hand-picked footage. If the directory is missing or empty, fetch a
    random visually-ASMR clip from Pexels (kinetic sand, slime, paint
    mixing, etc.) and cache it under ``temp_dir``.
    """
    if intro_dir.exists():
        local = [p for p in intro_dir.iterdir() if p.is_file() and p.suffix.lower() in _VIDEO_EXTS]
        if local:
            return random.choice(local)
    pexels_path = temp_dir / "intro_pexels.mp4"
    queries = config.satisfying_pexels_intro_queries or None
    return stock_footage.fetch_intro_clip(pexels_path, queries=queries)


def _pick_random_asset(directory: Path, exts: set[str], kind: str) -> Path:
    if not directory.exists():
        raise FileNotFoundError(
            f"Required {kind} directory missing: {directory}. "
            f"See channels/daniel_reed_shorts/assets/README.md."
        )
    candidates = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in exts]
    if not candidates:
        raise FileNotFoundError(
            f"No {kind} files found in {directory}. "
            f"Drop at least one {sorted(exts)} file in there. "
            f"See channels/daniel_reed_shorts/assets/README.md."
        )
    return random.choice(candidates)


def _ffmpeg_escape_path(p: Path) -> str:
    """Escape a filesystem path for use inside an ffmpeg filtergraph."""
    return str(p).replace("\\", "/").replace(":", "\\:").replace("'", r"\'")


def _ffmpeg_escape_text(text: str) -> str:
    """Escape text for ffmpeg drawtext: backslashes, colons, single quotes, percent."""
    return (
        text.replace("\\", "\\\\")
            .replace(":", "\\:")
            .replace("'", "\\'")
            .replace("%", "\\%")
    )


def assemble_satisfying_short(
    image_paths: list[Path],
    output_dir: Path,
    config: Config,
) -> Path:
    """Render the final 1080x1920 short to ``output_dir/video.mp4``.

    Picks a random ASMR intro clip and a random music track from the
    channel's ``assets/intro/`` and ``assets/music/`` directories.
    """
    output_path = output_dir / "video.mp4"
    temp_dir = output_dir / "temp_short"
    temp_dir.mkdir(parents=True, exist_ok=True)

    intro_dir = config.satisfying_intro_dir
    music_dir = config.satisfying_music_dir
    if intro_dir is None or music_dir is None:
        raise RuntimeError(
            "satisfying_shorts mode requires satisfying_intro_dir and "
            "satisfying_music_dir to be set in config (auto-derived from "
            "channel directory)."
        )

    intro_clip = _resolve_intro_clip(intro_dir, temp_dir, config)
    music_track = _pick_random_asset(music_dir, _AUDIO_EXTS, "music")
    logger.info(f"Intro clip: {intro_clip}")
    logger.info(f"Music track: {music_track.name}")

    intro_secs = int(config.satisfying_intro_seconds)
    sec_per_image = int(config.satisfying_seconds_per_image)
    photo_secs = sec_per_image * len(image_paths)
    total_secs = intro_secs + photo_secs
    fps = int(config.video_fps)

    # --- Step 1: Build the photo slideshow segment via concat demuxer ---
    photos_concat = temp_dir / "photos.txt"
    with open(photos_concat, "w", encoding="utf-8") as f:
        for img in image_paths:
            f.write(f"file '{img.resolve()}'\n")
            f.write(f"duration {sec_per_image}\n")
        # The concat demuxer requires the last image to be listed twice
        # without a duration so it knows when the stream ends.
        f.write(f"file '{image_paths[-1].resolve()}'\n")

    photos_video = temp_dir / "photos.mp4"
    photos_vf = (
        f"scale={W}:{H}:force_original_aspect_ratio=increase,"
        f"crop={W}:{H},"
        f"setsar=1,fps={fps},format=yuv420p"
    )
    cmd_photos = [
        FFMPEG, "-y",
        "-f", "concat", "-safe", "0", "-i", str(photos_concat),
        "-vf", photos_vf,
        "-r", str(fps),
        "-c:v", config.video_codec,
        "-b:v", config.video_bitrate,
        "-pix_fmt", "yuv420p",
        "-an",
        str(photos_video),
    ]
    _run(cmd_photos, "photos slideshow")

    # --- Step 2: Build the intro segment (intro clip + text overlay) ---
    font_path = _DEFAULT_FONT if _DEFAULT_FONT.exists() else None
    text = _ffmpeg_escape_text(config.satisfying_intro_text)
    fontfile_clause = f"fontfile='{_ffmpeg_escape_path(font_path)}':" if font_path else ""
    drawtext = (
        f"drawtext={fontfile_clause}"
        f"text='{text}':"
        f"fontcolor=white:"
        f"fontsize=72:"
        f"borderw=4:bordercolor=black@0.85:"
        f"x=(w-text_w)/2:"
        f"y=h*0.42"
    )
    intro_vf = (
        f"scale={W}:{H}:force_original_aspect_ratio=increase,"
        f"crop={W}:{H},"
        f"setsar=1,fps={fps},format=yuv420p,"
        f"{drawtext}"
    )
    # Intro segment is silent video only — Pexels clips usually have no
    # audio anyway, and music is muxed over the entire 60s in step 4.
    intro_video = temp_dir / "intro.mp4"
    cmd_intro = [
        FFMPEG, "-y",
        "-t", str(intro_secs),
        "-i", str(intro_clip),
        "-vf", intro_vf,
        "-r", str(fps),
        "-t", str(intro_secs),
        "-c:v", config.video_codec,
        "-b:v", config.video_bitrate,
        "-pix_fmt", "yuv420p",
        "-an",
        str(intro_video),
    ]
    _run(cmd_intro, "intro segment")

    # --- Step 3: Concat intro + photos into a single silent video ---
    silent_concat = temp_dir / "silent_concat.txt"
    silent_concat.write_text(
        f"file '{intro_video.resolve()}'\n"
        f"file '{photos_video.resolve()}'\n",
        encoding="utf-8",
    )
    silent_full = temp_dir / "silent_full.mp4"
    cmd_silent = [
        FFMPEG, "-y",
        "-f", "concat", "-safe", "0", "-i", str(silent_concat),
        "-c:v", "copy",
        "-an",
        "-t", str(total_secs),
        str(silent_full),
    ]
    _run(cmd_silent, "intro+photos concat")

    # --- Step 4: Mux background music across the full 60s ---
    # Music is faded in over 0.3s (so the first frame isn't a click) and
    # faded out over the final 1s.
    music_filter = (
        f"[1:a]aloop=loop=-1:size=2e9,"
        f"atrim=duration={total_secs},"
        f"asetpts=PTS-STARTPTS,"
        f"volume={config.satisfying_music_volume},"
        f"afade=t=in:st=0:d=0.3,"
        f"afade=t=out:st={max(0, total_secs - 1)}:d=1[a]"
    )
    cmd_final = [
        FFMPEG, "-y",
        "-i", str(silent_full),
        "-i", str(music_track),
        "-filter_complex", music_filter,
        "-map", "0:v",
        "-map", "[a]",
        "-c:v", "copy",
        "-c:a", config.audio_codec,
        "-ar", "44100",
        "-ac", "2",
        "-t", str(total_secs),
        "-movflags", "+faststart",
        str(output_path),
    ]
    _run(cmd_final, "mux music")

    shutil.rmtree(temp_dir, ignore_errors=True)
    logger.info(f"Satisfying short assembled: {output_path} ({total_secs}s)")
    return output_path


def make_thumbnail_from_photo(
    image_paths: list[Path],
    output_dir: Path,
    config: Config,
) -> Path:
    """Pick a representative photo as the Short's thumbnail.

    Shorts thumbnails are barely surfaced in YouTube's UI (the player uses
    a frame from the video), but the upload pipeline expects a thumbnail
    file. We use the first photo, scaled to thumbnail dimensions.
    """
    from PIL import Image

    src = image_paths[0]
    out = output_dir / "thumbnail.png"
    img = Image.open(src).convert("RGB")
    img = img.resize((config.thumbnail_width, config.thumbnail_height), Image.LANCZOS)
    img.save(out, "PNG")
    return out


def _run(cmd: list[str], description: str) -> None:
    logger.info(f"ffmpeg: {description}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed during {description}: {result.stderr[-800:]}"
        )
