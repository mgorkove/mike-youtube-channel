"""Video assembly with Ken Burns zoom effect.

Uses PIL affine transforms for sub-pixel smooth center-zoom on each image,
piping raw frames to ffmpeg for encoding. Clips are rendered in parallel,
then concatenated with the audio track.
"""

import logging
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

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
    """Render a single image into a video clip with smooth Ken Burns center-zoom.

    Uses PIL's Image.transform(AFFINE, LANCZOS) for sub-pixel precision,
    eliminating the integer-rounding jitter that ffmpeg's zoompan/crop filters
    produce. Raw RGB frames are piped to ffmpeg for encoding.
    """
    frames = int(clip_duration * config.video_fps)
    w = config.image_width
    h = config.image_height
    r = config.ken_burns_ratio
    fps = config.video_fps

    img = Image.open(img_path).convert("RGB")

    proc = subprocess.Popen(
        [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", str(fps),
            "-i", "pipe:",
            "-c:v", config.video_codec,
            "-b:v", config.video_bitrate,
            "-pix_fmt", "yuv420p",
            str(clip_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    for frame_n in range(frames):
        # zoom grows from 1.0 to 1.0 + ratio over (frames/fps) seconds
        zoom = 1.0 + r * frame_n / fps
        inv_z = 1.0 / zoom
        # Affine maps output pixel (x,y) → source pixel (inv_z*x + cx, inv_z*y + cy)
        # Center the crop: offset = center * (1 - 1/zoom)
        cx = w / 2.0 * (1.0 - inv_z)
        cy = h / 2.0 * (1.0 - inv_z)
        frame = img.transform(
            (w, h),
            Image.AFFINE,
            data=(inv_z, 0, cx, 0, inv_z, cy),
            resample=Image.BICUBIC,
        )
        proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for clip {index + 1}: {proc.stderr.read().decode()[-500:]}"
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
