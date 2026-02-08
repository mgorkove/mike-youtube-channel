"""Video assembly with Ken Burns zoom effect.

Combines generated images with a voiceover audio track into an MP4 video.
Each image gets a slow zoom-in (Ken Burns) effect for visual motion.
"""

import logging
from pathlib import Path

import numpy as np
from moviepy import (
    AudioFileClip,
    ImageClip,
    concatenate_videoclips,
)
from PIL import Image

from config_loader import Config

logger = logging.getLogger(__name__)


def assemble_video(
    image_paths: list[Path],
    audio_path: Path,
    output_dir: Path,
    config: Config,
) -> Path:
    """Combine images with Ken Burns zoom + audio into a final MP4.

    Each image is displayed for (audio_duration / num_images) seconds
    with a slow zoom effect. The audio track is laid over the result.
    """
    audio_clip = AudioFileClip(str(audio_path))
    audio_duration = audio_clip.duration
    num_images = len(image_paths)
    clip_duration = audio_duration / num_images

    logger.info(
        f"Assembling video: {num_images} images, "
        f"{clip_duration:.1f}s each, {audio_duration:.1f}s total"
    )

    clips = []
    for i, img_path in enumerate(image_paths):
        logger.info(f"Processing clip {i + 1}/{num_images}: {img_path.name}")

        # Load image as numpy array for MoviePy
        img = Image.open(img_path).convert("RGB")
        img_array = np.array(img)

        clip = ImageClip(img_array, duration=clip_duration)
        clip = clip.with_fps(config.video_fps)

        # Apply Ken Burns zoom effect
        clip = _apply_ken_burns(clip, config.ken_burns_ratio, config.video_fps)
        clips.append(clip)

    video = concatenate_videoclips(clips, method="compose")
    video = video.with_audio(audio_clip)

    output_path = output_dir / "video.mp4"
    logger.info(f"Writing video to {output_path}...")

    video.write_videofile(
        str(output_path),
        fps=config.video_fps,
        codec=config.video_codec,
        audio_codec=config.audio_codec,
        bitrate=config.video_bitrate,
        logger="bar",
    )

    # Clean up
    audio_clip.close()
    video.close()
    for c in clips:
        c.close()

    logger.info(f"Video assembled: {output_path}")
    return output_path


def _apply_ken_burns(clip, ratio: float, fps: int):
    """Apply a Ken Burns slow zoom-in effect centered on the image.

    The image progressively zooms from 1.0x to (1.0 + ratio * duration)x
    over the clip duration, cropping from the center to maintain
    the original frame size.
    """
    w, h = clip.size
    duration = clip.duration

    def zoom_filter(get_frame, t):
        frame = get_frame(t)

        # Calculate zoom factor at time t
        zoom = 1.0 + ratio * t

        # Calculate new dimensions (ensure even for video codecs)
        new_w = int(w * zoom)
        new_h = int(h * zoom)
        new_w += new_w % 2
        new_h += new_h % 2

        # Use PIL for high-quality resize
        img = Image.fromarray(frame)
        img_zoomed = img.resize((new_w, new_h), Image.LANCZOS)

        # Crop from center back to original dimensions
        left = (new_w - w) // 2
        top = (new_h - h) // 2
        img_cropped = img_zoomed.crop((left, top, left + w, top + h))

        return np.array(img_cropped)

    return clip.transform(zoom_filter)
