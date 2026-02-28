"""Re-assemble video from existing images + audio using the new smooth Ken Burns zoom."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from assembly.video import assemble_video
from config_loader import load_config

config = load_config("config.yaml")
output_dir = Path("output/net-worth-levels-where-the-rules-quietly-change")
audio_path = output_dir / "audio.wav"
images_dir = output_dir / "images"

# Collect images in order
image_paths = sorted(images_dir.glob("*.png"))
print(f"Found {len(image_paths)} images, audio: {audio_path}")

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

video_path = assemble_video(image_paths, audio_path, output_dir, config)
print(f"Done! Video: {video_path}")
