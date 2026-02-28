"""Upload the re-rendered video to YouTube."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config_loader import load_config
from upload.youtube import upload_video

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

config = load_config("config.yaml")
output_dir = Path("output/net-worth-levels-where-the-rules-quietly-change")

title = (output_dir / "title.txt").read_text().strip()
description = (output_dir / "description.txt").read_text().strip()
video_path = output_dir / "video.mp4"
thumbnail_path = output_dir / "thumbnail.png"

print(f"Title: {title}")
print(f"Video: {video_path} ({video_path.stat().st_size / 1024 / 1024:.1f} MB)")
print(f"Thumbnail: {thumbnail_path}")
print("Uploading...")

result = upload_video(video_path, title, description, thumbnail_path, config)
print(f"Uploaded! {result.video_url}")
