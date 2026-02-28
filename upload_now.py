"""Upload the already-assembled video to YouTube."""

import logging
import sys
from pathlib import Path

from config_loader import load_config
from upload import youtube

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output/how-the-money-system-separates-winners-from-everyone-else")
TITLE = "How the Money System Separates Winners from Everyone Else"


def main():
    config = load_config("config.yaml")

    video_path = OUTPUT_DIR / "video.mp4"
    description = (OUTPUT_DIR / "description.txt").read_text(encoding="utf-8")
    thumb_path = OUTPUT_DIR / "thumbnail.png"

    logger.info(f"Uploading: {video_path}")
    logger.info(f"Title: {TITLE}")

    upload_result = youtube.upload_video(video_path, TITLE, description, thumb_path, config)
    logger.info(f"Uploaded: {upload_result.video_url}")
    print(f"\nDone! Video URL: {upload_result.video_url}")


if __name__ == "__main__":
    main()
