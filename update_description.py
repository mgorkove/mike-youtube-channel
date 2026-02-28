"""Update the YouTube video description."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config_loader import load_config
from upload.youtube import get_youtube_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

config = load_config("config.yaml")
description = Path("output/net-worth-levels-where-the-rules-quietly-change/description.txt").read_text().strip()

service = get_youtube_service(config)

video_id = "dT5YTZbLUWg"

service.videos().update(
    part="snippet",
    body={
        "id": video_id,
        "snippet": {
            "title": "Net Worth Levels Where the Rules Quietly Change",
            "description": description,
            "tags": config.youtube_tags,
            "categoryId": config.youtube_category_id,
        },
    },
).execute()

print(f"Description updated for https://www.youtube.com/watch?v={video_id}")
