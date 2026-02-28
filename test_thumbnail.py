#!/usr/bin/env python3
"""Quick test: generate a single thumbnail for heartbreak_chronicles."""

import logging
import sys
from pathlib import Path

from google import genai

from config_loader import load_config
from generators import thumbnail

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

config = load_config("channels/heartbreak_chronicles/config.yaml")
client = genai.Client()

output_dir = Path("output/test_thumbnail")
output_dir.mkdir(parents=True, exist_ok=True)

title = "Wife Came Home at 3AM Smelling Like His Best Friend's Cologne"
topic = "A husband notices his wife sneaking in late smelling like cologne and sets up a hidden camera"

thumb_path = thumbnail.generate_thumbnail(title, topic, output_dir, config, client)
print(f"\nThumbnail saved to: {thumb_path}")
