#!/usr/bin/env python3
"""Quick end-to-end test: generates a ~30-second video with 3 images.

Uses shortened script/config to minimize API calls and processing time.
Runs as dry-run (no YouTube upload) unless --upload is passed.
"""

import dataclasses
import json
import logging
import math
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from config_loader import load_config, Config
from generators import text, speech, images, thumbnail
from assembly import video
from quality import checks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("test_e2e")


def main():
    do_upload = "--upload" in sys.argv

    config = load_config()
    # Override for short test
    config = dataclasses.replace(
        config,
        script_min_words=50,
        script_max_words=200,
        video_count=1,
        dry_run=not do_upload,
    )

    from google import genai
    client = genai.Client()

    output_dir = config.output_base_dir / "test-30s"
    output_dir.mkdir(parents=True, exist_ok=True)

    topic = "Why Banks Treat You Differently After $100K"

    # --- Step 1: Title ---
    logger.info("Step 1: Generating title...")
    title = text.generate_title(topic, config, client)
    logger.info(f"Title: {title}")
    (output_dir / "title.txt").write_text(title)

    # --- Step 2: Short script (~75 words for ~30s of speech) ---
    logger.info("Step 2: Generating short script...")
    short_script_prompt = (
        f"Write a 75-word YouTube script intro about: {topic}. "
        f"Open with a dramatic hook. Use blunt, direct language. "
        f"No prescriptive language like 'you should'. "
        f"Just the spoken narration, no stage directions."
    )
    response = client.models.generate_content(
        model=config.text_model_name,
        contents=short_script_prompt,
    )
    script_text = response.text.strip()
    word_count = len(script_text.split())
    logger.info(f"Script: {word_count} words")
    logger.info(f"Script text:\n{script_text}\n")
    (output_dir / "script.txt").write_text(script_text)

    # --- Step 3: Description ---
    logger.info("Step 3: Generating description...")
    description = text.generate_description(topic, title, script_text, config, client)
    logger.info(f"Description length: {len(description)} chars")
    (output_dir / "description.txt").write_text(description)

    # Check keywords + disclaimer
    kw_check = checks.check_keywords_present(description, config.required_keywords)
    disc_check = checks.check_disclaimer_present(description, config.disclaimer)
    logger.info(f"Keywords check: {kw_check}")
    logger.info(f"Disclaimer check: {disc_check}")

    # --- Step 4: Voiceover ---
    logger.info("Step 4: Generating voiceover...")
    tts_result = speech.generate_voiceover(script_text, output_dir, config, client)
    logger.info(f"Audio: {tts_result.duration_seconds:.1f}s at {tts_result.audio_path}")

    # --- Step 5: Image prompts ---
    num_images = max(3, math.ceil(tts_result.duration_seconds / config.seconds_per_image))
    logger.info(f"Step 5: Generating {num_images} image prompts...")
    image_prompts = text.extract_image_prompts(script_text, num_images, config, client)
    for i, p in enumerate(image_prompts):
        logger.info(f"  Image {i+1}: {p[:80]}...")

    # --- Step 6: Generate images ---
    logger.info("Step 6: Generating images...")
    image_paths = images.generate_images(image_prompts, output_dir, config, client)
    for ip in image_paths:
        img_check = checks.check_image_exists_and_dimensions(
            ip, config.image_width, config.image_height
        )
        logger.info(f"  {img_check}")

    # --- Step 7: Thumbnail ---
    logger.info("Step 7: Generating thumbnail...")
    thumb_path = thumbnail.generate_thumbnail(title, topic, output_dir, config, client)
    contrast_check = checks.check_contrast_ratio(thumb_path, config.thumbnail_min_contrast)
    logger.info(f"Thumbnail contrast: {contrast_check}")

    # --- Step 8: Assemble video ---
    logger.info("Step 8: Assembling video...")
    video_path = video.assemble_video(image_paths, tts_result.audio_path, output_dir, config)
    video_check = checks.check_video_file(video_path, tts_result.duration_seconds)
    logger.info(f"Video check: {video_check}")

    # --- Step 9: Upload (if requested) ---
    if do_upload:
        logger.info("Step 9: Uploading to YouTube (private)...")
        from upload import youtube
        upload_result = youtube.upload_video(
            video_path, title, description, thumb_path, config
        )
        logger.info(f"Uploaded: {upload_result.video_url}")
    else:
        logger.info("Step 9: Skipping upload (dry run). Pass --upload to upload.")

    logger.info(f"\nAll artifacts saved to: {output_dir}")
    logger.info("Test complete!")


if __name__ == "__main__":
    main()
