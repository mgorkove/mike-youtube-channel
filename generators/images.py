"""Image generation via Gemini 2.5 Flash Image.

Generates scene images with a consistent reference character
for use as video frames. Uses parallel workers and per-image retries.
"""

import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image

from config_loader import Config

logger = logging.getLogger(__name__)

MAX_WORKERS = 5
PER_IMAGE_RETRIES = 3
RETRY_DELAY = 2


def _generate_single_image(
    index: int,
    prompt: str,
    images_dir: Path,
    reference_img: Image.Image,
    config: Config,
    client: genai.Client,
    total: int,
) -> Path:
    """Generate a single image with per-image retry logic."""
    full_prompt = (
        f"Generate a photorealistic image in 16:9 aspect ratio. "
        f"The scene shows: {prompt}. "
        f"The man from the reference photo must appear in this scene "
        f"with the same face, build, and appearance as in the reference. "
        f"Cinematic lighting, high quality, YouTube video still frame style. "
        f"No text or watermarks in the image."
    )
    img_path = images_dir / f"{index + 1:03d}.png"

    for attempt in range(PER_IMAGE_RETRIES):
        try:
            response = client.models.generate_content(
                model=config.image_model,
                contents=[full_prompt, reference_img],
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                ),
            )

            for part in response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.mime_type and \
                   part.inline_data.mime_type.startswith("image/"):
                    img_data = part.inline_data.data
                    img = Image.open(io.BytesIO(img_data))
                    img = img.resize(
                        (config.image_width, config.image_height),
                        Image.LANCZOS,
                    )
                    img.save(img_path, "PNG")
                    logger.info(f"Saved image {index + 1}/{total}: {img_path}")
                    return img_path

            raise RuntimeError(f"No image data in response for prompt {index + 1}")

        except Exception as e:
            if attempt < PER_IMAGE_RETRIES - 1:
                logger.warning(
                    f"Image {index + 1}/{total} attempt {attempt + 1} failed: {e}. Retrying..."
                )
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                raise RuntimeError(
                    f"Image {index + 1} failed after {PER_IMAGE_RETRIES} attempts: {e}"
                ) from e


def generate_images(
    image_prompts: list[str],
    output_dir: Path,
    config: Config,
    client: genai.Client,
) -> list[Path]:
    """Generate one image per scene prompt using parallel workers.

    Each image is generated with the reference character image passed
    alongside the prompt. Failed images are retried individually.
    Images are saved at config dimensions (1920x1080).
    """
    reference_img = Image.open(config.reference_image_path)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    total = len(image_prompts)

    logger.info(f"Generating {total} images with {MAX_WORKERS} parallel workers")

    # Map of index -> Path for ordered results
    results: dict[int, Path] = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                _generate_single_image,
                i, prompt, images_dir, reference_img, config, client, total,
            ): i
            for i, prompt in enumerate(image_prompts)
        }

        for future in as_completed(futures):
            idx = futures[future]
            try:
                path = future.result()
                results[idx] = path
            except Exception as e:
                raise RuntimeError(
                    f"Image {idx + 1} generation failed: {e}"
                ) from e

    # Return paths in order
    return [results[i] for i in range(total)]
