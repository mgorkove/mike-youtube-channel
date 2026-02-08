"""Image generation via Gemini 2.5 Flash Image.

Generates scene images with a consistent reference character
for use as video frames.
"""

import logging
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image

from config_loader import Config

logger = logging.getLogger(__name__)


def generate_images(
    image_prompts: list[str],
    output_dir: Path,
    config: Config,
    client: genai.Client,
) -> list[Path]:
    """Generate one image per scene prompt, maintaining character consistency.

    Each image is generated with the reference character image passed
    alongside the prompt. Images are saved at config dimensions (1920x1080).
    """
    reference_img = Image.open(config.reference_image_path)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[Path] = []

    for i, prompt in enumerate(image_prompts):
        logger.info(f"Generating image {i + 1}/{len(image_prompts)}")

        full_prompt = (
            f"Generate a photorealistic image in 16:9 aspect ratio. "
            f"The scene shows: {prompt}. "
            f"The man from the reference photo must appear in this scene "
            f"with the same face, build, and appearance as in the reference. "
            f"Cinematic lighting, high quality, YouTube video still frame style. "
            f"No text or watermarks in the image."
        )

        response = client.models.generate_content(
            model=config.image_model,
            contents=[full_prompt, reference_img],
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
            ),
        )

        img_path = images_dir / f"{i + 1:03d}.png"
        saved = False

        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.mime_type and \
               part.inline_data.mime_type.startswith("image/"):
                # Decode image from response
                import io
                img_data = part.inline_data.data
                img = Image.open(io.BytesIO(img_data))
                # Resize to target dimensions
                img = img.resize(
                    (config.image_width, config.image_height),
                    Image.LANCZOS,
                )
                img.save(img_path, "PNG")
                image_paths.append(img_path)
                saved = True
                logger.info(f"Saved image {i + 1}: {img_path}")
                break

        if not saved:
            raise RuntimeError(
                f"No image data in response for prompt {i + 1}: {prompt[:100]}..."
            )

    return image_paths
