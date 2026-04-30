"""Image generation via Gemini 2.5 Flash Image.

Generates scene images with a consistent reference character
for use as video frames. Uses parallel workers and per-image retries.
"""

import io
import logging
import re
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

# Regex to detect [GUN] or [VEHICLE] prefix in image prompts.
# When present, the prefix selects which reference image to use:
#   [GUN]     → config.reference_image_path  (primary)
#   [VEHICLE] → config.reference_image_alt_path (alternate)
_REF_TYPE_PATTERN = re.compile(r"^\[(GUN|VEHICLE)\]\s*", re.IGNORECASE)


def _find_nearest_neighbor(index: int, images_dir: Path, total: int) -> Path | None:
    """Find the closest existing image by index (prefer previous, then next)."""
    for offset in range(1, total):
        for candidate_idx in [index - offset, index + offset]:
            if 0 <= candidate_idx < total:
                candidate = images_dir / f"{candidate_idx + 1:03d}.png"
                if candidate.exists():
                    return candidate
    return None


def _generate_single_image(
    index: int,
    prompt: str,
    images_dir: Path,
    reference_img: Image.Image | None,
    config: Config,
    client: genai.Client,
    total: int,
    reference_img_alt: Image.Image | None = None,
) -> Path:
    """Generate a single image with per-image retry logic."""
    # Check for [GUN]/[VEHICLE] prefix to select reference image
    ref_match = _REF_TYPE_PATTERN.match(prompt)
    if ref_match and reference_img_alt is not None:
        ref_type = ref_match.group(1).upper()
        prompt = _REF_TYPE_PATTERN.sub("", prompt)  # strip the prefix
        active_ref = reference_img if ref_type == "GUN" else reference_img_alt
    else:
        if ref_match:
            prompt = _REF_TYPE_PATTERN.sub("", prompt)
        active_ref = reference_img

    if config.video_mode == "satisfying_shorts":
        # Photographic mode: no cartoon wrapper, no reference character.
        # The LLM-written prompt already specifies vantage point, lighting,
        # location, and style cues — pass it through with a thin photo
        # wrapper that locks aspect ratio and forbids text/people-faces.
        full_prompt = (
            f"A single hyper-realistic photograph in vertical 9:16 aspect ratio. "
            f"Subject: {prompt} "
            f"Photorealistic, ultra-sharp, cinematic, rich color, professional "
            f"photography. Absolutely no text, no watermarks, no captions, no "
            f"logos. NO close-ups of human FACES — but hands, feet, and "
            f"anonymous figures photographed entirely from behind are perfectly "
            f"fine. Not an illustration, not a painting, not a render — a "
            f"photograph."
        )
        contents = [full_prompt]
    elif ref_match and active_ref:
        # Catalog mode: prompt is a direct replacement instruction (e.g.
        # "Change the gun to an M16. Change the release date to 1964.")
        # Pass it verbatim with the reference image.
        full_prompt = prompt
        contents = [full_prompt, active_ref]
    elif active_ref:
        # Character mode: wrap prompt in cartoon illustration instructions
        full_prompt = (
            f"Generate a muted, desaturated cartoon illustration in 16:9 aspect ratio. "
            f"The scene shows: {prompt}. "
            f"Draw the man from the reference photo as a cartoon character "
            f"with the same face, build, and appearance as in the reference. "
            f"Characters must NOT smile. Their facial expression should match the mood of the scene "
            f"(serious, focused, exhausted, determined, fearful, etc). No default happy faces. "
            f"Use a subdued, earthy color palette with muted greens, grays, tans, and olive tones. "
            f"Avoid bright or vibrant colors. Clean cartoon style with soft shading and clean outlines. "
            f"No text or watermarks in the image."
        )
        contents = [full_prompt, active_ref]
    else:
        full_prompt = (
            f"Generate a muted, desaturated digital art illustration in 16:9 aspect ratio. "
            f"The scene shows: {prompt}. "
            f"Characters must NOT smile. Their facial expression should match the mood of the scene "
            f"(serious, focused, exhausted, determined, fearful, etc). No default happy faces. "
            f"Use a subdued, earthy color palette with muted greens, grays, tans, and olive tones. "
            f"Avoid bright or vibrant colors. Stylized rendering with soft shading and clean outlines. "
            f"No text or watermarks in the image."
        )
        contents = [full_prompt]
    img_path = images_dir / f"{index + 1:03d}.png"

    if img_path.exists():
        logger.info(f"Skipping existing image {index + 1}/{total}: {img_path}")
        return img_path

    def _attempt_generate(attempt_contents):
        """Try to generate an image from the given contents. Returns img_path on success."""
        response = client.models.generate_content(
            model=config.image_model,
            contents=attempt_contents,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
            ),
        )

        if not response.candidates or not response.candidates[0].content or not response.candidates[0].content.parts:
            raise RuntimeError(f"Empty response for prompt {index + 1} (no image data returned)")

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

    for attempt in range(PER_IMAGE_RETRIES):
        try:
            return _attempt_generate(contents)
        except Exception as e:
            if attempt < PER_IMAGE_RETRIES - 1:
                logger.warning(
                    f"Image {index + 1}/{total} attempt {attempt + 1} failed: {e}. Retrying..."
                )
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                logger.warning(
                    f"Image {index + 1}/{total} failed all {PER_IMAGE_RETRIES} attempts. "
                    f"Retrying with sanitized prompt..."
                )

    # Fallback: retry with a generic/sanitized version of the prompt
    # (the original may have been blocked by content policy)
    if config.video_mode == "satisfying_shorts":
        sanitized_prompt = (
            f"A serene hyper-realistic photograph in vertical 9:16 aspect ratio "
            f"showing: {prompt}. Photorealistic, cinematic, ultra-sharp. "
            f"No text, no watermarks, no human faces."
        )
    else:
        sanitized_prompt = (
            f"Generate a muted, desaturated cartoon illustration in 16:9 aspect ratio. "
            f"A dramatic scene related to: {prompt}. "
            f"Use a subdued, earthy color palette with muted greens, grays, tans, and olive tones. "
            f"Avoid bright or vibrant colors. Clean cartoon style with soft shading and clean outlines. "
            f"No text or watermarks in the image. Keep the scene appropriate for all audiences."
        )
    sanitized_contents = [sanitized_prompt]
    if reference_img:
        sanitized_contents.append(reference_img)

    for attempt in range(PER_IMAGE_RETRIES):
        try:
            return _attempt_generate(sanitized_contents)
        except Exception as e:
            if attempt < PER_IMAGE_RETRIES - 1:
                logger.warning(
                    f"Image {index + 1}/{total} sanitized attempt {attempt + 1} failed: {e}. Retrying..."
                )
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                # Last resort: duplicate the nearest neighbor image
                logger.warning(
                    f"Image {index + 1}/{total} failed all attempts including sanitized fallback. "
                    f"Duplicating nearest neighbor image."
                )
                neighbor = _find_nearest_neighbor(index, images_dir, total)
                if neighbor:
                    import shutil
                    shutil.copy2(neighbor, img_path)
                    logger.info(f"Copied {neighbor} -> {img_path} as fallback for image {index + 1}")
                    return img_path
                raise RuntimeError(
                    f"Image {index + 1} failed after {PER_IMAGE_RETRIES} attempts "
                    f"(including sanitized fallback) and no neighbor found: {e}"
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
    reference_img = None
    if config.reference_image_path:
        reference_img = Image.open(config.reference_image_path)
    reference_img_alt = None
    if config.reference_image_alt_path:
        reference_img_alt = Image.open(config.reference_image_alt_path)
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
                reference_img_alt=reference_img_alt,
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
