"""Thumbnail generation via Gemini Image + Pillow text overlay.

Generates an eye-catching thumbnail with the reference character
and overlays bold, legible title text.
"""

import io
import logging
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont

from config_loader import Config

logger = logging.getLogger(__name__)


def generate_thumbnail(
    title: str,
    topic: str,
    output_dir: Path,
    config: Config,
    client: genai.Client,
) -> Path:
    """Generate a YouTube thumbnail with character and bold text overlay.

    Two-step process:
    1. Use Gemini to generate a dramatic base image with the reference character
    2. Use Pillow to overlay large, high-contrast text
    """
    # Step 1: Generate base image via Gemini
    reference_img = Image.open(config.reference_image_path)

    thumbnail_prompt = (
        f"A dramatic YouTube thumbnail photo. The man from the reference photo "
        f"appears prominently on the right side of the frame with an expressive, "
        f"engaging facial expression (surprised, intense, or thoughtful). "
        f"The background subtly suggests finance and money themes — "
        f"perhaps blurred city buildings, dollar signs, or financial charts. "
        f"Bold, vibrant colors. High contrast. Eye-catching composition. "
        f"The left side of the image should have a solid or gradient area "
        f"suitable for text overlay. "
        f"Topic context: {topic}. "
        f"No text in the image — text will be added separately. "
        f"16:9 aspect ratio, photorealistic, professional YouTube thumbnail style."
    )

    response = client.models.generate_content(
        model=config.image_model,
        contents=[thumbnail_prompt, reference_img],
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
        ),
    )

    base_img = None
    for part in response.candidates[0].content.parts:
        if part.inline_data and part.inline_data.mime_type and \
           part.inline_data.mime_type.startswith("image/"):
            img_data = part.inline_data.data
            base_img = Image.open(io.BytesIO(img_data))
            base_img = base_img.resize(
                (config.thumbnail_width, config.thumbnail_height),
                Image.LANCZOS,
            )
            break

    if base_img is None:
        raise RuntimeError("No image data in thumbnail generation response")

    # Step 2: Overlay text
    short_text = _shorten_for_thumbnail(title)
    base_img = _overlay_text(base_img, short_text)

    thumb_path = output_dir / "thumbnail.png"
    base_img.save(thumb_path, "PNG")
    logger.info(f"Thumbnail saved: {thumb_path}")
    return thumb_path


def _shorten_for_thumbnail(title: str) -> str:
    """Extract the most impactful 3-5 words from the title.

    Thumbnail text should be extremely short — readable at small sizes.
    Strips filler words to keep only the punch.
    """
    filler = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "to", "and", "for", "it", "its"}
    words = title.split()
    # Remove leading filler words (e.g., "Why The" → start at "Banking")
    key_words = [w for w in words if w.lower() not in filler]
    if len(key_words) <= 5:
        return " ".join(key_words).upper()
    return " ".join(key_words[:4]).upper()


def _overlay_text(img: Image.Image, text: str) -> Image.Image:
    """Overlay bold, high-contrast text on the left portion of the image."""
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Text area: left 55% of the image
    text_area_width = int(w * 0.55)
    margin = int(w * 0.04)
    max_text_width = text_area_width - 2 * margin

    # Auto-size font: start large, shrink until text fits within ~3-4 lines
    font_size = 80
    while font_size > 30:
        font = _get_bold_font(size=font_size)
        lines = _wrap_text(text, font, max_text_width, draw)
        if len(lines) <= 4:
            break
        font_size -= 4

    line_text = "\n".join(lines)

    # Calculate text position (vertically centered on left side)
    bbox = draw.multiline_textbbox((0, 0), line_text, font=font)
    text_height = bbox[3] - bbox[1]
    x = margin
    y = (h - text_height) // 2

    # Draw text with thick stroke for legibility
    draw.multiline_text(
        (x, y),
        line_text,
        fill="white",
        font=font,
        stroke_width=5,
        stroke_fill="black",
        spacing=8,
    )

    return img


def _wrap_text(
    text: str, font: ImageFont.ImageFont, max_width: int, draw: ImageDraw.ImageDraw
) -> list[str]:
    """Word-wrap text to fit within max_width pixels."""
    words = text.split()
    lines: list[str] = []
    current_line: list[str] = []

    for word in words:
        test_line = " ".join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]

    if current_line:
        lines.append(" ".join(current_line))

    return lines


def _get_bold_font(size: int) -> ImageFont.ImageFont:
    """Try to load a bold system font, fall back to default."""
    font_paths = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue

    logger.warning("No bold system font found, using default font")
    return ImageFont.load_default()
