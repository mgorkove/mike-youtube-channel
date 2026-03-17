"""Thumbnail generation via Gemini.

Two modes controlled by config.thumbnail_text_overlay:

**text_overlay=False** (default, e.g. Mike Explains Money):
  AI generates the full thumbnail (image + text baked in).

**text_overlay=True** (e.g. Heartbreak Chronicles):
  1. Gemini text model → overlay text + narrow portrait prompt
  2. Gemini image model → narrow portrait of the woman (~512×720)
  3. Pillow composites: dark bokeh background + woman on right + text on left
"""

import io
import logging
import random
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from config_loader import Config

logger = logging.getLogger(__name__)

# Colors for text overlay
COLOR_ORANGE = (255, 165, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_YELLOW = (255, 215, 0)
COLOR_PINK = (255, 105, 180)
COLOR_RED = (255, 40, 40)

# Map color names to RGB tuples (used by color-tagged overlay text)
COLOR_MAP = {
    "yellow": COLOR_YELLOW,
    "pink": COLOR_PINK,
    "white": COLOR_WHITE,
    "red": COLOR_RED,
    "orange": COLOR_ORANGE,
}

import re
_COLOR_TAG_RE = re.compile(r"^\[([a-z+\s]+)\]\s*", re.IGNORECASE)

# Portrait dimensions for the woman (narrow, will be pasted onto right side)
PORTRAIT_WIDTH = 512
PORTRAIT_HEIGHT = 720


def _create_bokeh_background(width: int, height: int) -> Image.Image:
    """Create a dark background with subtle red glow and soft bokeh lights."""
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)

    # Dark base gradient (fast horizontal line approach)
    for y in range(height):
        r = int(12 + 8 * (y / height))
        g = int(5 + 3 * (y / height))
        b = int(5 + 3 * (y / height))
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # Add a subtle red glow on the left side (text area) for urgency
    red_glow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(red_glow)
    # Large soft red ellipse centered on the left-center
    glow_draw.ellipse(
        [-width // 4, height // 6, width // 2, height * 5 // 6],
        fill=(120, 10, 10, 40),
    )
    red_glow = red_glow.filter(ImageFilter.GaussianBlur(radius=80))
    img = img.convert("RGBA")
    img = Image.alpha_composite(img, red_glow)
    img = img.convert("RGB")

    # Add bokeh dots (soft warm circles)
    bokeh_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    bokeh_draw = ImageDraw.Draw(bokeh_layer)

    rng = random.Random(42)  # deterministic seed for consistency
    for _ in range(35):
        x = rng.randint(-50, width + 50)
        y = rng.randint(-50, height + 50)
        radius = rng.randint(15, 80)
        # Warm red/amber tones with low opacity
        r = rng.randint(180, 255)
        g = rng.randint(60, 140)
        b = rng.randint(10, 40)
        alpha = rng.randint(15, 50)
        bokeh_draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=(r, g, b, alpha),
        )

    # Blur the bokeh for soft glow
    bokeh_layer = bokeh_layer.filter(ImageFilter.GaussianBlur(radius=25))

    # Composite bokeh onto base
    img = img.convert("RGBA")
    img = Image.alpha_composite(img, bokeh_layer)
    return img.convert("RGB")


def _word_wrap(text: str, font, max_width: int, draw: ImageDraw.Draw,
               stroke_width: int) -> list[str]:
    """Break text into lines that fit within max_width at the given font size."""
    words = text.split()
    if not words:
        return [text]
    lines = []
    current = words[0]
    for word in words[1:]:
        test = current + " " + word
        bbox = draw.textbbox((0, 0), test, font=font, stroke_width=stroke_width)
        if bbox[2] - bbox[0] <= max_width:
            current = test
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _parse_segment_colors(segment: str) -> tuple[str, list[tuple]]:
    """Parse color tag from a segment like '[yellow+white]SOME TEXT'.

    Returns (clean_text, [(color1, ...), (color2, ...)]).
    If no tag found, returns (segment, []).
    """
    m = _COLOR_TAG_RE.match(segment)
    if not m:
        return segment, []

    tag = m.group(1).lower().strip()
    text = segment[m.end():]
    colors = []
    for name in tag.split("+"):
        name = name.strip()
        if name in COLOR_MAP:
            colors.append(COLOR_MAP[name])
    return text, colors


def _draw_dual_color_line(
    draw: ImageDraw.Draw,
    x: int,
    y: int,
    text: str,
    font,
    color1: tuple,
    color2: tuple,
    outline_width: int,
):
    """Draw a single line where roughly the first half of words use color1
    and the second half use color2."""
    words = text.split()
    if not words:
        return
    split = max(1, len(words) // 2)
    part1 = " ".join(words[:split])
    part2 = " ".join(words[split:])

    # Draw first part
    draw.text(
        (x, y), part1, font=font, fill=color1,
        stroke_width=outline_width, stroke_fill=COLOR_BLACK,
    )
    # Measure first part width to position second part
    bbox1 = draw.textbbox((0, 0), part1 + " ", font=font, stroke_width=outline_width)
    x2 = x + bbox1[2] - bbox1[0]
    draw.text(
        (x2, y), part2, font=font, fill=color2,
        stroke_width=outline_width, stroke_fill=COLOR_BLACK,
    )


def _overlay_text(
    img: Image.Image,
    overlay_text: str,
    font_path: str,
) -> Image.Image:
    """Render multi-line overlay text onto the left side of a thumbnail.

    Text segments are separated by " / " in the overlay_text string.
    Long segments are word-wrapped to fill the available width.

    Supports two coloring modes:
    1. Color tags: segments prefixed with [yellow], [pink+white], etc.
       Each segment's wrapped lines get the tagged color(s).
    2. Legacy (no tags): first 2 and last 2 lines orange, middle white.
    """
    segments = [seg.strip() for seg in overlay_text.split("/")]
    if not segments:
        return img

    # Parse color tags from each segment
    parsed_segments = []  # list of (text, [colors])
    has_color_tags = False
    for seg in segments:
        text, colors = _parse_segment_colors(seg)
        if colors:
            has_color_tags = True
        parsed_segments.append((text, colors))

    img = img.copy()
    draw = ImageDraw.Draw(img)
    width, height = img.size

    # Target: text fills the left ~72% of the frame with tight margins
    text_area_width = int(width * 0.72)
    margin_left = int(width * 0.03)
    margin_top = int(height * 0.04)
    margin_bottom = int(height * 0.04)
    available_height = height - margin_top - margin_bottom

    # Find the largest font size where all word-wrapped lines fit.
    font_size = 200
    min_font_size = 24
    font = None
    # Each entry: (line_text, [colors_from_parent_segment])
    wrapped_with_colors: list[tuple[str, list[tuple]]] = []

    while font_size >= min_font_size:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except (OSError, IOError):
            logger.warning(f"Could not load font {font_path}, using default")
            font = ImageFont.load_default()
            break

        stroke = max(4, font_size // 10)
        wrapped_with_colors = []
        for text, colors in parsed_segments:
            for wline in _word_wrap(text, font, text_area_width, draw, stroke):
                wrapped_with_colors.append((wline, colors))

        total_height = 0
        for line, _ in wrapped_with_colors:
            bbox = draw.textbbox((0, 0), line, font=font, stroke_width=stroke)
            total_height += bbox[3] - bbox[1]

        if total_height <= available_height:
            break
        font_size -= 2

    if font is None:
        font = ImageFont.load_default()

    # Calculate line heights
    outline_width = max(4, font_size // 10)
    line_heights = []
    for line, _ in wrapped_with_colors:
        bbox = draw.textbbox((0, 0), line, font=font, stroke_width=outline_width)
        line_heights.append(bbox[3] - bbox[1])
    total_text_height = sum(line_heights)

    # Distribute extra vertical space between lines
    max_gap = int(font_size * 0.25)
    num_lines = len(wrapped_with_colors)
    if num_lines > 1:
        line_gap = min(
            (available_height - total_text_height) / (num_lines - 1),
            max_gap,
        )
    else:
        line_gap = 0

    # Center the block vertically if gap was capped
    used_height = total_text_height + int(line_gap) * max(num_lines - 1, 0)
    y_start = margin_top + (available_height - used_height) // 2

    # Draw each line
    y = y_start
    for i, ((line, colors), lh) in enumerate(zip(wrapped_with_colors, line_heights)):
        if has_color_tags and colors:
            if len(colors) == 1:
                # Single color tag
                draw.text(
                    (margin_left, y), line, font=font, fill=colors[0],
                    stroke_width=outline_width, stroke_fill=COLOR_BLACK,
                )
            else:
                # Dual color: first half of words in color1, second half in color2
                _draw_dual_color_line(
                    draw, margin_left, y, line, font,
                    colors[0], colors[1], outline_width,
                )
        elif has_color_tags:
            # Segment had no tag — default to white
            draw.text(
                (margin_left, y), line, font=font, fill=COLOR_WHITE,
                stroke_width=outline_width, stroke_fill=COLOR_BLACK,
            )
        else:
            # Legacy mode: first 2 and last 2 lines orange, rest white
            if i < 2 or i >= num_lines - 2:
                color = COLOR_ORANGE
            else:
                color = COLOR_WHITE
            draw.text(
                (margin_left, y), line, font=font, fill=color,
                stroke_width=outline_width, stroke_fill=COLOR_BLACK,
            )
        y += lh + int(line_gap)

    return img


def generate_thumbnail(
    title: str,
    topic: str,
    output_dir: Path,
    config: Config,
    client: genai.Client,
) -> Path:
    """Generate a YouTube thumbnail.

    text_overlay=False: AI generates the full thumbnail (legacy mode).
    text_overlay=True:  AI generates a narrow portrait → Pillow composites
                        dark background + woman + text overlay.
    """
    strategist_prompt = config.thumbnail_strategist_prompt
    if not strategist_prompt:
        raise RuntimeError(
            "No thumbnail strategist prompt configured. "
            "Ensure thumbnail_prompt.md exists next to the config file."
        )

    # Load reference image if available
    reference_img = None
    if config.reference_image_path:
        reference_img = Image.open(config.reference_image_path)

    # --- Step 1: Text model → overlay text + image prompt ---
    logger.info("Generating thumbnail prompt...")
    strategist_input = (
        f"{strategist_prompt}\n\n"
        f"Video Title: {title}\n"
        f"Video Topic: {topic}"
    )

    max_attempts = 3
    overlay_text = ""
    image_prompt = ""

    for attempt in range(1, max_attempts + 1):
        prompt_response = client.models.generate_content(
            model=config.text_model_name,
            contents=strategist_input,
            config=types.GenerateContentConfig(
                temperature=0.9,
                max_output_tokens=4096,
            ),
        )
        raw_output = prompt_response.text.strip()

        # Strip markdown code fences if present
        if raw_output.startswith("```"):
            raw_output = raw_output.split("\n", 1)[1]
            if raw_output.endswith("```"):
                raw_output = raw_output[: raw_output.rfind("```")]
            raw_output = raw_output.strip()

        # Parse EXACT_TEXT and image prompt
        image_prompt = raw_output
        if raw_output.startswith("EXACT_TEXT:"):
            parts = raw_output.split("\n", 1)
            overlay_text = parts[0].replace("EXACT_TEXT:", "").strip()
            image_prompt = parts[1].strip() if len(parts) > 1 else raw_output

        # Check if overlay text looks truncated (doesn't end with sentence-
        # ending punctuation).  Retry if so.
        if overlay_text and overlay_text.rstrip()[-1] in ".!?'\"…":
            break
        if attempt < max_attempts:
            logger.warning(
                f"Thumbnail text appears truncated (attempt {attempt}/{max_attempts}), "
                f"retrying... Last chars: ...{overlay_text[-30:]!r}"
            )
            overlay_text = ""
        else:
            logger.warning(
                f"Thumbnail text still truncated after {max_attempts} attempts, "
                f"using best result"
            )

    logger.info(f"Overlay text: {overlay_text}")

    # Use fixed image prompt from config if set (overrides LLM-generated prompt)
    if config.thumbnail_fixed_image_prompt:
        image_prompt = config.thumbnail_fixed_image_prompt
        logger.info(f"Using fixed image prompt from config")
    else:
        logger.info(f"Thumbnail prompt: {image_prompt[:120]}...")

    if config.thumbnail_text_overlay:
        # --- Composite mode: portrait + background + text ---
        return _generate_composite_thumbnail(
            image_prompt, overlay_text, output_dir, config, client, reference_img,
        )
    else:
        # --- Legacy mode: AI renders full thumbnail ---
        return _generate_full_thumbnail(
            image_prompt, overlay_text, output_dir, config, client, reference_img,
        )


def _generate_full_thumbnail(
    image_prompt: str,
    overlay_text: str,
    output_dir: Path,
    config: Config,
    client: genai.Client,
    reference_img: Image.Image | None,
) -> Path:
    """Legacy mode: AI generates the entire thumbnail with text baked in."""
    if overlay_text:
        reinforcement = (
            f'CRITICAL: The main overlay text is "{overlay_text}" in bold ALL-CAPS '
            f'with thick black outline.\n'
        )
        image_prompt = f'{reinforcement}\n{image_prompt}'

    contents = [image_prompt]
    if reference_img:
        contents.append(reference_img)

    response = client.models.generate_content(
        model=config.thumbnail_model,
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
        ),
    )

    base_img = _extract_image(response)
    base_img = base_img.resize(
        (config.thumbnail_width, config.thumbnail_height), Image.LANCZOS,
    )

    thumb_path = output_dir / "thumbnail.png"
    base_img.save(thumb_path, "PNG")
    logger.info(f"Thumbnail saved: {thumb_path}")
    return thumb_path


def _generate_composite_thumbnail(
    image_prompt: str,
    overlay_text: str,
    output_dir: Path,
    config: Config,
    client: genai.Client,
    reference_img: Image.Image | None,
) -> Path:
    """Composite mode: narrow portrait + dark bokeh background + text overlay."""
    thumb_w = config.thumbnail_width
    thumb_h = config.thumbnail_height

    # --- Generate narrow portrait of the woman ---
    logger.info(f"Generating portrait ({PORTRAIT_WIDTH}x{PORTRAIT_HEIGHT})...")
    contents = [image_prompt]
    if reference_img:
        contents.append(reference_img)

    response = client.models.generate_content(
        model=config.thumbnail_model,
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
        ),
    )

    portrait = _extract_image(response)
    # Resize to fill the right portion of the thumbnail
    portrait = portrait.resize((PORTRAIT_WIDTH, PORTRAIT_HEIGHT), Image.LANCZOS)

    # --- Create dark bokeh background ---
    logger.info("Creating background...")
    background = _create_bokeh_background(thumb_w, thumb_h)

    # --- Composite: paste portrait on the right ---
    paste_x = thumb_w - PORTRAIT_WIDTH
    paste_y = 0
    background.paste(portrait, (paste_x, paste_y))

    # --- Overlay text on the left ---
    if overlay_text:
        logger.info("Overlaying text...")
        background = _overlay_text(
            background, overlay_text, config.thumbnail_font_path,
        )

    thumb_path = output_dir / "thumbnail.png"
    background.save(thumb_path, "PNG")
    logger.info(f"Thumbnail saved: {thumb_path}")
    return thumb_path


def _extract_image(response) -> Image.Image:
    """Extract the first image from a Gemini response."""
    for part in response.candidates[0].content.parts:
        if part.inline_data and part.inline_data.mime_type and \
           part.inline_data.mime_type.startswith("image/"):
            return Image.open(io.BytesIO(part.inline_data.data))
    raise RuntimeError("No image data in thumbnail generation response")
