"""Thumbnail generation via Gemini.

Uses a two-step process:
1. Gemini text model generates an optimized image prompt based on the
   title/topic using a thumbnail strategist system prompt.
2. Gemini image model generates the final thumbnail with text baked in.

Style: Bold adult animated cartoon, thick outlines, bold typography,
high contrast, mobile-optimized, 1280x720.
"""

import io
import logging
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image

from config_loader import Config

logger = logging.getLogger(__name__)

THUMBNAIL_STRATEGIST_PROMPT = """You are an elite YouTube thumbnail strategist. Your job: create the most clickable finance thumbnail possible.

Generate ONE image-generation prompt for a YouTube thumbnail based on the video title/topic.

LAYOUT — The thumbnail should look like a popular YouTube finance channel thumbnail:
1. The cartoon man from the reference image — head and upper body or full body, taking up a significant portion of the frame
2. Bold ALL-CAPS text overlay — 2-4 words derived from the video title
3. Small illustrative icons, props, and visual elements that help tell the story of the video topic (e.g., dollar signs, money bags, bank buildings, chart arrows, documents, scales, locks, warning signs, level markers). These add visual interest and context.
4. A WHITE or very light background (this is the default — most thumbnails should use white/off-white). Only use dark backgrounds for especially dark/urgent topics.

CRITICAL: Every thumbnail on this channel must look DIFFERENT. Vary the layout, icons, and color accents each time.

TEXT — Must be bold and prominent:
- Choose 2-4 words extracted or paraphrased from the video title
- Thick, heavy, blocky Impact-style font with strong black outline/stroke
- Text appears ONCE only
- Use bold colors: green, red, black, or dark blue text on the white background
- The text should be large and immediately readable

Good text examples:
- "What Changes When You Cross the $250K Threshold" → "$250K CHANGES"
- "How the Banking System Assigns Internal Risk Scores" → "YOUR RISK SCORE"
- "The Invisible Tax Cliff..." → "TAX CLIFF"
- "Lombard Loans: How the Wealthy..." → "NEVER SELL"
- "How the Top 1% Use Life Insurance..." → "BE THE BANK"

BANNED text: "EPIC FAIL", "GONE WRONG", "YOU WON'T BELIEVE", "SHOCKING", "OMG", "MIND BLOWN", "EXPOSED", "NOT CLICKBAIT", "SECRET", "WHAT HAPPENED"

ILLUSTRATIVE ELEMENTS — These make the thumbnail informative and eye-catching:
- Add small icons and illustrations that visually explain the video topic
- Examples: dollar sign icons ($$$), money bags, bank buildings, upward/downward arrows, bar charts, pie charts, documents, locks, keys, scales of justice, warning triangles, percentage symbols, level/tier markers
- These should be drawn in the same cartoon style as the character
- Scatter them around the character or arrange them to tell a visual story
- Use green for positive/money elements, red for negative/danger elements
- Keep them simple and iconic — not photorealistic

CHARACTER — The man from the reference image:
- Cartoon version of the reference image with thick black outlines and smooth cel shading
- EXAGGERATED facial expression matching the topic's emotion:
  * Shock/fear: mouth WIDE open, eyes bulging, eyebrows raised, sweat drops
  * Anger/frustration: teeth clenched, brow furrowed, eyes narrowed
  * Confidence/empowerment: smug smirk, one eyebrow raised, pointing at viewer
- Same hair color/style and teal/sage green crewneck sweater as reference
- He can point at things, gesture, hold his head, cross arms, or interact with the icons

STYLE:
- Bold cartoon — thick black outlines, smooth cel shading
- WHITE or light background as the default
- Colorful illustrative icons and elements scattered around
- Bold, saturated accent colors (green, red, gold) against the white background
- NOT realistic, NOT 3D
- High contrast, eye-catching at phone-screen size
- 16:9 aspect ratio, 1280x720

STRICT BANS:
- Euro signs (€) — only dollar signs ($)
- Placeholder names like "John Smith"
- Black bars or letterboxing

Output format (strictly follow this):
- First line: EXACT_TEXT: followed by the 2-4 word ALL-CAPS overlay text
- Second line: the full image generation prompt as a single paragraph
- Nothing else"""


def generate_thumbnail(
    title: str,
    topic: str,
    output_dir: Path,
    config: Config,
    client: genai.Client,
) -> Path:
    """Generate a YouTube thumbnail using a two-step AI process.

    Step 1: Gemini text model creates an optimized image generation prompt.
    Step 2: Gemini image model generates the thumbnail with text baked in.
    """
    reference_img = Image.open(config.reference_image_path)

    # Step 1: Generate the image prompt via text model
    logger.info("Generating thumbnail prompt...")
    strategist_input = (
        f"{THUMBNAIL_STRATEGIST_PROMPT}\n\n"
        f"Video Title: {title}\n"
        f"Video Topic: {topic}"
    )

    prompt_response = client.models.generate_content(
        model=config.text_model_name,
        contents=strategist_input,
        config=types.GenerateContentConfig(
            temperature=0.9,
            max_output_tokens=1024,
        ),
    )
    raw_output = prompt_response.text.strip()

    # Parse the exact text and image prompt from strategist output
    overlay_text = ""
    image_prompt = raw_output
    if raw_output.startswith("EXACT_TEXT:"):
        lines = raw_output.split("\n", 1)
        overlay_text = lines[0].replace("EXACT_TEXT:", "").strip()
        image_prompt = lines[1].strip() if len(lines) > 1 else raw_output
    logger.info(f"Overlay text: {overlay_text}")
    logger.info(f"Thumbnail prompt: {image_prompt[:120]}...")

    # Reinforce style and exact text
    reinforcement = (
        'CRITICAL RULES — READ BEFORE GENERATING:\n'
        '- WHITE or light background (default)\n'
        '- Add small illustrative icons/props related to the topic (dollar signs, '
        'arrows, charts, money bags, etc.) scattered around in the same cartoon style\n'
        '- NO euro signs (€) — only dollar signs ($)\n'
        '- NO black bars or letterboxing\n'
    )
    if overlay_text:
        reinforcement += (
            f'- The main overlay text is "{overlay_text}" in bold ALL-CAPS '
            f'with thick black outline.\n'
        )
    image_prompt = f'{reinforcement}\n{image_prompt}'

    # Step 2: Generate the thumbnail image
    logger.info("Generating thumbnail image...")
    response = client.models.generate_content(
        model=config.image_model,
        contents=[image_prompt, reference_img],
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

    thumb_path = output_dir / "thumbnail.png"
    base_img.save(thumb_path, "PNG")
    logger.info(f"Thumbnail saved: {thumb_path}")
    return thumb_path
