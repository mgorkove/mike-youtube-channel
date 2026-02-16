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

THUMBNAIL_STRATEGIST_PROMPT = """You are an elite YouTube thumbnail strategist specializing in high-CTR finance content.

Your task: generate ONE image-generation prompt for a YouTube thumbnail based on the video title/topic.

The thumbnail MUST include the provided reference image of the man as the recurring protagonist.

COMPOSITION — Keep it extremely simple:
- The cartoon man from the reference image takes up 40-60% of the frame (upper body or half body)
- ONE simple prop or visual element related to the topic (e.g., a bank building, a money bag, a chart arrow, a document)
- Bold ALL-CAPS text overlay — 2-4 words derived from the video title
- NOTHING ELSE. No floating symbols, no scattered dollar signs, no extra decorations, no small text, no labels

BACKGROUND — Must be clean and simple:
- Use a solid color or simple gradient — white, light gray, pale green, dark navy, or dark red
- NO complex scenes, NO cityscapes, NO detailed environments
- The background should make the character and text POP, not compete with them

CHARACTER — Non-negotiable:
- Render the man from the reference image as a cartoon in bold adult animated style with thick black outlines
- Same hair, same face shape, same general appearance — must be recognizable as the same person
- He wears the same teal/sage green crewneck sweater as in the reference image
- Exaggerated facial expression: shock, concern, realization, excitement, or disbelief
- Upper body or half body — he should be LARGE in the frame
- He can point at something, hold something, react to something — one clear action

TEXT — Critical rules:
- Choose 2-4 words of bold ALL-CAPS text extracted or paraphrased from the video title
- The text must appear ONCE — do NOT duplicate it in multiple places
- Text should be in thick, heavy, blocky font (Impact style) with black outline/stroke
- Text fills roughly 30-40% of the thumbnail area
- Place text at the top OR bottom — not both
- Use white or yellow text on dark backgrounds, red or green text on light backgrounds
- BANNED overlay phrases: "EPIC FAIL", "GONE WRONG", "YOU WON'T BELIEVE", "SHOCKING", "OMG", "MIND BLOWN", "EXPOSED", "WHAT HAPPENED", "NOT CLICKBAIT", "SECRET"

Examples of good text choices:
- Title: "What Changes When You Cross the $250K Threshold" → "$250K CHANGES"
- Title: "How the Banking System Assigns Internal Risk Scores" → "YOUR RISK SCORE"
- Title: "The Invisible Tax Cliff..." → "TAX CLIFF"
- Title: "Lombard Loans: How the Wealthy..." → "NEVER SELL"

STRICT BANS — The image must NOT contain:
- Euro signs (€) — only use dollar signs ($) if any currency symbols appear
- Placeholder names like "John Smith", "Jane Doe", etc.
- Black bars or letterboxing at top/bottom
- Duplicate text (the overlay text must appear exactly ONCE)
- More than one prop/metaphor object — keep it to ONE
- Small unreadable text or labels on objects

STYLE:
- Bold adult animated cartoon — thick black outlines, smooth cel shading, rounded shapes
- NOT flat vector, NOT realistic, NOT 3D rendered
- High contrast colors — the thumbnail must be eye-catching at phone-screen size
- Aspect ratio 16:9, resolution 1280x720

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

    # Reinforce clean composition and exact text
    reinforcement = (
        'CRITICAL RULES FOR THIS IMAGE:\n'
        '- Clean, simple background — solid color or simple gradient only\n'
        '- NO black bars or letterboxing\n'
        '- NO euro signs — only dollar signs ($) if needed\n'
        '- NO placeholder names (no "John Smith" etc.)\n'
        '- NO scattered floating symbols or decorations\n'
    )
    if overlay_text:
        reinforcement += (
            f'- The text "{overlay_text}" must appear EXACTLY ONCE in bold '
            f'ALL-CAPS with black outline. Do NOT duplicate it.\n'
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
