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

LAYOUT — The thumbnail has exactly these elements:
1. GIANT TEXT (the dominant element — takes up ~40-50% of the frame)
2. The cartoon man from the reference image (head and upper body, ~35-40% of frame)
3. ONE simple background prop related to the topic (optional but encouraged — e.g., a bank vault door, a red cliff edge, a locked gate, a giant dollar sign, a cracking chart). This prop should be BEHIND or BESIDE the character, never in front. Keep it simple and recognizable as a single shape.
4. A clean, bold background color

CRITICAL: Every thumbnail on this channel must look DIFFERENT from the others. Vary the color scheme, layout direction, and prop each time.

TEXT — The #1 most important element:
- Choose 2-4 words extracted or paraphrased from the video title
- The text is HUGE — each letter should be roughly the same height as the character's head
- Thick, heavy, blocky Impact-style font with strong black outline/stroke
- Text appears ONCE only — never duplicate it
- Layout options (VARY these — don't always use the same one):
  * Text on LEFT, character on RIGHT
  * Text on RIGHT, character on LEFT
  * Text across TOP, character below
  * Text split TOP and BOTTOM with character in middle

COLOR SCHEMES — Pick one that matches the topic's emotion. Do NOT always use the same colors:
- DANGER/WARNING: Deep red background + white or yellow text
- MONEY/GROWTH: Bright green background + white or dark text
- EXCLUSION/SECRECY: Dark navy background + gold or yellow text
- SYSTEM/INSTITUTIONAL: Cool gray background + bold red or green text
- URGENCY/FEAR: Black or charcoal background + red or white text
- ASPIRATIONAL: White or light background + green or dark blue text

Good text examples:
- "What Changes When You Cross the $250K Threshold" → "$250K CHANGES"
- "How the Banking System Assigns Internal Risk Scores" → "YOUR RISK SCORE"
- "The Invisible Tax Cliff..." → "TAX CLIFF"
- "Lombard Loans: How the Wealthy..." → "NEVER SELL"
- "How the Top 1% Use Life Insurance..." → "BE THE BANK"

BANNED text: "EPIC FAIL", "GONE WRONG", "YOU WON'T BELIEVE", "SHOCKING", "OMG", "MIND BLOWN", "EXPOSED", "NOT CLICKBAIT", "SECRET", "WHAT HAPPENED"

CHARACTER — The man from the reference image:
- Bold adult animated cartoon style — thick black outlines, smooth cel shading
- Head and upper body, filling ~35-40% of the frame
- EXTREME facial expression — this is critical for clicks:
  * Mouth WIDE open (jaw dropped to chin), eyes BULGING out, eyebrows shot up to hairline, visible sweat drops
  * Or: teeth CLENCHED and visible, brow deeply furrowed, eyes narrowed, veins on forehead
  * Or: smug confident smirk with one eyebrow raised, pointing at the viewer (for empowering topics)
  * Pick the expression that matches the topic's emotion. The expression must be WILDLY exaggerated — think cartoon comedy levels of overreaction
- Same hair, face shape, and teal/sage green crewneck sweater as reference
- He can point at the text, gesture dramatically, hold his head in disbelief, or cross his arms confidently

STRICT BANS:
- Euro signs (€) — only dollar signs ($) if needed
- Floating scattered symbols, coins, or decorations
- Placeholder names like "John Smith"
- Black bars or letterboxing
- Any text other than the single overlay phrase
- Small or secondary text

STYLE:
- Bold adult animated cartoon — thick black outlines, smooth cel shading, rounded shapes
- NOT flat vector, NOT realistic, NOT 3D
- High contrast, eye-catching at phone-screen size
- 16:9 aspect ratio, 1280x720

Output format (strictly follow this):
- First line: EXACT_TEXT: followed by the 2-4 word ALL-CAPS overlay text
- Second line: the full image generation prompt as a single paragraph. You MUST specify: the exact background color, the text color, the text placement (top/left/right), the character placement, the character expression, and the background prop (if any).
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
        'CRITICAL RULES — READ BEFORE GENERATING:\n'
        '- Maximum elements: one cartoon man, big text, one optional background prop, and a colored background\n'
        '- NO floating scattered symbols, NO coins, NO euro signs (€)\n'
        '- NO black bars or letterboxing\n'
        '- NO small text, NO labels, NO secondary text\n'
        '- The character expression must be WILDLY exaggerated — mouth WIDE open or teeth clenched\n'
    )
    if overlay_text:
        reinforcement += (
            f'- The ONLY text in this image is "{overlay_text}" — written ONCE in giant '
            f'bold ALL-CAPS with thick black outline. No other text anywhere.\n'
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
