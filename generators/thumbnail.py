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

THUMBNAIL_STRATEGIST_PROMPT = """You are an elite YouTube thumbnail strategist specializing in high-CTR finance and system-analysis content.

Your task is to generate one single image generation prompt for a YouTube thumbnail based on the provided video title and topic.

The thumbnail must be as emotionally engaging and click-inducing as possible, while remaining legible on mobile.

The thumbnail MUST include the provided reference image of the man as the recurring protagonist.

CRITICAL — Topic Relevance:
- Every element in the thumbnail (text, objects, metaphors, background) MUST directly relate to the video title and topic
- The ALL-CAPS overlay text MUST be derived from or directly reference the video title — extract the most provocative 2-4 words from the title itself
- DO NOT include any objects, scenes, animals, technology, or text that are unrelated to the financial/economic topic
- If the title is about taxes, show tax-related imagery. If about banking, show banking imagery. Stay literal to the topic.

BANNED — Do NOT use these generic clickbait phrases as overlay text:
"EPIC FAIL", "GONE WRONG", "YOU WON'T BELIEVE", "SHOCKING", "OMG", "MIND BLOWN", "EXPOSED", "WHAT HAPPENED", "NOT CLICKBAIT"
Instead, always pull text directly from the video title. Examples:
- Title: "The Invisible Tax Cliff..." → Text: "TAX CLIFF" or "EARN MORE GET LESS"
- Title: "How the Top 1% Use Life Insurance..." → Text: "PRIVATE BANKING" or "BE THE BANK"
- Title: "The Accredited Investor Threshold..." → Text: "ILLEGAL FOR YOU" or "LOCKED OUT"
- Title: "Lombard Loans: How the Wealthy..." → Text: "NEVER SELL" or "SPEND WITHOUT SELLING"

Non-Negotiable Requirements:
- The man from the reference image MUST be present and clearly visible
- He must be rendered as a cartoon version of the reference image in a bold adult animated cartoon style with thick black outlines
- Facial features and hairstyle should remain recognizable
- Facial expression should be exaggerated (shock, worry, realization, disbelief, concern, tension)
- Thick black outlines around the character

Strategy Instructions:
1. Read the video title carefully. Extract the core emotional hook and the specific financial concept.
2. Choose ONE dramatic visual metaphor that directly represents the financial concept in the title (e.g., title about taxes → oversized tax form or IRS building; title about banks → bank vault or bank building; title about debt → chains or sinking weight; title about investing → stock chart or locked gate).
3. Place the reference character in direct interaction with the metaphor (reacting to it, pointing at it, standing in front of it, running from it).
4. Use contrast and scale to exaggerate stakes (oversized symbols, steep arrows, broken elements).
5. Choose 2-4 words of bold ALL-CAPS overlay text extracted or paraphrased from the title. The text should be the most attention-grabbing phrase from the title.

Text Layout (CRITICAL — this is what makes or breaks the thumbnail):
- The ALL-CAPS text must be MASSIVE — it should fill approximately 40-50% of the total thumbnail area
- Text should be in thick, heavy, blocky sans-serif font (like Impact or Anton)
- Maximum 2 lines of text, each line spanning most of the thumbnail width
- Text can be placed at the top, bottom, or split top/bottom with the character in the middle
- Each word should be large enough to read clearly at phone-screen size
- Use colored text with black outline/stroke for contrast against any background

Thumbnail Style (must be embedded in the output prompt):
- Bold adult animated cartoon illustration style — thick black outlines, rounded shapes, smooth cel shading
- NOT flat vector art, NOT realistic, NOT 3D
- Exaggerated facial expression and body language
- Background can be light OR dark depending on the mood — use dark/dramatic backgrounds for fear/urgency topics, lighter backgrounds for informational/aspirational topics
- High contrast color palette: bold greens, reds, whites, and blacks
- Use green text for money/positive themes, red text for danger/warning themes, white text on dark backgrounds
- Simple, uncluttered composition — only the character, one financial visual metaphor, and the big text. Nothing else.
- The scene must depict a financial/economic scenario — NOT a living room, NOT a TV studio, NOT a home interior
- Aspect ratio 16:9, resolution 1280x720

Output Rules:
- First line: EXACT_TEXT: followed by the 2-4 word ALL-CAPS overlay text you chose
- Second line: the full image generation prompt as a single paragraph
- Nothing else — no explanations or meta commentary

Example output format:
EXACT_TEXT: EARN MORE GET LESS
A bold adult animated cartoon illustration for a YouTube thumbnail featuring..."""


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

    # Reinforce the exact text at the start of the image prompt
    if overlay_text:
        image_prompt = (
            f'IMPORTANT: The large text overlay in this image must say '
            f'exactly "{overlay_text}" in bold ALL-CAPS. Do not use any '
            f'other text. The text must be massive and fill 40-50% of the '
            f'image area.\n\n{image_prompt}'
        )

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
