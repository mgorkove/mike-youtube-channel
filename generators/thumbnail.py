"""Thumbnail generation via Gemini.

Uses a two-step process:
1. Gemini text model generates an optimized image prompt based on the
   title/topic using a thumbnail strategist system prompt.
2. Gemini image model generates the final thumbnail with text baked in.

Style: American animated sitcom cartoon (Family Guy-like), bold typography,
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

Non-Negotiable Requirements:
- The man from the reference image MUST be present and clearly visible
- He must be rendered as a cartoon version of the reference image in American animated sitcom style (like Family Guy or similar adult animated shows)
- Facial features and hairstyle should remain recognizable
- Facial expression should be exaggerated (shock, worry, realization, disbelief, concern, tension)
- Thick black outlines around the character

Strategy Instructions:
1. Identify the emotional hook implied by the title (fear, surprise, imbalance, loss, urgency).
2. Choose one dramatic visual metaphor that amplifies this emotion (cracking dollar sign, collapsing graph, tipping scale, sinking ship, broken clock, vault snapping shut, cliff edge, system fracture, bank building, money stacks).
3. Place the reference character in direct interaction with the metaphor (reacting to it, pointing at it, standing in front of it, running from it).
4. Use contrast and scale to exaggerate stakes (oversized symbols, steep arrows, broken elements).
5. Add 2–5 words of bold ALL-CAPS text that creates curiosity or tension without giving answers.

Thumbnail Style (must be embedded in the output prompt):
- American animated sitcom cartoon style (like Family Guy, American Dad) — NOT flat vector art
- Thick black outlines, rounded shapes, smooth shading
- Exaggerated facial expression and body language
- Background can be light OR dark depending on the mood — use dark/dramatic backgrounds for fear/urgency topics, lighter backgrounds for informational/aspirational topics
- Large, heavy sans-serif ALL-CAPS typography — text should be very prominent and take up a significant portion of the thumbnail
- High contrast color palette: bold greens, reds, whites, and blacks
- Use green text for money/positive themes, red text for danger/warning themes, white text on dark backgrounds
- Simple, uncluttered composition — character on one side, text on the other, or text overlaid on top
- No photorealism
- No 3D rendering
- Optimized for mobile viewing (text must be readable at small sizes)
- Aspect ratio 16:9, resolution 1280x720

Output Rules:
- Output only one image generation prompt
- Single paragraph
- No explanations or meta commentary"""


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
    image_prompt = prompt_response.text.strip()
    logger.info(f"Thumbnail prompt: {image_prompt[:120]}...")

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
