"""Text generation via Gemini 3 Flash.

Handles: topic generation, titles, scripts, descriptions, and image prompts.
"""

import json
import logging
from pathlib import Path

from google import genai
from google.genai import types

from config_loader import Config

logger = logging.getLogger(__name__)


def generate_topics(count: int, config: Config, client: genai.Client) -> list[str]:
    """Generate a list of video topic ideas matching the channel theme."""
    prompt = f"""You are a YouTube content strategist for a finance education channel.

Channel theme: {config.channel_theme}

Generate exactly {count} unique video topic ideas. Each topic should:
- Address a specific, intriguing aspect of money, banking, or financial systems
- Be specific enough to write a full script about
- Target curiosity gaps that would interest a general audience
- Focus on hidden mechanisms, thresholds, or behaviors in financial systems

Return ONLY a JSON array of topic strings, nothing else. Example:
["How Banks Decide Who Gets Special Treatment", "The Hidden Math Behind Credit Card Minimum Payments"]"""

    response = client.models.generate_content(
        model=config.text_model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=config.text_model_temperature,
            max_output_tokens=2048,
        ),
    )

    text = response.text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        text = text.strip()

    topics = json.loads(text)
    if not isinstance(topics, list) or len(topics) < count:
        raise ValueError(f"Expected {count} topics, got: {topics}")
    return topics[:count]


def generate_title(topic: str, config: Config, client: genai.Client) -> str:
    """Generate a high-CTR YouTube title for the given topic."""
    prompt = f"""You are a YouTube title expert specializing in finance content.

Channel theme: {config.channel_theme}

Write ONE YouTube title for this topic: "{topic}"

Requirements:
- 40-80 characters
- Create a curiosity gap — make viewers feel they NEED to know
- Use specific numbers or thresholds when relevant
- Avoid clickbait that doesn't deliver (no "SHOCKING" or emoji)
- Style examples:
  "Net Worth Levels Where Rules Quietly Change"
  "Why Banks Treat You Differently After This Number"
  "The System Rewards This Type of Wealth"

Return ONLY the title text, nothing else. No quotes, no explanation."""

    response = client.models.generate_content(
        model=config.text_model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=config.text_model_temperature,
            max_output_tokens=256,
        ),
    )
    return response.text.strip().strip('"').strip("'")


def generate_script(topic: str, config: Config, client: genai.Client) -> str:
    """Generate a 3500-4500 word YouTube script using the prompt template."""
    prompt = config.script_generation_prompt.replace("[TOPIC]", topic)

    # Add channel context
    prompt += f"""

Channel context: {config.channel_theme}

CRITICAL RULES:
- The script must be between {config.script_min_words} and {config.script_max_words} words.
- NEVER use prescriptive language like "you should", "you need to", "I recommend", "you must", "my advice", "I suggest", or "you have to".
- Instead use observations: "what tends to happen is...", "the research shows...", "people in this situation often..."
- Write the script as a voiceover narration — no stage directions, no [brackets], just the spoken words.
- Include a subscribe CTA early in the script (within the first 500 words).
- Close by reinforcing that this information is rarely discussed and why.

Write the full script now."""

    response = client.models.generate_content(
        model=config.text_model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=config.text_model_temperature,
            max_output_tokens=config.text_model_max_tokens,
        ),
    )
    return response.text.strip()


def generate_description(
    topic: str,
    title: str,
    script: str,
    config: Config,
    client: genai.Client,
) -> str:
    """Generate a YouTube description with SEO keywords and disclaimer."""
    keywords_str = ", ".join(config.required_keywords)

    prompt = f"""Write a YouTube video description for this video:

Title: "{title}"
Topic: "{topic}"

Requirements:
1. Start with 2-3 engaging sentences summarizing what the viewer will learn.
2. Naturally weave in ALL of these keywords: {keywords_str}
3. Add a "What you'll learn:" section with 3-5 bullet points.
4. Include relevant hashtags at the end.
5. End with this exact disclaimer (copy it exactly):

{config.disclaimer}

The description should be 300-600 words. Return ONLY the description text, ready to paste into YouTube."""

    response = client.models.generate_content(
        model=config.text_model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=2048,
        ),
    )
    return response.text.strip()


def extract_image_prompts(
    script: str,
    num_images: int,
    config: Config,
    client: genai.Client,
) -> list[str]:
    """Split the script into visual segments and generate image prompts."""
    prompt = f"""You are a visual director for a YouTube finance education channel.

Given this script, divide it into exactly {num_images} sequential visual segments and create an image generation prompt for each one.

SCRIPT:
{script}

For each segment, write a detailed image generation prompt that:
- Describes a specific, concrete visual scene (not abstract concepts)
- Features a man (the host/narrator) acting out or reacting to what's being discussed
- If the script mentions buying something, show him buying it
- If the script discusses banks, show him in a bank setting
- If the script talks about wealth, show visual markers of wealth
- Uses cinematic, photorealistic style suitable for a YouTube video
- Specifies lighting, setting, and composition

Return ONLY a JSON array of exactly {num_images} prompt strings. Example:
["A man in a business suit standing confidently in a modern bank lobby, cinematic lighting, photorealistic", "The same man reviewing financial documents at a desk with city skyline visible through windows, warm lighting"]"""

    response = client.models.generate_content(
        model=config.text_model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=8192,
        ),
    )

    text = response.text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        text = text.strip()

    prompts = json.loads(text)
    if not isinstance(prompts, list):
        raise ValueError(f"Expected JSON array of image prompts, got: {type(prompts)}")

    # Pad or truncate to exact count
    if len(prompts) < num_images:
        logger.warning(f"Got {len(prompts)} prompts, expected {num_images}. Padding with last prompt.")
        while len(prompts) < num_images:
            prompts.append(prompts[-1])
    elif len(prompts) > num_images:
        prompts = prompts[:num_images]

    return prompts
