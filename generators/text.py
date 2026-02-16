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


def generate_topics(
    count: int,
    config: Config,
    client: genai.Client,
    existing_titles: list[str] | None = None,
) -> list[str]:
    """Generate a list of video topic ideas matching the channel theme."""
    dedup_block = ""
    if existing_titles:
        titles_list = "\n".join(f"- {t}" for t in existing_titles)
        dedup_block = f"""

IMPORTANT: The channel already has videos on these topics. Do NOT generate topics that overlap with or are too similar to any of these existing titles:
{titles_list}

Each new topic must be clearly distinct from all of the above."""

    prompt = f"""You are a YouTube content strategist for a finance education channel with 25-34 year old viewers.

Channel theme: {config.channel_theme}

Generate exactly {count} unique video topic ideas. Each topic should:
- Be something a real person would type into YouTube search or click on in their feed
- Focus on money milestones, wealth-building behaviors, or financial turning points that people actually care about
- Use specific dollar amounts, ages, or percentages when possible (e.g., "$100K", "before 40", "top 1%")
- Frame topics around what wealthy/successful people DO or what HAPPENS at certain thresholds — not abstract theory
- Mix these types: (1) aspirational "how millionaires/wealthy people do X", (2) milestone-based "what changes at $X", (3) eye-opening stats/data, (4) current events and trending financial news
- NEVER be about abstract institutional mechanics that don't connect to the individual (no dark pools, repo markets, correspondent banking, etc.)
- Be specific enough to write a full 20-minute script about
{dedup_block}
Return ONLY a JSON array of topic strings, nothing else. Example:
["How people actually retire before 40", "Why everything changes after your first $100K", "How millionaires use debt to avoid paying taxes"]"""

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


def generate_title(
    topic: str,
    config: Config,
    client: genai.Client,
    existing_titles: list[str] | None = None,
) -> str:
    """Generate a high-CTR YouTube title for the given topic."""
    dedup_block = ""
    if existing_titles:
        titles_list = "\n".join(f"- {t}" for t in existing_titles)
        dedup_block = f"""

IMPORTANT: The channel already has these video titles. Your new title must NOT be the same as or too similar to any of them:
{titles_list}
"""

    prompt = f"""You are writing YouTube video titles for a finance education channel. The titles need to GET CLICKS.

Topic: "{topic}"

Generate ONE YouTube title that:
- Makes someone scrolling YouTube STOP and click — create a strong curiosity gap
- Contains words and phrases people actually search for on YouTube
- Uses specific numbers, dollar amounts, or percentages when they add punch ($100K, $1M, 1%, "before 40")
- Can describe what wealthy people or millionaires DO, or what HAPPENS at certain financial thresholds
- Can use third-person framing: "How millionaires...", "How people retire before 40...", "Why the wealthy..."
- Do NOT give direct advice or commands (no "do this", "stop doing X", "you need to")
- Avoids hype words: secrets, hacks, tips, passive income, financial freedom
- Is a complete statement — do NOT end with a colon, dash, or ellipsis
- Keep under 65 characters when possible
- Use ONE or TWO words in ALL CAPS for emphasis when it feels natural (not every title)

High-performing title patterns (vary these):
"How Millionaires Use X to Y"
"Why EVERYTHING Changes After $X"
"How People Actually Retire Before X"
"The REAL Reason X Happens at $Y"
"What Happens to Your Money When X"
"How Many Americans Actually Have $X Saved"
"Why the Wealthy Never X (And What They Do Instead)"
"X Jaw-Dropping Money Stats Most People Don't Know"

Questions work well too:
"How Many People ACTUALLY Retire With $1M?"
"What Happens When You Save $X a Month for Y Years?"
{dedup_block}
Return ONLY the title text, nothing else. No quotes, no explanation."""

    response = client.models.generate_content(
        model=config.text_model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=config.text_model_temperature,
            max_output_tokens=1024,
        ),
    )
    return response.text.strip().strip('"').strip("'")


def generate_script(topic: str, config: Config, client: genai.Client) -> str:
    """Generate a 3500-4500 word YouTube script using the prompt template."""
    prompt = config.script_generation_prompt.replace("[TOPIC]", topic)

    banned_list = ", ".join(f'"{p}"' for p in config.banned_phrases)

    # Add channel context
    prompt += f"""

Channel context: {config.channel_theme}

CRITICAL RULES:
- The script must be between {config.script_min_words} and {config.script_max_words} words.
- ABSOLUTELY FORBIDDEN — do NOT use any of these phrases anywhere in the script: {banned_list}
  These are prescriptive phrases. This channel is observational and analytical, NOT advisory.
  Replace any urge to write prescriptive language with third-person observations like:
    "what tends to happen is…", "the pattern that emerges…", "people in this position often…",
    "the data suggests…", "institutions typically respond by…"
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
    script = response.text.strip()

    # Post-process: remove any banned phrases that slipped through
    for phrase in config.banned_phrases:
        script = script.replace(phrase, "")
        script = script.replace(phrase.capitalize(), "")
        script = script.replace(phrase.title(), "")

    return script


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
1. Start with a strong 2-3 sentence hook that makes people want to watch. Front-load the most searchable keywords.
2. Add a blank line, then a "In this video:" section with 4-6 bullet points summarizing key topics covered. Each bullet should contain searchable phrases people would type into YouTube.
3. Add a blank line, then a "Key topics:" line listing 5-8 comma-separated keyword phrases relevant to this specific video (these help YouTube's search algorithm).
4. Add a blank line, then 5-8 relevant hashtags.
5. End with this exact disclaimer (copy it verbatim):

{config.disclaimer}

Naturally weave these keywords into the hook paragraph: {keywords_str}

Return ONLY the description text, ready to paste into YouTube."""

    response = client.models.generate_content(
        model=config.text_model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=2048,
        ),
    )
    return response.text.strip()


def generate_tags(
    topic: str,
    title: str,
    config: Config,
    client: genai.Client,
) -> list[str]:
    """Generate topic-specific YouTube tags for a video."""
    default_tags = ", ".join(config.youtube_tags)

    prompt = f"""You are a YouTube SEO specialist for a finance education channel.

Video title: "{title}"
Topic: "{topic}"

Generate 15-20 YouTube tags that will help this specific video rank in search. Include:
- Exact phrases people would type into YouTube search to find this video
- Long-tail keyword variations (3-5 word phrases)
- Related subtopics and questions viewers might search
- Do NOT include single generic words like "money" or "finance" — those are already covered

The channel already uses these default tags: {default_tags}
Do NOT repeat any of those. Only generate NEW tags specific to this video's topic.

Return ONLY a JSON array of tag strings, nothing else. Example:
["life insurance wealth strategy", "infinite banking concept explained", "whole life insurance cash value", "how rich people use life insurance"]"""

    response = client.models.generate_content(
        model=config.text_model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=2048,
        ),
    )

    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        text = text.strip()

    tags = json.loads(text)
    if not isinstance(tags, list):
        raise ValueError(f"Expected JSON array of tags, got: {type(tags)}")

    # YouTube allows max 500 characters total for tags; trim if needed
    result = []
    total_len = 0
    for tag in tags:
        tag = tag.strip()
        if total_len + len(tag) + 1 > 480:  # leave room for default tags
            break
        result.append(tag)
        total_len += len(tag) + 1
    return result


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
- Uses a bold, colorful cartoon illustration style (like an animated explainer video)
- Specifies setting and composition
- Do NOT use the words "photorealistic" or "realistic" anywhere in the prompts

Return ONLY a JSON array of exactly {num_images} prompt strings. Example:
["A cartoon man in a business suit standing confidently in a modern bank lobby, bold colorful cartoon illustration style", "The same cartoon man reviewing financial documents at a desk with city skyline visible through windows, vibrant cartoon style"]"""

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
