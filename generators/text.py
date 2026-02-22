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

    prompt = f"""You are a YouTube content strategist. Your job is to generate video topics that match this channel's theme and will get clicks.

Channel theme: {config.channel_theme}

Generate exactly {count} unique video topic ideas. Each topic should:
- Be something people would type into YouTube search or click on in their feed
- Match the channel's theme and tone exactly
- Be specific enough to write a full script about
- Create curiosity or emotional pull — the viewer should NEED to know what happens
- NEVER reference specific years (no "in 2024", "in 2025", "in 2026") — keep topics evergreen
- Each topic should be distinct and cover a different angle or scenario
{dedup_block}
Return ONLY a JSON array of topic strings, nothing else."""

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

    if config.title_generation_prompt:
        # Use channel-specific title prompt
        prompt = config.title_generation_prompt.replace("[TOPIC]", topic)
        prompt += dedup_block
    else:
        # Default generic title prompt
        prompt = f"""You are writing YouTube video titles. The titles need to GET CLICKS.

Channel theme: {config.channel_theme}

Topic: "{topic}"

Generate ONE YouTube title that:
- Makes someone scrolling YouTube STOP and click — create a strong curiosity gap
- Contains words and phrases people actually search for on YouTube
- Matches the channel's tone and style
- Is a complete statement
- Keep under 70 characters when possible
- Use ONE or TWO words in ALL CAPS for emphasis when it feels natural
- Do NOT reference specific years (no "in 2024", "in 2025", "in 2026") — keep titles evergreen
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
    """Generate a YouTube script using the channel's prompt template."""
    prompt = config.script_generation_prompt.replace("[TOPIC]", topic)

    # Add channel context and word count constraints
    prompt += f"""

Channel context: {config.channel_theme}

CRITICAL RULES:
- The script must be between {config.script_min_words} and {config.script_max_words} words.
- Write the script as a voiceover narration — no stage directions, no [brackets], just the spoken words."""

    # Add banned phrases if configured
    if config.banned_phrases:
        banned_list = ", ".join(f'"{p}"' for p in config.banned_phrases)
        prompt += f"""
- ABSOLUTELY FORBIDDEN — do NOT use any of these phrases anywhere in the script: {banned_list}"""

    prompt += """

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

    if config.description_generation_prompt:
        # Use channel-specific description prompt
        prompt = config.description_generation_prompt
        prompt = prompt.replace("[TITLE]", title)
        prompt = prompt.replace("[TOPIC]", topic)
        prompt = prompt.replace("[DISCLAIMER]", config.disclaimer)
    else:
        # Default generic description prompt
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

    prompt = f"""You are a YouTube SEO specialist.

Channel theme: {config.channel_theme}

Video title: "{title}"
Topic: "{topic}"

Generate 10-15 YouTube tags that will help this specific video rank in search. Include:
- Exact phrases people would type into YouTube search to find this video
- Short keyword phrases (2-4 words each)
- EVERY tag MUST be 30 characters or fewer — YouTube rejects longer tags
- Related subtopics and questions viewers might search

The channel already uses these default tags: {default_tags}
Do NOT repeat any of those. Only generate NEW tags specific to this video's topic.

Return ONLY a JSON array of tag strings, nothing else."""

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


def extract_stock_footage_queries(
    script: str,
    num_clips: int,
    config: Config,
    client: genai.Client,
) -> list[str]:
    """Split the script into segments and generate Pexels search queries for each."""
    prompt = f"""You are a visual director for a YouTube video. Given this script, divide it into exactly {num_clips} sequential visual segments and generate a short stock footage search query for each one.

SCRIPT:
{script}

For each segment, write a 2-5 word search query that would find relevant stock footage on Pexels.com. Think about:
- What visual scene best represents what's being narrated?
- Use concrete, searchable terms (e.g., "couple arguing kitchen", "woman texting phone night", "man crying alone")
- Avoid abstract concepts — search for tangible scenes, actions, and emotions
- Each query should be visually distinct from the others

Return ONLY a JSON array of exactly {num_clips} search query strings. Example:
["couple arguing kitchen", "woman texting secretly night", "man discovering phone messages", "divorce papers table"]"""

    response = client.models.generate_content(
        model=config.text_model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=4096,
        ),
    )

    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        text = text.strip()

    queries = json.loads(text)
    if not isinstance(queries, list):
        raise ValueError(f"Expected JSON array, got: {type(queries)}")

    # Pad or truncate to exact count
    if len(queries) < num_clips:
        logger.warning(f"Got {len(queries)} queries, expected {num_clips}. Padding.")
        while len(queries) < num_clips:
            queries.append(queries[-1])
    elif len(queries) > num_clips:
        queries = queries[:num_clips]

    return queries


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
