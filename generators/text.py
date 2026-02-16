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

    prompt = f"""You are a YouTube content strategist for a finance education channel.

Channel theme: {config.channel_theme}

Generate exactly {count} unique video topic ideas. Each topic should:
- Either directly affect the viewer's personal finances OR be aspirational (e.g., "how the wealthy do X", "what changes when you reach X")
- NEVER be about abstract institutional mechanics that don't connect to the individual (no dark pools, repo markets, correspondent banking, etc.)
- Be specific enough to write a full script about
- Target curiosity gaps that would interest a general audience
- Focus on hidden mechanisms, thresholds, or behaviors in financial systems that impact real people
{dedup_block}
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

    prompt = f"""You are writing YouTube video titles for a faceless finance channel focused on money, banking, and power dynamics.

Topic: "{topic}"

Generate ONE YouTube title that:
- Targets a 25–34 year old audience interested in money, banking, wealth, and financial systems
- Contains words and phrases people actually search for on YouTube (think: what would someone type?)
- Creates a curiosity gap — the viewer should feel they NEED to click to understand
- Sounds analytical and insider-knowledge, not motivational or influencer-style
- Is descriptive, not prescriptive (no "how to get rich", no advice)
- Avoids hype words like secrets, hacks, tips, passive income, financial freedom
- Avoids direct commands or promises
- Is a complete statement — do NOT end with a colon, dash, or ellipsis
- Keep under 65 characters when possible

Good title patterns (mix these, don't always use the same one):
"How the Top 1% Use X as Y"
"Why Banks Treat X Differently After Y"
"What Happens to Your Money When X"
"The Real Reason X Changes at $Y"
"Why X Is the Most Misunderstood Thing in Finance"
"How X Actually Works Behind the Scenes"

Use concrete numbers when they add punch (e.g., $100K, $1M, 1%).
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
