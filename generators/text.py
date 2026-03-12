"""Text generation via Gemini 3 Flash.

Handles: topic generation, titles, scripts, descriptions, and image prompts.
"""

import json
import logging
import re
from pathlib import Path

from google import genai
from google.genai import types

from config_loader import Config

logger = logging.getLogger(__name__)

# Grammatically wrong "Made Her [PAST TENSE]" patterns → fix to base form.
# e.g. "Made Her PANICKED" → "Made Her PANIC"
_MADE_HER_FIXES = {
    "PANICKED": "PANIC",
    "COLLAPSED": "COLLAPSE",
    "SCREAMED": "SCREAM",
    "REGRETTED": "REGRET",
}
_MADE_HER_PATTERN = re.compile(
    r"(Made Her )(" + "|".join(_MADE_HER_FIXES.keys()) + r")\b",
    re.IGNORECASE,
)


def _fix_title_grammar(title: str) -> str:
    """Fix common grammatical errors in generated titles.

    Catches "Made Her PANICKED" → "Made Her PANIC" etc.
    """
    def _replace(m: re.Match) -> str:
        prefix = m.group(1)
        word = m.group(2)
        fixed = _MADE_HER_FIXES.get(word.upper(), word)
        # Preserve original casing style (all-caps vs title-case)
        if word.isupper():
            fixed = fixed.upper()
        return prefix + fixed

    return _MADE_HER_PATTERN.sub(_replace, title)


def _extract_json_array(text: str) -> list:
    """Robustly extract a JSON array from LLM output.

    Handles markdown fences, trailing commas, and unterminated strings.
    """
    # Strip markdown code fences
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if "```" in text:
            text = text[: text.rfind("```")]
        text = text.strip()

    # Try direct parse first
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Extract the JSON array substring (first [ to last ])
    match = re.search(r"\[", text)
    if match:
        start = match.start()
        # Find matching closing bracket
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        # Fix trailing commas before ]
                        fixed = re.sub(r",\s*]", "]", candidate)
                        try:
                            return json.loads(fixed)
                        except json.JSONDecodeError:
                            pass
                    break

    # Try to extract individual JSON objects from the text (handles
    # arrays of objects where the outer array is malformed but each
    # object is valid).
    objects = []
    for obj_match in re.finditer(r"\{[^{}]*\}", text):
        try:
            obj = json.loads(obj_match.group())
            if isinstance(obj, dict):
                objects.append(obj)
        except json.JSONDecodeError:
            pass
    if objects:
        logger.warning(
            f"JSON array parse failed, recovered {len(objects)} objects via regex"
        )
        return objects

    # Last resort: find all quoted strings and build the array
    strings = re.findall(r'"((?:[^"\\]|\\.)*)"', text)
    if strings:
        logger.warning(f"JSON parse failed, extracted {len(strings)} strings via regex")
        return strings

    raise ValueError(f"Could not extract JSON array from LLM output: {text[:200]}...")


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

    topic_guidance = ""
    if config.topic_generation_prompt:
        topic_guidance = f"\n\n{config.topic_generation_prompt}\n"

    prompt = f"""You are a YouTube content strategist. Your job is to generate video topics that match this channel's theme and will get clicks.

Channel theme: {config.channel_theme}
{topic_guidance}
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

    topics = _extract_json_array(response.text.strip())
    if not topics:
        raise ValueError("Got 0 topics from LLM")

    # If the LLM returned fewer topics than requested (common with malformed
    # JSON where some strings can't be extracted), generate the remaining ones
    # individually rather than failing the whole pipeline.
    while len(topics) < count:
        logger.warning(f"Only got {len(topics)}/{count} topics, generating 1 more...")
        extra_resp = client.models.generate_content(
            model=config.text_model_name,
            contents=(
                f"{prompt}\n\nGenerate exactly 1 topic. "
                f"Do NOT repeat any of these: {topics}"
            ),
            config=types.GenerateContentConfig(
                temperature=config.text_model_temperature,
                max_output_tokens=1024,
            ),
        )
        try:
            extra = _extract_json_array(extra_resp.text.strip())
        except ValueError:
            extra = []
        if extra:
            topics.append(extra[0])
        else:
            # If even a single-topic request fails, use the raw text
            raw = extra_resp.text.strip().strip('"').strip("'").strip("[]")
            if raw:
                topics.append(raw)

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
    title = response.text.strip().strip('"').strip("'")
    title = _fix_title_grammar(title)
    return title


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

    tags = _extract_json_array(response.text.strip())

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

    queries = _extract_json_array(response.text.strip())

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
    prompt = f"""You are a visual director for a YouTube channel.

Channel theme: {config.channel_theme}

Given this script, divide it into exactly {num_images} sequential visual segments and create an image generation prompt for each one.

SCRIPT:
{script}

For each segment, write a detailed image generation prompt that:
- Describes a specific, concrete visual scene (not abstract concepts)
- Shows characters, settings, and actions that match what's being narrated
- Each scene should be visually distinct from the others (different locations, characters, gear, lighting)
- Specifies the setting, character appearance, clothing/gear, and composition
- Uses a muted, desaturated cartoon art style with soft shading and clean outlines
- Color palette should be subdued and earthy: muted greens, grays, tans, olive tones. NO bright or vibrant colors.
- Do NOT use the words "photorealistic" or "realistic" anywhere in the prompts
- Do NOT include any text or watermarks in the image descriptions

Return ONLY a JSON array of exactly {num_images} prompt strings. Example:
["A muted, desaturated cartoon illustration of a person in uniform standing in a detailed environment, soft shading with clean outlines, earthy muted color palette", "A muted, desaturated cartoon illustration of a figure in a different setting performing an action, subdued lighting and muted olive-gray tones"]"""

    response = client.models.generate_content(
        model=config.text_model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=8192,
        ),
    )

    prompts = _extract_json_array(response.text.strip())

    # Pad or truncate to exact count
    if len(prompts) < num_images:
        logger.warning(f"Got {len(prompts)} prompts, expected {num_images}. Padding with last prompt.")
        while len(prompts) < num_images:
            prompts.append(prompts[-1])
    elif len(prompts) > num_images:
        prompts = prompts[:num_images]

    return prompts


def extract_image_prompts_with_segments(
    script: str,
    num_images: int,
    config: Config,
    client: genai.Client,
) -> list[dict]:
    """Split the script into natural visual segments with image prompts.

    Returns a list of dicts with 'prompt' (image generation prompt) and
    'segment' (the corresponding narration text).  Segment word counts
    are used downstream to calculate proportional display durations so
    that image transitions align with the narration.
    """
    prompt = f"""You are a visual director for a YouTube channel.

Channel theme: {config.channel_theme}

Given this script, divide it into approximately {num_images} sequential visual segments. Split at NATURAL break points — when the topic shifts, a new scene begins, or a new rank/stage/character is introduced. Do NOT split mid-sentence.

Most segments should cover roughly 2-3 seconds of narration, but it's fine for some to be shorter (1-2 seconds) or longer (up to 5 seconds) if the content naturally calls for it. Prioritize transitions that feel right over hitting an exact count.

SCRIPT:
{script}

For each segment, return:
- "prompt": a detailed image generation prompt for that segment
- "segment": the EXACT script text (copy-pasted verbatim) that this image covers

Image prompt guidelines:
- Describes a specific, concrete visual scene (not abstract concepts)
- Shows characters, settings, and actions that match what's being narrated
- Each scene should be visually distinct from the others (different locations, characters, gear, lighting)
- Specifies the setting, character appearance, clothing/gear, and composition
- Uses a muted, desaturated cartoon art style with soft shading and clean outlines
- Color palette should be subdued and earthy: muted greens, grays, tans, olive tones. NO bright or vibrant colors.
- Do NOT use the words "photorealistic" or "realistic" anywhere in the prompts
- Do NOT include any text or watermarks in the image descriptions

Return ONLY a JSON array of objects. Example:
[{{"prompt": "A muted, desaturated cartoon illustration of a young recruit stepping off a bus at a military base, dawn light, earthy muted tones", "segment": "You're 18. You just stepped off the bus at Fort Benning."}}, {{"prompt": "A muted, desaturated cartoon illustration of soldiers running an obstacle course, subdued lighting and muted olive-gray palette", "segment": "The drill sergeant is already screaming. You hit the mud and start crawling."}}]"""

    response = client.models.generate_content(
        model=config.text_model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=16384,
        ),
    )

    segments = _extract_json_array(response.text.strip())

    # Validate structure — each item must be a dict with prompt + segment
    valid = []
    plain_string_count = 0
    for item in segments:
        if isinstance(item, dict) and "prompt" in item and "segment" in item:
            # Filter out entries where the "prompt" value is just the key name
            if item["prompt"].strip().lower() in ("prompt", ""):
                continue
            valid.append(item)
        elif isinstance(item, str):
            plain_string_count += 1
            # Skip bare key names that the regex fallback may have captured
            if item.strip().lower() in ("prompt", "segment", ""):
                continue
            valid.append({"prompt": item, "segment": ""})

    if not valid:
        raise ValueError("LLM returned no valid image prompt segments")

    # If most items came back as plain strings, segments are missing —
    # raise so the retry logic can re-attempt and get proper objects.
    has_segments = sum(1 for v in valid if v["segment"].strip())
    if has_segments == 0 and plain_string_count > 0:
        raise ValueError(
            f"LLM returned {plain_string_count} plain strings instead of "
            f"prompt+segment objects — segment timing data is missing. "
            f"Retrying to get structured output."
        )

    logger.info(
        f"Extracted {len(valid)} image segments "
        f"(target was ~{num_images}, {has_segments} with segment text)"
    )
    return valid
