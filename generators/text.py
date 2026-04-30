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


# Patterns that indicate the model leaked its reasoning/drafting process
# instead of returning just the title.
_REASONING_PREFIX = re.compile(
    r"^(?:"
    r"Draft\s*\d+[^:]*:|"            # "Draft 8 (notes):"
    r"(?:If|Let me|I (?:need|should|want|think|will))\b[^—\"]*[.!]\s*|"  # "If I use X, I'm revealing... "
    r"Here(?:'s| is)[^:]*:\s*|"       # "Here's the title:"
    r"Title:\s*|"                      # "Title: ..."
    r"Final (?:title|version)[^:]*:\s*"  # "Final title:"
    r")",
    re.IGNORECASE,
)


def _clean_title(raw: str) -> str:
    """Extract the actual title from model output, stripping reasoning."""
    text = raw.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        text = text.strip()

    # If multi-line, the model likely included reasoning.  Take the last
    # non-empty line that looks like a title (starts with a quote, "I ", "My ",
    # or an uppercase word).
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) > 1:
        for line in reversed(lines):
            cleaned = line.strip().strip('"').strip("'").strip("*")
            # Looks like a real title if it starts with common story openers
            if re.match(r'^(?:[""\']|I |My |At |The |Her |His |We |He |She |")', cleaned):
                text = cleaned
                break
        else:
            # Fallback: use last line
            text = lines[-1]

    # Strip known reasoning prefixes
    text = _REASONING_PREFIX.sub("", text).strip()

    # Remove surrounding quotes, asterisks
    text = text.strip('"').strip("'").strip("*").strip('"').strip("'")

    return text


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
    max_extra_attempts = count * 3
    extra_attempts = 0
    while len(topics) < count:
        if extra_attempts >= max_extra_attempts:
            logger.error(f"Gave up after {max_extra_attempts} attempts to fill {count} topics (got {len(topics)})")
            break
        extra_attempts += 1
        logger.warning(f"Only got {len(topics)}/{count} topics, generating 1 more (attempt {extra_attempts}/{max_extra_attempts})...")
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
    title = _clean_title(response.text)
    title = _fix_title_grammar(title)
    if len(title) > 100:
        title = title[:97] + "..."
    return title


def generate_title_with_emv(
    topic: str,
    config: Config,
    client: genai.Client,
    existing_titles: list[str] | None = None,
    candidate_count: int = 5,
    min_score: float = 40.0,
    max_rounds: int = 3,
) -> str:
    """Generate multiple title candidates, score them via EMV, and pick the best.

    Generates `candidate_count` titles per round. If no title scores above
    `min_score`, regenerates up to `max_rounds` total. Returns the highest
    scoring title across all rounds.
    """
    from emv_score import get_emv_score
    import time

    best_title = None
    best_score = -1.0

    for round_num in range(1, max_rounds + 1):
        logger.info(f"EMV title selection round {round_num}/{max_rounds}: generating {candidate_count} candidates...")
        candidates = _generate_title_candidates(topic, config, client, existing_titles, candidate_count)

        for title in candidates:
            try:
                result = get_emv_score(title)
                score = result["score"]
                logger.info(f"  EMV {score:5.1f}% [{result['classification']:12s}] {title}")
            except Exception:
                logger.warning(f"  EMV scoring failed for: {title}, skipping")
                score = 0.0
            if score > best_score:
                best_score = score
                best_title = title
            time.sleep(1.0)  # polite delay between AMI requests

        if best_score >= min_score:
            logger.info(f"EMV winner ({best_score:.1f}%): {best_title}")
            return best_title

        logger.info(f"No title above {min_score}% EMV (best so far: {best_score:.1f}%). {'Retrying...' if round_num < max_rounds else 'Using best available.'}")

    logger.info(f"EMV final pick ({best_score:.1f}%): {best_title}")
    return best_title


def _generate_title_candidates(
    topic: str,
    config: Config,
    client: genai.Client,
    existing_titles: list[str] | None = None,
    count: int = 5,
) -> list[str]:
    """Generate multiple title candidates in a single LLM call."""
    dedup_block = ""
    if existing_titles:
        titles_list = "\n".join(f"- {t}" for t in existing_titles)
        dedup_block = f"""

IMPORTANT: The channel already has these video titles. Your new titles must NOT be the same as or too similar to any of them:
{titles_list}
"""

    if config.title_generation_prompt:
        # Adapt the channel-specific prompt to ask for multiple titles
        base_prompt = config.title_generation_prompt.replace("[TOPIC]", topic)
        # Replace the "Return ONLY the title" instruction with multi-title instruction
        base_prompt = re.sub(
            r"Return ONLY the title.*$",
            "",
            base_prompt,
            flags=re.IGNORECASE | re.DOTALL,
        ).rstrip()
        prompt = base_prompt + f"""
{dedup_block}
Generate {count} different title options, each using a DIFFERENT formula/angle.
Return them as a numbered list (1. ... 2. ... etc.), one per line. No quotes, no explanation."""
    else:
        prompt = f"""You are writing YouTube video titles. The titles need to GET CLICKS.

Channel theme: {config.channel_theme}

Topic: "{topic}"

Generate {count} YouTube title options, each using a different angle or hook. Each title should:
- Make someone scrolling YouTube STOP and click — create a strong curiosity gap
- Contains words and phrases people actually search for on YouTube
- Matches the channel's tone and style
- Is a complete statement
- Keep under 70 characters when possible
- Use ONE or TWO words in ALL CAPS for emphasis when it feels natural
- Do NOT reference specific years (no "in 2024", "in 2025", "in 2026") — keep titles evergreen
{dedup_block}
Return them as a numbered list (1. ... 2. ... etc.), one per line. No quotes, no explanation."""

    response = client.models.generate_content(
        model=config.text_model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=config.text_model_temperature,
            max_output_tokens=4096,  # thinking models need headroom for multiple titles
        ),
    )

    # Parse numbered list from response
    raw_lines = response.text.strip().splitlines()
    titles = []
    for line in raw_lines:
        # Match "1. Title here" or "1) Title here"
        m = re.match(r"^\s*\d+[\.\)]\s*(.+)", line)
        if m:
            title = _clean_title(m.group(1))
            title = _fix_title_grammar(title)
            if len(title) > 100:
                title = title[:97] + "..."
            if title:
                titles.append(title)

    if not titles:
        # Fallback: treat entire response as a single title
        title = _clean_title(response.text)
        title = _fix_title_grammar(title)
        if len(title) > 100:
            title = title[:97] + "..."
        titles = [title]

    logger.info(f"Generated {len(titles)} title candidates")
    return titles


def generate_script(topic: str, config: Config, client: genai.Client, title: str | None = None) -> str:
    """Generate a YouTube script using the channel's prompt template."""
    prompt = config.script_generation_prompt.replace("[TOPIC]", topic)

    # Add channel context and word count constraints
    prompt += f"""

Channel context: {config.channel_theme}

CRITICAL RULES:
- The script must be between {config.script_min_words} and {config.script_max_words} words.
- Write the script as a voiceover narration — no stage directions, no [brackets], just the spoken words."""

    if title:
        prompt += f"""
- The video title is: "{title}" — the script MUST match the angle and hook promised by this title. Build the narrative around it."""

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


def extract_satisfying_photo_prompts(
    topic: str,
    num_images: int,
    config: Config,
    client: genai.Client,
) -> list[str]:
    """Generate ~num_images unique 'perfectly satisfying' photo prompts.

    Each prompt describes a single hyper-real photograph whose satisfaction
    derives from geometry, vanishing-point perspective, repetition, or
    surprising vantage point — not from the literal subject. Output format
    is a single self-contained image-gen prompt per entry.
    """
    archetypes = (
        # Architecture / vantage / landscape
        "vanishing-point corridors (escalators, hallways, tunnels), "
        "aerial / bird's-eye geometric patterns (salt pans, terraced fields, parking grids, suburbs), "
        "ground-level rows of trees / vines / crops converging to the horizon (lavender, olives, tulips), "
        "natural tunnels (cherry blossom, wisteria, ivy, ice cave), "
        "mirror reflections (flooded rice terraces, salt flats, calm lakes), "
        "perfect facade grids (apartment windows at dusk, library bookshelves, tile walls), "
        "worm's-eye and bird's-eye views of staircases, atriums, escalators, organ pipes, "
        "color-block minimal compositions (Mediterranean alleys, painted walls), "
        "dead-on symmetrical architecture (cathedrals, temples, courtyards), "
        # Knolling / order / alignment
        "knolling — everyday objects laid out flat in a perfect grid (tools, art supplies, hardware, kitchen utensils), "
        "objects perfectly fitting other objects (eggs in a carton, tetris-like packing, pencils in a box), "
        "color-graded shelves and racks (paint chip rainbows, sock wall, sneaker rainbow, book spines), "
        "fresh untouched supplies (new pencils sharpened in spectrum order, sealed packaging, factory rows), "
        # Cuts / cross-sections
        "perfectly clean cross-sections (fruit, bread, soap, stone, layered cake, pomegranate, kiwi, agate), "
        "hydraulic-press / knife-through-butter style perfect cuts (kinetic sand cube cut, soap bar shaving), "
        "geode and crystal cross-sections, internal symmetry revealed, "
        # Liquid physics / macro
        "slow-motion liquid pours and droplets (honey, paint, milk, coffee crema, ink in water), "
        "water droplets on surfaces (leaves, glass, hydrophobic fabric, spider web jewels), "
        "ripple rings and crown splashes on the moment of impact, "
        # Nature patterns / macro
        "fibonacci spirals and fractal repetition (sunflower seeds, pinecone scales, pineapple, romanesco broccoli), "
        "frost and snowflake crystalline geometry macro, "
        "cracked-mud polygons, basalt columns, dried lakebed tiles, honeycombs, "
        "dewdrops ringing a leaf edge, raindrops on flowers, "
        # Fresh / pristine
        "fresh snow with a single perfect set of tracks, fresh-cut grass with mower stripes, "
        "freshly painted road lines, tennis court chalk, raked Zen sand gardens, untouched beach ripples, "
        # Color gradients
        "sunset gradient bands of pure color, paint-chip gradients, tile-glaze gradients, "
        "aerial color-graded farmland (tulip fields, lavender, canola, terraced rice), "
        # Food aesthetics
        "top-down food knolling (bento boxes, charcuterie, mise-en-place ingredients), "
        "pristine ice cream scoops, perfect espresso crema, latte art, layered pastries, "
        # Stacking / repetition
        "stacked firewood, stacked pottery, stacked bricks, shipping container yards, "
        "identical hot-air balloons in formation, lifeguard towers in a row, beach umbrellas in pattern, "
        # POV / hands-at-work / interaction
        "first-person POV looking down at one's own feet on a striking surface (metro grate, tile mosaic, painted crosswalk, fresh snow, raked sand, koi pond bridge), "
        "first-person POV of two hands at work (kneading dough, peeling an orange in one continuous strip, pouring honey, arranging objects in a grid, raking sand, cutting kinetic sand), "
        "anonymous figure photographed entirely from BEHIND walking through a satisfying scene (lavender row, cherry blossom tunnel, library spiral, salt flat, cathedral nave, neon-lit Tokyo alley)"
    )

    examples = (
        # Architecture
        '"Looking directly upward from the bottom of a very long escalator, chrome handrails and black rubber steps converging to a single bright point, symmetrical, underground metro, cool blue-silver light, ultra sharp, geometric"',
        '"Aerial drone photograph of geometric salt evaporation ponds, each rectangle a different shade of pink coral rust and white, razor-thin dikes dividing them, abstract, hyper-real colors"',
        '"Walking path canopied by arching cherry blossom trees forming a natural tunnel, petals falling in still air, path receding to a bright vanishing point, soft diffused spring light, Japan, romantic, ultra sharp foreground"',
        # Knolling / order
        '"Top-down knolling shot of a complete vintage tool set laid out on a dark walnut workbench, each tool spaced exactly equal distance apart, brass and steel reflecting warm overhead light, tack sharp, professional product photography"',
        '"Dead-on photograph of a wall of paint chip cards arranged as a continuous color spectrum from coral red to deep indigo, hardware store shelf, perfectly even lighting, flat composition, hyper-real colors"',
        # Cuts / cross-sections
        '"Macro photograph of a single perfect knife cut through a bar of pastel pink soap, the slice peeling back to reveal a pristine glossy interior, white seamless background, soft directional studio light, ultra sharp"',
        '"Cross-section macro of a halved kiwi held at the center of frame, jet-black seeds arranged in a perfect ring around the pale green star, dewy translucent flesh, white background, food photography, hyper-real"',
        # Liquid physics
        '"High-speed photograph of a single milk droplet hitting a black coffee surface, frozen in time as a perfect liquid coronet with seven beads on its rim, dark espresso-brown background, dramatic side light, ultra sharp"',
        '"Macro photograph of a fresh leaf rimmed with perfectly spaced clear water droplets along every serration of its edge, soft green out-of-focus background, overcast morning light, hyper-real, nature photography"',
        # Nature patterns / macro
        '"Top-down macro of a sunflower head in full bloom, the seeds spiraling outward in perfect Fibonacci geometry, deep mustard-yellow petals around the rim, late summer afternoon light, ultra sharp, hyper-real"',
        '"Aerial drone photograph of a dried lakebed cracked into a vast geometric mosaic of pale tan polygons, each tile a slightly different size, harsh midday sun raking shadows into the cracks, abstract, hyper-real"',
        # Fresh / pristine
        '"Aerial photograph of a freshly mown stadium lawn, alternating dark and light green stripes running diagonally across the frame in mathematically perfect bands, late golden afternoon, hyper-real, sports photography"',
        '"Top-down photograph of a Zen rock garden raked into perfect concentric ripples around three dark stones, white pebble background, soft diffused overcast light, Japan, minimalist, ultra sharp"',
        # Gradients
        '"Wide aerial drone photograph of a Dutch tulip field at peak bloom, ribbons of pure red, yellow, pink, and indigo planted in straight rows running to the horizon, golden hour, hyper-real saturated colors, cinematic"',
        # Food aesthetics
        '"Dead-on top-down photograph of a Japanese bento box, every compartment filled with a different food in a different geometric shape, vivid color contrast, soft diffused overhead light, food photography, ultra sharp"',
        # Stacking / repetition
        '"Aerial photograph of a row of fifty identical brightly colored hot air balloons floating at the same altitude over a misty Cappadocia valley at dawn, each balloon a different solid hue, hyper-real, cinematic"',
        # POV / hands-at-work / figure-from-behind
        '"First-person POV looking straight down at the photographer\'s own black sneakers standing on a metro station ventilation grate, polished steel bars receding in perfect parallel stripes around the feet, cool fluorescent light, ultra sharp, hyper-real"',
        '"First-person POV of two hands peeling a single orange in one continuous unbroken spiral over a marble countertop, thin curling rind dangling, soft daylight from a window, food photography, hyper-real, ultra sharp"',
        '"Photograph from directly behind a lone anonymous figure in a dark wool coat walking down the center aisle of a vast library between towering parallel bookshelves, reader unaware of camera, warm tungsten light, architectural photography, ultra sharp"',
        '"First-person POV looking down at the photographer\'s feet planted on either side of a freshly raked Zen garden line, white pebbles around the feet etched into perfect parallel grooves, soft overcast morning light, Japan, minimalist, hyper-real"',
    )
    examples_block = "\n".join(f"- {e}" for e in examples)

    prompt = f"""You are a photo-director writing image-generation prompts
for a YouTube Shorts channel of "perfectly satisfying" PHOTOGRAPHS — not
illustrations, not cartoons, not paintings. Each Short is a slideshow of
{num_images} still photographs held 2 seconds each.

The loose theme of this video is: "{topic}" — use it as a starting bias,
but the {num_images} photos must SPAN MULTIPLE oddly-satisfying
categories so a viewer scrolling past sees variety, not 28 of the same
shot. Aim to cover at least 6 of these categories across the {num_images}
prompts:

  1. ARCHITECTURE / VANTAGE POINT (vanishing-point corridors,
     symmetrical facades, worm's-eye atria)
  2. AERIAL / LANDSCAPE GEOMETRY (salt pans, terraced fields, tulip
     fields, suburb grids)
  3. KNOLLING / ALIGNMENT / ORDER (tools laid flat, color-graded
     shelves, perfect packing)
  4. PERFECT CUTS / CROSS-SECTIONS (fruit, soap, bread, geode,
     hydraulic press)
  5. LIQUID PHYSICS / DROPLETS (milk-coronet, ink in water, latte art,
     dewdrops on a leaf edge)
  6. NATURE PATTERNS / MACRO (fibonacci, fractal, frost, honeycomb,
     cracked mud, basalt columns)
  7. FRESH / PRISTINE STATE (untouched snow, mower stripes, raked
     Zen sand, fresh paint lines)
  8. COLOR GRADIENTS (sunset bands, paint-chip gradients, book-spine
     spectrum, tulip field stripes)
  9. FOOD AESTHETICS (top-down bento, espresso crema, layered cake
     cross-section, charcuterie knolling)
 10. STACKING / REPETITION (firewood, hot-air balloons, beach umbrellas,
     shipping containers, organ pipes)
 11. POV / HANDS-AT-WORK / FIGURE-FROM-BEHIND
     - First-person looking down at one's own feet on a striking surface
       (metro grate, fresh snow, raked Zen sand, mosaic floor, painted
       crosswalk, koi-pond bridge)
     - First-person two-handed shots of work in progress (peeling fruit
       in one strip, kneading dough, pouring honey, arranging objects,
       cutting kinetic sand, raking sand)
     - Anonymous figure photographed entirely from BEHIND, walking
       through a satisfying scene (lavender row, cherry tunnel, library
       spiral, salt flat, cathedral nave, neon Tokyo alley)
     IMPORTANT for category 11: at LEAST 3 of the {num_images} prompts
     should fall in this category — viewers strongly engage with POV.

Reference archetypes you can pull from:
{archetypes}.

OUTPUT FORMAT — match these EXACT examples in style and length. One
self-contained image prompt per entry, comma-separated clauses:

{examples_block}

EVERY prompt MUST contain in this order:
1. Vantage point ("Looking straight up...", "Aerial drone view...", "Dead-on symmetrical...", "Ground-level shot...", "Directly overhead...", "Worm's-eye view...")
2. Concrete subject (escalator, lavender row, library atrium, salt ponds, cherry tunnel, apartment facade, greenhouse, rice terraces, etc.)
3. The geometric pattern that does the work ("converging to a vanishing point", "concentric rings", "grid", "spiral", "mirror reflection", "razor-thin dividing lines")
4. Lighting cue ("golden hour", "blue hour", "dawn", "overcast", "soft diffused light", "harsh midday", "warm amber")
5. Location flavor ("Provence France", "Japan", "Bali", "Iceland", "Tokyo metro", "Mediterranean coast", "Southern Europe") — never copyrighted landmark names (no Eiffel Tower / Burj Khalifa / Times Square)
6. Photographic style cues at the end: "ultra sharp", "cinematic", "hyper-real colors", "architectural photography", "professional photography"

HARD RULES:
- NO illustrations, NO cartoons, NO paintings, NO renders, NO 3D — every prompt must read as describing a photograph.
- NO close-up human FACES. (Hands, feet, and anonymous figures from behind ARE allowed and encouraged for POV / hands-at-work shots — that's category 11.)
- NO text, signs, logos, or watermarks in the image.
- NO listicles inside a single prompt — exactly ONE photograph per entry.
- Do NOT use the words "satisfying" or "perfect" inside the prompts.
- Each of the {num_images} prompts must be visually DISTINCT — different archetype, different palette, different setting. Cycle through the categories; don't make 28 escalators.

Return ONLY a JSON array of exactly {num_images} prompt strings."""

    response = client.models.generate_content(
        model=config.text_model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=config.text_model_temperature,
            max_output_tokens=16384,
        ),
    )

    prompts = _extract_json_array(response.text.strip())

    # Pad or truncate to exact count
    if len(prompts) < num_images:
        logger.warning(
            f"Got {len(prompts)} satisfying-photo prompts, expected {num_images}. "
            f"Padding by repeating earlier prompts."
        )
        i = 0
        while len(prompts) < num_images:
            prompts.append(prompts[i % max(1, len(prompts))])
            i += 1
    elif len(prompts) > num_images:
        prompts = prompts[:num_images]

    return prompts


def extract_catalog_image_prompts(
    script: str,
    config: Config,
    client: genai.Client,
) -> list[str]:
    """Extract image prompts for catalog-style scripts with [GUN]/[VEHICLE] markers.

    Parses the script to find each cataloged item, then asks the LLM to
    produce a replacement instruction for each item. Each prompt is prefixed
    with [GUN] or [VEHICLE] so the image generator picks the correct
    reference template.
    """
    prompt = f"""You are a visual director for a military equipment catalog YouTube channel.

The following script describes military items, each marked with [GUN] or [VEHICLE].

SCRIPT:
{script}

For each item in the script (marked with [GUN] or [VEHICLE]), generate an image prompt that instructs an image model to modify a reference template image.

The reference images show a single military item on a clean background with:
- The item name as a label in the top-left
- "RELEASE DATE: [year]" in the top-right

For [GUN] items (firearms, weapons), the reference shows a gun. Your prompt should instruct the model to change it to the specific weapon.
For [VEHICLE] items (tanks, aircraft, ships, etc.), the reference shows a fighter jet. Your prompt should instruct the model to change it to the specific vehicle.

For each item, produce a prompt in this EXACT format:
"[TYPE] Change the [gun/vehicle] to a [ITEM DESCRIPTION]. Change the release date to [YEAR]. Change the label to "[ITEM NAME]". Do not change anything else in the image."

Where:
- [TYPE] is either [GUN] or [VEHICLE], matching the script marker
- For [GUN] items: use "Change the gun to a..." with a short description (e.g., "a rifle", "a submachine gun", "a heavy machine gun")
- For [VEHICLE] items: use "Change the vehicle to a..." with a short description (e.g., "a main battle tank", "a stealth fighter jet", "a nuclear submarine")
- [YEAR] is the year of introduction/service mentioned in the script
- [ITEM NAME] is the common name of the weapon/vehicle (e.g., "AK-47 Rifle", "M1 Abrams Tank", "F-22 Raptor")

Return ONLY a JSON array of prompt strings, one per item, in the same order they appear in the script. Example:
["[GUN] Change the gun to a rifle. Change the release date to 1964. Change the label to \"M16 Rifle\". Do not change anything else in the image.", "[VEHICLE] Change the vehicle to a main battle tank. Change the release date to 1980. Change the label to \"M1 Abrams\". Do not change anything else in the image."]"""

    response = client.models.generate_content(
        model=config.text_model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=16384,
        ),
    )

    prompts = _extract_json_array(response.text.strip())

    if not prompts:
        raise ValueError("LLM returned no catalog image prompts")

    logger.info(f"Extracted {len(prompts)} catalog image prompts")
    return prompts
