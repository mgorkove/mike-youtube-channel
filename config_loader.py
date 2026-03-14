from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv


@dataclass
class Config:
    """Immutable configuration loaded from config.yaml and .env."""

    # Topics (can be empty — pipeline will auto-generate)
    topics: list[str]
    video_count: int

    # Video mode: "ken_burns", "stock_footage", "slideshow", or "static_image"
    video_mode: str

    # Channel
    channel_theme: str
    required_keywords: list[str]
    disclaimer: str

    # Script
    script_min_words: int
    script_max_words: int
    banned_phrases: list[str]

    # Text model
    text_model_name: str
    text_model_temperature: float
    text_model_max_tokens: int

    # TTS
    tts_model: str
    tts_voice: str

    # Image generation (ken_burns mode)
    image_model: str
    seconds_per_image: int
    image_width: int
    image_height: int
    image_aspect_ratio: str
    reference_image_path: Path | None

    # Thumbnail
    thumbnail_model: str
    thumbnail_width: int
    thumbnail_height: int
    thumbnail_min_contrast: float

    # Video
    video_fps: int
    video_codec: str
    audio_codec: str
    video_bitrate: str
    ken_burns_ratio: float

    # YouTube
    youtube_client_secrets: str
    youtube_token_file: str
    youtube_category_id: str
    youtube_privacy_status: str
    youtube_tags: list[str]

    # Retry
    retry_max_attempts: int
    retry_base_delay: float
    retry_max_delay: float

    # Output
    output_base_dir: Path

    # Script generation prompt
    script_generation_prompt: str

    # Thumbnail strategist prompt
    thumbnail_strategist_prompt: str

    # Scheduling
    publish_timezone: str
    publish_times: list[list[int]]

    # Parallelism
    max_parallel_videos: int
    render_workers: int

    # Cleanup
    cleanup_after_upload: bool

    # --- Fields with defaults below ---

    # Title generation prompt (optional — channel-specific title formula)
    title_generation_prompt: str = ""

    # Description generation prompt (optional — channel-specific description style)
    description_generation_prompt: str = ""

    # Topic generation prompt (optional — channel-specific topic guidance)
    topic_generation_prompt: str = ""

    # Thumbnail text overlay (if true, overlay text with Pillow instead of AI rendering)
    thumbnail_text_overlay: bool = False
    thumbnail_font_path: str = "assets/Anton-Regular.ttf"
    thumbnail_fixed_image_prompt: str = ""

    # Stock footage settings (stock_footage mode)
    seconds_per_clip: int = 10
    subtitle_font_size: int = 56
    subtitle_margin_v: int = 40

    # Static image background (static_image mode)
    background_image_path: Path | None = None
    shorts_background_image_path: Path | None = None

    # Manual titles (one per topic, in order; empty = auto-generate)
    titles: list[str] = None

    # Target video length in seconds (None = use script word count range from config)
    target_video_length: int | None = None

    # Dry run (skip upload)
    dry_run: bool = False

    # Skip quality checks (title length, banned phrases, contrast, audio, image dimensions)
    skip_quality_checks: bool = False

    # Skip shorts generation and upload
    skip_shorts: bool = False

    # Schedule videos for today instead of next Monday
    schedule_same_day: bool = False


def load_config(config_path: str = "config.yaml") -> Config:
    """Load and validate configuration from YAML + .env files."""
    load_dotenv()

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file) as f:
        raw = yaml.safe_load(f)

    config_dir = config_file.parent

    video_mode = raw.get("video_mode", "ken_burns")

    # Load script generation prompt (next to config file)
    prompt_path = config_dir / "script_generation_prompt.md"
    script_prompt = ""
    if prompt_path.exists():
        script_prompt = prompt_path.read_text(encoding="utf-8").strip()

    # Load thumbnail strategist prompt (next to config file)
    thumb_prompt_path = config_dir / "thumbnail_prompt.md"
    thumb_prompt = ""
    if thumb_prompt_path.exists():
        thumb_prompt = thumb_prompt_path.read_text(encoding="utf-8").strip()

    # Load title generation prompt (optional, next to config file)
    title_prompt_path = config_dir / "title_generation_prompt.md"
    title_prompt = ""
    if title_prompt_path.exists():
        title_prompt = title_prompt_path.read_text(encoding="utf-8").strip()

    # Load description generation prompt (optional, next to config file)
    desc_prompt_path = config_dir / "description_generation_prompt.md"
    desc_prompt = ""
    if desc_prompt_path.exists():
        desc_prompt = desc_prompt_path.read_text(encoding="utf-8").strip()

    # Load topic generation prompt (optional, next to config file)
    topic_prompt_path = config_dir / "topic_generation_prompt.md"
    topic_prompt = ""
    if topic_prompt_path.exists():
        topic_prompt = topic_prompt_path.read_text(encoding="utf-8").strip()

    # Normalize the disclaimer (collapse whitespace from YAML multiline)
    disclaimer = " ".join(raw["channel"]["disclaimer"].split())

    # Reference image (optional for stock_footage mode)
    ref_image = None
    image_gen = raw.get("image_gen", {})
    if image_gen.get("reference_image"):
        ref_image = Path(image_gen["reference_image"])
        if not ref_image.exists():
            raise FileNotFoundError(f"Reference image not found: {ref_image}")

    # Background image (static_image mode)
    bg_image = None
    if raw.get("background_image"):
        bg_image = config_dir / raw["background_image"]
        if not bg_image.exists():
            raise FileNotFoundError(f"Background image not found: {bg_image}")

    shorts_bg_image = None
    if raw.get("shorts_background_image"):
        shorts_bg_image = config_dir / raw["shorts_background_image"]
        if not shorts_bg_image.exists():
            raise FileNotFoundError(f"Shorts background image not found: {shorts_bg_image}")

    config = Config(
        topics=raw.get("topics", []),
        video_count=raw.get("video_count", 1),
        video_mode=video_mode,
        channel_theme=" ".join(raw["channel"]["theme"].split()),
        required_keywords=raw["channel"]["required_keywords"],
        disclaimer=disclaimer,
        script_min_words=raw["script"]["min_words"],
        script_max_words=raw["script"]["max_words"],
        banned_phrases=raw["script"].get("banned_phrases", []),
        text_model_name=raw["text_model"]["name"],
        text_model_temperature=raw["text_model"]["temperature"],
        text_model_max_tokens=raw["text_model"]["max_output_tokens"],
        tts_model=raw["tts"]["model"],
        tts_voice=raw["tts"]["voice"],
        image_model=image_gen.get("model", "gemini-2.5-flash-image"),
        reference_image_path=ref_image,
        seconds_per_image=image_gen.get("seconds_per_image", 9),
        image_width=image_gen.get("image_width", 1920),
        image_height=image_gen.get("image_height", 1080),
        image_aspect_ratio=image_gen.get("aspect_ratio", "16:9"),
        thumbnail_model=raw["thumbnail"].get("model", image_gen.get("model", "gemini-2.5-flash-image")),
        thumbnail_width=raw["thumbnail"]["width"],
        thumbnail_height=raw["thumbnail"]["height"],
        thumbnail_min_contrast=raw["thumbnail"]["min_contrast_ratio"],
        video_fps=raw["video"]["fps"],
        ken_burns_ratio=raw["video"].get("ken_burns_ratio", 0.04),
        video_codec=raw["video"]["codec"],
        audio_codec=raw["video"]["audio_codec"],
        video_bitrate=raw["video"]["bitrate"],
        youtube_client_secrets=raw["youtube"]["client_secrets_file"],
        youtube_token_file=raw["youtube"]["token_file"],
        youtube_category_id=raw["youtube"]["category_id"],
        youtube_privacy_status=raw["youtube"]["privacy_status"],
        youtube_tags=raw["youtube"]["tags"],
        retry_max_attempts=raw["retry"]["max_attempts"],
        retry_base_delay=raw["retry"]["base_delay_seconds"],
        retry_max_delay=raw["retry"]["max_delay_seconds"],
        output_base_dir=Path(raw["output"]["base_dir"]),
        script_generation_prompt=script_prompt,
        thumbnail_strategist_prompt=thumb_prompt,
        title_generation_prompt=title_prompt,
        description_generation_prompt=desc_prompt,
        topic_generation_prompt=topic_prompt,
        publish_timezone=raw.get("scheduling", {}).get("timezone", "America/New_York"),
        publish_times=raw.get("scheduling", {}).get("publish_times", [[8, 0], [18, 0]]),
        max_parallel_videos=raw.get("max_parallel_videos", 1),
        render_workers=raw.get("render_workers", 4),
        cleanup_after_upload=raw.get("cleanup_after_upload", False),
        seconds_per_clip=raw.get("stock_footage", {}).get("seconds_per_clip", 10),
        subtitle_font_size=raw.get("subtitles", {}).get("font_size", 56),
        subtitle_margin_v=raw.get("subtitles", {}).get("margin_v", 40),
        thumbnail_text_overlay=raw.get("thumbnail", {}).get("text_overlay", False),
        thumbnail_font_path=raw.get("thumbnail", {}).get("font_path", "assets/Anton-Regular.ttf"),
        thumbnail_fixed_image_prompt=raw.get("thumbnail", {}).get("fixed_image_prompt", ""),
        background_image_path=bg_image,
        shorts_background_image_path=shorts_bg_image,
        skip_quality_checks=raw.get("skip_quality_checks", False),
        skip_shorts=raw.get("skip_shorts", False),
        schedule_same_day=raw.get("schedule_same_day", False),
    )

    return config
