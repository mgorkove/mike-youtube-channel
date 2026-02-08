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

    # Image generation
    image_model: str
    reference_image_path: Path
    seconds_per_image: int
    image_width: int
    image_height: int
    image_aspect_ratio: str

    # Thumbnail
    thumbnail_width: int
    thumbnail_height: int
    thumbnail_min_contrast: float

    # Video
    video_fps: int
    ken_burns_ratio: float
    video_codec: str
    audio_codec: str
    video_bitrate: str

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

    # Dry run (skip upload)
    dry_run: bool = False


def load_config(config_path: str = "config.yaml") -> Config:
    """Load and validate configuration from YAML + .env files."""
    load_dotenv()

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file) as f:
        raw = yaml.safe_load(f)

    # Load script generation prompt
    prompt_path = config_file.parent / "script_generation_prompt.md"
    script_prompt = ""
    if prompt_path.exists():
        script_prompt = prompt_path.read_text(encoding="utf-8").strip()

    # Normalize the disclaimer (collapse whitespace from YAML multiline)
    disclaimer = " ".join(raw["channel"]["disclaimer"].split())

    ref_image = Path(raw["image_gen"]["reference_image"])
    if not ref_image.exists():
        raise FileNotFoundError(f"Reference image not found: {ref_image}")

    config = Config(
        topics=raw.get("topics", []),
        video_count=raw.get("video_count", 1),
        channel_theme=" ".join(raw["channel"]["theme"].split()),
        required_keywords=raw["channel"]["required_keywords"],
        disclaimer=disclaimer,
        script_min_words=raw["script"]["min_words"],
        script_max_words=raw["script"]["max_words"],
        banned_phrases=raw["script"]["banned_phrases"],
        text_model_name=raw["text_model"]["name"],
        text_model_temperature=raw["text_model"]["temperature"],
        text_model_max_tokens=raw["text_model"]["max_output_tokens"],
        tts_model=raw["tts"]["model"],
        tts_voice=raw["tts"]["voice"],
        image_model=raw["image_gen"]["model"],
        reference_image_path=ref_image,
        seconds_per_image=raw["image_gen"]["seconds_per_image"],
        image_width=raw["image_gen"]["image_width"],
        image_height=raw["image_gen"]["image_height"],
        image_aspect_ratio=raw["image_gen"].get("aspect_ratio", "16:9"),
        thumbnail_width=raw["thumbnail"]["width"],
        thumbnail_height=raw["thumbnail"]["height"],
        thumbnail_min_contrast=raw["thumbnail"]["min_contrast_ratio"],
        video_fps=raw["video"]["fps"],
        ken_burns_ratio=raw["video"]["ken_burns_ratio"],
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
    )

    return config
