"""Quality validation functions for every pipeline stage.

All functions are pure — no API calls, no side effects.
Each returns a CheckResult indicating pass/fail with a message.
"""

import json
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass
class CheckResult:
    passed: bool
    message: str


# ---------------------------------------------------------------------------
# Script checks
# ---------------------------------------------------------------------------


def check_word_count(script: str, min_words: int, max_words: int) -> CheckResult:
    """Verify script word count is within the configured range."""
    count = len(script.split())
    if min_words <= count <= max_words:
        return CheckResult(True, f"Word count {count} is within [{min_words}, {max_words}]")
    return CheckResult(False, f"Word count {count} outside [{min_words}, {max_words}]")


def check_banned_phrases(script: str, banned: list[str]) -> CheckResult:
    """Check that the script contains no prescriptive language."""
    script_lower = script.lower()
    found = [p for p in banned if p.lower() in script_lower]
    if not found:
        return CheckResult(True, "No banned phrases found")
    return CheckResult(False, f"Banned phrases found: {found}")


# ---------------------------------------------------------------------------
# Title checks
# ---------------------------------------------------------------------------


def check_title_length(title: str, min_len: int = 20, max_len: int = 80) -> CheckResult:
    """Verify title length is within range and title is complete."""
    length = len(title)
    if not (min_len <= length <= max_len):
        return CheckResult(False, f"Title length {length} outside [{min_len}, {max_len}]")

    # Reject titles that look incomplete (end with colon, dash, ellipsis)
    if title.rstrip().endswith((":", " -", " —", "...", " |")):
        return CheckResult(False, f"Title appears incomplete (ends with trailing punctuation): '{title}'")

    # Reject titles that are just a label/prefix
    if ":" in title and len(title.split(":")[-1].strip()) < 5:
        return CheckResult(False, f"Title appears to be a fragment with short/empty suffix: '{title}'")

    return CheckResult(True, f"Title length {length} OK")


# ---------------------------------------------------------------------------
# Description checks
# ---------------------------------------------------------------------------


def check_keywords_present(description: str, keywords: list[str]) -> CheckResult:
    """Verify all required keywords appear in the description."""
    desc_lower = description.lower()
    missing = [kw for kw in keywords if kw.lower() not in desc_lower]
    if not missing:
        return CheckResult(True, "All required keywords present")
    return CheckResult(False, f"Missing keywords: {missing}")


def check_disclaimer_present(description: str, disclaimer: str) -> CheckResult:
    """Verify the full disclaimer text is present in the description."""
    norm_desc = " ".join(description.split()).lower()
    norm_disc = " ".join(disclaimer.split()).lower()
    if norm_disc in norm_desc:
        return CheckResult(True, "Disclaimer found in description")
    return CheckResult(False, "Disclaimer not found in description")


# ---------------------------------------------------------------------------
# Image checks
# ---------------------------------------------------------------------------


def check_image_exists_and_dimensions(
    path: Path, expected_w: int, expected_h: int
) -> CheckResult:
    """Verify image exists and has the correct dimensions."""
    if not path.exists():
        return CheckResult(False, f"Image not found: {path}")
    img = Image.open(path)
    w, h = img.size
    if w == expected_w and h == expected_h:
        return CheckResult(True, f"Image {path.name}: {w}x{h} OK")
    return CheckResult(False, f"Image {path.name}: expected {expected_w}x{expected_h}, got {w}x{h}")


# ---------------------------------------------------------------------------
# Thumbnail checks
# ---------------------------------------------------------------------------


def _relative_luminance(r: int, g: int, b: int) -> float:
    """Compute WCAG 2.0 relative luminance from sRGB values."""
    def linearize(c: int) -> float:
        c_norm = c / 255.0
        if c_norm <= 0.04045:
            return c_norm / 12.92
        return ((c_norm + 0.055) / 1.055) ** 2.4

    return 0.2126 * linearize(r) + 0.7152 * linearize(g) + 0.0722 * linearize(b)


def check_contrast_ratio(thumbnail_path: Path, min_ratio: float) -> CheckResult:
    """Check that the center region has enough contrast for text legibility.

    Samples the center 60% of the thumbnail, computes luminance distribution,
    and checks that the contrast between the darkest and brightest quartiles
    meets the WCAG 2.0 AA standard.
    """
    if not thumbnail_path.exists():
        return CheckResult(False, f"Thumbnail not found: {thumbnail_path}")

    img = Image.open(thumbnail_path).convert("RGB")
    w, h = img.size

    # Sample center region where text overlay is placed
    center = img.crop((int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8)))
    pixels = list(center.getdata())

    luminances = [_relative_luminance(r, g, b) for r, g, b in pixels]
    luminances.sort()

    n = len(luminances)
    q = max(n // 4, 1)
    dark_avg = sum(luminances[:q]) / q
    light_avg = sum(luminances[-q:]) / q

    l1 = max(light_avg, dark_avg)
    l2 = min(light_avg, dark_avg)
    ratio = (l1 + 0.05) / (l2 + 0.05)

    if ratio >= min_ratio:
        return CheckResult(True, f"Contrast ratio {ratio:.2f}:1 >= {min_ratio}:1")
    return CheckResult(False, f"Contrast ratio {ratio:.2f}:1 < {min_ratio}:1")


# ---------------------------------------------------------------------------
# Audio checks
# ---------------------------------------------------------------------------


def check_audio_file(audio_path: Path, script_word_count: int) -> CheckResult:
    """Verify audio file exists and has a reasonable duration for the word count."""
    if not audio_path.exists():
        return CheckResult(False, f"Audio file not found: {audio_path}")

    with wave.open(str(audio_path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        duration = frames / rate

    if duration <= 0:
        return CheckResult(False, "Audio duration is 0")

    # ~2.5 words/sec average speaking rate
    expected = script_word_count / 2.5
    if expected * 0.5 <= duration <= expected * 1.5:
        return CheckResult(
            True,
            f"Audio duration {duration:.1f}s reasonable for {script_word_count} words",
        )
    return CheckResult(
        False,
        f"Audio duration {duration:.1f}s suspicious for {script_word_count} words "
        f"(expected ~{expected:.0f}s)",
    )


# ---------------------------------------------------------------------------
# Video checks
# ---------------------------------------------------------------------------


def check_video_file(video_path: Path) -> CheckResult:
    """Verify final video exists and has both audio+video streams."""
    if not video_path.exists():
        return CheckResult(False, f"Video not found: {video_path}")
    if video_path.stat().st_size == 0:
        return CheckResult(False, "Video file is empty")

    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", "-show_format", str(video_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return CheckResult(False, f"ffprobe failed: {result.stderr}")

    probe = json.loads(result.stdout)
    stream_types = [s["codec_type"] for s in probe.get("streams", [])]

    if "video" not in stream_types:
        return CheckResult(False, "No video stream found")
    if "audio" not in stream_types:
        return CheckResult(False, "No audio stream found")

    return CheckResult(True, "Video file has both video and audio streams")
