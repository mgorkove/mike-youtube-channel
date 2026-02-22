"""Subtitle (SRT) generation from script text.

Splits the script into short subtitle chunks and distributes them
proportionally across the audio duration.
"""

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Maximum words per subtitle line
MAX_WORDS_PER_SUB = 10


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using punctuation boundaries."""
    # Split on sentence-ending punctuation followed by whitespace
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter empty strings
    return [s.strip() for s in parts if s.strip()]


def _chunk_sentences(sentences: list[str], max_words: int = MAX_WORDS_PER_SUB) -> list[str]:
    """Break long sentences into shorter subtitle chunks."""
    chunks = []
    for sentence in sentences:
        words = sentence.split()
        if len(words) <= max_words:
            chunks.append(sentence)
        else:
            # Split into chunks of max_words
            for i in range(0, len(words), max_words):
                chunk = " ".join(words[i:i + max_words])
                chunks.append(chunk)
    return chunks


def _format_srt_time(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_srt(
    script: str,
    audio_duration: float,
    output_path: Path,
) -> Path:
    """Generate an SRT subtitle file from a script.

    Splits the script into short chunks and distributes them evenly
    across the audio duration, proportional to word count.
    """
    sentences = _split_into_sentences(script)
    chunks = _chunk_sentences(sentences)

    if not chunks:
        raise ValueError("Script produced no subtitle chunks")

    # Calculate total words for proportional timing
    total_words = sum(len(c.split()) for c in chunks)
    if total_words == 0:
        raise ValueError("Script has no words for subtitle generation")

    # Distribute time proportionally by word count
    srt_lines = []
    current_time = 0.0

    for i, chunk in enumerate(chunks):
        word_count = len(chunk.split())
        duration = (word_count / total_words) * audio_duration

        start_time = current_time
        end_time = current_time + duration
        current_time = end_time

        srt_lines.append(f"{i + 1}")
        srt_lines.append(f"{_format_srt_time(start_time)} --> {_format_srt_time(end_time)}")
        srt_lines.append(chunk)
        srt_lines.append("")  # blank line separator

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(srt_lines), encoding="utf-8")
    logger.info(f"Generated {len(chunks)} subtitles → {output_path}")
    return output_path
