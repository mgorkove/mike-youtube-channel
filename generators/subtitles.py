"""Subtitle (SRT) generation using Whisper speech recognition.

Transcribes the TTS audio with faster-whisper to get accurate word-level
timestamps, then groups words into subtitle chunks.
"""

import logging
from pathlib import Path

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# Maximum words per subtitle line
MAX_WORDS_PER_SUB = 10

# Whisper model size — "base" is fast and accurate enough for clean TTS audio
WHISPER_MODEL = "base"


def _format_srt_time(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def transcribe_words(audio_path: Path) -> list[tuple[str, float, float]]:
    """Transcribe audio and return word-level timestamps.

    Returns a list of (word, start_seconds, end_seconds) tuples.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    logger.info(f"Transcribing audio for word timestamps: {audio_path}")
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    segments, _info = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        language="en",
    )

    words = []
    for segment in segments:
        if segment.words:
            for w in segment.words:
                words.append((w.word.strip(), w.start, w.end))

    if not words:
        raise ValueError("Whisper produced no words from the audio")

    logger.info(f"Transcribed {len(words)} words with timestamps")
    return words


def generate_srt(
    script: str,
    audio_duration: float,
    output_path: Path,
) -> Path:
    """Generate an SRT subtitle file by transcribing the audio.

    Uses faster-whisper to get word-level timestamps from the TTS audio,
    then groups words into subtitle chunks of up to MAX_WORDS_PER_SUB words.
    """
    audio_path = output_path.parent / "audio.wav"
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    logger.info(f"Transcribing audio for subtitles: {audio_path}")
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    segments, _info = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        language="en",
    )

    # Collect all words with timestamps
    words = []
    for segment in segments:
        if segment.words:
            for w in segment.words:
                words.append((w.word.strip(), w.start, w.end))

    if not words:
        raise ValueError("Whisper produced no words from the audio")

    logger.info(f"Transcribed {len(words)} words with timestamps")

    # Group words into subtitle chunks
    srt_lines = []
    sub_index = 1
    i = 0

    while i < len(words):
        chunk_words = words[i:i + MAX_WORDS_PER_SUB]
        text = " ".join(w[0] for w in chunk_words)
        start = chunk_words[0][1]
        end = chunk_words[-1][2]

        srt_lines.append(f"{sub_index}")
        srt_lines.append(f"{_format_srt_time(start)} --> {_format_srt_time(end)}")
        srt_lines.append(text)
        srt_lines.append("")

        sub_index += 1
        i += MAX_WORDS_PER_SUB

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(srt_lines), encoding="utf-8")
    logger.info(f"Generated {sub_index - 1} subtitles → {output_path}")
    return output_path
