"""Text-to-Speech via Gemini 2.5 Flash Lite Preview TTS.

Converts script text to WAV audio, chunking long scripts to stay within
API limits and concatenating the PCM data into a single file.
"""

import io
import logging
import wave
from dataclasses import dataclass
from pathlib import Path

from google import genai
from google.genai import types

from config_loader import Config

logger = logging.getLogger(__name__)

# Gemini TTS output: 24kHz, mono, 16-bit PCM
SAMPLE_RATE = 24000
SAMPLE_WIDTH = 2  # 16-bit = 2 bytes
CHANNELS = 1

# Max characters per TTS chunk (conservative limit)
MAX_CHUNK_CHARS = 4000


@dataclass
class TTSResult:
    audio_path: Path
    duration_seconds: float


def generate_voiceover(
    script: str,
    output_dir: Path,
    config: Config,
    client: genai.Client,
) -> TTSResult:
    """Convert script text to a WAV file using Gemini TTS.

    Long scripts are split into chunks at paragraph boundaries,
    each chunk is synthesized separately, and the PCM data is
    concatenated into a single WAV file.
    """
    chunks = _split_script_for_tts(script)
    logger.info(f"TTS: processing {len(chunks)} chunks")

    all_pcm_data = bytearray()

    for i, chunk in enumerate(chunks):
        logger.info(f"TTS chunk {i + 1}/{len(chunks)} ({len(chunk)} chars)")
        pcm = _synthesize_chunk(chunk, config, client)
        all_pcm_data.extend(pcm)

    audio_path = output_dir / "audio.wav"
    _write_wav(audio_path, bytes(all_pcm_data))

    duration = len(all_pcm_data) / (SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS)
    logger.info(f"TTS complete: {duration:.1f}s, saved to {audio_path}")

    return TTSResult(audio_path=audio_path, duration_seconds=duration)


def _synthesize_chunk(text: str, config: Config, client: genai.Client) -> bytes:
    """Synthesize a single chunk of text to PCM audio bytes."""
    response = client.models.generate_content(
        model=config.tts_model,
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=config.tts_voice,
                    )
                )
            ),
        ),
    )

    # Extract audio data from response
    for part in response.candidates[0].content.parts:
        if part.inline_data and part.inline_data.data:
            data = part.inline_data.data
            mime = part.inline_data.mime_type or ""

            # If the response is a WAV container, extract raw PCM
            if mime.startswith("audio/wav") or mime.startswith("audio/x-wav"):
                return _extract_pcm_from_wav(data)

            # If raw PCM (audio/L16 or similar)
            return data if isinstance(data, bytes) else bytes(data)

    raise RuntimeError("No audio data in TTS response")


def _extract_pcm_from_wav(wav_data: bytes) -> bytes:
    """Extract raw PCM frames from a WAV container."""
    with wave.open(io.BytesIO(wav_data), "rb") as wf:
        return wf.readframes(wf.getnframes())


def _split_script_for_tts(script: str) -> list[str]:
    """Split script into chunks suitable for the TTS API.

    Splits on double-newline (paragraph breaks) first, then groups
    paragraphs to stay under MAX_CHUNK_CHARS while keeping natural
    pause points.
    """
    paragraphs = [p.strip() for p in script.split("\n\n") if p.strip()]

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_length = 0

    for para in paragraphs:
        para_len = len(para)

        # If a single paragraph exceeds the limit, split it on sentences
        if para_len > MAX_CHUNK_CHARS:
            # Flush current chunk first
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_length = 0
            # Split long paragraph on sentence boundaries
            sentences = _split_sentences(para)
            for sentence in sentences:
                if current_length + len(sentence) + 1 > MAX_CHUNK_CHARS and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                current_chunk.append(sentence)
                current_length += len(sentence) + 1
            continue

        if current_length + para_len + 2 > MAX_CHUNK_CHARS and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_length = 0

        current_chunk.append(para)
        current_length += para_len + 2

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def _split_sentences(text: str) -> list[str]:
    """Naively split text into sentences on period/question/exclamation."""
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def _write_wav(path: Path, pcm_data: bytes) -> None:
    """Write raw PCM data to a WAV file."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_data)
