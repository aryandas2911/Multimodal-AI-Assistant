"""Tests for audio validation, speech-to-text, text-to-speech, and routing."""

import wave
from io import BytesIO

import pytest
import speech_recognition as sr

from modules.voice_processor import (
    MAX_AUDIO_BYTES,
    MAX_AUDIO_SECONDS,
    MAX_TTS_CHARS,
    VOICE_CONTEXTS,
    VoiceProcessingError,
    audio_signature,
    get_audio_info,
    resolve_voice_context,
    speech_to_text,
    text_to_speech,
    transcribe_audio,
    validate_audio,
)


def make_wav(
    seconds: float = 1.0,
    sample_rate: int = 16000,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    """Create a valid silent PCM WAV file entirely in memory."""
    frame_count = int(seconds * sample_rate)
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00" * frame_count * channels * sample_width)
    return buffer.getvalue()


class FakeRecognizer:
    """Stand-in for sr.Recognizer so tests never contact Google."""

    def __init__(self, text="What is in this image?", error=None):
        self.text = text
        self.error = error
        self.recorded = False
        self.calls = 0

    def record(self, source):
        # Touching the source proves the WAV was opened through sr.AudioFile.
        self.recorded = source is not None
        return object()

    def recognize_google(self, audio_data, **kwargs):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.text


class FakeEngine:
    """Minimal pyttsx3-style engine that writes a valid WAV file."""

    def __init__(self, fail=False):
        self.fail = fail
        self.spoken_text = None
        self.stopped = False

    def save_to_file(self, text, path):
        self.spoken_text = text
        if self.fail:
            raise RuntimeError("synthesis failed")
        with open(path, "wb") as audio_file:
            audio_file.write(make_wav(0.1))

    def runAndWait(self):
        if self.fail:
            raise RuntimeError("engine failed")

    def stop(self):
        self.stopped = True


def test_valid_wav_is_accepted():
    audio_bytes = make_wav(seconds=2.0)

    validate_audio(audio_bytes)
    info = get_audio_info(audio_bytes)

    assert info["duration_seconds"] == 2.0
    assert info["sample_rate"] == 16000
    assert info["channels"] == 1
    assert info["size_bytes"] == len(audio_bytes)


def test_empty_audio_is_rejected():
    with pytest.raises(VoiceProcessingError, match="empty"):
        validate_audio(b"")


def test_invalid_audio_is_rejected():
    with pytest.raises(VoiceProcessingError, match="valid WAV"):
        validate_audio(b"this is not audio")


def test_truncated_wav_is_rejected():
    # A RIFF header without a complete WAV structure should fail cleanly.
    with pytest.raises(VoiceProcessingError, match="valid WAV"):
        validate_audio(b"RIFF" + b"\x00" * 40)


def test_oversized_audio_is_rejected():
    # The size check runs before decoding, so fake bytes are enough.
    oversized = b"RIFF" + b"\x00" * (MAX_AUDIO_BYTES + 1)

    with pytest.raises(VoiceProcessingError, match="too large"):
        validate_audio(oversized)


def test_audio_longer_than_limit_is_rejected():
    audio_bytes = make_wav(seconds=MAX_AUDIO_SECONDS + 2)

    with pytest.raises(VoiceProcessingError, match="too long"):
        validate_audio(audio_bytes)


def test_successful_transcription_returns_text():
    audio_bytes = make_wav()
    recognizer = FakeRecognizer(text="Explain paging on page 2.")

    text = speech_to_text(audio_bytes, recognizer=recognizer)

    assert text == "Explain paging on page 2."
    assert recognizer.recorded is True
    assert recognizer.calls == 1


def test_unclear_speech_raises_helpful_error():
    recognizer = FakeRecognizer(error=sr.UnknownValueError())

    with pytest.raises(VoiceProcessingError, match="could not be understood"):
        speech_to_text(make_wav(), recognizer=recognizer)


def test_speech_service_failure_raises_helpful_error():
    recognizer = FakeRecognizer(error=sr.RequestError("service unavailable"))

    with pytest.raises(VoiceProcessingError, match="service is unavailable"):
        speech_to_text(make_wav(), recognizer=recognizer)


def test_empty_transcription_is_rejected():
    recognizer = FakeRecognizer(text="   ")

    with pytest.raises(VoiceProcessingError, match="No speech was recognized"):
        speech_to_text(make_wav(), recognizer=recognizer)


def test_unchanged_audio_is_not_transcribed_twice():
    audio_bytes = make_wav()
    recognizer = FakeRecognizer(text="What is this?")

    first_signature, first_text, first_transcribed = transcribe_audio(
        audio_bytes, recognizer=recognizer
    )
    second_signature, second_text, second_transcribed = transcribe_audio(
        audio_bytes,
        cached_signature=first_signature,
        cached_text=first_text,
        recognizer=recognizer,
    )

    assert first_transcribed is True
    assert second_transcribed is False
    assert first_signature == second_signature
    assert second_text == "What is this?"
    # The speech service must be contacted only once.
    assert recognizer.calls == 1


def test_audio_signature_changes_with_content():
    assert audio_signature(make_wav(1.0)) != audio_signature(make_wav(2.0))


def test_empty_tts_text_is_rejected():
    with pytest.raises(VoiceProcessingError, match="no answer text"):
        text_to_speech("   ")


def test_tts_text_is_limited_before_synthesis():
    engine = FakeEngine()

    text_to_speech("word " * (MAX_TTS_CHARS), engine_factory=lambda: engine)

    assert engine.spoken_text.endswith("...")
    assert len(engine.spoken_text) <= MAX_TTS_CHARS + 3


def test_tts_returns_playable_wav_bytes():
    engine = FakeEngine()

    audio_bytes = text_to_speech("A short answer.", engine_factory=lambda: engine)

    assert audio_bytes[:4] == b"RIFF"
    assert audio_bytes[8:12] == b"WAVE"
    assert engine.spoken_text == "A short answer."
    assert engine.stopped is True


def test_unavailable_tts_engine_degrades_gracefully():
    def failing_engine():
        raise RuntimeError("no speech engine")

    with pytest.raises(VoiceProcessingError, match="unavailable"):
        text_to_speech("Hello", engine_factory=failing_engine)


def test_tts_synthesis_failure_is_reported_safely():
    engine = FakeEngine(fail=True)

    with pytest.raises(VoiceProcessingError, match="unavailable"):
        text_to_speech("Hello", engine_factory=lambda: engine)


def test_general_context_needs_no_file():
    assert resolve_voice_context("General", has_image=False, has_pdf=False) == (
        "General"
    )


def test_image_context_requires_an_image():
    with pytest.raises(VoiceProcessingError, match="No image is loaded"):
        resolve_voice_context("Current Image", has_image=False, has_pdf=True)


def test_pdf_context_requires_a_pdf():
    with pytest.raises(VoiceProcessingError, match="No PDF is loaded"):
        resolve_voice_context("Current PDF", has_image=True, has_pdf=False)


def test_invalid_voice_context_is_rejected():
    with pytest.raises(VoiceProcessingError, match="valid voice context"):
        resolve_voice_context("Something else", has_image=True, has_pdf=True)


def test_voice_contexts_are_explicit():
    assert VOICE_CONTEXTS == ("General", "Current Image", "Current PDF")
