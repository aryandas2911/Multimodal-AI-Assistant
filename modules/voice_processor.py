"""Speech-to-text and text-to-speech helpers for the voice assistant.

The browser records audio with Streamlit's ``st.audio_input`` and sends WAV
bytes to this module. SpeechRecognition converts the WAV bytes to text, and
pyttsx3 can optionally turn an answer into a WAV file for ``st.audio``.

Important: ``speech_recognition.Microphone()`` is deliberately not used. That
class needs PyAudio and cannot access a remote user's browser microphone.
"""

import hashlib
import os
import tempfile
import wave
from io import BytesIO

import speech_recognition as sr

# A 30 second mono WAV at 16 kHz is about 1 MB, so this also allows stereo or
# higher sample rates recorded by the browser without accepting huge uploads.
MAX_AUDIO_BYTES = 12 * 1024 * 1024
MAX_AUDIO_SECONDS = 30

# Very long answers take a long time to synthesize and are tedious to listen
# to. The text answer stays complete; only the spoken version is shortened.
MAX_TTS_CHARS = 1200

# The browser may report a fraction over 30 seconds, so allow a small margin
# instead of rejecting an otherwise valid 30 second recording.
AUDIO_DURATION_TOLERANCE = 1.0

# Explicit contexts keep voice routing simple and predictable.
VOICE_CONTEXTS = ("General", "Current Image", "Current PDF")


class VoiceProcessingError(ValueError):
    """A safe, user-facing error raised by the voice pipeline."""


def _check_audio_bytes(audio_bytes: bytes) -> bytes:
    """Check the basic type and size before decoding the WAV structure."""
    if not isinstance(audio_bytes, (bytes, bytearray)):
        raise VoiceProcessingError("The recording does not contain audio data.")

    if len(audio_bytes) == 0:
        raise VoiceProcessingError(
            "The recording is empty. Please record a question and try again."
        )

    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise VoiceProcessingError(
            "The recording is too large. Please record a shorter question."
        )

    return bytes(audio_bytes)


def get_audio_info(audio_bytes: bytes) -> dict[str, object]:
    """Validate a WAV recording and return its basic properties.

    ``wave`` is part of the Python standard library, so no extra audio
    dependency is needed just to check the format or duration.
    """
    checked_bytes = _check_audio_bytes(audio_bytes)

    try:
        with wave.open(BytesIO(checked_bytes), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_rate = wav_file.getframerate()
            sample_width = wav_file.getsampwidth()
            frame_count = wav_file.getnframes()
    except (wave.Error, EOFError, OSError) as error:
        raise VoiceProcessingError(
            "The recording is not a valid WAV audio file. Please record again."
        ) from error

    if sample_rate <= 0 or channels <= 0 or sample_width <= 0:
        raise VoiceProcessingError(
            "The recording format is not supported. Please record again."
        )

    duration_seconds = frame_count / sample_rate
    if duration_seconds <= 0:
        raise VoiceProcessingError(
            "The recording is empty. Please record a question and try again."
        )

    if duration_seconds > MAX_AUDIO_SECONDS + AUDIO_DURATION_TOLERANCE:
        raise VoiceProcessingError(
            f"The recording is too long. Maximum length is {MAX_AUDIO_SECONDS} "
            "seconds."
        )

    return {
        "duration_seconds": round(duration_seconds, 2),
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width": sample_width,
        "frame_count": frame_count,
        "size_bytes": len(checked_bytes),
    }


def validate_audio(audio_bytes: bytes) -> None:
    """Validate an audio recording without keeping any resources open."""
    get_audio_info(audio_bytes)


def audio_signature(audio_bytes: bytes) -> str:
    """Return a stable identity for a recording.

    The app uses this to transcribe each recording once, even though
    Streamlit reruns the script many times.
    """
    checked_bytes = _check_audio_bytes(audio_bytes)
    return hashlib.sha256(checked_bytes).hexdigest()


def speech_to_text(audio_bytes: bytes, recognizer=None) -> str:
    """Transcribe WAV bytes with SpeechRecognition.

    ``recognize_google`` sends the audio to Google's online speech service,
    so recognition needs an internet connection. It is not offline or
    guaranteed to be available. Pass a recognizer in tests to avoid network
    calls.
    """
    validate_audio(audio_bytes)

    if recognizer is None:
        recognizer = sr.Recognizer()

    try:
        # sr.AudioFile reads WAV bytes through BytesIO, so no PyAudio
        # microphone object is created and no temporary file is needed.
        with sr.AudioFile(BytesIO(audio_bytes)) as source:
            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(audio_data, language="en-US")
    except sr.UnknownValueError:
        raise VoiceProcessingError(
            "The speech could not be understood. Please speak clearly and try "
            "again, or type the question manually."
        ) from None
    except sr.RequestError:
        raise VoiceProcessingError(
            "The speech recognition service is unavailable. Check your internet "
            "connection, or type the question manually."
        ) from None
    except (wave.Error, EOFError, ValueError, OSError):
        raise VoiceProcessingError(
            "The recording could not be read. Please record again."
        ) from None
    except Exception:
        # Never expose provider or decoder details in the UI.
        raise VoiceProcessingError(
            "Speech recognition failed. Please try again or type the question."
        ) from None

    if not isinstance(text, str) or not text.strip():
        raise VoiceProcessingError(
            "No speech was recognized. Please record the question again or type "
            "it manually."
        )

    return text.strip()


def transcribe_audio(
    audio_bytes: bytes,
    cached_signature: str | None = None,
    cached_text: str | None = None,
    recognizer=None,
) -> tuple[str, str, bool]:
    """Transcribe a recording, reusing a previous result for the same audio.

    Returns ``(signature, text, did_transcribe)``. When the signature matches
    the cached one and text exists, no speech service call is made. This is
    what prevents repeated transcription on Streamlit reruns.
    """
    signature = audio_signature(audio_bytes)

    if cached_signature == signature and cached_text:
        return signature, cached_text, False

    return signature, speech_to_text(audio_bytes, recognizer=recognizer), True


def resolve_voice_context(
    selected_context: str,
    has_image: bool,
    has_pdf: bool,
) -> str:
    """Check that the chosen voice context has the file it needs.

    Routing is intentionally explicit instead of using intent detection.
    """
    if selected_context not in VOICE_CONTEXTS:
        raise VoiceProcessingError(
            "Choose a valid voice context: General, Current Image, or Current PDF."
        )

    if selected_context == "Current Image" and not has_image:
        raise VoiceProcessingError(
            "No image is loaded. Upload an image in Image Question Answering "
            "mode first, or choose the General context."
        )

    if selected_context == "Current PDF" and not has_pdf:
        raise VoiceProcessingError(
            "No PDF is loaded. Upload a PDF in PDF Question Answering mode "
            "first, or choose the General context."
        )

    return selected_context


def _prepare_spoken_text(text: str) -> str:
    """Clean and limit text before synthesis, rejecting empty answers."""
    if not isinstance(text, str):
        raise VoiceProcessingError("There is no answer text to speak.")

    cleaned_text = " ".join(text.split())
    if not cleaned_text:
        raise VoiceProcessingError("There is no answer text to speak.")

    if len(cleaned_text) <= MAX_TTS_CHARS:
        return cleaned_text

    # Cut at a word boundary so the last word is not spoken in half.
    shortened = cleaned_text[:MAX_TTS_CHARS].rsplit(" ", 1)[0]
    return shortened + "..."


def _wav_bytes_are_playable(audio_bytes: bytes) -> bool:
    """Check for the RIFF/WAVE header pyttsx3 should produce."""
    return (
        len(audio_bytes) > 44
        and audio_bytes[:4] == b"RIFF"
        and audio_bytes[8:12] == b"WAVE"
    )


def text_to_speech(text: str, engine_factory=None) -> bytes:
    """Synthesize an answer locally with pyttsx3 and return WAV bytes.

    A unique temporary file per call avoids collisions between sessions, and
    the file is always removed before returning. The caller owns only the
    returned bytes.

    pyttsx3 uses SAPI5 on Windows and another system engine (for example
    espeak) on Linux. If no engine or audio output is available, a
    VoiceProcessingError is raised so the text answer still works.
    """
    # Validate before loading the engine, so empty input fails quickly.
    spoken_text = _prepare_spoken_text(text)

    engine = None
    temporary_path = None
    fallback_path = None

    try:
        if engine_factory is None:
            # Import lazily so a missing system speech engine cannot break the
            # rest of the application at import time.
            import pyttsx3

            engine = pyttsx3.init()
        else:
            engine = engine_factory()

        # mkstemp gives a unique name, which prevents two sessions from writing
        # to the same temporary file. The descriptor is closed before pyttsx3
        # writes, because Windows locks open files.
        descriptor, temporary_path = tempfile.mkstemp(
            prefix="multimodal_tts_", suffix=".wav"
        )
        os.close(descriptor)

        engine.save_to_file(spoken_text, temporary_path)
        engine.runAndWait()

        # Some engines append .wav to the requested path. Accept that file if
        # the exact path was not written.
        fallback_path = temporary_path + ".wav"
        output_path = temporary_path
        if not os.path.exists(output_path) and os.path.exists(fallback_path):
            output_path = fallback_path

        with open(output_path, "rb") as audio_file:
            audio_bytes = audio_file.read()

        if not _wav_bytes_are_playable(audio_bytes):
            raise VoiceProcessingError(
                "The speech engine did not produce a playable WAV file. "
                "The text answer is still available."
            )

        return audio_bytes
    except VoiceProcessingError:
        raise
    except Exception:
        raise VoiceProcessingError(
            "Speech output is unavailable on this system. The text answer is "
            "still available."
        ) from None
    finally:
        # Stop the engine and delete temporary files whether synthesis
        # succeeded or failed.
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass

        for path in (temporary_path, fallback_path):
            if path:
                try:
                    os.remove(path)
                except OSError:
                    pass
