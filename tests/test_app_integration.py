"""App-level integration tests for session state and mode wiring.

These tests run the real Streamlit script with AppTest, but mock every external
AI and speech service. No API key is used and no network request is made.
"""

import wave
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pymupdf
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

import modules.ai_assistant as ai
import modules.voice_processor as vp

# AppTest resolves relative paths against the test file, so use the repo root.
APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class FakeCompletions:
    """Minimal stand-in for Groq's chat.completions resource."""

    def __init__(self, content="Mocked answer."):
        self.content = content
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.content),
                )
            ]
        )


class FakeClient:
    """Minimal stand-in for groq.Groq so tests never use the network."""

    def __init__(self, content="Mocked answer."):
        self.completions = FakeCompletions(content=content)
        self.chat = SimpleNamespace(completions=self.completions)


@pytest.fixture
def fake_ai(monkeypatch):
    """Replace Groq with a local fake and count client creations."""
    state = {"clients": []}

    def create_client(settings=None):
        client = FakeClient()
        state["clients"].append(client)
        return client

    monkeypatch.setattr(
        ai,
        "get_settings",
        lambda: ai.AppSettings(
            api_key="test-key",
            model="test-text-model",
            vision_model="test-vision-model",
        ),
    )
    monkeypatch.setattr(ai, "create_client", create_client)
    return state


@pytest.fixture
def fake_speech(monkeypatch):
    """Replace Google speech recognition with a counter-based fake."""
    state = {"calls": 0}

    def speech_to_text(audio_bytes, recognizer=None):
        state["calls"] += 1
        return "What is artificial intelligence?"

    monkeypatch.setattr(vp, "speech_to_text", speech_to_text)
    return state


def make_image_bytes(color="red") -> bytes:
    """Create a small PNG in memory without writing a temporary file."""
    buffer = BytesIO()
    Image.new("RGB", (64, 48), color).save(buffer, format="PNG")
    return buffer.getvalue()


def make_pdf_bytes(text="Page one content.") -> bytes:
    """Create a one-page PDF with selectable text."""
    document = pymupdf.open()
    try:
        page = document.new_page()
        page.insert_text((72, 72), text)
        return document.tobytes()
    finally:
        # Always close the document, even if building the PDF fails.
        document.close()


def make_wav(seconds=1.0) -> bytes:
    """Create silent 16 kHz mono WAV bytes for the audio widget."""
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        # Two bytes per sample because sampwidth is 2.
        wav_file.writeframes(b"\x00" * int(seconds * 16000) * 2)
    return buffer.getvalue()


def submit_image_question(app, question="What is in this image?"):
    """Type a question and press the image page's Ask AI button."""
    app.text_input[0].set_value(question)
    next(button for button in app.button if button.label == "Ask AI").click()
    return app.run()


def submit_pdf_question(app, question="Summarize page 1."):
    """Type a question and press the PDF page's Ask AI button."""
    app.text_input[0].set_value(question)
    next(button for button in app.button if button.label == "Ask AI").click()
    return app.run()


def test_all_three_modes_render_without_exceptions(fake_ai):
    app = AppTest.from_file(APP_PATH).run()
    assert not list(app.exception)

    app.sidebar.radio[0].set_value("PDF Question Answering").run()
    assert not list(app.exception)

    app.sidebar.radio[0].set_value("Voice Assistant").run()
    assert not list(app.exception)


def test_image_answer_is_kept_and_not_requested_again_on_rerun(fake_ai):
    app = AppTest.from_file(APP_PATH).run()
    app.file_uploader[0].set_value(("image.png", make_image_bytes(), "image/png"))
    app.run()

    submit_image_question(app)

    assert not list(app.exception)
    assert len(app.session_state["image_history"]) == 1
    # Chat bubbles: one user question and one assistant answer.
    assert len(app.chat_message) == 2
    assert len(fake_ai["clients"]) == 1

    # A plain Streamlit rerun must not repeat the API request.
    app.run()
    assert len(fake_ai["clients"]) == 1
    assert len(app.session_state["image_history"]) == 1


def test_new_image_clears_old_image_history(fake_ai):
    app = AppTest.from_file(APP_PATH).run()
    app.file_uploader[0].set_value(("first.png", make_image_bytes(), "image/png"))
    app.run()
    submit_image_question(app)
    assert len(app.session_state["image_history"]) == 1

    # Uploading different content produces a new hash, so the old conversation
    # and cached bytes must be discarded.
    app.file_uploader[0].set_value(("second.png", make_image_bytes("blue"), "image/png"))
    app.run()

    assert app.session_state["image_history"] == []
    assert app.session_state["image_signature"] is not None


def test_new_pdf_clears_old_pdf_history(fake_ai):
    app = AppTest.from_file(APP_PATH).run()
    app.sidebar.radio[0].set_value("PDF Question Answering").run()
    app.file_uploader[0].set_value(
        ("first.pdf", make_pdf_bytes("First document."), "application/pdf")
    )
    app.run()
    submit_pdf_question(app)
    assert len(app.session_state["pdf_history"]) == 1

    # A new PDF must not inherit answers, page selection, or rendered pages
    # from the previous document.
    app.file_uploader[0].set_value(
        ("second.pdf", make_pdf_bytes("Second document."), "application/pdf")
    )
    app.run()

    assert app.session_state["pdf_history"] == []


def test_invalid_image_shows_a_friendly_error(fake_ai):
    app = AppTest.from_file(APP_PATH).run()
    # The uploader only filters by extension, so the content check must catch
    # this and show a message instead of raising.
    app.file_uploader[0].set_value(("bad.png", b"not an image", "image/png"))
    app.run()

    assert not list(app.exception)
    assert any("not a valid image" in error.value for error in app.error)


def test_voice_recording_is_transcribed_only_once(fake_ai, fake_speech):
    app = AppTest.from_file(APP_PATH).run()
    app.sidebar.radio[0].set_value("Voice Assistant").run()
    app.audio_input[0].upload("clip.wav", make_wav()).run()

    assert fake_speech["calls"] == 1
    assert app.text_area[0].value == "What is artificial intelligence?"

    # Rerunning with the same recording must reuse the stored transcript.
    app.run()
    assert fake_speech["calls"] == 1
    assert app.session_state["voice_transcript"] == "What is artificial intelligence?"


def test_clear_session_resets_documents_and_answers(fake_ai):
    app = AppTest.from_file(APP_PATH).run()
    app.file_uploader[0].set_value(("image.png", make_image_bytes(), "image/png"))
    app.run()
    submit_image_question(app)

    next(button for button in app.button if button.label == "Clear session").click()
    app.run()

    assert not list(app.exception)
    assert app.session_state["image_history"] == []
    assert app.session_state["image_bytes"] is None
    assert app.session_state["image_signature"] is None
    assert app.session_state["pdf_history"] == []
    assert app.session_state["voice_answer"] is None


def test_missing_configuration_shows_a_warning(monkeypatch):
    # get_settings() loads .env, so remove both variables to simulate an
    # unconfigured machine regardless of the developer's local file.
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)

    app = AppTest.from_file(APP_PATH).run()

    assert not list(app.exception)
    assert any("not configured" in warning.value for warning in app.warning)
