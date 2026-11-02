"""Tests for configuration and Groq image question answering."""

from types import SimpleNamespace

import httpx
import pytest
from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    Groq,
    NotFoundError,
    RateLimitError,
)

from modules.ai_assistant import (
    DEFAULT_VISION_MODEL,
    SYSTEM_INSTRUCTION,
    AIAssistantError,
    AppSettings,
    ConfigurationError,
    ask_image,
    create_client,
    get_settings,
)

TEST_SETTINGS = AppSettings(
    api_key="gsk_test_key",
    model="test-text-model",
    vision_model="test-vision-model",
)


class FakeCompletions:
    """Small stand-in for client.chat.completions in unit tests."""

    def __init__(self, content="A red apple.", error=None):
        self.content = content
        self.error = error
        self.last_call = None

    def create(self, **kwargs):
        self.last_call = kwargs
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.content),
                )
            ]
        )


class FakeClient:
    """Minimal object with the same shape used by ask_image()."""

    def __init__(self, content="A red apple.", error=None):
        self.completions = FakeCompletions(content=content, error=error)
        self.chat = SimpleNamespace(completions=self.completions)


def make_status_error(error_class, status_code):
    """Build a real Groq/httpx error without making a network request."""
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    return error_class("raw provider details", response=response, body=None)


def test_missing_api_key_raises_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_MODEL", "test-model")

    with pytest.raises(ConfigurationError, match="GROQ_API_KEY"):
        get_settings()


def test_placeholder_api_key_raises_error(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "your_api_key_here")
    monkeypatch.setenv("GROQ_MODEL", "test-model")

    with pytest.raises(ConfigurationError, match="example value"):
        get_settings()


def test_missing_model_raises_error(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key")
    monkeypatch.delenv("GROQ_MODEL", raising=False)

    with pytest.raises(ConfigurationError, match="GROQ_MODEL"):
        get_settings()


def test_valid_settings_are_loaded(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key")
    monkeypatch.setenv("GROQ_MODEL", "test-model")
    monkeypatch.delenv("GROQ_VISION_MODEL", raising=False)

    settings = get_settings()

    assert settings.api_key == "gsk_test_key"
    assert settings.model == "test-model"
    assert settings.vision_model == DEFAULT_VISION_MODEL


def test_vision_model_can_be_overridden(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key")
    monkeypatch.setenv("GROQ_MODEL", "test-model")
    monkeypatch.setenv("GROQ_VISION_MODEL", "custom-vision-model")

    settings = get_settings()

    assert settings.vision_model == "custom-vision-model"


def test_api_key_is_hidden_from_settings_repr():
    settings = AppSettings(api_key="gsk_secret_value", model="test-model")

    assert "gsk_secret_value" not in repr(settings)


def test_create_client_returns_groq_client():
    # This checks client construction only; no network request is made.
    client = create_client(TEST_SETTINGS)

    assert isinstance(client, Groq)


def test_ask_image_builds_multimodal_request():
    fake_client = FakeClient(content="The image shows a red apple.")

    answer = ask_image(
        "What is in this image?",
        b"fake-image-bytes",
        "image/png",
        client=fake_client,
        settings=TEST_SETTINGS,
    )

    assert answer == "The image shows a red apple."
    call = fake_client.completions.last_call
    assert call["model"] == "test-vision-model"
    assert call["messages"][0] == {
        "role": "system",
        "content": SYSTEM_INSTRUCTION,
    }

    user_content = call["messages"][1]["content"]
    assert {"type": "text", "text": "What is in this image?"} in user_content
    image_part = next(part for part in user_content if part["type"] == "image_url")
    assert image_part["image_url"]["url"] == (
        "data:image/png;base64,ZmFrZS1pbWFnZS1ieXRlcw=="
    )


def test_empty_question_is_rejected():
    with pytest.raises(AIAssistantError, match="enter a question"):
        ask_image(
            "   ",
            b"fake-image-bytes",
            "image/png",
            client=FakeClient(),
            settings=TEST_SETTINGS,
        )


def test_too_long_question_is_rejected():
    with pytest.raises(AIAssistantError, match="too long"):
        ask_image(
            "x" * 2001,
            b"fake-image-bytes",
            "image/png",
            client=FakeClient(),
            settings=TEST_SETTINGS,
        )


def test_empty_image_is_rejected():
    with pytest.raises(AIAssistantError, match="image data is empty"):
        ask_image(
            "What is this?",
            b"",
            "image/png",
            client=FakeClient(),
            settings=TEST_SETTINGS,
        )


def test_unsupported_mime_type_is_rejected():
    with pytest.raises(AIAssistantError, match="Unsupported image type"):
        ask_image(
            "What is this?",
            b"fake-image-bytes",
            "image/gif",
            client=FakeClient(),
            settings=TEST_SETTINGS,
        )


@pytest.mark.parametrize(
    ("error", "expected_message"),
    [
        (
            make_status_error(AuthenticationError, 401),
            "authentication failed",
        ),
        (
            make_status_error(RateLimitError, 429),
            "rate limit or quota",
        ),
        (
            APITimeoutError(
                request=httpx.Request(
                    "POST", "https://api.groq.com/openai/v1/chat/completions"
                )
            ),
            "timed out",
        ),
        (
            APIConnectionError(
                request=httpx.Request(
                    "POST", "https://api.groq.com/openai/v1/chat/completions"
                )
            ),
            "Could not connect",
        ),
        (
            make_status_error(NotFoundError, 404),
            "vision model is unavailable",
        ),
        (
            make_status_error(BadRequestError, 400),
            "rejected the request",
        ),
        (
            make_status_error(APIStatusError, 500),
            "returned an error",
        ),
    ],
)
def test_api_errors_become_safe_messages(error, expected_message):
    with pytest.raises(AIAssistantError, match=expected_message):
        ask_image(
            "What is this?",
            b"fake-image-bytes",
            "image/png",
            client=FakeClient(error=error),
            settings=TEST_SETTINGS,
        )


def test_api_error_does_not_expose_provider_details():
    error = make_status_error(AuthenticationError, 401)

    with pytest.raises(AIAssistantError) as caught_error:
        ask_image(
            "What is this?",
            b"fake-image-bytes",
            "image/png",
            client=FakeClient(error=error),
            settings=TEST_SETTINGS,
        )

    assert "raw provider details" not in str(caught_error.value)


def test_empty_model_response_is_rejected():
    with pytest.raises(AIAssistantError, match="empty response"):
        ask_image(
            "What is this?",
            b"fake-image-bytes",
            "image/png",
            client=FakeClient(content="   "),
            settings=TEST_SETTINGS,
        )


def test_missing_response_content_is_rejected():
    with pytest.raises(AIAssistantError, match="empty response"):
        ask_image(
            "What is this?",
            b"fake-image-bytes",
            "image/png",
            client=FakeClient(content=None),
            settings=TEST_SETTINGS,
        )
