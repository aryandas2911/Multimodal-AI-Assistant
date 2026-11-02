"""Configuration and Groq API helpers for the Multimodal AI Assistant.

Configuration is loaded from .env. The Groq client is created only when an AI
feature is used, so opening the Streamlit app does not send a request.
"""

import base64
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
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

# Load variables from .env once when the module is imported.
load_dotenv()

# Detect an unedited copy of .env.example instead of treating it as a real key.
PLACEHOLDER_API_KEY = "your_api_key_here"

# qwen/qwen3.8-27b is Groq's vision model in the current API documentation.
# It can be overridden with GROQ_VISION_MODEL in .env.
DEFAULT_VISION_MODEL = "qwen/qwen3.8-27b"

SUPPORTED_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_QUESTION_LENGTH = 2000
REQUEST_TIMEOUT_SECONDS = 60.0

SYSTEM_INSTRUCTION = (
    "You are a helpful multimodal assistant. Analyze the supplied image "
    "carefully and answer the user's question accurately. Describe only details "
    "supported by the image. If text is visible, read it carefully. If the "
    "image is unclear or the answer cannot be determined, state the limitation "
    "instead of inventing information."
)


class ConfigurationError(Exception):
    """Raised when required application configuration is missing or invalid."""


class AIAssistantError(Exception):
    """A safe, user-facing error raised when an AI request fails."""


@dataclass(frozen=True)
class AppSettings:
    """Validated application settings."""

    # Keep the secret out of repr output, logs, and debug displays.
    api_key: str = field(repr=False)
    model: str
    vision_model: str = DEFAULT_VISION_MODEL


def get_settings() -> AppSettings:
    """Read and validate the Groq configuration from environment variables.

    Raises:
        ConfigurationError: If the API key or text model is missing, or if the
            example placeholder key is still being used.
    """
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    model = os.getenv("GROQ_MODEL", "").strip()
    vision_model = os.getenv("GROQ_VISION_MODEL", "").strip() or DEFAULT_VISION_MODEL

    if not api_key:
        raise ConfigurationError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your "
            "Groq API key."
        )

    if api_key == PLACEHOLDER_API_KEY:
        raise ConfigurationError(
            "GROQ_API_KEY still contains the example value. Replace it with "
            "your real Groq API key in the .env file."
        )

    if not model:
        raise ConfigurationError(
            "GROQ_MODEL is not set. Add a Groq model name to the .env file."
        )

    return AppSettings(api_key=api_key, model=model, vision_model=vision_model)


def create_client(settings: AppSettings | None = None) -> Groq:
    """Create a Groq client without making a network request."""
    if settings is None:
        settings = get_settings()

    # Constructing the client only prepares HTTP settings; it sends no request.
    return Groq(api_key=settings.api_key)


def _validate_question(question: str) -> str:
    """Clean the question and reject empty or overly long input."""
    if not isinstance(question, str) or not question.strip():
        raise AIAssistantError("Please enter a question about the image.")

    cleaned_question = question.strip()
    if len(cleaned_question) > MAX_QUESTION_LENGTH:
        raise AIAssistantError(
            f"The question is too long. Maximum length is {MAX_QUESTION_LENGTH} characters."
        )

    return cleaned_question


def _validate_image_input(image_bytes: bytes, mime_type: str) -> None:
    """Check the image data passed to the AI layer."""
    if not isinstance(image_bytes, (bytes, bytearray)) or len(image_bytes) == 0:
        raise AIAssistantError("The image data is empty. Please upload an image again.")

    if mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
        raise AIAssistantError(
            "Unsupported image type. Please use a JPEG, PNG, or WEBP image."
        )


def ask_image(
    question: str,
    image_bytes: bytes,
    mime_type: str,
    client: Groq | None = None,
    settings: AppSettings | None = None,
) -> str:
    """Ask Groq a question about one image and return the answer text.

    The image and question are sent together in one multimodal chat request.
    Pass a pre-built client/settings in tests to avoid reading .env again.
    """
    cleaned_question = _validate_question(question)
    _validate_image_input(image_bytes, mime_type)

    if settings is None:
        settings = get_settings()
    if client is None:
        client = create_client(settings)

    # Groq accepts local images as a base64 data URL inside the chat message.
    encoded_image = base64.b64encode(image_bytes).decode("ascii")
    image_url = f"data:{mime_type};base64,{encoded_image}"

    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": cleaned_question},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        },
    ]

    try:
        response = client.chat.completions.create(
            model=settings.vision_model,
            messages=messages,
            temperature=0.2,
            max_completion_tokens=1024,
            stream=False,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except AuthenticationError:
        raise AIAssistantError(
            "Groq authentication failed. Check that GROQ_API_KEY is valid."
        ) from None
    except RateLimitError:
        raise AIAssistantError(
            "The Groq rate limit or quota was reached. Please wait and try again."
        ) from None
    except APITimeoutError:
        raise AIAssistantError("The AI request timed out. Please try again.") from None
    except APIConnectionError:
        raise AIAssistantError(
            "Could not connect to Groq. Check your internet connection and try again."
        ) from None
    except NotFoundError:
        raise AIAssistantError(
            "The configured vision model is unavailable. Check GROQ_VISION_MODEL."
        ) from None
    except BadRequestError:
        raise AIAssistantError(
            "The AI service rejected the request. Check the image and the "
            "vision model configuration."
        ) from None
    except APIStatusError:
        raise AIAssistantError(
            "The AI service returned an error. Please try again later."
        ) from None

    # Read the response defensively: a provider may return no usable text.
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        raise AIAssistantError(
            "The AI returned an unexpected response. Please try again."
        ) from None

    if not isinstance(content, str) or not content.strip():
        raise AIAssistantError("The AI returned an empty response. Please try again.")

    return content.strip()
