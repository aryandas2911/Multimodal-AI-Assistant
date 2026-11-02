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

# PDF requests send at most three rendered pages. This keeps payloads small and
# stays within the vision model's documented limit of three images per request.
MAX_PDF_CONTEXT_PAGES = 3
MAX_PDF_CONTEXT_TEXT_CHARS = 30_000
MAX_PDF_CONTEXT_IMAGE_BYTES = 8 * 1024 * 1024

# The system instruction guides every image request. It asks the model to stay
# grounded in the image and to admit when something cannot be determined.
SYSTEM_INSTRUCTION = (
    "You are a helpful multimodal assistant. Analyze the supplied image "
    "carefully and answer the user's question accurately. Describe only details "
    "supported by the image. If text is visible, read it carefully. If the "
    "image is unclear or the answer cannot be determined, state the limitation "
    "instead of inventing information."
)

# PDF answers must use both the extracted text and the rendered page images.
# Page citations make it easier for the user to check the answer.
PDF_SYSTEM_INSTRUCTION = (
    "You are a helpful document question-answering assistant. Use only the "
    "supplied PDF page text and rendered page images. The images show the full "
    "page, including diagrams, tables, charts, screenshots, and scanned text. "
    "Read visible text carefully and examine the visuals before answering. "
    "Cite page numbers in your answer when possible, for example (page 3). "
    "If the supplied pages do not contain enough information, say so clearly "
    "instead of inventing details."
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
    # Secrets and model names come from environment variables, never from code.
    # GROQ_MODEL is reserved for later text features; image questions use the
    # separate vision model so either can be changed without touching the code.
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


def _validate_question(question: str, subject: str = "the image") -> str:
    """Clean the question and reject empty or overly long input."""
    if not isinstance(question, str) or not question.strip():
        raise AIAssistantError(f"Please enter a question about {subject}.")

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


def _request_completion(
    messages: list[dict],
    client: Groq,
    settings: AppSettings,
    max_completion_tokens: int,
) -> str:
    """Send one chat request and convert provider errors into safe messages."""
    try:
        # Low temperature gives more consistent answers. A token limit controls
        # response length/cost, and the timeout prevents a frozen request.
        response = client.chat.completions.create(
            model=settings.vision_model,
            messages=messages,
            temperature=0.2,
            max_completion_tokens=max_completion_tokens,
            stream=False,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    # Convert provider-specific exceptions into short, safe messages. The
    # original exception is hidden with `from None` so details never reach the UI.
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
            "The AI service rejected the request. Check the supplied content "
            "and the vision model configuration."
        ) from None
    except APIStatusError:
        raise AIAssistantError(
            "The AI service returned an error. Please try again later."
        ) from None

    # Groq normally returns choices[0].message.content, but the nesting is
    # checked defensively so a malformed or blocked response cannot crash the app.
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        raise AIAssistantError(
            "The AI returned an unexpected response. Please try again."
        ) from None

    if not isinstance(content, str) or not content.strip():
        raise AIAssistantError("The AI returned an empty response. Please try again.")

    return content.strip()


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

    # JSON cannot carry raw bytes, so the image becomes a base64 data URL.
    # Groq accepts this inside the same message as the text question.
    encoded_image = base64.b64encode(image_bytes).decode("ascii")
    image_url = f"data:{mime_type};base64,{encoded_image}"

    # A multimodal message has two parts: the question text and the image.
    # The system message sets the behavior for the whole request.
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

    return _request_completion(messages, client, settings, max_completion_tokens=1024)


def _validate_pdf_context(page_context: list[dict]) -> None:
    """Check page context and enforce text/image size limits."""
    if not isinstance(page_context, (list, tuple)) or not page_context:
        raise AIAssistantError("No PDF page context was supplied.")

    if len(page_context) > MAX_PDF_CONTEXT_PAGES:
        raise AIAssistantError(
            f"At most {MAX_PDF_CONTEXT_PAGES} PDF pages can be sent at once. "
            "Please select fewer pages."
        )

    seen_pages = set()
    total_text_chars = 0
    total_image_bytes = 0

    # Validate every page before building the request. This prevents a large or
    # malformed payload from reaching the API and failing with a vague error.
    for page in page_context:
        if not isinstance(page, dict):
            raise AIAssistantError("The PDF page context is not valid.")

        page_number = page.get("page_number")
        if isinstance(page_number, bool) or not isinstance(page_number, int):
            raise AIAssistantError("Each PDF page needs a valid page number.")
        if page_number < 1:
            raise AIAssistantError("PDF page numbers must start at 1.")
        if page_number in seen_pages:
            raise AIAssistantError("The same PDF page was supplied twice.")
        seen_pages.add(page_number)

        text = page.get("text", "")
        if not isinstance(text, str):
            raise AIAssistantError("The PDF page text is not valid.")
        total_text_chars += len(text)

        image_bytes = page.get("image_bytes")
        mime_type = page.get("mime_type", "image/png")
        _validate_image_input(image_bytes, mime_type)
        total_image_bytes += len(image_bytes)

    if total_text_chars > MAX_PDF_CONTEXT_TEXT_CHARS:
        raise AIAssistantError(
            "The selected pages contain too much text for one request. "
            "Please ask about one page at a time."
        )

    if total_image_bytes > MAX_PDF_CONTEXT_IMAGE_BYTES:
        raise AIAssistantError(
            "The rendered PDF pages are too large for one request. "
            "Please ask about one page at a time."
        )


def ask_pdf(
    question: str,
    page_context: list[dict],
    client: Groq | None = None,
    settings: AppSettings | None = None,
) -> str:
    """Ask Groq a question using PDF text and rendered page images.

    Each page context is a dictionary with:
        page_number: 1-based page number
        text: selectable text for that page (may be empty on a scanned page)
        image_bytes: rendered PNG bytes for the whole page
        mime_type: image/png
    """
    cleaned_question = _validate_question(question, subject="the PDF")
    _validate_pdf_context(page_context)

    if settings is None:
        settings = get_settings()
    if client is None:
        client = create_client(settings)

    # Include the extracted text first, with a clear page marker for each page.
    context_blocks = []
    for page in page_context:
        page_number = page["page_number"]
        page_text = page.get("text", "").strip()
        if page_text:
            text_block = page_text
        else:
            # Tell the model that the empty string is expected on scanned pages
            # and that the rendered image is the source of truth there.
            text_block = (
                "[No selectable text was found on this page. "
                "Use the rendered page image.]"
            )
        context_blocks.append(f"--- PAGE {page_number} TEXT ---\n{text_block}")

    prompt = (
        f"Question: {cleaned_question}\n\n"
        "Answer using only the PDF context supplied below. Cite page numbers "
        "when possible. If the context is insufficient, say what is missing.\n\n"
        "PDF context:\n"
        + "\n\n".join(context_blocks)
    )

    # The text prompt is followed by one rendered image per page. This is what
    # allows the model to inspect diagrams, tables, and scanned pages.
    content_parts: list[dict] = [{"type": "text", "text": prompt}]
    for page in page_context:
        page_number = page["page_number"]
        encoded_image = base64.b64encode(page["image_bytes"]).decode("ascii")
        content_parts.append(
            {"type": "text", "text": f"Rendered image of page {page_number}:"}
        )
        content_parts.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{page['mime_type']};base64,{encoded_image}"
                },
            }
        )

    messages = [
        {"role": "system", "content": PDF_SYSTEM_INSTRUCTION},
        {"role": "user", "content": content_parts},
    ]

    return _request_completion(messages, client, settings, max_completion_tokens=1500)
