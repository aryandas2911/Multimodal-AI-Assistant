"""Configuration and Groq client setup for the Multimodal AI Assistant.

This module does not send any request to the Groq API. Creating a client
only prepares the connection; AI calls belong to the later feature phases.
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from groq import Groq

# Load variables from .env once when the module is imported.
load_dotenv()

PLACEHOLDER_API_KEY = "your_api_key_here"


class ConfigurationError(Exception):
    """Raised when required application configuration is missing or invalid."""


@dataclass(frozen=True)
class AppSettings:
    """Validated application settings."""

    api_key: str = field(repr=False)
    model: str


def get_settings() -> AppSettings:
    """Read and validate the Groq configuration from environment variables.

    Raises:
        ConfigurationError: If the API key or model name is missing, or if the
            example placeholder key is still being used.
    """
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    model = os.getenv("GROQ_MODEL", "").strip()

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

    return AppSettings(api_key=api_key, model=model)


def create_client(settings: AppSettings | None = None) -> Groq:
    """Create a Groq client without making a network request.

    Args:
        settings: Optional validated settings. If omitted, they are loaded from
            the environment.

    Returns:
        A configured Groq client.
    """
    if settings is None:
        settings = get_settings()

    return Groq(api_key=settings.api_key)
