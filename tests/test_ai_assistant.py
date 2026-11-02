"""Tests for configuration and Groq client setup."""

import pytest
from groq import Groq

from modules.ai_assistant import (
    AppSettings,
    ConfigurationError,
    create_client,
    get_settings,
)


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

    settings = get_settings()

    assert settings.api_key == "gsk_test_key"
    assert settings.model == "test-model"


def test_api_key_is_hidden_from_settings_repr():
    settings = AppSettings(api_key="gsk_secret_value", model="test-model")

    assert "gsk_secret_value" not in repr(settings)


def test_create_client_returns_groq_client():
    # This checks client construction only; no network request is made.
    settings = AppSettings(api_key="gsk_test_key", model="test-model")

    client = create_client(settings)

    assert isinstance(client, Groq)
