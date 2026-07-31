"""Tests for AI Core configuration models and loaders."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from ai_core.config.loader import ConfigLoader
from ai_core.config.models import GenerationConfig, LLMConfig, LLMProviderConfig
from ai_core.llm.exceptions import LLMConfigurationError


def test_generation_config_serialization():
    cfg = GenerationConfig(temperature=0.7, max_output_tokens=100, top_p=0.9)
    data = cfg.to_dict()
    assert data == {"temperature": 0.7, "max_output_tokens": 100, "top_p": 0.9}

    cfg2 = GenerationConfig.from_dict(data)
    assert cfg2.temperature == 0.7
    assert cfg2.max_output_tokens == 100
    assert cfg2.top_p == 0.9
    assert cfg2.top_k is None


def test_llm_provider_config_from_dict():
    raw = {
        "provider_name": "gemini",
        "model_name": "gemini-2.5-pro",
        "api_key": "secret-token",
        "timeout_seconds": 45.0,
        "default_generation_config": {"temperature": 0.2},
    }
    provider = LLMProviderConfig.from_dict(raw)
    assert provider.provider_name == "gemini"
    assert provider.model_name == "gemini-2.5-pro"
    assert provider.api_key == "secret-token"
    assert provider.timeout_seconds == 45.0
    assert provider.default_generation_config.temperature == 0.2


def test_config_loader_dict():
    raw = {
        "default_provider": "gemini",
        "providers": {
            "gemini": {
                "model_name": "gemini-2.5-flash",
                "api_key": "test-key",
            }
        },
    }
    cfg = ConfigLoader.load_from_dict(raw)
    assert cfg.default_provider == "gemini"
    provider_cfg = cfg.get_provider_config("gemini")
    assert provider_cfg.model_name == "gemini-2.5-flash"
    assert provider_cfg.api_key == "test-key"


def test_config_loader_json_file():
    with TemporaryDirectory() as tmp_dir:
        json_path = Path(tmp_dir) / "config.json"
        content = {
            "default_provider": "gemini",
            "providers": {
                "gemini": {
                    "model_name": "gemini-2.5-pro",
                    "api_key": "file-key",
                }
            },
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(content, f)

        cfg = ConfigLoader.load(json_path)
        assert cfg.default_provider == "gemini"
        assert cfg.get_provider_config().api_key == "file-key"


def test_config_loader_missing_file():
    with pytest.raises(LLMConfigurationError):
        ConfigLoader.load_from_json_file("/non/existent/path/config.json")


def test_config_loader_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    monkeypatch.setenv("GEMINI_TEMPERATURE", "0.5")

    cfg = ConfigLoader.load_from_env()
    provider_cfg = cfg.get_provider_config("gemini")
    assert provider_cfg.api_key == "env-key"
    assert provider_cfg.model_name == "gemini-2.5-flash-lite"
    assert provider_cfg.default_generation_config.temperature == 0.5
