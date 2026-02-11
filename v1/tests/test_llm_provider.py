"""Tests for _create_llm — LLM provider selection based on LLM_PROVIDER env var."""

import os
from unittest.mock import patch, MagicMock

from tests.conftest import _create_llm


class TestCreateLlm:
    def test_default_is_qwen(self):
        """Default (no LLM_PROVIDER set) should use Qwen via openai plugin."""
        with patch.dict(os.environ, {"LLM_PROVIDER": ""}, clear=False):
            with patch("agent_worker.openai.LLM") as mock_openai:
                mock_openai.return_value = MagicMock()
                _create_llm()
                mock_openai.assert_called_once()

    def test_qwen_explicit(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "qwen"}, clear=False):
            with patch("agent_worker.openai.LLM") as mock_openai:
                mock_openai.return_value = MagicMock()
                _create_llm()
                mock_openai.assert_called_once()

    def test_claude_provider(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "claude"}, clear=False):
            with patch("agent_worker.anthropic.LLM") as mock_anthropic:
                mock_anthropic.return_value = MagicMock()
                _create_llm()
                mock_anthropic.assert_called_once()
                call_kwargs = mock_anthropic.call_args.kwargs
                assert "haiku" in call_kwargs.get("model", "")

    def test_claude_case_insensitive(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "CLAUDE"}, clear=False):
            with patch("agent_worker.anthropic.LLM") as mock_anthropic:
                mock_anthropic.return_value = MagicMock()
                _create_llm()
                mock_anthropic.assert_called_once()

    def test_custom_model_env(self):
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "qwen",
            "LLM_MODEL": "CustomModel/v1",
            "LLM_BASE_URL": "http://10.0.0.1:8000/v1",
        }, clear=False):
            with patch("agent_worker.openai.LLM") as mock_openai:
                mock_openai.return_value = MagicMock()
                _create_llm()
                call_kwargs = mock_openai.call_args.kwargs
                assert call_kwargs["model"] == "CustomModel/v1"
                assert call_kwargs["base_url"] == "http://10.0.0.1:8000/v1"

    def test_temperature_set(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "claude"}, clear=False):
            with patch("agent_worker.anthropic.LLM") as mock_anthropic:
                mock_anthropic.return_value = MagicMock()
                _create_llm()
                assert mock_anthropic.call_args.kwargs["temperature"] == 0.7
