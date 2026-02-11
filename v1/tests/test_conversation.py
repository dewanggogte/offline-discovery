"""Tests for conversation quality — rule-based checks on transcripts and prompt structure."""

import json
import re
import pytest
from pathlib import Path

from tests.conftest import DEFAULT_INSTRUCTIONS

TRANSCRIPTS_DIR = Path(__file__).parent.parent / "transcripts"


def _load_transcripts():
    if not TRANSCRIPTS_DIR.exists():
        return []
    return [json.loads(f.read_text()) for f in sorted(TRANSCRIPTS_DIR.glob("*.json"))]


class TestConversationQuality:
    """Offline quality checks on real transcript data."""

    def test_agent_never_claims_shopkeeper_role(self):
        transcripts = _load_transcripts()
        if not transcripts:
            pytest.skip("No transcripts available")
        shopkeeper_phrases = ["i am the shopkeeper", "main dukandaar", "hamare yahan", "humara price"]
        for t in transcripts:
            for msg in t["messages"]:
                if msg["role"] == "assistant":
                    text = msg["text"].lower()
                    for phrase in shopkeeper_phrases:
                        assert phrase not in text, f"Shopkeeper phrase '{phrase}' in: {msg['text']}"

    def test_agent_messages_are_short(self):
        """Agent responses should be concise — under 300 chars."""
        transcripts = _load_transcripts()
        if not transcripts:
            pytest.skip("No transcripts available")
        for t in transcripts:
            for msg in t["messages"]:
                if msg["role"] == "assistant":
                    assert len(msg["text"]) < 300, (
                        f"Response too long ({len(msg['text'])} chars): {msg['text'][:100]}..."
                    )

    def test_no_action_markers_in_output(self):
        transcripts = _load_transcripts()
        if not transcripts:
            pytest.skip("No transcripts available")
        action_re = re.compile(r"[\*\(\[][a-zA-Z\s]+[\*\)\]]")
        for t in transcripts:
            for msg in t["messages"]:
                if msg["role"] == "assistant":
                    assert not action_re.search(msg["text"]), f"Action marker found: {msg['text']}"

    def test_no_think_tags_in_output(self):
        transcripts = _load_transcripts()
        if not transcripts:
            pytest.skip("No transcripts available")
        for t in transcripts:
            for msg in t["messages"]:
                if msg["role"] == "assistant":
                    assert "<think>" not in msg["text"]
                    assert "</think>" not in msg["text"]


class TestSystemPromptStructure:
    """Validate the system prompt has required XML sections."""

    def test_has_role_section(self):
        assert "<role>" in DEFAULT_INSTRUCTIONS
        assert "</role>" in DEFAULT_INSTRUCTIONS

    def test_has_conversation_flow(self):
        assert "<conversation_flow>" in DEFAULT_INSTRUCTIONS
        assert "</conversation_flow>" in DEFAULT_INSTRUCTIONS

    def test_has_output_format(self):
        assert "<output_format>" in DEFAULT_INSTRUCTIONS
        assert "</output_format>" in DEFAULT_INSTRUCTIONS

    def test_has_examples(self):
        assert "<examples>" in DEFAULT_INSTRUCTIONS
        assert "</examples>" in DEFAULT_INSTRUCTIONS

    def test_has_rules(self):
        assert "<rules>" in DEFAULT_INSTRUCTIONS
        assert "</rules>" in DEFAULT_INSTRUCTIONS

    def test_specifies_caller_role(self):
        assert "CALLER" in DEFAULT_INSTRUCTIONS
        assert "SHOPKEEPER" in DEFAULT_INSTRUCTIONS

    def test_specifies_end_call_tool(self):
        assert "end_call" in DEFAULT_INSTRUCTIONS
