"""Tests for pipeline/intake.py — suggestions parsing and requirements extraction."""

import pytest
import sys
import os
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline.intake import _REQUIREMENTS_RE, _SUGGESTIONS_RE


class TestSuggestionsRegex:
    def test_parses_suggestions(self):
        text = 'What type of AC?\n<suggestions>1.5 ton split AC|1 ton window AC|Not sure</suggestions>'
        match = _SUGGESTIONS_RE.search(text)
        assert match is not None
        chips = [s.strip() for s in match.group(1).split("|")]
        assert chips == ["1.5 ton split AC", "1 ton window AC", "Not sure"]

    def test_no_suggestions(self):
        text = "What type of AC are you looking for?"
        match = _SUGGESTIONS_RE.search(text)
        assert match is None

    def test_suggestions_with_whitespace(self):
        text = '<suggestions> Option A | Option B | Option C </suggestions>'
        match = _SUGGESTIONS_RE.search(text)
        assert match is not None
        chips = [s.strip() for s in match.group(1).strip().split("|")]
        assert chips == ["Option A", "Option B", "Option C"]


class TestRequirementsRegex:
    def test_parses_requirements(self):
        text = '''Got it!
<requirements>
{"product_type": "AC", "category": "1.5 ton split AC", "location": "Bangalore"}
</requirements>'''
        match = _REQUIREMENTS_RE.search(text)
        assert match is not None
        import json
        data = json.loads(match.group(1))
        assert data["product_type"] == "AC"

    def test_strips_both_tags(self):
        text = 'Response <requirements>{"a":1}</requirements> and <suggestions>A|B</suggestions>'
        cleaned = _REQUIREMENTS_RE.sub("", text)
        cleaned = _SUGGESTIONS_RE.sub("", cleaned).strip()
        assert cleaned == "Response  and"


class TestSystemPromptContent:
    """Verify system prompt includes product-specific questions and suggestion format."""

    def test_product_specific_questions_in_prompt(self):
        from pipeline.intake import SYSTEM_PROMPT
        assert "tonnage" in SYSTEM_PROMPT.lower() or "ton" in SYSTEM_PROMPT.lower()
        assert "front load" in SYSTEM_PROMPT.lower() or "top load" in SYSTEM_PROMPT.lower()
        assert "single or double door" in SYSTEM_PROMPT.lower() or "single door" in SYSTEM_PROMPT.lower()
        assert "use case" in SYSTEM_PROMPT.lower()

    def test_suggestion_format_in_prompt(self):
        from pipeline.intake import SYSTEM_PROMPT
        assert "<suggestions>" in SYSTEM_PROMPT
        assert "clickable chips" in SYSTEM_PROMPT.lower() or "chips" in SYSTEM_PROMPT.lower()
