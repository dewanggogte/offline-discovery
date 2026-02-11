# Testing Guide

## Overview

The test suite covers every component of the voice AI pipeline — from text normalization and chat context handling to live Sarvam API integration. Tests are split into **unit tests** (no API keys needed) and **live integration tests** (require `.env.local`).

## Quick Start

```bash
# One-time setup
pip install pytest pytest-asyncio

# Run unit tests (no API keys needed)
pytest tests/ -v

# Run everything including live API tests
pytest tests/ --live -v

# Run a single test file
pytest tests/test_normalization.py -v

# Run with short output
pytest tests/ -q
```

## Component Test Matrix

| Component | Test File | Tests | What It Covers |
|-----------|-----------|-------|---------------|
| TTS Normalization | `test_normalization.py` | ~55 | `_normalize_for_tts`, `_strip_think_tags`, `_ACTION_RE` |
| Chat Context Sanitization | `test_sanitize.py` | 8 | `SanitizedAgent._sanitize_chat_ctx` message ordering |
| LLM Provider Selection | `test_llm_provider.py` | 6 | `_create_llm` switching between Claude/Qwen |
| Transcript Saving | `test_transcript.py` | 10 | JSON schema, real file validation, save/reload roundtrip |
| Per-Call Logging | `test_logs.py` | 6 | `_setup_call_logger` file creation, handler lifecycle |
| Conversation Quality | `test_conversation.py` | 8 | Role adherence, response length, prompt structure |
| Sarvam STT (live) | `test_stt_live.py` | 2 | STT API response, language detection |
| Sarvam TTS (live) | `test_tts_live.py` | 4 | TTS audio output, normalized text, sample rates |

## Test Details

### TTS Normalization (`test_normalization.py`)

Tests the full text normalization pipeline that converts LLM output to TTS-friendly Hindi:

- **Abbreviation replacements** — AC→ए सी, EMI→ई एम आई, etc. Case-sensitive with word boundaries so "AC" doesn't match inside "Achha"
- **Word replacements** — rate→रेट, price→प्राइस, Samsung→सैमसंग, etc. Case-insensitive, no word boundaries (handles concatenated LLM output like "aapsepricepooch")
- **Script-boundary spacing** — Inserts spaces at Devanagari↔Latin/digit transitions. Fixes "सैमसंग1.5टन" → "सैमसंग 1.5 टन"
- **Action marker stripping** — Removes *confused*, (laughs), [pauses]
- **Think-tag stripping** — Removes Qwen3 `<think>...</think>` blocks
- **Full pipeline** — Tests with real LLM output patterns from actual transcripts

### Chat Context Sanitization (`test_sanitize.py`)

Tests `SanitizedAgent._sanitize_chat_ctx()` which ensures the first non-system message is from the user (required by vLLM/Qwen). Uses real `ChatContext` Pydantic objects — no mocking needed.

### LLM Provider Selection (`test_llm_provider.py`)

Tests `_create_llm()` returns the correct provider based on `LLM_PROVIDER` env var. Mocks `anthropic.LLM` and `openai.LLM` constructors to avoid actual API connections.

### Conversation Quality (`test_conversation.py`)

Rule-based checks on real transcript data (no judge LLM needed):
- Agent never uses shopkeeper phrases
- Responses stay under 300 characters
- No action markers or think tags leak through
- System prompt has all required XML sections

## Mocking Strategy

- **ChatContext** — Real Pydantic models from LiveKit SDK. Instantiated with `ChatContext()` + `add_message(role=..., content=...)`. No network needed.
- **LLM constructors** — Mocked with `unittest.mock.patch` to avoid API initialization.
- **Log handlers** — Tested directly since `_setup_call_logger` only creates files. Each test cleans up handlers in `try/finally`.
- **Live APIs** — Only called when `--live` flag is passed.

## Live Test Requirements

To run `pytest tests/ --live`, you need `.env.local` with:
```
SARVAM_API_KEY=<your_key>
```

Live tests generate audio via TTS, transcribe it via STT, and validate the API responses. They consume Sarvam API credits.

## Dashboard

```bash
python dashboard.py
# Opens http://localhost:9090
```

Shows: test results, latency metrics (TTFT, turn latency), token usage, conversation analytics, normalization report.

## Adding New Tests

When adding a new word to `_TTS_REPLACEMENTS` in `agent_worker.py`:
1. Add a test in `TestWordReplacements` or `TestAbbreviationReplacements`
2. If it's an abbreviation (all-caps, <=4 chars), test word-boundary protection
3. Test it in a concatenated context if it's commonly used in Hinglish
