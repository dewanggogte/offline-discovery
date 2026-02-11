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
| TTS Normalization | `test_normalization.py` | 53 | Hindi number conversion, spacing fixes, action markers, think tags, full pipeline |
| Chat Context Sanitization | `test_sanitize.py` | 8 | `SanitizedAgent._sanitize_chat_ctx` message ordering |
| LLM Provider Selection | `test_llm_provider.py` | 6 | `_create_llm` switching between Claude/Qwen |
| Transcript Saving | `test_transcript.py` | 10 | JSON schema, real file validation, save/reload roundtrip |
| Per-Call Logging | `test_logs.py` | 6 | `_setup_call_logger` file creation, handler lifecycle |
| Conversation Quality | `test_conversation.py` | 11 | Role adherence, response length, prompt structure |
| Sarvam STT (live) | `test_stt_live.py` | 2 | STT API response, language detection |
| Sarvam TTS (live) | `test_tts_live.py` | 4 | TTS audio output, normalized text, sample rates |

**Total: 95 passed, 6 skipped** (skipped = live tests requiring `--live` flag)

## Test Details

### TTS Normalization (`test_normalization.py`)

Tests the full text normalization pipeline that converts LLM output to TTS-friendly text:

- **Hindi number conversion** (13 tests) — `_number_to_hindi()` for single digits, double digits (1-99 unique words), hundreds, thousands (hazaar), lakhs. Edge cases: 0, 99.
- **Number replacement in text** (10 tests) — `_replace_numbers()` for in-sentence numbers, comma-separated (36,000), ranges (10-12), special cases (1.5→dedh, 2.5→dhaai).
- **Spacing fixes** (8 tests) — lowercase→uppercase transitions (`puraneAC` → `purane AC`), digit↔letter transitions (`5star` → `5 star`), multiple space collapse.
- **Action marker stripping** (6 tests) — Removes `*confused*`, `(laughs)`, `[pauses]`, markers mid-sentence, multiple markers.
- **Think-tag stripping** (8 tests) — Complete tags, multiline, unclosed (streaming), empty, multiple.
- **Full pipeline** (8 tests) — Combined normalization with real LLM output patterns.

### Chat Context Sanitization (`test_sanitize.py`)

Tests `SanitizedAgent._sanitize_chat_ctx()` which ensures the first non-system message is from the user (required by vLLM/Qwen). Uses real `ChatContext` Pydantic objects — no mocking needed.

### LLM Provider Selection (`test_llm_provider.py`)

Tests `_create_llm()` returns the correct provider based on `LLM_PROVIDER` env var. Mocks `anthropic.LLM` and `openai.LLM` constructors to avoid actual API connections.

### Conversation Quality (`test_conversation.py`)

Rule-based checks on real transcript data (no judge LLM needed):
- Agent never uses shopkeeper phrases
- Responses stay under 300 characters
- No action markers or think tags leak through (checks 3 most recent transcripts)
- System prompt has required sections (VOICE & TONE, CONVERSATION FLOW, OUTPUT RULES, EXAMPLES)
- Prompt specifies caller role and end_call tool

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

Also accessible via the browser test server at `GET /api/metrics` (returns JSON).

Shows: test results, latency metrics (TTFT, turn latency), token usage, conversation analytics, normalization report.

## Adding New Tests

When modifying normalization in `agent_worker.py`:
1. Add tests for the specific function (`_number_to_hindi`, `_replace_numbers`, `_normalize_for_tts`)
2. Test edge cases (empty input, no matches, special characters)
3. Add a full-pipeline test if the change affects multiple normalization stages
4. Run `pytest tests/ -v` to verify all 95 tests still pass
