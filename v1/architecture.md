# Architecture — AC Price Enquiry Agent

## Pipeline Overview

```
Browser (mic) ──WebRTC──▶ LiveKit Cloud ──▶ Agent Worker
                                                │
                                          ┌─────┴─────┐
                                          │  1. VAD    │  silero
                                          │            │  detects speech start/stop
                                          └─────┬─────┘
                                                │ audio frames
                                          ┌─────┴─────┐
                                          │  2. STT    │  Sarvam saaras:v3
                                          │            │  Hindi speech → text
                                          └─────┬─────┘
                                                │ text transcript
                                          ┌─────┴─────┐
                                          │  3. LLM    │  Claude Haiku 3.5 (default)
                                          │            │  or Qwen3-4B via vLLM
                                          └─────┬─────┘
                                                │ streamed text response
                                          ┌─────┴──────┐
                                          │ 3a. Normalize │  SanitizedAgent
                                          │    for TTS    │  strip think tags,
                                          │               │  Hindi phonetics,
                                          │               │  strip action markers
                                          └─────┬──────┘
                                                │ cleaned text
                                          ┌─────┴─────┐
                                          │  4. TTS    │  Sarvam bulbul:v3
                                          │            │  Hindi text → speech
                                          └─────┬─────┘
                                                │ audio frames
                                                ▼
                                          LiveKit Cloud ──WebRTC──▶ Browser (speaker)
```

## Components

| Stage | Provider | Model | Config |
|-------|----------|-------|--------|
| VAD | silero | silero-vad | Local inference, no API call |
| STT | Sarvam AI | saaras:v3 | hi-IN, 16kHz sample rate |
| LLM (default) | Anthropic | claude-3-5-haiku-20241022 | Via `livekit-plugins-anthropic` |
| LLM (alt) | Self-hosted vLLM | Qwen/Qwen3-4B-Instruct-2507-FP8 | `192.168.0.42:8000`, OpenAI-compatible |
| TTS | Sarvam AI | bulbul:v3 | Speaker: aditya (male), 24kHz browser / 8kHz SIP |
| Transport | LiveKit Cloud | — | WebRTC (browser) / SIP via Telnyx (phone) |

## Per-Turn Data Flow

1. **VAD** (silero) runs locally on the agent worker. Detects end-of-speech → triggers STT with the buffered audio chunk.
2. **STT** (Sarvam `saaras:v3`) receives 16kHz PCM audio, returns Hindi text transcript via `POST api.sarvam.ai/speech-to-text`. Supports 22+ Indian languages with auto-detection.
3. **LLM** (Claude Haiku 3.5 or Qwen3 via vLLM) receives the full chat context: `[system, user₁, assistant₁, user₂, ...]`. Streams back a Hindi text response.
4. **SanitizedAgent post-processing** intercepts streamed LLM output:
   - Strips `<think>...</think>` reasoning blocks (Qwen3-specific)
   - Strips roleplay action markers (`*confused*`, `(laughs)`, etc.)
   - Normalizes English terms to Hindi phonetics for TTS (e.g. "AC" → "ए सी", "Samsung" → "सैमसंग")
5. **TTS** (Sarvam `bulbul:v3`) receives cleaned text chunked into sentences by the LiveKit SDK. Each sentence is sent independently to `POST api.sarvam.ai/text-to-speech` and audio plays back as sentences complete.

## Greeting Flow

The agent speaks first to simulate initiating a phone call. This uses a special path:

```
session.say(greeting_text, add_to_chat_ctx=False)
    │
    ├── Skips VAD, STT, LLM entirely
    ├── Sends text straight to TTS → audio → browser
    └── add_to_chat_ctx=False keeps it out of chat history
```

The greeting confirms the shop identity ("Hello, yeh [store] hai? Aap log AC dealer ho?") before asking about products — matching natural Indian phone call conventions.

## SanitizedAgent

Custom `Agent` subclass (`agent_worker.py`) that wraps every LLM call with these protections:

### 1. Chat Context Sanitization

vLLM rejects requests where the first non-system message isn't from the user (HTTP 400). The agent inspects the chat context before each call and removes any stray assistant message that appears before the first user message.

```
Before: [system, assistant, user, ...]  → 400 error from vLLM
After:  [system, user, ...]             → works
```

### 2. Think-Tag Stripping

Qwen3 emits `<think>reasoning here</think>` blocks in its output. These are stripped via regex before text reaches TTS, so the agent doesn't speak its internal reasoning aloud. Handles both complete and streaming partial cases.

### 3. TTS Text Normalization

English abbreviations and brand names are replaced with Hindi phonetic equivalents so Sarvam TTS pronounces them correctly:

| English | Hindi phonetic |
|---------|---------------|
| AC | ए सी |
| Samsung | सैमसंग |
| EMI | ई एम आई |
| LG | एल जी |
| inverter | इन्वर्टर |
| ... | (30+ mappings) |

Roleplay action markers (`*confused*`, `(laughs)`, `[pauses]`) are also stripped.

### 4. LLM Request Logging

Every call logs:
- Message roles sent to the LLM (`[system, user, assistant, ...]`)
- Message count
- Truncated content at DEBUG level

Combined with event handlers for `user_input_transcribed`, `conversation_item_added`, `function_tools_executed`, `metrics_collected`, and `error` — gives full visibility into what's happening at each stage.

## System Prompt

The agent prompt uses XML-structured sections for reliable parsing by the LLM:

- `<role>` — Anchors the agent as the CALLER (not the shopkeeper). Explicitly states it does NOT know prices.
- `<voice_and_tone>` — Semi-formal Hindi/Hinglish, "bhai sahab"/"bhaiya" register, short responses.
- `<conversation_flow>` — 6 phases: confirm shop → ask about product → ask price → negotiate → extras → wrap up.
- `<rules>` — Hard guardrails against role-switching, spec-asking, or dragging out the call.
- `<output_format>` — TTS-specific constraints (no action markers, no echoing).
- `<examples>` — 3 few-shot examples showing correct tone and behavior.
- `<session_context>` — Per-call product and store name injected at dispatch time.

## LLM Provider Selection

Controlled by `LLM_PROVIDER` env var:

| Value | Provider | Model | Notes |
|-------|----------|-------|-------|
| `claude` | Anthropic API | claude-3-5-haiku-20241022 | Better role adherence, structured prompt support |
| `qwen` (default) | Self-hosted vLLM | Qwen/Qwen3-4B-Instruct-2507-FP8 | Free, requires GPU server at 192.168.0.42 |

## Session Modes

### Browser (WebRTC)

```
python test_browser.py
    │
    ├── Kills old agent workers automatically
    ├── Starts a fresh agent_worker.py dev process
    ├── Serves HTTP on port 8080
    │     ├── GET /           → HTML page with LiveKit JS SDK
    │     ├── GET /api/token  → generates JWT + dispatches agent
    │     └── GET /api/logs   → returns agent worker log tail
    └── Cleans up agent worker on exit (Ctrl+C)
```

Single command to start everything. Audio flows over WebRTC at 24kHz.

### SIP (Phone)

- Agent worker receives dispatch with phone number in metadata
- Creates SIP participant via Telnyx trunk → dials the shop
- Audio flows at 8kHz (telephony standard)
- 2-minute timeout auto-disconnects
- `end_call` function tool lets the LLM hang up when done

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `SARVAM_API_KEY` | Sarvam AI API key (STT + TTS) |
| `LLM_PROVIDER` | LLM backend: `claude` or `qwen` (default: `qwen`) |
| `ANTHROPIC_API_KEY` | Anthropic API key (when LLM_PROVIDER=claude) |
| `LLM_BASE_URL` | vLLM server endpoint (when LLM_PROVIDER=qwen) |
| `LLM_MODEL` | Model name on vLLM (when LLM_PROVIDER=qwen) |
| `LLM_API_KEY` | vLLM auth key (when LLM_PROVIDER=qwen) |
| `LIVEKIT_URL` | LiveKit Cloud WebSocket URL |
| `LIVEKIT_API_KEY` | LiveKit API key |
| `LIVEKIT_API_SECRET` | LiveKit API secret |
| `SIP_OUTBOUND_TRUNK_ID` | LiveKit SIP trunk ID (for phone calls) |

## Files

| File | Role |
|------|------|
| `agent_worker.py` | Core — SanitizedAgent, LLM provider switch, TTS normalization, tools, logging |
| `test_browser.py` | Browser test server — auto-manages agent worker, WebRTC UI on port 8080 |
| `test_sarvam.py` | Standalone Sarvam API tests (STT/TTS/LLM) |
| `.env.local` | API keys and config (gitignored) |
