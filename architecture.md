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
                                                │ streamed Romanized Hindi tokens
                                          ┌─────┴──────┐
                                          │ 3a. Normalize │  SanitizedAgent
                                          │    for TTS    │  strip think tags,
                                          │               │  numbers → Hindi words,
                                          │               │  strip action markers,
                                          │               │  fix spacing
                                          └─────┬──────┘
                                                │ cleaned text
                                          ┌─────┴─────┐
                                          │  4. TTS    │  Sarvam bulbul:v3
                                          │            │  enable_preprocessing=True
                                          │            │  Romanized Hindi → speech
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
| TTS | Sarvam AI | bulbul:v3 | Speaker: aditya (male), enable_preprocessing=True, 16kHz browser / 8kHz SIP |
| Transport | LiveKit Cloud | — | WebRTC (browser) / SIP via Telnyx (phone) |

## Per-Turn Data Flow

1. **VAD** (silero) runs locally on the agent worker. Detects end-of-speech → triggers STT with the buffered audio chunk.
2. **STT** (Sarvam `saaras:v3`) receives 16kHz PCM audio, returns Hindi text transcript. Supports 22+ Indian languages with auto-detection.
3. **LLM** (Claude Haiku 3.5 or Qwen3 via vLLM) receives the full chat context: `[system, user₁, assistant₁, user₂, ...]`. Streams back a Romanized Hindi text response.
4. **SanitizedAgent post-processing** intercepts streamed LLM tokens:
   - Strips `<think>...</think>` reasoning blocks (Qwen3-specific)
   - Strips roleplay action markers (`*confused*`, `(laughs)`, etc.)
   - Converts digit numbers to Hindi words (`36000` → `chhatees hazaar`, `1.5` → `dedh`)
   - Fixes spacing at case transitions (`puraneAC` → `purane AC`)
   - Preserves leading/trailing spaces on tokens (critical for TTS sentence splitting)
5. **TTS** (Sarvam `bulbul:v3` with `enable_preprocessing=True`) receives cleaned Romanized Hindi text chunked into sentences by the LiveKit SDK's `SentenceTokenizer`. Each sentence gets a separate WebSocket session for audio generation. Sarvam's preprocessing handles Romanized Hindi → native pronunciation internally.

## Greeting Flow

The agent speaks first to simulate initiating a phone call:

```
session.say(greeting_text, add_to_chat_ctx=True)
    │
    ├── Skips VAD, STT, LLM entirely
    ├── Sends text straight to TTS → audio → browser
    └── add_to_chat_ctx=True adds it to chat history so the LLM
        knows the conversation has started and maintains caller role
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

The LLM outputs Romanized Hindi (Latin script). Normalization handles:

- **Number → Hindi word conversion** — Full Hindi number system (1-99 unique words + hazaar/lakh/crore). Special cases: 1.5 → "dedh", 2.5 → "dhaai". Comma-separated numbers supported (36,000 → chhatees hazaar).
- **Action marker stripping** — Removes `*confused*`, `(laughs)`, `[pauses]` etc.
- **Spacing fixes** — Inserts spaces at lowercase→uppercase transitions (`puraneAC` → `purane AC`), digit↔letter transitions (`5star` → `5 star`). Collapses multiple spaces.
- **No `.strip()` on tokens** — Leading/trailing spaces from LLM tokens are preserved. This is critical: the LLM tokenizer places word-boundary spaces (e.g. `" Samsung"`) that must not be removed, otherwise TTS receives concatenated text and can't split sentences properly.

No Devanagari replacement maps — Sarvam TTS with `enable_preprocessing=True` handles Romanized Hindi pronunciation internally.

### 4. end_call Tool

The `end_call` function tool is defined as a method on `SanitizedAgent` using the `@function_tool()` decorator. This ensures it is automatically registered with the LLM (module-level `@function_tool()` does NOT auto-register). When called, it removes the SIP participant to hang up the phone call.

### 5. LLM Request Logging

Every call logs:
- Message roles sent to the LLM (`[system, user, assistant, ...]`)
- Message count
- Truncated content at DEBUG level

Combined with event handlers for `user_input_transcribed`, `conversation_item_added`, `function_tools_executed`, `metrics_collected`, and `error` — gives full visibility into what's happening at each stage.

## System Prompt

The agent prompt uses plain-text sections for natural LLM behavior:

- **VOICE & TONE** — Casual spoken Hindi/Hinglish, natural fillers, short 1-2 line responses.
- **WHAT YOU CARE ABOUT** — Price, installation, warranty, exchange, availability.
- **WHAT YOU DON'T CARE ABOUT** — Technical specs, Wi-Fi, smart features.
- **CONVERSATION FLOW** — Confirm shop → ask about product → get price → negotiate → wrap up → call end_call tool.
- **CRITICAL OUTPUT RULES** — Romanized Hindi only (no Devanagari), numbers as Hindi words, no action markers, no `[end_call]` as text.
- **EXAMPLES** — Few-shot examples showing correct tone and format.
- **Per-call context** — PRODUCT and STORE appended at dispatch time.

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
    │     ├── GET /             → HTML page with LiveKit JS SDK
    │     ├── GET /api/token    → generates JWT + dispatches agent
    │     ├── GET /api/logs     → returns agent worker log tail
    │     └── GET /api/metrics  → returns dashboard metrics as JSON
    └── Cleans up agent worker on exit (Ctrl+C)
```

Single command to start everything. Audio flows over WebRTC at 16kHz.

### SIP (Phone)

- Agent worker receives dispatch with phone number in metadata
- Creates SIP participant via Telnyx trunk → dials the shop
- Audio flows at 8kHz (telephony standard)
- 2-minute timeout auto-disconnects
- `end_call` function tool lets the LLM hang up when done

## Per-Call Logging & Transcripts

Each call creates:
- **Log file** in `logs/` — DEBUG-level logs from all components (agent, LLM, STT, TTS, LiveKit SDK)
- **Transcript JSON** in `transcripts/` — Structured record with store name, AC model, room, phone, timestamped messages

Log file naming: `{Store_Name}_{YYYYMMDD_HHMMSS}.log`
Transcript naming: `{Store_Name}_{YYYYMMDD_HHMMSS}.json`

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
| `agent_worker.py` | Core — SanitizedAgent, LLM provider switch, TTS normalization, end_call tool, logging |
| `test_browser.py` | Browser test server — auto-manages agent worker, WebRTC UI + metrics API on port 8080 |
| `dashboard.py` | Metrics dashboard — parses logs/transcripts, serves HTML on port 9090 |
| `test_sarvam.py` | Standalone Sarvam API tests (STT/TTS/LLM) |
| `tests/` | pytest test suite — 95 unit tests + 6 live API tests |
| `.env.local` | API keys and config (gitignored) |
