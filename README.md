# Hyperlocal Discovery — Voice AI Price Enquiry Agent

An automated voice AI agent that calls local AC shops to collect price quotes in natural Hindi/Hinglish. Supports **browser-based WebRTC** and **SIP phone calls**. Built with **LiveKit** + **Sarvam AI** (Hindi STT/TTS) + **Claude Haiku 3.5** (or self-hosted Qwen3).

## How It Works

```
Browser/Phone ──▶ LiveKit Room ──▶ Agent Worker
                                     │
                              ┌──────┼──────┐
                              ▼      ▼      ▼
                           Sarvam  LLM   Sarvam
                           STT    Claude  TTS
                         (saaras  Haiku  (bulbul
                           v3)    3.5     v3)
```

1. User connects via browser (WebRTC) or phone (SIP)
2. **Sarvam STT** (saaras:v3) transcribes Hindi/Hinglish speech at 16kHz
3. **LLM** (Claude Haiku 3.5 or Qwen3 via vLLM) generates natural Romanized Hindi responses as a customer enquiring about AC prices
4. **SanitizedAgent** normalizes LLM output — strips think tags, action markers, converts numbers to Hindi words, fixes spacing
5. **Sarvam TTS** (bulbul:v3, `enable_preprocessing=True`) speaks the response back, handling Romanized Hindi pronunciation internally
6. Agent follows a natural conversation flow: confirm shop → ask about product → get price → negotiate → wrap up

## Quick Start

### 1. Install

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env.local
# Fill in your API keys
```

You need:
- **Sarvam AI** key — [dashboard.sarvam.ai](https://dashboard.sarvam.ai) (free credits on signup)
- **LiveKit Cloud** — [cloud.livekit.io](https://cloud.livekit.io) (free tier: 5,000 participant-minutes/month)
- **Anthropic API** key — for Claude Haiku 3.5 (or use self-hosted Qwen3 via vLLM)
- **SIP trunk** (optional, for phone calls) — Telnyx or Twilio

### 3. Browser Test

```bash
python test_browser.py
```

This single command kills old agent workers, starts a fresh one, and serves the browser UI. Open http://localhost:8080, click **Start Conversation**, and talk in Hindi/Hinglish. Press Ctrl+C to stop everything.

### 4. SIP Phone Calls (production)

```bash
python agent_worker.py dev
python main.py
```

## Project Structure

```
├── agent_worker.py      # LiveKit agent — SanitizedAgent, LLM switch, TTS normalization
├── test_browser.py      # Browser test server — auto-manages agent, WebRTC UI on :8080
├── dashboard.py         # Metrics dashboard — parses logs/transcripts, serves on :9090
├── test_sarvam.py       # Standalone Sarvam API tests
├── tests/               # pytest test suite (95 unit tests + 6 live API tests)
├── architecture.md      # Detailed pipeline and component documentation
├── tests.md             # Testing guide
├── .env.local           # API keys and config (not committed)
└── requirements.txt     # Python dependencies
```

## Tech Stack

| Component | Service | Model/Config |
|-----------|---------|-------------|
| STT | Sarvam AI | saaras:v3, hi-IN, 16kHz |
| TTS | Sarvam AI | bulbul:v3, speaker: aditya, enable_preprocessing=True |
| LLM (default) | Anthropic | Claude Haiku 3.5 |
| LLM (alt) | Self-hosted vLLM | Qwen/Qwen3-4B-Instruct-2507-FP8 |
| Audio | LiveKit | WebRTC (browser, 16kHz) / SIP (phone, 8kHz) |
| VAD | Silero | Via LiveKit agents SDK |

Switch LLM provider via `LLM_PROVIDER` env var (`claude` or `qwen`).

## Browser Test Features

- Real-time **audio visualizers** (mic input + agent output)
- **Event log** with timestamps for connection, tracks, errors
- **Agent worker logs** viewer with auto-refresh and color coding
- **Auto agent management** — starts/stops agent worker automatically
- **Metrics dashboard** via `/api/metrics` endpoint

## Configuration

The agent prompt in `agent_worker.py` uses plain-text sections (VOICE & TONE, CONVERSATION FLOW, CRITICAL OUTPUT RULES, EXAMPLES) for natural LLM behavior. The agent speaks casual Hindi/Hinglish as a regular customer calling to ask about AC prices.

Key design decisions:
- LLM outputs **Romanized Hindi** (Latin script) — no Devanagari
- Sarvam TTS with `enable_preprocessing=True` handles pronunciation internally
- Numbers converted to Hindi words before TTS (e.g. "36000" → "chhatees hazaar")
- `end_call` tool is a method on `SanitizedAgent` for automatic registration with the LLM

See [architecture.md](architecture.md) for detailed pipeline documentation.
