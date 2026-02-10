# Offline Discovery — Voice AI Price Enquiry Agent

An automated voice AI agent that calls local electronics stores to collect AC price quotes. Supports both **SIP phone calls** and **browser-based WebRTC** conversations. Built with **Sarvam AI** (Hindi STT/TTS) + **LiveKit** (real-time audio) + **OpenAI-compatible LLM**.

## How It Works

```
Browser/Phone ──▶ LiveKit Room ──▶ Agent Worker
                                     │
                              ┌──────┼──────┐
                              ▼      ▼      ▼
                           Sarvam  LLM   Sarvam
                           STT    (any)   TTS
                         (saarika) OpenAI (bulbul v3)
                                  compat.
```

1. User connects via browser (WebRTC) or phone (SIP)
2. **Sarvam STT** (saarika:v2.5) transcribes Hindi/Hinglish speech
3. **LLM** generates natural Hindi responses as a customer enquiring about AC prices
4. **Sarvam TTS** (bulbul:v3) speaks the response back
5. Agent asks about price, exchange offers, warranty, installation, and availability

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
# Fill in your API keys (see .env.example for details)
```

You need:
- **Sarvam AI** key — [dashboard.sarvam.ai](https://dashboard.sarvam.ai) (free credits on signup)
- **LiveKit Cloud** — [cloud.livekit.io](https://cloud.livekit.io) (free tier: 5,000 participant-minutes/month)
- **SIP trunk** (optional, for phone calls) — Telnyx or Twilio

### 3. Test Sarvam APIs

```bash
python test_sarvam.py
```

### 4. Browser Test (no phone needed)

```bash
# Terminal 1
python agent_worker.py dev

# Terminal 2
python test_browser.py
```

Open http://localhost:8080, click **Start Conversation**, and talk to the agent in Hindi/Hinglish.

### 5. SIP Phone Calls (production)

```bash
# Start the agent worker
python agent_worker.py dev

# Run the orchestrator with store list
python main.py
```

## Project Structure

```
├── agent_worker.py      # LiveKit agent — handles voice conversation (SIP + browser)
├── test_browser.py      # Browser test server — WebRTC UI with audio viz & logs
├── test_sarvam.py       # Standalone Sarvam API tests
├── main.py              # Orchestrator — manages calling campaign
├── caller.py            # Call dispatch & transcript extraction
├── stores.json          # Store database (names, phone numbers)
├── outbound-trunk.json  # LiveKit SIP trunk config
├── .env.example         # Environment variables template
└── requirements.txt     # Python dependencies
```

## Tech Stack

| Component | Service | Model/Config |
|-----------|---------|-------------|
| STT | Sarvam AI | saarika:v2.5, hi-IN, 8kHz |
| TTS | Sarvam AI | bulbul:v3, speaker: aditya |
| LLM | Sarvam AI (OpenAI-compatible) | sarvam-m |
| Audio | LiveKit | WebRTC (browser) / SIP (phone) |
| VAD | Silero | Via LiveKit agents SDK |

## Browser Test Features

- Real-time **audio visualizers** (mic input + agent output)
- **Event log** with timestamps for connection, tracks, errors
- **Agent worker logs** viewer with auto-refresh and color coding
- Works with any browser that supports WebRTC

## Configuration

The agent behavior is controlled by the `DEFAULT_INSTRUCTIONS` prompt in `agent_worker.py`. The agent speaks casual Hindi/Hinglish and focuses on price, installation, warranty, and availability.

LLM, TTS model, and speaker can be changed in the `entrypoint()` function in `agent_worker.py`.
