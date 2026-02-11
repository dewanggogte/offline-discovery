"""
agent_worker.py — LiveKit Agent Worker
=======================================
Voice AI agent that calls local AC shops to enquire about prices.
Uses Sarvam AI for Hindi STT/TTS and Claude Haiku 3.5 or Qwen3 for LLM.
Includes SanitizedAgent for chat context sanitization, think-tag stripping,
and English→Hindi phonetic normalization for TTS.

Run with:
  python agent_worker.py dev       # local development
  python agent_worker.py start     # production

Or use test_browser.py which auto-manages the agent worker.
"""

import asyncio
import json
import os
import re
import logging
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env.local")

from livekit import api, agents
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RoomInputOptions,
    function_tool,
    RunContext,
    llm,
)
from livekit.plugins import anthropic, openai, silero, sarvam

logger = logging.getLogger("ac-price-caller.agent")


# ---------------------------------------------------------------------------
# Per-call file logger — saves all logs for each call session to logs/ dir
# ---------------------------------------------------------------------------
def _setup_call_logger(store_name: str) -> tuple[logging.FileHandler, str]:
    """Create a per-call log file and attach a file handler to the root logger.

    Returns (handler, log_filepath) so the handler can be removed when the call ends.
    """
    logs_dir = Path(__file__).parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"{store_name.replace(' ', '_')}_{ts}.log"

    handler = logging.FileHandler(str(log_file), encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    ))

    # Attach to root logger so it captures logs from all livekit.* loggers too
    root = logging.getLogger()
    root.addHandler(handler)
    logger.info(f"[LOG] Per-call log file: {log_file}")
    return handler, str(log_file)


# ---------------------------------------------------------------------------
# Custom Agent — sanitizes chat context + normalizes output for TTS
# ---------------------------------------------------------------------------
class SanitizedAgent(Agent):
    """Agent subclass that intercepts every LLM call to:
    1. Sanitize message ordering (vLLM/Qwen requires user-first after system)
    2. Log the exact messages sent to the LLM for debugging
    3. Strip <think>...</think> tags from Qwen3 output before TTS
    4. Normalize English terms to Hindi phonetics for TTS pronunciation
    5. Strip roleplay action markers (*confused*, etc.)
    """

    async def llm_node(self, chat_ctx, tools, model_settings):
        # --- Sanitize chat context ---
        chat_ctx = self._sanitize_chat_ctx(chat_ctx)

        # --- Log what we're sending to the LLM ---
        try:
            messages, _ = chat_ctx.to_provider_format("openai")
            roles = [m.get("role", "?") for m in messages]
            logger.info(f"[LLM REQUEST] roles={roles}, messages={len(messages)}")
            for i, m in enumerate(messages):
                content = m.get("content", "")
                if len(str(content)) > 200:
                    content = str(content)[:200] + "..."
                logger.debug(f"[LLM MSG {i}] role={m.get('role')} content={content}")
        except Exception as e:
            logger.warning(f"[LLM REQUEST] failed to log messages: {e}")

        # --- Forward to default LLM node, cleaning output for TTS ---
        # Per-chunk normalization to keep streaming smooth (no buffering).
        async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
            if isinstance(chunk, str):
                chunk = _strip_think_tags(chunk)
                chunk = _normalize_for_tts(chunk)
                if chunk:
                    yield chunk
            elif hasattr(chunk, "delta") and isinstance(getattr(chunk.delta, "content", None), str):
                chunk.delta.content = _strip_think_tags(chunk.delta.content)
                chunk.delta.content = _normalize_for_tts(chunk.delta.content)
                yield chunk
            else:
                yield chunk

    @staticmethod
    def _sanitize_chat_ctx(chat_ctx: llm.ChatContext) -> llm.ChatContext:
        """Ensure first non-system message is from the user.
        Required by vLLM/Qwen; also prevents stale assistant context with Claude."""
        ctx = chat_ctx.copy()
        items = ctx.items

        # Find first ChatMessage that isn't system
        for i, item in enumerate(items):
            if getattr(item, "type", None) != "message":
                continue
            if item.role == "system":
                continue
            # First non-system message found
            if item.role != "user":
                logger.warning(
                    f"[SANITIZE] First non-system message is role='{item.role}', "
                    f"expected 'user'. Removing it to prevent vLLM 400 error."
                )
                items.pop(i)
            break

        return ctx


# Regex to strip Qwen3 thinking blocks from streamed text (only applies when using Qwen LLM)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think>.*", re.DOTALL)


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from Qwen3 output so TTS doesn't read them."""
    text = _THINK_RE.sub("", text)
    text = _THINK_OPEN_RE.sub("", text)  # handle unclosed tag (streaming)
    return text.strip()


# ---------------------------------------------------------------------------
# TTS text normalization — fix pronunciation of English terms in Hindi TTS
# ---------------------------------------------------------------------------
# Map English abbreviations/words to Hindi phonetic equivalents so Sarvam TTS
# pronounces them correctly instead of reading them as Hindi words.
_TTS_REPLACEMENTS = {
    # Abbreviations
    "AC": "ए सी",
    "A.C.": "ए सी",
    "a.c.": "ए सी",
    "EMI": "ई एम आई",
    "E.M.I.": "ई एम आई",
    "GST": "जी एस टी",
    "MRP": "एम आर पी",
    "LED": "एल ई डी",
    "WiFi": "वाई फाई",
    "Wi-Fi": "वाई फाई",
    "BEE": "बी ई ई",
    # Product terms
    "ton": "टन",
    "star": "स्टार",
    "inverter": "इन्वर्टर",
    "split": "स्प्लिट",
    "window": "विंडो",
    "model": "मॉडल",
    "copper": "कॉपर",
    # Common Hinglish words the LLM uses in Latin script
    "rate": "रेट",
    "price": "प्राइस",
    "best": "बेस्ट",
    "budget": "बजट",
    "online": "ऑनलाइन",
    "exchange": "एक्सचेंज",
    "offer": "ऑफर",
    "free": "फ्री",
    "warranty": "वारंटी",
    "installation": "इंस्टॉलेशन",
    "delivery": "डिलीवरी",
    "discount": "डिस्काउंट",
    "cashback": "कैशबैक",
    "payment": "पेमेंट",
    "service": "सर्विस",
    "stock": "स्टॉक",
    "piping": "पाइपिंग",
    "compressor": "कंप्रेसर",
    "stabilizer": "स्टेबिलाइज़र",
    # Brand names
    "Samsung": "सैमसंग",
    "LG": "एल जी",
    "Daikin": "डायकिन",
    "Voltas": "वोल्टास",
    "Blue Star": "ब्लू स्टार",
    "Haier": "हायर",
    "Panasonic": "पैनासोनिक",
    "Whirlpool": "व्हर्लपूल",
    "Carrier": "कैरियर",
    "Godrej": "गोदरेज",
}

# Split replacements into two groups:
# 1. Abbreviations (ALL CAPS like AC, EMI, GST) — case-sensitive + word boundaries
#    to avoid matching inside words like "Achha" or "Legit"
# 2. Regular words — case-insensitive, no word boundaries (LLM concatenates words)
_ABBREV_REPLACEMENTS = {k: v for k, v in _TTS_REPLACEMENTS.items() if k.isupper() and len(k) <= 4}
_WORD_REPLACEMENTS = {k: v for k, v in _TTS_REPLACEMENTS.items() if k not in _ABBREV_REPLACEMENTS}

_ABBREV_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_ABBREV_REPLACEMENTS, key=len, reverse=True)) + r")\b",
) if _ABBREV_REPLACEMENTS else None

_WORD_REPLACEMENTS_LOWER = {k.lower(): v for k, v in _WORD_REPLACEMENTS.items()}
_WORD_PATTERN = re.compile(
    "(" + "|".join(re.escape(k) for k in sorted(_WORD_REPLACEMENTS_LOWER, key=len, reverse=True)) + ")",
    re.IGNORECASE,
) if _WORD_REPLACEMENTS_LOWER else None

# Regex to strip roleplay/action markers: *confused*, (laughs), [pauses], etc.
_ACTION_RE = re.compile(r"[\*\(\[][a-zA-Z\s]+[\*\)\]]")


# Regex to insert spaces at Devanagari↔Latin/digit script boundaries.
# Fixes concatenated output like "सैमसंग1.5टन" → "सैमसंग 1.5 टन"
_DEVA_TO_LATIN = re.compile(r"([\u0900-\u097F])([A-Za-z0-9])")
_LATIN_TO_DEVA = re.compile(r"([A-Za-z0-9])([\u0900-\u097F])")


def _normalize_for_tts(text: str) -> str:
    """Normalize LLM output for better Hindi TTS pronunciation."""
    # Strip roleplay action markers
    text = _ACTION_RE.sub("", text)
    # Replace abbreviations (case-sensitive, word-boundary protected)
    # Pad with spaces so adjacent replacements don't merge (e.g. "AC EMI" → " ए सी  ई एम आई ")
    if _ABBREV_PATTERN:
        text = _ABBREV_PATTERN.sub(lambda m: f" {_ABBREV_REPLACEMENTS[m.group(0)]} ", text)
    # Replace regular words (case-insensitive, no word boundaries for concatenated text)
    if _WORD_PATTERN:
        text = _WORD_PATTERN.sub(lambda m: f" {_WORD_REPLACEMENTS_LOWER[m.group(0).lower()]} ", text)
    # Insert spaces at Devanagari↔Latin/digit boundaries
    text = _DEVA_TO_LATIN.sub(r"\1 \2", text)
    text = _LATIN_TO_DEVA.sub(r"\1 \2", text)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    return text.strip()

# ---------------------------------------------------------------------------
# Conversation prompt
# ---------------------------------------------------------------------------
DEFAULT_INSTRUCTIONS = """
<role>
You are a middle-class Indian customer CALLING a local AC shop to ask about prices.
You are the CALLER. The person you are talking to is the SHOPKEEPER.
You do NOT know the price. You do NOT know availability. You are calling to FIND OUT.
You do NOT work at the shop. You have NO product information to share.
</role>

<voice_and_tone>
Speak in natural spoken Hindi/Hinglish — the way a real person talks on the phone.
Semi-formal and respectful. Keep responses SHORT — 1 line, max 2. This is a phone call, not a speech.

Address terms: Use "bhai sahab" ONLY ONCE at the start of the call. After that, just use "ji" or no address at all.
Most of your responses should have NO address term — just speak directly, like a normal phone conversation.

Do NOT use: "yaar", "bro", "dekho", "dost", "bhaiya" — these are too casual for a shopkeeper you don't know.
Do NOT overuse fillers. A simple "achha" or "ji" is enough. Do not stuff every sentence with "haan ji", "matlab", etc.
</voice_and_tone>

<conversation_flow>
The call should progress naturally through these phases. Do not rush — let the shopkeeper's responses guide you.

Phase 1 — CONFIRM THE SHOP:
Start by confirming you've called the right place. Ask if they sell ACs.
Example: "Hello, yeh [store name] hai? Aap log AC dealer ho?"

Phase 2 — ASK ABOUT THE PRODUCT:
Once confirmed, ask if they have the specific AC you want.
Example: "Bhai sahab, Samsung 1.5 ton ka 5 star inverter split AC hai aapke paas?"

Phase 3 — ASK THE PRICE:
Once they confirm they have it, ask about the price.
Example: "Achha, uska kya rate chal raha hai?"

Phase 4 — NEGOTIATE:
React to the price and try to get a better deal. Use these tactics naturally:
- Express surprise: "Itna? Thoda zyada lag raha hai"
- Compare: "Main 2-3 jagah se puch raha hoon" / "Online pe toh kam dikha raha tha"
- Ask for best price: "Aapka best price kya hoga?"
- Mention budget: "Budget thoda tight hai, kuch kam hoga kya?"

Phase 5 — ASK ABOUT EXTRAS (only after getting a price):
Pick 1-2 relevant questions, not all of them:
- "Installation free hai ya alag se?"
- "Warranty kitni milegi?"
- "Purana AC exchange pe le lo ge kya?"
- "Stock mein hai? Kitne din mein lag jayega?"

Phase 6 — WRAP UP AND END CALL:
Once you have the price and basic details, wrap up and IMMEDIATELY use the end_call tool.
Say something like "Theek hai ji, main soch ke bataata hoon. Dhanyavaad." and then call end_call.
Do NOT continue talking after saying goodbye. Do NOT ask more questions after wrapping up.
If the shopkeeper says "anything else?" after you've said goodbye, just say "Nahi ji, bas itna hi. Dhanyavaad." and use end_call.
</conversation_flow>

<rules>
- You are ONLY the customer. Never switch to the shopkeeper role.
- You do NOT know prices, specs, or product details. You are calling to ASK.
- If the shopkeeper asks you something you don't know, say so: "Woh toh mujhe nahi pata, aap bata dijiye"
- If the shopkeeper says "shop pe aa jao", respond: "Haan aa jaunga, pehle thoda idea chahiye tha price ka"
- Do not go through a checklist. Follow up based on what the shopkeeper says.
- Do not ask about technical specs (copper vs aluminium, cooling capacity, energy rating details, Wi-Fi features).
- If the shopkeeper mentions technical features, just say "achha" and move on.
- CALL ENDING: Once you have the price and 1-2 extras (installation/warranty/exchange), END THE CALL using the end_call tool. Do not drag it out. Do not keep asking more questions after getting the key info.
- If you've already said "Dhanyavaad" or "soch ke bataata hoon", you MUST use end_call immediately. Do not continue the conversation.
- If the shopkeeper refuses to share the price on the phone, politely end the call with end_call.
</rules>

<output_format>
Your output is fed directly into a text-to-speech engine and played as audio on a phone call.
Only output the exact words you would speak aloud. Nothing else.
Do not output action descriptions (*confused*, *pauses*), stage directions, or narration.
Do not echo or repeat what the shopkeeper just said back to them.
Do not prefix your response with the shopkeeper's words.
ALWAYS put a space between every word. Never concatenate words together.
Write in clean Romanized Hindi with proper spacing — e.g. "aap ka rate kya hai" not "aapkaratekya hai".
</output_format>

<examples>
Example 1 — Opening the call:
Shopkeeper: "Hello?"
You: "Hello, yeh Sharma Electronics hai? Aap log AC dealer ho?"
Shopkeeper: "Haan ji, boliye"
You: "Bhai sahab, Samsung 1.5 ton ka 5 star inverter split AC hai aapke paas?"

Example 2 — Reacting to a price:
Shopkeeper: "Woh 42,000 ka padega"
You: "42 hazaar? Thoda zyada lag raha hai. Online pe toh 38 mein dikha raha tha. Best price kya hoga?"

Example 3 — Wrapping up and ending:
Shopkeeper: "Installation 500 extra, warranty 5 saal"
You: "Theek hai ji, main soch ke bataata hoon. Dhanyavaad."
[Then IMMEDIATELY use end_call tool]

Example 4 — When shopkeeper asks "anything else?" after goodbye:
Shopkeeper: "Aur kuch poochna hai?"
You: "Nahi ji, bas itna hi tha. Dhanyavaad."
[Then IMMEDIATELY use end_call tool]
</examples>
"""


# ---------------------------------------------------------------------------
# Tools the LLM can call
# ---------------------------------------------------------------------------
@function_tool()
async def end_call(context: RunContext) -> str:
    """Call this when the conversation is complete and you have the price information, or if the shopkeeper refuses to give a price on the phone."""
    logger.info("Agent triggered end_call")
    # Graceful hangup
    room = context.session.room
    if room:
        for participant in room.remote_participants.values():
            if participant.kind == "SIP":
                # Disconnect SIP participant
                lk_api = api.LiveKitAPI()
                await lk_api.room.remove_participant(
                    api.RoomParticipantIdentity(
                        room=room.name,
                        identity=participant.identity,
                    )
                )
                await lk_api.aclose()
    return "Call ended. Thank you."


# ---------------------------------------------------------------------------
# LLM provider selection
# ---------------------------------------------------------------------------
def _create_llm():
    """Create the LLM instance based on LLM_PROVIDER env var.

    LLM_PROVIDER=qwen  (default) — Qwen3-4B via self-hosted vLLM
    LLM_PROVIDER=claude          — Claude Haiku 3.5 via Anthropic API
    """
    provider = os.environ.get("LLM_PROVIDER", "qwen").lower()

    if provider == "claude":
        logger.info("[LLM] Using Claude Haiku 3.5 (Anthropic)")
        return anthropic.LLM(
            model="claude-3-5-haiku-20241022",
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            temperature=0.7,
        )
    else:
        logger.info("[LLM] Using Qwen3-4B-Instruct (vLLM)")
        return openai.LLM(
            model=os.environ.get("LLM_MODEL", "Qwen/Qwen3-4B-Instruct-2507-FP8"),
            base_url=os.environ.get("LLM_BASE_URL", "http://192.168.0.42:8000/v1"),
            api_key=os.environ.get("LLM_API_KEY", "unused"),
            temperature=0.7,
        )


# ---------------------------------------------------------------------------
# Agent entrypoint
# ---------------------------------------------------------------------------
async def entrypoint(ctx: JobContext):
    """
    Called by LiveKit when a dispatch is created for this agent.
    Handles the full lifecycle: connect → dial → converse → hangup.
    """
    logger.info(f"Agent entrypoint called. Room: {ctx.room.name}")

    # Parse metadata from the dispatch
    metadata = json.loads(ctx.job.metadata or "{}")
    phone_number = metadata.get("phone", "")
    store_name = metadata.get("store_name", "Unknown Store")
    ac_model = metadata.get("ac_model", "Samsung 1.5 Ton Split AC")
    sip_trunk_id = metadata.get("sip_trunk_id", os.environ.get("SIP_OUTBOUND_TRUNK_ID", ""))

    is_browser = not phone_number
    logger.info(f"{'Browser session' if is_browser else f'Calling {store_name} at {phone_number}'} for {ac_model}")

    # Set up per-call log file (captures all agent, LLM, and session logs for this call)
    call_log_handler, call_log_path = _setup_call_logger(store_name)

    # Connect agent to the room
    await ctx.connect()

    # Build custom instructions with the specific AC model and store name
    instructions = DEFAULT_INSTRUCTIONS + f"""
<session_context>
Product you want to buy: {ac_model}
Shop you are calling: {store_name}
</session_context>
"""

    # Create the agent session with Sarvam STT/TTS + switchable LLM (Claude or Qwen)
    session = AgentSession(
        # Voice Activity Detection — detect when someone is speaking
        vad=silero.VAD.load(),
        # Speech-to-Text — Sarvam saaras:v3 for Hindi/Hinglish
        stt=sarvam.STT(
            language="hi-IN",
            model="saaras:v3",
            api_key=os.environ.get("SARVAM_API_KEY"),
            sample_rate=16000,
        ),
        # LLM — switchable via LLM_PROVIDER env var (qwen or claude)
        llm=_create_llm(),
        # Text-to-Speech — Sarvam bulbul:v3 for natural Hindi voice
        tts=sarvam.TTS(
            model="bulbul:v3",
            target_language_code="hi-IN",
            speaker="aditya",  # v3 male voice; others: rahul, rohan, amit, dev, varun, ratan; female: ritu, priya, neha, pooja, simran
            api_key=os.environ.get("SARVAM_API_KEY"),
            pace=1.0,
            pitch=0,
            loudness=1.5,
            speech_sample_rate=24000 if is_browser else 8000,  # Higher quality for browser
        ),
    )

    # Start the agent
    await session.start(
        room=ctx.room,
        agent=SanitizedAgent(instructions=instructions),
        room_input_options=RoomInputOptions(
            # Audio-only — no text or video input
            text_enabled=False,
            video_enabled=False,
        ),
    )

    # ---- Transcript collection & conversation logging ----
    transcript_lines = []  # Collect messages for saving to file

    @session.on("user_input_transcribed")
    def on_user_transcript(ev):
        if ev.is_final:
            logger.info(f"[USER] {ev.transcript}")
            transcript_lines.append({"role": "user", "text": ev.transcript, "time": datetime.now().isoformat()})

    @session.on("conversation_item_added")
    def on_conversation_item(ev):
        item = ev.item
        if item.role == "assistant":
            text = "".join(str(c) for c in item.content)
            logger.info(f"[LLM] {text}")
            transcript_lines.append({"role": "assistant", "text": text, "time": datetime.now().isoformat()})

    @session.on("function_tools_executed")
    def on_tools_executed(ev):
        for fc in ev.function_calls:
            logger.info(f"[TOOL CALL] {fc.name}({fc.arguments})")
        for out in ev.function_call_outputs:
            if out:
                logger.info(f"[TOOL RESULT] {out.name} → {out.output}")

    @session.on("metrics_collected")
    def on_metrics(ev):
        m = ev.metrics
        # Only log LLM metrics (has prompt_tokens attribute)
        if hasattr(m, "prompt_tokens"):
            logger.info(
                f"[LLM METRICS] tokens: {m.prompt_tokens}→{m.completion_tokens}, "
                f"TTFT: {m.ttft:.2f}s, duration: {m.duration:.2f}s"
            )

    @session.on("error")
    def on_error(ev):
        logger.error(f"[SESSION ERROR] source={type(ev.source).__name__}, error={ev.error}")

    # ---- Save transcript when a participant disconnects ----
    def _save_transcript():
        if not transcript_lines:
            return
        transcript_dir = Path(__file__).parent / "transcripts"
        transcript_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = transcript_dir / f"{store_name.replace(' ', '_')}_{ts}.json"
        data = {
            "store_name": store_name,
            "ac_model": ac_model,
            "room": ctx.room.name,
            "phone": phone_number or "browser",
            "timestamp": datetime.now().isoformat(),
            "messages": transcript_lines,
        }
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"[TRANSCRIPT] Saved to {filename}")
        except Exception as e:
            logger.error(f"[TRANSCRIPT] Failed to save: {e}")

    @ctx.room.on("participant_disconnected")
    def on_participant_left(participant):
        logger.info(f"Participant {participant.identity} left — saving transcript and closing call log")
        _save_transcript()
        # Close per-call log handler so the file is flushed and released
        logging.getLogger().removeHandler(call_log_handler)
        call_log_handler.close()
        logger.info(f"[LOG] Call log saved to {call_log_path}")

    # Now dial the store (or wait for browser participant)
    if phone_number and sip_trunk_id:
        logger.info(f"Dialing {phone_number} via SIP trunk {sip_trunk_id}")
        try:
            await ctx.api.sip.create_sip_participant(
                api.CreateSIPParticipantRequest(
                    sip_trunk_id=sip_trunk_id,
                    sip_call_to=phone_number,
                    room_name=ctx.room.name,
                    participant_identity=f"phone-{phone_number}",
                    participant_name=store_name,
                )
            )
            logger.info(f"SIP call initiated to {phone_number}")
        except Exception as e:
            logger.error(f"Failed to initiate SIP call: {e}")
            return
    elif not phone_number:
        # Browser session — wait for browser participant, then greet.
        # Greeting bypasses LLM (direct TTS) and stays out of chat context.
        logger.info("Browser session — waiting for browser participant to join")
        await ctx.wait_for_participant()
        logger.info("Browser participant joined — sending greeting")
        session.say("Hello, yeh Browser Test hai? Aap log ए सी dealer ho?", add_to_chat_ctx=False)

    if not is_browser:
        # Set a maximum call duration timer (SIP calls only)
        async def call_timeout():
            await asyncio.sleep(120)  # 2 minutes max
            logger.info("Call timeout reached, ending call")
            for participant in ctx.room.remote_participants.values():
                try:
                    await ctx.api.room.remove_participant(
                        api.RoomParticipantIdentity(
                            room=ctx.room.name,
                            identity=participant.identity,
                        )
                    )
                except Exception:
                    pass

        asyncio.create_task(call_timeout())


# ---------------------------------------------------------------------------
# Run the agent worker
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="ac-price-agent",  # Must match dispatch requests
        )
    )
