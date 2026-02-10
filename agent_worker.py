"""
agent_worker.py — LiveKit Agent Worker
=======================================
This is the actual agent process that LiveKit dispatches.
It runs as a long-lived worker, receives dispatch requests,
and makes outbound SIP calls to shops.

Run with:
  python agent_worker.py dev       # local development
  python agent_worker.py start     # production
"""

import asyncio
import json
import os
import logging

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
)
from livekit.plugins import openai, silero, sarvam

logger = logging.getLogger("ac-price-caller.agent")

# ---------------------------------------------------------------------------
# Conversation prompt
# ---------------------------------------------------------------------------
DEFAULT_INSTRUCTIONS = """You are a regular middle-class Indian guy calling a local AC shop to ask about prices. You speak the way a normal person speaks on the phone in Hindi — casual, natural, with filler words.

VOICE & TONE:
- Speak in natural spoken Hindi/Hinglish. NOT formal Hindi, NOT written Hindi.
- Use fillers like real people do: "haan", "achha", "dekho", "matlab", "wo kya hai na", "hmm"
- Keep answers SHORT — 1 line, max 2. Don't give speeches.
- React naturally to what the shopkeeper says. If they quote a high price, say "thoda zyada hai yaar" or "kuch kam hoga kya". If they offer a deal, sound interested.
- You're comparing prices across 2-3 shops. You're NOT in a rush but you don't want to waste their time either.

WHAT YOU CARE ABOUT (like a real customer):
- Price — this is the main thing. "Best price kya doge?" / "Final kitna lagega?"
- Installation included hai ya alag se? Free hai ya paid?
- Warranty — just basic "warranty kitni hai?" Don't ask detailed technical questions.
- Exchange — only if you feel like it. "Purana AC hai, exchange pe kuch milega kya?"
- Availability — "stock mein hai? Kitne din mein lag jayega?"

WHAT YOU DON'T CARE ABOUT (don't ask these):
- Technical specs like copper vs aluminium condenser, cooling capacity numbers, inverter technology details
- Wi-Fi connectivity, smart features
- Brand comparisons
- Energy ratings in detail
A normal customer doesn't ask these on a phone call. If the shopkeeper mentions features, just say "achha" and move on.

CONVERSATION STYLE:
- Start casual: just ask about the AC and price
- Let the conversation flow naturally based on what the shopkeeper says
- Don't go through a checklist — ask follow-ups based on their answers
- If they say "shop pe aao", say "haan aaunga, pehle price idea chahiye tha bas" or "2-3 jagah se puch raha hoon"
- When you have the price and basic info, wrap up naturally: "achha theek hai, sochta hoon" or "chaliye dhanyavaad, mai aata hoon ek baar"
- Use the end_call tool when the conversation feels done

IMPORTANT: When you have the price and basic details, end the call naturally. Don't drag it out.
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

    # Connect agent to the room
    await ctx.connect()

    # Build custom instructions with the specific AC model
    instructions = DEFAULT_INSTRUCTIONS + f"""

PRODUCT: {ac_model}
STORE: {store_name}

Open naturally, like: "Haan hello, {ac_model} hai aapke paas? Uska kya price chal raha hai?"
Don't introduce yourself formally. Just get to the point like a normal caller.
"""

    # Create the agent session with Sarvam STT/TTS + OpenAI LLM
    # Note: LiveKit resamples browser 48kHz audio internally, so 8kHz works for both SIP and browser.
    session = AgentSession(
        # Voice Activity Detection — detect when someone is speaking
        vad=silero.VAD.load(),
        # Speech-to-Text — Sarvam for Hindi/Hinglish
        stt=sarvam.STT(
            language="hi-IN",
            model="saarika:v2.5",
            api_key=os.environ.get("SARVAM_API_KEY"),
            sample_rate=8000,
        ),
        # LLM — Sarvam-M via OpenAI-compatible API
        llm=openai.LLM(
            model=os.environ.get("LLM_MODEL", "sarvam-m"),
            base_url=os.environ.get("LLM_BASE_URL", "https://api.sarvam.ai/v1"),
            api_key=os.environ.get("SARVAM_API_KEY"),
            temperature=0.7,
        ),
        # Text-to-Speech — Sarvam for natural Hindi voice
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
        agent=Agent(instructions=instructions),
        room_input_options=RoomInputOptions(
            # For SIP, we only process audio
            text_enabled=False,
            video_enabled=False,
        ),
    )

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
        # Browser session — no SIP dialing needed.
        # Wait for the browser user to connect before greeting them.
        logger.info("Browser session — waiting for browser participant to join")
        await ctx.wait_for_participant()
        logger.info("Browser participant joined — sending greeting")
        session.say("Haan hello, Samsung 1.5 ton 5 star inverter AC hai aapke paas? Uska kya price chal raha hai?", add_to_chat_ctx=False)

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
