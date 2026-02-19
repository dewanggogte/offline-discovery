"""
pipeline/config.py — Centralized pipeline configuration
========================================================
Single source of truth for all pipeline constants and tuning parameters.
Avoids scattering magic numbers across modules.
"""

import os


class PipelineConfig:
    """Central configuration for all pipeline stages."""

    # --- LLM ---
    CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001")

    # --- Research ---
    RESEARCH_MAX_ROUNDS = 4
    RESEARCH_MAX_SEARCHES = 3
    RESEARCH_TEMPERATURE = 0.3

    # --- Store Discovery ---
    STORE_MAX_RESULTS = 15
    STORE_RANK_TOP_N = 4
    STORE_RANK_WEIGHTS = {
        "rating": 0.30,
        "reviews": 0.20,
        "phone": 0.30,
        "maps_source": 0.10,
        "relevance": 0.10,
    }

    # --- Prompt Builder ---
    MAX_QUESTIONS = 10
    MAX_TOPICS = 10
    MAX_COMPETING_PRODUCTS = 5
    MAX_IMPORTANT_NOTES = 6
    MAX_RECOMMENDED_PRODUCTS = 3
    MAX_INSIDER_KNOWLEDGE = 3

    # --- Intake ---
    INTAKE_MAX_TOKENS = 1024

    # --- Voice / TTS ---
    DEFAULT_SPEAKER = "shubh"
    DEFAULT_PACE = 1.0
    TTS_MODEL = "bulbul:v3"
    STT_MODEL = "saaras:v3"

    # --- VAD / Endpointing ---
    MIN_SPEECH_DURATION = 0.15
    MIN_SILENCE_DURATION = 0.5
    ACTIVATION_THRESHOLD = 0.5
    MIN_ENDPOINTING_DELAY = 0.3
    MIN_INTERRUPTION_DURATION = 0.8
    MIN_INTERRUPTION_WORDS = 2
    FALSE_INTERRUPTION_TIMEOUT = 2.0

    # --- Session ---
    SESSION_MAX_AGE_SECONDS = 3600

    # --- Chain stores (used in prompt_builder for store-aware approach) ---
    CHAIN_STORES = {"croma", "reliance digital", "vijay sales", "poorvika",
                    "pai international", "bajaj electronics", "lot mobiles",
                    "sangeetha mobiles", "girias"}
