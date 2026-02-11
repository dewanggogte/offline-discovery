"""Live Sarvam TTS integration tests — require --live flag and valid SARVAM_API_KEY."""

import base64
import os
import pytest
import requests

from tests.conftest import _normalize_for_tts

SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "")
SARVAM_BASE_URL = "https://api.sarvam.ai"


def _tts_request(text, sample_rate=24000):
    payload = {
        "inputs": text,
        "target_language_code": "hi-IN",
        "speaker": "aditya",
        "model": "bulbul:v3",
        "speech_sample_rate": sample_rate,
    }
    headers = {"API-Subscription-Key": SARVAM_API_KEY, "Content-Type": "application/json"}
    return requests.post(f"{SARVAM_BASE_URL}/text-to-speech", json=payload, headers=headers)


@pytest.mark.live
class TestSarvamTTSLive:

    def test_tts_returns_audio(self):
        resp = _tts_request("Namaste, ए सी ka रेट kya hai?")
        assert resp.status_code == 200
        data = resp.json()
        assert "audios" in data
        audio_bytes = base64.b64decode(data["audios"][0])
        assert len(audio_bytes) > 1000  # Non-trivial audio

    def test_tts_with_normalized_text(self):
        """Test TTS with text that has been through _normalize_for_tts."""
        raw = "Samsung 1.5 ton ka AC ka price batao"
        normalized = _normalize_for_tts(raw)
        resp = _tts_request(normalized)
        assert resp.status_code == 200

    def test_tts_8khz_telephony(self):
        """Test TTS at 8kHz sample rate (used for SIP calls)."""
        resp = _tts_request("Hello, yeh Sharma Electronics hai?", sample_rate=8000)
        assert resp.status_code == 200

    def test_tts_empty_input_no_crash(self):
        """Empty input should not cause a 500 server error."""
        resp = _tts_request("")
        assert resp.status_code != 500
