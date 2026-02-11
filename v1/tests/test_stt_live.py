"""Live Sarvam STT integration tests — require --live flag and valid SARVAM_API_KEY."""

import base64
import os
import pytest
import requests

SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "")
SARVAM_BASE_URL = "https://api.sarvam.ai"


def _generate_test_audio():
    """Generate a short Hindi audio clip via TTS to use as STT input."""
    payload = {
        "inputs": "Namaste, Samsung AC ka price kya hai?",
        "target_language_code": "hi-IN",
        "speaker": "aditya",
        "model": "bulbul:v3",
        "speech_sample_rate": 24000,
    }
    headers = {"API-Subscription-Key": SARVAM_API_KEY, "Content-Type": "application/json"}
    resp = requests.post(f"{SARVAM_BASE_URL}/text-to-speech", json=payload, headers=headers)
    assert resp.status_code == 200, f"TTS failed: {resp.status_code} {resp.text[:200]}"
    return base64.b64decode(resp.json()["audios"][0])


@pytest.mark.live
class TestSarvamSTTLive:

    def test_stt_returns_transcript(self, tmp_path):
        audio_bytes = _generate_test_audio()
        audio_file = tmp_path / "test_stt.wav"
        audio_file.write_bytes(audio_bytes)

        headers = {"API-Subscription-Key": SARVAM_API_KEY}
        with open(audio_file, "rb") as f:
            resp = requests.post(
                f"{SARVAM_BASE_URL}/speech-to-text",
                headers=headers,
                files={"file": ("test.wav", f, "audio/wav")},
                data={"language_code": "hi-IN", "model": "saaras:v3"},
            )

        assert resp.status_code == 200, f"STT failed: {resp.status_code} {resp.text[:200]}"
        data = resp.json()
        assert "transcript" in data
        assert len(data["transcript"]) > 0

    def test_stt_language_detection(self, tmp_path):
        audio_bytes = _generate_test_audio()
        audio_file = tmp_path / "test_stt_lang.wav"
        audio_file.write_bytes(audio_bytes)

        headers = {"API-Subscription-Key": SARVAM_API_KEY}
        with open(audio_file, "rb") as f:
            resp = requests.post(
                f"{SARVAM_BASE_URL}/speech-to-text",
                headers=headers,
                files={"file": ("test.wav", f, "audio/wav")},
                data={"language_code": "hi-IN", "model": "saaras:v3"},
            )

        assert resp.status_code == 200
        data = resp.json()
        lang = data.get("language_code", "")
        assert "hi" in lang.lower() or lang == "", f"Unexpected language: {lang}"
