"""
test_sarvam.py — Test Sarvam AI APIs independently
====================================================
Run this FIRST to verify your Sarvam API key works and test
STT/TTS quality before integrating with LiveKit telephony.

Usage:
  python test_sarvam.py              # Run all tests
  python test_sarvam.py --tts-only   # Test only TTS
  python test_sarvam.py --stt-only   # Test only STT
"""

import asyncio
import base64
import os
import sys
import json
import argparse

import requests
from dotenv import load_dotenv

load_dotenv(".env.local")

SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "")
SARVAM_BASE_URL = "https://api.sarvam.ai"


def test_tts():
    """Test Sarvam Text-to-Speech with a sample Hindi greeting."""
    print("\n🔊 Testing Sarvam TTS (Text-to-Speech)...")
    print("-" * 50)

    # This is what your agent would say when calling a shop
    test_texts = [
        {
            "text": "Namaste, main ek AC ke baare mein poochna chahta tha. Samsung 1.5 ton split AC ka best price kya hoga?",
            "lang": "hi-IN",
            "desc": "Opening greeting (Hindi)"
        },
        {
            "text": "Exchange offer hai kya? Purana AC dene pe kitna discount milega?",
            "lang": "hi-IN",
            "desc": "Exchange offer question"
        },
        {
            "text": "Installation bhi price mein included hai? Aur warranty kitni milegi?",
            "lang": "hi-IN",
            "desc": "Installation & warranty question"
        },
    ]

    for i, item in enumerate(test_texts):
        print(f"\n  Test {i+1}: {item['desc']}")
        print(f"  Text: {item['text']}")

        payload = {
            "inputs": item["text"],
            "target_language_code": item["lang"],
            "speaker": "meera",  # Female voice
            "pitch": 0,
            "pace": 1.0,
            "loudness": 1.5,
            "speech_sample_rate": 8000,  # Telephony quality
            "enable_preprocessing": True,
            "model": "bulbul:v2",
        }

        headers = {
            "API-Subscription-Key": SARVAM_API_KEY,
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                f"{SARVAM_BASE_URL}/text-to-speech",
                json=payload,
                headers=headers,
            )

            if response.status_code == 200:
                data = response.json()
                # Sarvam returns base64 audio
                if "audios" in data and data["audios"]:
                    audio_b64 = data["audios"][0]
                    audio_bytes = base64.b64decode(audio_b64)
                    filename = f"test_tts_{i+1}.wav"
                    with open(filename, "wb") as f:
                        f.write(audio_bytes)
                    print(f"  ✅ Success! Audio saved to {filename} ({len(audio_bytes)} bytes)")
                else:
                    print(f"  ✅ Response OK but unexpected format: {list(data.keys())}")
            else:
                print(f"  ❌ Error {response.status_code}: {response.text[:200]}")

        except Exception as e:
            print(f"  ❌ Exception: {e}")


def test_stt():
    """Test Sarvam Speech-to-Text with a sample audio file."""
    print("\n🎤 Testing Sarvam STT (Speech-to-Text)...")
    print("-" * 50)

    # Check if we have any test audio files (from TTS test or user-provided)
    test_files = [f for f in os.listdir(".") if f.startswith("test_tts_") and f.endswith(".wav")]

    if not test_files:
        print("  ⚠️  No test audio files found. Run TTS test first to generate them.")
        print("  Or place a .wav file named 'test_audio.wav' in this directory.")

        # Check for user-provided test audio
        if os.path.exists("test_audio.wav"):
            test_files = ["test_audio.wav"]
        else:
            return

    for filename in test_files[:2]:  # Test first 2 files
        print(f"\n  Testing with: {filename}")

        headers = {
            "API-Subscription-Key": SARVAM_API_KEY,
        }

        try:
            with open(filename, "rb") as f:
                response = requests.post(
                    f"{SARVAM_BASE_URL}/speech-to-text",
                    headers=headers,
                    files={"file": (filename, f, "audio/wav")},
                    data={
                        "language_code": "hi-IN",
                        "model": "saarika:v2.5",
                    },
                )

            if response.status_code == 200:
                data = response.json()
                transcript = data.get("transcript", "")
                language = data.get("language_code", "unknown")
                print(f"  ✅ Transcript: {transcript}")
                print(f"  Language detected: {language}")
            else:
                print(f"  ❌ Error {response.status_code}: {response.text[:200]}")

        except Exception as e:
            print(f"  ❌ Exception: {e}")


def test_llm():
    """Test Sarvam's chat completion for price extraction."""
    print("\n🧠 Testing Sarvam Chat Completion...")
    print("-" * 50)

    # Simulate extracting price data from a transcript
    test_transcript = """
    Customer: Namaste, Samsung 1.5 ton 5 star inverter AC ka price kya hai? AR18CYNZABE model?
    Shopkeeper: Haan ji, 5 star inverter model hai na? Uska MRP toh 58,990 hai, but hum aapko 52,500 mein de sakte hain.
    Customer: Exchange offer hai kya?
    Shopkeeper: Haan, purana AC doge toh 4000 aur kam ho jayega. Matlab 48,500 mein pad jayega.
    Customer: Installation bhi included hai?
    Shopkeeper: Haan ji, free installation hai, copper piping 3 feet tak free. Usse zyada lagi toh extra charge lagega.
    Customer: Warranty kitni milegi?
    Shopkeeper: 1 saal comprehensive warranty hai company ki, aur compressor pe 10 saal. Extended warranty bhi karwa sakte hain 1499 mein.
    Customer: Features kya hain isme?
    Shopkeeper: Ye digital inverter compressor hai, copper condenser hai, Wi-Fi bhi hai SmartThings app se control kar sakte ho. Cooling capacity bhi bahut acchi hai 5 star rating ke saath.
    Customer: Koi freebies ya special deals hain?
    Shopkeeper: Free stabilizer mil raha hai saath mein, aur HDFC card pe 2000 ka cashback bhi hai. EMI bhi available hai no cost wala 12 mahine ka.
    Customer: Stock mein hai?
    Shopkeeper: Haan ji, abhi available hai. Kal tak delivery ho jayegi.
    """

    headers = {
        "Authorization": f"Bearer {SARVAM_API_KEY}",
        "Content-Type": "application/json",
    }

    extraction_prompt = f"""Extract price information from this transcript as JSON:

{test_transcript}

Return ONLY valid JSON:
{{"quoted_price": <number>, "mrp": <number>, "exchange_offer": "<desc>", "installation_included": <bool>, "warranty_info": "<desc>", "features": "<desc>", "freebies": "<desc>", "availability": "<desc>"}}"""

    payload = {
        "messages": [
            {"role": "user", "content": extraction_prompt}
        ],
        "max_tokens": 300,
        "temperature": 0.1,
    }

    try:
        response = requests.post(
            f"{SARVAM_BASE_URL}/v1/chat/completions",
            headers=headers,
            json=payload,
        )

        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            print(f"  ✅ Extraction result:\n{content}")
        else:
            print(f"  ⚠️  Sarvam LLM returned {response.status_code}")
            print(f"  Note: For production, use OpenAI GPT-4o-mini for extraction")
            print(f"  Sarvam LLM is best for translation, not structured extraction")

    except Exception as e:
        print(f"  ⚠️  Exception: {e}")
        print(f"  Note: Use OpenAI for extraction in production")


def test_credit_balance():
    """Check remaining Sarvam credits."""
    print("\n💳 Checking Sarvam Credit Balance...")
    print("-" * 50)
    print("  ℹ️  Check your balance at: https://dashboard.sarvam.ai/")
    print("  Free tier: ₹1,000 credits on signup")
    print("  STT: ~₹0.35/hour | TTS: ~₹0.18/10K chars | LLM: per-token")


def main():
    parser = argparse.ArgumentParser(description="Test Sarvam AI APIs")
    parser.add_argument("--tts-only", action="store_true", help="Test only TTS")
    parser.add_argument("--stt-only", action="store_true", help="Test only STT")
    parser.add_argument("--llm-only", action="store_true", help="Test only LLM extraction")
    args = parser.parse_args()

    if not SARVAM_API_KEY:
        print("❌ SARVAM_API_KEY not set!")
        print("   1. Sign up at https://dashboard.sarvam.ai/")
        print("   2. Copy your API key")
        print("   3. Add to .env.local: SARVAM_API_KEY=your_key_here")
        sys.exit(1)

    print("=" * 50)
    print("  Sarvam AI API Test Suite")
    print("=" * 50)
    print(f"  API Key: {SARVAM_API_KEY[:8]}...{SARVAM_API_KEY[-4:]}")

    if args.tts_only:
        test_tts()
    elif args.stt_only:
        test_stt()
    elif args.llm_only:
        test_llm()
    else:
        test_tts()
        test_stt()
        test_llm()
        test_credit_balance()

    print("\n✅ Tests complete!")


if __name__ == "__main__":
    main()
