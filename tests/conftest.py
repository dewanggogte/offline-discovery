"""Shared fixtures and configuration for the test suite."""

import os
import re
import sys
import pytest

# Set dummy env vars so agent_worker.py can be imported without .env.local
os.environ.setdefault("SARVAM_API_KEY", "test-dummy-key")
os.environ.setdefault("LIVEKIT_URL", "ws://localhost:7880")
os.environ.setdefault("LIVEKIT_API_KEY", "devkey")
os.environ.setdefault("LIVEKIT_API_SECRET", "devsecret")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-dummy-key")

# Ensure project root is on sys.path
sys.path.insert(0, str(os.path.dirname(os.path.dirname(__file__))))

from agent_worker import (
    _normalize_for_tts,
    _strip_think_tags,
    _ACTION_RE,
    _replace_numbers,
    _number_to_hindi,
    _transliterate_devanagari,
    _HINDI_ONES,
    SanitizedAgent,
    _create_llm,
    _setup_call_logger,
    DEFAULT_INSTRUCTIONS,
    CLAUDE_MODEL,
)
from livekit.agents.llm import ChatContext


# ---------------------------------------------------------------------------
# CLI option: --live
# ---------------------------------------------------------------------------
def pytest_addoption(parser):
    parser.addoption("--live", action="store_true", default=False, help="Run live API tests")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--live"):
        skip_live = pytest.mark.skip(reason="need --live option to run")
        for item in items:
            if "live" in item.keywords:
                item.add_marker(skip_live)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def normalize():
    """Return the _normalize_for_tts function."""
    return _normalize_for_tts


@pytest.fixture
def strip_think():
    """Return the _strip_think_tags function."""
    return _strip_think_tags


@pytest.fixture
def action_re():
    """Return the _ACTION_RE compiled regex."""
    return _ACTION_RE


@pytest.fixture
def make_chat_ctx():
    """Factory to create ChatContext with a specified role sequence."""
    def _make(roles_and_texts: list[tuple[str, str]]) -> ChatContext:
        ctx = ChatContext()
        for role, text in roles_and_texts:
            ctx.add_message(role=role, content=text)
        return ctx
    return _make


@pytest.fixture
def sample_transcript_data():
    """Return a dict matching the transcript JSON schema."""
    return {
        "store_name": "Test Store",
        "ac_model": "Samsung 1.5 Ton 5 Star Inverter Split AC",
        "room": "test-room-abc123",
        "phone": "+919876543210",
        "timestamp": "2026-02-11T17:30:23.924321",
        "messages": [
            {"role": "user", "text": "Hello", "time": "2026-02-11T17:27:41.875130"},
            {"role": "assistant", "text": "Namaste", "time": "2026-02-11T17:27:52.494117"},
        ],
    }


# ---------------------------------------------------------------------------
# Constraint checking & conversation scoring
# ---------------------------------------------------------------------------
class ConstraintChecker:
    """Validates agent responses against behavioral rules from DEFAULT_INSTRUCTIONS."""

    _DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')
    _QUESTION_RE = re.compile(r'\?')
    _ACTION_MARKER_RE = re.compile(r'[\*\(\[][a-zA-Z\s]+[\*\)\]]')
    _ENGLISH_PAREN_RE = re.compile(r'\([A-Z][a-z].*?\)')
    _END_CALL_TEXT_RE = re.compile(r'\[end_call\]', re.IGNORECASE)

    def check_no_devanagari(self, text: str) -> tuple[bool, str]:
        m = self._DEVANAGARI_RE.search(text)
        if m:
            return False, f"Devanagari found: '{m.group()}' in '{text[:80]}'"
        return True, ""

    def check_single_question(self, text: str) -> tuple[bool, str]:
        questions = self._QUESTION_RE.findall(text)
        if len(questions) > 2:
            return False, f"Stacked {len(questions)} questions: '{text[:100]}'"
        return True, ""

    def check_response_length(self, text: str, max_chars: int = 300) -> tuple[bool, str]:
        if len(text) > max_chars:
            return False, f"Response too long ({len(text)} chars): '{text[:80]}...'"
        return True, ""

    def check_no_action_markers(self, text: str) -> tuple[bool, str]:
        m = self._ACTION_MARKER_RE.search(text)
        if m:
            return False, f"Action marker: '{m.group()}'"
        return True, ""

    def check_no_newlines(self, text: str) -> tuple[bool, str]:
        if '\n' in text:
            return False, f"Newline found in: '{text[:80]}'"
        return True, ""

    def check_no_english_translations(self, text: str) -> tuple[bool, str]:
        m = self._ENGLISH_PAREN_RE.search(text)
        if m:
            return False, f"English translation: '{m.group()}'"
        return True, ""

    def check_no_end_call_text(self, text: str) -> tuple[bool, str]:
        m = self._END_CALL_TEXT_RE.search(text)
        if m:
            return False, f"end_call as text: '{m.group()}'"
        return True, ""

    def check_no_invented_details(self, text: str) -> tuple[bool, str]:
        patterns = [
            (r'\b(Voltas|LG|Daikin)\b.*\b(purana|old)\b', "invented old AC brand"),
            (r'\b\d+\s*(saal|year).*\bpurana\b', "invented specific age"),
            (r'\b(Andheri|Borivali|Malad|Bandra|Juhu)\b', "invented specific neighborhood"),
        ]
        for pat, desc in patterns:
            if re.search(pat, text, re.IGNORECASE):
                return False, f"{desc}: '{text[:100]}'"
        return True, ""

    def check_all(self, text: str) -> dict:
        checks = {
            'no_devanagari': self.check_no_devanagari(text),
            'single_question': self.check_single_question(text),
            'response_length': self.check_response_length(text),
            'no_action_markers': self.check_no_action_markers(text),
            'no_newlines': self.check_no_newlines(text),
            'no_english_translations': self.check_no_english_translations(text),
            'no_end_call_text': self.check_no_end_call_text(text),
            'no_invented_details': self.check_no_invented_details(text),
        }
        passed = all(v[0] for v in checks.values())
        failures = {k: v[1] for k, v in checks.items() if not v[0]}
        return {
            'passed': passed,
            'score': sum(1 for v in checks.values() if v[0]) / len(checks),
            'checks': {k: v[0] for k, v in checks.items()},
            'failures': failures,
            'text': text,
        }


class ConversationScorer:
    """Score a full conversation on multiple quality dimensions."""

    TOPIC_KEYWORDS = {
        'price': [r'rate', r'price', r'kitna', r'kitne', r'hazaar', r'rupay', r'₹', r'\d{4,}'],
        'warranty': [r'warranty', r'guarantee', r'saal ki'],
        'installation': [r'install', r'lagwa', r'fitting', r'pipe'],
        'delivery': [r'deliver', r'bhej', r'ghar pe', r'din mein', r'kab tak'],
        'exchange': [r'exchange', r'puran[ae]', r'old'],
    }

    def __init__(self, checker: ConstraintChecker):
        self.checker = checker

    def score_conversation(self, messages: list[dict]) -> dict:
        assistant_msgs = [m for m in messages if m.get('role') == 'assistant']
        if not assistant_msgs:
            return {'overall_score': 0.0, 'per_turn': [], 'topics_covered': set(),
                    'flags': ['no_assistant_messages']}

        per_turn = [self.checker.check_all(m['text']) for m in assistant_msgs]
        constraint_score = sum(r['score'] for r in per_turn) / len(per_turn)

        topics = self.detect_topics(messages)
        topic_score = min(len(topics) / 3.0, 1.0)

        price_echo_score = self.check_price_echo(messages)
        brevity_score = self._brevity_score(assistant_msgs)
        repetition_score = self._no_repetition_score(assistant_msgs)

        overall = (
            constraint_score * 0.40 +
            topic_score * 0.25 +
            price_echo_score * 0.15 +
            brevity_score * 0.10 +
            repetition_score * 0.10
        )

        return {
            'overall_score': round(overall, 3),
            'constraint_score': round(constraint_score, 3),
            'topic_score': round(topic_score, 3),
            'price_echo_score': round(price_echo_score, 3),
            'brevity_score': round(brevity_score, 3),
            'repetition_score': round(repetition_score, 3),
            'topics_covered': topics,
            'per_turn': per_turn,
            'turn_count': len(assistant_msgs),
        }

    def detect_topics(self, messages: list[dict]) -> set:
        all_text = ' '.join(m.get('text', '') for m in messages)
        topics = set()
        for topic, patterns in self.TOPIC_KEYWORDS.items():
            for pat in patterns:
                if re.search(pat, all_text, re.IGNORECASE):
                    topics.add(topic)
                    break
        return topics

    # Reverse mapping: Hindi word → number (e.g., "adtees" → 38)
    _HINDI_TO_NUM = {v: k for k, v in _HINDI_ONES.items() if v}

    def check_price_echo(self, messages: list[dict]) -> float:
        for i, msg in enumerate(messages):
            if msg.get('role') != 'user':
                continue
            text = msg.get('text', '')
            price_num = None

            # Try digit-based price: "38000 ka hai"
            price_match = re.search(r'(\d[\d,]*)\s*(?:ka|mein|rupay|₹|hai)', text)
            if price_match:
                price_str = price_match.group(1).replace(',', '')
                try:
                    price_num = int(price_str)
                except ValueError:
                    pass
                if price_num and price_num < 1000:
                    price_num = None

            # Try Hindi word price: "adtees hazaar ka hai"
            if price_num is None:
                hindi_match = re.search(r'(\w+)\s+hazaar', text.lower())
                if hindi_match:
                    word = hindi_match.group(1)
                    num = self._HINDI_TO_NUM.get(word)
                    if num:
                        price_num = num * 1000

            if price_num is None:
                continue

            price_str = str(price_num)
            hindi_form = _number_to_hindi(price_num) if price_num < 100000 else price_str
            # Also get just the thousands word for partial matching
            thousands_word = _HINDI_ONES.get(price_num // 1000, '') if price_num >= 1000 else ''

            for j in range(i + 1, min(i + 3, len(messages))):
                if messages[j].get('role') == 'assistant':
                    resp = messages[j].get('text', '').lower()
                    if (price_str in resp or
                            (hindi_form and hindi_form.lower() in resp) or
                            (thousands_word and thousands_word in resp)):
                        return 1.0
            return 0.0
        return 0.5  # no price given yet

    def check_call_readiness(self, messages: list[dict]) -> bool:
        topics = self.detect_topics(messages)
        return 'price' in topics and len(topics) >= 3

    def _brevity_score(self, assistant_msgs: list[dict]) -> float:
        lengths = [len(m.get('text', '')) for m in assistant_msgs]
        avg = sum(lengths) / len(lengths)
        if avg < 100:
            return 1.0
        if avg < 200:
            return 0.7
        if avg < 300:
            return 0.4
        return 0.0

    def _no_repetition_score(self, assistant_msgs: list[dict]) -> float:
        if len(assistant_msgs) < 2:
            return 1.0
        repetitions = 0
        for i in range(1, len(assistant_msgs)):
            prev_words = set(assistant_msgs[i - 1].get('text', '').lower().split())
            curr_words = set(assistant_msgs[i].get('text', '').lower().split())
            if not curr_words:
                continue
            overlap = len(prev_words & curr_words) / len(curr_words)
            if overlap > 0.6:
                repetitions += 1
        return max(0.0, 1.0 - repetitions / len(assistant_msgs))


@pytest.fixture
def constraint_checker():
    return ConstraintChecker()


@pytest.fixture
def conversation_scorer(constraint_checker):
    return ConversationScorer(constraint_checker)


@pytest.fixture
def build_chat_context():
    """Build a ChatContext with DEFAULT_INSTRUCTIONS as system + conversation turns."""
    def _build(turns: list[tuple[str, str]], store_name="Gupta Electronics",
               ac_model="Samsung 1.5 Ton 5 Star Inverter Split AC"):
        ctx = ChatContext()
        greeting = f"Hello, yeh {store_name} hai? Aap log AC dealer ho?"
        instructions = DEFAULT_INSTRUCTIONS + (
            f"\nPRODUCT: {ac_model}\nSTORE: {store_name}\n"
            f'\nNOTE: You have already greeted the shopkeeper with: "{greeting}"\n'
            "Do NOT repeat the greeting. Continue the conversation from the shopkeeper's response.\n"
        )
        ctx.add_message(role="system", content=instructions)
        for role, text in turns:
            ctx.add_message(role=role, content=text)
        return ctx
    return _build
