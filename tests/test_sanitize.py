"""Tests for SanitizedAgent._sanitize_chat_ctx — ensures correct message ordering for LLM."""

from tests.conftest import SanitizedAgent


class TestSanitizeChatCtx:
    def test_already_valid_user_first(self, make_chat_ctx):
        """No changes needed when user message is first after system."""
        ctx = make_chat_ctx([("system", "You are..."), ("user", "Hello"), ("assistant", "Hi")])
        result = SanitizedAgent._sanitize_chat_ctx(ctx)
        roles = [i.role for i in result.items if hasattr(i, "role")]
        assert roles == ["system", "user", "assistant"]

    def test_assistant_before_user_removed(self, make_chat_ctx):
        """Stale assistant message before first user message gets removed."""
        ctx = make_chat_ctx([("system", "You are..."), ("assistant", "stale"), ("user", "Hello")])
        result = SanitizedAgent._sanitize_chat_ctx(ctx)
        non_system = [i for i in result.items if hasattr(i, "role") and i.role != "system"]
        assert non_system[0].role == "user"

    def test_system_only(self, make_chat_ctx):
        """No crash when only system messages exist."""
        ctx = make_chat_ctx([("system", "You are...")])
        result = SanitizedAgent._sanitize_chat_ctx(ctx)
        assert len(result.items) == 1

    def test_empty_context(self, make_chat_ctx):
        ctx = make_chat_ctx([])
        result = SanitizedAgent._sanitize_chat_ctx(ctx)
        assert len(result.items) == 0

    def test_multiple_system_then_assistant(self, make_chat_ctx):
        ctx = make_chat_ctx([("system", "A"), ("system", "B"), ("assistant", "stale"), ("user", "Hi")])
        result = SanitizedAgent._sanitize_chat_ctx(ctx)
        non_system = [i for i in result.items if hasattr(i, "role") and i.role != "system"]
        assert non_system[0].role == "user"

    def test_returns_copy_not_original(self, make_chat_ctx):
        ctx = make_chat_ctx([("system", "A"), ("assistant", "stale"), ("user", "Hi")])
        result = SanitizedAgent._sanitize_chat_ctx(ctx)
        assert result is not ctx

    def test_user_first_no_system(self, make_chat_ctx):
        ctx = make_chat_ctx([("user", "Hello"), ("assistant", "Hi")])
        result = SanitizedAgent._sanitize_chat_ctx(ctx)
        assert len(result.items) == 2

    def test_assistant_first_no_system(self, make_chat_ctx):
        ctx = make_chat_ctx([("assistant", "stale"), ("user", "Hi")])
        result = SanitizedAgent._sanitize_chat_ctx(ctx)
        non_system = [i for i in result.items if hasattr(i, "role")]
        assert non_system[0].role == "user"
