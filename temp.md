# Issues to Fix (from Browser_Test_20260211_203622)

## 1. Greeting spoken twice / sanitizer stripping context
The greeting is added to chat context as an assistant message, but `_sanitize_chat_ctx` requires the first non-system message to be `user`. So it removes the greeting every turn. The LLM then regenerates its own greeting — user hears it twice.

**Fix:** Update `_sanitize_chat_ctx` to not strip the greeting, OR inject the greeting as a system message, OR use `generate_reply()` instead of `say()`.

## 2. Store name wrong in LLM-generated greeting
Because the original greeting gets stripped, the LLM generates its own using "Sharma Electronics" from the EXAMPLES section instead of "Browser Test" from STORE metadata.

**Fix:** Tied to #1. Also remove hardcoded "Sharma Electronics" from EXAMPLES — use a generic placeholder or dynamically insert the actual store name.

## 3. Incomplete/cut-off responses in transcript
Example: `"Hmm, theek hai. Warranty kitni milegi? Aur final mein total kitne"` — truncated mid-word when user interrupts (VAD detects speech). Partial response gets saved.

**Fix:** Either don't save truncated responses, or mark them as interrupted in the transcript.

## 4. Math error in negotiation
Agent says: `"bayaalees hazaar plus dedh hazaar, matlab bayaalees ke paanch hazaar upar"` — 42k + 1.5k ≠ "5 hazaar upar". LLM hallucinates arithmetic.

**Fix:** Prompt instruction to avoid doing math on the call. Just repeat what the shopkeeper says and ask for the final number.

## 5. `\n\n` in responses causing unnatural pauses
Two responses contain double newlines: `"...Theek hai ji. \n\nExchange pe..."`. Indicates multi-paragraph responses instead of the short 1-2 lines requested. TTS may interpret these as long pauses.

**Fix:** Strip `\n` characters in `_normalize_for_tts`. Also reinforce "one line only" in the prompt.

## 6. Hardcoded "Sharma Electronics" in prompt examples
The EXAMPLES section uses "Sharma Electronics" which bleeds into LLM behavior when context is lost (see #1).

**Fix:** Replace with a placeholder like `{store_name}` and dynamically substitute, or use a generic "yeh aapki dukaan hai?".
