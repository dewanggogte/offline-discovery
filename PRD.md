# PRD: Hyperlocal Discovery — Voice Price Comparison Agent

## Context

Hyperlocal Discovery is a consumer-facing tool that calls local shops to compare prices on appliances. A user describes what they want ("1.5 ton AC for a 2BHK in Koramangala"), the system researches the product, finds nearby stores, and dispatches an AI voice agent that calls each store pretending to be a regular customer asking for prices in Hindi. Results are compared and the best deal is recommended.

The core voice agent (LiveKit + Sarvam STT/TTS + Claude Haiku) works end-to-end ~~but has critical reliability issues discovered through log analysis of real calls~~. The pipeline (intake → research → store discovery → calling → analysis) is functional ~~but needs polish~~. ~~This PRD prioritizes fixing the identified bugs before any new features.~~

> **Status (Feb 17, 2026):** All 15 pitfalls identified below have been fixed and shipped. 188 tests passing (up from 141). See Implementation Plan section for per-item status.

### What prompted this

Analysis of call logs (Croma, Reliance Digital, Browser Test sessions from Feb 17) revealed 8 pitfalls — from price-corrupting number bugs to TTS crashes on English output. The most severe: streaming token boundaries split numbers like "28000" into "28" + "000", which get independently converted to "attaaees" + "zero" instead of "attaaees hazaar". This corrupts the exact data the product exists to collect.

---

## Product Vision

**One-liner:** "Tell us what you want to buy, and we'll call the shops for you."

**Target user:** A regular consumer in an Indian city who wants to compare prices before buying an appliance — but doesn't want to call 5 shops themselves.

**Deployment modes:**
- **Browser** — user plays the shopkeeper for testing/demo (current primary mode)
- **Phone** — agent calls real shops via SIP trunking for production data collection

---

## Current State

### What exists and works

| Component | Status | Key files |
|---|---|---|
| Voice agent (STT→LLM→TTS pipeline) | Working, has bugs | `agent_worker.py` |
| Browser WebRTC test interface | Working | `app.py` |
| Pipeline: intake chat | Working | `pipeline/intake.py` |
| Pipeline: product research (LLM + web search) | Working | `pipeline/research.py` |
| Pipeline: store discovery (Maps + web search) | Working, fragile | `pipeline/store_discovery.py` |
| Pipeline: dynamic prompt building | Working | `pipeline/prompt_builder.py` |
| Pipeline: cross-store comparison | Working | `pipeline/analysis.py` |
| Pipeline: session orchestrator | Working | `pipeline/session.py` |
| Full web UI (4-step wizard + quick call + dashboard) | Working | `app.py` |
| Post-call quality analysis | Working | `call_analysis.py` |
| Test suite (188 unit + 26 live) | Passing | `tests/` |
| Per-call logging & transcript saving | Working | `agent_worker.py` |
| Dev file watcher (auto-test, auto-analyze) | Working | `dev_watcher.py` |
| Metrics dashboard (TTFT, tokens, latency charts) | Working | `dashboard.py` |
| Docker + Homelab deployment (ArgoCD/K8s) | Working | `Dockerfile`, `.github/workflows/deploy.yaml` |
| GitHub Actions CI (build + push to GHCR + ArgoCD rollout) | Working | ` rkflows/deploy.yaml` |
| Shared agent lifecycle module | Working | `agent_lifecycle.py` |

### Architecture

```
User (browser/phone) ←→ LiveKit Server ←→ Agent Worker
                                             │
                              ┌───────────────┼───────────────┐
                              ▼               ▼               ▼
                         Silero VAD    Sarvam saaras:v3   Claude Haiku
                        (speech det)   (STT-translate)    (LLM, 0.7 temp)
                                              │               │
                              MultilingualModel               │
                              (turn detection)                │
                                                              ▼
                                                    SanitizedAgent
                                                    (llm_node hook)
                                                         │
                                              ┌──────────┼──────────┐
                                              ▼          ▼          ▼
                                     _strip_think   _normalize   _check_
                                       _tags()      _for_tts()   character
                                                         │        _break()
                                                         ▼
                                                  Sarvam bulbul:v3
                                                  (TTS, Hindi)
```

---

## Identified Pitfalls (from log analysis)

### P1 — CRITICAL: Streaming number splitting

**Bug:** `_normalize_for_tts()` runs per-streaming-chunk. LLM token boundaries can split numbers: `"28"` + `"000"` → `"attaaees"` + `"zero"` instead of `"attaaees hazaar"`.

**Evidence:** Reliance call — "attaaeeszero" for ₹28000, "solahzero" for ₹16000. Corrupts the exact price data that is the product's core value.

**Files:** `agent_worker.py:126-139` (llm_node streaming loop)

**Fix:** Buffer trailing digits at the end of each chunk. If a chunk ends with digits, hold them and prepend to the next chunk before applying `_replace_numbers()`. On stream end, flush the buffer.

### P2 — CRITICAL: No TTS error recovery

**Bug:** When the LLM outputs English text, Sarvam TTS crashes with `Text must contain at least one character from the allowed languages`. The error is logged but the call goes silent — no fallback, no retry, no graceful end.

**Evidence:** Croma call — TTS crash on English response, user hears nothing.

**Files:** `agent_worker.py:100-144` (llm_node), session error handler (~line 644)

**Fix:** In the `error` event handler, detect TTS language errors specifically. On TTS crash: (1) log `[TTS CRASH]`, (2) attempt to inject a canned Hindi fallback response ("Ek second, connection mein problem aa rahi hai"), (3) if repeated failures, trigger `end_call` with a graceful Hindi goodbye.

### P3 — HIGH: STT garbage transcripts with no quality signal

**Bug:** Sarvam STT returns garbled translations with no confidence score (hardcoded to 1.0 in plugin). "Table.", "Tell me the round from there.", "You can tell the Mohadarma or whatever." — all passed to the LLM as valid input.

**Evidence:** Reliance call ("Table."), Croma call ("Tell me the round from there."), Browser tests.

**Files:** `agent_worker.py:594-598` (user_input_transcribed handler)

**Fix:** Add a heuristic garbage filter in the `user_input_transcribed` handler. Flag transcripts that are: (a) under 3 words and contain no recognizable keywords, or (b) contain known STT artifacts like "Table.", "The.", "And." as the entire message. Log `[STT GARBAGE]` and optionally skip forwarding to LLM. Start with logging-only to measure frequency before filtering.

### P4 — HIGH: LLM latency (TTFT 0.7–1.7s, growing with context)

**Issue:** System prompt alone is ~2400 tokens. Over a 14-turn call, prompt tokens grow from 2419 to 3113. TTFT ranges from 0.72s to 1.70s. Total mouth-to-ear latency ~2.3s — well above the 1s threshold.

**Evidence:** Reliance call TTFT values: 0.97, 0.75, 1.17, 0.72, 0.96, 0.79, **1.70**, **1.50**, **1.47**, 1.17, 0.89, **1.45**, 1.02, 1.07.

**Files:** `agent_worker.py:462-484` (_create_llm), `agent_worker.py:373-455` (DEFAULT_INSTRUCTIONS), `pipeline/prompt_builder.py`

**Fix (multi-pronged):**
1. **Enable Anthropic prompt caching** — add `caching="ephemeral"` to the LLM config. Up to 80% TTFT reduction on cached prefix (highest-impact single change).
2. **Add `max_tokens=150`** to Claude LLM config — caps response length, faster generation.
3. **Trim prompt** — remove EXAMPLES section from system prompt (saves ~200 tokens). Move to few-shot in chat context only when needed.
4. **Measure** — add TTFT to the `[LLM METRICS]` log (already exists) and track p50/p90 trends.

### P5 — MEDIUM: Role reversal under adversarial input

**Bug:** When user speaks as a customer (asking questions), the agent reverses roles and becomes the shopkeeper. "Haan ji, hum AC repair bhi karte hain aur naye AC bhi bechte hain."

**Evidence:** Browser_Test_161408 — agent answered "Do you repair ACs?" as if it were the shop.

**Files:** `agent_worker.py:446-455` (STAY IN CHARACTER), `pipeline/prompt_builder.py:101-109`

**Fix:** Strengthen the prompt's role anchoring. Add after the STAY IN CHARACTER section:
```
- If the user asks YOU a question as if YOU are the shopkeeper (e.g. "Do you repair ACs?", "What brands do you have?"), DO NOT answer as the shopkeeper. Instead, redirect: "Nahi nahi, main toh customer hoon. Mujhe AC ka price chahiye."
```

### P6 — MEDIUM: Character breaks (English responses)

**Bug:** LLM occasionally responds in English, especially on confusing first messages or when asked to speak English.

**Evidence:** Browser_Test_155257 — "Okay, do you have Samsung..." and "I only speak Hindi. Let me try again..." Both are English.

**Files:** `agent_worker.py:190-210` (_check_character_break — already added), prompt sections

**Status:** Partially mitigated by the previous commit (STT translation explanation, "NEVER respond in English" rule, greeting in chat context, character break detection logging). The logging is in place; the remaining gap is active recovery — when a character break IS detected, retry with a canned Hindi response instead of sending English to TTS.

**Fix:** In `llm_node`, after `_check_character_break()` detects a break, replace the accumulated response text with a canned Hindi fallback: "Achha ji, aap AC ka price bata dijiye." Log the replacement. This prevents the English text from reaching TTS.

### P7 — LOW: LLM outputs Hindi words instead of digits

**Bug:** LLM sometimes writes "do sau pachaas" instead of "250", bypassing the deterministic number conversion.

**Evidence:** Reliance call — "Haan, do sau pachaas liter wala de do."

**Files:** `agent_worker.py:435-436` (digit instruction in prompt)

**Fix:** Reinforce in prompt with a negative example. Low priority since TTS handles Hindi words fine — this only matters for the transcript analysis number-echo checker.

### P8 — LOW: Greeting repeated (FIXED)

Already fixed in previous commit — greeting now in chat context with synthetic `[call connected]` user message.

### P9 — HIGH: Research intelligence not passed to voice agent

**Bug:** `prompt_builder.py` uses only 4 of 7 `ResearchOutput` fields (`questions_to_ask`, `topics_to_cover`, `topic_keywords`, `market_price_range`). Three fields are ignored: `product_summary`, `competing_products`, `important_notes`. The agent has zero product knowledge — when a shopkeeper asks "which model do you want?", it can only repeat "best model kya hai?" because it doesn't know any model names.

**Evidence:** Reliance Digital fridge call — shopkeeper asked "which model?" 4 times. Agent looped on the same question with no recovery.

**Files:** `pipeline/prompt_builder.py`

**Fix:** Add 3 new prompt sections: PRODUCT KNOWLEDGE (summary + top 3 competing products), BUYER NOTES (top 3 important_notes), WHEN STUCK (strategies for recovery). All conditional — empty data = section omitted. ~170-230 extra tokens, cached prefix.

### P10 — MEDIUM: Verbose greeting confuses shopkeepers

**Bug:** Greeting uses the raw `category` field (e.g. "Medium double door fridge with separate freezer section (220-280L)"). Shopkeeper literally said "I didn't understand anything."

**Evidence:** Reliance Digital fridge call — verbose greeting confused the shopkeeper.

**Files:** `pipeline/prompt_builder.py`, `pipeline/session.py`

**Fix:** Add `_casual_product_name()` helper that strips parenthetical specs, size adjectives, and "with ..." clauses. Add `build_greeting()` function. Replace verbose category with casual name in all spoken prompt sections.

### P11 — MEDIUM: No topic pivot strategy when stuck

**Bug:** When a conversation stalls (shopkeeper keeps asking "which model?", agent can't answer), the agent has no strategy for pivoting to a different topic or unblocking itself.

**Evidence:** Same Reliance Digital call — 4 failed "which model?" exchanges.

**Files:** `pipeline/prompt_builder.py`

**Fix:** Add WHEN STUCK section to prompt: (1) name a specific model from research, (2) "Achha theek hai" and pivot after 2 failed attempts, (3) anchor to lower end of price range if asked about budget.

### P12 — MEDIUM: Duplicate greeting in transcript

**Bug:** Greeting appears twice in the transcript. `session.say(greeting, add_to_chat_ctx=True)` fires the `conversation_item_added` handler which appends to `transcript_lines`, AND there was an explicit `transcript_lines.append()` right after — double recording.

**Evidence:** Girias call transcript — identical greeting at timestamps 17:47:19 and 17:47:25.

**Files:** `agent_worker.py` (~line 900)

**Fix:** Remove the explicit `transcript_lines.append()` after `session.say()`. The `conversation_item_added` handler already captures it.

### P13 — MEDIUM: LLM repeats greeting as first response (pipeline prompt)

**Bug:** `prompt_builder.build_prompt()` was missing the NOTE telling the LLM that the greeting was already spoken. Unlike `agent_worker.py`'s `DEFAULT_INSTRUCTIONS` (which had the NOTE), the pipeline prompt let the LLM generate the greeting again as its first response.

**Evidence:** Girias call — greeting spoken twice (once by `session.say()`, once by LLM).

**Files:** `pipeline/prompt_builder.py`

**Fix:** Add `greeting_note` to `build_prompt()` — `"NOTE: You have already greeted the shopkeeper with: '{greeting}'. Do NOT repeat the greeting."` Appended at the end of the prompt after the STORE line.

### P14 — MEDIUM: Research phase blocks HTTP server

**Bug:** `_handle_research()` in `app.py` called `asyncio.run(session.research_and_discover())` synchronously, blocking the single-threaded `HTTPServer`. During the 10-20 second research phase, the frontend couldn't poll for events — it appeared stuck with no progress updates.

**Files:** `app.py`

**Fix:** Run research in a background `threading.Thread`. Return immediately with `{"status": "started"}`. Add a GET endpoint `/api/session/{id}/research` for polling. Frontend POST starts research, then polls GET until `{"status": "done"}`.

### P15 — HIGH: Results table shows only store name (no price/warranty data)

**Bug:** `_collect_call_results_from_transcripts()` populated `extracted_data` from `analysis.get("scores", {})` — which gives constraint quality scores (`{"constraint": 1.0, "topic": 1.0}`), NOT actual price/warranty data. The comparison LLM received useless metrics. For single-store calls, the LLM was skipped entirely, returning raw `CallResult` with no structured price data.

**Evidence:** All calls — results table showed store name and "Best Deal" badge but no price, warranty, installation, or delivery information.

**Files:** `pipeline/session.py`, `pipeline/analysis.py`

**Fix:** (1) Include transcript messages in `extracted_data` so the comparison LLM has actual conversation to analyze. (2) Always run through the LLM to extract structured data (even for single-store). (3) Add `warranty` field to the LLM output schema. (4) Update frontend to display warranty in the extras column.

---

## Implementation Plan

> **All items below are COMPLETE.** Strikethrough indicates shipped code.

### ~~Phase 1: Critical Bug Fixes (P1 + P2)~~ DONE

#### ~~1a. Fix streaming number splitting~~ SHIPPED

**File:** `agent_worker.py` — `SanitizedAgent.llm_node()`

Add a digit buffer to the streaming loop. The current code:
```python
async for chunk in Agent.default.llm_node(...):
    chunk = _normalize_for_tts(chunk)  # BUG: splits numbers
    yield chunk
```

New approach — create `_NumberBufferedNormalizer` class:
- Maintains a `_digit_buffer: str` across chunks
- When a chunk ends with digits (regex `\d+$`), strip them and hold in buffer
- When next chunk arrives, prepend the buffer
- When a chunk does NOT end with digits and buffer is non-empty, flush buffer with current chunk
- On stream end (after the `async for` loop), flush any remaining buffer
- Apply `_normalize_for_tts()` only on flushed/complete text

**Tests to add:** `tests/test_normalization.py`
- `test_streaming_number_split_28000` — chunks `["Achha, 28", "000."]` → `"Achha, attaaees hazaar."`
- `test_streaming_number_split_16000` — chunks `["solah", "000"]` → handled (no digits to split here, this is the LLM writing words — no buffer needed)
- `test_streaming_no_split_needed` — chunks `["Achha, ", "38000", "."]` → `"Achha, adtees hazaar."`
- `test_streaming_number_at_end_of_stream` — chunk `["price 500"]` with no following chunk → flushed as `"price paanch sau"`

#### ~~1b. Add TTS crash recovery~~ SHIPPED

**File:** `agent_worker.py` — session error handler and `llm_node`

Two layers of defense:

**Layer 1 (llm_node):** After character break detection, if break detected, replace response with canned Hindi:
```python
if self._last_response_text:
    _check_character_break(self._last_response_text)
    if _is_character_break(self._last_response_text):  # same logic, returns bool
        logger.warning(f"[CHARACTER BREAK RECOVERY] Replacing English response")
        # Can't un-yield chunks already sent, but we can flag for TTS error handler
        self._character_break_detected = True
```

**Layer 2 (error handler):** Detect TTS language errors and inject fallback:
```python
@session.on("error")
def on_error(ev):
    error = ev.error
    if hasattr(error, 'error') and 'allowed languages' in str(error.error):
        logger.error(f"[TTS CRASH] English text sent to Hindi TTS")
        # The TTS will retry (retryable=True), but the text is still English.
        # We can't change it mid-stream. Log for now.
        # Future: intercept at llm_node level before TTS gets it.
```

The real fix is Layer 1 — prevent English from reaching TTS. Layer 2 is defense-in-depth logging.

**Tests:** Hard to unit test (requires mocking LiveKit session). Add to `test_normalization.py`:
- `test_is_character_break_english` — pure English → True
- `test_is_character_break_hindi` — Romanized Hindi → False
- `test_is_character_break_mixed` — mixed → False (has Hindi markers)

### ~~Phase 2: Quality Improvements (P3 + P4 + P5)~~ DONE

#### ~~2a. STT garbage detection~~ SHIPPED

**File:** `agent_worker.py` — `on_user_transcript` handler

Add heuristic filter:
```python
_GARBAGE_PATTERNS = {"table", "the", "and", "a", "an", "it", "is", "to", "of"}

def _is_likely_garbage(text: str) -> bool:
    words = text.strip().rstrip('.!?').lower().split()
    if len(words) <= 1 and words[0] in _GARBAGE_PATTERNS:
        return True
    return False
```

Log `[STT GARBAGE]` but still forward to LLM (logging-only phase). After collecting data on frequency, decide whether to filter.

**Tests:** `tests/test_normalization.py` (or new `test_stt_filter.py`)
- `test_garbage_single_word` — "Table." → garbage
- `test_garbage_the` — "The." → garbage
- `test_valid_short` — "Yes." → not garbage
- `test_valid_sentence` — "Tell me the price." → not garbage

#### ~~2b. Latency optimization — prompt caching~~ SHIPPED

**File:** `agent_worker.py` — `_create_llm()` and `SanitizedAgent.llm_node()`

Enable Anthropic prompt caching:
- The `livekit-plugins-anthropic` plugin supports `caching="ephemeral"` — add it to the `anthropic.LLM()` constructor.
- Add `max_tokens=150` to the LLM config.

**Verification:** Compare TTFT values in logs before/after. Target: p50 TTFT < 0.6s (down from ~1.0s).

#### ~~2c. Role reversal prevention~~ SHIPPED

**Files:** `agent_worker.py` (DEFAULT_INSTRUCTIONS), `pipeline/prompt_builder.py`

Add to STAY IN CHARACTER section in both files:
```
- If the user asks YOU a question as if YOU are the shopkeeper (e.g. "Do you repair ACs?", "What brands do you have?"), DO NOT answer. Redirect: "Nahi nahi, main toh customer hoon. Mujhe [product] ka price chahiye."
```

**Tests:** Add to `test_sanitize.py`:
- `test_prompt_has_role_reversal_guard` — DEFAULT_INSTRUCTIONS contains "main toh customer hoon"

### ~~Phase 3: Pipeline Polish~~ DONE

#### ~~3a. Fix `_active_session` global singleton~~ SHIPPED

**File:** `pipeline/session.py`

The `_active_session` module global means only one session captures log events. Fix: use a dict of active sessions keyed by `id(session)` so multiple concurrent sessions each get their own events.

#### ~~3b. Fix synchronous LLM calls in async functions~~ SHIPPED

**Files:** `pipeline/analysis.py`, `pipeline/store_discovery.py`

Wrap synchronous `client.messages.create()` calls in `asyncio.to_thread()` to avoid blocking the event loop.

#### ~~3c. Add tests to CI~~ SHIPPED

**File:** `.github/workflows/deploy.yaml`

Add a test job before the Docker build:
```yaml
- name: Run tests
  run: pip install -r requirements.txt && pytest tests/ -q --tb=short
```

#### ~~3d. Extract shared agent lifecycle code~~ SHIPPED

**Files:** `app.py`, `agent_lifecycle.py`

Extract `kill_old_agents()`, `start_agent_worker()`, `cleanup_agent()`, `find_agent_log()` into a shared `agent_lifecycle.py` module.

### ~~Phase 4: Test Coverage Gaps~~ DONE

Add tests for the most critical untested paths:

1. **Number buffer streaming** (Phase 1a tests above)
2. **Character break detection + recovery** (Phase 1b tests above)
3. **STT garbage filter** (Phase 2a tests above)
4. **Role reversal guard** (Phase 2c tests above)

### ~~Phase 5: Research Intelligence + Conversation Recovery (P9 + P10 + P11)~~ DONE {#phase-5}

#### ~~5a. Flow research data into voice agent prompt~~ SHIPPED

**File:** `pipeline/prompt_builder.py`

Add `_build_research_sections()` that generates 3 conditional prompt sections from previously-ignored research fields:
- **PRODUCT KNOWLEDGE** — `product_summary` + top 3 `competing_products` (name, price_range, pros). Tells agent to name a model when asked "which one?"
- **BUYER NOTES** — top 3 `important_notes` as bullets.
- **WHEN STUCK** — 3 strategies: name a model, pivot after 2 fails, anchor to low price.

All sections omitted when data is empty. ~170-230 tokens, negligible latency with prompt caching.

#### ~~5b. Fix verbose greeting~~ SHIPPED

**Files:** `pipeline/prompt_builder.py`, `pipeline/session.py`

Add `_casual_product_name()` helper: strips `(specs)`, leading size adjectives, `with ...` clauses. Falls back to `product_type` if result too short. Add `build_greeting()` that uses casual name. Replace `{product_desc}` with `{casual}` in all spoken prompt sections (opening line, CONVERSATION FLOW, ENDING THE CALL, STAY IN CHARACTER, EXAMPLES). Keep verbose `product_desc` in the `PRODUCT:` reference line at the end.

Update `session.py` to call `prompt_builder.build_greeting()` instead of building greeting inline.

#### ~~5c. Add "which model?" recovery to examples~~ SHIPPED

**File:** `pipeline/prompt_builder.py`

Update `_build_examples()` to include a model recovery exchange when competing_products exist:
```
Shopkeeper: "Kaun sa model chahiye?"
You: "Achha, [first_model_name] ka kya price hai?"
```

#### ~~5d. Tests for prompt builder~~ SHIPPED

**File:** `tests/test_prompt_builder.py` (new)

~15 tests covering:
- `TestCasualProductName` — parenthetical stripping, size adjective, tonnage preserved, with-clause, fallback
- `TestBuildGreeting` — casual name used, store name present, format correct
- `TestBuildPromptWithResearch` — PRODUCT KNOWLEDGE, BUYER NOTES, WHEN STUCK present/absent, caps at 3, casual in spoken sections, verbose in PRODUCT: line, model recovery in examples

### ~~Phase 6: Transcript & UI Fixes (P12 + P13 + P14 + P15)~~ DONE

#### ~~6a. Fix duplicate greeting in transcript (P12)~~ SHIPPED

**File:** `agent_worker.py`

Remove explicit `transcript_lines.append()` after `session.say(greeting, add_to_chat_ctx=True)`. The `conversation_item_added` handler already captures it — the explicit append created duplicates.

#### ~~6b. Add greeting NOTE to pipeline prompt (P13)~~ SHIPPED

**File:** `pipeline/prompt_builder.py`

Add `greeting_note` after the `STORE:` line:
```
NOTE: You have already greeted the shopkeeper with: "Hello, yeh Croma hai? split AC ke baare mein poochna tha."
Do NOT repeat the greeting. Continue the conversation from the shopkeeper's response.
```
Uses `build_greeting()` to generate the greeting text (with casual product name), ensuring consistency between what's spoken and what the NOTE says. 2 new tests added.

#### ~~6c. Non-blocking research with progress polling (P14)~~ SHIPPED

**File:** `app.py`, `pipeline/session.py`

- Research runs in a background `threading.Thread` instead of blocking `asyncio.run()`
- Session caches result on `_research_result` / `_research_error`
- New GET endpoint `/api/session/{id}/research` for polling
- Frontend: POST starts research → polls GET every 2s → renders results on completion
- Event log continues to populate during research (no longer blocked)

#### ~~6d. Fix results table data flow (P15)~~ SHIPPED

**Files:** `pipeline/session.py`, `pipeline/analysis.py`, `app.py`

- `_collect_call_results_from_transcripts()` now includes transcript messages in `extracted_data` (not constraint scores)
- `compare_stores()` formats transcript into readable conversation and sends to LLM for extraction
- Single-store calls now also go through LLM to extract structured price/warranty/delivery data
- Added `warranty` field to LLM output schema and frontend rendering
- `_format_transcript()` helper converts `[{role, text}]` messages into `"Agent: ... / Shopkeeper: ..."` text

**Tests:** 188 unit tests pass (22 prompt builder + 73 normalization + 25 sanitize + 34 offline scenarios + 11 conversation + 11 transcript + 6 logs + 6 LLM provider + 26 live skipped).

---

## Files Modified

| File | Changes |
|---|---|
| `agent_worker.py` | P1: digit buffer in llm_node streaming. P2: character break recovery + TTS error handling. P3: STT garbage logging. P4: max_tokens + prompt caching. P5: role reversal prompt. P12: remove duplicate greeting append. |
| `pipeline/prompt_builder.py` | P5: role reversal guard. P9: `_build_research_sections()` for PRODUCT KNOWLEDGE / BUYER NOTES / WHEN STUCK. P10: `_casual_product_name()` + `build_greeting()`, casual name in spoken sections. P11: WHEN STUCK strategies. P13: greeting NOTE to prevent LLM repeating greeting. |
| `pipeline/session.py` | P3a: per-session logging handler. P10: use `prompt_builder.build_greeting()`. P14: `_research_result`/`_research_error` caching. P15: transcript messages in `extracted_data`. |
| `pipeline/analysis.py` | P3b: asyncio.to_thread for sync LLM calls. P15: `_format_transcript()`, transcript-based extraction, single-store LLM analysis, `warranty` field. |
| `pipeline/store_discovery.py` | P3b: asyncio.to_thread for sync LLM calls. |
| `.github/workflows/deploy.yaml` | P3c: add test job before Docker build. |
| `app.py` | P3d: import from shared agent_lifecycle.py. P14: background threading for research, GET polling endpoint. P15: warranty in results table. |
| `agent_lifecycle.py` (new) | P3d: shared agent worker management functions. |
| `tests/conftest.py` | New imports for test helpers. |
| `tests/test_normalization.py` | P1a: 8 streaming number buffer tests. |
| `tests/test_sanitize.py` | P1b + P2a + P2c: 17 new tests (character break, STT garbage, role reversal). |
| `tests/test_prompt_builder.py` (new) | P9 + P10 + P11: 22 tests for casual names, greeting, research sections, model recovery, greeting NOTE. |

---

## Verification

### After Phase 1 (critical bugs)
1. `pytest tests/` — all tests pass including new streaming number tests
2. Manual test: start browser session, say "₹28000 hai" — verify agent echoes "attaaees hazaar" (not "attaaeeszero")
3. Manual test: speak English to agent — verify no TTS crash, agent stays in Hindi
4. Check logs: no `[TTS CRASH]` errors, `[CHARACTER BREAK]` warnings trigger recovery

### After Phase 2 (quality)
1. Check TTFT in logs — p50 should drop from ~1.0s to ~0.6s with prompt caching
2. `[STT GARBAGE]` warnings appear in logs for known garbage patterns
3. Manual test: ask agent "Do you repair ACs?" — agent redirects instead of answering as shopkeeper

### After Phase 3 (pipeline polish)
1. Two concurrent browser sessions both get correct event streams
2. GitHub Actions runs tests before building Docker image
3. `app.py` works with shared `agent_lifecycle.py`

### After Phase 4 (test coverage)
1. `pytest tests/` — total test count increases from 141 to 166
2. Coverage of streaming number buffer, character break detection, STT garbage filter, role reversal guard

### After Phase 5 (research intelligence)
1. `pytest tests/test_prompt_builder.py` — 22 tests pass for casual names, greeting, research sections
2. Generated prompt with full research data contains PRODUCT KNOWLEDGE, BUYER NOTES, WHEN STUCK
3. Generated prompt with empty research gracefully omits all three sections

### After Phase 6 (transcript & UI fixes)
1. Greeting appears exactly once in transcript (no duplicate)
2. LLM does not repeat greeting in its first response
3. Research phase shows progress in event log (not stuck/blank)
4. Results table shows price, installation, delivery, warranty — not empty
5. `pytest tests/` — 188 tests pass

---

## Phase 2: Next Improvements

> **Status (Feb 19, 2026):** All Phase 2 features DONE except F7 (logging server, skipped). 253 tests passing (up from 217). Implemented: F15, F10, F13, F2, F3, F6, F16, F17, F1, F4, F5, F8, F11, F18, F14, F9, F12.

### Priority / Effort Matrix

| # | Feature | Priority | Effort | Dependencies | Phase |
|---|---------|----------|--------|-------------|-------|
| F1 | Sub-1.5s → sub-1s voice latency | Critical | Large | None | 2F |
| F2 | Research → agent knowledge transfer | High | Medium | F13 | 2B |
| F3 | Prompt restructure + negotiation | High | Large | F2 | 2B |
| F4 | Male vs female voice A/B testing | Medium | Medium | None | 2C |
| F5 | TTS temperature / pace tuning | Low | Small | F4 | 2C |
| F6 | Automatic store selection | High | Small | None | 2B |
| F7 | Logging server on homelab | Medium | Large | None | 2F |
| F8 | Product-specific test cases + scoring | High | Large | F2, F3 | 2D |
| F9 | Blog for CallKaro | Low | Medium | None | 2G |
| F10 | Fix bold font in agent chat UI | Low | Trivial | None | 2A |
| F11 | Intake chatbot improvements | Medium | Medium | F13 | 2E |
| F12 | Market analysis for other use cases | Low | Small | None | 2G |
| F13 | Research module update | High | Medium | None | 2B |
| F14 | Subagent modularity / tweakability | Medium | Large | F2, F3, F13 | 2E |
| F15 | Remove quick call feature | Low | Trivial | None | 2A |
| F16 | Remove personal branding from UI | Medium | Trivial | None | 2A |
| F17 | Fix "bhai sahab" pronunciation | Medium | Trivial | None | 2A |
| F18 | Intake suggestion chips | High | Medium | F11 | 2E |

### Dependency Graph

```
F15 (remove quick call) ──────────────── standalone
F16 (remove personal branding) ───────── standalone
F17 (fix bhai sahab pronunciation) ───── standalone
F10 (fix bold font) ──────────────────── standalone

F13 (research module) ───┬── F2 (knowledge transfer) ───── F3 (prompt + negotiation)
                         │                                       │
                         └── F6 (auto store selection)           │
                                                                 ▼
F4 (voice A/B) ──── F5 (TTS temperature)            F8 (product tests) ← validates F2+F3

F11 (intake) ← F13
F18 (suggestion chips) ← F11
F14 (modularity) ← F2, F3, F13

F7 (logging server) ──────────────────── standalone (infra)
F1 (latency) ─────────────────────────── standalone (tuning)
F9 (blog) ────────────────────────────── standalone (non-code)
F12 (market analysis) ────────────────── standalone (non-code)
```

**Critical chain:** F13 → F2 → F3 → F8

---

### Phase 2A: Quick Wins {#phase-2a}

#### F15 — Remove Quick Call Feature

**Rationale:** Quick Call was a prototype shortcut. The full 4-step wizard (intake → research → store selection → calling) is now the primary UX. The Quick Call tab adds UI clutter and a maintenance surface for an unused path.

**Current state:** Quick Call tab, JS functions, API route, and token generation all live in `app.py`.

**Implementation:**

1. **Remove Quick Call tab HTML** — `app.py:487-503` (`<div id="tab-voice">` and contents)
2. **Remove tab button** — `app.py` search for `tab-voice` in the tab bar (~line 370-385)
3. **Remove JS functions** — `app.py:1180-1219` (`qcStart()`, `qcEnd()`, `qcRoom`/`qcMic`/`qcAgent`/`qcViz` declarations)
4. **Remove API route** — `app.py:1458` (`elif path == "/api/token":` → `self._serve_token()`)
5. **Remove token generation** — `app.py:1773-1812` (`create_legacy_token()` async function)
6. **Remove any remaining references** — grep for `qcStart`, `qcEnd`, `tab-voice`, `_serve_token`, `create_legacy_token`

**Tests:** Verify `pytest tests/` still passes. No new tests needed (removing code).

#### F16 — Remove Personal Branding from UI

**Rationale:** The frontend currently shows "Dewang Gogte" at the top of the page and in the footer. The UI should be purely about CallKaro as a product — no personal name or personal website references. This makes the app look more professional and product-focused.

**Current state:** Header and footer in `app.py` HTML contain personal name/branding.

**Implementation:**

1. **Remove name from header** — replace or remove any "Dewang Gogte" text in the top/header area of the HTML in `app.py`. The header should show only the CallKaro brand name/logo.
2. **Remove footer personal branding** — remove the footer section that references "Dewang Gogte" or links to a personal website. Either remove the footer entirely or replace with a minimal CallKaro-only footer.
3. **Grep for remaining references** — search `app.py` for any other mentions of "Dewang", "Gogte", or personal website URLs and remove them.

**Tests:** Manual — verify the UI shows only CallKaro branding with no personal name references.

#### F17 — Fix "bhai sahab" Pronunciation

**Rationale:** Sarvam TTS pronounces "bhai sahab" (two words) with intonation that sounds sarcastic rather than a plain respectful pronoun. Merging into "bhaisaab" (single word) produces a more natural, casual pronunciation.

**Current state:** "bhai sahab" appears in prompts and examples across multiple files:
- `agent_worker.py:540` — prompt instruction ("Use bhai sahab ONLY ONCE...")
- `agent_worker.py:611` — example dialogue
- `pipeline/prompt_builder.py:123` — same prompt instruction
- `pipeline/prompt_builder.py:383` — example dialogue
- `tests/test_normalization.py` — 2 test strings
- `tests/test_sanitize.py` — 1 test string
- `tests/test_scenario_offline.py` — 4 test strings
- `tests/test_scenario_live.py` — 1 test string

**Implementation:**

1. **Replace all occurrences** of `bhai sahab` / `Bhai sahab` with `bhaisaab` / `Bhaisaab` across all files listed above. Simple find-and-replace.
2. **Verify TTS output** — manual test to confirm "bhaisaab" sounds natural and non-sarcastic with Sarvam bulbul:v3.

**Tests:** `pytest tests/` — all existing tests pass after the rename. Manual TTS listen test.

#### F10 — Fix Bold Font in Agent Chat UI

**Rationale:** `addChatMsg()` uses `innerHTML` with only `\n→<br>` replacement. LLM markdown like `**bold**` passes through as raw text instead of rendering.

**Current state:** `app.py:714-721` — `addChatMsg()` function sets `div.innerHTML` directly.

**Implementation:**

1. Add `escapeHtml()` JS helper before `addChatMsg()`:
   ```javascript
   function escapeHtml(t) {
     return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
   }
   ```
2. In `addChatMsg()`, change `div.innerHTML = text.replace(/\n/g, '<br>')` to:
   ```javascript
   div.innerHTML = escapeHtml(text)
     .replace(/\*\*(.*?)\*\*/g, '<b>$1</b>')
     .replace(/\n/g, '<br>');
   ```

**Tests:** Manual — send a message containing `**bold**` and verify it renders as **bold** in the chat UI.

---

### Phase 2B: Research Intelligence Upgrade {#phase-2b}

#### F13 — Research Module Update

**Rationale:** Current research does generic product research and store discovery as disconnected modules. Research should recommend specific products based on user inputs, and store discovery should rank stores for the identified product.

**Current state:**
- `pipeline/schemas.py:57-93` — `ResearchOutput` has 7 fields (product_summary, market_price_range, questions_to_ask, topics_to_cover, topic_keywords, important_notes, competing_products)
- `pipeline/schemas.py:97-134` — `DiscoveredStore` has basic fields (name, address, phone, rating, review_count, area, city, source)
- `pipeline/research.py:22-65` — `SYSTEM_PROMPT` instructs generic research
- `pipeline/store_discovery.py:200-257` — `_deduplicate_and_structure()` does basic dedup

**Implementation:**

1. **Extend `ResearchOutput`** in `pipeline/schemas.py`:
   - Add `recommended_products: list[dict]` — top 3 recommended products with model names, specs, typical street prices, pros/cons
   - Add `negotiation_intelligence: dict` — common dealer margins, seasonal pricing, festival discount patterns
   - Add `insider_knowledge: list[str]` — e.g. "Samsung 2025 models had compressor issues — dealers may discount"

2. **Extend `DiscoveredStore`** in `pipeline/schemas.py`:
   - Add `specialist: bool` — whether this store specializes in the product category
   - Add `relevance_score: float` — product-aware ranking score

3. **Update research system prompt** in `pipeline/research.py:22-65`:
   - Instruct LLM to recommend specific products (not just research what user asked)
   - Search for common negotiation tactics for the product category
   - Search for known issues, recall notices, seasonal pricing for the product
   - Increase max search rounds from 3 to 4 (`pipeline/research.py:126`)

4. **Add product-aware store scoring** in `pipeline/store_discovery.py`:
   - In `_deduplicate_and_structure()` (~line 200), add scoring: ask LLM to assess store relevance to specific product
   - Populate `specialist` and `relevance_score` on each `DiscoveredStore`

**Tests:**
- `tests/test_prompt_builder.py` — verify new research fields flow into prompt sections
- Unit test for `relevance_score` calculation

#### F2 — Research → Agent Knowledge Transfer

**Rationale:** The voice agent should sound like a seasoned buyer. Current prompt uses research data but lacks negotiation intelligence, insider knowledge, and detailed competing product info.

**Current state:** `pipeline/prompt_builder.py:254-308` — `_build_research_sections()` builds PRODUCT KNOWLEDGE, BUYER NOTES, WHEN STUCK from `ResearchOutput`.

**Implementation:**

1. **Extend `_build_research_sections()`** in `pipeline/prompt_builder.py:254-308`:
   - Add **RECOMMENDED PRODUCTS** section — top 3 products with model name, typical street price, key specs. Agent uses these to sound knowledgeable ("Samsung AR18 ka kya rate hai? Online toh 38,000 dikh raha tha")
   - Add **NEGOTIATION INTELLIGENCE** section — dealer margins, seasonal pricing, competitor price references. Agent casually drops these mid-conversation
   - Add **INSIDER KNOWLEDGE** section — known issues, recall info. Agent uses strategically ("Suna hai Samsung ke 2025 models mein compressor issue tha...")

2. **All new sections are conditional** — empty data = section omitted. Target: system prompt stays under 3500 tokens (~14000 chars) with prompt caching.

**Tests:**
- `tests/test_prompt_builder.py` — add ~6 tests: new sections present when data exists, absent when empty, content accuracy, token count within budget

#### F3 — Prompt Restructure + Dynamic Conversation Flow + Negotiation

**Rationale:** Current conversation flow is static (confirm shop → ask price → negotiate → wrap up) regardless of product, location, or store type. Negotiation is basic. Flow should adapt to context.

**Current state:**
- `pipeline/prompt_builder.py:132-138` — static CONVERSATION FLOW section
- `agent_worker.py:533-619` — DEFAULT_INSTRUCTIONS with hardcoded flow

**Implementation:**

1. **New `_build_conversation_flow()` helper** in `pipeline/prompt_builder.py`:
   - **Product-aware flow:** AC needs tonnage confirmation first, washing machine needs capacity, fridge needs type (single/double door), laptop needs use case
   - **Store-aware flow:** Chain store (Croma/Reliance) — ask for offers/combos; local dealer — negotiate harder, mention competitor prices
   - **Location-aware flow:** Regional pricing norms (Bangalore vs Delhi), local negotiation culture

2. **New `_build_negotiation_section()` helper** in `pipeline/prompt_builder.py`:
   - Reference competitor prices from research ("Online toh X dikh raha tha")
   - Mention visiting multiple stores ("Main 2-3 shops se rate le raha hoon")
   - Use research knowledge strategically (mention specific models, features)
   - Anchor to lower end of market price range
   - Keep tone gentle — "seasoned buyer", not aggressive haggler

3. **Update `DEFAULT_INSTRUCTIONS`** in `agent_worker.py:533-619`:
   - Add negotiation note after line 558
   - Replace static conversation flow with dynamic placeholder that `instructions_override` from pipeline fills

**Tests:**
- `tests/test_prompt_builder.py` — add ~6 tests: product-specific flow for AC/fridge/washing machine, negotiation section content, store-type adaptation

#### F6 — Automatic Store Selection

**Rationale:** Users shouldn't need to manually pick stores. The system has enough data (rating, reviews, phone availability, source) to auto-select the best candidates.

**UX decision:** Automatic with confirmation — system picks stores, shows summary ("We'll call these 4 stores"), user clicks Confirm or Edit.

**Current state:**
- `pipeline/store_discovery.py` — returns up to 10 stores with `rating`/`review_count`
- `pipeline/session.py:134-181` — `research_and_discover()` returns stores for manual selection
- `app.py` — Step 2 UI shows checkboxes for store selection

**Implementation:**

1. **New `rank_stores()` function** in `pipeline/store_discovery.py`:
   ```python
   def rank_stores(stores: list[DiscoveredStore], top_n: int = 4) -> list[DiscoveredStore]:
       """Score and rank stores. Formula:
       rating (0-5, weight 30%) + log(review_count) (weight 20%)
       + phone_available (weight 30%) + google_maps_source (weight 20%)
       """
   ```
   - Returns top N stores sorted by score
   - Uses `specialist` and `relevance_score` from F13 when available

2. **Auto-rank after discovery** in `pipeline/session.py:134-181`:
   - After `research_and_discover()`, call `rank_stores()` to select top 4
   - Store ranked list on session for UI display

3. **Update Step 2 UI** in `app.py`:
   - Replace checkbox list with confirmation view: "We'll call these stores:" + ranked list
   - Show rank score badge on each store
   - Two buttons: "Start Calls" (confirm) and "Edit Selection" (reveals checkboxes)
   - "Edit Selection" falls back to current manual selection UX

**Tests:**
- Unit test for `rank_stores()` scoring formula
- Test that stores without phone numbers rank lower
- Test top_n parameter

---

### Phase 2C: Voice Experimentation {#phase-2c}

#### F4 — Male vs Female Voice A/B Testing

**Rationale:** Voice gender and style may significantly affect shopkeeper engagement and negotiation outcomes. Systematic A/B testing needed.

**Current state:** `agent_worker.py:716-726` — hardcoded `speaker="shubh"` (male, "Customer Care" style).

**Available Sarvam bulbul:v3 speakers:**
- **Male (14):** shubh, rahul, amit, ratan, rohan, dev, manan, sumit, aditya, kabir, varun, aayan, ashutosh, advait
- **Female (11):** ritu, priya, neha, pooja, simran, kavya, ishita, shreya, roopa, amelia, sophia

**Initial experiment variants:**
- **Control:** `shubh` (current default, male, "Customer Care")
- **Treatment A:** `ritu` (female, "Customer Care")
- **Treatment B:** `kabir` (male, "Content Creation")

**Statistical significance:** Minimum 30 calls per variant (90 total).

**Implementation:**

1. **New `experiment.py`** (new file):
   - `VoiceExperiment` dataclass: `name`, `variants: list[VoiceVariant]`, `created_at`
   - `VoiceVariant` dataclass: `speaker`, `gender`, `style`, `pace`, `loudness`
   - `ExperimentResult` dataclass: `variant`, `call_id`, `metrics` (price obtained, call duration, negotiation success, overall score)
   - `select_variant()` — random assignment with logging

2. **Parameterize voice in agent** — `agent_worker.py`:
   - In `entrypoint()` (~line 656), read voice config from room metadata
   - In TTS constructor (~line 716-726), use `metadata.get("speaker", "shubh")` instead of hardcoded value
   - Log voice variant in transcript metadata (~line 845-852)

3. **Dispatch voice config from pipeline** — `pipeline/session.py`:
   - In `start_call()` (~line 233-252), include `speaker` in agent dispatch metadata

4. **Voice comparison chart** — `dashboard.py`:
   - Add chart comparing metrics across voice variants
   - Group by speaker, show avg score, avg call duration, negotiation success rate

**Tests:**
- Unit test for `select_variant()` randomization
- Test that voice config flows through metadata to agent
- Test dashboard chart data aggregation

#### F5 — TTS Temperature / Pace Tuning

**Rationale:** Making the voice sound more natural improves shopkeeper engagement. Temperature controls variation in prosody.

**Current state:** Sarvam TTS temperature is NOT exposed in the LiveKit plugin. `SarvamTTSOptions` (in `livekit-plugins-sarvam`) has no `temperature` field. Only `pace` (0.5-2.0) is available.

**Blocked:** The LiveKit Sarvam plugin does not expose temperature, pitch (for v3), or loudness control beyond what's hardcoded. Options:
1. Fork/patch the LiveKit plugin to add temperature support
2. Use Sarvam REST API directly (loses streaming benefit)
3. Submit PR to `livekit-plugins-sarvam`

**Implementation (what's possible now):**

1. **Vary `pace` parameter** as part of F4 A/B experiment:
   - Add pace variants (0.9, 1.0, 1.1) to `VoiceVariant` in `experiment.py`
   - `agent_worker.py:716-726` — read `pace` from metadata

2. **Benchmark current TTS latency:**
   - Add TTS first-audio timing to `[LLM METRICS]` log in `agent_worker.py`
   - Track in dashboard

3. **Defer temperature** until confirming Sarvam API docs or plugin update.

**Tests:** Pace parameter flows through metadata to TTS constructor.

---

### Phase 2D: Product-Specific Testing {#phase-2d}

#### F8 — Product-Specific Test Cases + Scoring

**Rationale:** Current test suite is AC-centric (11 shopkeeper scenarios, all about ACs). The agent needs validation across product categories to ensure prompts generalize.

**Current state:**
- `tests/shopkeeper_scenarios.py` — 11 AC-focused scenarios
- `call_analysis.py:113-337` — `ConversationScorer` with 5 dimensions: constraint (40%), topic (25%), price_echo (15%), brevity (10%), repetition (10%)
- `call_analysis.py:28-107` — `ConstraintChecker` with 8 behavioral rules

**Implementation:**

**8.1 — Product-specific shopkeeper scenarios** (`tests/shopkeeper_scenarios.py`):

Restructure from flat `SCENARIOS` list to `PRODUCT_SCENARIOS` dict:
```python
PRODUCT_SCENARIOS = {
    "ac": [...],              # existing 11 + 3 new
    "washing_machine": [...], # 3 new scenarios
    "fridge": [...],          # 3 new scenarios
    "laptop": [...],          # 3 new scenarios
}
```

New scenarios (~12 total):
- **AC:** "Tonnage confused" (shopkeeper asks room size), "Window vs split debate", "Installation cost hidden"
- **Washing machine:** "Front load vs top load", "Capacity confusion" (kg vs family size), "Brand loyalty push"
- **Fridge:** "Single vs double door", "Convertible freezer upsell", "Energy rating lecture"
- **Laptop:** "Use case mismatch" (gaming vs work), "Extended warranty hard sell", "EMI vs cash discount"

**8.2 — New scoring dimensions** (`call_analysis.py`):

Add 3 new methods to `ConversationScorer`:
- `score_product_knowledge()` — did the agent demonstrate product knowledge? (mention model names, specs, competing products)
- `score_negotiation_effectiveness()` — did the agent negotiate? (reference competitor prices, mention multiple shops, push back on first price)
- `score_character_maintenance()` — did the agent stay in character throughout? (no English, no role reversal, consistent persona)

**Updated scoring weights:**
| Dimension | Old Weight | New Weight |
|-----------|-----------|------------|
| constraint | 40% | 30% |
| topic | 25% | 20% |
| price_echo | 15% | 10% |
| brevity | 10% | 5% |
| repetition | 10% | 5% |
| product_knowledge | — | 10% |
| negotiation | — | 10% |
| character | — | 10% |

New dimensions default to neutral (1.0) when `product_type` is not specified, preserving backward compatibility.

**8.3 — Product-specific live tests** (new `tests/test_product_scenarios_live.py`):
- Parametrized by product type
- Each test: build prompt for product → run scenario → score → assert thresholds
- ~16 live tests (4 products × 4 key scenarios each)

**8.4 — Offline product scoring tests** (`tests/test_scenario_offline.py`):
- `TestProductSpecificScoring` class — ~12 tests for new scoring dimensions
- Test product_knowledge scoring with/without model mentions
- Test negotiation scoring with/without competitor references
- Test character scoring with English leaks vs clean Hindi

**8.5 — Scenario runner update** (`tests/run_scenario_analysis.py`):
- Add `--product` flag to filter scenarios by product type
- Update summary output to show per-product breakdown

**Test count projection:**

| Test File | Current | Added | New Total |
|-----------|---------|-------|-----------|
| `test_scenario_offline.py` | 34 | ~12 | ~46 |
| `test_prompt_builder.py` | 22 | ~12 | ~34 |
| `shopkeeper_scenarios.py` | 11 scenarios | ~12 | ~23 scenarios |
| `test_product_scenarios_live.py` | 0 (new) | ~16 | ~16 |
| `test_scenario_live.py` | 20 | ~6 | ~26 |
| Others (F4, F6) | — | ~8 | ~8 |
| **Total** | **188 + 26 live** | **~66** | **~254 + 42 live** |

---

### Phase 2E: Intake and Pipeline {#phase-2e}

#### F11 — Intake Chatbot Improvements

**Rationale:** Current intake asks 5 generic questions (product, brand, budget, location, preferences). Product-specific clarifying questions would yield better research and more targeted calls.

**Current state:** `pipeline/intake.py:23-58` — `SYSTEM_PROMPT` with generic question flow. `IntakeAgent` class at line 63-108.

**Implementation:**

1. **Two-phase intake** in `pipeline/intake.py`:
   - **Phase 1 (2-3 turns):** Basic info — product type, budget range, location
   - **Phase 2 (1-2 turns):** Product-specific clarification:
     - AC → room size, floor, existing AC type
     - Washing machine → family size, front/top load preference, space constraints
     - Fridge → family size, single/double door, freezer usage
     - Laptop → primary use case (work/gaming/student), screen size preference

2. **Update `SYSTEM_PROMPT`** in `pipeline/intake.py:23-58`:
   - Add product-specific question trees
   - Vet existing questions for clarity and completeness
   - Add recommendation logic: based on answers, suggest specific specs (e.g., "For a 2BHK, 1.5 ton split AC is recommended")

3. **Recommendation output:** After Phase 2, intake returns `ProductRequirements` with inferred specs that feed into F13's research module.

**Tests:**
- Test product-specific question routing
- Test that recommendations match input patterns
- Test two-phase flow completion

#### F18 — Intake Suggestion Chips

**Rationale:** The intake chatbot currently relies entirely on free-text typing. Most users want common products (AC, fridge, washing machine) with common specs. Typing "1.5 ton split AC" is friction — especially on mobile. Clickable suggestion chips below each chatbot message let users tap through the ~80% common case in seconds, while still allowing free-text for the remaining 20%.

**Design principles:**
- **Chips are shortcuts, not constraints** — the text input always remains available. Chips accelerate; they never limit.
- **Chips are contextual** — they change based on what the chatbot just asked. Product type chips for the first message, tonnage chips after "AC" is selected, brand chips next, etc.
- **One tap = one message** — clicking a chip sends it as a user message immediately, identical to typing and pressing Send. The chatbot processes it normally.
- **Chips disappear after use** — once the user sends a message (via chip or typing), the current chip set is removed. The chatbot's next response may offer new chips.
- **LLM-driven chip generation** — the chatbot LLM outputs chip suggestions alongside its response, so chips are contextually aware and adapt to the conversation. A small set of default chips bootstraps the very first message before the LLM has responded.

**Current state:**
- `pipeline/intake.py:72-108` — `IntakeAgent.chat()` returns `{"response", "done", "requirements"}`. No suggestion data.
- `app.py:417-422` — intake chat UI has text input + Send button. No chip rendering.
- Chatbot asks one question at a time across 2-4 turns.

**Implementation:**

**18.1 — LLM-generated suggestions** (`pipeline/intake.py`):

Update `SYSTEM_PROMPT` to instruct the LLM to include a `<suggestions>` block in each response with 3-6 tappable chip labels relevant to the question just asked:

```
When you ask a question, also suggest 3-6 common answers as clickable chips.
Wrap them in <suggestions> tags, one per line:

<suggestions>
AC
Refrigerator
Washing Machine
Laptop
TV
</suggestions>

Guidelines for suggestions:
- Keep chip labels SHORT (1-4 words). E.g. "1.5 Ton", "Under ₹30K", "No preference".
- Offer the most common/popular options first.
- Always include a flexible option where appropriate ("No preference", "Not sure", "Skip").
- Adapt suggestions to the product type once known.
- Do NOT suggest chips for location — that requires free-text input.
- Do NOT suggest chips once you have enough info to extract requirements.
```

**18.2 — Parse suggestions from response** (`pipeline/intake.py`):

Add regex extraction (similar to `_REQUIREMENTS_RE`):
```python
_SUGGESTIONS_RE = re.compile(r"<suggestions>\s*(.*?)\s*</suggestions>", re.DOTALL)
```

Update `IntakeAgent.chat()` return value:
```python
return {
    "response": display_text,       # stripped of <suggestions> tags
    "done": self.done,
    "requirements": ...,
    "suggestions": suggestions,     # list[str] or empty list
}
```

**18.3 — Default chips for first message** (`app.py`):

Before the user types anything, show a welcome message with default product-type chips. These render immediately without waiting for an LLM call:

```python
DEFAULT_FIRST_CHIPS = ["AC", "Refrigerator", "Washing Machine", "Laptop", "TV", "Microwave"]
```

The frontend renders these under the initial "Hi! What product are you looking for?" welcome message. Once the user taps one or types their own message, these are replaced by LLM-generated chips.

**18.4 — Chip rendering in frontend** (`app.py`):

Add chip rendering below the chat messages area:

```html
<div class="suggestion-chips" id="intake-chips"></div>
```

CSS for chips:
```css
.suggestion-chips { display: flex; flex-wrap: wrap; gap: 0.5rem; padding: 0.5rem 0; }
.suggestion-chip {
    padding: 0.4rem 0.9rem; border-radius: 1rem; border: 1px solid var(--primary);
    color: var(--primary); background: transparent; cursor: pointer;
    font-size: 0.85rem; transition: all 0.15s;
}
.suggestion-chip:hover { background: var(--primary); color: white; }
```

JS behavior:
```javascript
function renderChips(suggestions) {
    const container = document.getElementById('intake-chips');
    container.innerHTML = '';
    suggestions.forEach(label => {
        const chip = document.createElement('button');
        chip.className = 'suggestion-chip';
        chip.textContent = label;
        chip.onclick = () => {
            document.getElementById('intake-input').value = label;
            sendIntake();
        };
        container.appendChild(chip);
    });
}
```

Update `sendIntake()`:
- Clear chips when user sends a message (chip click or manual typing)
- After receiving the chatbot response, call `renderChips(data.suggestions || [])`

**18.5 — Contextual chip examples by conversation phase:**

| Phase | Chatbot asks | Chip suggestions |
|-------|-------------|-----------------|
| Start (default) | "What are you looking to buy?" | AC, Refrigerator, Washing Machine, Laptop, TV, Microwave |
| Product spec (AC) | "What tonnage do you need?" | 1 Ton, 1.5 Ton, 2 Ton, Not sure |
| Product spec (AC) | "Split or window AC?" | Split AC, Window AC |
| Product spec (Fridge) | "Single or double door?" | Single Door, Double Door, Side-by-Side |
| Product spec (WM) | "Front load or top load?" | Front Load, Top Load, Not sure |
| Brand | "Any brand preference?" | No preference, Samsung, LG, Daikin, Voltas |
| Budget | "What's your budget range?" | Under ₹25K, ₹25-40K, ₹40-60K, ₹60K+, Flexible |
| Preferences | "Any specific features you want?" | Energy efficient, Quiet, WiFi-enabled, Not sure, Skip |

These are illustrative — the actual chips are generated by the LLM per-turn, so they naturally adapt to what the user has already said.

**18.6 — Edge cases:**
- **User types instead of tapping** — chips clear, LLM response generates fresh chips. No conflict.
- **User provides multiple details at once** ("I want a 1.5 ton Samsung split AC under 40K") — LLM recognizes this and skips ahead, generating chips only for remaining unknowns (e.g., location).
- **LLM doesn't output suggestions** — `suggestions` list is empty, no chips rendered. Graceful degradation.
- **Final confirmation message** — no chips. LLM is instructed to omit `<suggestions>` when extracting requirements.

**Tests:**
- `tests/test_intake.py` (new or extend existing):
  - `test_suggestions_parsed_from_response` — response with `<suggestions>` block returns list of strings
  - `test_suggestions_stripped_from_display` — display text has no `<suggestions>` tags
  - `test_no_suggestions_returns_empty_list` — response without block returns `[]`
  - `test_suggestions_not_generated_when_done` — final response with `<requirements>` has no chips
- Manual: tap through full flow (product → spec → brand → budget) using only chips, verify requirements extracted correctly

#### F14 — Subagent Modularity / Tweakability

**Rationale:** Current pipeline has configuration scattered across files, hardcoded model names, and coupled subagents. Tweaking one component (e.g., store scoring logic) requires reading multiple files.

**Current state:** Configuration is spread across `pipeline/research.py`, `pipeline/store_discovery.py`, `pipeline/analysis.py`, `pipeline/intake.py`, `agent_worker.py`.

**Implementation:**

1. **New `pipeline/config.py`** (new file):
   ```python
   @dataclass
   class PipelineConfig:
       # Research
       research_model: str = "claude-haiku-4-5-20250315"
       research_max_search_rounds: int = 3
       research_max_tokens: int = 4096
       # Store discovery
       store_max_results: int = 10
       store_auto_select_top_n: int = 4
       # Voice agent
       voice_model: str = "claude-haiku-4-5-20250315"
       voice_max_tokens: int = 150
       voice_temperature: float = 0.7
       voice_speaker: str = "shubh"
       voice_pace: float = 1.0
       # Analysis
       analysis_model: str = "claude-haiku-4-5-20250315"
       # Scoring weights
       scoring_weights: dict = field(default_factory=lambda: {
           "constraint": 0.30, "topic": 0.20, ...
       })
   ```

2. **Thread config through pipeline:**
   - `pipeline/session.py` — accept `PipelineConfig` in constructor, pass to subagents
   - Each subagent reads from config instead of hardcoded values
   - `agent_worker.py` — read voice config from metadata (set by session from config)

3. **Clear interfaces between subagents:**
   - Document input/output contracts for each pipeline phase
   - Type-check with existing dataclasses in `pipeline/schemas.py`

**Tests:**
- Test default config creates valid pipeline
- Test config overrides propagate to subagents
- Test that existing behavior unchanged with default config

---

### Phase 2F: Infrastructure {#phase-2f}

#### F7 — Logging Server on Homelab

**Rationale:** Logs and transcripts are saved to local filesystem. A centralized logging server enables cross-call analysis, trend tracking, and remote debugging.

**Current state:** `agent_worker.py` saves per-call logs to `logs/` and transcripts to `transcripts/`. `dashboard.py` reads from local files.

**Stack decision:** Requirements documented below. Specific stack (Loki+Grafana, ELK, custom) to be decided at implementation time.

**Requirements:**

1. **Structured JSON logging** — all log entries as JSON with fields: `timestamp`, `call_id`, `session_id`, `level`, `component`, `message`, `metadata`
2. **Transcript upload** — after each call, POST transcript JSON to logging server
3. **Log ingestion API** — REST endpoint accepting structured log entries and transcript uploads
4. **Search and query** — ability to search logs by call_id, time range, component, error level
5. **Dashboard integration** — `dashboard.py` can optionally read from remote server instead of local files
6. **Local fallback** — if logging server is unreachable, continue writing to local filesystem

**Implementation sketch:**

1. **Structured logging** in `agent_worker.py`:
   - Replace `logger.info(f"[TAG] message")` with structured JSON: `logger.info(json.dumps({"tag": "LLM_METRICS", "ttft": 0.6, ...}))`
   - After `_save_transcript()` (~line 836-858), POST to `LOG_SERVER_URL` env var (if set)

2. **Log collector service** — new `log_collector.py`:
   - REST API for log/transcript ingestion
   - Storage backend (filesystem initially, swappable to chosen stack)

3. **K8s deployment** — new `k8s/logging/` directory with manifests for chosen stack

4. **Dashboard update** — `dashboard.py`: add `--remote` flag to read from log server API instead of local files

**Tests:**
- Test structured log format
- Test log collector API endpoints
- Test local fallback when server unreachable

#### F1 — Sub-1.5s → Sub-1s Voice Latency

**Rationale:** Current mouth-to-ear latency is ~2.3s. For natural conversation, target ≤1.5s first (achievable with VAD/endpointing tuning), then ≤1s (requires speculative techniques).

**Current latency breakdown:**

| Component | Current | Target (Phase A) | Target (Phase B) | Files |
|-----------|---------|-------------------|-------------------|-------|
| VAD silence detection | 800ms | 500ms | 400ms | `agent_worker.py:700-704` |
| Endpointing delay | 500ms | 300ms | 200ms | `agent_worker.py:727-732` |
| STT | ~200ms | ~200ms | ~200ms | Streaming, limited optimization |
| LLM TTFT | ~600ms (cached) | ~500ms | ~300ms | `agent_worker.py:462-484` |
| TTS first audio | ~200ms | ~200ms | ~150ms | `agent_worker.py:716-726` |
| **Total** | **~2300ms** | **~1700ms** | **~1250ms** | |

**Phase A — Target ≤1.5s (VAD/endpointing tuning):**

1. **Reduce VAD silence duration** — `agent_worker.py:700-704`:
   - Change `min_silence_duration` from 0.8s to 0.5s
   - Risk: false turn endings. Mitigate with `min_speech_duration` bump (0.08 → 0.15s)

2. **Reduce endpointing delay** — `agent_worker.py:727-732`:
   - Change `min_endpointing_delay` from 0.5s to 0.3s
   - Risk: interrupting multi-sentence turns. Mitigate with `min_interruption_words` bump (2 → 3)

3. **Measure carefully** — A/B test old vs new VAD settings. Track:
   - False turn endings (agent responds mid-sentence)
   - Successful turn completions
   - Overall call quality scores

**Phase B — Target ≤1s (speculative techniques, future):**

1. **Speculative filler audio** — while LLM processes, play natural fillers ("Hmm", "Achha", "Haan ji") to mask latency
2. **Lower `max_tokens`** — from 150 to 100 (agent responses should be short)
3. **Prefill/speculative generation** — start generating likely responses before turn is fully detected
4. **TTS latency benchmarking** — measure Sarvam bulbul:v3 first-audio latency, explore lower sample rates

**Tests:**
- Measure TTFT p50/p90 before and after each change
- Track false turn ending rate in live calls
- A/B comparison of call quality scores

---

### Phase 2G: Non-Code {#phase-2g}

#### F9 — Blog for CallKaro

**Rationale:** Technical blog post to showcase the project, document learnings, and attract interest.

**Content outline:**
1. **Problem statement** — why phone-based price comparison matters in India
2. **Architecture deep-dive** — pipeline diagram, voice agent stack (LiveKit + Sarvam + Claude)
3. **Technical challenges** — number splitting bug, character breaks, STT garbage, latency optimization
4. **Sarvam integration** — STT/TTS in Hindi, monkey-patching the LiveKit plugin
5. **Testing strategy** — shopkeeper scenario simulation, constraint checking, conversation scoring
6. **Results** — sample call transcripts, quality scores, latency improvements
7. **What's next** — Phase 2 roadmap

**Diagrams to create:**
- Pipeline flow (intake → research → store discovery → calling → analysis)
- Voice latency breakdown waterfall chart
- Scoring dimensions radar chart

**Reference:** Study top Medium technical blogs for structure, use of diagrams, and storytelling.

#### F12 — Market Analysis for Other Use Cases

**Rationale:** Explore how the voice price comparison agent can adapt to other product categories beyond appliances.

**Assessment criteria for each category:**
1. Is phone-based price inquiry common for this category in India?
2. Is price comparison valuable (high variance between sellers)?
3. Can the conversation be kept short (under 3 minutes)?

**Categories to evaluate:**
- **Second-hand cameras** — pricing is condition-dependent, requires detailed questions
- **Used iPods/electronics** — high price variance, condition assessment needed
- **Furniture** — availability-focused more than price, local delivery matters
- **Grocery/produce** — high volume, low per-item price, frequent purchases
- **Services** (plumber, electrician) — rate comparison, availability check
- **Jewelry/gold** — daily rate + making charges, high-value negotiation
- **Auto parts** — compatibility verification + price comparison

**Deliverable:** Research document with opportunity assessment matrix, recommended next category, and required pipeline modifications per category.

---

### Suggested Implementation Order

1. **Week 1:** F15 + F10 (quick wins) → F13 (research module — foundational)
2. **Week 2:** F2 (knowledge transfer) → F6 (auto store selection)
3. **Week 3:** F3 (prompt restructure + negotiation) → F8 start (test infra + scenarios)
4. **Week 4:** F8 finish (scoring + live tests) → F4 (voice A/B)
5. **Week 5:** F11 (intake improvements) → F18 (suggestion chips) → F14 (modularity)
6. **Week 6+:** F7 (logging server) → F1 (latency tuning) → F5 (TTS, if feasible) → F9, F12 (non-code)

---

### Risks and Tradeoffs

| Risk | Impact | Mitigation |
|------|--------|------------|
| F1: Aggressive VAD cuts off speaker mid-sentence | High — broken conversations | Start conservative (600ms silence, 300ms endpointing). A/B test before going lower. Filler audio masks remaining latency. |
| F2/F3: Larger prompts increase TTFT | Medium — higher latency | Prompt caching enabled. New sections conditional. Monitor token count and TTFT after changes. |
| F3: Negotiation tactics may offend shopkeepers | Medium — bad UX | Keep tone gentle ("Main 2-3 shops se rate le raha hoon"). Test with live calls before full deployment. |
| F4: Voice changes may confuse shopkeepers | Low — most won't notice | Start with 2 voices, 30+ calls per variant before conclusions. |
| F5: TTS temperature not exposed in LiveKit plugin | High — blocks feature | Confirm Sarvam API docs first. Focus on `pace` parameter as alternative. |
| F8: New scoring dimensions change existing scores | Medium — test failures | New dimensions additive (default neutral). Update weights gradually. |
| F7: Logging server adds operational complexity | Medium — homelab maintenance | Keep local fallback. Start with simple collector, migrate to full stack later. |
| F14: Modularity refactor may break pipeline | High — regression | F8 tests must pass first as safety net. Refactor incrementally. |
| F18: LLM may generate poor/irrelevant chip labels | Low — cosmetic | Chips are optional shortcuts; text input always available. LLM prompt has explicit examples. Default chips for first turn are hardcoded. |
| F18: Extra `<suggestions>` tokens add latency | Low — ~20 tokens | Negligible vs response text. Chips omitted on final turn. |
