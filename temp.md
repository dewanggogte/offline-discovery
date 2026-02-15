# Issues to Fix

## Resolved

### ~~1. Greeting spoken twice / sanitizer stripping context~~
Fixed: Greeting sent with `add_to_chat_ctx=False`. LLM knows about it via NOTE in system instructions. Sanitizer no longer strips it.

### ~~2. Store name wrong in LLM-generated greeting~~
Fixed: "Sharma Electronics" removed from EXAMPLES section. Greeting text dynamically uses actual store name.

### ~~3. Incomplete/cut-off responses in transcript~~
Fixed: Interrupted messages now marked with `[interrupted]` flag in transcript JSON. Agent annotates truncated context with `[interrupted]` suffix for LLM awareness.

### ~~4. Math error / wrong Hindi number words~~
Fixed: LLM no longer writes Hindi number words. Prompt instructs LLM to write ALL numbers as digits (e.g. `39000`, `1.5 ton`, `2 saal`). The `_replace_numbers()` pipeline deterministically converts digits to correct Hindi words. Added "saadhe" pattern for half-thousands (37500 → "saadhe saintees hazaar").

### ~~5. `\n\n` in responses causing unnatural pauses~~
Fixed: `_normalize_for_tts()` now replaces all `\n` characters with spaces.

### ~~6. Hardcoded "Sharma Electronics" in prompt examples~~
Fixed: Examples section uses generic multi-turn format without specific store names.

### ~~7. Devanagari characters leaking through~~
Fixed: `_transliterate_devanagari()` safety net converts any leaked Devanagari to Romanized Hindi via static lookup table. Handles consonant+matra combinations correctly (e.g. `usका` → `uskaa`).

### ~~8. Duplicate goodbye in transcript~~
Fixed: Removed transcript append from `end_call` — `conversation_item_added` handler reliably captures it. Previous dedup check failed due to race condition.

### ~~9. Agent says "paas mein hi rehta hoon" when asked where they live~~
Fixed: `stores.json` now has `nearby_area` field per store. Prompt says to use the specific area name. Agent says "Koramangala mein rehta hoon" instead of being evasive.

### ~~10. Agent confused customer/shopkeeper roles with English-speaking shopkeeper~~
Fixed: Prompt now explicitly says "You are the CUSTOMER... NEVER confirm stock availability" and handles shopkeeper speaking English.

## Open

(No open issues currently.)
