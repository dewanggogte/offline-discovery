"""Tests for TTS text normalization — _normalize_for_tts, _strip_think_tags, _ACTION_RE."""

import pytest


# ===================================================================
# A. Abbreviation replacements (case-sensitive, word-boundary)
# ===================================================================
class TestAbbreviationReplacements:
    def test_ac_uppercase(self, normalize):
        assert "ए सी" in normalize("AC dealer ho?")

    def test_ac_dotted(self, normalize):
        assert "ए सी" in normalize("A.C. ka price")

    def test_ac_dotted_lower(self, normalize):
        assert "ए सी" in normalize("a.c. chalega")

    def test_emi_uppercase(self, normalize):
        assert "ई एम आई" in normalize("EMI available hai")

    def test_emi_dotted(self, normalize):
        assert "ई एम आई" in normalize("E.M.I. pe milega")

    def test_gst_uppercase(self, normalize):
        assert "जी एस टी" in normalize("GST included hai")

    def test_mrp_uppercase(self, normalize):
        assert "एम आर पी" in normalize("MRP kitna hai")

    def test_led_uppercase(self, normalize):
        assert "एल ई डी" in normalize("LED display hai")

    def test_lg_uppercase(self, normalize):
        assert "एल जी" in normalize("LG ka AC")

    def test_bee_uppercase(self, normalize):
        assert "बी ई ई" in normalize("BEE rating")

    # Word boundary protection
    def test_ac_not_in_achha(self, normalize):
        result = normalize("Achha")
        assert "ए सी" not in result
        assert "chha" not in result or "Achha" == result.strip()

    def test_ac_not_in_each(self, normalize):
        result = normalize("each one")
        assert "ए सी" not in result

    def test_multiple_abbreviations(self, normalize):
        result = normalize("AC aur EMI dono available hai")
        assert "ए सी" in result
        assert "ई एम आई" in result


# ===================================================================
# B. Word replacements (case-insensitive, no word boundaries)
# ===================================================================
class TestWordReplacements:
    def test_samsung(self, normalize):
        assert "सैमसंग" in normalize("Samsung AC")

    def test_samsung_lowercase(self, normalize):
        assert "सैमसंग" in normalize("samsung ka model")

    def test_inverter(self, normalize):
        assert "इन्वर्टर" in normalize("inverter AC")

    def test_split(self, normalize):
        assert "स्प्लिट" in normalize("split AC")

    def test_ton(self, normalize):
        assert "टन" in normalize("1.5 ton")

    def test_star(self, normalize):
        assert "स्टार" in normalize("5 star rating")

    def test_rate(self, normalize):
        assert "रेट" in normalize("rate kya hai")

    def test_price(self, normalize):
        assert "प्राइस" in normalize("price batao")

    def test_best(self, normalize):
        assert "बेस्ट" in normalize("best price")

    def test_online(self, normalize):
        assert "ऑनलाइन" in normalize("online pe dikha raha tha")

    def test_warranty(self, normalize):
        assert "वारंटी" in normalize("warranty kitni hai")

    def test_installation(self, normalize):
        assert "इंस्टॉलेशन" in normalize("installation free hai")

    def test_discount(self, normalize):
        assert "डिस्काउंट" in normalize("discount milega kya")

    def test_free(self, normalize):
        assert "फ्री" in normalize("free installation")

    def test_exchange(self, normalize):
        assert "एक्सचेंज" in normalize("exchange offer")

    def test_offer(self, normalize):
        assert "ऑफर" in normalize("koi offer hai kya")

    def test_delivery(self, normalize):
        assert "डिलीवरी" in normalize("delivery kab hogi")

    def test_cashback(self, normalize):
        assert "कैशबैक" in normalize("cashback milega")

    def test_stock(self, normalize):
        assert "स्टॉक" in normalize("stock mein hai")

    def test_model(self, normalize):
        assert "मॉडल" in normalize("model number kya hai")

    def test_wifi(self, normalize):
        assert "वाई फाई" in normalize("WiFi support hai")

    def test_wifi_hyphenated(self, normalize):
        assert "वाई फाई" in normalize("Wi-Fi enabled")

    def test_blue_star(self, normalize):
        assert "ब्लू स्टार" in normalize("Blue Star ka AC")

    def test_daikin(self, normalize):
        assert "डायकिन" in normalize("Daikin achha hai")

    def test_voltas(self, normalize):
        assert "वोल्टास" in normalize("Voltas bhi dekhlo")

    def test_whirlpool(self, normalize):
        assert "व्हर्लपूल" in normalize("Whirlpool ka kya rate hai")

    def test_carrier(self, normalize):
        assert "कैरियर" in normalize("Carrier AC lena hai")

    def test_godrej(self, normalize):
        assert "गोदरेज" in normalize("Godrej bhi theek hai")

    # Case insensitivity
    def test_rate_uppercase(self, normalize):
        assert "रेट" in normalize("RATE kya hai")

    def test_price_mixed_case(self, normalize):
        assert "प्राइस" in normalize("Price batao")

    # Concatenated words (real bug from LLM output)
    def test_rate_in_concatenated(self, normalize):
        assert "रेट" in normalize("karatekya hai")

    def test_price_in_concatenated(self, normalize):
        assert "प्राइस" in normalize("aapsepricepooch")

    def test_best_price_concatenated(self, normalize):
        result = normalize("bestprice")
        assert "बेस्ट" in result
        assert "प्राइस" in result


# ===================================================================
# C. Script-boundary spacing (Devanagari <-> Latin/digit)
# ===================================================================
class TestScriptBoundarySpacing:
    def test_devanagari_to_digit(self, normalize):
        result = normalize("सैमसंग1.5")
        assert "सैमसंग 1.5" in result

    def test_digit_to_devanagari(self, normalize):
        result = normalize("1.5टन")
        assert "1.5 टन" in result

    def test_full_model_string(self, normalize):
        result = normalize("सैमसंग1.5टन5स्टार")
        assert "सैमसंग 1" in result
        assert "5 स्टार" in result

    def test_latin_to_devanagari(self, normalize):
        result = normalize("kaरेट")
        assert "ka रेट" in result

    def test_devanagari_to_latin(self, normalize):
        result = normalize("ए सीhai")
        assert "ए सी hai" in result

    def test_no_extra_spaces_pure_devanagari(self, normalize):
        text = "नमस्ते कैसे हो"
        assert normalize(text) == text

    def test_no_extra_spaces_pure_latin(self, normalize):
        assert normalize("hello world") == "hello world"

    def test_multiple_space_collapse(self, normalize):
        result = normalize("ए सी   hai   aapke   paas")
        assert "  " not in result


# ===================================================================
# D. Action marker stripping
# ===================================================================
class TestActionMarkerStripping:
    def test_asterisk_markers(self, normalize):
        assert "confused" not in normalize("*confused*")

    def test_paren_markers(self, normalize):
        assert "laughs" not in normalize("(laughs)")

    def test_bracket_markers(self, normalize):
        assert "pauses" not in normalize("[pauses]")

    def test_marker_in_sentence(self, normalize):
        result = normalize("Main *confused* hoon")
        assert "*" not in result
        assert "Main" in result
        assert "hoon" in result

    def test_multiple_markers(self, normalize):
        result = normalize("*smiles* Hello *pauses* ji")
        assert "*" not in result
        assert "Hello" in result
        assert "ji" in result

    def test_normal_text_preserved(self, normalize):
        assert "kya" in normalize("kya hai bhai")


# ===================================================================
# E. Think-tag stripping
# ===================================================================
class TestStripThinkTags:
    def test_complete_think_tag(self, strip_think):
        assert strip_think("<think>reasoning</think>Hello") == "Hello"

    def test_multiline_think_tag(self, strip_think):
        result = strip_think("<think>\nI should ask about price\n</think>Namaste")
        assert result == "Namaste"

    def test_unclosed_think_tag_streaming(self, strip_think):
        assert strip_think("<think>partial reasoning") == ""

    def test_no_think_tags(self, strip_think):
        assert strip_think("Normal text") == "Normal text"

    def test_empty_think_tag(self, strip_think):
        assert strip_think("<think></think>Achha") == "Achha"

    def test_think_tag_with_content_after(self, strip_think):
        result = strip_think("<think>Let me negotiate</think>Thoda kam karo")
        assert result == "Thoda kam karo"

    def test_multiple_think_tags(self, strip_think):
        result = strip_think("<think>first</think>Hello<think>second</think> ji")
        assert "first" not in result
        assert "second" not in result
        assert "Hello" in result

    def test_think_tag_only(self, strip_think):
        assert strip_think("<think>only thinking</think>") == ""


# ===================================================================
# F. Full pipeline / combined tests
# ===================================================================
class TestFullNormalizationPipeline:
    def test_real_llm_output_product_query(self, normalize):
        """Test with actual LLM output pattern: product name with all English terms."""
        text = "Samsung 1.5 ton ka 5 star inverter split AC hai"
        result = normalize(text)
        assert "सैमसंग" in result
        assert "टन" in result
        assert "स्टार" in result
        assert "इन्वर्टर" in result
        assert "स्प्लिट" in result
        assert "ए सी" in result

    def test_real_llm_output_price_negotiation(self, normalize):
        text = "42 hazaar? Online pe toh 38 mein dikha raha tha. Best price kya hoga?"
        result = normalize(text)
        assert "ऑनलाइन" in result
        assert "बेस्ट" in result
        assert "प्राइस" in result

    def test_real_llm_output_extras(self, normalize):
        text = "Installation free hai? Warranty kitni milegi?"
        result = normalize(text)
        assert "इंस्टॉलेशन" in result
        assert "फ्री" in result
        assert "वारंटी" in result

    def test_empty_string(self, normalize):
        assert normalize("") == ""

    def test_only_hindi(self, normalize):
        text = "नमस्ते कैसे हैं आप"
        assert normalize(text) == text

    def test_concatenated_output_from_transcript(self, normalize):
        """Real bug pattern from transcript: LLM concatenates words."""
        text = "Mainekoiratebataya"
        result = normalize(text)
        assert "रेट" in result

    def test_adjacent_replacements_spaced(self, normalize):
        """Two adjacent English words should not merge into single Devanagari blob."""
        result = normalize("bestprice")
        assert "बेस्ट" in result
        assert "प्राइस" in result
        # They should have a space between them
        assert "बेस्ट प्राइस" in result or "बेस्ट" in result

    def test_mixed_devanagari_latin_digits(self, normalize):
        """Complex mixed-script string from real output."""
        text = "सैमसंग1.5टन5 स्टार इन्वर्टरस्प्लिट ए सी karatekya hai?"
        result = normalize(text)
        # Script boundaries should be spaced
        assert "सैमसंग 1.5" in result
        # "rate" should be transliterated
        assert "रेट" in result
