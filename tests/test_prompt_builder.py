"""Tests for pipeline/prompt_builder.py — casual names, greeting, research sections."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline.schemas import ProductRequirements, ResearchOutput, DiscoveredStore
from pipeline.prompt_builder import (
    _casual_product_name,
    build_greeting,
    build_prompt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _req(product_type="AC", category="", **kwargs):
    return ProductRequirements(
        product_type=product_type, category=category, location="Bangalore", **kwargs
    )


def _store(name="Croma", area="Koramangala"):
    return DiscoveredStore(name=name, area=area)


def _research(**overrides):
    defaults = dict(
        product_summary="",
        market_price_range=None,
        questions_to_ask=["Price kya hai?"],
        topics_to_cover=["price", "warranty", "delivery"],
        topic_keywords={},
        important_notes=[],
        competing_products=[],
    )
    defaults.update(overrides)
    return ResearchOutput(**defaults)


# ---------------------------------------------------------------------------
# TestCasualProductName
# ---------------------------------------------------------------------------
class TestCasualProductName:
    def test_strips_parenthetical(self):
        req = _req(category="double door fridge (220-280L)")
        assert _casual_product_name(req) == "double door fridge"

    def test_strips_size_adjective(self):
        req = _req(category="Medium double door fridge")
        assert _casual_product_name(req) == "double door fridge"

    def test_preserves_tonnage(self):
        req = _req(category="1.5 ton split AC")
        assert _casual_product_name(req) == "1.5 ton split AC"

    def test_strips_with_clause(self):
        req = _req(category="double door fridge with separate freezer section (220-280L)")
        assert _casual_product_name(req) == "double door fridge"

    def test_full_verbose_category(self):
        req = _req(category="Medium double door fridge with separate freezer section (220-280L)")
        assert _casual_product_name(req) == "double door fridge"

    def test_fallback_to_product_type(self):
        req = _req(product_type="AC", category="AC")
        assert _casual_product_name(req) == "AC"

    def test_short_category_unchanged(self):
        req = _req(category="split AC")
        assert _casual_product_name(req) == "split AC"

    def test_empty_category_uses_product_type(self):
        req = _req(product_type="washing machine", category="")
        assert _casual_product_name(req) == "washing machine"


# ---------------------------------------------------------------------------
# TestBuildGreeting
# ---------------------------------------------------------------------------
class TestBuildGreeting:
    def test_uses_casual_name_not_verbose(self):
        req = _req(category="Medium double door fridge with separate freezer section (220-280L)")
        greeting = build_greeting(req, _store("Reliance Digital"))
        assert "double door fridge" in greeting
        assert "(220-280L)" not in greeting
        assert "Medium" not in greeting

    def test_contains_store_name(self):
        greeting = build_greeting(_req(category="1.5 ton split AC"), _store("Croma"))
        assert "Croma" in greeting

    def test_greeting_format(self):
        greeting = build_greeting(_req(category="split AC"), _store("Vijay Sales"))
        assert greeting.startswith("Hello, yeh Vijay Sales hai?")
        assert "ke baare mein poochna tha." in greeting


# ---------------------------------------------------------------------------
# TestBuildPromptWithResearch
# ---------------------------------------------------------------------------
class TestBuildPromptWithResearch:
    def test_product_knowledge_present(self):
        research = _research(
            product_summary="Samsung and LG dominate the split AC market.",
            competing_products=[
                {"name": "Samsung AR18CY5ARWK", "price_range": "35000-40000", "pros": "energy efficient"},
                {"name": "LG PS-Q19YNZE", "price_range": "38000-42000", "pros": "low noise"},
                {"name": "Daikin MTKL50U", "price_range": "40000-45000", "pros": "best cooling"},
            ],
        )
        prompt = build_prompt(_req(category="1.5 ton split AC"), research, _store())
        assert "PRODUCT KNOWLEDGE:" in prompt
        assert "Samsung AR18CY5ARWK" in prompt
        assert "LG PS-Q19YNZE" in prompt
        assert "which model?" in prompt.lower()

    def test_buyer_notes_present(self):
        research = _research(
            important_notes=[
                "Check copper condenser vs aluminium",
                "Installation charges vary 1500-3000",
                "Warranty registration required within 30 days",
            ],
        )
        prompt = build_prompt(_req(category="split AC"), research, _store())
        assert "BUYER NOTES:" in prompt
        assert "copper condenser" in prompt

    def test_when_stuck_present(self):
        research = _research(
            product_summary="Popular segment.",
            competing_products=[
                {"name": "Samsung AR18CY5ARWK", "price_range": "35000-40000", "pros": "good"},
            ],
            market_price_range=(35000, 45000),
        )
        prompt = build_prompt(_req(category="1.5 ton split AC"), research, _store())
        assert "WHEN STUCK:" in prompt
        assert "Samsung AR18CY5ARWK" in prompt
        assert "35000" in prompt  # anchor to low end

    def test_empty_research_no_crash(self):
        research = _research()
        prompt = build_prompt(_req(category="split AC"), research, _store())
        assert "PRODUCT KNOWLEDGE:" not in prompt
        assert "BUYER NOTES:" not in prompt
        assert "WHEN STUCK:" not in prompt
        assert "PRODUCT:" in prompt  # reference field still present

    def test_competing_products_capped_at_5(self):
        research = _research(
            competing_products=[
                {"name": f"Model{i}", "price_range": "10000", "pros": "ok"}
                for i in range(8)
            ],
        )
        prompt = build_prompt(_req(category="AC"), research, _store())
        assert "Model0" in prompt
        assert "Model4" in prompt
        assert "Model5" not in prompt

    def test_important_notes_capped_at_6(self):
        research = _research(
            important_notes=[f"Note {i}" for i in range(9)],
        )
        prompt = build_prompt(_req(category="AC"), research, _store())
        assert "Note 0" in prompt
        assert "Note 5" in prompt
        assert "Note 6" not in prompt

    def test_casual_name_in_spoken_sections(self):
        """Spoken sections should use casual name, not verbose category."""
        research = _research(product_summary="Popular segment.")
        req = _req(category="Medium double door fridge with separate freezer section (220-280L)")
        prompt = build_prompt(req, research, _store())

        # Split at PRODUCT: line — everything before is spoken sections
        product_line_idx = prompt.index("PRODUCT:")
        spoken = prompt[:product_line_idx]

        assert "(220-280L)" not in spoken
        assert "Medium double door fridge with" not in spoken
        assert "double door fridge" in spoken

    def test_verbose_desc_in_product_reference(self):
        """PRODUCT: line should keep the full verbose category."""
        research = _research()
        req = _req(category="Medium double door fridge with separate freezer section (220-280L)")
        prompt = build_prompt(req, research, _store())
        # Find the PRODUCT: line
        for line in prompt.split("\n"):
            if line.startswith("PRODUCT:"):
                assert "Medium double door fridge with separate freezer section (220-280L)" in line
                break
        else:
            pytest.fail("PRODUCT: line not found")

    def test_model_recovery_in_examples(self):
        """EXAMPLES should include a 'which model?' recovery when competing_products exist."""
        research = _research(
            competing_products=[
                {"name": "Samsung AR18CY5ARWK", "price_range": "35000-40000", "pros": "good"},
            ],
        )
        prompt = build_prompt(_req(category="1.5 ton split AC"), research, _store())
        assert "Kaun sa model chahiye?" in prompt
        assert "Samsung AR18CY5ARWK ka kya price hai?" in prompt

    def test_greeting_note_present(self):
        """Prompt should include a NOTE about the greeting to prevent LLM from repeating it."""
        prompt = build_prompt(_req(category="split AC"), _research(), _store("Croma"))
        assert "NOTE: You have already greeted" in prompt
        assert "Do NOT repeat the greeting" in prompt
        assert "Croma" in prompt.split("NOTE")[1]  # greeting includes store name

    def test_greeting_note_uses_casual_name(self):
        """Greeting NOTE should use casual product name, not verbose category."""
        prompt = build_prompt(
            _req(category="Medium double door fridge with separate freezer section (220-280L)"),
            _research(), _store("Reliance"),
        )
        # The greeting note should contain the casual name, not the verbose one
        note_section = prompt.split("NOTE:")[1]
        assert "double door fridge" in note_section
        assert "(220-280L)" not in note_section


# ---------------------------------------------------------------------------
# TestResearchIntelligenceSections (F2)
# ---------------------------------------------------------------------------
class TestResearchIntelligenceSections:
    """Test new research sections: recommended products, negotiation intelligence, insider knowledge."""

    def test_recommended_products_present(self):
        research = _research(
            recommended_products=[
                {"model": "Samsung AR18CY5ARWK", "street_price": 36000, "specs": "1.5T 5-star inverter"},
                {"model": "LG PS-Q19YNZE", "street_price": 39000, "specs": "1.5T AI dual inverter"},
            ],
        )
        prompt = build_prompt(_req(category="1.5 ton split AC"), research, _store())
        assert "RECOMMENDED PRODUCTS:" in prompt
        assert "Samsung AR18CY5ARWK" in prompt
        assert "36000" in prompt
        assert "online dekha" in prompt.lower()

    def test_recommended_products_absent_when_empty(self):
        research = _research(recommended_products=[])
        prompt = build_prompt(_req(category="split AC"), research, _store())
        assert "RECOMMENDED PRODUCTS:" not in prompt

    def test_recommended_products_capped_at_3(self):
        research = _research(
            recommended_products=[
                {"model": f"Model{i}", "street_price": 10000 + i, "specs": "ok"}
                for i in range(5)
            ],
        )
        prompt = build_prompt(_req(category="AC"), research, _store())
        assert "Model0" in prompt
        assert "Model2" in prompt
        assert "Model3" not in prompt

    def test_negotiation_intelligence_present(self):
        research = _research(
            negotiation_intelligence={
                "typical_margin": "8-12% on split ACs",
                "seasonal_notes": "Summer end-of-season discounts in September",
                "bundle_tricks": "Dealers push expensive stabilizers",
                "online_reference": "Available for 36000 on Amazon",
            },
        )
        prompt = build_prompt(_req(category="1.5 ton split AC"), research, _store())
        assert "NEGOTIATION INTELLIGENCE:" in prompt
        assert "8-12%" in prompt
        assert "September" in prompt
        assert "stabilizer" in prompt.lower()
        assert "Amazon" in prompt

    def test_negotiation_intelligence_absent_when_empty(self):
        research = _research(negotiation_intelligence={})
        prompt = build_prompt(_req(category="split AC"), research, _store())
        assert "NEGOTIATION INTELLIGENCE:" not in prompt

    def test_negotiation_intelligence_partial(self):
        """Only present fields should appear."""
        research = _research(
            negotiation_intelligence={"typical_margin": "10% margin"},
        )
        prompt = build_prompt(_req(category="split AC"), research, _store())
        assert "NEGOTIATION INTELLIGENCE:" in prompt
        assert "10% margin" in prompt
        assert "Seasonal" not in prompt

    def test_insider_knowledge_present(self):
        research = _research(
            insider_knowledge=[
                "Samsung 2025 models had compressor issues",
                "LG launched new models in March — old stock may be discounted",
            ],
        )
        prompt = build_prompt(_req(category="split AC"), research, _store())
        assert "INSIDER KNOWLEDGE:" in prompt
        assert "compressor issues" in prompt
        assert "strategically" in prompt.lower()

    def test_insider_knowledge_absent_when_empty(self):
        research = _research(insider_knowledge=[])
        prompt = build_prompt(_req(category="split AC"), research, _store())
        assert "INSIDER KNOWLEDGE:" not in prompt

    def test_insider_knowledge_capped_at_3(self):
        research = _research(
            insider_knowledge=[f"Tip {i}" for i in range(5)],
        )
        prompt = build_prompt(_req(category="AC"), research, _store())
        assert "Tip 0" in prompt
        assert "Tip 2" in prompt
        assert "Tip 3" not in prompt

    def test_when_stuck_prefers_recommended_model(self):
        """WHEN STUCK should prefer recommended_products over competing_products for model name."""
        research = _research(
            recommended_products=[{"model": "Samsung AR18CY5ARWK", "street_price": 36000, "specs": "good"}],
            competing_products=[{"name": "LG PS-Q19YNZE", "price_range": "38000-42000", "pros": "quiet"}],
        )
        prompt = build_prompt(_req(category="1.5 ton split AC"), research, _store())
        assert "WHEN STUCK:" in prompt
        # Should prefer recommended model over competing
        stuck_section = prompt.split("WHEN STUCK:")[1].split("\n\n")[0]
        assert "Samsung AR18CY5ARWK" in stuck_section


# ---------------------------------------------------------------------------
# TestDynamicConversationFlow (F3)
# ---------------------------------------------------------------------------
class TestDynamicConversationFlow:
    """Test product-aware and store-aware conversation flow."""

    def test_ac_product_flow(self):
        prompt = build_prompt(
            _req(product_type="AC", category="1.5 ton split AC"),
            _research(), _store("Local AC Shop"),
        )
        assert "CONVERSATION FLOW:" in prompt
        assert "tonnage" in prompt.lower()

    def test_washing_machine_product_flow(self):
        prompt = build_prompt(
            _req(product_type="washing machine", category="front load 7kg washing machine"),
            _research(), _store("Appliance World"),
        )
        assert "CONVERSATION FLOW:" in prompt
        assert "capacity" in prompt.lower() or "front/top load" in prompt.lower()

    def test_fridge_product_flow(self):
        prompt = build_prompt(
            _req(product_type="fridge", category="double door fridge"),
            _research(), _store("Croma"),
        )
        assert "CONVERSATION FLOW:" in prompt
        assert "single or double door" in prompt.lower()

    def test_laptop_product_flow(self):
        prompt = build_prompt(
            _req(product_type="laptop", category="gaming laptop"),
            _research(), _store("CompuMart"),
        )
        assert "CONVERSATION FLOW:" in prompt
        assert "use case" in prompt.lower()

    def test_chain_store_approach(self):
        """Chain stores should get offer/combo approach."""
        prompt = build_prompt(
            _req(category="split AC"), _research(), _store("Croma"),
        )
        assert "chain store" in prompt.lower()
        assert "offers" in prompt.lower() or "combos" in prompt.lower()

    def test_local_dealer_approach(self):
        """Local dealers should get direct negotiation approach."""
        prompt = build_prompt(
            _req(category="split AC"), _research(), _store("Sharma Electronics"),
        )
        assert "local dealer" in prompt.lower()
        assert "negotiate" in prompt.lower() or "2-3 shops" in prompt.lower()


class TestNegotiationSection:
    """Test negotiation section in prompts."""

    def test_negotiation_present(self):
        prompt = build_prompt(_req(category="split AC"), _research(), _store())
        assert "NEGOTIATION:" in prompt

    def test_negotiation_references_market_price(self):
        research = _research(market_price_range=(35000, 42000))
        prompt = build_prompt(_req(category="split AC"), research, _store())
        assert "35000" in prompt
        assert "online" in prompt.lower()

    def test_negotiation_mentions_comparison_shopping(self):
        prompt = build_prompt(_req(category="split AC"), _research(), _store())
        assert "2-3 shops" in prompt

    def test_negotiation_uses_recommended_product(self):
        research = _research(
            recommended_products=[{"model": "Samsung AR18CY5ARWK", "street_price": 36000, "specs": "good"}],
        )
        prompt = build_prompt(_req(category="split AC"), research, _store())
        assert "Samsung AR18CY5ARWK" in prompt
        assert "36000" in prompt

    def test_negotiation_gentle_tone(self):
        prompt = build_prompt(_req(category="split AC"), _research(), _store())
        assert "GENTLE" in prompt or "gentle" in prompt.lower()
