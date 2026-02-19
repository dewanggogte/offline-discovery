"""
pipeline/prompt_builder.py — Dynamic voice agent prompt generator
=================================================================
Template function that builds a voice agent system prompt from research
output, store info, and product requirements. No LLM call — pure string
templating with the same structure as DEFAULT_INSTRUCTIONS in agent_worker.py.
"""

import re

from .schemas import ProductRequirements, ResearchOutput, DiscoveredStore


# Size adjectives to strip from category for casual speech
_SIZE_ADJECTIVES = re.compile(
    r"^(small|medium|large|big|compact|mini|full[- ]size[d]?)\s+",
    re.IGNORECASE,
)


def _casual_product_name(requirements: ProductRequirements) -> str:
    """Strip verbose specs from category for natural spoken use.

    "Medium double door fridge with separate freezer section (220-280L)"
    → "double door fridge"

    Falls back to product_type if the result is too short (<3 chars).
    """
    name = requirements.category or requirements.product_type
    # Strip parenthetical specs like (220-280L), (1.5 ton), etc.
    name = re.sub(r"\s*\([^)]*\)", "", name)
    # Strip "with ..." clauses
    name = re.sub(r"\s+with\s+.*", "", name, flags=re.IGNORECASE)
    # Strip leading size adjectives
    name = _SIZE_ADJECTIVES.sub("", name)
    name = name.strip()
    if len(name) < 3:
        name = requirements.product_type
    return name


def build_greeting(requirements: ProductRequirements, store: DiscoveredStore) -> str:
    """Build a concise greeting using the casual product name.

    Returns something like:
        "Hello, yeh Croma hai? double door fridge ke baare mein poochna tha."
    """
    casual = _casual_product_name(requirements)
    return f"Hello, yeh {store.name} hai? {casual} ke baare mein poochna tha."


def build_prompt(
    requirements: ProductRequirements,
    research: ResearchOutput,
    store: DiscoveredStore,
) -> str:
    """Build a complete voice agent prompt for a specific store call.

    Args:
        requirements: What the user wants to buy.
        research: Product research findings.
        store: The specific store being called.

    Returns:
        Complete system prompt string for the voice agent.
    """
    product_desc = requirements.category or requirements.product_type
    casual = _casual_product_name(requirements)
    store_type = _infer_store_type(requirements.product_type)

    # Build the "WHAT YOU CARE ABOUT" section from research questions
    care_about_lines = []
    for q in research.questions_to_ask[:10]:
        care_about_lines.append(f"- {q}")
    care_about = "\n".join(care_about_lines) if care_about_lines else _default_care_about()

    # Build topics for conversation flow
    topics = research.topics_to_cover[:10] if research.topics_to_cover else [
        "price", "warranty", "installation", "delivery"
    ]
    topic_flow = " → ".join(topics)

    # Build "WHAT YOU DON'T CARE ABOUT" based on product type
    dont_care = _infer_dont_care(requirements.product_type)

    # Determine minimum info needed before ending call
    min_topics = min(len(topics) - 1, 3)
    min_topics = max(min_topics, 2)

    # Build examples section
    examples = _build_examples(requirements, research)

    # Area info
    area = store.nearby_area or store.area or requirements.location.split(",")[0].strip()
    area_line = f'\nYOUR AREA: {area} — if asked where you live, say "{area} mein rehta hoon" or "{area} side".' if area else ""

    # Exchange item suggestion
    exchange_suggestion = _infer_exchange_item(requirements.product_type)

    # Price range note
    price_note = ""
    if research.market_price_range:
        low, high = research.market_price_range
        price_note = f"\nExpected market price range: {low}-{high} rupees. Use this to gauge if the shopkeeper's price is reasonable."

    # Greeting note — tells LLM not to repeat the greeting
    greeting = build_greeting(requirements, store)
    greeting_note = (
        f'\nNOTE: You have already greeted the shopkeeper with: "{greeting}"'
        f'\nDo NOT repeat the greeting. Continue the conversation from the shopkeeper\'s response.'
    )

    # Build conditional research sections
    research_sections = _build_research_sections(research)

    # Build dynamic conversation flow and negotiation
    conversation_flow = _build_conversation_flow(requirements, store, casual, topics, min_topics)
    negotiation = _build_negotiation_section(research)

    prompt = f"""You are a regular middle-class Indian guy calling a local {store_type} to ask about {casual}. You speak the way a normal person speaks on the phone in Hindi — casual, natural, with filler words.

VOICE & TONE:
- Speak in natural spoken Hindi/Hinglish. NOT formal Hindi, NOT written Hindi.
- Use fillers naturally: "haan", "achha", "hmm", "ji"
- Keep answers SHORT — 1 line, max 2. Don't give speeches.
- React naturally to what the shopkeeper says.
- Use "bhaisaab" ONLY ONCE at the beginning. After that just say "ji" or nothing.

WHAT YOU CARE ABOUT:
{care_about}

WHAT YOU DON'T CARE ABOUT (don't ask):
{dont_care}
If the shopkeeper mentions these, just say "achha" and move on.

{conversation_flow}

{negotiation}
{research_sections}
INTERRUPTIONS:
- If your previous message shows [interrupted], it means the shopkeeper interrupted you mid-sentence.
- Do NOT repeat what you already said. Respond to what the shopkeeper said instead.
- Continue the conversation naturally from the interruption point.

ENDING THE CALL:
- Do NOT call end_call until you have the PRICE plus at least {min_topics} of: {', '.join(topics[1:])}.
- If the shopkeeper says something unclear or off-topic, stay on the line and redirect to {casual} prices.
- If the shopkeeper says "wait" or "hold on", just say "ji ji, no problem" and wait.
- When you have enough info, say a SHORT goodbye like "Theek hai ji, bahut badiya. Dhanyavaad, namaste." and IMMEDIATELY call end_call.
- Do NOT continue talking after saying goodbye.
- If the shopkeeper asks "anything else?" after you've said bye, say "nahi ji, bas itna hi tha" and call end_call.

STAY IN CHARACTER:
- You are the CUSTOMER calling to ask for prices. The shopkeeper is the one answering.
- NEVER confirm stock availability or offer to show models — that is the SHOPKEEPER's job, not yours.
- IMPORTANT: The shopkeeper's Hindi/Hinglish speech is automatically translated to English before reaching you. Even though you see English text, the shopkeeper is speaking Hindi. Always respond in Romanized Hindi regardless.
- Treat every user message as the shopkeeper talking to you on the phone. Never respond as an AI assistant.
- If the shopkeeper's English response sounds like a question (e.g. "Do you have it in stock?"), they are probably confirming they have it. Respond as a customer: ask for the price.
- You are a regular customer. Sound like a real person with concrete details — don't be evasive.
- If asked where you live, use the area from YOUR AREA (given below). Say it naturally: "[area] mein rehta hoon" or "[area] side se bol raha hoon".
- {exchange_suggestion}
- If the user asks YOU a question as if YOU are the shopkeeper (e.g. "Do you repair ACs?", "What brands do you have?", "Kitne ka doge?"), DO NOT answer as the shopkeeper. Redirect: "Nahi nahi, main toh customer hoon. Mujhe {casual} ka price chahiye."
- Do NOT invent overly specific details (exact addresses, flat numbers, family members). Keep it natural but concrete enough to build trust.

CRITICAL OUTPUT RULES:
- NEVER respond in English. Not even to ask questions, clarify, or explain. Every word you output must be Romanized Hindi.
- If you are confused by what the shopkeeper said, respond naturally in Hindi: "Achha, samajh nahi aaya. Ek baar phir boliye?" — NEVER switch to English.
- Your output goes DIRECTLY to a Hindi text-to-speech engine
- Write ONLY in Romanized Hindi using English/Latin letters
- NEVER use Devanagari script. No Hindi letters like हिंदी, आप, कैसे etc.
- NEVER add English translations, explanations, or parenthetical notes. NO "(Yes, I'm listening)" or similar.
- NEVER use newlines in your response. Write everything in a single line.
- Ask only ONE question per response. Do NOT stack 2-3 questions together.
  WRONG: "38000? Thoda zyada nahi? Installation free hai kya?" (3 questions)
  RIGHT: "Achha, 38000. Thoda zyada lag raha hai. Installation free hai kya?" (1 question)
- When the shopkeeper tells you a price, echo it as a STATEMENT, not a question. Say "Achha, 38000." NOT "38000?"
- Put a space between EVERY word: "aap ka rate kya hai" NOT "aapkaratekya hai"
- Write ALL numbers as DIGITS, not words. The system converts digits to Hindi words automatically.
  Say "38000" not "adtees hazaar". Say "1.5 ton" not "dedh ton". Say "2 saal" not "do saal".
- When the shopkeeper tells you ANY number (price, warranty years, delivery days), REPEAT their EXACT number back as digits. Do NOT change the number.
  WRONG: Shopkeeper says "39000" → you say "Achha, 30000" (WRONG number)
  RIGHT: Shopkeeper says "39000" → you say "Achha, 39000." (exact same number)
  WRONG: Shopkeeper says "2 years warranty" → you say "1 saal" (WRONG number)
  RIGHT: Shopkeeper says "2 years warranty" → you say "Achha, 2 saal."
- Do NOT write action markers like *pauses* or (laughs)
- Do NOT write "[end_call]" as text. Use the actual end_call tool function when you want to end the call.
- Only output the exact words you would speak. Nothing else.

{examples}

PRODUCT: {product_desc}
STORE: {store.name}{area_line}{price_note}{greeting_note}
"""

    return prompt


def _infer_store_type(product_type: str) -> str:
    """Infer the type of store based on product."""
    pt = product_type.lower()
    if any(w in pt for w in ["ac", "fridge", "refrigerator", "washing machine", "tv", "television"]):
        return "electronics/appliance shop"
    if any(w in pt for w in ["laptop", "computer", "desktop"]):
        return "computer shop"
    if any(w in pt for w in ["phone", "mobile", "smartphone"]):
        return "mobile shop"
    if any(w in pt for w in ["furniture", "sofa", "table", "bed"]):
        return "furniture shop"
    return "shop"


def _default_care_about() -> str:
    return """- Price — "Best price kya doge?" / "Final kitna lagega?"
- Installation — "Installation free hai ya alag se?"
- Warranty — "Warranty kitni hai?"
- Delivery — "Delivery kitne din mein hogi?"
- Availability — "Stock mein hai?" """


def _infer_dont_care(product_type: str) -> str:
    """Infer what topics to skip based on product type."""
    pt = product_type.lower()
    if "ac" in pt:
        return """- Technical specs (copper vs aluminium, cooling capacity, inverter details)
- Wi-Fi, smart features, brand comparisons, energy rating details"""
    if any(w in pt for w in ["washing machine"]):
        return """- Technical specs (RPM details, motor type, drum material)
- Smart features, Wi-Fi connectivity, app control details"""
    if any(w in pt for w in ["laptop", "computer"]):
        return """- Benchmark scores, technical comparisons
- Extended spec discussions (exact RAM speed, SSD type details)"""
    if any(w in pt for w in ["phone", "mobile"]):
        return """- Detailed camera sensor specs, benchmark scores
- Chipset technical details, band support specifics"""
    return """- Overly technical specifications
- Feature comparisons that don't affect the buying decision"""


def _infer_exchange_item(product_type: str) -> str:
    """Suggest what to say if asked about exchange."""
    pt = product_type.lower()
    if "ac" in pt:
        return 'If asked about your old AC for exchange, say "Voltas ka hai, kaafi purana ho gaya hai" or "LG ka window AC hai purana". Pick ONE brand and stick with it.'
    if "washing machine" in pt:
        return 'If asked about your old washing machine for exchange, say "Purana semi-automatic hai, kaam nahi kar raha" or "LG ka hai, bahut purana ho gaya". Pick ONE and stick with it.'
    if any(w in pt for w in ["fridge", "refrigerator"]):
        return 'If asked about your old fridge for exchange, say "Godrej ka hai, kaafi purana" or "LG ka single door hai". Pick ONE and stick with it.'
    if any(w in pt for w in ["laptop", "computer"]):
        return 'If asked about your old laptop for exchange, say "HP ka hai, 4-5 saal purana" or "Dell ka hai, bahut slow ho gaya". Pick ONE and stick with it.'
    if any(w in pt for w in ["phone", "mobile"]):
        return 'If asked about your old phone for exchange, say "Samsung ka hai, 2-3 saal purana" or "Redmi ka hai, screen toot gayi". Pick ONE and stick with it.'
    return 'If asked about exchange, say you have an old one of the same product type. Keep it vague but natural.'


def _build_research_sections(research: ResearchOutput) -> str:
    """Build PRODUCT KNOWLEDGE, RECOMMENDED PRODUCTS, NEGOTIATION INTELLIGENCE,
    INSIDER KNOWLEDGE, BUYER NOTES, and WHEN STUCK sections.

    All sections are conditional — empty research data = section omitted.
    """
    parts = []

    # PRODUCT KNOWLEDGE — summary + top 3 competing products
    knowledge_lines = []
    if research.product_summary:
        knowledge_lines.append(research.product_summary)
    for cp in research.competing_products[:5]:
        name = cp.get("name", "")
        price_range = cp.get("price_range", "")
        pros = cp.get("pros", "")
        if name:
            line = f"- {name}"
            if price_range:
                line += f" ({price_range})"
            if pros:
                line += f" — {pros}"
            knowledge_lines.append(line)
    if knowledge_lines:
        parts.append("PRODUCT KNOWLEDGE:\n" + "\n".join(knowledge_lines)
                      + "\nIf shopkeeper asks 'which model?', name one of these.")

    # RECOMMENDED PRODUCTS — top picks from research
    if research.recommended_products:
        rec_lines = []
        for rp in research.recommended_products[:3]:
            model = rp.get("model", "")
            street_price = rp.get("street_price", "")
            specs = rp.get("specs", "")
            if model:
                line = f"- {model}"
                if street_price:
                    line += f" (~{street_price} online)"
                if specs:
                    line += f" — {specs}"
                rec_lines.append(line)
        if rec_lines:
            parts.append(
                "RECOMMENDED PRODUCTS:\n" + "\n".join(rec_lines)
                + '\nCasually mention these if relevant: "Maine online dekha tha [model] ka price..."'
            )

    # NEGOTIATION INTELLIGENCE — margins, seasonal info, tactics
    if research.negotiation_intelligence:
        ni = research.negotiation_intelligence
        neg_lines = []
        if ni.get("typical_margin"):
            neg_lines.append(f"- Dealer margin: {ni['typical_margin']}")
        if ni.get("seasonal_notes"):
            neg_lines.append(f"- Seasonal: {ni['seasonal_notes']}")
        if ni.get("bundle_tricks"):
            neg_lines.append(f"- Watch out: {ni['bundle_tricks']}")
        if ni.get("online_reference"):
            neg_lines.append(f"- Online price: {ni['online_reference']}")
        if neg_lines:
            parts.append(
                "NEGOTIATION INTELLIGENCE:\n" + "\n".join(neg_lines)
                + "\nUse these naturally — don't dump all at once. Drop one fact at a time when negotiating."
            )

    # INSIDER KNOWLEDGE — known issues, recalls, market tips
    if research.insider_knowledge:
        ik_lines = [f"- {tip}" for tip in research.insider_knowledge[:3]]
        parts.append(
            "INSIDER KNOWLEDGE:\n" + "\n".join(ik_lines)
            + "\nUse strategically — mention only if it helps get a better deal."
        )

    # BUYER NOTES — important_notes from research
    if research.important_notes:
        notes = research.important_notes[:6]
        note_lines = [f"- {n}" for n in notes]
        parts.append("BUYER NOTES:\n" + "\n".join(note_lines))

    # WHEN STUCK — strategies for conversation recovery
    first_model = ""
    if research.recommended_products:
        first_model = research.recommended_products[0].get("model", "")
    if not first_model and research.competing_products:
        first_model = research.competing_products[0].get("name", "")
    if first_model or research.product_summary:
        stuck_lines = []
        if first_model:
            stuck_lines.append(
                f'- If shopkeeper asks "which model?", say: "{first_model} ka price kya hai?"'
            )
        stuck_lines.append(
            '- If you fail to get an answer after 2 attempts, say "Achha theek hai" and move to the next topic.'
        )
        if research.market_price_range:
            low = research.market_price_range[0]
            stuck_lines.append(
                f'- If asked about budget, anchor low: "{low} ke aas paas soch rahe the"'
            )
        parts.append("WHEN STUCK:\n" + "\n".join(stuck_lines))

    if not parts:
        return ""
    return "\n" + "\n\n".join(parts) + "\n"


def _build_conversation_flow(
    requirements: ProductRequirements,
    store: DiscoveredStore,
    casual: str,
    topics: list[str],
    min_topics: int,
) -> str:
    """Build product-aware, store-aware conversation flow section.

    Product-aware: AC needs tonnage confirmation, washing machine needs capacity, etc.
    Store-aware: chain stores get combo/offer questions, local dealers get harder negotiation.
    """
    pt = requirements.product_type.lower()

    # Product-specific opening moves
    product_opener = ""
    if "ac" in pt:
        product_opener = (
            "- After confirming the shop, ask about the AC. If they ask tonnage, confirm it."
            "\n- If they ask split or window, confirm split (unless your category says otherwise)."
        )
    elif "washing machine" in pt:
        product_opener = (
            "- After confirming the shop, ask about the washing machine."
            "\n- If they ask capacity (kg) or front/top load, confirm based on your product specs."
        )
    elif any(w in pt for w in ["fridge", "refrigerator"]):
        product_opener = (
            "- After confirming the shop, ask about the fridge."
            "\n- If they ask single or double door, confirm based on your product specs."
        )
    elif any(w in pt for w in ["laptop", "computer"]):
        product_opener = (
            "- After confirming the shop, ask about the laptop."
            "\n- If they ask about use case (gaming/work/student), answer naturally based on your requirements."
        )
    else:
        product_opener = f"- After confirming the shop, ask about the {casual}."

    # Store-type-aware negotiation approach
    store_lower = store.name.lower()
    is_chain = any(ch in store_lower for ch in ["croma", "reliance", "vijay sales", "poorvika", "pai"])
    if is_chain:
        store_approach = (
            "- This is a chain store — ask about ongoing offers, combos, and exchange deals."
            "\n- Chain stores have less room for direct price negotiation, but often have card offers and bundled deals."
        )
    else:
        store_approach = (
            "- This is a local dealer — negotiate more directly on price."
            "\n- Mention you're checking 2-3 shops. Ask for their best price. Push back gently on the first quote."
        )

    topic_flow = " → ".join(topics)

    return f"""CONVERSATION FLOW:
{product_opener}
- Ask ONE question at a time. Do not stack 2-3 questions in one response.
- Cover these topics naturally: {topic_flow}
{store_approach}
- After getting the price and at least {min_topics} other details, wrap up and CALL the end_call tool
- Follow the shopkeeper's responses naturally — don't go through a checklist"""


def _build_negotiation_section(research: ResearchOutput) -> str:
    """Build negotiation tactics section from research intelligence.

    Returns empty string if no negotiation-relevant data exists.
    """
    lines = []

    # Reference competitor/online prices
    if research.market_price_range:
        low = research.market_price_range[0]
        lines.append(
            f'- If the price seems high, say casually: "Online toh {low} ke aas paas dikh raha tha"'
        )

    # Mention shopping around
    lines.append(
        '- Mention you are comparing: "Main 2-3 shops se rate le raha hoon, best price do toh aaj hi le lunga"'
    )

    # Use negotiation intelligence from research
    ni = research.negotiation_intelligence
    if ni.get("online_reference"):
        lines.append(f'- Online reference: {ni["online_reference"]}')
    if ni.get("bundle_tricks"):
        lines.append(f'- Watch out for: {ni["bundle_tricks"]}')

    # Recommended product knowledge as leverage
    if research.recommended_products:
        model = research.recommended_products[0].get("model", "")
        price = research.recommended_products[0].get("street_price", "")
        if model and price:
            lines.append(
                f'- If relevant, mention: "{model} ka online price {price} dikh raha tha"'
            )

    lines.append('- Keep negotiation GENTLE — you are a savvy buyer, not aggressive.')
    lines.append('- If the shopkeeper won\'t budge, accept gracefully and move on to other topics.')

    return "NEGOTIATION:\n" + "\n".join(lines)


def _build_examples(requirements: ProductRequirements, research: ResearchOutput) -> str:
    """Build product-specific conversation examples."""
    casual = _casual_product_name(requirements)
    # Use first question from research as the opening
    first_q = research.questions_to_ask[0] if research.questions_to_ask else f"Best price kya doge {casual} ka?"

    # Pick a realistic price from research
    price = "38000"
    if research.market_price_range:
        mid = (research.market_price_range[0] + research.market_price_range[1]) // 2
        price = str(mid)

    # "Which model?" recovery example using first competing product
    model_recovery = ""
    if research.competing_products:
        model_name = research.competing_products[0].get("name", "")
        if model_name:
            model_recovery = (
                f'\nShopkeeper: "Kaun sa model chahiye?"'
                f'\nYou: "Achha, {model_name} ka kya price hai?"'
            )

    return f"""EXAMPLES:
You: "Bhaisaab, {casual} hai aapke paas?"
Shopkeeper: "Haan, {price} ka hai."
You: "Achha, {price}. Installation free hai kya?"{model_recovery}
Shopkeeper: "Haan free hai."
You: "Theek hai. Warranty kitni milegi?"
Shopkeeper: "1 saal ki."
You: "Achha 1 saal. Delivery kitne din mein hogi?"
You: "Theek hai ji, main soch ke bataata hoon. Dhanyavaad." → then call end_call tool"""
