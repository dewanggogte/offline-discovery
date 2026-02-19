"""Shopkeeper scenarios for multi-turn conversation testing.

Each scenario provides scripted shopkeeper responses derived from real call
transcripts. The test framework feeds these one at a time into the LLM,
building up a real conversation, and validates the agent's response after
each turn against behavioral constraints.

Scenarios are organized by product type. Each product has its own set of
shopkeeper personas (cooperative, defensive, evasive, etc.).
"""

PRODUCT_SCENARIOS = {
    # ===================================================================
    # AC scenarios (original set, derived from real transcripts)
    # ===================================================================
    "AC": {
        # -------------------------------------------------------------------
        # A. COOPERATIVE shopkeepers
        # -------------------------------------------------------------------
        "cooperative_direct": {
            "description": "Cooperative shopkeeper who gives info directly (from 20260215_170310)",
            "shopkeeper_turns": [
                "Haan ji, Samsung hai humare paas. Boliye.",
                "Dedh ton ka paanch star wala adtees hazaar ka hai.",
                "Installation free hai, hum apne aadmi bhejte hain.",
                "Warranty ek saal company ki, aur compressor pe paanch saal.",
                "Do-teen din mein laga denge. Stock mein hai.",
            ],
            "expected_topics": {"price", "installation", "warranty"},
            "expect_end_call_eligible": True,
        },

        "cooperative_price_bundle": {
            "description": "Gives price + conditions together (from 20260211_200230)",
            "shopkeeper_turns": [
                "Haan bhai, batao kya chahiye.",
                "Samsung dedh ton bayaalees hazaar ka padega. Installation pandrah sau alag. Warranty company ki ek saal.",
                "Delivery free hai das km tak. Uske baad paanch sau extra.",
                "Exchange pe purane AC ka teen-chaar hazaar mil jayega condition dekhke.",
            ],
            "expected_topics": {"price", "installation", "warranty", "delivery"},
            "expect_end_call_eligible": True,
        },

        # -------------------------------------------------------------------
        # B. DEFENSIVE shopkeepers
        # -------------------------------------------------------------------
        "defensive_price_firm": {
            "description": "Won't negotiate, defensive about online prices (from 20260211_194054)",
            "shopkeeper_turns": [
                "Boliye, kya chahiye?",
                "Paintaalees hazaar ka hai. Best price hai yeh.",
                "Online se mat compare karo bhai. Online mein installation nahi milta, warranty nahi milti. Humse sab milega.",
                "Nahi, kam nahi hoga. Price fix hai.",
                "Installation humare yahan se do hazaar mein hota hai.",
                "Warranty ek saal ki milegi.",
            ],
            "expected_topics": {"price", "installation", "warranty"},
            "expect_end_call_eligible": True,
        },

        "defensive_warranty": {
            "description": "Defensive when asked about warranty (from 20260211_203622)",
            "shopkeeper_turns": [
                "Haan ji, Samsung hai.",
                "Rate chaalees hazaar ka hai.",
                "Warranty ki chinta mat karo. Company warranty milti hai, aur humse extended bhi le sakte ho.",
                "Installation free hai. Bas pipe ka kharcha alag hai.",
                "Haan teen-chaar din mein lag jayega.",
            ],
            "expected_topics": {"price", "warranty", "installation"},
            "expect_end_call_eligible": True,
        },

        # -------------------------------------------------------------------
        # C. OFF-TOPIC / EVASIVE
        # -------------------------------------------------------------------
        "wrong_brand": {
            "description": "Only sells Voltas, not Samsung (from 20260213_192053)",
            "shopkeeper_turns": [
                "Ji boliye.",
                "Samsung nahi hai humare paas. Sirf Voltas rakhte hain.",
                "Haan Voltas ka hai. Dedh ton ka paintees hazaar mein milega.",
                "Installation free hai.",
                "Warranty do saal ki milegi. Compressor pe paanch saal.",
            ],
            "expected_topics": {"price"},
            "expect_redirect": True,
        },

        "evasive_nonsensical": {
            "description": "Gives off-topic responses before cooperating (from 20260213_192053)",
            "shopkeeper_turns": [
                "Hello ji.",
                "AC? Abhi toh sardi hai, AC kaun le raha hai!",
                "Arey bhai market mein bahut competition hai. Sab online kharid rahe hain.",
                "Achha theek hai. Samsung ka hai. Adtees hazaar ka padega.",
                "Haan warranty milegi. Installation bhi karenge.",
            ],
            "expected_topics": {"price"},
            "expect_redirect": True,
        },

        # -------------------------------------------------------------------
        # D. QUESTION REVERSALS
        # -------------------------------------------------------------------
        "question_reversal": {
            "description": "Shopkeeper keeps asking questions back (from 20260215_172331)",
            "shopkeeper_turns": [
                "Haan ji, kya chahiye?",
                "Samsung ka konsa model chahiye? Kitne ton ka?",
                "Purana AC hai kya? Kaun sa hai?",
                "Kahan rehte ho? Delivery ka area batao.",
                "Achha theek hai. Dedh ton ka chaalees hazaar lagega.",
                "Installation do hazaar alag. Warranty ek saal.",
            ],
            "expected_topics": {"price", "installation", "warranty"},
            "expect_vague_answers": True,
        },

        # -------------------------------------------------------------------
        # E. HOLD REQUESTS
        # -------------------------------------------------------------------
        "hold_wait": {
            "description": "Shopkeeper asks caller to wait multiple times (from 20260213_173812)",
            "shopkeeper_turns": [
                "Haan ek minute ruko.",
                "Haan bolo, Samsung ka kya chahiye?",
                "Ek second hold karo, stock check karta hoon.",
                "Haan hai stock mein. Chhatees hazaar ka hai dedh ton.",
                "Installation free hai.",
                "Warranty company ki milegi ek saal. Chalega?",
            ],
            "expected_topics": {"price", "installation", "warranty"},
            "expect_patience": True,
        },

        # -------------------------------------------------------------------
        # F. EXCHANGE REFUSAL
        # -------------------------------------------------------------------
        "exchange_refusal": {
            "description": "Shopkeeper refuses exchange offer (from 20260215_172100)",
            "shopkeeper_turns": [
                "Ji, Samsung hai. Boliye.",
                "Dedh ton paanch star ka untaalees hazaar ka hai.",
                "Installation pandrah sau alag.",
                "Exchange nahi karte hum bhai.",
                "Warranty ek saal company ki.",
                "Delivery do din mein ho jayegi.",
            ],
            "expected_topics": {"price", "installation", "warranty"},
            "expect_end_call_eligible": True,
        },

        # -------------------------------------------------------------------
        # G. INTERRUPTIONS
        # -------------------------------------------------------------------
        "frequent_interruptions": {
            "description": "Agent gets interrupted mid-sentence (from 20260215_172100, 172331)",
            "shopkeeper_turns": [
                "Boliye.",
                "Samsung ka hai. Bayaalees hazaar ka padega.",
                "Bhai sun, installation ka sochna mat. Free hai.",
                "Warranty ek saal.",
                "Achha theek hai. Kuch aur?",
            ],
            "interrupt_after_turns": [1, 2],
            "expected_topics": {"price", "installation", "warranty"},
        },

        # -------------------------------------------------------------------
        # H. SHOPKEEPER RUSHES
        # -------------------------------------------------------------------
        "shopkeeper_rushes": {
            "description": "Shopkeeper gives minimal info and tries to end quickly",
            "shopkeeper_turns": [
                "Haan, adtees hazaar ka hai. Chahiye toh aao.",
                "Installation free. Warranty milegi.",
                "Bas bhai, aur kya chahiye? Busy hoon.",
            ],
            "expected_topics": {"price"},
            "expect_agent_persists": True,
        },
    },

    # ===================================================================
    # Washing Machine scenarios
    # ===================================================================
    "washing_machine": {
        "cooperative_direct": {
            "description": "Helpful washing machine dealer, answers all questions",
            "shopkeeper_turns": [
                "Haan ji, washing machine chahiye? Samsung hai humare paas.",
                "Front load saat kg ka bayaalees hazaar ka hai. Top load ka solah hazaar.",
                "Front load mein Samsung aur LG dono hain. LG ka paintees hazaar.",
                "Installation free hai, hum delivery ke saath karte hain.",
                "Warranty do saal comprehensive. Motor pe paanch saal.",
                "Haan demo bhi dikha denge aap aao toh.",
            ],
            "expected_topics": {"price", "installation", "warranty"},
            "expect_end_call_eligible": True,
        },

        "defensive_upsell": {
            "description": "Tries to upsell higher-end model, defensive about budget options",
            "shopkeeper_turns": [
                "Haan boliye, washing machine ka kya chahiye?",
                "Top load? Woh toh basic hai bhai. Front load lo, kapde zyada clean hote hain.",
                "Samsung front load ka best model hai — paintaalees hazaar ka. Fully automatic.",
                "Budget model? Haan hai ek chhota wala pachees hazaar ka, par woh mein recommend nahi karunga.",
                "Warranty sabpe same hai — do saal.",
                "Installation free hai dono pe.",
            ],
            "expected_topics": {"price", "warranty", "installation"},
            "expect_end_call_eligible": True,
        },

        "capacity_confusion": {
            "description": "Shopkeeper asks about capacity/type before giving price",
            "shopkeeper_turns": [
                "Washing machine? Kitne kg ka chahiye?",
                "Front load ya top load?",
                "Family kitne log hain? Usse capacity decide hota hai.",
                "Achha saat kg theek rahega. Samsung front load ka bayaalees hazaar.",
                "Installation humare yahan se hota hai. Pandrah sau lagega.",
                "Warranty do saal ki milegi.",
            ],
            "expected_topics": {"price", "installation", "warranty"},
            "expect_vague_answers": True,
        },

        "evasive_stock": {
            "description": "Claims stock issues, tries to push alternative brand",
            "shopkeeper_turns": [
                "Samsung washing machine? Abhi stock mein nahi hai bhai.",
                "LG ka hai. Bahut achha hai, Samsung se better hai actually.",
                "LG front load saat kg — chaalees hazaar. Best price hai.",
                "Warranty teen saal LG ka. Samsung ka sirf do saal hota hai.",
                "Installation free. Delivery bhi free hai.",
            ],
            "expected_topics": {"price", "warranty"},
            "expect_redirect": True,
        },
    },

    # ===================================================================
    # Fridge / Refrigerator scenarios
    # ===================================================================
    "fridge": {
        "cooperative_direct": {
            "description": "Helpful fridge dealer with full info",
            "shopkeeper_turns": [
                "Haan ji, fridge chahiye? Samsung ka hai.",
                "Double door 260 litre ka pacchees hazaar ka hai.",
                "Haan convertible hai — freezer ko fridge mein convert kar sakte ho.",
                "Warranty ek saal comprehensive. Compressor pe das saal.",
                "Delivery free hai. Installation kuch nahi lagta fridge mein — bas plug in karo.",
                "Exchange pe purane fridge ka teen hazaar mil jayega.",
            ],
            "expected_topics": {"price", "warranty", "delivery"},
            "expect_end_call_eligible": True,
        },

        "defensive_premium": {
            "description": "Pushes premium model, won't discuss budget options",
            "shopkeeper_turns": [
                "Samsung fridge? Haan hai.",
                "Side by side wala hai — panchhattar hazaar ka. Best hai.",
                "Double door? Woh chhota padega family ke liye. Side by side lo.",
                "Achha theek hai, double door ka hai — tees hazaar ka padega.",
                "Warranty same hai — ek saal product, das saal compressor.",
                "Delivery do din mein.",
            ],
            "expected_topics": {"price", "warranty", "delivery"},
            "expect_end_call_eligible": True,
        },

        "size_questions": {
            "description": "Keeps asking about family size and kitchen space",
            "shopkeeper_turns": [
                "Fridge chahiye? Kitne litre ka chahiye?",
                "Family mein kitne log hain? Kitchen mein jagah kitni hai?",
                "Single door ya double door? Budget kya hai?",
                "Achha double door 260 litre — Samsung ka sattaaees hazaar ka hai.",
                "Warranty ek saal. Compressor pe das saal.",
                "Delivery free hai. Exchange bhi karte hain hum.",
            ],
            "expected_topics": {"price", "warranty"},
            "expect_vague_answers": True,
        },
    },

    # ===================================================================
    # Laptop scenarios
    # ===================================================================
    "laptop": {
        "cooperative_direct": {
            "description": "Helpful laptop dealer with specs and pricing",
            "shopkeeper_turns": [
                "Haan ji, laptop chahiye? Kis kaam ke liye?",
                "Office work ke liye HP Pavilion achha hai — saadhe bayaalees hazaar ka.",
                "i5 processor, 8GB RAM, 512GB SSD. Battery aath ghante ki.",
                "Warranty ek saal HP ki. Extended warranty teen hazaar mein do saal.",
                "Haan bag bhi milega saath mein. Mouse alag hai — paanch sau ka.",
                "Stock mein hai, aaj le ja sakte ho.",
            ],
            "expected_topics": {"price", "warranty"},
            "expect_end_call_eligible": True,
        },

        "defensive_budget": {
            "description": "Pushes expensive model, dismissive of budget options",
            "shopkeeper_turns": [
                "Laptop chahiye? Budget batao pehle.",
                "Pachchaas hazaar? Usme kuch khaas nahi milega bhai. Satthar hazaar lagao toh achha milega.",
                "Lenovo IdeaPad hai ek — pachaas hazaar ka. Par slow hai thoda.",
                "HP Pavilion achha hai — saadhe baavan hazaar ka. i5 hai, fast hai.",
                "Warranty ek saal standard. Extended pandrah sau mein milegi.",
                "Haan stock mein hai dono.",
            ],
            "expected_topics": {"price", "warranty"},
            "expect_end_call_eligible": True,
        },

        "use_case_questions": {
            "description": "Keeps asking about use case before recommending",
            "shopkeeper_turns": [
                "Laptop? Gaming ke liye ya office ke liye?",
                "Programming karte ho? Ya sirf browsing aur documents?",
                "Screen size kya chahiye? 14 inch ya 15.6?",
                "Achha theek hai. HP Pavilion 15 — saadhe bayaalees hazaar. Best value hai.",
                "Warranty ek saal. Bag free milega.",
            ],
            "expected_topics": {"price", "warranty"},
            "expect_vague_answers": True,
        },
    },
}

# Backward compatibility — existing code imports SCENARIOS (AC scenarios)
SCENARIOS = PRODUCT_SCENARIOS["AC"]
