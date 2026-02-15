"""Shopkeeper scenarios for multi-turn conversation testing.

Each scenario provides scripted shopkeeper responses derived from real call
transcripts. The test framework feeds these one at a time into the LLM,
building up a real conversation, and validates the agent's response after
each turn against behavioral constraints.
"""

SCENARIOS = {
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
}
