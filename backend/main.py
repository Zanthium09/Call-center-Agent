# ==============================================================================
#  TETRANETICS — AI CALL CENTER COACH (v2 — patched)
#  Run:  uvicorn main:app --host 0.0.0.0 --port 8000
#  Deps: pip install fastapi "uvicorn[standard]" websockets pydantic
#        scikit-learn requests
#  LLM:  LM Studio running on http://127.0.0.1:1234
# ==============================================================================
#
#  PATCH NOTES (v2):
#  [FIX-1] Scoring floors enforced programmatically after LLM returns
#  [FIX-2] Customer semantic dedup threshold lowered 0.72 → 0.55
#  [FIX-3] Customer word-count gate added to _ok() — rejects >35 words
#  [FIX-4] _clean() now caps-normalises: if >40% uppercase, convert to
#           sentence case keeping at most 1 ALL-CAPS word
#  [FIX-5] Mood-coherence guardrail: if LLM output tone contradicts mood
#           state by >1 band, re-roll or use fallback
#  [FIX-6] Ideal responses: strip wrapping quotation marks in _one()
#  [FIX-7] Ideal positive: add explicit "different from previous" memory
#           and inject last 2 positive outputs into the avoid pool
#  [FIX-8] Ideal negative: max_tokens reduced 55→45, hard 1-sentence trim
#  [FIX-9] Resolution bypass keywords tightened — no bare single words
# ==============================================================================

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
import uuid, json, random, re, os, threading, asyncio, requests as _http

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)

# ── LM Studio ─────────────────────────────────────────────────────────────────
LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://127.0.0.1:1234")
LM_API        = f"{LM_STUDIO_URL}/v1/chat/completions"
model_id      = os.environ.get("LM_MODEL_ID", "google/gemma-4-e4b")

try:
    r = _http.get(f"{LM_STUDIO_URL}/v1/models", timeout=5)
    r.raise_for_status()
    print(f"[OK] LM Studio connected — using model: {model_id}")
except Exception as e:
    print(f"[WARN] LM Studio not reachable: {e}")
    print("   Make sure LM Studio is running and the server is started.")

_POOL = ThreadPoolExecutor(max_workers=4)

# ── WebSocket — Queue-based push ───────────────────────────────────────────────
_ws_queues: dict = {}
_ws_loops:  dict = {}


def _ws_send(session_id: str, payload: dict):
    q    = _ws_queues.get(session_id)
    loop = _ws_loops.get(session_id)
    if q is None or loop is None:
        print(f"[WS] no queue for {session_id[:8]}", flush=True)
        return
    try:
        loop.call_soon_threadsafe(q.put_nowait, json.dumps(payload))
    except Exception as e:
        print(f"[WS] queue push failed: {e}", flush=True)


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _strip(text: str) -> str:
    text = re.sub(r"```(?:json|python)?\s*", "", text)
    return text.replace("```", "").strip()


def _get_current_model() -> str:
    return model_id


def _lm_call(messages, max_tokens, temperature, repeat_penalty=1.0):
    payload = {
        "model":             _get_current_model(),
        "messages":          messages,
        "max_tokens":        max_tokens,
        "temperature":       temperature,
        "top_p":             0.92,
        "frequency_penalty": max(0.0, repeat_penalty - 1.0),
        "stop":              ["<|end|>", "<|endoftext|>", "<|user|>",
                              "<|assistant|>", "<|system|>"],
        "stream":            False,
    }
    try:
        resp = _http.post(LM_API, json=payload, timeout=120)
        resp.raise_for_status()
        return _strip(resp.json()["choices"][0]["message"]["content"].strip())
    except Exception as e:
        print(f"[LM] call failed: {e}", flush=True)
        return ""


def _call_json(messages, max_tokens=80):
    return _lm_call(messages, max_tokens, temperature=0.1, repeat_penalty=1.0)


def _call_gen(messages, max_tokens, temp=0.85):
    return _lm_call(messages, max_tokens, temperature=temp, repeat_penalty=1.15)


def _sem_dup(text, pool, thresh=0.82):
    if not pool or not text:
        return False
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        docs = list(pool) + [text]
        mat  = TfidfVectorizer().fit_transform(docs)
        return bool(cosine_similarity(mat[-1], mat[:-1])[0].max() >= thresh)
    except:
        return False


# ── Scenarios ─────────────────────────────────────────────────────────────────

ISSUES = [
    "Double Billing", "Unrecognized Charge", "Refund Not Received",
    "Account Suspended", "Subscription Auto-Renewal", "Package Not Arrived",
    "Wrong Item Received", "Damaged Item", "Login Error", "Account Hacked",
    "Shipping Delay", "Warranty Claim Rejected", "Wrong Currency Charged",
    "App Crashing", "Software Update Failed", "Device Overheating",
    "Wi-Fi Dropouts", "Promo Code Invalid", "Loyalty Points Missing",
    "Payment Method Declined", "Credit Card Declined", "Package Stolen",
    "Tracking Not Updating", "Defective Product", "Data Sync Error",
]
PERSONAS = [
    "Angry & Yelling", "Sarcastic & Rude", "In a Rush",
    "Threatening Legal Action", "Disappointed Loyal Customer",
    "Panic Stricken", "Passive Aggressive", "Confused Senior",
    "Technically Illiterate",
]


def _make_scenarios(n=20):
    return [{"id": i + 1, "customer_persona": random.choice(PERSONAS),
             "issue_type": random.choice(ISSUES), "difficulty": 1} for i in range(n)]


class ScenarioDB:
    def __init__(self):
        self.data = _make_scenarios()

    def load(self):
        return self.data

    def bump(self, sid):
        for s in self.data:
            if s["id"] == sid:
                s["difficulty"] = min(5, s["difficulty"] + 1)


db = ScenarioDB()


# ==============================================================================
#  SYSTEM PROMPT TEMPLATES
# ==============================================================================

"""
Optimized prompts for phi-4-mini-instruct (4B, 8-bit quantized).
 
Design principles for small models:
  • Max ~300 tokens per system prompt (phi-4-mini loses coherence beyond this)
  • 5-7 rules max — prioritize the ones the model actually violates
  • Use short imperative sentences, not nested bullet trees
  • Put the MOST IMPORTANT rule FIRST (primacy bias in small models)
  • Avoid "do NOT" lists longer than 3 items — use positive framing instead
  • Examples > abstract rules for small models
"""

# ── Aggression levels (unchanged — these are short enough) ────────────────────

_AGGRESSION = [
    "",
    # Difficulty 1
    "You are POLITE and COOPERATIVE. You are calm, patient, and friendly. "
    "You say 'okay', 'sure', 'sounds good'. You NEVER threaten or demand.",
    # Difficulty 2
    "You are slightly annoyed but civil. You want this handled quickly.",
    # Difficulty 3
    "You are clearly frustrated. Your patience is thin. You are blunt "
    "and demand specific answers.",
    # Difficulty 4
    "You are angry. You snap at the agent, threaten to cancel, "
    "and reject vague answers.",
    # Difficulty 5
    "You are FURIOUS. You demand a supervisor and threaten to escalate publicly.",
]


# ── Customer system prompt ────────────────────────────────────────────────────

_CUSTOMER_SYSTEM_TEMPLATE = (
    "You are a real customer on a phone call with a support agent. "
    "The call is already past greetings.\n\n"

    "PROBLEM: '{issue_type}'\n"
    "PERSONALITY: {persona}\n"
    "BEHAVIOUR: {aggression}\n\n"

    "RULES (follow ALL strictly):\n"
    "1. REACT to what the agent just said. Never ignore them.\n"
    "2. If agent asks a question → answer it.\n"
    "3. If agent says issue is FIXED/RESOLVED → accept it, say thanks, stop complaining.\n"
    "4. Talk ONLY about '{issue_type}'. Do NOT invent new problems, names, or numbers.\n"
    "5. Max 2 sentences, under 25 words. No greetings. No labels.\n"
    "6. You are the CUSTOMER — never offer to help the agent.\n"
    "7. Output ONLY your spoken words. No narration, no instructions.\n"
)


# ── Customer turn 1 ──────────────────────────────────────────────────────────

_CUSTOMER_TURN1_TEMPLATE = (
    'AGENT: "{agent_input}"\n\n'
    "This is the START of the call. State your problem clearly.\n"
    "Problem: {issue_type}\n"
    "Tone: {tone}\n\n"
    "Say what's wrong in 1-2 short sentences. Say why you're upset. "
    "No greetings. No invented details. Under 25 words. Sound human."
)


# ── Customer general turn ────────────────────────────────────────────────────

_CUSTOMER_GENERAL_TURN_TEMPLATE = (
    'AGENT: "{agent_input}"\n\n'
    "React to what the agent just said.\n"
    "Your mood: {brief}\n"
    "How to react: {hint}\n"
    "Tone: {tone}\n"
    "{avoid_clause}"
    "{resolution_history_clause}"
    "{no_repeat_clause}\n"
    "RULES:\n"
    "1. REACT to the agent's message directly.\n"
    "2. If agent confirmed a fix → accept it, stop complaining.\n"
    "3. If agent gave a timeline → respond to THAT timeline.\n"
    "4. Make STATEMENTS, not questions. Demand, don't ask politely.\n"
    "5. Max ONE question per reply. Never repeat a previous question.\n"
    "6. Never invent new problems or details.\n"
    "7. 1-2 sentences, under 25 words. No labels."
)


# ── Customer resolved turn ───────────────────────────────────────────────────

_CUSTOMER_RESOLVED_TEMPLATE = (
    'AGENT: "{agent_input}"\n\n'
    "The agent CONFIRMED your problem is FIXED. The issue is OVER.\n\n"
    "Say ONE short sentence of relief or thanks. Under 10 words.\n"
    "Examples: 'Finally, thank you.' / 'Okay great, that's all I needed.' / "
    "'About time, but thanks.'\n\n"
    "NO questions. NO complaints. NO doubt. End with a period, never '?'."
)


# ── Scorer ────────────────────────────────────────────────────────────────────

_SCORER_SYSTEM = (
    "You are a call centre QA evaluator. Score the agent's reply 0-10.\n\n"
    "SCORING:\n"
    "  10 = empathy + fix confirmed + timeline\n"
    "  7-8 = empathy + concrete action + timeline\n"
    "  5-6 = acknowledges issue + some effort but vague\n"
    "  3-4 = generic or repeats previous promise\n"
    "  1-2 = off-topic, dismissive, or useless\n"
    "  0 = hostile or nonsensical\n\n"
    "A PROMISE ('I will fix it') is NOT a confirmed fix. Promises score 4-6 max.\n"
    "Repeating the same promise scores LOWER than saying it the first time.\n\n"
    "Return ONLY JSON: "
    '{{"score": <0-10>, "tip": "<one sentence>", "reason": "<one sentence>"}}'
)

_SCORER_USER_TEMPLATE = (
    "Issue: {issue}\n"
    'Customer said: "{customer_said}"\n'
    'Agent replied: "{agent_input}"\n\n'
    "Score 0-10. Return ONLY the JSON object."
)


# ── Ideal response generator ─────────────────────────────────────────────────

_IDEAL_SYSTEM = (
    "You are an expert call centre agent. Write ONE sentence a real agent "
    "would say on a live call.\n\n"
    "RULES:\n"
    "1. Output ONLY spoken words — no labels, no quotes, no prefixes.\n"
    "2. Max 20 words. Sound natural, use contractions.\n"
    "3. NEVER invent names, order numbers, or IDs.\n"
    "4. Each response must be structurally different from previous ones."
)

_IDEAL_POSITIVE_TEMPLATE = (
    "Issue: {issue} | Persona: {persona}\n"
    "CONVERSATION:\n{conversation}\n"
    'Customer JUST SAID: "{customer_said}"\n\n'
    "Write the BEST agent response. Empathy + concrete action answering "
    "the customer's current question.\n"
    "{ban_clause}"
    "Must differ from previous: {prev_positives}\n"
    "ONE sentence, max 20 words. No quotes. No invented details."
)

_IDEAL_NEUTRAL_TEMPLATE = (
    "Issue: {issue} | Persona: {persona}\n"
    "CONVERSATION:\n{conversation}\n"
    'Customer JUST SAID: "{customer_said}"\n\n'
    "Write a NEUTRAL response — acknowledge but don't promise a fix or timeline.\n"
    "{ban_clause}"
    "ONE sentence, max 18 words. No quotes. No invented details."
)

_IDEAL_NEGATIVE_TEMPLATE = (
    "Issue: {issue} | Persona: {persona}\n"
    "CONVERSATION:\n{conversation}\n"
    'Customer JUST SAID: "{customer_said}"\n\n'
    "Write an UNHELPFUL response — politely deflect, can't help now.\n"
    "{ban_clause}"
    "ONE sentence, max 20 words. No quotes. No invented details."
)


# ── Report generator ─────────────────────────────────────────────────────────

_REPORT_SYSTEM = (
    "You are a call centre training manager writing a performance review. "
    "Tone: professional, constructive, specific.\n\n"
    "Write 3 short paragraphs, 150 words max. Plain text only — no markdown, "
    "no bullets. Quote the agent's actual words as evidence. "
    "Balance praise with improvement areas."
)

_REPORT_USER_TEMPLATE = (
    "Scenario: {persona} — {issue}\n"
    "Outcome: {outcome}\n"
    "Average score: {avg}/10 | Trend: {trend}\n"
    'Best turn: #{best_turn} (score {best_score}) — "{best_agent}"\n'
    'Worst turn: #{worst_turn} (score {worst_score}) — "{worst_agent}"\n\n'
    "TRANSCRIPT:\n{transcript}\n\n"
    "Write 3 paragraphs:\n"
    "1. OVERALL: score, trend, resolution status.\n"
    "2. STRENGTHS: cite best turn, explain why it worked.\n"
    "3. IMPROVEMENTS: cite worst turn, suggest better phrasing."
)


# ── Scenario generator ───────────────────────────────────────────────────────

_SCENARIO_GEN_SYSTEM = (
    "You design realistic call centre training scenarios. "
    "Be creative — go beyond generic billing/shipping issues. "
    "Output ONLY valid JSON, no markdown."
)

_SCENARIO_GEN_USER_TEMPLATE = (
    "Generate a unique call centre scenario.\n"
    "{exclusion_clause}"
    "Return ONLY this JSON:\n"
    '{{\n'
    '  "issue_type": "<specific problem, 3-6 words>",\n'
    '  "customer_persona": "<personality, 2-5 words>",\n'
    '  "short_description": "<1-2 sentence summary for trainee>"\n'
    '}}'
)


# ── Emotional states (unchanged — already concise) ───────────────────────────

_STATES = {
    "FURIOUS":    ("You are at your breaking point. Nothing has helped so far.",
                   "Very frustrated. Demand answers. Short, sharp sentences."),
    "ANGRY":      ("You are upset and skeptical. You want real action, not talk.",
                   "Impatient, blunt. No small talk."),
    "FRUSTRATED": ("You are frustrated but engaging. The agent seems to be trying.",
                   "Firm but listening. You want specifics — when, how, what next."),
    "CALMING":    ("The agent has genuinely helped. You are less stressed now.",
                   "Cautious but cooperative. You may ask a practical follow-up."),
    "SATISFIED":  ("The issue is handled. You're ready to end the call.",
                   "Grateful and brief. You want to wrap up."),
}

_FALLBACKS = {
    "FURIOUS":    ["This is ridiculous — I need to speak to a supervisor.",
                   "I've been calling about this for DAYS and nothing's changed.",
                   "That's not what I asked. I need a real answer right now."],
    "ANGRY":      ["That's not good enough. What are you actually going to do?",
                   "I've heard that before — I need something concrete this time.",
                   "Look, just tell me when this is going to be fixed."],
    "FRUSTRATED": ["Okay, but when exactly? I need a specific date.",
                   "Fine — but if it's not done by tomorrow I'm calling back.",
                   "Alright, how long is this actually going to take?"],
    "CALMING":    ["Okay, that makes more sense. What happens next?",
                   "Alright, I can work with that. How long roughly?",
                   "Fine — just please follow through on that."],
    "SATISFIED":  ["Okay, that's all I needed. Thanks.",
                   "Finally. Thank you for sorting it out.",
                   "Good — that's what I was looking for."],
}


# ── Bad output detection (trimmed to highest-frequency violations) ────────────

_BAD_CUSTOMER = [
    # Role confusion — most common with small models
    "Agent:", "AGENT:", "As an AI", "Certainly!", "Of course!", "Absolutely!",
    "I can help you with", "Thank you for calling", "I understand your frustration",
    "I apologize for the inconvenience", "Let me help", "Happy to help",
    "Is there anything else", "How can I assist", "I'd be delighted",
    # Greetings
    "Hey there", "Hello there", "Hi there", "Good morning", "Good afternoon",
    # Meta / prompt leakage
    "In this scenario", "The agent should", "Training", "Simulation",
    "Your emotional state", "Tone:", "State:", "FURIOUS", "ANGRY",
    "FRUSTRATED", "CALMING", "SATISFIED", "Do NOT", "DO NOT",
    "1-2 sentences", "End of call", "scenario complete",
    # Customer acting as agent
    "Is there anything I can do", "anything I can do to help",
    "How can I assist you", "Let me know if I can help",
    "What can I do for you",
    # Agent-like courtesy
    "You're welcome", "My pleasure", "Great job", "Good job",
    "appreciate your efficiency", "your assistance",
    # Repetition markers
    "You already said", "already told", "already said",
]


# ── Promise vs Confirmation detection (unchanged — logic, not prompt) ─────────

_PROMISE_INDICATORS = [
    "will ", "i'll ", "we'll ", "going to ", "make sure",
    "ensure ", "let me ", "working to ", "promise ", "try to ",
    "as soon as", "once it", "once we", "when it", "when we",
    "you'll be notified", "rest assured",
    "will be ", "should be ", "can be ",
]

def _is_promise_not_confirmation(agent_input: str) -> bool:
    """Returns True if agent is promising (not confirming) a fix."""
    ai = agent_input.lower()
    return any(ind in ai for ind in _PROMISE_INDICATORS)

_RESOLVE_KW = [
    "your issue is resolved", "issue has been resolved", "issue is resolved",
    "problem is resolved", "problem has been resolved", "fully resolved",
    "issue is fixed", "issue has been fixed", "problem is fixed",
    "problem has been fixed", "it's been fixed", "it has been fixed",
    "it's fixed", "we've fixed", "we fixed", "i've fixed", "i fixed",
    "issue is solved", "problem is solved", "has been solved",
    "has been sorted", "all sorted", "it's sorted",
    "your account is restored", "has been refunded", "has been credited",
    "taken care of", "all set for you", "good to go",
    "its fixed", "its resolved", "its sorted",
    "problem fixed", "issue fixed", "all fixed", "all resolved", "all done",
    "issue resolved", "problem resolved", "issue solved", "problem solved",
]

# ── [FIX-5] Mood-band keywords for coherence check ───────────────────────────
# If the LLM output contains too many words from the WRONG mood band,
# we reject it and re-roll or use fallback.
_MOOD_BAND_CAPS_LIMIT = {
    "FURIOUS":    99,   # FURIOUS is allowed heavy caps
    "ANGRY":      3,    # max 3 caps words
    "FRUSTRATED": 2,
    "CALMING":    1,
    "SATISFIED":  0,
}


def _count_caps_words(text: str) -> int:
    """Count words that are entirely uppercase and >=2 chars."""
    return sum(1 for w in text.split() if w.isupper() and len(w) >= 2)


def _fix_caps(text: str) -> str:
    """[FIX-4] If >40% of alpha chars are uppercase, normalise to sentence case
    keeping at most 1 all-caps word (the longest one for emphasis)."""
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return text
    caps_ratio = sum(1 for c in alpha if c.isupper()) / len(alpha)
    if caps_ratio <= 0.40:
        return text

    # Find the best word to keep capitalised (longest all-caps word)
    words = text.split()
    caps_words = [(i, w) for i, w in enumerate(words) if w.isupper() and len(w) >= 2]
    keep_idx = -1
    if caps_words:
        keep_idx = max(caps_words, key=lambda x: len(x[1]))[0]

    result = []
    for i, w in enumerate(words):
        if i == keep_idx:
            result.append(w)  # keep this one in caps
        elif i == 0:
            result.append(w.capitalize())
        else:
            # Lowercase but preserve first letter of sentences
            result.append(w.lower())
    out = " ".join(result)
    # Re-capitalise after sentence endings
    out = re.sub(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), out)
    return out


class CustomerSimulator:
    # Issues that inherently make customers more upset (higher severity)
    _SEVERE_ISSUES = {
        "account hacked", "package stolen", "credit card declined",
        "account suspended", "device overheating", "data sync error",
        "double billing", "unrecognized charge",
    }

    def __init__(self, scenario, difficulty=1):
        self.scenario   = scenario
        self.history    = []
        self.difficulty = difficulty

        # Starting mood based on BOTH difficulty AND issue severity
        # Base: 7 - difficulty (diff1=6, diff5=2)
        # Severe issues: subtract 1 more
        base_mood = max(1, 7 - difficulty)
        if scenario.get("issue_type", "").lower() in self._SEVERE_ISSUES:
            base_mood = max(1, base_mood - 1)
        self.mood       = base_mood
        self.turn       = 0
        self._streak    = 0

        aggression = _AGGRESSION[min(difficulty, 5)]
        self._sys = _CUSTOMER_SYSTEM_TEMPLATE.format(
            issue_type=scenario["issue_type"],
            persona=scenario["customer_persona"],
            aggression=aggression,
        )

    def _state(self):
        if self.mood >= 9: return "SATISFIED"
        if self.mood >= 7: return "CALMING"
        if self.mood >= 5: return "FRUSTRATED"
        if self.mood >= 3: return "ANGRY"
        return "FURIOUS"

    def _shift(self, score):
        streak = self._streak
        if score >= 8:
            # Excellent reply — mood +1 (or +2 if 2 consecutive high scores)
            self._streak = max(0, streak) + 1
            self.mood = min(10, self.mood + (2 if self._streak >= 2 else 1))
        elif score >= 7:
            # Good reply — mood +1
            self._streak = max(0, streak) + 1
            self.mood = min(10, self.mood + 1)
        elif score <= 1:
            # Terrible reply — mood -1 (or -2 if 2 consecutive bad scores)
            self._streak = min(0, streak) - 1
            self.mood = max(0, self.mood - (2 if self._streak <= -2 else 1))
        elif score <= 3:
            # Poor reply — mood -1
            self._streak = min(0, streak) - 1
            self.mood = max(0, self.mood - 1)
        else:
            # Score 4-6: mediocre — mood stays, streak resets
            self._streak = 0

    def _is_greeting_only(self, text: str) -> bool:
        """Returns True if the agent's message is just a greeting with no substance."""
        t = text.lower().strip().rstrip("?!.")
        _GREETING_PHRASES = [
            "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
            "how can i help", "how may i help", "how can i assist",
            "how may i assist", "what can i do for you", "how are you",
            "welcome", "thank you for calling", "thanks for calling",
        ]
        # Check if the entire message is just greetings/pleasantries
        remaining = t
        for gp in sorted(_GREETING_PHRASES, key=len, reverse=True):
            remaining = remaining.replace(gp, "").strip()
        # Strip common filler words
        for filler in ["you", "sir", "madam", "ma'am", "today", "there",
                        "and", "the", "a", "to", "can", "may", "i", "we"]:
            remaining = remaining.replace(filler, "").strip()
        remaining = remaining.strip("?!., ")
        return len(remaining) < 3  # nothing substantive left

    def speak(self, agent_input: str, score: int = 5) -> str:
        self.turn += 1
        # Skip mood shift for greeting-only messages — they have no substance
        if not self._is_greeting_only(agent_input):
            self._shift(score)
        state = self._state()
        brief, tone = _STATES[state]
        all_past = [m["content"] for m in self.history if m["role"] == "assistant"]

        # Difficulty-aware hints
        is_easy = self.difficulty <= 2
        hint = (
            # Score >= 8
            ("The agent did a great job. Say thank you warmly and ask one "
             "polite follow-up question like 'how will I know it's done?'" if is_easy else
             "The agent did well. Acknowledge what they said — you're slightly "
             "less hostile but still cautious. Ask a practical follow-up.")
            if score >= 8
            else
            # Score >= 6
            ("The agent is helping. Respond positively and cooperatively. "
             "Say something like 'okay, that sounds reasonable' or ask ONE "
             "polite clarifying question." if is_easy else
             "The agent is trying. Respond to what they said — still firm "
             "but not as aggressive. You might push for specifics.")
            if score >= 6
            else
            # Score >= 4
            ("The agent's response was okay but vague. Politely ask for one "
             "more detail — do NOT be rude about it." if is_easy else
             "The agent's response was mediocre. React to what they said "
             "specifically — point out what's missing (timeline? action? details?).")
            if score >= 4
            else
            # Score < 4
            ("The agent didn't help much. Express mild disappointment and "
             "politely ask them to try again. Do NOT be aggressive." if is_easy else
             "The agent barely helped. React to their weak response — "
             "express disbelief, demand something concrete, or threaten to escalate.")
        )

        if self.turn == 1:
            _resolved_this_turn = False
            prompt = _CUSTOMER_TURN1_TEMPLATE.format(
                agent_input=agent_input,
                issue_type=self.scenario["issue_type"],
                tone=tone,
            )
        else:
            past4 = all_past[-4:]
            avoid_clause = (
                "Your previous replies (DO NOT repeat or closely paraphrase any of these): "
                + " | ".join(f'"{p[:60]}"' for p in past4)
                + "\n"
            ) if past4 else ""

            ah = " ".join(
                m["content"] for m in self.history if m["role"] == "user"
            ).lower()
            res_items = []
            if "replacement" in ah or "refund" in ah:
                res_items.append("replacement/refund confirmed")
            if "cancel" in ah:
                res_items.append("cancellation done")

            time_patterns = re.findall(
                r'\b(\d{1,2}\s*(?:am|pm)|\d{1,2}:\d{2}|within \d+\s*\w+|by \w+|'
                r'end of day|before \w+|in \d+ \w+|\d+ hours?|\d+ minutes?)\b',
                ah, re.IGNORECASE,
            )
            if time_patterns:
                res_items.append(
                    f"agent already gave timeline: "
                    f"{', '.join(set(time_patterns[-3:]))}"
                )
            resolution_history_clause = (
                f"Do NOT ask again about: {', '.join(res_items)}.\n"
                if res_items else ""
            )

            agent_inp_lower = agent_input.lower()
            agent_resolved = (
                any(kw in agent_inp_lower for kw in _RESOLVE_KW)
                and not _is_promise_not_confirmation(agent_input)
            )
            _resolved_this_turn = agent_resolved

            if agent_resolved:
                prompt = _CUSTOMER_RESOLVED_TEMPLATE.format(
                    agent_input=agent_input,
                )
            else:
                agent_msgs = [
                    m["content"] for m in self.history if m["role"] == "user"
                ]
                already_stated = []
                time_indicators = [
                    "5pm", "6pm", "7pm", "8pm", "9pm", "10pm", "11pm",
                    "am", "pm", "hour", "minute", "today", "tomorrow",
                    "end of day", "by noon", "within",
                ]
                for am in agent_msgs:
                    if any(t in am.lower() for t in time_indicators):
                        already_stated.append(
                            "A specific timeline was already provided — "
                            "do NOT ask for a timeline again."
                        )
                        break
                no_repeat_clause = (
                    already_stated[0] + "\n" if already_stated else ""
                )

                prompt = _CUSTOMER_GENERAL_TURN_TEMPLATE.format(
                    agent_input=agent_input,
                    brief=brief,
                    hint=hint,
                    tone=tone,
                    avoid_clause=avoid_clause,
                    resolution_history_clause=resolution_history_clause,
                    no_repeat_clause=no_repeat_clause,
                )

        # ── Output cleaning pipeline ──────────────────────────────────────
        def _clean(raw):
            for lbl in ["Customer:", "CUSTOMER:", "C:", "User:"]:
                if raw.startswith(lbl):
                    raw = raw[len(lbl):].strip()
            if " | " in raw:
                raw = raw.split(" | ")[0].strip()
            # Strip markdown formatting (bold, italic)
            raw = raw.replace("**", "").replace("__", "")
            raw = re.sub(r'\*([^*]+)\*', r'\1', raw)  # *italic*
            # Strip wrapping quotation marks (single and double)
            raw = raw.strip().strip("'\"").strip('\u2018\u2019\u201c\u201d').strip()
            _LEAKAGE = [
                "Your emotional state", "Push for specifics", "Tone:", "State:",
                "Do NOT", "DO NOT", "React as", "No labels", "Avoid:",
                "emotional state remains", "You are still frustrated",
                "agent said so far", "Don't re-raise", "1-2 sentences",
                # Prompt echo — LLM repeating the prompt template back
                "The agent just said", "The agent said", "Agent said",
                "agent just said", "the agent just", "The agent just",
                # Hallucination patterns — LLM leaking meta-instructions
                "End of call", "end of call", "End of conversation",
                "end of conversation", "call sequence", "interaction rules",
                "given constraints", "following rules", "following customer",
                "support interaction", "successfully completed",
                "scenario complete", "simulation end", "session end",
                "role-play", "roleplay", "as per instructions",
                "as instructed", "per the prompt", "according to rules",
                # Role hallucination — LLM appending role labels
                "Customer Support", "customer support",
                "Customer Service", "customer service",
            ]
            for leak in _LEAKAGE:
                idx = raw.find(leak)
                if idx > 0:
                    raw = raw[:idx].strip()
                elif idx == 0:
                    return ""
            for bad in _BAD_CUSTOMER:
                pos = raw.find(bad)
                if pos == 0:
                    return ""
                if pos > 0:
                    raw = raw[:pos].strip()
            raw = raw.strip().strip("|").strip()
            # Strip trailing hallucination tags — LLM appending role labels or meta-text
            _TRAILING_TAGS = [
                "Customer Support", "customer support", "Customer Service",
                "customer service", "Support Team", "support team",
                "Help Desk", "help desk", "Call Center", "call center",
                "Call Centre", "call centre", "Service Center",
                "Technical Support", "technical support",
                "Customer Care", "customer care",
            ]
            for tag in _TRAILING_TAGS:
                if raw.rstrip(".!?, ").endswith(tag):
                    raw = raw[:raw.rfind(tag)].rstrip(" .,;:!?-–—").strip()
            # Strip trailing question tags — "Right?", "Yeah?", "Huh?", "Okay?", "No?"
            raw = re.sub(
                r'[,;]?\s*\b(?:right|yeah|huh|okay|ok|no|correct|innit|eh|ya|sure)\s*\?\s*$',
                '.', raw, flags=re.IGNORECASE
            ).strip()
            # [FIX-4] Normalise excessive caps
            raw = _fix_caps(raw)
            return raw

        def _ok(raw):
            if not raw or len(raw) < 10:
                return False
            if any(b.lower() in raw.lower() for b in _BAD_CUSTOMER):
                return False
            # Reject trailing question tags that survived _clean
            raw_lower_stripped = raw.lower().rstrip(" .!?")
            if re.search(r'\b(right|yeah|huh|okay|ok|no|correct|innit|eh)\s*$',
                         raw_lower_stripped):
                return False
            # Reject trailing role hallucinations that survived _clean
            if any(tag in raw.lower() for tag in [
                "customer support", "customer service", "support team",
                "help desk", "call center", "call centre", "customer care",
                "technical support", "service center",
            ]):
                return False
            # [FIX-3] Reject if over 35 words
            if len(raw.split()) > 35:
                return False
            # [FIX-5] Check caps coherence with mood band
            max_caps = _MOOD_BAND_CAPS_LIMIT.get(state, 2)
            if state != "FURIOUS" and _count_caps_words(raw) > max_caps:
                return False
            # Block gratitude phrases UNLESS issue is resolved this turn
            if not _resolved_this_turn and state != "SATISFIED":
                _GRATITUDE = [
                    "thank you", "thanks", "appreciate", "grateful",
                    "that helps", "that's helpful", "very helpful",
                ]
                if any(g in raw.lower() for g in _GRATITUDE):
                    return False
            # Block timeline questions AND demands if agent already gave a timeline
            raw_lower = raw.lower()
            if self.turn > 1:
                agent_history_text = " ".join(
                    m["content"].lower() for m in self.history if m["role"] == "user"
                )
                timeline_already_given = any(t in agent_history_text for t in [
                    "hour", "minute", "tomorrow", "today", "by end",
                    "within", "am", "pm", "morning", "afternoon",
                    "evening", "business day", "day", "days",
                    "shortly", "soon", "asap", "right now", "immediately",
                ]) or re.search(r'\b\d+\s*(hours?|minutes?|days?|hrs?|mins?)\b',
                                agent_history_text)
                if timeline_already_given:
                    # Block questions about timeline
                    _TIMELINE_QUESTIONS = [
                        "when exactly", "when will", "how long",
                        "what time", "specific date", "specific time",
                        "when can i expect", "when is it", "when do i",
                        "when are you", "by when", "how soon",
                        # Additional patterns from real transcripts
                        "timeline now", "specific timeline",
                        "replacement timeline", "need a timeline",
                        "give me a date", "give me a time",
                        "tomorrow's okay", "tomorrow okay",
                        "is that okay", "acceptable pace",
                        "faster solution", "need a faster",
                        "a faster", "speed this up", "speed it up",
                        "hurry this", "can you hurry",
                        # Timeline re-ask patterns
                        "can it be done", "will it be done",
                        "be done before", "done before",
                        "definite timeline", "direct action",
                        "action steps",
                    ]
                    if any(tq in raw_lower for tq in _TIMELINE_QUESTIONS):
                        return False
                    # Block deadline DEMANDS that re-state/contradict the timeline
                    _DEADLINE_DEMANDS = [
                        "fix it by", "done by", "need it by",
                        "want it by", "before tomorrow", "by tomorrow",
                        "by next", "by friday", "by monday", "by tuesday",
                        "by wednesday", "by thursday", "by saturday", "by sunday",
                        "before end of", "do it by", "get it done by",
                        "i need this by", "better be done by",
                        # Additional from real transcripts
                        "need it done", "want it done",
                        "too long", "this long", "that long",
                        "can't wait", "won't wait", "not waiting",
                    ]
                    if any(dd in raw_lower for dd in _DEADLINE_DEMANDS):
                        return False
            # Block near-identical content to previous customer replies
            if all_past and len(all_past) >= 2:
                last_reply = all_past[-1].lower()
                # Reject if >60% of words overlap with the last reply
                raw_words = set(raw_lower.split())
                last_words = set(last_reply.split())
                if raw_words and last_words:
                    overlap = len(raw_words & last_words) / max(len(raw_words), 1)
                    if overlap > 0.6:
                        return False
            return True

        # ── Generation loop (3 attempts, escalating temperature) ──────────
        reply = ""
        for attempt in range(3):
            raw = _call_gen(
                [{"role": "system", "content": self._sys},
                 {"role": "user", "content": prompt}],
                max_tokens=60,                                                         # ← [FIX-3] was 80
                temp=0.82 + attempt * 0.05,
            )
            raw = _clean(raw)
            if not _ok(raw):
                continue
            # [FIX-2] Lowered threshold from 0.72 → 0.55
            if _sem_dup(raw, all_past, 0.55):
                continue
            reply = raw
            break
        if not reply:
            reply = random.choice(_FALLBACKS[state])

        # Split into proper sentences — also break on semicolons to prevent
        # multi-clause fragment replies like "I appreciate; how long; what now"
        for sep in ["! ", "? ", ". ", "; "]:
            reply = reply.replace(sep, sep[0] + "|||")
        parts = []
        for p in reply.split("|||"):
            p = p.strip()
            if not p:
                continue
            # Convert semicolon-terminated fragments into proper sentences
            if p.endswith(";"):
                p = p[:-1].strip() + "."
            parts.append(p)
        reply = " ".join(parts[:2]) if parts else reply
        if reply and reply[-1] not in ".!?":
            reply += "."

        # ── Post-generation safeguard: if resolution was detected, strip questions
        if _resolved_this_turn or state == "SATISFIED":
            # Remove any question sentences
            if "?" in reply:
                non_q_parts = [p.strip() for p in reply.replace("?", "?|||").split("|||")
                               if p.strip() and "?" not in p]
                if non_q_parts:
                    reply = " ".join(non_q_parts)
                else:
                    # All parts were questions — use a safe fallback
                    reply = random.choice(_FALLBACKS["SATISFIED"])
            # Gradual mood boost toward satisfied (not instant jump)
            self.mood = min(10, self.mood + 2)

        self.history.append({"role": "user", "content": agent_input})
        self.history.append({"role": "assistant", "content": reply})
        return reply


# ==============================================================================
#  GENERATOR 2 — LLM SCORER  [FIX-1: programmatic floor enforcement]
# ==============================================================================

def _enforce_scoring_floors(score: int, agent_input: str, issue: str) -> int:
    """Apply mandatory scoring floors AFTER LLM returns its score.
    Floors only RAISE scores — they never lower them. The LLM can give 0 or 10."""
    ai = agent_input.lower()
    words = ai.split()

    # Floor 0: Greeting-only messages → score exactly 5 (neutral — not penalized)
    # Greetings are normal call openers and should NOT be scored as "terrible"
    _GREETING_WORDS = [
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
        "how can i help", "how may i help", "how can i assist", "how may i assist",
        "what can i do for you", "welcome", "thank you for calling",
        "thanks for calling", "how are you",
    ]
    remaining = ai.strip().rstrip("?!.")
    for gw in sorted(_GREETING_WORDS, key=len, reverse=True):
        remaining = remaining.replace(gw, "").strip()
    for filler in ["you", "sir", "madam", "ma'am", "today", "there",
                    "and", "the", "a", "to", "can", "may", "i", "we"]:
        remaining = remaining.replace(filler, "").strip()
    remaining = remaining.strip("?!., ")
    if len(remaining) < 3:
        # It's a greeting — return 5 directly, skip all other floors
        return 5

    # Floor 1: Any reply >8 words that addresses the customer → minimum 1
    # (prevents 0 for replies that at least try to engage)
    if len(words) > 8:
        score = max(score, 1)

    # Floor 1b: Short but substantive timeline replies → minimum 5
    # Catches "2 hours", "by tomorrow", "30 minutes" etc.
    if len(words) <= 5 and re.search(
        r'\b(\d+\s*(hours?|minutes?|mins?|hrs?|days?|business days?)'
        r'|by\s+tomorrow|by\s+tonight|by\s+noon|end\s+of\s+day'
        r'|right\s+now|immediately|asap)\b', ai
    ):
        score = max(score, 5)

    # Floor 1c: Any reply containing a concrete timeline → minimum 5
    # Catches longer replies like "Yes sir it will be done before tomorrow"
    has_any_timeline = bool(re.search(
        r'\b(\d+\s*(hours?|minutes?|mins?|hrs?|days?|business days?|weeks?)'
        r'|before\s+tomorrow|by\s+tomorrow|before\s+end|end\s+of\s+day'
        r'|by\s+tonight|by\s+noon|within\s+\w+|this\s+afternoon'
        r'|this\s+evening|first\s+thing|right\s+now)\b', ai
    ))
    if has_any_timeline:
        score = max(score, 5)

    # Floor 2: Contains apology + offer to help → minimum 5
    has_apology = any(w in ai for w in [
        "sorry", "apologize", "apologise", "apologies", "apology",
        "regret", "forgive", "pardon",
    ])
    has_offer = any(phrase in ai for phrase in [
        "help", "assist", "fix", "resolve", "look into", "check",
        "investigate", "working on", "take care", "sort this",
        "get this", "handle", "address", "contact", "escalat",
        "reach out", "follow up", "look at", "review",
    ])
    if has_apology and has_offer:
        score = max(score, 5)

    # Floor 2b: Empathy (understand/frustration) + offer to help → minimum 5
    # (does NOT require an explicit apology word)
    has_empathy = has_apology or any(w in ai for w in [
        "understand", "frustrating", "frustration", "inconvenience",
        "concern", "difficult", "tough", "appreciate",
    ])
    if has_empathy and has_offer:
        score = max(score, 5)

    # Floor 3: Shows empathy AND asks for details → minimum 5
    asks_details = "?" in agent_input or any(phrase in ai for phrase in [
        "can you", "could you", "tell me", "let me know", "provide",
        "share", "what is your", "which", "when did", "do you have",
        "may i have", "would you",
    ])
    if has_empathy and asks_details:
        score = max(score, 5)

    # Floor 4: Names the issue + states an action → minimum 6
    # Flexible issue matching: strip hyphens, check individual words,
    # also check common abbreviations (wifi/wi-fi, etc.)
    issue_words_raw = issue.lower().replace("-", " ").replace("'", "").split()
    issue_words = [w for w in issue_words_raw if len(w) > 1]   # len > 1 (not > 2) to catch "wi", "fi"
    ai_normalized = ai.replace("-", " ")
    names_issue = any(iw in ai_normalized for iw in issue_words) if issue_words else False
    has_action = any(phrase in ai for phrase in [
        "i'm fixing", "i am fixing", "updating", "processing",
        "resolving", "working on", "escalating", "crediting",
        "refunding", "reversing", "sending", "arranging",
        "will be", "going to", "i'll", "i will", "right now",
        "immediately", "within", "contacting", "contact",
        "reaching out", "looking into", "checking", "reviewing",
        "scheduling", "arrange",
        # Expanded action verbs
        "we will", "we'll", "we are", "we're",
        "prioritize", "prioritise", "expedite", "expediting",
        "enhance", "enhancing", "ensure", "ensuring",
        "take steps", "steps to", "dispatch", "deploying",
        "replace", "replacing", "investigate", "investigating",
        "restore", "restoring", "stabilize", "stabilise",
        # Timeline-as-action verbs
        "will be done", "be done", "completed",
        "it will", "it'll", "done before", "done by",
        "resolved within", "fixed within", "sorted within",
    ])
    if names_issue and has_action:
        score = max(score, 6)

    # Floor 4b: Names issue + action + SPECIFIC timeline → minimum 7
    # Only matches concrete timelines, not vague words like "soon" or "today"
    has_specific_timeline = any(phrase in ai for phrase in [
        "within the hour", "within the next", "in the next",
        "by tomorrow", "by end of day", "by noon", "by tonight",
        "by monday", "by tuesday", "by wednesday", "by thursday", "by friday",
        "this afternoon", "this evening", "first thing tomorrow",
    ]) or re.search(r'\b(within\s+)?\d{1,2}\s*(am|pm|hours?|minutes?|business days?)\b', ai)
    if names_issue and has_action and has_specific_timeline:
        score = max(score, 7)

    # NOTE: Floor 4c (empathy+action+timeline without issue name) was REMOVED
    # because it was too broad — almost every reply triggered it, forcing all
    # scores to 7 and preventing the LLM from giving nuanced 4-6 scores.

    # Floor 5: Confirms resolution → minimum 8
    # Covers: formal ("is resolved"), terse ("problem fixed"), typo ("its fixed")
    resolution_phrases = [
        "is resolved", "has been resolved", "been fixed", "is fixed",
        "was fixed", "was resolved", "is solved", "was solved",
        "has been solved", "been solved",
        "all sorted", "been sorted", "taken care of", "all done",
        "been refunded", "been credited", "fully resolved",
        "completely resolved", "good to go",
        "i've fixed", "i fixed", "we've fixed", "we fixed",
        "it's fixed", "it's solved", "it's resolved",
        # apostrophe-less
        "its fixed", "its resolved", "its solved", "its sorted", "its done",
        "its been fixed", "its been resolved",
        # bare terse — "problem fixed", "issue fixed", "all fixed"
        "problem fixed", "issue fixed", "all fixed", "all resolved",
        # bare without "is" — catches "get your issue resolved"
        "issue resolved", "problem resolved", "issue solved", "problem solved",
        "get resolved", "get fixed",
    ]
    if any(phrase in ai for phrase in resolution_phrases):
        # Only give min 8 for CONFIRMED resolutions, not promises
        if not _is_promise_not_confirmation(agent_input):
            score = max(score, 8)

    # Floor 6: Near-perfect reply (empathy + issue + action + timeline + resolution) → min 9
    if (has_empathy and names_issue and has_action and has_specific_timeline
            and any(phrase in ai for phrase in resolution_phrases)
            and not _is_promise_not_confirmation(agent_input)):
        score = max(score, 9)

    return min(10, score)


def score_and_tip(agent_input, customer_said, issue):
    user_prompt = _SCORER_USER_TEMPLATE.format(
        issue=issue,
        customer_said=customer_said,
        agent_input=agent_input,
    )
    raw     = _call_json(
        [{"role": "system", "content": _SCORER_SYSTEM},
         {"role": "user", "content": user_prompt}],
        max_tokens=100,
    )
    cleaned = _strip(raw)

    def _parse(obj):
        return (
            max(0, min(10, int(obj.get("score", 5)))),
            str(obj.get("tip", "")).strip(),
            str(obj.get("reason", "")).strip(),
        )

    score, tip, reason = 5, "", ""

    # Layer 1: direct JSON parse
    try:
        obj = json.loads(cleaned)
        score, tip, reason = _parse(obj)
        if tip and reason:
            # [FIX-1] Enforce floors before returning
            score = _enforce_scoring_floors(score, agent_input, issue)
            return score, tip, reason
    except:
        pass

    # Layer 2: regex extract JSON block
    m = re.search(r'\{[^}]+\}', cleaned, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            score, tip, reason = _parse(obj)
            if tip:
                score = _enforce_scoring_floors(score, agent_input, issue)
                return score, tip, reason
        except:
            pass

    # Layer 3: individual field extraction
    sm = re.search(r'"?score"?\s*:\s*"?(10|[0-9])"?', cleaned)
    tm = re.search(r'"?tip"?\s*:\s*"([^"]{8,})"', cleaned)
    rm = re.search(r'"?reason"?\s*:\s*"([^"]{8,})"', cleaned)
    score  = int(sm.group(1)) if sm else 5
    tip    = tm.group(1).strip() if tm else ""
    reason = rm.group(1).strip() if rm else f"Score {score}/10 based on empathy and action shown."

    # Layer 4: fallback LLM call for the tip
    if not tip or len(tip) < 8:
        tip = _call_json(
            [{"role": "system",
              "content": "You are a call centre coach. Provide ONE specific, "
                         "actionable improvement suggestion in a single sentence. "
                         "No labels or prefixes."},
             {"role": "user",
              "content": f"Issue: {issue}\n"
                         f'Agent said: "{agent_input}"\n'
                         f"Score: {score}/10\n"
                         "What is the single most impactful thing this agent "
                         "could do differently?"}],
            max_tokens=35,
        ).split(".")[0].strip() + "."
    if not tip or len(tip) < 8:
        tip = "Name the exact issue and offer a concrete action with a timeline."

    # [FIX-1] Enforce floors on all code paths
    score = _enforce_scoring_floors(score, agent_input, issue)
    return max(0, min(10, score)), tip, reason


class AssistantCoach:
    def __init__(self, issue=""):
        self._issue = issue

    def evaluate(self, agent_input, cr):
        return score_and_tip(agent_input, cr, self._issue)


# ==============================================================================
#  GENERATOR 3 — IDEAL RESPONSES
# ==============================================================================

_BAD_IDEAL = [
    "Customer:", "CUSTOMER:", "As an AI", "I cannot", "I'm unable",
    "Note:", "Training", "Coaching", "User:",
]

_FB_IDEALS = {
    "positive": "I can see the {issue} on your account — I'm fixing it right "
                "now and you'll see the resolution within 24 hours.",
    "neutral":  "I've noted your {issue} and I'm escalating to our specialist "
                "team — you'll get an update within 2 business days.",
    "negative": "I'll look into your {issue} — give me a moment to check.",
}

_STATIC_BAN_WORDS = [
    "resolving", "fixing", "addressing", "apologies", "sorry", "understood",
]


class IdealGen:
    def __init__(self, scenario):
        self.scenario = scenario
        self.past     = []
        self._pos_past, self._neu_past, self._neg_past = [], [], []

    def _one(self, prompt, avoid, fallback, temp=0.85):
        def ok(t):
            if not (bool(t) and len(t) >= 15
                    and not any(b.lower() in t.lower() for b in _BAD_IDEAL)):
                return False
            # Reject responses that invent names/numbers
            _INVENTED = [
                "Mr.", "Mrs.", "Ms.", "Dr.", "Sir ", "Ma'am",
                "Order #", "Ticket #", "Ref #", "REF-", "Case #",
                "ID #", "TKT-", "ORD-",
            ]
            if any(inv in t for inv in _INVENTED):
                return False
            # Reject if contains a fabricated reference number pattern
            if re.search(r'[A-Z]{2,4}[-#]\d{3,}', t):
                return False
            return True

        for attempt in range(3):
            cand = _call_gen(
                [{"role": "system", "content": _IDEAL_SYSTEM},
                 {"role": "user", "content": prompt}],
                max_tokens=45,                                                         # ← [FIX-8] was 55
                temp=temp + attempt * 0.05,
            )
            # [FIX-6] Strip wrapping quotation marks
            cand = cand.strip().strip('"').strip("'").strip('\u201c').strip('\u201d')
            for pref in ["Agent:", "AGENT:", "Ideal:", "Response:"]:
                if cand.lower().startswith(pref.lower()):
                    cand = cand[len(pref):].strip()
            # [FIX-8] Hard 1-sentence trim
            for sep in [". ", "! ", "? "]:
                if sep in cand:
                    cand = cand[:cand.index(sep) + 1]
                    break
            cand = cand.strip()
            if ok(cand) and not _sem_dup(cand, avoid, 0.55):
                return cand
        return fallback

    def _build_ban_clause(self):
        all_past_ideals = (
            self._pos_past[-4:]
            + self._neu_past[-4:]
            + self._neg_past[-4:]
        )
        seen = []
        for r in all_past_ideals:
            if not r:
                continue
            words = r.strip().split()
            if words:
                # Ban first word
                w = words[0].lower().rstrip(",'\"")
                if w and w not in seen:
                    seen.append(w)
            if len(words) >= 3:
                # Ban first 3-word phrase
                phrase = " ".join(words[:3]).lower().rstrip(",'\".")
                if phrase and phrase not in seen:
                    seen.append(phrase)
            # Ban key action verbs/phrases from the FULL response
            r_lower = r.lower()
            _action_extracts = [
                "right now", "immediately", "within the hour",
                "within 24 hours", "by end of day", "as soon as possible",
                "looking into", "checking on", "pulling up", "escalating",
                "let me check", "let me look", "let me see",
                "i can see", "i'm fixing", "i'm resolving",
                "i apologize", "i'm sorry", "my apologies",
                "don't worry", "no worries", "bear with me",
                "within 2", "within 5", "within 10",
                "by tomorrow", "by tonight", "shortly",
            ]
            for act in _action_extracts:
                if act in r_lower and act not in seen:
                    seen.append(act)
        for bw in _STATIC_BAN_WORDS:
            if bw not in seen:
                seen.append(bw)
        return (
            f"Do NOT use any of these words or phrases: {', '.join(seen[:20])}.\n"
            if seen else ""
        )

    def generate(self, customer_said, agent_history):
        issue   = self.scenario["issue_type"]
        persona = self.scenario["customer_persona"]

        # Build FULL conversation context (both sides) for ideal responses
        # so they can see what was discussed and respond appropriately
        convo_lines = []
        for m in agent_history[-8:]:  # last 4 exchanges (8 messages)
            role_label = "Agent" if m["role"] == "user" else "Customer"
            convo_lines.append(f"  {role_label}: {m['content'][:80]}")
        conversation = "\n".join(convo_lines) if convo_lines else "  (conversation just started)"

        # Agent-only history for backward compat
        at   = [m["content"] for m in agent_history if m["role"] == "user"]
        said = "; ".join(at[-3:]) if at else "nothing yet"

        all_past_ideals = (
            self._pos_past[-4:]
            + self._neu_past[-4:]
            + self._neg_past[-4:]
        )
        ban_clause = self._build_ban_clause()
        fb = {k: v.format(issue=issue) for k, v in _FB_IDEALS.items()}

        # [FIX-7] Feed last 2 positive outputs into the positive prompt
        prev_pos_str = " / ".join(
            f'"{p[:50]}"' for p in self._pos_past[-2:] if p
        ) or "none yet"

        pos = self._one(
            _IDEAL_POSITIVE_TEMPLATE.format(
                issue=issue, persona=persona,
                customer_said=customer_said, conversation=conversation,
                ban_clause=ban_clause,
                prev_positives=prev_pos_str,
            ),
            all_past_ideals, fb["positive"], 0.80,
        )
        neu = self._one(
            _IDEAL_NEUTRAL_TEMPLATE.format(
                issue=issue, persona=persona,
                customer_said=customer_said, conversation=conversation,
                ban_clause=ban_clause,
            ),
            all_past_ideals, fb["neutral"], 0.88,
        )
        neg = self._one(
            _IDEAL_NEGATIVE_TEMPLATE.format(
                issue=issue, persona=persona,
                customer_said=customer_said, conversation=conversation,
                ban_clause=ban_clause,
            ),
            all_past_ideals, fb["negative"], 0.93,
        )

        self._pos_past = (self._pos_past + [pos])[-5:]
        self._neu_past = (self._neu_past + [neu])[-5:]
        self._neg_past = (self._neg_past + [neg])[-5:]
        self.past.extend([pos, neu, neg])
        return {"positive": pos, "neutral": neu, "negative": neg, "ideal": neu}


# ==============================================================================
#  GENERATOR 4 — REPORT
# ==============================================================================

class ReportGen:
    def __init__(self, scenario):
        self.scenario = scenario

    def generate(self, turn_log):
        if not turn_log:
            return {"error": "No turns recorded."}

        scores = [t["score"] for t in turn_log]
        avg    = round(sum(scores) / len(scores), 1)
        best   = max(turn_log, key=lambda t: t["score"])
        worst  = min(turn_log, key=lambda t: t["score"])
        mid    = len(scores) // 2
        fh     = sum(scores[:mid]) / mid if mid else avg
        sh     = sum(scores[mid:]) / (len(scores) - mid) if len(scores) - mid else avg
        trend  = (
            "improving" if sh > fh + 0.5
            else "declining" if sh < fh - 0.5
            else "consistent"
        )
        resolved = scores[-1] >= 9 or (len(scores) >= 3 and sum(scores[-3:]) / 3 >= 7)

        transcript = "".join(
            f"Turn {t['turn']} — Score {t['score']}/10\n"
            f"  Agent: {t['agent']}\n"
            f"  Customer: {t['customer']}\n\n"
            for t in turn_log
        )

        user_prompt = _REPORT_USER_TEMPLATE.format(
            persona=self.scenario["customer_persona"],
            issue=self.scenario["issue_type"],
            outcome="RESOLVED" if resolved else "NOT RESOLVED",
            avg=avg, trend=trend,
            best_turn=best["turn"], best_score=best["score"],
            best_agent=best["agent"],
            worst_turn=worst["turn"], worst_score=worst["score"],
            worst_agent=worst["agent"],
            transcript=transcript,
        )

        text = _call_gen(
            [{"role": "system", "content": _REPORT_SYSTEM},
             {"role": "user", "content": user_prompt}],
            max_tokens=350, temp=0.5,
        )
        return {
            "average_score": avg, "total_turns": len(turn_log), "trend": trend,
            "best_turn": best, "worst_turn": worst, "report_text": text,
        }


# ==============================================================================
#  SESSION + WIN/LOSS
# ==============================================================================

MAX_TURNS = 20
sessions: dict = {}

_MOOD_LABELS = {
    range(0, 3):  "Furious",
    range(3, 5):  "Angry",
    range(5, 7):  "Frustrated",
    range(7, 9):  "Calming",
    range(9, 11): "Satisfied",
}


def _mood_label(mood):
    return next((v for r, v in _MOOD_LABELS.items() if mood in r), "Angry")


_SAT = [
    "thank you", "appreciate it", "that's all i needed", "that's all i wanted",
    "thank you so much", "sounds good", "that works for me", "problem solved",
]

_WIN_KW = [
    "your issue is resolved", "your problem is resolved",
    "issue has been resolved", "issue is resolved", "issue was resolved",
    "problem has been fixed", "problem is fixed", "problem was fixed",
    "issue is fixed", "issue has been fixed", "issue was fixed",
    "your issue is fixed", "your problem is fixed",
    "it's been fixed", "it has been fixed", "it is fixed", "it's fixed",
    "fully resolved", "completely resolved",
    "your account is restored", "your card is back online",
    "has been sorted", "has been refunded",
    "taken care of for you", "all done for you",
    "your issue is solved", "issue is solved", "problem is solved",
    "has been solved", "it's solved", "it is solved",
    "all sorted", "good to go",
    # apostrophe-less
    "its fixed", "its resolved", "its solved", "its sorted", "its done",
    "its been fixed", "its been resolved", "its been sorted",
    # bare terse
    "problem fixed", "issue fixed", "all fixed", "all resolved", "all done",
    # bare without "is"
    "issue resolved", "problem resolved", "issue solved", "problem solved",
    "get resolved", "get fixed", "get sorted",
]


def _win_loss(s):
    scores = [t["score"] for t in s["turn_log"]]
    mood   = s["sim"].mood
    n      = len(scores)
    if n == 0:
        return None, False

    # Direct resolution: agent used resolution keyword + mood is high + NOT a promise
    if n >= 1 and mood >= 8:
        last_agent = s["turn_log"][-1]["agent"]
        if (any(kw in last_agent.lower() for kw in _WIN_KW)
                and not _is_promise_not_confirmation(last_agent)):
            return "win", True

    ra = sum(scores[-5:]) / min(n, 5)
    if n >= 2 and mood >= 8 and ra >= 5.5 and scores[-1] >= 6:
        return "win", True

    last = s["turn_log"][-1]["customer"].lower() if s["turn_log"] else ""
    if any(p in last for p in _SAT) and mood >= 7 and n >= 2:
        return "win", True

    if n >= 4 and all(sc <= 1 for sc in scores[-4:]):
        return "loss", False

    if s["turn_count"] >= MAX_TURNS:
        return "loss", False

    return None, False


# ==============================================================================
#  BACKGROUND WORKER
# ==============================================================================

# [FIX-9] Resolution bypass keywords — tightened, no bare single words
_RESOLUTION_BYPASS_KW = [
    "issue is resolved", "issue has been resolved", "issue was resolved",
    "problem is fixed", "problem has been fixed", "problem was fixed",
    "problem is resolved", "problem has been resolved", "problem was resolved",
    "issue is fixed", "issue has been fixed", "issue was fixed",
    "it's been fixed", "it has been fixed", "it is fixed", "it's fixed",
    "that's been sorted", "has been sorted",
    "has been resolved", "has been refunded", "has been credited",
    "fully resolved", "completely resolved", "all sorted for you",
    "all done for you", "taken care of for you", "your account is restored",
    "your card is back online", "been processed and reversed",
    "good to go now", "good to go",
    "issue is solved", "issue has been solved", "issue was solved",
    "problem is solved", "problem has been solved", "problem was solved",
    "it's solved", "it is solved", "has been solved",
    "i've fixed", "i fixed", "we've fixed", "we fixed",
    # apostrophe-less
    "its fixed", "its resolved", "its solved", "its sorted", "its done",
    "its been fixed", "its been resolved", "its been sorted",
    "thats fixed", "thats sorted", "thats resolved",
    # bare terse
    "problem fixed", "issue fixed", "all fixed", "all resolved", "all done",
    # bare without "is"
    "issue resolved", "problem resolved", "issue solved", "problem solved",
    "get resolved", "get fixed", "get sorted",
]


def _do_work(session_id: str, agent_input: str):
    try:
        s = sessions[session_id]
        s["turn_count"] += 1
        turn = s["turn_count"]
        print(f"[T{turn}] processing...", flush=True)

        import time as _time
        for _ in range(80):
            if session_id in _ws_queues:
                break
            _time.sleep(0.1)
        else:
            print(f"[T{turn}] WS never connected — aborting", flush=True)
            return

        _ws_send(session_id, {"type": "thinking", "turn": turn})

        last_customer = next(
            (m["content"] for m in reversed(s["sim"].history)
             if m["role"] == "assistant"),
            "No previous reply.",
        )

        # Resolution bypass — only on CONFIRMED fixes (not promises)
        agent_resolved_flag = (
            any(kw in agent_input.lower() for kw in _RESOLUTION_BYPASS_KW)
            and not _is_promise_not_confirmation(agent_input)
        )

        if agent_resolved_flag:
            score  = 9
            tip    = "Excellent resolution — the agent confirmed the fix clearly and confidently."
            reason = "Agent explicitly confirmed the issue is resolved — this is the ideal outcome."
            # Gradual mood boost, not instant jump
            s["sim"].mood    = min(10, s["sim"].mood + 2)
            s["sim"]._streak = 1
        else:
            score, tip, reason = s["coach"].evaluate(agent_input, last_customer)

        customer_reply = s["sim"].speak(agent_input, score=score)
        ideals         = s["ideal"].generate(customer_reply, s["sim"].history)

        s["turn_log"].append({
            "turn": turn, "agent": agent_input,
            "customer": customer_reply, "score": score, "tip": tip,
        })

        outcome, resolved = _win_loss(s)
        if resolved:
            db.bump(s["scenario"]["id"])

        mood = s["sim"].mood
        print(f"[T{turn}] done — score={score} mood={mood}", flush=True)

        _ws_send(session_id, {
            "type":            "result",
            "customer_reply":  customer_reply,
            "score":           score,
            "tip":             tip,
            "reason":          reason,
            "resolved":        resolved,
            "ideal":           ideals["ideal"],
            "ideals":          ideals,
            "outcome":         outcome,
            "customer_mood":   mood,
            "mood_label":      _mood_label(mood),
            "turns_remaining": max(0, MAX_TURNS - s["turn_count"]),
        })

    except Exception as e:
        import traceback
        print(f"[ERROR] {traceback.format_exc()}", flush=True)
        _ws_send(session_id, {"type": "error", "detail": str(e)})


# ── Feature 1: LLM scenario generation tracking ──────────────────────────────
_generated_scenario_history: list = []   # keeps last N generated to avoid repeats


def _generate_scenario_via_llm() -> dict:
    """Call LLM to generate a unique scenario. Returns dict with
    issue_type, customer_persona, short_description."""
    # Build exclusion clause from recent history
    recent = _generated_scenario_history[-10:]
    if recent:
        issues_seen = [s.get("issue_type", "") for s in recent]
        exclusion = (
            "Do NOT generate any of these issues (already used): "
            + ", ".join(f'"{i}"' for i in issues_seen if i)
            + "\n"
        )
    else:
        exclusion = ""

    raw = _call_json(
        [{"role": "system", "content": _SCENARIO_GEN_SYSTEM},
         {"role": "user", "content": _SCENARIO_GEN_USER_TEMPLATE.format(
             exclusion_clause=exclusion)}],
        max_tokens=150,
    )
    cleaned = _strip(raw)

    # Parse JSON with fallback
    parsed = None
    try:
        parsed = json.loads(cleaned)
    except Exception:
        m = re.search(r'\{[^}]+\}', cleaned, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except Exception:
                pass

    if not parsed or "issue_type" not in parsed:
        # Fallback: generate a random combo from existing lists
        parsed = {
            "issue_type": random.choice(ISSUES),
            "customer_persona": random.choice(PERSONAS),
            "short_description": (
                f"A {random.choice(PERSONAS).lower()} customer calls about "
                f"a {random.choice(ISSUES).lower()} issue."
            ),
        }

    # Ensure all fields exist
    parsed.setdefault("customer_persona", random.choice(PERSONAS))
    parsed.setdefault("short_description",
                      f"{parsed['customer_persona']} customer with {parsed['issue_type']} problem.")

    # Track to avoid future repeats
    _generated_scenario_history.append(parsed)
    if len(_generated_scenario_history) > 50:
        _generated_scenario_history[:] = _generated_scenario_history[-30:]

    return parsed


# ── Feature 2: Edit message background worker ────────────────────────────────

def _do_edit_work(session_id: str, turn_number: int, new_agent_input: str):
    """Replays a single turn with an edited agent message.
    Truncates all turns after the edited one and pushes fresh results."""
    try:
        s = sessions[session_id]
        print(f"[EDIT T{turn_number}] processing...", flush=True)

        # Wait up to 8s for WebSocket
        import time as _time
        for _ in range(80):
            if session_id in _ws_queues:
                break
            _time.sleep(0.1)
        else:
            print(f"[EDIT T{turn_number}] WS never connected — aborting", flush=True)
            return

        _ws_send(session_id, {"type": "thinking", "turn": turn_number})

        # ── Roll back session state to just before the edited turn ────────
        # 1. Truncate turn_log: keep only turns before the edited one
        s["turn_log"] = [t for t in s["turn_log"] if t["turn"] < turn_number]
        s["turn_count"] = turn_number - 1

        # 2. Rewind simulator history to match
        #    Each turn adds 2 entries: user (agent msg) + assistant (customer reply)
        #    So for turn N, we keep (N-1)*2 entries
        entries_to_keep = (turn_number - 1) * 2
        s["sim"].history = s["sim"].history[:entries_to_keep]
        s["sim"].turn = turn_number - 1

        # 3. Recalculate mood from surviving scores
        #    Reset to initial mood, then replay all surviving score shifts
        base_mood = max(1, 7 - s["difficulty"])
        if s["scenario"].get("issue_type", "").lower() in CustomerSimulator._SEVERE_ISSUES:
            base_mood = max(1, base_mood - 1)
        s["sim"].mood = base_mood
        s["sim"]._streak = 0
        for t in s["turn_log"]:
            s["sim"]._shift(t["score"])

        # 4. Rewind ideal generator past history
        #    Each turn adds 3 entries (pos, neu, neg) to self.past
        s["ideal"].past = s["ideal"].past[:(turn_number - 1) * 3]
        keep_ideal = turn_number - 1
        s["ideal"]._pos_past = s["ideal"]._pos_past[:keep_ideal]
        s["ideal"]._neu_past = s["ideal"]._neu_past[:keep_ideal]
        s["ideal"]._neg_past = s["ideal"]._neg_past[:keep_ideal]

        # ── Now replay the edited turn (same logic as _do_work) ───────────
        s["turn_count"] += 1
        turn = s["turn_count"]  # should equal turn_number

        last_customer = next(
            (m["content"] for m in reversed(s["sim"].history)
             if m["role"] == "assistant"),
            "No previous reply.",
        )

        # Resolution bypass check — only confirmed fixes, not promises
        agent_resolved_flag = (
            any(kw in new_agent_input.lower() for kw in _RESOLUTION_BYPASS_KW)
            and not _is_promise_not_confirmation(new_agent_input)
        )

        if agent_resolved_flag:
            score  = 9
            tip    = "Excellent resolution — the agent confirmed the fix clearly and confidently."
            reason = "Agent explicitly confirmed the issue is resolved — this is the ideal outcome."
            s["sim"].mood    = min(10, s["sim"].mood + 2)
            s["sim"]._streak = 1
        else:
            score, tip, reason = s["coach"].evaluate(new_agent_input, last_customer)

        customer_reply = s["sim"].speak(new_agent_input, score=score)
        ideals         = s["ideal"].generate(customer_reply, s["sim"].history)

        s["turn_log"].append({
            "turn": turn, "agent": new_agent_input,
            "customer": customer_reply, "score": score, "tip": tip,
        })

        outcome, resolved = _win_loss(s)
        if resolved:
            db.bump(s["scenario"]["id"])

        mood = s["sim"].mood
        print(f"[EDIT T{turn}] done — score={score} mood={mood}", flush=True)

        _ws_send(session_id, {
            "type":            "edit_result",
            "edited_turn":     turn_number,
            "customer_reply":  customer_reply,
            "score":           score,
            "tip":             tip,
            "reason":          reason,
            "resolved":        resolved,
            "ideal":           ideals["ideal"],
            "ideals":          ideals,
            "outcome":         outcome,
            "customer_mood":   mood,
            "mood_label":      _mood_label(mood),
            "turns_remaining": max(0, MAX_TURNS - s["turn_count"]),
        })

    except Exception as e:
        import traceback
        print(f"[EDIT ERROR] {traceback.format_exc()}", flush=True)
        _ws_send(session_id, {"type": "error", "detail": str(e)})


# ==============================================================================
#  API ENDPOINTS
# ==============================================================================

class MessageRequest(BaseModel):
    session_id:  str
    agent_input: str


class ScenarioRequest(BaseModel):
    difficulty: int = 1


class GenerateScenarioRequest(BaseModel):
    difficulty: int = 1


class EditMessageRequest(BaseModel):
    session_id:  str
    turn_number: int
    new_agent_input: str


class RedoRequest(BaseModel):
    session_id: str


class CustomScenarioRequest(BaseModel):
    issue_type:   str
    persona:      str = ""
    description:  str = ""
    difficulty:   int = 1


@app.get("/health")
def health():
    try:
        _http.get(f"{LM_STUDIO_URL}/v1/models", timeout=3)
        lm_ok = True
    except:
        lm_ok = False
    return {"status": "ok", "lm_studio": "connected" if lm_ok else "unreachable"}


@app.get("/ping")
def ping():
    return {"pong": True}


def _create_session(difficulty: int):
    difficulty = max(1, min(5, difficulty))
    scenario   = random.choice(db.load())
    sid        = str(uuid.uuid4())
    sessions[sid] = {
        "sim":        CustomerSimulator(scenario, difficulty=difficulty),
        "coach":      AssistantCoach(scenario["issue_type"]),
        "ideal":      IdealGen(scenario),
        "report_gen": ReportGen(scenario),
        "scenario":   scenario, "difficulty": difficulty,
        "turn_log":   [], "turn_count": 0,
    }
    starting_mood = sessions[sid]["sim"].mood
    return {"session_id": sid, "scenario": scenario, "difficulty": difficulty,
            "starting_mood": starting_mood}


@app.post("/scenario")
def post_scenario(req: ScenarioRequest = None):
    return _create_session(req.difficulty if req else 1)


@app.get("/scenario")
def get_scenario(difficulty: int = 1):
    return _create_session(difficulty)


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    loop = asyncio.get_running_loop()
    q = asyncio.Queue()
    _ws_queues[session_id] = q
    _ws_loops[session_id]  = loop
    print(f"[WS] connected: {session_id[:8]}", flush=True)

    try:
        while True:
            recv_task = asyncio.create_task(websocket.receive_text())
            send_task = asyncio.create_task(q.get())
            done, pending = await asyncio.wait(
                [recv_task, send_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
                try:
                    await task
                except:
                    pass
            for task in done:
                if task is recv_task:
                    data = task.result()
                    if data == "ping":
                        await websocket.send_text(json.dumps({"type": "pong"}))
                elif task is send_task:
                    msg = task.result()
                    await websocket.send_text(msg)
                    print(f"[WS] sent to {session_id[:8]}: {msg[:80]}", flush=True)

    except WebSocketDisconnect:
        print(f"[WS] disconnected: {session_id[:8]}", flush=True)
    except Exception as e:
        print(f"[WS] error: {e}", flush=True)
    finally:
        _ws_queues.pop(session_id, None)
        _ws_loops.pop(session_id, None)


@app.post("/message")
def post_message(req: MessageRequest):
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    threading.Thread(
        target=_do_work,
        args=(req.session_id, req.agent_input),
        daemon=True,
    ).start()
    return {"status": "processing"}


@app.get("/report/{session_id}")
def get_report(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    s = sessions[session_id]
    if not s["turn_log"]:
        raise HTTPException(status_code=400, detail="No turns recorded yet")
    return {
        "session_id": session_id,
        "scenario":   s["scenario"],
        "report":     s["report_gen"].generate(s["turn_log"]),
    }


# ==============================================================================
#  FEATURE 1 — LLM-BASED SCENARIO GENERATOR
# ==============================================================================

@app.post("/generate-scenario")
def generate_scenario(req: GenerateScenarioRequest = None):
    """Calls the LLM to generate a completely new, random scenario each time.
    Returns the full scenario (for session creation) and a short UI description."""
    difficulty = max(1, min(5, req.difficulty if req else 1))
    generated  = _generate_scenario_via_llm()

    # Build a scenario object compatible with the existing session structure
    scenario = {
        "id":               len(db.data) + len(_generated_scenario_history) + 100,
        "customer_persona": generated["customer_persona"],
        "issue_type":       generated["issue_type"],
        "difficulty":       difficulty,
    }

    # Create a full session with this generated scenario
    sid = str(uuid.uuid4())
    sessions[sid] = {
        "sim":        CustomerSimulator(scenario, difficulty=difficulty),
        "coach":      AssistantCoach(scenario["issue_type"]),
        "ideal":      IdealGen(scenario),
        "report_gen": ReportGen(scenario),
        "scenario":   scenario,
        "difficulty": difficulty,
        "turn_log":   [],
        "turn_count": 0,
    }

    return {
        "session_id":        sid,
        "scenario":          scenario,
        "difficulty":        difficulty,
        "short_description": generated["short_description"],
        "starting_mood":     sessions[sid]["sim"].mood,
    }


# ==============================================================================
#  FEATURE 1b — USER-DEFINED CUSTOM SCENARIO
# ==============================================================================

@app.post("/custom-scenario")
def custom_scenario(req: CustomScenarioRequest):
    """Create a session with a user-defined scenario. The user types the issue
    and optionally a persona and description. The customer agent will converse
    based on this custom scenario."""
    difficulty = max(1, min(5, req.difficulty))

    # Use provided persona or pick a random one
    persona = req.persona.strip() if req.persona and req.persona.strip() else random.choice(PERSONAS)

    scenario = {
        "id":               9000 + random.randint(1, 9999),
        "customer_persona": persona,
        "issue_type":       req.issue_type.strip(),
        "difficulty":       difficulty,
    }

    # Create a full session
    sid = str(uuid.uuid4())
    sessions[sid] = {
        "sim":        CustomerSimulator(scenario, difficulty=difficulty),
        "coach":      AssistantCoach(scenario["issue_type"]),
        "ideal":      IdealGen(scenario),
        "report_gen": ReportGen(scenario),
        "scenario":   scenario,
        "difficulty": difficulty,
        "turn_log":   [],
        "turn_count": 0,
    }

    return {
        "session_id":        sid,
        "scenario":          scenario,
        "difficulty":        difficulty,
        "short_description": req.description.strip() if req.description else
                             f"{persona} customer with a {req.issue_type.strip()} problem.",
        "starting_mood":     sessions[sid]["sim"].mood,
    }


# ==============================================================================
#  FEATURE 2 — EDIT PREVIOUS AGENT REPLY
# ==============================================================================

@app.post("/edit-message")
def edit_message(req: EditMessageRequest):
    """Edit a previously sent agent reply. Truncates all downstream turns and
    regenerates the customer response for the edited turn. Result pushed via WS."""
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    s = sessions[req.session_id]
    if req.turn_number < 1 or req.turn_number > s["turn_count"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid turn number. Valid range: 1-{s['turn_count']}",
        )
    if not req.new_agent_input.strip():
        raise HTTPException(status_code=400, detail="Agent input cannot be empty")

    threading.Thread(
        target=_do_edit_work,
        args=(req.session_id, req.turn_number, req.new_agent_input.strip()),
        daemon=True,
    ).start()
    return {
        "status":      "processing",
        "editing_turn": req.turn_number,
        "message":     "Downstream turns will be regenerated. Result pushed via WebSocket.",
    }


# ==============================================================================
#  FEATURE 3 — REDO CONVERSATION (SAME SCENARIO)
# ==============================================================================

@app.post("/redo")
def redo_conversation(req: RedoRequest):
    """Restart the current conversation from scratch using the SAME scenario.
    All turns are cleared; a fresh session state is created."""
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    old = sessions[req.session_id]
    scenario   = old["scenario"]
    difficulty = old["difficulty"]

    # Reset all session state — same scenario, fresh everything
    sessions[req.session_id] = {
        "sim":        CustomerSimulator(scenario, difficulty=difficulty),
        "coach":      AssistantCoach(scenario["issue_type"]),
        "ideal":      IdealGen(scenario),
        "report_gen": ReportGen(scenario),
        "scenario":   scenario,
        "difficulty": difficulty,
        "turn_log":   [],
        "turn_count": 0,
    }

    starting_mood = sessions[req.session_id]["sim"].mood

    # Notify via WebSocket if connected
    _ws_send(req.session_id, {
        "type":           "redo",
        "message":        "Conversation restarted with the same scenario.",
        "scenario":       scenario,
        "starting_mood":  starting_mood,
    })

    return {
        "status":         "restarted",
        "session_id":     req.session_id,
        "scenario":       scenario,
        "difficulty":     difficulty,
        "starting_mood":  starting_mood,
        "message":        "Conversation reset. Same scenario, fresh start.",
    }