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
#  [FIX-10] All prompts changed from "phone call" to "chat conversation"
#  [FIX-11] Customer system prompt rewritten — airtight anti-hallucination
#  [FIX-12] Scorer prompt rewritten — requires specific reasons
#  [FIX-13] Turn-1 fallback uses problem statement, not mood-based
#  [FIX-14] Reason fallback is score-range-specific, not generic
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
model_id      = os.environ.get("LM_MODEL_ID", "qwen/qwen3-4b-instruct-2507")

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
#  SYSTEM PROMPT TEMPLATES  [FIX-10/11/12 — rewritten for chat, airtight]
# ==============================================================================

"""
Optimized prompts for small-to-medium LLMs (4B–12B).

Design principles:
  • Max ~350 tokens per system prompt
  • 5-8 rules max — prioritize the ones the model actually violates
  • Use short imperative sentences, not nested bullet trees
  • Put the MOST IMPORTANT rule FIRST (primacy bias)
  • Avoid "do NOT" lists longer than 5 items — use positive framing instead
  • Examples > abstract rules
  • STRICT role boundaries: each generator has ZERO knowledge of others
"""

# ── Aggression levels ─────────────────────────────────────────────────────────

_AGGRESSION = [
    "",
    # Difficulty 1
    "You are a patient, easygoing customer. You give the agent time to help. "
    "You accept reasonable answers. A simple apology with a clear plan makes you happy.",
    # Difficulty 2
    "You are mildly annoyed but reasonable. You want clarity and a plan. "
    "You appreciate effort. A genuine apology and concrete action calms you.",
    # Difficulty 3
    "You are frustrated and skeptical. You need the agent to show they understand "
    "your problem. Vague promises annoy you, but real action earns your trust. "
    "You CAN be won over with genuine effort.",
    # Difficulty 4
    "You are angry and impatient. You push back on vague answers and demand specifics. "
    "You might threaten to cancel, but if the agent shows genuine competence and "
    "takes real ownership, you grudgingly cooperate. Hard to please but not impossible.",
    # Difficulty 5
    "You are furious. You are hostile and sarcastic. You threaten legal action and "
    "demand a supervisor. BUT if the agent stays calm, takes full ownership, gives "
    "a concrete fix with a timeline, and shows genuine empathy, you will eventually "
    "calm down. Even the angriest customer can be won over by exceptional service.",
]


_CUSTOMER_SYSTEM_TEMPLATE = (
    "You are a real person chatting with customer support about a problem you have.\n\n"

    "YOUR PROBLEM: '{issue_type}'\n"
    "YOUR PERSONALITY: {persona}\n"
    "YOUR ATTITUDE: {aggression}\n\n"

    "HOW TO BEHAVE — think and respond like a real human customer:\n"
    "- If the agent helps you well, acknowledge it and cooperate.\n"
    "- If the agent is vague or unhelpful, push back and express frustration.\n"
    "- If the agent asks you a question, answer it with relevant details.\n"
    "- If the agent takes a concrete action (checking your account, contacting "
    "a team, sending an email), acknowledge that it happened.\n"
    "- If the agent gives you a timeline, respond to THAT timeline.\n"
    "- If the agent says your issue is fixed, verify it makes sense before accepting.\n"
    "- Each reply should say something NEW. Don't repeat what you already said.\n\n"

    "CONVERSATION FLOW — a real conversation progresses naturally:\n"
    "Turn 1: State your problem clearly.\n"
    "Next turns: React to the agent -> provide info they asked for -> respond to "
    "their actions -> ask practical follow-up questions -> accept resolution when "
    "it is genuine.\n"
    "Do not get stuck repeating the same demand. Move the conversation forward.\n\n"

    "OUTPUT: Write ONLY your words as the customer. 1-3 sentences, 10-30 words. "
    "Sound like a real person typing — natural, conversational, human.\n"
    "No labels, no 'Customer:', no agent phrases like 'certainly' or 'absolutely'.\n"
)


# ── Customer turn 1 ──────────────────────────────────────────────────────────

_CUSTOMER_TURN1_TEMPLATE = (
    'The agent typed: "{agent_input}"\n\n'
    "This is the start of the conversation. Tell the agent your problem: {issue_type}\n"
    "Your tone: {tone}\n\n"
    "State what is wrong in 1-3 natural sentences. Be specific about what happened "
    "and why you are contacting support. Include a relevant detail about your "
    "situation (when it started, what you tried, how it affects you).\n"
    "Do not invent fake names or order numbers.\n"
    "10-30 words. Sound like a real person typing in a chat."
)


# ── Customer general turn ────────────────────────────────────────────────────

_CUSTOMER_GENERAL_TURN_TEMPLATE = (
    'The agent just typed: "{agent_input}"\n\n'
    "Your current mood: {brief}\n"
    "{hint}\n"
    "{avoid_clause}"
    "{resolution_history_clause}"
    "{no_repeat_clause}"
    "{timeline_block_clause}\n"
    "React naturally to what the agent said. Consider:\n"
    "- Did they actually address your concern, or dodge it?\n"
    "- Did they take a concrete action, or just make a vague promise?\n"
    "- Did they answer your question, or ignore it?\n"
    "- What is the next logical thing to discuss in this conversation?\n\n"
    "Respond like a real person would. Move the conversation forward — "
    "do not repeat what you already said. If they helped, acknowledge it. "
    "If they did not, push for something specific. If they asked a question, answer it.\n"
    "You can ask a natural follow-up question about next steps, costs, or confirmation.\n"
    "1-3 sentences, 10-30 words. No labels. Sound human and natural."
)


# ── Customer resolved turn ───────────────────────────────────────────────────

_CUSTOMER_RESOLVED_TEMPLATE = (
    'The agent typed: "{agent_input}"\n\n'
    "The agent says your problem is fixed.\n"
    "Your satisfaction level: {resolution_tone}\n\n"
    "Respond naturally based on how the conversation went. "
    "If the agent earned your trust through good service, thank them genuinely. "
    "If it was a struggle, you can be relieved but still show some irritation.\n"
    "1-2 sentences, 10-20 words. End with a period, not a question mark.\n"
    "Sound like a real person, not a robot."
)

_RESOLUTION_TONES = [
    "",
    "You are genuinely grateful. The agent helped well. Say a warm thank you.",
    "You are satisfied. It went okay. Brief thanks, nothing more.",
    "You are relieved but still a bit annoyed it took effort. Grudging thanks.",
    "You are barely satisfied. It took way too long. Show irritation but accept it.",
    "You accept the fix but you are still upset about the experience. Curt, no warmth.",
]


# ── Scorer — merit-based, not defaulting to 5 ────────────────────────────────

_SCORER_SYSTEM = (
    "You are a strict call centre QA evaluator. Score the agent reply 0-10 "
    "based PURELY on quality and merit of their response.\n\n"
    "SCORING — be precise, do NOT default to 5:\n"
    "  10 = Perfect: empathy + names issue + confirms fix + specific timeline\n"
    "  9 = Excellent: empathy + concrete action + specific timeline\n"
    "  8 = Very good: empathy + action + general timeline\n"
    "  7 = Good: acknowledges issue + states clear action plan\n"
    "  6 = Decent: shows effort, addresses concern but lacks specifics\n"
    "  5 = Average: acknowledges issue but vague, no clear action\n"
    "  4 = Below average: generic response, does not address specific concern\n"
    "  3 = Poor: mostly filler, repeats previous promise, no new information\n"
    "  2 = Very poor: dismissive, off-topic, or unhelpful\n"
    "  1 = Terrible: rude, nonsensical, or ignores customer entirely\n"
    "  0 = Hostile or completely irrelevant\n\n"
    "IMPORTANT:\n"
    "- A simple greeting ('Hi, how can I help?') = exactly 5\n"
    "- A promise ('I will fix it') without details = 4-5, NOT higher\n"
    "- Repeating the same thing = score LOWER than first time\n"
    "- Agent who shows empathy AND takes action AND gives timeline = 8+\n"
    "- Use the FULL range 0-10. Most replies should NOT be 5.\n\n"
    "Return ONLY valid JSON:\n"
    '{{\"score\": <0-10>, '
    '\"tip\": \"<ONE specific coaching suggestion, under 20 words>\", '
    '\"reason\": \"<what agent did well or missed, under 20 words>\"}}'
)

_SCORER_USER_TEMPLATE = (
    "Issue the customer contacted about: {issue}\n"
    'What the customer said: \"{customer_said}\"\n'
    'What the agent replied: \"{agent_input}\"\n\n'
    "Evaluate the agent reply on merit. Consider:\n"
    "- Did they show empathy? (acknowledge feelings)\n"
    "- Did they take action? (concrete steps, not just words)\n"
    "- Did they give a timeline? (when will it be resolved)\n"
    "- Did they address what the customer specifically said?\n"
    "- Did they add new information or just repeat themselves?\n\n"
    "Score 0-10 based on quality. Return ONLY the JSON."
)


# ── Ideal response generator ─────────────────────────────────────────────────

_IDEAL_SYSTEM = (
    "You write example agent replies for a call centre training tool. "
    "Your responses should sound like a real, competent support agent "
    "typing in a live chat — natural, professional, and helpful.\n\n"
    "RULES:\n"
    "1. Output ONLY the agent words — no labels, no quotes, no prefixes.\n"
    "2. Sound natural and human. Use contractions. Be conversational.\n"
    "3. NEVER invent names, order numbers, or ticket IDs.\n"
    "4. Directly address what the customer just said.\n"
    "5. Each response must use a DIFFERENT sentence structure and opening word."
)

_IDEAL_POSITIVE_TEMPLATE = (
    "Issue: {issue} | Customer personality: {persona}\n"
    "CONVERSATION SO FAR:\n{conversation}\n"
    'Customer just said: \"{customer_said}\"\n\n'
    "Write an EXCELLENT agent response — the kind that would make a customer "
    "feel heard and confident their issue will be resolved. Include empathy, "
    "a concrete action, and if appropriate a timeline or next step.\n"
    "{ban_clause}"
    "Must differ from: {prev_positives}\n"
    "Start with a DIFFERENT word than previous responses.\n"
    "1-2 sentences, 15-30 words. Sound like a real person. No invented details."
)

_IDEAL_NEUTRAL_TEMPLATE = (
    "Issue: {issue} | Customer personality: {persona}\n"
    "CONVERSATION SO FAR:\n{conversation}\n"
    'Customer just said: \"{customer_said}\"\n\n'
    "Write an AVERAGE agent response — professional but not outstanding. "
    "The agent acknowledges the concern and shows some effort, but does not "
    "promise a fix or give a specific timeline. It is okay but not impressive.\n"
    "{ban_clause}"
    "Start with a DIFFERENT opening word than 'I' or 'Let'.\n"
    "1-2 sentences, 15-30 words. Sound professional. No invented details."
)

_IDEAL_NEGATIVE_TEMPLATE = (
    "Issue: {issue} | Customer personality: {persona}\n"
    "CONVERSATION SO FAR:\n{conversation}\n"
    'Customer just said: \"{customer_said}\"\n\n'
    "Write a POOR agent response — an example of what NOT to do. "
    "The agent is vague, deflects, gives no plan, or dismisses the concern. "
    "It should be realistic (not absurdly bad) but clearly unhelpful.\n"
    "{ban_clause}"
    "Start with a DIFFERENT opening word than 'I' or 'Let'.\n"
    "1-2 sentences, 15-30 words. Sound like a lazy agent. No invented details."
)

# ── Report generator — [FIX-10] chat-based ───────────────────────────────────

_REPORT_SYSTEM = (
    "You are a call centre training manager writing a performance review "
    "for a chat-based customer support session. "
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


# ── Emotional states ─────────────────────────────────────────────────────────

_STATES = {
    "FURIOUS":    ("You are furious. You feel ignored and disrespected. Nothing has worked.",
                   "Hostile and sharp. You want immediate action or you are done with this company."),
    "ANGRY":      ("You are upset. The agent has not earned your trust yet.",
                   "Skeptical and blunt. You need real proof they can actually help."),
    "FRUSTRATED": ("You are frustrated but still willing to engage. The agent seems to be trying.",
                   "Impatient but listening. You expect results and want to see progress."),
    "CALMING":    ("You are starting to feel better. The agent is actually helping.",
                   "Cautious but warming up. Cooperative if they keep this up."),
    "SATISFIED":  ("You feel the issue is being handled. You are almost done.",
                   "Relieved. Ready to thank the agent and wrap up the conversation."),
}

# [FIX-13] Turn-1 specific fallbacks — these state the problem, not ask about timelines
_TURN1_FALLBACKS = {
    "Double Billing":           "I just noticed I was charged twice for the same order.",
    "Unrecognized Charge":      "There's a charge on my account I don't recognize.",
    "Refund Not Received":      "I was supposed to get a refund but it hasn't come through.",
    "Account Suspended":        "My account got suspended and I don't know why.",
    "Subscription Auto-Renewal":"My subscription renewed automatically and I wanted it canceled.",
    "Package Not Arrived":      "My package was supposed to arrive days ago and it still hasn't.",
    "Wrong Item Received":      "I received the wrong item in my delivery.",
    "Damaged Item":             "The item I received is damaged.",
    "Login Error":              "I can't log into my account — it keeps giving me an error.",
    "Account Hacked":           "I think my account has been hacked — there's activity I didn't do.",
    "Shipping Delay":           "My order has been delayed and I need to know when it's arriving.",
    "Warranty Claim Rejected":  "My warranty claim was rejected and I don't understand why.",
    "Wrong Currency Charged":   "I was charged in the wrong currency for my order.",
    "App Crashing":             "The app keeps crashing every time I try to open it.",
    "Software Update Failed":   "The software update failed and now nothing is working right.",
    "Device Overheating":       "My device keeps overheating during normal use.",
    "Wi-Fi Dropouts":           "My Wi-Fi keeps dropping out randomly throughout the day.",
    "Promo Code Invalid":       "I tried to use a promo code but it says invalid.",
    "Loyalty Points Missing":   "My loyalty points disappeared from my account.",
    "Payment Method Declined":  "My payment keeps getting declined even though my card is fine.",
    "Credit Card Declined":     "My credit card is being declined but I have enough funds.",
    "Package Stolen":           "I think my package was stolen — it says delivered but I don't have it.",
    "Tracking Not Updating":    "My tracking number hasn't updated in days.",
    "Defective Product":        "The product I bought is defective — it stopped working.",
    "Data Sync Error":          "My data isn't syncing properly across my devices.",
}

_FALLBACKS = {
    "FURIOUS":    ["This is ridiculous — I need to speak to a supervisor.",
                   "I've been dealing with this for DAYS and nothing's changed.",
                   "That's not what I asked. I need a real answer right now."],
    "ANGRY":      ["That's not good enough. What are you actually going to do?",
                   "I've heard that before — I need something concrete this time.",
                   "I'm losing patience. Give me a straight answer."],
    "FRUSTRATED": ["I need to know exactly what you're going to do about this.",
                   "Fine — but I expect this to actually get handled.",
                   "Okay, I'm trusting you on this. Don't let me down."],
    "CALMING":    ["Okay, that makes more sense. I'll hold you to that.",
                   "Alright, I can work with that. Just make sure it happens.",
                   "Fine — just please follow through on that."],
    "SATISFIED":  ["Okay, that's all I needed. Thanks.",
                   "Finally. Thank you for sorting it out.",
                   "Good — that's what I was looking for."],
}

# Difficulty-aware satisfied fallbacks for resolution
_RESOLVED_FALLBACKS = [
    [],
    # Difficulty 1 — warm
    ["Great, thank you so much.", "Perfect, that's exactly what I needed.", "Wonderful, thanks for your help."],
    # Difficulty 2 — brief
    ["Okay good, thanks.", "That works, appreciate it.", "Alright, thanks."],
    # Difficulty 3 — grudging
    ["About time. Thanks.", "Finally. Took long enough.", "Okay, that works I guess."],
    # Difficulty 4 — barely grateful
    ["Should not have taken this long.", "Finally. Don't let it happen again.", "About time. I expected better."],
    # Difficulty 5 — curt with warning
    ["Fine. This better not happen again.", "Whatever. Just don't let it repeat.", "Took way too long. We're done here."],
]


# ── Bad output detection — [FIX-11] expanded for chat context ─────────────────

_BAD_CUSTOMER = [
    # Role confusion — most common with small models
    "Agent:", "AGENT:", "As an AI", "Certainly!", "Of course!", "Absolutely!",
    "I can help you with", "Thank you for calling", "I understand your frustration",
    "I apologize for the inconvenience", "Let me help", "Happy to help",
    "Is there anything else", "How can I assist", "I'd be delighted",
    # Chat-specific agent phrases [FIX-10]
    "Thank you for reaching out", "Thank you for contacting",
    "Thanks for reaching out", "Thanks for contacting",
    "Thank you for your patience", "Thanks for your patience",
    "I appreciate your patience", "I appreciate you reaching out",
    "Let me look into that for you", "I'll be happy to help",
    "Allow me to", "Please allow me",
    # Greetings (agent-style)
    "Hey there", "Hello there", "Hi there", "Good morning", "Good afternoon",
    "Welcome to", "Welcome,",
    # Meta / prompt leakage
    "In this scenario", "The agent should", "Training", "Simulation",
    "Your emotional state", "Tone:", "State:", "FURIOUS", "ANGRY",
    "FRUSTRATED", "CALMING", "SATISFIED", "Do NOT", "DO NOT",
    "1-2 sentences", "End of call", "scenario complete",
    "end of chat", "chat complete", "session complete",
    # Customer acting as agent
    "Is there anything I can do", "anything I can do to help",
    "How can I assist you", "Let me know if I can help",
    "What can I do for you", "I can assist",
    "I can help", "Let me assist",
    # Agent-like courtesy
    "You're welcome", "My pleasure", "Great job", "Good job",
    "appreciate your efficiency", "your assistance",
    # Repetition markers
    "You already said", "already told", "already said",
    # Role labels / system prompt leakage [FIX-11]
    "IDENTITY RULES", "ABSOLUTE RULES", "THINGS YOU MUST NEVER",
    "YOUR PROBLEM:", "YOUR PERSONALITY:", "YOUR BEHAVIOUR:",
    "REACT to what", "Make STATEMENTS",
    # [FIX-15] New system prompt section headers
    "YOUR SITUATION", "RULES YOU MUST", "FORBIDDEN PHRASES",
    "=== YOUR", "=== RULES", "=== FORBIDDEN",
    "PROBLEM (the issue", "PERSONALITY (how you",
    "BEHAVIOUR (your current",
]


# ── Promise vs Confirmation detection ─────────────────────────────────────────

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
    "has been delivered", "is delivered", "it is delivered",
    "package delivered", "order delivered", "been delivered",
    "successfully delivered", "already delivered",
]

# ── [FIX-5] Mood-band keywords for coherence check ───────────────────────────
_MOOD_BAND_CAPS_LIMIT = {
    "FURIOUS":    99,
    "ANGRY":      3,
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

    words = text.split()
    caps_words = [(i, w) for i, w in enumerate(words) if w.isupper() and len(w) >= 2]
    keep_idx = -1
    if caps_words:
        keep_idx = max(caps_words, key=lambda x: len(x[1]))[0]

    result = []
    for i, w in enumerate(words):
        if i == keep_idx:
            result.append(w)
        elif i == 0:
            result.append(w.capitalize())
        else:
            result.append(w.lower())
    out = " ".join(result)
    out = re.sub(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), out)
    return out


class CustomerSimulator:
    _SEVERE_ISSUES = {
        "account hacked", "package stolen", "credit card declined",
        "account suspended", "device overheating", "data sync error",
        "double billing", "unrecognized charge",
    }

    def __init__(self, scenario, difficulty=1):
        self.scenario   = scenario
        self.history    = []
        self.difficulty = difficulty

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
            self._streak = max(0, streak) + 1
            self.mood = min(10, self.mood + (2 if self._streak >= 2 else 1))
        elif score >= 7:
            self._streak = max(0, streak) + 1
            self.mood = min(10, self.mood + 1)
        elif score <= 1:
            self._streak = min(0, streak) - 1
            self.mood = max(0, self.mood - (2 if self._streak <= -2 else 1))
        elif score <= 3:
            self._streak = min(0, streak) - 1
            self.mood = max(0, self.mood - 1)
        else:
            self._streak = 0

    def _is_greeting_only(self, text: str) -> bool:
        """Returns True if the agent's message is just a greeting with no substance."""
        t = text.lower().strip().rstrip("?!.")
        _GREETING_PHRASES = [
            "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
            "how can i help", "how may i help", "how can i assist",
            "how may i assist", "what can i do for you", "how are you",
            "welcome", "thank you for reaching out", "thanks for reaching out",
            "thank you for contacting", "thanks for contacting",
        ]
        remaining = t
        for gp in sorted(_GREETING_PHRASES, key=len, reverse=True):
            remaining = remaining.replace(gp, "").strip()
        for filler in ["you", "sir", "madam", "ma'am", "today", "there",
                        "and", "the", "a", "to", "can", "may", "i", "we",
                        "us", "me", "with"]:
            remaining = remaining.replace(filler, "").strip()
        remaining = remaining.strip("?!., ")
        return len(remaining) < 3

    def speak(self, agent_input: str, score: int = 5) -> str:
        self.turn += 1
        if not self._is_greeting_only(agent_input):
            self._shift(score)
        state = self._state()
        brief, tone = _STATES[state]
        all_past = [m["content"] for m in self.history if m["role"] == "assistant"]

        # Context-aware hints — based on what the agent actually did
        ai_lower = agent_input.lower()

        agent_took_action = any(p in ai_lower for p in [
            "mailed", "emailed", "sent you", "sending you", "contacted",
            "escalated", "assigned", "dispatched", "scheduled", "booked",
            "arranged", "filed", "submitted", "logged", "raised",
            "team is working", "developers are", "engineers are",
            "technician", "specialist", "checking", "looking into",
            "investigating", "processing", "working on",
        ])
        agent_gave_timeline = any(p in ai_lower for p in [
            "hour", "minute", "tomorrow", "today", "tonight",
            "within", "by end", "shortly", "right now",
        ]) or bool(re.search(r'\b\d+\s*(hours?|minutes?|days?)\b', ai_lower))
        agent_confirmed_delivery = any(p in ai_lower for p in [
            "delivered", "delivery", "arrived", "received",
            "check your", "been shipped", "on its way",
        ])
        agent_apologized = any(p in ai_lower for p in [
            "sorry", "apologize", "apologies", "apology", "regret",
        ])

        if agent_confirmed_delivery:
            hint = (
                "The agent says your item has been delivered or is arriving. "
                "React to this — confirm you will check, or express doubt "
                "if you have not received it. Do not ignore what they said."
            )
        elif agent_took_action and agent_gave_timeline:
            hint = (
                "The agent took action AND gave you a timeline. Acknowledge both. "
                "You can express cautious acceptance or warn about consequences "
                "if they do not follow through. Move the conversation forward."
            )
        elif agent_took_action:
            hint = (
                "The agent took a concrete action. Acknowledge it — do not ignore "
                "what they did. Ask what happens next, or express cautious hope. "
                "Do not repeat your previous demand."
            )
        elif agent_gave_timeline:
            hint = (
                "The agent gave you a timeline. Respond to THAT timeline. "
                "Accept it, push back on it, or set expectations about what "
                "happens if they miss it. Do not ask for a timeline again."
            )
        elif agent_apologized and score >= 5:
            hint = (
                "The agent apologized. Acknowledge their apology but steer "
                "toward a solution. You want action, not just words."
            )
        elif score >= 8:
            hint = (
                "The agent gave a strong, helpful response. Acknowledge their "
                "effort. Cooperate, ask a follow-up, or start warming up. "
                "Move the conversation toward resolution."
            )
        elif score >= 5:
            hint = (
                "The agent is trying but their response could be better. "
                "React to what they actually said — push for more detail "
                "or ask a follow-up that moves things forward."
            )
        else:
            hint = (
                "The agent response was weak or unhelpful. Express frustration "
                "with what they specifically said or failed to say. "
                "Demand something concrete or threaten to escalate."
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
                resolution_tone = _RESOLUTION_TONES[min(self.difficulty, 5)]
                prompt = _CUSTOMER_RESOLVED_TEMPLATE.format(
                    agent_input=agent_input,
                    resolution_tone=resolution_tone,
                )
            else:
                agent_msgs = [
                    m["content"] for m in self.history if m["role"] == "user"
                ]
                # Check BOTH history AND current agent_input for timelines
                all_agent_text = " ".join(
                    am.lower() for am in agent_msgs
                ) + " " + agent_input.lower()
                already_stated = []
                time_indicators = [
                    "5pm", "6pm", "7pm", "8pm", "9pm", "10pm", "11pm",
                    "am", "pm", "hour", "minute", "today", "tomorrow",
                    "end of day", "by noon", "within", "shortly",
                    "right now", "immediately", "asap",
                ]
                has_timeline_in_convo = (
                    any(t in all_agent_text for t in time_indicators)
                    or bool(re.search(r'\b\d+\s*(hours?|minutes?|days?|hrs?|mins?)\b', all_agent_text))
                )
                if has_timeline_in_convo:
                    already_stated.append(
                        "A specific timeline was already provided — "
                        "do NOT ask for a timeline again."
                    )
                no_repeat_clause = (
                    already_stated[0] + "\n" if already_stated else ""
                )

                # Build a strong timeline block clause for the prompt
                timeline_block_clause = ""
                if has_timeline_in_convo:
                    timeline_block_clause = (
                        "IMPORTANT: The agent has ALREADY given you a timeline. "
                        "Do NOT ask 'when', 'how long', 'how soon', or any timeline question. "
                        "Instead, respond to what the agent said — acknowledge their plan or push for action.\n"
                    )

                prompt = _CUSTOMER_GENERAL_TURN_TEMPLATE.format(
                    agent_input=agent_input,
                    brief=brief,
                    hint=hint,
                    tone=tone,
                    avoid_clause=avoid_clause,
                    resolution_history_clause=resolution_history_clause,
                    no_repeat_clause=no_repeat_clause,
                    timeline_block_clause=timeline_block_clause,
                )

        # ── Output cleaning pipeline ──────────────────────────────────────
        def _clean(raw):
            for lbl in ["Customer:", "CUSTOMER:", "C:", "User:"]:
                if raw.startswith(lbl):
                    raw = raw[len(lbl):].strip()
            if " | " in raw:
                raw = raw.split(" | ")[0].strip()
            raw = raw.replace("**", "").replace("__", "")
            raw = re.sub(r'\*([^*]+)\*', r'\1', raw)
            raw = raw.strip().strip("'\"").strip('\u2018\u2019\u201c\u201d').strip()
            _LEAKAGE = [
                "Your emotional state", "Push for specifics", "Tone:", "State:",
                "Do NOT", "DO NOT", "React as", "No labels", "Avoid:",
                "emotional state remains", "You are still frustrated",
                "agent said so far", "Don't re-raise", "1-2 sentences",
                "The agent just said", "The agent said", "Agent said",
                "agent just said", "the agent just", "The agent just",
                "End of call", "end of call", "End of conversation",
                "end of conversation", "call sequence", "interaction rules",
                "given constraints", "following rules", "following customer",
                "support interaction", "successfully completed",
                "scenario complete", "simulation end", "session end",
                "role-play", "roleplay", "as per instructions",
                "as instructed", "per the prompt", "according to rules",
                "Customer Support", "customer support",
                "Customer Service", "customer service",
                # [FIX-10/11] Additional chat-context leakage
                "End of chat", "end of chat", "chat complete",
                "IDENTITY RULES", "THINGS YOU MUST NEVER",
                "YOUR PROBLEM:", "YOUR PERSONALITY:", "YOUR BEHAVIOUR:",
                "REACT to what", "Make STATEMENTS",
                "under 25 words", "Max 2 sentences",
                "No agent phrases", "agent phrases",
                # [FIX-15] New system prompt section headers
                "YOUR SITUATION", "RULES YOU MUST", "FORBIDDEN PHRASES",
                "=== YOUR", "=== RULES", "=== FORBIDDEN",
                "PROBLEM (the issue", "PERSONALITY (how you",
                "BEHAVIOUR (your current", "IMPORTANT: The agent",
                "timeline_block_clause", "resolution_history_clause",
                "no_repeat_clause", "avoid_clause",
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
            raw = re.sub(
                r'[,;]?\s*\b(?:right|yeah|huh|okay|ok|no|correct|innit|eh|ya|sure)\s*\?\s*$',
                '.', raw, flags=re.IGNORECASE
            ).strip()
            raw = _fix_caps(raw)
            return raw

        def _ok(raw):
            if not raw or len(raw) < 10:
                return False
            if any(b.lower() in raw.lower() for b in _BAD_CUSTOMER):
                return False
            raw_lower_stripped = raw.lower().rstrip(" .!?")
            if re.search(r'\b(right|yeah|huh|okay|ok|no|correct|innit|eh)\s*$',
                         raw_lower_stripped):
                return False
            if any(tag in raw.lower() for tag in [
                "customer support", "customer service", "support team",
                "help desk", "call center", "call centre", "customer care",
                "technical support", "service center",
            ]):
                return False
            if len(raw.split()) > 50:
                return False
            max_caps = _MOOD_BAND_CAPS_LIMIT.get(state, 2)
            if state != "FURIOUS" and _count_caps_words(raw) > max_caps:
                return False
            if not _resolved_this_turn and state != "SATISFIED":
                _GRATITUDE = [
                    "thank you", "thanks", "appreciate", "grateful",
                    "that helps", "that's helpful", "very helpful",
                ]
                if any(g in raw.lower() for g in _GRATITUDE):
                    return False
            raw_lower = raw.lower()
            if self.turn > 1:
                # Check BOTH history AND current agent_input for timelines
                agent_history_text = " ".join(
                    m["content"].lower() for m in self.history if m["role"] == "user"
                ) + " " + agent_input.lower()
                timeline_already_given = any(t in agent_history_text for t in [
                    "hour", "minute", "tomorrow", "today", "by end",
                    "within", "am", "pm", "morning", "afternoon",
                    "evening", "business day", "day", "days",
                    "shortly", "soon", "asap", "right now", "immediately",
                    "tonight", "5pm", "6pm", "7pm", "around",
                ]) or re.search(r'\b\d+\s*(hours?|minutes?|days?|hrs?|mins?)\b',
                                agent_history_text)
                if timeline_already_given:
                    _TIMELINE_QUESTIONS = [
                        "when exactly", "when will", "how long",
                        "what time", "specific date", "specific time",
                        "when can i expect", "when is it", "when do i",
                        "when are you", "by when", "how soon",
                        "timeline now", "specific timeline",
                        "replacement timeline", "need a timeline",
                        "give me a date", "give me a time",
                        "tomorrow's okay", "tomorrow okay",
                        "is that okay", "acceptable pace",
                        "faster solution", "need a faster",
                        "a faster", "speed this up", "speed it up",
                        "hurry this", "can you hurry",
                        "can it be done", "will it be done",
                        "be done before", "done before",
                        "definite timeline", "direct action",
                        "action steps",
                        # Additional timeline re-ask patterns
                        "how long roughly", "long roughly",
                        "how long will", "how long does",
                        "how long is", "when roughly",
                        "roughly how", "around when",
                        "what's the timeline", "what is the timeline",
                        "expected timeline", "estimated time",
                        "how much longer", "much longer",
                        "when should i", "when do you",
                        "any timeline", "any idea when",
                        "idea when", "know when",
                    ]
                    if any(tq in raw_lower for tq in _TIMELINE_QUESTIONS):
                        return False
                    _DEADLINE_DEMANDS = [
                        "fix it by", "done by", "need it by",
                        "want it by", "before tomorrow", "by tomorrow",
                        "by next", "by friday", "by monday", "by tuesday",
                        "by wednesday", "by thursday", "by saturday", "by sunday",
                        "before end of", "do it by", "get it done by",
                        "i need this by", "better be done by",
                        "need it done", "want it done",
                        "too long", "this long", "that long",
                        "can't wait", "won't wait", "not waiting",
                    ]
                    if any(dd in raw_lower for dd in _DEADLINE_DEMANDS):
                        return False
            if all_past and len(all_past) >= 2:
                last_reply = all_past[-1].lower()
                raw_words = set(raw_lower.split())
                last_words = set(last_reply.split())
                if raw_words and last_words:
                    overlap = len(raw_words & last_words) / max(len(raw_words), 1)
                    if overlap > 0.6:
                        return False
            return True

        # ── Generation loop (2 attempts — reduced from 3 to minimize LLM calls) ──
        reply = ""
        for attempt in range(2):
            raw = _call_gen(
                [{"role": "system", "content": self._sys},
                 {"role": "user", "content": prompt}],
                max_tokens=90,
                temp=0.82 + attempt * 0.10,                                            # ← bigger jump on retry
            )
            raw = _clean(raw)
            if not _ok(raw):
                continue
            if _sem_dup(raw, all_past, 0.55):
                continue
            reply = raw
            break

        # [FIX-13] Turn-1 fallback: use issue-specific problem statement
        if not reply and self.turn == 1:
            issue_type = self.scenario.get("issue_type", "")
            reply = _TURN1_FALLBACKS.get(
                issue_type,
                f"I'm having a problem with {issue_type.lower()} and I need help."
            )
        elif not reply:
            reply = random.choice(_FALLBACKS[state])

        for sep in ["! ", "? ", ". ", "; "]:
            reply = reply.replace(sep, sep[0] + "|||")
        parts = []
        for p in reply.split("|||"):
            p = p.strip()
            if not p:
                continue
            if p.endswith(";"):
                p = p[:-1].strip() + "."
            parts.append(p)
        reply = " ".join(parts[:2]) if parts else reply
        if reply and reply[-1] not in ".!?":
            reply += "."

        if _resolved_this_turn or state == "SATISFIED":
            # Detect if generated reply sounds like a complaint instead of acceptance
            _COMPLAINT_WORDS = [
                "expect", "better", "should", "handled", "not good",
                "terrible", "awful", "useless", "waste", "ridiculous",
                "unacceptable", "pathetic", "worst", "don't believe",
            ]
            is_complaint = any(cw in reply.lower() for cw in _COMPLAINT_WORDS)
            if "?" in reply or is_complaint:
                diff_idx = min(self.difficulty, 5)
                reply = random.choice(_RESOLVED_FALLBACKS[diff_idx])
            self.mood = min(10, self.mood + 2)

        self.history.append({"role": "user", "content": agent_input})
        self.history.append({"role": "assistant", "content": reply})
        return reply


# ==============================================================================
#  GENERATOR 2 — LLM SCORER  [FIX-1: programmatic floor enforcement]
# ==============================================================================

def _enforce_scoring_floors(score: int, agent_input: str, issue: str) -> int:
    """Apply mandatory scoring floors AFTER LLM returns its score.
    Floors only RAISE scores — they never lower them."""
    ai = agent_input.lower()
    words = ai.split()

    # Floor 0: Greeting-only messages → score exactly 5
    _GREETING_WORDS = [
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
        "how can i help", "how may i help", "how can i assist", "how may i assist",
        "what can i do for you", "welcome", "thank you for reaching out",
        "thanks for reaching out", "thank you for contacting",
        "thanks for contacting", "how are you",
    ]
    remaining = ai.strip().rstrip("?!.")
    for gw in sorted(_GREETING_WORDS, key=len, reverse=True):
        remaining = remaining.replace(gw, "").strip()
    for filler in ["you", "sir", "madam", "ma'am", "today", "there",
                    "and", "the", "a", "to", "can", "may", "i", "we"]:
        remaining = remaining.replace(filler, "").strip()
    remaining = remaining.strip("?!., ")
    if len(remaining) < 3:
        return 5

    # Floor 1: Any reply >8 words → minimum 1
    if len(words) > 8:
        score = max(score, 1)

    # Floor 1b: Short timeline replies → minimum 5
    if len(words) <= 5 and re.search(
        r'\b(\d+\s*(hours?|minutes?|mins?|hrs?|days?|business days?)'
        r'|by\s+tomorrow|by\s+tonight|by\s+noon|end\s+of\s+day'
        r'|right\s+now|immediately|asap)\b', ai
    ):
        score = max(score, 5)

    # Floor 1c: Any reply with a concrete timeline → minimum 5
    has_any_timeline = bool(re.search(
        r'\b(\d+\s*(hours?|minutes?|mins?|hrs?|days?|business days?|weeks?)'
        r'|before\s+tomorrow|by\s+tomorrow|before\s+end|end\s+of\s+day'
        r'|by\s+tonight|by\s+noon|within\s+\w+|this\s+afternoon'
        r'|this\s+evening|first\s+thing|right\s+now)\b', ai
    ))
    if has_any_timeline:
        score = max(score, 5)

    # Floor 2: Apology + offer to help → minimum 5
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

    # Floor 2b: Empathy + offer to help → minimum 5
    has_empathy = has_apology or any(w in ai for w in [
        "understand", "frustrating", "frustration", "inconvenience",
        "concern", "difficult", "tough", "appreciate",
    ])
    if has_empathy and has_offer:
        score = max(score, 5)

    # Floor 3: Empathy AND asks for details → minimum 5
    asks_details = "?" in agent_input or any(phrase in ai for phrase in [
        "can you", "could you", "tell me", "let me know", "provide",
        "share", "what is your", "which", "when did", "do you have",
        "may i have", "would you",
    ])
    if has_empathy and asks_details:
        score = max(score, 5)

    # Floor 4: Names issue + states action → minimum 6
    issue_words_raw = issue.lower().replace("-", " ").replace("'", "").split()
    issue_words = [w for w in issue_words_raw if len(w) > 1]
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
        "we will", "we'll", "we are", "we're",
        "prioritize", "prioritise", "expedite", "expediting",
        "enhance", "enhancing", "ensure", "ensuring",
        "take steps", "steps to", "dispatch", "deploying",
        "replace", "replacing", "investigate", "investigating",
        "restore", "restoring", "stabilize", "stabilise",
        "will be done", "be done", "completed",
        "it will", "it'll", "done before", "done by",
        "resolved within", "fixed within", "sorted within",
    ])
    if names_issue and has_action:
        score = max(score, 6)

    # Floor 4b: Names issue + action + SPECIFIC timeline → minimum 7
    has_specific_timeline = any(phrase in ai for phrase in [
        "within the hour", "within the next", "in the next",
        "by tomorrow", "by end of day", "by noon", "by tonight",
        "by monday", "by tuesday", "by wednesday", "by thursday", "by friday",
        "this afternoon", "this evening", "first thing tomorrow",
    ]) or re.search(r'\b(within\s+)?\d{1,2}\s*(am|pm|hours?|minutes?|business days?)\b', ai)
    if names_issue and has_action and has_specific_timeline:
        score = max(score, 7)

    # Floor 5: Confirms resolution → minimum 8
    resolution_phrases = [
        "is resolved", "has been resolved", "been fixed", "is fixed",
        "was fixed", "was resolved", "is solved", "was solved",
        "has been solved", "been solved",
        "all sorted", "been sorted", "taken care of", "all done",
        "been refunded", "been credited", "fully resolved",
        "completely resolved", "good to go",
        "i've fixed", "i fixed", "we've fixed", "we fixed",
        "it's fixed", "it's solved", "it's resolved",
        "its fixed", "its resolved", "its solved", "its sorted", "its done",
        "its been fixed", "its been resolved",
        "problem fixed", "issue fixed", "all fixed", "all resolved",
        "issue resolved", "problem resolved", "issue solved", "problem solved",
        "get resolved", "get fixed",
    ]
    if any(phrase in ai for phrase in resolution_phrases):
        if not _is_promise_not_confirmation(agent_input):
            score = max(score, 8)

    # Floor 6: Near-perfect reply → min 9
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
        max_tokens=150,                                                                # ← [FIX-12] was 100, more room for detailed reason
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
    reason = rm.group(1).strip() if rm else ""

    # [FIX-14] Concise score-range-specific reason fallback
    if not reason or len(reason) < 10:
        if score >= 8:
            reason = "Strong empathy with concrete action and timeline."
        elif score >= 6:
            reason = "Acknowledged the issue but lacked specific timeline or action."
        elif score >= 4:
            reason = "Vague response — missing empathy, action, or timeline."
        elif score >= 2:
            reason = "Did not address the customer's concern meaningfully."
        else:
            reason = "Off-topic or unhelpful response."

    # [FIX-28] Programmatic tip fallback — no second LLM call (saves 1 call per turn)
    if not tip or len(tip) < 8:
        ai = agent_input.lower()
        if not any(w in ai for w in ["sorry", "apologize", "understand", "frustrat"]):
            tip = "Start with empathy — acknowledge the customer's frustration first."
        elif not any(w in ai for w in ["fix", "resolve", "check", "look into", "investigate"]):
            tip = "State a concrete action you will take to resolve this."
        elif not re.search(r'\b(\d+\s*(hours?|minutes?|days?)|tomorrow|tonight)\b', ai):
            tip = "Add a specific timeline so the customer knows when to expect resolution."
        else:
            tip = "Be more specific about what exactly you are doing to fix this."

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
    "positive": "I completely understand how frustrating this {issue} must be — "
                "let me pull up your account right now and get this sorted within the next 2 hours.",
    "neutral":  "Thank you for bringing this {issue} to our attention. "
                "Our team is looking into it and we will keep you updated on the progress.",
    "negative": "These things can happen sometimes with {issue} situations. "
                "You might want to try again later or check our FAQ page for common solutions.",
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
            _INVENTED = [
                "Mr.", "Mrs.", "Ms.", "Dr.", "Sir ", "Ma'am",
                "Order #", "Ticket #", "Ref #", "REF-", "Case #",
                "ID #", "TKT-", "ORD-",
            ]
            if any(inv in t for inv in _INVENTED):
                return False
            if re.search(r'[A-Z]{2,4}[-#]\d{3,}', t):
                return False
            # Reject if starts with same word as any previous ideal in avoid pool
            first_word = t.strip().split()[0].lower().rstrip(",'\"") if t.strip() else ""
            if first_word and avoid:
                prev_first_words = [
                    a.strip().split()[0].lower().rstrip(",'\"")
                    for a in avoid if a and a.strip()
                ]
                # Allow up to 2 repeats of the same first word across all ideals
                if prev_first_words.count(first_word) >= 2:
                    return False
            return True

        for attempt in range(2):                                                       # ← 2 attempts (was 4) — saves up to 6 LLM calls
            cand = _call_gen(
                [{"role": "system", "content": _IDEAL_SYSTEM},
                 {"role": "user", "content": prompt}],
                max_tokens=70,
                temp=temp + attempt * 0.12,                                            # ← bigger jump on retry
            )
            cand = cand.strip().strip('"').strip("'").strip('\u201c').strip('\u201d')
            for pref in ["Agent:", "AGENT:", "Ideal:", "Response:", "Positive:", "Neutral:", "Negative:"]:
                if cand.lower().startswith(pref.lower()):
                    cand = cand[len(pref):].strip()
            # Allow up to 2 sentences — trim at 3rd sentence boundary
            sent_count = 0
            trim_idx = len(cand)
            for si, sc in enumerate(cand):
                if sc in ".!?" and si < len(cand) - 1 and cand[si+1] == " ":
                    sent_count += 1
                    if sent_count >= 2:
                        trim_idx = si + 1
                        break
            cand = cand[:trim_idx]
            cand = cand.strip()
            if ok(cand) and not _sem_dup(cand, avoid, 0.50):                           # ← tighter dedup (was 0.55)
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
                w = words[0].lower().rstrip(",'\"")
                if w and w not in seen:
                    seen.append(w)
            if len(words) >= 3:
                phrase = " ".join(words[:3]).lower().rstrip(",'\".")
                if phrase and phrase not in seen:
                    seen.append(phrase)
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
            f"Do NOT use any of these words or phrases: {', '.join(seen[:30])}.\n"
            if seen else ""
        )

    def generate(self, customer_said, agent_history):
        issue   = self.scenario["issue_type"]
        persona = self.scenario["customer_persona"]

        convo_lines = []
        for m in agent_history[-8:]:
            role_label = "Agent" if m["role"] == "user" else "Customer"
            convo_lines.append(f"  {role_label}: {m['content'][:80]}")
        conversation = "\n".join(convo_lines) if convo_lines else "  (conversation just started)"

        at   = [m["content"] for m in agent_history if m["role"] == "user"]
        said = "; ".join(at[-3:]) if at else "nothing yet"

        all_past_ideals = (
            self._pos_past[-4:]
            + self._neu_past[-4:]
            + self._neg_past[-4:]
        )
        ban_clause = self._build_ban_clause()
        fb = {k: v.format(issue=issue) for k, v in _FB_IDEALS.items()}

        prev_pos_str = " / ".join(
            f'"{p[:50]}"' for p in self._pos_past[-2:] if p
        ) or "none yet"

        # Generate positive first
        pos = self._one(
            _IDEAL_POSITIVE_TEMPLATE.format(
                issue=issue, persona=persona,
                customer_said=customer_said, conversation=conversation,
                ban_clause=ban_clause,
                prev_positives=prev_pos_str,
            ),
            all_past_ideals, fb["positive"], 0.80,
        )
        # Neutral avoids all past + this turn's positive
        neu = self._one(
            _IDEAL_NEUTRAL_TEMPLATE.format(
                issue=issue, persona=persona,
                customer_said=customer_said, conversation=conversation,
                ban_clause=ban_clause,
            ),
            all_past_ideals + [pos], fb["neutral"], 0.88,
        )
        # Negative avoids all past + this turn's positive AND neutral
        neg = self._one(
            _IDEAL_NEGATIVE_TEMPLATE.format(
                issue=issue, persona=persona,
                customer_said=customer_said, conversation=conversation,
                ban_clause=ban_clause,
            ),
            all_past_ideals + [pos, neu], fb["negative"], 0.93,
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
    "its fixed", "its resolved", "its solved", "its sorted", "its done",
    "its been fixed", "its been resolved", "its been sorted",
    "problem fixed", "issue fixed", "all fixed", "all resolved", "all done",
    "issue resolved", "problem resolved", "issue solved", "problem solved",
    "get resolved", "get fixed", "get sorted",
    "has been delivered", "is delivered", "it is delivered",
    "package delivered", "order delivered", "been delivered",
    "successfully delivered", "already delivered",
]


def _win_loss(s):
    scores = [t["score"] for t in s["turn_log"]]
    mood   = s["sim"].mood
    n      = len(scores)
    if n == 0:
        return None, False

    if n >= 1 and mood >= 7:
        last_agent = s["turn_log"][-1]["agent"]
        last_customer = s["turn_log"][-1]["customer"].lower()
        if (any(kw in last_agent.lower() for kw in _WIN_KW)
                and not _is_promise_not_confirmation(last_agent)):
            _REJECTION = ["not good", "terrible", "useless", "don't believe",
                          "prove it", "doubt", "liar", "unacceptable", "worst"]
            if not any(rw in last_customer for rw in _REJECTION):
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
    "its fixed", "its resolved", "its solved", "its sorted", "its done",
    "its been fixed", "its been resolved", "its been sorted",
    "thats fixed", "thats sorted", "thats resolved",
    "problem fixed", "issue fixed", "all fixed", "all resolved", "all done",
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

        agent_resolved_flag = (
            any(kw in agent_input.lower() for kw in _RESOLUTION_BYPASS_KW)
            and not _is_promise_not_confirmation(agent_input)
        )

        if agent_resolved_flag:
            current_mood = s["sim"].mood
            if current_mood <= 3:
                score  = 7
                tip    = "Good resolution statement, but the customer is still upset — rebuild trust first."
                reason = "Claimed resolution while customer mood is still low."
                s["sim"].mood = min(10, current_mood + 1)
            elif current_mood <= 5:
                score  = 8
                tip    = "Resolution confirmed. Customer is cautious — follow up to ensure satisfaction."
                reason = "Resolution stated. Customer may need one more reassurance."
                s["sim"].mood = min(10, current_mood + 2)
                s["sim"]._streak = 1
            else:
                score  = 9
                tip    = "Excellent resolution — the agent confirmed the fix clearly and confidently."
                reason = "Agent confirmed resolution with good customer rapport."
                s["sim"].mood = min(10, current_mood + 2)
                s["sim"]._streak = 1
        else:
            score, tip, reason = s["coach"].evaluate(agent_input, last_customer)

        customer_reply = s["sim"].speak(agent_input, score=score)

        s["turn_log"].append({
            "turn": turn, "agent": agent_input,
            "customer": customer_reply, "score": score, "tip": tip,
        })

        outcome, resolved = _win_loss(s)
        if resolved:
            db.bump(s["scenario"]["id"])

        mood = s["sim"].mood
        print(f"[T{turn}] score={score} mood={mood} — sending partial, generating ideals...", flush=True)

        # ── PHASE 1: Send score + customer reply IMMEDIATELY ──
        # User sees the response right away, ideals come later
        _ws_send(session_id, {
            "type":            "result",
            "customer_reply":  customer_reply,
            "score":           score,
            "tip":             tip,
            "reason":          reason,
            "resolved":        resolved,
            "ideal":           "",
            "ideals":          {"positive": "", "neutral": "", "negative": "", "ideal": ""},
            "outcome":         outcome,
            "customer_mood":   mood,
            "mood_label":      _mood_label(mood),
            "turns_remaining": max(0, MAX_TURNS - s["turn_count"]),
        })

        # ── PHASE 2: Generate ideals and push them as a follow-up ──
        ideals = s["ideal"].generate(customer_reply, s["sim"].history)
        print(f"[T{turn}] ideals done — sending update", flush=True)

        _ws_send(session_id, {
            "type":   "ideals_update",
            "turn":   turn,
            "ideal":  ideals["ideal"],
            "ideals": ideals,
        })

    except Exception as e:
        import traceback
        print(f"[ERROR] {traceback.format_exc()}", flush=True)
        _ws_send(session_id, {"type": "error", "detail": str(e)})


# ── Feature 1: LLM scenario generation tracking ──────────────────────────────
_generated_scenario_history: list = []


def _generate_scenario_via_llm() -> dict:
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
        parsed = {
            "issue_type": random.choice(ISSUES),
            "customer_persona": random.choice(PERSONAS),
            "short_description": (
                f"A {random.choice(PERSONAS).lower()} customer contacts support about "
                f"a {random.choice(ISSUES).lower()} issue."
            ),
        }

    parsed.setdefault("customer_persona", random.choice(PERSONAS))
    parsed.setdefault("short_description",
                      f"{parsed['customer_persona']} customer with {parsed['issue_type']} problem.")

    _generated_scenario_history.append(parsed)
    if len(_generated_scenario_history) > 50:
        _generated_scenario_history[:] = _generated_scenario_history[-30:]

    return parsed


# ── Feature 2: Edit message background worker ────────────────────────────────

def _do_edit_work(session_id: str, turn_number: int, new_agent_input: str):
    try:
        s = sessions[session_id]
        print(f"[EDIT T{turn_number}] processing...", flush=True)

        import time as _time
        for _ in range(80):
            if session_id in _ws_queues:
                break
            _time.sleep(0.1)
        else:
            print(f"[EDIT T{turn_number}] WS never connected — aborting", flush=True)
            return

        _ws_send(session_id, {"type": "thinking", "turn": turn_number})

        s["turn_log"] = [t for t in s["turn_log"] if t["turn"] < turn_number]
        s["turn_count"] = turn_number - 1

        entries_to_keep = (turn_number - 1) * 2
        s["sim"].history = s["sim"].history[:entries_to_keep]
        s["sim"].turn = turn_number - 1

        base_mood = max(1, 7 - s["difficulty"])
        if s["scenario"].get("issue_type", "").lower() in CustomerSimulator._SEVERE_ISSUES:
            base_mood = max(1, base_mood - 1)
        s["sim"].mood = base_mood
        s["sim"]._streak = 0
        for t in s["turn_log"]:
            s["sim"]._shift(t["score"])

        s["ideal"].past = s["ideal"].past[:(turn_number - 1) * 3]
        keep_ideal = turn_number - 1
        s["ideal"]._pos_past = s["ideal"]._pos_past[:keep_ideal]
        s["ideal"]._neu_past = s["ideal"]._neu_past[:keep_ideal]
        s["ideal"]._neg_past = s["ideal"]._neg_past[:keep_ideal]

        s["turn_count"] += 1
        turn = s["turn_count"]

        last_customer = next(
            (m["content"] for m in reversed(s["sim"].history)
             if m["role"] == "assistant"),
            "No previous reply.",
        )

        agent_resolved_flag = (
            any(kw in new_agent_input.lower() for kw in _RESOLUTION_BYPASS_KW)
            and not _is_promise_not_confirmation(new_agent_input)
        )

        if agent_resolved_flag:
            current_mood = s["sim"].mood
            if current_mood <= 3:
                score  = 7
                tip    = "Good resolution statement, but the customer is still upset — rebuild trust first."
                reason = "Claimed resolution while customer mood is still low."
                s["sim"].mood = min(10, current_mood + 1)
            elif current_mood <= 5:
                score  = 8
                tip    = "Resolution confirmed. Customer is cautious — follow up to ensure satisfaction."
                reason = "Resolution stated. Customer may need one more reassurance."
                s["sim"].mood = min(10, current_mood + 2)
                s["sim"]._streak = 1
            else:
                score  = 9
                tip    = "Excellent resolution — the agent confirmed the fix clearly and confidently."
                reason = "Agent confirmed resolution with good customer rapport."
                s["sim"].mood = min(10, current_mood + 2)
                s["sim"]._streak = 1
        else:
            score, tip, reason = s["coach"].evaluate(new_agent_input, last_customer)

        customer_reply = s["sim"].speak(new_agent_input, score=score)

        s["turn_log"].append({
            "turn": turn, "agent": new_agent_input,
            "customer": customer_reply, "score": score, "tip": tip,
        })

        outcome, resolved = _win_loss(s)
        if resolved:
            db.bump(s["scenario"]["id"])

        mood = s["sim"].mood
        print(f"[EDIT T{turn}] score={score} mood={mood} — sending partial, generating ideals...", flush=True)

        # PHASE 1: Send edit result immediately (no ideals yet)
        _ws_send(session_id, {
            "type":            "edit_result",
            "edited_turn":     turn_number,
            "customer_reply":  customer_reply,
            "score":           score,
            "tip":             tip,
            "reason":          reason,
            "resolved":        resolved,
            "ideal":           "",
            "ideals":          {"positive": "", "neutral": "", "negative": "", "ideal": ""},
            "outcome":         outcome,
            "customer_mood":   mood,
            "mood_label":      _mood_label(mood),
            "turns_remaining": max(0, MAX_TURNS - s["turn_count"]),
        })

        # PHASE 2: Generate ideals and push update
        ideals = s["ideal"].generate(customer_reply, s["sim"].history)
        print(f"[EDIT T{turn}] ideals done — sending update", flush=True)

        _ws_send(session_id, {
            "type":   "ideals_update",
            "turn":   turn,
            "ideal":  ideals["ideal"],
            "ideals": ideals,
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
    difficulty = max(1, min(5, req.difficulty if req else 1))
    generated  = _generate_scenario_via_llm()

    scenario = {
        "id":               len(db.data) + len(_generated_scenario_history) + 100,
        "customer_persona": generated["customer_persona"],
        "issue_type":       generated["issue_type"],
        "difficulty":       difficulty,
    }

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
    difficulty = max(1, min(5, req.difficulty))
    persona = req.persona.strip() if req.persona and req.persona.strip() else random.choice(PERSONAS)

    scenario = {
        "id":               9000 + random.randint(1, 9999),
        "customer_persona": persona,
        "issue_type":       req.issue_type.strip(),
        "difficulty":       difficulty,
    }

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
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    old = sessions[req.session_id]
    scenario   = old["scenario"]
    difficulty = old["difficulty"]

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