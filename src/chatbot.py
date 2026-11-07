"""
chatbot.py - conversational AI helper for criminal-law problems in India (LangGraph + LangChain).

Scope (one problem area, in depth): offences against the person - threats, cheating/fraud, theft,
assault, harassment - under the Bharatiya Nyaya Sanhita (BNS), using the index built in retriever.py.
Other problems (deposits, salary, refunds...) still get case-building help, but NO statute citations,
because there is no verified source for them here.

What it does
  1. Builds the case      - turns a messy story into what happened / when / who / what proof exists
  2. Explains where they stand - only from retrieved BNS sections, in plain language, with options
  3. Drafts documents     - police complaint, demand/legal notice, or a case brief for a lawyer
  4. Practice mode        - plays the other side (e.g. the police officer) and then gives feedback

How it avoids invented law (layers)
  - Distance gate  : if the best retrieved section is too far away, no section is offered at all.
  - LLM judge      : every retrieved section must be judged strong/possible fit; only those are "allowed".
  - Allow-list     : the writer may cite ONLY allowed sections (numbers are checked in code, not by the LLM).
  - Verify node    : regex-extracts every section number in the reply; unknown ones trigger one rewrite,
                     then are redacted. Case-law citations are blocked outright.
  - Exact lookup   : if the user names a section, it is checked against the index (a fake one is reported as missing).
  - Escalation     : emergency / accused / notice-received situations are routed away from "DIY" mode.

Usage (notebook or script):
    pip install langgraph langchain langchain-google-genai python-dotenv
    # put GOOGLE_API_KEY=your-key in src/.env
    from src.chatbot import chat, show_case      # run from the project root
    print(chat("Someone keeps threatening to kill me over WhatsApp", thread_id="user1"))
    print(chat("What can I do?", thread_id="user1"))
    show_case("user1")
"""
import json
import os
import re
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from .retriever import get_section, search

# Reads GOOGLE_API_KEY from src/.env (or a .env FILE in the project root). A root ".env" that is a
# folder (e.g. a virtual environment) is ignored.
_SRC_DIR = Path(__file__).resolve().parent
for _env in (_SRC_DIR / ".env", _SRC_DIR.parent / ".env"):
    if _env.is_file():
        load_dotenv(_env)

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
# Cheapest current Gemini text model (about $0.10 in / $0.40 out per 1M tokens, has a free tier).
# Needs: pip install langchain-google-genai   and the GOOGLE_API_KEY environment variable.
# Any other LangChain model string also works, e.g. "google_genai:gemini-3.1-flash-lite" for a stronger
# (and pricier) one, if the cheap model's replies or structured outputs aren't good enough.
LLM_MODEL = "google_genai:gemini-2.5-flash-lite"

TOP_K = 5
# BGE-M3 cosine distance. From your tests: good matches ~0.40, no-match queries >= 0.51.
# Tune this on your own test queries.
NO_MATCH_DISTANCE = 0.50
MAX_SOURCE_CHARS = 1800   # per section text passed to the writer
HISTORY_MESSAGES = 14

DEFAULT_PRACTICE_ROLE = "the police officer at the station who receives the complaint"

# ----------------------------------------------------------------------------
# Structured-output schemas
# ----------------------------------------------------------------------------
class SectionMention(BaseModel):
    code: Literal["BNS", "IPC", "other", "unknown"] = "unknown"
    number: int


class Triage(BaseModel):
    intent: Literal["discuss", "draft_document", "start_practice", "end_practice"] = "discuss"
    domain: Literal["criminal", "civil_or_other", "unclear"] = "unclear"
    emergency: bool = False
    accused_or_notice: bool = False
    serious_matter: bool = False
    legal_query: str = ""
    document_type: str = ""
    practice_role: str = ""
    sections_mentioned: list[SectionMention] = Field(default_factory=list)


class TimelineEvent(BaseModel):
    when: str = "unknown"
    what: str


class Evidence(BaseModel):
    item: str
    status: Literal["have", "can_get", "dont_have", "unknown"] = "unknown"


class CaseFile(BaseModel):
    summary: str = ""
    other_party: str = ""
    place: str = ""
    timeline: list[TimelineEvent] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    harm: str = ""
    steps_already_taken: list[str] = Field(default_factory=list)
    missing_info: list[str] = Field(default_factory=list)


class Fit(BaseModel):
    section: int
    fit: Literal["strong", "possible", "no"]
    reason: str


class Judgement(BaseModel):
    fits: list[Fit] = Field(default_factory=list)


# ----------------------------------------------------------------------------
# Graph state
# ----------------------------------------------------------------------------
class State(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    case: dict               # CaseFile as dict (persists across turns)
    triage_result: dict      # Triage as dict (this turn)
    lookups: list            # results of exact section lookups the user asked for
    fits: list               # judged sections: section, name, fit, reason, text
    allowed: list            # section numbers the reply may mention
    retrieval_status: str    # match | weak | none | out_of_scope | skipped
    mode: str                # normal | practice
    practice_role: str
    practice_start: int      # index in messages where the practice session began
    attempts: int
    retry: bool
    verify_feedback: str
    last_node: str


# ----------------------------------------------------------------------------
# LLM helpers (everything that touches the model goes through these two)
# ----------------------------------------------------------------------------
_LLMS = {}


def _llm(temperature):
    is_gemini = LLM_MODEL.startswith("google_genai")
    if is_gemini and not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        raise RuntimeError("GOOGLE_API_KEY not found. Put GOOGLE_API_KEY=your-key in a .env file next to "
                           "chatbot.py (see .env.example) or set it in your environment.")
    if temperature not in _LLMS:
        try:
            # Pin the API version: the "v1" endpoint rejects system instructions, "v1beta" accepts them.
            extra = {"api_version": "v1beta"} if is_gemini else {}
            _LLMS[temperature] = init_chat_model(LLM_MODEL, temperature=temperature, **extra)
        except (TypeError, ValueError):  # older integration without an api_version option
            _LLMS[temperature] = init_chat_model(LLM_MODEL, temperature=temperature)
    return _LLMS[temperature]


# Some API versions refuse a separate system instruction. If that happens once, we switch to putting the
# instructions at the top of the first user message instead (works everywhere, slightly less strict).
_FOLD_SYSTEM = False


def _with_system(system, messages):
    if not _FOLD_SYSTEM:
        return [SystemMessage(system), *messages]
    first, rest = messages[0], list(messages[1:])
    folded = HumanMessage(f"INSTRUCTIONS (follow them for everything below):\n{system}\n\n---\n\n"
                          f"{_as_text(first.content)}")
    return [folded, *rest]


def _invoke(model, system, messages):
    global _FOLD_SYSTEM
    try:
        return model.invoke(_with_system(system, messages))
    except Exception as e:  # noqa: BLE001
        if not _FOLD_SYSTEM and "developer instruction" in str(e).lower():
            _FOLD_SYSTEM = True
            return model.invoke(_with_system(system, messages))
        raise


def _as_text(content):
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)


def _structured(schema, system, user):
    """One structured-output call; retried once, never silently defaulted."""
    model = _llm(0.0).with_structured_output(schema)
    last_err = None
    for _ in range(2):
        try:
            out = _invoke(model, system, [HumanMessage(user)])
            if out is not None:
                return out
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"Structured call for {schema.__name__} failed: {last_err}")


def _text(system, messages, temperature=0.3):
    return _as_text(_invoke(_llm(temperature), system, messages).content)


def _history(state, n=HISTORY_MESSAGES):
    msgs = list(state["messages"][-n:])
    while msgs and not isinstance(msgs[0], HumanMessage):  # providers want a user turn first
        msgs.pop(0)
    return msgs


def _last_user_text(state):
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            return _as_text(m.content)
    return ""


def _convo(state, n=8):
    return "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {_as_text(m.content)[:700]}"
        for m in _history(state, n)
    )


def _clip(text, n=MAX_SOURCE_CHARS):
    return text if len(text) <= n else text[:n] + " ...[truncated]"


# ----------------------------------------------------------------------------
# Citation guard (deterministic - this is the anti-hallucination backstop)
# ----------------------------------------------------------------------------
# Matches "Section 318(4), 319 and 335", "s. 351", "BNS 103", "धारा 351". Needs >=2 digits after a list
# connector so "Section 351 and 3 witnesses" does not count "3" as a section.
CITE_RE = re.compile(
    r"(?<![A-Za-z])(?:sections?|sec\.?|s\.|धारा|BNS)\s*"
    r"(\b\d{1,3}\b(?:\s*\([^)]*\))*(?:\s*(?:,|and|&|/|or)\s*\b\d{2,3}\b(?:\s*\([^)]*\))*)*)",
    re.I,
)
# Case-law style citations: "X v. Y", "AIR 1999 SC", "SCC", "Cri LJ"
CASE_RE = re.compile(r"\b(?:AIR|SCC|SCR|Cri\.?\s?L\.?J\.?)\b|\b[A-Z][\w.]+\s+v(?:s)?\.\s+[A-Z]")


def cited_sections(text):
    nums = set()
    for m in CITE_RE.finditer(text):
        chunk = re.sub(r"\([^)]*\)", "", m.group(1))
        nums.update(int(n) for n in re.findall(r"\d+", chunk))
    return nums


def _redact(text, allowed):
    def fix(m):
        return re.sub(
            r"\b(\d{1,3})\b(?!\s*\))",
            lambda n: n.group(0) if int(n.group(1)) in allowed else "[unverified]",
            m.group(0),
        )

    return CITE_RE.sub(fix, text)


def _guard(text, allowed=frozenset()):
    """Redact unknown section numbers and drop lines that cite case law."""
    text = _redact(text, set(allowed))
    return "\n".join(l for l in text.split("\n") if not CASE_RE.search(l))


# ----------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------
LEGAL_AID = "free legal aid is available through your District Legal Services Authority (NALSA helpline 15100)"

EMERGENCY_TEXT = (
    "**If you are in danger right now, please call 112 (India's emergency number) immediately**, "
    "or move to a safe place with other people around you.\n\n"
    "अगर आप अभी खतरे में हैं, तो तुरंत **112** पर कॉल करें या किसी सुरक्षित जगह पर लोगों के पास चले जाएँ।\n\n"
    "Other helplines: women in distress **181** · child helpline **1098** · online financial fraud **1930**.\n\n"
    "Once you are safe, tell me what happened and I will help you write it down and plan the next step."
)

TRIAGE_SYSTEM = """TRIAGE
You triage messages for an AI helper that assists ordinary people in India with CRIMINAL-law problems
(threats, cheating/fraud, theft, assault, harassment, ...), using the Bharatiya Nyaya Sanhita (BNS, in force
since 1 July 2024). Classify the user's LATEST message, using the conversation for context.

- intent: "discuss" (tells their story, asks where they stand or what to do, anything else) |
  "draft_document" (wants a complaint, notice, letter, application or a summary written) |
  "start_practice" (wants to rehearse / role-play the other side) |
  "end_practice" (wants to stop practising or asks for feedback while MODE is practice).
- domain: "criminal" = someone committed an offence against the user (threat, cheating, theft, assault,
  harassment, online fraud, ...). Someone lying to get your money IS criminal even though money is involved.
  "civil_or_other" = deposit, salary, refund, contract, tenancy, consumer, family or property disputes with no
  clear offence. "unclear" = not enough information yet.
- emergency: true ONLY if someone is in immediate physical danger right now.
- accused_or_notice: the user is accused / arrested / called by police as a suspect, or received a police or
  court notice, summons, FIR copy or legal notice they must answer.
- serious_matter: sexual offence, a child under 18 involved, death or serious injury, domestic violence.
- legal_query: ENGLISH, 1-2 plain sentences describing what was done to whom and how, for searching a statute
  database. Empty string if no problem has been described yet (e.g. a greeting).
- document_type: when drafting, what kind (e.g. "police complaint", "legal notice", "case summary for a lawyer").
- practice_role: when starting practice, who to play (default: the police officer receiving the complaint).
- sections_mentioned: every section number the user names, with its code (BNS, IPC, other, unknown)."""

CASE_SYSTEM = """CASE FILE
You maintain a structured case file for a person describing a legal problem in India.
Return the FULL updated case file. Rules:
- Use only what the user has said. Never invent names, dates, amounts or facts.
- Keep earlier facts unless the user corrects them. Write in English (translate if needed), keep names as given.
- timeline: events in order; 'when' exactly as the user said it, or "unknown".
- evidence: each item of proof mentioned (messages, screenshots, bank statement, witness, medical report, ...)
  with status have / can_get / dont_have / unknown.
- missing_info: the most important missing facts for a complaint (max 5), e.g. dates, the other person's
  identity, whether police were approached."""

JUDGE_SYSTEM = """SECTION FIT CHECK
You check whether statute sections fit a person's situation. For each candidate decide:
strong   = the facts clearly match what the section covers;
possible = some facts match but an important element is unclear;
no       = it only shares words (e.g. 'threat', 'property', 'owner') but does not fit the situation.
Be strict. Judge ONLY from the section text given. Give a one-sentence reason for each."""

COMMON_RULES = """
Rules you must follow:
- Write in the user's own language and script (English, Hindi, Hinglish, ...). Keep section numbers as digits.
- Plain, warm, short sentences. No legal jargon without a one-line explanation.
- Cite law ONLY from the SOURCES block, as "BNS Section N". NEVER cite any other section number (not BNSS, IPC,
  CrPC, IT Act, or any other Act) and NEVER cite court cases. Describe procedures in words instead.
- Quote punishments only if they appear in the SOURCES text. Never promise or predict an outcome.
- Say "may apply" - whether an offence is made out is for the police and the court to decide.
- Separate what the law text says (cited) from general practical guidance (say it is general guidance).
- Be honest about what you do not know. If the matter is serious or time-sensitive, say a lawyer is needed."""

STATUS_TEXT = {
    "match": "The SOURCES below were checked against the user's facts. Explain how they may apply.",
    "weak": "The SOURCES below fit only partly. Say clearly that this is uncertain and a lawyer should confirm.",
    "none": "NO section in the legal database fits these facts. Say that plainly. Do NOT cite any section number. "
            "Give only general next steps and recommend a lawyer.",
    "out_of_scope": "This looks like a civil/consumer/labour/other matter. Your database covers only criminal law "
                    "(BNS), so you have NO verified source for it. Say that plainly. Do NOT cite any law or section "
                    "number. You can still organise their story, explain in plain words (as general guidance, not "
                    "verified) what route such problems usually take, and recommend a lawyer or legal aid. If the "
                    "facts suggest deliberate lying or theft, mention a criminal angle may exist for a lawyer to assess.",
    "skipped": "No legal problem has been described yet. Greet briefly and ask what happened. Do not cite law.",
}

ADVISE_SYSTEM = """ADVISE
You are a careful AI legal helper for ordinary people in India. You are NOT a lawyer.
Reply structure (skip a part if empty; keep the whole reply compact):
1. **What I understood** - 2-3 lines.
2. **Where you stand** - the sections that may apply, with the plain meaning of each, grounded in SOURCES.
3. **Your options** - numbered, practical (e.g. write the complaint down, preserve evidence, approach the police
   station, what to do if they refuse, whether a lawyer is needed). Mark this as general guidance.
4. **Your proof** - what they have and what to collect or keep safe.
5. **What I still need to know** - at most 3 short questions.
6. Offer next steps: drafting the complaint/letter, or practising the conversation with the other side.
""" + COMMON_RULES

DRAFT_SYSTEM = """DRAFT
You draft documents for ordinary people in India from their case file. You are NOT a lawyer.
Rules for the draft:
- Use ONLY facts in the case file. For anything unknown write a placeholder in [SQUARE BRACKETS]
  (name, address, date, police station, amount...). Never invent facts.
- Police complaint: addressed to the Station House Officer; subject line; facts in time order; evidence list;
  where SOURCES allow, one line saying the facts appear to disclose offences under those BNS sections "as
  understood by the complainant - for the police to decide"; request to register an FIR and investigate;
  declaration, date, signature.
- Legal notice / demand letter (non-criminal matters): facts, the demand, a reasonable deadline. Do NOT cite any
  law or section.
- Case summary for a lawyer: one page - parties, timeline, proof, what the person wants, open questions.
- Write in the language the user asked for; if not stated, write in English and say they can ask for Hindi
  or another language.
After the draft add a short "Before you send this" checklist (placeholders to fill, proof to attach).
""" + COMMON_RULES

PRACTICE_SYSTEM = """PRACTICE
You are role-playing the OTHER SIDE so the user can rehearse a real conversation. Your role: {role}.
- Stay in character. Be realistic, not cruel: ask pointed questions, raise the objections a real person would
  (e.g. "this is a civil matter, settle it", "do you have proof?", "why did you wait so long?").
- One question or objection per turn, 1-3 sentences. React to what the user actually said.
- Do NOT give legal advice and do NOT cite any section numbers or cases while in character.
- Use the case file only to know the facts; do not leak facts the user has not told your character yet.
- Reply in the user's language.
CASE FILE: {case}"""

FEEDBACK_SYSTEM = """PRACTICE FEEDBACK
The user just finished a role-play rehearsal. Give friendly, concrete feedback in the user's language:
- 2 things they did well; up to 3 gaps (facts missing or inconsistent with the case file, unclear dates,
  proof they did not mention, being pushed around by objections);
- for the toughest objection, a short model answer they could use;
- one next step. Do NOT cite section numbers or cases.
CASE FILE: {case}"""

HANDOFF_SYSTEM = """HANDOFF
The user says they are accused / have been arrested / have received a police or court notice or summons.
This needs a real lawyer now. In the user's language, warmly and briefly:
- acknowledge it is stressful; explain WHY a lawyer matters here (deadlines, what they say or sign can matter);
- practical safe steps: do not ignore the notice or summons, note the dates, keep every paper, do not sign what
  they do not understand, they may ask to speak to a lawyer before giving a detailed statement;
- offer to prepare a one-page summary of their situation to take to a lawyer.
Do NOT cite any section numbers or cases, and do not judge guilt or innocence."""


# ----------------------------------------------------------------------------
# Nodes
# ----------------------------------------------------------------------------
def triage(state: State):
    case = state.get("case") or {}
    user = (
        f"MODE: {state.get('mode', 'normal')}\n"
        f"CASE SO FAR: {case.get('summary') or '(none)'}\n\n"
        f"CONVERSATION:\n{_convo(state, 6)}\n\nClassify the LATEST user message."
    )
    t = _structured(Triage, TRIAGE_SYSTEM, user)
    return {"triage_result": t.model_dump()}


def emergency(state: State):
    return {"messages": [AIMessage(EMERGENCY_TEXT)]}


def handoff(state: State):
    reply = _guard(_text(HANDOFF_SYSTEM, _history(state, 8)))
    return {"messages": [AIMessage(reply + f"\n\n*{LEGAL_AID[0].upper() + LEGAL_AID[1:]}.*")]}


def update_case(state: State):
    old = state.get("case") or CaseFile().model_dump()
    user = (
        f"EXISTING CASE FILE:\n{json.dumps(old, ensure_ascii=False)}\n\n"
        f"CONVERSATION:\n{_convo(state, 10)}\n\nReturn the full updated case file."
    )
    return {"case": _structured(CaseFile, CASE_SYSTEM, user).model_dump()}


def _merge_hits(*hit_lists):
    best = {}
    for hits in hit_lists:
        for h in hits:
            cur = best.get(h["section"])
            if cur is None or h["distance"] < cur["distance"]:
                best[h["section"]] = h
    return sorted(best.values(), key=lambda h: h["distance"])[:TOP_K]


def retrieve(state: State):
    """Exact lookups -> semantic search -> distance gate -> LLM fit check -> allow-list."""
    t = state["triage_result"]
    out = {"hits": [], "fits": [], "allowed": [], "lookups": [], "retrieval_status": "skipped",
           "attempts": 0, "retry": False, "verify_feedback": ""}

    # 1) sections the user named: check they really exist (never trust the number)
    candidates = []
    user_numbers = set()
    for m in t.get("sections_mentioned", []):
        n, code = m["number"], m["code"]
        user_numbers.add(n)
        if code in ("BNS", "unknown"):
            sec = get_section(n)
            out["lookups"].append({"code": code, "number": n, "exists": sec is not None,
                                   "name": sec["metadata"]["Section_name"] if sec else ""})
            if sec:
                candidates.append({"section": n, "name": sec["metadata"]["Section_name"],
                                   "distance": 0.0, "text": sec["text"]})
        else:
            out["lookups"].append({"code": code, "number": n, "exists": None, "name": ""})
    out["allowed"] = sorted(user_numbers)  # echoing a number the user typed is fine

    if t["domain"] == "civil_or_other":
        out["retrieval_status"] = "out_of_scope"
        return out
    if not t.get("legal_query", "").strip() and not candidates:
        return out

    # 2) semantic search on the plain-English query AND the raw message (BGE-M3 is multilingual)
    hits = _merge_hits(search(t["legal_query"], k=TOP_K) if t.get("legal_query") else [],
                       search(_last_user_text(state), k=TOP_K))
    if hits and hits[0]["distance"] > NO_MATCH_DISTANCE and not candidates:
        out["hits"], out["retrieval_status"] = hits, "none"
        return out
    seen = {c["section"] for c in candidates}
    candidates += [h for h in hits if h["section"] not in seen and h["distance"] <= NO_MATCH_DISTANCE]
    if not candidates:
        out["retrieval_status"] = "none"
        return out

    # 3) LLM checks each candidate against the facts; only strong/possible survive
    case = state.get("case") or {}
    listing = "\n\n".join(f"[Section {c['section']}] {c['name']}\n{_clip(c['text'], 900)}" for c in candidates)
    user = (f"SITUATION: {case.get('summary', '')}\nFACTS SEARCHED: {t.get('legal_query', '')}\n\n"
            f"CANDIDATE SECTIONS:\n{listing}")
    judged = _structured(Judgement, JUDGE_SYSTEM, user)
    by_num = {c["section"]: c for c in candidates}
    fits = []
    for f in judged.fits:
        if f.fit != "no" and f.section in by_num:  # ignore anything not actually retrieved
            c = by_num[f.section]
            fits.append({"section": f.section, "name": c["name"], "fit": f.fit,
                         "reason": f.reason, "text": c["text"]})
    out["hits"] = hits
    out["fits"] = fits
    out["allowed"] = sorted(user_numbers | {f["section"] for f in fits})
    if not fits:
        out["retrieval_status"] = "none"
    else:
        out["retrieval_status"] = "match" if any(f["fit"] == "strong" for f in fits) else "weak"
    return out


def _context_block(state):
    """Everything the writer is allowed to rely on."""
    parts = [STATUS_TEXT[state.get("retrieval_status", "skipped")]]
    fits = state.get("fits", [])
    if fits:
        parts.append("SOURCES (the ONLY law you may cite):\n\n" + "\n\n".join(
            f"[BNS Section {f['section']}] {f['name']} - fit: {f['fit']} ({f['reason']})\n{_clip(f['text'])}"
            for f in fits))
    else:
        parts.append("SOURCES: (none - cite no section numbers)")
    for l in state.get("lookups", []):
        n = l["number"]
        if l["exists"] is True:
            parts.append(f"The user mentioned BNS Section {n}: it exists - '{l['name']}'.")
        elif l["exists"] is False:
            parts.append(f"The user mentioned Section {n}: there is NO Section {n} in the BNS database. "
                         f"Tell them so; do not guess what it means.")
        elif l["code"] == "IPC":
            parts.append(f"The user mentioned IPC Section {n}. The IPC was replaced by the BNS on 1 July 2024 and "
                         f"numbers changed. You have no verified mapping, so do not say which BNS section matches; "
                         f"work from the facts.")
        else:
            parts.append(f"The user mentioned Section {n} of another law. You have no source for it; do not explain it.")
    parts.append("CASE FILE:\n" + json.dumps(state.get("case") or {}, ensure_ascii=False, indent=1))
    if state["triage_result"].get("serious_matter"):
        parts.append("This is a SERIOUS matter (sexual offence, child, death/serious injury or domestic violence). "
                     "Say early that a lawyer and a support organisation should lead; be gentle.")
    if state.get("verify_feedback"):
        parts.append("CORRECTION: " + state["verify_feedback"])
    return "\n\n".join(parts)


def advise(state: State):
    reply = _text(ADVISE_SYSTEM + "\n\n" + _context_block(state), _history(state))
    return {"messages": [AIMessage(reply)], "last_node": "advise", "retry": False}


def draft(state: State):
    t = state["triage_result"]
    doc = t.get("document_type") or ("police complaint" if state.get("retrieval_status") in ("match", "weak")
                                      else "case summary for a lawyer")
    system = DRAFT_SYSTEM + f"\n\nDOCUMENT TO WRITE: {doc}\n\n" + _context_block(state)
    return {"messages": [AIMessage(_text(system, _history(state), temperature=0.2))],
            "last_node": "draft", "retry": False}


def _footer(state):
    if state.get("retrieval_status") in ("match", "weak"):
        basis = "general information based on the text of the Bharatiya Nyaya Sanhita, 2023"
    else:
        basis = "general information that I could not check against a legal source"
    return (f"\n\n---\n*I'm an AI, not a lawyer. This is {basis}, not legal advice. For anything serious or "
            f"time-sensitive please speak to a lawyer; {LEGAL_AID}.*")


def verify(state: State):
    """Check every section number in the reply against the allow-list. One rewrite, then redact."""
    last = state["messages"][-1]
    reply = _as_text(last.content)
    allowed = set(state.get("allowed", []))
    bad = sorted(cited_sections(reply) - allowed)
    case_cite = bool(CASE_RE.search(reply))

    if (bad or case_cite) and state.get("attempts", 0) < 1:
        problems = []
        if bad:
            problems.append(f"You mentioned section number(s) {bad} that are not in SOURCES. Remove them; "
                            f"cite only sections from SOURCES (or none).")
        if case_cite:
            problems.append("You included a court-case citation. Remove it; never cite cases.")
        return {"retry": True, "attempts": state.get("attempts", 0) + 1,
                "verify_feedback": " ".join(problems),
                "messages": [RemoveMessage(id=last.id)]}

    body = _guard(reply, allowed)
    if len(body.strip()) < 20:  # guard removed (almost) everything -> never send an empty answer
        body = ("I couldn't put together an answer here that I can fully stand behind. Could you tell me a bit "
                "more about what happened (who, when, where, and what proof you have)? For anything urgent, "
                "please speak to a lawyer.")
    return {"retry": False, "messages": [AIMessage(content=body + _footer(state), id=last.id)]}


def practice(state: State):
    starting = state.get("mode") != "practice"
    role = (state["triage_result"].get("practice_role") or DEFAULT_PRACTICE_ROLE) if starting \
        else state.get("practice_role", DEFAULT_PRACTICE_ROLE)
    system = PRACTICE_SYSTEM.format(role=role, case=json.dumps(state.get("case") or {}, ensure_ascii=False))
    if starting:
        system += ("\n\nFIRST TURN: begin with one short line saying the practice has started, that you are playing "
                   f"{role}, and that they can type 'stop' for feedback. Then ask your first in-character question.")
    reply = _guard(_text(system, _history(state, 12)))
    update = {"messages": [AIMessage(reply)], "mode": "practice", "practice_role": role}
    if starting:
        update["practice_start"] = len(state["messages"]) - 1
    return update


def practice_feedback(state: State):
    start = state.get("practice_start", 0)
    transcript = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Other side'}: {_as_text(m.content)}"
        for m in state["messages"][start:-1]
    )
    system = FEEDBACK_SYSTEM.format(case=json.dumps(state.get("case") or {}, ensure_ascii=False))
    reply = _guard(_text(system, [HumanMessage("REHEARSAL TRANSCRIPT:\n" + transcript)]))
    return {"messages": [AIMessage(reply)], "mode": "normal"}


# ----------------------------------------------------------------------------
# Routing + graph
# ----------------------------------------------------------------------------
def route_after_triage(state: State):
    t = state["triage_result"]
    if t["emergency"]:
        return "emergency"
    if state.get("mode") == "practice":
        return "practice_feedback" if t["intent"] == "end_practice" else "practice"
    if t["accused_or_notice"] and t["intent"] != "draft_document":
        return "handoff"
    return "update_case"


def route_after_case(state: State):
    return "practice" if state["triage_result"]["intent"] == "start_practice" else "retrieve"


def route_after_retrieve(state: State):
    wants_draft = state["triage_result"]["intent"] == "draft_document"
    # not enough facts to draft -> let the advisor ask questions instead
    return "draft" if wants_draft and (state.get("case") or {}).get("summary") else "advise"


def route_after_verify(state: State):
    return state["last_node"] if state.get("retry") else END


def build_graph():
    g = StateGraph(State)
    for name, fn in [("triage", triage), ("emergency", emergency), ("handoff", handoff),
                     ("update_case", update_case), ("retrieve", retrieve), ("advise", advise),
                     ("draft", draft), ("verify", verify), ("practice", practice),
                     ("practice_feedback", practice_feedback)]:
        g.add_node(name, fn)
    g.add_edge(START, "triage")
    g.add_conditional_edges("triage", route_after_triage,
                            ["emergency", "handoff", "update_case", "practice", "practice_feedback"])
    g.add_conditional_edges("update_case", route_after_case, ["practice", "retrieve"])
    g.add_conditional_edges("retrieve", route_after_retrieve, ["draft", "advise"])
    g.add_edge("advise", "verify")
    g.add_edge("draft", "verify")
    g.add_conditional_edges("verify", route_after_verify, ["advise", "draft", END])
    for end_node in ("emergency", "handoff", "practice", "practice_feedback"):
        g.add_edge(end_node, END)
    return g.compile(checkpointer=MemorySaver())


graph = build_graph()


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def chat(message: str, thread_id: str = "default") -> str:
    """Send one user message; conversation + case file are remembered per thread_id."""
    out = graph.invoke({"messages": [HumanMessage(message)]},
                       {"configurable": {"thread_id": thread_id}})
    return _as_text(out["messages"][-1].content)


def show_case(thread_id: str = "default") -> dict:
    """The structured case file built so far (summary, timeline, evidence, missing info)."""
    snap = graph.get_state({"configurable": {"thread_id": thread_id}})
    return snap.values.get("case", {})


def get_state(thread_id: str = "default") -> dict:
    """Small, UI-friendly view of a conversation: case file, mode, and which sections the last answer used."""
    v = graph.get_state({"configurable": {"thread_id": thread_id}}).values
    return {
        "case": v.get("case", {}),
        "mode": v.get("mode", "normal"),
        "status": v.get("retrieval_status", ""),
        "sections": [{"section": f["section"], "name": f["name"], "fit": f["fit"], "reason": f["reason"]}
                     for f in v.get("fits", [])],
        "lookups": v.get("lookups", []),
    }