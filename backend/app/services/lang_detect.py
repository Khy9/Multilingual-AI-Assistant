"""Language detection for code-mixed regional input.

Why this module exists
----------------------
Off-the-shelf detectors (langdetect, fastText, CLD3) assume ONE language per
document and decide using character n-grams. That assumption breaks twice for our
users:

  1. Code-mixing. "Naaku ee document lo revenue figures kavali" is Telugu grammar
     with English nouns. A single-label detector must pick one and is wrong either
     way.
  2. Romanization. Telugu and Hindi typed in Latin script have no Telugu/Devanagari
     characters at all, so a script- or n-gram-based detector sees only Latin and
     guesses something like Portuguese, Indonesian or Somali with high confidence.

So detection runs in two stages:

  Stage 1 (free, instant, offline) - detect_scripts()
      Classify each token by Unicode block. This reliably separates English from
      native-script Telugu/Hindi, and detects mixed-script input. If the text
      contains non-Latin script, we are done — no API call needed.

  Stage 2 (only for all-Latin, ambiguous text) - classify_with_llm()
      Ask Gemini itself to name the language(s) present. An LLM handles romanized
      code-mixing well because it reads words, not character statistics. Kept as a
      separate, clearly-named function so the cheap path stays independent of it.
      Falls back to a keyword heuristic if the API is unavailable or rate-limited.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Literal

log = logging.getLogger(__name__)

Register = Literal["formal", "casual"]

# --- Unicode blocks -----------------------------------------------------------
TELUGU_RANGE = (0x0C00, 0x0C7F)
DEVANAGARI_RANGE = (0x0900, 0x097F)

LANGUAGE_NAMES = {
    "en": "English",
    "te": "Telugu",
    "hi": "Hindi",
    "te-rom": "Telugu (romanized)",
    "hi-rom": "Hindi (romanized)",
}

# Function words that are strong romanized markers. Deliberately words that do NOT
# collide with common English words — this is the offline safety net for stage 2,
# not the primary detector.
TELUGU_ROMAN_MARKERS = {
    "meeku", "naaku", "nenu", "meeru", "emiti", "emi", "kavali", "cheppu",
    "ela", "ekkada", "endhuku", "enduku", "cheyyi", "chey", "unnaru", "undi",
    "ledu", "kada", "chala", "bagundi", "telusu", "teliyadu", "gurinchi",
    "lo", "loni", "tho", "kosam", "ante", "avunu", "sari", "ippudu", "cheyyandi",
}
HINDI_ROMAN_MARKERS = {
    "aap", "aapko", "mujhe", "mera", "meri", "kya", "kaise", "kaisa", "kahan",
    "kyun", "kyon", "hai", "hain", "tha", "thi", "nahi", "nahin", "haan",
    "chahiye", "bata", "batao", "karo", "karna", "kar", "raha", "rahi", "bahut",
    "thoda", "abhi", "kuch", "sab", "bhi", "matlab", "samajh",
    # NOTE: "yaar" and "bhai" are deliberately excluded. They are so absorbed into
    # everyday Indian English ("hey yaar, send me the file") that they signal a
    # casual register, not Hindi grammar. They live in CASUAL_MARKERS instead.
}

# Casual-register cues (romanized + English chat style).
CASUAL_MARKERS = {
    "yaar", "bro", "dude", "hey", "hi", "yo", "plz", "pls", "u", "ur", "gonna",
    "wanna", "lol", "ra", "ba", "anna", "akka", "bhai", "arre", "abey",
}
FORMAL_MARKERS = {
    "kindly", "please", "regards", "sir", "madam", "sincerely", "request",
    "would", "could", "may", "hereby", "respectfully", "andi", "gaaru", "garu",
    "aap", "aapko", "kripya", "dhanyavaad",
}

_TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)


@dataclass
class DetectionResult:
    """Everything the chat route needs to build a system prompt."""

    languages: list[str] = field(default_factory=list)  # e.g. ["te-rom", "en"]
    code_mixed: bool = False
    register: Register = "casual"
    method: str = "script"  # "script" | "llm" | "heuristic" | "profile"
    confidence: float = 0.0
    script_counts: dict[str, int] = field(default_factory=dict)

    @property
    def label(self) -> str:
        """Human-readable badge for the UI, e.g. 'Telugu (romanized) + English'."""
        names = [LANGUAGE_NAMES.get(code, code) for code in self.languages]
        if not names:
            return "Unknown"
        return " + ".join(names)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["label"] = self.label
        return data


def _in_range(char: str, bounds: tuple[int, int]) -> bool:
    return bounds[0] <= ord(char) <= bounds[1]


def detect_scripts(text: str) -> dict[str, int]:
    """Stage 1: count tokens per Unicode script.

    Returns counts keyed by "telugu", "devanagari", "latin", "other". Purely
    offline and instant — this runs on every request.
    """
    counts = {"telugu": 0, "devanagari": 0, "latin": 0, "other": 0}

    for token in _TOKEN_RE.findall(text):
        scripts = set()
        for char in token:
            if _in_range(char, TELUGU_RANGE):
                scripts.add("telugu")
            elif _in_range(char, DEVANAGARI_RANGE):
                scripts.add("devanagari")
            elif char.isascii() and char.isalpha():
                scripts.add("latin")
            elif char.isalpha():
                scripts.add("other")
        # A token counts toward every script it contains, so "Telugu-లో" is mixed.
        for script in scripts:
            counts[script] += 1

    return counts


def detect_register(text: str) -> Register:
    """Cheap register guess from lexical cues and punctuation style.

    Shapes the system prompt so the model mirrors the user's formality instead of
    flattening everything into neutral textbook prose.
    """
    words = {word.lower() for word in _TOKEN_RE.findall(text)}
    formal_hits = len(words & FORMAL_MARKERS)
    casual_hits = len(words & CASUAL_MARKERS)

    if formal_hits > casual_hits:
        return "formal"
    if casual_hits > formal_hits:
        return "casual"
    # Tie-break on surface style: long sentences without chat punctuation read formal.
    if len(words) >= 12 and not re.search(r"[!?]{2,}|\.\.\.|:\)|😊|😂", text):
        return "formal"
    return "casual"


def _heuristic_roman_languages(text: str) -> tuple[list[str], float]:
    """Offline fallback for romanized input: match known function words.

    Used when the LLM classifier is unavailable (no key, rate limited, bad JSON).
    """
    words = [word.lower() for word in _TOKEN_RE.findall(text)]
    if not words:
        return ["en"], 0.0

    word_set = set(words)
    telugu_hits = len(word_set & TELUGU_ROMAN_MARKERS)
    hindi_hits = len(word_set & HINDI_ROMAN_MARKERS)

    if telugu_hits == 0 and hindi_hits == 0:
        return ["en"], 0.6

    languages: list[str] = []
    if telugu_hits >= hindi_hits and telugu_hits > 0:
        languages.append("te-rom")
    if hindi_hits > telugu_hits or (hindi_hits > 0 and telugu_hits == 0):
        languages.append("hi-rom")

    # Any Latin word that is not a regional marker is treated as the English part.
    regional = TELUGU_ROMAN_MARKERS | HINDI_ROMAN_MARKERS
    if any(word not in regional for word in word_set):
        languages.append("en")

    hits = max(telugu_hits, hindi_hits)
    confidence = min(0.5 + 0.1 * hits, 0.9)
    return languages, confidence


_LLM_CLASSIFIER_PROMPT = """You are a language identification system for Indian code-mixed text.

Classify the languages present in the user's text. The text is written in Latin
script but may be romanized Telugu or romanized Hindi, possibly mixed with English
("Tenglish" / "Hinglish").

Use exactly these language codes:
  en      = English
  te-rom  = Telugu written in Latin script (e.g. "meeku em kavali")
  hi-rom  = Hindi written in Latin script (e.g. "aapko kya chahiye")

Respond with ONLY a JSON object, no markdown fences, no commentary:
{"languages": ["te-rom", "en"], "code_mixed": true, "register": "casual"}

register is "formal" or "casual". code_mixed is true when more than one language
is present."""


async def classify_with_llm(text: str) -> DetectionResult | None:
    """Stage 2: ask Gemini to identify romanized / code-mixed languages.

    Returns None on any failure (no API key, 429, unparseable output) so the caller
    can fall back to the offline heuristic. Deliberately a separate function: the
    cheap script pass must never depend on the network.
    """
    # Imported here to keep this module importable (and testable) without the SDK
    # configured.
    from app.services import llm

    try:
        raw = await llm.complete(
            prompt=f"Text: {text}",
            system_prompt=_LLM_CLASSIFIER_PROMPT,
            temperature=0.0,
        )
    except llm.LLMError as exc:
        log.warning("LLM language classifier unavailable (%s); using heuristic.", exc)
        return None

    # Models sometimes wrap JSON in ```json fences despite instructions.
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        log.warning("LLM classifier returned non-JSON: %r", raw[:200])
        return None

    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        log.warning("LLM classifier returned invalid JSON: %r", raw[:200])
        return None

    languages = [str(code) for code in payload.get("languages", []) if code]
    if not languages:
        return None

    register: Register = "formal" if payload.get("register") == "formal" else "casual"

    return DetectionResult(
        languages=languages,
        code_mixed=bool(payload.get("code_mixed", len(languages) > 1)),
        register=register,
        method="llm",
        confidence=0.95,
    )


async def detect(text: str, *, preferred_languages: list[str] | None = None) -> DetectionResult:
    """Full two-stage detection. This is the function routers call.

    `preferred_languages` comes from the persistent user profile and only breaks
    ties — it never overrides positive evidence in the text itself.
    """
    text = (text or "").strip()
    if not text:
        return DetectionResult(languages=["en"], method="empty")

    counts = detect_scripts(text)
    register = detect_register(text)
    non_latin = counts["telugu"] + counts["devanagari"]

    # --- Case A: native script present. Script evidence is conclusive. ---------
    if non_latin > 0:
        languages: list[str] = []
        if counts["telugu"]:
            languages.append("te")
        if counts["devanagari"]:
            languages.append("hi")
        if counts["latin"]:
            languages.append("en")
        return DetectionResult(
            languages=languages,
            code_mixed=len(languages) > 1,
            register=register,
            method="script",
            confidence=0.99,
            script_counts=counts,
        )

    # --- Case B: all Latin. Script tells us nothing; escalate. ----------------
    result = await classify_with_llm(text)
    if result is None:
        languages, confidence = _heuristic_roman_languages(text)
        result = DetectionResult(
            languages=languages,
            code_mixed=len(languages) > 1,
            register=register,
            method="heuristic",
            confidence=confidence,
        )

    result.script_counts = counts

    # Returning-user bias: only when the text gave us nothing but plain English at
    # low confidence, prefer the pair this user habitually writes in.
    if preferred_languages and result.languages == ["en"] and result.confidence < 0.7:
        result.languages = list(preferred_languages)
        result.code_mixed = len(preferred_languages) > 1
        result.method = "profile"

    return result
