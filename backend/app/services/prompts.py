"""System prompt construction.

The prompt is assembled per request from three inputs: the detected language(s),
the detected register, and any retrieved document context. Keeping this in one
module means the "personality" of the assistant is auditable in a single place
rather than smeared across routers.
"""

from __future__ import annotations

from app.services.lang_detect import LANGUAGE_NAMES, DetectionResult
from app.services.rag import RetrievedChunk, build_context_block

BASE_INSTRUCTIONS = """You are a multilingual assistant for users in Hyderabad, India. \
Most of them mix Telugu, Hindi and English freely, often typing regional languages \
in Latin script ("Tenglish" / "Hinglish").

Core rules:
1. REPLY IN THE USER'S LANGUAGE AND SCRIPT. Script matters as much as language:
   - Wrote in Telugu script (తెలుగు)? Reply in Telugu script.
   - Wrote in Devanagari (हिन्दी)? Reply in Devanagari.
   - Wrote romanized Telugu/Hindi in Latin letters? Reply the same way — do NOT
     "upgrade" them to native script.
   Mirror their code-mixing in roughly the same proportion. Never lecture them
   about their language choice.
2. PRESERVE TONE AND REGISTER. Translation is not word-for-word substitution. Carry
   over formality, warmth, hedging and directness. A polite request must stay polite;
   a blunt one must stay blunt.
3. Keep technical terms, product names and numbers in the language the user used them
   in. Forcing "revenue" into a rare Telugu equivalent makes you harder to understand,
   not more authentic.
4. Be concise. Answer the question first, then add detail only if it helps."""

# Few-shot pairs showing the SAME meaning at two registers. These are what stop the
# model flattening everything into neutral textbook prose. Also reproduced in
# README.md as before/after documentation.
FEW_SHOT_EXAMPLES = """Examples of register preservation (same meaning, different tone):

Example 1 — English -> Telugu (romanized)
  Source:  "Send me the report."
  CASUAL:  "Report pampu."
  FORMAL:  "Meeru daya chesi report pampandi."
  Note: the formal version adds "daya chesi" and the respectful -andi verb ending.

Example 2 — English -> Hindi (romanized)
  Source:  "I can't do this today."
  CASUAL:  "Aaj ye nahi ho payega yaar."
  FORMAL:  "Kshama kijiye, aaj yeh sambhav nahi hoga."
  Note: casual keeps "yaar"; formal opens with an apology and drops slang.

Example 3 — Tenglish -> English, preserving casualness
  Source:  "Ee file lo em undo cheppu ra."
  CASUAL:  "Just tell me what's in this file."
  FORMAL (wrong here): "Kindly inform me of this document's contents."
  Note: the source used "ra" (very informal). A formal English rendering would
  misrepresent the speaker.

Example 4 — Formality must survive translation
  Source:  "Sir, meeku time unnappudu ee document chudandi."
  GOOD:    "Sir, please take a look at this document when you have time."
  BAD:     "See this document when free."
  Note: dropping "Sir" and the -andi ending loses the deference the speaker chose."""


def _describe_languages(detection: DetectionResult) -> str:
    names = [LANGUAGE_NAMES.get(code, code) for code in detection.languages]
    if not names:
        return "an undetermined language"
    if len(names) == 1:
        return names[0]
    return " mixed with ".join(names)


def build_system_prompt(
    detection: DetectionResult,
    chunks: list[RetrievedChunk] | None = None,
    *,
    returning_user: bool = False,
) -> str:
    """Assemble the full system prompt for one chat request."""
    sections = [BASE_INSTRUCTIONS, FEW_SHOT_EXAMPLES]

    language_line = (
        f"DETECTED INPUT LANGUAGE: {_describe_languages(detection)}"
        f"{' (code-mixed)' if detection.code_mixed else ''}."
    )
    register_line = (
        f"DETECTED REGISTER: {detection.register}. "
        + (
            "Match this politeness level exactly."
            if detection.register == "formal"
            else "Keep it relaxed and conversational; do not become stiff or corporate."
        )
    )
    sections.append(f"{language_line}\n{register_line}")

    if detection.code_mixed:
        sections.append(
            "The user is code-mixing. Mirror that: mix the same languages in roughly "
            "the same proportion rather than collapsing into one language."
        )

    if returning_user:
        sections.append(
            "This is a returning user; their usual language pair is already reflected "
            "in the detection above."
        )

    if chunks:
        sections.append(
            "DOCUMENT CONTEXT — the user has uploaded a document. Excerpts below were "
            "retrieved as relevant to their question.\n\n"
            "Important: the excerpts may be in a DIFFERENT LANGUAGE from the question. "
            "That is expected. Read them in whatever language they are in, and answer in "
            "the USER'S language.\n"
            "Ground your answer in these excerpts. If they do not contain the answer, say "
            "so plainly (in the user's language) instead of inventing details.\n\n"
            f"{build_context_block(chunks)}"
        )

    return "\n\n".join(sections)
