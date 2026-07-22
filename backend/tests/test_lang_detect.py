"""Language detection test script.

Run it directly:
    python tests/test_lang_detect.py

Works with or without a GEMINI_API_KEY:
  - With a key, romanized cases go through the Gemini classifier (method="llm").
  - Without one, they fall back to the offline keyword heuristic (method="heuristic").
Either way it prints the detection output for every case, so you can see what the
two-stage detector actually decided.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Allow running this file directly from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.services import lang_detect  # noqa: E402

CASES: list[tuple[str, str, str]] = [
    (
        "Pure English",
        "Could you please summarise the third quarter revenue figures for me?",
        "en only, not code-mixed",
    ),
    (
        "Pure Telugu (Telugu script)",
        "మూడవ త్రైమాసికం ఆదాయ వివరాలు చెప్పండి",
        "te only, detected by Unicode script pass",
    ),
    (
        "Tenglish (romanized Telugu + English)",
        "Naaku ee document lo revenue figures kavali, cheppandi",
        "te-rom + en, code-mixed (needs stage 2 — all Latin script)",
    ),
    (
        "Hinglish (romanized Hindi + English)",
        "Aapko ye report kaise chahiye, PDF format mein bhejun kya?",
        "hi-rom + en, code-mixed (needs stage 2 — all Latin script)",
    ),
    (
        "Mixed script (Telugu script + English)",
        "ఈ report లో ఏముంది cheppandi",
        "te + en, code-mixed, detected by script pass alone",
    ),
    (
        "Casual English (register check)",
        "hey yaar can u just tell me what's in this file lol",
        "en, register should be casual",
    ),
]


async def main() -> None:
    settings = get_settings()
    print("=" * 78)
    print("LANGUAGE DETECTION TEST")
    print(f"Gemini key configured: {settings.llm_configured}  "
          f"(stage 2 uses {'the LLM classifier' if settings.llm_configured else 'the offline heuristic'})")
    print("=" * 78)

    for name, text, expectation in CASES:
        print(f"\n--- {name} ---")
        print(f"  input     : {text}")
        print(f"  expecting : {expectation}")

        result = await lang_detect.detect(text)

        print(f"  languages : {result.languages}   -> {result.label!r}")
        print(f"  code_mixed: {result.code_mixed}")
        print(f"  register  : {result.register}")
        print(f"  method    : {result.method}   (confidence {result.confidence})")
        print(f"  scripts   : {result.script_counts}")

    print("\n" + "=" * 78)
    print("Stage 1 (Unicode script pass) is offline and runs on every request.")
    print("Stage 2 (LLM classifier) runs ONLY for all-Latin text, where script")
    print("evidence cannot separate English from romanized Telugu/Hindi.")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
