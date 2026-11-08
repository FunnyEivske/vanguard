"""Local Name and Wake Word Spotting for V.A.N.G.U.A.R.D.

Detects when an utterance addresses a specific persona or refers to the Vanguard substrate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class NameSpottingResult:
    """Detection result for spotted persona or substrate invocation."""

    spotted: bool
    target_id: str | None = None
    matched_phrase: str | None = None
    is_substrate_reference: bool = False
    clean_text: str = ""


class NameSpotter:
    """Pattern-based and phonetic name spotter for Vanguard household."""

    PATTERNS: dict[str, list[re.Pattern[str]]] = {
        "jarvis": [
            re.compile(r"\b(jarvis|j\.a\.r\.v\.i\.s\.)\b", re.IGNORECASE),
        ],
        "friday": [
            re.compile(r"\b(friday|firday|f\.r\.i\.d\.a\.y\.)\b", re.IGNORECASE),
        ],
        "edith": [
            re.compile(r"\b(edith|eath|e\.d\.i\.t\.h\.)\b", re.IGNORECASE),
        ],
        "dume": [
            re.compile(r"\b(dum-?e|d\.u\.m\.-?e|dumper|dummy)\b", re.IGNORECASE),
        ],
        "u": [
            re.compile(r"\b(unit|u)\b", re.IGNORECASE),
        ],
        "vanguard": [
            re.compile(r"\b(vanguard|v\.a\.n\.g\.u\.a\.r\.d\.|the\s+house|huset|nettverket|the\s+network)\b", re.IGNORECASE),
        ],
    }

    def spot(self, text: str) -> NameSpottingResult:
        """Scan input text for persona or system invocations."""
        normalized = text.strip()

        for persona_id, patterns in self.PATTERNS.items():
            for pat in patterns:
                match = pat.search(normalized)
                if match:
                    matched = match.group(0)
                    is_substrate = persona_id == "vanguard"
                    return NameSpottingResult(
                        spotted=True,
                        target_id=persona_id,
                        matched_phrase=matched,
                        is_substrate_reference=is_substrate,
                        clean_text=normalized,
                    )

        return NameSpottingResult(spotted=False, target_id=None, clean_text=normalized)
