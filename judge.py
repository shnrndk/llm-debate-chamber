"""
Judge Agent — Phase 3 of the debate pipeline.

The judge receives the complete debate transcript and produces a structured
verdict with chain-of-thought analysis, argument evaluation, a final answer,
and a confidence score.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional

from llm_client import call_llm
from prompts import JUDGE_SYSTEM, JUDGE_USER
from baselines import extract_yes_no
from config import JUDGE_TEMPERATURE


# ─── Result Model ─────────────────────────────────────────────────────────────

@dataclass
class JudgeVerdict:
    raw_output: str
    cot_analysis: str
    debater_a_strongest: str
    debater_a_weakest: str
    debater_b_strongest: str
    debater_b_weakest: str
    winner_reasoning: str
    verdict: str          # "Yes" or "No"
    confidence: int       # 1–5


# ─── Parsing ──────────────────────────────────────────────────────────────────

def _extract_section(text: str, tag: str) -> str:
    """
    Extract content after a section tag like "COT_ANALYSIS:" until the next tag.

    Handles all these formats the model might produce:
      WINNER_REASONING:\nsome text...
      WINNER_REASONING: some text...
      winner_reasoning:\nsome text...
    Stops at the next ALL-CAPS section header or end of string.
    """
    # Step 1: find where our tag ends
    tag_pattern = rf"(?i){re.escape(tag)}\s*:?\s*"
    m_tag = re.search(tag_pattern, text)
    if not m_tag:
        return ""

    # Step 2: grab everything from after the tag until the next section header
    remainder = text[m_tag.end():]

    # A section header is a line that is ALL_CAPS_WITH_UNDERSCORES followed by optional colon
    # We stop before it (or at end of string)
    stop = re.search(r"\n\s*[A-Z][A-Z_]{2,}\s*:", remainder)
    if stop:
        body = remainder[:stop.start()]
    else:
        body = remainder

    return body.strip()


def _extract_confidence(text: str, tag: str = "CONFIDENCE") -> int:
    """Extract an integer confidence score 1-5 from the judge's output.

    Args:
        text: The raw LLM output to search.
        tag:  The label to look for (default 'CONFIDENCE', use 'FINAL_CONFIDENCE'
              for deliberation outputs).
    """
    m = re.search(rf"{re.escape(tag)}:\s*([1-5])", text, re.IGNORECASE)
    if m:
        val = int(m.group(1))
        return max(1, min(5, val))
    # Fallback: look for any digit 1-5 near the tag word
    m2 = re.search(rf"{re.escape(tag)}[^0-9]*([1-5])", text, re.IGNORECASE)
    if m2:
        return int(m2.group(1))
    return 3  # Default mid-confidence if not found


# ─── Judge Class ──────────────────────────────────────────────────────────────

class Judge:
    """
    LLM judge that evaluates a completed debate transcript.
    """

    def evaluate(self, question: str, transcript: str) -> JudgeVerdict:
        """
        Evaluate the full debate transcript and produce a structured verdict.

        Args:
            question:   The original question.
            transcript: Full formatted debate transcript (all rounds).

        Returns:
            JudgeVerdict with all structured fields populated.
        """
        prompt = JUDGE_USER.format(question=question, transcript=transcript)
        raw = call_llm(
            messages=[{"role": "user", "content": prompt}],
            system=JUDGE_SYSTEM,
            temperature=JUDGE_TEMPERATURE,
        )

        verdict = JudgeVerdict(
            raw_output=raw,
            cot_analysis=_extract_section(raw, "COT_ANALYSIS"),
            debater_a_strongest=_extract_section(raw, "DEBATER_A_STRONGEST"),
            debater_a_weakest=_extract_section(raw, "DEBATER_A_WEAKEST"),
            debater_b_strongest=_extract_section(raw, "DEBATER_B_STRONGEST"),
            debater_b_weakest=_extract_section(raw, "DEBATER_B_WEAKEST"),
            winner_reasoning=_extract_section(raw, "WINNER_REASONING"),
            verdict=extract_yes_no(raw),
            confidence=_extract_confidence(raw),
        )

        # If extraction failed, try to recover from raw VERDICT: line
        if verdict.verdict == "Unknown":
            verdict.verdict = extract_yes_no(raw)

        return verdict