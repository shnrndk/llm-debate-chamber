"""
Debater Agents — Phase 1 & Phase 2 of the debate pipeline.

Debater A argues the "Yes" side.
Debater B argues the "No" side.

Both share the same underlying structure; the difference is the system prompt
and the position they are instructed to defend.
"""

from __future__ import annotations
from dataclasses import dataclass

from llm_client import call_llm
from prompts import (
    DEBATER_A_SYSTEM, DEBATER_B_SYSTEM,
    INITIAL_POSITION_USER, DEBATE_ROUND_USER,
)
from baselines import extract_yes_no
from config import TEMPERATURE


# ─── Data Models ──────────────────────────────────────────────────────────────

@dataclass
class DebaterTurn:
    debater: str          # "A" or "B"
    role: str             # "Yes" or "No"
    round_num: int        # 0 = initial position; 1+ = debate rounds
    raw_output: str
    position: str         # extracted "Yes" or "No"


# ─── Debater Class ────────────────────────────────────────────────────────────

class Debater:
    """
    A single debater agent.

    Args:
        name:     "A" or "B"
        position: "Yes" (Debater A) or "No" (Debater B)
    """

    def __init__(self, name: str, position: str):
        assert name in ("A", "B"), "Debater name must be 'A' or 'B'"
        assert position in ("Yes", "No"), "Position must be 'Yes' or 'No'"
        self.name = name
        self.position = position
        self.system_prompt = DEBATER_A_SYSTEM if name == "A" else DEBATER_B_SYSTEM

    # ── Phase 1: Initial Position ────────────────────────────────────────────

    def initial_position(self, question: str) -> DebaterTurn:
        """
        Generate the debater's opening position on the question.
        Both debaters call this independently (no cross-visibility).
        """
        prompt = INITIAL_POSITION_USER.format(question=question)
        raw = call_llm(
            messages=[{"role": "user", "content": prompt}],
            system=self.system_prompt,
            temperature=TEMPERATURE,
        )
        extracted = extract_yes_no(raw)
        # Override extraction with assigned role if model drifts
        if extracted == "Unknown":
            extracted = self.position

        return DebaterTurn(
            debater=self.name,
            role=self.position,
            round_num=0,
            raw_output=raw,
            position=extracted,
        )

    # ── Phase 2: Debate Round ────────────────────────────────────────────────

    def debate_round(self, question: str, transcript: str, round_num: int) -> DebaterTurn:
        """
        Generate a round argument given the full debate transcript so far.

        Args:
            question:   The original debate question.
            transcript: Formatted string of all previous turns.
            round_num:  Current round number (1-indexed).
        """
        prompt = DEBATE_ROUND_USER.format(
            question=question,
            transcript=transcript,
            round_num=round_num,
        )
        raw = call_llm(
            messages=[{"role": "user", "content": prompt}],
            system=self.system_prompt,
            temperature=TEMPERATURE,
        )
        extracted = extract_yes_no(raw)
        if extracted == "Unknown":
            extracted = self.position

        return DebaterTurn(
            debater=self.name,
            role=self.position,
            round_num=round_num,
            raw_output=raw,
            position=extracted,
        )


# ─── Transcript Formatting ────────────────────────────────────────────────────

def format_transcript(turns: list[DebaterTurn]) -> str:
    """
    Convert a list of DebaterTurns into a human-readable debate transcript
    that is passed as context to agents and the judge.
    """
    lines = []
    for turn in turns:
        if turn.round_num == 0:
            label = f"[INITIAL POSITION — Debater {turn.debater} ({turn.role})]"
        else:
            label = f"[ROUND {turn.round_num} — Debater {turn.debater} ({turn.role})]"
        lines.append(label)
        lines.append(turn.raw_output.strip())
        lines.append("")   # blank line separator
    return "\n".join(lines)