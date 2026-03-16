"""
Multi-Agent Jury Panel — Bonus Component (Kalra et al., VERDICT 2025).

Architecture:
  Phase A — Independent Evaluation
      Each of N jurors reads the full debate transcript independently and
      produces a structured verdict (CoT analysis + vote + confidence).
      Jurors have distinct personas to encourage diverse reasoning angles.

  Phase B — Deliberation (optional, controlled by JURY_DELIBERATION in config)
      Each juror sees every other juror's initial verdict and may MAINTAIN
      or REVISE their vote with a written justification.

  Phase C — Foreman Synthesis
      A neutral Foreman agent tallies votes, identifies agreement/disagreement
      patterns, and issues the official panel ruling with a difficulty rating.

Key research questions this enables:
  - Does the jury outperform a single judge?
  - Does deliberation improve accuracy vs. independent voting?
  - Does panel disagreement correlate with question difficulty?
"""

from __future__ import annotations
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from llm_client import call_llm
from baselines import extract_yes_no
from config import JURY_SIZE, JURY_DELIBERATION, JUDGE_TEMPERATURE
from prompts import (
    JUROR_PERSONAS,
    JUROR_SYSTEM, JUROR_INITIAL_USER,
    DELIBERATION_SYSTEM, DELIBERATION_USER,
    FOREMAN_SYSTEM, FOREMAN_USER,
)


# ─── Data Models ──────────────────────────────────────────────────────────────

@dataclass
class JurorVerdict:
    juror_id: str              # "J1", "J2", …
    juror_title: str           # "Logical Analyst", …
    raw_output: str
    cot_analysis: str
    debater_a_strongest: str
    debater_a_weakest: str
    debater_b_strongest: str
    debater_b_weakest: str
    winner_reasoning: str
    initial_verdict: str       # "Yes" or "No"
    initial_confidence: int    # 1–5
    # Deliberation fields (populated only if JURY_DELIBERATION=True)
    deliberation_analysis: str = ""
    position_changed: bool = False
    final_verdict: str = ""    # "Yes" or "No"  (same as initial if no deliberation)
    final_confidence: int = 0  # 0 = not yet set


@dataclass
class PanelRuling:
    """Final synthesized ruling produced by the Foreman agent."""
    raw_output: str
    individual_verdicts: list[JurorVerdict]
    vote_tally: str
    agreement_points: str
    disagreement_points: str
    majority_verdict: str       # "Yes" or "No"
    panel_confidence: float     # average of final confidences
    difficulty: str             # "Easy" / "Medium" / "Hard"
    # Computed stats
    initial_vote_counts: dict[str, int] = field(default_factory=dict)
    final_vote_counts: dict[str, int] = field(default_factory=dict)
    num_changed: int = 0        # jurors who revised during deliberation
    panel_correct: Optional[bool] = None   # set after comparing to ground truth


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _extract_section(text: str, tag: str) -> str:
    pattern = rf"{re.escape(tag)}[\s:]*(.*?)(?=\n[A-Z_]{{3,}}:|$)"
    m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""


def _extract_confidence(text: str, tag: str = "CONFIDENCE") -> int:
    m = re.search(rf"{tag}:\s*([1-5])", text, re.IGNORECASE)
    if m:
        return max(1, min(5, int(m.group(1))))
    m2 = re.search(r"confidence[^0-9]*([1-5])", text, re.IGNORECASE)
    return int(m2.group(1)) if m2 else 3


def _extract_panel_confidence(text: str) -> float:
    m = re.search(r"PANEL_CONFIDENCE:\s*([0-9.]+)", text, re.IGNORECASE)
    if m:
        try:
            return round(float(m.group(1)), 1)
        except ValueError:
            pass
    return 3.0


def _extract_difficulty(text: str) -> str:
    m = re.search(r"DIFFICULTY_ASSESSMENT:\s*(Easy|Medium|Hard)", text, re.IGNORECASE)
    return m.group(1).capitalize() if m else "Unknown"


def _format_other_verdicts(verdicts: list[JurorVerdict], exclude_id: str) -> str:
    """
    Format fellow jurors' reasoning for deliberation — VOTE HIDDEN.
    Only the reasoning is shown, not the verdict, to prevent conformity bias.
    Each juror must engage with the arguments on merit, not conform to a count.
    """
    lines = []
    for v in verdicts:
        if v.juror_id == exclude_id:
            continue
        # Show full reasoning but deliberately omit the verdict and confidence
        lines.append(
            f"Juror {v.juror_id} ({v.juror_title}) — {v.juror_title} perspective:\n"
            f"  Analysis: {v.cot_analysis[:200] or v.winner_reasoning[:200]}...\n"
            f"  Key argument: {v.winner_reasoning[:300]}..."
        )
    return "\n\n".join(lines)


def _format_final_verdicts(verdicts: list[JurorVerdict]) -> str:
    """Format all jurors' final verdicts for the Foreman."""
    lines = []
    for v in verdicts:
        verdict = v.final_verdict or v.initial_verdict
        conf = v.final_confidence if v.final_confidence > 0 else v.initial_confidence
        changed = " [REVISED]" if v.position_changed else ""
        lines.append(
            f"Juror {v.juror_id} ({v.juror_title}){changed}:\n"
            f"  Final Vote: {verdict} | Confidence: {conf}/5\n"
            f"  Final Reasoning: {(v.deliberation_analysis or v.winner_reasoning)[:300]}..."
        )
    return "\n\n".join(lines)


# ─── Jury Panel ───────────────────────────────────────────────────────────────

class JuryPanel:
    """
    Multi-agent jury panel with optional deliberation.

    Args:
        size:        Number of jurors (default from config.JURY_SIZE, max 5).
        deliberate:  Whether to run the deliberation phase.
    """

    def __init__(self, size: int = JURY_SIZE, deliberate: bool = JURY_DELIBERATION):
        self.size = min(size, len(JUROR_PERSONAS))  # cap at available personas
        self.deliberate = deliberate
        self.personas = JUROR_PERSONAS[:self.size]

    # ── Phase A: Independent Evaluation ──────────────────────────────────────

    def _evaluate_independently(self, question: str, transcript: str) -> list[JurorVerdict]:
        """Each juror independently evaluates the debate."""
        verdicts: list[JurorVerdict] = []

        for persona in self.personas:
            jid    = persona["id"]
            title  = persona["title"]
            focus  = persona["focus"]

            system = JUROR_SYSTEM.format(
                juror_id=jid, juror_title=title, juror_focus=focus
            )
            prompt = JUROR_INITIAL_USER.format(
                juror_id=jid, juror_title=title, juror_focus=focus,
                question=question, transcript=transcript,
            )

            raw = call_llm(
                messages=[{"role": "user", "content": prompt}],
                system=system,
                temperature=JUDGE_TEMPERATURE,
            )

            initial_verdict = extract_yes_no(raw)
            if initial_verdict == "Unknown":
                initial_verdict = "Yes"  # fallback

            v = JurorVerdict(
                juror_id=jid,
                juror_title=title,
                raw_output=raw,
                cot_analysis=_extract_section(raw, "COT_ANALYSIS"),
                debater_a_strongest=_extract_section(raw, "DEBATER_A_STRONGEST"),
                debater_a_weakest=_extract_section(raw, "DEBATER_A_WEAKEST"),
                debater_b_strongest=_extract_section(raw, "DEBATER_B_STRONGEST"),
                debater_b_weakest=_extract_section(raw, "DEBATER_B_WEAKEST"),
                winner_reasoning=_extract_section(raw, "WINNER_REASONING"),
                initial_verdict=initial_verdict,
                initial_confidence=_extract_confidence(raw, "CONFIDENCE"),
                final_verdict=initial_verdict,          # default = initial
                final_confidence=_extract_confidence(raw, "CONFIDENCE"),
            )
            verdicts.append(v)

        return verdicts

    # ── Phase B: Deliberation ─────────────────────────────────────────────────

    def _deliberate(self, question: str, verdicts: list[JurorVerdict]) -> list[JurorVerdict]:
        """
        Each juror sees the others' initial verdicts and may revise their own.
        Updates verdicts in-place and returns the list.
        """
        for v in verdicts:
            other_text = _format_other_verdicts(verdicts, exclude_id=v.juror_id)

            system = DELIBERATION_SYSTEM.format(
                juror_id=v.juror_id, juror_title=v.juror_title,
            )
            prompt = DELIBERATION_USER.format(
                question=question,
                my_verdict=v.initial_verdict,
                my_confidence=v.initial_confidence,
                my_reasoning=v.winner_reasoning[:400],
                other_verdicts=other_text,
            )

            raw = call_llm(
                messages=[{"role": "user", "content": prompt}],
                system=system,
                temperature=JUDGE_TEMPERATURE,
            )

            # Extract deliberation outputs
            final_v = extract_yes_no(raw)
            if final_v == "Unknown":
                final_v = v.initial_verdict  # keep original if parsing fails

            # Check for FINAL_VERDICT tag specifically
            m = re.search(r"FINAL_VERDICT:\s*(Yes|No)", raw, re.IGNORECASE)
            if m:
                final_v = m.group(1).capitalize()

            final_c = _extract_confidence(raw, "FINAL_CONFIDENCE")
            if final_c == 3 and "FINAL_CONFIDENCE" not in raw.upper():
                final_c = v.initial_confidence  # keep original if tag not found

            changed_match = re.search(r"POSITION_CHANGE:\s*(MAINTAINED|REVISED)", raw, re.IGNORECASE)
            position_changed = bool(changed_match and changed_match.group(1).upper() == "REVISED")

            v.deliberation_analysis = _extract_section(raw, "DELIBERATION_ANALYSIS")
            v.position_changed = position_changed
            v.final_verdict = final_v
            v.final_confidence = final_c

        return verdicts

    # ── Phase C: Foreman Synthesis ────────────────────────────────────────────

    def _synthesize(self, question: str, verdicts: list[JurorVerdict]) -> PanelRuling:
        """The Foreman tallies votes and issues the official panel ruling."""
        final_verdicts_text = _format_final_verdicts(verdicts)

        prompt = FOREMAN_USER.format(
            question=question,
            final_verdicts=final_verdicts_text,
        )

        raw = call_llm(
            messages=[{"role": "user", "content": prompt}],
            system=FOREMAN_SYSTEM,
            temperature=JUDGE_TEMPERATURE,
        )

        # Compute vote counts ourselves (don't trust the LLM's tally)
        initial_votes = Counter(v.initial_verdict for v in verdicts)
        final_votes   = Counter(
            (v.final_verdict if v.final_verdict else v.initial_verdict)
            for v in verdicts
        )
        num_changed = sum(1 for v in verdicts if v.position_changed)

        majority_m = re.search(r"MAJORITY_VERDICT:\s*(Yes|No)", raw, re.IGNORECASE)
        majority = majority_m.group(1).capitalize() if majority_m else max(final_votes, key=final_votes.get)

        # Compute average confidence from jurors directly (more reliable than LLM calc)
        confs = [
            (v.final_confidence if v.final_confidence > 0 else v.initial_confidence)
            for v in verdicts
        ]
        avg_conf = round(sum(confs) / len(confs), 1) if confs else 3.0

        return PanelRuling(
            raw_output=raw,
            individual_verdicts=verdicts,
            vote_tally=_extract_section(raw, "VOTE_TALLY"),
            agreement_points=_extract_section(raw, "AGREEMENT_POINTS"),
            disagreement_points=_extract_section(raw, "DISAGREEMENT_POINTS"),
            majority_verdict=majority,
            panel_confidence=avg_conf,
            difficulty=_extract_difficulty(raw),
            initial_vote_counts=dict(initial_votes),
            final_vote_counts=dict(final_votes),
            num_changed=num_changed,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(self, question: str, transcript: str,
                 verbose: bool = True) -> PanelRuling:
        """
        Run the full jury evaluation pipeline.

        Args:
            question:   The original debate question.
            transcript: Full formatted debate transcript.
            verbose:    Print progress to stdout.

        Returns:
            PanelRuling with all individual verdicts + panel synthesis.
        """
        if verbose:
            print(f"\n  [Jury] Phase A — {self.size} jurors evaluating independently...")

        verdicts = self._evaluate_independently(question, transcript)

        if verbose:
            for v in verdicts:
                print(f"    {v.juror_id} ({v.juror_title}): {v.initial_verdict} ({v.initial_confidence}/5)")

        if self.deliberate:
            if verbose:
                print(f"\n  [Jury] Phase B — Deliberation...")
            verdicts = self._deliberate(question, verdicts)

            if verbose:
                for v in verdicts:
                    tag = " [REVISED]" if v.position_changed else ""
                    print(f"    {v.juror_id} final: {v.final_verdict} ({v.final_confidence}/5){tag}")

        if verbose:
            print(f"\n  [Jury] Phase C — Foreman synthesis...")

        ruling = self._synthesize(question, verdicts)

        if verbose:
            yes_v = ruling.final_vote_counts.get("Yes", 0)
            no_v  = ruling.final_vote_counts.get("No", 0)
            print(f"\n  [Jury] Panel ruling: {ruling.majority_verdict}  "
                  f"(Yes: {yes_v} | No: {no_v})")
            print(f"  [Jury] Avg confidence: {ruling.panel_confidence}/5 | "
                  f"Difficulty: {ruling.difficulty}")
            if self.deliberate:
                print(f"  [Jury] Jurors who revised: {ruling.num_changed}/{self.size}")

        return ruling


# ─── Comparison Utility ───────────────────────────────────────────────────────

def compare_single_vs_panel(single_verdict: str, panel_ruling: PanelRuling,
                             ground_truth: str) -> dict:
    """
    Compare single judge vs. jury panel performance for one question.
    Returns a dict suitable for logging / CSV export.
    """
    gt = ground_truth.strip().lower()
    return {
        "single_judge_verdict":   single_verdict,
        "single_judge_correct":   single_verdict.lower() == gt,
        "panel_majority_verdict": panel_ruling.majority_verdict,
        "panel_correct":          panel_ruling.majority_verdict.lower() == gt,
        "panel_confidence":       panel_ruling.panel_confidence,
        "difficulty":             panel_ruling.difficulty,
        "initial_yes_votes":      panel_ruling.initial_vote_counts.get("Yes", 0),
        "initial_no_votes":       panel_ruling.initial_vote_counts.get("No", 0),
        "final_yes_votes":        panel_ruling.final_vote_counts.get("Yes", 0),
        "final_no_votes":         panel_ruling.final_vote_counts.get("No", 0),
        "jurors_revised":         panel_ruling.num_changed,
        "panel_unanimous":        len(panel_ruling.final_vote_counts) == 1,
    }