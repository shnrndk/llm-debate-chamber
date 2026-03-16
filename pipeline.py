"""
Debate Orchestrator — coordinates Phases 1–4 for a single question.

Phase 1: Independent initialization (both debaters state their position)
Phase 2: Multi-round debate with adaptive stopping
Phase 3: Judge evaluates the full transcript
Phase 4: Compare judge verdict against ground truth; log everything
"""

from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Optional

from debaters import Debater, DebaterTurn, format_transcript
from judge import Judge, JudgeVerdict
from jury import JuryPanel, PanelRuling, compare_single_vs_panel
from baselines import BaselineResult, run_all_baselines
from data_loader import StrategyQAExample
from config import (
    MAX_DEBATE_ROUNDS, MIN_DEBATE_ROUNDS, CONVERGENCE_ROUNDS,
    LOG_DIR, MODEL_NAME, TEMPERATURE, JUDGE_TEMPERATURE,
    JURY_SIZE, JURY_DELIBERATION,
)


# ─── Result Model ─────────────────────────────────────────────────────────────

@dataclass
class DebateResult:
    # Metadata
    qid: str
    question: str
    ground_truth: str          # "Yes" or "No"
    timestamp: str

    # Phase 1
    debater_a_initial: dict
    debater_b_initial: dict
    initial_consensus: bool    # Did both agree in Phase 1?

    # Phase 2
    debate_rounds: list[dict]  # List of (debater_a_turn, debater_b_turn) per round
    num_rounds: int
    early_stop: bool

    # Phase 3
    full_transcript: str
    judge_verdict: dict

    # Phase 4
    debate_correct: bool
    baselines: dict[str, dict]
    baseline_correctness: dict[str, bool]

    # Phase 3 — Jury Panel (bonus)
    jury_panel: Optional[dict] = None       # None if jury not enabled
    jury_correct: Optional[bool] = None
    jury_vs_single: Optional[dict] = None   # comparison stats

    # Extras
    model: str = MODEL_NAME


# ─── Correctness Check ────────────────────────────────────────────────────────

def _is_correct(predicted: str, ground_truth: str) -> bool:
    return predicted.strip().lower() == ground_truth.strip().lower()


# ─── Orchestrator ─────────────────────────────────────────────────────────────

class DebatePipeline:
    """
    Full LLM Debate + Judge pipeline.

    Usage:
        pipeline = DebatePipeline()
        result   = pipeline.run(example)
    """

    def __init__(self):
        self.debater_a = Debater(name="A", position="Yes")
        self.debater_b = Debater(name="B", position="No")
        self.judge     = Judge()
        self.jury      = JuryPanel()   # uses JURY_SIZE + JURY_DELIBERATION from config
        os.makedirs(LOG_DIR, exist_ok=True)

    # ── Main Entry Point ─────────────────────────────────────────────────────

    def run(self, example: StrategyQAExample,
            run_baselines: bool = True,
            run_jury: bool = True,
            verbose: bool = True) -> DebateResult:
        """
        Run the complete pipeline for one StrategyQA example.

        Args:
            example:       The question + ground truth.
            run_baselines: Whether to also run all baselines.
            verbose:       Print progress to stdout.

        Returns:
            DebateResult with all phases recorded.
        """
        q    = example.question
        gt   = example.answer_str
        qid  = example.qid
        ts   = datetime.now().isoformat()

        if verbose:
            self._header(f"QUESTION [{qid}]: {q}")
            print(f"  Ground truth: {gt}")

        # ── BASELINES ────────────────────────────────────────────────────────
        baseline_results: dict[str, BaselineResult] = {}
        baseline_correctness: dict[str, bool] = {}

        if run_baselines:
            baseline_results = run_all_baselines(q)
            for name, res in baseline_results.items():
                baseline_correctness[name] = _is_correct(res.predicted, gt)

        # ── PHASE 1: Initialization ──────────────────────────────────────────
        if verbose:
            self._header("PHASE 1 — Initialization")

        turn_a0 = self.debater_a.initial_position(q)
        turn_b0 = self.debater_b.initial_position(q)

        if verbose:
            print(f"  Debater A initial: {turn_a0.position}")
            print(f"  Debater B initial: {turn_b0.position}")

        all_turns: list[DebaterTurn] = [turn_a0, turn_b0]

        initial_consensus = (turn_a0.position == turn_b0.position)
        if initial_consensus and verbose:
            print(f"  >> Both debaters agree: {turn_a0.position}. Skipping to Phase 3.")

        # ── PHASE 2: Multi-Round Debate ───────────────────────────────────────
        debate_rounds_log: list[dict] = []
        num_rounds = 0
        early_stop = False

        if not initial_consensus:
            if verbose:
                self._header("PHASE 2 — Multi-Round Debate")

            consecutive_same = 0

            for round_num in range(1, MAX_DEBATE_ROUNDS + 1):
                transcript_so_far = format_transcript(all_turns)

                if verbose:
                    print(f"\n  --- Round {round_num} ---")

                # Debater A argues
                turn_a = self.debater_a.debate_round(q, transcript_so_far, round_num)
                all_turns.append(turn_a)
                transcript_so_far = format_transcript(all_turns)

                # Debater B responds
                turn_b = self.debater_b.debate_round(q, transcript_so_far, round_num)
                all_turns.append(turn_b)

                if verbose:
                    print(f"    Debater A: {turn_a.position}")
                    print(f"    Debater B: {turn_b.position}")

                debate_rounds_log.append({
                    "round": round_num,
                    "debater_a": asdict(turn_a),
                    "debater_b": asdict(turn_b),
                })
                num_rounds = round_num

                # Adaptive stopping: both agree for CONVERGENCE_ROUNDS consecutive rounds
                if turn_a.position == turn_b.position:
                    consecutive_same += 1
                else:
                    consecutive_same = 0

                if round_num >= MIN_DEBATE_ROUNDS and consecutive_same >= CONVERGENCE_ROUNDS:
                    early_stop = True
                    if verbose:
                        print(f"  >> Early stop: both agreed for {CONVERGENCE_ROUNDS} consecutive rounds.")
                    break

        # ── PHASE 3: Judgment ─────────────────────────────────────────────────
        if verbose:
            self._header("PHASE 3 — Judgment")

        full_transcript = format_transcript(all_turns)
        verdict = self.judge.evaluate(q, full_transcript)

        if verbose:
            print(f"  Judge verdict:    {verdict.verdict}")
            print(f"  Confidence:       {verdict.confidence}/5")
            print(f"  Winner reasoning: {verdict.winner_reasoning[:200]}...")

        # ── PHASE 4: Evaluation ───────────────────────────────────────────────
        debate_correct = _is_correct(verdict.verdict, gt)

        if verbose:
            self._header("PHASE 4 — Evaluation")
            status = "✓ CORRECT" if debate_correct else "✗ INCORRECT"
            print(f"  Debate pipeline: {verdict.verdict}  [{status}]")
            if run_baselines:
                for name, res in baseline_results.items():
                    ok = baseline_correctness[name]
                    sym = "✓" if ok else "✗"
                    print(f"  {name:<20}: {res.predicted}  [{sym}]")

        # ── JURY PANEL (Bonus) ───────────────────────────────────────────────────
        jury_panel_dict = None
        jury_correct    = None
        jury_vs_single  = None

        if run_jury:
            if verbose:
                self._header("PHASE 3B — Multi-Agent Jury Panel (Bonus)")
            panel = self.jury.evaluate(q, full_transcript, verbose=verbose)
            panel.panel_correct = _is_correct(panel.majority_verdict, gt)
            jury_correct = panel.panel_correct
            jury_vs_single = compare_single_vs_panel(
                single_verdict=verdict.verdict,
                panel_ruling=panel,
                ground_truth=gt,
            )
            from dataclasses import asdict as _asdict
            jury_panel_dict = {
                "majority_verdict":    panel.majority_verdict,
                "panel_confidence":    panel.panel_confidence,
                "difficulty":          panel.difficulty,
                "num_changed":         panel.num_changed,
                "initial_vote_counts": panel.initial_vote_counts,
                "final_vote_counts":   panel.final_vote_counts,
                "agreement_points":    panel.agreement_points,
                "disagreement_points": panel.disagreement_points,
                "vote_tally":          panel.vote_tally,
                "individual_verdicts": [
                    {
                        "juror_id":              v.juror_id,
                        "juror_title":           v.juror_title,
                        "initial_verdict":       v.initial_verdict,
                        "initial_confidence":    v.initial_confidence,
                        "final_verdict":         v.final_verdict,
                        "final_confidence":      v.final_confidence,
                        "position_changed":      v.position_changed,
                        "winner_reasoning":      v.winner_reasoning,
                        "deliberation_analysis": v.deliberation_analysis,
                    }
                    for v in panel.individual_verdicts
                ],
            }
            if verbose:
                status = "CORRECT" if jury_correct else "INCORRECT"
                print(f"  Jury verdict: {panel.majority_verdict}  [{status}]")
                print(f"  Single judge: {verdict.verdict}  "
                      f"[{'CORRECT' if debate_correct else 'INCORRECT'}]")

        # ── Build result ──────────────────────────────────────────────────────
        result = DebateResult(
            qid=qid,
            question=q,
            ground_truth=gt,
            timestamp=ts,
            debater_a_initial=asdict(turn_a0),
            debater_b_initial=asdict(turn_b0),
            initial_consensus=initial_consensus,
            debate_rounds=debate_rounds_log,
            num_rounds=num_rounds,
            early_stop=early_stop,
            full_transcript=full_transcript,
            judge_verdict=asdict(verdict),
            debate_correct=debate_correct,
            baselines={k: asdict(v) for k, v in baseline_results.items()},
            baseline_correctness=baseline_correctness,
            jury_panel=jury_panel_dict,
            jury_correct=jury_correct,
            jury_vs_single=jury_vs_single,
        )

        # ── Logging ────────────────────────────────────────────────────────────
        self._save_log(result)

        return result

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _header(self, title: str):
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")

    def _save_log(self, result: DebateResult):
        """Save the full debate log as JSON."""
        fname = f"{LOG_DIR}/{result.qid}_{result.timestamp[:10]}.json"
        with open(fname, "w") as f:
            json.dump(asdict(result), f, indent=2)
        print(f"\n[Log saved → {fname}]")