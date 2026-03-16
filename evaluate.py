"""
Evaluation Module — generates four report-ready CSV files from debate logs.

Output files (all saved to results/):
  1. accuracy_summary.csv     — per-method accuracy, the main comparison table
  2. per_question_detail.csv  — every question: all predictions vs. ground truth
  3. jury_analysis.csv        — per-question jury stats (votes, changes, difficulty)
  4. debate_efficiency.csv    — rounds used, early stops, confidence per question

Run standalone:
    python evaluate.py
Or it is called automatically at the end of main.py.
"""

from __future__ import annotations
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config import LOG_DIR, RESULTS_DIR


# ─── Aggregated Stats Model ───────────────────────────────────────────────────

@dataclass
class AggregatedResults:
    num_questions: int
    method_accuracy: dict
    method_correct: dict
    debate_accuracy: float
    avg_confidence: float
    avg_rounds: float
    early_stops: int
    avg_jury_confidence: float = 0.0


# ─── Log Loading ──────────────────────────────────────────────────────────────

def load_logs(log_dir: str = LOG_DIR) -> list:
    logs = []
    for path in sorted(Path(log_dir).glob("*.json")):
        with open(path) as f:
            logs.append(json.load(f))
    return logs


# ─── Aggregation ──────────────────────────────────────────────────────────────

def compute_accuracy(logs: list) -> AggregatedResults:
    if not logs:
        raise ValueError("No log files found. Run the pipeline first.")

    n = len(logs)
    debate_correct   = 0
    total_confidence = 0
    total_rounds     = 0
    early_stops      = 0
    total_jury_conf  = 0
    jury_count       = 0

    baseline_methods = set()
    for log in logs:
        baseline_methods.update(log.get("baseline_correctness", {}).keys())

    method_correct = {m: 0 for m in baseline_methods}
    method_correct["debate_single_judge"] = 0
    method_correct["debate_jury_panel"]   = 0

    for log in logs:
        if log.get("debate_correct", False):
            debate_correct += 1

        jv = log.get("judge_verdict", {})
        total_confidence += jv.get("confidence", 3)
        total_rounds     += log.get("num_rounds", 0)
        if log.get("early_stop", False):
            early_stops += 1

        if log.get("jury_correct") is not None:
            if log["jury_correct"]:
                method_correct["debate_jury_panel"] += 1
            jp = log.get("jury_panel") or {}
            if jp.get("panel_confidence"):
                total_jury_conf += jp["panel_confidence"]
                jury_count += 1

        for method, correct in log.get("baseline_correctness", {}).items():
            if correct:
                method_correct[method] = method_correct.get(method, 0) + 1

    method_correct["debate_single_judge"] = debate_correct
    method_accuracy = {m: c / n for m, c in method_correct.items()}
    avg_jury_conf = (total_jury_conf / jury_count) if jury_count else 0.0

    return AggregatedResults(
        num_questions=n,
        method_accuracy=method_accuracy,
        method_correct=method_correct,
        debate_accuracy=debate_correct / n,
        avg_confidence=total_confidence / n,
        avg_rounds=total_rounds / n,
        early_stops=early_stops,
        avg_jury_confidence=avg_jury_conf,
    )


# ─── CSV 1: Accuracy Summary ──────────────────────────────────────────────────

def _save_accuracy_summary(logs, agg, results_dir):
    path = os.path.join(results_dir, "accuracy_summary.csv")

    METHOD_LABELS = {
        "zero_shot":            "Zero-Shot",
        "one_shot":             "One-Shot",
        "few_shot":             "Few-Shot",
        "chain_of_thought":     "Chain-of-Thought (CoT)",
        "self_consistency":     "Self-Consistency",
        "debate_single_judge":  "Debate + Single Judge",
        "debate_jury_panel":    "Debate + Jury Panel (Bonus)",
    }

    ranked = sorted(agg.method_accuracy.items(), key=lambda x: x[1], reverse=True)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "rank", "method", "method_label",
            "correct", "total", "accuracy_pct",
            "avg_confidence", "notes",
        ])
        writer.writeheader()

        for rank, (method, acc) in enumerate(ranked, 1):
            notes = ""
            conf  = ""
            if method == "debate_single_judge":
                notes = f"Avg confidence: {agg.avg_confidence:.2f}/5 | Avg rounds: {agg.avg_rounds:.1f}"
                conf  = f"{agg.avg_confidence:.2f}"
            elif method == "debate_jury_panel" and agg.avg_jury_confidence:
                notes = f"Avg jury confidence: {agg.avg_jury_confidence:.2f}/5"
                conf  = f"{agg.avg_jury_confidence:.2f}"
            elif method == "self_consistency":
                try:
                    from config import SELF_CONSISTENCY_N
                    notes = f"N={SELF_CONSISTENCY_N} samples, majority vote"
                except Exception:
                    pass

            writer.writerow({
                "rank":           rank,
                "method":         method,
                "method_label":   METHOD_LABELS.get(method, method),
                "correct":        agg.method_correct.get(method, 0),
                "total":          agg.num_questions,
                "accuracy_pct":   f"{acc * 100:.1f}%",
                "avg_confidence": conf,
                "notes":          notes,
            })

    return path


# ─── CSV 2: Per-Question Detail ───────────────────────────────────────────────

def _save_per_question_detail(logs, results_dir):
    path = os.path.join(results_dir, "per_question_detail.csv")

    baseline_methods = sorted(set(
        m for log in logs
        for m in log.get("baseline_correctness", {}).keys()
    ))

    fieldnames = (
        ["qid", "question", "ground_truth",
         "single_judge_verdict", "single_judge_correct", "single_judge_confidence",
         "jury_verdict", "jury_correct", "jury_confidence", "jury_difficulty",
         "num_debate_rounds", "early_stop", "initial_consensus"]
        + [f"{m}_pred" for m in baseline_methods]
        + [f"{m}_correct" for m in baseline_methods]
        + ["all_methods_agree", "methods_correct_count"]
    )

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for log in logs:
            baselines = log.get("baselines", {})
            jv        = log.get("judge_verdict", {})
            jp        = log.get("jury_panel") or {}

            all_preds = [baselines.get(m, {}).get("predicted", "") for m in baseline_methods]
            if jv.get("verdict"):
                all_preds.append(jv["verdict"])
            if jp.get("majority_verdict"):
                all_preds.append(jp["majority_verdict"])
            all_preds = [p for p in all_preds if p]
            all_agree = len(set(all_preds)) == 1 if all_preds else False

            correct_count = sum([
                bool(log.get("debate_correct", False)),
                bool(log.get("jury_correct", False)),
                *[bool(log.get("baseline_correctness", {}).get(m, False)) for m in baseline_methods],
            ])
            total_methods = 2 + len(baseline_methods)

            row = {
                "qid":                     log["qid"],
                "question":                log["question"][:100],
                "ground_truth":            log.get("ground_truth", ""),
                "single_judge_verdict":    jv.get("verdict", ""),
                "single_judge_correct":    log.get("debate_correct", ""),
                "single_judge_confidence": jv.get("confidence", ""),
                "jury_verdict":            jp.get("majority_verdict", "N/A"),
                "jury_correct":            log.get("jury_correct", "N/A"),
                "jury_confidence":         jp.get("panel_confidence", "N/A"),
                "jury_difficulty":         jp.get("difficulty", "N/A"),
                "num_debate_rounds":       log.get("num_rounds", 0),
                "early_stop":              log.get("early_stop", False),
                "initial_consensus":       log.get("initial_consensus", False),
                "all_methods_agree":       all_agree,
                "methods_correct_count":   f"{correct_count}/{total_methods}",
            }
            for m in baseline_methods:
                row[f"{m}_pred"]    = baselines.get(m, {}).get("predicted", "")
                row[f"{m}_correct"] = log.get("baseline_correctness", {}).get(m, "")
            writer.writerow(row)

    return path


# ─── CSV 3: Jury Analysis ─────────────────────────────────────────────────────

def _save_jury_analysis(logs, results_dir):
    path = os.path.join(results_dir, "jury_analysis.csv")

    max_jurors = max(
        (len((log.get("jury_panel") or {}).get("individual_verdicts", []))
         for log in logs),
        default=0
    )
    juror_cols = []
    for i in range(1, max_jurors + 1):
        juror_cols += [
            f"j{i}_id", f"j{i}_title",
            f"j{i}_initial", f"j{i}_final",
            f"j{i}_changed", f"j{i}_confidence",
        ]

    fieldnames = [
        "qid", "question", "ground_truth",
        "single_judge_verdict", "single_judge_correct",
        "jury_verdict", "jury_correct",
        "panel_confidence", "difficulty",
        "initial_yes_votes", "initial_no_votes",
        "final_yes_votes", "final_no_votes",
        "jurors_revised", "panel_unanimous",
        "single_vs_jury_agree",
    ] + juror_cols

    jury_logs = [log for log in logs if log.get("jury_panel")]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for log in jury_logs:
            jp  = log.get("jury_panel") or {}
            jv  = log.get("judge_verdict", {})
            ivs = jp.get("individual_verdicts", [])
            single = jv.get("verdict", "")
            jury   = jp.get("majority_verdict", "")

            row = {
                "qid":                  log["qid"],
                "question":             log["question"][:100],
                "ground_truth":         log.get("ground_truth", ""),
                "single_judge_verdict": single,
                "single_judge_correct": log.get("debate_correct", ""),
                "jury_verdict":         jury,
                "jury_correct":         log.get("jury_correct", ""),
                "panel_confidence":     jp.get("panel_confidence", ""),
                "difficulty":           jp.get("difficulty", ""),
                "initial_yes_votes":    jp.get("initial_vote_counts", {}).get("Yes", 0),
                "initial_no_votes":     jp.get("initial_vote_counts", {}).get("No", 0),
                "final_yes_votes":      jp.get("final_vote_counts", {}).get("Yes", 0),
                "final_no_votes":       jp.get("final_vote_counts", {}).get("No", 0),
                "jurors_revised":       jp.get("num_changed", 0),
                "panel_unanimous":      len(jp.get("final_vote_counts", {})) == 1,
                "single_vs_jury_agree": single == jury,
            }
            for i, iv in enumerate(ivs, 1):
                row[f"j{i}_id"]         = iv.get("juror_id", "")
                row[f"j{i}_title"]      = iv.get("juror_title", "")
                row[f"j{i}_initial"]    = iv.get("initial_verdict", "")
                row[f"j{i}_final"]      = iv.get("final_verdict", "")
                row[f"j{i}_changed"]    = iv.get("position_changed", False)
                row[f"j{i}_confidence"] = iv.get("final_confidence", "")
            writer.writerow(row)

    return path


# ─── CSV 4: Debate Efficiency ─────────────────────────────────────────────────

def _save_debate_efficiency(logs, results_dir):
    path = os.path.join(results_dir, "debate_efficiency.csv")

    try:
        from config import MAX_DEBATE_ROUNDS
    except Exception:
        MAX_DEBATE_ROUNDS = 5

    fieldnames = [
        "qid", "question", "ground_truth",
        "num_rounds", "max_rounds", "rounds_used_pct",
        "early_stop", "initial_consensus",
        "debater_a_initial", "debater_b_initial",
        "single_judge_verdict", "single_judge_correct", "single_judge_confidence",
        "jury_verdict", "jury_correct", "jury_panel_confidence", "jury_difficulty",
        "rounds_to_in_round_consensus",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for log in logs:
            jv     = log.get("judge_verdict", {})
            jp     = log.get("jury_panel") or {}
            rounds = log.get("debate_rounds", [])
            nr     = log.get("num_rounds", 0)

            rounds_to_consensus = ""
            for r in rounds:
                da = r.get("debater_a", {}).get("position", "")
                db = r.get("debater_b", {}).get("position", "")
                if da and db and da == db:
                    rounds_to_consensus = r["round"]
                    break

            pct = f"{(nr / MAX_DEBATE_ROUNDS * 100):.0f}%" if MAX_DEBATE_ROUNDS else ""

            writer.writerow({
                "qid":                          log["qid"],
                "question":                     log["question"][:100],
                "ground_truth":                 log.get("ground_truth", ""),
                "num_rounds":                   nr,
                "max_rounds":                   MAX_DEBATE_ROUNDS,
                "rounds_used_pct":              pct,
                "early_stop":                   log.get("early_stop", False),
                "initial_consensus":            log.get("initial_consensus", False),
                "debater_a_initial":            log.get("debater_a_initial", {}).get("position", ""),
                "debater_b_initial":            log.get("debater_b_initial", {}).get("position", ""),
                "single_judge_verdict":         jv.get("verdict", ""),
                "single_judge_correct":         log.get("debate_correct", ""),
                "single_judge_confidence":      jv.get("confidence", ""),
                "jury_verdict":                 jp.get("majority_verdict", "N/A"),
                "jury_correct":                 log.get("jury_correct", "N/A"),
                "jury_panel_confidence":        jp.get("panel_confidence", "N/A"),
                "jury_difficulty":              jp.get("difficulty", "N/A"),
                "rounds_to_in_round_consensus": rounds_to_consensus,
            })

    return path


# ─── Master Save Function ─────────────────────────────────────────────────────

def save_results(logs: list, results_dir: str = RESULTS_DIR) -> AggregatedResults:
    """
    Generate all four CSV report files from debate logs.

      results/accuracy_summary.csv     — main accuracy table  (report Table 1)
      results/per_question_detail.csv  — all predictions per question
      results/jury_analysis.csv        — jury voting, revisions, difficulty
      results/debate_efficiency.csv    — rounds, early stops, consensus timing
    """
    os.makedirs(results_dir, exist_ok=True)
    agg = compute_accuracy(logs)

    p1 = _save_accuracy_summary(logs, agg, results_dir)
    p2 = _save_per_question_detail(logs, results_dir)
    p3 = _save_jury_analysis(logs, results_dir)
    p4 = _save_debate_efficiency(logs, results_dir)

    print(f"\n{'='*62}")
    print(f"  CSV FILES SAVED TO '{results_dir}/'")
    print(f"{'='*62}")
    print(f"  accuracy_summary.csv     ← Main accuracy table for report")
    print(f"  per_question_detail.csv  ← All predictions per question")
    print(f"  jury_analysis.csv        ← Jury votes, revisions, difficulty")
    print(f"  debate_efficiency.csv    ← Rounds, early stops, consensus")
    print(f"{'='*62}")

    return agg


# ─── Terminal Table ───────────────────────────────────────────────────────────

def print_accuracy_table(agg: AggregatedResults):
    METHOD_LABELS = {
        "zero_shot":            "Zero-Shot",
        "one_shot":             "One-Shot",
        "few_shot":             "Few-Shot",
        "chain_of_thought":     "CoT (Chain-of-Thought)",
        "self_consistency":     "Self-Consistency",
        "debate_single_judge":  "Debate + Single Judge",
        "debate_jury_panel":    "Debate + Jury Panel",
    }
    ORDER = [
        "zero_shot", "one_shot", "few_shot",
        "chain_of_thought", "self_consistency",
        "debate_single_judge", "debate_jury_panel",
    ]

    def sort_key(item):
        try:
            return (ORDER.index(item[0]), "")
        except ValueError:
            return (99, item[0])

    methods = sorted(agg.method_accuracy.items(), key=sort_key)

    W = 64
    print("\n" + "=" * W)
    print("  RESULTS SUMMARY")
    print("=" * W)
    print(f"  {'Method':<30} {'Correct':>9}  {'Accuracy':>9}")
    print("-" * W)
    for method, acc in methods:
        label   = METHOD_LABELS.get(method, method)
        correct = agg.method_correct.get(method, 0)
        marker  = " ◄" if method in ("debate_single_judge", "debate_jury_panel") else ""
        print(f"  {label:<30} {correct:>5}/{agg.num_questions:<3}  {acc:>8.1%}{marker}")
    print("-" * W)
    print(f"  Questions evaluated:         {agg.num_questions}")
    print(f"  Avg single judge confidence: {agg.avg_confidence:.2f} / 5")
    if agg.avg_jury_confidence:
        print(f"  Avg jury panel confidence:   {agg.avg_jury_confidence:.2f} / 5")
    print(f"  Avg debate rounds:           {agg.avg_rounds:.1f}")
    print(f"  Early stops:                 {agg.early_stops} / {agg.num_questions}")
    print("=" * W)


# ─── Standalone entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    logs = load_logs()
    if not logs:
        print(f"No log files found in '{LOG_DIR}/'. Run the pipeline first.")
    else:
        print(f"Found {len(logs)} log file(s). Generating CSVs...")
        agg = save_results(logs)
        print_accuracy_table(agg)