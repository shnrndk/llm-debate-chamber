"""
main.py — Entry point for the LLM Debate Pipeline.

Usage examples:
    # Load from local SciFact/StrategyQA file (recommended)
    python main.py --data path/to/strategy_qa_test.json --n 100

    # Demo mode with 5 built-in questions (no dataset needed)
    python main.py

    # Skip baselines or jury for faster runs
    python main.py --data strategy_qa_test.json --n 50 --no-baseline
    python main.py --data strategy_qa_test.json --n 50 --no-jury

    # Re-evaluate existing logs without re-running the pipeline
    python main.py --eval-only
"""

from __future__ import annotations
import argparse
import sys

from data_loader import load_from_jsonl, load_strategyqa, _builtin_samples
from pipeline import DebatePipeline
from evaluate import load_logs, save_results, print_accuracy_table
from config import NUM_QUESTIONS


def parse_args():
    parser = argparse.ArgumentParser(description="LLM Debate Pipeline")

    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to local .jsonl dataset file (SciFact or StrategyQA). "
             "If omitted, falls back to HuggingFace or built-in demo samples.",
    )
    parser.add_argument(
        "--n", type=int, default=NUM_QUESTIONS,
        help=f"Number of questions to process (default: {NUM_QUESTIONS})",
    )
    parser.add_argument(
        "--no-baseline", action="store_true",
        help="Skip baseline methods (zero/one/few-shot, CoT, self-consistency)",
    )
    parser.add_argument(
        "--no-jury", action="store_true",
        help="Skip jury panel — run single judge only",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip running pipeline; only re-evaluate existing logs",
    )
    parser.add_argument(
        "--split", default="train",
        help="HuggingFace dataset split: train | validation (ignored if --data is set)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for question sampling",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Evaluate-only mode ───────────────────────────────────────────────────
    if args.eval_only:
        print("[main] Re-evaluating existing logs...")
        logs = load_logs()
        if not logs:
            print("[main] No logs found in logs/. Run the pipeline first.")
            sys.exit(1)
        agg = save_results(logs)
        print_accuracy_table(agg)
        return

    # ── Load dataset ─────────────────────────────────────────────────────────
    if args.data:
        examples = load_from_jsonl(args.data, n=args.n, seed=args.seed)
    else:
        if args.n > 5:
            print(f"\n[main] No --data file provided and you requested --n {args.n}.")
            print(f"       Only 5 built-in demo questions are available without a dataset.")
            print(f"\n  Provide your local file:")
            print(f"    python main.py --data path/to/strategy_qa_test.json --n {args.n}")
            print(f"\n  Or run a quick demo with 5 questions:")
            print(f"    python main.py --n 5\n")
            sys.exit(1)
        examples = _builtin_samples(n=args.n)

    if not examples:
        print("[main] No examples loaded. Exiting.")
        sys.exit(1)

    print(f"\n[main] Running pipeline on {len(examples)} question(s)...\n")

    # ── Run pipeline ─────────────────────────────────────────────────────────
    pipeline = DebatePipeline()
    for i, ex in enumerate(examples, 1):
        print(f"\n{'#'*60}")
        print(f"  Question {i}/{len(examples)}")
        print(f"{'#'*60}")
        try:
            pipeline.run(
                ex,
                run_baselines=not args.no_baseline,
                run_jury=not args.no_jury,
                verbose=True,
            )
        except Exception as e:
            print(f"[main] ERROR on {ex.qid}: {e}")
            import traceback; traceback.print_exc()

    # ── Generate CSVs + print table ──────────────────────────────────────────
    logs = load_logs()
    if logs:
        agg = save_results(logs)
        print_accuracy_table(agg)
    else:
        print("[main] No logs to evaluate.")


if __name__ == "__main__":
    main()