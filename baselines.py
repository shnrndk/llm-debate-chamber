"""
Baseline Methods for StrategyQA Yes/No QA.

Implements:
  - Zero-shot
  - One-shot
  - Few-shot
  - Chain-of-Thought (CoT)
  - Self-Consistency (majority vote over N CoT samples)

Each function returns a BaselineResult dataclass with the predicted answer,
raw model output, and metadata for logging.
"""

from __future__ import annotations
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from llm_client import call_llm
from prompts import (
    ZERO_SHOT_SYSTEM, ZERO_SHOT_USER,
    ONE_SHOT_SYSTEM, ONE_SHOT_USER,
    FEW_SHOT_SYSTEM, FEW_SHOT_USER,
    COT_SYSTEM, COT_USER,
    SELF_CONSISTENCY_SYSTEM, SELF_CONSISTENCY_USER,
)
from config import (
    MODEL_NAME, TEMPERATURE, JUDGE_TEMPERATURE,
    SELF_CONSISTENCY_N, SELF_CONSISTENCY_TEMP,
)


# ─── Result Model ─────────────────────────────────────────────────────────────

@dataclass
class BaselineResult:
    method: str               # e.g. "zero_shot", "cot", "self_consistency"
    question: str
    predicted: str            # "Yes" or "No"
    raw_output: str           # Full model response
    samples: list[str] = field(default_factory=list)   # For self-consistency
    votes: dict[str, int] = field(default_factory=dict) # Vote tallies


# ─── Answer Extraction ────────────────────────────────────────────────────────

def extract_yes_no(text: str) -> str:
    """
    Extract Yes/No from model output.  Handles various formats:
      - "Yes", "No"  (exact)
      - "ANSWER: Yes"
      - "MY POSITION: No"
      - First word of response
    Returns "Yes", "No", or "Unknown".
    """
    # Look for explicit answer tags first
    for pattern in [
        r"ANSWER:\s*(Yes|No)",
        r"MY POSITION:\s*(Yes|No)",
        r"VERDICT:\s*(Yes|No)",
        r"^(Yes|No)\b",
    ]:
        m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).capitalize()

    # Fallback: look for standalone Yes/No anywhere
    for word in text.split():
        clean = re.sub(r"[^a-zA-Z]", "", word)
        if clean.lower() == "yes":
            return "Yes"
        if clean.lower() == "no":
            return "No"

    return "Unknown"


# ─── Baseline Functions ───────────────────────────────────────────────────────

def zero_shot(question: str) -> BaselineResult:
    """Direct yes/no prediction with no examples."""
    prompt = ZERO_SHOT_USER.format(question=question)
    raw = call_llm(
        messages=[{"role": "user", "content": prompt}],
        system=ZERO_SHOT_SYSTEM,
        temperature=JUDGE_TEMPERATURE,  # Low temp for direct QA
    )
    return BaselineResult(
        method="zero_shot",
        question=question,
        predicted=extract_yes_no(raw),
        raw_output=raw,
    )


def one_shot(question: str) -> BaselineResult:
    """Prediction with one demonstrating example."""
    prompt = ONE_SHOT_USER.format(question=question)
    raw = call_llm(
        messages=[{"role": "user", "content": prompt}],
        system=ONE_SHOT_SYSTEM,
        temperature=JUDGE_TEMPERATURE,
    )
    return BaselineResult(
        method="one_shot",
        question=question,
        predicted=extract_yes_no(raw),
        raw_output=raw,
    )


def few_shot(question: str) -> BaselineResult:
    """Prediction with four demonstrating examples."""
    prompt = FEW_SHOT_USER.format(question=question)
    raw = call_llm(
        messages=[{"role": "user", "content": prompt}],
        system=FEW_SHOT_SYSTEM,
        temperature=JUDGE_TEMPERATURE,
    )
    return BaselineResult(
        method="few_shot",
        question=question,
        predicted=extract_yes_no(raw),
        raw_output=raw,
    )


def chain_of_thought(question: str) -> BaselineResult:
    """CoT prompting: step-by-step reasoning before the final answer."""
    prompt = COT_USER.format(question=question)
    raw = call_llm(
        messages=[{"role": "user", "content": prompt}],
        system=COT_SYSTEM,
        temperature=TEMPERATURE,
    )
    return BaselineResult(
        method="chain_of_thought",
        question=question,
        predicted=extract_yes_no(raw),
        raw_output=raw,
    )


def self_consistency(question: str, n: int = SELF_CONSISTENCY_N) -> BaselineResult:
    """
    Self-Consistency (Wang et al., 2023):
    Sample N independent CoT completions, take the majority vote.
    """
    prompt = SELF_CONSISTENCY_USER.format(question=question)
    samples = []
    answers = []

    for i in range(n):
        raw = call_llm(
            messages=[{"role": "user", "content": prompt}],
            system=SELF_CONSISTENCY_SYSTEM,
            temperature=SELF_CONSISTENCY_TEMP,   # High temp for diversity
        )
        ans = extract_yes_no(raw)
        samples.append(raw)
        answers.append(ans)

    vote_counts = dict(Counter(answers))
    majority = max(vote_counts, key=vote_counts.get)

    return BaselineResult(
        method="self_consistency",
        question=question,
        predicted=majority,
        raw_output="\n\n---\n\n".join(samples),
        samples=samples,
        votes=vote_counts,
    )


# ─── Run all baselines ────────────────────────────────────────────────────────

def run_all_baselines(question: str) -> dict[str, BaselineResult]:
    """
    Convenience function: runs all five baselines for a single question.
    Returns a dict keyed by method name.
    """
    print("\n" + "="*60)
    print("RUNNING BASELINES")
    print("="*60)

    results = {}
    methods = [
        ("zero_shot",        zero_shot),
        ("one_shot",         one_shot),
        ("few_shot",         few_shot),
        ("chain_of_thought", chain_of_thought),
        ("self_consistency", self_consistency),
    ]

    for name, fn in methods:
        print(f"  [{name}] ...", end="", flush=True)
        result = fn(question)
        results[name] = result
        print(f" → {result.predicted}")

    return results
