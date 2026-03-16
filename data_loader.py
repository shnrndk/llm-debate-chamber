"""
Data loader — supports:

  1. StrategyQA JSON file  (.json  — a JSON array of objects)
  2. SciFact JSONL file    (.jsonl — one JSON object per line)
  3. Built-in demo samples (5 questions, no file needed)

StrategyQA JSON format (strategy_qa_test.json):
  [
    {
      "qid": "b8677742616fef051f00",
      "term": "Genghis Khan",
      "description": "...",
      "question": "Are more people today related to Genghis Khan than Julius Caesar?",
      "answer": true,
      "facts": ["Julius Caesar had three children.", ...],
      "decomposition": ["How many kids did Julius Caesar have?", ...]
    },
    ...
  ]

Ground truth:
  - "answer": true/false  → Yes / No  (present in train AND test splits)
  - facts are used for logging and analysis only

Usage:
    python main.py --data strategy_qa_test.json --n 100
    python main.py --data claims_train.jsonl --n 100
    python main.py --n 5   # built-in demo
"""

from __future__ import annotations
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ─── Data Model ───────────────────────────────────────────────────────────────

@dataclass
class StrategyQAExample:
    qid: str
    question: str
    answer: Optional[bool]      # True=Yes, False=No, None=unknown
    facts: list
    term: str = ""              # StrategyQA: key entity
    description: str = ""      # StrategyQA: entity description
    decomposition: list = field(default_factory=list)

    @property
    def answer_str(self) -> str:
        if self.answer is None:
            return "Unknown"
        return "Yes" if self.answer else "No"


# ─── Main loader ──────────────────────────────────────────────────────────────

def load_from_jsonl(filepath: str, n: Optional[int] = None,
                    seed: int = 42) -> list:
    """
    Load examples from a local file.

    Auto-detects format:
      - .json  with a leading '[' → JSON array  (StrategyQA)
      - .jsonl or one-object-per-line            → JSONL      (SciFact)

    Args:
        filepath: Path to the file.
        n:        Max number of examples. None = all.
        seed:     Random seed for sampling when n < total.
    """
    path = Path(filepath)
    if not path.exists():
        print(f"\n[DataLoader] ERROR: File not found: {filepath}")
        print(f"  Check the path and try again.")
        sys.exit(1)

    # Sniff the first non-whitespace character to decide format
    with open(path, "r") as f:
        first_char = ""
        for ch in f.read(512):
            if ch.strip():
                first_char = ch
                break

    if first_char == "[":
        examples = _load_json_array(path)
    else:
        examples = _load_jsonl(path)

    if not examples:
        print(f"[DataLoader] ERROR: No valid examples parsed from {filepath}")
        print(f"  Make sure the file is valid StrategyQA JSON or SciFact JSONL.")
        sys.exit(1)

    if n is not None and n < len(examples):
        random.seed(seed)
        examples = random.sample(examples, n)

    examples.sort(key=lambda x: x.qid)

    dataset_name = _detect_name(path)
    no_gt = sum(1 for e in examples if e.answer is None)
    print(f"[DataLoader] Loaded {len(examples)} examples from {dataset_name} ({path.name})")
    if no_gt == len(examples):
        print(f"[DataLoader] Warning: No ground-truth labels found — accuracy metrics will be skipped.")
    elif no_gt > 0:
        print(f"[DataLoader] Note: {no_gt}/{len(examples)} examples missing ground truth.")

    return examples


# backward-compat alias
load_from_file = load_from_jsonl


# ─── JSON array loader (StrategyQA .json) ────────────────────────────────────

def _load_json_array(path: Path) -> list:
    """Load a JSON file whose top-level value is an array of question objects."""
    with open(path, "r") as f:
        raw = json.load(f)

    # Handle wrapped formats: {"data": [...]} or {"questions": [...]}
    if isinstance(raw, dict):
        for key in ("data", "examples", "questions", "train", "dev", "test"):
            if key in raw and isinstance(raw[key], list):
                raw = raw[key]
                break

    if not isinstance(raw, list):
        print(f"[DataLoader] ERROR: Expected a JSON array, got {type(raw).__name__}.")
        sys.exit(1)

    examples = []
    for row in raw:
        ex = _parse_strategyqa_row(row)
        if ex:
            examples.append(ex)
    return examples


# ─── JSONL loader (SciFact .jsonl) ───────────────────────────────────────────

def _load_jsonl(path: Path) -> list:
    """Load a file with one JSON object per line."""
    examples = []
    with open(path, "r") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[DataLoader] Warning: skipping line {lineno} — {e}")
                continue
            ex = _parse_scifact_row(row) or _parse_strategyqa_row(row)
            if ex:
                examples.append(ex)
    return examples


# ─── Row parsers ──────────────────────────────────────────────────────────────

def _parse_strategyqa_row(row: dict) -> Optional[StrategyQAExample]:
    """
    Parse one StrategyQA question object.

    Ground truth handling:
      - "answer": true  → Yes  (correct answer)
      - "answer": false → No   (correct answer)
      - key absent      → None (test split without labels)
    """
    if "question" not in row:
        return None

    qid   = str(row.get("qid", row.get("id", "")))
    question = row["question"].strip()

    # Ground truth — present in both train and test splits of StrategyQA
    if "answer" in row and row["answer"] is not None:
        answer = bool(row["answer"])
    else:
        answer = None

    # Facts are the human-annotated evidence sentences
    # Decomposition contains the sub-questions used to reason about the answer
    facts = row.get("facts", [])
    decomposition = row.get("decomposition", [])

    # If facts are missing, fall back to decomposition steps as a proxy
    if not facts and decomposition:
        facts = decomposition

    return StrategyQAExample(
        qid=qid,
        question=question,
        answer=answer,
        facts=facts,
        term=row.get("term", ""),
        description=row.get("description", ""),
        decomposition=decomposition,
    )


def _parse_scifact_row(row: dict) -> Optional[StrategyQAExample]:
    """
    Parse one SciFact claims row.

    Ground truth handling:
      - "label": "SUPPORTS" → True  (Yes)
      - "label": "REFUTES"  → False (No)
      - key absent          → None  (test split)
    """
    if "claim" not in row:
        return None

    qid   = str(row.get("id", row.get("qid", "")))
    claim = row["claim"].strip()

    label = row.get("label", None)
    if label is not None:
        answer = label.upper() in ("SUPPORTS", "SUPPORT", "SUPPORTED", "TRUE")
    else:
        answer = None

    facts = []
    evidence = row.get("evidence", {})
    if isinstance(evidence, dict):
        for ev in evidence.values():
            if isinstance(ev, list):
                for item in ev:
                    if isinstance(item, dict):
                        facts.extend(item.get("sentences", []))
            elif isinstance(ev, dict):
                facts.extend(ev.get("sentences", []))

    return StrategyQAExample(qid=qid, question=claim, answer=answer, facts=facts)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _detect_name(path: Path) -> str:
    name = path.stem.lower()
    if any(x in name for x in ("strategy", "strategyqa", "train", "dev", "test")):
        return "StrategyQA"
    if any(x in name for x in ("claim", "scifact", "corpus")):
        return "SciFact"
    return "dataset"


# ─── HuggingFace fallback ─────────────────────────────────────────────────────

def load_strategyqa(split: str = "train", n: Optional[int] = None,
                    seed: int = 42) -> list:
    """Try HuggingFace; falls back to built-in samples."""
    try:
        from datasets import load_dataset
        ds = load_dataset("wics/strategy-qa", split=split)
        examples = [ex for row in ds if (ex := _parse_strategyqa_row(row))]
        if n is not None:
            random.seed(seed)
            examples = random.sample(examples, min(n, len(examples)))
        print(f"[DataLoader] Loaded {len(examples)} StrategyQA examples from HuggingFace.")
        return examples
    except Exception as e:
        if n is not None and n > 5:
            print(f"\n[DataLoader] ERROR: HuggingFace unavailable: {e}")
            print(f"  Use your local file:  python main.py --data strategy_qa_test.json --n {n}")
            sys.exit(1)
        print(f"[DataLoader] HuggingFace unavailable. Using built-in demo examples.")
        return _builtin_samples(n)


# ─── Built-in demo samples ────────────────────────────────────────────────────

def _builtin_samples(n: Optional[int] = None) -> list:
    samples = [
        StrategyQAExample(
            qid="demo_001",
            question="Did the Roman Empire exist at the same time as the Mayan civilization?",
            answer=True,
            facts=["Roman Empire: ~27 BC–476 AD.", "Maya: ~2000 BC–1500s AD."],
        ),
        StrategyQAExample(
            qid="demo_002",
            question="Could Albert Einstein have used a calculator in high school?",
            answer=False,
            facts=["Einstein attended high school in the 1890s.",
                   "Electronic calculators weren't invented until the 1960s."],
        ),
        StrategyQAExample(
            qid="demo_003",
            question="Is the Great Wall of China visible from the Moon?",
            answer=False,
            facts=["The wall is ~15-30 ft wide.",
                   "Astronauts confirmed it's not visible from the Moon."],
        ),
        StrategyQAExample(
            qid="demo_004",
            question="Would a person survive on Mars surface without a spacesuit?",
            answer=False,
            facts=["Mars atmosphere is 95% CO2.", "Pressure <1% of Earth's."],
        ),
        StrategyQAExample(
            qid="demo_005",
            question="Can you legally drive a car in Japan at age 16?",
            answer=False,
            facts=["Minimum driving age in Japan is 18."],
        ),
    ]
    result = samples[:n] if n is not None else samples
    print(f"[DataLoader] Using {len(result)} built-in demo examples.")
    return result


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        examples = load_from_jsonl(sys.argv[1], n=5)
        for ex in examples:
            print(f"[{ex.qid[:8]}] answer={ex.answer_str:3s}  "
                  f"facts={len(ex.facts)}  q={ex.question[:70]}")
    else:
        print("Usage: python data_loader.py path/to/strategy_qa_test.json")
        print("Running built-in demo...\n")
        for ex in _builtin_samples(n=3):
            print(f"[{ex.qid}] {ex.question[:70]}  →  {ex.answer_str}")