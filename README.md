# LLM Debate Pipeline — Assignment 2

A multi-agent debate system where two LLM agents argue opposing sides of a StrategyQA yes/no question, evaluated by an LLM judge.

## Project Structure

```
llm_debate/
├── config.py          # All hyperparameters (edit this first)
├── data_loader.py     # StrategyQA dataset loader
├── llm_client.py      # Anthropic API wrapper
├── prompts.py         # All prompt templates (editable)
├── baselines.py       # Zero-shot, One-shot, Few-shot, CoT, Self-Consistency
├── debaters.py        # Debater A (Yes) and Debater B (No) agents
├── judge.py           # Judge agent
├── pipeline.py        # 4-phase debate orchestrator
├── evaluate.py        # Accuracy aggregation and CSV export
├── main.py            # Entry point
├── requirements.txt
├── logs/              # Auto-created: JSON transcripts per run
└── results/           # Auto-created: CSV accuracy tables
```

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY='sk-ant-...'
```

## Usage

```bash
# Demo: run one question end-to-end
python main.py

# Batch: run 100 questions
python main.py --n 100

# Skip baselines (faster)
python main.py --n 10 --no-baseline

# Re-evaluate existing logs (no API calls)
python main.py --eval-only
```

## Pipeline Phases

| Phase | Description |
|-------|-------------|
| 1 — Init | Both debaters independently state their position |
| 2 — Debate | N rounds of argument + rebuttal (min 3, max 5) |
| 3 — Judgment | Judge evaluates full transcript → verdict + confidence |
| 4 — Evaluation | Compare verdict to ground truth; log everything |

## Baselines Compared

| Method | Description |
|--------|-------------|
| `zero_shot` | Direct yes/no answer, no examples |
| `one_shot` | One demonstration example |
| `few_shot` | Four demonstration examples |
| `chain_of_thought` | Step-by-step reasoning before answer |
| `self_consistency` | Majority vote over 5 CoT samples |
| `debate` | This pipeline's judge verdict |

## Configuration (config.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_NAME` | claude-sonnet-4-20250514 | Model for all agents |
| `MAX_DEBATE_ROUNDS` | 5 | Hard cap on debate rounds |
| `MIN_DEBATE_ROUNDS` | 3 | Rounds before early stop allowed |
| `CONVERGENCE_ROUNDS` | 2 | Consecutive agreement rounds for early stop |
| `SELF_CONSISTENCY_N` | 5 | Samples for self-consistency |
| `NUM_QUESTIONS` | 1 | Default demo question count |

## Outputs

- `logs/<qid>_<date>.json` — full debate transcript per question
- `results/summary.csv` — per-question predictions vs. ground truth
- `results/accuracy.csv` — per-method accuracy summary

## References

- Irving, Christiano & Amodei (2018). AI Safety via Debate.
- Liang et al. (EMNLP 2024). Multi-Agent Debate framework.
- Wang et al. (2023). Self-Consistency improves CoT reasoning.
- Geva et al. (2021). StrategyQA dataset.
# llm-debate-chamber
