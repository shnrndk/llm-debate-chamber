# LLM Debate Pipeline — Assignment 2

**LLM & Agentic Systems | Graduate Course**  
*Building Adversarial Multi-Agent Reasoning Systems*

A multi-agent debate system where two LLM agents argue opposing sides of a StrategyQA yes/no question, supervised by an LLM judge and a multi-agent jury panel. Includes a full baseline comparison suite, a live-streaming web UI, and a REST API.

📄 **[View Full Report](https://shnrndk.github.io/llm-debate-chamber/)** — Methodology, experiments, analysis, and prompt engineering


 **[Live Preview](https://llm-debate-chamber.onrender.com)** - is hosted in https://llm-debate-chamber.onrender.com

---

## Results (100 questions, StrategyQA, gpt-4o-mini)

| Rank | Method | Correct / 100 | Accuracy |
|------|--------|--------------|----------|
| 1 | Chain-of-Thought (CoT) | 85 | 85.0% |
| 2 | Self-Consistency | 84 | 84.0% |
| 3 | One-Shot | 75 | 75.0% |
| 4 | Zero-Shot | 72 | 72.0% |
| — | **Debate + Jury Panel** ✦ | **71** | **71.0%** |
| 5 | Few-Shot | 69 | 69.0% |
| — | **Debate + Single Judge** ✦ | **68** | **68.0%** |

✦ = debate pipeline components  
Avg judge confidence: 4.07/5 · Avg rounds: 4.4 · Early stops: 38/100

---

## Report

The full written report is in **`https://shnrndk.github.io/llm-debate-chamber/`** — open it directly in a browser or host via GitHub Pages.

Sections: Methodology · Experiments · Analysis · Prompt Engineering · Prompt Appendix

---

## Project Structure

```
llm_debate/
├── .env.example          # API key template — copy to .env and fill in
├── .gitignore            # Excludes .env, logs/, results/, __pycache__/
├── config.py             # All hyperparameters — edit this first
├── llm_client.py         # OpenAI API wrapper (streaming + standard)
├── prompts.py            # All prompt templates for every agent
├── data_loader.py        # StrategyQA JSON + SciFact JSONL loader
├── baselines.py          # Zero-shot, One-shot, Few-shot, CoT, Self-Consistency
├── debaters.py           # Debater A (Yes) and Debater B (No) agents
├── judge.py              # Single judge agent
├── jury.py               # Multi-agent jury panel (bonus)
├── pipeline.py           # 4-phase debate orchestrator
├── evaluate.py           # Accuracy aggregation + 4 CSV report files
├── main.py               # CLI entry point
├── api.py                # FastAPI REST + SSE streaming backend
├── report.html           # GitHub Pages report — dark/light mode toggle, all 4 sections
├── frontend/
│   └── index.html        # Web UI with live token streaming
├── requirements.txt
├── logs/                 # Auto-created: JSON transcript per question
└── results/              # Auto-created: CSV accuracy tables
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API key

```bash
cp .env.example .env
# Edit .env and add:  OPENAI_API_KEY=sk-...
```

### 3. Test the connection

```bash
python llm_client.py
# → [OpenAI] Connection OK — gpt-4o-mini replied: 'OK'
```

---

## Running Experiments

### Command line

```bash
# Quick test — 3 questions
python main.py --data strategy_qa_test.json --n 3

# Full experiment — 100 questions
python main.py --data strategy_qa_test.json --n 100

# Skip baselines (faster, fewer API calls)
python main.py --data strategy_qa_test.json --n 100 --no-baseline

# Skip jury panel
python main.py --data strategy_qa_test.json --n 100 --no-jury

# Re-generate CSVs from existing logs (no API calls)
python main.py --eval-only
```

### Web UI (live token streaming)

```bash
uvicorn api:app --reload --port 8000
open http://localhost:8000
```

The UI streams every token live as each agent thinks, shows round-by-round debate transcripts, judge verdict panel, jury deliberation, and baseline comparison table.


---

## Pipeline Phases

| Phase | Description |
|-------|-------------|
| **1 — Init** | Both debaters independently generate an opening position. If they agree, skip to Phase 3. |
| **2 — Debate** | Min 3, max 5 rounds. Each round: Debater A argues, Debater B counters with full transcript context. Early stop if both converge for 2 consecutive rounds. |
| **3 — Judgment** | Judge produces: CoT analysis, strongest/weakest args per side, verdict, confidence (1–5). |
| **3B — Jury** *(Bonus)* | 5 jurors evaluate independently → blind deliberation (votes hidden) → Foreman ruling. |
| **4 — Evaluation** | Compare verdicts to ground truth. Log everything to JSON. Generate 4 CSVs. |

---

## Baselines

| Method | Description |
|--------|-------------|
| `zero_shot` | Direct yes/no, no examples |
| `one_shot` | One demonstration example |
| `few_shot` | Four demonstration examples |
| `chain_of_thought` | Step-by-step reasoning before answer |
| `self_consistency` | Majority vote over 5 independent CoT samples |

---

## Dataset

Supports two formats — auto-detected from file content:

**StrategyQA** (`.json` — JSON array):
```json
[{"qid": "...", "question": "...", "answer": true, "facts": [...], "decomposition": [...]}]
```


---

## Configuration (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_NAME` | `gpt-4o-mini` | OpenAI model for all agents |
| `TEMPERATURE` | `0.7` | Debater sampling temperature |
| `JUDGE_TEMPERATURE` | `0.2` | Lower = more deterministic verdicts |
| `MAX_DEBATE_ROUNDS` | `5` | Hard cap on rounds |
| `MIN_DEBATE_ROUNDS` | `3` | Minimum before early stop is eligible |
| `CONVERGENCE_ROUNDS` | `2` | Consecutive agreement rounds to trigger early stop |
| `SELF_CONSISTENCY_N` | `5` | Samples for self-consistency majority vote |
| `JURY_SIZE` | `5` | Jurors in the panel |
| `JURY_DELIBERATION` | `True` | Enable blind deliberation phase |

---

## Output Files

| File | Contents |
|------|----------|
| `results/accuracy_summary.csv` | Per-method accuracy, rank, confidence — **Table 1 for report** |
| `results/per_question_detail.csv` | Every method's prediction per question side-by-side |
| `results/jury_analysis.csv` | Per-juror votes, revisions, panel confidence, difficulty |
| `results/debate_efficiency.csv` | Rounds used, early stops, consensus timing |
| `logs/<qid>_<date>.json` | Full transcript: all turns, judge reasoning, jury deliberation |

---

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/health` | Health check |
| `POST /api/debate` | Submit question async — returns `debate_id` |
| `GET /api/debate/{id}` | Poll for result |
| `GET /api/debates` | List all past debates |
| `GET /api/stream?question=...` | Live SSE token stream |

---

## Jury Panel — Bonus Component

5 jurors with distinct evaluation lenses deliberate using **blind voting** (votes hidden during deliberation to prevent conformity cascades):

| Juror | Persona | Focus |
|-------|---------|-------|
| J1 | Logical Analyst | Argument validity, deductive chains |
| J2 | Evidence Critic | Factual accuracy, evidence quality |
| J3 | Devil's Advocate | Hidden assumptions, steelmanning |
| J4 | Synthesis Judge | Narrative coherence across rounds |
| J5 | Empirical Realist | Real-world plausibility |

**Key finding:** Jury (71%) outperforms single judge (68%) on medium-difficulty questions. Blind deliberation was critical — revealed-vote deliberation caused conformity cascades that hurt accuracy.

---

## References

- Irving, Christiano & Amodei (2018). *AI Safety via Debate.*
- Liang et al. (EMNLP 2024). *Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate.*
- Kenton et al. (NeurIPS 2024). *Scalable Oversight via Multi-Agent Debate.*
- Wang et al. (2023). *Self-Consistency Improves Chain of Thought Reasoning in Language Models.*
- Geva et al. (2021). *Did Aristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies (StrategyQA).*
- Kalra et al. (2025). *VERDICT: A Multi-Agent Jury Framework for LLM Evaluation.*
- Wei et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.*