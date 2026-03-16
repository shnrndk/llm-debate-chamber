# LLM Debate Pipeline — Assignment 2

**LLM & Agentic Systems | Graduate Course**  
*Building Adversarial Multi-Agent Reasoning Systems*

A multi-agent debate system where two LLM agents argue opposing sides of a StrategyQA yes/no question, supervised by an LLM judge. Includes a full baseline comparison suite and a multi-agent jury panel (bonus component).

---

## Results (13 questions, StrategyQA, gpt-4o-mini)

| Method | Accuracy |
|--------|----------|
| Zero-Shot | 30.8% |
| One-Shot | 30.8% |
| Few-Shot | 30.8% |
| Chain-of-Thought (CoT) | 38.5% |
| Self-Consistency | 38.5% |
| **Debate + Single Judge** | **61.5%** |
| Debate + Jury Panel (Bonus) | 15.4% |

Debate outperforms the best baseline by **+23 percentage points**.

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
```

Edit `.env` and add your OpenAI key:

```
OPENAI_API_KEY=sk-...
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
# Run 3 questions (quick test)
python main.py --data strategy_qa_test.json --n 3

# Full experiment — 100 questions
python main.py --data strategy_qa_test.json --n 100

# Skip baselines (faster, fewer API calls)
python main.py --data strategy_qa_test.json --n 100 --no-baseline

# Skip jury panel
python main.py --data strategy_qa_test.json --n 100 --no-jury

# Re-generate CSVs from existing logs without any API calls
python main.py --eval-only
```

### Web UI

```bash
uvicorn api:app --reload --port 8000
open http://localhost:8000
```

The UI streams tokens live as each agent thinks, shows round-by-round debate transcripts, judge verdict, jury panel deliberation, and baseline comparison.

---

## Pipeline Phases

| Phase | Description |
|-------|-------------|
| **1 — Initialization** | Both debaters independently generate an opening position without seeing each other's response. If they agree, skip to Phase 3. |
| **2 — Multi-Round Debate** | Min 3, max 5 rounds. Each round: Debater A argues, then Debater B counters with full transcript context. Adaptive early stopping if both converge for 2 consecutive rounds. |
| **3 — Judgment** | Judge reads the full transcript and produces: CoT analysis, strongest/weakest arguments per side, final verdict, confidence score (1–5). |
| **4 — Evaluation** | Compare judge verdict to ground truth. Log everything to JSON. |
| **3B — Jury Panel** *(Bonus)* | 5 jurors with distinct personas evaluate independently, then deliberate (with votes hidden to prevent conformity bias), then a Foreman synthesizes the panel ruling. |

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

Supports two formats out of the box:

**StrategyQA** (JSON array — `.json`):
```json
[{"qid": "...", "question": "...", "answer": true, "facts": [...], "decomposition": [...]}]
```

**SciFact** (JSONL — `.jsonl`):
```
{"id": 7, "claim": "...", "label": "SUPPORTS"}
```

The loader auto-detects format from the first character of the file.

---

## Configuration (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_NAME` | `gpt-4o-mini` | OpenAI model for all agents |
| `TEMPERATURE` | `0.7` | Debater sampling temperature |
| `JUDGE_TEMPERATURE` | `0.2` | Judge temperature (lower = more deterministic) |
| `MAX_DEBATE_ROUNDS` | `5` | Hard cap on debate rounds |
| `MIN_DEBATE_ROUNDS` | `3` | Minimum rounds before early stop |
| `CONVERGENCE_ROUNDS` | `2` | Consecutive agreement rounds to trigger early stop |
| `SELF_CONSISTENCY_N` | `5` | Samples for self-consistency baseline |
| `JURY_SIZE` | `5` | Number of jurors in the panel |
| `JURY_DELIBERATION` | `True` | Whether jurors deliberate after independent evaluation |

---

## Output Files

Every run produces four CSV files in `results/` and one JSON log per question in `logs/`:

| File | Contents |
|------|----------|
| `results/accuracy_summary.csv` | Per-method accuracy, rank, confidence — **main Table 1 for report** |
| `results/per_question_detail.csv` | Every method's prediction per question side-by-side |
| `results/jury_analysis.csv` | Per-juror votes, revisions, panel confidence, difficulty |
| `results/debate_efficiency.csv` | Rounds used, early stops, consensus timing |
| `logs/<qid>_<date>.json` | Full transcript: all turns, judge reasoning, jury deliberation |

---

## Jury Panel — Bonus Component

5 jurors with distinct evaluation personas:

| Juror | Persona | Focus |
|-------|---------|-------|
| J1 | Logical Analyst | Argument validity and deductive reasoning |
| J2 | Evidence Critic | Factual accuracy and evidence quality |
| J3 | Devil's Advocate | Hidden assumptions, steelmanning the losing side |
| J4 | Synthesis Judge | Overall narrative coherence across rounds |
| J5 | Empirical Realist | Real-world plausibility and epistemic humility |

Deliberation uses **blind voting** — jurors see each other's reasoning but not their verdicts, preventing conformity cascades.

---

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/health` | Health check |
| `POST /api/debate` | Submit question (async, returns debate_id) |
| `GET /api/debate/{id}` | Poll for result |
| `GET /api/debates` | List all past debates |
| `GET /api/stream?question=...` | Live SSE token stream |

---

## Cost Estimate (gpt-4o-mini)

With 5 debate rounds, 5 baselines, and 5 jury jurors per question:

| Scale | Estimated Cost |
|-------|----------------|
| 10 questions | ~$0.30–0.50 |
| 100 questions | ~$3–5 |
| 200 questions | ~$6–10 |

Use `--no-jury` and `--no-baseline` to reduce costs during development.

---

## References

- Irving, Christiano & Amodei (2018). *AI Safety via Debate.*
- Liang et al. (EMNLP 2024). *Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate.*
- Kenton et al. (NeurIPS 2024). *Scalable Oversight via Multi-Agent Debate.*
- Wang et al. (2023). *Self-Consistency Improves Chain of Thought Reasoning in Language Models.*
- Geva et al. (2021). *Did Aristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies (StrategyQA).*
- Kalra et al. (2025). *VERDICT: A Multi-Agent Jury Framework for LLM Evaluation.*
- Wei et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.*