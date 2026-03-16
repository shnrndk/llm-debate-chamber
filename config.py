"""
Configuration for the LLM Debate Pipeline.
All hyperparameters are centralized here — nothing is hardcoded in other modules.
"""

# ─── Local LM Studio Settings ─────────────────────────────────────────────────
# LM Studio exposes an OpenAI-compatible API. Set the base URL and model name
# exactly as shown in LM Studio's "Local Server" tab.
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"   # LM Studio OpenAI-compatible endpoint
LM_STUDIO_API_KEY  = "lm-studio"                      # Any non-empty string works

# ─── Model Settings ───────────────────────────────────────────────────────────
MODEL_NAME        = "openai/gpt-oss-20b"   # Model identifier as shown in LM Studio
TEMPERATURE       = 0.7                    # Sampling temperature for debaters
JUDGE_TEMPERATURE = 0.2                    # Lower temp for deterministic judging
MAX_TOKENS        = 2048                   # Max tokens per LLM call

# ─── Debate Settings ──────────────────────────────────────────────────────────
MAX_DEBATE_ROUNDS = 5         # Maximum rounds before forcing judgment
MIN_DEBATE_ROUNDS = 3         # Minimum rounds before early stopping is allowed
CONVERGENCE_ROUNDS = 2        # Consecutive agreement rounds needed to stop early

# ─── Self-Consistency Settings ────────────────────────────────────────────────
SELF_CONSISTENCY_N = 5        # Number of samples for self-consistency baseline
SELF_CONSISTENCY_TEMP = 0.8   # Higher temp for diverse sampling

# ─── Dataset Settings ─────────────────────────────────────────────────────────
DATASET_NAME = "tau/commonsense_qa"   # HuggingFace dataset handle (fallback)
STRATEGYQA_HF = "wics/strategy-qa"   # Primary StrategyQA handle
NUM_QUESTIONS = 1                     # Questions to run in demo mode (set to 100+ for experiments)

# ─── Jury Panel Settings (Bonus) ─────────────────────────────────────────────
JURY_SIZE = 5             # Number of independent judges in the panel (>=3)
JURY_DELIBERATION = True  # If True, judges see each other's verdicts and may revise

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_DIR = "logs"          # Directory for saving JSON transcripts
RESULTS_DIR = "results"   # Directory for evaluation CSV outputs