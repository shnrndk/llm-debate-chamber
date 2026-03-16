"""
api.py — FastAPI REST backend for the LLM Debate Pipeline UI.

Endpoints:
  POST /api/debate          — run full debate on a custom question
  GET  /api/debate/{id}     — retrieve a stored debate result
  GET  /api/debates         — list all past debates
  GET  /api/health          — health check

Run with:
    pip install fastapi uvicorn
    uvicorn api:app --reload --port 8000

The frontend at frontend/index.html connects to http://localhost:8000
"""

from __future__ import annotations
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# ─── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="LLM Debate Pipeline API",
    description="Multi-agent debate system for yes/no question answering",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for active debates (also persisted to logs/)
debate_store: dict[str, dict] = {}

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


# ─── Request / Response Models ────────────────────────────────────────────────

class DebateRequest(BaseModel):
    question: str
    run_baselines: bool = True
    run_jury: bool = True


class DebateStatus(BaseModel):
    debate_id: str
    status: str          # "pending" | "running" | "complete" | "error"
    question: str
    created_at: str
    result: Optional[dict] = None
    error: Optional[str] = None


# ─── Helper: flatten result for frontend ─────────────────────────────────────

def _flatten_result(raw: dict) -> dict:
    """
    Convert the pipeline DebateResult dict into a clean, frontend-friendly shape.
    """
    jv = raw.get("judge_verdict", {})
    jp = raw.get("jury_panel") or {}

    # Build round-by-round timeline for the UI
    rounds_timeline = []

    # Phase 1 — initial positions
    rounds_timeline.append({
        "phase": "initialization",
        "round": 0,
        "label": "Opening Positions",
        "debater_a": {
            "text": raw.get("debater_a_initial", {}).get("raw_output", ""),
            "position": raw.get("debater_a_initial", {}).get("position", ""),
        },
        "debater_b": {
            "text": raw.get("debater_b_initial", {}).get("raw_output", ""),
            "position": raw.get("debater_b_initial", {}).get("position", ""),
        },
    })

    # Phase 2 — debate rounds
    for r in raw.get("debate_rounds", []):
        rounds_timeline.append({
            "phase": "debate",
            "round": r["round"],
            "label": f"Round {r['round']}",
            "debater_a": {
                "text": r.get("debater_a", {}).get("raw_output", ""),
                "position": r.get("debater_a", {}).get("position", ""),
            },
            "debater_b": {
                "text": r.get("debater_b", {}).get("raw_output", ""),
                "position": r.get("debater_b", {}).get("position", ""),
            },
        })

    # Baselines
    baselines_out = {}
    for method, res in raw.get("baselines", {}).items():
        baselines_out[method] = {
            "predicted": res.get("predicted", ""),
            "correct": raw.get("baseline_correctness", {}).get(method),
            "raw_output": res.get("raw_output", "")[:500],
        }

    # Jury jurors
    jurors_out = []
    for iv in jp.get("individual_verdicts", []):
        jurors_out.append({
            "id": iv.get("juror_id", ""),
            "title": iv.get("juror_title", ""),
            "initial_verdict": iv.get("initial_verdict", ""),
            "final_verdict": iv.get("final_verdict", ""),
            "confidence": iv.get("final_confidence", 0),
            "position_changed": iv.get("position_changed", False),
            "reasoning": iv.get("winner_reasoning", ""),
            "deliberation": iv.get("deliberation_analysis", ""),
        })

    return {
        "qid": raw.get("qid", ""),
        "question": raw.get("question", ""),
        "ground_truth": raw.get("ground_truth", "Unknown"),
        "timestamp": raw.get("timestamp", ""),
        # Debate dynamics
        "initial_consensus": raw.get("initial_consensus", False),
        "num_rounds": raw.get("num_rounds", 0),
        "early_stop": raw.get("early_stop", False),
        "rounds_timeline": rounds_timeline,
        # Single judge verdict
        "judge": {
            "verdict": jv.get("verdict", ""),
            "confidence": jv.get("confidence", 0),
            "cot_analysis": jv.get("cot_analysis", ""),
            "debater_a_strongest": jv.get("debater_a_strongest", ""),
            "debater_a_weakest": jv.get("debater_a_weakest", ""),
            "debater_b_strongest": jv.get("debater_b_strongest", ""),
            "debater_b_weakest": jv.get("debater_b_weakest", ""),
            "winner_reasoning": jv.get("winner_reasoning", ""),
            "correct": raw.get("debate_correct"),
        },
        # Jury panel
        "jury": {
            "enabled": bool(jp),
            "majority_verdict": jp.get("majority_verdict", ""),
            "panel_confidence": jp.get("panel_confidence", 0),
            "difficulty": jp.get("difficulty", ""),
            "initial_vote_counts": jp.get("initial_vote_counts", {}),
            "final_vote_counts": jp.get("final_vote_counts", {}),
            "num_changed": jp.get("num_changed", 0),
            "agreement_points": jp.get("agreement_points", ""),
            "disagreement_points": jp.get("disagreement_points", ""),
            "vote_tally": jp.get("vote_tally", ""),
            "correct": raw.get("jury_correct"),
            "jurors": jurors_out,
        },
        # Baselines
        "baselines": baselines_out,
    }


# ─── Background Task: run the debate ─────────────────────────────────────────

def _run_debate_task(debate_id: str, question: str,
                     run_baselines: bool, run_jury: bool):
    """Runs in a background thread so the POST returns immediately."""
    debate_store[debate_id]["status"] = "running"

    try:
        from data_loader import StrategyQAExample
        from pipeline import DebatePipeline

        ex = StrategyQAExample(
            qid=debate_id,
            question=question,
            answer=None,   # no ground truth for user-submitted questions
            facts=[],
        )

        pipeline = DebatePipeline()
        result = pipeline.run(
            ex,
            run_baselines=run_baselines,
            run_jury=run_jury,
            verbose=False,
        )

        from dataclasses import asdict
        result_dict = asdict(result)
        flat = _flatten_result(result_dict)

        debate_store[debate_id]["status"] = "complete"
        debate_store[debate_id]["result"] = flat

    except Exception as e:
        import traceback
        debate_store[debate_id]["status"] = "error"
        debate_store[debate_id]["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.post("/api/debate", response_model=DebateStatus)
def start_debate(req: DebateRequest, background_tasks: BackgroundTasks):
    """
    Submit a question and start the debate pipeline asynchronously.
    Returns a debate_id immediately. Poll GET /api/debate/{id} for results.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    debate_id = str(uuid.uuid4())[:8]
    created_at = datetime.now().isoformat()

    debate_store[debate_id] = {
        "debate_id": debate_id,
        "status": "pending",
        "question": req.question,
        "created_at": created_at,
        "result": None,
        "error": None,
    }

    background_tasks.add_task(
        _run_debate_task, debate_id, req.question,
        req.run_baselines, req.run_jury,
    )

    return DebateStatus(**debate_store[debate_id])


@app.get("/api/debate/{debate_id}", response_model=DebateStatus)
def get_debate(debate_id: str):
    """Poll this endpoint until status == 'complete' or 'error'."""
    if debate_id not in debate_store:
        # Try loading from log file
        matches = list(Path(LOG_DIR).glob(f"{debate_id}_*.json"))
        if matches:
            with open(matches[0]) as f:
                raw = json.load(f)
            flat = _flatten_result(raw)
            return DebateStatus(
                debate_id=debate_id,
                status="complete",
                question=flat["question"],
                created_at=flat["timestamp"],
                result=flat,
            )
        raise HTTPException(status_code=404, detail=f"Debate '{debate_id}' not found.")

    entry = debate_store[debate_id]
    return DebateStatus(**entry)


@app.get("/api/debates")
def list_debates():
    """List all debates (in-memory + from log files)."""
    results = []

    # From memory
    for entry in debate_store.values():
        results.append({
            "debate_id": entry["debate_id"],
            "status": entry["status"],
            "question": entry["question"][:80],
            "created_at": entry["created_at"],
        })

    # From log files (past sessions)
    seen_ids = {e["debate_id"] for e in results}
    for path in sorted(Path(LOG_DIR).glob("*.json"), reverse=True):
        try:
            with open(path) as f:
                raw = json.load(f)
            qid = raw.get("qid", path.stem)
            if qid not in seen_ids:
                results.append({
                    "debate_id": qid,
                    "status": "complete",
                    "question": raw.get("question", "")[:80],
                    "created_at": raw.get("timestamp", ""),
                })
                seen_ids.add(qid)
        except Exception:
            pass

    results.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return {"debates": results, "total": len(results)}


# ─── Streaming SSE endpoint ──────────────────────────────────────────────────────────────

def _sse(event: str, data: dict) -> str:
    """Format a Server-Sent Event message."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _stream_debate(question: str, run_baselines: bool, run_jury: bool):
    """
    Generator that runs the full debate pipeline and yields SSE events.

    Event types:
      phase_start   — a new pipeline phase is beginning
      token         — a single token from the current LLM call
      turn_done     — a debater/judge/juror turn is complete (full text)
      baseline_done — one baseline method finished
      jury_juror    — one juror verdict complete
      complete      — full result payload
      error         — something went wrong
    """
    import traceback
    from data_loader import StrategyQAExample
    from debaters import Debater, format_transcript
    from judge import Judge, _extract_section, _extract_confidence
    from baselines import (
        extract_yes_no, zero_shot, one_shot, few_shot,
        chain_of_thought, self_consistency,
    )
    from jury import JuryPanel, compare_single_vs_panel
    from prompts import (
        DEBATER_A_SYSTEM, DEBATER_B_SYSTEM,
        INITIAL_POSITION_USER, DEBATE_ROUND_USER,
        JUDGE_SYSTEM, JUDGE_USER,
        JUROR_SYSTEM, JUROR_INITIAL_USER,
        DELIBERATION_SYSTEM, DELIBERATION_USER,
        FOREMAN_SYSTEM, FOREMAN_USER,
        JUROR_PERSONAS,
    )
    from llm_client import call_llm_stream, call_llm
    from config import (
        MAX_DEBATE_ROUNDS, MIN_DEBATE_ROUNDS, CONVERGENCE_ROUNDS,
        JUDGE_TEMPERATURE, TEMPERATURE, JURY_SIZE, JURY_DELIBERATION,
    )
    from dataclasses import dataclass
    import re

    try:
        # ── helper: stream one LLM call, return full text ─────────────────
        def stream_turn(system, prompt, event_label, temperature=TEMPERATURE):
            yield _sse("phase_start", {"label": event_label})
            full = ""
            for token in call_llm_stream(
                messages=[{"role": "user", "content": prompt}],
                system=system,
                temperature=temperature,
            ):
                full += token
                yield _sse("token", {"token": token, "label": event_label})
            yield _sse("turn_done", {"label": event_label, "text": full})
            return full

        # ── helper: collect full text from stream_turn generator ──────────
        def run_stream(system, prompt, event_label, temperature=TEMPERATURE):
            full = ""
            for msg in stream_turn(system, prompt, event_label, temperature):
                yield msg
                if msg.startswith("event: turn_done"):
                    # extract text from the data line
                    data_line = msg.split("\n")[1]
                    full = json.loads(data_line[5:]).get("text", "")
            return full

        # We need both yielding and capturing — use a list accumulator
        def collect_stream(system, prompt, event_label, temperature=TEMPERATURE):
            """Yields SSE events AND returns the full accumulated text."""
            full_text = []
            events = []
            yield _sse("phase_start", {"label": event_label})
            for token in call_llm_stream(
                messages=[{"role": "user", "content": prompt}],
                system=system,
                temperature=temperature,
            ):
                full_text.append(token)
                yield _sse("token", {"token": token, "label": event_label})
            text = "".join(full_text)
            yield _sse("turn_done", {"label": event_label, "text": text})
            # stash the result so caller can read it after exhausting the generator
            full_text.clear()
            full_text.append(text)

        # ── BASELINES ─────────────────────────────────────────────────────
        baseline_results = {}
        baseline_correctness = {}

        if run_baselines:
            yield _sse("phase_start", {"label": "Running baselines…"})
            for name, fn in [
                ("zero_shot",        zero_shot),
                ("one_shot",         one_shot),
                ("few_shot",         few_shot),
                ("chain_of_thought", chain_of_thought),
                ("self_consistency", self_consistency),
            ]:
                yield _sse("phase_start", {"label": f"Baseline: {name.replace('_',' ')}…"})
                res = fn(question)
                baseline_results[name] = res
                yield _sse("baseline_done", {
                    "method": name,
                    "predicted": res.predicted,
                    "raw_output": res.raw_output[:400],
                })

        # ── PHASE 1: Initial positions ─────────────────────────────────────
        yield _sse("phase_start", {"label": "Phase 1 — Debater A opening position…"})
        a0_text = []
        prompt_a0 = INITIAL_POSITION_USER.format(question=question)
        for token in call_llm_stream(
            messages=[{"role": "user", "content": prompt_a0}],
            system=DEBATER_A_SYSTEM, temperature=TEMPERATURE,
        ):
            a0_text.append(token)
            yield _sse("token", {"token": token, "label": "debater_a_init"})
        a0_full = "".join(a0_text)
        a0_pos = extract_yes_no(a0_full) or "Yes"
        yield _sse("turn_done", {
            "label": "debater_a_init", "text": a0_full,
            "phase": "init", "debater": "A", "position": a0_pos, "round": 0,
        })

        yield _sse("phase_start", {"label": "Phase 1 — Debater B opening position…"})
        b0_text = []
        prompt_b0 = INITIAL_POSITION_USER.format(question=question)
        for token in call_llm_stream(
            messages=[{"role": "user", "content": prompt_b0}],
            system=DEBATER_B_SYSTEM, temperature=TEMPERATURE,
        ):
            b0_text.append(token)
            yield _sse("token", {"token": token, "label": "debater_b_init"})
        b0_full = "".join(b0_text)
        b0_pos = extract_yes_no(b0_full) or "No"
        yield _sse("turn_done", {
            "label": "debater_b_init", "text": b0_full,
            "phase": "init", "debater": "B", "position": b0_pos, "round": 0,
        })

        # Build turn tracking for transcript
        from debaters import DebaterTurn, format_transcript
        all_turns = [
            DebaterTurn(debater="A", role="Yes", round_num=0, raw_output=a0_full, position=a0_pos),
            DebaterTurn(debater="B", role="No",  round_num=0, raw_output=b0_full, position=b0_pos),
        ]

        initial_consensus = (a0_pos == b0_pos)
        num_rounds = 0
        early_stop = False
        debate_rounds_log = []

        # ── PHASE 2: Multi-round debate ────────────────────────────────────
        if not initial_consensus:
            consecutive_same = 0
            for round_num in range(1, MAX_DEBATE_ROUNDS + 1):
                transcript = format_transcript(all_turns)

                yield _sse("phase_start", {"label": f"Round {round_num} — Debater A…"})
                a_text = []
                prompt_a = DEBATE_ROUND_USER.format(
                    question=question, transcript=transcript, round_num=round_num)
                for token in call_llm_stream(
                    messages=[{"role": "user", "content": prompt_a}],
                    system=DEBATER_A_SYSTEM, temperature=TEMPERATURE,
                ):
                    a_text.append(token)
                    yield _sse("token", {"token": token, "label": f"debater_a_r{round_num}"})
                a_full = "".join(a_text)
                a_pos = extract_yes_no(a_full) or "Yes"
                yield _sse("turn_done", {
                    "label": f"debater_a_r{round_num}", "text": a_full,
                    "phase": "debate", "debater": "A", "position": a_pos, "round": round_num,
                })
                turn_a = DebaterTurn(debater="A", role="Yes", round_num=round_num,
                                     raw_output=a_full, position=a_pos)
                all_turns.append(turn_a)
                transcript = format_transcript(all_turns)

                yield _sse("phase_start", {"label": f"Round {round_num} — Debater B…"})
                b_text = []
                prompt_b = DEBATE_ROUND_USER.format(
                    question=question, transcript=transcript, round_num=round_num)
                for token in call_llm_stream(
                    messages=[{"role": "user", "content": prompt_b}],
                    system=DEBATER_B_SYSTEM, temperature=TEMPERATURE,
                ):
                    b_text.append(token)
                    yield _sse("token", {"token": token, "label": f"debater_b_r{round_num}"})
                b_full = "".join(b_text)
                b_pos = extract_yes_no(b_full) or "No"
                yield _sse("turn_done", {
                    "label": f"debater_b_r{round_num}", "text": b_full,
                    "phase": "debate", "debater": "B", "position": b_pos, "round": round_num,
                })
                turn_b = DebaterTurn(debater="B", role="No", round_num=round_num,
                                     raw_output=b_full, position=b_pos)
                all_turns.append(turn_b)

                debate_rounds_log.append({
                    "round": round_num,
                    "debater_a": {"position": a_pos, "raw_output": a_full},
                    "debater_b": {"position": b_pos, "raw_output": b_full},
                })
                num_rounds = round_num

                if a_pos == b_pos:
                    consecutive_same += 1
                else:
                    consecutive_same = 0

                if round_num >= MIN_DEBATE_ROUNDS and consecutive_same >= CONVERGENCE_ROUNDS:
                    early_stop = True
                    yield _sse("phase_start", {"label": f"Early stop — both agreed for {CONVERGENCE_ROUNDS} rounds"})
                    break

        # ── PHASE 3: Judge ─────────────────────────────────────────────────
        full_transcript = format_transcript(all_turns)
        yield _sse("phase_start", {"label": "Phase 3 — Judge evaluating transcript…"})
        judge_text = []
        judge_prompt = JUDGE_USER.format(question=question, transcript=full_transcript)
        for token in call_llm_stream(
            messages=[{"role": "user", "content": judge_prompt}],
            system=JUDGE_SYSTEM, temperature=JUDGE_TEMPERATURE,
        ):
            judge_text.append(token)
            yield _sse("token", {"token": token, "label": "judge"})
        judge_full = "".join(judge_text)

        def ext(tag): return _extract_section(judge_full, tag)
        judge_verdict = {
            "verdict": extract_yes_no(judge_full),
            "confidence": _extract_confidence(judge_full),
            "cot_analysis": ext("COT_ANALYSIS"),
            "debater_a_strongest": ext("DEBATER_A_STRONGEST"),
            "debater_a_weakest": ext("DEBATER_A_WEAKEST"),
            "debater_b_strongest": ext("DEBATER_B_STRONGEST"),
            "debater_b_weakest": ext("DEBATER_B_WEAKEST"),
            "winner_reasoning": ext("WINNER_REASONING"),
            "raw_output": judge_full,
        }
        yield _sse("turn_done", {"label": "judge", "text": judge_full, **judge_verdict})

        # ── PHASE 3B: Jury ─────────────────────────────────────────────────
        jury_panel_data = None
        if run_jury:
            personas = JUROR_PERSONAS[:min(JURY_SIZE, len(JUROR_PERSONAS))]
            juror_verdicts = []

            for persona in personas:
                jid, title, focus = persona["id"], persona["title"], persona["focus"]
                yield _sse("phase_start", {"label": f"Jury — {jid} ({title}) evaluating…"})

                j_text = []
                j_prompt = JUROR_INITIAL_USER.format(
                    juror_id=jid, juror_title=title, juror_focus=focus,
                    question=question, transcript=full_transcript,
                )
                j_system = JUROR_SYSTEM.format(
                    juror_id=jid, juror_title=title, juror_focus=focus)
                for token in call_llm_stream(
                    messages=[{"role": "user", "content": j_prompt}],
                    system=j_system, temperature=JUDGE_TEMPERATURE,
                ):
                    j_text.append(token)
                    yield _sse("token", {"token": token, "label": f"juror_{jid}"})
                j_full = "".join(j_text)
                j_verdict = extract_yes_no(j_full) or "Yes"
                j_conf = _extract_confidence(j_full)
                jv_obj = {
                    "juror_id": jid, "juror_title": title,
                    "initial_verdict": j_verdict, "initial_confidence": j_conf,
                    "final_verdict": j_verdict, "final_confidence": j_conf,
                    "winner_reasoning": _extract_section(j_full, "WINNER_REASONING"),
                    "position_changed": False, "deliberation_analysis": "",
                    "raw_output": j_full,
                }
                juror_verdicts.append(jv_obj)
                yield _sse("jury_juror", {
                    "juror_id": jid, "juror_title": title,
                    "verdict": j_verdict, "confidence": j_conf,
                    "text": j_full,
                })

            # Deliberation
            if JURY_DELIBERATION:
                for jv in juror_verdicts:
                    jid, title = jv["juror_id"], jv["juror_title"]
                    yield _sse("phase_start", {"label": f"Jury deliberation — {jid}…"})
                    others = "\n\n".join(
                        f"{x['juror_id']} ({x['juror_title']}): {x['initial_verdict']} "
                        f"(conf {x['initial_confidence']}/5)\n{x['winner_reasoning'][:200]}"
                        for x in juror_verdicts if x["juror_id"] != jid
                    )
                    d_prompt = DELIBERATION_USER.format(
                        question=question,
                        my_verdict=jv["initial_verdict"],
                        my_confidence=jv["initial_confidence"],
                        my_reasoning=jv["winner_reasoning"][:400],
                        other_verdicts=others,
                    )
                    d_system = DELIBERATION_SYSTEM.format(juror_id=jid, juror_title=title)
                    d_text = []
                    for token in call_llm_stream(
                        messages=[{"role": "user", "content": d_prompt}],
                        system=d_system, temperature=JUDGE_TEMPERATURE,
                    ):
                        d_text.append(token)
                        yield _sse("token", {"token": token, "label": f"deliberation_{jid}"})
                    d_full = "".join(d_text)
                    m = re.search(r"FINAL_VERDICT:\s*(Yes|No)", d_full, re.IGNORECASE)
                    final_v = m.group(1).capitalize() if m else jv["initial_verdict"]
                    final_c = _extract_confidence(d_full, "FINAL_CONFIDENCE")
                    changed = final_v != jv["initial_verdict"]
                    jv.update({
                        "final_verdict": final_v, "final_confidence": final_c,
                        "position_changed": changed,
                        "deliberation_analysis": _extract_section(d_full, "DELIBERATION_ANALYSIS"),
                    })
                    yield _sse("jury_juror", {
                        "juror_id": jid, "juror_title": title,
                        "verdict": final_v, "confidence": final_c,
                        "position_changed": changed, "phase": "deliberation",
                    })

            # Foreman
            yield _sse("phase_start", {"label": "Jury — Foreman synthesizing verdict…"})
            from collections import Counter
            final_votes = Counter(jv["final_verdict"] for jv in juror_verdicts)
            majority = final_votes.most_common(1)[0][0]
            formatted_verdicts = "\n\n".join(
                f"{jv['juror_id']} ({jv['juror_title']}): {jv['final_verdict']} "
                f"(conf {jv['final_confidence']}/5)\n{jv['deliberation_analysis'] or jv['winner_reasoning']}"
                for jv in juror_verdicts
            )
            f_prompt = FOREMAN_USER.format(
                question=question, final_verdicts=formatted_verdicts)
            f_text = []
            for token in call_llm_stream(
                messages=[{"role": "user", "content": f_prompt}],
                system=FOREMAN_SYSTEM, temperature=JUDGE_TEMPERATURE,
            ):
                f_text.append(token)
                yield _sse("token", {"token": token, "label": "foreman"})
            f_full = "".join(f_text)
            avg_conf = round(
                sum(jv["final_confidence"] for jv in juror_verdicts) / len(juror_verdicts), 1)
            m_diff = re.search(r"DIFFICULTY_ASSESSMENT:\s*(Easy|Medium|Hard)", f_full, re.IGNORECASE)

            jury_panel_data = {
                "majority_verdict": majority,
                "panel_confidence": avg_conf,
                "difficulty": m_diff.group(1).capitalize() if m_diff else "Unknown",
                "initial_vote_counts": dict(Counter(jv["initial_verdict"] for jv in juror_verdicts)),
                "final_vote_counts": dict(final_votes),
                "num_changed": sum(1 for jv in juror_verdicts if jv["position_changed"]),
                "agreement_points": _extract_section(f_full, "AGREEMENT_POINTS"),
                "disagreement_points": _extract_section(f_full, "DISAGREEMENT_POINTS"),
                "vote_tally": _extract_section(f_full, "VOTE_TALLY"),
                "individual_verdicts": juror_verdicts,
            }
            yield _sse("turn_done", {"label": "foreman", "text": f_full, **{
                k: v for k, v in jury_panel_data.items()
                if k not in ("individual_verdicts",)
            }})

        # ── Complete ───────────────────────────────────────────────────────
        from dataclasses import asdict as _asdict
        result_raw = {
            "qid": "stream",
            "question": question,
            "ground_truth": "Unknown",
            "timestamp": datetime.now().isoformat(),
            "debater_a_initial": {"position": a0_pos, "raw_output": a0_full},
            "debater_b_initial": {"position": b0_pos, "raw_output": b0_full},
            "initial_consensus": initial_consensus,
            "debate_rounds": debate_rounds_log,
            "num_rounds": num_rounds,
            "early_stop": early_stop,
            "full_transcript": full_transcript,
            "judge_verdict": judge_verdict,
            "debate_correct": None,
            "baselines": {k: {"predicted": v.predicted, "raw_output": v.raw_output}
                          for k, v in baseline_results.items()},
            "baseline_correctness": {},
            "jury_panel": jury_panel_data,
            "jury_correct": None,
            "jury_vs_single": None,
        }
        flat = _flatten_result(result_raw)
        yield _sse("complete", flat)

    except Exception as e:
        yield _sse("error", {"message": str(e), "traceback": traceback.format_exc()})


@app.get("/api/stream")
async def debate_stream(
    question: str,
    run_baselines: bool = True,
    run_jury: bool = True,
):
    """
    SSE endpoint — streams debate progress token-by-token.
    Connect with:  new EventSource('/api/debate/stream?question=...')
    """
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    def sync_generator():
        return list(_stream_debate(question, run_baselines, run_jury))

    async def async_generator():
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=1)
        # Run the synchronous generator in a thread pool, yielding chunks as ready
        gen = _stream_debate(question, run_baselines, run_jury)
        for chunk in gen:
            yield chunk
            await asyncio.sleep(0)  # yield control to event loop

    return StreamingResponse(
        async_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ─── Serve frontend ───────────────────────────────────────────────────────────

frontend_path = Path(__file__).parent / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")