"""
Gemma 3 4B Baseline — GTBench Pig (Dice Game)
==============================================
Pig: non-zero-sum, probabilistic, dynamic dice game.
On your turn, repeatedly roll a die. Accumulate the sum, but if you roll a 1
you lose all points for that turn. At any point you can HOLD to bank your
turn total. First to reach 100 wins.

GTBench conventional agent: "hold at 20" strategy (well-known strong heuristic)
Actions: ROLL or HOLD

Metrics:
  - Win rate (primary)
  - NRA vs hold-at-20
  - Avg turn score banked per turn
  - Avg hold threshold (inferred from play)
  - Parse failure rate

Usage
-----
python gemma_pig.py --backend openai --base-url http://localhost:11434/v1 --model gemma3:4b
python gemma_pig.py --reasoning cot --backend openai --base-url http://localhost:11434/v1 --model gemma3:4b
"""

from __future__ import annotations
import argparse, json, re, time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np

# ─── Config ──────────────────────────────────────────────────────────────────
WIN_SCORE   = 100
MAX_TURNS   = 200   # safety cap to prevent infinite games

# ─── Prompts ─────────────────────────────────────────────────────────────────
SYSTEM = """You are playing the dice game Pig.

Rules:
- On your turn, you repeatedly roll a 6-sided die
- If you roll 2-6: add that number to your TURN TOTAL (not banked yet)
- If you roll 1 (PIG): you lose ALL points accumulated this turn, and your turn ends
- At any time you can HOLD: your turn total is added to your SCORE and your turn ends
- First player to reach 100 points wins

Respond with ONLY: ROLL or HOLD"""

COT_SYSTEM = """You are playing the dice game Pig.

Rules:
- On your turn, you repeatedly roll a 6-sided die
- If you roll 2-6: add that number to your TURN TOTAL (not banked yet)
- If you roll 1 (PIG): you lose ALL points accumulated this turn, and your turn ends
- At any time you can HOLD: your turn total is added to your SCORE and your turn ends
- First player to reach 100 points wins

Think step by step about the risk vs. reward of rolling again given your current
turn total and score, then end with ROLL or HOLD on its own line."""

def build_prompt(my_score: int, opp_score: int, turn_total: int,
                 last_roll: int, roll_history: list[int]) -> str:
    lines = [
        f"Your banked score: {my_score}",
        f"Opponent's score:  {opp_score}",
        f"Your turn total:   {turn_total}  (not yet banked)",
        f"Last roll:         {last_roll}",
    ]
    if roll_history:
        lines.append(f"This turn's rolls:  {roll_history}")
    lines.append(f"Points needed to win: {WIN_SCORE - my_score}")
    lines.append("\nROLL or HOLD?")
    return "\n".join(lines)

def parse_action(text: str) -> Optional[str]:
    text_up = text.upper()
    if "HOLD" in text_up: return "HOLD"
    if "ROLL" in text_up: return "ROLL"
    return None

# ─── Opponents ───────────────────────────────────────────────────────────────

def hold_at_20(my_score, opp_score, turn_total, rng) -> str:
    """Classic strong heuristic: hold when turn_total >= 20, or can win."""
    if my_score + turn_total >= WIN_SCORE:
        return "HOLD"
    return "HOLD" if turn_total >= 20 else "ROLL"

def hold_at_25(my_score, opp_score, turn_total, rng) -> str:
    if my_score + turn_total >= WIN_SCORE: return "HOLD"
    return "HOLD" if turn_total >= 25 else "ROLL"

def random_pig(my_score, opp_score, turn_total, rng) -> str:
    return "HOLD" if rng.random() < 0.3 else "ROLL"

OPPONENTS = {"hold_at_20": hold_at_20, "hold_at_25": hold_at_25, "random": random_pig}

# ─── Game runner ─────────────────────────────────────────────────────────────

@dataclass
class GameResult:
    llm_won: bool
    llm_score: int
    opp_score: int
    turns: int
    llm_turn_scores: list[int]
    parse_fails: int

def play_turn_llm(agent, my_score: int, opp_score: int, rng) -> tuple[int, int, list[int]]:
    """Returns (points_banked, parse_fails, roll_history)."""
    turn_total = 0
    roll_history = []
    parse_fails = 0
    sys = COT_SYSTEM if agent.reasoning == "cot" else SYSTEM

    while True:
        roll = int(rng.integers(1, 7))
        if roll == 1:
            return 0, parse_fails, roll_history   # pig — lose turn
        turn_total += roll
        roll_history.append(roll)

        if my_score + turn_total >= WIN_SCORE:
            return turn_total, parse_fails, roll_history   # auto-hold to win

        prompt = build_prompt(my_score, opp_score, turn_total, roll, roll_history)
        raw = agent.backend.chat(sys, prompt)
        action = parse_action(raw)
        if action is None:
            parse_fails += 1
            action = "HOLD"   # safe fallback

        if action == "HOLD":
            return turn_total, parse_fails, roll_history

def play_turn_opp(opp_fn, my_score: int, opp_score: int, rng) -> int:
    """Returns points banked."""
    turn_total = 0
    while True:
        roll = int(rng.integers(1, 7))
        if roll == 1: return 0
        turn_total += roll
        if opp_score + turn_total >= WIN_SCORE: return turn_total
        action = opp_fn(opp_score, my_score, turn_total, rng)
        if action == "HOLD": return turn_total

def run_game(agent, opp_fn, rng, llm_goes_first: bool) -> GameResult:
    llm_score = opp_score = 0
    turn = 0
    llm_turn_scores = []
    total_parse_fails = 0

    players = ["llm", "opp"] if llm_goes_first else ["opp", "llm"]

    for _ in range(MAX_TURNS):
        player = players[turn % 2]
        if player == "llm":
            gained, pf, _ = play_turn_llm(agent, llm_score, opp_score, rng)
            llm_score += gained
            llm_turn_scores.append(gained)
            total_parse_fails += pf
            if llm_score >= WIN_SCORE:
                return GameResult(True, llm_score, opp_score, turn+1,
                                  llm_turn_scores, total_parse_fails)
        else:
            gained = play_turn_opp(opp_fn, opp_score, llm_score, rng)
            opp_score += gained
            if opp_score >= WIN_SCORE:
                return GameResult(False, llm_score, opp_score, turn+1,
                                  llm_turn_scores, total_parse_fails)
        turn += 1

    return GameResult(llm_score >= opp_score, llm_score, opp_score,
                      turn, llm_turn_scores, total_parse_fails)

def compute_metrics(agent, n_games: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    all_results = []

    for opp_name, opp_fn in OPPONENTS.items():
        for ep in range(n_games):
            for first in [True, False]:   # alternate who goes first
                r = run_game(agent, opp_fn, rng, first)
                all_results.append({**vars(r), "opponent": opp_name, "llm_first": first})

    wins        = [r["llm_won"]   for r in all_results]
    llm_scores  = [r["llm_score"] for r in all_results]
    opp_scores  = [r["opp_score"] for r in all_results]
    failures    = [r["parse_fails"] for r in all_results]
    turn_scores = [s for r in all_results for s in r["llm_turn_scores"] if s > 0]

    llm_total = sum(wins)
    opp_total = len(wins) - llm_total
    nra = (llm_total - opp_total) / (llm_total + opp_total + 1e-9)

    return {
        "label":             "Gemma-3-4B Pig",
        "win_rate":          round(float(np.mean(wins)),       3),
        "avg_llm_score":     round(float(np.mean(llm_scores)), 3),
        "nra":               round(float(nra),                 3),
        "avg_turn_score":    round(float(np.mean(turn_scores)) if turn_scores else 0, 3),
        "parse_failure_rate":round(float(np.mean(failures)),   3),
        "n_games":           len(all_results),
    }

# ─── Backends ────────────────────────────────────────────────────────────────
class TransformersBackend:
    def __init__(self, model, max_tokens, temperature):
        import torch
        from transformers import pipeline
        self.pipe = pipeline("text-generation", model=model, device_map="auto",
                             torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
        self.max_tokens = max_tokens; self.temperature = temperature
    def chat(self, system, user):
        msgs = [{"role":"system","content":system},{"role":"user","content":user}]
        do_s = self.temperature > 0
        out = self.pipe(msgs, max_new_tokens=self.max_tokens, do_sample=do_s,
                        temperature=self.temperature if do_s else None,
                        pad_token_id=self.pipe.tokenizer.eos_token_id)
        return out[0]["generated_text"][-1]["content"]

class OpenAIBackend:
    def __init__(self, base_url, api_key, model, max_tokens, temperature):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model; self.max_tokens = max_tokens; self.temperature = temperature
    def chat(self, system, user):
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            max_tokens=self.max_tokens, temperature=self.temperature)
        return r.choices[0].message.content

class LLMAgent:
    def __init__(self, backend, reasoning="prompt"):
        self.backend = backend; self.reasoning = reasoning

# ─── Main ────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backend",     default="transformers", choices=["transformers","openai"])
    p.add_argument("--model",       default="google/gemma-3-4b-it")
    p.add_argument("--base-url",    default="http://localhost:11434/v1")
    p.add_argument("--api-key",     default="ollama")
    p.add_argument("--games",       type=int, default=20, help="Games per opponent")
    p.add_argument("--max-tokens",  type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--output-dir",  default=".")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--reasoning",   default="prompt", choices=["prompt","cot"])
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "transformers":
        backend = TransformersBackend(args.model, args.max_tokens, args.temperature)
    else:
        backend = OpenAIBackend(args.base_url, args.api_key, args.model,
                                args.max_tokens, args.temperature)
    agent = LLMAgent(backend, reasoning=args.reasoning)

    print(f"\nPig | reasoning={args.reasoning} | games={args.games}")
    t0 = time.time()
    metrics = compute_metrics(agent, args.games, args.seed)
    metrics["elapsed_seconds"] = round(time.time() - t0, 1)
    metrics["model"] = args.model; metrics["reasoning"] = args.reasoning

    print(f"\n{'─'*45}")
    print(f"Win rate:          {metrics['win_rate']:.3f}")
    print(f"Avg score/game:    {metrics['avg_llm_score']:.3f}")
    print(f"NRA:               {metrics['nra']:.3f}  (>0 = better than opponents)")
    print(f"Avg turn score:    {metrics['avg_turn_score']:.3f}  (hold-at-20 ≈ 18)")
    print(f"Parse failures:    {metrics['parse_failure_rate']:.3f}")
    print(f"Elapsed:           {metrics['elapsed_seconds']}s")
    print(f"{'─'*45}\n")

    suffix = f"_{args.reasoning}"
    (out_dir / f"gemma_pig{suffix}.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved → gemma_pig{suffix}.json")
