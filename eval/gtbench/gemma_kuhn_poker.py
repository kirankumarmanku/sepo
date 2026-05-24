"""
Gemma 3 4B Baseline — GTBench Kuhn Poker
=========================================
Kuhn Poker: incomplete-information, zero-sum, probabilistic card game.
Each player is dealt one card (J, Q, or K). Players bet or pass.
Higher card wins at showdown.

GTBench conventional agent: MCTS
Actions: <BET>, <PASS>, <CALL>, <FOLD>

Metrics computed:
  - Win rate (primary, zero-sum game)
  - NRA: Normalised Relative Advantage (GTBench standard)
  - Bluff rate: how often agent bets with a Jack (weakest card)
  - Fold rate: how often agent folds to a bet
  - Parse failure rate

Usage
-----
# Ollama:
python gemma_kuhn_poker.py --backend openai --base-url http://localhost:11434/v1 --model gemma3:4b

# HuggingFace local:
python gemma_kuhn_poker.py

# With CoT:
python gemma_kuhn_poker.py --reasoning cot --backend openai --base-url http://localhost:11434/v1 --model gemma3:4b
"""

from __future__ import annotations
import argparse, json, random, re, time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np

# ─── Cards & Actions ─────────────────────────────────────────────────────────
JACK, QUEEN, KING = 0, 1, 2
CARD_NAME = {JACK: "Jack", QUEEN: "Queen", KING: "King"}
BET, PASS, CALL, FOLD = "BET", "PASS", "CALL", "FOLD"

# ─── Prompts ─────────────────────────────────────────────────────────────────
SYSTEM = """You are playing Kuhn Poker against an opponent.

Rules:
- Each player is dealt one card: Jack (lowest), Queen (middle), or King (highest)
- Each player antes 1 chip at the start
- Player 1 acts first: BET (add 1 chip) or PASS
- If Player 1 passes, Player 2 can PASS (showdown) or BET (add 1 chip)
  - If Player 2 bets, Player 1 can CALL (add 1 chip, then showdown) or FOLD (lose ante)
- If Player 1 bets, Player 2 can CALL (showdown) or FOLD (lose ante)
- At showdown the higher card wins the pot

Respond with ONLY your action from the legal actions listed. Nothing else."""

COT_SYSTEM = """You are playing Kuhn Poker against an opponent.

Rules:
- Each player is dealt one card: Jack (lowest), Queen (middle), or King (highest)
- Each player antes 1 chip at the start
- Player 1 acts first: BET (add 1 chip) or PASS
- If Player 1 passes, Player 2 can PASS (showdown) or BET (add 1 chip)
  - If Player 2 bets, Player 1 can CALL (add 1 chip, then showdown) or FOLD (lose ante)
- If Player 1 bets, Player 2 can CALL (showdown) or FOLD (lose ante)
- At showdown the higher card wins the pot

Think step by step about your card strength and bluffing opportunities, then end with your action on its own line."""

def build_prompt(card: int, role: int, history: list[str], legal: list[str]) -> str:
    lines = [
        f"Your card: {CARD_NAME[card]}",
        f"You are Player {role + 1}",
        f"Action history: {' -> '.join(history) if history else 'none'}",
        f"Legal actions: {', '.join(legal)}",
        "\nWhat is your action?",
    ]
    return "\n".join(lines)

def parse_action(text: str, legal: list[str]) -> Optional[str]:
    text_up = text.upper()
    for a in legal:
        if a in text_up:
            return a
    return None

# ─── Kuhn Poker Engine ───────────────────────────────────────────────────────

@dataclass
class GameResult:
    llm_profit: float      # net chips won/lost by LLM
    opp_profit: float
    llm_card: int
    opp_card: int
    history: list[str]
    llm_role: int          # 0=P1, 1=P2
    bluffed: bool          # LLM bet/called with Jack
    folded: bool           # LLM folded
    raw_output: str
    parse_fail: bool

def run_game(agent, opponent_strategy, rng: np.random.Generator,
             llm_role: int) -> GameResult:
    """
    llm_role: 0 = LLM is Player 1 (acts first), 1 = LLM is Player 2
    opponent_strategy: function(card, role, history, legal) -> action
    """
    cards = list(rng.permutation([JACK, QUEEN, KING]))
    llm_card = cards[llm_role]
    opp_card = cards[1 - llm_role]

    history = []
    pot = 2   # both antes
    llm_contrib = 1
    opp_contrib = 1
    raw_output = ""
    parse_fail = False
    bluffed = False
    folded = False

    def get_llm_action(legal):
        nonlocal raw_output, parse_fail
        prompt = build_prompt(llm_card, llm_role, history, legal)
        sys = COT_SYSTEM if agent.reasoning == "cot" else SYSTEM
        raw_output = agent.backend.chat(sys, prompt)
        action = parse_action(raw_output, legal)
        if action is None:
            parse_fail = True
            action = legal[0]
        return action

    def get_opp_action(legal):
        return opponent_strategy(opp_card, 1 - llm_role, history, legal)

    # ── Game tree ────────────────────────────────────────────────────────────
    if llm_role == 0:   # LLM is P1, acts first
        action = get_llm_action([BET, PASS])
        history.append(action)
        if action == BET:
            pot += 1; llm_contrib += 1
            if llm_card == JACK: bluffed = True
            # P2 responds
            opp_action = get_opp_action([CALL, FOLD])
            history.append(opp_action)
            if opp_action == FOLD:
                return GameResult(opp_contrib, -opp_contrib, llm_card, opp_card,
                                  history, llm_role, bluffed, False, raw_output, parse_fail)
            else:  # CALL
                pot += 1; opp_contrib += 1
        else:  # PASS
            opp_action = get_opp_action([BET, PASS])
            history.append(opp_action)
            if opp_action == PASS:
                pass  # showdown
            else:  # opp BET
                pot += 1; opp_contrib += 1
                final = get_llm_action([CALL, FOLD])
                history.append(final)
                if final == FOLD:
                    folded = True
                    return GameResult(-llm_contrib, llm_contrib, llm_card, opp_card,
                                     history, llm_role, bluffed, folded, raw_output, parse_fail)
                else:  # CALL
                    pot += 1; llm_contrib += 1
    else:  # LLM is P2, opp acts first
        opp_action = get_opp_action([BET, PASS])
        history.append(opp_action)
        if opp_action == BET:
            pot += 1; opp_contrib += 1
            action = get_llm_action([CALL, FOLD])
            history.append(action)
            if action == FOLD:
                folded = True
                return GameResult(-llm_contrib, llm_contrib, llm_card, opp_card,
                                  history, llm_role, bluffed, folded, raw_output, parse_fail)
            else:
                if llm_card == JACK: bluffed = True
                pot += 1; llm_contrib += 1
        else:  # opp PASS
            action = get_llm_action([BET, PASS])
            history.append(action)
            if action == BET:
                if llm_card == JACK: bluffed = True
                pot += 1; llm_contrib += 1
                opp_final = get_opp_action([CALL, FOLD])
                history.append(opp_final)
                if opp_final == FOLD:
                    return GameResult(opp_contrib, -opp_contrib, llm_card, opp_card,
                                      history, llm_role, bluffed, False, raw_output, parse_fail)
                else:
                    pot += 1; opp_contrib += 1

    # Showdown
    if llm_card > opp_card:
        return GameResult(opp_contrib, -opp_contrib, llm_card, opp_card,
                          history, llm_role, bluffed, False, raw_output, parse_fail)
    else:
        return GameResult(-llm_contrib, llm_contrib, llm_card, opp_card,
                          history, llm_role, bluffed, False, raw_output, parse_fail)

# ─── MCTS-style opponent: Nash equilibrium approximation ─────────────────────
# Kuhn Poker Nash strategy (alpha=1/3):
# P1: Bet with K always, bet with J with prob 1/3, pass with Q
# P2: Call with K/Q always, call with J with prob 1/3
def nash_opponent(card: int, role: int, history: list[str], legal: list[str]) -> str:
    rng = random.Random()
    if role == 0:  # P1
        if card == KING:  return BET if BET in legal else CALL
        if card == QUEEN: return PASS if PASS in legal else FOLD
        if card == JACK:  return BET if (BET in legal and rng.random() < 1/3) else (PASS if PASS in legal else FOLD)
    else:  # P2
        if card == KING:  return CALL if CALL in legal else BET
        if card == QUEEN: return CALL if CALL in legal else PASS
        if card == JACK:  return CALL if (CALL in legal and rng.random() < 1/3) else (FOLD if FOLD in legal else PASS)
    return legal[0]

def random_opponent(card, role, history, legal):
    return random.choice(legal)

OPPONENTS = {"nash": nash_opponent, "random": random_opponent}

# ─── Metrics ─────────────────────────────────────────────────────────────────
def compute_metrics(agent, n_games: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    results = []
    for opp_name, opp_fn in OPPONENTS.items():
        for ep in range(n_games):
            for llm_role in [0, 1]:  # play both positions
                r = run_game(agent, opp_fn, rng, llm_role)
                results.append({**vars(r), "opponent": opp_name})

    profits   = [r["llm_profit"]  for r in results]
    wins      = [1 if r["llm_profit"] > 0 else 0 for r in results]
    bluffs    = [r["bluffed"]     for r in results]
    folds     = [r["folded"]      for r in results]
    failures  = [r["parse_fail"]  for r in results]

    # NRA (GTBench standard for zero-sum)
    llm_total = sum(r["llm_profit"] for r in results)
    opp_total = sum(r["opp_profit"] for r in results)
    nra = (llm_total - opp_total) / (abs(llm_total) + abs(opp_total) + 1e-9)

    return {
        "label":             "Gemma-3-4B Kuhn Poker",
        "win_rate":          round(float(np.mean(wins)),    3),
        "avg_profit":        round(float(np.mean(profits)), 3),
        "nra":               round(float(nra),              3),
        "bluff_rate":        round(float(np.mean(bluffs)),  3),
        "fold_rate":         round(float(np.mean(folds)),   3),
        "parse_failure_rate":round(float(np.mean(failures)),3),
        "n_games":           len(results),
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
    p.add_argument("--games",       type=int, default=20, help="Games per opponent per role")
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

    print(f"\nKuhn Poker | reasoning={args.reasoning} | games={args.games}")
    t0 = time.time()
    metrics = compute_metrics(agent, args.games, args.seed)
    metrics["elapsed_seconds"] = round(time.time() - t0, 1)
    metrics["model"] = args.model; metrics["reasoning"] = args.reasoning

    print(f"\n{'─'*45}")
    print(f"Win rate:          {metrics['win_rate']:.3f}")
    print(f"Avg profit/game:   {metrics['avg_profit']:.3f}")
    print(f"NRA:               {metrics['nra']:.3f}  (>0 = better than opponent)")
    print(f"Bluff rate:        {metrics['bluff_rate']:.3f}  (bets with Jack)")
    print(f"Fold rate:         {metrics['fold_rate']:.3f}")
    print(f"Parse failures:    {metrics['parse_failure_rate']:.3f}")
    print(f"Elapsed:           {metrics['elapsed_seconds']}s")
    print(f"{'─'*45}\n")

    suffix = f"_{args.reasoning}"
    (out_dir / f"gemma_kuhn_poker{suffix}.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved → gemma_kuhn_poker{suffix}.json")
