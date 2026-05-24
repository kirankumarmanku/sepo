"""
Gemma 3 4B Baseline — GTBench Negotiation
==========================================
Negotiation: non-zero-sum, incomplete-information, probabilistic.
Two players negotiate over a bundle of items (books, hats, balls).
Each player has private values for each item type.
Players alternate making offers over multiple rounds.

Based on the Facebook Negotiation dataset format used in GTBench.
Actions: propose a split or accept/reject

Metrics:
  - Avg score per game (primary)
  - Deal rate: fraction of games ending in agreement
  - NRA vs rule-based opponent
  - Fairness index: 1 - |my_score - opp_score| / (my_score + opp_score)
  - Parse failure rate

Usage
-----
python gemma_negotiation.py --backend openai --base-url http://localhost:11434/v1 --model gemma3:4b
python gemma_negotiation.py --reasoning cot --backend openai --base-url http://localhost:11434/v1 --model gemma3:4b
"""

from __future__ import annotations
import argparse, json, re, time, random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np

# ─── Config ──────────────────────────────────────────────────────────────────
MAX_ROUNDS  = 5     # max negotiation rounds before disagreement
ITEMS       = ["books", "hats", "balls"]
QUANTITIES  = [3, 2, 2]   # total of each item available

# ─── Prompts ─────────────────────────────────────────────────────────────────
SYSTEM = """You are negotiating over items with an opponent.

Available items: 3 books, 2 hats, 2 balls

Rules:
- You and your opponent have secret values for each item (you only know your own)
- Players alternate proposing how to split the items (how many you take of each)
- The other player can ACCEPT or make a counter-proposal
- If no deal is reached in 5 rounds, both players get 0 points
- Your score = sum of (items you receive × your values)

Format for proposals: "books=X hats=Y balls=Z" where X+Y+Z covers your share.
To accept: respond with "ACCEPT".

Maximise your total score."""

COT_SYSTEM = """You are negotiating over items with an opponent.

Available items: 3 books, 2 hats, 2 balls

Rules:
- You and your opponent have secret values for each item (you only know your own)
- Players alternate proposing how to split the items (how many you take of each)
- The other player can ACCEPT or make a counter-proposal
- If no deal is reached in 5 rounds, both players get 0 points
- Your score = sum of (items you receive × your values)

Format for proposals: "books=X hats=Y balls=Z" where X+Y+Z covers your share.
To accept: respond with "ACCEPT".

Think step by step about your values and what the opponent might value, then make your move."""

def build_prompt(my_values: dict, round_idx: int, max_rounds: int,
                 history: list[str], last_offer: Optional[dict],
                 i_am_responder: bool) -> str:
    val_str = ", ".join(f"{k}: {v} pts each" for k, v in my_values.items())
    lines = [
        f"Round {round_idx + 1} of {max_rounds}",
        f"Your values: {val_str}",
    ]
    if history:
        lines.append("Negotiation history:")
        for h in history:
            lines.append(f"  {h}")
    if last_offer and i_am_responder:
        offer_str = ", ".join(f"{k}: {v}" for k, v in last_offer.items())
        opp_gets = {k: QUANTITIES[i] - v for i, (k, v) in enumerate(last_offer.items())}
        opp_str  = ", ".join(f"{k}: {v}" for k, v in opp_gets.items())
        my_score = sum(my_values[k] * v for k, v in last_offer.items())
        lines += [
            f"Opponent's offer — you get: {offer_str}",
            f"Opponent would get: {opp_str}",
            f"Your score if you ACCEPT: {my_score}",
            "\nRespond with ACCEPT or a counter-proposal (books=X hats=Y balls=Z).",
        ]
    else:
        lines.append("\nMake a proposal (books=X hats=Y balls=Z) — how many of each item YOU want.")
    return "\n".join(lines)

def parse_proposal(text: str) -> Optional[dict]:
    """Parse 'books=X hats=Y balls=Z' or 'ACCEPT'."""
    if "ACCEPT" in text.upper():
        return "ACCEPT"
    pattern = r'books\s*=\s*(\d+).*?hats\s*=\s*(\d+).*?balls\s*=\s*(\d+)'
    m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if m:
        b, h, ba = int(m.group(1)), int(m.group(2)), int(m.group(3))
        # Clamp to valid range
        b  = min(b,  QUANTITIES[0])
        h  = min(h,  QUANTITIES[1])
        ba = min(ba, QUANTITIES[2])
        return {"books": b, "hats": h, "balls": ba}
    # fallback: try any three numbers
    nums = re.findall(r'\d+', text)
    if len(nums) >= 3:
        b  = min(int(nums[0]), QUANTITIES[0])
        h  = min(int(nums[1]), QUANTITIES[1])
        ba = min(int(nums[2]), QUANTITIES[2])
        return {"books": b, "hats": h, "balls": ba}
    return None

def score(values: dict, allocation: dict) -> float:
    return float(sum(values[k] * allocation[k] for k in ITEMS))

# ─── Opponent strategies ─────────────────────────────────────────────────────

def greedy_opponent(opp_values: dict, round_idx: int, last_offer: Optional[dict],
                    i_am_responder: bool) -> tuple[str, Optional[dict]]:
    """Always demands max for itself, accepts if offer gives ≥ half its max possible."""
    if i_am_responder and last_offer and last_offer != "ACCEPT":
        opp_alloc = {k: QUANTITIES[i] - last_offer[k]
                     for i, k in enumerate(ITEMS)}
        my_score   = score(opp_values, opp_alloc)
        max_score  = score(opp_values, dict(zip(ITEMS, QUANTITIES)))
        if my_score >= max_score * 0.4:
            return "ACCEPT", None
    # Make a greedy proposal (take everything)
    proposal = dict(zip(ITEMS, QUANTITIES))
    return "PROPOSE", proposal

def fair_opponent(opp_values: dict, round_idx: int, last_offer: Optional[dict],
                  i_am_responder: bool) -> tuple[str, Optional[dict]]:
    """Proposes 50/50 split, accepts anything ≥ 40% of max."""
    if i_am_responder and last_offer and last_offer != "ACCEPT":
        opp_alloc = {k: QUANTITIES[i] - last_offer[k]
                     for i, k in enumerate(ITEMS)}
        if score(opp_values, opp_alloc) >= score(opp_values, dict(zip(ITEMS, QUANTITIES))) * 0.4:
            return "ACCEPT", None
    proposal = {k: q // 2 for k, q in zip(ITEMS, QUANTITIES)}
    return "PROPOSE", proposal

OPPONENTS = {"greedy": greedy_opponent, "fair": fair_opponent}

# ─── Game runner ─────────────────────────────────────────────────────────────

@dataclass
class GameResult:
    my_score:    float
    opp_score:   float
    deal_reached:bool
    rounds_taken:int
    parse_fails: int

def run_game(agent, opp_fn, my_values: dict, opp_values: dict, rng) -> GameResult:
    history    = []
    last_offer = None
    parse_fails = 0
    sys = COT_SYSTEM if agent.reasoning == "cot" else SYSTEM

    for t in range(MAX_ROUNDS):
        llm_goes_first = (t == 0)

        if llm_goes_first or (t % 2 == 0):
            # LLM's turn to propose or respond
            i_am_responder = (last_offer is not None)
            prompt = build_prompt(my_values, t, MAX_ROUNDS, history,
                                  last_offer, i_am_responder)
            raw = agent.backend.chat(sys, prompt)
            parsed = parse_proposal(raw)
            if parsed is None:
                parse_fails += 1
                parsed = {k: q // 2 for k, q in zip(ITEMS, QUANTITIES)}
            if parsed == "ACCEPT" and last_offer:
                my_alloc  = last_offer
                opp_alloc = {k: QUANTITIES[i] - my_alloc[k]
                             for i, k in enumerate(ITEMS)}
                history.append(f"LLM: ACCEPT")
                return GameResult(score(my_values, my_alloc),
                                  score(opp_values, opp_alloc),
                                  True, t + 1, parse_fails)
            last_offer = parsed
            history.append(f"LLM proposes: {parsed}")
        else:
            # Opponent's turn
            opp_action, opp_proposal = opp_fn(opp_values, t, last_offer,
                                               last_offer is not None)
            if opp_action == "ACCEPT" and last_offer:
                my_alloc  = last_offer
                opp_alloc = {k: QUANTITIES[i] - my_alloc[k]
                             for i, k in enumerate(ITEMS)}
                history.append(f"Opponent: ACCEPT")
                return GameResult(score(my_values, my_alloc),
                                  score(opp_values, opp_alloc),
                                  True, t + 1, parse_fails)
            if opp_proposal:
                # opp proposes: LLM gets the complement
                llm_alloc = {k: QUANTITIES[i] - opp_proposal[k]
                             for i, k in enumerate(ITEMS)}
                last_offer = llm_alloc
                history.append(f"Opponent proposes (you'd get): {llm_alloc}")

    return GameResult(0.0, 0.0, False, MAX_ROUNDS, parse_fails)

def sample_values(rng) -> dict:
    """Random private values for each item type (0-3)."""
    return {k: int(rng.integers(0, 4)) for k in ITEMS}

def compute_metrics(agent, n_episodes: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    all_results = []
    for opp_name, opp_fn in OPPONENTS.items():
        for ep in range(n_episodes):
            mv  = sample_values(rng)
            ov  = sample_values(rng)
            res = run_game(agent, opp_fn, mv, ov, rng)
            all_results.append({**vars(res), "opponent": opp_name,
                                 "my_values": mv, "opp_values": ov})

    my_scores   = [r["my_score"]    for r in all_results]
    opp_scores  = [r["opp_score"]   for r in all_results]
    deals       = [r["deal_reached"]for r in all_results]
    failures    = [r["parse_fails"] for r in all_results]

    fairness = []
    for r in all_results:
        total = r["my_score"] + r["opp_score"]
        if total > 0:
            fairness.append(1 - abs(r["my_score"] - r["opp_score"]) / total)

    my_total  = sum(my_scores)
    opp_total = sum(opp_scores)
    nra = (my_total - opp_total) / (abs(my_total) + abs(opp_total) + 1e-9)

    return {
        "label":             "Gemma-3-4B Negotiation",
        "avg_score":         round(float(np.mean(my_scores)),  3),
        "deal_rate":         round(float(np.mean(deals)),      3),
        "nra":               round(float(nra),                 3),
        "fairness_index":    round(float(np.mean(fairness)) if fairness else 0, 3),
        "parse_failure_rate":round(float(np.mean(failures)) / MAX_ROUNDS, 3),
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
    p.add_argument("--episodes",    type=int, default=20)
    p.add_argument("--max-tokens",  type=int, default=256)
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

    print(f"\nNegotiation | reasoning={args.reasoning} | episodes={args.episodes}")
    t0 = time.time()
    metrics = compute_metrics(agent, args.episodes, args.seed)
    metrics["elapsed_seconds"] = round(time.time() - t0, 1)
    metrics["model"] = args.model; metrics["reasoning"] = args.reasoning

    print(f"\n{'─'*45}")
    print(f"Avg score/game:    {metrics['avg_score']:.3f}")
    print(f"Deal rate:         {metrics['deal_rate']:.3f}  (higher = more agreements)")
    print(f"NRA:               {metrics['nra']:.3f}  (>0 = better than opponent)")
    print(f"Fairness index:    {metrics['fairness_index']:.3f}  (1.0 = perfectly fair)")
    print(f"Parse failures:    {metrics['parse_failure_rate']:.3f}")
    print(f"Elapsed:           {metrics['elapsed_seconds']}s")
    print(f"{'─'*45}\n")

    suffix = f"_{args.reasoning}"
    (out_dir / f"gemma_negotiation{suffix}.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved → gemma_negotiation{suffix}.json")
