"""
Gemma 3 4B Baseline — GTBench Blind Auction
============================================
Blind Auction (First-Price Sealed-Bid Auction): static, incomplete-information,
probabilistic. Each player receives a private value and submits a sealed bid.
Highest bid wins the item, pays their bid. If you win, profit = value - bid.

GTBench conventional agent: Nash equilibrium bidder
  Nash strategy in symmetric first-price auction: bid = value * (n-1)/n
  With 2 players: bid = value / 2

Actions: integer bid from 0 to max_value

Metrics:
  - Avg profit per auction (primary)
  - NRA vs Nash baseline
  - Win rate
  - Bid shading ratio: (value - bid) / value  (higher = more strategic)
  - Overbid rate: bid > value (always negative EV)
  - Parse failure rate

Usage
-----
python gemma_blind_auction.py --backend openai --base-url http://localhost:11434/v1 --model gemma3:4b
python gemma_blind_auction.py --reasoning cot --backend openai --base-url http://localhost:11434/v1 --model gemma3:4b
"""

from __future__ import annotations
import argparse, json, re, time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np

# ─── Config ──────────────────────────────────────────────────────────────────
MAX_VALUE = 10  # private values drawn uniformly from [1, MAX_VALUE]
N_ROUNDS = 6  # auctions per episode (matches GTBench auction config)

# ─── Prompts ─────────────────────────────────────────────────────────────────
SYSTEM = """You are playing a Blind Auction (First-Price Sealed-Bid Auction).

Rules:
- You and your opponent each receive a private value for the item (integer 1-10)
- You simultaneously submit sealed bids (integer 0-10)
- The highest bid wins the item
- If you win, your profit = your private value - your bid
- If you lose, your profit = 0
- If bids tie, neither player wins

Your goal is to maximise your total profit across all rounds.
Respond with ONLY an integer bid from 0 to 10. Nothing else."""

COT_SYSTEM = """You are playing a Blind Auction (First-Price Sealed-Bid Auction).

Rules:
- You and your opponent each receive a private value for the item (integer 1-10)
- You simultaneously submit sealed bids (integer 0-10)
- The highest bid wins the item
- If you win, your profit = your private value - your bid
- If you lose, your profit = 0
- If bids tie, neither player wins

Your goal is to maximise your total profit across all rounds.
Think step by step about bid shading and expected profit, then end with your integer bid on its own line."""


def build_prompt(
    value: int, round_idx: int, n_rounds: int, my_history: list, opp_history: list
) -> str:
    lines = [
        f"Round {round_idx + 1} of {n_rounds}",
        f"Your private value this round: {value}",
    ]
    if my_history:
        pairs = [
            f"R{i+1}: you bid {m}, opp bid {o}"
            for i, (m, o) in enumerate(zip(my_history, opp_history))
        ]
        lines.append("Past rounds: " + " | ".join(pairs))
    lines.append("\nWhat is your bid (0-10)?")
    return "\n".join(lines)


def parse_bid(text: str, max_val: int = MAX_VALUE) -> Optional[int]:
    nums = re.findall(r"\b(\d+)\b", text)
    for n in reversed(nums):  # take last number (after CoT reasoning)
        val = int(n)
        if 0 <= val <= max_val:
            return val
    return None


# ─── Auction Engine ───────────────────────────────────────────────────────────


@dataclass
class RoundResult:
    my_value: int
    opp_value: int
    my_bid: int
    opp_bid: int
    my_profit: float
    opp_profit: float
    won: bool


def run_episode(
    agent, opponent_strategy, rng: np.random.Generator, n_rounds: int
) -> list[RoundResult]:
    my_bids, opp_bids = [], []
    results = []
    raw_outputs = []
    parse_fails = []

    for t in range(n_rounds):
        my_val = int(rng.integers(1, MAX_VALUE + 1))
        opp_val = int(rng.integers(1, MAX_VALUE + 1))

        # LLM bids
        sys = COT_SYSTEM if agent.reasoning == "cot" else SYSTEM
        prompt = build_prompt(my_val, t, n_rounds, my_bids, opp_bids)
        raw = agent.backend.chat(sys, prompt)
        raw_outputs.append(raw)
        my_bid = parse_bid(raw)
        if my_bid is None:
            parse_fails.append(True)
            my_bid = my_val // 2  # fallback: Nash bid
        else:
            parse_fails.append(False)

        # Opponent bids
        opp_bid = opponent_strategy(opp_val, t, my_bids, opp_bids, rng)

        # Outcome
        if my_bid > opp_bid:
            my_profit, opp_profit, won = float(my_val - my_bid), 0.0, True
        elif opp_bid > my_bid:
            my_profit, opp_profit, won = 0.0, float(opp_val - opp_bid), False
        else:
            my_profit, opp_profit, won = 0.0, 0.0, False

        my_bids.append(my_bid)
        opp_bids.append(opp_bid)
        results.append(
            RoundResult(my_val, opp_val, my_bid, opp_bid, my_profit, opp_profit, won)
        )

    # tag parse failures onto results
    for r, pf in zip(results, parse_fails):
        r.parse_fail = pf

    return results


# ─── Opponent strategies ─────────────────────────────────────────────────────


def nash_bidder(value, t, my_h, opp_h, rng):
    """Nash equilibrium: bid = value / 2 for uniform [0, max_val]."""
    return max(0, value // 2)


def truthful_bidder(value, t, my_h, opp_h, rng):
    """Truthful: bid = value."""
    return value


def random_bidder(value, t, my_h, opp_h, rng):
    return int(rng.integers(0, MAX_VALUE + 1))


OPPONENTS = {"nash": nash_bidder, "truthful": truthful_bidder, "random": random_bidder}

# ─── Metrics ─────────────────────────────────────────────────────────────────


def compute_metrics(agent, n_episodes: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    all_results = []

    for opp_name, opp_fn in OPPONENTS.items():
        for ep in range(n_episodes):
            ep_results = run_episode(agent, opp_fn, rng, N_ROUNDS)
            for r in ep_results:
                all_results.append({**vars(r), "opponent": opp_name})

    profits = [r["my_profit"] for r in all_results]
    opp_profits = [r["opp_profit"] for r in all_results]
    wins = [r["won"] for r in all_results]
    shading = [
        (r["my_value"] - r["my_bid"]) / r["my_value"]
        for r in all_results
        if r["my_value"] > 0
    ]
    overbids = [r["my_bid"] > r["my_value"] for r in all_results]
    failures = [r.get("parse_fail", False) for r in all_results]

    # NRA
    my_total = sum(profits)
    opp_total = sum(opp_profits)
    nra = (my_total - opp_total) / (abs(my_total) + abs(opp_total) + 1e-9)

    return {
        "label": "Gemma-3-4B Blind Auction",
        "avg_profit": round(float(np.mean(profits)), 3),
        "win_rate": round(float(np.mean(wins)), 3),
        "nra": round(float(nra), 3),
        "bid_shading_ratio": round(float(np.mean(shading)), 3),
        "overbid_rate": round(float(np.mean(overbids)), 3),
        "parse_failure_rate": round(float(np.mean(failures)), 3),
        "n_rounds_total": len(all_results),
    }


# ─── Backends ────────────────────────────────────────────────────────────────
class TransformersBackend:
    def __init__(self, model, max_tokens, temperature):
        import torch
        from transformers import pipeline

        self.pipe = pipeline(
            "text-generation",
            model=model,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        self.max_tokens = max_tokens
        self.temperature = temperature

    def chat(self, system, user):
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        do_s = self.temperature > 0
        out = self.pipe(
            msgs,
            max_new_tokens=self.max_tokens,
            do_sample=do_s,
            temperature=self.temperature if do_s else None,
            pad_token_id=self.pipe.tokenizer.eos_token_id,
        )
        return out[0]["generated_text"][-1]["content"]


class OpenAIBackend:
    def __init__(self, base_url, api_key, model, max_tokens, temperature):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def chat(self, system, user):
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return r.choices[0].message.content


class LLMAgent:
    def __init__(self, backend, reasoning="prompt"):
        self.backend = backend
        self.reasoning = reasoning


# ─── Main ────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--backend", default="transformers", choices=["transformers", "openai"]
    )
    p.add_argument("--model", default="google/gemma-3-4b-it")
    p.add_argument("--base-url", default="http://localhost:11434/v1")
    p.add_argument("--api-key", default="ollama")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--output-dir", default=".")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reasoning", default="prompt", choices=["prompt", "cot"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "transformers":
        backend = TransformersBackend(args.model, args.max_tokens, args.temperature)
    else:
        backend = OpenAIBackend(
            args.base_url, args.api_key, args.model, args.max_tokens, args.temperature
        )
    agent = LLMAgent(backend, reasoning=args.reasoning)

    print(f"\nBlind Auction | reasoning={args.reasoning} | episodes={args.episodes}")
    t0 = time.time()
    metrics = compute_metrics(agent, args.episodes, args.seed)
    metrics["elapsed_seconds"] = round(time.time() - t0, 1)
    metrics["model"] = args.model
    metrics["reasoning"] = args.reasoning

    print(f"\n{'─'*45}")
    print(f"Avg profit/round:  {metrics['avg_profit']:.3f}")
    print(f"Win rate:          {metrics['win_rate']:.3f}")
    print(f"NRA:               {metrics['nra']:.3f}  (>0 = better than opponents)")
    print(f"Bid shading ratio: {metrics['bid_shading_ratio']:.3f}  (>0.5 = strategic)")
    print(f"Overbid rate:      {metrics['overbid_rate']:.3f}  (should be 0)")
    print(f"Parse failures:    {metrics['parse_failure_rate']:.3f}")
    print(f"Elapsed:           {metrics['elapsed_seconds']}s")
    print(f"{'─'*45}\n")

    suffix = f"_{args.reasoning}"
    (out_dir / f"gemma_blind_auction{suffix}.json").write_text(
        json.dumps(metrics, indent=2)
    )
    print(f"Saved → gemma_blind_auction{suffix}.json")
