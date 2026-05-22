"""
SFT data generation for GTBench-style multi-issue negotiation.

Generates episodes where strategies play according to SEPO-optimal policies.
Each example: system prompt + user prompt (state) + assistant response (reasoning + [a,b,c]).
"""

from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from typing import List, Dict

import numpy as np

from games.negotiation_gtbench import (
    NegotiationGTBenchGame, _sample_pool, _sample_values,
    _resolve, ITEMS, N_ITEMS, MAX_VAL, MAX_POOL,
    GreedyNeg, FairNeg, ConcedeNeg, TFTNeg, HardballNeg
)

# ── Strategy classes (for SFT demonstrations) ────────────────────────────────

class ProportionalStrategy:
    """Demand items proportional to own values — maximise expected payoff."""
    name = "proportional"

    def act(self, my_values, pool, h_llm, h_opp, rng):
        total_v = sum(my_values) or 1
        demand = []
        for i in range(N_ITEMS):
            share = my_values[i] / total_v
            d = max(0, min(pool[i], round(pool[i] * share + 0.5)))
            demand.append(d)
        return demand

    def reasoning(self, my_values, pool, demand, h_llm, h_opp):
        reason_parts = []
        for i, item in enumerate(ITEMS):
            if my_values[i] > 0:
                reason_parts.append(f"{item} (value={my_values[i]})")
        r = f"My highest-value items are {', '.join(reason_parts)}. "
        r += f"I'll demand proportional to my values to maximise payoff while leaving room for a deal."
        return r


class ConservativeStrategy:
    """Demand a bit less than proportional — prioritise deal-making."""
    name = "conservative"

    def act(self, my_values, pool, h_llm, h_opp, rng):
        total_v = sum(my_values) or 1
        demand = []
        for i in range(N_ITEMS):
            share = my_values[i] / total_v
            d = max(0, min(pool[i], int(pool[i] * share * 0.8)))
            demand.append(d)
        return demand

    def reasoning(self, my_values, pool, demand, h_llm, h_opp):
        breakdowns = sum(1 for p, op in zip(
            [sum(d[j]*my_values[j] for j in range(N_ITEMS)) for d in h_llm],
            [0]*len(h_llm)
        ) if p == 0) if h_llm else 0
        if breakdowns > 0:
            r = f"We had {breakdowns} breakdown(s). I'll reduce my demands to secure a deal."
        else:
            r = "I'll demand slightly below my proportional share to ensure we reach a deal."
        return r


class AdaptiveStrategy:
    """Concede on low-value items while protecting high-value ones."""
    name = "adaptive"

    def act(self, my_values, pool, h_llm, h_opp, rng):
        if not h_opp:
            # First round: demand most of high-value items
            total_v = sum(my_values) or 1
            return [max(0, min(pool[i], int(pool[i] * my_values[i] / total_v * 1.2)))
                    for i in range(N_ITEMS)]
        # Check last round outcome
        last_llm = h_llm[-1]
        last_opp = h_opp[-1]
        last_deal = all(last_llm[i] + last_opp[i] <= pool[i] for i in range(N_ITEMS))
        if last_deal:
            # Deal worked, keep same demand
            return list(last_llm)
        else:
            # Breakdown: concede 1 unit on lowest-value item
            sorted_items = sorted(range(N_ITEMS), key=lambda i: my_values[i])
            demand = list(last_llm)
            for i in sorted_items:
                if demand[i] > 0:
                    demand[i] -= 1
                    break
            return [max(0, d) for d in demand]

    def reasoning(self, my_values, pool, demand, h_llm, h_opp):
        if not h_opp:
            top_item = ITEMS[int(np.argmax(my_values))]
            return f"First round. I value {top_item} most — demand more of it while leaving space for a deal."
        last_llm = h_llm[-1]
        last_opp = h_opp[-1]
        last_deal = all(last_llm[i] + last_opp[i] <= pool[i] for i in range(N_ITEMS))
        if last_deal:
            return "Last round resulted in a deal. Maintaining my demand."
        else:
            low_item = ITEMS[int(np.argmin(my_values))]
            return f"Breakdown last round. Conceding on {low_item} (my lowest-value item) to secure a deal."


class DefensiveStrategy:
    """Demand minimum needed based on opponent pattern — resist exploitation."""
    name = "defensive"

    def act(self, my_values, pool, h_llm, h_opp, rng):
        if not h_opp:
            total_v = sum(my_values) or 1
            return [max(0, min(pool[i], int(pool[i] * my_values[i] / total_v)))
                    for i in range(N_ITEMS)]
        # Infer opponent's likely demand from history
        avg_opp = [
            int(np.mean([h[i] for h in h_opp]))
            for i in range(N_ITEMS)
        ]
        # Demand remainder after opponent's expected demand
        demand = [max(0, pool[i] - avg_opp[i]) for i in range(N_ITEMS)]
        return demand

    def reasoning(self, my_values, pool, demand, h_llm, h_opp):
        if not h_opp:
            return "First round. Demanding my proportional share."
        avg_opp = [int(np.mean([h[i] for h in h_opp])) for i in range(N_ITEMS)]
        opp_summary = ", ".join(f"{ITEMS[i]}={avg_opp[i]}" for i in range(N_ITEMS))
        return (f"Opponent typically demands [{opp_summary}]. "
                f"I'll demand the remainder to secure a deal and protect my payoff.")


STRATEGIES = [
    ProportionalStrategy(),
    ConservativeStrategy(),
    AdaptiveStrategy(),
    DefensiveStrategy(),
]

STRATEGY_WEIGHTS = [0.35, 0.25, 0.25, 0.15]

OPPONENTS = [FairNeg(), TFTNeg(), ConcedeNeg(), GreedyNeg(), HardballNeg()]
OPPONENT_WEIGHTS = [0.25, 0.25, 0.20, 0.20, 0.10]


# ── Episode simulation ────────────────────────────────────────────────────────

def simulate_episode(strategy, opponent, game, rng) -> List[Dict]:
    """Simulate one episode and return list of (state, action, reasoning) dicts."""
    state   = game.reset(opponent, rng)
    examples = []
    done    = False

    while not done:
        user_prompt = game.user_prompt(state)
        pool        = state["pool"]
        my_values   = state["my_values"]
        h_llm       = state["h_llm"]
        h_opp       = state["h_opp"]

        action   = strategy.act(my_values, pool, h_llm, h_opp, rng)
        # Clamp to valid range
        action   = [max(0, min(pool[i], action[i])) for i in range(N_ITEMS)]
        reasoning = strategy.reasoning(my_values, pool, action, h_llm, h_opp)

        examples.append({
            "user":      user_prompt,
            "action":    action,
            "reasoning": reasoning,
            "strategy":  strategy.name,
        })

        state, _, _, done = game.step(action, state, rng)

    return examples


def to_chat_jsonl(example: Dict, system_prompt: str) -> Dict:
    action_str = f"[{example['action'][0]}, {example['action'][1]}, {example['action'][2]}]"
    content    = f"{example['reasoning']}\n{action_str}"
    return {
        "messages": [
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": example["user"]},
            {"role": "assistant", "content": content},
        ]
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Generate GTBench negotiation SFT data")
    ap.add_argument("--episodes-per-opponent", type=int, default=200)
    ap.add_argument("--output-dir",            default="sepo_sft_neg_gtbench")
    ap.add_argument("--train-frac",            type=float, default=0.8)
    ap.add_argument("--seed",                  type=int,   default=42)
    ap.add_argument("--dry-run",               action="store_true")
    args = ap.parse_args()

    rng  = np.random.default_rng(args.seed)
    game = NegotiationGTBenchGame(n_rounds=4)

    all_examples = []
    for opp_idx, opponent in enumerate(OPPONENTS):
        for ep in range(args.episodes_per_opponent):
            strategy = rng.choice(STRATEGIES, p=STRATEGY_WEIGHTS)
            examples = simulate_episode(strategy, opponent, game, rng)
            all_examples.extend(examples)
        print(f"  [{opponent.name}] {args.episodes_per_opponent} episodes done", flush=True)

    # Shuffle
    rng_py = random.Random(args.seed)
    rng_py.shuffle(all_examples)

    split = int(len(all_examples) * args.train_frac)
    train = all_examples[:split]
    valid = all_examples[split:]

    if args.dry_run:
        print(f"Dry run: {len(all_examples)} examples ({len(train)} train / {len(valid)} valid)")
        print("Sample:", json.dumps(to_chat_jsonl(train[0], game.system_prompt()), indent=2)[:500])
        return

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sys_prompt = game.system_prompt()
    with open(out / "train.jsonl", "w") as f:
        for ex in train:
            f.write(json.dumps(to_chat_jsonl(ex, sys_prompt)) + "\n")
    with open(out / "valid.jsonl", "w") as f:
        for ex in valid:
            f.write(json.dumps(to_chat_jsonl(ex, sys_prompt)) + "\n")

    stats = {
        "total": len(all_examples),
        "train": len(train),
        "valid": len(valid),
        "episodes_per_opponent": args.episodes_per_opponent,
        "opponents": [o.name for o in OPPONENTS],
        "strategies": [s.name for s in STRATEGIES],
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\nGenerated {len(all_examples)} examples → {out}/")
    print(f"  Train: {len(train)}  Valid: {len(valid)}")


if __name__ == "__main__":
    main()
