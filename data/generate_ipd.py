"""
SFT Warm Start Data Generation — IPD
======================================
Generates supervised fine-tuning data for the SEPO warm start stage by running
the SEPO-optimal rule-based MixedPolicy against opponent pools and recording
every (game state → action) decision as a chat JSONL example.

Paper §2.2 Stage 1: "Supervised warm start from solver- and simulation-derived
strategy traces."

SEPO-optimal weights for IPD (from sepo_gtbench_ipd.py):
  TFT=85.2%  Grim=11.5%  AlwaysD=2.0%  GenTFT=0.8%  AlwaysC=0.5%

Design: per-episode strategy sampling
  One strategy is sampled at the START of each episode and used for all 8
  rounds.  This produces clean, consistent within-episode demonstrations
  (e.g. a full TFT episode, a full Grim episode) rather than noisy round-by-
  round mixtures.  The model learns each strategy's pattern before GRPO blends
  them via reward.

Output format (MLX-LM chat JSONL):
  {"messages": [
    {"role": "system",    "content": "..."},
    {"role": "user",      "content": "Round 1 of 8..."},
    {"role": "assistant", "content": "<SILENT>"}
  ]}

Files written:
  <output-dir>/train.jsonl   80% of examples  (shuffled)
  <output-dir>/valid.jsonl   20% of examples
  <output-dir>/stats.json    generation metadata + action distribution

Usage:
  python sft_data_gen.py                          # default 200 eps/opponent
  python sft_data_gen.py --episodes-per-opponent 5 --dry-run   # quick check
  python sft_data_gen.py --episodes-per-opponent 500           # larger dataset
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np

# ── Reuse strategy classes and game constants from sepo_gtbench_ipd.py ────────
from sepo_gtbench_ipd import (
    COOPERATE, DEFECT, PAYOFF_TABLE, N_ROUNDS,
    AlwaysCooperate, AlwaysDefect, TitForTat, GenerousTitForTat, GrimTrigger,
    ALL_STRATEGIES, N_STRAT,
)

ACTION_LABEL = {COOPERATE: "COOPERATE", DEFECT: "DEFECT"}

# ─────────────────────────────────────────────────────────────────────────────
# 1. SEPO-optimal policy weights
#    Source: sepo_gtbench_ipd.py cross-entropy search output
#    Order:  [AlwaysCooperate, AlwaysDefect, TitForTat, GenTFT, GrimTrigger]
# ─────────────────────────────────────────────────────────────────────────────

SEPO_WEIGHTS = np.array([0.05, 0.30, 0.35, 0.05, 0.25])
SEPO_WEIGHTS = SEPO_WEIGHTS / SEPO_WEIGHTS.sum()   # normalise

ALL_OPPONENTS = [AlwaysCooperate, AlwaysDefect, TitForTat,
                 GenerousTitForTat, GrimTrigger]

# ─────────────────────────────────────────────────────────────────────────────
# 2. System prompt — neutral baseline
#    MUST match the prompt used during GRPO evaluation later.
#    We do NOT use the SEPO-steered prompt here: we want the model to learn
#    SEPO-aligned behavior from the demonstrations, not from instructions.
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are playing the Iterated Prisoner's Dilemma game.

Rules:
- Each round you choose one of two actions: COOPERATE or DEFECT
- If both players choose COOPERATE: you each get 3 points
- If you choose DEFECT and opponent chooses COOPERATE: you get 5 points, opponent gets 0
- If you choose COOPERATE and opponent chooses DEFECT: you get 0 points, opponent gets 5
- If both choose DEFECT: you each get 1 point

Your goal is to maximise your total score over all rounds.

Think briefly about the opponent's pattern, then end your response with your action on the last line: COOPERATE or DEFECT."""


# ─────────────────────────────────────────────────────────────────────────────
# 3. Prompt builder  (identical to llm_sepo_optimizer.py)
# ─────────────────────────────────────────────────────────────────────────────

def build_user_prompt(round_idx: int, n_rounds: int,
                      my_history: List[int], opp_history: List[int]) -> str:
    lines = [f"Round {round_idx + 1} of {n_rounds}."]
    if my_history:
        lines.append("Your past actions:       " +
                     ", ".join(ACTION_LABEL[a] for a in my_history))
        lines.append("Opponent's past actions: " +
                     ", ".join(ACTION_LABEL[a] for a in opp_history))
        my_sc  = sum(PAYOFF_TABLE[(m, o)][0] for m, o in zip(my_history, opp_history))
        opp_sc = sum(PAYOFF_TABLE[(m, o)][1] for m, o in zip(my_history, opp_history))
        lines.append(f"Scores so far — you: {my_sc}, opponent: {opp_sc}")
    else:
        lines.append("This is the first round. No history yet.")
    lines.append("\nWhat is your action?")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Core data generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_episode(strategy, opponent, n_rounds: int,
                     seed: int, rng_strat: np.random.Generator):
    """
    Run one episode and return a list of (user_prompt, action) pairs.

    strategy   — one instantiated rule-based strategy (sampled once per episode)
    opponent   — one instantiated opponent strategy
    rng_strat  — shared RNG used by the strategy (for GenTFT forgiveness etc.)
    """
    rng_opp = np.random.default_rng(seed)
    h_policy, h_opp = [], []
    examples = []

    for t in range(n_rounds):
        user_prompt = build_user_prompt(t, n_rounds, h_policy, h_opp)
        action      = strategy.act(h_policy, h_opp, rng_strat)
        opp_action  = opponent.act(h_opp, h_policy, rng_opp)

        examples.append({
            "user":     user_prompt,
            "action":   action,
            "strategy": strategy.name,
            "opponent": opponent.name,
            "round":    t,
            "h_opp":    list(h_opp),
        })

        h_policy.append(action)
        h_opp.append(opp_action)

    return examples


def generate_dataset(episodes_per_opponent: int, seed: int) -> List[dict]:
    """
    Generate all training examples across all opponents.

    For each (opponent, episode):
      1. Sample one strategy from SEPO_WEIGHTS (per-episode sampling)
      2. Run a full episode with that fixed strategy
      3. Record all N_ROUNDS (user_prompt, action) pairs
    """
    rng_select = np.random.default_rng(seed)
    rng_strat  = np.random.default_rng(seed + 1)

    all_examples = []
    strat_instances = [S() for S in ALL_STRATEGIES]

    for opp_cls in ALL_OPPONENTS:
        for ep in range(episodes_per_opponent):
            # Sample strategy for this whole episode
            strat_idx = rng_select.choice(N_STRAT, p=SEPO_WEIGHTS)
            strategy  = strat_instances[strat_idx]
            opponent  = opp_cls()
            seed_ep   = seed + ALL_OPPONENTS.index(opp_cls) * 10000 + ep

            episode_examples = generate_episode(
                strategy, opponent, N_ROUNDS, seed_ep, rng_strat)
            all_examples.extend(episode_examples)

    return all_examples


REASONING = {
    # (strategy, opponent_pattern) -> reasoning template
    "always-defect":    "Opponent always defects. I must defect to avoid being exploited.",
    "always-cooperate": "Opponent always cooperates. I can cooperate for mutual benefit.",
    "tit-for-tat":      "Opponent mirrors my last action. I'll cooperate to maintain mutual cooperation.",
    "generous-tit-for-tat": "Opponent is forgiving. Cooperation is stable here.",
    "grim-trigger":     "Opponent punishes any defection permanently. Cooperation is essential.",
}


def make_reasoning(example: dict) -> str:
    strategy = example["strategy"]
    action   = example["action"]
    h_opp    = example.get("h_opp", [])

    if strategy == "always-defect":
        reason = "Opponent always defects. Defecting limits my losses."
    elif strategy == "always-cooperate":
        reason = "Opponent always cooperates. Cooperating maximises mutual payoff."
    elif strategy in ("tit-for-tat", "generous-tit-for-tat"):
        if h_opp and h_opp[-1] == 1:  # opponent defected last
            reason = "Opponent defected last round. I should retaliate to discourage exploitation."
        else:
            reason = "Opponent cooperated last round. Cooperating maintains mutual benefit."
    elif strategy == "grim-trigger":
        if 1 in h_opp:
            reason = "Opponent triggered grim — permanent defection. I must defect."
        else:
            reason = "No defection yet. Cooperating keeps the relationship stable."
    else:
        reason = "Choosing based on the opponent's pattern."

    return f"{reason}\n{ACTION_LABEL[action]}"


def to_chat_jsonl(example: dict) -> dict:
    """Convert raw example to chat JSONL format with reasoning."""
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": example["user"]},
            {"role": "assistant", "content": make_reasoning(example)},
        ]
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Statistics
# ─────────────────────────────────────────────────────────────────────────────

def compute_stats(examples: List[dict], episodes_per_opponent: int) -> dict:
    total = len(examples)
    n_cooperate = sum(1 for e in examples if e["action"] == COOPERATE)
    n_defect    = total - n_cooperate

    # Per-strategy action distribution
    by_strategy = defaultdict(lambda: {"cooperate": 0, "defect": 0, "total": 0})
    for e in examples:
        s = e["strategy"]
        by_strategy[s]["total"] += 1
        if e["action"] == COOPERATE:
            by_strategy[s]["cooperate"] += 1
        else:
            by_strategy[s]["defect"] += 1

    # Per-opponent action distribution
    by_opponent = defaultdict(lambda: {"cooperate": 0, "defect": 0, "total": 0})
    for e in examples:
        o = e["opponent"]
        by_opponent[o]["total"] += 1
        if e["action"] == COOPERATE:
            by_opponent[o]["cooperate"] += 1
        else:
            by_opponent[o]["defect"] += 1

    # Expected strategy sample counts
    expected_strategy_counts = {
        ALL_STRATEGIES[i].name: round(SEPO_WEIGHTS[i] * episodes_per_opponent
                                      * len(ALL_OPPONENTS), 1)
        for i in range(N_STRAT)
    }

    return {
        "total_examples":       total,
        "n_opponents":          len(ALL_OPPONENTS),
        "episodes_per_opponent": episodes_per_opponent,
        "rounds_per_episode":   N_ROUNDS,
        "total_episodes":       episodes_per_opponent * len(ALL_OPPONENTS),
        "action_distribution": {
            "cooperate": n_cooperate,
            "defect":    n_defect,
            "cooperate_pct": round(100 * n_cooperate / total, 1),
            "defect_pct":    round(100 * n_defect    / total, 1),
        },
        "by_strategy":  {k: dict(v) for k, v in by_strategy.items()},
        "by_opponent":  {k: dict(v) for k, v in by_opponent.items()},
        "expected_strategy_episode_counts": expected_strategy_counts,
        "sepo_weights": {
            ALL_STRATEGIES[i].name: round(float(SEPO_WEIGHTS[i]), 4)
            for i in range(N_STRAT)
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Output
# ─────────────────────────────────────────────────────────────────────────────

def save_jsonl(records: List[dict], path: Path):
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"  Saved {len(records):>6} examples → {path}")


def print_stats(stats: dict):
    print()
    print(f"  Total examples : {stats['total_examples']}")
    print(f"  Opponents      : {stats['n_opponents']}")
    print(f"  Episodes/opp   : {stats['episodes_per_opponent']}")
    print(f"  Rounds/episode : {stats['rounds_per_episode']}")
    print()
    ad = stats["action_distribution"]
    print(f"  Action distribution:")
    print(f"    <SILENT>  (cooperate): {ad['cooperate']:>5}  ({ad['cooperate_pct']}%)")
    print(f"    <TESTIFY> (defect):    {ad['defect']:>5}  ({ad['defect_pct']}%)")
    print()
    print(f"  Per-strategy breakdown (actions):")
    for name, s in stats["by_strategy"].items():
        coop_pct = round(100 * s["cooperate"] / s["total"], 1) if s["total"] else 0
        bar = "█" * int(coop_pct / 5)
        print(f"    {name:<25}  coop={coop_pct:>5.1f}%  {bar}")
    print()
    print(f"  Per-opponent breakdown (actions):")
    for name, s in stats["by_opponent"].items():
        coop_pct = round(100 * s["cooperate"] / s["total"], 1) if s["total"] else 0
        print(f"    vs. {name:<22}  coop={coop_pct:>5.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="Generate SFT warm-start data for SEPO IPD")
    ap.add_argument("--episodes-per-opponent", type=int, default=200,
                    help="Episodes per opponent (default 200 → 8,000 examples total)")
    ap.add_argument("--output-dir", default="sepo_sft_data",
                    help="Output directory (default: sepo_sft_data/)")
    ap.add_argument("--train-frac", type=float, default=0.8,
                    help="Fraction of data for training set (default 0.8)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print one sample example and stats without writing files")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = Path(args.output_dir)

    print("=" * 60)
    print("  SEPO SFT Warm Start — Data Generation (IPD)")
    print("=" * 60)
    print(f"  Episodes/opponent : {args.episodes_per_opponent}")
    print(f"  Opponents         : {[c.name for c in ALL_OPPONENTS]}")
    print(f"  Total episodes    : {args.episodes_per_opponent * len(ALL_OPPONENTS)}")
    print(f"  Expected examples : {args.episodes_per_opponent * len(ALL_OPPONENTS) * N_ROUNDS}")
    print(f"  Strategy weights  : TFT={SEPO_WEIGHTS[2]:.3f}  "
          f"Grim={SEPO_WEIGHTS[4]:.3f}  "
          f"AlwaysD={SEPO_WEIGHTS[1]:.3f}  "
          f"GenTFT={SEPO_WEIGHTS[3]:.3f}  "
          f"AlwaysC={SEPO_WEIGHTS[0]:.3f}")
    print()

    # Generate
    print("Generating episodes...")
    raw_examples = generate_dataset(args.episodes_per_opponent, seed=args.seed)

    # Stats
    stats = compute_stats(raw_examples, args.episodes_per_opponent)
    print_stats(stats)

    if args.dry_run:
        print()
        print("── Sample training example (dry run) ──")
        sample = to_chat_jsonl(raw_examples[0])
        print(json.dumps(sample, indent=2))
        print()
        print("Dry run complete. No files written.")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

        # Convert to MLX-LM chat format and shuffle
        chat_examples = [to_chat_jsonl(e) for e in raw_examples]
        rng_shuffle = random.Random(args.seed)
        rng_shuffle.shuffle(chat_examples)

        n_train = int(len(chat_examples) * args.train_frac)
        train_set = chat_examples[:n_train]
        valid_set = chat_examples[n_train:]

        print()
        print(f"Writing dataset to {out_dir}/")
        save_jsonl(train_set, out_dir / "train.jsonl")
        save_jsonl(valid_set, out_dir / "valid.jsonl")

        stats["train_examples"] = len(train_set)
        stats["valid_examples"] = len(valid_set)
        stats["train_frac"]     = args.train_frac
        stats["seed"]           = args.seed

        stats_path = out_dir / "stats.json"
        stats_path.write_text(json.dumps(stats, indent=2))
        print(f"  Saved stats       → {stats_path}")

        print()
        print("Done. Next steps:")
        print("  1. pip install mlx-lm")
        print(f"  2. mlx_lm.lora \\")
        print(f"       --model google/gemma-3-4b-it \\")
        print(f"       --train \\")
        print(f"       --data ./{out_dir} \\")
        print(f"       --iters 1000 \\")
        print(f"       --batch-size 4 \\")
        print(f"       --lora-layers 8 \\")
        print(f"       --dtype bfloat16")
        print()
        print("  3. mlx_lm.fuse \\")
        print(f"       --model google/gemma-3-4b-it \\")
        print(f"       --adapter-path mlx_model \\")
        print(f"       --save-path ./gemma-sepo-sft-merged")
        print()
        print("  4. huggingface-cli upload <username>/gemma-sepo-sft-ipd ./gemma-sepo-sft-merged")
