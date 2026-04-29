"""
Multi-Game SFT Warm Start Data Generation
==========================================
Generates SFT training data for all 4 SEPO games in one combined dataset:
  IPD          — COOPERATE / DEFECT (8 rounds)
  Resource     — LOW / MEDIUM / HIGH extraction (8 rounds)
  Auction      — LOW / MEDIUM / HIGH bid (6 rounds)
  Negotiation  — integer demand 1–9 (4 rounds)

Each game uses SEPO-optimal rule-based policies as demonstrations, plus a small
random-strategy weight (~8%) to expose the model to recovery from suboptimal
moves and prevent over-specialisation.

Strategy weight design (per game):
  ~50-60% cooperative/adaptive strategies  (teach baseline cooperative play)
  ~20-25% punishment/credible-threat        (teach exploitation resistance)
  ~8%     random exploration               (teach recovery, add diversity)

Output:
  <output-dir>/train.jsonl   80% of combined examples (shuffled across all games)
  <output-dir>/valid.jsonl   20%
  <output-dir>/stats.json    generation metadata + per-game counts
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List

import numpy as np

from games.ipd import (
    IPDGame, COOPERATE, DEFECT, PAYOFF_TABLE,
    AlwaysCooperate, AlwaysDefect, TitForTat, GenerousTitForTat, GrimTrigger,
)
from games.resource import ResourceGame, HighExtract, MediumExtract, ScarcityAware
from games.auction import AuctionGame, AggressiveBid, TruthfulBid, ConservativeBid
from games.negotiation import NegotiationGame, SelfishSplit, FairSplit, BalancedSplit

TOTAL_PIE = 10


# ─────────────────────────────────────────────────────────────────────────────
# IPD strategies + weights
# ─────────────────────────────────────────────────────────────────────────────

class RandomIPD:
    name = "random-ipd"
    def act(self, h_llm, h_opp, rng):
        return int(rng.integers(0, 2))  # 0=COOPERATE, 1=DEFECT


IPD_STRATEGIES = [AlwaysCooperate, AlwaysDefect, TitForTat, GenerousTitForTat, GrimTrigger, RandomIPD]
IPD_WEIGHTS    = np.array([0.05, 0.27, 0.33, 0.05, 0.22, 0.08])
IPD_WEIGHTS    = IPD_WEIGHTS / IPD_WEIGHTS.sum()
IPD_OPPONENTS  = [AlwaysCooperate, AlwaysDefect, TitForTat, GenerousTitForTat, GrimTrigger]
IPD_LABEL      = {COOPERATE: "COOPERATE", DEFECT: "DEFECT"}


def _ipd_reasoning(strategy_name: str, action: int, h_opp: List[int]) -> str:
    if strategy_name == "random-ipd":
        reason = "Testing the opponent's response with a different action to gather information."
    elif "grim" in strategy_name:
        reason = ("Opponent triggered grim — defecting permanently to avoid further exploitation."
                  if DEFECT in h_opp else
                  "No defection yet. Cooperating keeps the relationship stable.")
    elif "always-defect" in strategy_name:
        reason = "Opponent always defects. Defecting limits my losses."
    elif "always-cooperate" in strategy_name:
        reason = "Opponent always cooperates. Cooperating maximises mutual payoff."
    elif "tit-for-tat" in strategy_name or "generous" in strategy_name:
        reason = ("Opponent defected last round. Retaliating to discourage exploitation."
                  if h_opp and h_opp[-1] == DEFECT else
                  "Opponent cooperated. Maintaining mutual cooperation.")
    else:
        reason = "Choosing based on opponent's pattern."
    return f"{reason}\n{IPD_LABEL[action]}"


# ─────────────────────────────────────────────────────────────────────────────
# Resource strategies + weights
# ─────────────────────────────────────────────────────────────────────────────

class ResAlwaysLow:
    name = "res-always-low"
    def act(self, h_llm, h_opp, stock, rng): return 1

class ResTFT:
    name = "res-tft"
    def act(self, h_llm, h_opp, stock, rng):
        if not h_opp: return 1
        return h_opp[-1]

class ResGrim:
    name = "res-grim"
    def act(self, h_llm, h_opp, stock, rng):
        return 3 if any(e == 3 for e in h_opp) else 1

class ResScarcity:
    name = "res-scarcity"
    def act(self, h_llm, h_opp, stock, rng):
        return 1 if stock <= 6 else 2

class RandomRes:
    name = "random-res"
    def act(self, h_llm, h_opp, stock, rng):
        return int(rng.integers(1, 4))  # 1, 2, or 3


RES_STRATEGIES = [ResAlwaysLow, ResTFT, ResGrim, ResScarcity, RandomRes]
RES_WEIGHTS    = np.array([0.12, 0.40, 0.22, 0.18, 0.08])
RES_WEIGHTS    = RES_WEIGHTS / RES_WEIGHTS.sum()
RES_OPPONENTS  = [HighExtract, MediumExtract, ScarcityAware]
RES_LABEL      = {1: "LOW", 2: "MEDIUM", 3: "HIGH"}


def _resource_reasoning(strategy_name: str, action: int, h_opp: List[int], stock: int) -> str:
    if strategy_name == "random-res":
        reason = f"Testing a different extraction level to learn the opponent's pattern."
    elif stock <= 4:
        reason = f"Stock critically low at {stock}. Extracting LOW to prevent depletion penalty."
    elif "grim" in strategy_name and any(e == 3 for e in h_opp):
        reason = "Opponent over-extracted HIGH. Matching HIGH to signal retaliation."
    elif "tft" in strategy_name and h_opp:
        reason = f"Opponent extracted {RES_LABEL[h_opp[-1]]} last round. Mirroring to enforce sustainable extraction."
    elif stock <= 6:
        reason = f"Stock at {stock}. Extracting LOW to conserve the resource."
    else:
        reason = f"Stock healthy at {stock}. Extracting {RES_LABEL[action]} sustainably."
    return f"{reason}\n{RES_LABEL[action]}"


# ─────────────────────────────────────────────────────────────────────────────
# Auction strategies + weights
# ─────────────────────────────────────────────────────────────────────────────

class AucValueBid:
    name = "auc-value-bid"
    def act(self, h_llm, h_opp, my_val, rng):
        return 1 if my_val == 2 else 2

class AucAggressiveValue:
    name = "auc-aggressive-value"
    def act(self, h_llm, h_opp, my_val, rng):
        return 1 if my_val == 2 else 3

class AucAdaptive:
    name = "auc-adaptive"
    def act(self, h_llm, h_opp, my_val, rng):
        if not h_opp:
            return 1 if my_val == 2 else 2
        avg_opp = sum(h_opp) / len(h_opp)
        if my_val == 4:
            return 3 if avg_opp >= 2.5 else 2
        return 1

class RandomAuc:
    name = "random-auc"
    def act(self, h_llm, h_opp, my_val, rng):
        return int(rng.integers(1, 4))  # 1, 2, or 3


AUC_STRATEGIES = [AucValueBid, AucAggressiveValue, AucAdaptive, RandomAuc]
AUC_WEIGHTS    = np.array([0.44, 0.28, 0.20, 0.08])
AUC_WEIGHTS    = AUC_WEIGHTS / AUC_WEIGHTS.sum()
AUC_OPPONENTS  = [AggressiveBid, TruthfulBid, ConservativeBid]
AUC_LABEL      = {1: "LOW", 2: "MEDIUM", 3: "HIGH"}


def _auction_reasoning(strategy_name: str, action: int, my_val: int, h_opp: List[int]) -> str:
    if strategy_name == "random-auc":
        reason = f"Trying a different bid level to gather information about the opponent's strategy."
    elif my_val == 2:
        reason = "My item value is 2. A LOW bid of 1 gives profit of 1 if I win — optimal for low value."
    elif h_opp and sum(h_opp) / len(h_opp) >= 2.5:
        reason = (f"My value is 4. Opponent bids aggressively "
                  f"(avg {sum(h_opp)/len(h_opp):.1f}). Bidding {AUC_LABEL[action]} to compete.")
    else:
        profit = my_val - action
        reason = f"My item value is 4. Bidding {AUC_LABEL[action]} gives profit of {profit} if I win."
    return f"{reason}\n{AUC_LABEL[action]}"


# ─────────────────────────────────────────────────────────────────────────────
# Negotiation strategies + weights
# ─────────────────────────────────────────────────────────────────────────────

class NegFair:
    name = "neg-fair"
    def act(self, h_llm, h_opp, rng): return 5

class NegBalanced:
    name = "neg-balanced"
    def act(self, h_llm, h_opp, rng):
        if h_opp and h_opp[-1] >= 7: return 3
        return 6

class NegConcede:
    name = "neg-concede"
    def act(self, h_llm, h_opp, rng):
        if h_llm and h_opp and h_llm[-1] + h_opp[-1] > TOTAL_PIE:
            return max(3, h_llm[-1] - 1)
        return 6

class NegTFT:
    name = "neg-tft"
    def act(self, h_llm, h_opp, rng):
        if not h_opp: return 5
        return max(1, min(9, TOTAL_PIE - h_opp[-1]))

class RandomNeg:
    name = "random-neg"
    def act(self, h_llm, h_opp, rng):
        return int(rng.integers(3, 8))  # 3–7: avoids extreme demands


NEG_STRATEGIES = [NegFair, NegBalanced, NegConcede, NegTFT, RandomNeg]
NEG_WEIGHTS    = np.array([0.32, 0.27, 0.18, 0.15, 0.08])
NEG_WEIGHTS    = NEG_WEIGHTS / NEG_WEIGHTS.sum()
NEG_OPPONENTS  = [SelfishSplit, FairSplit, BalancedSplit]


def _negotiation_reasoning(strategy_name: str, action: int,
                            h_llm: List[int], h_opp: List[int]) -> str:
    if strategy_name == "random-neg":
        reason = f"Testing demand of {action} to gauge how the opponent responds."
    elif h_opp and h_opp[-1] >= 7:
        reason = (f"Opponent demands {h_opp[-1]}. Conceding to {action} "
                  "to ensure a deal — better than breakdown.")
    elif h_llm and h_opp and h_llm[-1] + h_opp[-1] > TOTAL_PIE:
        reason = f"Last round broke down. Lowering demand to {action} to restart cooperation."
    elif action == 5:
        reason = "Demanding a fair 5 — guarantees a deal with most opponents."
    else:
        reason = f"Demanding {action} — leaves {TOTAL_PIE - action} for opponent, ensuring deal."
    return f"{reason}\n{action}"


# ─────────────────────────────────────────────────────────────────────────────
# Episode simulators (use each game's state machine for prompt consistency)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_ipd(strategy, opponent_inst, game, rng) -> List[dict]:
    state = game.reset(opponent_inst, rng)
    examples = []
    while True:
        user_prompt = game.user_prompt(state)
        action = strategy.act(state["h_llm"], state["h_opp"], rng)
        examples.append({
            "game": "ipd", "user": user_prompt, "action": action,
            "strategy": strategy.name, "opponent": opponent_inst.name,
            "h_opp": list(state["h_opp"]),
        })
        state, _, _, done = game.step(action, state, rng)
        if done: break
    return examples


def simulate_resource(strategy, opponent_inst, game, rng) -> List[dict]:
    state = game.reset(opponent_inst, rng)
    examples = []
    while True:
        user_prompt = game.user_prompt(state)
        action = strategy.act(state["h_llm"], state["h_opp"], state["stock"], rng)
        examples.append({
            "game": "resource", "user": user_prompt, "action": action,
            "strategy": strategy.name, "opponent": opponent_inst.name,
            "stock": state["stock"], "h_opp": list(state["h_opp"]),
        })
        state, _, _, done = game.step(action, state, rng)
        if done: break
    return examples


def simulate_auction(strategy, opponent_inst, game, rng) -> List[dict]:
    state = game.reset(opponent_inst, rng)
    examples = []
    while True:
        user_prompt = game.user_prompt(state)
        action = strategy.act(state["h_llm"], list(state["h_opp"]), state["my_value"], rng)
        examples.append({
            "game": "auction", "user": user_prompt, "action": action,
            "strategy": strategy.name, "opponent": opponent_inst.name,
            "my_val": state["my_value"], "h_opp": list(state["h_opp"]),
        })
        state, _, _, done = game.step(action, state, rng)
        if done: break
    return examples


def simulate_negotiation(strategy, opponent_inst, game, rng) -> List[dict]:
    state = game.reset(opponent_inst, rng)
    examples = []
    while True:
        user_prompt = game.user_prompt(state)
        action = strategy.act(state["h_llm"], state["h_opp"], rng)
        examples.append({
            "game": "negotiation", "user": user_prompt, "action": action,
            "strategy": strategy.name, "opponent": opponent_inst.name,
            "h_llm": list(state["h_llm"]), "h_opp": list(state["h_opp"]),
        })
        state, _, _, done = game.step(action, state, rng)
        if done: break
    return examples


# ─────────────────────────────────────────────────────────────────────────────
# Chat format
# ─────────────────────────────────────────────────────────────────────────────

def to_chat_jsonl(example: dict, system_prompt: str) -> dict:
    g = example["game"]
    if g == "ipd":
        reasoning = _ipd_reasoning(example["strategy"], example["action"], example["h_opp"])
    elif g == "resource":
        reasoning = _resource_reasoning(example["strategy"], example["action"],
                                        example["h_opp"], example["stock"])
    elif g == "auction":
        reasoning = _auction_reasoning(example["strategy"], example["action"],
                                       example["my_val"], example["h_opp"])
    else:
        reasoning = _negotiation_reasoning(example["strategy"], example["action"],
                                           example["h_llm"], example["h_opp"])
    return {
        "messages": [
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": example["user"]},
            {"role": "assistant", "content": reasoning},
        ]
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dataset generation
# ─────────────────────────────────────────────────────────────────────────────

GAME_CONFIG = {
    "ipd":         (lambda: IPDGame(n_rounds=8),         IPD_STRATEGIES, IPD_WEIGHTS, IPD_OPPONENTS, simulate_ipd),
    "resource":    (lambda: ResourceGame(n_rounds=8),    RES_STRATEGIES, RES_WEIGHTS, RES_OPPONENTS, simulate_resource),
    "auction":     (lambda: AuctionGame(n_rounds=6),     AUC_STRATEGIES, AUC_WEIGHTS, AUC_OPPONENTS, simulate_auction),
    "negotiation": (lambda: NegotiationGame(n_rounds=4), NEG_STRATEGIES, NEG_WEIGHTS, NEG_OPPONENTS, simulate_negotiation),
}


def generate_game(game_name: str, episodes_per_opponent: int, seed: int):
    game_cls, strat_cls, weights, opp_cls, simulate = GAME_CONFIG[game_name]
    game     = game_cls()
    strats   = [S() for S in strat_cls]
    opps     = [O() for O in opp_cls]
    rng_sel  = np.random.default_rng(seed)
    examples = []

    for opp in opps:
        for ep in range(episodes_per_opponent):
            idx      = rng_sel.choice(len(strats), p=weights)
            strategy = strats[idx]
            ep_rng   = np.random.default_rng(seed + abs(hash(game_name + opp.name)) % 100000 + ep)
            examples.extend(simulate(strategy, opp, game, ep_rng))

    return examples, game.system_prompt()


def generate_all(episodes_per_opponent: int, seed: int):
    all_examples, system_prompts, counts = [], {}, {}
    for name in GAME_CONFIG:
        print(f"  [{name}] generating...", flush=True)
        examples, sys_prompt = generate_game(name, episodes_per_opponent, seed)
        all_examples.extend(examples)
        system_prompts[name] = sys_prompt
        counts[name] = len(examples)
        print(f"  [{name}] {len(examples)} examples")
    return all_examples, system_prompts, counts


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description="Generate multi-game SFT warm-start data")
    ap.add_argument("--episodes-per-opponent", type=int, default=200,
                    help="Episodes per opponent per game (default 200)")
    ap.add_argument("--output-dir", default="sepo_sft_data_multi")
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print one example per game and exit without writing files")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = Path(args.output_dir)

    print("=" * 60)
    print("  SEPO Multi-Game SFT Data Generation")
    print("=" * 60)
    print(f"  Games             : IPD, Resource, Auction, Negotiation")
    print(f"  Episodes/opponent : {args.episodes_per_opponent}")
    print(f"  Random strategy   : 8% weight per game (exploration)")
    print()

    raw_examples, system_prompts, counts = generate_all(args.episodes_per_opponent, args.seed)

    total = len(raw_examples)
    print(f"\n  Total examples : {total}")
    for g, n in counts.items():
        print(f"    {g:<14}: {n:>6}  ({100*n/total:.1f}%)")

    if args.dry_run:
        print("\n── Sample example per game (dry run) ──")
        for g in GAME_CONFIG:
            ex = next(e for e in raw_examples if e["game"] == g)
            chat = to_chat_jsonl(ex, system_prompts[g])
            print(f"\n[{g.upper()}]")
            print(json.dumps(chat, indent=2))
        print("\nDry run complete. No files written.")
    else:
        chat_examples = [to_chat_jsonl(e, system_prompts[e["game"]]) for e in raw_examples]
        rng_shuffle = random.Random(args.seed)
        rng_shuffle.shuffle(chat_examples)

        n_train   = int(len(chat_examples) * args.train_frac)
        train_set = chat_examples[:n_train]
        valid_set = chat_examples[n_train:]

        out_dir.mkdir(parents=True, exist_ok=True)

        def _save(records, path):
            with path.open("w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
            print(f"  Saved {len(records):>6} examples → {path}")

        print()
        _save(train_set, out_dir / "train.jsonl")
        _save(valid_set, out_dir / "valid.jsonl")

        stats = {
            "total": total, "train": len(train_set), "valid": len(valid_set),
            "per_game": counts,
            "episodes_per_opponent": args.episodes_per_opponent,
            "seed": args.seed,
            "strategy_weights": {
                "ipd":         {IPD_STRATEGIES[i].name: round(float(IPD_WEIGHTS[i]), 3) for i in range(len(IPD_STRATEGIES))},
                "resource":    {RES_STRATEGIES[i].name: round(float(RES_WEIGHTS[i]), 3) for i in range(len(RES_STRATEGIES))},
                "auction":     {AUC_STRATEGIES[i].name: round(float(AUC_WEIGHTS[i]), 3) for i in range(len(AUC_STRATEGIES))},
                "negotiation": {NEG_STRATEGIES[i].name: round(float(NEG_WEIGHTS[i]), 3) for i in range(len(NEG_STRATEGIES))},
            },
        }
        (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
        print(f"  Saved stats       → {out_dir}/stats.json")
        print(f"\n  Done. Next: python sft_train.py --data-dir {out_dir}")
