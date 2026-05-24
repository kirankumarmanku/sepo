"""
SFT Warm Start Data Generation — Kuhn Poker
=============================================
Generates supervised fine-tuning data for Kuhn Poker by running approximate
Nash and well-known heuristic strategies against an opponent pool, recording
every (game state → action) decision as a chat JSONL example.

Strategy weights (SEPO-style mix):
  Nash-approx           : 45%  (theoretical optimal)
  Tight-value           : 25%  (only bet/call with K)
  Pot-control           : 20%  (bet K, pass Q, bet J as bluff sometimes)
  Random-legal          :  8%  (exploration / recovery from suboptimal lines)
  Always-pass-with-Q    :  2%  (low-quality variant for diversity)

Output format (chat JSONL):
  {"messages": [
    {"role": "system",    "content": "..."},
    {"role": "user",      "content": "Hand 1 of 6.\\nYour card: K\\n..."},
    {"role": "assistant", "content": "I have the strongest card. Betting for value.\\nBET"}
  ]}

Files written:
  <output-dir>/train.jsonl   80% of examples (shuffled)
  <output-dir>/valid.jsonl   20%
  <output-dir>/stats.json    generation metadata

Usage:
  python sft_data_gen_kuhn.py
  python sft_data_gen_kuhn.py --episodes-per-opponent 500
  python sft_data_gen_kuhn.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List

import numpy as np

from games.kuhn import (
    ACTION_LABEL,
    BET,
    CALL,
    CARD_J,
    CARD_K,
    CARD_LABEL,
    CARD_Q,
    FOLD,
    PASS,
    AlwaysBet,
    AlwaysPass,
    KuhnPokerGame,
    LooseAggressive,
    MaxExploiter,
    NashApprox,
    TightPassive,
    _legal_actions,
)

# ─────────────────────────────────────────────────────────────────────────────
# Strategies — each takes (card, history, rng) and returns an action
# ─────────────────────────────────────────────────────────────────────────────


class StrategyNashApprox:
    """Approximate Nash equilibrium strategy (alpha ≈ 1/3)."""

    name = "nash-approx"

    def __init__(self, alpha: float = 1 / 3):
        self.alpha = alpha

    def act(self, card: int, history: List[int], rng) -> int:
        legal = _legal_actions(history)
        facing_bet = history and history[-1] == BET

        if facing_bet:
            if card == CARD_K:
                return CALL
            if card == CARD_J:
                return FOLD
            return CALL if rng.random() < 1 / 3 else FOLD

        # Acting first or after a check
        if card == CARD_K:
            return BET
        if card == CARD_Q:
            return PASS
        return BET if rng.random() < self.alpha else PASS


class StrategyTightValue:
    """Only bet/call with K. Pass/fold everything else."""

    name = "tight-value"

    def act(self, card: int, history: List[int], rng) -> int:
        facing_bet = history and history[-1] == BET
        if card == CARD_K:
            return CALL if facing_bet else BET
        return FOLD if facing_bet else PASS


class StrategyPotControl:
    """Bet K, pass Q, occasionally bluff J. Call only with K."""

    name = "pot-control"

    def act(self, card: int, history: List[int], rng) -> int:
        facing_bet = history and history[-1] == BET

        if facing_bet:
            if card == CARD_K:
                return CALL
            return FOLD

        if card == CARD_K:
            return BET
        if card == CARD_Q:
            return PASS
        # J: bluff occasionally
        return BET if rng.random() < 0.2 else PASS


class StrategyAlwaysPassQ:
    """Always pass with Q. Otherwise standard. Demonstrates suboptimal play."""

    name = "always-pass-q"

    def act(self, card: int, history: List[int], rng) -> int:
        facing_bet = history and history[-1] == BET

        if card == CARD_Q:
            return FOLD if facing_bet else PASS
        if card == CARD_K:
            return CALL if facing_bet else BET
        # J
        return FOLD if facing_bet else PASS


class StrategyRandomLegal:
    """Random legal action — exploration for diversity."""

    name = "random-legal"

    def act(self, card: int, history: List[int], rng) -> int:
        legal = _legal_actions(history)
        return int(rng.choice(legal))


STRATEGIES = [
    StrategyNashApprox,
    StrategyTightValue,
    StrategyPotControl,
    StrategyRandomLegal,
    StrategyAlwaysPassQ,
]
WEIGHTS = np.array([0.45, 0.25, 0.20, 0.08, 0.02])
WEIGHTS = WEIGHTS / WEIGHTS.sum()


# ─────────────────────────────────────────────────────────────────────────────
# Opponent pool (used to drive the game state — strategy plays against these)
# ─────────────────────────────────────────────────────────────────────────────

OPPONENT_CLASSES = [
    NashApprox,
    TightPassive,
    LooseAggressive,
    AlwaysPass,
    AlwaysBet,
]


# ─────────────────────────────────────────────────────────────────────────────
# Reasoning templates per (card, action, history)
# ─────────────────────────────────────────────────────────────────────────────


def make_reasoning(
    card: int, action: int, history: List[int], strategy_name: str
) -> str:
    card_str = CARD_LABEL[card]
    facing_bet = history and history[-1] == BET

    if strategy_name == "random-legal":
        return (
            f"My card is {card_str}. Trying {ACTION_LABEL[action]} to learn the opponent's response pattern.\n"
            f"{ACTION_LABEL[action]}"
        )

    if action == BET:
        if card == CARD_K:
            reason = f"I have the K — the strongest card. Betting for value to build the pot."
        elif card == CARD_J:
            reason = (
                f"I have the J — a weak card. Betting as a bluff to represent strength."
            )
        else:
            reason = f"I have the Q — middling card. Betting to apply pressure."
    elif action == CALL:
        if card == CARD_K:
            reason = f"I have the K and opponent bet. Calling for the value — I have the best hand."
        elif card == CARD_Q:
            reason = f"I have the Q and opponent bet. Calling occasionally to defend against bluffs."
        else:
            reason = f"I have the J and opponent bet. Calling here is risky but balances my range."
    elif action == FOLD:
        if card == CARD_J:
            reason = f"I have the J — a weak card. Folding to a bet to minimise losses."
        elif card == CARD_Q:
            reason = f"I have the Q and opponent bet. Folding because I likely lose at showdown."
        else:
            reason = f"Folding the K is unusual but signals deception."
    elif action == PASS:
        if card == CARD_Q:
            reason = f"I have the Q. Passing to avoid commitment without strong hand."
        elif card == CARD_J:
            reason = (
                f"I have the J. Passing to avoid investing in a likely-losing hand."
            )
        else:
            reason = (
                f"I have the K but passing to slow-play and induce a bet from opponent."
            )
    else:
        reason = f"My card is {card_str}. Choosing {ACTION_LABEL[action]}."

    return f"{reason}\n{ACTION_LABEL[action]}"


# ─────────────────────────────────────────────────────────────────────────────
# Episode generation — replicates KuhnPokerGame state transitions
# ─────────────────────────────────────────────────────────────────────────────


def simulate_episode(strategy, opponent, game: KuhnPokerGame, rng) -> List[dict]:
    """Run one episode; return list of (user_prompt, action) records for the strategy."""
    state = game.reset(opponent, rng)
    examples = []

    # The game already handles the case where opponent moves first via state["first_to_act"];
    # however in reset() the LLM always acts first on hand 0.  We just run the loop.
    done = False
    while not done:
        # Only the LLM's decisions become training examples
        if state["first_to_act"] == "llm" or state["history"]:
            # Build the user prompt the way the game would
            user_prompt = game.user_prompt(state)
            action = strategy.act(state["my_card"], state["history"], rng)

            # Coerce to legal
            legal = _legal_actions(state["history"])
            if action not in legal:
                action = legal[0]

            examples.append(
                {
                    "user": user_prompt,
                    "action": action,
                    "card": state["my_card"],
                    "history": list(state["history"]),
                    "strategy": strategy.name,
                    "opponent": opponent.name,
                }
            )

            state, _, _, done = game.step(action, state, rng)
        else:
            # Shouldn't happen with reset's default — skip just in case
            break

    return examples


def generate_dataset(episodes_per_opponent: int, seed: int) -> List[dict]:
    rng_select = np.random.default_rng(seed)
    rng_strat = np.random.default_rng(seed + 1)
    all_examples = []

    strat_instances = [S() for S in STRATEGIES]
    game = KuhnPokerGame(n_hands=6)

    for opp_cls in OPPONENT_CLASSES:
        for ep in range(episodes_per_opponent):
            strat_idx = rng_select.choice(len(STRATEGIES), p=WEIGHTS)
            strategy = strat_instances[strat_idx]
            opponent = opp_cls()
            ep_seed = seed + OPPONENT_CLASSES.index(opp_cls) * 10000 + ep
            ep_rng = np.random.default_rng(ep_seed)

            episode_examples = simulate_episode(strategy, opponent, game, ep_rng)
            all_examples.extend(episode_examples)

    return all_examples


# ─────────────────────────────────────────────────────────────────────────────
# Chat format
# ─────────────────────────────────────────────────────────────────────────────


def to_chat_jsonl(example: dict, system_prompt: str) -> dict:
    reasoning = make_reasoning(
        example["card"], example["action"], example["history"], example["strategy"]
    )
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["user"]},
            {"role": "assistant", "content": reasoning},
        ]
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stats
# ─────────────────────────────────────────────────────────────────────────────


def compute_stats(examples: List[dict], episodes_per_opponent: int) -> dict:
    total = len(examples)
    action_counts = {ACTION_LABEL[a]: 0 for a in (PASS, BET, CALL, FOLD)}
    for e in examples:
        action_counts[ACTION_LABEL[e["action"]]] += 1

    by_strategy = {}
    for e in examples:
        s = e["strategy"]
        by_strategy.setdefault(s, {"total": 0})
        by_strategy[s]["total"] += 1
        by_strategy[s].setdefault(ACTION_LABEL[e["action"]], 0)
        by_strategy[s][ACTION_LABEL[e["action"]]] += 1

    by_card = {CARD_LABEL[c]: {"total": 0} for c in (CARD_J, CARD_Q, CARD_K)}
    for e in examples:
        c = CARD_LABEL[e["card"]]
        by_card[c]["total"] += 1
        a = ACTION_LABEL[e["action"]]
        by_card[c].setdefault(a, 0)
        by_card[c][a] += 1

    return {
        "total_examples": total,
        "n_opponents": len(OPPONENT_CLASSES),
        "episodes_per_opponent": episodes_per_opponent,
        "total_episodes": episodes_per_opponent * len(OPPONENT_CLASSES),
        "action_distribution": action_counts,
        "by_strategy": by_strategy,
        "by_card": by_card,
        "strategy_weights": {
            STRATEGIES[i].__name__: round(float(WEIGHTS[i]), 4)
            for i in range(len(STRATEGIES))
        },
    }


def print_stats(stats: dict):
    print()
    print(f"  Total examples : {stats['total_examples']}")
    print(f"  Opponents      : {stats['n_opponents']}")
    print(f"  Episodes/opp   : {stats['episodes_per_opponent']}")
    print()
    print(f"  Action distribution:")
    for a, n in stats["action_distribution"].items():
        pct = 100 * n / stats["total_examples"] if stats["total_examples"] else 0
        print(f"    {a:<6}: {n:>5}  ({pct:.1f}%)")
    print()
    print(f"  Per-card action breakdown:")
    for c, d in stats["by_card"].items():
        total = d["total"]
        if not total:
            continue
        parts = []
        for a in ("PASS", "BET", "CALL", "FOLD"):
            n = d.get(a, 0)
            if n:
                parts.append(f"{a}={100 * n / total:.0f}%")
        print(f"    {c}: total={total:>5}  {' '.join(parts)}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    ap = argparse.ArgumentParser(
        description="Generate SFT warm-start data for Kuhn Poker"
    )
    ap.add_argument(
        "--episodes-per-opponent",
        type=int,
        default=200,
        help="Episodes per opponent (default 200)",
    )
    ap.add_argument(
        "--output-dir",
        default="sepo_sft_data_kuhn",
        help="Output directory (default: sepo_sft_data_kuhn/)",
    )
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sample examples without writing files",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = Path(args.output_dir)

    print("=" * 60)
    print("  SEPO SFT Warm Start — Kuhn Poker Data Generation")
    print("=" * 60)
    print(f"  Episodes/opponent : {args.episodes_per_opponent}")
    print(f"  Opponents         : {[c.name for c in [O() for O in OPPONENT_CLASSES]]}")
    print(
        f"  Strategy weights  : {dict(zip([S.__name__ for S in STRATEGIES], WEIGHTS.round(3)))}"
    )
    print()

    print("Generating episodes...")
    raw_examples = generate_dataset(args.episodes_per_opponent, seed=args.seed)

    stats = compute_stats(raw_examples, args.episodes_per_opponent)
    print_stats(stats)

    game = KuhnPokerGame(n_hands=6)
    system_prompt = game.system_prompt()

    if args.dry_run:
        print()
        print("── Sample examples (dry run, one per card) ──")
        for c in (CARD_J, CARD_Q, CARD_K):
            ex = next((e for e in raw_examples if e["card"] == c), None)
            if ex:
                print(f"\n[card={CARD_LABEL[c]}]")
                chat = to_chat_jsonl(ex, system_prompt)
                print(json.dumps(chat, indent=2))
        print("\nDry run complete. No files written.")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

        chat_examples = [to_chat_jsonl(e, system_prompt) for e in raw_examples]
        rng_shuffle = random.Random(args.seed)
        rng_shuffle.shuffle(chat_examples)

        n_train = int(len(chat_examples) * args.train_frac)
        train_set = chat_examples[:n_train]
        valid_set = chat_examples[n_train:]

        def _save(records, path):
            with path.open("w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
            print(f"  Saved {len(records):>6} examples → {path}")

        print()
        print(f"Writing dataset to {out_dir}/")
        _save(train_set, out_dir / "train.jsonl")
        _save(valid_set, out_dir / "valid.jsonl")

        stats["train_examples"] = len(train_set)
        stats["valid_examples"] = len(valid_set)
        stats["train_frac"] = args.train_frac
        stats["seed"] = args.seed

        (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
        print(f"  Saved stats       → {out_dir}/stats.json")

        print()
        print("Done. Next steps:")
        print(f"  1. Register KuhnPokerGame in games/__init__.py")
        print(f"  2. Add 'kuhn' to GAME_REGISTRY in grpo_sepo.py and eval_sepo.py")
        print(f"  3. Train SFT with --data-dir {out_dir}")
