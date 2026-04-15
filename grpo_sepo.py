"""
grpo_sepo.py — Stage 2: GRPO with SEPO Objective
=================================================
Pluggable GRPO training that works with any Game + any HuggingFace model.

SEPO objective:
  J(π) = u(π) - λe·e(π) - λc·c(π) - λx·x(π)

GRPO algorithm (per step):
  1. Sample episodes from diverse opponent pools
  2. Generate G rollouts per episode using the policy model
  3. Compute SEPO reward for each rollout
  4. Advantage = (r - mean(r)) / std(r)  within group
  5. Loss = -mean(A · log π(a|s)) + β · KL(π || π_ref)
  6. Gradient update

Usage:
  # Gemma 3 4B SFT checkpoint, IPD game
  python grpo_sepo.py --model kartiinx/gemma-3-4b-sepo-sft --game ipd

  # Gemma 4 e2b, IPD game
  python grpo_sepo.py --model kartiinx/gemma-4-e2b-sepo-sft --game ipd

  # Custom SEPO weights
  python grpo_sepo.py --model <path> --game ipd --lambda-e 0.5 --lambda-c 2.0 --lambda-x 1.0

  # LoRA mode (lower VRAM)
  python grpo_sepo.py --model <path> --game ipd --lora --lora-rank 16
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from games import Game, Episode
from games.ipd import IPDGame

# ── Game registry ─────────────────────────────────────────────────────────────
# Add new games here as they are implemented.
GAME_REGISTRY: Dict[str, Game] = {
    "ipd": IPDGame(n_rounds=8),
    # "resource":    ResourceGame(),    # TODO
    # "auction":     AuctionGame(),     # TODO
    # "negotiation": NegotiationGame(), # TODO
    # "pressure":    PressureGame(),    # TODO
}


# ── Episode runner ─────────────────────────────────────────────────────────────

@torch.no_grad()
def run_episode(
    model,
    tokenizer,
    game: Game,
    opponent,
    pool: str,
    seed: int,
    device,
    temperature: float = 0.8,
) -> Tuple[Episode, List[torch.Tensor]]:
    """
    Run one full episode. Returns (Episode, list_of_per_step_log_probs).
    log_probs are computed with no_grad — recomputed with grad during loss step.
    """
    rng = np.random.default_rng(seed)
    state = game.reset(opponent, rng)

    all_input_ids = []      # store for log prob recomputation
    all_gen_ids   = []
    actions, opp_actions = [], []
    payoffs, opp_payoffs = [], []

    done = False
    while not done:
        user_msg = game.user_prompt(state)
        messages = [
            {"role": "system", "content": game.system_prompt()},
            {"role": "user",   "content": user_msg},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(device)

        out = model.generate(
            input_ids,
            max_new_tokens=8,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_ids = out[0, input_ids.shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        action = game.parse_action(gen_text)
        if action is None:
            action = game.fallback_action

        state, pay, opp_pay, done = game.step(action, state, rng)

        all_input_ids.append(input_ids)
        all_gen_ids.append(gen_ids)
        actions.append(action)
        opp_actions.append(state["h_opp"][-1])
        payoffs.append(pay)
        opp_payoffs.append(opp_pay)

    episode = Episode(
        opponent_name=opponent.name,
        pool=pool,
        actions=actions,
        opp_actions=opp_actions,
        payoffs=payoffs,
        opp_payoffs=opp_payoffs,
    )
    return episode, (all_input_ids, all_gen_ids)


def recompute_log_probs(
    model, input_ids_list: List[torch.Tensor], gen_ids_list: List[torch.Tensor]
) -> List[torch.Tensor]:
    """
    Recompute log probs of generated tokens WITH gradients.
    One log_prob scalar per step (sum over generated tokens in that step).
    """
    log_probs = []
    for input_ids, gen_ids in zip(input_ids_list, gen_ids_list):
        full_ids = torch.cat([input_ids[0], gen_ids], dim=0).unsqueeze(0)
        outputs = model(full_ids)
        logits = outputs.logits[0]  # [seq_len, vocab]

        # Logits at positions predicting the generated tokens
        n_input = input_ids.shape[1]
        n_gen   = gen_ids.shape[0]
        pred_logits = logits[n_input - 1 : n_input - 1 + n_gen]  # [n_gen, vocab]

        lp = F.log_softmax(pred_logits, dim=-1)
        step_lp = lp[torch.arange(n_gen), gen_ids].sum()
        log_probs.append(step_lp)
    return log_probs


# ── SEPO reward ────────────────────────────────────────────────────────────────

def sepo_reward(episodes: List[Episode], game: Game, lambda_e: float, lambda_c: float, lambda_x: float) -> float:
    metrics = game.sepo_metrics(episodes)
    r = (metrics["utility"]
         - lambda_e * metrics["exploitability"]
         - lambda_c * metrics["collusion"]
         - lambda_x * metrics["externality"])
    return r, metrics


# ── GRPO training step ────────────────────────────────────────────────────────

def grpo_step(
    model,
    ref_model,
    tokenizer,
    game: Game,
    device,
    n_groups: int,
    n_rollouts: int,
    temperature: float,
    lambda_e: float,
    lambda_c: float,
    lambda_x: float,
    beta: float,
    seed_offset: int,
):
    """
    One GRPO step.

    Groups: each group = one opponent from train pool.
    Rollouts: G independent episodes per group.
    Reward: SEPO objective computed from all episodes in the rollout
            (train pool utility + exploitability vs exploiter pool + collusion vs collusive pool).
    """
    all_log_probs  = []  # List[Tensor] — one per (group, rollout, step)
    all_ref_lps    = []
    all_advantages = []
    step_metrics   = []

    for g_idx in range(n_groups):
        rollout_rewards = []
        rollout_data    = []   # (episodes, input_ids_list, gen_ids_list) per rollout

        for r_idx in range(n_rollouts):
            seed_base = seed_offset + g_idx * 1000 + r_idx * 100
            rollout_episodes = []
            rollout_ids = []

            # Play vs all pools to compute full SEPO reward
            pools = {
                "train":    game.train_pool,
                "exploiter": game.exploiter_pool,
                "collusive": game.collusive_pool,
            }
            for pool_name, opponents in pools.items():
                for opp in opponents:
                    ep, (inp_ids, gen_ids) = run_episode(
                        model, tokenizer, game, opp, pool_name,
                        seed=seed_base + hash(opp.name) % 97,
                        device=device, temperature=temperature,
                    )
                    rollout_episodes.append(ep)
                    if pool_name == "train":   # only train pool episodes enter gradient
                        rollout_ids.append((inp_ids, gen_ids))

            reward, metrics = sepo_reward(rollout_episodes, game, lambda_e, lambda_c, lambda_x)
            rollout_rewards.append(reward)
            rollout_data.append((rollout_ids, metrics))

        # GRPO advantage normalization within this group
        rewards = np.array(rollout_rewards, dtype=np.float32)
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8) if rewards.std() > 1e-8 else np.zeros_like(rewards)

        for r_idx, (rollout_ids, metrics) in enumerate(rollout_data):
            A = float(adv[r_idx])
            for inp_ids, gen_ids in rollout_ids:
                # Recompute with grad
                lps = recompute_log_probs(model, inp_ids, gen_ids)
                with torch.no_grad():
                    ref_lps = recompute_log_probs(ref_model, inp_ids, gen_ids)
                for lp, ref_lp in zip(lps, ref_lps):
                    all_log_probs.append(lp)
                    all_ref_lps.append(ref_lp.detach())
                    all_advantages.append(A)

        step_metrics.append(rollout_data[-1][1])  # last rollout's metrics for logging

    if not all_log_probs:
        return None, {}

    log_probs_t  = torch.stack(all_log_probs)
    ref_lps_t    = torch.stack(all_ref_lps)
    advantages_t = torch.tensor(all_advantages, device=device, dtype=log_probs_t.dtype)

    # Policy gradient loss
    pg_loss = -(advantages_t * log_probs_t).mean()

    # KL penalty: KL(π || π_ref) ≈ mean(log π - log π_ref)
    kl = (log_probs_t - ref_lps_t).mean()
    loss = pg_loss + beta * kl

    avg_metrics = {k: float(np.mean([m[k] for m in step_metrics])) for k in step_metrics[0]}
    avg_metrics["kl"] = float(kl.detach())
    avg_metrics["pg_loss"] = float(pg_loss.detach())

    return loss, avg_metrics


# ── Main training loop ────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    game = GAME_REGISTRY[args.game]
    print(f"Game: {game.name}")

    # Load tokenizer + model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto")

    if args.lora:
        from peft import get_peft_model, LoraConfig, TaskType
        base_model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
        )
        model = get_peft_model(base_model, lora_cfg)
        model.print_trainable_parameters()
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)

    model.train()

    # Reference model — frozen
    print("Loading reference model (frozen)...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_4bit=args.ref_4bit,
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log = []
    print(f"\nStarting GRPO training — {args.iters} steps")
    print(f"SEPO weights: λe={args.lambda_e}  λc={args.lambda_c}  λx={args.lambda_x}\n")

    for step in range(args.iters):
        optimizer.zero_grad()

        loss, metrics = grpo_step(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            game=game,
            device=device,
            n_groups=args.n_groups,
            n_rollouts=args.n_rollouts,
            temperature=args.temperature,
            lambda_e=args.lambda_e,
            lambda_c=args.lambda_c,
            lambda_x=args.lambda_x,
            beta=args.beta,
            seed_offset=step * 10000,
        )

        if loss is None:
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()

        log_entry = {"step": step, "loss": float(loss), **metrics}
        log.append(log_entry)

        if step % args.log_every == 0:
            print(
                f"Step {step:4d} | loss={float(loss):.4f} | "
                f"u={metrics.get('utility', 0):.3f} | "
                f"e={metrics.get('exploitability', 0):.3f} | "
                f"c={metrics.get('collusion', 0):.3f} | "
                f"x={metrics.get('externality', 0):.3f} | "
                f"kl={metrics.get('kl', 0):.4f}"
            )

        if step % args.save_every == 0 and step > 0:
            ckpt = output_dir / f"step_{step:04d}"
            model.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            with open(output_dir / "log.json", "w") as f:
                json.dump(log, f, indent=2)
            print(f"  → Saved checkpoint: {ckpt}")

    # Final save
    final = output_dir / "final"
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    with open(output_dir / "log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nDone. Final model: {final}")


def main():
    p = argparse.ArgumentParser(description="GRPO + SEPO Stage 2 Training")

    # Model
    p.add_argument("--model", required=True, help="HF model path or repo (SFT checkpoint)")
    p.add_argument("--output-dir", default="grpo_output")
    p.add_argument("--lora", action="store_true", help="Use LoRA (lower VRAM)")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--ref-4bit", action="store_true", help="Load reference model in 4-bit (saves VRAM)")

    # Game
    p.add_argument("--game", default="ipd", choices=list(GAME_REGISTRY.keys()),
                   help="Game environment to train on")

    # SEPO objective weights
    p.add_argument("--lambda-e", type=float, default=0.5,  help="Exploitability penalty weight")
    p.add_argument("--lambda-c", type=float, default=2.0,  help="Collusion penalty weight")
    p.add_argument("--lambda-x", type=float, default=1.0,  help="Externality penalty weight")

    # GRPO hyperparameters
    p.add_argument("--iters",        type=int,   default=500)
    p.add_argument("--n-groups",     type=int,   default=4,   help="Episodes per GRPO step")
    p.add_argument("--n-rollouts",   type=int,   default=4,   help="Rollouts per group (G)")
    p.add_argument("--temperature",  type=float, default=0.8, help="Sampling temperature")
    p.add_argument("--lr",           type=float, default=1e-5)
    p.add_argument("--beta",         type=float, default=0.01, help="KL penalty weight")
    p.add_argument("--log-every",    type=int,   default=10)
    p.add_argument("--save-every",   type=int,   default=100)

    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
