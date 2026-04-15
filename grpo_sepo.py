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
    Run one full episode.
    Returns (Episode, (input_ids_list, gen_ids_list, old_log_probs_list)).
    old_log_probs are computed under the current policy at generation time (no_grad).
    """
    rng = np.random.default_rng(seed)
    state = game.reset(opponent, rng)

    all_input_ids  = []
    all_gen_ids    = []
    all_old_lps    = []   # log probs at generation time, for importance ratio
    actions, opp_actions = [], []
    payoffs, opp_payoffs = [], []

    done = False
    while not done:
        user_msg = game.user_prompt(state)
        messages = [
            {"role": "system", "content": game.system_prompt()},
            {"role": "user",   "content": user_msg},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoding = tokenizer(text, return_tensors="pt").to(device)
        encoding["token_type_ids"] = torch.zeros_like(encoding["input_ids"])

        out = model.generate(
            **encoding,
            max_new_tokens=8,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_ids = out[0, encoding["input_ids"].shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # Compute old log prob for the generated tokens (no_grad, generation-time policy)
        full_ids = torch.cat([encoding["input_ids"][0], gen_ids], dim=0).unsqueeze(0)
        tti = torch.zeros_like(full_ids)
        logits = model(full_ids, token_type_ids=tti).logits[0]
        n_inp = encoding["input_ids"].shape[1]
        n_gen = gen_ids.shape[0]
        pred_lp = F.log_softmax(logits[n_inp - 1: n_inp - 1 + n_gen], dim=-1)
        old_lp = pred_lp[torch.arange(n_gen), gen_ids].sum().cpu()

        action = game.parse_action(gen_text)
        if action is None:
            action = game.fallback_action

        state, pay, opp_pay, done = game.step(action, state, rng)

        all_input_ids.append(encoding["input_ids"])
        all_gen_ids.append(gen_ids)
        all_old_lps.append(old_lp)
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
    return episode, (all_input_ids, all_gen_ids, all_old_lps)


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
        outputs = model(full_ids, token_type_ids=torch.zeros_like(full_ids))
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
    clip_eps: float,
    seed_offset: int,
):
    """
    One GRPO step.

    Groups: each group = one opponent from train pool.
    Rollouts: G independent episodes per group.
    Reward: SEPO objective computed from all episodes in the rollout
            (train pool utility + exploitability vs exploiter pool + collusion vs collusive pool).
    """
    all_pg_losses  = []  # clipped surrogate per token-step
    all_kl_terms   = []  # KL per token-step
    all_advantages = []
    step_metrics   = []

    for g_idx in range(n_groups):
        rollout_rewards = []
        rollout_data    = []

        for r_idx in range(n_rollouts):
            seed_base = seed_offset + g_idx * 1000 + r_idx * 100
            rollout_episodes = []
            rollout_ids = []

            pools = {
                "train":     game.train_pool,
                "exploiter": game.exploiter_pool,
                "collusive": game.collusive_pool,
            }
            for pool_name, opponents in pools.items():
                for opp in opponents:
                    ep, (inp_ids, gen_ids, old_lps) = run_episode(
                        model, tokenizer, game, opp, pool_name,
                        seed=seed_base + hash(opp.name) % 97,
                        device=device, temperature=temperature,
                    )
                    rollout_episodes.append(ep)
                    if pool_name == "train":
                        rollout_ids.append((inp_ids, gen_ids, old_lps))

            reward, metrics = sepo_reward(rollout_episodes, game, lambda_e, lambda_c, lambda_x)
            rollout_rewards.append(reward)
            rollout_data.append((rollout_ids, metrics))

        # GRPO advantage normalisation within this group
        rewards = np.array(rollout_rewards, dtype=np.float32)
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8) if rewards.std() > 1e-8 else np.zeros_like(rewards)

        for r_idx, (rollout_ids, metrics) in enumerate(rollout_data):
            A = float(adv[r_idx])
            for inp_ids, gen_ids, old_lps in rollout_ids:
                new_lps = recompute_log_probs(model, inp_ids, gen_ids)
                with torch.no_grad():
                    ref_lps = recompute_log_probs(ref_model, inp_ids, gen_ids)
                for new_lp, old_lp, ref_lp in zip(new_lps, old_lps, ref_lps):
                    old_lp = old_lp.to(new_lp.device)
                    ratio = torch.exp(new_lp - old_lp.detach())
                    clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
                    pg = -torch.min(ratio * A, clipped * A)
                    kl = (new_lp - ref_lp.detach()).clamp(min=0)
                    all_pg_losses.append(pg)
                    all_kl_terms.append(kl)
                    all_advantages.append(A)

        step_metrics.append(rollout_data[-1][1])

    if not all_pg_losses:
        return None, {}

    pg_loss = torch.stack(all_pg_losses).mean()
    kl      = torch.stack(all_kl_terms).mean()
    loss    = pg_loss + beta * kl

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
    # args.model may be a PEFT adapter repo (LoRA only, no base weights).
    # If so, load the base model separately and merge the SFT adapter in.
    from peft import PeftModel, get_peft_model, LoraConfig, TaskType
    from pathlib import Path as _Path

    is_peft = (_Path(args.model) / "adapter_config.json").exists() or (
        # HF hub adapter repos always have adapter_config.json at root
        not (_Path(args.model) / "config.json").exists()
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(dtype=torch.bfloat16, device_map="auto")

    def load_merged(model_id, base_model_id, kwargs):
        """Load base model + fuse SFT LoRA adapter → plain HF model."""
        print(f"  Loading base: {base_model_id}")
        base = AutoModelForCausalLM.from_pretrained(base_model_id, **kwargs)
        print(f"  Applying SFT adapter: {model_id}")
        peft_m = PeftModel.from_pretrained(base, model_id, autocast_adapter_dtype=False)
        return peft_m.merge_and_unload()

    if is_peft:
        print(f"Detected PEFT adapter: {args.model}  (base: {args.base_model})")
        merged = load_merged(args.model, args.base_model, load_kwargs)
    else:
        print(f"Loading full model: {args.model}")
        merged = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)

    if args.lora:
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
        )
        model = get_peft_model(merged, lora_cfg, autocast_adapter_dtype=False)
        model.print_trainable_parameters()
    else:
        model = merged

    model.train()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # Reference model — frozen SFT checkpoint (same merge, different copy)
    print("Loading reference model (frozen)...")
    ref_kwargs = dict(dtype=torch.bfloat16, device_map="auto")
    if args.ref_4bit:
        from transformers import BitsAndBytesConfig
        ref_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        del ref_kwargs["dtype"]  # incompatible with 4bit quant
    if is_peft:
        ref_model = load_merged(args.model, args.base_model, ref_kwargs)
    else:
        ref_model = AutoModelForCausalLM.from_pretrained(args.model, **ref_kwargs)
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
            clip_eps=args.clip_eps,
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
    p.add_argument("--model", required=True, help="HF model path or repo (SFT checkpoint; full model or PEFT adapter)")
    p.add_argument("--base-model", default="google/gemma-3-4b-it",
                   help="Base model to load before applying PEFT adapter (only used if --model is a LoRA adapter repo)")
    p.add_argument("--output-dir", default="grpo_output")
    p.add_argument("--lora", action="store_true", help="Use LoRA for GRPO policy (lower VRAM)")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--ref-4bit", action="store_true", help="Load reference model in 4-bit (saves VRAM)")

    # Game
    p.add_argument("--game", default="ipd", choices=list(GAME_REGISTRY.keys()),
                   help="Game environment to train on")

    # SEPO objective weights
    # Lambda values from paper (sepo_gtbench_ipd_results.md)
    p.add_argument("--lambda-e", type=float, default=3.6,  help="Exploitability penalty weight")
    p.add_argument("--lambda-c", type=float, default=3.2,  help="Collusion penalty weight")
    p.add_argument("--lambda-x", type=float, default=2.4,  help="Externality penalty weight")

    # GRPO hyperparameters
    p.add_argument("--iters",        type=int,   default=500)
    p.add_argument("--n-groups",     type=int,   default=2,   help="Groups per GRPO step (one opponent each)")
    p.add_argument("--n-rollouts",   type=int,   default=16,  help="Rollouts per group (G) — more = better advantage estimates")
    p.add_argument("--temperature",  type=float, default=0.8, help="Sampling temperature")
    p.add_argument("--lr",           type=float, default=1e-5)
    p.add_argument("--beta",         type=float, default=0.01, help="KL penalty weight")
    p.add_argument("--clip-eps",     type=float, default=0.2,  help="PPO-style clip epsilon (DeepSeek-R1 default)")
    p.add_argument("--log-every",    type=int,   default=10)
    p.add_argument("--save-every",   type=int,   default=100)

    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
