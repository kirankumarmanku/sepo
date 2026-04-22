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
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import transformers
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


# ── Action stopping criteria ──────────────────────────────────────────────────

class ActionStoppingCriteria(transformers.StoppingCriteria):
    """Stop generation as soon as COOPERATE or DEFECT appears after any thinking block."""
    def __init__(self, tokenizer, input_len: int):
        self.tokenizer = tokenizer
        self.input_len = input_len

    def __call__(self, input_ids, scores, **kwargs):
        generated = self.tokenizer.decode(
            input_ids[0, self.input_len:], skip_special_tokens=True
        )
        if "<think>" in generated and "</think>" not in generated:
            return False
        up = generated.upper()
        return "COOPERATE" in up or "DEFECT" in up or "SILENT" in up or "TESTIFY" in up


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
    max_new_tokens: int = 512,
    use_token_type_ids: bool = False,
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
        if use_token_type_ids:
            encoding["token_type_ids"] = torch.zeros_like(encoding["input_ids"])

        stopping = transformers.StoppingCriteriaList([
            ActionStoppingCriteria(tokenizer, encoding["input_ids"].shape[1])
        ])
        out = model.generate(
            **encoding,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping,
        )
        gen_ids = out[0, encoding["input_ids"].shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # Compute old log prob for the generated tokens (no_grad, generation-time policy)
        full_ids = torch.cat([encoding["input_ids"][0], gen_ids], dim=0).unsqueeze(0)
        fwd_kwargs = {}
        if use_token_type_ids:
            fwd_kwargs["token_type_ids"] = torch.zeros_like(full_ids)
        logits = model(full_ids, **fwd_kwargs).logits[0]
        n_inp = encoding["input_ids"].shape[1]
        n_gen = gen_ids.shape[0]
        pred_lp = F.log_softmax(logits[n_inp - 1: n_inp - 1 + n_gen], dim=-1)
        old_lp = pred_lp[torch.arange(n_gen), gen_ids].sum().cpu()

        action = game.parse_action(gen_text)
        if action is None:
            print(f"      [PARSE FAIL] full response:\n{gen_text}\n---", flush=True)
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
    model, input_ids_list: List[torch.Tensor], gen_ids_list: List[torch.Tensor],
    use_token_type_ids: bool = False,
) -> List[torch.Tensor]:
    """
    Recompute log probs of generated tokens WITH gradients.
    One log_prob scalar per step (sum over generated tokens in that step).
    """
    log_probs = []
    for input_ids, gen_ids in zip(input_ids_list, gen_ids_list):
        full_ids = torch.cat([input_ids[0], gen_ids], dim=0).unsqueeze(0)
        fwd_kwargs = {}
        if use_token_type_ids:
            fwd_kwargs["token_type_ids"] = torch.zeros_like(full_ids)
        outputs = model(full_ids, **fwd_kwargs)
        logits = outputs.logits[0]  # [seq_len, vocab]

        # Logits at positions predicting the generated tokens
        n_input = input_ids.shape[1]
        n_gen   = gen_ids.shape[0]
        pred_logits = logits[n_input - 1 : n_input - 1 + n_gen]  # [n_gen, vocab]

        # Chunked log_softmax: Gemma 4 vocab=262144, n_gen up to 512 → 268 MB per
        # log_softmax output. Process in chunks of 32 to stay within 24 GB VRAM.
        _CHUNK = 32
        selected = []
        for _i in range(0, n_gen, _CHUNK):
            _lp = F.log_softmax(pred_logits[_i : _i + _CHUNK], dim=-1)
            selected.append(_lp[torch.arange(_lp.shape[0]), gen_ids[_i : _i + _CHUNK]])
        step_lp = torch.cat(selected).sum()
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
    n_rollouts: int,
    temperature: float,
    lambda_e: float,
    lambda_c: float,
    lambda_x: float,
    beta: float,
    clip_eps: float,
    seed_offset: int,
    max_new_tokens: int = 512,
    use_token_type_ids: bool = False,
):
    """
    One GRPO step — per-round advantage normalisation.

    Group = one train-pool opponent × n_rollouts episodes.
    Advantage is computed PER ROUND across rollouts, not per episode.

    reward_t_r = payoff_t_r - SEPO_penalty_r
    where SEPO_penalty_r = λe·e + λc·c + λx·x  (episode-level, shared across rounds)

    Normalising per round means even 1 defect out of 16 rollouts at round t
    produces a non-zero advantage signal for that decision.
    """
    # Pre-count terms so each backward() call is correctly normalised.
    # Only one computation graph lives in memory at a time instead of n_total.
    n_total = len(game.train_pool) * n_rollouts * game.n_steps
    pg_loss_accum = 0.0
    kl_accum      = 0.0
    step_metrics  = []

    for g_idx, train_opp in enumerate(game.train_pool):
        # Collect n_rollouts episodes for this opponent
        episodes = []  # (ep, inp_ids, gen_ids, old_lps, sepo_penalty, metrics)
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] rollout opp={train_opp.name} ({g_idx+1}/{len(game.train_pool)})", flush=True)

        # Shared aux episodes for SEPO — run ONCE per opponent group, not per rollout.
        # If train_opp is already in the exploiter/collusive pool, reuse a train
        # episode (pool tag swapped) instead of running a duplicate game.
        exploit_names  = {o.name for o in game.exploiter_pool}
        collusive_names = {o.name for o in game.collusive_pool}
        _seed_aux = seed_offset + g_idx * 1000 + 999
        if train_opp.name in exploit_names:
            shared_exploit_eps = None   # will reuse first train ep below
        else:
            shared_exploit_eps = []
            for opp in game.exploiter_pool:
                ep, _ = run_episode(model, tokenizer, game, opp, "exploiter",
                                    seed=_seed_aux, device=device, temperature=temperature,
                                    max_new_tokens=max_new_tokens, use_token_type_ids=use_token_type_ids)
                shared_exploit_eps.append(ep)

        if train_opp.name in collusive_names:
            shared_collusive_eps = None  # will reuse first train ep below
        else:
            shared_collusive_eps = []
            for opp in game.collusive_pool:
                ep, _ = run_episode(model, tokenizer, game, opp, "collusive",
                                    seed=_seed_aux + 1, device=device, temperature=temperature,
                                    max_new_tokens=max_new_tokens, use_token_type_ids=use_token_type_ids)
                shared_collusive_eps.append(ep)

        first_train_ep = None  # set on first rollout for reuse

        for r_idx in range(n_rollouts):
            seed_base = seed_offset + g_idx * 1000 + r_idx * 100

            ep_train, (inp_ids, gen_ids, old_lps) = run_episode(
                model, tokenizer, game, train_opp, "train",
                seed=seed_base, device=device, temperature=temperature,
                max_new_tokens=max_new_tokens, use_token_type_ids=use_token_type_ids,
            )
            if first_train_ep is None:
                first_train_ep = ep_train

            # Build aux list — reuse train ep (pool-tag swapped) when possible
            from dataclasses import replace as _replace
            aux = [ep_train]
            aux += shared_exploit_eps if shared_exploit_eps is not None else \
                   [_replace(first_train_ep, pool="exploiter")]
            aux += shared_collusive_eps if shared_collusive_eps is not None else \
                   [_replace(first_train_ep, pool="collusive")]

            _, metrics = sepo_reward(aux, game, lambda_e, lambda_c, lambda_x)
            sepo_penalty = (lambda_e * metrics["exploitability"]
                          + lambda_c * metrics["collusion"]
                          + lambda_x * metrics["externality"])
            episodes.append((ep_train, inp_ids, gen_ids, old_lps, sepo_penalty, metrics))
            actions_str = "".join("C" if a == 0 else "D" for a in ep_train.actions)
            opp_str     = "".join("C" if a == 0 else "D" for a in ep_train.opp_actions)
            print(f"    [{datetime.now().strftime('%H:%M:%S')}] r{r_idx+1:02d} llm={actions_str} opp={opp_str} u={sum(ep_train.payoffs):.1f} pen={sepo_penalty:.3f}", flush=True)

        # Per-round advantage: normalise across rollouts at each round t
        n_steps = len(episodes[0][0].payoffs)
        for t in range(n_steps):
            round_rewards = np.array(
                [ep.payoffs[t] - sepo_pen for ep, _, _, _, sepo_pen, _ in episodes],
                dtype=np.float32,
            )
            adv = ((round_rewards - round_rewards.mean()) / (round_rewards.std() + 1e-8)
                   if round_rewards.std() > 1e-8 else np.zeros_like(round_rewards))

            for r_idx, (_, inp_ids, gen_ids, old_lps, _, _) in enumerate(episodes):
                A = float(adv[r_idx])
                new_lps = recompute_log_probs(model, [inp_ids[t]], [gen_ids[t]], use_token_type_ids)
                with torch.no_grad():
                    if ref_model is None:
                        # LoRA mode: disable adapters to get base model log probs
                        with model.disable_adapter():
                            ref_lps = recompute_log_probs(model, [inp_ids[t]], [gen_ids[t]], use_token_type_ids)
                    else:
                        ref_lps = recompute_log_probs(ref_model, [inp_ids[t]], [gen_ids[t]], use_token_type_ids)
                for new_lp, ref_lp in zip(new_lps, ref_lps):
                    old_lp = old_lps[t].to(new_lp.device)
                    ratio   = torch.exp(new_lp - old_lp.detach())
                    clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
                    pg  = -torch.min(ratio * A, clipped * A)
                    kl  = (new_lp - ref_lp.detach()).clamp(min=0)
                    pg_loss_accum += pg.detach().item()
                    kl_accum      += kl.detach().item()
                    # Backward immediately — one graph at a time, no accumulation
                    ((pg + beta * kl) / n_total).backward()

        step_metrics.append(episodes[-1][-1])

    if not step_metrics:
        return None, {}

    avg_pg = pg_loss_accum / n_total
    avg_kl = kl_accum / n_total
    loss_val = avg_pg + beta * avg_kl

    avg_metrics = {k: float(np.mean([m[k] for m in step_metrics])) for k in step_metrics[0]}
    avg_metrics["kl"]      = avg_kl
    avg_metrics["pg_loss"] = avg_pg

    return loss_val, avg_metrics


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

    # PEFT adapter if: local dir has adapter_config.json, OR --base-model was explicitly provided
    _local = _Path(args.model)
    is_peft = (_local.exists() and (_local / "adapter_config.json").exists()) or (args.base_model is not None)

    tokenizer_id = args.base_model if is_peft else args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
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
        # Gemma 4 wraps projections in Gemma4ClippableLinear — need inner .linear
        # Gemma 3 and most models use plain q_proj/v_proj
        named = {n for n, _ in merged.named_modules()}
        if "model.layers.0.self_attn.q_proj.linear" in named:
            lora_targets = ["q_proj.linear", "v_proj.linear"]
        else:
            lora_targets = ["q_proj", "v_proj"]
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            target_modules=lora_targets,
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

    # Reference model
    # When using LoRA: base weights ARE the reference — disable adapters in-place
    # instead of loading a second copy (saves ~10GB VRAM on 24GB GPU).
    # When full fine-tune: load a separate frozen copy.
    if args.lora:
        print("LoRA mode: using base model (adapters disabled) as reference — no second copy loaded.")
        ref_model = None  # signal to grpo_step to use disable_adapter()
    else:
        print("Loading reference model (frozen)...")
        ref_kwargs = dict(dtype=torch.bfloat16, device_map="auto")
        if args.ref_4bit:
            from transformers import BitsAndBytesConfig
            ref_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            del ref_kwargs["dtype"]
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
            n_rollouts=args.n_rollouts,
            temperature=args.temperature,
            lambda_e=args.lambda_e,
            lambda_c=args.lambda_c,
            lambda_x=args.lambda_x,
            beta=args.beta,
            clip_eps=args.clip_eps,
            seed_offset=step * 10000,
            max_new_tokens=args.max_new_tokens,
            use_token_type_ids=args.token_type_ids,
        )

        if loss is None:
            continue

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()

        log_entry = {"step": step, "loss": float(loss), **metrics}
        log.append(log_entry)

        if step % args.log_every == 0:
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
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
    p.add_argument("--base-model", default=None,
                   help="Base model to load before applying PEFT adapter. If set, --model is treated as a LoRA adapter repo.")
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
    p.add_argument("--iters",           type=int,   default=500)
    p.add_argument("--n-rollouts",      type=int,   default=8,   help="Rollouts per train-pool opponent per step")
    p.add_argument("--temperature",     type=float, default=0.8, help="Sampling temperature")
    p.add_argument("--max-new-tokens",  type=int,   default=1024, help="Max tokens per generation (use 1024+ for thinking models)")
    p.add_argument("--token-type-ids",  action="store_true",     help="Pass token_type_ids=zeros (required for Gemma 3, not Gemma 4)")
    p.add_argument("--lr",           type=float, default=1e-5)
    p.add_argument("--beta",         type=float, default=0.01, help="KL penalty weight")
    p.add_argument("--clip-eps",     type=float, default=0.2,  help="PPO-style clip epsilon (DeepSeek-R1 default)")
    p.add_argument("--log-every",    type=int,   default=10)
    p.add_argument("--save-every",   type=int,   default=100)

    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
