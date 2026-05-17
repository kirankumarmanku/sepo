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

from games import Episode, Game
from games.auction import AuctionGame
from games.ipd import IPDGame
from games.negotiation import NegotiationGame
from games.kuhn import KuhnPokerGame
from games.negotiation_gtbench import NegotiationGTBenchGame
# ── Game registry ─────────────────────────────────────────────────────────────
GAME_REGISTRY: Dict[str, Game] = {
    "ipd": IPDGame(n_rounds=8),
    "resource": ResourceGame(n_rounds=8),
    "auction": AuctionGame(n_rounds=6),
    "negotiation": NegotiationGame(n_rounds=4),
    "kuhn": KuhnPokerGame(n_hands=6),  # ← new
    "negotiation_gt":  NegotiationGTBenchGame(n_rounds=4),
}


# ── Action stopping criteria ──────────────────────────────────────────────────


class ActionStoppingCriteria(transformers.StoppingCriteria):
    """Stop generation when a valid action word appears on the last line."""

    def __init__(self, tokenizer, input_len: int, game):
        self.tokenizer = tokenizer
        self.input_len = input_len
        self.game = game

    def __call__(self, input_ids, scores, **kwargs):
        import re as _re

        generated = self.tokenizer.decode(
            input_ids[0, self.input_len :], skip_special_tokens=True
        )
        if "<think>" in generated and "</think>" not in generated:
            return False
        lines = [l.strip() for l in generated.split("\n") if l.strip()]
        if not lines:
            return False
        last = lines[-1].upper()
        return self.game.action_on_last_line(last)


# ── Constrained action decode ─────────────────────────────────────────────────


@torch.no_grad()
def forced_action_decode(
    model, tokenizer, messages, gen_text, game, device, use_token_type_ids
):
    """
    When parse_action fails, force a valid action by restricting the next token
    to only the first tokens of each valid action string.

    Appends the reasoning as an assistant turn, adds a short action-elicitation
    user turn, then masks all logits except valid action first-tokens.
    Always returns a valid action — no regex, no fallback needed.
    """
    vocab = game.action_vocab
    if not vocab:
        return game.fallback_action

    # Build token ID → action mapping using first token of each action string
    tok_to_action = {}
    for action_str, action_val in vocab.items():
        # Encode with a leading space (how sub-words appear mid-sentence)
        for prefix in (" " + action_str, action_str):
            ids = tokenizer.encode(prefix, add_special_tokens=False)
            if ids:
                tok_to_action[ids[0]] = action_val

    if not tok_to_action:
        return game.fallback_action

    action_list = "/".join(vocab.keys())
    elicit_messages = messages + [
        {"role": "assistant", "content": gen_text},
        {"role": "user", "content": f"State your final action ({action_list}):"},
    ]
    text = tokenizer.apply_chat_template(
        elicit_messages, tokenize=False, add_generation_prompt=True
    )
    enc = tokenizer(text, return_tensors="pt").to(device)
    if use_token_type_ids:
        enc["token_type_ids"] = torch.zeros_like(enc["input_ids"])

    outputs = model(**enc)
    logits = outputs.logits[0, -1]  # next-token logits

    # Mask all tokens except valid action first-tokens
    masked = torch.full_like(logits, float("-inf"))
    for tok_id, action_val in tok_to_action.items():
        masked[tok_id] = logits[tok_id]

    best_tok = masked.argmax().item()
    return tok_to_action.get(best_tok, game.fallback_action)


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
    # Switch to eval mode for generation — dropout + gradient checkpointing in
    # train mode causes repetitive garbage output during inference.
    was_training = model.training
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    rng = np.random.default_rng(seed)
    state = game.reset(opponent, rng)

    all_input_ids = []
    all_gen_ids = []
    all_old_lps = []  # log probs at generation time, for importance ratio
    actions, opp_actions = [], []
    payoffs, opp_payoffs = [], []

    done = False
    while not done:
        user_msg = game.user_prompt(state)
        messages = [
            {"role": "system", "content": game.system_prompt()},
            {"role": "user", "content": user_msg},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        encoding = tokenizer(text, return_tensors="pt").to(device)
        if use_token_type_ids:
            encoding["token_type_ids"] = torch.zeros_like(encoding["input_ids"])

        stopping = transformers.StoppingCriteriaList(
            [ActionStoppingCriteria(tokenizer, encoding["input_ids"].shape[1], game)]
        )
        out = model.generate(
            **encoding,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping,
        )
        gen_ids = out[0, encoding["input_ids"].shape[1] :]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        if getattr(run_episode, "_show_gen", False):
            print(f"        [gen] {repr(gen_text)}", flush=True)

        # Compute old log prob for the generated tokens (no_grad, generation-time policy)
        full_ids = torch.cat([encoding["input_ids"][0], gen_ids], dim=0).unsqueeze(0)
        fwd_kwargs = {}
        if use_token_type_ids:
            fwd_kwargs["token_type_ids"] = torch.zeros_like(full_ids)
        logits = model(full_ids, **fwd_kwargs).logits[0]
        n_inp = encoding["input_ids"].shape[1]
        n_gen = gen_ids.shape[0]
        pred_lp = F.log_softmax(logits[n_inp - 1 : n_inp - 1 + n_gen], dim=-1)
        old_lp = pred_lp[torch.arange(n_gen), gen_ids].sum().cpu()

        action = game.parse_action(gen_text)
        if action is None:
            print(f"      [PARSE FAIL] → constrained decode", flush=True)
            action = forced_action_decode(
                model, tokenizer, messages, gen_text, game, device, use_token_type_ids
            )

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

    if was_training:
        model.train()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

    return episode, (all_input_ids, all_gen_ids, all_old_lps)


def recompute_log_probs(
    model,
    input_ids_list: List[torch.Tensor],
    gen_ids_list: List[torch.Tensor],
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
        n_gen = gen_ids.shape[0]
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


def sepo_reward(
    episodes: List[Episode],
    game: Game,
    lambda_e: float,
    lambda_c: float,
    lambda_x: float,
) -> float:
    metrics = game.sepo_metrics(episodes)
    scale = 3.0 / game.max_payoff  # normalise utility/exploit to common IPD scale
    r = (
        metrics["utility"] * scale
        - lambda_e * metrics["exploitability"] * scale
        - lambda_c * metrics["collusion"]
        - lambda_x * metrics["externality"]
    )
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
    kl_accum = 0.0
    step_metrics = []

    for g_idx, train_opp in enumerate(game.train_pool):
        # Collect n_rollouts episodes for this opponent
        episodes = []  # (ep, inp_ids, gen_ids, old_lps, sepo_penalty, metrics)
        print(
            f"  [{datetime.now().strftime('%H:%M:%S')}] rollout opp={train_opp.name} ({g_idx + 1}/{len(game.train_pool)})",
            flush=True,
        )

        # Shared aux episodes for SEPO — run ONCE per opponent group, not per rollout.
        # Aux episodes for SEPO metrics — skipped when a cached penalty is provided
        # (caller refreshes the cache every sepo_eval_every steps).
        # When not cached: run once per opponent group, reusing train ep if possible.
        from dataclasses import replace as _replace

        exploit_names = {o.name for o in game.exploiter_pool}
        collusive_names = {o.name for o in game.collusive_pool}
        _seed_aux = seed_offset + g_idx * 1000 + 999

        if cached_sepo_penalty is None:
            if train_opp.name in exploit_names:
                shared_exploit_eps = None
            else:
                shared_exploit_eps = []
                for opp in game.exploiter_pool:
                    ep, _ = run_episode(
                        model,
                        tokenizer,
                        game,
                        opp,
                        "exploiter",
                        seed=_seed_aux,
                        device=device,
                        temperature=temperature,
                        max_new_tokens=max_new_tokens,
                        use_token_type_ids=use_token_type_ids,
                    )
                    shared_exploit_eps.append(ep)

            if train_opp.name in collusive_names:
                shared_collusive_eps = None
            else:
                shared_collusive_eps = []
                for opp in game.collusive_pool:
                    ep, _ = run_episode(
                        model,
                        tokenizer,
                        game,
                        opp,
                        "collusive",
                        seed=_seed_aux + 1,
                        device=device,
                        temperature=temperature,
                        max_new_tokens=max_new_tokens,
                        use_token_type_ids=use_token_type_ids,
                    )
                    shared_collusive_eps.append(ep)

        first_train_ep = None

        for r_idx in range(n_rollouts):
            seed_base = seed_offset + g_idx * 1000 + r_idx * 100

            ep_train, (inp_ids, gen_ids, old_lps) = run_episode(
                model,
                tokenizer,
                game,
                train_opp,
                "train",
                seed=seed_base,
                device=device,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                use_token_type_ids=use_token_type_ids,
            )

            if cached_sepo_penalty is not None:
                sepo_penalty = cached_sepo_penalty
                metrics = {
                    "exploitability": 0.0,
                    "collusion": 0.0,
                    "externality": 0.0,
                    "utility": float(sum(ep_train.payoffs)) / game.n_steps,
                }
            else:
                # Use first_train_ep as fixed reference so SEPO penalty is
                # constant across rollouts — penalty variance was creating
                # false advantage signal unrelated to action differences.
                aux = [first_train_ep]
                aux += (
                    shared_exploit_eps
                    if shared_exploit_eps is not None
                    else [_replace(first_train_ep, pool="exploiter")]
                )
                aux += (
                    shared_collusive_eps
                    if shared_collusive_eps is not None
                    else [_replace(first_train_ep, pool="collusive")]
                )
                _, metrics = sepo_reward(aux, game, lambda_e, lambda_c, lambda_x)
                sepo_penalty = (
                    lambda_e * metrics["exploitability"]
                    + lambda_c * metrics["collusion"]
                    + lambda_x * metrics["externality"]
                )
            episodes.append(
                (ep_train, inp_ids, gen_ids, old_lps, sepo_penalty, metrics)
            )
            actions_str = "".join(game.action_label(a) for a in ep_train.actions)
            opp_str     = "".join(game.action_label(a) for a in ep_train.opp_actions)
            exp_str     = "".join(game.action_label(a) for ep in exploit_eps_r for a in ep.actions)
            col_str     = "".join(game.action_label(a) for ep in collusive_eps_r for a in ep.actions)
            print(f"    [{datetime.now().strftime('%H:%M:%S')}] r{r_idx+1:02d} llm={actions_str} opp={opp_str} exp={exp_str} col={col_str} u={sum(ep_train.payoffs):.1f} pen={sepo_penalty:.3f}", flush=True)

        # Per-round advantage: normalise across rollouts at each round t
        n_steps = min(len(ep.payoffs) for ep, _, _, _, _, _ in episodes)
        for t in range(n_steps):
            round_rewards = np.array(
                [ep.payoffs[t] - sepo_pen for ep, _, _, _, sepo_pen, _ in episodes],
                dtype=np.float32,
            )
            adv = (
                (round_rewards - round_rewards.mean()) / (round_rewards.std() + 1e-8)
                if round_rewards.std() > 1e-8
                else np.zeros_like(round_rewards)
            )

            for r_idx, (_, inp_ids, gen_ids, old_lps, _, _) in enumerate(episodes):
                A = float(adv[r_idx])
                new_lps = recompute_log_probs(
                    model, [inp_ids[t]], [gen_ids[t]], use_token_type_ids
                )
                with torch.no_grad():
                    if ref_model is None:
                        # LoRA mode: disable adapters to get base model log probs
                        with model.disable_adapter():
                            ref_lps = recompute_log_probs(
                                model, [inp_ids[t]], [gen_ids[t]], use_token_type_ids
                            )
                    else:
                        ref_lps = recompute_log_probs(
                            ref_model, [inp_ids[t]], [gen_ids[t]], use_token_type_ids
                        )
                for new_lp, ref_lp in zip(new_lps, ref_lps):
                    old_lp = old_lps[t].to(new_lp.device)
                    ratio = torch.exp(new_lp - old_lp.detach())
                    clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
                    pg = -torch.min(ratio * A, clipped * A)
                    kl = (new_lp - ref_lp.detach()).clamp(min=0)
                    pg_loss_accum += pg.detach().item()
                    kl_accum += kl.detach().item()
                    # Backward immediately — one graph at a time, no accumulation
                    ((pg + beta * kl) / n_total).backward()

        step_metrics.append(episodes[-1][-1])

    if not step_metrics:
        return None, {}

    avg_pg = pg_loss_accum / n_total
    avg_kl = kl_accum / n_total
    loss_val = avg_pg + beta * avg_kl

    avg_metrics = {
        k: float(np.mean([m[k] for m in step_metrics])) for k in step_metrics[0]
    }
    avg_metrics["kl"] = avg_kl
    avg_metrics["pg_loss"] = avg_pg

    return loss_val, avg_metrics


# ── Main training loop ────────────────────────────────────────────────────────


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.game == "all":
        games = list(GAME_REGISTRY.values())
        print(f"Game: ALL ({', '.join(g.name for g in games)}) — joint multi-game GRPO")
    else:
        g = GAME_REGISTRY[args.game]
        if args.n_rounds != g.n_steps:
            g = type(g)(n_rounds=args.n_rounds)
        games = [g]
        print(f"Game: {g.name}  rounds={g.n_steps}")

    # Load tokenizer + model
    # args.model may be a PEFT adapter repo (LoRA only, no base weights).
    # If so, load the base model separately and merge the SFT adapter in.
    from pathlib import Path as _Path

    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    # PEFT adapter if: local dir has adapter_config.json, OR --base-model was explicitly provided
    _local = _Path(args.model)
    is_peft = (_local.exists() and (_local / "adapter_config.json").exists()) or (
        args.base_model is not None
    )

    tokenizer_id = args.model  # args.base_model if is_peft else args.model
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
        # Scope LoRA targets to the language model only.
        # Multimodal Gemma 4 has Gemma4ClippableLinear wrappers in vision/audio
        # towers — these never run during text-only training, so LoRA on them
        # gets zero gradients. Target plain nn.Linear projections in the
        # language_model submodule (and fall back to suffix match for text-only
        # models like Qwen / Gemma 3).
        import torch.nn as _nn
        lora_targets = []
        for _n, _mod in merged.named_modules():
            if not any(_n.endswith(f"self_attn.{_p}") for _p in ("q_proj", "k_proj", "v_proj", "o_proj")):
                continue
            if not isinstance(_mod, _nn.Linear):
                continue  # skip Gemma4ClippableLinear wrappers
            if "language_model" in _n or _n.startswith("model.layers"):
                lora_targets.append(_n)
        if not lora_targets:
            lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
        print(f"  LoRA targets: {len(lora_targets)} modules")
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
        print(
            "LoRA mode: using base model (adapters disabled) as reference — no second copy loaded."
        )
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
    # Per-game SEPO cache and KL tracker (works for single-game too)
    sepo_caches = {g.name: None for g in games}
    kl_since_refresh = {g.name: 0.0 for g in games}

    # Build per-game λe dict — falls back to global --lambda-e for unspecified games
    lambda_e_per_game = {}
    if args.lambda_e_override:
        for token in args.lambda_e_override.split(","):
            gname, val = token.strip().split(":")
            lambda_e_per_game[gname.strip()] = float(val.strip())

    start_step = args.start_step
    print(f"\nStarting GRPO training — {args.iters} steps (from step {start_step})")
    le_str = "  ".join(
        f"{g.name}:λe={lambda_e_per_game.get(g.name, args.lambda_e)}" for g in games
    )
    print(f"SEPO weights: {le_str}  λc={args.lambda_c}  λx={args.lambda_x}")
    print(
        f"SEPO refresh: every {args.sepo_eval_every} steps OR when cumulative KL > {args.sepo_kl_threshold}\n"
    )

    for _i in range(args.iters):
        step = start_step + _i
        optimizer.zero_grad()

        step_losses, step_metrics_list = [], []

        for game in games:
            refresh = (
                sepo_caches[game.name] is None
                or step % args.sepo_eval_every == 0
                or kl_since_refresh[game.name] > args.sepo_kl_threshold
            )
            if refresh and step > 0:
                print(
                    f"  [SEPO refresh] game={game.name} step={step} cumKL={kl_since_refresh[game.name]:.3f}",
                    flush=True,
                )
                kl_since_refresh[game.name] = 0.0

            loss, metrics = grpo_step(
                model=model,
                ref_model=ref_model,
                tokenizer=tokenizer,
                game=game,
                device=device,
                n_rollouts=args.n_rollouts,
                temperature=args.temperature,
                lambda_e=lambda_e_per_game.get(game.name, args.lambda_e),
                lambda_c=args.lambda_c,
                lambda_x=args.lambda_x,
                beta=args.beta,
                clip_eps=args.clip_eps,
                seed_offset=step * 10000 + abs(hash(game.name)) % 10000,
                max_new_tokens=args.max_new_tokens,
                use_token_type_ids=args.token_type_ids,
            )

            if loss is not None:
                step_losses.append(loss)
                step_metrics_list.append(metrics)

        if not step_losses:
            continue

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()

        # Average metrics across games for logging
        avg_loss = float(np.mean(step_losses))
        avg_metrics = {
            k: float(np.mean([m[k] for m in step_metrics_list]))
            for k in step_metrics_list[0]
        }

        log_entry = {
            "step": step,
            "loss": avg_loss,
            "games": [g.name for g in games],
            **avg_metrics,
        }
        log.append(log_entry)

        if step % args.log_every == 0:
            game_tag = f"[{','.join(g.name for g in games)}] " if len(games) > 1 else ""
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"Step {step:4d} {game_tag}| loss={avg_loss:.6f} | "
                f"u={avg_metrics.get('utility', 0):.3f} | "
                f"e={avg_metrics.get('exploitability', 0):.3f} | "
                f"c={avg_metrics.get('collusion', 0):.3f} | "
                f"x={avg_metrics.get('externality', 0):.3f} | "
                f"kl={avg_metrics.get('kl', 0):.6f} | "
                f"pg={avg_metrics.get('pg_loss', 0):.6f}"
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
    p.add_argument(
        "--model",
        required=True,
        help="HF model path or repo (SFT checkpoint; full model or PEFT adapter)",
    )
    p.add_argument(
        "--base-model",
        default=None,
        help="Base model to load before applying PEFT adapter. If set, --model is treated as a LoRA adapter repo.",
    )
    p.add_argument("--output-dir", default="grpo_output")
    p.add_argument(
        "--lora", action="store_true", help="Use LoRA for GRPO policy (lower VRAM)"
    )
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument(
        "--ref-4bit",
        action="store_true",
        help="Load reference model in 4-bit (saves VRAM)",
    )

    # Game
    p.add_argument(
        "--game",
        default="ipd",
        choices=list(GAME_REGISTRY.keys()) + ["all"],
        help="Game environment: ipd | resource | auction | negotiation | all (joint multi-game GRPO)",
    )

    # SEPO objective weights
    p.add_argument(
        "--lambda-e",
        type=float,
        default=2.4,
        help="Exploitability penalty weight (global default)",
    )
    p.add_argument(
        "--lambda-c", type=float, default=2.4, help="Collusion penalty weight"
    )
    p.add_argument(
        "--lambda-x", type=float, default=2.4, help="Externality penalty weight"
    )
    p.add_argument(
        "--lambda-e-override",
        type=str,
        default=None,
        help="Per-game λe overrides, comma-separated: ipd:3.0,negotiation:4.0"
        " — unspecified games use --lambda-e",
    )

    # GRPO hyperparameters
    p.add_argument("--iters", type=int, default=500)
    p.add_argument(
        "--n-rollouts",
        type=int,
        default=8,
        help="Rollouts per train-pool opponent per step",
    )
    p.add_argument(
        "--n-rounds", type=int, default=8, help="Rounds per episode (default 8)"
    )
    p.add_argument(
        "--temperature", type=float, default=0.8, help="Sampling temperature"
    )
    p.add_argument(
        "--max-new-tokens", type=int, default=128, help="Max tokens per generation"
    )
    p.add_argument(
        "--token-type-ids",
        action="store_true",
        help="Pass token_type_ids=zeros (required for Gemma 3, not Gemma 4)",
    )
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--beta", type=float, default=0.01, help="KL penalty weight")
    p.add_argument(
        "--clip-eps",
        type=float,
        default=0.2,
        help="PPO-style clip epsilon (DeepSeek-R1 default)",
    )
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument(
        "--sepo-eval-every",
        type=int,
        default=5,
        help="Recompute SEPO aux episodes every N steps (default 5)",
    )
    p.add_argument(
        "--sepo-kl-threshold",
        type=float,
        default=0.5,
        help="Also refresh SEPO when cumulative KL since last refresh exceeds this (default 0.5)",
    )

    p.add_argument(
        "--show-gen",
        action="store_true",
        help="Print generated text for each round (verify model output)",
    )
    p.add_argument(
        "--start-step",
        type=int,
        default=0,
        help="Resume offset: step counter starts here (use with a checkpoint as --model)",
    )
    args = p.parse_args()
    if args.show_gen:
        run_episode._show_gen = True
    train(args)


if __name__ == "__main__":
    main()
