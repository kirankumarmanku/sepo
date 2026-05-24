# Qwen3.5-4B SEPO Pipeline — Commands

## Environment setup

```bash
pip install --force-reinstall torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
```

```bash
pip install flash-linear-attention
```

```bash
apt install -y vim bc
```

## SFT diagnostics

```bash
python - <<'PY'
import json
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")

with open("sepo_sft_data_multi/train.jsonl") as f:
    lengths = []
    for line in f:
        row = json.loads(line)
        text = tok.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
        n = len(tok.encode(text))
        lengths.append(n)

import statistics
print(f"Examples: {len(lengths)}")
print(f"Median: {statistics.median(lengths)}")
print(f"Mean:   {statistics.mean(lengths):.0f}")
print(f"Max:    {max(lengths)}")
print(f"Over 256: {sum(1 for l in lengths if l > 256)} / {len(lengths)}")
PY
```

```bash
python - <<'PY'
import json
with open("sepo_sft_data_multi/train.jsonl") as f:
    for i, line in enumerate(f):
        if i >= 5: break
        row = json.loads(line)
        print(f"=== Example {i} ===")
        for m in row["messages"]:
            print(f"  [{m['role']}] {m['content']}")
        print()
PY
```

## SFT training

```bash
CUDA_VISIBLE_DEVICES=0 python -m train.sft --model Qwen/Qwen3.5-4B --data-dir sepo_sft_data_multi --epochs 1 --output-dir ./sft_qwen_test 2>&1 | tee sft_qwen_test.log
```

```bash
pkill -f "train.sft"
```

```bash
tmux new -s sft
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m train.sft \
  --model Qwen/Qwen3.5-4B \
  --data-dir sepo_sft_data_multi \
  --output-dir ./sft_qwen_v2 \
  --epochs 3 \
  --hf-repo kirankumarmanku/Qwen3.5-4B-sepo-sft-v2 \
  2>&1 | tee sft_qwen_v2.log
```

```bash
grep -E "'loss'|'eval_loss'|train_runtime|train_loss" sft_qwen_v2.log | tail -20
```

## SFT inference verification

```bash
python - <<'PY'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-4B', dtype=torch.bfloat16, device_map='auto')
tok = AutoTokenizer.from_pretrained('./sft_qwen_v2/final_adapter')
model = PeftModel.from_pretrained(base, './sft_qwen_v2/final_adapter').merge_and_unload()
sys_prompt = """You are playing the Iterated Prisoner's Dilemma game.
Rules:
- Each round you choose one of two actions: COOPERATE or DEFECT
- If both players choose COOPERATE: you each get 3 points
- If you choose DEFECT and opponent chooses COOPERATE: you get 5, opponent gets 0
- If you choose COOPERATE and opponent chooses DEFECT: you get 0, opponent gets 5
- If both choose DEFECT: you each get 1 point
Your goal is to maximise your total score over all rounds.
Think briefly about the opponent's pattern, then end your response with your action on the last line: COOPERATE or DEFECT."""
user = """Round 4 of 8.
Your past actions:       COOPERATE, COOPERATE, COOPERATE
Opponent's past actions: COOPERATE, COOPERATE, COOPERATE
Scores so far — you: 9, opponent: 9
What is your action?"""
messages = [{'role':'system','content':sys_prompt},{'role':'user','content':user}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(text, return_tensors='pt').to(model.device)
out = model.generate(**inputs, max_new_tokens=512, do_sample=False, pad_token_id=tok.eos_token_id)
print(tok.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))
PY
```

## GRPO script updates

```bash
cp train/grpo.py train/grpo.py.bak
python apply_grpo_updates.py
python verify_grpo_updates.py
```

## GRPO probing

```bash
chmod +x probe_temperature.sh probe_lr.sh launch_grpo.sh monitor_grpo.sh
```

```bash
./probe_temperature.sh
```

```bash
./probe_lr.sh 0.8
```

```bash
for LR in 1e-6 3e-6 1e-5; do
  echo "=== lr=$LR ==="
  grep -oP "kl=\d+\.\d+" probes/probe_lr_${LR}.log | head -5
done
```

```bash
for T in 0.6 0.8 1.0; do
  echo "=== T=$T ==="
  grep -oP "pg=\-?\d+\.\d+" probes/probe_temp_${T}.log | head -3
done
```

## GRPO launch

```bash
./launch_grpo.sh 0.8 1e-5 0.1
```

```bash
./monitor_grpo.sh
```

## GRPO checkpoint verification

```bash
ls grpo_qwen_v2_final/
```

```bash
python - <<'PY'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-4B', dtype=torch.bfloat16, device_map='auto')
tok = AutoTokenizer.from_pretrained('grpo_qwen_v2_final/step_0025')
base = PeftModel.from_pretrained(base, 'kirankumarmanku/Qwen3.5-4B-sepo-sft-v2').merge_and_unload()
model = PeftModel.from_pretrained(base, 'grpo_qwen_v2_final/step_0025').merge_and_unload()

msgs = [
    {'role':'system','content':"You are playing the Iterated Prisoner's Dilemma. Choose COOPERATE or DEFECT. End with action on last line."},
    {'role':'user','content':"Round 4 of 8. Your past: COOPERATE, COOPERATE, COOPERATE. Opponent past: DEFECT, DEFECT, DEFECT. Scores: you=0, opp=15. Action?"},
]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = tok(text, return_tensors='pt').to(model.device)
out = model.generate(**inputs, max_new_tokens=256, do_sample=False, pad_token_id=tok.eos_token_id)
print(tok.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))
PY
```

## Eval script patch

```bash
sed -i 's|, enable_thinking=False||' eval/eval_sepo.py
```

```bash
python - <<'PY'
src = open("eval/eval_sepo.py").read()

old = '''def load_model(model_path: str, adapter_path: Optional[str], device):
    print(f"  Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    if adapter_path:
        print(f"  Loading adapter from {adapter_path}...")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        print("  Adapter merged.")

    model.eval()
    return model, tokenizer'''

new = '''def load_model(model_path: str, adapter_path: Optional[str], device,
               sft_adapter: Optional[str] = None):
    """Load base -> optional SFT adapter (merged) -> optional final adapter (merged)."""
    tok_source = adapter_path or sft_adapter or model_path
    print(f"  Loading tokenizer from {tok_source}...")
    tokenizer = AutoTokenizer.from_pretrained(tok_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading base model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    from peft import PeftModel
    if sft_adapter:
        print(f"  Applying SFT adapter: {sft_adapter}")
        model = PeftModel.from_pretrained(model, sft_adapter)
        model = model.merge_and_unload()
        print("  SFT adapter merged.")

    if adapter_path:
        print(f"  Applying final adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        print("  Final adapter merged.")

    model.eval()
    return model, tokenizer'''

src = src.replace(old, new)

old_arg = 'ap.add_argument("--adapter",    default=None,  help="LoRA adapter path (optional)")'
new_arg = '''ap.add_argument("--adapter",    default=None,  help="LoRA adapter path (optional)")
    ap.add_argument("--sft-adapter", default=None,  help="SFT adapter merged before --adapter")'''
src = src.replace(old_arg, new_arg)

old_call = "model, tokenizer = load_model(args.model, args.adapter, device)"
new_call = "model, tokenizer = load_model(args.model, args.adapter, device, sft_adapter=args.sft_adapter)"
src = src.replace(old_call, new_call)

open("eval/eval_sepo.py", "w").write(src)
print("Patched eval/eval_sepo.py")
PY
```

## Evaluation runs

```bash
CUDA_VISIBLE_DEVICES=1 python -m eval.eval_sepo \
    --model Qwen/Qwen3.5-4B \
    --game all --episodes 8 --temperature 0.0 --max-tokens 512 \
    --label "base" --output-dir eval_results/base_e8 2>&1 | tee eval_base_e8.log
```

```bash
CUDA_VISIBLE_DEVICES=1 python -m eval.eval_sepo \
    --model Qwen/Qwen3.5-4B \
    --adapter kirankumarmanku/Qwen3.5-4B-sepo-sft-v2 \
    --game all --episodes 8 --temperature 0.0 --max-tokens 512 \
    --label "sft_v2" --output-dir eval_results/sft_v2_e8 2>&1 | tee eval_sft_v2_e8.log
```

```bash
for STEP in step_0025 step_0050 step_0075 final; do
    CUDA_VISIBLE_DEVICES=1 python -m eval.eval_sepo \
        --model Qwen/Qwen3.5-4B \
        --sft-adapter kirankumarmanku/Qwen3.5-4B-sepo-sft-v2 \
        --adapter "grpo_qwen_v2_final/$STEP" \
        --game all --episodes 8 --temperature 0.0 --max-tokens 512 \
        --label "grpo_$STEP" --output-dir "eval_results/grpo_${STEP}_e8" \
        2>&1 | tee "eval_${STEP}_e8.log"
done
```

## Process management

```bash
tmux list-sessions
tmux kill-session -t grpo2
```

```bash
ps aux | grep eval_sepo | grep -v grep
pkill -9 -f eval_sepo
```

```bash
nvidia-smi
```

## Training log inspection

```bash
python - <<'PY'
import json
with open("grpo_qwen_v2_final/log.json") as f:
    log = json.load(f)

print(f"Total steps logged: {len(log)}")
print(f"Last step: {log[-1]['step']}")
print()
print(f"{'step':>6} {'utility':>8} {'exploit':>8} {'collusion':>10} {'externality':>12} {'kl':>8} {'pg':>8}")
print("-" * 70)
for entry in log:
    if entry.get('exploitability', 0) > 0:
        print(f"{entry['step']:>6} {entry['utility']:>8.3f} {entry['exploitability']:>8.3f} "
              f"{entry['collusion']:>10.3f} {entry['externality']:>12.3f} "
              f"{entry.get('kl', 0):>8.3f} {entry.get('pg_loss', 0):>8.3f}")
PY
```
