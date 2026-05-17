import json
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")

with open("sepo_sft_data_multi/train.jsonl") as f:
    rows = [json.loads(line) for line in f]

# Bucket by game (assuming game name is in the prompt or a metadata field)
from collections import defaultdict
by_game = defaultdict(list)
for r in rows:
    text = tok.apply_chat_template(r["messages"], tokenize=False)
    n_tokens = len(tok(text)["input_ids"])
    # crude: detect game by keyword in user message
    user = next(m["content"] for m in r["messages"] if m["role"] == "user").lower()
    if "offer" in user or "split" in user or "negotiat" in user:
        game = "negotiation"
    elif "bid" in user or "auction" in user:
        game = "auction"
    elif "extract" in user or "resource" in user:
        game = "resource"
    elif "cooperate" in user or "defect" in user or "dilemma" in user:
        game = "ipd"
    elif "threat" in user or "pressure" in user or "principal" in user:
        game = "pressure"
    else:
        game = "unknown"
    by_game[game].append(n_tokens)

for game, lengths in by_game.items():
    n = len(lengths)
    clipped = sum(1 for l in lengths if l > 256)
    print(f"{game:15s} n={n:5d}  median={sorted(lengths)[n//2]:4d}  max={max(lengths):4d}  clipped@256={clipped}/{n} ({100*clipped/n:.1f}%)")
