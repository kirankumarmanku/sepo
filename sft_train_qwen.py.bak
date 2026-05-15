import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-4B', dtype=torch.bfloat16, device_map='auto')
tok = AutoTokenizer.from_pretrained('./sft_qwen_v2/final_adapter')
model = PeftModel.from_pretrained(base, './sft_qwen_v2/final_adapter').merge_and_unload()

# Test one example from each game
seen = set()
with open("sepo_sft_data_multi/train.jsonl") as f:
    for line in f:
        row = json.loads(line)
        sys = row["messages"][0]["content"]
        if "Prisoner" in sys: key = "IPD"
        elif "auction" in sys.lower(): key = "AUCTION"
        elif "resource" in sys.lower() or "extract" in sys.lower(): key = "RESOURCE"
        elif "negotiat" in sys.lower() or "demand" in sys.lower(): key = "NEGOTIATION"
        else: continue
        if key in seen: continue
        seen.add(key)
        print(f"=== {key} ===")
        infer_msgs = [m for m in row["messages"] if m["role"] != "assistant"]
        text = tok.apply_chat_template(infer_msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors='pt').to(model.device)
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False, pad_token_id=tok.eos_token_id)
        print(tok.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))
        print()
        if len(seen) == 4: break
