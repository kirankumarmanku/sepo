# RunPod Setup Guide — SEPO Training

End-to-end guide: provision a RunPod GPU, connect via VSCode Remote SSH, clone the repo, authenticate HuggingFace, and run SFT + GRPO training.

---

## 1. Provision a RunPod Instance

1. Go to [runpod.io](https://runpod.io) → **Secure Cloud** → **Deploy**
2. Select a GPU pod:

| Use case | GPU | VRAM | Est. cost |
|---|---|---|---|
| SFT (Gemma 3 4B LoRA) | RTX 4090 | 24 GB | ~$0.44/hr |
| GRPO (Gemma 3 4B LoRA) | RTX 4090 | 24 GB | ~$0.44/hr |
| GRPO (Gemma 4 e2b LoRA) | A40 | 48 GB | ~$0.79/hr |

3. **Template:** `RunPod PyTorch 2.x` (has CUDA + Python pre-installed)
4. **Disk:** Set container disk ≥ 40 GB (model weights + adapters)
5. **Expose SSH port:** Under "Advanced" → Enable SSH → note the **host** and **port**

---

## 2. Add Your SSH Key to RunPod

1. On your local machine, copy your public key:
   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```
2. In RunPod → **Settings** → **SSH Keys** → paste the key and save
3. Test connection after pod starts:
   ```bash
   ssh root@<pod-ip> -p <port>
   ```

---

## 3. Add Pod to Local SSH Config

Edit `~/.ssh/config` on your local machine:

```
Host runpod-sepo
    HostName <pod-ip>
    User root
    Port <port>
    IdentityFile ~/.ssh/id_ed25519
    StrictHostKeyChecking no
```

Now you can connect with just:
```bash
ssh runpod-sepo
```

> **Note:** RunPod IP and port change every time you restart a pod. Update `~/.ssh/config` after each restart.

---

## 4. Connect VSCode via Remote SSH

1. Install the **Remote - SSH** extension in VSCode
2. Open command palette → `Remote-SSH: Connect to Host` → select `runpod-sepo`
3. VSCode opens a new window connected to the pod
4. Open the terminal inside VSCode (`Ctrl+` `` ` ``) — you are now in the pod shell
5. Open folder → `/root/workspace/sepo` (after cloning in step 6)

---

## 5. Start a tmux Session

**Always run training inside tmux** — SSH disconnections will not kill the process.

```bash
# Start a new session
tmux new -s train

# Detach from session (training keeps running)
Ctrl+B, then D

# Re-attach after reconnecting SSH
tmux attach -t train

# List sessions
tmux ls

# Kill a session
tmux kill-session -t train
```

> **Rule:** Never start a training run without tmux. If SSH drops mid-run without tmux, the process dies and all progress is lost.

---

## 6. Clone the Repo

```bash
cd /root/workspace
git clone -b grpo-stage2 https://github.com/kirankumarmanku/sepo.git
cd sepo
```

If you need to authenticate git for pushing:
```bash
git remote set-url origin https://<YOUR_GITHUB_TOKEN>@github.com/kirankumarmanku/sepo.git
```

---

## 7. Install Dependencies

```bash
pip install torch transformers accelerate peft trl datasets \
            bitsandbytes huggingface_hub numpy scipy openai
```

Or use the setup script (does all of the above):
```bash
bash setup_runpod.sh
```

---

## 8. Authenticate HuggingFace

Gemma 3/4 are gated models — you need a HF token with **read access** to download them. You also need **write access** if you want to push checkpoints to HF Hub.

### Get a token
1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a token with **Read** role (add **Write** if pushing checkpoints)
3. Accept the Gemma model license at:
   - [huggingface.co/google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it)
   - [huggingface.co/google/gemma-4-e2b-it](https://huggingface.co/google/gemma-4-e2b-it) (if using Gemma 4)

### Log in on RunPod
```bash
huggingface-cli login
# Paste your token when prompted
```

### Verify access
```bash
python -c "from huggingface_hub import whoami; print(whoami()['name'])"
```

---

## 9. Training & Evaluation Commands

See **[commands.md](commands.md)** for complete data generation, SFT, GRPO, and eval commands for both Qwen3.5-4B and Gemma 4 E4B-it across all games.

---

## 13. Monitor Training

### Watch live log
```bash
tail -f grpo_run.log
```

### Check GPU usage
```bash
nvidia-smi
watch -n 2 nvidia-smi   # refresh every 2 seconds
```

### Reconnect to tmux after SSH drop
```bash
ssh runpod-sepo
tmux attach -t grpo
```

---

## 14. Checkpoints

Checkpoints are saved every 100 steps by default (`--save-every 100`) to `grpo_output/step_XXXX/`.

```
grpo_output/
├── step_0100/          ← LoRA adapter at step 100
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files
├── step_0200/
└── final/             ← saved at end of run
```

### Push a checkpoint to HuggingFace

```bash
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='grpo_output/step_0100',
    repo_id='<your-hf-username>/gemma-3-4b-sepo-grpo',
    repo_type='model',
)
"
```

---

## 15. Pulling Updates from GitHub

```bash
# On RunPod, inside the repo directory
git pull origin grpo-stage2
```

If you hit auth errors after SSH reconnect:
```bash
git remote set-url origin https://<YOUR_GITHUB_TOKEN>@github.com/kirankumarmanku/sepo.git
git pull origin grpo-stage2
```

---

## Common Issues

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: datasets` | `pip install datasets` |
| `CUDA out of memory` | Add `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before the command |
| `[PARSE FAIL]` every round | Check `--show-gen`, verify model outputs COOPERATE/DEFECT not garbage |
| `loss=0.000000` every step | Generation is producing identical outputs — check temperature and opponent diversity |
| Git auth fails after reconnect | Re-set remote URL with token (see step 15) |
| HF 401 error on save (warning only) | Harmless — PEFT can't fetch base model config without auth, adapter saves correctly |
| tmux session not found | Pod was restarted — create new session with `tmux new -s grpo` |
