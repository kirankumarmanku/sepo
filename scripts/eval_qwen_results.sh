# Base
CUDA_VISIBLE_DEVICES=1 python eval_sepo.py \
	    --model Qwen/Qwen3.5-4B \
	        --game all --episodes 4 --temperature 0.0 --max-tokens 512 \
		    --label "base" --output-dir eval_results/base

# SFT v2
CUDA_VISIBLE_DEVICES=1 python eval_sepo.py \
	    --model Qwen/Qwen3.5-4B \
	        --adapter kirankumarmanku/Qwen3.5-4B-sepo-sft-v2 \
		    --game all --episodes 4 --temperature 0.0 --max-tokens 512 \
		        --label "sft_v2" --output-dir eval_results/sft_v2
