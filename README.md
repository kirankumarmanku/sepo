## Running gemma_ipd_baseline: 

### Install prerequisites:



```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run the model:
With Ollama

```
ollama pull gemma3:4b
python gemma_ipd_baseline.py --backend openai \
  --base-url http://localhost:11434/v1 \
  --model gemma3:4b
```
