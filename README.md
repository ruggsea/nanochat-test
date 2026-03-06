# nanochat-test

Running [karpathy/nanochat-d32](https://huggingface.co/karpathy/nanochat-d32) with HuggingFace transformers.

## Setup

nanochat is only in transformers main (not released yet), so you need to install from git:

```bash
pip install torch
pip install git+https://github.com/huggingface/transformers.git
```

## Usage

```bash
python inference_transformers.py
```

Or in your own code:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "karpathy/nanochat-d32"
revision = "refs/pr/1"  # transformers-compatible weights

tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    revision=revision,
    torch_dtype=torch.bfloat16,
).to("cuda")

inputs = tokenizer("The capital of Belgium is", return_tensors="pt").to("cuda")

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

generated = outputs[0, inputs["input_ids"].shape[1]:]
print(tokenizer.decode(generated, skip_special_tokens=True))
```

## Benchmark

On NVIDIA H200, bfloat16, using HF `.generate()` with greedy decoding:

| Tokens | Time | Speed |
|--------|------|-------|
| 32 | 0.91s | 35.2 tok/s |
| 64 | 1.88s | 34.1 tok/s |
| 128 | 3.81s | 33.6 tok/s |
| 256 | 6.92s | 37.0 tok/s |
| Batch 5x64 | 2.06s | 155.3 tok/s |

Peak GPU memory: 3.7 GB

## Notes

- The model needs `revision="refs/pr/1"` - that's the PR adding native transformers support to karpathy's repo
- Works with `AutoModelForCausalLM` and standard `.generate()` API
- bfloat16 on GPU, float32 on CPU
- ONNX export currently broken - `torch.onnx.export` (both jit and dynamo) chokes on the causal mask implementation in transformers 5.x dev. Waiting on upstream fixes.

## Model info

- **Model**: [karpathy/nanochat-d32](https://huggingface.co/karpathy/nanochat-d32) (nanoGPT, 32 layers)
- **License**: MIT
