#!/usr/bin/env python3
"""Run nanochat-d32 inference using HuggingFace transformers."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "karpathy/nanochat-d32"
REVISION = "refs/pr/1"  # PR that adds native transformers support


def load_model(device: str = "auto"):
    """Load model and tokenizer from HuggingFace."""
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        torch_dtype=torch.bfloat16 if "cuda" in device else torch.float32,
    ).to(device)

    return model, tokenizer, device


def generate(model, tokenizer, prompt: str, device: str, max_new_tokens: int = 64):
    """Generate text from a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = outputs[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


if __name__ == "__main__":
    print(f"Loading {MODEL_ID}...")
    model, tokenizer, device = load_model()
    print(f"Loaded on {device}")

    prompts = [
        "The capital of Belgium is",
        "The Eiffel Tower stands in Paris and",
    ]

    for prompt in prompts:
        result = generate(model, tokenizer, prompt, device)
        print(f"\n> {prompt}")
        print(f"  {result}")
