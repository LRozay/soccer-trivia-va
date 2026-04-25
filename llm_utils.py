"""
llm_utils.py — Model loading, inference, and helper utilities.

Supports:
  • Any HuggingFace causal-LM (local, via transformers)
  • Optional 4-bit quantization with bitsandbytes
  • Automatic chat-template formatting
  • JSON extraction from model output
  • Simple prompt-level caching (for repeated system prompts / few-shot blocks)
"""

import re
import json
import time
import hashlib
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ─────────────────────────────────────────────
# Recommended model IDs
# ─────────────────────────────────────────────

SMALL_MODEL  = "microsoft/Phi-3.5-mini-instruct"   # ~3.8B — fits Colab free T4
LARGE_MODEL  = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # ~8B — Colab Pro A100


# ─────────────────────────────────────────────
# Load a model
# ─────────────────────────────────────────────

def load_model(
    model_name: str,
    quantize_4bit: bool = False,
    device_map: str = "auto",
) -> tuple:
    """
    Load a HuggingFace causal-LM and its tokenizer.

    Returns:
        (model, tokenizer)

    Usage:
        model, tok = load_model(SMALL_MODEL)
        model, tok = load_model(LARGE_MODEL, quantize_4bit=True)
    """
    print(f"⏳ Loading {model_name} ...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if quantize_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_cfg,
            device_map=device_map,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
        )

    model.eval()
    print(f"✅ Loaded in {time.time() - t0:.1f}s")
    return model, tokenizer


# ─────────────────────────────────────────────
# Generate a response
# ─────────────────────────────────────────────

def generate(
    model,
    tokenizer,
    messages: list[dict],
    max_new_tokens: int = 512,
    temperature: float = 0.3,
    do_sample: bool = True,
) -> str:
    """
    Generate a response given a list of chat messages.

    messages format:
        [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]

    Returns the assistant's reply as a plain string.
    """
    # Use the tokenizer's built-in chat template (works for Phi, Llama, Mistral, etc.)
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,          # enables KV-cache for speed
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ─────────────────────────────────────────────
# JSON extraction
# ─────────────────────────────────────────────

def extract_json(text: str) -> dict | list | None:
    """
    Robustly extract the first JSON object or array from model output.
    Handles markdown fences, leading/trailing text, etc.
    """
    # Strip markdown fences
    text = re.sub(r"```(?:json)?", "", text).strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the first {...} or [...]
    for pattern in [r"\{.*\}", r"\[.*\]"]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

    return None


# ─────────────────────────────────────────────
# Prompt-level cache
# ─────────────────────────────────────────────

_prompt_cache: dict[str, str] = {}

def cached_generate(
    model,
    tokenizer,
    messages: list[dict],
    **kwargs,
) -> tuple[str, bool]:
    """
    Like generate(), but caches results by a hash of the input messages.
    Returns (response_text, cache_hit).
    Useful for demonstrating latency savings on repeated system prompts.
    """
    key = hashlib.md5(json.dumps(messages, sort_keys=True).encode()).hexdigest()

    if key in _prompt_cache:
        return _prompt_cache[key], True

    response = generate(model, tokenizer, messages, **kwargs)
    _prompt_cache[key] = response
    return response, False


def clear_cache():
    _prompt_cache.clear()


# ─────────────────────────────────────────────
# Latency benchmark helper (for eval section)
# ─────────────────────────────────────────────

def benchmark(
    model,
    tokenizer,
    messages: list[dict],
    runs: int = 3,
    use_cache: bool = False,
) -> dict:
    """
    Run the same prompt N times and report average latency.
    Pass use_cache=True to compare cached vs uncached.
    """
    times = []
    for _ in range(runs):
        if use_cache:
            clear_cache()
        t0 = time.time()
        if use_cache:
            cached_generate(model, tokenizer, messages)
        else:
            generate(model, tokenizer, messages)
        times.append(time.time() - t0)

    return {
        "runs": runs,
        "avg_latency_s": round(sum(times) / len(times), 3),
        "min_s": round(min(times), 3),
        "max_s": round(max(times), 3),
    }
