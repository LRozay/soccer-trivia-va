"""
llm_utils.py — Model loading and inference.

Backends (in priority order):
  1. Groq API  — free, fast (500+ tok/s), Llama 3.1 8B & 70B
  2. Local HuggingFace — fallback if no Groq key

Set GROQ_API_KEY in Colab secrets (or as env var) to use Groq.
"""

import os
import re
import json
import time
import hashlib

# ─────────────────────────────────────────────
# Model identifiers
# ─────────────────────────────────────────────

# Groq model IDs
GROQ_SMALL = "llama-3.1-8b-instant"    # fast, free
GROQ_LARGE = "llama-3.3-70b-versatile" # stronger reasoning, still free

# Local HuggingFace fallback
SMALL_MODEL = "microsoft/Phi-3.5-mini-instruct"
LARGE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"


# ─────────────────────────────────────────────
# Groq client (lazy init)
# ─────────────────────────────────────────────

_groq_client = None

def _get_groq_key() -> str | None:
    # Try Colab secrets first, then env var
    try:
        from google.colab import userdata
        return userdata.get("GROQ_API_KEY")
    except Exception:
        return os.environ.get("GROQ_API_KEY")

def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        try:
            from groq import Groq
            key = _get_groq_key()
            if not key:
                return None
            _groq_client = Groq(api_key=key)
        except ImportError:
            return None
    return _groq_client


# ─────────────────────────────────────────────
# Load a local model (only used if no Groq key)
# ─────────────────────────────────────────────

def load_model(model_name: str = SMALL_MODEL,
               quantize_4bit: bool = False,
               device_map: str = "auto") -> tuple:
    """
    If a Groq API key is available, returns (None, None) — no local model needed.
    Otherwise loads the HuggingFace model.
    """
    if _get_groq_key():
        print("✅ Groq API key found — using Groq backend (no local model needed).")
        print(f"   Small model : {GROQ_SMALL}")
        print(f"   Large model : {GROQ_LARGE}")
        return None, None

    print(f"⚠️  No Groq key found — loading local model: {model_name}")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print(f"✅ Loaded in {time.time() - t0:.1f}s")
    return model, tokenizer


# ─────────────────────────────────────────────
# Generate — Groq path
# ─────────────────────────────────────────────

def _generate_groq(messages: list[dict], max_new_tokens: int = 256,
                   temperature: float = 0.3, model_size: str = "small") -> str:
    client = _get_groq_client()
    if client is None:
        return "[Groq unavailable]"

    groq_model = GROQ_SMALL if model_size == "small" else GROQ_LARGE

    completion = client.chat.completions.create(
        model=groq_model,
        messages=messages,
        max_tokens=max_new_tokens,
        temperature=temperature,
    )
    return completion.choices[0].message.content.strip()


# ─────────────────────────────────────────────
# Generate — Local HuggingFace path
# ─────────────────────────────────────────────

def _generate_local(model, tokenizer, messages: list[dict],
                    max_new_tokens: int = 256, temperature: float = 0.3,
                    do_sample: bool = True) -> str:
    import torch

    result = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", tokenize=True)

    if hasattr(result, "input_ids"):
        input_ids      = result.input_ids.to(model.device)
        attention_mask = result.attention_mask.to(model.device)
        prompt_length  = input_ids.shape[-1]
    else:
        input_ids      = result.to(model.device)
        attention_mask = torch.ones_like(input_ids)
        prompt_length  = input_ids.shape[-1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_token_ids = output_ids[0][prompt_length:]
    reply = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

    del input_ids, attention_mask, output_ids
    torch.cuda.empty_cache()
    return reply


# ─────────────────────────────────────────────
# Unified generate() — called by pipeline
# ─────────────────────────────────────────────

def generate(model, tokenizer, messages: list[dict],
             max_new_tokens: int = 256, temperature: float = 0.3,
             do_sample: bool = True, model_size: str = "small") -> str:
    """
    Route to Groq or local depending on what's available.
    model/tokenizer can be None if using Groq.
    """
    if _get_groq_key():
        return _generate_groq(messages, max_new_tokens, temperature, model_size)
    elif model is not None and tokenizer is not None:
        return _generate_local(model, tokenizer, messages, max_new_tokens,
                                temperature, do_sample)
    else:
        return "[No model available — set GROQ_API_KEY or load a local model]"


# ─────────────────────────────────────────────
# JSON extraction
# ─────────────────────────────────────────────

def extract_json(text: str) -> dict | list | None:
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for pattern in [r"\{.*\}", r"\[.*\]"]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


# ─────────────────────────────────────────────
# Prompt cache
# ─────────────────────────────────────────────

_prompt_cache: dict[str, str] = {}

def cached_generate(model, tokenizer, messages: list[dict], **kwargs) -> tuple[str, bool]:
    key = hashlib.md5(json.dumps(messages, sort_keys=True).encode()).hexdigest()
    if key in _prompt_cache:
        return _prompt_cache[key], True
    response = generate(model, tokenizer, messages, **kwargs)
    _prompt_cache[key] = response
    return response, False

def clear_cache():
    _prompt_cache.clear()


# ─────────────────────────────────────────────
# Latency benchmark (for eval)
# ─────────────────────────────────────────────

def benchmark(model, tokenizer, messages: list[dict],
              runs: int = 3, use_cache: bool = False) -> dict:
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
        "min_s":         round(min(times), 3),
        "max_s":         round(max(times), 3),
        "backend":       "groq" if _get_groq_key() else "local",
    }
