"""
llm_utils.py — Model loading, inference, and helper utilities.
"""

import re
import json
import time
import hashlib

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

SMALL_MODEL = "microsoft/Phi-3.5-mini-instruct"
LARGE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"


def load_model(model_name: str, quantize_4bit: bool = False, device_map: str = "auto") -> tuple:
    """
    Load model in float16 — fits on T4 and avoids bitsandbytes version conflicts.
    quantize_4bit param kept for API compatibility but ignored.
    """
    print(f"⏳ Loading {model_name} ...")
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


def generate(model, tokenizer, messages: list[dict], max_new_tokens: int = 256,
             temperature: float = 0.3, do_sample: bool = True) -> str:
    """Generate a response and free GPU memory immediately after."""
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

    # Decode only the newly generated tokens
    new_token_ids = output_ids[0][prompt_length:]
    reply = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

    # Free GPU memory
    del input_ids, attention_mask, output_ids
    torch.cuda.empty_cache()

    return reply


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

def benchmark(model, tokenizer, messages: list[dict], runs: int = 3, use_cache: bool = False) -> dict:
    times = []
    for _ in range(runs):
        if use_cache:
            clear_cache()
        t0 = time.time()
        cached_generate(model, tokenizer, messages) if use_cache else generate(model, tokenizer, messages)
        times.append(time.time() - t0)
    return {"runs": runs, "avg_latency_s": round(sum(times)/len(times), 3),
            "min_s": round(min(times), 3), "max_s": round(max(times), 3)}
