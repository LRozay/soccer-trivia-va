"""
llm_utils.py — Model loading, inference, and helper utilities.
"""

import re
import json
import time
import hashlib

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

SMALL_MODEL = "microsoft/Phi-3.5-mini-instruct"
LARGE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"


def load_model(model_name: str, quantize_4bit: bool = False, device_map: str = "auto") -> tuple:
    print(f"⏳ Loading {model_name} ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if quantize_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_cfg,
            device_map=device_map, trust_remote_code=True)
        # Skip model.eval() for quantized models — .to() is not supported
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16,
            device_map=device_map, trust_remote_code=True)
        model.eval()
    print(f"✅ Loaded in {time.time() - t0:.1f}s")
    return model, tokenizer


def generate(model, tokenizer, messages: list[dict], max_new_tokens: int = 512,
             temperature: float = 0.3, do_sample: bool = True) -> str:
    result = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", tokenize=True)
    if hasattr(result, "input_ids"):
        input_ids      = result.input_ids.to(model.device)
        attention_mask = result.attention_mask.to(model.device)
    else:
        input_ids      = result.to(model.device)
        attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, temperature=temperature,
            do_sample=do_sample, pad_token_id=tokenizer.eos_token_id)
    new_tokens = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


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
