"""Compare Qwen2.5 decode logits: CPU (float32) vs NPU (bfloat16).

Runs a single prefill + 5 decode steps and compares logit distributions
at each step using Pearson correlation, top-k overlap, and argmax agreement.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
from scipy.stats import pearsonr
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from hf_utils import load_weights_from_hf

import npu_compiler
from npu_runtime.buffer import NPUBuffer
from npu_runtime.device import Device
from npu_runtime.executor import Executor

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT = "The capital of France is"
MAX_DECODE_STEPS = 5


def get_cpu_logits(prompt: str, max_steps: int):
    """Run CPU inference and collect logits at each decode step."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    all_logits = []
    all_tokens = []

    with torch.no_grad():
        # Prefill
        out = model(input_ids, use_cache=True)
        prefill_logits = out.logits[0, -1, :].numpy()  # (vocab,)
        all_logits.append(prefill_logits)
        next_token = int(np.argmax(prefill_logits))
        all_tokens.append(next_token)
        past = out.past_key_values

        # Decode steps
        for _ in range(max_steps - 1):
            next_input = torch.tensor([[next_token]])
            out = model(next_input, past_key_values=past, use_cache=True)
            step_logits = out.logits[0, -1, :].numpy()
            all_logits.append(step_logits)
            next_token = int(np.argmax(step_logits))
            all_tokens.append(next_token)
            past = out.past_key_values

    return all_logits, all_tokens, tokenizer


def get_npu_logits(prompt: str, max_steps: int, tokenizer):
    """Run NPU inference and collect logits at each decode step."""
    device = Device()

    # Compile
    prefill_program = npu_compiler.compile("qwen2_prefill_ir.json")
    decode_program = npu_compiler.compile("qwen2_decode_ir.json")

    # Load weights
    prefill_weights = load_weights_from_hf(MODEL_ID, prefill_program, device)
    decode_weights = load_weights_from_hf(MODEL_ID, decode_program, device)

    # KV mapping
    with open("qwen2_kv_mapping.json") as f:
        kv_map = json.load(f)
    max_cache_len = kv_map["max_cache_len"]

    # Executors
    prefill_executor = Executor(prefill_program, device)
    decode_executor = Executor(decode_program, device)

    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors="np")[0]
    prefill_seq_len = prefill_program.input_specs[0].shape[1]
    actual_len = min(len(input_ids), prefill_seq_len)

    # Prefill inputs
    padded_ids = np.zeros((1, prefill_seq_len), dtype=np.int64)
    padded_ids[0, :actual_len] = input_ids[:actual_len]

    _NEG_INF = np.float16(-np.inf)
    causal_mask = np.triu(
        np.full((1, 1, prefill_seq_len, prefill_seq_len), _NEG_INF, dtype=np.float16), k=1
    )
    causal_mask[:, :, :, actual_len:] = _NEG_INF
    causal_mask[:, :, actual_len:, :] = _NEG_INF
    position_ids = np.arange(prefill_seq_len, dtype=np.int64).reshape(1, -1)
    cache_position = np.arange(prefill_seq_len, dtype=np.int64)

    prefill_np_map = {
        "input_ids": padded_ids,
        "attention_mask": causal_mask,
        "position_ids": position_ids,
        "cache_position": cache_position,
    }
    prefill_inputs = {}
    for spec in prefill_program.input_specs:
        prefill_inputs[spec.name] = NPUBuffer.from_numpy(prefill_np_map[spec.name], device, spec=spec)

    # Prefill
    prefill_outputs = prefill_executor.run(inputs=prefill_inputs, weights=prefill_weights)
    logits_spec = prefill_program.output_specs[0]
    logits = prefill_outputs[logits_spec.name].to_numpy(spec=logits_spec)
    prefill_logits = logits[0, actual_len - 1, :]

    all_logits = [prefill_logits]
    next_token = int(np.argmax(prefill_logits))
    all_tokens = [next_token]

    # Pad prefill KV from (B,H,prefill_seq_len,D) to (B,H,max_cache_len,D)
    _FIXED_INPUTS = {"input_ids", "attention_mask", "position_ids", "cache_position"}
    decode_kv_specs = {s.name: s for s in decode_program.input_specs if s.name not in _FIXED_INPUTS}

    kv_buffers = {}
    for layer in kv_map["layers"]:
        for pkey, dkey in [(layer["prefill_key_output"], layer["decode_key_input"]),
                           (layer["prefill_value_output"], layer["decode_value_input"])]:
            prefill_kv = prefill_outputs[pkey]
            decode_spec = decode_kv_specs[dkey]
            prefill_spec = next(s for s in prefill_program.output_specs if s.name == pkey)
            kv_np = prefill_kv.to_numpy(spec=prefill_spec)
            padded_kv = np.zeros(decode_spec.shape, dtype=kv_np.dtype)
            padded_kv[:, :, :prefill_seq_len, :] = kv_np
            kv_buffers[dkey] = NPUBuffer.from_numpy(padded_kv, device, spec=decode_spec)

    # Decode steps
    for step in range(max_steps - 1):
        cur_pos = actual_len + step
        token_arr = np.array([[next_token]], dtype=np.int64)
        decode_mask = np.full((1, 1, 1, max_cache_len), _NEG_INF, dtype=np.float16)
        decode_mask[0, 0, 0, :cur_pos + 1] = 0.0
        decode_pos_ids = np.array([[cur_pos]], dtype=np.int64)
        decode_cache_pos = np.array([cur_pos], dtype=np.int64)

        decode_np_map = {
            "input_ids": token_arr,
            "attention_mask": decode_mask,
            "position_ids": decode_pos_ids,
            "cache_position": decode_cache_pos,
        }
        decode_inputs = {}
        for spec in decode_program.input_specs:
            if spec.name in decode_np_map:
                decode_inputs[spec.name] = NPUBuffer.from_numpy(decode_np_map[spec.name], device, spec=spec)
            elif spec.name in kv_buffers:
                decode_inputs[spec.name] = kv_buffers[spec.name]

        decode_outputs = decode_executor.run(inputs=decode_inputs, weights=decode_weights)
        dec_logits_name = decode_program.output_specs[0].name
        dec_logits = decode_outputs[dec_logits_name].to_numpy(spec=decode_program.output_specs[0])
        step_logits = dec_logits[0, -1, :]

        all_logits.append(step_logits)
        next_token = int(np.argmax(step_logits))
        all_tokens.append(next_token)

        for layer_info in kv_map["layers"]:
            kv_buffers[layer_info["decode_key_input"]] = decode_outputs[layer_info["decode_key_output"]]
            kv_buffers[layer_info["decode_value_input"]] = decode_outputs[layer_info["decode_value_output"]]

    return all_logits, all_tokens


def compare_logits(cpu_logits_list, npu_logits_list, cpu_tokens, npu_tokens, tokenizer):
    """Compare logit distributions at each step."""
    print("\n" + "=" * 70)
    print("LOGIT COMPARISON: CPU (float32) vs NPU (bfloat16)")
    print("=" * 70)

    all_ok = True

    for i, (cpu_logits, npu_logits) in enumerate(zip(cpu_logits_list, npu_logits_list)):
        cpu_f = cpu_logits.astype(np.float64)
        npu_f = npu_logits.astype(np.float64)

        # Pearson correlation
        r, p_val = pearsonr(cpu_f, npu_f)

        # Top-k overlap
        cpu_top10 = set(np.argsort(cpu_f)[-10:])
        npu_top10 = set(np.argsort(npu_f)[-10:])
        top10_overlap = len(cpu_top10 & npu_top10)

        cpu_top5 = set(np.argsort(cpu_f)[-5:])
        npu_top5 = set(np.argsort(npu_f)[-5:])
        top5_overlap = len(cpu_top5 & npu_top5)

        # Argmax agreement
        cpu_argmax = int(np.argmax(cpu_f))
        npu_argmax = int(np.argmax(npu_f))
        argmax_match = cpu_argmax == npu_argmax

        # Max abs difference
        max_diff = np.max(np.abs(cpu_f - npu_f))
        mean_diff = np.mean(np.abs(cpu_f - npu_f))

        # Relative error on top logits
        cpu_top_val = cpu_f[cpu_argmax]
        npu_top_val = npu_f[cpu_argmax]
        rel_err_top = abs(cpu_top_val - npu_top_val) / (abs(cpu_top_val) + 1e-8)

        step_name = "Prefill" if i == 0 else f"Decode {i}"
        status = "OK" if (r > 0.99 and argmax_match) else "MISMATCH" if not argmax_match else "WARN"
        if status == "MISMATCH":
            all_ok = False

        print(f"\n--- Step {i} ({step_name}) ---")
        print(f"  Pearson r:       {r:.6f}")
        print(f"  Top-1 (argmax):  CPU={cpu_argmax} ({tokenizer.decode([cpu_argmax])!r})"
              f"  NPU={npu_argmax} ({tokenizer.decode([npu_argmax])!r})"
              f"  {'MATCH' if argmax_match else 'MISMATCH'}")
        print(f"  Top-5 overlap:   {top5_overlap}/5")
        print(f"  Top-10 overlap:  {top10_overlap}/10")
        print(f"  Max |diff|:      {max_diff:.4f}")
        print(f"  Mean |diff|:     {mean_diff:.4f}")
        print(f"  Rel err (top):   {rel_err_top:.6f}")
        print(f"  Status:          {status}")

    print("\n" + "=" * 70)
    print(f"CPU tokens:  {cpu_tokens}")
    print(f"NPU tokens:  {npu_tokens}")
    print(f"CPU text:    {tokenizer.decode(cpu_tokens)!r}")
    print(f"NPU text:    {tokenizer.decode(npu_tokens)!r}")
    token_match = cpu_tokens == npu_tokens
    print(f"Token match: {'ALL MATCH' if token_match else 'MISMATCH'}")
    print("=" * 70)

    return all_ok


def main():
    print("1/3: Running CPU inference...")
    cpu_logits, cpu_tokens, tokenizer = get_cpu_logits(PROMPT, MAX_DECODE_STEPS)
    print(f"     CPU tokens: {cpu_tokens} = {tokenizer.decode(cpu_tokens)!r}")

    print("2/3: Running NPU inference...")
    npu_logits, npu_tokens = get_npu_logits(PROMPT, MAX_DECODE_STEPS, tokenizer)
    print(f"     NPU tokens: {npu_tokens} = {tokenizer.decode(npu_tokens)!r}")

    print("3/3: Comparing logits...")
    all_ok = compare_logits(cpu_logits, npu_logits, cpu_tokens, npu_tokens, tokenizer)

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
