"""Run Qwen2.5-1.5B-Instruct on NPU (Metal GPU).

Uses only the generic compiler/runtime API — no LLM-specific code in the backend.
KV-cache is handled as regular NPUBuffer I/O.
Model weights are automatically downloaded from HuggingFace Hub if not cached.

Prerequisites:
    1. Run extract_qwen_ir.py first to generate IR files.
    2. pip install transformers huggingface_hub safetensors

Usage:
    python examples/run_qwen.py --prompt "Hello, how are you?"
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from hf_utils import load_weights_from_hf

import npu_compiler
from npu_runtime.buffer import NPUBuffer
from npu_runtime.device import Device
from npu_runtime.executor import Executor

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def main():
    parser = argparse.ArgumentParser(description="Run Qwen2.5 on NPU")
    parser.add_argument("--prompt", default="Hello, how are you?", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=32, help="Max new tokens to generate")
    parser.add_argument("--model-id", default=MODEL_ID, help="HuggingFace model ID or local path")
    parser.add_argument("--prefill-ir", default="qwen2_prefill_ir.json", help="Prefill IR path")
    parser.add_argument("--decode-ir", default="qwen2_decode_ir.json", help="Decode IR path")
    parser.add_argument("--kv-mapping", default="qwen2_kv_mapping.json", help="KV mapping path")
    args = parser.parse_args()

    print("=" * 60)
    print("Qwen2.5-1.5B-Instruct on NPU (Metal GPU)")
    print("=" * 60)

    # 1. Tokenize
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    input_ids = tokenizer.encode(args.prompt, return_tensors="np")[0]  # (seq_len,)
    print(f"\nPrompt: {args.prompt}")
    print(f"Token IDs: {input_ids[:10]}... (len={len(input_ids)})")

    # 2. Compile
    print("\nCompiling...")
    t0 = time.time()
    prefill_program = npu_compiler.compile(args.prefill_ir)
    decode_program = npu_compiler.compile(args.decode_ir)
    print(f"  Compile time: {time.time() - t0:.2f}s")
    print(f"  Prefill kernels: {len(prefill_program.kernel_calls)}")
    print(f"  Decode kernels: {len(decode_program.kernel_calls)}")

    # 3. Load weights (auto-downloads from HuggingFace Hub if not cached)
    print(f"\nLoading weights from '{args.model_id}'...")
    device = Device()
    t0 = time.time()
    prefill_weights = load_weights_from_hf(args.model_id, prefill_program, device)
    decode_weights = load_weights_from_hf(args.model_id, decode_program, device)
    print(f"  Load time: {time.time() - t0:.2f}s")
    print(f"  Prefill weight buffers: {len(prefill_weights)}")
    print(f"  Decode weight buffers: {len(decode_weights)}")

    # 4. Load KV mapping
    with open(args.kv_mapping) as f:
        kv_map = json.load(f)
    # Infer max_cache_len from decode program's first KV input spec
    first_kv_name = kv_map["layers"][0]["decode_key_input"]
    max_cache_len = next(s.shape[2] for s in decode_program.input_specs if s.name == first_kv_name)
    print(f"\nKV mapping: {kv_map['num_layers']} layers, max_cache_len={max_cache_len}")

    # 5. Create executors
    prefill_executor = Executor(prefill_program, device)
    decode_executor = Executor(decode_program, device)

    # 6. Prefill
    print("\nPrefill...")
    # Pad input_ids to prefill seq_len
    prefill_seq_len = prefill_program.input_specs[0].shape[1]
    padded_ids = np.zeros((1, prefill_seq_len), dtype=np.int64)
    actual_len = min(len(input_ids), prefill_seq_len)
    padded_ids[0, :actual_len] = input_ids[:actual_len]

    # Build causal attention mask (4D) with padding masking
    _NEG_INF = np.float16(-np.inf)
    causal_mask = np.triu(
        np.full((1, 1, prefill_seq_len, prefill_seq_len), _NEG_INF, dtype=np.float16), k=1
    )
    # Mask out padding positions (queries and keys beyond actual_len)
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
        data = prefill_np_map[spec.name]
        prefill_inputs[spec.name] = NPUBuffer.from_numpy(data, device, spec=spec)

    t0 = time.time()
    prefill_outputs = prefill_executor.run(inputs=prefill_inputs, weights=prefill_weights)
    prefill_time = time.time() - t0
    print(f"  Prefill time: {prefill_time:.3f}s")

    # Extract logits and KV cache from outputs
    logits_name = prefill_program.output_specs[0].name
    logits = prefill_outputs[logits_name].to_numpy(spec=prefill_program.output_specs[0])

    # Get next token from last position
    next_token = int(np.argmax(logits[0, actual_len - 1, :]))
    generated = [next_token]
    print(f"  First token: {next_token} ({tokenizer.decode([next_token])})")

    # 7. Decode loop
    print(f"\nDecoding (max {args.max_tokens} tokens)...")

    # Pad prefill KV from (B,H,prefill_seq_len,D) to (B,H,max_cache_len,D) for decode
    _FIXED_INPUTS = {"input_ids", "attention_mask", "position_ids", "cache_position"}
    decode_kv_specs = {s.name: s for s in decode_program.input_specs if s.name not in _FIXED_INPUTS}

    kv_buffers = {}
    for layer in kv_map["layers"]:
        for pkey, dkey in [(layer["prefill_key_output"], layer["decode_key_input"]),
                           (layer["prefill_value_output"], layer["decode_value_input"])]:
            prefill_kv = prefill_outputs[pkey]
            decode_spec = decode_kv_specs[dkey]
            # Prefill KV shape: (B, H, prefill_seq_len, D) → pad to (B, H, max_cache_len, D)
            prefill_spec = next(s for s in prefill_program.output_specs if s.name == pkey)
            kv_np = prefill_kv.to_numpy(spec=prefill_spec)
            padded_kv = np.zeros(decode_spec.shape, dtype=kv_np.dtype)
            padded_kv[:, :, :prefill_seq_len, :] = kv_np
            kv_buffers[dkey] = NPUBuffer.from_numpy(padded_kv, device, spec=decode_spec)

    t0 = time.time()
    for step in range(args.max_tokens - 1):
        cur_pos = actual_len + step
        # Prepare decode input
        token_arr = np.array([[next_token]], dtype=np.int64)
        # Decode attention mask: (1, 1, 1, max_cache_len)
        # Unmask positions 0..cur_pos (inclusive), mask the rest
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

        # Extract logits
        dec_logits_name = decode_program.output_specs[0].name
        dec_logits = decode_outputs[dec_logits_name].to_numpy(spec=decode_program.output_specs[0])
        next_token = int(np.argmax(dec_logits[0, -1, :]))

        if next_token == tokenizer.eos_token_id:
            break

        generated.append(next_token)

        # Update KV cache: decode output → next step's input (same shape, no padding needed)
        for layer in kv_map["layers"]:
            kv_buffers[layer["decode_key_input"]] = decode_outputs[layer["decode_key_output"]]
            kv_buffers[layer["decode_value_input"]] = decode_outputs[layer["decode_value_output"]]

    decode_time = time.time() - t0
    print(f"  Decode time: {decode_time:.3f}s ({len(generated)} tokens)")
    print(f"  Tokens/sec: {len(generated) / decode_time:.1f}")

    # 8. Print result
    output_text = tokenizer.decode(generated)
    print("\n--- Generated ---")
    print(f"{args.prompt}{output_text}")
    print("--- End ---")


if __name__ == "__main__":
    main()
