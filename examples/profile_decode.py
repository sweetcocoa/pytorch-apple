"""Profile decode step breakdown to find bottlenecks."""
from __future__ import annotations
import json, time, sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hf_utils import load_weights_from_hf
import npu_compiler
from npu_runtime.buffer import NPUBuffer
from npu_runtime.device import Device
from npu_runtime.executor import Executor

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

def main():
    # Compile
    prefill_program = npu_compiler.compile("qwen2_prefill_ir.json")
    decode_program = npu_compiler.compile("qwen2_decode_ir.json")
    device = Device()

    # Load weights
    prefill_weights = load_weights_from_hf(MODEL_ID, prefill_program, device)
    decode_weights = load_weights_from_hf(MODEL_ID, decode_program, device)

    with open("qwen2_kv_mapping.json") as f:
        kv_map = json.load(f)

    # Tokenize
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    input_ids = tokenizer.encode("Hello", return_tensors="np")[0]

    # Prefill
    prefill_seq_len = prefill_program.input_specs[0].shape[1]
    padded_ids = np.zeros((1, prefill_seq_len), dtype=np.int64)
    actual_len = min(len(input_ids), prefill_seq_len)
    padded_ids[0, :actual_len] = input_ids[:actual_len]
    _NEG_INF = np.float16(-np.inf)
    causal_mask = np.triu(np.full((1, 1, prefill_seq_len, prefill_seq_len), _NEG_INF, dtype=np.float16), k=1)
    causal_mask[:, :, :, actual_len:] = _NEG_INF
    causal_mask[:, :, actual_len:, :] = _NEG_INF
    position_ids = np.arange(prefill_seq_len, dtype=np.int64).reshape(1, -1)
    cache_position = np.arange(prefill_seq_len, dtype=np.int64)
    np_map = {"input_ids": padded_ids, "attention_mask": causal_mask,
              "position_ids": position_ids, "cache_position": cache_position}
    inputs = {spec.name: NPUBuffer.from_numpy(np_map[spec.name], device, spec=spec)
              for spec in prefill_program.input_specs}

    prefill_executor = Executor(prefill_program, device)
    decode_executor = Executor(decode_program, device)

    prefill_outputs = prefill_executor.run(inputs=inputs, weights=prefill_weights)
    logits_name = prefill_program.output_specs[0].name
    logits = prefill_outputs[logits_name].to_numpy(spec=prefill_program.output_specs[0])
    next_token = int(np.argmax(logits[0, actual_len - 1, :]))

    max_cache_len = kv_map["max_cache_len"]

    # Pad prefill KV to max_cache_len for decode
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

    # Profile one decode step
    print(f"Decode program: {len(decode_program.kernel_calls)} kernel calls")
    print(f"  Input specs: {len(decode_program.input_specs)}")

    # Count kernel types
    kernel_counts = {}
    for call in decode_program.kernel_calls:
        kernel_counts[call.kernel_name] = kernel_counts.get(call.kernel_name, 0) + 1
    print("\nKernel type counts:")
    for name, count in sorted(kernel_counts.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count}")

    # Prepare decode inputs
    cur_pos = actual_len
    token_arr = np.array([[next_token]], dtype=np.int64)
    decode_mask = np.full((1, 1, 1, max_cache_len), _NEG_INF, dtype=np.float16)
    decode_mask[0, 0, 0, :cur_pos + 1] = 0.0
    decode_pos_ids = np.array([[cur_pos]], dtype=np.int64)
    decode_cache_pos = np.array([cur_pos], dtype=np.int64)

    decode_np_map = {"input_ids": token_arr, "attention_mask": decode_mask,
                     "position_ids": decode_pos_ids, "cache_position": decode_cache_pos}

    # Time: input preparation
    t0 = time.time()
    decode_inputs = {}
    for spec in decode_program.input_specs:
        if spec.name in decode_np_map:
            decode_inputs[spec.name] = NPUBuffer.from_numpy(decode_np_map[spec.name], device, spec=spec)
        elif spec.name in kv_buffers:
            decode_inputs[spec.name] = kv_buffers[spec.name]
    t_prep = time.time() - t0

    # Time: executor.run()
    t0 = time.time()
    decode_outputs = decode_executor.run(inputs=decode_inputs, weights=decode_weights)
    t_run = time.time() - t0

    # Time: output extraction
    t0 = time.time()
    dec_logits_name = decode_program.output_specs[0].name
    dec_logits = decode_outputs[dec_logits_name].to_numpy(spec=decode_program.output_specs[0])
    next_token2 = int(np.argmax(dec_logits[0, -1, :]))
    t_extract = time.time() - t0

    print(f"\nDecode step timing:")
    print(f"  Input prep:     {t_prep*1000:.1f}ms")
    print(f"  executor.run(): {t_run*1000:.1f}ms")
    print(f"  Output extract: {t_extract*1000:.1f}ms")
    print(f"  Total:          {(t_prep+t_run+t_extract)*1000:.1f}ms")
    print(f"  Token: {next_token2} ({tokenizer.decode([next_token2])})")

    # Run 5 more steps for average
    times = []
    for step in range(5):
        cur_pos = actual_len + step + 1
        token_arr = np.array([[next_token2]], dtype=np.int64)
        decode_mask = np.full((1, 1, 1, max_cache_len), _NEG_INF, dtype=np.float16)
        decode_mask[0, 0, 0, :cur_pos + 1] = 0.0
        decode_np_map = {"input_ids": token_arr, "attention_mask": decode_mask,
                         "position_ids": np.array([[cur_pos]], dtype=np.int64),
                         "cache_position": np.array([cur_pos], dtype=np.int64)}

        decode_inputs = {}
        for spec in decode_program.input_specs:
            if spec.name in decode_np_map:
                decode_inputs[spec.name] = NPUBuffer.from_numpy(decode_np_map[spec.name], device, spec=spec)
            elif spec.name in kv_buffers:
                decode_inputs[spec.name] = kv_buffers[spec.name]

        t0 = time.time()
        decode_outputs = decode_executor.run(inputs=decode_inputs, weights=decode_weights)
        t_run = time.time() - t0
        times.append(t_run)

        dec_logits = decode_outputs[dec_logits_name].to_numpy(spec=decode_program.output_specs[0])
        next_token2 = int(np.argmax(dec_logits[0, -1, :]))

        for layer in kv_map["layers"]:
            kv_buffers[layer["decode_key_input"]] = decode_outputs[layer["decode_key_output"]]
            kv_buffers[layer["decode_value_input"]] = decode_outputs[layer["decode_value_output"]]

    print(f"\n5 decode steps run() times: {[f'{t*1000:.0f}ms' for t in times]}")
    print(f"  Average: {sum(times)/len(times)*1000:.0f}ms")


if __name__ == "__main__":
    main()
