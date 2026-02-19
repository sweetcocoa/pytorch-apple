"""Run Qwen2.5-1.5B-Instruct on NPU via graph partition pipeline (DAGExecutor).

Usage:
    python examples/run_qwen_graph.py
    python examples/run_qwen_graph.py --prompt "The capital of France is"
    python examples/run_qwen_graph.py --force-cpu-ops aten.mul.Tensor
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import ml_dtypes  # noqa: F401
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "torch_to_ir"))

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def load_numpy_weights(model_id: str, ir_dict: dict) -> dict[str, np.ndarray]:
    """Load HuggingFace weights as bfloat16 numpy arrays for DAGExecutor."""
    import glob as _glob

    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file as _load_file_torch

    if os.path.isdir(model_id):
        model_dir = model_id
    else:
        model_dir = snapshot_download(model_id, allow_patterns=["*.safetensors", "*.json"])

    def _to_bf16_numpy(v: torch.Tensor) -> np.ndarray:
        if v.dtype != torch.bfloat16:
            v = v.to(torch.bfloat16)
        return v.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)

    state_dict: dict[str, np.ndarray] = {}
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f).get("weight_map", {})
        for shard_file in set(weight_map.values()):
            tensors = _load_file_torch(os.path.join(model_dir, shard_file), device="cpu")
            for k, v in tensors.items():
                state_dict[k] = _to_bf16_numpy(v)
    else:
        for st_file in sorted(_glob.glob(os.path.join(model_dir, "*.safetensors"))):
            tensors = _load_file_torch(st_file, device="cpu")
            for k, v in tensors.items():
                state_dict[k] = _to_bf16_numpy(v)

    needed_keys = {w["name"] for w in ir_dict.get("weights", [])}
    remapped: dict[str, np.ndarray] = {}
    for ir_key in needed_keys:
        if ir_key in state_dict:
            remapped[ir_key] = state_dict[ir_key]
        elif ir_key.startswith("model.") and ir_key[len("model.") :] in state_dict:
            remapped[ir_key] = state_dict[ir_key[len("model.") :]]

    for key in needed_keys:
        if key not in remapped and "lm_head.weight" in key:
            embed_key = key.replace("lm_head.weight", "model.embed_tokens.weight")
            if embed_key in remapped:
                remapped[key] = remapped[embed_key]

    for w in ir_dict.get("weights", []):
        if w["name"] in remapped:
            continue
        if "inv_freq" in w["name"]:
            dim = w["shape"][0]
            inv_freq = 1.0 / (10000.0 ** (np.arange(dim, dtype=np.float32) / dim))
            remapped[w["name"]] = inv_freq.astype(ml_dtypes.bfloat16)
        elif "lifted_tensor" in w["name"] and w["shape"] == [0]:
            remapped[w["name"]] = np.array([], dtype=ml_dtypes.bfloat16)

    return remapped


def extract_prefill_ir(prompt_len: int) -> tuple[dict, object]:
    """Extract Qwen prefill IR and return (ir_dict, tokenizer)."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from torch_ir import extract_ir

    from extract_qwen_ir import QwenPrefillWrapper, _clean_ir_attrs

    from contextlib import nullcontext
    import transformers.models.qwen2.modeling_qwen2 as _qwen2_mod

    _qwen2_mod.maybe_autocast = lambda **_kwargs: nullcontext()

    config = AutoConfig.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    with torch.device("meta"):
        meta_model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    meta_model.eval()

    wrapper = QwenPrefillWrapper(meta_model)
    meta_input = (
        torch.randint(0, config.vocab_size, (1, prompt_len), device="meta"),
        torch.zeros(1, 1, prompt_len, prompt_len, device="meta"),
        torch.arange(prompt_len, device="meta").unsqueeze(0),
        torch.arange(prompt_len, device="meta"),
    )

    ir = extract_ir(wrapper, meta_input, model_name="Qwen2.5_Prefill_Graph")
    _clean_ir_attrs(ir)

    import tempfile

    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
    ir.save(tmp.name)
    tmp.close()
    with open(tmp.name) as f:
        ir_dict = json.load(f)
    os.unlink(tmp.name)

    return ir_dict, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Run Qwen2.5 on NPU via graph partition pipeline")
    parser.add_argument("--prompt", default="Hello, how are you?", help="Input prompt")
    parser.add_argument("--model-id", default=MODEL_ID, help="HuggingFace model ID or local path")
    parser.add_argument(
        "--force-cpu-ops",
        nargs="*",
        default=[],
        help="Force specific op types to run on CPU (e.g. aten.mul.Tensor)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Qwen2.5-1.5B on NPU — Graph Partition Pipeline")
    print("=" * 60)

    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt")
    actual_len = input_ids.shape[1]
    print(f"\nPrompt: {args.prompt}")
    print(f"Tokens: {actual_len}")

    print("\nExtracting IR...")
    t0 = time.time()
    ir_dict, _ = extract_prefill_ir(actual_len)
    print(f"  IR extraction: {time.time() - t0:.2f}s")
    print(f"  Nodes: {len(ir_dict['nodes'])}")
    print(f"  Weights: {len(ir_dict['weights'])}")

    from npu_compiler.op_support import is_op_supported
    from npu_compiler.partitioner import Partition, TransferOp, partition

    force_cpu = set(args.force_cpu_ops)

    def support_fn(op_type, _attrs=None):
        if op_type in force_cpu:
            return False
        return is_op_supported(op_type)

    print("\nPartitioning...")
    t0 = time.time()
    plan = partition(ir_dict, support_fn)
    partition_time = time.time() - t0

    partitions = [s for s in plan.steps if isinstance(s, Partition)]
    transfers = [s for s in plan.steps if isinstance(s, TransferOp)]
    npu_parts = [p for p in partitions if p.target == "npu"]
    cpu_parts = [p for p in partitions if p.target == "cpu"]

    print(f"  Partition time: {partition_time:.3f}s")
    print(f"  Partitions: {len(partitions)} ({len(npu_parts)} NPU, {len(cpu_parts)} CPU)")
    print(f"  Transfer ops: {len(transfers)}")
    show_limit = 10
    for p in partitions[:show_limit]:
        print(f"    [{p.target.upper()}] partition {p.partition_id}: {len(p.nodes)} nodes")
    if len(partitions) > show_limit:
        print(f"    ... ({len(partitions) - show_limit} more partitions)")

    from npu_runtime.dag_executor import DAGExecutor
    from npu_runtime.metal_backend import MetalBackend

    print("\nCompiling partitions...")
    t0 = time.time()
    backend = MetalBackend()
    dag = DAGExecutor(plan, backend)
    print(f"  Compile time: {time.time() - t0:.2f}s")
    print(f"  Metal device: {backend.device.name}")

    print(f"\nLoading weights from '{args.model_id}'...")
    t0 = time.time()
    weights_np = load_numpy_weights(args.model_id, ir_dict)
    dag.load_weights(weights_np)
    print(f"  Weight load time: {time.time() - t0:.2f}s")
    print(f"  Weight tensors: {len(weights_np)}")

    input_np = input_ids.numpy().astype(np.int64)
    causal_mask = np.triu(np.full((1, 1, actual_len, actual_len), -3.389e38, dtype=np.float16), k=1)
    position_ids_np = np.arange(actual_len, dtype=np.int64).reshape(1, -1)
    cache_position_np = np.arange(actual_len, dtype=np.int64)

    inputs = {
        "input_ids": input_np,
        "attention_mask": causal_mask,
        "position_ids": position_ids_np,
        "cache_position": cache_position_np,
    }

    print("\nRunning prefill on NPU...")
    t0 = time.time()
    result = dag.execute(inputs=inputs)
    npu_time = time.time() - t0
    print(f"  NPU prefill time: {npu_time:.3f}s")

    npu_logits = list(result.values())[0]
    next_token = int(np.argmax(npu_logits[0, actual_len - 1, :]))
    print(f"  Next token: {next_token} ({tokenizer.decode([next_token])})")

    print("\nRunning CPU reference...")
    from transformers import AutoModelForCausalLM

    cpu_model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float32)
    cpu_model.eval()

    t0 = time.time()
    with torch.no_grad():
        cpu_logits = cpu_model(input_ids).logits.numpy()
    cpu_time = time.time() - t0
    print(f"  CPU prefill time: {cpu_time:.3f}s")

    cpu_next = int(np.argmax(cpu_logits[0, actual_len - 1, :]))
    print(f"  Next token: {cpu_next} ({tokenizer.decode([cpu_next])})")

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    print(f"\n  Top-1 agreement: {'YES' if next_token == cpu_next else 'NO'}")

    for pos in range(actual_len):
        cpu_top5 = set(np.argsort(cpu_logits[0, pos])[-5:])
        npu_top5 = set(np.argsort(npu_logits[0, pos])[-5:])
        overlap = len(cpu_top5 & npu_top5)
        print(f"  Position {pos}: top-5 overlap {overlap}/5")

    print(f"\n  NPU time: {npu_time:.3f}s")
    print(f"  CPU time: {cpu_time:.3f}s")

    if force_cpu:
        print(f"\n  Forced CPU ops: {sorted(force_cpu)}")
        print(f"  Partitions: {len(npu_parts)} NPU + {len(cpu_parts)} CPU")

    print()


if __name__ == "__main__":
    main()
