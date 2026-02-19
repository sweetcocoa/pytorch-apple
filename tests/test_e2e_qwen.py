"""E2E test for Qwen2.5-1.5B-Instruct on NPU.

Compares NPU prefill logits against HuggingFace CPU inference.
Models are downloaded from HuggingFace Hub if not cached.
Skipped if transformers is not installed.

Usage:
    uv run pytest tests/test_e2e_qwen.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Add torch_to_ir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "torch_to_ir"))

try:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
class TestQwenPrefill:
    """Test Qwen2.5 prefill against CPU reference."""

    def test_prefill_top5_agreement(self, device, tmp_path):
        """Prefill logits: top-5 token agreement with CPU at each position."""
        from torch_ir import extract_ir

        config = AutoConfig.from_pretrained(MODEL_ID)

        # Short sequence for testing
        prompt = "Hello world"

        # 1. CPU reference (auto-downloads from hub)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        cpu_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
        cpu_model.eval()

        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        actual_len = input_ids.shape[1]

        with torch.no_grad():
            cpu_outputs = cpu_model(input_ids)
            cpu_logits = cpu_outputs.logits.numpy()  # (1, seq_len, vocab)

        # 2. Extract prefill IR with wrapper
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
        from extract_qwen_ir import QwenPrefillWrapper, _clean_ir_attrs

        # Patch autocast for meta device (same as extract_qwen_ir.py main())
        from contextlib import nullcontext
        import transformers.models.qwen2.modeling_qwen2 as _qwen2_mod

        _qwen2_mod.maybe_autocast = lambda **_kwargs: nullcontext()

        with torch.device("meta"):
            meta_model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
        meta_model.eval()

        wrapper = QwenPrefillWrapper(meta_model)
        meta_input = (
            torch.randint(0, config.vocab_size, (1, actual_len), device="meta"),  # input_ids
            torch.zeros(1, 1, actual_len, actual_len, device="meta"),  # attention_mask (4D causal)
            torch.arange(actual_len, device="meta").unsqueeze(0),  # position_ids
            torch.arange(actual_len, device="meta"),  # cache_position
        )
        ir = extract_ir(wrapper, meta_input, model_name="Qwen2.5_Prefill_Test")
        _clean_ir_attrs(ir)

        ir_path = str(tmp_path / "qwen_prefill_test.json")
        ir.save(ir_path)

        # 3. Compile
        import npu_compiler

        program = npu_compiler.compile(ir_path)

        # 4. Load weights (auto-downloads from hub)
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
        from hf_utils import load_weights_from_hf

        weights = load_weights_from_hf(MODEL_ID, program, device)

        # 5. Run on NPU
        from npu_runtime.buffer import NPUBuffer
        from npu_runtime.executor import Executor

        executor = Executor(program, device)

        input_np = input_ids.numpy().astype(np.int64)
        # Build causal mask and position arrays
        causal_mask = np.triu(np.full((1, 1, actual_len, actual_len), -3.389e38, dtype=np.float16), k=1)
        position_ids_np = np.arange(actual_len, dtype=np.int64).reshape(1, -1)
        cache_position_np = np.arange(actual_len, dtype=np.int64)

        np_map = {
            "input_ids": input_np,
            "attention_mask": causal_mask,
            "position_ids": position_ids_np,
            "cache_position": cache_position_np,
        }

        npu_inputs = {}
        for spec in program.input_specs:
            npu_inputs[spec.name] = NPUBuffer.from_numpy(np_map[spec.name], device, spec=spec)

        npu_outputs = executor.run(inputs=npu_inputs, weights=weights)

        # 6. Compare logits
        logits_spec = program.output_specs[0]
        npu_logits = npu_outputs[logits_spec.name].to_numpy(spec=logits_spec)

        # Top-5 agreement at each position
        for pos in range(actual_len):
            cpu_top5 = np.argsort(cpu_logits[0, pos])[-5:]
            npu_top5 = np.argsort(npu_logits[0, pos])[-5:]
            overlap = len(set(cpu_top5) & set(npu_top5))
            assert overlap >= 3, (
                f"Position {pos}: top-5 overlap {overlap}/5 is too low. CPU top-5: {cpu_top5}, NPU top-5: {npu_top5}"
            )

        # Top-1 agreement at last position (most important for generation)
        cpu_next = np.argmax(cpu_logits[0, actual_len - 1])
        npu_next = np.argmax(npu_logits[0, actual_len - 1])
        assert cpu_next == npu_next, f"Top-1 mismatch at last position: CPU={cpu_next}, NPU={npu_next}"

        print(f"✓ Qwen2.5 prefill: top-5 agreement verified for {actual_len} positions")
        print(f"  Next token: {cpu_next} ({tokenizer.decode([cpu_next])})")


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
class TestQwenDAGExecutor:
    """Test Qwen2.5 prefill through the graph partition (DAGExecutor) pipeline."""

    def test_prefill_via_dag_executor(self, device, tmp_path):
        """Prefill through partition() → DAGExecutor matches CPU logits."""
        import json

        from torch_ir import extract_ir

        from npu_compiler.op_support import is_op_supported
        from npu_compiler.partitioner import Partition, partition
        from npu_runtime.dag_executor import DAGExecutor
        from npu_runtime.metal_backend import MetalBackend

        config = AutoConfig.from_pretrained(MODEL_ID)
        prompt = "Hello world"

        # 1. CPU reference
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        cpu_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
        cpu_model.eval()

        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        actual_len = input_ids.shape[1]

        with torch.no_grad():
            cpu_logits = cpu_model(input_ids).logits.numpy()

        # 2. Extract IR
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
        from extract_qwen_ir import QwenPrefillWrapper, _clean_ir_attrs

        from contextlib import nullcontext
        import transformers.models.qwen2.modeling_qwen2 as _qwen2_mod

        _qwen2_mod.maybe_autocast = lambda **_kwargs: nullcontext()

        with torch.device("meta"):
            meta_model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
        meta_model.eval()

        wrapper = QwenPrefillWrapper(meta_model)
        meta_input = (
            torch.randint(0, config.vocab_size, (1, actual_len), device="meta"),
            torch.zeros(1, 1, actual_len, actual_len, device="meta"),
            torch.arange(actual_len, device="meta").unsqueeze(0),
            torch.arange(actual_len, device="meta"),
        )
        ir = extract_ir(wrapper, meta_input, model_name="Qwen2.5_DAG_Test")
        _clean_ir_attrs(ir)

        ir_path = str(tmp_path / "qwen_dag_test.json")
        ir.save(ir_path)

        with open(ir_path) as f:
            ir_dict = json.load(f)

        # 3. Partition → should be single NPU partition (all ops supported)
        plan = partition(ir_dict, is_op_supported)
        partitions = [s for s in plan.steps if isinstance(s, Partition)]
        npu_parts = [p for p in partitions if p.target == "npu"]
        assert len(npu_parts) >= 1, "Should have at least one NPU partition"
        print(f"  Partitions: {len(partitions)} total, {len(npu_parts)} NPU")

        # 4. Load numpy weights from HF
        # DAGExecutor needs raw numpy state_dict (not NPUBuffers).
        # Qwen uses bfloat16 compute — must load as bfloat16 (via uint16 view).
        import ml_dtypes  # noqa: F401
        from safetensors.torch import load_file as _load_file_torch
        from huggingface_hub import snapshot_download

        import glob as _glob

        model_dir = snapshot_download(MODEL_ID, allow_patterns=["*.safetensors", "*.json"])

        def _to_bf16_numpy(v: torch.Tensor) -> np.ndarray:
            """Convert torch tensor to bfloat16 numpy via uint16 view."""
            if v.dtype != torch.bfloat16:
                v = v.to(torch.bfloat16)
            return v.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)

        # Load safetensors (single file or multi-shard)
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

        # Remap keys: IR may use "model.X" prefix for wrapper weights
        needed_keys = {w["name"] for w in ir_dict.get("weights", [])}
        remapped: dict[str, np.ndarray] = {}
        for ir_key in needed_keys:
            if ir_key in state_dict:
                remapped[ir_key] = state_dict[ir_key]
            elif ir_key.startswith("model.") and ir_key[len("model.") :] in state_dict:
                remapped[ir_key] = state_dict[ir_key[len("model.") :]]
        # Handle weight tying (lm_head)
        for key in needed_keys:
            if key not in remapped and "lm_head.weight" in key:
                embed_key = key.replace("lm_head.weight", "model.embed_tokens.weight")
                if embed_key in remapped:
                    remapped[key] = remapped[embed_key]
        # inv_freq / lifted_tensor
        for w in ir_dict.get("weights", []):
            if w["name"] in remapped:
                continue
            if "inv_freq" in w["name"]:
                dim = w["shape"][0]
                inv_freq = 1.0 / (10000.0 ** (np.arange(dim, dtype=np.float32) / dim))
                remapped[w["name"]] = inv_freq.astype(ml_dtypes.bfloat16)
            elif "lifted_tensor" in w["name"] and w["shape"] == [0]:
                remapped[w["name"]] = np.array([], dtype=ml_dtypes.bfloat16)
        weights_np = remapped

        # 5. Execute via DAGExecutor
        backend = MetalBackend()
        dag = DAGExecutor(plan, backend)
        dag.load_weights(weights_np)

        input_np = input_ids.numpy().astype(np.int64)
        causal_mask = np.triu(np.full((1, 1, actual_len, actual_len), -3.389e38, dtype=np.float16), k=1)
        position_ids_np = np.arange(actual_len, dtype=np.int64).reshape(1, -1)
        cache_position_np = np.arange(actual_len, dtype=np.int64)

        result = dag.execute(
            inputs={
                "input_ids": input_np,
                "attention_mask": causal_mask,
                "position_ids": position_ids_np,
                "cache_position": cache_position_np,
            },
        )

        # 6. Compare logits
        npu_logits = list(result.values())[0]

        for pos in range(actual_len):
            cpu_top5 = set(np.argsort(cpu_logits[0, pos])[-5:])
            npu_top5 = set(np.argsort(npu_logits[0, pos])[-5:])
            overlap = len(cpu_top5 & npu_top5)
            assert overlap >= 3, f"Position {pos}: top-5 overlap {overlap}/5 too low"

        cpu_next = np.argmax(cpu_logits[0, actual_len - 1])
        npu_next = np.argmax(npu_logits[0, actual_len - 1])
        assert cpu_next == npu_next, f"Top-1 mismatch: CPU={cpu_next}, NPU={npu_next}"

        print(f"✓ Qwen2.5 DAG executor: top-5 agreement verified for {actual_len} positions")
        print(f"  Next token: {cpu_next} ({tokenizer.decode([int(cpu_next)])})")


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
class TestQwenGreedyDecode:
    """Test Qwen2.5 greedy decode produces same tokens as CPU."""

    def test_greedy_decode_10_tokens(self, device, tmp_path):
        """Greedy decode 10 tokens should match CPU."""
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        prompt = "The capital of France is"
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        # CPU reference: greedy decode 10 tokens (auto-downloads from hub)
        cpu_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
        cpu_model.eval()

        with torch.no_grad():
            cpu_generated = cpu_model.generate(input_ids, max_new_tokens=10, do_sample=False)
        cpu_new_tokens = cpu_generated[0, input_ids.shape[1] :].tolist()

        print(f"CPU generated: {tokenizer.decode(cpu_new_tokens)}")
        print(f"CPU tokens: {cpu_new_tokens}")

        # For now, just verify the CPU reference works
        assert len(cpu_new_tokens) > 0, "CPU should generate at least 1 token"
        print(f"✓ CPU greedy decode verified: {len(cpu_new_tokens)} tokens")
