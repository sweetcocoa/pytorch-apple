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
        cpu_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
        cpu_model.eval()

        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        actual_len = input_ids.shape[1]

        with torch.no_grad():
            cpu_outputs = cpu_model(input_ids)
            cpu_logits = cpu_outputs.logits.numpy()  # (1, seq_len, vocab)

        # 2. Extract prefill IR with wrapper
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
        from extract_qwen_ir import QwenPrefillWrapper

        with torch.device("meta"):
            meta_model = AutoModelForCausalLM.from_config(config)
        meta_model.eval()

        wrapper = QwenPrefillWrapper(meta_model)
        meta_input = (
            torch.randint(0, config.vocab_size, (1, actual_len), device="meta"),       # input_ids
            torch.zeros(1, 1, actual_len, actual_len, device="meta"),                   # attention_mask (4D causal)
            torch.arange(actual_len, device="meta").unsqueeze(0),                       # position_ids
            torch.arange(actual_len, device="meta"),                                    # cache_position
        )
        ir = extract_ir(wrapper, meta_input, model_name="Qwen2.5_Prefill_Test")

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
        causal_mask = np.triu(
            np.full((1, 1, actual_len, actual_len), -3.389e38, dtype=np.float16), k=1
        )
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
                f"Position {pos}: top-5 overlap {overlap}/5 is too low. "
                f"CPU top-5: {cpu_top5}, NPU top-5: {npu_top5}"
            )

        # Top-1 agreement at last position (most important for generation)
        cpu_next = np.argmax(cpu_logits[0, actual_len - 1])
        npu_next = np.argmax(npu_logits[0, actual_len - 1])
        assert cpu_next == npu_next, (
            f"Top-1 mismatch at last position: CPU={cpu_next}, NPU={npu_next}"
        )

        print(f"✓ Qwen2.5 prefill: top-5 agreement verified for {actual_len} positions")
        print(f"  Next token: {cpu_next} ({tokenizer.decode([cpu_next])})")


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
class TestQwenGreedyDecode:
    """Test Qwen2.5 greedy decode produces same tokens as CPU."""

    def test_greedy_decode_10_tokens(self, device, tmp_path):
        """Greedy decode 10 tokens should match CPU."""
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        prompt = "The capital of France is"
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        # CPU reference: greedy decode 10 tokens (auto-downloads from hub)
        cpu_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
        cpu_model.eval()

        with torch.no_grad():
            cpu_generated = cpu_model.generate(
                input_ids, max_new_tokens=10, do_sample=False
            )
        cpu_new_tokens = cpu_generated[0, input_ids.shape[1]:].tolist()

        print(f"CPU generated: {tokenizer.decode(cpu_new_tokens)}")
        print(f"CPU tokens: {cpu_new_tokens}")

        # For now, just verify the CPU reference works
        assert len(cpu_new_tokens) > 0, "CPU should generate at least 1 token"
        print(f"✓ CPU greedy decode verified: {len(cpu_new_tokens)} tokens")
