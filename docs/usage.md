# Usage Guide

## Workflow Overview

```mermaid
graph LR
    A[PyTorch Model] -->|torch_to_ir| B[IR JSON]
    B -->|compile_ir| C[.npubin]
    C -->|Executor| D[GPU Output]
```

## ResNet Example

### Extract IR and Run

```python
import torch
import torchvision.models as models
from torch_ir import extract_ir

# Load model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval()
example = torch.randn(1, 3, 224, 224)

# Extract IR
ir = extract_ir(model, example, model_name="resnet18")
ir.save("resnet18_ir.json")
```

### Compile and Execute

```python
from npu_compiler import compile_ir
from npu_compiler.compiled_program import CompiledProgram
from npu_runtime.device import Device
from npu_runtime.executor import Executor
from npu_runtime.weight_loader import load_weights
from npu_runtime.buffer import NPUBuffer

# Compile
program = compile_ir("resnet18_ir.json")
program.save("resnet18.npubin")

# Setup runtime
device = Device()
program = CompiledProgram.load("resnet18.npubin")
executor = Executor(program, device)
weights = load_weights("resnet18.safetensors", program, device)

# Run inference
import numpy as np
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
input_buf = NPUBuffer.from_numpy(input_data, device, spec=program.input_specs[0])
outputs = executor.run(inputs={program.input_specs[0].name: input_buf}, weights=weights)
logits = outputs[program.output_specs[0].name].to_numpy(spec=program.output_specs[0])
```

## Qwen2.5-1.5B Example

Qwen uses a two-phase approach: **prefill** (process prompt) and **decode** (generate tokens).

### Prefill Phase
```python
# Compile prefill IR (variable sequence length)
prefill_program = compile_ir("qwen_prefill_ir.json")
prefill_executor = Executor(prefill_program, device)

# Run prefill with prompt tokens
prefill_outputs = prefill_executor.run(
    inputs={
        "input_ids": token_buf,
        "attention_mask": mask_buf,
        "position_ids": pos_buf,
    },
    weights=weights,
)
```

### Decode Phase
```python
# Compile decode IR (single token, KV cache)
decode_program = compile_ir("qwen_decode_ir.json")
decode_executor = Executor(decode_program, device)

# Autoregressive generation
for step in range(max_tokens):
    outputs = decode_executor.run(
        inputs={
            "input_ids": next_token_buf,
            "cache_position": cache_pos_buf,
            "position_ids": pos_buf,
            "attention_mask": mask_buf,
        },
        weights=weights,
    )
    next_token = outputs["logits"].to_numpy(spec=...).argmax()
```

## Profiling

Use the built-in profiler to measure kernel performance:

```python
from npu_runtime.profiler import Profiler

profiler = Profiler(executor)
outputs = profiler.run(inputs=inputs, weights=weights)
profiler.print_summary()
```

This prints per-kernel timing with total time, call count, and average duration.
