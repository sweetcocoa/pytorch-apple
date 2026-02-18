"""Example: run a compiled ResNet on Metal GPU and compare with CPU."""

import os
import time

import numpy as np
import torch
import torchvision.models as models

from npu_compiler.compiled_program import CompiledProgram
from npu_runtime import Device, Executor, NPUBuffer, load_weights, profile


def run_resnet(model_name: str = "resnet18", npubin_path: str | None = None):
    """Run ResNet on Metal GPU and compare with CPU."""

    # 1. Load or compile
    if npubin_path and os.path.exists(npubin_path):
        program = CompiledProgram.load(npubin_path)
        print(f"Loaded: {npubin_path}")
    else:
        from examples.compile_resnet import compile_resnet
        npubin_path = compile_resnet(model_name)
        program = CompiledProgram.load(npubin_path)

    # 2. Create model for weights and CPU reference
    torch.manual_seed(42)
    model_fn = getattr(models, model_name)
    model = model_fn(weights=None)
    model.eval()

    # 3. Initialize Metal device
    device = Device()
    print(f"Metal device: {device.name}")

    # 4. Load weights
    state_dict_np = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    weights = load_weights(state_dict_np, program, device)
    print(f"Loaded {len(weights)} weight buffers")

    # 5. Create executor
    executor = Executor(program, device)

    # 6. Prepare input
    input_tensor = torch.randn(1, 3, 224, 224)
    input_spec = program.input_specs[0]
    input_buf = {input_spec.name: NPUBuffer.from_numpy(input_tensor.numpy(), device, spec=input_spec)}

    # 7. Run on NPU
    outputs = executor.run(input_buf, weights)
    output_spec = program.output_specs[0]
    npu_output = list(outputs.values())[0].to_numpy(spec=output_spec)

    # 8. Run on CPU
    with torch.no_grad():
        cpu_output = model(input_tensor).numpy()

    # 9. Compare
    max_diff = np.max(np.abs(npu_output - cpu_output))
    argmax_match = np.argmax(npu_output) == np.argmax(cpu_output)
    print("\nAccuracy comparison:")
    print(f"  Max absolute difference: {max_diff:.6f}")
    print(f"  Classification agreement: {'Yes' if argmax_match else 'No'}")
    print(f"  NPU top-5: {np.argsort(npu_output[0])[-5:][::-1]}")
    print(f"  CPU top-5: {np.argsort(cpu_output[0])[-5:][::-1]}")

    # 10. Profile
    print(f"\nProfiling ({model_name})...")
    result = profile(executor, input_buf, weights, warmup=3, iterations=10)
    print(f"  NPU: {result.total_ms:.2f} ms/inference")

    # CPU timing
    start = time.perf_counter()
    for _ in range(10):
        with torch.no_grad():
            model(input_tensor)
    cpu_ms = (time.perf_counter() - start) / 10 * 1000
    print(f"  CPU: {cpu_ms:.2f} ms/inference")
    print(f"  Speedup: {cpu_ms / result.total_ms:.2f}x")

    # Cleanup
    if os.path.exists(npubin_path):
        os.unlink(npubin_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--npubin", default=None)
    args = parser.parse_args()
    run_resnet(args.model, args.npubin)
