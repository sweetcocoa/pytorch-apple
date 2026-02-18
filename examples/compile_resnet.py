"""Example: compile a ResNet model IR to .npubin."""

import os

import torch
import torchvision.models as models
from torch_ir import extract_ir

import npu_compiler


def compile_resnet(model_name: str = "resnet18"):
    """Extract IR from ResNet and compile to .npubin."""
    # 1. Create meta model (no weights)
    model_fn = getattr(models, model_name)
    with torch.device("meta"):
        meta_model = model_fn(weights=None)
    meta_model.eval()

    # 2. Extract IR
    meta_input = (torch.randn(1, 3, 224, 224, device="meta"),)
    ir = extract_ir(meta_model, meta_input, model_name=model_name.capitalize())

    # 3. Save IR
    ir_path = f"{model_name}_ir.json"
    ir.save(ir_path)
    print(f"IR saved: {ir_path}")

    # 4. Compile
    program = npu_compiler.compile(ir_path)
    print(f"Compiled: {len(program.kernel_calls)} kernel calls")
    print(f"  Inputs: {[(s.name, s.shape) for s in program.input_specs]}")
    print(f"  Outputs: {[(s.name, s.shape) for s in program.output_specs]}")
    print(f"  Weights: {len(program.weight_specs)} tensors")
    print(f"  BN fold recipes: {len(program.weight_recipes)}")

    # 5. Save .npubin
    npubin_path = f"{model_name}.npubin"
    program.save(npubin_path)
    print(f"Saved: {npubin_path}")

    # Cleanup
    os.unlink(ir_path)

    return npubin_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet18", choices=["resnet18", "resnet50"])
    args = parser.parse_args()
    compile_resnet(args.model)
