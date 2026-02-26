"""Fused elementwise CUDA kernel code generation.

Generates CUDA C source code for chains of elementwise operations
that will be JIT-compiled via NVRTC at runtime.
"""

from __future__ import annotations

import numpy as np

from npu_compiler.ir_reader import OpNode

# Map ATen op_type to CUDA C expression template.
# {v_prev} = previous stage output, {v_in} = additional input read
_CUDA_EXPR_MAP: dict[str, str] = {
    "aten.relu.default": "__hmax({v_prev}, (__half)0.0)",
    "aten.relu_.default": "__hmax({v_prev}, (__half)0.0)",
    "aten.silu.default": "({v_prev} / ((__half)1.0 + hexp(-{v_prev})))",
    "aten.neg.default": "(-{v_prev})",
    "aten.rsqrt.default": "hrsqrt({v_prev})",
    "aten.cos.default": "hcos({v_prev})",
    "aten.sin.default": "hsin({v_prev})",
    # Binary ops: {v_in} is the second operand
    "aten.add.Tensor": "({v_prev} + {v_in})",
    "aten.add_.Tensor": "({v_prev} + {v_in})",
    "aten.mul.Tensor": "({v_prev} * {v_in})",
    "aten.div.Tensor": "({v_prev} / {v_in})",
}

# Ops that need a second input buffer
_BINARY_OPS = {
    "aten.add.Tensor",
    "aten.add_.Tensor",
    "aten.mul.Tensor",
    "aten.div.Tensor",
}

# pow is special — needs scalar exponent
_POW_OP = "aten.pow.Tensor_Scalar"


def is_fusible_elementwise(op_type: str) -> bool:
    """Check if an op can participate in elementwise fusion."""
    return op_type in _CUDA_EXPR_MAP or op_type == _POW_OP


def generate_fused_kernel(
    kernel_name: str,
    chain: list[OpNode],
    input_names: list[str],
    output_name: str,
) -> str:
    """Generate CUDA C source for a fused elementwise kernel.

    Args:
        kernel_name: Unique name for the generated kernel function.
        chain: Ordered list of OpNodes to fuse.
        input_names: Buffer names for external inputs (in order).
        output_name: Buffer name for the output.

    Returns:
        CUDA C source code string suitable for NVRTC compilation.
    """
    # Build function signature
    params: list[str] = []
    input_idx_map: dict[str, int] = {}
    for i, name in enumerate(input_names):
        params.append(f"const __half* in{i}")
        input_idx_map[name] = i
    params.append("__half* out0")
    params.append("int N")

    sig = ", ".join(params)

    # Build kernel body
    lines: list[str] = []
    lines.append("#include <cuda_fp16.h>")
    lines.append('extern "C" {')
    lines.append(f"__global__ void {kernel_name}({sig}) {{")
    lines.append("    int idx = blockIdx.x * blockDim.x + threadIdx.x;")
    lines.append("    if (idx >= N) return;")

    var_counter = 0
    # Map tensor name -> CUDA variable name
    tensor_to_var: dict[str, str] = {}

    # Read initial inputs
    for name, idx in input_idx_map.items():
        vname = f"v{var_counter}"
        lines.append(f"    __half {vname} = in{idx}[idx];")
        tensor_to_var[name] = vname
        var_counter += 1

    # Generate fused ops
    for node in chain:
        op_type = node.op_type

        # Find the primary input variable
        primary_input = node.inputs[0].name
        v_prev = tensor_to_var.get(primary_input, "in0[idx]")

        if op_type == _POW_OP:
            exponent = float(node.attrs.get("exponent", 2.0))
            vname = f"v{var_counter}"
            lines.append(f"    __half {vname} = __float2half(powf(__half2float({v_prev}), {exponent}f));")
            tensor_to_var[node.outputs[0].name] = vname
            var_counter += 1
        elif op_type in _BINARY_OPS:
            if len(node.inputs) >= 2:
                second_input = node.inputs[1].name
                v_in = tensor_to_var.get(second_input)
                if v_in is None:
                    # Need to read from an external input buffer
                    buf_idx = input_idx_map.get(second_input)
                    if buf_idx is not None:
                        v_in = f"in{buf_idx}[idx]"
                    else:
                        v_in = "(__half)0.0"
            else:
                # Scalar op: second operand is in attrs['other']
                scalar_val = float(node.attrs.get("other", 0.0))
                v_in = f"__float2half({scalar_val}f)"
            expr = _CUDA_EXPR_MAP[op_type].format(v_prev=v_prev, v_in=v_in)
            vname = f"v{var_counter}"
            lines.append(f"    __half {vname} = {expr};")
            tensor_to_var[node.outputs[0].name] = vname
            var_counter += 1
        elif op_type in _CUDA_EXPR_MAP:
            expr = _CUDA_EXPR_MAP[op_type].format(v_prev=v_prev, v_in="")
            vname = f"v{var_counter}"
            lines.append(f"    __half {vname} = {expr};")
            tensor_to_var[node.outputs[0].name] = vname
            var_counter += 1

    # Write output — last variable written
    last_output = chain[-1].outputs[0].name
    last_var = tensor_to_var.get(last_output, f"v{var_counter - 1}")
    lines.append(f"    out0[idx] = {last_var};")
    lines.append("}")
    lines.append("}")

    return "\n".join(lines)


def compute_total_elements(shape: list[int]) -> int:
    """Compute total number of elements for a shape."""
    return int(np.prod(shape))
