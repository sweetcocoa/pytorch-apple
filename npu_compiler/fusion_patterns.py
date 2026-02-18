"""Pattern matching for op fusion.

Architecture:
    FusionPatternRegistry allows registering new fusion patterns without modifying
    find_fusion_groups(). Each pattern is a (trigger_op, match_fn) pair. The match
    function receives the trigger node and returns a FusedGroup or None.

    Built-in patterns (Conv+BN+ReLU, Add+ReLU, RMSNorm, SiLU+Gate, Masked Softmax,
    Decode Attention) are registered at module load time via register_fusion_pattern().

Design trade-offs:
    - Available-set tracking: Fusion patterns are matched in graph order, not globally.
      A fused kernel runs at the first node's position, so all its inputs must be
      produced before that point. Without this check, silu+gate fusion reads
      uninitialized up_proj buffer data (discovered as garbage output in Qwen2.5).

    - Passthrough skip limit (_MAX_PASSTHROUGH_SKIP=4): Between two fusible ops,
      the IR may insert expand/to.dtype/assert/dropout. We skip up to 4 such nodes
      during pattern matching. Higher limits risk false matches across layer boundaries.

    - Fused decode attention (M=1 only): The kernel uses shared memory to store
      per-head attention scores, sized for max_seq_len. For M>1 (prefill), this
      would require O(M × seq_len) shared memory, exceeding Metal's 32KB limit.
      Prefill attention uses separate matmul+softmax kernels instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from npu_compiler.ir_reader import IRGraph, OpNode


@dataclass
class FusedGroup:
    """A group of ops to be executed as a single fused kernel."""
    name: str
    kernel_type: str  # "conv_bn_relu", "add_relu", "linear_relu", etc.
    nodes: list[OpNode]
    # First node's inputs (minus intermediate) become fused group's inputs
    # Last node's outputs become fused group's outputs
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Fusion pattern registry — enables adding new patterns without modifying
# find_fusion_groups(). Register with:
#   register_fusion_pattern("aten.my_op.default", my_match_fn)
# ---------------------------------------------------------------------------

# Match function signature:
#   (node, graph, consumers, fused_names, available) -> FusedGroup | None
FusionMatchFn = Callable[
    [OpNode, IRGraph, dict, set, set],
    FusedGroup | None,
]

_FUSION_PATTERN_REGISTRY: list[tuple[str, FusionMatchFn]] = []


def register_fusion_pattern(trigger_op: str, match_fn: FusionMatchFn):
    """Register a fusion pattern triggered by a specific op type.

    Args:
        trigger_op: The ATen op type that starts this pattern (e.g. "aten.pow.Tensor_Scalar").
        match_fn: Function(node, graph, consumers, fused_names, available) -> FusedGroup | None.
                  Return a FusedGroup if the pattern matches, None otherwise.
    """
    _FUSION_PATTERN_REGISTRY.append((trigger_op, match_fn))


def _build_consumer_map(graph: IRGraph) -> dict[str, list[OpNode]]:
    """Map tensor names to the nodes that consume them."""
    consumers: dict[str, list[OpNode]] = {}
    for node in graph.nodes:
        for inp in node.inputs:
            consumers.setdefault(inp.name, []).append(node)
    return consumers


def _has_single_consumer(tensor_name: str, consumers: dict[str, list[OpNode]]) -> bool:
    return len(consumers.get(tensor_name, [])) == 1


def _follow_single_consumer(tensor_name: str, consumers: dict[str, list[OpNode]],
                             expected_op: str, fused: set[str]) -> OpNode | None:
    """Follow a single-consumer edge to a node of the expected op type."""
    next_nodes = consumers.get(tensor_name, [])
    if len(next_nodes) == 1 and next_nodes[0].op_type == expected_op and next_nodes[0].name not in fused:
        return next_nodes[0]
    return None


# Ops that produce no computation (zero-cost aliases); safely skippable during
# pattern matching because they don't affect the fused kernel's data flow.
_PASSTHROUGH_OPS = {
    "aten.expand.default",
    "aten._assert_tensor_metadata.default",
    "aten.to.dtype",
    "aten.dropout.default",
}


# Maximum passthrough ops to skip when looking for a fusion target.
# 4 covers the worst case: expand → to.dtype → _assert → dropout between
# two fused ops (observed in Qwen attention: matmul → expand → to.dtype → softmax).
_MAX_PASSTHROUGH_SKIP = 4


def _follow_skip_passthrough(tensor_name: str, consumers: dict[str, list[OpNode]],
                              expected_op: str, fused: set[str],
                              max_skip: int = _MAX_PASSTHROUGH_SKIP) -> tuple[OpNode | None, list[OpNode]]:
    """Follow consumer, skipping passthrough ops (expand, assert, to.dtype)."""
    skipped: list[OpNode] = []
    current = tensor_name
    for _ in range(max_skip):
        next_nodes = consumers.get(current, [])
        if not next_nodes:
            return None, []
        # _assert_tensor_metadata may have 0 outputs but still appears as consumer
        # For nodes like _assert, the same tensor may have multiple consumers
        # Filter to non-fused nodes
        candidates = [n for n in next_nodes if n.name not in fused]
        if not candidates:
            return None, []
        # If any candidate is our target, return it
        for c in candidates:
            if c.op_type == expected_op:
                return c, skipped
        # If there's exactly one non-assert candidate, follow it if passthrough
        # But _assert has no outputs, so it doesn't block — just skip it
        non_assert = [n for n in candidates if n.op_type != "aten._assert_tensor_metadata.default"]
        assert_nodes = [n for n in candidates if n.op_type == "aten._assert_tensor_metadata.default"]
        for a in assert_nodes:
            skipped.append(a)
        if len(non_assert) != 1:
            return None, []
        node = non_assert[0]
        if node.op_type in _PASSTHROUGH_OPS:
            skipped.append(node)
            if node.outputs:
                current = node.outputs[0].name
            else:
                return None, []
        else:
            return None, []
    return None, []


def _try_match_rmsnorm(pow_node: OpNode, idx: int, graph: IRGraph,
                       consumers: dict[str, list[OpNode]],
                       fused: set[str]) -> tuple[list[OpNode], set[str]] | None:
    """Try to match RMSNorm pattern starting from pow node.

    Fusing 8 individual dispatches into a single kernel reduces launch overhead
    dramatically (×57 RMSNorm layers in Qwen2.5). The fused kernel also avoids
    materializing intermediate tensors (squared, mean, rsqrt) in global memory.

    Pattern: pow(x,2) → mean → add(eps) → rsqrt → [expand] → mul(x) → [expand] → mul(weight)
    """
    # Check pow exponent is 2
    exp = pow_node.attrs.get("exponent", 0)
    if exp != 2.0 and exp != 2:
        return None

    chain = [pow_node]
    fused_names = {pow_node.name}
    rmsnorm_input = pow_node.inputs[0].name  # x

    # pow → mean
    mean_node = _follow_single_consumer(pow_node.outputs[0].name, consumers, "aten.mean.dim", fused)
    if mean_node is None:
        return None
    chain.append(mean_node)
    fused_names.add(mean_node.name)

    # mean → add(eps)
    add_node = _follow_single_consumer(mean_node.outputs[0].name, consumers, "aten.add.Tensor", fused)
    if add_node is None:
        return None
    chain.append(add_node)
    fused_names.add(add_node.name)

    # add → rsqrt
    rsqrt_node = _follow_single_consumer(add_node.outputs[0].name, consumers, "aten.rsqrt.default", fused)
    if rsqrt_node is None:
        return None
    chain.append(rsqrt_node)
    fused_names.add(rsqrt_node.name)

    # rsqrt → [passthrough ops] → mul(x)
    mul1_node, skip1 = _follow_skip_passthrough(rsqrt_node.outputs[0].name, consumers, "aten.mul.Tensor", fused)
    if mul1_node is None or len(mul1_node.inputs) != 2:
        return None
    # Verify one input is rmsnorm_input (x) — may come through passthrough aliases
    mul1_input_names = {inp.name for inp in mul1_node.inputs}
    # Also check if x was aliased through a to.dtype or expand
    valid_x_names = {rmsnorm_input}
    for node in graph.nodes:
        if node.op_type in ("aten.to.dtype", "aten.expand.default") and node.inputs:
            if node.inputs[0].name in valid_x_names and node.outputs:
                valid_x_names.add(node.outputs[0].name)
    if not mul1_input_names & valid_x_names:
        return None
    for e in skip1:
        chain.append(e)
        fused_names.add(e.name)
    chain.append(mul1_node)
    fused_names.add(mul1_node.name)

    # mul(x*rsqrt) → [passthrough ops] → mul(weight)
    mul2_node, skip2 = _follow_skip_passthrough(mul1_node.outputs[0].name, consumers, "aten.mul.Tensor", fused)
    if mul2_node is None or len(mul2_node.inputs) != 2:
        return None
    for e in skip2:
        chain.append(e)
        fused_names.add(e.name)
    chain.append(mul2_node)
    fused_names.add(mul2_node.name)

    return chain, fused_names


def _try_match_decode_attention(transpose_node: OpNode, graph: IRGraph,
                                consumers: dict[str, list[OpNode]],
                                fused: set[str]) -> tuple[list[OpNode], set[str], dict] | None:
    """Try to match decode attention pattern starting from transpose(dim0=2,dim1=3).

    Fusing the full attention sequence into a single kernel eliminates 6+ dispatch
    round-trips per head (×28 heads ×28 layers), and allows keeping intermediate
    scores in shared memory instead of global buffers.  Only valid for decode (M=1)
    because the shared-memory score array is sized for max sequence length.

    Pattern: transpose(K) → matmul(Q×K^T) → mul(scale) → [alias] → add(mask) → softmax
             → [passthrough: _assert, to.dtype, dropout] → matmul(attn×V)

    Returns (nodes_chain, fused_names, metadata) or None.
    """
    # Entry: transpose with dim0=2, dim1=3 on a 4D tensor
    dim0 = transpose_node.attrs.get("dim0", 0)
    dim1 = transpose_node.attrs.get("dim1", 1)
    if dim0 != 2 or dim1 != 3:
        return None
    out_shape = transpose_node.outputs[0].shape
    if len(out_shape) != 4:
        return None

    chain = [transpose_node]
    fused_names = {transpose_node.name}

    # 1. transpose → single consumer matmul (Q × K^T)
    t_out = transpose_node.outputs[0].name
    matmul1 = _follow_single_consumer(t_out, consumers, "aten.matmul.default", fused)
    if matmul1 is None:
        return None
    # Q must have M=1 (decode: shape is [B,H,1,D])
    q_shape = matmul1.inputs[0].shape
    if len(q_shape) != 4 or q_shape[-2] != 1:
        return None
    chain.append(matmul1)
    fused_names.add(matmul1.name)
    q_name = matmul1.inputs[0].name

    # 2. matmul → mul.Tensor (scale)
    matmul1_out = matmul1.outputs[0].name
    mul_node = _follow_single_consumer(matmul1_out, consumers, "aten.mul.Tensor", fused)
    if mul_node is None:
        return None
    # Extract scale: mul has 1 input (scalar attr) or 2 inputs
    if len(mul_node.inputs) == 1:
        scale = float(mul_node.attrs.get("other", 1.0))
    else:
        # Scale could be a scalar constant — check attrs
        scale = float(mul_node.attrs.get("other", 0.0))
        if scale == 0.0:
            return None
    chain.append(mul_node)
    fused_names.add(mul_node.name)

    # 3. mul → skip alias → add.Tensor (mask addition)
    cur_out = mul_node.outputs[0].name
    cur_consumers = consumers.get(cur_out, [])
    if len(cur_consumers) != 1:
        return None
    next_node = cur_consumers[0]
    if next_node.name in fused:
        return None

    # Skip alias nodes between mul and add
    while next_node.op_type in ("aten.alias.default", "aten.to.dtype"):
        chain.append(next_node)
        fused_names.add(next_node.name)
        if not next_node.outputs:
            return None
        cur_out = next_node.outputs[0].name
        cur_consumers = consumers.get(cur_out, [])
        # alias may have multiple consumers if _assert also consumes it
        _ASSERT_OP = "aten._assert_tensor_metadata.default"
        non_assert = [n for n in cur_consumers if n.op_type != _ASSERT_OP and n.name not in fused]
        assert_nodes = [n for n in cur_consumers if n.op_type == _ASSERT_OP and n.name not in fused]
        for a in assert_nodes:
            chain.append(a)
            fused_names.add(a.name)
        if len(non_assert) != 1:
            return None
        next_node = non_assert[0]

    # Now next_node should be add.Tensor
    if next_node.op_type != "aten.add.Tensor" or next_node.name in fused:
        return None
    add_node = next_node
    if len(add_node.inputs) != 2:
        return None
    chain.append(add_node)
    fused_names.add(add_node.name)

    # Identify mask: the add input that is NOT the mul/alias output
    mask_name = None
    for inp in add_node.inputs:
        if inp.name != cur_out:
            mask_name = inp.name
            break
    if mask_name is None:
        return None

    # 4. add → softmax.int
    add_out = add_node.outputs[0].name
    softmax_node = _follow_single_consumer(add_out, consumers, "aten.softmax.int", fused)
    if softmax_node is None:
        return None
    chain.append(softmax_node)
    fused_names.add(softmax_node.name)

    # 5. softmax → skip passthrough (_assert, to.dtype, dropout) → matmul (attn×V)
    matmul2, skipped = _follow_skip_passthrough(softmax_node.outputs[0].name, consumers,
                                                 "aten.matmul.default", fused)
    if matmul2 is None:
        return None

    # Also skip dropout as passthrough
    if matmul2.op_type != "aten.matmul.default":
        return None

    for s in skipped:
        chain.append(s)
        fused_names.add(s.name)
    chain.append(matmul2)
    fused_names.add(matmul2.name)

    # Extract V name: matmul2 input[1]
    v_name = matmul2.inputs[1].name

    # Extract K_notrans name: transpose's input
    k_notrans_name = transpose_node.inputs[0].name

    # Get dimensions from transpose input shape: (B, H, S, D)
    k_shape = transpose_node.inputs[0].shape
    B, H, S, D = k_shape

    metadata = {
        "q_name": q_name,
        "k_notrans_name": k_notrans_name,
        "v_name": v_name,
        "mask_name": mask_name,
        "scale": scale,
        "B": B, "H": H, "S": S, "D": D,
        "final_output_name": matmul2.outputs[0].name,
    }

    return chain, fused_names, metadata


# ---------------------------------------------------------------------------
# Built-in fusion match functions (registered via register_fusion_pattern)
# ---------------------------------------------------------------------------

def _match_conv_bn_relu(node, graph, consumers, fused, available):
    """Conv2d + BN + ReLU — folding BN into conv weight at load time
    eliminates BN's 5 param buffers and 4 dispatch calls per layer.
    ReLU fusion adds the activation at zero cost (single flag in conv kernel)."""
    chain = [node]
    fused_names = {node.name}
    output_name = node.outputs[0].name
    kernel_type = "conv"

    next_nodes = consumers.get(output_name, [])
    if (len(next_nodes) == 1 and next_nodes[0].op_type == "aten.batch_norm.default"
            and next_nodes[0].name not in fused):
        bn_node = next_nodes[0]
        chain.append(bn_node)
        fused_names.add(bn_node.name)
        output_name = bn_node.outputs[0].name
        kernel_type = "conv_bn"

    next_nodes = consumers.get(output_name, [])
    if (len(next_nodes) == 1 and next_nodes[0].op_type == "aten.relu.default"
            and next_nodes[0].name not in fused):
        relu_node = next_nodes[0]
        chain.append(relu_node)
        fused_names.add(relu_node.name)
        kernel_type = kernel_type + "_relu"

    if len(chain) > 1:
        return FusedGroup(name=f"fused_{chain[0].name}", kernel_type=kernel_type, nodes=chain)
    return None


def _match_add_relu(node, graph, consumers, fused, available):
    """Add + ReLU fusion."""
    next_nodes = consumers.get(node.outputs[0].name, [])
    if (len(next_nodes) == 1 and next_nodes[0].op_type == "aten.relu.default"
            and next_nodes[0].name not in fused):
        relu_node = next_nodes[0]
        return FusedGroup(name=f"fused_{node.name}", kernel_type="add_relu",
                          nodes=[node, relu_node])
    return None


def _match_rmsnorm(node, graph, consumers, fused, available):
    """RMSNorm: pow(x,2) → mean → add(eps) → rsqrt → mul(x) → mul(weight)."""
    result = _try_match_rmsnorm(node, 0, graph, consumers, fused)
    if result is not None:
        chain, _fused_names = result
        return FusedGroup(name=f"fused_{node.name}", kernel_type="rmsnorm", nodes=chain)
    return None


def _match_decode_attention(node, graph, consumers, fused, available):
    """Fused decode attention: transpose→matmul→scale→add(mask)→softmax→matmul."""
    result = _try_match_decode_attention(node, graph, consumers, fused)
    if result is not None:
        chain, _fused_names, metadata = result
        return FusedGroup(name=f"fused_{node.name}", kernel_type="decode_attention",
                          nodes=chain, metadata=metadata)
    return None


def _match_silu_mul(node, graph, consumers, fused, available):
    """SiLU + mul (GatedMLP: silu(gate) * up)."""
    silu_out = node.outputs[0].name
    silu_consumers = consumers.get(silu_out, [])
    if (len(silu_consumers) == 1
            and silu_consumers[0].op_type == "aten.mul.Tensor"
            and silu_consumers[0].name not in fused
            and len(silu_consumers[0].inputs) == 2):
        mul_node = silu_consumers[0]
        # CRITICAL: The fused kernel runs at silu's position in the graph.
        # If mul's other input (e.g. up_proj output) hasn't been produced yet,
        # the kernel reads uninitialized buffer data → garbage output.
        # Only fuse when all non-silu inputs are already available.
        other_inputs = [inp.name for inp in mul_node.inputs if inp.name != silu_out]
        if other_inputs and all(name in available for name in other_inputs):
            return FusedGroup(name=f"fused_{node.name}", kernel_type="silu_mul",
                              nodes=[node, mul_node])
    return None


def _match_masked_softmax(node, graph, consumers, fused, available):
    """Add + Softmax (masked softmax: scores + mask → softmax)."""
    if len(node.inputs) != 2:
        return None
    add_out = node.outputs[0].name
    add_consumers = consumers.get(add_out, [])
    if (len(add_consumers) == 1
            and add_consumers[0].op_type == "aten.softmax.int"
            and add_consumers[0].name not in fused
            and _has_single_consumer(add_out, consumers)):
        softmax_node = add_consumers[0]
        return FusedGroup(name=f"fused_{node.name}", kernel_type="masked_softmax",
                          nodes=[node, softmax_node])
    return None


# Register all built-in patterns. Order matters: Conv+BN+ReLU must match before
# Add+ReLU to avoid the add_relu pattern stealing conv's relu.
register_fusion_pattern("aten.conv2d.default", _match_conv_bn_relu)
register_fusion_pattern("aten.add.Tensor", _match_add_relu)
register_fusion_pattern("aten.pow.Tensor_Scalar", _match_rmsnorm)
register_fusion_pattern("aten.transpose.int", _match_decode_attention)
register_fusion_pattern("aten.silu.default", _match_silu_mul)
register_fusion_pattern("aten.add.Tensor", _match_masked_softmax)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def find_fusion_groups(graph: IRGraph) -> list[FusedGroup | OpNode]:
    """Find fusible patterns and return a list of FusedGroups and standalone OpNodes.

    Consults the fusion pattern registry for each node. Patterns are tried in
    registration order, so earlier-registered patterns have priority.

    Note: expects ops to be normalized (no in-place variants) before calling.
    """
    consumers = _build_consumer_map(graph)
    fused_node_names: set[str] = set()
    result: list[FusedGroup | OpNode] = []

    # Track which tensors are available (produced by earlier nodes, or external inputs/weights).
    # Required to prevent fusing ops whose inputs haven't been computed yet (see silu+gate).
    available: set[str] = {inp.name for inp in graph.graph_inputs}
    for w in graph.weights:
        available.add(w.name)
    for placeholder in graph.weight_name_mapping:
        available.add(placeholder)

    # Build per-op-type index into registry for O(1) lookup
    patterns_by_op: dict[str, list[FusionMatchFn]] = {}
    for trigger_op, match_fn in _FUSION_PATTERN_REGISTRY:
        patterns_by_op.setdefault(trigger_op, []).append(match_fn)

    i = 0
    while i < len(graph.nodes):
        node = graph.nodes[i]

        if node.name in fused_node_names:
            i += 1
            continue

        # Try registered patterns for this node's op type
        matched = False
        for match_fn in patterns_by_op.get(node.op_type, []):
            group = match_fn(node, graph, consumers, fused_node_names, available)
            if group is not None:
                for n in group.nodes:
                    fused_node_names.add(n.name)
                result.append(group)
                matched = True
                break

        if matched:
            i += 1
            continue

        # No fusion: standalone node
        if node.name not in fused_node_names:
            result.append(node)

        # Mark this node's outputs as available for subsequent fusions
        for out in node.outputs:
            available.add(out.name)

        i += 1

    return result
