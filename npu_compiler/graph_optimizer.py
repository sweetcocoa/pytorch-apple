"""Graph optimizer: no-op elimination, BN folding, op fusion, layout optimization.

Tiling(채널 64바이트 정렬)은 codegen/runtime에서 activation 패딩으로 구현.
Layout 변환(NCHW → tiled layout)은 현재 불필요 (NCHW 유지, 채널 패딩만 적용).
"""

from __future__ import annotations

from dataclasses import dataclass

from npu_compiler.ir_reader import IRGraph, OpNode, TensorSpec

# Ops that are identity pass-throughs: output[0] == input[0]
_IDENTITY_OPS = {
    "aten.alias.default",
    "aten.detach_.default",
}


def eliminate_noop_ops(graph: IRGraph) -> int:
    """Remove no-op nodes from the graph. Returns count of removed nodes.

    Handles:
    - aten._assert_tensor_metadata.default: no outputs, pure assertion → remove
    - aten.alias.default: identity → rewire consumers to input
    - aten.detach_.default: identity → rewire consumers to input
    - aten.dropout.default (p=0, train=False): identity → rewire consumers to input
    """
    removed = 0
    new_nodes: list[OpNode] = []

    # Map: old tensor name → replacement tensor name
    rewire: dict[str, str] = {}

    for node in graph.nodes:
        op = node.op_type

        # 1. No-output assertions → drop
        if op == "aten._assert_tensor_metadata.default":
            removed += 1
            continue

        # Apply pending rewires to this node's inputs
        for inp in node.inputs:
            if inp.name in rewire:
                inp.name = rewire[inp.name]

        # 2. Identity ops → rewire output to input
        if op in _IDENTITY_OPS and len(node.inputs) >= 1 and len(node.outputs) >= 1:
            rewire[node.outputs[0].name] = node.inputs[0].name
            removed += 1
            continue

        # 3. Dropout with p=0 → identity
        if (op == "aten.dropout.default"
                and node.attrs.get("p", 1.0) == 0.0
                and len(node.inputs) >= 1 and len(node.outputs) >= 1):
            rewire[node.outputs[0].name] = node.inputs[0].name
            removed += 1
            continue

        new_nodes.append(node)

    # Rewire graph outputs
    for out in graph.graph_outputs:
        if out.name in rewire:
            out.name = rewire[out.name]

    graph.nodes = new_nodes
    return removed


@dataclass
class WeightTransformRecipe:
    """Recipe for transforming a weight tensor at load time."""
    original_name: str  # state_dict key
    transform: str  # "bn_fold", "none", "fp16_convert"
    params: dict  # transform-specific params


@dataclass
class BNFoldResult:
    """Result of BN folding: updated graph + weight transform recipes.

    Named fields prevent accidental swap of (graph, recipes) return values
    and enable pattern matching: result.graph, result.weight_recipes.
    """
    graph: IRGraph
    weight_recipes: list[WeightTransformRecipe]


def fold_batch_norms(graph: IRGraph) -> BNFoldResult:
    """Fold BatchNorm into preceding Conv2d weights.

    For each Conv2d→BN pair:
    - BN parameters (gamma, beta, mean, var) are absorbed into conv weight/bias
    - BN node is removed from graph
    - A WeightTransformRecipe is generated for the weight loader

    Returns updated graph and recipes.
    """
    recipes: list[WeightTransformRecipe] = []
    new_nodes: list[OpNode] = []
    i = 0
    while i < len(graph.nodes):
        node = graph.nodes[i]

        if (node.op_type == "aten.batch_norm.default"
                and i > 0
                and graph.nodes[i - 1].op_type == "aten.conv2d.default"
                and graph.nodes[i - 1].outputs[0].name == node.inputs[0].name):

            conv_node = new_nodes[-1]  # Already added

            # Get BN parameter names from inputs
            # BN inputs: [input, weight(gamma), bias(beta), running_mean, running_var]
            bn_gamma_name = node.inputs[1].name if len(node.inputs) > 1 else None
            bn_beta_name = node.inputs[2].name if len(node.inputs) > 2 else None
            bn_mean_name = node.inputs[3].name if len(node.inputs) > 3 else None
            bn_var_name = node.inputs[4].name if len(node.inputs) > 4 else None

            # Map placeholder names to state_dict keys
            conv_weight_key = graph.weight_name_mapping.get(
                conv_node.inputs[1].name, conv_node.inputs[1].name
            )
            conv_bias_key = None
            if len(conv_node.inputs) > 2:
                conv_bias_key = graph.weight_name_mapping.get(
                    conv_node.inputs[2].name, conv_node.inputs[2].name
                )

            recipe = WeightTransformRecipe(
                original_name=conv_weight_key,
                transform="bn_fold",
                params={
                    "conv_weight": conv_weight_key,
                    "conv_bias": conv_bias_key,
                    "bn_gamma": graph.weight_name_mapping.get(bn_gamma_name, bn_gamma_name) if bn_gamma_name else None,
                    "bn_beta": graph.weight_name_mapping.get(bn_beta_name, bn_beta_name) if bn_beta_name else None,
                    "bn_mean": graph.weight_name_mapping.get(bn_mean_name, bn_mean_name) if bn_mean_name else None,
                    "bn_var": graph.weight_name_mapping.get(bn_var_name, bn_var_name) if bn_var_name else None,
                    "eps": node.attrs.get("eps", 1e-5),
                },
            )
            recipes.append(recipe)

            # Update conv output to take BN's output shape/name
            conv_node.outputs[0] = TensorSpec(
                name=node.outputs[0].name,
                shape=node.outputs[0].shape,
                dtype=node.outputs[0].dtype,
            )

            # Skip BN node
            i += 1
            continue

        new_nodes.append(node)
        i += 1

    graph.nodes = new_nodes
    return BNFoldResult(graph=graph, weight_recipes=recipes)
