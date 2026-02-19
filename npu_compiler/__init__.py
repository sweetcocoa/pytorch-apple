from npu_compiler.codegen import generate_execution_plan
from npu_compiler.compiled_program import CompiledProgram
from npu_compiler.constraint_checker import check_constraints
from npu_compiler.fusion_patterns import find_fusion_groups as find_fusion_groups
from npu_compiler.graph_optimizer import eliminate_noop_ops, fold_batch_norms
from npu_compiler.ir_reader import IRGraph, load_ir
from npu_compiler.ir_reader import load_ir_from_dict as load_ir_from_dict
from npu_compiler.op_support import get_supported_ops as get_supported_ops
from npu_compiler.op_support import is_op_supported as is_op_supported
from npu_compiler.partitioner import Partition as Partition
from npu_compiler.partitioner import PartitionPlan as PartitionPlan
from npu_compiler.partitioner import TransferOp as TransferOp
from npu_compiler.partitioner import partition as partition
from npu_compiler.target_config import METAL_GPU as METAL_GPU
from npu_compiler.target_config import TargetConfig as TargetConfig


def compile(ir: str | dict) -> CompiledProgram:
    """Compile an IR to a CompiledProgram.

    Args:
        ir: Either a path to an IR JSON file (str) or an in-memory IR dict.
    """
    if isinstance(ir, str):
        graph = load_ir(ir)
    else:
        graph = load_ir_from_dict(ir)
    return compile_graph(graph)


def compile_graph(graph: IRGraph) -> CompiledProgram:
    """Compile an IRGraph to a CompiledProgram."""
    # 1. Eliminate no-op nodes (before constraint check)
    eliminate_noop_ops(graph)

    # 2. Check constraints
    violations = check_constraints(graph)
    if violations:
        msgs = "\n".join(f"  - [{v.node_name}] {v.message}" for v in violations)
        raise ValueError(f"NPU constraint violations:\n{msgs}")

    # 3. BN folding
    bn_result = fold_batch_norms(graph)

    # 4. Generate execution plan (includes fusion)
    plan = generate_execution_plan(bn_result.graph)

    return CompiledProgram(
        model_name=graph.model_name,
        execution_plan=plan,
        weight_recipes=bn_result.weight_recipes,
    )
