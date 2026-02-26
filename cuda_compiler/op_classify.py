"""Op classification for CUDA subgraph compiler.

Each ATen op is classified into one of four categories that determines
how it is handled during subgraph compilation:

- ANCHOR_BLAS: Heavy compute delegated to cuBLAS (fusion barrier)
- ELEMENTWISE: Chainable into fused CUDA kernels
- REDUCTION: Custom CUDA kernels (fusion barrier)
- SHAPE_ALIAS: Zero-cost view/reshape (no GPU dispatch)
"""

from __future__ import annotations

from enum import Enum, auto


class OpCategory(Enum):
    ANCHOR_BLAS = auto()
    ELEMENTWISE = auto()
    REDUCTION = auto()
    SHAPE_ALIAS = auto()
    SPECIAL = auto()  # embedding, rope, index_copy — hand-written kernels
    NOOP = auto()  # assertions, dropout (eval), etc.


_OP_CATEGORY_TABLE: dict[str, OpCategory] = {
    # ANCHOR_BLAS: cuBLAS-delegated ops (fusion barriers)
    "aten.matmul.default": OpCategory.ANCHOR_BLAS,
    "aten.linear.default": OpCategory.ANCHOR_BLAS,
    "aten.addmm.default": OpCategory.ANCHOR_BLAS,
    "aten.conv2d.default": OpCategory.ANCHOR_BLAS,
    # ELEMENTWISE: chainable into fused kernels
    "aten.relu.default": OpCategory.ELEMENTWISE,
    "aten.relu_.default": OpCategory.ELEMENTWISE,
    "aten.silu.default": OpCategory.ELEMENTWISE,
    "aten.add.Tensor": OpCategory.ELEMENTWISE,
    "aten.add_.Tensor": OpCategory.ELEMENTWISE,
    "aten.mul.Tensor": OpCategory.ELEMENTWISE,
    "aten.div.Tensor": OpCategory.ELEMENTWISE,
    "aten.neg.default": OpCategory.ELEMENTWISE,
    "aten.pow.Tensor_Scalar": OpCategory.ELEMENTWISE,
    "aten.rsqrt.default": OpCategory.ELEMENTWISE,
    "aten.cos.default": OpCategory.ELEMENTWISE,
    "aten.sin.default": OpCategory.ELEMENTWISE,
    # REDUCTION: custom CUDA kernels (fusion barriers)
    "aten.softmax.int": OpCategory.REDUCTION,
    "aten.mean.dim": OpCategory.REDUCTION,
    "aten.batch_norm.default": OpCategory.REDUCTION,
    "aten.max_pool2d.default": OpCategory.REDUCTION,
    "aten.adaptive_avg_pool2d.default": OpCategory.REDUCTION,
    # SHAPE_ALIAS: zero-cost view/reshape
    "aten.reshape.default": OpCategory.SHAPE_ALIAS,
    "aten.view.default": OpCategory.SHAPE_ALIAS,
    "aten.flatten.using_ints": OpCategory.SHAPE_ALIAS,
    "aten.contiguous.default": OpCategory.SHAPE_ALIAS,
    "aten.unsqueeze.default": OpCategory.SHAPE_ALIAS,
    "aten.alias.default": OpCategory.SHAPE_ALIAS,
    "aten.detach_.default": OpCategory.SHAPE_ALIAS,
    "aten.to.dtype": OpCategory.SHAPE_ALIAS,
    "aten.dropout.default": OpCategory.SHAPE_ALIAS,
    "<built-in function getitem>": OpCategory.SHAPE_ALIAS,
    "aten.t.default": OpCategory.SHAPE_ALIAS,
    "aten.expand.default": OpCategory.SHAPE_ALIAS,
    "aten.slice.Tensor": OpCategory.SHAPE_ALIAS,
    "aten.transpose.int": OpCategory.SHAPE_ALIAS,
    "aten.cat.default": OpCategory.SHAPE_ALIAS,
    # SPECIAL: hand-written CUDA kernels
    "aten.embedding.default": OpCategory.SPECIAL,
    "wrap_with_set_grad_enabled": OpCategory.SPECIAL,  # RoPE
    "aten.index_copy.default": OpCategory.SPECIAL,
    "aten.full.default": OpCategory.SPECIAL,
    # NOOP
    "aten._assert_tensor_metadata.default": OpCategory.NOOP,
}


def classify_op(op_type: str) -> OpCategory:
    """Classify an ATen op into a CUDA compilation category."""
    return _OP_CATEGORY_TABLE.get(op_type, OpCategory.SPECIAL)
