"""Op support table: determines which ops can run on the NPU.

This module codifies the supported operations from docs/operators.md
and codegen.HANDLED_OPS. The partitioner uses is_op_supported() to
decide whether each node runs on NPU or falls back to CPU.
"""

from __future__ import annotations

_SUPPORTED_OPS: set[str] = {
    # CNN
    "aten.conv2d.default",
    "aten.relu.default",
    "aten.relu_.default",
    "aten.batch_norm.default",
    "aten.max_pool2d.default",
    "aten.adaptive_avg_pool2d.default",
    # Linear algebra
    "aten.linear.default",
    "aten.matmul.default",
    "aten.addmm.default",
    "aten.t.default",
    # Element-wise binary
    "aten.add.Tensor",
    "aten.add_.Tensor",
    "aten.mul.Tensor",
    "aten.div.Tensor",
    # Element-wise unary
    "aten.silu.default",
    "aten.neg.default",
    "aten.rsqrt.default",
    "aten.cos.default",
    "aten.sin.default",
    "aten.pow.Tensor_Scalar",
    # Tensor manipulation
    "aten.embedding.default",
    "aten.transpose.int",
    "aten.cat.default",
    "aten.slice.Tensor",
    "aten.expand.default",
    "aten.index_copy.default",
    # Reduction / normalization
    "aten.softmax.int",
    "aten.mean.dim",
    # Scalar ops
    "aten.full.default",
    # Zero-cost aliases (no dispatch)
    "aten.reshape.default",
    "aten.view.default",
    "aten.flatten.using_ints",
    "aten.contiguous.default",
    "aten.unsqueeze.default",
    "aten.alias.default",
    "aten.detach_.default",
    "aten.to.dtype",
    "aten.dropout.default",
    "<built-in function getitem>",
    "aten._assert_tensor_metadata.default",
    # Positional encoding
    "wrap_with_set_grad_enabled",
}


def is_op_supported(op_type: str, _attrs: dict | None = None) -> bool:
    """Check whether an op can execute on the NPU.

    The _attrs parameter is accepted but unused — it exists so this function
    matches the callback signature expected by partition(is_supported_fn).
    """
    return op_type in _SUPPORTED_OPS


def get_supported_ops() -> frozenset[str]:
    return frozenset(_SUPPORTED_OPS)
