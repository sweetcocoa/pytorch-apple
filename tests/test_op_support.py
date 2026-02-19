"""Tests for op support table."""

from npu_compiler.codegen import HANDLED_OPS
from npu_compiler.op_support import get_supported_ops, is_op_supported


class TestOpSupport:
    def test_supported_ops_match_handled_ops(self):
        """_SUPPORTED_OPS must be a superset of codegen.HANDLED_OPS."""
        supported = get_supported_ops()
        missing = HANDLED_OPS - supported
        assert not missing, f"HANDLED_OPS not in _SUPPORTED_OPS: {missing}"

    def test_handled_ops_match_supported(self):
        """_SUPPORTED_OPS must equal codegen.HANDLED_OPS (no stale entries)."""
        supported = get_supported_ops()
        extra = supported - HANDLED_OPS
        assert not extra, f"_SUPPORTED_OPS not in HANDLED_OPS: {extra}"

    def test_known_supported_ops(self):
        assert is_op_supported("aten.conv2d.default")
        assert is_op_supported("aten.relu.default")
        assert is_op_supported("aten.linear.default")
        assert is_op_supported("aten.matmul.default")
        assert is_op_supported("aten.softmax.int")
        assert is_op_supported("aten.embedding.default")
        assert is_op_supported("<built-in function getitem>")
        assert is_op_supported("wrap_with_set_grad_enabled")

    def test_known_unsupported_ops(self):
        assert not is_op_supported("aten.gelu.default")
        assert not is_op_supported("aten.layer_norm.default")
        assert not is_op_supported("aten.tanh.default")
        assert not is_op_supported("aten.group_norm.default")
        assert not is_op_supported("aten.leaky_relu.default")
        assert not is_op_supported("aten.pixel_shuffle.default")

    def test_zero_cost_aliases_supported(self):
        aliases = [
            "aten.reshape.default",
            "aten.view.default",
            "aten.flatten.using_ints",
            "aten.contiguous.default",
            "aten.unsqueeze.default",
            "aten.alias.default",
            "aten.detach_.default",
            "aten.to.dtype",
            "aten.dropout.default",
        ]
        for op in aliases:
            assert is_op_supported(op), f"{op} should be supported (zero-cost alias)"

    def test_callback_interface_accepts_attrs(self):
        """is_op_supported accepts attrs for callback compatibility with partition()."""
        assert is_op_supported("aten.conv2d.default", {"kernel_size": [3, 3]})
        assert not is_op_supported("aten.gelu.default", {})

    def test_get_supported_ops_returns_frozenset(self):
        ops = get_supported_ops()
        assert isinstance(ops, frozenset)
        assert len(ops) > 40
