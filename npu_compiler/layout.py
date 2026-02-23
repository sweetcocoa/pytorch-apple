"""Layout constraint system for tensor memory layout management.

Centralizes layout decisions (channel padding, dense row-major, tiled) that were
previously scattered across codegen.py (_PADDED_KERNELS), target_config.py
(padded_shape_4d), and per-op heuristics.

Architecture:
    Two-level system:
    1. LayoutConstraint — per-op abstract requirement (e.g., "conv needs PADDED_NCHW")
    2. LayoutProfile — per-backend overrides (e.g., CPU profile forces all DENSE)

    resolve_layouts() runs after fusion, before kernel emission, to produce a
    ResolvedLayout for every tensor in the graph. This replaces the old approach
    of checking _PADDED_KERNELS at buffer allocation time.

Design trade-offs:
    - ANY layout propagates from input[0]: elementwise ops like add/relu don't
      care about layout, so they inherit whatever their first input uses. This
      avoids inserting unnecessary pad/depad conversions in mixed conv+linear models.

    - Flatten/reshape forces DENSE output: when going from 4D (potentially padded)
      to non-4D, we must strip padding. The layout system detects this automatically
      and codegen inserts depad_4d_kernel.

    - Weight/bias inputs use DENSE by default since weights are stored in dense
      format and transformed at load time (conv kernels handle the padding internally).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from npu_compiler.target_config import METAL_GPU, TargetConfig


class LayoutKind(Enum):
    """Tensor memory layout types."""

    DENSE = auto()  # Row-major contiguous, no padding (matmul, linear, transformer)
    PADDED_NCHW = auto()  # NCHW with channel dim padded to tile multiple (conv, pool, BN)
    TILED_NCHW = auto()  # (N, C/tile, H, W, tile) block layout (future NPU/TPU)
    ANY = auto()  # Accepts any layout (elementwise, reshape passthrough)


@dataclass(frozen=True)
class Layout:
    """A specific tensor memory layout specification."""

    kind: LayoutKind
    channel_tile: int = 0  # Only meaningful for PADDED_NCHW / TILED_NCHW

    def physical_shape(self, logical_shape: list[int], config: TargetConfig = METAL_GPU) -> list[int]:
        """Compute physical allocation shape from logical shape."""
        if self.kind == LayoutKind.PADDED_NCHW and len(logical_shape) == 4:
            tile = self.channel_tile or config.channel_tile
            N, C, H, W = logical_shape
            return [N, ((C + tile - 1) // tile) * tile, H, W]
        if self.kind == LayoutKind.TILED_NCHW and len(logical_shape) == 4:
            tile = self.channel_tile or config.channel_tile
            N, C, H, W = logical_shape
            return [N, (C + tile - 1) // tile, H, W, tile]
        return list(logical_shape)

    def is_compatible_with(self, other: Layout) -> bool:
        """Check if two layouts are compatible (no conversion needed)."""
        if self.kind == LayoutKind.ANY or other.kind == LayoutKind.ANY:
            return True
        return self.kind == other.kind and self.channel_tile == other.channel_tile


# ---------------------------------------------------------------------------
# Predefined layout constants
# ---------------------------------------------------------------------------

DENSE = Layout(LayoutKind.DENSE)
PADDED_NCHW_32 = Layout(LayoutKind.PADDED_NCHW, channel_tile=32)
ANY_LAYOUT = Layout(LayoutKind.ANY)


# ---------------------------------------------------------------------------
# LayoutConstraint — per-op layout requirements
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayoutConstraint:
    """Layout requirements for an operation's inputs and outputs.

    input_layouts:  Required layout per input (index-matched to op inputs).
                    Extra inputs beyond this list default to DENSE.
    output_layouts: Required layout per output.
    propagate_input: When True and output layout is ANY, propagate input[0]'s
                     resolved layout to outputs. This allows elementwise ops
                     to work in whatever layout their input uses.
    """

    input_layouts: tuple[Layout, ...]
    output_layouts: tuple[Layout, ...]
    propagate_input: bool = True


# Default constraint: ANY inputs, ANY outputs, propagate from input
_DEFAULT_CONSTRAINT = LayoutConstraint(
    input_layouts=(ANY_LAYOUT,),
    output_layouts=(ANY_LAYOUT,),
    propagate_input=True,
)


# ---------------------------------------------------------------------------
# Layout registry — maps op_type to LayoutConstraint
# ---------------------------------------------------------------------------

_LAYOUT_REGISTRY: dict[str, LayoutConstraint] = {}


def register_layout(op_type: str, constraint: LayoutConstraint) -> None:
    """Register a layout constraint for an op type."""
    _LAYOUT_REGISTRY[op_type] = constraint


def get_layout_constraint(op_type: str) -> LayoutConstraint:
    """Get the layout constraint for an op type (default: ANY passthrough)."""
    return _LAYOUT_REGISTRY.get(op_type, _DEFAULT_CONSTRAINT)


# ---------------------------------------------------------------------------
# Built-in op registrations
# ---------------------------------------------------------------------------

# Conv2d: padded NCHW input activation, dense weight/bias, padded output
register_layout(
    "aten.conv2d.default",
    LayoutConstraint(
        input_layouts=(PADDED_NCHW_32, DENSE, DENSE),  # activation, weight, bias
        output_layouts=(PADDED_NCHW_32,),
    ),
)

# BatchNorm: padded NCHW (consumed by BN folding, but declare for completeness)
register_layout(
    "aten.batch_norm.default",
    LayoutConstraint(
        input_layouts=(PADDED_NCHW_32, DENSE, DENSE, DENSE, DENSE),
        output_layouts=(PADDED_NCHW_32,),
    ),
)

# MaxPool: padded NCHW
register_layout(
    "aten.max_pool2d.default",
    LayoutConstraint(
        input_layouts=(PADDED_NCHW_32,),
        output_layouts=(PADDED_NCHW_32,),
    ),
)

# AdaptiveAvgPool: padded NCHW
register_layout(
    "aten.adaptive_avg_pool2d.default",
    LayoutConstraint(
        input_layouts=(PADDED_NCHW_32,),
        output_layouts=(PADDED_NCHW_32,),
    ),
)

# Linear/Matmul: DENSE
register_layout(
    "aten.linear.default",
    LayoutConstraint(
        input_layouts=(DENSE, DENSE, DENSE),
        output_layouts=(DENSE,),
        propagate_input=False,
    ),
)

register_layout(
    "aten.matmul.default",
    LayoutConstraint(
        input_layouts=(DENSE, DENSE),
        output_layouts=(DENSE,),
        propagate_input=False,
    ),
)

register_layout(
    "aten.addmm.default",
    LayoutConstraint(
        input_layouts=(DENSE, DENSE, DENSE),
        output_layouts=(DENSE,),
        propagate_input=False,
    ),
)

# Elementwise unary: ANY (propagate from input)
for _op in [
    "aten.relu.default",
    "aten.relu_.default",
    "aten.silu.default",
    "aten.neg.default",
    "aten.rsqrt.default",
    "aten.cos.default",
    "aten.sin.default",
    "aten.pow.Tensor_Scalar",
    "aten.contiguous.default",
    "aten.to.dtype",
    "aten.dropout.default",
    "aten.alias.default",
    "aten.detach_.default",
    "aten.unsqueeze.default",
]:
    register_layout(
        _op,
        LayoutConstraint(
            input_layouts=(ANY_LAYOUT,),
            output_layouts=(ANY_LAYOUT,),
            propagate_input=True,
        ),
    )

# Elementwise binary: ANY (propagate from input)
for _op in [
    "aten.add.Tensor",
    "aten.add_.Tensor",
    "aten.mul.Tensor",
    "aten.div.Tensor",
]:
    register_layout(
        _op,
        LayoutConstraint(
            input_layouts=(ANY_LAYOUT, ANY_LAYOUT),
            output_layouts=(ANY_LAYOUT,),
            propagate_input=True,
        ),
    )

# Flatten/reshape: ANY input, DENSE output (forces depad for 4D→non-4D)
for _op in [
    "aten.flatten.using_ints",
    "aten.view.default",
    "aten.reshape.default",
]:
    register_layout(
        _op,
        LayoutConstraint(
            input_layouts=(ANY_LAYOUT,),
            output_layouts=(DENSE,),
            propagate_input=False,
        ),
    )

# Mean: can be 4D (spatial) or last-dim reduction
register_layout(
    "aten.mean.dim",
    LayoutConstraint(
        input_layouts=(ANY_LAYOUT,),
        output_layouts=(ANY_LAYOUT,),
        propagate_input=True,
    ),
)

# Transpose: DENSE (explicit data movement)
register_layout(
    "aten.transpose.int",
    LayoutConstraint(
        input_layouts=(DENSE,),
        output_layouts=(DENSE,),
        propagate_input=False,
    ),
)

register_layout(
    "aten.t.default",
    LayoutConstraint(
        input_layouts=(DENSE,),
        output_layouts=(DENSE,),
        propagate_input=False,
    ),
)

# Cat: ANY (propagate from first input)
register_layout(
    "aten.cat.default",
    LayoutConstraint(
        input_layouts=(ANY_LAYOUT, ANY_LAYOUT),
        output_layouts=(ANY_LAYOUT,),
        propagate_input=True,
    ),
)

# Slice: ANY (propagate from input)
register_layout(
    "aten.slice.Tensor",
    LayoutConstraint(
        input_layouts=(ANY_LAYOUT,),
        output_layouts=(ANY_LAYOUT,),
        propagate_input=True,
    ),
)

# Expand: ANY (propagate from input)
register_layout(
    "aten.expand.default",
    LayoutConstraint(
        input_layouts=(ANY_LAYOUT,),
        output_layouts=(ANY_LAYOUT,),
        propagate_input=True,
    ),
)

# Softmax: DENSE
register_layout(
    "aten.softmax.int",
    LayoutConstraint(
        input_layouts=(DENSE,),
        output_layouts=(DENSE,),
        propagate_input=False,
    ),
)

# Embedding: DENSE output (indices are int, weight is dense)
register_layout(
    "aten.embedding.default",
    LayoutConstraint(
        input_layouts=(DENSE, DENSE),
        output_layouts=(DENSE,),
        propagate_input=False,
    ),
)

# Full: DENSE
register_layout(
    "aten.full.default",
    LayoutConstraint(
        input_layouts=(),
        output_layouts=(DENSE,),
        propagate_input=False,
    ),
)

# Getitem: ANY (propagate from input)
register_layout(
    "<built-in function getitem>",
    LayoutConstraint(
        input_layouts=(ANY_LAYOUT,),
        output_layouts=(ANY_LAYOUT,),
        propagate_input=True,
    ),
)

# RoPE: DENSE
register_layout(
    "wrap_with_set_grad_enabled",
    LayoutConstraint(
        input_layouts=(DENSE, DENSE),
        output_layouts=(DENSE, DENSE),
        propagate_input=False,
    ),
)

# Assert: no outputs, ANY input
register_layout(
    "aten._assert_tensor_metadata.default",
    LayoutConstraint(
        input_layouts=(ANY_LAYOUT,),
        output_layouts=(),
        propagate_input=False,
    ),
)

# Index copy: DENSE
register_layout(
    "aten.index_copy.default",
    LayoutConstraint(
        input_layouts=(DENSE, DENSE, DENSE),
        output_layouts=(DENSE,),
        propagate_input=False,
    ),
)


# ---------------------------------------------------------------------------
# ResolvedLayout — result of layout resolution for a single tensor
# ---------------------------------------------------------------------------


@dataclass
class ResolvedLayout:
    """The resolved layout for a specific tensor after layout analysis."""

    layout: Layout
    logical_shape: list[int]
    physical_shape: list[int]

    @property
    def needs_padding(self) -> bool:
        return self.physical_shape != self.logical_shape


# ---------------------------------------------------------------------------
# LayoutConversion — inserted when adjacent ops have incompatible layouts
# ---------------------------------------------------------------------------


@dataclass
class LayoutConversion:
    """A layout conversion that must be inserted between two ops."""

    tensor_name: str
    from_layout: Layout
    to_layout: Layout
    logical_shape: list[int]


# ---------------------------------------------------------------------------
# LayoutProfile — backend-specific layout overrides
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayoutProfile:
    """Backend-specific layout overrides.

    Allows different backends (Metal GPU, CPU, future NPU/TPU) to override
    the default layout constraints for specific ops.
    """

    name: str
    overrides: dict[str, LayoutConstraint] = field(default_factory=dict)

    def get_constraint(self, op_type: str) -> LayoutConstraint:
        """Get layout constraint for an op, checking overrides first."""
        if op_type in self.overrides:
            return self.overrides[op_type]
        return get_layout_constraint(op_type)


# Predefined profiles
METAL_PROFILE = LayoutProfile(name="metal_gpu")  # Uses default registry (PADDED_NCHW for conv)

CPU_PROFILE = LayoutProfile(
    name="cpu",
    overrides={
        # CPU doesn't need channel padding — everything is DENSE
        "aten.conv2d.default": LayoutConstraint(
            input_layouts=(DENSE, DENSE, DENSE),
            output_layouts=(DENSE,),
        ),
        "aten.batch_norm.default": LayoutConstraint(
            input_layouts=(DENSE, DENSE, DENSE, DENSE, DENSE),
            output_layouts=(DENSE,),
        ),
        "aten.max_pool2d.default": LayoutConstraint(
            input_layouts=(DENSE,),
            output_layouts=(DENSE,),
        ),
        "aten.adaptive_avg_pool2d.default": LayoutConstraint(
            input_layouts=(DENSE,),
            output_layouts=(DENSE,),
        ),
    },
)


# ---------------------------------------------------------------------------
# resolve_layouts() — main layout resolution pass
# ---------------------------------------------------------------------------


def resolve_layouts(
    nodes: list,
    graph_inputs: list,
    weights: list,
    weight_name_mapping: dict[str, str],
    profile: LayoutProfile | None = None,
    config: TargetConfig = METAL_GPU,
) -> dict[str, ResolvedLayout]:
    """Resolve layouts for all tensors in the graph.

    Runs after fusion, before kernel emission. Produces a ResolvedLayout for
    every named tensor, which codegen uses for buffer allocation and I/O transforms.

    Args:
        nodes: list of FusedGroup | OpNode from find_fusion_groups()
        graph_inputs: list of TensorSpec for model inputs
        weights: list of TensorSpec for weight tensors
        weight_name_mapping: FX placeholder → state_dict key mapping
        profile: optional backend profile for layout overrides
        config: target hardware config

    Returns:
        dict mapping tensor name → ResolvedLayout
    """
    from npu_compiler.fusion_patterns import FusedGroup
    from npu_compiler.ir_reader import OpNode

    if profile is None:
        profile = METAL_PROFILE

    resolved: dict[str, ResolvedLayout] = {}

    # Initialize graph inputs as DENSE
    for spec in graph_inputs:
        resolved[spec.name] = ResolvedLayout(
            layout=DENSE,
            logical_shape=list(spec.shape),
            physical_shape=list(spec.shape),
        )

    # Initialize weights as DENSE
    for spec in weights:
        resolved[spec.name] = ResolvedLayout(
            layout=DENSE,
            logical_shape=list(spec.shape),
            physical_shape=list(spec.shape),
        )

    # Also initialize weight placeholders
    for placeholder in weight_name_mapping:
        if placeholder not in resolved:
            resolved[placeholder] = ResolvedLayout(
                layout=DENSE,
                logical_shape=[],
                physical_shape=[],
            )

    def _resolve_op_outputs(op_node: OpNode) -> None:
        """Resolve output layouts for a single op node."""
        constraint = profile.get_constraint(op_node.op_type)

        # Determine the effective input layout (for propagation)
        input_layout = DENSE
        if op_node.inputs:
            first_input = op_node.inputs[0].name
            if first_input in resolved:
                input_layout = resolved[first_input].layout

        # Resolve each output
        for i, out_spec in enumerate(op_node.outputs):
            if i < len(constraint.output_layouts):
                out_constraint = constraint.output_layouts[i]
            else:
                out_constraint = ANY_LAYOUT

            # Determine effective output layout
            if out_constraint.kind == LayoutKind.ANY:
                if constraint.propagate_input:
                    effective = input_layout
                else:
                    effective = DENSE
            else:
                effective = out_constraint

            physical = effective.physical_shape(list(out_spec.shape), config)
            resolved[out_spec.name] = ResolvedLayout(
                layout=effective,
                logical_shape=list(out_spec.shape),
                physical_shape=physical,
            )

    # Process nodes in topological order
    for item in nodes:
        if isinstance(item, FusedGroup):
            # For fused groups, resolve based on the primary op
            # The fused group's layout is determined by its first (main) op
            primary_node = item.nodes[0]
            constraint = profile.get_constraint(primary_node.op_type)

            # Resolve intermediate outputs within the group
            for node in item.nodes:
                _resolve_op_outputs(node)

            # Override the final output with the primary op's output constraint
            last_node = item.nodes[-1]
            if last_node.outputs:
                # Use the primary op's output constraint for the fused output
                if constraint.output_layouts:
                    out_constraint = constraint.output_layouts[0]
                else:
                    out_constraint = ANY_LAYOUT

                # For fused conv groups, the output is padded
                if out_constraint.kind == LayoutKind.ANY:
                    # Propagate from primary's input
                    if primary_node.inputs:
                        first_input = primary_node.inputs[0].name
                        if first_input in resolved:
                            out_constraint = resolved[first_input].layout
                        else:
                            out_constraint = DENSE
                    else:
                        out_constraint = DENSE

                for out_spec in last_node.outputs:
                    physical = out_constraint.physical_shape(list(out_spec.shape), config)
                    resolved[out_spec.name] = ResolvedLayout(
                        layout=out_constraint,
                        logical_shape=list(out_spec.shape),
                        physical_shape=physical,
                    )
        elif isinstance(item, OpNode):
            _resolve_op_outputs(item)

    return resolved


def needs_padded_output(op_type: str, profile: LayoutProfile | None = None) -> bool:
    """Check if an op type produces padded NCHW output.

    Replaces the old _PADDED_KERNELS set check in codegen.py.
    """
    if profile is None:
        profile = METAL_PROFILE
    constraint = profile.get_constraint(op_type)
    if constraint.output_layouts:
        return constraint.output_layouts[0].kind == LayoutKind.PADDED_NCHW
    return False


def get_padded_op_types(profile: LayoutProfile | None = None) -> set[str]:
    """Get all op types that produce padded NCHW output.

    Replaces the old _PADDED_KERNELS set in codegen.py, used for checking
    whether I/O tensors need padding.
    """
    if profile is None:
        profile = METAL_PROFILE
    result = set()
    for op_type, constraint in _LAYOUT_REGISTRY.items():
        # Check overrides first
        effective = profile.get_constraint(op_type)
        if effective.output_layouts and effective.output_layouts[0].kind == LayoutKind.PADDED_NCHW:
            result.add(op_type)
    return result
